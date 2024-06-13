from typing import Dict, Tuple, Union, Callable
import copy
import torch
import torch.nn as nn
import torchvision

from cleandiffuser.utils.crop_randomizer import CropRandomizer
from cleandiffuser.nn_condition import BaseNNCondition


def replace_submodules(
        root_module: nn.Module, 
        predicate: Callable[[nn.Module], bool], 
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all BN are replaced
    bn_list = [k.split('.') for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module


def get_resnet(name, weights=None, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", "r3m"
    """
    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)
    resnet.fc = torch.nn.Identity()
    return resnet


class MultiImageObsCondition(BaseNNCondition):

    """
    Input:
        - condition: {"cond1": (b, *cond1_shape), "cond2": (b, *cond2_shape), ...} or (b, *cond_in_shape)
        - mask :     (b, *mask_shape) or None, None means no mask

    Output:
        - condition: (b, *cond_out_shape)
    
    Assumes rgb input: B, C, H, W or B, seq_len, C,H,W
    Assumes low_dim input: B, D or B, seq_len, D
    """
    def __init__(self,
            shape_meta: dict,
            rgb_model_name: str,
            emb_dim: int = 256, 
            resize_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,
            crop_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,
            random_crop: bool=True,
            # replace BatchNorm with GroupNorm
            use_group_norm: bool=False,
            # use single rgb model for all rgb inputs
            share_rgb_model: bool=False,
            # renormalize rgb input with imagenet normalization
            # assuming input in [0,1]
            imagenet_norm: bool=False,
            # use_seq: B, seq_len, C, H, W or B, C, H, W
            use_seq=False, 
            # if True: (bs, seq_len, embed_dim)
            keep_horizon_dims=False
        ):
        super().__init__()
        rgb_keys = list()
        low_dim_keys = list()
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = dict()

        # rgb_model
        if 'resnet' in rgb_model_name:
            rgb_model = get_resnet(rgb_model_name)
        else:
            raise ValueError("Fatal rgb_model")

        # handle sharing vision backbone
        if share_rgb_model:
            assert isinstance(rgb_model, nn.Module)
            key_model_map['rgb'] = rgb_model

        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            # print(key, attr)
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            key_shape_map[key] = shape
            if type == 'rgb':
                rgb_keys.append(key)
                # configure model for this key
                this_model = None
                if not share_rgb_model:
                    if isinstance(rgb_model, dict):
                        # have provided model for each key
                        this_model = rgb_model[key]
                    else:
                        assert isinstance(rgb_model, nn.Module)
                        # have a copy of the rgb model
                        this_model = copy.deepcopy(rgb_model)
                
                if this_model is not None:
                    if use_group_norm:
                        this_model = replace_submodules(
                            root_module=this_model,
                            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                            func=lambda x: nn.GroupNorm(
                                num_groups=x.num_features//16, 
                                num_channels=x.num_features)
                        )
                    key_model_map[key] = this_model
                
                # configure resize
                input_shape = shape
                this_resizer = nn.Identity()
                if resize_shape is not None:
                    if isinstance(resize_shape, dict):
                        h, w = resize_shape[key]
                    else:
                        h, w = resize_shape
                    this_resizer = torchvision.transforms.Resize(
                        size=(h, w)
                    )
                    input_shape = (shape[0], h, w)

                # configure randomizer
                this_randomizer = nn.Identity()
                if crop_shape is not None:
                    if isinstance(crop_shape, dict):
                        h, w = crop_shape[key]
                    else:
                        h, w = crop_shape
                    if random_crop:
                        this_randomizer = CropRandomizer(
                            input_shape=input_shape,
                            crop_height=h,
                            crop_width=w,
                            num_crops=1,
                            pos_enc=False
                        )
                    else:
                        this_randomizer = torchvision.transforms.CenterCrop(
                            size=(h,w)
                        )
                # configure normalizer
                this_normalizer = nn.Identity()
                if imagenet_norm:
                    this_normalizer = torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                
                this_transform = nn.Sequential(this_resizer, this_randomizer, this_normalizer)
                key_transform_map[key] = this_transform
            elif type == 'low_dim':
                low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)

        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map

        self.use_seq = use_seq
        self.keep_horizon_dims = keep_horizon_dims
        self.mlp = nn.Sequential(
            nn.Linear(self.output_shape(), emb_dim), nn.LeakyReLU(), nn.Linear(emb_dim, emb_dim))

    def multi_image_forward(self, obs_dict):
        batch_size = None
        features = list()

        if self.use_seq:
            # input: (bs, horizon, c, h, w)
            for k in obs_dict.keys():
                obs_dict[k] = obs_dict[k].flatten(end_dim=1)

        # process rgb input
        if self.share_rgb_model:
            # pass all rgb obs to rgb model
            imgs = list()
            for key in self.rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert img.shape[1:] == self.key_shape_map[key]
                img = self.key_transform_map[key](img)
                imgs.append(img)
            # (N*B,C,H,W)
            imgs = torch.cat(imgs, dim=0)
            # (N*B,D)
            feature = self.key_model_map['rgb'](imgs)
            # (N,B,D)
            feature = feature.reshape(-1,batch_size,*feature.shape[1:])
            # (B,N,D)
            feature = torch.moveaxis(feature,0,1)
            # (B,N*D)
            feature = feature.reshape(batch_size,-1)
            features.append(feature)
        else:
            # run each rgb obs to independent models
            for key in self.rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert img.shape[1:] == self.key_shape_map[key]
                img = self.key_transform_map[key](img)
                feature = self.key_model_map[key](img)
                features.append(feature)
        
        # process lowdim input
        for key in self.low_dim_keys:
            data = obs_dict[key]
            if batch_size is None:
                batch_size = data.shape[0]
            else:
                assert batch_size == data.shape[0]
            assert data.shape[1:] == self.key_shape_map[key]
            features.append(data)
        
        # concatenate all features
        features = torch.cat(features, dim=-1)
        return features

    def forward(self, obs_dict, mask=None):
        ori_batch_size, ori_seq_len = self.get_batch_size(obs_dict)
        features = self.multi_image_forward(obs_dict)
        # linear embedding
        result = self.mlp(features)
        if self.use_seq:
            if self.keep_horizon_dims:
                result = result.reshape(ori_batch_size, ori_seq_len, -1)
            else:
                result = result.reshape(ori_batch_size, -1)
        return result
    
    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        batch_size = 1
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            if self.use_seq:
                prefix = (batch_size, 1)
            else:
                prefix = (batch_size,)
            this_obs = torch.zeros(
                prefix + shape, 
                dtype=self.dtype,
                device=self.device)
            example_obs_dict[key] = this_obs
        example_output = self.multi_image_forward(example_obs_dict)
        output_shape = example_output.shape[1:]
        return output_shape[0]
    
    def get_batch_size(self, obs_dict):
        any_key = next(iter(obs_dict))
        any_tensor = obs_dict[any_key]
        return any_tensor.size(0), any_tensor.size(1)

    @property
    def device(self):
        return next(iter(self.parameters())).device
    
    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
    

# if __name__ == "__main__":
#     shape_meta = {
#             'obs': {
#                 'image': {
#                     'shape': (3, 96, 96),
#                     'type': 'rgb'
#                 },
#                 'agent_pos': {
#                     'shape': (2, ),
#                     'type': 'low_dim'
#                 }
#             }
#         }
#     rgb_model = "resnet18"

#     resize_shape=None
#     crop_shape=(84, 84)
#     random_crop=True
#     use_group_norm=True
#     share_rgb_model=False
#     imagenet_norm=False
#     im = MultiImageObsCondition(shape_meta, rgb_model=rgb_model, resize_shape=resize_shape, crop_shape=crop_shape,
#                               random_crop=random_crop, use_group_norm=use_group_norm, share_rgb_model=share_rgb_model,
#                               imagenet_norm=imagenet_norm)
#     im.output_shape()