import json
from typing import List

import pytorch_lightning as L
import torch
import torch.nn as nn
import torchvision.transforms as T
import zarr
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import AutoModel, T5EncoderModel, T5Tokenizer
import cv2
import numpy as np
import re

class T5LanguageEncoder:
    def __init__(
        self,
        pretrained_model_name_or_path: str = "google-t5/t5-base",
        device: str = "cpu",
    ):
        self.device = device
        self.tokenizer = T5Tokenizer.from_pretrained(pretrained_model_name_or_path)
        self.model = T5EncoderModel.from_pretrained(pretrained_model_name_or_path).to(device).eval()

    @torch.no_grad()
    def encode(self, sentences: List[str]):
        inputs = self.tokenizer(
            sentences, return_tensors="pt", padding="max_length", max_length=32, truncation=True
        )
        input_ids = inputs.input_ids
        outputs = self.model(input_ids=input_ids.to(self.device))
        last_hidden_states = outputs.last_hidden_state
        attention_mask = inputs.attention_mask
        return last_hidden_states, attention_mask.to(last_hidden_states.device)

    def __call__(self, sentences: List[str]):
        return self.encode(sentences)


class Predictor(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.dino = AutoModel.from_pretrained("facebook/dinov2-base").train()
        self.normalizer = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

        self.trace_embed = nn.Parameter(torch.randn(1, 8, 768) * 0.02)
        self.image_proj = nn.Sequential(nn.Linear(768, 768), nn.SiLU(), nn.Linear(768, 768))
        self.lang_proj = nn.Sequential(nn.Linear(768, 768), nn.SiLU(), nn.Linear(768, 768))
        self.pos_emb = nn.Parameter(torch.randn(1, 8 + 257 + 32, 768) * 0.02)

        # self.transformer = nn.ModuleList(
        #     [nn.TransformerDecoderLayer(768, 12, 3072, batch_first=True) for _ in range(6)]
        # )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(768, 12, 3072, batch_first=True), num_layers=6
        )

        self.pooler = nn.Sequential(nn.Linear(768, 2), nn.Tanh())

    def configure_optimizers(self):
        x = "dino"
        return torch.optim.Adam(
            [
                {"params": [p for n, p in self.named_parameters() if x not in n], "lr": 1e-4},
                {"params": [p for n, p in self.named_parameters() if x in n], "lr": 1e-5},
            ]
        )

    def forward(self, image, lang_emb, lang_mask):
        image = self.normalizer(image / 255.0)
        image_emb = self.dino(image)["last_hidden_state"]
        image_emb = self.image_proj(image_emb)
        lang_emb = self.lang_proj(lang_emb)

        x = torch.cat(
            [self.trace_embed.repeat(image_emb.size(0), 1, 1), image_emb, lang_emb], dim=1
        )
        x = x + self.pos_emb

        lang_mask = torch.logical_not(lang_mask).to(torch.bool)
        mask = torch.nn.functional.pad(lang_mask, (8 + 257, 0), value=False)

        x = self.transformer(x, src_key_padding_mask=mask)[:, :8]

        return self.pooler(x)

    def training_step(self, batch):
        points, image, lang_emb, lang_mask = batch
        pred = self(image, lang_emb, lang_mask)
        loss = nn.functional.mse_loss(pred, points)
        self.log("train_loss", loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
        return loss


class SVTDataset(torch.utils.data.Dataset):
    def __init__(self, root: zarr.hierarchy.Group):
        self.root = root
        self.sz = len(self.root["images"])

    def __len__(self):
        return self.sz

    def __getitem__(self, idx):
        image = self.root["images"][idx]
        lang_emb = self.root["lang_embs"][idx]
        lang_mask = self.root["lang_masks"][idx]
        points = self.root["points"][idx]
        points = 2 * points - 1
        return points, image, lang_emb, lang_mask

def process_filename(filename):
    pattern = r"^im_\d+\.jpg$"
    if re.match(pattern, filename):
        return "im_0.jpg"
    else:
        return filename

if __name__ == "__main__":
    root = zarr.open("/mnt/new_driver_18t/dzb/svt.zarr", mode="r")
    dataset = SVTDataset(root)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=True, persistent_workers=True, num_workers=8
    )

    predictor = Predictor.load_from_checkpoint(
        "lightning_logs/version_0/checkpoints/epoch=999-step=50000.ckpt"
    )

    device = "cuda:3"
    predictor = predictor.to(device)
    t5_encoder = T5LanguageEncoder(device=device)
    
    with open("pred_test/questions.jsonl", "r") as f:
        datas = [json.loads(line) for line in f]
    
    images, texts = [], []
    gt_traces = []
    for data in datas:
        text = data['task_instruction']
        gt = np.array(data['visual_trace'])
        gt[:, 0] = gt[:, 0] / 999
        gt[:, 1] = gt[:, 1] / 999
        image_path = data['rel_image_path']
        image = cv2.imread('pred_test/images/' + image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224)).transpose(2, 0, 1)
        texts.append(text)
        images.append(image)
        gt_traces.append(gt)
    images = torch.tensor(np.array(images)).to(device)
    lang_embs, lang_masks = t5_encoder(texts)
    lang_embs = lang_embs.to(device)
    lang_masks = lang_masks.to(device)
    
    with torch.no_grad():
        predictor.eval()
        traces = predictor(images, lang_embs, lang_masks)
    traces = traces * 0.5 + 0.5
    traces = traces.cpu().numpy()

    gt_traces = np.array(gt_traces)
    
    mse = np.sqrt(((traces - gt_traces) ** 2).mean())

    # images, texts = [], []
    # heights, widths = [], []
    # orig_images = []
    # for data in datas:
    #     image_path = data["image"]
    #     image = cv2.imread("pred_test/" + image_path)
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     orig_images.append(image)
    #     h, w = image.shape[:2]
    #     image = cv2.resize(image, (224, 224)).transpose(2, 0, 1)
        
    #     heights.append(h)
    #     widths.append(w)
    #     images.append(image)
    #     texts.append(data["text"])
    
    # images = torch.tensor(np.array(images)).to(device)
    # lang_embs, lang_masks = t5_encoder(texts)
    # lang_embs = lang_embs.to(device)
    # lang_masks = lang_masks.to(device)
    # heights = torch.tensor(heights).to(device)
    # widths = torch.tensor(widths).to(device)
    
    # with torch.no_grad():
    #     predictor.eval()
    #     trace = predictor(images, lang_embs, lang_masks)
    # trace = (trace * 0.5 + 0.5) * torch.stack([heights, widths], dim=1).unsqueeze(1)
    
    # # plot scatters in images
    # trace = trace.cpu().numpy()
    # orig_images = np.array(orig_images)
    # for i in range(len(orig_images)):
    #     image = orig_images[i]
    #     trace_ = trace[i]
    #     for j in range(len(trace_)):
    #         x, y = trace_[j]
    #         cv2.circle(image, (int(x), int(y)), 3, (0, 255, 0), -1)
    #     cv2.imwrite(f"pred_test/{i}.jpg", image)
        
    
    
    # callback = ModelCheckpoint(
    #     every_n_train_steps=10_000,
    #     save_top_k=-1,
    # )
    # trainer = L.Trainer(
    #     devices=[0, 1, 2, 3],
    #     max_steps=50_000,
    #     callbacks=[callback],
    #     strategy="ddp_find_unused_parameters_true",
    # )
    # trainer.fit(predictor, dataloader)
