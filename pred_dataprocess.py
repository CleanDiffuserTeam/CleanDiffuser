import json
import re
from typing import List

import cv2
import numpy as np
import torch
import zarr
from transformers import T5EncoderModel, T5Tokenizer


def process_filename(filename):
    pattern = r"^im_\d+\.jpg$"
    if re.match(pattern, filename):
        return "im_0.jpg"
    else:
        return filename


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


if __name__ == "__main__":
    with open("level_5_spatial_visual_trace_prediction_final.json", "r") as f:
        data = json.load(f)

    langs = []
    images = []
    points = []

    for i in range(len(data)):
        example = data[i]

        try:
            lang = (
                example["conversations"][0]["value"]
                .split("Your task instruction: ")[1]
                .split("\n")[0]
                .split("You need to understand free space and ")[0]
            )
            while lang[-1] in [" ", "."]:
                lang = lang[:-1]

            image_path = example["image"]
            prefix = image_path.split("/")[0]
            postfix = image_path[len(prefix) :]
            if prefix == "droid":
                image_path = "/mnt/new_driver_18t/droid_filtered" + postfix
            elif prefix == "bridge_data_v2":
                image_path = "/mnt/new_driver_18t/datasets/bridgev2/raw" + postfix
            elif prefix == "rtx":
                image_path = "/mnt/new_driver_18t/datasets/openx_processed" + postfix
            else:
                raise ValueError("Unknown prefix: {}".format(prefix))

            file_name = image_path.split("/")[-1]
            new_file_name = process_filename(file_name)
            image_path = image_path.replace(file_name, new_file_name)

            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
            image = cv2.resize(image, (224, 224)).transpose(2, 0, 1)

            try:
                point_answer = example["conversations"][1]["value"].split("<Answer>")[1]
                point = point_answer.split("<point>")
                point = point[1].split("</point>")[0]
                point = json.loads(point)
                while len(point) < 8:
                    point.append(point[-1])
            except Exception as e:
                print(f"Point error: {e}. Use default points.")
                continue

        except Exception as e:
            print(e)
            continue

        langs.append(lang)
        images.append(image)
        points.append(np.array(point) / 999)

        print(f"({i + 1} / {len(data)})")

    images = np.array(images)
    points = np.array(points)
    lang_embs = []
    lang_masks = []

    device = "cuda:3"
    t5_encoder = T5LanguageEncoder(device=device)
    bs = 1000
    for i in range(0, len(langs), bs):
        lang_emb, lang_mask = t5_encoder(langs[i : i + bs])
        lang_emb = lang_emb.cpu().numpy()
        lang_mask = lang_mask.cpu().numpy()
        lang_embs.append(lang_emb * lang_mask[:, :, None])
        lang_masks.append(lang_mask)
    lang_embs = np.concatenate(lang_embs, axis=0)
    lang_masks = np.concatenate(lang_masks, axis=0)

    root = zarr.open("/mnt/new_driver_18t/dzb/svt.zarr", mode="w")
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=zarr.Blosc.AUTOSHUFFLE)
    root.create_dataset(
        "images", data=images, chunks=(1, 3, 224, 224), compressor=compressor, dtype=np.uint8
    )
    root.create_dataset(
        "points", data=points, chunks=(1, 8, 2), compressor=compressor, dtype=np.float32
    )
    root.create_dataset(
        "lang_embs", data=lang_embs, chunks=(1, 32, 768), compressor=compressor, dtype=np.float32
    )
    root.create_dataset(
        "lang_masks", data=lang_masks, chunks=(1, 32), compressor=compressor, dtype=np.float32
    )
    root.attrs["langs"] = langs
