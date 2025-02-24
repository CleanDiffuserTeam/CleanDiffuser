from typing import List

import torch
from transformers import T5EncoderModel, T5Tokenizer


class T5LanguageEncoder:
    def __init__(
        self,
        pretrained_model_name_or_path: str = "google-t5/t5-base",
        max_length: int = 32,
        device: str = "cpu",
    ):
        self.device = device
        self.max_length = max_length
        self.tokenizer = T5Tokenizer.from_pretrained(pretrained_model_name_or_path)
        self.model = T5EncoderModel.from_pretrained(pretrained_model_name_or_path).to(device).eval()

    @torch.no_grad()
    def encode(self, sentences: List[str]):
        inputs = self.tokenizer(
            sentences,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state
        last_hidden_states = last_hidden_states * attention_mask.unsqueeze(-1)
        return last_hidden_states, attention_mask

    def __call__(self, sentences: List[str]):
        return self.encode(sentences)
