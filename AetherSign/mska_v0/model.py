from __future__ import annotations

import torch

from .recognition import Recognition


class SignLanguageModel(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg["task"] != "S2G":
            raise ValueError("The self-contained v0 package only implements the S2G task")
        self.task = cfg["task"]
        self.device_name = cfg["device"]
        self.recognition_network = Recognition(cfg=cfg["model"]["RecognitionNetwork"])
        self.gloss_tokenizer = self.recognition_network.gloss_tokenizer

    def forward(self, src_input, **kwargs):
        recognition_outputs = self.recognition_network(src_input)
        return {**recognition_outputs, "total_loss": recognition_outputs["recognition_loss"]}

    def predict_gloss_from_logits(self, gloss_logits, beam_size, input_lengths):
        return self.recognition_network.decode(
            gloss_logits=gloss_logits,
            beam_size=beam_size,
            input_lengths=input_lengths,
        )
