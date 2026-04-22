from __future__ import annotations

import pickle
from collections import defaultdict
from typing import Iterable, List, Sequence

import torch


class GlossTokenizerS2G:
    def __init__(self, tokenizer_cfg: dict):
        with open(tokenizer_cfg["gloss2id_file"], "rb") as handle:
            raw_gloss2id = pickle.load(handle)

        if "<si>" in raw_gloss2id:
            silence_token = "<si>"
        elif "<s>" in raw_gloss2id:
            silence_token = "<s>"
        else:
            raise ValueError("gloss vocabulary must contain <si> or <s> for CTC blank")

        if raw_gloss2id[silence_token] != 0:
            raise ValueError(f"CTC blank token {silence_token!r} must map to id 0")

        self.lower_case = tokenizer_cfg.get("lower_case", True)
        self.gloss2id = defaultdict(lambda: raw_gloss2id["<unk>"], raw_gloss2id)
        self.id2gloss = {idx: gloss for gloss, idx in raw_gloss2id.items()}
        self.silence_token = silence_token
        self.silence_id = raw_gloss2id[silence_token]
        self.pad_token = "<pad>"
        self.pad_id = raw_gloss2id[self.pad_token]

    def __len__(self) -> int:
        return len(self.id2gloss)

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, list):
            return [self.convert_tokens_to_ids(token) for token in tokens]
        return self.gloss2id[self._normalize(tokens)]

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, list):
            return [self.convert_ids_to_tokens(idx) for idx in ids]
        return self.id2gloss[int(ids)]

    def __call__(self, batch_gloss_seq: Sequence[str]) -> dict:
        max_length = max(len(gloss_seq.split()) for gloss_seq in batch_gloss_seq)
        gloss_lengths = []
        batch_gloss_ids = []

        for gloss_seq in batch_gloss_seq:
            gloss_ids = [self.gloss2id[self._normalize(token)] for token in gloss_seq.split()]
            gloss_lengths.append(len(gloss_ids))
            padded_ids = gloss_ids + (max_length - len(gloss_ids)) * [self.pad_id]
            batch_gloss_ids.append(padded_ids)

        return {
            "gls_lengths": torch.tensor(gloss_lengths, dtype=torch.long),
            "gloss_labels": torch.tensor(batch_gloss_ids, dtype=torch.long),
        }

    def _normalize(self, token: str) -> str:
        return token.lower() if self.lower_case else token
