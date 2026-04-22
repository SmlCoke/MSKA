from __future__ import annotations

import datetime
import json
import os
import random
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class SmoothedValue:
    def __init__(self, window_size: int = 20, fmt: Optional[str] = None):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt or "{median:.4f} ({global_avg:.4f})"

    def update(self, value, n: int = 1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        values = torch.tensor(list(self.deque))
        return values.median().item()

    @property
    def avg(self):
        values = torch.tensor(list(self.deque), dtype=torch.float32)
        return values.mean().item()

    @property
    def global_avg(self):
        return self.total / max(self.count, 1)

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self) -> str:
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger:
    def __init__(self, delimiter: str = "\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            if isinstance(value, (float, int)):
                self.meters[key].update(value)

    def add_meter(self, name: str, meter: SmoothedValue):
        self.meters[name] = meter

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        raise AttributeError(attr)

    def __str__(self) -> str:
        return self.delimiter.join(f"{name}: {meter}" for name, meter in self.meters.items())

    def log_every(self, iterable: Iterable, print_freq: int, header: str = ""):
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        length = len(iterable)

        for index, obj in enumerate(iterable):
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)

            if index % print_freq == 0 or index == length - 1:
                eta_seconds = iter_time.global_avg * (length - index - 1)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                log_items = [
                    header,
                    f"[{index}/{length}]",
                    f"eta: {eta_string}",
                    str(self),
                    f"time: {iter_time}",
                    f"data: {data_time}",
                ]
                if torch.cuda.is_available():
                    log_items.append(f"max mem: {torch.cuda.max_memory_allocated() / (1024 ** 2):.0f}")
                print(self.delimiter.join(item for item in log_items if item))
            end = time.time()

        total_time = time.time() - start_time
        print(f"{header} Total time: {str(datetime.timedelta(seconds=int(total_time)))} ({total_time / max(length, 1):.4f} s / it)")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters_in_mb(model: torch.nn.Module) -> float:
    return np.sum(np.prod(param.size()) for _, param in model.named_parameters()) / 1e6


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def resolve_path(path_str: str, base_dir: Path) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (base_dir / path).resolve()


def save_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_checkpoint(path: Path, model, optimizer=None, scheduler=None, epoch: int = 0, best_metric: Optional[float] = None):
    payload = {
        "model": model.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    torch.save(payload, path)


class PositionalEncoding(nn.Module):
    def __init__(self, size: int, max_len: int = 5000):
        if size % 2 != 0:
            raise ValueError(f"Positional encoding expects even dimension, got {size}")
        super().__init__()
        pe = torch.zeros(max_len, size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, size, 2, dtype=torch.float) * -(np.log(10000.0) / size))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, emb: Tensor):
        return emb + self.pe[:, : emb.size(1)]


class MaskedNorm(nn.Module):
    def __init__(self, num_features: int = 512, norm_type: str = "batch", num_groups: int = 1):
        super().__init__()
        self.norm_type = norm_type
        self.num_features = num_features

        if norm_type == "batch":
            self.norm = nn.BatchNorm1d(num_features=num_features)
        elif norm_type == "group":
            self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=num_features)
        elif norm_type == "layer":
            self.norm = nn.LayerNorm(normalized_shape=num_features)
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}")

    def forward(self, x: Tensor, mask: Tensor):
        if self.training and self.norm_type == "batch":
            reshaped = x.reshape([-1, self.num_features])
            reshaped_mask = mask.reshape([-1, 1]) > 0
            selected = torch.masked_select(reshaped, reshaped_mask).reshape([-1, self.num_features])
            normalized = self.norm(selected)
            scattered = reshaped.masked_scatter(reshaped_mask, normalized)
            return scattered.reshape([x.shape[0], -1, self.num_features])

        if self.norm_type == "layer":
            return self.norm(x)

        reshaped = x.reshape([-1, self.num_features])
        normalized = self.norm(reshaped)
        return normalized.reshape([x.shape[0], -1, self.num_features])


class PositionwiseFeedForward(nn.Module):
    def __init__(self, input_size, ff_size, dropout=0.1, kernel_size=1, skip_connection=True):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_size, eps=1e-6)
        self.skip_connection = skip_connection

        if isinstance(kernel_size, int):
            self.pwff_layer = nn.Sequential(
                nn.Conv1d(input_size, ff_size, kernel_size=kernel_size, stride=1, padding="same"),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(ff_size, input_size, kernel_size=kernel_size, stride=1, padding="same"),
                nn.Dropout(dropout),
            )
        elif isinstance(kernel_size, list):
            layers = [
                nn.Conv1d(input_size, ff_size, kernel_size=kernel_size[0], stride=1, padding="same"),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            for kernel in kernel_size[1:-1]:
                layers.extend(
                    [
                        nn.Conv1d(ff_size, ff_size, kernel_size=kernel, stride=1, padding="same"),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                    ]
                )
            layers.extend(
                [
                    nn.Conv1d(ff_size, input_size, kernel_size=kernel_size[-1], stride=1, padding="same"),
                    nn.Dropout(dropout),
                ]
            )
            self.pwff_layer = nn.Sequential(*layers)
        else:
            raise ValueError("kernel_size must be int or list[int]")

    def forward(self, x):
        normalized = self.layer_norm(x)
        transformed = self.pwff_layer(normalized.transpose(1, 2)).transpose(1, 2)
        return transformed + x if self.skip_connection else transformed


class MLPHead(nn.Module):
    def __init__(self, embedding_size: int, projection_hidden_size: int):
        super().__init__()
        self.embedding_size = embedding_size
        self.net = nn.Sequential(
            nn.Linear(embedding_size, projection_hidden_size),
            nn.BatchNorm1d(projection_hidden_size),
            nn.ReLU(True),
            nn.Linear(projection_hidden_size, embedding_size),
        )

    def forward(self, x):
        batch_size, time_steps, channels = x.shape
        projected = self.net(x.reshape(-1, channels))
        return projected.reshape(batch_size, time_steps, channels)
