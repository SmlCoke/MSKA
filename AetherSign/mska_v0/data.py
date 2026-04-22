from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SampleRecord:
    index: str
    name: str
    length: int
    gloss: str


def load_label_csv(csv_path: Path) -> List[SampleRecord]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        records = []
        for row in reader:
            records.append(
                SampleRecord(
                    index=row["index"],
                    name=row["name"],
                    length=int(row["length"]),
                    gloss=row["gloss"].strip(),
                )
            )
    return records


def get_selected_index(vlen: int, clip_len: int, tmin: float, tmax: float):
    if tmin == 1 and tmax == 1:
        if vlen <= clip_len:
            frame_index = np.arange(vlen)
            valid_len = vlen
        else:
            start = (vlen - clip_len) // 2
            end = start + clip_len
            frame_index = np.arange(vlen)[start:end]
            valid_len = clip_len

        if valid_len % 4 != 0:
            valid_len -= valid_len % 4
            frame_index = frame_index[:valid_len]
        return frame_index, valid_len

    min_len = min(int(tmin * vlen), clip_len)
    max_len = min(clip_len, int(tmax * vlen))
    selected_len = np.random.randint(min_len, max_len + 1)
    if selected_len % 4 != 0:
        selected_len += 4 - (selected_len % 4)

    if selected_len <= vlen:
        selected_index = sorted(np.random.permutation(np.arange(vlen))[:selected_len])
    else:
        copied_index = np.random.randint(0, vlen, selected_len - vlen)
        selected_index = sorted(np.concatenate([np.arange(vlen), copied_index]))

    if selected_len > clip_len:
        raise ValueError(f"selected length {selected_len} exceeds clip length {clip_len}")

    return np.asarray(selected_index), selected_len


def downsample_lengths(src_lengths: torch.Tensor) -> torch.Tensor:
    new_lengths = (((src_lengths - 1) / 2) + 1).long()
    new_lengths = (((new_lengths - 1) / 2) + 1).long()
    return new_lengths


def preprocess_keypoints(keypoints: torch.Tensor, phase: str, data_cfg: dict) -> torch.Tensor:
    keypoints = keypoints.clone().to(torch.float32)
    if data_cfg.get("convert_to_centered_coords", True):
        keypoints[:, 0, :, :] = 2 * keypoints[:, 0, :, :] - 1
        keypoints[:, 1, :, :] = 1 - 2 * keypoints[:, 1, :, :]

    if phase == "train":
        keypoints = random_rotate_xy(keypoints, max_degree=data_cfg.get("random_rotate_degree", 15.0))
    return keypoints


def random_rotate_xy(keypoints: torch.Tensor, max_degree: float = 15.0) -> torch.Tensor:
    if keypoints.numel() == 0:
        return keypoints

    batch_size = keypoints.shape[0]
    apply_mask = torch.rand(batch_size, device=keypoints.device) >= 0.5
    angles = (torch.rand(batch_size, device=keypoints.device) * 2 - 1) * math.radians(max_degree)
    cos_theta = torch.cos(angles)
    sin_theta = torch.sin(angles)
    rotation = torch.stack(
        [
            torch.stack([cos_theta, -sin_theta], dim=-1),
            torch.stack([sin_theta, cos_theta], dim=-1),
        ],
        dim=1,
    )

    xy = keypoints[:, :2].permute(0, 2, 3, 1)
    rotated_xy = torch.einsum("btvc,bcd->btvd", xy, rotation)
    xy = torch.where(apply_mask.view(batch_size, 1, 1, 1), rotated_xy, xy)
    keypoints[:, :2] = xy.permute(0, 3, 1, 2)
    return keypoints


def collate_keypoint_samples(samples, tokenizer, data_cfg: dict, phase: str):
    clip_len = data_cfg.get("max_length", 400)
    if phase == "train":
        tmin, tmax = data_cfg.get("time_augment_min", 0.5), data_cfg.get("time_augment_max", 1.5)
    else:
        tmin, tmax = 1.0, 1.0

    gloss_batch = []
    keypoint_batch = []
    src_length_batch = []
    name_batch = []

    for name, keypoint, gloss, length in samples:
        frame_index, valid_len = get_selected_index(length, clip_len=clip_len, tmin=tmin, tmax=tmax)
        keypoint_batch.append(torch.stack([keypoint[:, frame_id, :] for frame_id in frame_index], dim=1))
        src_length_batch.append(valid_len)
        name_batch.append(name)
        gloss_batch.append(gloss)

    max_length = max(src_length_batch)
    padded_keypoints = []
    for keypoints, valid_len in zip(keypoint_batch, src_length_batch):
        if valid_len < max_length:
            padding = keypoints[:, -1, :].unsqueeze(1).repeat(1, max_length - valid_len, 1)
            keypoints = torch.cat([keypoints, padding], dim=1)
        padded_keypoints.append(keypoints)

    keypoint_tensor = torch.stack(padded_keypoints, dim=0)
    keypoint_tensor = preprocess_keypoints(keypoint_tensor, phase=phase, data_cfg=data_cfg)

    src_lengths = torch.tensor(src_length_batch, dtype=torch.long)
    new_src_lengths = downsample_lengths(src_lengths)
    gloss_input = tokenizer(gloss_batch)

    max_new_len = int(new_src_lengths.max().item())
    mask = torch.zeros(new_src_lengths.shape[0], 1, max_new_len, dtype=torch.bool)
    for idx, seq_len in enumerate(new_src_lengths):
        mask[idx, :, : int(seq_len.item())] = True

    return {
        "name": name_batch,
        "keypoint": keypoint_tensor,
        "gloss": gloss_batch,
        "mask": mask,
        "new_src_lengths": new_src_lengths,
        "gloss_input": gloss_input,
        "src_length": src_lengths,
    }


def build_inference_batch(name: str, keypoint: torch.Tensor, gloss: str, tokenizer, data_cfg: dict):
    return collate_keypoint_samples(
        [(name, keypoint, gloss, int(keypoint.shape[1]))],
        tokenizer=tokenizer,
        data_cfg=data_cfg,
        phase="test",
    )


class SignGlossDataset(Dataset):
    def __init__(self, split_csv: Path, npy_dir: Path, tokenizer, config: dict, phase: str):
        self.split_csv = Path(split_csv)
        self.npy_dir = Path(npy_dir)
        self.tokenizer = tokenizer
        self.config = config
        self.phase = phase
        self.records = load_label_csv(self.split_csv)
        self.num_keypoints = config["data"]["num_keypoints"]
        self.keypoint_format = config["data"].get("keypoint_format", "CTV")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        npy_path = self.npy_dir / f"{record.name}.npy"
        keypoint = np.load(npy_path, allow_pickle=False)
        keypoint = self._validate_keypoint(keypoint=keypoint, record=record, npy_path=npy_path)
        return record.name, torch.from_numpy(keypoint).to(torch.float32), record.gloss, record.length

    def collate_fn(self, batch):
        return collate_keypoint_samples(
            batch,
            tokenizer=self.tokenizer,
            data_cfg=self.config["data"],
            phase=self.phase,
        )

    def _validate_keypoint(self, keypoint: np.ndarray, record: SampleRecord, npy_path: Path) -> np.ndarray:
        if keypoint.ndim != 3:
            raise ValueError(f"{npy_path} should be a 3D array, got shape {keypoint.shape}")

        if self.keypoint_format != "CTV":
            raise ValueError(f"Unsupported keypoint format: {self.keypoint_format}")

        channels, time_steps, num_keypoints = keypoint.shape
        if channels != 3:
            raise ValueError(f"{npy_path} should have C=3, got {channels}")
        if num_keypoints != self.num_keypoints:
            raise ValueError(f"{npy_path} should have V={self.num_keypoints}, got {num_keypoints}")
        if time_steps != record.length:
            raise ValueError(
                f"{npy_path} length mismatch: csv says {record.length}, npy says {time_steps}"
            )
        return keypoint
