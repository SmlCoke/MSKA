from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import yaml
from torch.utils.data import DataLoader

from .data import SignGlossDataset, build_inference_batch
from .metrics import wer_list
from .model import SignLanguageModel
from .optimizer import build_optimizer, build_scheduler
from .tokenizer import GlossTokenizerS2G
from .utils import MetricLogger, count_parameters_in_mb, ensure_dir, resolve_path, save_checkpoint, save_jsonl, set_seed

HEAD_TO_LOGITS_KEY = {
    "ensemble_last": "ensemble_last_gloss_logits",
    "fuse": "fuse_gloss_logits",
    "left": "left_gloss_logits",
    "right": "right_gloss_logits",
    "body": "body_gloss_logits",
}


def load_config(config_path: Path) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def prepare_runtime_config(
    config_path: Path,
    device: Optional[str] = None,
    batch_size: Optional[int] = None,
    epochs: Optional[int] = None,
    num_workers: Optional[int] = None,
) -> dict:
    config = load_config(config_path)
    config = copy.deepcopy(config)
    config_dir = config_path.parent.resolve()
    data_cfg = config["data"]

    dataset_root = resolve_path(data_cfg.get("dataset_root", "../dataset"), config_dir)
    label_dir = resolve_path(data_cfg.get("label_dir", str(dataset_root / "label")), config_dir)
    npy_dir = resolve_path(data_cfg.get("npy_dir", str(dataset_root / "npy")), config_dir)
    data_cfg["dataset_root"] = str(dataset_root)
    data_cfg["label_dir"] = str(label_dir)
    data_cfg["npy_dir"] = str(npy_dir)
    data_cfg["train_label_path"] = str(resolve_path(data_cfg.get("train_label_path", str(label_dir / "train.csv")), config_dir))
    data_cfg["dev_label_path"] = str(resolve_path(data_cfg.get("dev_label_path", str(label_dir / "dev.csv")), config_dir))
    data_cfg["test_label_path"] = str(resolve_path(data_cfg.get("test_label_path", str(label_dir / "test.csv")), config_dir))

    config["gloss"]["gloss2id_file"] = str(resolve_path(config["gloss"]["gloss2id_file"], config_dir))
    config["model"]["RecognitionNetwork"]["GlossTokenizer"]["gloss2id_file"] = config["gloss"]["gloss2id_file"]

    training_cfg = config["training"]
    training_cfg["model_dir"] = str(resolve_path(training_cfg["model_dir"], config_dir))
    training_cfg["batch_size"] = int(batch_size if batch_size is not None else training_cfg.get("batch_size", config.get("batch_size", 8)))
    training_cfg["epochs"] = int(epochs if epochs is not None else training_cfg.get("epochs", 100))
    training_cfg["num_workers"] = int(num_workers if num_workers is not None else training_cfg.get("num_workers", 4))

    config["device"] = device or config.get("device", "cuda")
    config["_meta"] = {"config_path": str(config_path.resolve()), "config_dir": str(config_dir)}
    return config


def build_tokenizer(config: dict) -> GlossTokenizerS2G:
    return GlossTokenizerS2G(config["gloss"])


def build_dataloader(config: dict, tokenizer, split: str, batch_size: Optional[int] = None, shuffle: Optional[bool] = None):
    split = split.lower()
    if split not in {"train", "dev", "test"}:
        raise ValueError(f"Unsupported split: {split}")

    phase = "train" if split == "train" else "val" if split == "dev" else "test"
    label_path = Path(config["data"][f"{split}_label_path"])
    dataset = SignGlossDataset(
        split_csv=label_path,
        npy_dir=Path(config["data"]["npy_dir"]),
        tokenizer=tokenizer,
        config=config,
        phase=phase,
    )
    effective_batch_size = batch_size or config["training"]["batch_size"]
    effective_shuffle = shuffle if shuffle is not None else split == "train"
    loader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=effective_shuffle,
        num_workers=config["training"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
        drop_last=split == "train",
        collate_fn=dataset.collate_fn,
    )
    return dataset, loader


def build_model(config: dict):
    device = torch.device(config["device"])
    model = SignLanguageModel(cfg=config)
    model.to(device)
    return model, device


def build_training_components(config: dict, model):
    optimizer = build_optimizer(config["training"]["optimization"], model)
    scheduler, scheduler_step_at = build_scheduler(config["training"]["optimization"], optimizer)
    return optimizer, scheduler, scheduler_step_at


def load_checkpoint(path: Path, model, optimizer=None, scheduler=None, strict: bool = True):
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=strict)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint


def decode_head_predictions(model, tokenizer, output: dict, beam_size: int, head_name: str):
    logits_key = HEAD_TO_LOGITS_KEY[head_name]
    decoded = model.predict_gloss_from_logits(
        gloss_logits=output[logits_key],
        beam_size=beam_size,
        input_lengths=output["input_lengths"],
    )
    batch_tokens = tokenizer.convert_ids_to_tokens(decoded)
    return [" ".join(tokens) for tokens in batch_tokens]


def evaluate_loader(
    model,
    tokenizer,
    data_loader,
    config: dict,
    beam_size: int,
    prediction_head: Optional[str] = None,
    output_jsonl: Optional[Path] = None,
):
    prediction_head = prediction_head or config["testing"]["recognition"].get("prediction_head", "ensemble_last")
    report_heads = config["testing"]["recognition"].get(
        "report_heads",
        ["ensemble_last", "fuse", "body", "left", "right"],
    )
    report_heads = [head for head in report_heads if head in HEAD_TO_LOGITS_KEY]

    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    all_rows = []
    references_by_name = {}
    predictions_by_head = {head: {} for head in report_heads}

    with torch.no_grad():
        for batch in metric_logger.log_every(data_loader, print_freq=10, header="Eval"):
            output = model(batch)
            metric_logger.update(loss=output["total_loss"].item())
            for head_name in report_heads:
                predictions = decode_head_predictions(model, tokenizer, output, beam_size=beam_size, head_name=head_name)
                for name, prediction, reference in zip(batch["name"], predictions, batch["gloss"]):
                    predictions_by_head[head_name][name] = prediction
                    references_by_name[name] = reference

    rows = []
    for name, reference in references_by_name.items():
        row = {"name": name, "gloss_ref": reference}
        for head_name in report_heads:
            row[f"{head_name}_gloss_pred"] = predictions_by_head[head_name].get(name, "")
        row["prediction_head"] = prediction_head
        row["gloss_pred"] = row.get(f"{prediction_head}_gloss_pred", "")
        rows.append(row)

    if output_jsonl is not None:
        save_jsonl(output_jsonl, rows)

    summary = {
        "loss": metric_logger.loss.global_avg if "loss" in metric_logger.meters else 0.0,
        "prediction_head": prediction_head,
        "report_heads": report_heads,
        "num_samples": len(rows),
    }

    if rows and all(row["gloss_ref"] for row in rows):
        wer_by_head = {}
        for head_name in report_heads:
            hypotheses = [row[f"{head_name}_gloss_pred"] for row in rows]
            references = [row["gloss_ref"] for row in rows]
            wer_by_head[head_name] = wer_list(references=references, hypotheses=hypotheses)
        summary["wer_by_head"] = wer_by_head
        summary["prediction_head_wer"] = wer_by_head[prediction_head]["wer"]
        summary["best_wer"] = min(result["wer"] for result in wer_by_head.values())
    else:
        summary["wer_by_head"] = {}
        summary["prediction_head_wer"] = None
        summary["best_wer"] = None

    return summary, rows


def run_single_npy_inference(
    model,
    tokenizer,
    config: dict,
    npy_path: Path,
    beam_size: int,
    prediction_head: Optional[str] = None,
    gloss: str = "",
):
    npy_path = Path(npy_path)
    keypoint = torch.from_numpy(__import__("numpy").load(npy_path, allow_pickle=False)).to(torch.float32)
    batch = build_inference_batch(
        name=npy_path.stem,
        keypoint=keypoint,
        gloss=gloss,
        tokenizer=tokenizer,
        data_cfg=config["data"],
    )

    model.eval()
    with torch.no_grad():
        output = model(batch)

    prediction_head = prediction_head or config["testing"]["recognition"].get("prediction_head", "ensemble_last")
    prediction = decode_head_predictions(model, tokenizer, output, beam_size=beam_size, head_name=prediction_head)[0]

    result = {
        "name": npy_path.stem,
        "prediction_head": prediction_head,
        "gloss_pred": prediction,
        "gloss_ref": gloss,
    }
    if gloss:
        result["wer"] = wer_list(references=[gloss], hypotheses=[prediction])["wer"]
    return result


def train_one_epoch(model, data_loader, optimizer, device, epoch: int, epochs: int, grad_clipper=None):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", metric_logger.meters["lr"])
    header = f"Epoch: [{epoch}/{epochs}]"

    for batch in metric_logger.log_every(data_loader, print_freq=10, header=header):
        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        output["total_loss"].backward()
        if grad_clipper is not None:
            grad_clipper(model.parameters())
        optimizer.step()
        metric_logger.update(loss=output["total_loss"].item(), lr=optimizer.param_groups[0]["lr"])

    return {name: meter.global_avg for name, meter in metric_logger.meters.items()}


def dump_metrics_jsonl(path: Path, row: dict):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize_model(model) -> float:
    return count_parameters_in_mb(model)


__all__ = [
    "HEAD_TO_LOGITS_KEY",
    "build_dataloader",
    "build_model",
    "build_tokenizer",
    "build_training_components",
    "dump_metrics_jsonl",
    "evaluate_loader",
    "load_checkpoint",
    "prepare_runtime_config",
    "run_single_npy_inference",
    "save_checkpoint",
    "set_seed",
    "summarize_model",
]
