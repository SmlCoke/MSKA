from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn
from torch.optim import Optimizer, lr_scheduler


def build_gradient_clipper(config: dict):
    if "clip_grad_val" in config and "clip_grad_norm" in config:
        raise ValueError("Only one of clip_grad_val or clip_grad_norm can be set")

    if "clip_grad_val" in config:
        clip_value = config["clip_grad_val"]
        return lambda params: nn.utils.clip_grad_value_(parameters=params, clip_value=clip_value)

    if "clip_grad_norm" in config:
        max_norm = config["clip_grad_norm"]
        return lambda params: nn.utils.clip_grad_norm_(parameters=params, max_norm=max_norm)

    return None


def build_optimizer(config: dict, model) -> Optimizer:
    optimizer_name = config.get("optimizer", "adam").lower()
    weight_decay = config.get("weight_decay", 0.0)
    eps = config.get("eps", 1.0e-8)
    betas = tuple(config.get("betas", (0.9, 0.999)))
    base_lr = config["learning_rate"]["default"]

    parameter_groups = []
    for module_name, module in model.named_children():
        lr_value = base_lr
        for prefix, group_lr in config["learning_rate"].items():
            if prefix != "default" and prefix in module_name:
                lr_value = group_lr
        parameter_groups.append({"params": module.parameters(), "lr": lr_value})

    if optimizer_name == "adam":
        return torch.optim.Adam(
            params=parameter_groups,
            lr=base_lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=config.get("amsgrad", False),
        )

    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            params=parameter_groups,
            lr=base_lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=config.get("amsgrad", False),
        )

    if optimizer_name == "sgd":
        return torch.optim.SGD(
            params=parameter_groups,
            lr=base_lr,
            momentum=config.get("momentum", 0.0),
            weight_decay=weight_decay,
        )

    raise ValueError(f"Unknown optimizer: {optimizer_name}")


def build_scheduler(config: dict, optimizer: Optimizer) -> Tuple[Optional[lr_scheduler._LRScheduler], Optional[str]]:
    scheduler_name = config.get("scheduler", "").lower()
    if not scheduler_name:
        return None, None

    if scheduler_name == "cosineannealing":
        return (
            lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                eta_min=config.get("eta_min", 0.0),
                T_max=config.get("t_max", 20),
            ),
            "epoch",
        )

    if scheduler_name == "plateau":
        return (
            lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                mode=config.get("mode", "min"),
                threshold_mode="abs",
                factor=config.get("decrease_factor", 0.1),
                patience=config.get("patience", 10),
            ),
            "validation",
        )

    raise ValueError(f"Unsupported scheduler: {scheduler_name}")
