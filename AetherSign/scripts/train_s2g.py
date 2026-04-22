from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mska_v0.engine import (  # noqa: E402
    build_dataloader,
    build_model,
    build_tokenizer,
    build_training_components,
    dump_metrics_jsonl,
    evaluate_loader,
    load_checkpoint,
    prepare_runtime_config,
    save_checkpoint,
    set_seed,
    summarize_model,
)
from mska_v0.optimizer import build_gradient_clipper  # noqa: E402
from mska_v0.utils import ensure_dir  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Train AetherSign MSKA-SLR v0 on direct CSV+NPY data")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "aethersign_s2g_v0.yaml")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    config = prepare_runtime_config(
        config_path=args.config,
        device=args.device,
        batch_size=args.batch_size,
        epochs=args.epochs,
        num_workers=args.num_workers,
    )
    set_seed(args.seed)

    output_dir = Path(config["training"]["model_dir"])
    ensure_dir(output_dir)
    (output_dir / "resolved_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    tokenizer = build_tokenizer(config)
    _, train_loader = build_dataloader(config, tokenizer, split="train")
    _, dev_loader = build_dataloader(config, tokenizer, split="dev", shuffle=False)
    _, test_loader = build_dataloader(config, tokenizer, split="test", shuffle=False)

    model, device = build_model(config)
    optimizer, scheduler, scheduler_step_at = build_training_components(config, model)
    grad_clipper = build_gradient_clipper(config["training"]["optimization"])

    start_epoch = 0
    best_wer = float("inf")
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer=optimizer, scheduler=scheduler, strict=True)
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        if checkpoint.get("best_metric") is not None:
            best_wer = float(checkpoint["best_metric"])

    print(f"Device: {device}")
    print(f"Parameters: {summarize_model(model):.3f}M")
    print(f"Train batches: {len(train_loader)}  Dev batches: {len(dev_loader)}  Test batches: {len(test_loader)}")

    for epoch in range(start_epoch, config["training"]["epochs"]):
        model.train()
        running_loss = 0.0
        running_steps = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            output["total_loss"].backward()
            if grad_clipper is not None:
                grad_clipper(model.parameters())
            optimizer.step()
            running_loss += output["total_loss"].item()
            running_steps += 1

        if scheduler is not None and scheduler_step_at == "epoch":
            scheduler.step()

        train_loss = running_loss / max(running_steps, 1)
        dev_pred_path = output_dir / f"dev_predictions_epoch_{epoch:03d}.jsonl"
        dev_summary, _ = evaluate_loader(
            model=model,
            tokenizer=tokenizer,
            data_loader=dev_loader,
            config=config,
            beam_size=config["training"]["validation"]["recognition"]["beam_size"],
            prediction_head=config["testing"]["recognition"].get("prediction_head", "ensemble_last"),
            output_jsonl=dev_pred_path,
        )

        save_checkpoint(
            output_dir / "checkpoint_last.pth",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_metric=best_wer,
        )

        current_wer = dev_summary["best_wer"] if dev_summary["best_wer"] is not None else float("inf")
        if current_wer <= best_wer:
            best_wer = current_wer
            save_checkpoint(
                output_dir / "checkpoint_best.pth",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_metric=best_wer,
            )

        log_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "dev_loss": dev_summary["loss"],
            "dev_prediction_head": dev_summary["prediction_head"],
            "dev_prediction_head_wer": dev_summary["prediction_head_wer"],
            "dev_best_wer": dev_summary["best_wer"],
            "best_checkpoint_wer": best_wer,
            "lr": optimizer.param_groups[0]["lr"],
        }
        dump_metrics_jsonl(output_dir / "metrics.jsonl", log_row)
        print(json.dumps(log_row, ensure_ascii=False))

    best_checkpoint_path = output_dir / "checkpoint_best.pth"
    if best_checkpoint_path.exists():
        load_checkpoint(best_checkpoint_path, model, strict=True)
        test_summary, _ = evaluate_loader(
            model=model,
            tokenizer=tokenizer,
            data_loader=test_loader,
            config=config,
            beam_size=config["testing"]["recognition"]["beam_size"],
            prediction_head=config["testing"]["recognition"].get("prediction_head", "ensemble_last"),
            output_jsonl=output_dir / "test_predictions_best.jsonl",
        )
        (output_dir / "test_summary_best.json").write_text(
            json.dumps(test_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print("Best checkpoint test summary:")
        print(json.dumps(test_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
