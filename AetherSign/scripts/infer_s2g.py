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
    evaluate_loader,
    load_checkpoint,
    prepare_runtime_config,
    run_single_npy_inference,
)
from mska_v0.utils import ensure_dir  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Infer AetherSign MSKA-SLR v0 from split files or a single npy")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "aethersign_s2g_v0.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--split", type=str, choices=["dev", "test"], default=None)
    parser.add_argument("--npy", type=Path, default=None)
    parser.add_argument("--gloss", type=str, default="")
    parser.add_argument("--beam-size", type=int, default=None)
    parser.add_argument("--prediction-head", type=str, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if bool(args.split) == bool(args.npy):
        raise ValueError("Specify exactly one of --split or --npy")

    config = prepare_runtime_config(config_path=args.config, device=args.device)
    tokenizer = build_tokenizer(config)
    model, _ = build_model(config)
    load_checkpoint(args.checkpoint, model, strict=True)

    beam_size = args.beam_size
    if beam_size is None:
        beam_size = config["testing"]["recognition"]["beam_size"]
    prediction_head = args.prediction_head or config["testing"]["recognition"].get("prediction_head", "ensemble_last")

    if args.split:
        _, data_loader = build_dataloader(config, tokenizer, split=args.split, shuffle=False)
        output_dir = args.output or (Path(config["training"]["model_dir"]) / "inference")
        ensure_dir(output_dir)
        summary, rows = evaluate_loader(
            model=model,
            tokenizer=tokenizer,
            data_loader=data_loader,
            config=config,
            beam_size=beam_size,
            prediction_head=prediction_head,
            output_jsonl=output_dir / f"{args.split}_predictions.jsonl",
        )
        (output_dir / f"{args.split}_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"Saved {len(rows)} predictions to {output_dir}")
    else:
        result = run_single_npy_inference(
            model=model,
            tokenizer=tokenizer,
            config=config,
            npy_path=args.npy,
            beam_size=beam_size,
            prediction_head=prediction_head,
            gloss=args.gloss,
        )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
