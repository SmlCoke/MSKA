from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Build gloss2ids.pkl for AetherSign MSKA-SLR v0")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "aethersign_s2g_v0.yaml")
    parser.add_argument("--gloss-map", type=Path, default=None, help="Override gloss_map.txt path")
    parser.add_argument("--output", type=Path, default=None, help="Override gloss2ids.pkl output path")
    return parser.parse_args()


def resolve_path(path_str: str, base_dir: Path) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (base_dir / path).resolve()


def build_vocab(gloss_map_path: Path, output_path: Path):
    special_tokens = {"<si>": 0, "<unk>": 1, "<pad>": 2, "</s>": 3}
    vocab = dict(special_tokens)
    next_id = len(vocab)

    with gloss_map_path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            gloss = line.strip()
            if not gloss or gloss in vocab:
                continue
            vocab[gloss] = next_id
            next_id += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(vocab, handle)
    return vocab


def main():
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config_dir = args.config.parent.resolve()

    default_gloss_map = resolve_path(
        config["data"].get("gloss_map_path", config["data"].get("label_dir", "../dataset/label") + "/gloss_map.txt"),
        config_dir,
    )
    default_output = resolve_path(config["gloss"]["gloss2id_file"], config_dir)

    gloss_map_path = args.gloss_map.resolve() if args.gloss_map else default_gloss_map
    output_path = args.output.resolve() if args.output else default_output
    vocab = build_vocab(gloss_map_path=gloss_map_path, output_path=output_path)

    print(f"Built gloss vocabulary with {len(vocab)} entries")
    print(f"gloss_map: {gloss_map_path}")
    print(f"output: {output_path}")
    print(f"special tokens: {{'<si>': {vocab['<si>']}, '<unk>': {vocab['<unk>']}, '<pad>': {vocab['<pad>']}, '</s>': {vocab['</s>']}}}")


if __name__ == "__main__":
    main()
