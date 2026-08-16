#!/usr/bin/env python3
"""Calibrate a transformer slider so user scale ±1 is one trained concept.

Loads the flow transformer and pos/neg/neu conditions (from the weights sidecar
prompts or a cache dir). Measures ||vel_pos-vel_neg|| and the trained target
||guidance*(vel_pos-vel_neg)||, sweeps LoRA scales on the NEUTRAL condition,
and writes unit_scale into the weights sidecar JSON.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_HF_HOME = "/ml2/music/.cache/huggingface"
os.environ["HF_HOME"] = _HF_HOME
os.environ["HUGGINGFACE_HUB_CACHE"] = f"{_HF_HOME}/hub"
os.environ["HF_HUB_CACHE"] = f"{_HF_HOME}/hub"
os.environ["TRANSFORMERS_CACHE"] = f"{_HF_HOME}/hub"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _early_visible_device(argv: list[str]) -> str:
    for i, arg in enumerate(argv):
        if arg == "--device" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--device="):
            return arg.split("=", 1)[1]
    return os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]


os.environ["CUDA_VISIBLE_DEVICES"] = _early_visible_device(sys.argv[1:])

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from conceptmod.textsliders.calibrate_scale import (  # noqa: E402
    DEFAULT_MODEL_DIR,
    DEFAULT_SCALES,
    calibrate_weights,
    guess_cache_dir,
    read_sidecar,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="path to last.safetensors")
    parser.add_argument(
        "--cache_dir",
        default=None,
        help="condition cache (pos/neg/neu). guessed from sidecar name if omitted",
    )
    parser.add_argument("--model_dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument(
        "--scales",
        default=",".join(str(scale) for scale in DEFAULT_SCALES),
        help="positive LoRA multipliers to sweep",
    )
    parser.add_argument("--guidance", type=float, default=None, help="override sidecar guidance")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--no_write",
        action="store_true",
        help="print the table only; do not update the sidecar JSON",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    import torch

    args = parse_args(argv)
    weights = Path(args.weights)
    meta = read_sidecar(weights)
    cache_dir = Path(args.cache_dir) if args.cache_dir else guess_cache_dir(weights, meta)
    if cache_dir is None:
        raise FileNotFoundError(f"pass --cache_dir; could not guess cache for {weights}")
    scales = [float(part) for part in args.scales.split(",") if part.strip()]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    calibrate_weights(
        weights,
        cache_dir=cache_dir,
        model_dir=Path(args.model_dir),
        device=device,
        scales=scales,
        seed=int(args.seed),
        write=not args.no_write,
        guidance=args.guidance,
    )


if __name__ == "__main__":
    main()
