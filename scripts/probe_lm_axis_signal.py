#!/usr/bin/env python3
"""Does the AR language model actually encode this axis?

For each prompt YAML, encode the neutral / positive / negative captions through
the Qwen3 language model and measure how far the audio-start hidden state moves.
That hidden state is the plan the flow transformer renders from, so a large
separation means an LM slider has something to grab; a small one means the axis
is decided downstream (mix/tone) and belongs to the transformer.

  sep      ||pos - neg|| / ||neu||   how far apart the two poles sit
  cos      cos(pos-neu, neg-neu)     >0 means both poles shift the same way
                                     (shared caption verbosity, not the axis)
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

import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from conceptmod.textsliders.train_lm_slider_music3 import (  # noqa: E402
    DEFAULT_MODEL,
    _assemble,
    _encode_static,
    _load_rows,
    _tokenize,
)

DEFAULT_AXES = [
    "prompts-gender-v3.yaml",
    "prompts-rapslow-v3.yaml",
    "prompts-triphop-v3.yaml",
    "prompts-energy.yaml",
    "prompts-tempo.yaml",
    "prompts-distortion.yaml",
    "prompts-space.yaml",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default=str(_REPO_ROOT / "conceptmod/textsliders/data"))
    parser.add_argument("--model_dir", default=str(DEFAULT_MODEL))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("axes", nargs="*", default=DEFAULT_AXES)
    args = parser.parse_args()

    device = torch.device(f"cuda:{int(args.device)}" if torch.cuda.is_available() else "cpu")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(Path(args.model_dir) / "tokenizer"), local_files_only=True)
    lm = AutoModelForCausalLM.from_pretrained(
        str(Path(args.model_dir) / "language_model"), torch_dtype=torch.bfloat16, local_files_only=True
    )
    lm.to(device).eval().requires_grad_(False)

    print(f"{'axis':<26} {'rows':>4} {'|pos-neu|':>10} {'|neg-neu|':>10} {'sep':>7} {'cos':>7}  verdict")
    for name in args.axes:
        path = Path(args.data_dir) / name
        if not path.exists():
            print(f"{name:<26} missing")
            continue
        rows, _meta = _load_rows(path)
        seps, coss, pnorms, nnorms = [], [], [], []
        for row in rows:
            lyrics = str(row.get("lyrics") or "")
            toks = {
                key: _tokenize(tokenizer, _assemble(str(row.get(key) or row["target"]), lyrics), device)
                for key in ("neutral", "positive", "negative")
            }
            with torch.no_grad():
                pos = _encode_static(lm, *toks["positive"])
                neg = _encode_static(lm, *toks["negative"])
                neu = _encode_static(lm, *toks["neutral"])
            base = torch.norm(neu).clamp_min(1e-6)
            pnorms.append((torch.norm(pos - neu) / base).item())
            nnorms.append((torch.norm(neg - neu) / base).item())
            seps.append((torch.norm(pos - neg) / base).item())
            coss.append(F.cosine_similarity(pos - neu, neg - neu, dim=-1).mean().item())

        sep = sum(seps) / len(seps)
        cos = sum(coss) / len(coss)
        if sep >= 0.25:
            verdict = "strong LM signal"
        elif sep >= 0.15:
            verdict = "usable LM signal"
        else:
            verdict = "weak - transformer axis"
        print(
            f"{name.replace('prompts-', '').replace('.yaml', ''):<26} {len(rows):>4} "
            f"{sum(pnorms)/len(pnorms):>10.3f} {sum(nnorms)/len(nnorms):>10.3f} "
            f"{sep:>7.3f} {cos:>7.3f}  {verdict}"
        )


if __name__ == "__main__":
    main()
