#!/usr/bin/env python3
"""Smoke the LTX-2.5 video slider on the CPU dummy and print the live card.

Never downloads ``Lightricks/LTX-2.5-Diffusers``. Never trains on GPU.
Music 3 defaults are untouched.

Live train card
---------------
model_id              Lightricks/LTX-2.5-Diffusers
transformer           transformer/ (distilled; exclude transformer_full/)
text encoder          LTX Gemma 4 12B on CPU
LoRA                  LTX2Attention attn1+attn2 to_q/to_k/to_v/to_out.0
lora_up               N(0, 0.02)
sample                49 frames, 544x960, conv VAE
sigmas                DISTILLED_SIGMA_VALUES (do not pass num_inference_steps)
guidance / STG / mod  1.0 / 0 / 0
prompt enhancer       OFF
infer                 neu caption at scale 1
GPU                   A100 80GB community ~$1.19/hr. Not 4090. Not B300.

Dummy / CI:

python scripts/smoke_ltx25_slider.py
PYTHONPATH=. python conceptmod/textsliders/train_lora_ltx25.py --dummy --steps 2
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from conceptmod.textsliders.train_lora_ltx25 import main as train_main


def main(argv: list[str] | None = None) -> dict:
    del argv
    print("=== LTX-2.5 dummy smoke (no Hub, no GPU) ===")
    sidecar = train_main(
        [
            "--dummy",
            "--steps",
            "2",
            "--name",
            "smile-ltx25-smoke",
            "--prompts_file",
            str(_REPO_ROOT / "conceptmod/textsliders/data/prompts-ltx25-smile.yaml"),
            "--save_dir",
            str(_REPO_ROOT / "models/ltx25-smoke"),
            "--sample_scales",
            "0,0.5,1",
            "--seed",
            "0",
        ]
    )
    print(json.dumps(sidecar, indent=2, default=str))
    return sidecar


if __name__ == "__main__":
    main()
