#!/usr/bin/env python3
"""Smoke the LTX-2.5 video slider on the CPU dummy and print the live card.

Never downloads ``Lightricks/LTX-2.5-Diffusers``. Never trains on GPU.
Music 3 defaults are untouched.

Live train card
---------------
model_id              Lightricks/LTX-2.5-Diffusers
recipe                ltx25_uni_embed (default; velocity is opt-in only)
transformer           transformer/ FROZEN (exclude transformer_full/)
LoRA                  video connectors + TE last-N q/k/v/o (not DiT attn1/attn2)
lora_up               N(0, 0.02)
rank / lr / steps     16 / 2e-4 / 700 / te_last_n=4 / seed 7
sample                49 frames, 544x960, conv VAE, scales -1,0,0.5,1 on neu
sigmas                DISTILLED_SIGMA_VALUES (do not pass num_inference_steps)
guidance / STG / mod  1.0 / 0 / 1.0 (pipeline treats >1.0 as on)
prompt enhancer       OFF
infer                 neu caption at every scale
GPU                   dual RTX A6000: TE+connectors cuda:1, DiT cuda:0

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
            "--sample_scales=-1,0,0.5,1",
            "--seed",
            "0",
        ]
    )
    print(json.dumps(sidecar, indent=2, default=str))
    return sidecar


if __name__ == "__main__":
    main()
