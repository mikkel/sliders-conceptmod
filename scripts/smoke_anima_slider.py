#!/usr/bin/env python3
"""Smoke the Anima image slider on the CPU fake and print the live train card.

Never downloads ``circlestone-labs/Anima-Base-v1.0-Diffusers``. Never trains
on GPU. Music 3 defaults are untouched.

Live train card
---------------
model_id        circlestone-labs/Anima-Base-v1.0-Diffusers
arch            2B Cosmos-Predict2 DiT, Qwen3+T5, Qwen-Image VAE
lora            rank 16 on attn to_q / to_k / to_v / to_out.0
                do not train text_conditioner
frozen ref      base transformer with adapter disabled
resolution      768
sample_steps    40
cfg             4
lr              1e-4
sample          in-process PEFT pipe(prompt=...) at 0 / 0.25 / 0.5 / 1.0
device          cuda:0

Live command (local weights, HF_HUB_OFFLINE=1):

HF_HUB_OFFLINE=1 python conceptmod/textsliders/train_lora_anima.py \\
  --name smile-anima \\
  --prompts_file conceptmod/textsliders/data/prompts-anima.yaml \\
  --model_id circlestone-labs/Anima-Base-v1.0-Diffusers \\
  --rank 16 --resolution 768 --sample_steps 40 --cfg 4 \\
  --lr 1e-4 --device cuda:0 --save_dir models/smile-anima

Dummy / CI:

python scripts/smoke_anima_slider.py
python conceptmod/textsliders/train_lora_anima.py --dummy --steps 8 --device cpu
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from conceptmod.textsliders.anima_slider import (
    live_train_card,
    live_train_command,
    stock_teacher_smoke_captions,
)
from conceptmod.textsliders.train_lora_anima import main as train_main


def main(argv: list[str] | None = None) -> dict:
    del argv
    card = live_train_card()
    print("=== Anima live train card ===")
    print(json.dumps(card, indent=2))
    print()
    print("=== Live train command ===")
    print(live_train_command())
    print()
    print("=== Stock teacher smoke (no Hub; live weights required to render) ===")
    print(json.dumps(stock_teacher_smoke_captions(), indent=2))
    print()
    print("=== Dummy smoke (no Hub, no GPU) ===")
    sidecar = train_main(
        [
            "--dummy",
            "--steps",
            "8",
            "--device",
            "cpu",
            "--name",
            "smile-anima-smoke",
            "--prompts_file",
            str(_REPO_ROOT / "conceptmod/textsliders/data/prompts-anima.yaml"),
            "--save_dir",
            str(_REPO_ROOT / "models/anima-smoke"),
            "--rank",
            "16",
            "--resolution",
            "768",
            "--sample_steps",
            "40",
            "--cfg",
            "4",
        ]
    )
    return sidecar


if __name__ == "__main__":
    main()
