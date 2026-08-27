#!/usr/bin/env python3
"""Smoke the Anima image slider on the CPU fake and print the live train card.

Never downloads ``circlestone-labs/Anima-Base-v1.0-Diffusers``. Never trains
on GPU. Music 3 defaults are untouched. Anima-Turbo v1.1 is preview-only
(``scripts/convert_anima_turbo_diffusers.py``); this smoke stays on Base.

Live train card
---------------
model_id        circlestone-labs/Anima-Base-v1.0-Diffusers
arch            2B Cosmos-Predict2 DiT, Qwen3+T5, Qwen-Image VAE
lora            --lora_targets conditioner (smile default-on)
                AnimaTextConditioner q_proj / k_proj / v_proj / o_proj
                dit = old transformer-only; Qwen3 text_encoder not adapted
frozen ref      base modules with adapters disabled
resolution      768 (4090 smile retrain: 512)
sample_steps    40
cfg             4
lr              1e-4
lm_target       trajectory (K-step FlowMatch Euler; direct / cfg_delta kept)
traj_steps      4
sample_every    100 (end-of-train gate always runs)
sample          in-process PEFT pipe(prompt=...) at 0 / 0.25 / 0.5 / 1.0
                same bare infer/neu captions as train
device          cuda:0

Live command (local weights, HF_HUB_OFFLINE=1):

HF_HUB_OFFLINE=1 python conceptmod/textsliders/train_lora_anima.py \\
  --name smile-anima \\
  --prompts_file conceptmod/textsliders/data/prompts-anima.yaml \\
  --model_id circlestone-labs/Anima-Base-v1.0-Diffusers \\
  --lora_targets conditioner --rank 16 --resolution 768 --sample_steps 40 --cfg 4 \\
  --lr 1e-4 --lm_target trajectory --traj_steps 4 \\
  --teacher_gap_boost 1 --sample_every 100 \\
  --device cuda:0 --save_dir models/smile-anima

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
