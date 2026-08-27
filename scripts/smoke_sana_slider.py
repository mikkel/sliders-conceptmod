#!/usr/bin/env python3
"""Smoke the Sana 0.6B image slider on the CPU dummy and print the live card.

Never downloads ``Efficient-Large-Model/Sana_600M_512px_diffusers``.
Never trains on GPU. Music 3 defaults are untouched.

Live train card
---------------
model_id        Efficient-Large-Model/Sana_600M_512px_diffusers
arch            0.6B flow-matching linear DiT, Gemma-2, DC-AE
train           xattn (conceptmod 0.6B default; --lora RANK optional)
resolution      512
sample_steps    20
cfg             4.5
control         a bowl of fruit on a table
device          cuda:0

Live command (Modal / RunPod — first GPU look):

CUDA_VISIBLE_DEVICES=0 python conceptmod/textsliders/train_lora_sana.py \\
  --name age-sana \\
  --prompts_file conceptmod/textsliders/data/prompts-sana.yaml \\
  --model_id Efficient-Large-Model/Sana_600M_512px_diffusers \\
  --train_method xattn \\
  --resolution 512 \\
  --sample_steps 20 --sample_guidance 4.5 \\
  --control_prompt "a bowl of fruit on a table" \\
  --steps 500 --lr 2e-5 --seed 7 --device 0 \\
  --save_dir models/sana-slider

Dummy / CI:

python scripts/smoke_sana_slider.py
python conceptmod/textsliders/train_lora_sana.py --dummy --steps 2 --device 0
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from conceptmod.textsliders.slider_targets import (
    sana_live_train_card,
    sana_live_train_command,
)
from conceptmod.textsliders.train_lora_sana import main as train_main


def main(argv: list[str] | None = None) -> dict:
    del argv
    card = sana_live_train_card()
    print("=== Sana live train card ===")
    print(json.dumps(card, indent=2))
    print()
    print("=== Live train command ===")
    print(sana_live_train_command())
    print()
    print("=== Dummy smoke (no Hub, no GPU) ===")
    weights = train_main(
        [
            "--dummy",
            "--steps",
            "2",
            "--name",
            "age-sana-smoke",
            "--prompts_file",
            str(_REPO_ROOT / "conceptmod/textsliders/data/prompts-sana.yaml"),
            "--save_dir",
            str(_REPO_ROOT / "models/sana-smoke"),
            "--train_method",
            "xattn",
            "--resolution",
            "512",
            "--sample_steps",
            "20",
            "--sample_guidance",
            "4.5",
            "--control_prompt",
            "a bowl of fruit on a table",
        ]
    )
    sidecar = Path(str(weights).replace(".safetensors", ".json"))
    meta = json.loads(sidecar.read_text())
    print(json.dumps({"weights": str(weights), "sidecar": meta}, indent=2, default=str))
    return meta


if __name__ == "__main__":
    main()
