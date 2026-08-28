#!/usr/bin/env python3
"""Dummy (and optional live) smoke for the Anima same-crop smile teacher.

Caption-only plus from the same z_T still jumps crop: stock Anima goes
full-body → close-up when the caption goes closed-mouth → teeth. This
script proves the invert/img2img teacher path runs and that dummy neu/plus
share crop (channel 2) while expression (channel 1) still moves.

Never a 500-step train. Never rents a GPU. CI uses ``--dummy`` (default).
Live ``--live`` needs local Base weights; it only builds the two teachers
(K-step Euler, no LoRA fit).

    PYTHONPATH=. python scripts/smoke_anima_same_crop_teacher.py
    PYTHONPATH=. python scripts/smoke_anima_same_crop_teacher.py --dummy

4090 / L40S short train smoke (not this script; not 500 steps):

    HF_HUB_OFFLINE=1 python conceptmod/textsliders/train_lora_anima.py \\
      --name smile-anima-same-crop-smoke \\
      --lora_targets conditioner --rank 16 --resolution 512 \\
      --lm_target same_crop --teacher same_crop --teacher_strength 0.5 \\
      --traj_steps 4 --steps 8 --sample_steps 8 --sample_every 0 \\
      --device cuda:0 --save_dir models/smile-anima-same-crop-smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from conceptmod.textsliders.anima_fake import FakeAnimaBackend
from conceptmod.textsliders.anima_slider import (
    DEFAULT_TEACHER_STRENGTH,
    DEFAULT_TRAJ_STEPS,
    WOMAN_NEU,
    WOMAN_PLUS,
    anima_teacher_crop_gap,
    anima_teacher_expr_gap,
    anima_teacher_pair,
    same_crop_smoke_command,
)
from conceptmod.textsliders.train_lora_anima import main as train_main


def dummy_teacher_report(
    *,
    seed: int = 0,
    traj_steps: int = DEFAULT_TRAJ_STEPS,
    strength: float = DEFAULT_TEACHER_STRENGTH,
) -> dict:
    backend = FakeAnimaBackend(device="cpu", rank=4, seed=seed, lora_targets="dit")
    g = torch.Generator().manual_seed(seed + 11)
    z = torch.randn((1, *backend.latent_shape), generator=g)
    caption = anima_teacher_pair(
        backend, WOMAN_NEU, WOMAN_PLUS, z, num_steps=traj_steps, teacher="caption"
    )
    locked = anima_teacher_pair(
        backend,
        WOMAN_NEU,
        WOMAN_PLUS,
        z,
        num_steps=traj_steps,
        teacher="same_crop",
        strength=strength,
    )
    cap_crop = anima_teacher_crop_gap(caption.x_neu, caption.x_plus)
    lock_crop = anima_teacher_crop_gap(locked.x_neu, locked.x_plus)
    lock_expr = anima_teacher_expr_gap(locked.x_neu, locked.x_plus)
    ok = (
        locked.shared_crop
        and lock_crop < cap_crop
        and lock_expr > 1e-4
        and locked.z_mid is not None
    )
    return {
        "dummy": True,
        "verdict": "PASS" if ok else "FAIL",
        "caption_crop_gap": cap_crop,
        "same_crop_crop_gap": lock_crop,
        "same_crop_expr_gap": lock_expr,
        "shared_crop": locked.shared_crop,
        "start_index": locked.start_index,
        "start_sigma": locked.start_sigma,
        "teacher_strength": strength,
        "traj_steps": traj_steps,
        "note": (
            "caption plus from z_T writes crop at high σ; same_crop invert "
            "keeps neu crop and still moves expression"
        ),
    }


def dummy_train_smoke(tmp: Path) -> dict:
    return train_main(
        [
            "--dummy",
            "--steps",
            "6",
            "--device",
            "cpu",
            "--name",
            "smile-anima-same-crop-dummy",
            "--prompts_file",
            str(_REPO_ROOT / "conceptmod/textsliders/data/prompts-anima.yaml"),
            "--save_dir",
            str(tmp),
            "--rank",
            "4",
            "--lm_target",
            "same_crop",
            "--teacher",
            "same_crop",
            "--teacher_strength",
            "0.5",
            "--traj_steps",
            "4",
            "--sample_every",
            "0",
        ]
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dummy", action="store_true", default=True)
    parser.add_argument(
        "--live",
        action="store_true",
        help="refuse: this smoke is dummy-only in CI; use the 8-step GPU command",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    if args.live:
        raise SystemExit(
            "this script does not load Hub weights or rent a GPU. "
            "Use the 8-step command in docs/anima-slider.md:\n"
            + same_crop_smoke_command()
        )
    report = dummy_teacher_report()
    print("=== same-crop teacher (dummy crop constraints) ===")
    print(json.dumps(report, indent=2))
    if report["verdict"] != "PASS":
        raise SystemExit("same-crop dummy teacher failed crop/expression checks")
    print()
    print("=== dummy train --lm_target same_crop ===")
    sidecar = dummy_train_smoke(_REPO_ROOT / "models" / "anima-same-crop-smoke")
    print()
    print("=== recommended 4090/L40S smoke (not run here) ===")
    print(same_crop_smoke_command())
    return {"teacher": report, "train": sidecar}


if __name__ == "__main__":
    main()
