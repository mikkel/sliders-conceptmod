#!/usr/bin/env python3
"""Dummy smoke for the Anima embed+structure split.

Caption-only plus from the same z_T bundles crop. same_crop invert (#71)
held crop but smile was too weak, and --teacher_strength is per-concept.
embed_struct puts concept in conditioner embeds
(MSE(E_θ(neu), sg E_frozen(plus))) and locks structure to the frozen
neu short traj. No smile-tuned σ. Any attribute yaml.

Never a 500-step train. Never rents a GPU. CI uses --dummy (default).

    PYTHONPATH=. python scripts/smoke_anima_embed_struct.py
    PYTHONPATH=. python scripts/smoke_anima_embed_struct.py --dummy

4090 / L40S short train smoke (not this script; not 500 steps):

    HF_HUB_OFFLINE=1 python conceptmod/textsliders/train_lora_anima.py \\
      --name smile-anima-embed-struct-smoke \\
      --lora_targets conditioner --rank 16 --resolution 512 \\
      --lm_target embed_struct --concept_target caption \\
      --traj_steps 4 --steps 8 --sample_steps 8 --sample_every 0 \\
      --device cuda:0 --save_dir models/smile-anima-embed-struct-smoke
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
    WOMAN_NEU,
    WOMAN_PLUS,
    anima_embed_delta_cosine,
    anima_embed_struct_loss,
    anima_short_trajectory,
    anima_teacher_crop_gap,
    embed_struct_smoke_command,
)
from conceptmod.textsliders.train_lora_anima import main as train_main


def dummy_split_report(*, seed: int = 0, traj_steps: int = 4) -> dict:
    backend = FakeAnimaBackend(
        device="cpu", rank=8, seed=seed, lora_targets="conditioner"
    )
    neu, pos = WOMAN_NEU, WOMAN_PLUS
    g = torch.Generator().manual_seed(seed + 11)
    z = torch.randn((1, *backend.latent_shape), generator=g)

    def _cos() -> float:
        e_s, _ = backend.encode_text(neu)
        with backend.disable_adapter():
            e_n, _ = backend.encode_text(neu)
            e_p, _ = backend.encode_text(pos)
        return float(anima_embed_delta_cosine(e_s, e_n, e_p))

    before = _cos()
    opt = torch.optim.AdamW(backend.trainable_parameters(), lr=5e-2)
    for _ in range(20):
        e_student, _ = backend.encode_text(neu)
        with torch.no_grad(), backend.disable_adapter():
            e_plus, _ = backend.encode_text(pos)
            e_neu, _ = backend.encode_text(neu)
            x_neu = anima_short_trajectory(
                backend, neu, z, num_steps=traj_steps, frozen=True
            )
        x_student = anima_short_trajectory(
            backend, neu, z, num_steps=traj_steps, frozen=False, scale=1.0
        )
        loss = anima_embed_struct_loss(
            e_student,
            e_plus.detach(),
            None,
            e_neu.detach(),
            x_student,
            x_neu,
            embed_identity_weight=0.0,
            identity_weight=0.0,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    after = _cos()
    with torch.no_grad():
        x_s = anima_short_trajectory(
            backend, neu, z, num_steps=traj_steps, frozen=False, scale=1.0
        )
        x_n = anima_short_trajectory(
            backend, neu, z, num_steps=traj_steps, frozen=True
        )
        x_p = anima_short_trajectory(
            backend, pos, z, num_steps=traj_steps, frozen=True
        )
    crop_student = anima_teacher_crop_gap(x_n, x_s)
    crop_plus = anima_teacher_crop_gap(x_n, x_p)
    ok = after > before and crop_student < crop_plus and backend.lora_B_norm() > 0.0
    return {
        "dummy": True,
        "verdict": "PASS" if ok else "FAIL",
        "embed_cos_before": before,
        "embed_cos_after": after,
        "crop_student": crop_student,
        "crop_plus": crop_plus,
        "lora_B_norm": backend.lora_B_norm(),
        "note": (
            "E_θ(neu) moves toward E_frozen(plus); student traj crop "
            "stays nearer neu than caption-plus zoom"
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
            "smile-anima-embed-struct-dummy",
            "--prompts_file",
            str(_REPO_ROOT / "conceptmod/textsliders/data/prompts-anima.yaml"),
            "--save_dir",
            str(tmp),
            "--rank",
            "4",
            "--lm_target",
            "embed_struct",
            "--lora_targets",
            "conditioner",
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
            + embed_struct_smoke_command()
        )
    report = dummy_split_report()
    print("=== embed_struct split (dummy concept vs crop) ===")
    print(json.dumps(report, indent=2))
    if report["verdict"] != "PASS":
        raise SystemExit("embed_struct dummy split failed embed/crop checks")
    print()
    print("=== dummy train --lm_target embed_struct ===")
    sidecar = dummy_train_smoke(_REPO_ROOT / "models" / "anima-embed-struct-smoke")
    print()
    print("=== recommended 4090/L40S smoke (not run here) ===")
    print(embed_struct_smoke_command())
    return {"split": report, "train": sidecar}


if __name__ == "__main__":
    main()
