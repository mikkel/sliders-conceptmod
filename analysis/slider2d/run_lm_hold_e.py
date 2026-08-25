#!/usr/bin/env python3
"""Prove full-odd teacher + hold-on-ê vs Hub and short-û project."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from analysis.slider2d.energy import EnergyLiveField2D, energy_pairs, energy_verdicts, energy_all_right, _score_energy_residual
from analysis.slider2d.field import E_ATTR, Field2D
from analysis.slider2d.mismatch import (
    MismatchField2D,
    mismatch_pairs,
    score_against_odd,
    mismatch_verdicts,
    mismatch_all_right,
    leak_cell_odd,
    leak_cell_align,
)
from analysis.slider2d.train import music3_pairs, score_residual, train_lm
from conceptmod.textsliders.slider_targets import leftover_bipolar, lm_odd_align


SAME_DIR_BAND = 0.06


def _score_gender(*, leak_dir, hold_weight, project_odd, slider_dir, steps, seed):
    field = MismatchField2D()
    residual = train_lm(
        field,
        mismatch_pairs(),
        symmetric=True,
        target_mode="symmetric",
        project_odd=project_odd,
        hold_weight=hold_weight,
        slider_dir=slider_dir,
        leak_dir=leak_dir,
        steps=steps,
        seed=seed,
    )
    metrics = score_against_odd(residual, field.odd(), junk=field.junk)
    leftover = leftover_bipolar(residual.delta(1.0), residual.delta(-1.0))
    metrics.update(leftover)
    metrics["axis"] = mismatch_verdicts(metrics)
    metrics["pass"] = mismatch_all_right(metrics)
    metrics["same_dir_ok"] = leftover["same_dir"] <= SAME_DIR_BAND
    metrics["odd_align"] = float(lm_odd_align(*field.rich_pair()[:2], field.declared_u))
    return metrics


def _score_energy(*, leak_dir, hold_weight, project_odd, slider_dir, steps, seed):
    field = EnergyLiveField2D()
    residual = train_lm(
        field,
        energy_pairs(field),
        symmetric=True,
        target_mode="symmetric",
        project_odd=project_odd,
        hold_weight=hold_weight,
        slider_dir=slider_dir,
        leak_dir=leak_dir,
        steps=steps,
        seed=seed,
    )
    decisions = [False] * len(field.aligns)
    if project_odd:
        decisions = [True] * len(field.aligns)
    metrics = _score_energy_residual(residual, field, decisions=decisions)
    leftover = leftover_bipolar(residual.delta(1.0), residual.delta(-1.0))
    metrics.update(leftover)
    metrics["axis"] = energy_verdicts(metrics)
    metrics["pass"] = energy_all_right(metrics)
    metrics["same_dir_ok"] = leftover["same_dir"] <= SAME_DIR_BAND
    return metrics


def _score_old_leak(*, leak_dir, hold_weight, project_odd, steps, seed):
    field = Field2D()
    residual = train_lm(
        field,
        music3_pairs(False),
        symmetric=True,
        target_mode="symmetric",
        project_odd=project_odd,
        hold_weight=hold_weight,
        leak_dir=leak_dir,
        steps=steps,
        seed=seed,
    )
    raw = score_residual(residual)
    leftover = leftover_bipolar(residual.delta(1.0), residual.delta(-1.0))
    leak = abs(raw["leak_ratio"])
    return {
        **raw,
        **leftover,
        "odd_align": leak_cell_align(field),
        "pass": raw["cos_slider_plus"] >= 0.90 and leak <= 0.20 and raw["cos_plus_minus"] <= -0.85,
        "same_dir_ok": leftover["same_dir"] <= SAME_DIR_BAND,
        "norm_odd": float(leak_cell_odd(field).norm()),
    }


def score_recipe(name: str, *, steps: int, seed: int, hold_weight: float = 1.0) -> dict:
    gender_field = MismatchField2D()
    energy_field = EnergyLiveField2D()
    if name == "hub":
        g = _score_gender(
            leak_dir=None, hold_weight=0.0, project_odd=False,
            slider_dir=None, steps=steps, seed=seed,
        )
        # Hub uses floor/anchor; pair-odd is the same teacher for leftover.
        e = _score_energy(
            leak_dir=None, hold_weight=0.0, project_odd=False,
            slider_dir=None, steps=steps, seed=seed,
        )
        leak = _score_old_leak(leak_dir=None, hold_weight=0.0, project_odd=False, steps=steps, seed=seed)
    elif name == "project_short_u":
        g = _score_gender(
            leak_dir=None, hold_weight=1.0, project_odd=True,
            slider_dir=gender_field.declared_u, steps=steps, seed=seed,
        )
        e = _score_energy(
            leak_dir=None, hold_weight=1.0, project_odd=True,
            slider_dir=energy_field.declared_u, steps=steps, seed=seed,
        )
        leak = _score_old_leak(leak_dir=None, hold_weight=1.0, project_odd=True, steps=steps, seed=seed)
    elif name.startswith("hold_e"):
        g = _score_gender(
            leak_dir=None, hold_weight=0.0, project_odd=False,
            slider_dir=None, steps=steps, seed=seed,
        )
        e = _score_energy(
            leak_dir=energy_field.unused, hold_weight=hold_weight, project_odd=False,
            slider_dir=None, steps=steps, seed=seed,
        )
        leak = _score_old_leak(
            leak_dir=E_ATTR, hold_weight=hold_weight, project_odd=False, steps=steps, seed=seed,
        )
    else:
        raise ValueError(name)
    return {"name": name, "gender": g, "energy": e, "old_leak_cell": leak, "hold_weight": hold_weight}


def _row(cell: dict) -> str:
    return (
        f"pass={cell['pass']} leak={cell.get('leak_ratio', 0):+.3f} "
        f"str={cell.get('strength', cell.get('strength_on_u', 0)):.3f} "
        f"cos={cell.get('cos_intended', cell.get('cos_concept', cell.get('cos_slider_plus', 0))):+.3f} "
        f"col={cell.get('cos_plus_minus', 0):+.3f} "
        f"same_dir={cell.get('same_dir', 0):.4f} leak_frac={cell.get('leak_frac', 0):+.3f}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    recipes = [
        ("hub", 0.0),
        ("project_short_u", 1.0),
        ("hold_e_l1", 1.0),
        ("hold_e_l4", 4.0),
        ("hold_e_l8", 8.0),
        ("hold_e_l16", 16.0),
        ("hold_e_l64", 64.0),
    ]
    blob = {}
    for name, lam in recipes:
        result = score_recipe(name, steps=args.steps, seed=args.seed, hold_weight=lam)
        blob[name] = {
            "hold_weight": lam,
            "gender_pass": result["gender"]["pass"],
            "energy_pass": result["energy"]["pass"],
            "old_leak_pass": result["old_leak_cell"]["pass"],
            "gender": {k: (float(v) if isinstance(v, (int, float)) else v)
                       for k, v in result["gender"].items() if k != "axis" and not isinstance(v, list)},
            "energy": {k: (float(v) if isinstance(v, (int, float)) else v)
                       for k, v in result["energy"].items() if k != "axis" and not isinstance(v, list)},
            "old_leak": {k: (float(v) if isinstance(v, (int, float)) else v)
                         for k, v in result["old_leak_cell"].items() if k != "axis" and not isinstance(v, list)},
        }
        print(f"== {name} λ={lam} ==")
        print(f"  gender   {_row(result['gender'])}")
        print(f"  energy   {_row(result['energy'])}")
        print(f"  old_leak {_row(result['old_leak_cell'])}")

    out = Path("/tmp/hold_e_proof.json")
    out.write_text(json.dumps(blob, indent=2) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
