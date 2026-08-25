"""Score Hub / project-short-û / hold-ê (and discarded gates) on both live cells."""

from __future__ import annotations

from conceptmod.textsliders.slider_targets import SLIDER_ALIGN_MIN
from analysis.slider2d.energy import EnergyLiveField2D, energy_policy_table, score_energy_policy
from analysis.slider2d.mismatch import score_mismatch_policy


POLICY_ORDER = (
    "hub",
    "always_project_hold",
    "gated_row_0.50",
    "slider_align_0.50",
    "hold_e",
)


def gender_policy_table(*, steps: int = 200, seed: int = 0) -> list[dict]:
    return [
        score_mismatch_policy(
            "hub",
            project_odd=False,
            hold_weight=0.0,
            use_short_u=False,
            leakage_floor=-0.9,
            anchor_weight=0.3,
            steps=steps,
            seed=seed,
        ),
        score_mismatch_policy(
            "always_project_hold",
            project_odd=True,
            hold_weight=1.0,
            use_short_u=True,
            steps=steps,
            seed=seed,
        ),
        score_mismatch_policy(
            "gated_row_0.50",
            project_odd=True,
            hold_weight=1.0,
            use_short_u=True,
            project_align_min=SLIDER_ALIGN_MIN,
            project_align_scope="row",
            steps=steps,
            seed=seed,
        ),
        score_mismatch_policy(
            "slider_align_0.50",
            project_odd=True,
            hold_weight=1.0,
            use_short_u=True,
            project_align_min=SLIDER_ALIGN_MIN,
            project_align_scope="slider",
            steps=steps,
            seed=seed,
        ),
        score_mismatch_policy(
            "hold_e",
            project_odd=False,
            hold_weight=0.0,
            use_short_u=False,
            steps=steps,
            seed=seed,
        ),
    ]


def live_policy_table(*, steps: int = 200, seed: int = 0) -> dict:
    gender = {r["name"]: r for r in gender_policy_table(steps=steps, seed=seed)}
    energy = {r["name"]: r for r in energy_policy_table(steps=steps, seed=seed)}
    return {
        "gender": [gender[name] for name in POLICY_ORDER],
        "energy": [energy[name] for name in POLICY_ORDER],
        "energy_cheat": score_energy_policy(
            "u_is_pole_odd",
            project_odd=True,
            hold_weight=1.0,
            use_declared_u=False,
            use_pole_odd_u=True,
            field=EnergyLiveField2D(aligns=(0.48,)),
            steps=steps,
            seed=seed,
        ),
        "floor": SLIDER_ALIGN_MIN,
    }


def table_row(cell: str, row: dict) -> dict:
    """Compact PR/doc row: leak, ||d+||/||odd||, cos intended, mixed."""
    return {
        "cell": cell,
        "policy": row["name"],
        "leak": float(row["leak_ratio"]),
        "strength": float(row["strength"]),
        "cos_intended": float(row.get("cos_intended", row.get("cos_concept"))),
        "mixed": bool(row.get("mixed", False)),
        "pass": bool(row["pass"]),
        "norm_plus": float(row["norm_plus"]),
        "norm_odd": float(row["norm_odd"]),
        "odd_align": float(row["odd_align"]),
        "same_dir": float(row.get("same_dir", 0.0)),
        "leak_frac": float(row.get("leak_frac", row.get("cos_plus_minus", 0.0))),
    }
