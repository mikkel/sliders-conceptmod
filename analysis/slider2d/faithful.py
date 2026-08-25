"""Can v6 faithful (raw-pole MSE) be fixed on the leaky 2-D field?

Faithful means the teacher stays ``h+`` / ``h−``. Hold-ê does not replace
those poles, but on this field ``ê`` already lives in ``h± − h0``, so the
hold fights the teacher.

LM scores a single embed at ``t = 0.5``. Slider ⊥ ê, and the even mode of
the ungated pair is parallel to ê, so the ê components close:

    student_ê(±1) = teacher_ê(±1) / (1 + λ)

Leftover leak is therefore teacher leak / (1+λ). Pair-odd already dropped
the even ê, so the same λ leaves a much smaller leak. Endreg / planreg /
semantic-KL are AR-only and are not on this field.

CPU only. No Hub, no GPU, no Music 3 weights.
"""

from __future__ import annotations

from typing import Iterable

import torch

from analysis.slider2d.energy import (
    EnergyLiveField2D,
    _score_energy_residual,
    energy_all_right,
    energy_pairs,
    energy_verdicts,
)
from analysis.slider2d.field import E_ATTR, E_SLIDER, Field2D, cosine
from analysis.slider2d.mismatch import (
    MismatchField2D,
    mismatch_all_right,
    mismatch_pairs,
    mismatch_verdicts,
    score_against_odd,
)
from analysis.slider2d.train import (
    axis_verdicts,
    music3_pairs,
    score_residual,
    train_lm,
)
from conceptmod.textsliders.slider_targets import leftover_bipolar


# Same λ grid as the hold-ê proof, plus 32 (first value that can push
# faithful leftover leak under ~0.05 on this pair).
HOLD_LAMBDAS = (0.0, 1.0, 4.0, 8.0, 16.0, 32.0, 64.0)

# Pole-copy gates. The leaky teacher itself has leak ~1.69, so a student
# that is still close to the poles cannot also be leak-right.
POLE_COS_MIN = 0.90
POLE_REL_ERR_MAX = 0.20


def hold_e_shrink(teacher_along: float, hold_weight: float) -> float:
    """Equilibrium along ê for MSE-to-teacher + λ · ((h−h0)·ê)².

    ``F.mse_loss`` on a 2-vector and ``lm_axis_hold`` share a ½ factor, so
    the ê stationarity condition is ``(s − t) + λ s = 0``.
    """
    return float(teacher_along) / (1.0 + float(hold_weight))


def leak_teacher(field: Field2D | None = None, t: float = 0.5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    field = field or Field2D()
    return field.embed("energetic", t), field.embed("calm", t), field.embed("song", t)


def teacher_geometry(
    pos: torch.Tensor,
    neg: torch.Tensor,
    neu: torch.Tensor,
    *,
    leak_dir: torch.Tensor | None = None,
) -> dict[str, float]:
    """Even / odd split of the raw poles, and how much ê they already contain."""
    leak_dir = E_ATTR if leak_dir is None else leak_dir
    unit = leak_dir.flatten() / leak_dir.flatten().norm().clamp_min(1e-8)
    slider = E_SLIDER.flatten()
    d_plus = (pos - neu).flatten()
    d_minus = (neg - neu).flatten()
    even = 0.5 * (d_plus + d_minus)
    odd = 0.5 * (d_plus - d_minus)
    e_plus = float(d_plus @ unit)
    e_minus = float(d_minus @ unit)
    slider_plus = float(d_plus @ slider)
    return {
        "teacher_e_plus": e_plus,
        "teacher_e_minus": e_minus,
        "teacher_slider_plus": slider_plus,
        "teacher_leak": e_plus / (abs(slider_plus) + 1e-8),
        "teacher_even_e": float(even @ unit),
        "teacher_odd_e": float(odd @ unit),
        "teacher_even_norm": float(even.norm()),
        "teacher_odd_norm": float(odd.norm()),
        "even_cos_e": cosine(even, unit),
        "even_is_parallel_e": abs(cosine(even, unit)) >= 0.99,
        "r_raw": cosine(d_plus, d_minus),
    }


def score_pole_fit(
    residual,
    pos: torch.Tensor,
    neg: torch.Tensor,
    neu: torch.Tensor,
    *,
    leak_dir: torch.Tensor | None = None,
) -> dict[str, float]:
    """How much of the raw poles the learned residual actually copied."""
    leak_dir = E_ATTR if leak_dir is None else leak_dir
    unit = leak_dir.flatten() / leak_dir.flatten().norm().clamp_min(1e-8)
    d_plus = residual.delta(1.0).flatten()
    d_minus = residual.delta(-1.0).flatten()
    t_plus = (pos - neu).flatten()
    t_minus = (neg - neu).flatten()
    err_plus = d_plus - t_plus
    e_plus = float(d_plus @ unit)
    t_e_plus = float(t_plus @ unit)
    pole_rel = float(err_plus.norm() / t_plus.norm().clamp_min(1e-8))
    if abs(t_e_plus) < 1e-6:
        copied = 1.0 if abs(e_plus) < 1e-6 else 0.0
    else:
        copied = e_plus / t_e_plus
    return {
        "pole_cos_plus": cosine(d_plus, t_plus),
        "pole_cos_minus": cosine(d_minus, t_minus),
        "pole_rel_err": pole_rel,
        "pole_err_e": abs(e_plus - t_e_plus),
        "student_e_plus": e_plus,
        "student_e_minus": float(d_minus @ unit),
        "e_copied_frac": copied,
        "faithful_fit": cosine(d_plus, t_plus) >= POLE_COS_MIN and pole_rel <= POLE_REL_ERR_MAX,
    }


def _pack(name: str, residual, pos, neg, neu, *, teacher_faithful: bool) -> dict:
    metrics = score_residual(residual)
    metrics.update(score_pole_fit(residual, pos, neg, neu))
    leftover = leftover_bipolar(residual.delta(1.0), residual.delta(-1.0))
    metrics.update(leftover)
    axis = axis_verdicts(metrics)
    leak_ok = axis["leak"] == "right"
    slider_ok = axis["slider"] == "right"
    collapse_ok = axis["collapse"] == "right"
    gates = leak_ok and slider_ok and collapse_ok
    return {
        "name": name,
        "teacher_faithful": teacher_faithful,
        "axis": axis,
        "gates": gates,
        "wins_while_faithful": bool(teacher_faithful and gates and metrics["faithful_fit"]),
        **metrics,
    }


def score_leak_lm(
    name: str,
    *,
    target_mode: str,
    hold_weight: float = 0.0,
    with_attrs: bool = False,
    leakage_floor: float | None = None,
    anchor_weight: float = 0.0,
    project_odd: bool = False,
    leak_dir: torch.Tensor | None = None,
    subtract_dir: torch.Tensor | None = None,
    common_beta: float = 0.0,
    steps: int = 200,
    seed: int = 0,
) -> dict:
    field = Field2D()
    pairs = music3_pairs(with_attrs)
    used_e = None if with_attrs else (leak_dir if leak_dir is not None else (E_ATTR if hold_weight > 0.0 else None))
    residual = train_lm(
        field,
        pairs,
        target_mode=target_mode,
        hold_weight=hold_weight,
        leak_dir=used_e,
        subtract_dir=subtract_dir,
        project_odd=project_odd,
        leakage_floor=leakage_floor,
        anchor_weight=anchor_weight,
        common_beta=common_beta,
        steps=steps,
        seed=seed,
    )
    pos, neg, neu = leak_teacher(field)
    if with_attrs:
        # Pinned captions: score against the male pair (identical residual to female).
        pos = field.embed("male energetic", 0.5)
        neg = field.embed("male calm", 0.5)
        neu = field.embed("male song", 0.5)
    return _pack(name, residual, pos, neg, neu, teacher_faithful=(target_mode == "faithful"))


def leak_field_table(*, steps: int = 200, seed: int = 0) -> list[dict]:
    """Faithful knobs plus the non-faithful comparables on energetic×gender."""
    rows = []
    for lam in HOLD_LAMBDAS:
        suffix = "raw" if lam == 0.0 else f"hold_l{int(lam)}"
        rows.append(
            score_leak_lm(
                f"lm_faithful_{suffix}",
                target_mode="faithful",
                hold_weight=lam,
                leak_dir=E_ATTR if lam > 0.0 else None,
                steps=steps,
                seed=seed,
            )
        )
    rows.append(
        score_leak_lm(
            "lm_faithful_hub",
            target_mode="faithful",
            leakage_floor=-0.9,
            anchor_weight=0.3,
            steps=steps,
            seed=seed,
        )
    )
    rows.append(
        score_leak_lm(
            "lm_faithful_attrs",
            target_mode="faithful",
            with_attrs=True,
            steps=steps,
            seed=seed,
        )
    )
    rows.append(score_leak_lm("lm_symmetric", target_mode="symmetric", steps=steps, seed=seed))
    rows.append(
        score_leak_lm(
            "lm_v9_hub",
            target_mode="symmetric",
            leakage_floor=-0.9,
            anchor_weight=0.3,
            steps=steps,
            seed=seed,
        )
    )
    rows.append(
        score_leak_lm(
            "lm_v9_project",
            target_mode="symmetric",
            project_odd=True,
            hold_weight=1.0,
            steps=steps,
            seed=seed,
        )
    )
    rows.append(
        score_leak_lm(
            "lm_v9",
            target_mode="symmetric",
            hold_weight=8.0,
            leak_dir=E_ATTR,
            steps=steps,
            seed=seed,
        )
    )
    return rows


def pair_odd_hold_sweep(*, steps: int = 200, seed: int = 0) -> list[dict]:
    """Same λ grid with pair-odd teacher, for the 1/(1+λ) overlay."""
    return [
        score_leak_lm(
            f"lm_odd_hold_l{int(lam)}" if lam > 0.0 else "lm_odd_raw",
            target_mode="symmetric",
            hold_weight=lam,
            leak_dir=E_ATTR if lam > 0.0 else None,
            steps=steps,
            seed=seed,
        )
        for lam in HOLD_LAMBDAS
    ]


def score_energy_faithful(*, hold_weight: float = 0.0, steps: int = 200, seed: int = 0) -> dict:
    field = EnergyLiveField2D()
    residual = train_lm(
        field,
        energy_pairs(field),
        target_mode="faithful",
        hold_weight=hold_weight,
        leak_dir=field.unused if hold_weight > 0.0 else None,
        steps=steps,
        seed=seed,
    )
    metrics = _score_energy_residual(residual, field, decisions=[False] * len(field.aligns))
    metrics.update(leftover_bipolar(residual.delta(1.0), residual.delta(-1.0)))
    pos = field.embed(field.pole_names[0][0])
    neg = field.embed(field.pole_names[0][1])
    neu = field.embed("song")
    metrics.update(score_pole_fit(residual, pos, neg, neu, leak_dir=field.unused))
    metrics["name"] = f"energy_faithful_l{int(hold_weight)}"
    metrics["axis"] = energy_verdicts(metrics)
    metrics["pass"] = energy_all_right(metrics)
    return metrics


def score_energy_odd(*, hold_weight: float = 0.0, steps: int = 200, seed: int = 0) -> dict:
    field = EnergyLiveField2D()
    residual = train_lm(
        field,
        energy_pairs(field),
        target_mode="symmetric",
        hold_weight=hold_weight,
        leak_dir=field.unused if hold_weight > 0.0 else None,
        steps=steps,
        seed=seed,
    )
    metrics = _score_energy_residual(residual, field, decisions=[False] * len(field.aligns))
    metrics.update(leftover_bipolar(residual.delta(1.0), residual.delta(-1.0)))
    metrics["name"] = f"energy_odd_l{int(hold_weight)}"
    metrics["axis"] = energy_verdicts(metrics)
    metrics["pass"] = energy_all_right(metrics)
    return metrics


def score_mismatch_faithful(*, steps: int = 200, seed: int = 0) -> dict:
    field = MismatchField2D()
    residual = train_lm(
        field,
        mismatch_pairs(),
        target_mode="faithful",
        steps=steps,
        seed=seed,
    )
    metrics = score_against_odd(residual, field.odd(), junk=field.junk)
    metrics.update(leftover_bipolar(residual.delta(1.0), residual.delta(-1.0)))
    pos, neg, neu = field.rich_pair()
    metrics.update(score_pole_fit(residual, pos, neg, neu, leak_dir=field.junk))
    metrics["name"] = "mismatch_faithful"
    metrics["axis"] = mismatch_verdicts(metrics)
    metrics["pass"] = mismatch_all_right(metrics)
    return metrics


def floatable(row: dict) -> dict:
    """JSON-safe subset (drop tensors / long lists)."""
    skip = {"delta_plus", "delta_minus", "decisions", "row_aligns"}
    out = {}
    for key, value in row.items():
        if key in skip:
            continue
        if isinstance(value, dict):
            out[key] = {k: (float(v) if isinstance(v, (int, float, bool)) else v) for k, v in value.items()}
        elif isinstance(value, (int, float, bool, str)):
            out[key] = value if isinstance(value, (bool, str)) else float(value)
        elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            continue
        else:
            out[key] = value
    return out
