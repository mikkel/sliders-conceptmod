"""Live Music 3 hold-failure signatures the orthonormal 2-D cells omit.

Existing overlap / rich / faithful cells are orthonormal 2-D. They show
ê·û overlap and ê_⊥û locking slider-cos. They do **not** make trainer
c+ (cos with a) vs slider-cos (cos with û) a test, and they never break
±1 polarity. This module is those missing live bullets on a CPU field.

CPU hidden-space only. No Hub, no GPU, no Music 3 weights. Does not
change the live ``--lm_target v9`` default.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from analysis.slider2d.energy import (
    EnergyLiveField2D,
    _score_energy_residual,
    energy_pairs,
)
from analysis.slider2d.field import cosine
from analysis.slider2d.mismatch import MismatchField2D, mismatch_pairs, score_against_odd
from analysis.slider2d.overlap import live_fit_metrics, score_overlap_policy
from analysis.slider2d.train import Pair, Residual, train_lm
from conceptmod.textsliders.slider_targets import (
    LEAK_HOLD_WEIGHT,
    leftover_bipolar,
    lm_axis_hold,
    lm_hidden_targets,
    lm_hold_dir,
    lm_slider_loss,
    lm_unit,
)


# Live energy-v14 caption-axis (short loud/calm vs leak ê).
LIVE_CAPTION_AXIS = 0.37
LIVE_ENERGY_ALIGN = 0.58
HIGH_D = 32
CONCEPT_SCALE = 1.2
STRENGTH_UNDEF = 0.05
LEFTOVER_LAMBDAS = (0.3, 1.0, 8.0)

# Copied pair-odd / gender-v14 / Hub-odd look. Hold working is *not* this.
V12_CPLUS = 0.90
V12_LOSS = 0.05
V12_COLLAPSE = -0.85
HOLD_CPLUS = 0.80  # ê_⊥û λ=8 on 2-D is ~0.70
HOLD_PERC = 0.60


def _unit(v: torch.Tensor) -> torch.Tensor:
    return lm_unit(v)


def shrink_factor(dim: int, hold_weight: float) -> float:
    """``student_ê = teacher_ê / (1 + λ D/2)`` when MSE is mean over D."""
    return 1.0 / (1.0 + float(hold_weight) * float(dim) / 2.0)


def looks_like_v12(row: dict) -> bool:
    """Copied pair-odd: high c+, locked ~0.02 loss, collapse ≲ −0.9."""
    return (
        float(row.get("cos_teacher", 0.0)) >= V12_CPLUS
        and float(row.get("loss", 1.0)) <= V12_LOSS
        and float(row.get("collapse", 1.0)) <= V12_COLLAPSE
    )


def hold_working_not_v12(row: dict) -> bool:
    """ê_⊥û leftover lock: slider high, leak small, c+ mid, perc mid."""
    return (
        float(row.get("hold_weight", 0.0)) > 0.0
        and float(row.get("cos_slider", row.get("cos_intended", 0.0))) >= 0.90
        and abs(float(row.get("leak_ratio", 1.0))) <= 0.20
        and float(row.get("cos_teacher", 1.0)) < HOLD_CPLUS
        and float(row.get("perc", 0.0)) >= HOLD_PERC
        and not looks_like_v12(row)
    )


def polarity_undefined(row: dict) -> bool:
    """Near-zero residual: live cosine is undefined / noisy, not a locked −1."""
    return float(row.get("strength", 1.0)) < STRENGTH_UNDEF


def pair_fit(
    residual: Residual,
    pos: torch.Tensor,
    neg: torch.Tensor,
    neu: torch.Tensor,
    *,
    leak_dir: torch.Tensor | None = None,
    hold_weight: float = 0.0,
    slider_dir: torch.Tensor | None = None,
) -> dict[str, float]:
    """Live-log numbers on one pair: c+ / slider-cos / collapse / perc / loss."""
    tgt_plus, tgt_minus = lm_hidden_targets(pos, neg, neu, target_mode="symmetric")
    pred_plus = neu + residual.delta(1.0)
    pred_minus = neu + residual.delta(-1.0)
    v_pos = pred_plus - neu
    v_neg = pred_minus - neu
    teacher = tgt_plus - neu
    hold = None
    used = 0.0
    if leak_dir is not None and hold_weight > 0.0:
        hold = lm_axis_hold(pred_plus, pred_minus, neu, leak_dir)
        used = float(hold_weight)
    loss = float(
        lm_slider_loss(
            pred_plus,
            pred_minus,
            tgt_plus,
            tgt_minus,
            hold=hold,
            hold_weight=used,
        )
    )
    strength = float(v_pos.norm() / teacher.norm().clamp_min(1e-8))
    raw_col = cosine(v_pos, v_neg)
    undef = strength < STRENGTH_UNDEF
    slider = 0.0 if slider_dir is None else cosine(v_pos, slider_dir)
    return {
        "cos_teacher": cosine(v_pos, teacher),
        "cos_slider": slider,
        "collapse": raw_col,
        "collapse_live": 0.0 if undef else raw_col,
        "polarity_undefined": undef,
        "perc": float(torch.norm(pred_plus - tgt_plus) / teacher.norm().clamp_min(1e-8)),
        "loss": loss,
        "strength": strength,
        "norm_plus": float(v_pos.norm()),
        "norm_minus": float(v_neg.norm()),
        "norm_teacher": float(teacher.norm()),
    }


@dataclass(frozen=True)
class HighDEnergyField:
    """Energy-like pair-odd in D dimensions.

    dim 0 is short loud/calm û. dim 1 is leftover unused of the poles
    (mix / BPM / genre inside a). Extra dims are empty on the poles so
    the MSE 1/D factor is real. ê is supplied by the caller — this field
    does not invent a leak pair.
    """

    dim: int = HIGH_D
    align: float = LIVE_ENERGY_ALIGN
    scale: float = CONCEPT_SCALE

    @property
    def u(self) -> torch.Tensor:
        out = torch.zeros(self.dim)
        out[0] = 1.0
        return out

    @property
    def unused(self) -> torch.Tensor:
        out = torch.zeros(self.dim)
        out[1] = 1.0
        return out

    @property
    def intended(self) -> torch.Tensor:
        return self.u.clone()

    def extra(self, index: int = 2) -> torch.Tensor:
        out = torch.zeros(self.dim)
        out[int(index)] = 1.0
        return out

    def odd(self) -> torch.Tensor:
        leftover = (1.0 - float(self.align) ** 2) ** 0.5
        return float(self.scale) * (float(self.align) * self.u + leftover * self.unused)

    def embed(self, prompt, t: float = 0.5) -> torch.Tensor:
        del t
        name = prompt.name if hasattr(prompt, "name") else str(prompt)
        if name in ("song", "neu"):
            return torch.zeros(self.dim)
        if name == "pos":
            return self.odd()
        if name == "neg":
            return -self.odd()
        raise KeyError(name)


def highd_pairs() -> list[Pair]:
    from analysis.slider2d.field import Prompt

    return [
        Pair(
            target=Prompt("song", 0.0, 0.0),
            positive=Prompt("pos", 1.0, 0.0),
            negative=Prompt("neg", -1.0, 0.0),
            neutral=Prompt("song", 0.0, 0.0),
        )
    ]


def short_u_at_caption_axis(
    e_hat: torch.Tensor,
    axis: float = LIVE_CAPTION_AXIS,
    junk: torch.Tensor | None = None,
) -> torch.Tensor:
    """Unit û with ``ê·û = axis``. Junk fills the rest (not leftover of a)."""
    e = _unit(e_hat)
    if junk is None:
        junk = torch.zeros_like(e)
        junk[min(2, e.numel() - 1)] = 1.0
    rest = junk.flatten() - (junk.flatten() @ e) * e
    if float(rest.norm()) < 1e-8:
        rest = torch.zeros_like(e)
        rest[0] = 1.0
        rest = rest - (rest @ e) * e
    rest = _unit(rest)
    a = float(axis)
    return a * e + ((1.0 - a * a) ** 0.5) * rest


def _annotate(metrics: dict, **extra) -> dict:
    metrics.update(extra)
    metrics["looks_like_v12"] = looks_like_v12(metrics)
    metrics["hold_working"] = hold_working_not_v12(metrics)
    if "polarity_undefined" not in metrics:
        metrics["polarity_undefined"] = polarity_undefined(metrics)
    if "collapse_live" not in metrics:
        metrics["collapse_live"] = (
            0.0 if metrics["polarity_undefined"] else float(metrics.get("collapse", 0.0))
        )
    return metrics


def score_gender_like(*, steps: int = 200, seed: int = 0) -> dict:
    """Clean pair, hold 0, no junk ê. The copied pair-odd / gender-v14 look."""
    field = MismatchField2D()
    residual = train_lm(
        field,
        mismatch_pairs(),
        target_mode="symmetric",
        hold_weight=0.0,
        steps=steps,
        seed=seed,
    )
    pos, neg, neu = field.rich_pair()
    odd = field.odd()
    metrics = score_against_odd(residual, odd, junk=field.junk)
    metrics.update(leftover_bipolar(residual.delta(1.0), residual.delta(-1.0)))
    fit = pair_fit(residual, pos, neg, neu, slider_dir=odd)
    metrics.update(fit)
    # On a clean pair the intended slider *is* a. Short û is a name.
    metrics["cos_slider"] = fit["cos_teacher"]
    metrics["cos_intended"] = metrics["cos_concept"]
    return _annotate(
        metrics,
        name="gender_like_hold0",
        cell="gender_like",
        hold_weight=0.0,
        leak_kind="none",
        dim=2,
        e_dot_u=float("nan"),
        e_dot_a=float("nan"),
        hold_norm=0.0,
    )


def score_ortho_perp_not_v12(*, steps: int = 200, seed: int = 0) -> dict:
    """Orthonormal 2-D: ê synonym + ê_⊥û λ=8. PASS leftover, FAIL v12-looking."""
    row = score_overlap_policy(
        "energy_perp_l8",
        overlap=0.5,
        hold_weight=LEAK_HOLD_WEIGHT,
        leak_mode="opposite",
        ortho="slider",
        steps=steps,
        seed=seed,
    )
    row["cos_slider"] = row["cos_intended"]
    row["e_dot_a"] = row.get("e_dot_odd", float("nan"))
    row["cell"] = "ortho_2d_perp"
    row["leak_kind"] = "synonym_perp_u"
    row["dim"] = 2
    if "polarity_undefined" not in row:
        row["strength"] = float(row.get("norm_plus", 0.0) / max(row.get("norm_odd", 1.0), 1e-8))
    return _annotate(row)


def score_highd(
    name: str,
    *,
    leak_kind: str,
    hold_weight: float,
    ortho: str = "raw",
    dim: int = HIGH_D,
    caption_axis: float = LIVE_CAPTION_AXIS,
    leftover_norm: float = 0.05,
    steps: int = 200,
    seed: int = 0,
) -> dict:
    field = HighDEnergyField(dim=dim)
    a = field.odd()
    a_hat = _unit(a)
    if leak_kind == "unused":
        raw_e = field.unused.clone()
    elif leak_kind == "synonym_a":
        raw_e = a_hat.clone()
    elif leak_kind == "tiny_unused":
        raw_e = _unit(field.u + float(leftover_norm) * field.unused)
    elif leak_kind == "synonym_pin":
        # Medium-energy on both leak captions cancels in ê. Density / genre
        # still restates the poles (ê≈a). Short û is loud/calm at caption-axis
        # 0.37 — dropping it leaves ê_⊥ still ≈ a.
        raw_e = a_hat.clone()
    else:
        raise ValueError(f"unknown leak_kind {leak_kind!r}")

    if leak_kind == "synonym_pin":
        slider = short_u_at_caption_axis(raw_e, axis=caption_axis, junk=field.extra(2))
    else:
        slider = field.u
    held = lm_hold_dir(raw_e, slider_dir=slider, odd_dir=a_hat, mode=ortho)
    residual = train_lm(
        field,
        highd_pairs(),
        target_mode="symmetric",
        hold_weight=0.0 if held is None else float(hold_weight),
        leak_dir=None if held is None else held,
        slider_dir=None if ortho == "raw" else slider,
        steps=steps,
        seed=seed,
    )
    neu = torch.zeros(field.dim)
    pos, neg = a, -a
    fit = pair_fit(
        residual,
        pos,
        neg,
        neu,
        leak_dir=held,
        hold_weight=0.0 if held is None else float(hold_weight),
        slider_dir=field.u,
    )
    d_plus = residual.delta(1.0)
    leak = float(d_plus @ field.unused) / (abs(float(d_plus @ field.u)) + 1e-8)
    held_unit = None if held is None else _unit(held)
    metrics = {
        "name": name,
        "cell": "highd",
        "leak_kind": leak_kind,
        "ortho": ortho,
        "dim": dim,
        "hold_weight": float(hold_weight),
        "used_hold": 0.0 if held is None else float(hold_weight),
        "hold_off": held is None,
        "hold_norm": 0.0 if held is None else float(held.norm()),
        "hold_dot_a": 0.0 if held_unit is None else float(held_unit @ a_hat),
        "hold_dot_u": 0.0 if held_unit is None else float(held_unit @ field.u),
        "e_dot_u": float(_unit(raw_e) @ slider) if leak_kind == "synonym_pin" else float(_unit(raw_e) @ field.u),
        "e_dot_a": float(_unit(raw_e) @ a_hat),
        "caption_axis": float(_unit(raw_e) @ slider),
        "cos_intended": fit["cos_slider"],
        "leak_ratio": leak,
        "norm_odd": float(a.norm()),
        "shrink_closed": shrink_factor(dim, hold_weight) if leak_kind == "synonym_a" and ortho == "raw" else float("nan"),
        "pass": abs(leak) <= 0.20 and fit["cos_slider"] >= 0.90 and fit["collapse"] <= -0.85,
        **leftover_bipolar(d_plus, residual.delta(-1.0)),
        **fit,
    }
    return _annotate(metrics)


def score_leftover_lambda(
    hold_weight: float,
    *,
    steps: int = 200,
    seed: int = 0,
) -> dict:
    """Unused-only ê (genre+BPM, no density) on the live energy poles."""
    row = score_overlap_policy(
        f"leftover_only_l{hold_weight:g}",
        overlap=0.0,
        hold_weight=float(hold_weight),
        leak_mode="opposite",
        ortho="raw",
        use_unused_e=True,
        steps=steps,
        seed=seed,
    )
    row["cos_slider"] = row["cos_intended"]
    row["e_dot_a"] = row.get("e_dot_odd", float("nan"))
    row["cell"] = "leftover_only"
    row["leak_kind"] = "unused"
    row["dim"] = 2
    if "strength" not in row:
        row["strength"] = float(row.get("norm_plus", 0.0) / max(row.get("norm_odd", 1.0), 1e-8))
    return _annotate(row)


def score_pair_odd_sub_e(*, steps: int = 200, seed: int = 0) -> dict:
    """Teacher = pair-odd − ê on the leaky energy poles. Not wired live."""
    field = EnergyLiveField2D()
    residual = train_lm(
        field,
        energy_pairs(field),
        target_mode="symmetric",
        subtract_dir=field.unused,
        steps=steps,
        seed=seed,
    )
    metrics = _score_energy_residual(residual, field, decisions=[False] * len(field.aligns))
    metrics.update(leftover_bipolar(residual.delta(1.0), residual.delta(-1.0)))
    fit = live_fit_metrics(residual, field, leak_dir=None, hold_weight=0.0)
    metrics.update(fit)
    metrics["cos_slider"] = metrics["cos_intended"]
    # c+ is alignment with full pair-odd a, which still contains unused.
    # After subtract-ê the student is û, so c+ ≈ |a·û| = 0.58.
    metrics["cell"] = "pair_odd_sub_e"
    metrics["leak_kind"] = "unused"
    metrics["dim"] = 2
    metrics["hold_weight"] = 0.0
    metrics["name"] = "pair_odd_sub_e"
    if "strength" not in metrics:
        metrics["strength"] = float(metrics.get("norm_plus", 0.0) / max(metrics.get("norm_odd", 1.0), 1e-8))
    metrics["pass"] = (
        abs(metrics["leak_ratio"]) <= 0.20
        and metrics["cos_intended"] >= 0.90
        and metrics["cos_plus_minus"] <= -0.85
    )
    return _annotate(metrics)


def score_no_hold_energy(*, steps: int = 200, seed: int = 0) -> dict:
    """Full pair-odd, no ê. The Hub / v12-looking copy of leaky a."""
    row = score_overlap_policy(
        "pair_odd_no_hold",
        overlap=0.0,
        hold_weight=0.0,
        steps=steps,
        seed=seed,
    )
    row["cos_slider"] = row["cos_intended"]
    row["cell"] = "pair_odd_no_hold"
    row["leak_kind"] = "none"
    row["dim"] = 2
    if "strength" not in row:
        row["strength"] = float(row.get("norm_plus", 0.0) / max(row.get("norm_odd", 1.0), 1e-8))
    return _annotate(row)


def signature_table(*, steps: int = 200, seed: int = 0) -> list[dict]:
    """Every live bullet the orthonormal 2-D cells were missing."""
    rows = [
        score_gender_like(steps=steps, seed=seed),
        score_no_hold_energy(steps=steps, seed=seed),
        score_ortho_perp_not_v12(steps=steps, seed=seed),
        score_leftover_lambda(0.3, steps=steps, seed=seed),
        score_leftover_lambda(1.0, steps=steps, seed=seed),
        score_leftover_lambda(8.0, steps=steps, seed=seed),
        score_pair_odd_sub_e(steps=steps, seed=seed),
        score_highd(
            "highd_synonym_raw_l8",
            leak_kind="synonym_a",
            hold_weight=LEAK_HOLD_WEIGHT,
            ortho="raw",
            steps=steps,
            seed=seed,
        ),
        score_highd(
            "highd_synonym_slider_l8",
            leak_kind="synonym_a",
            hold_weight=LEAK_HOLD_WEIGHT,
            ortho="slider",
            steps=steps,
            seed=seed,
        ),
        score_highd(
            "highd_tiny_unused_slider_l8",
            leak_kind="tiny_unused",
            hold_weight=LEAK_HOLD_WEIGHT,
            ortho="slider",
            steps=steps,
            seed=seed,
        ),
        score_highd(
            "highd_pin_caption037_l8",
            leak_kind="synonym_pin",
            hold_weight=LEAK_HOLD_WEIGHT,
            ortho="slider",
            caption_axis=LIVE_CAPTION_AXIS,
            steps=steps,
            seed=seed,
        ),
        score_highd(
            "highd_unused_slider_l8",
            leak_kind="unused",
            hold_weight=LEAK_HOLD_WEIGHT,
            ortho="slider",
            steps=steps,
            seed=seed,
        ),
    ]
    return rows


def compact(row: dict) -> dict:
    skip = {"history", "axis", "delta_plus", "delta_minus", "decisions", "row_aligns"}
    out = {}
    for key, value in row.items():
        if key in skip:
            continue
        if isinstance(value, (int, float, bool, str)):
            out[key] = value
        elif isinstance(value, list) and value and isinstance(value[0], (int, float, bool)):
            out[key] = [float(v) if isinstance(v, (int, float)) else v for v in value]
    return out
