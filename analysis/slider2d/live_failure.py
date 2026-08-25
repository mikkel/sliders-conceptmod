"""CPU fixture for the live Music 3 hold failure.

The old overlap field has a direct odd residual, ``d(-1) = -d(+1)``.
That is useful for the caption geometry, but makes a polarity failure
impossible by construction.  This module keeps that 2-D control and adds
one deliberately small local-Jacobian model:

* the pair-odd teacher is still ``(+a, -a)``;
* one shared parameter follows that teacher exactly;
* one shared, hold-free response mode has different +1/-1 Jacobians.

The second mode is the minimum extra capacity needed to ask whether a
high-dimensional scalar hold can select a non-bipolar solution.  It is an
analogue, not a claim about the exact Qwen Jacobian.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from conceptmod.textsliders.slider_targets import (
    lm_axis_hold,
    lm_hold_dir,
    lm_slider_loss,
    lm_unit,
)


PAIR_ALIGN = 0.58
PAIR_SCALE = 1.20
HIGH_DIM = 64
TINY_LEFTOVER = 0.02


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.flatten()[None], b.flatten()[None]))


def _metrics(
    plus: torch.Tensor,
    minus: torch.Tensor,
    teacher: torch.Tensor,
    slider: torch.Tensor,
    leftover: torch.Tensor,
    *,
    hold_dir: torch.Tensor | None,
    hold_weight: float,
) -> dict[str, float]:
    zero = torch.zeros_like(teacher)
    hold = (
        lm_axis_hold(plus, minus, zero, hold_dir)
        if hold_dir is not None and hold_weight > 0.0
        else None
    )
    loss = lm_slider_loss(
        plus,
        minus,
        teacher,
        -teacher,
        hold=hold,
        hold_weight=hold_weight,
    )
    pair_odd_mse = F.mse_loss(plus, teacher) + F.mse_loss(minus, -teacher)
    perc = 0.5 * (
        (plus - teacher).norm() / teacher.norm()
        + (minus + teacher).norm() / teacher.norm()
    )
    return {
        "trainer_c_plus": _cos(plus, teacher),
        "slider_cos": _cos(plus, slider),
        "collapse": _cos(plus, minus),
        "leftover": float(abs(plus @ lm_unit(leftover)) / abs(plus @ lm_unit(slider)).clamp_min(1e-8)),
        "perc": float(perc),
        "loss": float(loss),
        "pair_odd_mse": float(pair_odd_mse),
        "norm_plus": float(plus.norm()),
    }


def direct_2d(
    name: str,
    *,
    hold_weight: float,
    subtract_leftover: bool = False,
    gender_like: bool = False,
) -> dict:
    """Closed-form direct residual using the same pole-MSE and scalar hold.

    For D=2 the optimum along held ê is ``teacher_e / (1 + λ)``.
    This is the existing orthonormal PASS, now with c+ and slider-cos named
    side by side.
    """
    slider = torch.tensor([1.0, 0.0])
    leftover = torch.tensor([0.0, 1.0])
    if gender_like:
        # The singer is the intended axis.  No leak axis is declared.
        teacher = PAIR_SCALE * slider
        plus, minus = teacher.clone(), -teacher.clone()
        held = None
    else:
        teacher = PAIR_SCALE * torch.tensor(
            [PAIR_ALIGN, math.sqrt(1.0 - PAIR_ALIGN**2)]
        )
        if subtract_leftover:
            plus = (teacher @ slider) * slider
        else:
            plus = torch.stack(
                [teacher[0], teacher[1] / (1.0 + float(hold_weight))]
            )
        minus = -plus
        held = leftover if hold_weight > 0.0 and not subtract_leftover else None
    row = _metrics(
        plus,
        minus,
        teacher,
        slider,
        leftover,
        hold_dir=held,
        hold_weight=0.0 if subtract_leftover else hold_weight,
    )
    row.update(
        {
            "name": name,
            "dim": 2,
            "hold_weight": float(hold_weight),
            "teacher": "pair_odd_sub_e" if subtract_leftover else "pair_odd",
            "hold_norm": 0.0 if held is None else float(held.norm()),
        }
    )
    if subtract_leftover:
        # This recipe changes the teacher to the projected vector, which it
        # fits exactly.  Keep pair_odd_mse/perc to expose the teacher change.
        row["loss"] = 0.0
    return row


def _high_dim_geometry(dim: int = HIGH_DIM) -> dict[str, torch.Tensor]:
    """Teacher and two response modes for the high-D analogue.

    ``q`` is a messy residual of true leftover + syntax noise.  ``b+`` and
    ``b-`` are q-orthogonal shared-capacity responses chosen so the finite-λ
    fit approaches the live signature: c+≈0.31, caption/slider cosine≈0.37,
    collapse≈+0.18.
    """
    if dim < 7:
        raise ValueError("high-D fixture needs at least seven dimensions")
    u = torch.zeros(dim)
    true_e = torch.zeros(dim)
    syntax = torch.zeros(dim)
    common = torch.zeros(dim)
    private_plus = torch.zeros(dim)
    private_minus = torch.zeros(dim)
    u[0], true_e[1], syntax[2] = 1.0, 1.0, 1.0
    common[3], private_plus[4], private_minus[5] = 1.0, 1.0, 1.0
    q = lm_unit(true_e + syntax)

    teacher_unit = PAIR_ALIGN * u + math.sqrt(1.0 - PAIR_ALIGN**2) * true_e
    # The limiting mode is slightly beyond the desired finite-λ result because
    # the small surviving pair-odd response pulls c+ up and collapse down.
    slider_part = 0.35
    leftover_part = (
        0.275 - PAIR_ALIGN * slider_part
    ) / math.sqrt(1.0 - PAIR_ALIGN**2)
    base_sq = slider_part**2 + 2.0 * leftover_part**2
    common_sq = 0.21 + base_sq
    private_sq = 1.0 - base_sq - common_sq
    if private_sq <= 0.0:
        raise AssertionError("invalid response-mode geometry")
    shared = math.sqrt(common_sq) * common
    private = math.sqrt(private_sq)
    bp = (
        slider_part * u
        + leftover_part * true_e
        - leftover_part * syntax
        + shared
        + private * private_plus
    )
    bm = (
        -slider_part * u
        - leftover_part * true_e
        + leftover_part * syntax
        + shared
        + private * private_minus
    )
    rms_scale = PAIR_SCALE * math.sqrt(float(dim))
    return {
        "slider": u,
        "leftover": true_e,
        "syntax": syntax,
        "hold": q,
        "teacher": rms_scale * teacher_unit,
        "plus_mode": rms_scale * bp,
        "minus_mode": rms_scale * bm,
    }


def high_dim_synonym(
    name: str,
    *,
    hold_weight: float = 8.0,
    pin_same_loudness: bool = False,
    dim: int = HIGH_DIM,
) -> dict:
    """Fit the two-parameter high-D response under ê_⊥û hold.

    Raw ê is almost all slider.  Orthogonalization leaves a tiny residual;
    ``lm_axis_hold`` normalizes it, so its norm does not weaken λ.  The
    "same loudness" pin changes only how close raw ê is to û: density/genre
    still leave exactly the same residual direction.
    """
    geo = _high_dim_geometry(dim)
    u, q, teacher = geo["slider"], geo["hold"], geo["teacher"]
    eps = TINY_LEFTOVER * (1.5 if pin_same_loudness else 1.0)
    raw_e = math.sqrt(1.0 - eps**2) * u + eps * q
    held = lm_hold_dir(raw_e, slider_dir=u, mode="slider")
    if held is None:
        raise AssertionError("synonym fixture must retain a tiny ê_⊥û")

    # p = x*a + y*b+, m = -x*a + y*b-.  Solve the exact convex live loss.
    bp = torch.stack([teacher, geo["plus_mode"]], dim=1)
    bm = torch.stack([-teacher, geo["minus_mode"]], dim=1)
    held_unit = lm_unit(held)
    strength = math.sqrt(float(hold_weight) * dim / 2.0)
    design = torch.cat(
        [
            bp,
            bm,
            strength * (held_unit @ bp)[None],
            strength * (held_unit @ bm)[None],
        ],
        dim=0,
    )
    target = torch.cat([teacher, -teacher, torch.zeros(2)])
    params = torch.linalg.lstsq(design, target).solution
    plus, minus = bp @ params, bm @ params
    row = _metrics(
        plus,
        minus,
        teacher,
        u,
        geo["leftover"],
        hold_dir=held,
        hold_weight=hold_weight,
    )
    row.update(
        {
            "name": name,
            "dim": dim,
            "hold_weight": float(hold_weight),
            "teacher": "pair_odd",
            "hold_norm": float(held.norm()),
            "raw_e_dot_u": float(raw_e @ u),
            "effective_factor": 1.0 + float(hold_weight) * dim / 2.0,
            "pin_same_loudness": bool(pin_same_loudness),
            "response_x": float(params[0]),
            "response_y": float(params[1]),
        }
    )
    return row


def fixture_table() -> list[dict]:
    rows = [
        direct_2d("gender_pair_odd_no_hold", hold_weight=0.0, gender_like=True),
        direct_2d("energy_2d_hold_l8", hold_weight=8.0),
        high_dim_synonym("highd_synonym_hold_l8", hold_weight=8.0),
        high_dim_synonym(
            "highd_same_loudness_pin_l8",
            hold_weight=8.0,
            pin_same_loudness=True,
        ),
    ]
    for weight in (0.3, 1.0, 8.0):
        rows.append(
            direct_2d(f"leftover_only_hold_l{weight:g}", hold_weight=weight)
        )
    rows.append(
        direct_2d(
            "pair_odd_sub_e",
            hold_weight=0.0,
            subtract_leftover=True,
        )
    )
    return rows
