"""Hold-ê when ê is *not* unused gender — the live energy geometry.

Existing hold-ê cells PASS because they set ê = unused gender, which is
orthogonal to energetic. Live energy-v4 leak captions *are* energy
(slammed / 168 / pop-punk vs airy / 52 / lullaby), so ê overlaps the
slider and the pair-odd teacher. This cell makes that overlap a knob.

    ê(ρ) = ρ û + √(1−ρ²) unused          # opposite-energy leak captions
    ê·û = ρ

At ρ = 0 this is the old unused-ê cell. At ρ ≈ mean |odd·û|/||odd||
(0.58 on the energy rows) ê is a synonym of the poles. At ρ = 1
ê is the short loud/calm slider.

CPU hidden-space only. No Hub, no GPU, no Music 3 weights.
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
from analysis.slider2d.field import E_ATTR, E_SLIDER, cosine
from analysis.slider2d.train import Residual
from conceptmod.textsliders.slider_targets import (
    LEAK_HOLD_WEIGHT,
    SLIDER_ALIGN_MIN,
    leftover_bipolar,
    lm_hidden_targets,
    lm_hold_dir,
    lm_odd_align,
    lm_project_decisions,
    lm_slider_loss,
    lm_teachers_mixed,
    lm_unit,
)


OVERLAPS = (0.0, 0.3, 0.5, 0.7, 0.9, 1.0)
HOLD_LAMBDAS = (0.0, 1.0, 8.0, 32.0)
ORTHO_MODES = ("raw", "slider", "odd")
LEAK_MODES = ("opposite", "same_energy")

# Live early-stop band. 2-D cannot copy high-D cosine(near-zero, teacher)
# when ê ∥ â; those rows still have to show slider-cos collapse / perc
# stuck away from a locked fit.
LIVE_COS_FAIL = 0.25
LIVE_PERC_FAIL = 1.00
SLIDER_LOCK = 0.90
LEAK_LOCK = 0.20


def leak_axis(
    overlap: float,
    *,
    slider: torch.Tensor | None = None,
    unused: torch.Tensor | None = None,
    leak_mode: str = "opposite",
) -> torch.Tensor:
    """Declared ê. Opposite-energy leak mixes unused-attr vs slider-axis."""
    slider = E_SLIDER if slider is None else slider
    unused = E_ATTR if unused is None else unused
    mode = str(leak_mode).strip().lower()
    if mode == "same_energy":
        # Both leak captions loud; difference is mix / BPM / genre.
        return unused.flatten().clone()
    if mode != "opposite":
        raise ValueError(f"leak_mode must be opposite/same_energy, got {leak_mode!r}")
    rho = float(overlap)
    if not 0.0 <= rho <= 1.0:
        raise ValueError(f"overlap must be in [0, 1], got {overlap!r}")
    leftover = (1.0 - rho * rho) ** 0.5
    axis = rho * slider.flatten() + leftover * unused.flatten()
    return lm_unit(axis)


def mean_odd_unit(field: EnergyLiveField2D | None = None) -> torch.Tensor:
    field = field or EnergyLiveField2D()
    acc = torch.zeros(2)
    for odd in field.odds():
        acc = acc + lm_unit(odd)
    acc = acc / max(len(field.aligns), 1)
    return lm_unit(acc)


def leak_geometry(
    overlap: float,
    *,
    field: EnergyLiveField2D | None = None,
    leak_mode: str = "opposite",
    ortho: str = "raw",
) -> dict[str, float]:
    field = field or EnergyLiveField2D()
    raw = leak_axis(overlap, slider=field.intended, unused=field.unused, leak_mode=leak_mode)
    odd = mean_odd_unit(field)
    held = lm_hold_dir(raw, slider_dir=field.intended, odd_dir=odd, mode=ortho)
    held_unit = None if held is None else lm_unit(held)
    return {
        "overlap": float(overlap),
        "leak_mode": leak_mode,
        "ortho": ortho,
        "e_dot_u": float(raw @ lm_unit(field.intended)),
        "e_dot_odd": float(raw @ odd),
        "hold_norm": 0.0 if held is None else float(held.norm()),
        "hold_dot_u": 0.0 if held_unit is None else float(held_unit @ lm_unit(field.intended)),
        "hold_dot_odd": 0.0 if held_unit is None else float(held_unit @ odd),
        "hold_off": held is None,
    }


def live_fit_metrics(
    residual: Residual,
    field: EnergyLiveField2D,
    *,
    leak_dir: torch.Tensor | None,
    hold_weight: float,
) -> dict[str, float]:
    """Live-log numbers: c+ / collapse / perc / pole MSE, mean over rows."""
    pairs = energy_pairs(field)
    cos_s: list[float] = []
    cols: list[float] = []
    percs: list[float] = []
    losses: list[float] = []
    for pair in pairs:
        pos = field.embed(pair.positive)
        neg = field.embed(pair.negative)
        neu = field.embed(pair.neutral)
        tgt_plus, tgt_minus = lm_hidden_targets(pos, neg, neu, target_mode="symmetric")
        pred_plus = neu + residual.delta(1.0)
        pred_minus = neu + residual.delta(-1.0)
        v_pos = pred_plus - neu
        v_neg = pred_minus - neu
        v_pos_t = tgt_plus - neu
        v_neg_t = tgt_minus - neu
        cos_s.append(cosine(v_pos, v_pos_t))
        cols.append(cosine(v_pos, v_neg))
        perc = float(torch.norm(pred_plus - tgt_plus) / v_pos_t.norm().clamp_min(1e-8))
        percs.append(perc)
        hold = None
        used = 0.0
        if leak_dir is not None and hold_weight > 0.0:
            unit = lm_unit(leak_dir)

            def _along(pred: torch.Tensor) -> torch.Tensor:
                return ((pred - neu).flatten() @ unit).pow(2)

            hold = 0.5 * (_along(pred_plus) + _along(pred_minus))
            used = float(hold_weight)
        losses.append(
            float(
                lm_slider_loss(
                    pred_plus,
                    pred_minus,
                    tgt_plus,
                    tgt_minus,
                    hold=hold,
                    hold_weight=used,
                )
            )
        )
    n = max(len(cos_s), 1)
    return {
        "cos_teacher": sum(cos_s) / n,
        "collapse": sum(cols) / n,
        "perc": sum(percs) / n,
        "loss": sum(losses) / n,
    }


def _history_train(
    field: EnergyLiveField2D,
    *,
    leak_dir: torch.Tensor | None,
    hold_weight: float,
    project_odd: bool,
    slider_dir: torch.Tensor | None,
    project_align_min: float | None,
    leakage_floor: float | None,
    anchor_weight: float,
    steps: int,
    seed: int,
    every: int,
) -> tuple[Residual, list[dict]]:
    """Same loss as ``train_lm``, snapshot live-fit numbers along the way."""
    pairs = energy_pairs(field)
    residual = Residual.create("odd_even")
    t = 0.5
    packed = []
    aligns: list[float] = []
    for pair in pairs:
        pos = field.embed(pair.positive, t)
        neg = field.embed(pair.negative, t)
        neu = field.embed(pair.neutral, t)
        declared = slider_dir
        align = float(lm_odd_align(pos, neg, declared)) if declared is not None else 0.0
        packed.append((pos, neg, neu))
        aligns.append(align)
    decisions = (
        lm_project_decisions(aligns, project_align_min, "slider")
        if project_odd
        else [False] * len(packed)
    )
    from conceptmod.textsliders.slider_targets import (
        lm_anchor_kappa,
        lm_anchor_targets,
        lm_axis_hold,
        lm_project_odd_axis,
    )

    def loss_fn(res: Residual) -> torch.Tensor:
        total = residual.w_odd.new_zeros(())
        n = 0
        for (pos, neg, neu), do_project in zip(packed, decisions):
            if do_project and declared is not None:
                tgt_plus, tgt_minus = lm_project_odd_axis(pos, neg, neu, declared)
            else:
                tgt_plus, tgt_minus = lm_hidden_targets(pos, neg, neu, target_mode="symmetric")
            pred_plus = neu + res.delta(1.0)
            pred_minus = neu + res.delta(-1.0)
            hold = None
            used_hold = 0.0
            if (not do_project) and leak_dir is not None and hold_weight > 0.0:
                hold = lm_axis_hold(pred_plus, pred_minus, neu, leak_dir)
                used_hold = float(hold_weight)
            elif do_project and hold_weight > 0.0 and declared is not None:
                from conceptmod.textsliders.slider_targets import lm_ortho_hold

                hold = lm_ortho_hold(pred_plus, pred_minus, neu, declared)
                used_hold = float(hold_weight)
            anc_plus = anc_minus = None
            weight = float(anchor_weight)
            if weight > 0.0:
                if leakage_floor is None:
                    raise ValueError("anchor_autocal requires leakage_floor")
                kappa = lm_anchor_kappa(pos, neg, neu, float(leakage_floor), autocal=True)
                anc_plus, anc_minus = lm_anchor_targets(pos, neg, neu, kappa)
            total = total + lm_slider_loss(
                pred_plus,
                pred_minus,
                tgt_plus,
                tgt_minus,
                anchor_plus=anc_plus,
                anchor_minus=anc_minus,
                anchor_weight=weight,
                hold=hold,
                hold_weight=used_hold,
            )
            n += 1
        return total / n

    def snap(res: Residual, loss_val: float, step: int) -> dict:
        frozen = Residual(res.kind, res.w_odd.detach().clone(), None if res.w_even is None else res.w_even.detach().clone())
        fit = live_fit_metrics(frozen, field, leak_dir=leak_dir, hold_weight=hold_weight)
        return {"step": step, "loss": float(loss_val), **fit}

    torch.manual_seed(seed)
    opt = torch.optim.Adam(residual.parameters(), lr=0.08)
    hist = [snap(residual, float(loss_fn(residual).detach()), 0)]
    for i in range(steps):
        loss = loss_fn(residual)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (i + 1) % every == 0 or (i + 1) == steps:
            hist.append(snap(residual, float(loss.detach()), i + 1))
    even = None if residual.w_even is None else residual.w_even.detach().clone()
    return Residual(residual.kind, residual.w_odd.detach().clone(), even), hist


def score_overlap_policy(
    name: str,
    *,
    overlap: float = 0.0,
    hold_weight: float = 0.0,
    leak_mode: str = "opposite",
    ortho: str = "raw",
    project_odd: bool = False,
    project_align_min: float | None = None,
    leakage_floor: float | None = None,
    anchor_weight: float = 0.0,
    use_unused_e: bool = False,
    field: EnergyLiveField2D | None = None,
    steps: int = 200,
    seed: int = 0,
    history_every: int = 50,
) -> dict:
    field = field or EnergyLiveField2D()
    raw_e = (
        field.unused.clone()
        if use_unused_e
        else leak_axis(overlap, slider=field.intended, unused=field.unused, leak_mode=leak_mode)
    )
    geo = leak_geometry(
        0.0 if use_unused_e else overlap,
        field=field,
        leak_mode="same_energy" if use_unused_e else leak_mode,
        ortho=ortho,
    )
    held = lm_hold_dir(
        raw_e, slider_dir=field.intended, odd_dir=mean_odd_unit(field), mode=ortho
    )
    used_hold = 0.0 if project_odd else (float(hold_weight) if held is not None else 0.0)
    residual, history = _history_train(
        field,
        leak_dir=None if project_odd else held,
        hold_weight=hold_weight if project_odd else used_hold,
        project_odd=project_odd,
        slider_dir=field.declared_u if (project_odd or used_hold > 0.0) else field.declared_u,
        project_align_min=project_align_min,
        leakage_floor=leakage_floor,
        anchor_weight=anchor_weight,
        steps=steps,
        seed=seed,
        every=history_every,
    )
    decisions = (
        lm_project_decisions(list(field.aligns), project_align_min, "slider")
        if project_odd
        else [False] * len(field.aligns)
    )
    metrics = _score_energy_residual(residual, field, decisions=decisions)
    metrics.update(leftover_bipolar(residual.delta(1.0), residual.delta(-1.0)))
    fit = live_fit_metrics(residual, field, leak_dir=held, hold_weight=used_hold)
    metrics.update(fit)
    metrics.update(geo)
    metrics.update(
        {
            "name": name,
            "hold_weight": float(hold_weight),
            "used_hold": used_hold,
            "odd_align": metrics["mean_align"],
            "axis": energy_verdicts(metrics),
            "pass": energy_all_right(metrics),
            "live_fail": (
                fit["cos_teacher"] < LIVE_COS_FAIL and fit["perc"] >= LIVE_PERC_FAIL
            ),
            "looks_like_v12": (
                fit["cos_teacher"] >= 0.90
                and fit["loss"] <= 0.05
                and fit["collapse"] <= -0.85
            ),
            "slider_collapsed": metrics["cos_intended"] < 0.30,
            "history": history,
        }
    )
    return metrics


def overlap_sweep(
    *,
    overlaps: Iterable[float] = OVERLAPS,
    lambdas: Iterable[float] = HOLD_LAMBDAS,
    orthos: Iterable[str] = ORTHO_MODES,
    leak_modes: Iterable[str] = ("opposite",),
    steps: int = 200,
    seed: int = 0,
) -> list[dict]:
    rows = []
    for leak_mode in leak_modes:
        used_overlaps = (0.0,) if leak_mode == "same_energy" else tuple(overlaps)
        for rho in used_overlaps:
            for lam in lambdas:
                for ortho in orthos:
                    if lam == 0.0 and ortho != "raw":
                        continue
                    name = f"{leak_mode}_o{rho:.1f}_l{int(lam)}_{ortho}"
                    rows.append(
                        score_overlap_policy(
                            name,
                            overlap=rho,
                            hold_weight=lam,
                            leak_mode=leak_mode,
                            ortho=ortho,
                            steps=steps,
                            seed=seed,
                        )
                    )
    return rows


def baseline_table(*, steps: int = 200, seed: int = 0) -> list[dict]:
    """ê-independent recipes + unused-only v9, on the same energy poles."""
    return [
        score_overlap_policy(
            "pair_odd_no_hold",
            overlap=0.0,
            hold_weight=0.0,
            leak_mode="opposite",
            ortho="raw",
            steps=steps,
            seed=seed,
        ),
        score_overlap_policy(
            "hub",
            overlap=0.0,
            hold_weight=0.0,
            leakage_floor=-0.9,
            anchor_weight=0.3,
            steps=steps,
            seed=seed,
        ),
        score_overlap_policy(
            "gated_project_0.50",
            overlap=0.0,
            hold_weight=1.0,
            project_odd=True,
            project_align_min=SLIDER_ALIGN_MIN,
            steps=steps,
            seed=seed,
        ),
        score_overlap_policy(
            "v9_unused_e",
            overlap=0.0,
            hold_weight=LEAK_HOLD_WEIGHT,
            use_unused_e=True,
            ortho="raw",
            steps=steps,
            seed=seed,
        ),
        score_overlap_policy(
            "same_energy_hold_l8",
            overlap=0.0,
            hold_weight=LEAK_HOLD_WEIGHT,
            leak_mode="same_energy",
            ortho="raw",
            steps=steps,
            seed=seed,
        ),
        score_overlap_policy(
            "energy_slider_e_raw_l8",
            overlap=1.0,
            hold_weight=LEAK_HOLD_WEIGHT,
            leak_mode="opposite",
            ortho="raw",
            steps=steps,
            seed=seed,
        ),
        score_overlap_policy(
            "energy_slider_e_slider_l8",
            overlap=1.0,
            hold_weight=LEAK_HOLD_WEIGHT,
            leak_mode="opposite",
            ortho="slider",
            steps=steps,
            seed=seed,
        ),
        score_overlap_policy(
            "pole_synonym_raw_l8",
            overlap=0.5,
            hold_weight=LEAK_HOLD_WEIGHT,
            leak_mode="opposite",
            ortho="raw",
            steps=steps,
            seed=seed,
        ),
        score_overlap_policy(
            "pole_synonym_slider_l8",
            overlap=0.5,
            hold_weight=LEAK_HOLD_WEIGHT,
            leak_mode="opposite",
            ortho="slider",
            steps=steps,
            seed=seed,
        ),
        score_overlap_policy(
            "pole_synonym_odd_l8",
            overlap=0.5,
            hold_weight=LEAK_HOLD_WEIGHT,
            leak_mode="opposite",
            ortho="odd",
            steps=steps,
            seed=seed,
        ),
    ]


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
    out["history"] = [
        {
            k: (int(v) if k == "step" else (float(v) if isinstance(v, (int, float)) else v))
            for k, v in snap.items()
        }
        for snap in row.get("history", [])
    ]
    return out

