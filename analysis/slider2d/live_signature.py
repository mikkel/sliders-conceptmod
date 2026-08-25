"""Live Music 3 failure signatures on the CPU 2-D / high-D fixture.

Existing overlap / rich / faithful cells show ê·û overlap and ê_⊥û
locking slider-cos. They do not line up trainer c+ (cos with pair-odd ``a``)
vs slider-cos (cos with declared û), high-D tiny-ê_⊥ polarity stress, the
medium-energy synonym pin, leftover-only ê at several λ, or pair_odd_sub_e
on leaky energy poles.

This module scores those stories in one table with real numbers. It does not
change the live ``--lm_target v9`` default.

CPU hidden-space only. No Hub, no GPU, no Music 3 weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from analysis.slider2d.energy import (
    EnergyLiveField2D,
    _score_energy_residual,
    energy_pairs,
)
from analysis.slider2d.field import E_ATTR, E_SLIDER, Prompt, cosine
from analysis.slider2d.mismatch import score_against_odd
from analysis.slider2d.overlap import leak_axis, score_overlap_policy
from analysis.slider2d.train import Pair, Residual, train_lm
from conceptmod.textsliders.slider_targets import (
    LEAK_HOLD_WEIGHT,
    lm_hidden_targets,
    lm_hold_dir,
    lm_unit,
)


# Live bands referenced in the task brief.
GENDER_CPLUS_MIN = 0.90
GENDER_COLLAPSE_MAX = -0.85
ENERGY_SLIDER_LOCK = 0.90
ENERGY_LEAK_LOCK = 0.20
ENERGY_CPLUS_V12_LIKE = 0.85
ENERGY_CPLUS_HOLD_WORKING = 0.65
ENERGY_PERC_FAIL = 0.70
ENERGY_LOSS_FAIL = 0.50
LEFTOVER_CANARY_LAM = 1.0
LEFTOVER_CANARY_LEAK_MIN = 0.20
SYNONYM_PIN_OVERLAP = 0.95
HIGH_D_DEFAULT_DIM = 64


@dataclass(frozen=True)
class GenderLiveField2D:
    """Clean gender pair: ``a`` is the singer, no leak axis, hold λ=0.

    Short declared û aligns with the pair-odd (not the gender-v1 mismatch at
    0.20). Copying the full pair-odd teacher is the gender-v14 / v12-looking
    success path: high c+, antipodal ±1, no junk ê.
    """

    scale: float = 1.2

    POLE_POS = "female structured lead vocal"
    POLE_NEG = "male structured lead vocal"
    POLE_NEU = "song"
    SHORT_POS = "A woman is singing, her voice is feminine."
    SHORT_NEG = "A man is singing, his voice is masculine."

    @property
    def intended(self) -> torch.Tensor:
        return E_ATTR.clone()

    @property
    def declared_u(self) -> torch.Tensor:
        return self.intended.clone()

    def embed(self, prompt: Prompt | str, t: float = 0.5) -> torch.Tensor:
        del t
        name = prompt.name if isinstance(prompt, Prompt) else prompt
        locs = {
            self.POLE_POS: torch.tensor([0.0, self.scale]),
            self.POLE_NEG: torch.tensor([0.0, -self.scale]),
            self.POLE_NEU: torch.tensor([0.0, 0.0]),
            "song": torch.tensor([0.0, 0.0]),
            self.SHORT_POS: 0.40 * self.declared_u,
            self.SHORT_NEG: -0.40 * self.declared_u,
        }
        if name not in locs:
            raise KeyError(f"unknown gender prompt {name!r}")
        return locs[name].clone()

    def odd(self) -> torch.Tensor:
        pos = self.embed(self.POLE_POS)
        neg = self.embed(self.POLE_NEG)
        return (pos - neg) / 2.0


@dataclass(frozen=True)
class HighDLeakField:
    """High-D analogue of live energy: tiny ê_⊥ after dropping û.

    Pole MSE uses ``F.mse_loss`` mean over ``dim`` components; hold along
  ê_⊥ does not divide by ``dim``. As ``dim`` grows, the student projection
    on ê_⊥ shrinks toward zero while slider-cos can stay locked — the live
    c+ floor (~0.58 here) is the pair-odd slider fraction ``align``.

    Symmetric ``odd_even`` training keeps ±1 antipodal on this fixture; live
    energy-v14 collapse +0.18 is listed as not reproducible here.
    """

    dim: int = HIGH_D_DEFAULT_DIM
    align: float = 0.58
    scale: float = 1.2
    rho_e: float = 0.995
    eps: float = 0.04

    @property
    def intended(self) -> torch.Tensor:
        u = torch.zeros(self.dim)
        u[0] = 1.0
        return u

    @property
    def unused(self) -> torch.Tensor:
        v = torch.zeros(self.dim)
        v[1] = 1.0
        return v

    def odd_vec(self) -> torch.Tensor:
        a = float(self.align)
        v = torch.zeros(self.dim)
        v[0] = a * self.scale
        v[1] = (1.0 - a * a) ** 0.5 * self.scale
        return v

    def embed(self, prompt: Prompt | str, t: float = 0.5) -> torch.Tensor:
        del t
        name = prompt.name if isinstance(prompt, Prompt) else prompt
        if name in ("song", "neutral"):
            return torch.zeros(self.dim)
        odd = self.odd_vec()
        return odd.clone() if name in ("pos", "positive") else (-odd).clone()

    @property
    def leak_e(self) -> torch.Tensor:
        u = self.intended
        perp = torch.zeros(self.dim)
        perp[1] = 1.0
        spread = max(self.dim - 2, 1)
        for i in range(2, self.dim):
            perp[i] = self.eps / spread**0.5
        raw = self.rho_e * u + (1.0 - self.rho_e**2) ** 0.5 * lm_unit(perp)
        return lm_unit(raw)

    def pairs(self) -> list[Pair]:
        return [
            Pair(
                target=Prompt("song", 0.0, 0.0),
                positive=Prompt("positive", 1.0, 0.0),
                negative=Prompt("negative", -1.0, 0.0),
                neutral=Prompt("song", 0.0, 0.0),
            )
        ]


def gender_pairs(field: GenderLiveField2D | None = None) -> list[Pair]:
    field = field or GenderLiveField2D()
    return [
        Pair(
            target=Prompt(field.POLE_NEU, 0.0, 0.0),
            positive=Prompt(field.POLE_POS, 1.0, 0.0),
            negative=Prompt(field.POLE_NEG, -1.0, 0.0),
            neutral=Prompt(field.POLE_NEU, 0.0, 0.0),
        )
    ]


def hold_e_perp_norm(
    leak_dir: torch.Tensor,
    *,
    slider_dir: torch.Tensor,
    ortho: str = "slider",
    odd_dir: torch.Tensor | None = None,
) -> float:
    held = lm_hold_dir(leak_dir, slider_dir=slider_dir, odd_dir=odd_dir, mode=ortho)
    return 0.0 if held is None else float(held.norm())


def _live_fit(
    residual: Residual,
    field,
    pairs: list[Pair],
    *,
    leak_dir: torch.Tensor | None,
    hold_weight: float,
) -> dict[str, float]:
    """Mean pole MSE / c+ / collapse / perc — same columns as overlap cell."""
    from conceptmod.textsliders.slider_targets import lm_axis_hold, lm_slider_loss

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
        cos_s.append(cosine(v_pos, v_pos_t))
        cols.append(cosine(v_pos, v_neg))
        percs.append(float(torch.norm(pred_plus - tgt_plus) / v_pos_t.norm().clamp_min(1e-8)))
        hold = None
        used = 0.0
        if leak_dir is not None and hold_weight > 0.0:
            hold = lm_axis_hold(pred_plus, pred_minus, neu, leak_dir)
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


def score_gender_pair_odd(*, steps: int = 200, seed: int = 0) -> dict:
    """Gender-like: hold 0, copied pair-odd (v12-looking c+, not û confusion)."""
    field = GenderLiveField2D()
    residual = train_lm(
        field,
        gender_pairs(field),
        symmetric=True,
        target_mode="symmetric",
        project_odd=False,
        hold_weight=0.0,
        steps=steps,
        seed=seed,
    )
    odd = field.odd()
    geom = score_against_odd(residual, odd, junk=field.intended)
    fit = _live_fit(
        residual,
        field,
        gender_pairs(field),
        leak_dir=None,
        hold_weight=0.0,
    )
    d_plus = residual.delta(1.0)
    da = float(d_plus[0])  # gender pair is ⊥ energetic; no unused leak
    ds = float(d_plus[1])
    leak = da / (abs(ds) + 1e-8)
    return _signature_row(
        "gender_pair_odd",
        live_story="gender-v14 / v12-looking copy of pair-odd, hold λ=0",
        dims=2,
        hold_weight=0.0,
        overlap=0.0,
        hold_e_perp_norm=0.0,
        cos_slider=cosine(d_plus, field.declared_u),
        cos_teacher=fit["cos_teacher"],
        collapse=fit["collapse"],
        leak=leak,
        perc=fit["perc"],
        loss=fit["loss"],
        pass_slider=geom["cos_concept"] >= GENDER_CPLUS_MIN and abs(leak) <= ENERGY_LEAK_LOCK,
        pass_collapse=fit["collapse"] <= GENDER_COLLAPSE_MAX,
        notes="No leak axis. c+ ≈ cos(concept): both high — the misread as slider lock.",
    )


def score_energy_perp_synonym(*, steps: int = 200, seed: int = 0) -> dict:
    """Energy ê≈â, hold ê_⊥û λ=8: slider locks, c+ ~0.70 — PASS leftover, FAIL v12 look."""
    row = score_overlap_policy(
        "energy_pole_synonym_perp",
        overlap=0.5,
        hold_weight=LEAK_HOLD_WEIGHT,
        ortho="slider",
        steps=steps,
        seed=seed,
    )
    return _signature_row(
        "energy_pole_synonym_perp",
        live_story="energy-v14 ê_⊥û on orthonormal 2-D at ρ=0.5",
        dims=2,
        hold_weight=LEAK_HOLD_WEIGHT,
        overlap=row["overlap"],
        hold_e_perp_norm=row.get("hold_norm", 0.0),
        cos_slider=row["cos_intended"],
        cos_teacher=row["cos_teacher"],
        collapse=row["cos_plus_minus"],
        leak=row["leak_ratio"],
        perc=row["perc"],
        loss=row["loss"],
        pass_slider=row["cos_intended"] >= ENERGY_SLIDER_LOCK
        and abs(row["leak_ratio"]) <= ENERGY_LEAK_LOCK,
        pass_collapse=row["cos_plus_minus"] <= GENDER_COLLAPSE_MAX,
        notes="Slider-cos high, leftover ~0.15, c+ ~0.70 — hold working, not v12-looking.",
    )


def score_high_d_tiny_eperp(
    *,
    dim: int = HIGH_D_DEFAULT_DIM,
    hold_weight: float = LEAK_HOLD_WEIGHT,
    steps: int = 400,
    seed: int = 0,
) -> dict:
    """High-D tiny ê_⊥: c+ drops toward align floor; ±1 stays antipodal in fixture."""
    field = HighDLeakField(dim=dim)
    held = lm_hold_dir(field.leak_e, slider_dir=field.intended, mode="slider")
    residual = train_lm(
        field,
        field.pairs(),
        symmetric=True,
        target_mode="symmetric",
        project_odd=False,
        hold_weight=hold_weight,
        leak_dir=held,
        slider_dir=field.intended,
        steps=steps,
        seed=seed,
    )
    fit = _live_fit(
        residual,
        field,
        field.pairs(),
        leak_dir=held,
        hold_weight=hold_weight,
    )
    d_plus = residual.delta(1.0)
    eperp_norm = hold_e_perp_norm(field.leak_e, slider_dir=field.intended)
    # Alignment of student with teacher along ê_⊥ only (live residual → 0).
    pos, neg, neu = field.embed("pos"), field.embed("neg"), field.embed("song")
    tp, tm = lm_hidden_targets(pos, neg, neu, target_mode="symmetric")
    vpt = tp - neu
    if held is not None and float(held.norm()) > 1e-8:
        hu = lm_unit(held)
        st_e = float(torch.dot(d_plus, hu))
        tt_e = float(torch.dot(vpt, hu))
        c_plus_eperp = st_e / (abs(tt_e) + 1e-8)
    else:
        c_plus_eperp = float("nan")
    return _signature_row(
        "high_d_tiny_eperp",
        live_story=f"live high-D energy-v14 analogue (D={dim})",
        dims=dim,
        hold_weight=hold_weight,
        overlap=float(field.leak_e @ field.intended),
        hold_e_perp_norm=eperp_norm,
        cos_slider=cosine(d_plus, field.intended),
        cos_teacher=fit["cos_teacher"],
        collapse=fit["collapse"],
        leak=float(torch.dot(d_plus, field.unused) / (abs(float(torch.dot(d_plus, field.intended))) + 1e-8)),
        perc=fit["perc"],
        loss=fit["loss"],
        pass_slider=fit["cos_teacher"] < ENERGY_CPLUS_V12_LIKE,
        pass_collapse=False,
        notes=(
            f"ê_⊥ norm {eperp_norm:.3f}; c+_eperp shrink {c_plus_eperp:.2f}. "
            "Collapse +0.18 not seen in symmetric odd_even — live-only."
        ),
        extra={"c_plus_eperp": c_plus_eperp},
    )


def score_synonym_pin(*, steps: int = 200, seed: int = 0) -> dict:
    """Medium-energy disguise: both captions loud, density/genre still ∥ û; raw hold λ=8."""
    field = EnergyLiveField2D()
    raw_e = leak_axis(
        SYNONYM_PIN_OVERLAP,
        slider=field.intended,
        unused=field.unused,
        leak_mode="opposite",
    )
    row = score_overlap_policy(
        "synonym_pin_raw",
        overlap=SYNONYM_PIN_OVERLAP,
        hold_weight=LEAK_HOLD_WEIGHT,
        leak_mode="opposite",
        ortho="raw",
        steps=steps,
        seed=seed,
    )
    eperp = hold_e_perp_norm(raw_e, slider_dir=field.intended)
    return _signature_row(
        "synonym_pin_medium_energy",
        live_story="same-loudness rewrite; ê still slider synonym, ê_⊥ tiny",
        dims=2,
        hold_weight=LEAK_HOLD_WEIGHT,
        overlap=SYNONYM_PIN_OVERLAP,
        hold_e_perp_norm=eperp,
        cos_slider=row["cos_intended"],
        cos_teacher=row["cos_teacher"],
        collapse=row["cos_plus_minus"],
        leak=row["leak_ratio"],
        perc=row["perc"],
        loss=row["loss"],
        pass_slider=False,
        pass_collapse=row["cos_plus_minus"] <= GENDER_COLLAPSE_MAX,
        notes="Raw hold punches slider (cos may go negative). Loss stays high; not the 2-D PASS band.",
    )


def score_leftover_only(
    hold_weight: float,
    *,
    steps: int = 250,
    seed: int = 0,
) -> dict:
    """Leftover-only ê (genre+BPM, no density) on energy poles."""
    field = EnergyLiveField2D()
    held = field.unused.clone()
    residual = train_lm(
        field,
        energy_pairs(field),
        symmetric=True,
        target_mode="symmetric",
        project_odd=False,
        hold_weight=hold_weight,
        leak_dir=held,
        slider_dir=field.intended,
        steps=steps,
        seed=seed,
    )
    metrics = _score_energy_residual(residual, field, decisions=[False] * len(field.aligns))
    fit = _live_fit(
        residual,
        field,
        energy_pairs(field),
        leak_dir=held,
        hold_weight=hold_weight,
    )
    return _signature_row(
        f"leftover_only_l{hold_weight:g}",
        live_story=f"genre+BPM-only ê, λ={hold_weight}",
        dims=2,
        hold_weight=hold_weight,
        overlap=0.0,
        hold_e_perp_norm=1.0,
        cos_slider=metrics["cos_intended"],
        cos_teacher=fit["cos_teacher"],
        collapse=fit["collapse"],
        leak=metrics["leak_ratio"],
        perc=fit["perc"],
        loss=fit["loss"],
        pass_slider=hold_weight == LEFTOVER_CANARY_LAM
        and abs(metrics["leak_ratio"]) > LEFTOVER_CANARY_LEAK_MIN
        and fit["loss"] < 1.0,
        pass_collapse=fit["collapse"] <= GENDER_COLLAPSE_MAX,
        notes="Canary at λ=1: bipolar, trainable loss, leftover still >0.20.",
    )


def score_pair_odd_sub_e(*, steps: int = 250, seed: int = 0) -> dict:
    """Teacher pair_odd − ê on leaky energy poles (compare to hold, do not wire live)."""
    field = EnergyLiveField2D()
    residual = train_lm(
        field,
        energy_pairs(field),
        symmetric=True,
        target_mode="symmetric",
        project_odd=False,
        hold_weight=0.0,
        subtract_dir=field.unused,
        steps=steps,
        seed=seed,
    )
    metrics = _score_energy_residual(residual, field, decisions=[False] * len(field.aligns))
    fit = _live_fit(
        residual,
        field,
        energy_pairs(field),
        leak_dir=None,
        hold_weight=0.0,
    )
    return _signature_row(
        "pair_odd_sub_e",
        live_story="drop declared ê from teacher on leaky energy poles",
        dims=2,
        hold_weight=0.0,
        overlap=0.0,
        hold_e_perp_norm=0.0,
        cos_slider=metrics["cos_intended"],
        cos_teacher=fit["cos_teacher"],
        collapse=fit["collapse"],
        leak=metrics["leak_ratio"],
        perc=fit["perc"],
        loss=fit["loss"],
        pass_slider=abs(metrics["leak_ratio"]) <= ENERGY_LEAK_LOCK,
        pass_collapse=fit["collapse"] <= GENDER_COLLAPSE_MAX,
        notes="Leak-0 on unused; c+ ~align — v12-looking teacher, not pair-odd copy.",
    )


def _signature_row(
    name: str,
    *,
    live_story: str,
    dims: int,
    hold_weight: float,
    overlap: float,
    hold_e_perp_norm: float,
    cos_slider: float,
    cos_teacher: float,
    collapse: float,
    leak: float,
    perc: float,
    loss: float,
    pass_slider: bool,
    pass_collapse: bool,
    notes: str,
    extra: dict | None = None,
) -> dict:
    out = {
        "name": name,
        "live_story": live_story,
        "dims": dims,
        "hold_weight": float(hold_weight),
        "overlap": float(overlap),
        "hold_e_perp_norm": float(hold_e_perp_norm),
        "cos_slider": float(cos_slider),
        "cos_teacher": float(cos_teacher),
        "collapse": float(collapse),
        "leak": float(leak),
        "perc": float(perc),
        "loss": float(loss),
        "v12_looking": float(cos_teacher) >= ENERGY_CPLUS_V12_LIKE
        and float(loss) < 0.05,
        "hold_working_not_v12": float(cos_slider) >= ENERGY_SLIDER_LOCK
        and float(cos_teacher) < ENERGY_CPLUS_V12_LIKE
        and float(loss) >= ENERGY_LOSS_FAIL,
        "notes": notes,
        "pass_slider_gate": pass_slider,
        "pass_collapse_gate": pass_collapse,
    }
    if extra:
        out.update(extra)
    return out


def signature_table(
    *,
    steps: int = 200,
    seed: int = 0,
    leftover_lambdas: Iterable[float] = (0.3, 1.0, 8.0),
    high_d_dim: int = HIGH_D_DEFAULT_DIM,
) -> list[dict]:
    """All live-signature rows in display order."""
    rows = [
        score_gender_pair_odd(steps=steps, seed=seed),
        score_energy_perp_synonym(steps=steps, seed=seed),
        score_high_d_tiny_eperp(dim=high_d_dim, steps=max(steps, 400), seed=seed),
        score_synonym_pin(steps=steps, seed=seed),
    ]
    for lam in leftover_lambdas:
        rows.append(score_leftover_only(lam, steps=max(steps, 250), seed=seed))
    rows.append(score_pair_odd_sub_e(steps=max(steps, 250), seed=seed))
    return rows


def verdict_table(rows: list[dict] | None = None) -> list[dict]:
    """Map each live bullet to fixture visibility."""
    rows = rows or signature_table()
    by_name = {r["name"]: r for r in rows}
    return [
        {
            "live_bullet": "gender-like copy pair-odd",
            "fixture": "gender_pair_odd",
            "visible": "yes",
            "cos_slider": by_name["gender_pair_odd"]["cos_slider"],
            "cos_teacher": by_name["gender_pair_odd"]["cos_teacher"],
            "verdict": "2-D cell; c+ high, collapse antipodal, hold λ=0",
        },
        {
            "live_bullet": "energy ê synonym + ê_⊥û λ=8",
            "fixture": "energy_pole_synonym_perp",
            "visible": "yes",
            "cos_slider": by_name["energy_pole_synonym_perp"]["cos_slider"],
            "cos_teacher": by_name["energy_pole_synonym_perp"]["cos_teacher"],
            "verdict": "slider-cos high, c+ ~0.70, loss ~0.85 — PASS leftover, FAIL v12 look",
        },
        {
            "live_bullet": "high-D tiny ê_⊥ + λ=8 collapse +0.18",
            "fixture": "high_d_tiny_eperp",
            "visible": "partial",
            "cos_slider": by_name["high_d_tiny_eperp"]["cos_slider"],
            "cos_teacher": by_name["high_d_tiny_eperp"]["cos_teacher"],
            "verdict": (
                "c+ drops vs 2-D; ê_⊥ residual shrinks. "
                f"collapse {by_name['high_d_tiny_eperp']['collapse']:+.2f} stays antipodal in fixture — live +0.18 not reproduced."
            ),
        },
        {
            "live_bullet": "synonym pin (medium energy, density ∥ û)",
            "fixture": "synonym_pin_medium_energy",
            "visible": "yes",
            "cos_slider": by_name["synonym_pin_medium_energy"]["cos_slider"],
            "cos_teacher": by_name["synonym_pin_medium_energy"]["cos_teacher"],
            "verdict": "raw hold still fights; ê_⊥ tiny at high ρ",
        },
        {
            "live_bullet": "leftover-only ê + λ=1 canary",
            "fixture": "leftover_only_l1",
            "visible": "yes",
            "cos_slider": by_name["leftover_only_l1"]["cos_slider"],
            "cos_teacher": by_name["leftover_only_l1"]["cos_teacher"],
            "verdict": (
                f"leak {by_name['leftover_only_l1']['leak']:+.3f}, loss {by_name['leftover_only_l1']['loss']:.3f} — "
                "trainable; try live before pair_odd_sub_e"
            ),
        },
        {
            "live_bullet": "pair_odd_sub_e vs hold",
            "fixture": "pair_odd_sub_e",
            "visible": "yes",
            "cos_slider": by_name["pair_odd_sub_e"]["cos_slider"],
            "cos_teacher": by_name["pair_odd_sub_e"]["cos_teacher"],
            "verdict": "leak-0 unused; teacher projects like short-û — next step if λ=1 canary fails live",
        },
    ]


def compact(row: dict) -> dict:
    skip = {"history", "axis", "delta_plus", "delta_minus"}
    return {k: v for k, v in row.items() if k not in skip}
