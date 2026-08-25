"""Can poles be richer without hosing leak?

Live Music 3 poles are structured captions, not two short words. Leak is
unused mix / BPM / gender *inside* ``h±``. Slider-synonym words are a
different direction: they should be free if they lie on the intended
axis.

This cell keeps the existing Field2D / EnergyLive / mismatch harness and
adds a 4-D rich field so those two kinds of extra words can be swept
apart:

    dim 0  short û          loud / calm / energetic
    dim 1  unused ê         gender (declared leak axis)
    dim 2  unused BPM       a second unused axis, not ê
    dim 3  slider richness  structured adjectives ⊥ short û

Hard-subtract ê is ``t± = h± − ((h±−h0)·ê)ê`` (then faithful or pair-odd).
That keeps every other component, unlike project-onto-short-û.

CPU only. No Hub, no GPU, no Music 3 weights. Does not change the live
trainer default.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from analysis.slider2d.energy import (
    EnergyLiveField2D,
    _score_energy_residual,
    energy_all_right,
    energy_pairs,
    energy_verdicts,
)
from analysis.slider2d.faithful import (
    score_energy_faithful,
    score_energy_odd,
    score_leak_lm,
    score_mismatch_faithful,
    score_pole_fit,
)
from analysis.slider2d.field import E_ATTR, E_SLIDER, Field2D, Prompt, cosine
from analysis.slider2d.mismatch import (
    MismatchField2D,
    mismatch_all_right,
    mismatch_pairs,
    mismatch_verdicts,
    score_against_odd,
    score_mismatch_policy,
)
from analysis.slider2d.train import (
    Pair,
    Residual,
    score_residual,
    train_lm,
)
from conceptmod.textsliders.slider_targets import leftover_bipolar, lm_odd_align


# Live numbers this cell must keep visible.
ALIGN_GENDER_V1 = 0.20
ALIGN_ENERGY = (0.48, 0.68)
ALIGN_CHEAT = 0.95
HOLD_LAMBDAS = (0.0, 1.0, 4.0, 8.0, 16.0, 32.0)
SLIDER_RICH_GRID = (0.0, 0.5, 0.8, 1.2, 2.0)
UNUSED_RICH_GRID = (0.0, 0.3, 0.6, 1.0, 1.5)
ONAXIS_SLIDER_GRID = (0.5, 1.0, 1.5, 2.0, 3.0)
ALIGN_GRID = (0.20, 0.48, 0.68, 0.95)

# Default rich leaky poles: Field2D-like even+odd gender, extra BPM unused,
# plus off-axis slider adjectives. Slider words are *not* unused.
DEFAULT_SLIDER = 1.00
DEFAULT_RICH = 0.80
DEFAULT_ODD_GENDER = 0.34
DEFAULT_EVEN_GENDER = 1.35
DEFAULT_ODD_BPM = 0.40

E_U = torch.tensor([1.0, 0.0, 0.0, 0.0])
E_GENDER = torch.tensor([0.0, 1.0, 0.0, 0.0])
E_BPM = torch.tensor([0.0, 0.0, 1.0, 0.0])
E_RICH = torch.tensor([0.0, 0.0, 0.0, 1.0])

POLE_POS = "loud aggressive driving pop-punk, bright timbre"
POLE_NEG = "quiet calm hushed lullaby, dark timbre"
SHORT_POS = "loud"
SHORT_NEG = "calm"
RICH_U_POS = "loud aggressive driving, bright timbre"
RICH_U_NEG = "quiet calm hushed, dark timbre"
POLE_NEU = "song"

POLE_COS_MIN = 0.90
RICH_KEPT_MIN = 0.70
LEAK_MAX = 0.20


def _unit(v: torch.Tensor) -> torch.Tensor:
    flat = v.flatten()
    return flat / flat.norm().clamp_min(1e-8)


@dataclass(frozen=True)
class ScaledField2D:
    """Existing energetic×gender field with on-axis slider / unused scales.

    ``slider_scale`` multiplies the intended component (extra energetic/calm
    synonyms that lie on û). ``unused_scale`` multiplies unused gender.
    ``(1, 1)`` is Field2D at ``t = 0.5``.
    """

    slider_scale: float = 1.0
    unused_scale: float = 1.0

    def embed(self, prompt: Prompt | str, t: float = 0.5) -> torch.Tensor:
        inner = Field2D()
        loc = inner.embed(prompt, t)
        neu = inner.embed("song", t)
        delta = loc - neu
        return neu + torch.stack(
            [self.slider_scale * delta[0], self.unused_scale * delta[1]]
        )


@dataclass(frozen=True)
class RichField:
    """4-D poles: short û, unused gender, unused BPM, slider richness."""

    slider: float = DEFAULT_SLIDER
    rich: float = DEFAULT_RICH
    odd_gender: float = DEFAULT_ODD_GENDER
    even_gender: float = DEFAULT_EVEN_GENDER
    odd_bpm: float = DEFAULT_ODD_BPM
    even_bpm: float = 0.0
    pin_gender: bool = False
    pin_bpm: bool = False

    @property
    def intended(self) -> torch.Tensor:
        raw = self.slider * E_U + self.rich * E_RICH
        if float(raw.norm()) < 1e-8:
            return E_U.clone()
        return _unit(raw)

    @property
    def short_u(self) -> torch.Tensor:
        return E_U.clone()

    @property
    def rich_u(self) -> torch.Tensor:
        return self.intended.clone()

    @property
    def leak_e(self) -> torch.Tensor:
        return E_GENDER.clone()

    @property
    def unused_dirs(self) -> tuple[torch.Tensor, torch.Tensor]:
        return E_GENDER, E_BPM

    def _gender(self) -> tuple[float, float]:
        if self.pin_gender:
            return 0.0, 0.0
        return self.even_gender, self.odd_gender

    def _bpm(self) -> tuple[float, float]:
        if self.pin_bpm:
            return 0.0, 0.0
        return self.even_bpm, self.odd_bpm

    def delta_plus(self) -> torch.Tensor:
        even_g, odd_g = self._gender()
        even_b, odd_b = self._bpm()
        return torch.tensor(
            [self.slider, even_g + odd_g, even_b + odd_b, self.rich],
            dtype=torch.float32,
        )

    def delta_minus(self) -> torch.Tensor:
        even_g, odd_g = self._gender()
        even_b, odd_b = self._bpm()
        return torch.tensor(
            [-self.slider, even_g - odd_g, even_b - odd_b, -self.rich],
            dtype=torch.float32,
        )

    def odd(self) -> torch.Tensor:
        return 0.5 * (self.delta_plus() - self.delta_minus())

    def even(self) -> torch.Tensor:
        return 0.5 * (self.delta_plus() + self.delta_minus())

    def embed(self, prompt: Prompt | str, t: float = 0.5) -> torch.Tensor:
        del t
        name = prompt.name if isinstance(prompt, Prompt) else prompt
        if name in (POLE_NEU, "song"):
            return torch.zeros(4)
        if name in (POLE_POS, RICH_U_POS):
            return self.delta_plus() if name == POLE_POS else self.slider * E_U + self.rich * E_RICH
        if name in (POLE_NEG, RICH_U_NEG):
            return self.delta_minus() if name == POLE_NEG else -(self.slider * E_U + self.rich * E_RICH)
        if name == SHORT_POS:
            return 0.40 * E_U
        if name == SHORT_NEG:
            return -0.40 * E_U
        raise KeyError(f"unknown rich prompt {name!r}")

    def align_short(self) -> float:
        odd = self.odd()
        return float(odd @ E_U) / float(odd.norm().clamp_min(1e-8))

    def align_rich(self) -> float:
        odd = self.odd()
        return float((odd @ self.rich_u).abs()) / float(odd.norm().clamp_min(1e-8))


def rich_pairs() -> list[Pair]:
    return [
        Pair(
            target=Prompt(POLE_NEU, 0.0, 0.0),
            positive=Prompt(POLE_POS, 1.0, 0.0),
            negative=Prompt(POLE_NEG, -1.0, 0.0),
            neutral=Prompt(POLE_NEU, 0.0, 0.0),
        )
    ]


def field_at_short_align(
    align: float,
    *,
    slider: float = DEFAULT_SLIDER,
    rich: float = 0.0,
    even_gender: float = 0.0,
) -> RichField:
    """Already-odd poles with ``|odd·û_short|/||odd|| = align`` when possible.

    Off-axis richness caps short-û align at ``s / sqrt(s²+r²)``. Past that
    the unused component would be imaginary — the field sits at the cap.
    """
    a = float(align)
    s = float(slider)
    r = float(rich)
    cap2 = s * s / max(s * s + r * r, 1e-8)
    cap = cap2 ** 0.5
    if a >= cap - 1e-6:
        odd_g = 0.0
        used_align = cap
    else:
        # a = s / sqrt(s² + g² + r²)  →  g² = s²/a² − s² − r²
        odd_g = max(s * s / max(a * a, 1e-8) - s * s - r * r, 0.0) ** 0.5
        used_align = a
    del used_align
    return RichField(
        slider=s,
        rich=r,
        odd_gender=odd_g,
        even_gender=even_gender,
        odd_bpm=0.0,
        even_bpm=0.0,
    )


def short_u_vs(concept: torch.Tensor, align: float, junk: torch.Tensor) -> torch.Tensor:
    """Unit û with ``|concept·û| = align``, junk filling the rest."""
    c = _unit(concept)
    j = junk.flatten() - (junk.flatten() @ c) * c
    if float(j.norm()) < 1e-8:
        j = E_GENDER.clone()
        j = j - (j @ c) * c
    j = _unit(j)
    a = float(align)
    return a * c + ((1.0 - a * a) ** 0.5) * j


def score_rich_residual(residual: Residual, field: RichField) -> dict:
    d_plus = residual.delta(1.0).flatten()
    d_minus = residual.delta(-1.0).flatten()
    teacher = field.delta_plus()
    intended = field.intended
    proj_u = float(d_plus @ E_U)
    proj_rich = float(d_plus @ E_RICH)
    proj_g = float(d_plus @ E_GENDER)
    proj_b = float(d_plus @ E_BPM)
    unused = (proj_g * proj_g + proj_b * proj_b) ** 0.5
    proj_int = float(d_plus @ intended)
    t_u = float(teacher @ E_U)
    t_rich = float(teacher @ E_RICH)
    t_g = float(teacher @ E_GENDER)
    t_b = float(teacher @ E_BPM)
    t_unused = (t_g * t_g + t_b * t_b) ** 0.5
    return {
        "delta_plus": [float(x) for x in d_plus],
        "delta_minus": [float(x) for x in d_minus],
        "proj_slider": proj_u,
        "proj_rich": proj_rich,
        "proj_gender": proj_g,
        "proj_bpm": proj_b,
        "proj_intended": proj_int,
        "unused_norm": unused,
        "teacher_unused": t_unused,
        "cos_slider": cosine(d_plus, E_U),
        "cos_intended": cosine(d_plus, intended),
        "cos_plus_minus": cosine(d_plus, d_minus),
        "pole_cos_plus": cosine(d_plus, teacher),
        "leak_ratio": unused / (abs(proj_u) + 1e-8),
        "leak_intended": unused / (abs(proj_int) + 1e-8),
        "leak_gender": proj_g / (abs(proj_u) + 1e-8),
        "leak_bpm": proj_b / (abs(proj_u) + 1e-8),
        "slider_kept": proj_u / (t_u + 1e-8) if abs(t_u) > 1e-8 else 1.0,
        "rich_kept": proj_rich / (t_rich + 1e-8) if abs(t_rich) > 1e-8 else 1.0,
        "gender_kept": proj_g / (t_g + 1e-8) if abs(t_g) > 1e-8 else (1.0 if abs(proj_g) < 1e-6 else 0.0),
        "norm_plus": float(d_plus.norm()),
        "norm_minus": float(d_minus.norm()),
        "align_short": field.align_short(),
        "align_rich": field.align_rich(),
        **leftover_bipolar(d_plus, d_minus),
    }


def rich_verdicts(metrics: dict) -> dict[str, str]:
    return {
        "slider": "right" if metrics["cos_intended"] >= 0.90 else "needs_help",
        "leak": "right" if abs(metrics["leak_ratio"]) <= LEAK_MAX else "needs_help",
        "collapse": "right" if metrics["cos_plus_minus"] <= -0.85 else "needs_help",
        "rich": "right" if metrics["rich_kept"] >= RICH_KEPT_MIN else "needs_help",
    }


def rich_all_right(metrics: dict) -> bool:
    return all(v == "right" for v in rich_verdicts(metrics).values())


def train_rich(
    field: RichField,
    *,
    target_mode: str = "symmetric",
    hold_weight: float = 0.0,
    leak_dir=None,
    subtract_dir=None,
    project_odd: bool = False,
    slider_dir: torch.Tensor | None = None,
    steps: int = 250,
    seed: int = 0,
) -> Residual:
    return train_lm(
        field,
        rich_pairs(),
        target_mode=target_mode,
        hold_weight=hold_weight,
        leak_dir=leak_dir,
        subtract_dir=subtract_dir,
        project_odd=project_odd,
        slider_dir=slider_dir,
        steps=steps,
        seed=seed,
    )


def score_rich(
    name: str,
    field: RichField | None = None,
    *,
    target_mode: str = "symmetric",
    hold_weight: float = 0.0,
    leak_dir=None,
    subtract_dir=None,
    project_odd: bool = False,
    slider_dir: torch.Tensor | None = None,
    steps: int = 250,
    seed: int = 0,
) -> dict:
    field = field or RichField()
    residual = train_rich(
        field,
        target_mode=target_mode,
        hold_weight=hold_weight,
        leak_dir=leak_dir,
        subtract_dir=subtract_dir,
        project_odd=project_odd,
        slider_dir=slider_dir,
        steps=steps,
        seed=seed,
    )
    metrics = score_rich_residual(residual, field)
    axis = rich_verdicts(metrics)
    metrics.update(
        {
            "name": name,
            "teacher_faithful": target_mode == "faithful" and subtract_dir is None and not project_odd,
            "axis": axis,
            "gates": all(v == "right" for k, v in axis.items() if k != "rich"),
            "keeps_rich_and_leak_right": axis["leak"] == "right" and axis["rich"] == "right" and axis["slider"] == "right",
            "field_slider": field.slider,
            "field_rich": field.rich,
            "field_odd_gender": field.odd_gender,
            "field_odd_bpm": field.odd_bpm,
            "field_even_gender": field.even_gender,
            "pin_gender": field.pin_gender,
            "pin_bpm": field.pin_bpm,
        }
    )
    return metrics


def teacher_variant_table(
    field: RichField | None = None,
    *,
    steps: int = 250,
    seed: int = 0,
) -> list[dict]:
    """Every asked teacher on the same rich leaky poles."""
    field = field or RichField()
    odd = field.odd()
    rows = [
        score_rich("faithful", field, target_mode="faithful", steps=steps, seed=seed),
        score_rich("pair_odd", field, target_mode="symmetric", steps=steps, seed=seed),
        score_rich(
            "faithful_sub_e",
            field,
            target_mode="faithful",
            subtract_dir=field.leak_e,
            steps=steps,
            seed=seed,
        ),
        score_rich(
            "pair_odd_sub_e",
            field,
            target_mode="symmetric",
            subtract_dir=field.leak_e,
            steps=steps,
            seed=seed,
        ),
        score_rich(
            "pair_odd_sub_all",
            field,
            target_mode="symmetric",
            subtract_dir=list(field.unused_dirs),
            steps=steps,
            seed=seed,
        ),
    ]
    for lam in HOLD_LAMBDAS:
        suffix = "raw" if lam == 0.0 else f"l{int(lam)}"
        rows.append(
            score_rich(
                f"hold_e_{suffix}",
                field,
                target_mode="symmetric",
                hold_weight=lam,
                leak_dir=field.leak_e if lam > 0.0 else None,
                steps=steps,
                seed=seed,
            )
        )
    rows.append(
        score_rich(
            "v9",
            field,
            target_mode="symmetric",
            hold_weight=8.0,
            leak_dir=field.leak_e,
            steps=steps,
            seed=seed,
        )
    )
    rows.append(
        score_rich(
            "v9_hold_all",
            field,
            target_mode="symmetric",
            hold_weight=8.0,
            leak_dir=list(field.unused_dirs),
            steps=steps,
            seed=seed,
        )
    )
    rows.append(
        score_rich(
            "project_short",
            field,
            project_odd=True,
            slider_dir=field.short_u,
            steps=steps,
            seed=seed,
        )
    )
    rows.append(
        score_rich(
            "project_rich",
            field,
            project_odd=True,
            slider_dir=field.rich_u,
            steps=steps,
            seed=seed,
        )
    )
    rows.append(
        score_rich(
            "project_odd",
            field,
            project_odd=True,
            slider_dir=odd,
            steps=steps,
            seed=seed,
        )
    )
    return rows


def slider_richness_sweep(
    *,
    unused_gender: float = DEFAULT_ODD_GENDER,
    unused_bpm: float = DEFAULT_ODD_BPM,
    even_gender: float = DEFAULT_EVEN_GENDER,
    steps: int = 250,
    seed: int = 0,
) -> list[dict]:
    """Extra slider synonyms at fixed unused. Off-axis richness (dim 3)."""
    rows = []
    for rich in SLIDER_RICH_GRID:
        field = RichField(
            slider=DEFAULT_SLIDER,
            rich=rich,
            odd_gender=unused_gender,
            even_gender=even_gender,
            odd_bpm=unused_bpm,
        )
        for name, kwargs in (
            ("pair_odd", {"target_mode": "symmetric"}),
            ("v9", {"target_mode": "symmetric", "hold_weight": 8.0, "leak_dir": field.leak_e}),
            ("pair_odd_sub_e", {"target_mode": "symmetric", "subtract_dir": field.leak_e}),
            ("project_short", {"project_odd": True, "slider_dir": field.short_u}),
            ("project_rich", {"project_odd": True, "slider_dir": field.rich_u}),
        ):
            row = score_rich(f"slider_{rich:.1f}_{name}", field, steps=steps, seed=seed, **kwargs)
            row["sweep"] = "slider_rich"
            row["sweep_x"] = rich
            row["recipe"] = name
            rows.append(row)
    return rows


def onaxis_slider_sweep(*, steps: int = 250, seed: int = 0) -> list[dict]:
    """Extra energetic/calm synonyms that lie on û. Existing 2-D field."""
    rows = []
    for scale in ONAXIS_SLIDER_GRID:
        field = ScaledField2D(slider_scale=scale, unused_scale=1.0)
        pairs = [
            Pair(
                target=Prompt("song", 0.0, 0.0),
                positive=Prompt("energetic", 1.0, 0.0),
                negative=Prompt("calm", -1.0, 0.0),
                neutral=Prompt("song", 0.0, 0.0),
            )
        ]
        for name, kwargs in (
            ("pair_odd", {"target_mode": "symmetric"}),
            ("v9", {"target_mode": "symmetric", "hold_weight": 8.0, "leak_dir": E_ATTR}),
            ("pair_odd_sub_e", {"target_mode": "symmetric", "subtract_dir": E_ATTR}),
            ("project_short", {"target_mode": "symmetric", "project_odd": True, "slider_dir": E_SLIDER}),
        ):
            residual = train_lm(field, pairs, steps=steps, seed=seed, **kwargs)
            metrics = score_residual(residual)
            pos, neg, neu = field.embed("energetic"), field.embed("calm"), field.embed("song")
            metrics.update(score_pole_fit(residual, pos, neg, neu))
            metrics.update(
                {
                    "name": f"onaxis_{scale:.1f}_{name}",
                    "recipe": name,
                    "sweep": "onaxis_slider",
                    "sweep_x": scale,
                    "rich_kept": 1.0,
                    "cos_intended": metrics["cos_slider_plus"],
                    "align_short": float(lm_odd_align(pos, neg, E_SLIDER)),
                }
            )
            leak = abs(metrics["leak_ratio"])
            metrics["axis"] = {
                "slider": "right" if metrics["cos_slider_plus"] >= 0.90 else "needs_help",
                "leak": "right" if leak <= LEAK_MAX else "needs_help",
                "collapse": "right" if metrics["cos_plus_minus"] <= -0.85 else "needs_help",
                "rich": "right",
            }
            rows.append(metrics)
    return rows


def unused_richness_sweep(
    *,
    which: str = "gender",
    slider: float = DEFAULT_SLIDER,
    rich: float = DEFAULT_RICH,
    steps: int = 250,
    seed: int = 0,
) -> list[dict]:
    """Extra unused words at fixed slider richness."""
    rows = []
    for unused in UNUSED_RICH_GRID:
        if which == "gender":
            field = RichField(slider=slider, rich=rich, odd_gender=unused, even_gender=0.0, odd_bpm=0.0)
        elif which == "bpm":
            field = RichField(
                slider=slider,
                rich=rich,
                odd_gender=DEFAULT_ODD_GENDER,
                even_gender=0.0,
                odd_bpm=unused,
            )
        else:
            raise ValueError(which)
        for name, kwargs in (
            ("pair_odd", {"target_mode": "symmetric"}),
            ("v9", {"target_mode": "symmetric", "hold_weight": 8.0, "leak_dir": field.leak_e}),
            ("pair_odd_sub_e", {"target_mode": "symmetric", "subtract_dir": field.leak_e}),
            ("pair_odd_sub_all", {"target_mode": "symmetric", "subtract_dir": list(field.unused_dirs)}),
        ):
            row = score_rich(f"unused_{which}_{unused:.1f}_{name}", field, steps=steps, seed=seed, **kwargs)
            row["sweep"] = f"unused_{which}"
            row["sweep_x"] = unused
            row["recipe"] = name
            rows.append(row)
    return rows


def partial_pin_table(*, steps: int = 250, seed: int = 0) -> list[dict]:
    """Pin gender only / BPM only / both. Slider adjectives stay rich."""
    rows = []
    for pin_g, pin_b, label in (
        (False, False, "free"),
        (True, False, "pin_gender"),
        (False, True, "pin_bpm"),
        (True, True, "pin_both"),
    ):
        field = RichField(pin_gender=pin_g, pin_bpm=pin_b)
        for name, kwargs in (
            ("faithful", {"target_mode": "faithful"}),
            ("pair_odd", {"target_mode": "symmetric"}),
            ("v9", {"target_mode": "symmetric", "hold_weight": 8.0, "leak_dir": field.leak_e}),
        ):
            row = score_rich(f"{label}_{name}", field, steps=steps, seed=seed, **kwargs)
            row["sweep"] = "partial_pin"
            row["pin"] = label
            row["recipe"] = name
            rows.append(row)
    return rows


def align_sweep(*, rich: float = DEFAULT_RICH, steps: int = 250, seed: int = 0) -> list[dict]:
    """Project short û vs rich û at live aligns. Already-odd poles."""
    rows = []
    for align in ALIGN_GRID:
        field = field_at_short_align(align, rich=rich)
        for name, kwargs in (
            ("pair_odd", {"target_mode": "symmetric"}),
            ("v9", {"target_mode": "symmetric", "hold_weight": 8.0, "leak_dir": field.leak_e}),
            ("project_short", {"project_odd": True, "slider_dir": field.short_u}),
            ("project_rich", {"project_odd": True, "slider_dir": field.rich_u}),
        ):
            row = score_rich(f"align_{align:.2f}_{name}", field, steps=steps, seed=seed, **kwargs)
            row["sweep"] = "align"
            row["sweep_x"] = align
            row["recipe"] = name
            row["short_align_cap"] = field.align_short()
            rows.append(row)
    # Clean pair + rich adjectives, short û at gender-v1 0.20 (kills singer / richness).
    clean = RichField(slider=1.2, rich=0.80, odd_gender=0.0, even_gender=0.0, odd_bpm=0.0)
    concept = clean.odd()
    tilted = short_u_vs(concept, ALIGN_GENDER_V1, E_GENDER)
    killed = score_rich(
        "mismatch_rich_project_short",
        clean,
        project_odd=True,
        slider_dir=tilted,
        steps=steps,
        seed=seed,
    )
    killed["sweep"] = "align"
    killed["sweep_x"] = ALIGN_GENDER_V1
    killed["recipe"] = "project_tilted_short"
    killed["kills_gender_v1"] = True
    rows.append(killed)
    kept = score_rich(
        "mismatch_rich_pair_odd",
        clean,
        target_mode="symmetric",
        steps=steps,
        seed=seed,
    )
    kept["sweep"] = "align"
    kept["recipe"] = "pair_odd_clean"
    rows.append(kept)
    return rows


def energy_mismatch_table(*, steps: int = 250, seed: int = 0) -> dict[str, dict]:
    """Do not overfit the even∥ê coincidence. Reuse the live cells."""
    energy_f = score_energy_faithful(hold_weight=0.0, steps=steps, seed=seed)
    energy_o = score_energy_odd(hold_weight=0.0, steps=steps, seed=seed)
    energy_v9 = score_energy_odd(hold_weight=8.0, steps=steps, seed=seed)
    efield = EnergyLiveField2D()
    residual = train_lm(
        efield,
        energy_pairs(efield),
        target_mode="symmetric",
        subtract_dir=efield.unused,
        steps=steps,
        seed=seed,
    )
    energy_sub = _score_energy_residual(residual, efield, decisions=[False] * len(efield.aligns))
    energy_sub.update(leftover_bipolar(residual.delta(1.0), residual.delta(-1.0)))
    energy_sub["name"] = "energy_odd_sub_e"
    energy_sub["axis"] = energy_verdicts(energy_sub)
    energy_sub["pass"] = energy_all_right(energy_sub)

    mismatch = score_mismatch_faithful(steps=steps, seed=seed)
    mismatch_v9 = score_mismatch_policy(
        "mismatch_hold_e_none",
        project_odd=False,
        hold_weight=0.0,
        use_short_u=False,
        steps=steps,
        seed=seed,
    )
    mismatch_proj = score_mismatch_policy(
        "mismatch_project_short",
        project_odd=True,
        hold_weight=1.0,
        use_short_u=True,
        steps=steps,
        seed=seed,
    )
    mfield = MismatchField2D()
    residual_m = train_lm(
        mfield,
        mismatch_pairs(),
        target_mode="symmetric",
        subtract_dir=mfield.junk,
        steps=steps,
        seed=seed,
    )
    mismatch_sub = score_against_odd(residual_m, mfield.odd(), junk=mfield.junk)
    mismatch_sub.update(leftover_bipolar(residual_m.delta(1.0), residual_m.delta(-1.0)))
    mismatch_sub["name"] = "mismatch_sub_junk"
    mismatch_sub["axis"] = mismatch_verdicts(mismatch_sub)
    mismatch_sub["pass"] = mismatch_all_right(mismatch_sub)
    return {
        "energy_faithful": energy_f,
        "energy_odd": energy_o,
        "energy_v9": energy_v9,
        "energy_sub_e": energy_sub,
        "mismatch_faithful": mismatch,
        "mismatch_pair_odd": mismatch_v9,
        "mismatch_project_short": mismatch_proj,
        "mismatch_sub_junk": mismatch_sub,
    }


def field2d_baselines(*, steps: int = 250, seed: int = 0) -> list[dict]:
    """Reuse the scored leaky energetic×gender numbers as baselines, not the only table."""
    return [
        score_leak_lm("lm_faithful_raw", target_mode="faithful", steps=steps, seed=seed),
        score_leak_lm(
            "lm_faithful_hold_l8",
            target_mode="faithful",
            hold_weight=8.0,
            leak_dir=E_ATTR,
            steps=steps,
            seed=seed,
        ),
        score_leak_lm("lm_faithful_attrs", target_mode="faithful", with_attrs=True, steps=steps, seed=seed),
        score_leak_lm("lm_symmetric", target_mode="symmetric", steps=steps, seed=seed),
        score_leak_lm(
            "lm_v9",
            target_mode="symmetric",
            hold_weight=8.0,
            leak_dir=E_ATTR,
            steps=steps,
            seed=seed,
        ),
        score_leak_lm(
            "lm_v9_project",
            target_mode="symmetric",
            project_odd=True,
            hold_weight=1.0,
            steps=steps,
            seed=seed,
        ),
        score_leak_lm(
            "lm_faithful_sub_e",
            target_mode="faithful",
            subtract_dir=E_ATTR,
            steps=steps,
            seed=seed,
        ),
        score_leak_lm(
            "lm_odd_sub_e",
            target_mode="symmetric",
            subtract_dir=E_ATTR,
            steps=steps,
            seed=seed,
        ),
    ]


def misaligned_e_cell(*, steps: int = 250, seed: int = 0) -> dict[str, dict]:
    """Oracle ê vs a slightly tilted ê. Hard-subtract of a wrong ê nicks the slider."""
    field = RichField()
    wrong = _unit(E_GENDER + 0.25 * E_U)
    return {
        "sub_true": score_rich(
            "sub_true_e",
            field,
            target_mode="symmetric",
            subtract_dir=field.leak_e,
            steps=steps,
            seed=seed,
        ),
        "sub_wrong": score_rich(
            "sub_wrong_e",
            field,
            target_mode="symmetric",
            subtract_dir=wrong,
            steps=steps,
            seed=seed,
        ),
        "hold_true": score_rich(
            "hold_true_e",
            field,
            target_mode="symmetric",
            hold_weight=8.0,
            leak_dir=field.leak_e,
            steps=steps,
            seed=seed,
        ),
        "hold_wrong": score_rich(
            "hold_wrong_e",
            field,
            target_mode="symmetric",
            hold_weight=8.0,
            leak_dir=wrong,
            steps=steps,
            seed=seed,
        ),
    }


def core_table(*, steps: int = 200, seed: int = 0) -> list[dict]:
    """Compact table for tests: default rich leaky poles, seed 0."""
    field = RichField()
    return [
        score_rich("faithful", field, target_mode="faithful", steps=steps, seed=seed),
        score_rich("pair_odd", field, target_mode="symmetric", steps=steps, seed=seed),
        score_rich("v9", field, target_mode="symmetric", hold_weight=8.0, leak_dir=field.leak_e, steps=steps, seed=seed),
        score_rich("pair_odd_sub_e", field, target_mode="symmetric", subtract_dir=field.leak_e, steps=steps, seed=seed),
        score_rich(
            "pair_odd_sub_all",
            field,
            target_mode="symmetric",
            subtract_dir=list(field.unused_dirs),
            steps=steps,
            seed=seed,
        ),
        score_rich("project_short", field, project_odd=True, slider_dir=field.short_u, steps=steps, seed=seed),
        score_rich("project_rich", field, project_odd=True, slider_dir=field.rich_u, steps=steps, seed=seed),
        score_rich("project_odd", field, project_odd=True, slider_dir=field.odd(), steps=steps, seed=seed),
    ]


def beats_v9_on_rich(row: dict, v9: dict) -> bool:
    """A live-default change needs a real win on rich leaky poles, not just leak-0."""
    if row.get("kills_gender_v1"):
        return False
    if abs(row["leak_ratio"]) > abs(v9["leak_ratio"]) - 0.02:
        return False
    if row["rich_kept"] < v9["rich_kept"] - 0.05:
        return False
    if row["cos_intended"] < 0.90:
        return False
    if row["cos_plus_minus"] > -0.85:
        return False
    return True
