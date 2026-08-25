"""Live energy geometry: leaky structured poles vs a short intended û.

The old energetic×gender leak cell sets û from the pole names
(energetic↔calm). There ``|odd·û|/||odd|| ≈ 0.95``, so project+hold is
almost identity on the intended axis and cannot see live energy.

Live energy-v9/v12 on Music 3: structured poles leak unused mix / genre /
BPM, and the short loud/calm û *is* the intended slider, but only at
**0.48 and 0.68** across four rows. A hard per-row 0.50 gate splits
those rows (mixed teacher on one LoRA). Hub / pair-symmetric still
leaks the unused attr.

This field is that geometry. Tests fail if someone sets û = pole-odd
(the 0.95 cheat) and calls it energy.

CPU hidden-space only. No Hub, no GPU, no Music 3 weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from analysis.slider2d.field import E_ATTR, E_SLIDER, Prompt, cosine
from analysis.slider2d.train import Pair, Residual, pair_slider_dir, train_lm
from conceptmod.textsliders.slider_targets import (
    SLIDER_ALIGN_MIN,
    lm_odd_align,
    lm_project_decisions,
    lm_teachers_mixed,
)


# Live energy short loud/calm û vs structured poles (four rows).
LIVE_ENERGY_ALIGNS = (0.48, 0.48, 0.68, 0.68)

# Same ballpark as the gender mismatch cell / leak field.
CONCEPT_SCALE = 1.2

SHORT_POS = "Extremely high energy, aggressive and loud."
SHORT_NEG = "Extremely quiet and calm, almost silent."

# Structured poles — not the short û, not energetic/calm pole-name cheats.
ENERGY_POLES = (
    ("loud structured pop-punk", "quiet structured lullaby"),
    ("loud structured hardcore", "quiet structured drone"),
    ("loud structured hard-dance", "quiet structured downtempo"),
    ("loud structured anthem", "quiet structured hush"),
)

POLE_NEU = "song"


def energy_odd(align: float, scale: float = CONCEPT_SCALE) -> torch.Tensor:
    """Leaky pair-odd: intended û plus unused attr, with ``|odd·û|/||odd|| = align``."""
    a = float(align)
    if not 0.0 <= a <= 1.0:
        raise ValueError(f"align must be in [0, 1], got {align!r}")
    return scale * torch.tensor([a, (1.0 - a * a) ** 0.5])


@dataclass(frozen=True)
class EnergyLiveField2D:
    """Four leaky energy pairs; declared û is the intended short loud/calm axis."""

    aligns: tuple[float, ...] = LIVE_ENERGY_ALIGNS
    scale: float = CONCEPT_SCALE

    @property
    def intended(self) -> torch.Tensor:
        return E_SLIDER.clone()

    @property
    def unused(self) -> torch.Tensor:
        return E_ATTR.clone()

    @property
    def declared_u(self) -> torch.Tensor:
        return self.intended.clone()

    @property
    def declared_captions(self) -> tuple[str, str]:
        return SHORT_POS, SHORT_NEG

    @property
    def pole_names(self) -> tuple[tuple[str, str], ...]:
        return ENERGY_POLES[: len(self.aligns)]

    def _index_for(self, name: str) -> tuple[int, str] | None:
        if name == POLE_NEU or name == "song":
            return None
        if name == SHORT_POS:
            return -1, "short_pos"
        if name == SHORT_NEG:
            return -1, "short_neg"
        for i, (pos, neg) in enumerate(self.pole_names):
            if name == pos:
                return i, "pos"
            if name == neg:
                return i, "neg"
        raise KeyError(f"unknown energy prompt {name!r}")

    def embed(self, prompt: Prompt | str, t: float = 0.5) -> torch.Tensor:
        del t
        name = prompt.name if isinstance(prompt, Prompt) else prompt
        found = self._index_for(name)
        if found is None:
            return torch.tensor([0.0, 0.0])
        index, kind = found
        if kind == "short_pos":
            return 0.40 * self.declared_u
        if kind == "short_neg":
            return -0.40 * self.declared_u
        odd = energy_odd(self.aligns[index], self.scale)
        if kind == "pos":
            return odd.clone()
        return (-odd).clone()

    def odd(self, index: int) -> torch.Tensor:
        return energy_odd(self.aligns[index], self.scale)

    def odds(self) -> list[torch.Tensor]:
        return [self.odd(i) for i in range(len(self.aligns))]


def energy_pairs(field: EnergyLiveField2D | None = None) -> list[Pair]:
    field = field or EnergyLiveField2D()
    rows = []
    for pos_name, neg_name in field.pole_names:
        rows.append(
            Pair(
                target=Prompt(POLE_NEU, 0.0, 0.0),
                positive=Prompt(pos_name, 1.0, 0.0),
                negative=Prompt(neg_name, -1.0, 0.0),
                neutral=Prompt(POLE_NEU, 0.0, 0.0),
            )
        )
    return rows


def energy_row_aligns(field: EnergyLiveField2D | None = None) -> list[float]:
    field = field or EnergyLiveField2D()
    out = []
    for i, (pos_name, neg_name) in enumerate(field.pole_names):
        pos = field.embed(pos_name)
        neg = field.embed(neg_name)
        out.append(float(lm_odd_align(pos, neg, field.declared_u)))
        del i
    return out


def pair_odd_as_u_hides_energy(field: EnergyLiveField2D | None = None) -> bool:
    """True when û = a row's pole-odd prints align 1.0 while the short û does not.

    That is the old leak-0 lie: call the pair names the axis and project
    becomes identity, so unused leak never shows up as middling 0.48 / 0.68.
    """
    field = field or EnergyLiveField2D()
    pos, neg = field.pole_names[0]
    hide = float(lm_odd_align(field.embed(pos), field.embed(neg), field.odd(0)))
    live = energy_row_aligns(field)
    return hide > 0.99 and max(live) < 0.80


def energy_verdicts(metrics: dict) -> dict[str, str]:
    leak = abs(metrics["leak_ratio"])
    return {
        "slider": "right" if metrics["cos_intended"] >= 0.90 else "needs_help",
        "leak": "right" if leak <= 0.20 else "needs_help",
        "collapse": "right" if metrics["cos_plus_minus"] <= -0.85 else "needs_help",
        "strength": "right" if metrics["strength_on_u"] >= 0.50 else "needs_help",
        "mixed": "right" if not metrics["mixed"] else "needs_help",
    }


def energy_all_right(metrics: dict) -> bool:
    return all(v == "right" for v in energy_verdicts(metrics).values())


def _score_energy_residual(
    residual: Residual,
    field: EnergyLiveField2D,
    *,
    decisions: list[bool],
) -> dict:
    d_plus = residual.delta(1.0)
    d_minus = residual.delta(-1.0)
    odds = field.odds()
    mean_odd = sum(float(o.norm()) for o in odds) / len(odds)
    mean_u = sum(abs(float(o @ field.intended)) for o in odds) / len(odds)
    ds = float(d_plus @ field.intended)
    da = float(d_plus @ field.unused)
    return {
        "delta_plus": [float(d_plus[0]), float(d_plus[1])],
        "delta_minus": [float(d_minus[0]), float(d_minus[1])],
        "cos_intended": cosine(d_plus, field.intended),
        "cos_concept": cosine(d_plus, field.intended),
        "cos_plus_minus": cosine(d_plus, d_minus),
        "norm_plus": float(d_plus.norm()),
        "norm_minus": float(d_minus.norm()),
        "norm_odd": mean_odd,
        "strength": float(d_plus.norm() / max(mean_odd, 1e-8)),
        "strength_on_u": float(d_plus.norm() / max(mean_u, 1e-8)),
        "proj_intended": ds,
        "proj_unused": da,
        "leak_ratio": da / (abs(ds) + 1e-8),
        "row_aligns": energy_row_aligns(field),
        "mean_align": sum(energy_row_aligns(field)) / len(field.aligns),
        "decisions": [bool(d) for d in decisions],
        "mixed": lm_teachers_mixed(decisions),
        "n_rows": len(field.aligns),
    }


def score_energy_policy(
    name: str,
    *,
    project_odd: bool,
    hold_weight: float = 0.0,
    project_align_min: float | None = None,
    project_align_scope: str = "row",
    use_declared_u: bool = True,
    use_pole_odd_u: bool = False,
    leakage_floor: float | None = None,
    anchor_weight: float = 0.0,
    field: EnergyLiveField2D | None = None,
    steps: int = 200,
    seed: int = 0,
) -> dict:
    field = field or EnergyLiveField2D()
    if use_pole_odd_u:
        slider = field.odd(0)
    elif use_declared_u:
        slider = field.declared_u
    else:
        slider = None
    residual = train_lm(
        field,
        energy_pairs(field),
        symmetric=True,
        target_mode="symmetric",
        project_odd=project_odd,
        hold_weight=hold_weight,
        slider_dir=slider,
        project_align_min=project_align_min,
        project_align_scope=project_align_scope,
        leakage_floor=leakage_floor,
        anchor_weight=anchor_weight,
        steps=steps,
        seed=seed,
    )
    used = slider if slider is not None else field.declared_u
    aligns = [
        float(lm_odd_align(field.embed(pos), field.embed(neg), used))
        for pos, neg in field.pole_names
    ]
    if slider is None or not project_odd:
        decisions = [False] * len(field.aligns)
    else:
        decisions = lm_project_decisions(aligns, project_align_min, project_align_scope)
    metrics = _score_energy_residual(residual, field, decisions=decisions)
    metrics["row_aligns"] = aligns
    metrics["mean_align"] = sum(aligns) / len(aligns)
    metrics.update(
        {
            "name": name,
            "odd_align": metrics["mean_align"],
            "axis": energy_verdicts(metrics),
            "pass": energy_all_right(metrics),
        }
    )
    return metrics


def energy_policy_table(*, steps: int = 200, seed: int = 0) -> list[dict]:
    """Hub / always-project / per-row 0.50 / slider-level 0.50 on the energy cell."""
    return [
        score_energy_policy(
            "hub",
            project_odd=False,
            hold_weight=0.0,
            leakage_floor=-0.9,
            anchor_weight=0.3,
            steps=steps,
            seed=seed,
        ),
        score_energy_policy(
            "always_project_hold",
            project_odd=True,
            hold_weight=1.0,
            use_declared_u=True,
            steps=steps,
            seed=seed,
        ),
        score_energy_policy(
            "gated_row_0.50",
            project_odd=True,
            hold_weight=1.0,
            project_align_min=SLIDER_ALIGN_MIN,
            project_align_scope="row",
            steps=steps,
            seed=seed,
        ),
        score_energy_policy(
            "slider_align_0.50",
            project_odd=True,
            hold_weight=1.0,
            project_align_min=SLIDER_ALIGN_MIN,
            project_align_scope="slider",
            steps=steps,
            seed=seed,
        ),
    ]


def assert_not_pole_name_cheat(field: EnergyLiveField2D | None = None) -> None:
    """Raise if this cell collapsed back to û = pole names / energetic-calm."""
    field = field or EnergyLiveField2D()
    names = [n for pair in field.pole_names for n in pair]
    if "energetic" in names or "calm" in names:
        raise AssertionError("energy poles must not be the energetic/calm name cheat")
    if field.declared_captions == field.pole_names[0]:
        raise AssertionError("declared û captions must not be the structured poles")
    if field.declared_captions[0] in names or field.declared_captions[1] in names:
        raise AssertionError("declared short û must not equal a pole name")
    aligns = energy_row_aligns(field)
    if any(abs(a - 1.0) < 1e-3 for a in aligns):
        raise AssertionError("energy aligns must not be 1.0 (û = pole-odd cheat)")
    if set(round(a, 2) for a in aligns) != {0.48, 0.68}:
        raise AssertionError(f"energy aligns must be 0.48/0.68, got {aligns}")
    pair = energy_pairs(field)[0]
    # pair_slider_dir from pole polarity is the intended axis — not the cheat.
    # The cheat is û = embed(pos)−embed(neg).
    if not torch.allclose(pair_slider_dir(pair), field.intended):
        raise AssertionError("energy pole polarity should declare the intended axis")
    if torch.allclose(field.odd(0) / field.odd(0).norm(), field.declared_u, atol=1e-4):
        raise AssertionError("declared û must not be the leaky pair-odd")
