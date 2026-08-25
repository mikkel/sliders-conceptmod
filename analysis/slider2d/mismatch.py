"""Gender-v1 mismatch: a clean pair-odd vs a short declared û.

The energetic×gender field in ``field.py`` sets û from pole names
(energetic↔calm = ``E_SLIDER``). There û is the intended concept, so
project-odd looks leak-0 and cannot see gender-v1.

This field is the opposite geometry, matching the live gender-v1 log
``odd·û / ||odd|| = 0.20``:

- poles are a rich/clean gender pair; ``(pos − neg)`` *is* the concept
- declared û is a *different* short phrase, not the pole names
- current ``lm_v9`` (always project + hold) must fail: tiny ``||d+||``,
  low cos to the true pair axis, hold eating the concept
- Hub / pair-symmetric (full odd, κ=0) must pass

CPU hidden-space only. No Hub, no GPU, no Music 3 weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from analysis.slider2d.field import E_SLIDER, Field2D, Prompt, cosine
from analysis.slider2d.train import Pair, Residual, music3_pairs, pair_slider_dir, train_lm
from conceptmod.textsliders.slider_targets import (
    lm_odd_align,
    lm_project_decisions,
    lm_teachers_mixed,
)


# Live gender-v1 train log: short declared captions vs the structured pair.
LIVE_GENDER_V1_ALIGN = 0.20

# Opt-in floor in the measured gap between this cell (0.20) and the leak
# cell (~0.95). Project+hold's slider-cos knee on *this* geometry is 0.90
# (cos(d+, pair-odd) = alignment after an exact project). 0.50 is the
# majority-of-odd rule that is right on both scored cells.
PROJECT_ALIGN_RECOMMENDED = 0.50
PROJECT_ALIGN_SLIDER_KNEE = 0.90

POLE_POS = "female structured lead"
POLE_NEG = "male structured lead"
POLE_NEU = "song"
SHORT_POS = "A woman is singing, her voice is feminine."
SHORT_NEG = "A man is singing, his voice is masculine."

# ||odd|| = 1.2 so pair strength is in the same ballpark as the leak field.
CONCEPT_SCALE = 1.2


def short_u(align: float = LIVE_GENDER_V1_ALIGN) -> torch.Tensor:
    """Unit declared û with ``|e_concept · û| = align``."""
    a = float(align)
    if not 0.0 <= a <= 1.0:
        raise ValueError(f"align must be in [0, 1], got {align!r}")
    return torch.tensor([a, (1.0 - a * a) ** 0.5])


@dataclass(frozen=True)
class MismatchField2D:
    """Clean gender pair on x; short declared û at ``align`` to that odd."""

    align: float = LIVE_GENDER_V1_ALIGN

    @property
    def concept(self) -> torch.Tensor:
        return torch.tensor([1.0, 0.0])

    @property
    def junk(self) -> torch.Tensor:
        return torch.tensor([0.0, 1.0])

    @property
    def declared_u(self) -> torch.Tensor:
        return short_u(self.align)

    @property
    def pole_names(self) -> tuple[str, str]:
        return POLE_POS, POLE_NEG

    @property
    def declared_captions(self) -> tuple[str, str]:
        return SHORT_POS, SHORT_NEG

    def embed(self, prompt: Prompt | str, t: float = 0.5) -> torch.Tensor:
        del t
        name = prompt.name if isinstance(prompt, Prompt) else prompt
        u = self.declared_u
        locs = {
            POLE_POS: torch.tensor([CONCEPT_SCALE, 0.0]),
            POLE_NEG: torch.tensor([-CONCEPT_SCALE, 0.0]),
            POLE_NEU: torch.tensor([0.0, 0.0]),
            "song": torch.tensor([0.0, 0.0]),
            SHORT_POS: 0.40 * u,
            SHORT_NEG: -0.40 * u,
        }
        if name not in locs:
            raise KeyError(f"unknown mismatch prompt {name!r}")
        return locs[name].clone()

    def rich_pair(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.embed(POLE_POS), self.embed(POLE_NEG), self.embed(POLE_NEU)

    def odd(self) -> torch.Tensor:
        pos, neg, _neu = self.rich_pair()
        return (pos - neg) / 2.0


def mismatch_pairs() -> list[Pair]:
    """One clean gender pair. Pole names are *not* the declared short û."""
    return [
        Pair(
            target=Prompt(POLE_NEU, 0.0, 0.0),
            positive=Prompt(POLE_POS, 1.0, 0.0),
            negative=Prompt(POLE_NEG, -1.0, 0.0),
            neutral=Prompt(POLE_NEU, 0.0, 0.0),
        )
    ]


def score_against_odd(
    residual: Residual,
    odd: torch.Tensor,
    *,
    junk: torch.Tensor | None = None,
) -> dict[str, float]:
    """Slider *strength* and leak vs the true pair-odd (not vs û)."""
    d_plus = residual.delta(1.0)
    d_minus = residual.delta(-1.0)
    odd_f = odd.flatten()
    if junk is None:
        junk_f = torch.tensor([-float(odd_f[1]), float(odd_f[0])])
    else:
        junk_f = junk.flatten()
    ds = float(d_plus.flatten() @ (odd_f / odd_f.norm().clamp_min(1e-8)))
    da = float(d_plus.flatten() @ (junk_f / junk_f.norm().clamp_min(1e-8)))
    return {
        "delta_plus": [float(d_plus[0]), float(d_plus[1])],
        "delta_minus": [float(d_minus[0]), float(d_minus[1])],
        "cos_concept": cosine(d_plus, odd_f),
        "cos_plus_minus": cosine(d_plus, d_minus),
        "norm_plus": float(d_plus.norm()),
        "norm_minus": float(d_minus.norm()),
        "norm_odd": float(odd_f.norm()),
        "strength": float(d_plus.norm() / odd_f.norm().clamp_min(1e-8)),
        "proj_concept": ds,
        "proj_junk": da,
        "leak_ratio": da / (abs(ds) + 1e-8),
        "concept_kept": abs(ds) / float(odd_f.norm().clamp_min(1e-8)),
    }


def mismatch_verdicts(metrics: dict) -> dict[str, str]:
    """Independent gates. Strength is the measurement the leak cell omitted."""
    leak = abs(metrics["leak_ratio"])
    return {
        "slider": "right" if metrics["cos_concept"] >= 0.90 else "needs_help",
        "leak": "right" if leak <= 0.20 else "needs_help",
        "collapse": "right" if metrics["cos_plus_minus"] <= -0.85 else "needs_help",
        "strength": "right" if metrics["strength"] >= 0.50 else "needs_help",
    }


def mismatch_all_right(metrics: dict) -> bool:
    ax = mismatch_verdicts(metrics)
    return all(v == "right" for v in ax.values())


def leak_cell_odd(field: Field2D | None = None, t: float = 0.5) -> torch.Tensor:
    field = field or Field2D()
    pos = field.embed("energetic", t)
    neg = field.embed("calm", t)
    return (pos - neg) / 2.0


def leak_cell_align(field: Field2D | None = None, t: float = 0.5) -> float:
    """|odd·û|/||odd|| on the energetic×gender cell (û = pole polarity)."""
    field = field or Field2D()
    pos = field.embed("energetic", t)
    neg = field.embed("calm", t)
    return float(lm_odd_align(pos, neg, E_SLIDER))


def train_mismatch(
    field: MismatchField2D | None = None,
    *,
    project_odd: bool,
    hold_weight: float = 0.0,
    slider_dir: torch.Tensor | None = None,
    project_align_min: float | None = None,
    project_align_scope: str = "row",
    steps: int = 200,
    seed: int = 0,
) -> Residual:
    field = field or MismatchField2D()
    return train_lm(
        field,
        mismatch_pairs(),
        symmetric=True,
        target_mode="symmetric",
        project_odd=project_odd,
        hold_weight=hold_weight,
        slider_dir=slider_dir,
        project_align_min=project_align_min,
        project_align_scope=project_align_scope,
        steps=steps,
        seed=seed,
    )


def score_mismatch_policy(
    name: str,
    *,
    project_odd: bool,
    hold_weight: float = 0.0,
    use_short_u: bool = True,
    project_align_min: float | None = None,
    project_align_scope: str = "row",
    leakage_floor: float | None = None,
    anchor_weight: float = 0.0,
    field: MismatchField2D | None = None,
    steps: int = 200,
    seed: int = 0,
) -> dict:
    field = field or MismatchField2D()
    declared = field.declared_u if use_short_u else None
    residual = train_lm(
        field,
        mismatch_pairs(),
        symmetric=True,
        target_mode="symmetric",
        project_odd=project_odd,
        hold_weight=hold_weight,
        slider_dir=declared,
        project_align_min=project_align_min,
        project_align_scope=project_align_scope,
        leakage_floor=leakage_floor,
        anchor_weight=anchor_weight,
        steps=steps,
        seed=seed,
    )
    odd = field.odd()
    pos, neg, _neu = field.rich_pair()
    metrics = score_against_odd(residual, odd, junk=field.junk)
    used_u = field.declared_u if use_short_u else field.concept
    align = float(lm_odd_align(pos, neg, used_u))
    if declared is None or not project_odd:
        decisions = [False]
    else:
        decisions = lm_project_decisions([align], project_align_min, project_align_scope)
    metrics.update(
        {
            "name": name,
            "odd_align": align,
            "cos_intended": metrics["cos_concept"],
            "strength_on_u": metrics["strength"],
            "row_aligns": [align],
            "mean_align": align,
            "decisions": decisions,
            "mixed": lm_teachers_mixed(decisions),
            "n_rows": 1,
            "axis": mismatch_verdicts(metrics),
            "pass": mismatch_all_right(metrics),
        }
    )
    return metrics


def score_leak_policy(
    name: str,
    *,
    project_odd: bool,
    hold_weight: float = 0.0,
    project_align_min: float | None = None,
    steps: int = 200,
    seed: int = 0,
) -> dict:
    """Same recipes on the existing energetic×gender cell (û = pole names)."""
    from analysis.slider2d.train import score_residual

    field = Field2D()
    residual = train_lm(
        field,
        music3_pairs(False),
        symmetric=True,
        target_mode="symmetric",
        project_odd=project_odd,
        hold_weight=hold_weight,
        project_align_min=project_align_min,
        steps=steps,
        seed=seed,
    )
    raw = score_residual(residual)
    odd = leak_cell_odd(field)
    # Strength vs the *intended* slider component, not the leaked odd.
    # ||d+|| / |odd · E_SLIDER| is 1.0 when project-odd recovers the slider.
    slider_comp = abs(float(odd @ E_SLIDER))
    d_plus = residual.delta(1.0)
    metrics = {
        "name": name,
        "cos_slider_plus": raw["cos_slider_plus"],
        "cos_concept": raw["cos_slider_plus"],
        "cos_plus_minus": raw["cos_plus_minus"],
        "leak_ratio": raw["leak_ratio"],
        "norm_plus": raw["norm_plus"],
        "norm_odd": float(odd.norm()),
        "strength": float(d_plus.norm() / max(slider_comp, 1e-8)),
        "odd_align": leak_cell_align(field),
    }
    leak = abs(metrics["leak_ratio"])
    metrics["axis"] = {
        "slider": "right" if metrics["cos_slider_plus"] >= 0.90 else "needs_help",
        "leak": "right" if leak <= 0.20 else "needs_help",
        "collapse": "right" if metrics["cos_plus_minus"] <= -0.85 else "needs_help",
        "strength": "right" if metrics["strength"] >= 0.50 else "needs_help",
    }
    metrics["pass"] = all(v == "right" for v in metrics["axis"].values())
    return metrics


def policy_table(*, steps: int = 200, seed: int = 0, align_min: float = PROJECT_ALIGN_RECOMMENDED) -> dict:
    """Score the four asked policies on both cells."""
    mismatch = [
        score_mismatch_policy(
            "pair_symmetric", project_odd=False, hold_weight=0.0, use_short_u=False,
            steps=steps, seed=seed,
        ),
        score_mismatch_policy(
            "always_project_hold", project_odd=True, hold_weight=1.0, use_short_u=True,
            steps=steps, seed=seed,
        ),
        score_mismatch_policy(
            "gated_align", project_odd=True, hold_weight=1.0, use_short_u=True,
            project_align_min=align_min, steps=steps, seed=seed,
        ),
        score_mismatch_policy(
            "u_is_pair_odd", project_odd=True, hold_weight=1.0, use_short_u=False,
            steps=steps, seed=seed,
        ),
    ]
    leak = [
        score_leak_policy(
            "pair_symmetric", project_odd=False, hold_weight=0.0,
            steps=steps, seed=seed,
        ),
        score_leak_policy(
            "always_project_hold", project_odd=True, hold_weight=1.0,
            steps=steps, seed=seed,
        ),
        score_leak_policy(
            "gated_align", project_odd=True, hold_weight=1.0,
            project_align_min=align_min, steps=steps, seed=seed,
        ),
        score_leak_policy(
            "u_is_pair_odd", project_odd=True, hold_weight=1.0,
            # Explicit û = leaked (pos−neg) is identity project — same as symmetric.
            project_align_min=None, steps=steps, seed=seed,
        ),
    ]
    # The last leak row above still uses pair_slider_dir (E_SLIDER), not pair-odd.
    # Score a true "û = pair odd" leak row separately.
    field = Field2D()
    odd = leak_cell_odd(field)
    residual = train_lm(
        field,
        music3_pairs(False),
        symmetric=True,
        target_mode="symmetric",
        project_odd=True,
        hold_weight=1.0,
        slider_dir=odd,
        steps=steps,
        seed=seed,
    )
    from analysis.slider2d.train import score_residual

    raw = score_residual(residual)
    leak[-1] = {
        "name": "u_is_pair_odd",
        "cos_slider_plus": raw["cos_slider_plus"],
        "cos_concept": raw["cos_slider_plus"],
        "cos_plus_minus": raw["cos_plus_minus"],
        "leak_ratio": raw["leak_ratio"],
        "norm_plus": raw["norm_plus"],
        "norm_odd": float(odd.norm()),
        "strength": float(residual.delta(1.0).norm() / max(abs(float(odd @ E_SLIDER)), 1e-8)),
        "odd_align": float(lm_odd_align(field.embed("energetic"), field.embed("calm"), odd)),
    }
    leak[-1]["axis"] = {
        "slider": "right" if raw["cos_slider_plus"] >= 0.90 else "needs_help",
        "leak": "right" if abs(raw["leak_ratio"]) <= 0.20 else "needs_help",
        "collapse": "right" if raw["cos_plus_minus"] <= -0.85 else "needs_help",
        "strength": "right" if leak[-1]["strength"] >= 0.50 else "needs_help",
    }
    leak[-1]["pass"] = all(v == "right" for v in leak[-1]["axis"].values())
    return {"mismatch": mismatch, "leak": leak, "align_min": align_min}


def sweep_project_hold_align(
    aligns: list[float] | None = None,
    *,
    steps: int = 120,
    seed: int = 0,
) -> list[dict]:
    """Knee: smallest alignment where always-project+hold passes this cell."""
    if aligns is None:
        aligns = [round(0.05 * i, 2) for i in range(1, 21)]
        if LIVE_GENDER_V1_ALIGN not in aligns:
            aligns = sorted(set(aligns + [LIVE_GENDER_V1_ALIGN]))
    rows = []
    for align in aligns:
        field = MismatchField2D(align=align)
        residual = train_mismatch(
            field,
            project_odd=True,
            hold_weight=1.0,
            slider_dir=field.declared_u,
            steps=steps,
            seed=seed,
        )
        metrics = score_against_odd(residual, field.odd(), junk=field.junk)
        metrics["align"] = float(align)
        metrics["axis"] = mismatch_verdicts(metrics)
        metrics["pass"] = mismatch_all_right(metrics)
        rows.append(metrics)
    return rows


def knee_from_sweep(rows: list[dict], key: str) -> float | None:
    """Smallest swept align where the named verdict is right (None if never)."""
    for row in rows:
        if row["axis"][key] == "right":
            return float(row["align"])
    return None


def pair_slider_dir_hides_mismatch() -> bool:
    """True when û-from-pole-names is the concept axis (old fixture's trap)."""
    pair = mismatch_pairs()[0]
    direction = pair_slider_dir(pair)
    field = MismatchField2D()
    return bool(torch.allclose(direction, field.concept, atol=1e-6))
