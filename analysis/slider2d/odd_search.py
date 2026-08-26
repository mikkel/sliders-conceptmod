"""Odd enough to flip ``leak_frac`` negative, without becoming a midpoint.

The question this cell answers is a gate, not a vibe:

    ``exam_divergent`` is True  **and**  ``leak_frac < 0``

``exam_divergent`` is the energy-v4 two-track cell of the pair exam
(:mod:`analysis.slider2d.exam`) — five gates, every one of them something
the student *sings over a continuation*. ``leak_frac`` is
``cos(d₊, d₋)`` of the fitted ±1 student, out of
:func:`conceptmod.textsliders.slider_targets.leftover_bipolar`.

Before this cell the only row on the board clearing both was
``faithful_attrs``, and that is a caption rewrite (pin the unused
attribute in the yaml) rather than a loss. Everything that had ever
pushed ``leak_frac`` negative from the loss side —  ``pair_odd_midpoint``,
``pair_odd_sub_e``, ``dual_band_midpoint``, hold-ê, ``hub``, project —
landed at ``leak_frac ≈ −1`` and **failed** the divergent exam. Everything
that passed the divergent exam — caption poles, the leftover gate,
``faithful_guard_e``, ``dual_band_poles``, ``hidden_kl`` — sat at
``leak_frac`` between +0.03 and +0.12. The two clusters looked like two
sides of a wall.

They are not. There is no wall, and this cell's first job is to say why in
one line of algebra.

Where ``leak_frac`` actually comes from
=======================================

A live LoRA is one set of weights added at the prompt-last position of
every row, so the student is a shared residual ``d(σ) = σ·w + |σ|·w_even``
and the only two things it can spend on are the pair's even and odd
halves. On an attainable target that means

    ``d± = β·c ± γ·a``     with ``c = ½(h₊+h₋) − h0``, ``a = ½(h₊−h₋)``
    ``leak_frac = cos(d₊, d₋) = (β²‖c‖² − γ²‖a‖²) / (β²‖c‖² + γ²‖a‖²)``

:func:`conceptmod.textsliders.slider_targets.teacher_leak_frac` is that
expression, and on ``--pole_mode hidden`` it predicts the *fitted* number
to four decimals for every recipe on this board. Three things follow, and
all three are measured here rather than asserted:

1. **For a caption-faithful teacher (β = γ = 1) it is exactly the pair
   cosine the trainer already logs.** energy-v4 logs ``−0.11 … +0.14``
   across its three genre rows. So the *sign* of ``leak_frac`` on a
   caption teacher is a property of the row, not of the loss: on the
   ``−0.11`` row plain ``faithful`` is already a hit, and on the ``+0.14``
   row nothing caption-faithful can be one. The cell uses the midpoint
   ``+0.015``, which is why ``faithful_raw`` sits a hair positive.
2. **On a divergent pair the track cancels out of it.** With
   ``a = ½·track·(p̂−q̂) + o`` and ``c = ½·track·(p̂+q̂) + shared·ŝ``, both
   halves carry ``½·track²`` and it subtracts off. What is left is
   ``shared²`` against ``‖o‖²`` — "is there more content both poles state
   than content that flips sign". energy-v4 sits at 1.40 against 1.30. A
   7% margin is the entire wall.
3. **So ``leak_frac`` is a dial, and the only interesting question is what
   the exam charges for turning it.** That is the rest of this module.
   ``β < 1`` shrinks the even half; ``γ > 1`` lengthens the odd half.
   Both flip the sign at ``γ/β > 1.008`` on this pair.

The three directions of "more odd", and their price
===================================================

``faithful_gain``   ``t± = mid ± γ·a``
    Keep the caption pair's own midpoint; scale only the axis. Cheapest by
    a wide margin, because the pole's own track content scales with it and
    the divergent exam's live gate is whether the pole's genre word still
    beats its intensity word. :func:`lm_blend_guard` admits it at every γ.

``faithful_gain_blind``   the same gain, restricted to ``P_blind``
    Only the half of the axis the next-token KL cannot read at all. It
    sounds like the free lunch — extra axis at zero cost to the scored
    distribution — and it is about five times more expensive per unit of
    ``leak_frac``, because the delivery content the residual stream
    carries forward lands on the *axis adjective* and competes with the
    pole's own track word.

``faithful_common_agree``   ``t± = h0 + agree(h₊,h₋) ± a``
    Do not lengthen anything; delete the part of ``c`` that no single
    caption occupies (:func:`lm_common_agree`). On a close pair this is
    the caption pair exactly. On a divergent pair it strips the blend —
    and it **fails**, at the same gate and the same value as
    ``pair_odd_midpoint``. That is the useful negative: on a divergent
    pair the blend inside ``c`` is load-bearing, because half of the +
    pole's own track is sitting in it.

What a hit here does and does not mean
======================================

``leak_frac`` is a *ratio*. ``faithful_gain`` moves it by growing the
denominator, so ``even_norm`` is bit-identical at every γ — the same blend
is still in the residual, it is just outweighed. The board reports
``even_norm`` and ``odd_norm`` next to it so nobody reads a negative
``leak_frac`` as "the same-direction content went away".

The leftover column is worse than that and the cell publishes it: on the
``unused_e`` cell ``leak_tok`` falls monotonically with γ, from 0.228 at
γ=1 to 0.016 at γ=3, and **none of it is a real leftover drop**. The
hidden-space ratio ``(d₊·ĝ)/(d₊·û)`` is 0.450 at every γ to four
decimals. Driving the axis harder squeezes the unused attribute out of the
top of the next-token distribution while the hidden state carries exactly
as much of it, proportionally. A mass-ratio leak column saturates; that is
a Goodhart and it is charted rather than banked.

No live listen is claimed for anything here. The live default is unchanged
(``--lm_target v9`` / ``--pole_mode hidden``), and the leftover gate that
already sounds better on ears is same-direction on ``leak_frac``.

CPU only. No Hub, no GPU, no Music 3 weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from analysis.slider2d.exam import (
    CELL_IS,
    CELLS,
    GATE_BOUNDS,
    LIVE_PAIR_COS,
    PairField,
    divergent_field,
    fit_exam,
    score_exam,
)
from analysis.slider2d.sheet import (
    gender_like_field,
    leaky_field,
    score_sheet,
)
from conceptmod.textsliders.slider_targets import (
    leftover_bipolar,
    lm_pair_even_odd,
    teacher_leak_frac,
)


# The gate. Strictly negative — 0.0 is the faithful teacher on a pair whose
# logged cosine happens to be 0, and that is not a result.
LEAK_FRAC_WIN = 0.0
# The pair the criterion names. ``close`` and ``unused_e`` are reported for
# every candidate but do not decide the hit, exactly as the brief says.
WIN_CELL = "divergent"
# A hit has to survive every one of these, not the lucky one.
SEEDS = (0, 1, 2, 3, 4, 5)


@dataclass(frozen=True)
class Candidate:
    """One recipe, its ``score_exam`` kwargs, and what it is trying."""

    name: str
    family: str
    kwargs: dict
    idea: str


def candidates() -> list[Candidate]:
    """Every recipe this search scores, baselines and traps included.

    The traps are on the list on purpose. A search that only scores its own
    ideas cannot show that the thing it found is not the same thing the
    board already rejected.
    """
    out = [
        Candidate(
            "faithful_raw",
            "baseline",
            {"pole_mode": "hidden", "teacher": "faithful"},
            "the caption pair; leak_frac is then the pair's own logged cosine",
        ),
        Candidate(
            "semantic_kl_poles",
            "baseline",
            {"pole_mode": "semantic_kl", "teacher": "faithful"},
            "the live v18 recipe",
        ),
        Candidate(
            "dual_band_poles",
            "baseline",
            {"pole_mode": "dual_band", "teacher": "faithful"},
            "#35's shipped loss on caption poles",
        ),
        Candidate(
            "pair_odd_midpoint",
            "trap",
            {"pole_mode": "hidden", "teacher": "pair_odd"},
            "t± = h0 ± a; deletes c, leak_frac −1, the named trap",
        ),
        Candidate(
            "dual_band_midpoint",
            "trap",
            {"pole_mode": "dual_band", "teacher": "pair_odd"},
            "the same midpoint under #35's loss",
        ),
    ]
    # β: keep only part of the common term. This is the Hub's κ-anchor
    # geometry and the trainer's own ``--common_beta``; the board had
    # scored β=0 (the trap) and β=1 (faithful) and nothing in between.
    for beta in (0.9, 0.7, 0.5, 0.3, 0.1):
        out.append(
            Candidate(
                f"common_beta_{beta:g}",
                "shrink the even half",
                {"pole_mode": "hidden", "teacher": "pair_odd", "common_beta": beta},
                f"t± = h0 ± a + {beta:g}·c",
            )
        )
    # γ: keep all of c, lengthen the axis around the caption midpoint.
    for gain in (1.1, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0):
        out.append(
            Candidate(
                f"faithful_gain_{gain:g}",
                "lengthen the odd half",
                {
                    "pole_mode": "hidden",
                    "teacher": "faithful_gain",
                    "target_scale": gain,
                },
                f"t± = mid ± {gain:g}·a",
            )
        )
    # The same gain, restricted to the band a semantic KL cannot read.
    for gain in (1.5, 2.0, 3.0, 4.0):
        out.append(
            Candidate(
                f"blind_gain_{gain:g}",
                "lengthen the blind half only",
                {
                    "pole_mode": "hidden",
                    "teacher": "faithful_gain",
                    "blind_gain": gain,
                },
                f"t± = mid ± (a + {gain - 1:g}·P_blind(a))",
            )
        )
    out += [
        Candidate(
            "faithful_common_agree",
            "delete the blend",
            {"pole_mode": "hidden", "teacher": "faithful_common_agree"},
            "t± = h0 + agree(h₊,h₋) ± a; keeps shared, drops the blend",
        ),
        # The teacher and the loss that can actually deliver its band. A
        # semantic KL has zero gradient on P_blind, so the same target
        # under ``semantic_kl`` is a control for "did the gain arrive".
        Candidate(
            "blind_gain_3_dual_band",
            "lengthen the blind half only",
            {
                "pole_mode": "dual_band",
                "teacher": "faithful_gain",
                "blind_gain": 3.0,
            },
            "the blind over-drive under the loss that has a gradient there",
        ),
        Candidate(
            "blind_gain_3_semantic_kl",
            "lengthen the blind half only",
            {
                "pole_mode": "semantic_kl",
                "teacher": "faithful_gain",
                "blind_gain": 3.0,
            },
            "the same target under a loss with no gradient in that band",
        ),
        Candidate(
            "faithful_gain_2_dual_band",
            "lengthen the odd half",
            {
                "pole_mode": "dual_band",
                "teacher": "faithful_gain",
                "target_scale": 2.0,
            },
            "t± = mid ± 2·a under #35's loss",
        ),
    ]
    return out


# -- gate margins ---------------------------------------------------------


def gate_margins(row: dict) -> dict[str, float]:
    """Signed distance to each gate, in units of that gate's own tolerance.

    The five exam columns do not share a scale — the off-caption cap is
    0.05 out of 1 and the coherence floor is 0.90 — so "how safely does
    this pass" is only comparable after dividing by the per-gate near-gate
    tolerance the exam already declares in ``GATE_BOUNDS``. Positive is
    inside the gate. ``1.0`` means exactly one tolerance clear, which is
    the line ``near_gate`` draws.
    """
    out: dict[str, float] = {}
    for gate, (key, bound, kind, tol) in GATE_BOUNDS.items():
        value = row.get(key)
        if value is None or gate not in row.get("axis", {}):
            continue
        size = abs(float(value)) if kind == "cap" else float(value)
        slack = (float(bound) - size) if kind == "cap" else (size - float(bound))
        out[gate] = slack / float(tol)
    return out


def worst_margin(row: dict) -> float | None:
    """The gate this row is closest to, signed. Negative means it failed."""
    margins = gate_margins(row)
    return min(margins.values()) if margins else None


# -- the search -----------------------------------------------------------


def score_candidate(
    cand: Candidate, cell: str, *, seed: int = 0, steps: int = 400
) -> dict:
    """One candidate, one cell, one seed, with the leak columns attached."""
    field = CELLS[cell](seed=seed)
    row = score_exam(cand.name, field, steps=steps, seed=seed, **cand.kwargs)
    row["family"] = cand.family
    row["idea"] = cand.idea
    row["seed"] = int(seed)
    row["worst_margin"] = worst_margin(row)
    row["gate_margins"] = gate_margins(row)
    row["leak_frac_wins"] = bool(row["leak_frac"] < LEAK_FRAC_WIN)
    row["wins"] = bool(row["pass"] and row["leak_frac_wins"])
    return row


def search(
    *,
    cells: tuple[str, ...] = tuple(CELLS),
    seeds: tuple[int, ...] = SEEDS,
    steps: int = 400,
) -> list[dict]:
    """Score every candidate on every cell at every seed."""
    return [
        score_candidate(cand, cell, seed=seed, steps=steps)
        for cand in candidates()
        for cell in cells
        for seed in seeds
    ]


def by_candidate(rows: list[dict]) -> dict[str, dict]:
    """Collapse the seed axis: a hit has to hold at every seed.

    ``leak_frac`` on ``--pole_mode hidden`` does not move with the seed at
    all (the target is attainable, so the fit lands on it), but the rollout
    is sampled and the exam verdict does move. Reporting the seed-worst
    margin instead of the seed-0 boolean is the whole reason this is not
    just the pair-exam table with one more column.
    """
    out: dict[str, dict] = {}
    for row in rows:
        entry = out.setdefault(
            row["name"],
            {
                "name": row["name"],
                "family": row["family"],
                "idea": row["idea"],
                "pole_mode": row["pole_mode"],
                "teacher": row["teacher"],
                "cells": {},
            },
        )
        cell = entry["cells"].setdefault(
            row["cell"],
            {
                "seeds": 0,
                "passes": 0,
                "leak_frac": row["leak_frac"],
                "target_leak_frac": row["target_leak_frac"],
                "even_norm": row["even_norm"],
                "odd_norm": row["odd_norm"],
                "same_dir": row["same_dir"],
                "leak_tok": row["leak_tok"],
                "blend_teacher": row["blend_teacher"],
                "worst_margin": row["worst_margin"],
                "near_gate": list(row["near_gate"]),
                "reason": row["reason"],
            },
        )
        cell["seeds"] += 1
        cell["passes"] += int(bool(row["pass"]))
        if row["worst_margin"] is not None and (
            cell["worst_margin"] is None or row["worst_margin"] < cell["worst_margin"]
        ):
            cell["worst_margin"] = row["worst_margin"]
            cell["near_gate"] = list(row["near_gate"])
            cell["reason"] = row["reason"]
    for entry in out.values():
        win = entry["cells"].get(WIN_CELL)
        entry["exam_divergent"] = (
            None if win is None else bool(win["passes"] == win["seeds"])
        )
        entry["leak_frac"] = None if win is None else win["leak_frac"]
        entry["leak_frac_wins"] = (
            None if win is None else bool(win["leak_frac"] < LEAK_FRAC_WIN)
        )
        entry["wins"] = bool(entry["exam_divergent"] and entry["leak_frac_wins"])
        entry["all_cells_pass"] = all(
            cell["passes"] == cell["seeds"] for cell in entry["cells"].values()
        )
        entry["margin"] = None if win is None else win["worst_margin"]
    return out


def hits(summary: dict[str, dict]) -> list[dict]:
    """Everything clearing both halves of the criterion, best margin first."""
    found = [e for e in summary.values() if e["wins"]]
    found.sort(key=lambda e: (-(e["margin"] or 0.0), e["leak_frac"]))
    return found


def frontier(summary: dict[str, dict]) -> dict:
    """The two edges the brief asks for if the hit set comes back empty.

    Reported whether or not it is empty, because the edges are what say
    how wide the win is rather than that there is one.
    """
    divergent_passers = [
        e for e in summary.values() if e["exam_divergent"] and e["leak_frac"] is not None
    ]
    negative = [
        e
        for e in summary.values()
        if e["leak_frac"] is not None and e["leak_frac"] < LEAK_FRAC_WIN
    ]
    best_leak = min(divergent_passers, key=lambda e: e["leak_frac"], default=None)
    # Several candidates tie at the top of the margin column, because four
    # of the five divergent gates saturate. Break that tie on ``leak_frac``
    # so the edge reported is the useful corner and not an ordering
    # accident.
    best_exam = max(
        negative,
        key=lambda e: (e["margin"] if e["margin"] is not None else -9e9, -e["leak_frac"]),
        default=None,
    )
    return {
        "best_leak_frac_among_divergent_passers": None
        if best_leak is None
        else {"name": best_leak["name"], "leak_frac": best_leak["leak_frac"]},
        "best_divergent_among_leak_frac_negative": None
        if best_exam is None
        else {
            "name": best_exam["name"],
            "margin": best_exam["margin"],
            "leak_frac": best_exam["leak_frac"],
            "exam_divergent": best_exam["exam_divergent"],
        },
        "divergent_passers": len(divergent_passers),
        "leak_frac_negative": len(negative),
    }


# -- the cost curve -------------------------------------------------------


GAIN_GRID = (1.0, 1.05, 1.1, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)


def gain_sweep(
    cell: str = WIN_CELL,
    grid: tuple[float, ...] = GAIN_GRID,
    *,
    blind: bool = False,
    seeds: tuple[int, ...] = SEEDS,
    steps: int = 300,
) -> list[dict]:
    """What the exam charges per unit of ``leak_frac``, along one direction.

    ``blind=False`` scales the whole axis; ``blind=True`` scales only the
    part inside ``P_blind``. Both are the same teacher function with a
    different knob, so the two curves are directly comparable and the
    x-axis they share is ``leak_frac`` itself.
    """
    out = []
    for gain in grid:
        kwargs = (
            {"blind_gain": gain} if blind else {"target_scale": gain}
        )
        rows = [
            score_exam(
                f"gain{gain:g}",
                CELLS[cell](seed=seed),
                pole_mode="hidden",
                teacher="faithful_gain",
                steps=steps,
                seed=seed,
                **kwargs,
            )
            for seed in seeds
        ]
        margins = [r["worst_margin"] for r in map(_with_margin, rows)]
        out.append(
            {
                "gain": float(gain),
                "blind": bool(blind),
                "cell": cell,
                "leak_frac": rows[0]["leak_frac"],
                "even_norm": rows[0]["even_norm"],
                "odd_norm": rows[0]["odd_norm"],
                "leak_tok": rows[0]["leak_tok"],
                "seeds": len(rows),
                "passes": sum(1 for r in rows if r["pass"]),
                "worst_margin": min(m for m in margins if m is not None),
                "roll_match_kept": min(r["roll_match_kept"] for r in rows),
                "roll_overlap": min(r["roll_overlap"] for r in rows),
                "roll_off_corpus": max(r["roll_off_corpus"] for r in rows),
                "roll_coherence": min(r["roll_coherence"] for r in rows),
                "roll_swing_kept": min(r["roll_swing_kept"] for r in rows),
            }
        )
    return out


def _with_margin(row: dict) -> dict:
    row["worst_margin"] = worst_margin(row)
    return row


def gain_window(sweep: list[dict]) -> dict:
    """The interval of gains that clears both halves at every seed."""
    good = [
        r
        for r in sweep
        if r["passes"] == r["seeds"] and r["leak_frac"] < LEAK_FRAC_WIN
    ]
    return {
        "low": min((r["gain"] for r in good), default=None),
        "high": max((r["gain"] for r in good), default=None),
        "points": len(good),
        "best": min(good, key=lambda r: r["leak_frac"], default=None),
    }


# -- the claims this cell has to be able to defend ------------------------


def algebra_check(
    cell: str = WIN_CELL, *, gains: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0)
) -> list[dict]:
    """Fitted ``leak_frac`` against the four-caption prediction.

    If these do not agree there is no algebra to reason with and every
    statement in this module's docstring is unsupported.
    """
    field = CELLS[cell]()
    pos, neg, neu = field.poles(0)
    out = []
    for gain in gains:
        row = score_exam(
            f"gain{gain:g}",
            field,
            pole_mode="hidden",
            teacher="faithful_gain",
            target_scale=gain,
            steps=400,
            seed=0,
        )
        common, axis = lm_pair_even_odd(pos, neg, neu)
        even = float(common.norm()) ** 2
        odd = (gain * float(axis.norm())) ** 2
        out.append(
            {
                "gain": float(gain),
                "fitted": row["leak_frac"],
                "predicted": row["target_leak_frac"],
                "closed_form": (even - odd) / (even + odd),
                "even_norm": row["even_norm"],
            }
        )
    return out


def strength_invariance(cell: str = WIN_CELL, *, steps: int = 400) -> list[dict]:
    """Turning the slider up at inference does not move ``leak_frac``.

    ``d±(σ) = σ(c ± a)`` and a cosine is scale-free, so a caption-faithful
    student read at σ = 3 prints the same number it printed at σ = 1. That
    is the reason a *teacher* gain is a different object from an inference
    gain, and it is the sentence this cell would otherwise be handwaving.
    """
    field = CELLS[cell]()
    residual, _room, _final = fit_exam(
        field, pole_mode="hidden", teacher="faithful", steps=steps, seed=0
    )
    return [
        {"sigma": float(sigma)}
        | leftover_bipolar(residual.delta(sigma), residual.delta(-sigma))
        for sigma in (0.5, 1.0, 2.0, 3.0)
    ]


def leftover_saturation(*, gains: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0, 4.0)) -> list[dict]:
    """``leak_tok`` falls with γ; the hidden leftover does not move at all.

    On the ``unused_e`` cell the unpinned attribute ``ĝ`` sits inside
    ``a``, so a gain scales it exactly as fast as the axis. The
    hidden-space ratio is therefore a constant, and every bit of the
    improvement the token-mass column reports is the concept tokens
    saturating the top of the distribution and squeezing the attribute
    out. Published so the gain is not banked as a leftover fix.
    """
    field = CELLS["unused_e"]()
    g_hat = field.unused_dir()
    u_hat = field.short_u()
    out = []
    for gain in gains:
        row = score_exam(
            f"gain{gain:g}",
            field,
            pole_mode="hidden",
            teacher="faithful_gain",
            target_scale=gain,
            steps=400,
            seed=0,
        )
        residual, _room, _final = fit_exam(
            field,
            pole_mode="hidden",
            teacher="faithful_gain",
            target_scale=gain,
            steps=400,
            seed=0,
        )
        d_plus = residual.delta(1.0)
        on_g = float(d_plus @ g_hat)
        on_u = float(d_plus @ u_hat)
        out.append(
            {
                "gain": float(gain),
                "leak_tok": row["leak_tok"],
                "hidden_unused": on_g,
                "hidden_axis": on_u,
                "hidden_ratio": on_g / on_u if abs(on_u) > 1e-8 else None,
            }
        )
    return out


ROW_COS_GRID = (-0.11, -0.05, 0.0, 0.015, 0.07, 0.14)


def row_cos_sweep(*, steps: int = 300, seed: int = 0) -> list[dict]:
    """``faithful`` across the logged energy-v4 pair-cosine range.

    energy-v4's three genre rows log ``cos(pos−neu, neg−neu)`` from −0.11
    to +0.14 and the divergent cell is calibrated to the midpoint. A
    caption teacher realizes that cosine as its ``leak_frac``, so this
    sweep is the same recipe on the same yaml scoring on both sides of the
    criterion depending on which row you look at. It is the reason this
    cell reports ``leak_frac`` as a pair coordinate first and a recipe
    column second.
    """
    out = []
    for cos in ROW_COS_GRID:
        field = divergent_field(probe_cos_target=cos, seed=seed)
        row = score_exam(
            "faithful_raw",
            field,
            pole_mode="hidden",
            teacher="faithful",
            steps=steps,
            seed=seed,
        )
        out.append(
            {
                "probe_cos": float(cos),
                "shared": field.shared_size(),
                "leak_frac": row["leak_frac"],
                "pass": bool(row["pass"]),
                "wins": bool(row["pass"] and row["leak_frac"] < LEAK_FRAC_WIN),
                "roll_match_kept": row["roll_match_kept"],
            }
        )
    return out


def pair_budget(cell: str = WIN_CELL) -> dict:
    """Why the wall is 7% wide, from the field's own numbers.

    On a divergent pair the track split contributes ``½·track²`` to *both*
    halves and cancels. What decides the sign is the shared specificity
    against the part of the pole content that flips.
    """
    field: PairField = CELLS[cell]()
    pos, neg, neu = field.poles(0)
    common, axis = lm_pair_even_odd(pos, neg, neu)
    track_half = 0.5 * float(field.track) ** 2
    return {
        "cell": cell,
        "even_sq": float(common.norm()) ** 2,
        "odd_sq": float(axis.norm()) ** 2,
        "track_in_both": track_half,
        "shared_sq": float(field.shared_size()) ** 2,
        "flipping_sq": float(field.odd_shape().norm()) ** 2,
        "probe_cos": field.probe_cos(0),
        "faithful_leak_frac": teacher_leak_frac(pos, neg, neu),
        "break_even_gain": (
            float(common.norm()) / float(axis.norm().clamp_min(1e-8))
        ),
    }


# The #22 sheet fixtures the compiled board actually reads ``leak_frac``
# off (``bipolar_from``: leftover, then gender, then the exam row). Not
# every candidate here is expressible on that cell — it has no blind
# projector knob and no ``dual_band`` over-drive — and the ones that are
# not get ``None`` rather than a guessed number.
SHEET_TEACHERS = frozenset({"faithful", "pair_odd", "faithful_gain"})


def sheet_leak(cand: Candidate, *, steps: int = 400, seed: int = 0) -> dict | None:
    """``leak_frac`` on the sheet fixtures, so the hit holds on either read.

    The criterion names ``leftover_bipolar`` without naming a fixture, and
    the compiled board's ``leak_frac`` column is the #22 sheet cell's, not
    the divergent exam cell's — which is why the board prints +0.03 for
    ``faithful_raw`` where the divergent cell prints +0.015. Same
    quantity, different pair. A recipe that only clears the criterion on
    one of the two is not a hit, so both are reported.
    """
    if cand.kwargs.get("teacher", "pair_odd") not in SHEET_TEACHERS:
        return None
    if cand.kwargs.get("blind_gain", 1.0) != 1.0:
        return None
    kwargs = {
        key: value
        for key, value in cand.kwargs.items()
        if key in ("pole_mode", "teacher", "common_beta", "target_scale")
    }
    field = leaky_field()
    leftover = score_sheet(
        cand.name, field, leak_dir=field.leak_e(), steps=steps, seed=seed, **kwargs
    )
    clean = score_sheet(
        cand.name, gender_like_field(), steps=steps, seed=seed, **kwargs
    )
    return {
        "leftover_leak_frac": leftover["leak_frac"],
        "leftover_same_dir": leftover["same_dir"],
        "leftover_leak_tok": leftover["leak_tok"],
        "leftover_pass": bool(leftover["pass"]),
        "leftover_garble": leftover["garble"],
        "gender_leak_frac": clean["leak_frac"],
        "gender_pass": bool(clean["pass"]),
        "board_leak_frac": leftover["leak_frac"],
        "board_wins": bool(leftover["leak_frac"] < LEAK_FRAC_WIN),
    }


def sheet_table(*, steps: int = 400, seed: int = 0) -> dict[str, dict | None]:
    return {
        cand.name: sheet_leak(cand, steps=steps, seed=seed) for cand in candidates()
    }


def cell_notes() -> dict[str, str]:
    return dict(CELL_IS)


def live_pair_cos() -> dict:
    return {k: v for k, v in LIVE_PAIR_COS.items()}
