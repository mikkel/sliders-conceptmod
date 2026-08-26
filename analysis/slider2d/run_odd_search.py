#!/usr/bin/env python3
"""Search for a TRAIN recipe that passes exam_divergent with leak_frac < 0."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from analysis.slider2d.exam import (
    EXAM_COHERENCE,
    EXAM_LEAK_LOCK,
    EXAM_MATCH_KEPT,
    EXAM_ROLL_OFF_MAX,
    EXAM_ROLL_OVERLAP,
    EXAM_ROLL_SWING,
)
from analysis.slider2d.odd_search import (
    GAIN_GRID,
    LEAK_FRAC_WIN,
    SEEDS,
    WIN_CELL,
    algebra_check,
    by_candidate,
    candidates,
    cell_notes,
    frontier,
    gain_sweep,
    gain_window,
    hits,
    leftover_saturation,
    live_pair_cos,
    pair_budget,
    row_cos_sweep,
    search,
    sheet_table,
    sibling_concordance,
    strength_invariance,
)

DEFAULT_OUT = _REPO / "docs" / "lm-odd-leak-frac"

FAMILY_COLOR = {
    "baseline": "#2c3e50",
    "trap": "#c0392b",
    "shrink the even half": "#8e44ad",
    "lengthen the odd half": "#1e8449",
    "lengthen the blind half only": "#b9770e",
    "delete the blend": "#16a085",
}


def _f(value, spec: str, empty: str = "N/A") -> str:
    if value is None:
        return empty
    return format(float(value), spec)


def plot_criterion(summary: dict, path: Path) -> None:
    """The criterion as a plane: leak_frac against how safely it passes.

    One quadrant is the win. Nothing else on the panel matters, so the
    other three are only there to show what used to fill them.
    """
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    plotted = [
        e
        for e in summary.values()
        if e["leak_frac"] is not None and e["margin"] is not None
    ]
    ax.fill_betweenx([0.0, 6.6], -1.05, LEAK_FRAC_WIN, color="#1e8449", alpha=0.06)
    seen = set()
    for entry in plotted:
        color = FAMILY_COLOR.get(entry["family"], "#7f8c8d")
        label = entry["family"] if entry["family"] not in seen else None
        seen.add(entry["family"])
        ax.scatter(
            entry["leak_frac"],
            entry["margin"],
            c=color,
            s=64,
            zorder=3,
            marker="o" if entry["wins"] else "X",
            edgecolors="white",
            linewidths=0.7,
            label=label,
        )
    # Two corners are crowded — the ``leak_frac ≈ −1`` trap pile and the
    # passing cluster near 0 — and every point in each is a variation on one
    # knob. Labelling all of them turns the panel into a smudge, so a name
    # is dropped when it would land on one already placed.
    placed: list[tuple[float, float]] = []
    for entry in sorted(plotted, key=lambda e: (not e["wins"], e["leak_frac"])):
        x, y = entry["leak_frac"], entry["margin"]
        if any(abs(x - px) < 0.075 and abs(y - py) < 0.55 for px, py in placed):
            continue
        placed.append((x, y))
        ax.annotate(
            entry["name"],
            (x, y),
            fontsize=6.4,
            xytext=(5, 4),
            textcoords="offset points",
            color=FAMILY_COLOR.get(entry["family"], "#7f8c8d"),
        )
    ax.axvline(LEAK_FRAC_WIN, color="#7f8c8d", ls=":", lw=0.9)
    ax.axhline(0.0, color="#7f8c8d", ls="-", lw=1.0)
    ax.axhline(1.0, color="#bdc3c7", ls="--", lw=0.8)
    ax.annotate(
        "the gate", (-1.02, 0.12), fontsize=7, color="#7f8c8d"
    )
    ax.annotate(
        "one tolerance clear", (-1.02, 1.12), fontsize=7, color="#95a5a6"
    )
    ax.set_xlim(-1.05, 0.22)
    ax.set_ylim(-3.6, 4.2)
    ax.set_xlabel("leak_frac = cos(d₊, d₋) of the fitted ±1 student")
    ax.set_ylabel("worst exam_divergent gate, in that gate's own tolerances")
    ax.set_title(
        "circles clear both halves of the criterion; the shaded quadrant is the win"
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, loc="lower right", title="family", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


FLOOR = -4.0


def plot_cost(whole: list[dict], blind: list[dict], path: Path) -> None:
    """What the exam charges per unit of leak_frac, along two directions.

    The γ = 10 end of each curve is tens of tolerances past the gate — the
    ±1 ends are off every caption by then — so the panel clips at
    ``FLOOR`` and marks what it clipped rather than flattening everything
    that still matters into one line at the top.
    """
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for sweep, color, label in (
        (whole, "#1e8449", "scale the whole axis  (mid ± γ·a)"),
        (blind, "#b9770e", "scale only P_blind(a)"),
    ):
        xs = [r["leak_frac"] for r in sweep]
        ys = [max(FLOOR, r["worst_margin"]) for r in sweep]
        ax.plot(xs, ys, "o-", color=color, label=label, ms=4.5)
        for row, y in zip(sweep, ys):
            if row["gain"] not in (1.0, 1.5, 2.0, 3.0, 10.0):
                continue
            clipped = row["worst_margin"] < FLOOR
            ax.annotate(
                f"γ={row['gain']:g}" + (f" ({row['worst_margin']:.0f})" if clipped else ""),
                (row["leak_frac"], y),
                fontsize=6.6,
                xytext=(4, -10 if clipped else 5),
                textcoords="offset points",
                color=color,
            )
    ax.axhline(0.0, color="#7f8c8d", ls="-", lw=1.0, label="the gate")
    ax.axvline(LEAK_FRAC_WIN, color="#7f8c8d", ls=":", lw=0.9)
    ax.set_ylim(FLOOR - 0.4, 4.0)
    ax.set_xlabel("leak_frac bought  ←  more negative")
    ax.set_ylabel("worst exam_divergent gate, seed-worst over 6 seeds")
    ax.set_title(
        "both directions buy leak_frac; the whole axis buys three times more "
        "before the exam goes"
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_rows(sweep: list[dict], path: Path) -> None:
    """leak_frac of the caption teacher across energy-v4's own row spread."""
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    xs = [r["probe_cos"] for r in sweep]
    lo, hi = live_pair_cos()["energy-v4"]
    ax.axhspan(-0.16, 0.0, color="#1e8449", alpha=0.09)
    ax.axhspan(0.0, 0.19, color="#c0392b", alpha=0.06)
    ax.plot(xs, xs, "--", color="#bdc3c7", lw=1.0, label="the pair's own logged cosine")
    ax.plot(
        xs,
        [r["leak_frac"] for r in sweep],
        "-",
        color="#2c3e50",
        lw=1.2,
        label="faithful, fitted leak_frac",
    )
    for row in sweep:
        ax.scatter(
            row["probe_cos"],
            row["leak_frac"],
            c="#1e8449" if row["wins"] else "#c0392b",
            s=70,
            zorder=4,
            edgecolors="white",
            linewidths=0.7,
        )
    for cos, tag in ((lo, "energy-v4's lowest row"), (hi, "and its highest")):
        ax.axvline(cos, color="#2c3e50", ls="-.", lw=0.9)
        ax.annotate(
            tag, (cos, 0.165), fontsize=7, rotation=90, va="top",
            xytext=(3, 0), textcoords="offset points", color="#2c3e50",
        )
    ax.axhline(0.0, color="#7f8c8d", ls="-", lw=1.0)
    ax.annotate(
        "leak_frac < 0 — clears the criterion",
        (0.148, -0.148), fontsize=7.5, color="#1e8449", ha="right",
    )
    ax.annotate(
        "leak_frac ≥ 0 — does not",
        (-0.015, 0.175), fontsize=7.5, color="#c0392b", ha="right",
    )
    ax.set_xlim(-0.125, 0.155)
    ax.set_ylim(-0.16, 0.19)
    ax.set_xlabel("cos(pos−neu, neg−neu) — a property of the prompt row")
    ax.set_ylabel("leak_frac of the caption teacher")
    ax.set_title("the same recipe on the same yaml, on both sides of the criterion")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


BOARD_HEADER = (
    "| recipe | family | idea | leak_frac *(divergent)* | leak_frac *(board / #22 sheet)* | "
    "even ‖·‖ | odd ‖·‖ | exam_divergent | worst gate | exam_close | exam_unused_e | "
    "sheet leftover | verdict |\n"
    "|---|---|---|---:|---:|---:|---:|---|---:|---|---|---|---|"
)


def board_row(entry: dict, sheet: dict | None) -> str:
    cells = entry["cells"]

    def verdict(cell: str) -> str:
        row = cells.get(cell)
        if row is None:
            return "N/A"
        if row["passes"] == row["seeds"]:
            tag = "PASS"
            if row["near_gate"]:
                tag += " *(near " + ", ".join(row["near_gate"]) + ")*"
            return tag
        if row["passes"] == 0:
            return "FAIL"
        return f"FAIL *({row['passes']}/{row['seeds']} seeds)*"

    win = cells.get(WIN_CELL, {})
    return (
        f"| `{entry['name']}` | {entry['family']} | {entry['idea']} | "
        f"{_f(entry['leak_frac'], '+.3f')} | "
        f"{_f(None if sheet is None else sheet['board_leak_frac'], '+.3f', '—')} | "
        f"{_f(win.get('even_norm'), '.3f')} | "
        f"{_f(win.get('odd_norm'), '.3f')} | {verdict('divergent')} | "
        f"{_f(entry['margin'], '+.1f')} | {verdict('close')} | "
        f"{verdict('unused_e')} | "
        f"{'—' if sheet is None else ('PASS' if sheet['leftover_pass'] else 'FAIL')} | "
        f"{'**HIT**' if entry['wins'] else '—'} |"
    )


def write_report(blob: dict, path: Path) -> None:
    summary = blob["summary"]
    order = [c.name for c in candidates()]
    rows = [summary[name] for name in order if name in summary]
    won = blob["hits"]
    front = blob["frontier"]
    budget = blob["pair_budget"]
    whole = blob["gain_sweep"]
    blind = blob["blind_sweep"]
    window = blob["gain_window"]
    blind_window = blob["blind_window"]
    lo, hi = blob["live_pair_cos"]["energy-v4"]
    sat = blob["leftover_saturation"]
    sheet = blob["sheet"]
    best = won[0] if won else None
    best_sheet = None if best is None else sheet.get(best["name"])
    # Smallest gain whose sheet-leftover verdict flips, for the caveat below.
    flip = next(
        (
            f"{row['gain']:g}"
            for row in whole
            if (bag := sheet.get(f"faithful_gain_{row['gain']:g}")) is not None
            and bag["leftover_pass"]
        ),
        "N/A",
    )

    lines = [
        "# Odd enough for `leak_frac < 0`, without becoming a midpoint",
        "",
        "Generated by `analysis/slider2d/run_odd_search.py`. CPU only, no Hub,",
        "no GPU, no Music 3 weights. **The live default is unchanged**",
        "(`--lm_target v9` / `--pole_mode hidden`), no v4 yaml is rewritten, and",
        "no live listen is claimed for anything on this page.",
        "",
        "The criterion is a gate, not a vibe:",
        "",
        "> `exam_divergent` is True **and** `leak_frac < 0`,",
        "> where `leak_frac = cos(d₊, d₋)` of the fitted ±1 student",
        "> (`leftover_bipolar` in `slider_targets.py`).",
        "",
        "Before this cell the only row clearing both was `faithful_attrs`, which",
        "is a caption rewrite rather than a loss. Every recipe that had pushed",
        "`leak_frac` negative from the loss side — `pair_odd_midpoint`,",
        "`pair_odd_sub_e`, `dual_band_midpoint`, hold-ê, `hub`, project — landed",
        "at `leak_frac ≈ −1` and **failed** the divergent exam. Every recipe that",
        "passed it sat between +0.03 and +0.12. Two clusters, and a wall between",
        "them.",
        "",
        "**Which fixture `leak_frac` is read off matters, so this page reports",
        "both.** `leftover_bipolar` names a quantity, not a pair. The compiled",
        "board's column (`bipolar_from`) prefers the #22 sheet leftover cell,",
        "which is where +0.03 for `faithful_raw` comes from; the divergent exam",
        "cell prints +0.015 for the same recipe, because it is a different pair.",
        "The `HIT` column follows the criterion as written — the divergent cell —",
        "and the board's fixture is printed beside it, so a result that only",
        "works on one reading is visible as one.",
        "",
        "## The answer, first",
        "",
    ]
    if best is None:
        lines += [
            "**The hit set is empty.** The frontier is below.",
            "",
        ]
    else:
        lines += [
            f"**The set is not empty: {len(won)} recipes clear both halves.** The",
            f"widest-margin one is `{best['name']}` — `{best['idea']}` — at",
            f"`leak_frac {best['leak_frac']:+.3f}` on the divergent cell and",
            f"`{best_sheet['board_leak_frac']:+.3f}` on the board's sheet fixture,"
            if best_sheet
            else "and on the divergent cell,",
            f"passing every divergent gate at all {len(SEEDS)} seeds with"
            f" {best['margin']:.1f} tolerances of room on the tightest one, and",
            "passing the close and `unused_e` cells as well.",
            "",
            "**But there is no wall, and that is the more useful half of the",
            "result.** `leak_frac` is not something a loss discovers. For a shared",
            "±1 residual it is an exact function of the teacher's even and odd",
            "halves, readable off four captions before a single step:",
            "",
            "```",
            "d± = β·c ± γ·a          c = ½(h₊+h₋) − h0,  a = ½(h₊−h₋)",
            "leak_frac = (β²‖c‖² − γ²‖a‖²) / (β²‖c‖² + γ²‖a‖²)",
            "```",
            "",
            "(`teacher_leak_frac` in `slider_targets.py`.) So the honest question",
            "was never *whether* something can be odd enough — anything with",
            f"`γ/β > {budget['break_even_gain']:.3f}` on this pair is — but what the",
            "exam charges for it, and which direction of \"more odd\" is cheapest.",
            "",
        ]
    lines += [
        "## Why the wall is 7% wide",
        "",
        "On a divergent pair the two tracks are separate features:",
        "`a = ½·track·(p̂−q̂) + o` and `c = ½·track·(p̂+q̂) + shared·ŝ`. Both halves",
        f"therefore carry the same `½·track² = {budget['track_in_both']:.2f}`, and it",
        "subtracts straight out of `leak_frac`. What is left deciding the sign is",
        "the shared specificity against the part of the pole content that flips:",
        "",
        f"| | value |",
        f"|---|---:|",
        f"| `‖c‖²` | {budget['even_sq']:.4f} |",
        f"| `‖a‖²` | {budget['odd_sq']:.4f} |",
        f"| the track, in **both** | {budget['track_in_both']:.4f} |",
        f"| `shared²` (even, after the track cancels) | {budget['shared_sq']:.4f} |",
        f"| `‖o‖²` (odd, after the track cancels) | {budget['flipping_sq']:.4f} |",
        f"| break-even gain `‖c‖/‖a‖` | {budget['break_even_gain']:.4f} |",
        "",
        "That is the whole wall: 1.40 against 1.30. A caption-faithful teacher",
        f"lands at `{budget['faithful_leak_frac']:+.4f}`, which is not a coincidence",
        "— it is the pair's own `cos(pos−neu, neg−neu)`, the number the live",
        "trainer already prints.",
        "",
        "### So on a caption teacher, the sign belongs to the *row*",
        "",
        f"energy-v4 logs that cosine from {lo:+.2f} to {hi:+.2f} across its three",
        "genre rows, and the divergent cell is calibrated to the midpoint",
        f"({0.5 * (lo + hi):+.3f}). Sweeping the cell across that logged range,",
        "with `faithful` and nothing else changed:",
        "",
        "| row cosine | `shared` | fitted `leak_frac` | exam_divergent | clears both |",
        "|---:|---:|---:|---|---|",
    ]
    for row in blob["row_cos_sweep"]:
        lines.append(
            f"| {row['probe_cos']:+.3f} | {row['shared']:.3f} | "
            f"{row['leak_frac']:+.3f} | {'PASS' if row['pass'] else 'FAIL'} | "
            f"{'**yes**' if row['wins'] else 'no'} |"
        )
    lines += [
        "",
        "Same recipe, same yaml, both sides of the criterion depending on which",
        "prompt row you read. `leak_frac` is a pair coordinate first and a recipe",
        "column second, and a board that sorts recipes by it is partly sorting",
        "prompt rows.",
        "",
        "![rows](lm-odd-leak-frac/rows.png)",
        "",
        "## The three directions of \"more odd\", and their price",
        "",
        "| teacher | what it does | on a close pair |",
        "|---|---|---|",
        "| `faithful_gain` | `t± = mid ± γ·a` — keep the caption pair's own "
        "midpoint, scale only the axis | a bigger slider, `c` intact |",
        "| `faithful_gain` *(blind)* | the same gain restricted to `P_blind` — the "
        "half of the axis a next-token KL cannot read at all | the axis *is* "
        "mostly that band there |",
        "| `faithful_common_agree` | `t± = h0 + agree(h₊,h₋) ± a` — delete the "
        "part of `c` no single caption occupies | **identical to `faithful`**: on "
        "one song `agree = c` |",
        "",
        "`lm_blend_guard` admits `faithful_gain` at every γ, structurally:",
        "`‖t₊ − pos‖ = (γ−1)‖a‖` is below `‖t₊ − mid‖ = γ‖a‖` for all of them. So",
        "unlike every `sub_e` variant it cannot drift into a blend — it only ever",
        "moves the two ends further apart around a midpoint it never touches.",
        "",
        "## The board",
        "",
        f"Every candidate, on all three pair-exam cells, at {len(SEEDS)} seeds. A",
        "cell passes only if it passes at **every** seed. `worst gate` is the",
        "tightest divergent gate in units of that gate's own near-gate tolerance,",
        "seed-worst — `+1.0` is exactly one tolerance clear, which is the line",
        "`near_gate` draws.",
        "",
        "The two `leak_frac` columns are the same quantity on two different",
        "pairs. `—` in the sheet columns is a candidate that fixture cannot",
        "express (it has no blind projector and no `dual_band` over-drive),",
        "reported missing rather than guessed.",
        "",
        BOARD_HEADER,
    ]
    lines += [board_row(entry, sheet.get(entry["name"])) for entry in rows]
    lines += [
        "",
        "![criterion](lm-odd-leak-frac/criterion.png)",
        "",
        "Five things on that board are worth reading twice.",
        "",
        "**`common_beta` was never swept.** The board had scored β = 0 (the",
        "`pair_odd_midpoint` trap) and β = 1 (`faithful`) and nothing between",
        "them, and the entire interior passes: β = 0.1 still clears the divergent",
        "exam at `leak_frac −0.98`. That is the Hub's own κ-anchor geometry —",
        "`lm_anchor_kappa` solves for exactly this κ from a declared",
        "`leakage_floor` — so \"declare the `leak_frac` you want and get it\" has",
        "been in the trainer the whole time. It is also why it is a poor",
        "criterion on its own.",
        "",
        "**`faithful_common_agree` fails, and it is the useful negative.** It is",
        "the one candidate here that removes the blend rather than outweighing",
        "it, and on a close pair it is provably the caption pair. On the divergent",
        "pair it fails at the same gate and the same value as",
        "`pair_odd_midpoint`. The reason is that half of the + pole's own track",
        "is sitting inside `c`: strip it and the pole says pop-punk no louder",
        "than it says slammed, and the continuation stops matching. **On a",
        "divergent pair the blend is load-bearing**, which is why every β → 0",
        "recipe on the trap list fails and why the only affordable direction is",
        "to lengthen the axis rather than shorten the common term.",
        "",
        "**The blind-band restriction is the expensive direction.** It looks like",
        "the free lunch — extra axis in the band the scored distribution cannot",
        "read — and it is the worse buy. Inside the window where every seed still",
        f"passes, scaling the whole axis reaches `leak_frac"
        f" {window['best']['leak_frac']:+.3f}` and the blind-only variant only",
        f"reaches `{blind_window['best']['leak_frac']:+.3f}`, a factor of"
        f" {blind_window['best']['leak_frac'] / window['best']['leak_frac']:.2f}."
        " The reason is that the",
        "delivery content the residual stream carries forward lands on the *axis",
        "adjective* and competes with the pole's own track word, while scaling",
        "the whole axis scales that track word too.",
        "",
        "**And under `semantic_kl` the blind over-drive does not arrive at all.**",
        "`blind_gain_3_semantic_kl` prints the same `leak_frac` as",
        "`semantic_kl_poles` to four decimals — the target asked for three times",
        "the delivery content and the loss has exactly zero gradient there, so",
        "nothing moved. Under `dual_band`, which supplies one, the same target",
        "lands at a genuinely different number. That is the #35 blind-band story",
        "reproduced from the target side, and it is why a teacher and a loss are",
        "not interchangeable knobs.",
        "",
        "**The traps are seed-fragile, and so is anything near them.**",
        "`pair_odd_midpoint` and `dual_band_midpoint` clear the divergent exam at",
        "2 of 6 seeds and `faithful_common_agree` at the same 2. Their published",
        "verdict is FAIL because a cell has to pass at every seed here, but a",
        "single-seed board would have printed some of them green. That is the",
        "reason this cell reports seed counts rather than a boolean.",
        "",
        "## The cost curve",
        "",
        "Both curves are the same teacher function with a different knob, so the",
        "x-axis they share is `leak_frac` itself.",
        "",
        "| γ | leak_frac | worst gate | overlap | same-words | off-caption | coherence | swing | seeds passed |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in whole:
        lines.append(
            f"| {row['gain']:g} | {row['leak_frac']:+.3f} | "
            f"{row['worst_margin']:+.1f} | {row['roll_overlap']:.3f} | "
            f"{row['roll_match_kept']:.3f} | {row['roll_off_corpus']:.3f} | "
            f"{row['roll_coherence']:.3f} | {row['roll_swing_kept']:+.2f} | "
            f"{row['passes']}/{row['seeds']} |"
        )
    lines += [
        "",
        f"The window where `mid ± γ·a` clears both halves at every seed is",
        f"**γ ∈ [{_f(window['low'], 'g')}, {_f(window['high'], 'g')}]** on this grid,",
        f"and the same window for the blind-only variant is",
        f"**[{_f(blind_window['low'], 'g')}, {_f(blind_window['high'], 'g')}]**.",
        "It is bounded on both sides and the cell says so: below the break-even",
        f"gain {budget['break_even_gain']:.3f} the sign has not flipped yet, and far",
        "enough above it the ±1 ends are driven past anything a caption reaches",
        "and the continuation garbles — at γ = 10 the off-caption column goes",
        f"from 0.000 to {whole[-1]['roll_off_corpus']:.3f}, which is the #22 sheet",
        "cell's garble device firing on purpose.",
        "",
        "![cost](lm-odd-leak-frac/cost.png)",
        "",
        "## Three other searches found the same curve",
        "",
        "Three sibling searches were open on this criterion at the same time,",
        "each describing a different mechanism: [#37](https://github.com/mikkel/"
        "sliders-conceptmod/pull/37) caps the common term at `0.9‖odd‖`,",
        "[#38](https://github.com/mikkel/sliders-conceptmod/pull/38) sweeps",
        "`--common_beta`, and [#40](https://github.com/mikkel/sliders-conceptmod/"
        "pull/40) adds a penalty on the *student's* even half and leaves the",
        "teacher a real caption. A teacher scale, a target cap and a loss term",
        "are not the same object.",
        "",
        "They land on one curve anyway. Inverting the closed form for the",
        "odd-over-even ratio each published `leak_frac` implies, and re-measuring",
        "`mid ± γ·a` at that ratio:",
        "",
        "| source | mechanism | reported `leak_frac` | implied odd/even | this cell at the same ratio |",
        "|---|---|---:|---:|---:|",
    ]
    for row in blob["sibling_concordance"]:
        lines.append(
            f"| {row['source']} | {row['mechanism']} | "
            f"{row['reported_leak_frac']:+.3f} | "
            f"{row['implied_odd_over_even']:.3f} | {row['closed_form']:+.3f} |"
        )
    lines += [
        "",
        "Every one agrees to the precision it was quoted at, and the",
        "mechanism-to-ratio map is closed form in each case: #38's β gives",
        "`1/β` (0.90 → 1.111, 0.60 → 1.667, 0.50 → 1.999) and #40's even-ridge",
        "weight `w` gives `1 + w/2` (0.25 → 1.125, 1 → 1.500, 4 → 3.004), which",
        "is just the ridge solution for shrinking one half of a quadratic.",
        "",
        "None of that makes any of them wrong — they are three correct routes to",
        "the same point, and #40's is the only one that keeps the teacher a real",
        "caption, which is a real advantage this recipe does not have. It does",
        "mean the four PRs are **not** four independent hits. They are one",
        "one-parameter family reported four times, and the exam cannot tell the",
        "members apart because it only ever sees the point they land on. If a",
        "board ends up carrying all four as separate rows, that is four names",
        "for one dial.",
        "",
        "## What a hit here does not mean",
        "",
        "### `leak_frac` is a ratio, and this buys it from the denominator",
        "",
        "`‖even‖` is bit-identical at every γ — the board prints it. The same",
        "blend is still in the residual; it is outweighed, not removed. A",
        "negative `leak_frac` from `faithful_gain` says *the axis now outweighs",
        "the same-direction content*, and nothing at all about that content",
        "having gone anywhere.",
        "",
        "This is also why the recipe is a different object from turning the",
        "slider up at inference. A student read at σ has `d± = σ(c ± a)`, and a",
        "cosine is scale-free:",
        "",
        "| σ | leak_frac | even ‖·‖ | odd ‖·‖ |",
        "|---:|---:|---:|---:|",
    ]
    for row in blob["strength_invariance"]:
        lines.append(
            f"| {row['sigma']:g} | {row['leak_frac']:+.4f} | "
            f"{row['even_norm']:.3f} | {row['odd_norm']:.3f} |"
        )
    lines += [
        "",
        "An inference gain cannot move `leak_frac` by construction. A *teacher*",
        "gain scales the odd half without the even half, so it can. Those are",
        "genuinely different deltas and only one of them is on this page.",
        "",
        "### The leftover column improves for a reason that is not real",
        "",
        "On the `unused_e` cell `leak_tok` falls monotonically with γ and crosses",
        f"the {EXAM_LEAK_LOCK} lock the sheet cell scores. None of it is a",
        "leftover drop. The unpinned attribute `ĝ` sits inside `a`, so a gain",
        "scales it exactly as fast as the axis, and the hidden-space ratio does",
        "not move at all:",
        "",
        "| γ | `leak_tok` (token mass) | `d₊·ĝ` | `d₊·û` | hidden ratio |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in sat:
        lines.append(
            f"| {row['gain']:g} | {row['leak_tok']:+.3f} | "
            f"{row['hidden_unused']:.3f} | {row['hidden_axis']:.3f} | "
            f"{row['hidden_ratio']:.4f} |"
        )
    lines += [
        "",
        "Every bit of that improvement is the concept tokens saturating the top",
        "of the next-token distribution and squeezing the attribute out. A",
        "mass-ratio leak column saturates under any gain; charted, not banked.",
        "This recipe is **orthogonal** to the leftover question and composes with",
        "the pair-aware leftover gate rather than replacing it.",
        "",
        "The consequence is worth stating in the strongest form available,",
        "because it is a mark against this page's own result: **on the #22 sheet",
        f"leftover cell, `faithful_gain` flips FAIL to PASS from γ ≥ {flip}, and",
        "that pass is this artifact.** `faithful_raw` fails that cell on the",
        f"{EXAM_LEAK_LOCK} leak lock at "
        f"{_f(sheet['faithful_raw']['leftover_leak_tok'], '+.3f')} and the gain",
        "walks it under the lock without removing anything. Do not read the",
        "sheet-leftover column on the gain rows as a leftover fix. The one",
        "column on that cell the gain does not flatter is `garble`, and that",
        "one gets *worse* with γ.",
        "",
        "### And the exam is doing less work here than elsewhere",
        "",
        "Across this whole family four of the five divergent gates are pinned at",
        "their ceiling — overlap 1.000, off-caption 0.000, coherence 1.000, swing",
        "at the ceiling. Only `same_words` moves, and it is essentially asking",
        "whether the pole's own genre word still beats its intensity adjective.",
        "So a claimed win in this region rests on one gate, and the honest",
        "reading of the margin column is \"how much room does that one gate",
        "have\", not \"how good does this sound\".",
        "",
        "## The frontier",
        "",
        f"- Divergent passers: **{front['divergent_passers']}**; recipes with",
        f"  `leak_frac < 0`: **{front['leak_frac_negative']}**.",
    ]
    bl = front["best_leak_frac_among_divergent_passers"]
    be = front["best_divergent_among_leak_frac_negative"]
    lines += [
        f"- Best `leak_frac` among divergent passers: "
        f"`{bl['name']}` at {bl['leak_frac']:+.3f}." if bl else
        "- Best `leak_frac` among divergent passers: none.",
        f"- Best divergent margin among `leak_frac < 0`: "
        f"`{be['name']}` at {be['margin']:+.1f} tolerances "
        f"(`leak_frac {be['leak_frac']:+.3f}`)." if be else
        "- Best divergent margin among `leak_frac < 0`: none.",
        "",
        "## What is wired live",
        "",
        "`--lm_target faithful_gain`, non-default, requiring an explicit",
        "`--target_scale` (at 1.0 it is byte-for-byte `--lm_target faithful`, so",
        "the trainer refuses rather than shipping a renamed duplicate):",
        "",
        "```bash",
        "python conceptmod/textsliders/train_lm_slider_music3.py \\",
        "  --prompts_file prompts/prompts-energy-v4.yaml \\",
        "  --lm_target faithful_gain --target_scale 1.5 --pole_mode hidden",
        "```",
        "",
        "No `--pole_mode` is added, and the default stays `--lm_target v9` /",
        "`--pole_mode hidden`. The blind-band variant is **not** wired live: it is",
        "the control that shows the cheap direction is cheap, and it needs a",
        "readout projector to mean anything.",
        "",
        "## The claims this page has to be able to defend",
        "",
        "`tests/test_lm_odd_leak_frac.py` checks each of these, and they are the",
        "reason the algebra above is quoted rather than asserted.",
        "",
        "| claim | check |",
        "|---|---|",
    ]
    for row in blob["algebra_check"]:
        lines.append(
            f"| fitted `leak_frac` at γ={row['gain']:g} equals the four-caption "
            f"prediction | {row['fitted']:+.6f} vs {row['closed_form']:+.6f} |"
        )
    lines += [
        "| `faithful_gain` at γ=1 is the caption pair | exact tensor equality |",
        "| `lm_common_agree` on a close pair is `c` | exact, to 1e-6 |",
        "| `lm_blend_guard` admits `faithful_gain` at every γ | structural |",
        "| `‖even‖` does not move with γ | exact |",
        "| the hit is negative on the board's fixture too | both `leak_frac` columns |",
        "",
        "## What this cell still cannot see",
        "",
        "- **Whether any of it sounds better.** `leak_frac` has never been shown",
        "  to order a live listen, and the live run this repo liked most",
        "  (`energy-lm-v18`, caption poles + KL) is *positive* on it. This page",
        "  reports a stated criterion being met; it does not claim ears.",
        "- **A real Qwen hidden state.** Nine dimensions, ten tokens, one frozen",
        "  linear readout and one hand-written transition. The break-even gain",
        f"  {budget['break_even_gain']:.3f} is a property of this field's calibration",
        "  to the logged energy-v4 cosine, not a number to type into a run.",
        "- **The blend-versus-shared split at scale.** `lm_common_agree` asks a",
        "  per-coordinate question, and this fixture's basis is interpretable by",
        "  construction. In 3584 noisy dimensions the sign test is a soft AND",
        "  whose behaviour on real states is untested here. It is on the board",
        "  because it *fails*, which is a claim that survives the doubt.",
        "",
        "## The cheapest next live measurement",
        "",
        "It needs no training and it is the one that would falsify the whole",
        "page. Encode the three energy-v4 rows' pole and neutral captions and",
        "print `cos(h₊−h0, h₋−h0)` per row. If the live spread really does",
        f"straddle zero the way the log says ({lo:+.2f} … {hi:+.2f}), then",
        "`leak_frac` is a per-row property on the live model too and no recipe",
        "column should be sorted by it without saying which row it came from.",
        "",
        "## Related cells",
        "",
        "- [lm-pair-exam.md](lm-pair-exam.md) — the exam this criterion's first",
        "  half comes from.",
        "- [lm-2d-scoreboard.md](lm-2d-scoreboard.md) — the compiled board, where",
        "  the `leak_frac` column lives.",
        "- [lm-sheet-goodhart.md](lm-sheet-goodhart.md) — the single-token sheet",
        "  cell that scores leftover leak, and the garble device γ trips at 10.",
        "- [lm-highd-leftover.md](lm-highd-leftover.md) — leftover ê and the",
        "  trainer's `c+` ceiling.",
        "",
        "## How to run",
        "",
        "```bash",
        "PYTHONPATH=. python analysis/slider2d/run_odd_search.py --out docs/lm-odd-leak-frac",
        "PYTHONPATH=. pytest tests/test_lm_odd_leak_frac.py -q",
        "```",
        "",
        "CPU only. No Hub, no GPU, no Music 3 weights.",
        "",
        f"Seeds `{list(SEEDS)}`, `{blob['steps']}` Adam steps, "
        f"`{blob['sweep_steps']}` on the sweeps.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--sweep-steps", type=int, default=300)
    args = parser.parse_args(argv)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    rows = search(steps=args.steps)
    summary = by_candidate(rows)
    whole = gain_sweep(blind=False, steps=args.sweep_steps)
    blind = gain_sweep(blind=True, steps=args.sweep_steps)
    blob = {
        "criterion": {
            "cell": WIN_CELL,
            "leak_frac_below": LEAK_FRAC_WIN,
            "seeds": list(SEEDS),
            "gates": {
                "overlap": EXAM_ROLL_OVERLAP,
                "match_kept": EXAM_MATCH_KEPT,
                "off_caption": EXAM_ROLL_OFF_MAX,
                "coherence": EXAM_COHERENCE,
                "swing": EXAM_ROLL_SWING,
                "leak_on_sheet_cell": EXAM_LEAK_LOCK,
            },
        },
        "steps": args.steps,
        "sweep_steps": args.sweep_steps,
        "cells": cell_notes(),
        "live_pair_cos": live_pair_cos(),
        "pair_budget": pair_budget(),
        "summary": summary,
        "hits": hits(summary),
        "frontier": frontier(summary),
        "gain_grid": list(GAIN_GRID),
        "gain_sweep": whole,
        "blind_sweep": blind,
        "gain_window": gain_window(whole),
        "blind_window": gain_window(blind),
        "sheet": sheet_table(steps=args.steps),
        "sibling_concordance": sibling_concordance(),
        "row_cos_sweep": row_cos_sweep(steps=args.sweep_steps),
        "algebra_check": algebra_check(),
        "strength_invariance": strength_invariance(),
        "leftover_saturation": leftover_saturation(),
        "live_default_unchanged": True,
        "live_wired": {
            "lm_target": "faithful_gain",
            "requires": "--target_scale != 1",
            "default_changed": False,
        },
    }
    (out / "metrics.json").write_text(json.dumps(blob, indent=2) + "\n", encoding="utf-8")
    plot_criterion(summary, out / "criterion.png")
    plot_cost(whole, blind, out / "cost.png")
    plot_rows(blob["row_cos_sweep"], out / "rows.png")
    write_report(blob, out.parent / "lm-odd-leak-frac.md")

    won = blob["hits"]
    print(f"criterion: exam_{WIN_CELL} PASS at every seed and leak_frac < {LEAK_FRAC_WIN}")
    print(f"hits: {len(won)}")
    for entry in won:
        board = blob["sheet"].get(entry["name"])
        print(
            f"  {entry['name']:26s} leak_frac={entry['leak_frac']:+.3f} "
            f"board={_f(None if board is None else board['board_leak_frac'], '+.3f', '  n/a')} "
            f"margin={entry['margin']:+.1f} all_cells={entry['all_cells_pass']}"
        )
    print(f"frontier: {json.dumps(blob['frontier'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
