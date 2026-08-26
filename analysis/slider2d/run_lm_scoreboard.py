#!/usr/bin/env python3
"""Compile every scored 2-D / high-D / sheet recipe into one gated table."""

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

from analysis.slider2d.exam import LIVE_EXAM
from analysis.slider2d.scoreboard import (
    BIPOLAR_MIRROR_FLOOR,
    CELL_LABEL,
    CELL_ORDER,
    FAILS,
    RACE_RECIPES,
    UNSCORED,
    WORKS,
    WORKS_SOME,
    collect_scoreboard,
    floatable_row,
    gates_blob,
    live_exam_report,
    sort_rows,
)


DEFAULT_OUT = _REPO / "docs" / "lm-2d-scoreboard"


def _fmt(value, spec: str, empty: str = "N/A") -> str:
    if value is None:
        return empty
    return format(float(value), spec)


def _verdict_md(verdict: str) -> str:
    if verdict == WORKS:
        return "**works**"
    if verdict == WORKS_SOME:
        return "works-on-some-pairs"
    if verdict == UNSCORED:
        return "unscored"
    return "**fails**"


CELL_MARK = {True: "pass", False: "**fail**", None: "—"}


PAIRS_HEADER = (
    "| recipe | exam_score | " + " | ".join(c.replace("exam_", "") for c in CELL_ORDER)
    + " | predicts live | compiled |\n"
    "|---|---:|" + "---|" * len(CELL_ORDER) + "---|---|"
)


def _pairs_md(row: dict) -> str:
    cells = " | ".join(CELL_MARK[row["cells"].get(name)] for name in CELL_ORDER)
    runs = row.get("predicts") or {}
    named = ", ".join(f"`{run}` ({cell.replace('exam_', '')})" for cell, run in runs.items())
    return (
        f"| `{row['id']}` | {_fmt(row.get('exam_score'), '.3f')} | {cells} | "
        f"{named or '—'} | {_verdict_md(row['compiled'])} |"
    )


TABLE_HEADER = (
    "| recipe | exam_score | leftover leak | leak_frac *(log)* | same_dir *(log)* | "
    "on-sheet | kept | off-sheet | argmax | "
    "swing | pair-odd cos *(log)* | ±1 *(log)* | intended cos | "
    "c+ | perc | rich-kept | compiled |\n"
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"
)


def _table_md(row: dict) -> str:
    return (
        f"| `{row['id']}` | {_fmt(row.get('exam_score'), '.3f')} | "
        f"{_fmt(row['leftover_leak'], '+.3f')} | "
        f"{_fmt(row.get('leak_frac'), '+.3f')} | "
        f"{_fmt(row.get('same_dir'), '.3f')} | "
        f"{_fmt(row['on_sheet'], '.3f')} | {_fmt(row['on_sheet_kept'], '.3f')} | "
        f"{_fmt(row['off_sheet'], '.3f')} | {_fmt(row['argmax_on_sheet'], '.2f')} | "
        f"{_fmt(row['swing_kept'], '.2f')} | {_fmt(row['pair_odd_cos'], '+.3f')} | "
        f"{_fmt(row['collapse'], '+.3f')} | {_fmt(row['intended_cos'], '+.3f')} | "
        f"{_fmt(row['trainer_c_plus'], '+.3f')} | {_fmt(row['perc'], '.0f')} | "
        f"{_fmt(row['rich_kept'], '.2f')} | {_verdict_md(row['compiled'])} |"
    )


def plot_scoreboard(rows: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    colors = {
        WORKS: "#1e8449",
        WORKS_SOME: "#b9770e",
        FAILS: "#c0392b",
        UNSCORED: "#7f8c8d",
    }
    for verdict, color in colors.items():
        pts = [r for r in rows if r["compiled"] == verdict]
        xs = [0.0 if r["leftover_leak"] is None else r["leftover_leak"] for r in pts]
        ys = [0.0 if r["on_sheet_kept"] is None else r["on_sheet_kept"] for r in pts]
        ax.scatter(xs, ys, c=color, s=42, label=verdict, zorder=3)
        for row, x, y in zip(pts, xs, ys):
            ax.annotate(row["id"], (x, y), fontsize=6.4, xytext=(4, 3), textcoords="offset points")
    ax.axvline(0.20, color="#7f8c8d", ls=":", lw=0.9)
    ax.axhline(0.90, color="#7f8c8d", ls=":", lw=0.9)
    ax.set_xlabel("leftover leak  (unused mix / BPM / gender)")
    ax.set_ylabel("on-sheet kept  (N/A plotted at 0)")
    ax.set_title("compiled gate: leak small and, when a sheet exists, stay on it")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


VERDICT_COLOR = {
    WORKS: "#1e8449",
    WORKS_SOME: "#b9770e",
    FAILS: "#c0392b",
    UNSCORED: "#7f8c8d",
}


def _pair_mark(row: dict) -> str:
    marks = []
    for cell in ("exam_divergent", "exam_close"):
        value = row.get("cells", {}).get(cell)
        marks.append({True: "P", False: "F", None: "—"}[value])
    return "/".join(marks)


def plot_exam_score(rows: list[dict], path: Path) -> None:
    """Horizontal exam_score ranking. Nulls omitted; race rows highlighted."""
    scored = [r for r in rows if r.get("exam_score") is not None]
    scored = sorted(scored, key=lambda r: float(r["exam_score"]))
    if not scored:
        return
    height = max(4.8, 0.42 * len(scored) + 1.6)
    fig, ax = plt.subplots(figsize=(8.8, height))
    y = list(range(len(scored)))
    colors = [VERDICT_COLOR.get(r["compiled"], "#7f8c8d") for r in scored]
    widths = [float(r["exam_score"]) for r in scored]
    bars = ax.barh(y, widths, color=colors, height=0.62, zorder=3)
    for bar, row in zip(bars, scored):
        if row["id"] in RACE_RECIPES:
            bar.set_hatch("///")
            bar.set_linewidth(1.8)
            bar.set_edgecolor("#1a252f")
            ax.plot(
                -0.012,
                bar.get_y() + bar.get_height() / 2.0,
                marker=">",
                color="#1a252f",
                markersize=7,
                clip_on=False,
                zorder=4,
            )
        score = float(row["exam_score"])
        ax.text(
            min(score + 0.012, 1.02),
            bar.get_y() + bar.get_height() / 2.0,
            f"{score:.3f}  {_pair_mark(row)}",
            va="center",
            ha="left",
            fontsize=7.4,
        )
    ax.set_yticks(y)
    ax.set_yticklabels([r["id"] for r in scored], fontsize=8)
    ax.set_xlim(0.0, 1.18)
    ax.set_xlabel("exam_score = min(overlap, swing) on exam_divergent + exam_close")
    ax.set_title("pair-exam ranking (live pairs only; race rows hatched)")
    ax.grid(axis="x", alpha=0.25)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=VERDICT_COLOR[WORKS], label=WORKS),
        plt.Rectangle((0, 0), 1, 1, color=VERDICT_COLOR[WORKS_SOME], label=WORKS_SOME),
        plt.Rectangle((0, 0), 1, 1, color=VERDICT_COLOR[FAILS], label=FAILS),
        plt.Rectangle((0, 0), 1, 1, facecolor="#7f8c8d", hatch="///", edgecolor="#1a252f", label="2026-08-25 race"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _hatch_race(bar, row: dict) -> None:
    if row["id"] not in RACE_RECIPES:
        return
    bar.set_hatch("///")
    bar.set_linewidth(1.8)
    bar.set_edgecolor("#1a252f")


def plot_leak_frac(rows: list[dict], path: Path) -> None:
    """Horizontal leak_frac bars. Same recipe order as exam-score; race hatched."""
    plotted = [r for r in rows if r.get("leak_frac") is not None]
    scored = [r for r in plotted if r.get("exam_score") is not None]
    scored = sorted(scored, key=lambda r: float(r["exam_score"]))
    nulls = sorted(
        [r for r in plotted if r.get("exam_score") is None],
        key=lambda r: float(r["leak_frac"]),
    )
    plotted = nulls + scored
    if not plotted:
        return
    height = max(4.8, 0.42 * len(plotted) + 1.6)
    fig, ax = plt.subplots(figsize=(8.8, height))
    y = list(range(len(plotted)))
    colors = [VERDICT_COLOR.get(r["compiled"], "#7f8c8d") for r in plotted]
    widths = [float(r["leak_frac"]) for r in plotted]
    bars = ax.barh(y, widths, color=colors, height=0.62, zorder=3)
    for bar, row, value in zip(bars, plotted, widths):
        _hatch_race(bar, row)
        if row["id"] in RACE_RECIPES:
            ax.plot(
                -1.12,
                bar.get_y() + bar.get_height() / 2.0,
                marker=">",
                color="#1a252f",
                markersize=7,
                clip_on=False,
                zorder=4,
            )
        side = 1 if value >= 0 else -1
        ax.text(
            value + 0.035 * side,
            bar.get_y() + bar.get_height() / 2.0,
            f"{value:+.3f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=7.4,
        )
    ax.axvline(
        BIPOLAR_MIRROR_FLOOR,
        color="#1a252f",
        ls="--",
        lw=1.0,
        zorder=2,
        label=f"bipolar-mirror floor ({BIPOLAR_MIRROR_FLOOR:.2f})",
    )
    ax.axvline(0.0, color="#7f8c8d", ls=":", lw=1.0, zorder=2, label="same-dir starts (0)")
    ax.set_yticks(y)
    ax.set_yticklabels([r["id"] for r in plotted], fontsize=8)
    ax.set_xlim(-1.22, 1.22)
    ax.set_xlabel("leak_frac = cos(d+, d−)  (leftover_bipolar; logged, never scored)")
    ax.set_title("caption-pole even motion (leftover leak ≠ leak_frac; race rows hatched)")
    ax.grid(axis="x", alpha=0.25)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=VERDICT_COLOR[WORKS], label=WORKS),
        plt.Rectangle((0, 0), 1, 1, color=VERDICT_COLOR[WORKS_SOME], label=WORKS_SOME),
        plt.Rectangle((0, 0), 1, 1, color=VERDICT_COLOR[FAILS], label=FAILS),
        plt.Rectangle((0, 0), 1, 1, facecolor="#7f8c8d", hatch="///", edgecolor="#1a252f", label="2026-08-25 race"),
        plt.Line2D([0], [0], color="#1a252f", ls="--", label=f"bipolar-mirror floor ({BIPOLAR_MIRROR_FLOOR:.2f})"),
        plt.Line2D([0], [0], color="#7f8c8d", ls=":", label="same-dir starts (0)"),
    ]
    ax.legend(handles=handles, fontsize=7.4, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_leftover_vs_leak_frac(rows: list[dict], path: Path) -> None:
    """Optional scatter: unused-ê leftover leak vs caption-pole leak_frac."""
    pts = [
        r
        for r in rows
        if r.get("leak_frac") is not None and r.get("leftover_leak") is not None
    ]
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    for verdict, color in VERDICT_COLOR.items():
        group = [r for r in pts if r["compiled"] == verdict]
        if not group:
            continue
        ax.scatter(
            [float(r["leftover_leak"]) for r in group],
            [float(r["leak_frac"]) for r in group],
            c=color,
            s=42,
            label=verdict,
            zorder=3,
        )
        for row in group:
            ax.annotate(
                row["id"],
                (float(row["leftover_leak"]), float(row["leak_frac"])),
                fontsize=6.0,
                xytext=(4, 3),
                textcoords="offset points",
            )
    ax.axhline(BIPOLAR_MIRROR_FLOOR, color="#1a252f", ls="--", lw=0.9)
    ax.axhline(0.0, color="#7f8c8d", ls=":", lw=0.9)
    ax.axvline(0.20, color="#7f8c8d", ls=":", lw=0.9)
    ax.set_xlabel("leftover leak  (unused ê)")
    ax.set_ylabel("leak_frac = cos(d+, d−)")
    ax.set_title("leftover leak ≠ leak_frac")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_report(rows: list[dict], blob: dict, path: Path) -> None:
    gates = blob["gates"]
    live = blob["live_exam"]
    winners = [r for r in rows if r["compiled"] == WORKS]
    gender_only = [r for r in rows if r["compiled"] == WORKS_SOME]
    fails = [r for r in rows if r["compiled"] == FAILS]
    agree = sum(1 for r in live if r["agrees"])
    lines = [
        "# 2-D / high-D / sheet / pair-exam scoreboard",
        "",
        "Generated by `analysis/slider2d/run_lm_scoreboard.py`. CPU only, no Hub,",
        "no GPU, no Music 3 weights. Does not change the live trainer default.",
        "",
        "Every other cell in this repo scores one question. This page joins",
        "those real fixture numbers into one table and applies one gate. It",
        "does not invent a loss.",
        "",
        "## The exam this board is graded on",
        "",
        "Three live Music 3 runs, 2026-08-25, after the #24 wire. rank 8 /",
        "alpha 8 / lr 5e-4 / 800 steps / seed 7 / `--no-early_stop` / endreg 1.0",
        "/ hold 0, pair-odd early-stop gates deliberately off.",
        "",
        "| live run | recipe | prompts | c+ | ±1 | p% / n% | loss | ears |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for run, info in LIVE_EXAM.items():
        lines.append(
            f"| `{run}` | `--lm_target {info['teacher']} --pole_mode "
            f"{info['pole_mode']}` | `{info['prompts']}` | {info['c_plus']:+.3f} | "
            f"{info['collapse']:+.3f} | {info['pperc']:.2f} / {info['nperc']:.2f} | "
            f"{info['loss']:.4f} | "
            f"{'PASS' if info['listen'] == 'pass' else 'FAIL'} — {info['heard']} |"
        )
    lines += [
        "",
        "Two of those are the **same recipe** on different prompt files, and",
        "the one with the smallest loss, the best `c+` and the lowest `p%` is",
        "one of the two that garbled. So a recipe is not one row: it is a",
        "**recipe × pair** grid. The sortable number is `exam_score` =",
        "`min(overlap, swing)` over the live exam pairs that row is read on.",
        "",
        f"| live run | fixture row | pair | predicted | ears | agrees | why |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in live:
        lines.append(
            f"| `{row['run']}` | `{row['recipe']}` | `{row['cell']}` | "
            f"**{row['predicted'].upper()}** | **{row['listen'].upper()}** | "
            f"{'yes' if row['agrees'] else 'NO'} | {row['reason']} |"
        )
    lines += [
        "",
        f"{agree} of {len(live)} agree. Full derivation, sweeps and caveats in",
        "[lm-pair-exam.md](lm-pair-exam.md).",
        "",
        "## The compiled gate",
        "",
        gates["prose"],
        "",
        "Pair-exam cells, on what the student sings over its own 8-token",
        "continuation:",
        "",
        f"- continuation overlap with the pole's words: `≥ {gates['exam_overlap']}`",
        f"- position-wise agreement / the pole's own cross-draw agreement: `≥ {gates['exam_match_kept']}`",
        f"- off-caption mass: `≤ {gates['exam_off_caption']}`",
        f"- coherence, consecutive words from the same song: `≥ {gates['exam_coherence']}`",
        f"- audible swing kept: `≥ {gates['exam_swing']}`",
        "",
        "#22 sheet cells, on the one token at `<|audio_start|>`:",
        "",
        f"- leftover leak lock: `{gates['leak_lock']}`",
        f"- on-sheet kept: `≥ {gates['sheet_lock']}`",
        f"- off-sheet mass: `≤ {gates['garble_max']}`",
        f"- argmax-on-sheet: `{gates['argmax_lock']:g}`",
        f"- concept swing kept: `≥ {gates['swing_floor']}`",
        "",
        "Never an input to any gate:",
        "",
        f"- pair-odd cos scored: `{gates['pair_odd_cos_scored']}`",
        f"- ±1 collapse scored: `{gates['collapse_scored']}`",
        f"- pole loss scored: `{gates['pole_loss_scored']}`",
        f"- p% / n% scored: `{gates['perc_scored']}`",
        f"- leak_frac scored: `{gates['leak_frac_scored']}`",
        f"- same_dir scored: `{gates['same_dir_scored']}`",
        "",
        "A missing cell is `—`, not a free pass. On the sheet cells a missing",
        "sheet column is not a free pass for a midpoint teacher either: those",
        "rows inherit the leftover sheet of `t± = h0 ± a`, because deleting",
        "`c` is a fact about the target point, not about the hidden width.",
        "",
        "## Which pairs each recipe passes",
        "",
        "Sorted by `exam_score` descending, nulls last. `exam_score` is",
        "`min(overlap, swing)` on `exam_divergent` (energy-v4) and",
        "`exam_close` (gender-v4) only — unused_e and the sheet cells are",
        "other questions. A pair with no reading is skipped, not a free 1.0.",
        "The verdict stays a label; the number is what a human sorts by.",
        "",
        PAIRS_HEADER,
    ]
    lines += [_pairs_md(r) for r in rows]
    lines += [
        "",
        "What each cell is:",
        "",
    ]
    lines += [f"- `{name.replace('exam_', '')}` — {CELL_LABEL[name]}" for name in CELL_ORDER]
    lines += [
        "",
        "The two `semantic_kl_poles` cells are the load-bearing row: it is the",
        "live energy win on a divergent pair and the live gender garble on a",
        "close one, so no single verdict for the recipe is honest and",
        "`works-on-some-pairs` names which. The combined 2026-08-25 race rows",
        "(`faithful_sub_e_if_unused`, `semantic_kl_null`, `hidden_kl_poles`,",
        "fixture-only `unrolled_kl`, plus the #35 Opus rows `faithful_guard_e`,",
        "`dual_band_poles`, `dual_band_guard_e`, `dual_band_midpoint`) sit",
        "next to the #28 baselines; `faithful_raw` / `faithful_attrs` remain",
        "the hidden-MSE caption pair.",
        "",
        "## Short verdict",
        "",
    ]
    if winners:
        names = ", ".join(f"`{r['id']}`" for r in winners)
        lines.append(
            f"**Works on every pair it is read on:** {names}. "
            "`faithful_sub_e_if_unused` is the leftover-gated sibling of "
            "`faithful_raw`: subtract leftover ê only when `|ê̂_⊥ · â| < 0.50`. "
            "`faithful_attrs` is the data fix — unused gender/BPM pinned in "
            "the captions — so leftover ê is not in the text."
        )
    else:
        lines.append("No recipe passed on every pair it is read on.")
    if gender_only:
        names = ", ".join(f"`{r['id']}`" for r in gender_only)
        lines += [
            "",
            f"**Works on some pairs:** {names}. Each passes at least one pair "
            "and fails another. `faithful_raw` (and `hidden_beta1`, which is "
            "the same target reached through `--lm_target symmetric "
            "--common_beta 1`) and `hidden_kl_poles` pass all three pair cells "
            "and are charged only by the unused-ê sheet — a leak gender-v4 "
            "has no `leak_*` to trip. `semantic_kl_null` is the one hybrid "
            "from PRs #29 / #32 / #33 (trainer aliases "
            "`semantic_kl_plus_hidden` and `semantic_kl_pin`). "
            "`semantic_kl_poles` is the row the live exam is about: the "
            "energy win and the gender garble.",
        ]
    lines += [
        "",
        f"**Fails:** {len(fails)} recipes — hub, hold-ê raw, short-û and rich-û "
        "project, and the high-D leftover holds. None of them has a pair-exam "
        "reading; they fail on leftover leak and the sheet alone. A perfect "
        "pair-odd lock and a solved pole loss are both failure modes here.",
        "",
        "## The full joined table",
        "",
        "Every column the older cells contribute, in `exam_score` order.",
        "Pair-odd cos, ±1, `leak_frac` and `same_dir` are **logged, never scored**.",
        "`leftover leak` is unused ê. `leak_frac` is `cos(d+, d−)` of the",
        "fitted ±1 student — caption-pole even motion, not leftover ê.",
        "",
        TABLE_HEADER,
    ]
    lines += [_table_md(r) for r in rows]
    lines += [
        "",
        "![exam_score ranking](lm-2d-scoreboard/exam-score.png)",
        "",
        "`exam_score` = min(overlap, swing) on `exam_divergent` + `exam_close` only.",
        "Hatched bars are the combined 2026-08-25 race recipes (leftover-gate,",
        "hybrid, hidden_kl, unrolled_kl, and the #35 Opus rows); #28 baselines stay solid.",
        "",
        "![leak_frac = cos(d+, d−)](lm-2d-scoreboard/leak-frac.png)",
        "",
        "`leak_frac` = `cos(d+, d−)` from `leftover_bipolar` on the fitted ±1",
        "student. This is **not** leftover leak. Clean bipolar wants",
        f"`leak_frac` ≤ {gates['bipolar_mirror_floor']:.2f}; caption-pole / leftover-gate",
        "no-op recipes keep the common component and sit near 0. Leftover-gate",
        "clears unused ê and does **not** clear this leak. Hatched the same",
        "way as `exam-score.png`. Vertical lines: bipolar-mirror floor",
        f"({gates['bipolar_mirror_floor']:.2f}) and same-dir starts (0).",
        "",
        "![leftover leak vs leak_frac](lm-2d-scoreboard/leftover-vs-leak-frac.png)",
        "",
        "![leak vs on-sheet kept](lm-2d-scoreboard/scoreboard.png)",
        "",
        "## What each row is",
        "",
    ]
    for row in rows:
        extra = []
        if row.get("content_cos") is not None:
            extra.append(f"content-cos {_fmt(row['content_cos'], '+.3f')}")
        if row.get("polarity_distinct") is not None:
            extra.append(f"polarity {_fmt(row['polarity_distinct'], '+.3f')}")
        if row.get("trainer_c_plus_distinct") is not None:
            extra.append(f"c+ distinct {_fmt(row['trainer_c_plus_distinct'], '+.3f')}")
        if row.get("loss") is not None:
            extra.append(f"loss {_fmt(row['loss'], '.3f')}")
        suffix = f" ({'; '.join(extra)})" if extra else ""
        lines.append(
            f"- `{row['id']}` — {row['label']}. {row['notes']} "
            f"Fixture: {row['fixture']}.{suffix}"
        )
        for cell, reason in (row.get("exam_reason") or {}).items():
            run = (row.get("predicts") or {}).get(cell)
            live = f" — live `{run}`" if run else ""
            lines.append(
                f"  - {cell.replace('exam_', '')} pair{live}: {reason}"
            )
    lines += [
        "",
        "## The next live cards the board points at",
        "",
        "Live techniques from the 2026-08-25 race plus the #35 Opus flags.",
        "None of these has a listen on this branch. PRs #29 / #32 / #33 are",
        "**one hybrid** (`semantic_kl_null`); the other two names are trainer",
        "aliases, not extra board rows. `dual_band` is a distinct live flag",
        "from that hybrid (centered SVD projector, not uncentered `ker(W)`).",
        "",
        "```bash",
        "# leftover gate: subtract ê only when unused",
        "python conceptmod/textsliders/train_lm_slider_music3.py \\",
        "  --name energy-lm-v19 \\",
        "  --prompts_file conceptmod/textsliders/data/prompts-energy-v4.yaml \\",
        "  --lm_target faithful_sub_e_if_unused --pole_mode hidden \\",
        "  --rank 8 --alpha 8 --lr 5e-4 --steps 800 --seed 7 \\",
        "  --no-early_stop --endreg_weight 1.0",
        "",
        "# hybrid KL + unread hidden (aliases: semantic_kl_plus_hidden, semantic_kl_pin)",
        "python conceptmod/textsliders/train_lm_slider_music3.py \\",
        "  --name gender-lm-v20 \\",
        "  --prompts_file conceptmod/textsliders/data/prompts-gender-v4.yaml \\",
        "  --lm_target faithful --pole_mode semantic_kl_null \\",
        "  --rank 8 --alpha 8 --lr 5e-4 --steps 800 --seed 7 \\",
        "  --no-early_stop --endreg_weight 1.0",
        "",
        "# hidden MSE + tiny semantic KL",
        "python conceptmod/textsliders/train_lm_slider_music3.py \\",
        "  --name gender-lm-hidden-kl \\",
        "  --prompts_file conceptmod/textsliders/data/prompts-gender-v4.yaml \\",
        "  --lm_target faithful --pole_mode hidden_kl \\",
        "  --rank 8 --alpha 8 --lr 5e-4 --steps 800 --seed 7 \\",
        "  --no-early_stop --endreg_weight 1.0",
        "",
        "# blend-guarded leftover ê (threshold-free)",
        "python conceptmod/textsliders/train_lm_slider_music3.py \\",
        "  --name energy-lm-guard-e \\",
        "  --prompts_file conceptmod/textsliders/data/prompts-energy-v4.yaml \\",
        "  --lm_target faithful_guard_e --pole_mode hidden \\",
        "  --rank 8 --alpha 8 --lr 5e-4 --steps 800 --seed 7 \\",
        "  --no-early_stop --endreg_weight 1.0",
        "",
        "# dual-band KL + blind-band MSE on real poles",
        "python conceptmod/textsliders/train_lm_slider_music3.py \\",
        "  --name gender-lm-dual-band \\",
        "  --prompts_file conceptmod/textsliders/data/prompts-gender-v4.yaml \\",
        "  --lm_target faithful --pole_mode dual_band \\",
        "  --rank 8 --alpha 8 --lr 5e-4 --steps 800 --seed 7 \\",
        "  --no-early_stop --endreg_weight 1.0",
        "```",
        "",
        "`unrolled_kl` is fixture-only: the live trainer has no frozen mix.",
        "What to watch on a live run: energy should keep the genre/BPM ride",
        "(`energy-lm-v18`) instead of the midpoint pull (`energy-lm-v16`);",
        "gender `p%` / `n%` should leave `gender-lm-v16`'s 0.523 / 0.777.",
        "`c+` will print *worse* than v9 under a caption target; per #22 that",
        "is expected. The listen is the gate. This does not change the live",
        "default, which is still `--lm_target v9` / `--pole_mode hidden`.",
        "",
        "## Why these columns are logged and never scored",
        "",
        "**leftover leak ≠ leak_frac.** leftover leak is unused ê on the #22",
        "sheet (gated at 0.2). `leak_frac` = `cos(d+, d−)` is caption-pole",
        "even motion from `leftover_bipolar` on the fitted ±1 student, the",
        "same class as pair-odd cos and the ±1 collapse. A leftover-gate",
        "recipe can clear unused ê and still sit at `leak_frac` around",
        "−0.4 to +0.1. Neither `leak_frac` nor `same_dir` is an input to",
        "`exam_score` or the compiled works/fails gate.",
        "",
        "**Pair-odd cos and the ±1 collapse** (#22). On the leftover sheet",
        "`v9_hidden` prints `cos(d+, a) = +1.000` and `cos(d+, d−) = −1.000`",
        "while keeping about a third of the caption's on-sheet mass.",
        "",
        "**The pole loss and p% / n%** (2026-08-25). `gender-lm-v16` printed",
        "the campaign's smallest pole loss (0.0091) with `p% / n% = 0.523 /",
        "0.777` on the same step, and its lyrics came out garbled. A semantic-KL",
        "loss has zero gradient on the part of the hidden state the semantic",
        "band does not read, so on a close pair it reaches its floor without",
        "the axis arriving at all: the loss cannot distinguish \"solved\" from",
        "\"there was nothing here I could see\".",
        "",
        "That is why the ordering rule is the pair, and why the previous",
        "version of this board ranked `semantic_kl_sub_e` (live",
        "`energy-lm-v16`, garbled) above `semantic_kl_poles` (live",
        "`energy-lm-v18`, the win): it scored the energy recipes on a",
        "same-song field with an unpinned attribute, which is not what",
        "energy-v4 is. That cell is still here and still the place leak is",
        "scored — it is no longer the energy stand-in.",
        "",
        "## Related cells",
        "",
        "- [lm-pair-exam.md](lm-pair-exam.md) — the pair cells, the rollout,",
        "  the divergence and visible-share sweeps, and the live exam.",
        "- [lm-sheet-goodhart.md](lm-sheet-goodhart.md) — the sheet readout",
        "  and the four locked recipes.",
        "- [lm-live-cells.md](lm-live-cells.md) — gender-like vs energy-like",
        "  hidden leak.",
        "- [lm-highd-leftover.md](lm-highd-leftover.md) — leftover ê, λ·D/2,",
        "  trainer c+.",
        "- [lm-rich-2d.md](lm-rich-2d.md) — project short û vs rich û.",
        "- [lm-faithful-2d.md](lm-faithful-2d.md) — raw poles vs attributes.",
        "- [lm-hold-overlap.md](lm-hold-overlap.md) — hold-ê raw vs ê_⊥û.",
        "",
        "## How to run",
        "",
        "```bash",
        "PYTHONPATH=. python analysis/slider2d/run_lm_scoreboard.py --out docs/lm-2d-scoreboard",
        "PYTHONPATH=. pytest tests/test_lm_2d_scoreboard.py -q",
        "```",
        "",
        "CPU only. No Hub, no GPU, no Music 3 weights.",
        "",
        f"Sheet seed `{blob['seed']}`, `{blob['sheet_steps']}` Adam steps; "
        f"pair-exam `{blob['exam_steps']}` steps, 8-token rollout, 4 draws; "
        f"other fixtures `{blob['other_steps']}` steps.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sheet-steps", type=int, default=400)
    parser.add_argument("--other-steps", type=int, default=200)
    parser.add_argument("--exam-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    rows = collect_scoreboard(
        sheet_steps=args.sheet_steps,
        other_steps=args.other_steps,
        exam_steps=args.exam_steps,
        seed=args.seed,
    )
    rows = sort_rows(rows)
    live = live_exam_report(exam_steps=args.exam_steps, seed=args.seed)
    blob = {
        "sheet_steps": args.sheet_steps,
        "other_steps": args.other_steps,
        "exam_steps": args.exam_steps,
        "seed": args.seed,
        "gates": gates_blob(),
        "cells": list(CELL_ORDER),
        "live_exam": [
            {k: v for k, v in row.items() if k != "row"} for row in live
        ],
        "rows": [floatable_row(r) for r in rows],
        "live_default_unchanged": True,
    }
    (out / "metrics.json").write_text(json.dumps(blob, indent=2) + "\n", encoding="utf-8")
    plot_scoreboard(rows, out / "scoreboard.png")
    plot_exam_score(rows, out / "exam-score.png")
    plot_leak_frac(rows, out / "leak-frac.png")
    plot_leftover_vs_leak_frac(rows, out / "leftover-vs-leak-frac.png")
    write_report(rows, blob, out.parent / "lm-2d-scoreboard.md")
    for row in rows:
        cells = "".join(
            {True: "P", False: "F", None: "-"}[row["cells"].get(name)]
            for name in CELL_ORDER
        )
        print(
            f"{row['compiled']:20s} {row['id']:26s} "
            f"exam={_fmt(row.get('exam_score'), '.3f')} {cells} "
            f"leak={_fmt(row['leftover_leak'], '+.3f')} "
            f"leak_frac={_fmt(row.get('leak_frac'), '+.3f')} "
            f"cos={_fmt(row['pair_odd_cos'], '+.3f')}"
        )
    for row in live:
        print(
            f"{'live':20s} {row['run']:26s} {row['cell']:10s} "
            f"predicted={row['predicted']:5s} ears={row['listen']:5s} "
            f"agrees={row['agrees']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
