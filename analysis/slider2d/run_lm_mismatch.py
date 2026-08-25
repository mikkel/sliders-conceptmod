#!/usr/bin/env python3
"""Score LM v9 policies on the gender-v1 mismatch cell (CPU)."""

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

from analysis.slider2d.mismatch import (
    LIVE_GENDER_V1_ALIGN,
    PROJECT_ALIGN_RECOMMENDED,
    PROJECT_ALIGN_SLIDER_KNEE,
    MismatchField2D,
    knee_from_sweep,
    leak_cell_align,
    policy_table,
    sweep_project_hold_align,
)


DEFAULT_OUT = _REPO / "docs" / "lm-v9-mismatch"


def plot_compare(mismatch_rows: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.axhline(0, color="#dddddd", lw=0.6)
    ax.axvline(0, color="#dddddd", lw=0.6)
    styles = {
        "pair_symmetric": ("#1e8449", "pair-symmetric (full odd, κ=0)"),
        "always_project_hold": ("#c0392b", "always project+hold (today v9)"),
        "gated_align": ("#1a5276", "gated |odd·û|/||odd|| ≥ 0.50"),
        "u_is_pair_odd": ("#6c3483", "û = pair-odd (identity project)"),
    }
    for row in mismatch_rows:
        color, label = styles[row["name"]]
        dp = row["delta_plus"]
        dm = row["delta_minus"]
        ax.arrow(
            0, 0, dp[0], dp[1],
            color=color, width=0.02, length_includes_head=True, head_width=0.08, alpha=0.95,
        )
        ax.arrow(
            0, 0, dm[0], dm[1],
            color=color, width=0.02, length_includes_head=True, head_width=0.08, alpha=0.40,
        )
        ax.annotate(label, (dp[0] + 0.04, dp[1] + 0.04), fontsize=8, color=color)
    ax.annotate("true pair-odd (gender)", (1.15, 0.08), fontsize=8, color="#444444")
    ax.set_xlabel("concept  (male ← → female)")
    ax.set_ylabel("short-û junk  (orthogonal leftover)")
    ax.set_title("+1 (solid) / −1 (faint)  ·  project+hold keeps 0.20 of a clean pair")
    ax.set_aspect("equal")
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.2, 1.6)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_knee(rows: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    xs = [r["align"] for r in rows]
    ax.plot(xs, [r["cos_concept"] for r in rows], label="cos(d+, pair-odd)", color="#1a5276")
    ax.plot(xs, [r["strength"] for r in rows], label="||d+|| / ||odd||", color="#b9770e")
    ax.plot(xs, [abs(r["leak_ratio"]) for r in rows], label="|leak|", color="#922b21")
    ax.axvline(LIVE_GENDER_V1_ALIGN, color="#c0392b", ls="--", lw=0.8, label="live gender-v1 0.20")
    ax.axvline(PROJECT_ALIGN_SLIDER_KNEE, color="#1e8449", ls=":", lw=0.8, label="slider-cos knee 0.90")
    ax.axhline(0.90, color="#888888", ls=":", lw=0.6)
    ax.axhline(0.50, color="#888888", ls=":", lw=0.6)
    ax.axhline(0.20, color="#888888", ls=":", lw=0.6)
    ax.set_xlabel("|odd · û| / ||odd||")
    ax.set_ylabel("project+hold fit")
    ax.set_title("Knee on a clean pair: project+hold is identity only when û ≈ odd")
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(-0.05, 1.35)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _row_line(r: dict, *, leak_cell: bool = False) -> str:
    ax = r["axis"]
    cos = r["cos_slider_plus"] if leak_cell else r["cos_concept"]
    return (
        f"| `{r['name']}` | **{ax['slider']}** | **{ax['leak']}** | "
        f"**{ax['collapse']}** | **{ax['strength']}** | "
        f"{r['odd_align']:.3f} | {r['norm_plus']:.3f} | {r['strength']:.3f} | "
        f"{cos:.3f} | {r['leak_ratio']:+.3f} | {r['cos_plus_minus']:+.3f} |"
    )


def write_report(
    table: dict,
    sweep: list[dict],
    knees: dict,
    leak_align: float,
    blob: dict,
    path: Path,
) -> None:
    mm = {r["name"]: r for r in table["mismatch"]}
    lk = {r["name"]: r for r in table["leak"]}
    v9 = mm["always_project_hold"]
    sym = mm["pair_symmetric"]
    gated = mm["gated_align"]
    lines = [
        "# LM v9 mismatch: clean pair vs a short declared û",
        "",
        "The energetic×gender cell in [lm-v9-2d.md](lm-v9-2d.md) sets û from",
        "the pole names (energetic↔calm). There û **is** the intended slider,",
        "so `project_odd` + hold looks leak-0 and **cannot see gender-v1**.",
        "Someone who only checks “û = pole names” will re-blind the suite.",
        "",
        "Energy is a different geometry (leaky pair, short û at 0.48/0.68).",
        "Do not read the leak-0 row below as energy — û there is still the",
        "pole names. Both live cells and the one default loss are in",
        "[lm-live-cells.md](lm-live-cells.md).",
        "",
        "This cell is the live gender geometry. Poles are a rich/clean pair",
        "whose `(pos − neg)` is already the concept. Declared û is a",
        "*different* short phrase with `|odd·û|/||odd|| ≈ 0.20` — the number",
        "logged on gender-v1 against",
        "`A woman is singing, her voice is feminine.` /",
        "`A man is singing, his voice is masculine.`. Always-project+hold",
        "(`--lm_target v9_always`) **fails**. Hub / pair-symmetric and the",
        "new slider-level default **pass**.",
        "",
        "CPU only. No Hub, no GPU, no Music 3 weights.",
        "",
        "## Why the old fixture was blind",
        "",
        "On the leak cell, `pair_slider_dir` returns `E_SLIDER` from the",
        "energetic/calm polarity. Ungated `energetic − calm` already lies",
        f"mostly on that axis (`|odd·û|/||odd|| = {leak_align:.3f}`), so",
        "projecting drops unused gender and the hold has nothing left to",
        "eat. Gender-v1 is the opposite: the structured poles *are* the",
        "singer, and the short declared captions are a weak, tilted û.",
        "Projecting keeps `0.20` of the pair; hold treats the other `80%`",
        "as unused leak. Slider *strength* after hold — `||d+|| / ||odd||`",
        "— is the measurement the leak-ratio-only suite omitted.",
        "",
        "## Verdict",
        "",
        f"**Current `lm_v9` fails this cell.** `odd·û/||odd|| = {v9['odd_align']:.2f}`,",
        f"`||d+|| = {v9['norm_plus']:.3f}` (strength {v9['strength']:.3f}),",
        f"`cos(d+, pair-odd) = {v9['cos_concept']:.2f}`, leak {v9['leak_ratio']:+.3f}.",
        f"Pair-symmetric keeps the concept (`||d+|| = {sym['norm_plus']:.3f}`,",
        f"cos {sym['cos_concept']:.3f}, leak {sym['leak_ratio']:+.3f},",
        f"collapse {sym['cos_plus_minus']:+.3f}).",
        "2-D collapse stays −1 for both (targets remain odd); live gender-v1’s",
        "−0.562 is high-D / LoRA capacity, not this field.",
        "",
        "Mismatch cell (clean pair, short û at 0.20):",
        "",
        "| policy | slider | leak | ±1 | strength | odd·û | ||d+|| | ||d+||/||odd|| | cos | leak | collapse |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in table["mismatch"]:
        lines.append(_row_line(r, leak_cell=False))
    lines += [
        "",
        "Leak cell (energetic×gender, û = pole names) — `lm_v9` must stay leak-0:",
        "",
        "| policy | slider | leak | ±1 | strength | odd·û | ||d+|| | strength | cos | leak | collapse |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in table["leak"]:
        lines.append(_row_line(r, leak_cell=True))
    lines += [
        "",
        "![project+hold vs pair-symmetric on the mismatch cell](lm-v9-mismatch/compare.png)",
        "",
        "![Alignment sweep: when project+hold starts to pass](lm-v9-mismatch/knee.png)",
        "",
        "## Knee",
        "",
        "On a clean pair, an exact project onto û realizes",
        "`cos(d+, odd) = |odd·û|/||odd||` and `||d+||/||odd||` equal to the",
        "same number. Sweeping û’s tilt:",
        "",
        f"- slider-cos ≥ 0.90 at align **{knees['slider']}**",
        f"- strength ≥ 0.50 at align **{knees['strength']}**",
        f"- |leak| ≤ 0.20 at align **{knees['leak']}**",
        f"- all four right at align **{knees['all']}**",
        "",
        f"Live gender-v1 sat at **{LIVE_GENDER_V1_ALIGN:.2f}** — well below every",
        f"knee. The leak cell sits at **{leak_align:.3f}**. Any floor in",
        f"`({LIVE_GENDER_V1_ALIGN:.2f}, {leak_align:.2f})` is right on both",
        f"scored cells. The opt-in `--project_align_min {PROJECT_ALIGN_RECOMMENDED}`",
        "is the majority-of-odd rule in that gap. The slider-cos knee",
        f"({PROJECT_ALIGN_SLIDER_KNEE}) is the stricter “û must already be",
        "the pair” line; at 0.90, projecting only drops a small residual.",
        "",
        "|odd·û|/||odd|| **alone cannot tell energy from gender** in the",
        "abstract: a leaky pair + a clean û can print the same 0.20 as a",
        "clean pair + a weak û. The two cells we have are not that tie.",
        "The leak cell’s short û *is* the pole polarity and is already",
        f"{leak_align:.2f}-aligned; gender-v1’s short û was not. When the",
        "number is small, the conservative fallback (keep the pair, drop",
        "hold) is what gender needed. Do not treat 0.20 as “û is the",
        "intended concept, pair is junk” without another check.",
        "",
        "## What each policy does",
        "",
        f"- `pair_symmetric` / Hub-on-pair (full odd, κ=0): **pass** mismatch,",
        f"  **fail** leak (unused gender stays in `(pos−neg)/2`).",
        f"- `always_project_hold` / today’s `--lm_target v9`: **fail** mismatch",
        f"  (strength {v9['strength']:.3f}, cos {v9['cos_concept']:.2f}),",
        f"  **pass** leak (leak-0).",
        f"- `gated_align` (`--project_align_min {PROJECT_ALIGN_RECOMMENDED}`):",
        f"  **pass** both. Mismatch alignment {gated['odd_align']:.2f} < 0.50",
        "  → pair-symmetric, hold off. Leak alignment "
        f"{lk['gated_align']['odd_align']:.3f} ≥ 0.50 → project+hold.",
        "- `u_is_pair_odd`: project onto `(pos−neg)` itself. Identity on a",
        "  clean pair (**pass** mismatch) and a no-op on a leaky pair",
        "  (**fail** leak). Same geometry as pair-symmetric.",
        "",
        "## Train recommendation (gender vs energy)",
        "",
        "This cell is gender. Energy is **not** the leak-0 row above — that",
        "row still sets û from pole names (0.95). Live energy was 0.48 / 0.68",
        "and a hard per-row 0.50 gate mixed teachers. The one default that is",
        "right on both live cells is slider-level `--lm_target v9` (mean",
        f"`|odd·û|/||odd||` ≥ {PROJECT_ALIGN_RECOMMENDED}, same teacher on",
        "every row). See [lm-live-cells.md](lm-live-cells.md). Old",
        "always-project is `--lm_target v9_always`. Hub still leaks unused",
        "attr on a leaky pair.",
        "",
        "## How to run",
        "",
        "```bash",
        "PYTHONPATH=. python analysis/slider2d/run_lm_mismatch.py --out docs/lm-v9-mismatch",
        "PYTHONPATH=. pytest tests/test_lm_live_cells.py tests/test_lm_v9_mismatch.py tests/test_lm_v9_2d.py tests/test_lm_trainer_v9.py -q",
        "```",
        "",
        f"Seed `{blob['seed']}`, `{blob['steps']}` Adam steps.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    table = policy_table(steps=args.steps, seed=args.seed)
    sweep = sweep_project_hold_align(steps=min(args.steps, 120), seed=args.seed)
    knees = {
        "slider": knee_from_sweep(sweep, "slider"),
        "strength": knee_from_sweep(sweep, "strength"),
        "leak": knee_from_sweep(sweep, "leak"),
        "collapse": knee_from_sweep(sweep, "collapse"),
        "all": next((float(r["align"]) for r in sweep if r["pass"]), None),
    }
    leak_align = leak_cell_align()
    blob = {
        "steps": args.steps,
        "seed": args.seed,
        "live_gender_v1_align": LIVE_GENDER_V1_ALIGN,
        "recommended_align_min": PROJECT_ALIGN_RECOMMENDED,
        "slider_cos_knee": PROJECT_ALIGN_SLIDER_KNEE,
        "leak_cell_align": leak_align,
        "knees": knees,
        "mismatch": table["mismatch"],
        "leak": table["leak"],
        "sweep": [
            {
                "align": r["align"],
                "cos_concept": r["cos_concept"],
                "strength": r["strength"],
                "leak_ratio": r["leak_ratio"],
                "pass": r["pass"],
                "axis": r["axis"],
            }
            for r in sweep
        ],
    }
    (out / "metrics.json").write_text(json.dumps(blob, indent=2) + "\n", encoding="utf-8")
    plot_compare(table["mismatch"], out / "compare.png")
    plot_knee(sweep, out / "knee.png")
    write_report(table, sweep, knees, leak_align, blob, out.parent / "lm-v9-mismatch.md")
    print(f"leak-cell |odd·û|/||odd|| = {leak_align:.3f}")
    print(f"knees: {knees}")
    for cell, rows in (("mismatch", table["mismatch"]), ("leak", table["leak"])):
        print(f"== {cell} ==")
        for r in rows:
            ax = r["axis"]
            print(
                f"  {r['name']:22s} pass={str(r['pass']):5s} "
                f"align={r['odd_align']:.3f} ||d+||={r['norm_plus']:.3f} "
                f"str={r['strength']:.3f} cos={r.get('cos_concept', r.get('cos_slider_plus')):+.3f} "
                f"leak={r['leak_ratio']:+.3f} ±1={r['cos_plus_minus']:+.3f} "
                f"slider={ax['slider']} leak={ax['leak']} str={ax['strength']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
