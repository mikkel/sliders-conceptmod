#!/usr/bin/env python3
"""Score live gender-like and energy-like cells; write the comparison table."""

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

from analysis.slider2d.energy import LIVE_ENERGY_ALIGNS, energy_row_aligns
from analysis.slider2d.live_compare import live_policy_table, table_row
from analysis.slider2d.mismatch import LIVE_GENDER_V1_ALIGN
from conceptmod.textsliders.slider_targets import LEAK_HOLD_WEIGHT, SLIDER_ALIGN_MIN


DEFAULT_OUT = _REPO / "docs" / "lm-live-cells"

BARE_TRAIN = (
    "python conceptmod/textsliders/train_lm_slider_music3.py "
    "--prompts_file conceptmod/textsliders/data/prompts-energy-v4.yaml"
)
BARE_TRAIN_GENDER = (
    "python conceptmod/textsliders/train_lm_slider_music3.py "
    "--prompts_file conceptmod/textsliders/data/prompts-gender-v4.yaml"
)


def _fmt(row: dict) -> str:
    mixed = "yes" if row["mixed"] else "no"
    verdict = "PASS" if row["pass"] else "FAIL"
    return (
        f"| `{row['policy']}` | {row['leak']:+.3f} | {row['strength']:.3f} | "
        f"{row['cos_intended']:.3f} | {mixed} | **{verdict}** |"
    )


def plot_energy(energy_rows: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.axhline(0, color="#dddddd", lw=0.6)
    ax.axvline(0, color="#dddddd", lw=0.6)
    styles = {
        "hub": ("#b9770e", "Hub / pair-odd"),
        "always_project_hold": ("#c0392b", "always project+hold"),
        "gated_row_0.50": ("#6c3483", "per-row gate 0.50 (mixed)"),
        "slider_align_0.50": ("#7f8c8d", "slider-level project (v9_project)"),
        "hold_e": ("#1e8449", "hold ê (NEW v9)"),
    }
    for row in energy_rows:
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
        ax.annotate(label, (dp[0] + 0.03, dp[1] + 0.03), fontsize=8, color=color)
    ax.annotate("intended û (energy)", (1.05, 0.08), fontsize=8, color="#444444")
    ax.annotate("unused leak", (0.08, 1.05), fontsize=8, color="#444444")
    ax.set_xlabel("intended energy  (loud ← → quiet)")
    ax.set_ylabel("unused mix / genre / BPM")
    ax.set_title("+1 (solid) / −1 (faint)  ·  live energy 0.48 / 0.68")
    ax.set_aspect("equal")
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.4, 1.6)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_report(table: dict, blob: dict, path: Path) -> None:
    gender = {r["name"]: table_row("gender", r) for r in table["gender"]}
    energy = {r["name"]: table_row("energy", r) for r in table["energy"]}
    cheat = table["energy_cheat"]
    g_new = next(r for r in table["gender"] if r["name"] == "hold_e")
    e_new = next(r for r in table["energy"] if r["name"] == "hold_e")
    e_row = next(r for r in table["energy"] if r["name"] == "gated_row_0.50")
    lines = [
        "# Live 2-D cells: gender-like and energy-like",
        "",
        "The old energetic×gender leak cell set û from the pole names, so",
        "`|odd·û|/||odd|| ≈ 0.95` and project+hold looked leak-0. That hid",
        "live energy the same way the first fixture hid gender.",
        "",
        "Two CPU cells match the live Music 3 logs. One default loss has to",
        "be right on both. Hub flags are not required.",
        "",
        "CPU only. No Hub weights, no GPU, no Music 3 downloads.",
        "",
        "## Cells",
        "",
        f"- **Gender-like:** clean pair, short û at `|odd·û|/||odd|| = {LIVE_GENDER_V1_ALIGN:.2f}`.",
        "  Always-project+hold must FAIL (tiny `||d+||`, hold eats the singer).",
        "  Pair-symmetric / Hub-odd must PASS. NEW: teacher = full odd,",
        "  **no hold on û⊥**. Hold only if a real leak ê is declared.",
        f"- **Energy-like:** leaky pair (unused attr in `pos−neg`), short û **is**",
        f"  the intended axis, alignments `{list(LIVE_ENERGY_ALIGNS)}` on four rows.",
        "  Per-row 0.50 splits the rows (mixed teacher) and must FAIL.",
        "  Hub / symmetric-on-pair must still leak. Short-û project+hold may",
        "  look leak-0 but is the live-fragile path. NEW: teacher = full odd,",
        "  hold along the unused-attr direction ê.",
        "- **Old leak-0 cell** (û = energetic/calm pole names) is not energy.",
        "  Project+hold is still leak-0 there; that is the cheat.",
        "- **+/− leftover** (leak_frac / same-dir) must stay in the live-good",
        "  band (≲ 6% same-dir). It is not the thing we optimize.",
        "",
        "## Verdict",
        "",
        f"**`--lm_target v9` is now full pair-odd + hold-on-ê** (λ={LEAK_HOLD_WEIGHT}).",
        "Teacher stays `a = ½(h+−h−)`, `t± = h0 ± a`, κ = 0. Do not replace",
        "`a` with `(a·û)û`. Short `slider_positive` is a name / probe.",
        "Declare unused mix / BPM / genre as YAML `leak_positive` /",
        "`leak_negative` (or `leak: [pos, neg]`). If `attributes` already",
        "pin the unused axis, `a` is clean and hold can be 0.",
        "",
        f"Gender-like NEW: leak {g_new['leak_ratio']:+.3f}, "
        f"`||d+||/||odd||` {g_new['strength']:.3f}, "
        f"cos {g_new['cos_concept']:.3f}, mixed {g_new['mixed']}, "
        f"same-dir {g_new.get('same_dir', 0):.4f}.",
        f"Energy-like NEW: leak {e_new['leak_ratio']:+.3f}, "
        f"`||d+||/||odd||` {e_new['strength']:.3f}, "
        f"cos to û {e_new['cos_intended']:.3f}, mixed {e_new['mixed']}, "
        f"same-dir {e_new.get('same_dir', 0):.4f}.",
        f"Per-row 0.50 on energy: mixed={e_row['mixed']}, "
        f"leak {e_row['leak_ratio']:+.3f}, cos {e_row['cos_intended']:.3f}.",
        "",
        "Discarded: always-project / short-û project (kills gender at 0.20),",
        "Hub leash (still leaks unused attr), slider-level 0.50 gate (û is",
        "still the teacher on energy), per-row 0.50 (mixed energy teacher),",
        "soft per-row blend of pair-odd and `(a·û)û`. λ=1 hold-on-ê leaves",
        "energy leak ~0.69 — too weak when ê fights the full-odd teacher.",
        "û = pole-odd on the energy cell is identity project and still leaks —",
        f"cheat leak {cheat['leak_ratio']:+.3f}, align {cheat['odd_align']:.3f}.",
        "",
        "## Table (Hub / project-short-û / NEW)",
        "",
        "### Gender-like (intended axis = clean pair-odd)",
        "",
        "| policy | leak | ||d+||/||odd|| | cos intended | mixed-row | verdict |",
        "|---|---:|---:|---:|---|---|",
    ]
    for name in ("hub", "always_project_hold", "hold_e"):
        lines.append(_fmt(gender[name]))
    lines += [
        "",
        "### Energy-like (intended axis = energy, unused attr = ê)",
        "",
        "| policy | leak | ||d+||/||odd|| | cos intended | mixed-row | verdict |",
        "|---|---:|---:|---:|---|---|",
    ]
    for name in ("hub", "always_project_hold", "hold_e"):
        lines.append(_fmt(energy[name]))
    lines += [
        "",
        "![energy-like residuals](lm-live-cells/energy.png)",
        "",
        "## What each policy does",
        "",
        "- `hub`: pair-odd + published floor/anchor. **PASS** gender (clean pair),",
        "  **FAIL** energy (unused attr stays in `(pos−neg)/2`).",
        "- `always_project_hold` / `--lm_target v9_always` (and `v9_project`",
        "  on energy): **FAIL** gender (strength 0.20, hold eats the singer),",
        "  **PASS** energy (û is the axis). Live-fragile: û is a name, not the",
        "  teacher.",
        "- `hold_e` / default `--lm_target v9`: **PASS** both. Teacher = full",
        "  odd on every row. Gender declares no ê (hold 0). Energy holds the",
        "  unused-attr direction ê at λ=8.",
        "- `gated_row_0.50` / `slider_align_0.50`: old project path, kept as",
        "  `--lm_target v9_project`. Not the default.",
        "",
        "## Bare Music 3 LM train",
        "",
        "The one argv is `--lm_target v9` (default). Declare ê in YAML as",
        "`leak_positive` / `leak_negative` (or `leak: [pos, neg]`). CLI",
        "`--leak_positive` / `--leak_negative` wins. `attributes` prefixes",
        "captions (makes `a` clean) and is not ê. Omit leak on a clean pair.",
        "",
        "Old short-û project is `--lm_target v9_project` (slider-level gate)",
        "or `--lm_target v9_always`.",
        "",
        "```bash",
        BARE_TRAIN_GENDER,
        BARE_TRAIN,
        "```",
        "",
        "## How to run",
        "",
        "```bash",
        "PYTHONPATH=. python analysis/slider2d/run_lm_live.py --out docs/lm-live-cells",
        "PYTHONPATH=. pytest tests/test_lm_live_cells.py tests/test_lm_v9_mismatch.py tests/test_lm_v9_2d.py tests/test_lm_trainer_v9.py -q",
        "```",
        "",
        f"Seed `{blob['seed']}`, `{blob['steps']}` Adam steps.",
        "",
        "Unused-ê is the wrong ê for live energy-v4 leak captions (they",
        "*are* energy). That cell is [lm-hold-overlap.md](lm-hold-overlap.md).",
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
    table = live_policy_table(steps=args.steps, seed=args.seed)
    compact = {
        "gender": [table_row("gender", r) for r in table["gender"]],
        "energy": [table_row("energy", r) for r in table["energy"]],
        "energy_cheat": table_row("energy", table["energy_cheat"]),
    }
    blob = {
        "steps": args.steps,
        "seed": args.seed,
        "live_gender_align": LIVE_GENDER_V1_ALIGN,
        "live_energy_aligns": list(LIVE_ENERGY_ALIGNS),
        "measured_energy_aligns": energy_row_aligns(),
        "slider_align_min": SLIDER_ALIGN_MIN,
        "bare_train_energy": BARE_TRAIN,
        "bare_train_gender": BARE_TRAIN_GENDER,
        "table": compact,
        "gender": table["gender"],
        "energy": table["energy"],
        "energy_cheat": table["energy_cheat"],
    }
    (out / "metrics.json").write_text(json.dumps(blob, indent=2) + "\n", encoding="utf-8")
    plot_energy(table["energy"], out / "energy.png")
    write_report(table, blob, out.parent / "lm-live-cells.md")
    print(f"energy aligns = {blob['measured_energy_aligns']}")
    for cell in ("gender", "energy"):
        print(f"== {cell} ==")
        for r in table[cell]:
            print(
                f"  {r['name']:22s} pass={str(r['pass']):5s} mixed={str(r['mixed']):5s} "
                f"align={r['odd_align']:.3f} ||d+||={r['norm_plus']:.3f} "
                f"str={r['strength']:.3f} cos={r.get('cos_intended', r.get('cos_concept')):+.3f} "
                f"leak={r['leak_ratio']:+.3f}"
            )
    cheat = table["energy_cheat"]
    print(
        f"  {'u_is_pole_odd':22s} pass={str(cheat['pass']):5s} "
        f"align={cheat['odd_align']:.3f} leak={cheat['leak_ratio']:+.3f} "
        f"(pole-name cheat must still leak)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
