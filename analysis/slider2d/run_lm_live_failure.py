#!/usr/bin/env python3
"""Write the CPU live-failure fixture table and short verdict."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from analysis.slider2d.live_failure import fixture_table


DEFAULT_OUT = _REPO / "docs" / "lm-live-failure"


def _table_row(row: dict) -> str:
    return (
        f"| `{row['name']}` | {row['dim']} | {row['teacher']} | "
        f"{row['hold_weight']:g} | {row['trainer_c_plus']:+.3f} | "
        f"{row['slider_cos']:+.3f} | {row['collapse']:+.3f} | "
        f"{row['leftover']:+.3f} | {row['perc']*100:.0f} | "
        f"{row['loss']:.3f} |"
    )


def write_report(rows: list[dict], path: Path) -> None:
    by_name = {row["name"]: row for row in rows}
    held = by_name["energy_2d_hold_l8"]
    highd = by_name["highd_synonym_hold_l8"]
    canary = by_name["leftover_only_hold_l1"]
    hard = by_name["leftover_only_hold_l8"]
    sub = by_name["pair_odd_sub_e"]
    lines = [
        "# Live Music 3 hold failure on a CPU fixture",
        "",
        "Analysis/test/docs only. The live `--lm_target v9` default remains",
        "full pair-odd; this fixture does not wire a teacher or trainer change.",
        "",
        "## Verdict",
        "",
        (
            "The orthonormal 2-D PASS was real but easy to misread: at λ=8 it "
            f"has slider-cos {held['slider_cos']:.3f} while trainer c+ is only "
            f"{held['trainer_c_plus']:.3f}, perc is {held['perc']*100:.0f}%, "
            f"and loss is {held['loss']:.3f}. It locks û and suppresses leftover "
            "ê; it is not a v12-looking copy of pair-odd."
        ),
        "",
        (
            "A D=64 local-Jacobian analogue now reaches the live shape: a tiny "
            f"||ê_⊥û||={highd['hold_norm']:.3f} is normalized before hold, the "
            f"effective scalar-vs-MSE factor is {highd['effective_factor']:.0f}, "
            f"c+={highd['trainer_c_plus']:.3f}, slider-cos="
            f"{highd['slider_cos']:.3f}, collapse={highd['collapse']:+.3f}, "
            f"and loss={highd['loss']:.3f}. Dimension makes the hold hard; "
            "the polarity break additionally requires a shared sign-asymmetric "
            "response mode. Dimension alone cannot move an exactly odd residual "
            "away from collapse −1."
        ),
        "",
        (
            "The same-loudness pin is unchanged after orthogonalization: reducing "
            "the raw slider coefficient does not remove the normalized "
            "density/genre/syntax residual. This reproduces why adding “medium "
            "energy” does not prove ê became leftover-only."
        ),
        "",
        (
            "Leftover-only ê at λ=1 is still the canary, not a result to wire: "
            f"c+={canary['trainer_c_plus']:.3f}, collapse="
            f"{canary['collapse']:+.3f}, loss={canary['loss']:.3f}, leftover="
            f"{canary['leftover']:.3f}. λ=8 stays bipolar and reaches leftover "
            f"{hard['leftover']:.3f} in the direct field. `pair_odd_sub_e` is "
            f"leak-free but has c+={sub['trainer_c_plus']:.3f} against the "
            "original pair-odd: it succeeds by changing the teacher. Try the "
            "leftover-only captions + λ=1 live first; if the normalized high-D "
            "hold still breaks polarity, compare the subtract-ê teacher in a "
            "separate PR."
        ),
        "",
        "## Real fixture numbers",
        "",
        "| cell | D | teacher | λ | trainer c+ | slider cos | collapse | leftover | perc% | loss |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(_table_row(row) for row in rows)
    lines += [
        "",
        "`loss` is the actual fixture objective. For `pair_odd_sub_e` it is",
        "zero against the changed teacher; `pair_odd_mse` in `metrics.json`",
        "keeps the cost against the original full pair-odd visible.",
        "",
        "## What is now visible",
        "",
        "- Gender-like/no-ê copies pair-odd: high c+, high intended-axis cosine,",
        "  bipolar ±1. No junk ê is invented for this row.",
        "- The old D=2 λ=8 PASS exposes trainer c+ and slider-cos separately.",
        "- A tiny normalized high-D ê_⊥ plus sign-asymmetric shared capacity can",
        "  reproduce low c+, high loss, and collapse near the live +0.18.",
        "- Same-loudness wording does not help while the residual density/genre",
        "  direction is unchanged.",
        "- Leftover-only λ∈{0.3,1,8} and pair-odd−ê are directly comparable.",
        "",
        "## Still not visible",
        "",
        "- This is not Qwen and does not prove its exact Jacobian or identify",
        "  which attention/MLP parameters create the shared response mode.",
        "- It does not embed the actual captions, model multi-token hidden states,",
        "  or reproduce the step-13 loss explosion. The synonym and pin geometry",
        "  are declared fixture inputs, not inferred language semantics.",
        "- The fixture shows a possible mechanism for polarity break, not that",
        "  high dimensionality by itself causes one.",
        "",
        "## Geometry",
        "",
        "The live pole loss uses per-element MSE, while hold is a scalar squared",
        "projection. Along a unit held direction in D dimensions, the relative",
        "normal-equation factor is:",
        "",
        "```text",
        "student_e = teacher_e / (1 + λ D / 2)",
        "```",
        "",
        "For D=2 this is `1+λ`, so shrink stays parallel and ±1 remains exactly",
        "opposite. The D=64 cell adds one two-parameter local response:",
        "`p = x a + y b+`, `m = −x a + y b−`. The b modes are hold-free,",
        "share parameters, and have different sign-conditioned Jacobians. Hard",
        "hold suppresses x and selects y, making polarity failure measurable.",
        "",
        "## Run",
        "",
        "```bash",
        "PYTHONPATH=. python3 analysis/slider2d/run_lm_live_failure.py",
        "PYTHONPATH=. python3 -m pytest tests/test_lm_live_failure_fixture.py -q",
        "```",
        "",
        "CPU only. No Hub, GPU, Music 3, or Qwen weights.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = fixture_table()
    (args.out / "metrics.json").write_text(
        json.dumps({"rows": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(rows, args.out.parent / "lm-live-failure.md")
    for row in rows:
        print(
            f"{row['name']:30s} c+={row['trainer_c_plus']:+.3f} "
            f"slider={row['slider_cos']:+.3f} col={row['collapse']:+.3f} "
            f"left={row['leftover']:.3f} loss={row['loss']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
