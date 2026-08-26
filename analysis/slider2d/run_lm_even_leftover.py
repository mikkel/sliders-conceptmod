#!/usr/bin/env python3
"""Search EVEN leftover recipes: exam_divergent True and leak_frac < 0.

Leftover-gate already subtracts unused ê from the odd part and still sits
at leak_frac +0.105. This scores subtract/hold of leftover/blend even
only — caption even stays. CPU only. Does not change the live default.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from analysis.slider2d.exam import CELLS, score_exam
from analysis.slider2d.sheet import leaky_field, score_sheet


DEFAULT_OUT = _REPO / "docs" / "lm-even-leftover"

# Baselines + even leftover candidates. Hidden MSE. Live default stays v9.
CANDIDATES: list[dict] = [
    {"name": "faithful_raw", "teacher": "faithful"},
    {"name": "faithful_sub_e_if_unused", "teacher": "faithful_sub_e_if_unused"},
    {"name": "faithful_guard_e", "teacher": "faithful_guard_e"},
    {"name": "pair_odd_midpoint", "teacher": "pair_odd"},
    {"name": "faithful_sub_even_e", "teacher": "faithful_sub_even_e"},
    {"name": "faithful_sub_even_e_if_unused", "teacher": "faithful_sub_even_e_if_unused"},
    {"name": "faithful_sub_even_e_guard", "teacher": "faithful_sub_even_e_guard"},
    {"name": "faithful_sub_even_blend", "teacher": "faithful_sub_even_blend"},
    {"name": "faithful_sub_even_blend_if_unused", "teacher": "faithful_sub_even_blend_if_unused"},
    {"name": "faithful_sub_even_blend_guard", "teacher": "faithful_sub_even_blend_guard"},
    {"name": "faithful_gate_odd_sub_even", "teacher": "faithful_gate_odd_sub_even"},
    {"name": "faithful_gate_odd_sub_even_blend", "teacher": "faithful_gate_odd_sub_even_blend"},
    {
        "name": "faithful_sub_even_blend_s50",
        "teacher": "faithful_sub_even_blend",
        "even_scale": 0.5,
    },
    {
        "name": "faithful_sub_even_blend_s25",
        "teacher": "faithful_sub_even_blend",
        "even_scale": 0.25,
    },
    {
        "name": "faithful_hold_even_e_l8",
        "teacher": "faithful",
        "hold_weight": 8.0,
        "even_hold": True,
    },
    {
        "name": "gate_odd_hold_even_e_l8",
        "teacher": "faithful_sub_e_if_unused",
        "hold_weight": 8.0,
        "even_hold": True,
    },
]


def _kwargs(cand: dict) -> dict:
    out = {
        "teacher": cand["teacher"],
        "even_scale": float(cand.get("even_scale", 1.0)),
        "even_hold": bool(cand.get("even_hold", False)),
        "hold_weight": float(cand.get("hold_weight", 0.0)),
        "pole_mode": "hidden",
    }
    return out


def score_candidate(cand: dict, *, exam_steps: int, sheet_steps: int, seed: int) -> dict:
    kwargs = _kwargs(cand)
    exam: dict[str, dict] = {}
    for cell in ("divergent", "close", "unused_e"):
        field = CELLS[cell](seed=seed)
        e = field.declared_e()
        row = score_exam(
            cand["name"],
            field,
            leak_dir=e,
            steps=exam_steps,
            seed=seed,
            **kwargs,
        )
        exam[cell] = {
            "pass": bool(row["pass"]),
            "leak_frac": float(row["collapse"]),
            "leftover_leak": row.get("leak_tok"),
            "overlap": row.get("roll_overlap"),
            "swing": row.get("roll_swing_kept"),
            "coherence": row.get("roll_coherence"),
            "reason": row.get("reason"),
            "blend_teacher": row.get("blend_teacher"),
        }
    sheet_field = leaky_field(seed=seed)
    sheet = score_sheet(
        cand["name"],
        sheet_field,
        leak_dir=sheet_field.leak_e(),
        teacher=kwargs["teacher"],
        hold_weight=0.0 if kwargs["even_hold"] else kwargs["hold_weight"],
        steps=sheet_steps,
        seed=seed,
    )
    sheet_out = {
        "pass": bool(sheet["pass"]),
        "leak_frac": float(sheet["collapse"]),
        "leftover_leak": sheet.get("leak_tok"),
        "on_sheet_kept": sheet.get("on_sheet_kept"),
        "sheet_dir_kept": sheet.get("sheet_dir_kept"),
    }
    exam_div = exam["divergent"]
    # Official #36 leak_frac is leftover-sheet collapse. Also log exam collapse.
    hit = bool(exam_div["pass"]) and (
        float(sheet_out["leak_frac"]) < 0.0 or float(exam_div["leak_frac"]) < 0.0
    )
    # Banned Goodhart: leak_frac ≈ −1 and divergent fail.
    goodhart = (not exam_div["pass"]) and float(exam_div["leak_frac"]) < -0.80
    return {
        "name": cand["name"],
        "teacher": cand["teacher"],
        "even_scale": kwargs["even_scale"],
        "even_hold": kwargs["even_hold"],
        "hold_weight": kwargs["hold_weight"],
        "exam_divergent": exam_div["pass"],
        "exam_close": exam["close"]["pass"],
        "exam_unused_e": exam["unused_e"]["pass"],
        "leak_frac_sheet": sheet_out["leak_frac"],
        "leak_frac_divergent": exam_div["leak_frac"],
        "leftover_leak": sheet_out["leftover_leak"],
        "sheet_leftover": sheet_out["pass"],
        "divergent_reason": exam_div["reason"],
        "blend_teacher": exam_div["blend_teacher"],
        "hit": hit and not goodhart,
        "goodhart": goodhart,
        "exam": exam,
        "sheet": sheet_out,
    }


def _f(value, spec: str = "+.3f", empty: str = "N/A") -> str:
    if value is None:
        return empty
    return format(float(value), spec)


def write_markdown(rows: list[dict], path: Path) -> None:
    hits = [r for r in rows if r["hit"]]
    lines = [
        "# Even leftover search: exam_divergent and leak_frac < 0",
        "",
        "Generated by `analysis/slider2d/run_lm_even_leftover.py`. CPU only,",
        "no Hub, no GPU, no Music 3 weights. Does not change the live trainer",
        "default (`--lm_target v9` / `--pole_mode hidden`).",
        "",
        "Leftover-gate subtracts unused ê from the **odd** part and still sits",
        "at leftover-sheet `leak_frac` +0.105. These rows subtract or hold only",
        "the even that is leftover/blend (`ê_⊥` or leak-pair even `ê_even`).",
        "Caption even stays. `t± = h0 ± a` is the banned control.",
        "",
        "A hit is `exam_divergent` True and `leak_frac < 0`. More negative is",
        "better only if divergent still passes. `leak_frac ≈ −1` + fail",
        "divergent is pair-odd / midpoint Goodhart.",
        "",
        f"Hits: **{len(hits)}** of {len(rows)}.",
        "",
        "| recipe | divergent | close | unused_e | leak_frac sheet | leak_frac div | leftover leak | sheet | hit |",
        "|---|---|---|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| `{name}` | {div} | {close} | {unused} | {lfs} | {lfd} | {leak} | {sheet} | {hit} |".format(
                name=row["name"],
                div="pass" if row["exam_divergent"] else "**fail**",
                close="pass" if row["exam_close"] else "**fail**",
                unused="pass" if row["exam_unused_e"] else "**fail**",
                lfs=_f(row["leak_frac_sheet"]),
                lfd=_f(row["leak_frac_divergent"]),
                leak=_f(row["leftover_leak"]),
                sheet="pass" if row["sheet_leftover"] else "**fail**",
                hit="**HIT**" if row["hit"] else ("goodhart" if row["goodhart"] else "—"),
            )
        )
    lines += [
        "",
        "## Why each row",
        "",
    ]
    for row in rows:
        lines.append(
            f"- `{row['name']}` — divergent {row['divergent_reason']}"
            + (" (blend teacher)" if row["blend_teacher"] else "")
        )
    lines += [
        "",
        "## How to run",
        "",
        "```bash",
        "PYTHONPATH=. python analysis/slider2d/run_lm_even_leftover.py --out docs/lm-even-leftover",
        "PYTHONPATH=. pytest tests/test_lm_even_leftover.py -q",
        "```",
        "",
        "CPU only. No Hub, no GPU, no Music 3 weights.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--exam-steps", type=int, default=200)
    parser.add_argument("--sheet-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = [
        score_candidate(
            cand,
            exam_steps=args.exam_steps,
            sheet_steps=args.sheet_steps,
            seed=args.seed,
        )
        for cand in CANDIDATES
    ]
    (args.out / "metrics.json").write_text(
        json.dumps(rows, indent=2, default=str) + "\n", encoding="utf-8"
    )
    write_markdown(rows, args.out / "README.md")
    hits = [r["name"] for r in rows if r["hit"]]
    print(f"wrote {args.out}  hits={hits or 'none'}")


if __name__ == "__main__":
    main()
