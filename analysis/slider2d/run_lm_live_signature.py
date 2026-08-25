#!/usr/bin/env python3
"""Generate the live-signature fixture table (c+ vs slider-cos, high-D, canary λ)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from analysis.slider2d.live_signature import (
    HIGH_D_DEFAULT_DIM,
    signature_table,
    verdict_table,
)
from conceptmod.textsliders.slider_targets import LEAK_HOLD_WEIGHT

DEFAULT_OUT = _REPO / "docs" / "lm-live-signature"


def _row_md(row: dict) -> str:
    return (
        f"| `{row['name']}` | {row['dims']} | {row['hold_weight']:g} | "
        f"{row['overlap']:.2f} | {row['hold_e_perp_norm']:.3f} | "
        f"{row['cos_slider']:+.3f} | {row['cos_teacher']:+.3f} | "
        f"{row['collapse']:+.3f} | {row['leak']:+.3f} | "
        f"{row['perc']*100:.0f} | {row['loss']:.3f} | "
        f"{'yes' if row['v12_looking'] else 'no'} | "
        f"{'yes' if row['hold_working_not_v12'] else 'no'} |"
    )


def write_report(blob: dict, path: Path) -> None:
    rows = blob["signature"]
    verdicts = blob["verdict"]
    lines = [
        "# Live Music 3 failure signatures (CPU fixture)",
        "",
        "Maps live energy-v14 / gender-v14 trainer logs onto orthonormal 2-D",
        "and a minimal high-D cell. **c+** is trainer alignment with pair-odd",
        "``a``; **slider-cos** is alignment with declared ``û``. They diverge",
        "when hold is working on energy — do not read c+ as slider lock.",
        "",
        "CPU only. No Hub, no GPU, no Music 3 weights. Does not change",
        "``--lm_target v9`` default.",
        "",
        "## Signature table",
        "",
        "| cell | D | λ | ê·û | ‖ê_⊥‖ | slider-cos | c+ | collapse | leak | perc% | loss | v12-looking | hold≠v12 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(_row_md(row))
    lines += [
        "",
        "## Verdict: live bullet → fixture",
        "",
        "| live bullet | fixture | visible | slider-cos | c+ | verdict |",
        "|---|---|---:|---:|---:|---|",
    ]
    for v in verdicts:
        lines.append(
            f"| {v['live_bullet']} | `{v['fixture']}` | {v['visible']} | "
            f"{v['cos_slider']:+.3f} | {v['cos_teacher']:+.3f} | {v['verdict']} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- **Gender-like** copies full pair-odd (hold λ=0). c+ and slider-cos",
        "  are both high — the look people misread as “slider locked”.",
        "- **Energy ê_⊥û** at ρ=0.5 locks slider-cos and leftover leak but",
        "  c+ ~0.70 and loss ~0.85 — hold working, not v12-looking.",
        "- **High-D** shrinks the ê_⊥ residual; c+ floors near |odd·û|.",
        "  Symmetric ``odd_even`` keeps collapse antipodal; live +0.18 is",
        "  not reproduced on this fixture.",
        "- **Synonym pin** (ρ=0.95, raw hold): ê_⊥ is tiny; λ=8 still fights.",
        "- **Leftover-only ê + λ=1** is the canary before ``pair_odd_sub_e``.",
        "- **pair_odd_sub_e** zeros unused leak but uses a projected teacher",
        "  (c+ ~ align) — compare to hold, do not wire live without a cell win.",
        "",
        "## How to run",
        "",
        "```bash",
        "PYTHONPATH=. python analysis/slider2d/run_lm_live_signature.py --out docs/lm-live-signature",
        "PYTHONPATH=. pytest tests/test_lm_live_signature.py -q",
        "```",
        "",
        f"Seed `{blob['seed']}`, `{blob['steps']}` Adam steps, high-D dim `{blob['high_d_dim']}`.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--high-d-dim", type=int, default=HIGH_D_DEFAULT_DIM)
    args = parser.parse_args(argv)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    rows = signature_table(steps=args.steps, seed=args.seed, high_d_dim=args.high_d_dim)
    blob = {
        "steps": args.steps,
        "seed": args.seed,
        "high_d_dim": args.high_d_dim,
        "leak_hold_weight": LEAK_HOLD_WEIGHT,
        "signature": rows,
        "verdict": verdict_table(rows),
    }
    (out / "metrics.json").write_text(json.dumps(blob, indent=2) + "\n", encoding="utf-8")
    write_report(blob, out.parent / "lm-live-signature.md")
    print(f"wrote {out / 'metrics.json'}")
    for row in rows:
        print(
            f"  {row['name']:32s} slider={row['cos_slider']:+.3f} "
            f"c+={row['cos_teacher']:+.3f} leak={row['leak']:+.3f} "
            f"col={row['collapse']:+.3f} loss={row['loss']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
