#!/usr/bin/env python3
"""Score every honest faithful knob on the leaky energetic×gender CPU field."""

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

from analysis.slider2d.faithful import (
    HOLD_LAMBDAS,
    floatable,
    leak_field_table,
    leak_teacher,
    pair_odd_hold_sweep,
    score_energy_faithful,
    score_energy_odd,
    score_mismatch_faithful,
    teacher_geometry,
    hold_e_shrink,
)
from analysis.slider2d.field import Field2D


DEFAULT_OUT = _REPO / "docs" / "lm-faithful-2d"


def _by_name(rows: list[dict]) -> dict[str, dict]:
    return {r["name"]: r for r in rows}


def plot_compare(rows: list[dict], path: Path) -> None:
    want = _by_name(rows)
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.axhline(0, color="#dddddd", lw=0.6)
    ax.axvline(0, color="#dddddd", lw=0.6)
    styles = {
        "lm_faithful_raw": ("#922b21", "faithful (raw poles)"),
        "lm_faithful_hold_l8": ("#b9770e", "faithful + hold ê λ=8"),
        "lm_faithful_hold_l32": ("#af601a", "faithful + hold ê λ=32"),
        "lm_v9": ("#1e8449", "v9 pair-odd + hold ê λ=8"),
        "lm_faithful_attrs": ("#6c3483", "faithful + attributes"),
    }
    for name, (color, label) in styles.items():
        row = want[name]
        dp = row["delta_plus"]
        dm = row["delta_minus"]
        ax.arrow(
            0, 0, dp[0], dp[1],
            color=color, width=0.02, length_includes_head=True, head_width=0.10, alpha=0.95,
        )
        ax.arrow(
            0, 0, dm[0], dm[1],
            color=color, width=0.02, length_includes_head=True, head_width=0.10, alpha=0.40,
        )
        ax.annotate(label, (dp[0] + 0.04, dp[1] + 0.04), fontsize=8, color=color)
    ax.set_xlabel("slider  (calm ← → energetic)")
    ax.set_ylabel("attribute  (female ← → male)")
    ax.set_title("+1 (solid) / −1 (faint)  ·  hold-ê shrinks pole ê, does not copy it")
    ax.set_aspect("equal")
    ax.set_xlim(-2.0, 2.4)
    ax.set_ylim(-1.2, 2.4)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_sweep(faithful: list[dict], odd: list[dict], geo: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    # Hold-grid rows only (skip hub / attrs).
    f_rows = [r for r in faithful if r["name"].startswith("lm_faithful_") and "hub" not in r["name"] and "attrs" not in r["name"]]
    o_rows = odd
    f_x = list(HOLD_LAMBDAS)
    f_y = [abs(r["leak_ratio"]) for r in f_rows]
    o_y = [abs(r["leak_ratio"]) for r in o_rows]
    theory_f = [abs(hold_e_shrink(geo["teacher_leak"], lam)) for lam in f_x]
    theory_o = [abs(hold_e_shrink(geo["teacher_odd_e"] / abs(geo["teacher_slider_plus"]), lam)) for lam in f_x]
    ax.plot(f_x, f_y, "o-", color="#922b21", label="faithful + hold ê (fit)")
    ax.plot(f_x, theory_f, "--", color="#922b21", alpha=0.45, label="faithful  teacher_ê/(1+λ)")
    ax.plot(f_x, o_y, "s-", color="#1e8449", label="pair-odd + hold ê (fit)")
    ax.plot(f_x, theory_o, "--", color="#1e8449", alpha=0.45, label="pair-odd  odd_ê/(1+λ)")
    ax.axhline(0.20, color="#888888", ls=":", lw=0.8, label="leak gate 0.20")
    ax.set_xlabel("hold-ê weight λ")
    ax.set_ylabel("|attr| / |slider| leak")
    ax.set_title("Hold shrinks whatever ê the teacher asked for")
    ax.legend(fontsize=8)
    ax.set_xlim(-1, 68)
    ax.set_ylim(-0.05, 1.85)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_report(
    rows: list[dict],
    odd_sweep: list[dict],
    geo: dict,
    energy_f: dict,
    energy_o: dict,
    energy_fh: dict,
    energy_oh: dict,
    mismatch: dict,
    blob: dict,
    path: Path,
) -> None:
    by_name = _by_name(rows)
    raw = by_name["lm_faithful_raw"]
    h8 = by_name["lm_faithful_hold_l8"]
    h32 = by_name["lm_faithful_hold_l32"]
    attrs = by_name["lm_faithful_attrs"]
    hub_f = by_name["lm_faithful_hub"]
    v9 = by_name["lm_v9"]
    sym = by_name["lm_symmetric"]
    wins = [r for r in rows if r["wins_while_faithful"]]
    if wins:
        recipe = ", ".join(f"`{r['name']}` (leak {r['leak_ratio']:+.3f})" for r in wins)
        verdict = (
            f"**Yes, but only by cleaning the captions** — {recipe}. "
            "On the leaky ungated pair, no knob that keeps raw-pole MSE as the "
            "teacher also copies those poles *and* gets leak ≤ 0.20: hold-ê "
            f"at the live λ=8 leaves leak {h8['leak_ratio']:+.3f} "
            f"(pole cos {h8['pole_cos_plus']:.3f}); higher λ can pass the leak "
            "gate only by refusing to copy ê. Pair-odd + hold-ê (`lm_v9`, "
            f"leak {v9['leak_ratio']:+.3f}) is the nearest recipe that works "
            "on leaky captions without rewriting them. Do not wire "
            "faithful+hold as the live default."
        )
    else:
        verdict = (
            "**No.** On the leaky pair, ê already lives in `h± − h0`. "
            "A student that still copies the poles keeps that leak; a student "
            "that kills leak has stopped being a faithful fit. Pair-odd + "
            f"hold-ê (`lm_v9`, leak {v9['leak_ratio']:+.3f}) is the nearest "
            "recipe that works on leaky captions. Do not wire faithful+hold "
            "as the live default."
        )

    main_names = [
        "lm_faithful_raw",
        "lm_faithful_hold_l1",
        "lm_faithful_hold_l8",
        "lm_faithful_hold_l32",
        "lm_faithful_hold_l64",
        "lm_faithful_hub",
        "lm_faithful_attrs",
        "lm_symmetric",
        "lm_v9_hub",
        "lm_v9_project",
        "lm_v9",
    ]
    lines = [
        "# Can faithful / v6 be fixed on the 2-D field?",
        "",
        "Same energetic×gender geometry as [2d-analysis.md](2d-analysis.md)",
        "and [lm-v9-2d.md](lm-v9-2d.md). Teacher for `faithful` is the raw",
        "poles: MSE to `h+` / `h−` (`--lm_target faithful` / v6). This scores",
        "every honest knob that keeps that teacher, then the non-faithful",
        "comparables. CPU only. No Hub, no GPU, no Music 3 weights.",
        "",
        "Endreg / planreg stay AR-only. `pole_mode: semantic_kl` vs hidden",
        "MSE onto the pair-odd midpoint is the sheet cell in",
        "[lm-lyric-garble.md](lm-lyric-garble.md) — this field cannot see",
        "off-sheet singing. `--common_beta` is ignored in faithful mode (the",
        "function returns `pos, neg`). `--target_scale` is symmetric-only.",
        "",
        "## Verdict",
        "",
        verdict,
        "",
        f"Ungated teacher leak **{geo['teacher_leak']:.3f}** (even ê {geo['teacher_even_e']:.3f},",
        f"odd ê {geo['teacher_odd_e']:.3f}; even ∥ ê, `cos={geo['even_cos_e']:.3f}`).",
        "Closed form: `student_ê = teacher_ê / (1+λ)`.",
        "",
        "## Leaky energetic×gender",
        "",
        "| method | teacher | slider | leak | ±1 | faithful fit | slider cos | leak | ±1 cos | pole cos | ê copied |",
        "|---|---|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for name in main_names:
        r = by_name[name]
        teacher = "raw poles" if r["teacher_faithful"] else "not faithful"
        fit = "yes" if r["faithful_fit"] else "no"
        lines.append(
            f"| `{r['name']}` | {teacher} | **{r['axis']['slider']}** | **{r['axis']['leak']}** | "
            f"**{r['axis']['collapse']}** | {fit} | {r['cos_slider_plus']:.3f} | "
            f"{r['leak_ratio']:+.3f} | {r['cos_plus_minus']:+.3f} | "
            f"{r['pole_cos_plus']:.3f} | {r['e_copied_frac']:.2f} |"
        )
    lines += [
        "",
        "![faithful vs hold vs v9 vs attributes](lm-faithful-2d/compare.png)",
        "",
        "![leak vs λ: faithful hold vs pair-odd hold](lm-faithful-2d/sweep.png)",
        "",
        "## What each knob does",
        "",
        f"- `lm_faithful_raw` / v6: MSE to `h+`/`h−`. Leak {raw['leak_ratio']:+.3f},",
        f"  ±1 cos {raw['cos_plus_minus']:+.3f}, pole cos {raw['pole_cos_plus']:.3f}.",
        "  Copies the poles, including unused gender and the shared even mode.",
        f"- `lm_faithful_hold_l*` : same teacher + hold-ê. Closed form",
        f"  leftover leak = {geo['teacher_leak']:.3f}/(1+λ).",
        f"  λ=8 (live v9 weight): leak {h8['leak_ratio']:+.3f}, pole cos {h8['pole_cos_plus']:.3f},",
        f"  ê copied {h8['e_copied_frac']:.2f} — gates can look close, but the fit",
        f"  is no longer the poles. λ=32: leak {h32['leak_ratio']:+.3f}, pole cos",
        f"  {h32['pole_cos_plus']:.3f}. Passing the leak gate means abandoning the poles.",
        f"- `lm_faithful_hub`: raw-pole MSE + published floor/anchor (κ blend).",
        f"  Leak {hub_f['leak_ratio']:+.3f}. Anchor only sizes even blend-back; it",
        "  does not take ê out of the odd teacher, and it does not beat raw leak.",
        f"- `lm_faithful_attrs`: `--attributes male,female` pins gender on every",
        f"  caption. Teacher is still raw poles, but those poles are clean.",
        f"  Leak {attrs['leak_ratio']:+.3f}, ±1 {attrs['cos_plus_minus']:+.3f},",
        f"  pole cos {attrs['pole_cos_plus']:.3f}. This is the data fix.",
        f"- `lm_symmetric` / pair-odd: `t± = h0 ± (h+−h−)/2`. Collapse fixed",
        f"  ({sym['cos_plus_minus']:+.3f}); leak {sym['leak_ratio']:+.3f} stays in `a`.",
        "- `lm_v9_hub`: published Hub leash. Same leaked odd axis as pair-odd.",
        "- `lm_v9_project`: short-û project+hold. Leak-0 here because û *is*",
        "  the pole names — the old cheat, not energy, not the default.",
        f"- `lm_v9`: current default, pair-odd + hold-ê λ=8. Leak {v9['leak_ratio']:+.3f},",
        f"  ±1 {v9['cos_plus_minus']:+.3f}. Nearest thing that works on *leaky*",
        "  captions without rewriting them.",
        "",
        "## Energy-like and mismatch (already-odd poles)",
        "",
        "On the energy-like cell the structured poles are already odd around",
        "neutral (`h+ = −h−`). Faithful and pair-odd are the same teacher.",
        "Hold-ê is then exactly current v9 — not a new faithful fix.",
        "Mismatch is a clean pair: faithful already has leak 0 because ê is",
        "not in the poles.",
        "",
        "| cell | method | leak | ±1 cos | pass |",
        "|---|---|---:|---:|---|",
        f"| energy-like | faithful λ=0 | {energy_f['leak_ratio']:+.3f} | {energy_f['cos_plus_minus']:+.3f} | {energy_f['pass']} |",
        f"| energy-like | pair-odd λ=0 | {energy_o['leak_ratio']:+.3f} | {energy_o['cos_plus_minus']:+.3f} | {energy_o['pass']} |",
        f"| energy-like | faithful + hold λ=8 | {energy_fh['leak_ratio']:+.3f} | {energy_fh['cos_plus_minus']:+.3f} | {energy_fh['pass']} |",
        f"| energy-like | pair-odd + hold λ=8 | {energy_oh['leak_ratio']:+.3f} | {energy_oh['cos_plus_minus']:+.3f} | {energy_oh['pass']} |",
        f"| mismatch (clean) | faithful | {mismatch['leak_ratio']:+.3f} | {mismatch['cos_plus_minus']:+.3f} | {mismatch['pass']} |",
        "",
        "## Why hold-ê cannot keep the name and kill leak",
        "",
        "```",
        "faithful teacher:   t± = h±                         # ê is inside",
        "pair-odd teacher:   t± = h0 ± (h+ − h−)/2           # even ê dropped",
        "hold:               L += λ · ((h(±1)−h0) · ê)²",
        "equilibrium:        student_ê = teacher_ê / (1+λ)",
        "```",
        "",
        f"Teacher +1 is already leak {geo['teacher_leak']:.3f}. Any residual with",
        "leak ≤ 0.20 has cosine ≲ 0.67 to that pole — below the 0.90 copy gate.",
        "The only way to stay faithful *and* leak-0 is to change the captions",
        "so the poles no longer contain ê.",
        "",
        "## What this field cannot see",
        "",
        "- AR endreg / planreg (v6 had those; not expressed here).",
        "  Semantic-KL vs the v15 midpoint Goodhart is",
        "  [lm-lyric-garble.md](lm-lyric-garble.md).",
        "- Real Music 3 hidden geometry, Hub weights, multi-row yaml averaging.",
        "- A third unused axis that is not the even mode. Here even is parallel",
        "  to ê, so hold-ê also fixes collapse. That coincidence is this field,",
        "  not a general promise.",
        "",
        "## How to run",
        "",
        "```bash",
        "PYTHONPATH=. python analysis/slider2d/run_lm_faithful.py --out docs/lm-faithful-2d",
        "PYTHONPATH=. pytest tests/test_lm_faithful_2d.py tests/test_lm_v9_2d.py -q",
        "```",
        "",
        "CPU only. No Hub, no GPU, no Music 3 weights.",
        "",
        f"Seed `{blob['seed']}`, `{blob['steps']}` Adam steps.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    rows = leak_field_table(steps=args.steps, seed=args.seed)
    odd_sweep = pair_odd_hold_sweep(steps=args.steps, seed=args.seed)
    geo = teacher_geometry(*leak_teacher(Field2D()))
    energy_f = score_energy_faithful(hold_weight=0.0, steps=args.steps, seed=args.seed)
    energy_o = score_energy_odd(hold_weight=0.0, steps=args.steps, seed=args.seed)
    energy_fh = score_energy_faithful(hold_weight=8.0, steps=args.steps, seed=args.seed)
    energy_oh = score_energy_odd(hold_weight=8.0, steps=args.steps, seed=args.seed)
    mismatch = score_mismatch_faithful(steps=args.steps, seed=args.seed)

    blob = {
        "steps": args.steps,
        "seed": args.seed,
        "geometry": geo,
        "methods": {r["name"]: floatable(r) for r in rows},
        "odd_sweep": {r["name"]: floatable(r) for r in odd_sweep},
        "energy_faithful": floatable(energy_f),
        "energy_odd": floatable(energy_o),
        "energy_faithful_hold": floatable(energy_fh),
        "energy_odd_hold": floatable(energy_oh),
        "mismatch_faithful": floatable(mismatch),
    }
    (out / "metrics.json").write_text(json.dumps(blob, indent=2) + "\n", encoding="utf-8")
    plot_compare(rows, out / "compare.png")
    plot_sweep(rows, odd_sweep, geo, out / "sweep.png")
    write_report(
        rows, odd_sweep, geo, energy_f, energy_o, energy_fh, energy_oh, mismatch, blob,
        out.parent / "lm-faithful-2d.md",
    )
    for r in rows:
        print(
            f"{r['name']:24s} leak={r['leak_ratio']:+.3f} ±1={r['cos_plus_minus']:+.3f} "
            f"pole={r['pole_cos_plus']:.3f} ê={r['e_copied_frac']:.2f} "
            f"fit={r['faithful_fit']} gates={r['gates']} win={r['wins_while_faithful']}"
        )
    print(
        f"energy faithful/odd λ=0  leak={energy_f['leak_ratio']:+.3f}/{energy_o['leak_ratio']:+.3f} "
        f"λ=8 leak={energy_fh['leak_ratio']:+.3f}/{energy_oh['leak_ratio']:+.3f}"
    )
    print(
        f"mismatch faithful leak={mismatch['leak_ratio']:+.3f} "
        f"±1={mismatch['cos_plus_minus']:+.3f} pass={mismatch['pass']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
