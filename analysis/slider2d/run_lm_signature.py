#!/usr/bin/env python3
"""Score live Music 3 hold-failure signatures on the CPU fixture."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from analysis.slider2d.signature import (
    HIGH_D,
    LIVE_CAPTION_AXIS,
    compact,
    shrink_factor,
    signature_table,
)
from conceptmod.textsliders.slider_targets import LEAK_HOLD_WEIGHT
from conceptmod.textsliders.train_lm_slider_music3 import parse_args, resolve_lm_recipe


DEFAULT_OUT = _REPO / "docs" / "lm-live-signature"


def _fmt(ok: bool) -> str:
    return "yes" if ok else "no"


def _num(value, digits=3) -> str:
    if value is None:
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    if math.isnan(v):
        return "—"
    return f"{v:+.{digits}f}"


def _row_md(row: dict) -> str:
    col = row.get("collapse_live", row.get("collapse", 0.0))
    if row.get("polarity_undefined"):
        col_s = f"{row.get('collapse', 0):+.3f}†"
    else:
        col_s = f"{row.get('collapse', 0):+.3f}"
    return (
        f"| `{row['name']}` | {row.get('dim', 2)} | {row.get('hold_weight', 0):g} | "
        f"{_num(row.get('e_dot_u'))} | {_num(row.get('e_dot_a'))} | "
        f"{row.get('cos_slider', 0):+.3f} | {row.get('cos_teacher', 0):+.3f} | "
        f"{row.get('leak_ratio', 0):+.3f} | {row.get('perc', 0)*100:.0f} | "
        f"{row.get('loss', 0):.3f} | {col_s} | "
        f"{_fmt(row.get('looks_like_v12', False))} | "
        f"{_fmt(row.get('hold_working', False))} |"
    )


def plot_cplus_vs_slider(rows: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    colors = {
        "gender_like": "#1e8449",
        "pair_odd_no_hold": "#7d3c98",
        "ortho_2d_perp": "#1a5276",
        "leftover_only": "#b9770e",
        "pair_odd_sub_e": "#117a65",
        "highd": "#c0392b",
    }
    for row in rows:
        cell = row.get("cell", "")
        ax.scatter(
            row.get("cos_slider", 0.0),
            row.get("cos_teacher", 0.0),
            c=colors.get(cell, "#444444"),
            s=70,
            zorder=3,
        )
        ax.annotate(
            row["name"].replace("highd_", "").replace("leftover_only_", "λ="),
            (row.get("cos_slider", 0.0), row.get("cos_teacher", 0.0)),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
        )
    ax.axhspan(0.90, 1.02, facecolor="#1e8449", alpha=0.08, label="v12-looking c+")
    ax.axvspan(0.90, 1.02, facecolor="#1a5276", alpha=0.06, label="slider lock")
    ax.axhline(0.70, color="#888888", ls="--", lw=0.7, label="hold-working c+ ~0.70")
    ax.set_xlabel("slider-cos  (cos with û)")
    ax.set_ylabel("trainer c+  (cos with a)")
    ax.set_title("c+ vs slider-cos  ·  copied pair-odd is not slider lock")
    ax.set_xlim(-0.15, 1.05)
    ax.set_ylim(-0.15, 1.08)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_shrink(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    dims = (2, 8, 16, 32, 64)
    xs = list(dims)
    ys = [shrink_factor(d, LEAK_HOLD_WEIGHT) for d in dims]
    ax.plot(xs, ys, "o-", color="#c0392b", label="1 / (1 + λ D/2), λ=8")
    ax.axhline(1.0 / 9.0, color="#888888", ls="--", lw=0.7, label="2-D 1/(1+λ)")
    ax.set_xlabel("hidden dim D")
    ax.set_ylabel("student_ê / teacher_ê")
    ax.set_title("High-D hold shrink when ê ∥ a")
    ax.set_ylim(0.0, 0.15)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_report(rows: list[dict], blob: dict, path: Path) -> None:
    by = {r["name"]: r for r in rows}
    gender = by["gender_like_hold0"]
    perp = by["energy_perp_l8"]
    nohold = by["pair_odd_no_hold"]
    lam1 = by["leftover_only_l1"]
    lam8 = by["leftover_only_l8"]
    sub = by["pair_odd_sub_e"]
    raw = by["highd_synonym_raw_l8"]
    pin = by["highd_pin_caption037_l8"]
    tiny = by["highd_tiny_unused_slider_l8"]
    highd_unused = by["highd_unused_slider_l8"]
    closed = shrink_factor(HIGH_D, LEAK_HOLD_WEIGHT)
    args = parse_args(["--prompts_file", "prompts.yaml"])
    default_v9 = args.lm_target == "v9" and resolve_lm_recipe(lm_target="v9", symmetric=True) == "v9"

    leftover_canary = (
        abs(lam1["leak_ratio"]) > 0.20
        and lam1["cos_teacher"] > 0.90
        and lam1["collapse"] <= -0.85
        and lam1["loss"] < 1.0
    )
    next_is_sub = leftover_canary and abs(sub["leak_ratio"]) <= 0.20 and sub["cos_slider"] > 0.90

    lines = [
        "# Live Music 3 hold failure on the CPU fixture",
        "",
        "Existing overlap / rich / faithful cells on main are orthonormal 2-D.",
        "They show ê·û overlap and ê_⊥û locking slider-cos. They do **not**",
        "make trainer c+ (cos with a) vs slider-cos (cos with û) a test, and",
        "they never break ±1 polarity. This page is those missing live bullets",
        "with real fixture numbers.",
        "",
        "CPU only. No Hub, no GPU, no Music 3 weights.",
        f"Live `--lm_target v9` default is unchanged ({default_v9}).",
        "",
        "## Verdict",
        "",
    ]
    lines.append(
        f"Gender-like hold 0 copies pair-odd: c+ {gender['cos_teacher']:+.3f}, "
        f"slider-cos {gender['cos_slider']:+.3f}, collapse {gender['collapse']:+.3f}, "
        f"loss {gender['loss']:.3f} — the v12-looking / gender-v14 look. "
        f"No junk ê on that cell."
    )
    lines.append("")
    lines.append(
        f"Energy ê synonym + ê_⊥û λ=8 on orthonormal 2-D: slider-cos "
        f"{perp['cos_slider']:+.3f}, leftover {perp['leak_ratio']:+.3f}, "
        f"c+ {perp['cos_teacher']:+.3f}, perc {perp['perc']*100:.0f}%, "
        f"loss {perp['loss']:.3f}. PASS leftover lock, FAIL looks-like-v12 "
        f"(pair-odd no-hold on the same poles is c+ {nohold['cos_teacher']:+.3f}, "
        f"loss {nohold['loss']:.3f}, perc {nohold['perc']*100:.0f}%). "
        f"Hold is *supposed* to make fit-to-pair-odd worse. Slider lock is "
        f"alignment with û, not c+."
    )
    lines.append("")
    lines.append(
        f"High-D (D={HIGH_D}) ê≈a raw λ=8: student / teacher "
        f"{raw['strength']:.4f} vs closed form {closed:.4f}. "
        f"c+ {raw['cos_teacher']:+.3f} stays parallel; raw collapse "
        f"{raw['collapse']:+.3f} is still −1 because a linear odd residual "
        f"of εa and −εa is antipodal. ||d+|| / ||a|| < 0.05 so the live-log "
        f"cosine is undefined (reported collapse_live "
        f"{raw['collapse_live']:+.3f}) — the closest analogue this field "
        f"can make to energy-v14 collapse +0.18. Loss {raw['loss']:.3f} is "
        f"not in the 0.02 band; perc {raw['perc']*100:.0f}%."
    )
    lines.append("")
    lines.append(
        f"Synonym pin (medium energy on both leak captions, density/genre "
        f"still = poles, short û at caption-axis {LIVE_CAPTION_AXIS}): "
        f"ê·û {pin['e_dot_u']:+.3f}, hold·â {pin.get('hold_dot_a', 0):+.3f}, "
        f"c+ {pin['cos_teacher']:+.3f}, slider-cos {pin['cos_slider']:+.3f}, "
        f"perc {pin['perc']*100:.0f}%, collapse {pin['collapse']:+.3f}. "
        f"ê_⊥ after dropping the 0.37 short û is still ≈ a, so λ=8 still "
        f"fights the teacher. Tiny unused leftover after ortho is **not** "
        f"this miss: unit-normalize turns it into leftover-only hold "
        f"(high-D unused slider-cos {highd_unused['cos_slider']:+.3f}, "
        f"tiny-unused {tiny['cos_slider']:+.3f})."
    )
    lines.append("")
    if next_is_sub:
        lines.append(
            f"**leftover-only ê + λ=1 is the canary, not the next live wire.** "
            f"It trains (c+ {lam1['cos_teacher']:+.3f}, loss {lam1['loss']:.3f}, "
            f"collapse {lam1['collapse']:+.3f}) but leftover stays "
            f"{lam1['leak_ratio']:+.3f}. λ=8 leftover-only locks leak "
            f"({lam8['leak_ratio']:+.3f}) and is hold-working, not v12-looking "
            f"(c+ {lam8['cos_teacher']:+.3f}, perc {lam8['perc']*100:.0f}%). "
            f"`pair_odd_sub_e` zeros leak ({sub['leak_ratio']:+.3f}) and locks "
            f"slider-cos {sub['cos_slider']:+.3f}, but c+ vs full pair-odd is "
            f"{sub['cos_teacher']:+.3f} — the teacher changed. Compare only. "
            f"Do not wire it. Do not change `--lm_target v9`."
        )
    else:
        lines.append(
            "leftover-only ê + λ=1 did not land the expected canary band; "
            "see the table. Do not wire pair_odd_sub_e. Do not change "
            "`--lm_target v9`."
        )
    lines += [
        "",
        "v12 / Hub is not leak-free on energy. Gender’s 0.97 c+ is “I copied",
        "pair-odd.” Energy will never look like that if hold is working.",
        "",
        "## Table",
        "",
        "† = residual so small the live-log ±1 cosine is undefined (closest",
        "analogue to energy-v14 collapse +0.18). Linear odd residual of εa",
        "and −εa still prints raw collapse −1.",
        "",
        "| cell | D | λ | ê·û | ê·â | slider-cos | c+ | leak | perc% | loss | ±1 | v12-looking | hold-working |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(_row_md(row))
    lines += [
        "",
        "![c+ vs slider-cos](lm-live-signature/cplus_vs_slider.png)",
        "",
        "![high-D shrink closed form](lm-live-signature/shrink.png)",
        "",
        "## What is now a 2-D / high-D cell",
        "",
        "- Gender-like copied pair-odd (hold 0, no junk ê): c+ / collapse /",
        "  slider-cos as first-class columns. This *is* the v12-looking look.",
        "- Energy ê synonym + ê_⊥û λ=8 on orthonormal 2-D: leftover PASS,",
        "  looks-like-v12 FAIL (c+ ~0.70, perc ~72%).",
        "- leftover-only ê × λ ∈ {0.3, 1, 8} vs `pair_odd_sub_e` on the same",
        "  leaky energy poles.",
        "- High-D ê≈a raw λ=8: closed-form shrink `1/(1+λD/2)`, residual → 0,",
        "  perc stuck, polarity cosine undefined.",
        "- Synonym pin: short û at live caption-axis 0.37, ê≈a, ê_⊥û still",
        "  fights a. λ=8 still bad.",
        "",
        "## What this field still cannot see",
        "",
        "- Live collapse **+0.18** as a converged linear-odd number. A",
        "  multiplier residual of εa and −εa is antipodal even when ε→0.",
        "  Live Qwen LoRA is nonlinear in hidden space; +1 and −1 need not",
        "  stay opposite once the residual is numerically gone. The fixture",
        "  reports that as undefined / 0, not +0.18.",
        "- Loss 278 by step 13. High-D MSE is a mean over D, so the same",
        "  fight prints a *small* loss (~0.09 at D=32) and a huge perc.",
        "  The explosion is an AR / optimizer event this field does not have.",
        "- A tiny unused leftover that is *weaker* than leftover-only hold.",
        "  `lm_axis_hold` unit-normalizes ê_⊥. Tiny unused = unused.",
        "",
        "## Geometry",
        "",
        "```",
        "MSE  = mean_D ||student − teacher||²     # 1/D factor",
        "hold = ½ ((d+·ê)² + (d−·ê)²)            # not divided by D",
        "w_ê  = a_ê / (1 + λ D/2)",
        "```",
        "",
        f"At D=2, λ=8 this is `1/9`. At D={HIGH_D} it is `{closed:.4f}`.",
        "2-D stays parallel to the teacher (c+ high) whenever ê ∥ a.",
        "High-D zeros that component. If ê≈a, the whole residual → 0.",
        "",
        "Synonym pin: leak captions both say medium energy (even cancels).",
        "Density / slammed vs airy still encodes the poles, so ê≈a.",
        f"Short `slider_positive` is loud/calm at ê·û = {LIVE_CAPTION_AXIS}.",
        "ê_⊥ = ê − (ê·û)û still ≈ a. λ=8 still fights.",
        "",
        "## How to run",
        "",
        "```bash",
        "PYTHONPATH=. python analysis/slider2d/run_lm_signature.py --out docs/lm-live-signature",
        "PYTHONPATH=. pytest tests/test_lm_signature.py tests/test_lm_hold_overlap.py tests/test_lm_live_cells.py -q",
        "```",
        "",
        f"Seed `{blob['seed']}`, `{blob['steps']}` Adam steps.",
        "",
        "See [lm-hold-overlap.md](lm-hold-overlap.md) for the orthonormal",
        "ê·û sweep and [lm-live-cells.md](lm-live-cells.md) for the default",
        "v9 gender/energy cells. Do not change `--lm_target v9`.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = signature_table(steps=args.steps, seed=args.seed)
    blob = {
        "steps": args.steps,
        "seed": args.seed,
        "high_d": HIGH_D,
        "caption_axis": LIVE_CAPTION_AXIS,
        "shrink_closed": shrink_factor(HIGH_D, LEAK_HOLD_WEIGHT),
        "lm_target_default": parse_args(["--prompts_file", "prompts.yaml"]).lm_target,
        "rows": [compact(r) for r in rows],
    }
    (out / "metrics.json").write_text(json.dumps(blob, indent=2) + "\n", encoding="utf-8")
    plot_cplus_vs_slider(rows, out / "cplus_vs_slider.png")
    plot_shrink(out / "shrink.png")
    write_report(rows, blob, out.parent / "lm-live-signature.md")
    for row in rows:
        print(
            f"{row['name']:32s} c+={row.get('cos_teacher', 0):+.3f} "
            f"sl={row.get('cos_slider', 0):+.3f} leak={row.get('leak_ratio', 0):+.3f} "
            f"col={row.get('collapse', 0):+.3f} perc={row.get('perc', 0)*100:.0f} "
            f"loss={row.get('loss', 0):.3f} v12={row.get('looks_like_v12')} "
            f"hold={row.get('hold_working')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
