#!/usr/bin/env python3
"""Score the Hub v9 LM recipe on the existing energetic×gender CPU field."""

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

from analysis.slider2d.field import Field2D
from analysis.slider2d.run_analysis import plot_quiver
from analysis.slider2d.train import (
    MethodResult,
    axis_verdicts,
    lm_v9_specs,
    run_method,
)
from conceptmod.textsliders.slider_targets import (
    lm_anchor_kappa,
    lm_pair_collapse,
    lm_perfect_fit_collapse,
)


DEFAULT_OUT = _REPO / "docs" / "lm-v9-2d"


def run_v9(steps: int = 250, seed: int = 0) -> list[MethodResult]:
    field = Field2D()
    return [run_method(spec, field, steps=steps, seed=seed) for spec in lm_v9_specs()]


def _pair_geometry(field: Field2D) -> dict:
    pos = field.embed("energetic", 0.5)
    neg = field.embed("calm", 0.5)
    neu = field.embed("song", 0.5)
    r = float(lm_pair_collapse(pos, neg, neu))
    kappa = float(lm_anchor_kappa(pos, neg, neu, -0.9))
    rho2 = (1.0 - r) / (1.0 + r)
    return {
        "r_raw_collapse": r,
        "rho2": rho2,
        "kappa_floor_m09": kappa,
        "perfect_fit_collapse": float(lm_perfect_fit_collapse(kappa, rho2)),
        "odd_leak": float((pos - neg)[1] / ((pos - neg)[0].abs() + 1e-8)),
    }


def plot_compare(results: list[MethodResult], path: Path) -> None:
    """One-axes overlay of raw / symmetric / v9 residuals."""
    want = {r.name: r for r in results}
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.axhline(0, color="#dddddd", lw=0.6)
    ax.axvline(0, color="#dddddd", lw=0.6)
    styles = {
        "lm_raw": ("#922b21", "lm_raw (faithful / v6)"),
        "lm_symmetric": ("#1a5276", "lm_symmetric (v4 polarity)"),
        "lm_v9_hub": ("#b9770e", "lm_v9_hub (published floor+anchor)"),
        "lm_v9": ("#1e8449", "lm_v9 (project odd + hold, κ=0)"),
        "lm_raw_attrs": ("#6c3483", "lm_raw_attrs"),
    }
    for name, (color, label) in styles.items():
        result = want[name]
        dp = result.residual.delta(1.0).detach()
        dm = result.residual.delta(-1.0).detach()
        ax.arrow(
            0, 0, float(dp[0]), float(dp[1]),
            color=color, width=0.02, length_includes_head=True, head_width=0.10, alpha=0.95,
        )
        ax.arrow(
            0, 0, float(dm[0]), float(dm[1]),
            color=color, width=0.02, length_includes_head=True, head_width=0.10, alpha=0.45,
        )
        ax.annotate(label, (float(dp[0]) + 0.04, float(dp[1]) + 0.04), fontsize=8, color=color)
    ax.set_xlabel("slider  (calm ← → energetic)")
    ax.set_ylabel("attribute  (female ← → male)")
    ax.set_title("+1 (solid) / −1 (faint)  ·  fixed v9 drops unused-gender leak")
    ax.set_aspect("equal")
    ax.set_xlim(-2.0, 2.2)
    ax.set_ylim(-1.4, 2.2)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_axis_scatter(results: list[MethodResult], path: Path) -> None:
    """Green only when slider, leak, and ±1 collapse are all right."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for result in results:
        leak = abs(result.metrics["leak_ratio"])
        slider = result.metrics["cos_slider_plus"] * result.metrics["want_sign"]
        axv = axis_verdicts(result.metrics)
        all_right = axv["slider"] == axv["leak"] == axv["collapse"] == "right"
        color = "#1e8449" if all_right else "#c0392b"
        ax.scatter(leak, slider, c=color, s=50)
        ax.annotate(result.name, (leak + 0.01, slider), fontsize=7)
    ax.axvline(0.20, color="#888888", ls="--", lw=0.8)
    ax.axhline(0.90, color="#888888", ls="--", lw=0.8)
    ax.set_xlabel("|attr| / |slider| leak ratio")
    ax.set_ylabel("signed slider cosine")
    ax.set_title("Leak vs slider  (green = all three axes right)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_report(results: list[MethodResult], geo: dict, blob: dict, path: Path) -> None:
    by_name = {r.name: r for r in results}
    v9 = by_name["lm_v9"]
    hub = by_name["lm_v9_hub"]
    raw = by_name["lm_raw"]
    sym = by_name["lm_symmetric"]
    floor = by_name["lm_symmetric_floor"]
    attrs = by_name["lm_raw_attrs"]
    tf = by_name["m3_nmse_axis"]
    v9_ax = axis_verdicts(v9.metrics)
    hub_ax = axis_verdicts(hub.metrics)
    lines = [
        "# LM v9 formulation on the 2-D CPU field",
        "",
        "Same geometry as [2d-analysis.md](2d-analysis.md): energetic ↔ calm",
        "versus unused male ↔ female. This scores the **loss / target math**,",
        "not Hub weights and not caption BPM.",
        "",
        "Published Hub v9 (`leakage_floor` + `anchor_weight`) cannot see",
        "unused-gender leak: that component already lives in the odd teacher",
        "`(pos − neg) / 2`. The default scored here (`lm_v9`) keeps Hub",
        "positives (odd ±1, slider-axis track) and **projects that odd",
        "teacher onto the declared slider direction**, then holds the",
        "orthogonal residual. κ = 0 — no even blend-back. Ungated captions",
        "only; `--attributes` is not used.",
        "",
        "The live `train_lm_slider_music3.py` defaults to this recipe",
        "(`--lm_target v9`) and calls the same functions. Endreg / planreg /",
        "semantic-KL poles are AR-only and are not on this field.",
        "",
        "## Verdict",
        "",
        f"**`lm_v9` is right on all three.** Slider **{v9_ax['slider']}**,",
        f"unused-gender leak **{v9_ax['leak']}**, ±1 collapse",
        f"**{v9_ax['collapse']}**. Published Hub v9 (`lm_v9_hub`) stays",
        f"slider **{hub_ax['slider']}**, leak **{hub_ax['leak']}**,",
        f"collapse **{hub_ax['collapse']}** — same leak as `--symmetric`.",
        "",
        "| method | slider | leak | ±1 collapse | slider cos | leak ratio | ±1 cos |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for r in results:
        ax = axis_verdicts(r.metrics)
        m = r.metrics
        lines.append(
            f"| `{r.name}` | **{ax['slider']}** | **{ax['leak']}** | **{ax['collapse']}** | "
            f"{m['cos_slider_plus']:.3f} | {m['leak_ratio']:.3f} | {m['cos_plus_minus']:.3f} |"
        )
    lines += [
        "",
        "![v9 vs raw / symmetric residuals](lm-v9-2d/compare.png)",
        "",
        "![Learned residuals on the teacher field](lm-v9-2d/quiver.png)",
        "",
        "![Leak vs slider cosine](lm-v9-2d/scatter.png)",
        "",
        "## What each flag does here",
        "",
        f"- `lm_raw` / v6 `target_mode: faithful`: poles are raw `pos`/`neg`.",
        f"  ±1 cos={raw.metrics['cos_plus_minus']:.3f} (even-mode collapse).",
        f"  Leak {raw.metrics['leak_ratio']:.3f} — both poles sit above quiet `song`",
        "  *and* `energetic` already leaks male.",
        f"- `lm_symmetric` / v4 polarity: `tgt(±1) = neu ± (pos−neg)/2`.",
        f"  Collapse **right** ({sym.metrics['cos_plus_minus']:.3f}). Leak",
        f"  {sym.metrics['leak_ratio']:.3f} remains — it lives in the odd teacher.",
        f"- `lm_symmetric_floor`: `leakage_floor=-0.9` with `anchor_weight=0`.",
        f"  Identical to symmetric (leak {floor.metrics['leak_ratio']:.3f},",
        f"  ±1 {floor.metrics['cos_plus_minus']:.3f}). The floor is inert",
        "  without an anchor term.",
        f"- `lm_v9_hub`: published Hub recipe (symmetric + `anchor_weight=0.3`",
        f"  + autocal κ from `leakage_floor=-0.9`). On this pair r={geo['r_raw_collapse']:.3f},",
        f"  κ={geo['kappa_floor_m09']:.3f}. Leak {hub.metrics['leak_ratio']:.3f} —",
        "  same leaked odd axis as `--symmetric`. Blend-back can make leak",
        "  slightly *worse* than plain symmetric.",
        f"- `lm_v9` (default): κ=0 (no even blend-back) + odd teacher",
        f"  projected onto the declared slider direction + orthogonal hold.",
        f"  Leak {v9.metrics['leak_ratio']:.3f}, ±1 {v9.metrics['cos_plus_minus']:.3f},",
        f"  slider cos {v9.metrics['cos_slider_plus']:.3f}. Ungated captions.",
        f"- `lm_raw_attrs`: `--attributes male,female` also zeros leak",
        f"  ({attrs.metrics['leak_ratio']:.3f}) by changing the *captions*.",
        "  That row is the paper disentangle, not the v9 formulation.",
        f"- TF `nmse`+`axis`: leak {tf.metrics['leak_ratio']:.3f}, already",
        "  measured in [tf-leak.md](tf-leak.md). Default TF is not the v9 question.",
        "",
        "## Why the published floor cannot kill unused-gender leak",
        "",
        "Hub v9 as shipped:",
        "",
        "```",
        "a = (h+ − h−) / 2",
        "t± = h0 ± a                         # still the leaked pos−neg",
        "κ = √(ρ² · (1 + floor) / (1 − floor))",
        "anchor± = (1 − κ)(h0 ± a) + κ h±    # even blend-back only",
        "```",
        "",
        f"Ungated pair: r = {geo['r_raw_collapse']:.3f}, odd leak = {geo['odd_leak']:.3f}.",
        "`leakage_floor` sizes how much even mode may return. Unused gender",
        "is already inside `a`.",
        "",
        "Fixed default:",
        "",
        "```",
        "a = (h+ − h−) / 2",
        "â = (a · û) û                       # û = declared slider direction",
        "t± = h0 ± â                         # κ = 0",
        "L = MSE(h(±1), t±) + hold · ||(h(±1)−h0)_⊥û||²",
        "```",
        "",
        "û is the caption pair's declared polarity (energetic↔calm /",
        "`E_SLIDER`), not a male/female prefix. Hold stops leftover even",
        "capacity from walking onto the unused axis.",
        "",
        "## What this field cannot see",
        "",
        "- **Declared û ≠ pole names.** This field sets û from energetic↔calm",
        "  polarity, so project-odd is nearly identity on the intended axis",
        "  (`|odd·û|/||odd|| ≈ 0.95`) and cannot reproduce gender-v1. That",
        "  failure is the mismatch cell in [lm-v9-mismatch.md](lm-v9-mismatch.md).",
        "- AR endreg / planreg / `pole_mode: semantic_kl` / `collapse_weight`",
        "  (v6 had those; v9 turns planreg and collapse off).",
        "- Real Music 3 hidden geometry, Hub weights, v7 prompt yamls.",
        "- Caption-BPM leak (that is a TF pair fact; see tf-leak.md).",
        "",
        "## How to run",
        "",
        "```bash",
        "PYTHONPATH=. python analysis/slider2d/run_lm_v9.py --out docs/lm-v9-2d",
        "PYTHONPATH=. pytest tests/test_lm_v9_2d.py tests/test_2d_slider_geometry.py -q",
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
    results = run_v9(steps=args.steps, seed=args.seed)
    field = Field2D()
    geo = _pair_geometry(field)
    blob = {
        "steps": args.steps,
        "seed": args.seed,
        "geometry": geo,
        "methods": {
            r.name: {
                "family": r.family,
                "verdict": r.verdict,
                "reason": r.reason,
                "axis": axis_verdicts(r.metrics),
                **{k: v for k, v in r.metrics.items() if k != "action"},
            }
            for r in results
        },
    }
    (out / "metrics.json").write_text(json.dumps(blob, indent=2) + "\n", encoding="utf-8")
    plot_compare(results, out / "compare.png")
    plot_quiver(results, field, out / "quiver.png")
    plot_axis_scatter(results, out / "scatter.png")
    write_report(results, geo, blob, out.parent / "lm-v9-2d.md")
    for r in results:
        ax = axis_verdicts(r.metrics)
        print(
            f"{r.name:22s} slider={ax['slider']:11s} leak={ax['leak']:11s} "
            f"collapse={ax['collapse']:11s}  "
            f"cos={r.metrics['cos_slider_plus']:+.3f} leak={r.metrics['leak_ratio']:+.3f} "
            f"±1={r.metrics['cos_plus_minus']:+.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
