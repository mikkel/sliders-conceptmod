#!/usr/bin/env python3
"""Score hold-ê when ê overlaps the slider / pair-odd (live energy)."""

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

from analysis.slider2d.energy import LIVE_ENERGY_ALIGNS
from analysis.slider2d.overlap import (
    HOLD_LAMBDAS,
    OVERLAPS,
    baseline_table,
    compact,
    leak_geometry,
    mean_odd_unit,
    overlap_sweep,
)
from conceptmod.textsliders.slider_targets import LEAK_HOLD_WEIGHT


DEFAULT_OUT = _REPO / "docs" / "lm-hold-overlap"


def _fmt_bool(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def _row_md(row: dict) -> str:
    return (
        f"| `{row['name']}` | {row.get('overlap', 0):.1f} | {row.get('e_dot_odd', 0):+.3f} | "
        f"{int(row.get('hold_weight', 0))} | {row.get('ortho', 'raw')} | "
        f"{row.get('cos_intended', 0):+.3f} | {row.get('leak_ratio', 0):+.3f} | "
        f"{row.get('cos_teacher', 0):+.3f} | {row.get('perc', 0)*100:.0f} | "
        f"{row.get('loss', 0):.3f} | {row.get('cos_plus_minus', 0):+.3f} | "
        f"**{_fmt_bool(row['pass'])}** |"
    )


def plot_sweep(rows: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
    styles = {
        "raw": ("#c0392b", "o", "raw ê"),
        "slider": ("#1e8449", "s", "ê ⊥ û"),
        "odd": ("#6c3483", "D", "ê ⊥ â"),
    }
    for ortho, (color, mark, label) in styles.items():
        xs, cos_y, leak_y = [], [], []
        for rho in OVERLAPS:
            match = [
                r
                for r in rows
                if r["leak_mode"] == "opposite"
                and r["ortho"] == ortho
                and abs(r["overlap"] - rho) < 1e-9
                and abs(r["hold_weight"] - 8.0) < 1e-9
            ]
            if not match:
                continue
            xs.append(rho)
            cos_y.append(match[0]["cos_intended"])
            leak_y.append(abs(match[0]["leak_ratio"]))
        axes[0].plot(xs, cos_y, mark + "-", color=color, label=label)
        axes[1].plot(xs, leak_y, mark + "-", color=color, label=label)
    axes[0].axhline(0.90, color="#888888", ls="--", lw=0.8, label="lock 0.90")
    axes[1].axhline(0.20, color="#888888", ls="--", lw=0.8, label="leak 0.20")
    axes[0].set_xlabel("ê · û")
    axes[0].set_ylabel("slider cos")
    axes[0].set_title("λ=8 opposite-energy leak  ·  slider lock")
    axes[1].set_xlabel("ê · û")
    axes[1].set_ylabel("|leak|")
    axes[1].set_title("λ=8 opposite-energy leak  ·  unused leftover")
    for ax in axes:
        ax.set_xlim(-0.05, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_residuals(rows: list[dict], path: Path) -> None:
    want = {
        "opposite_o0.0_l8_raw": ("#1e8449", "ρ=0 raw (unused ê)"),
        "opposite_o0.5_l8_raw": ("#c0392b", "ρ=0.5 raw (ê ≈ â)"),
        "opposite_o1.0_l8_raw": ("#922b21", "ρ=1 raw (ê = û)"),
        "opposite_o0.5_l8_slider": ("#148f77", "ρ=0.5 ê ⊥ û"),
        "opposite_o1.0_l8_slider": ("#7f8c8d", "ρ=1 ê ⊥ û (hold off)"),
    }
    by_name = {r["name"]: r for r in rows}
    fig, ax = plt.subplots(figsize=(6.6, 6.4))
    ax.axhline(0, color="#dddddd", lw=0.6)
    ax.axvline(0, color="#dddddd", lw=0.6)
    for name, (color, label) in want.items():
        row = by_name[name]
        dp = row["delta_plus"]
        ax.arrow(
            0, 0, dp[0], dp[1],
            color=color, width=0.02, length_includes_head=True, head_width=0.08, alpha=0.95,
        )
        ax.annotate(label, (dp[0] + 0.03, dp[1] + 0.03), fontsize=8, color=color)
    ax.annotate("intended û", (1.05, 0.08), fontsize=8, color="#444444")
    ax.annotate("unused leftover", (0.08, 1.05), fontsize=8, color="#444444")
    ax.set_xlabel("intended energy  (loud ← → quiet)")
    ax.set_ylabel("unused mix / genre / BPM")
    ax.set_title("+1 residual  ·  opposite-energy leak  ·  λ=8")
    ax.set_aspect("equal")
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.2, 1.6)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_report(blob: dict, path: Path) -> None:
    sweep = blob["sweep"]
    base = {r["name"]: r for r in blob["baselines"]}
    by_name = {r["name"]: r for r in sweep}
    live_raw = by_name["opposite_o1.0_l8_raw"]
    pole_raw = by_name["opposite_o0.5_l8_raw"]
    unused_raw = by_name["opposite_o0.0_l8_raw"]
    pole_slider = by_name["opposite_o0.5_l8_slider"]
    pole_odd = by_name["opposite_o0.5_l8_odd"]
    slider_slider = by_name["opposite_o1.0_l8_slider"]
    same = next(r for r in blob["same_energy"] if abs(r["hold_weight"] - 8.0) < 1e-9 and r["ortho"] == "raw")
    odd_u = float(blob["mean_odd_dot_u"])
    lines = [
        "# Hold-ê when ê overlaps the slider (live energy)",
        "",
        "Existing hold-ê cells PASS because they set ê = unused gender,",
        "orthogonal to energetic. Live energy-v4 already declares",
        "`slider_positive` = loud energy and `leak_positive` =",
        "\"Dense slammed mix, BPM 168, pop-punk.\" — that ê *is* energy.",
        "This cell makes `ê·û` a knob on the same energy-like poles",
        f"(aligns `{list(LIVE_ENERGY_ALIGNS)}`).",
        "",
        "CPU only. No Hub, no GPU, no Music 3 weights.",
        "",
        "## Verdict",
        "",
    ]
    # Verdict paragraph is filled from real numbers after the sweep exists.
    lines += [
        (
            f"Raw hold at λ=8 **reproduces the live miss** when ê is a "
            f"pole synonym or the slider: at `ê·û=1` slider cos "
            f"{live_raw['cos_intended']:+.3f}, leak "
            f"{live_raw['leak_ratio']:+.3f}, c+ "
            f"{live_raw['cos_teacher']:+.3f}, perc "
            f"{live_raw['perc']*100:.0f}%, loss {live_raw['loss']:.3f}. "
            f"At `ê·û=0.5` (ê·â={pole_raw['e_dot_odd']:+.3f} ≈ 1) "
            f"c+ {pole_raw['cos_teacher']:+.3f}, perc "
            f"{pole_raw['perc']*100:.0f}%, loss {pole_raw['loss']:.3f} — "
            f"the 2-D shrink-in-place of `student = teacher/(1+λ)`, not a "
            f"locked ~0.01 loss. Unused-only ê still locks "
            f"(slider {unused_raw['cos_intended']:+.3f}, leak "
            f"{unused_raw['leak_ratio']:+.3f})."
        ),
        "",
        (
            f"**Allowed ê is leftover unused, not a synonym of the poles "
            f"or of `slider_positive`.** Energy-v4 leak captions are the "
            f"wrong ê (opposite-energy restates the structured poles; "
            f"mean |odd·û|/||odd|| = {odd_u:.2f}, so `ê·û≈0.5` is ê≈â). "
            f"Hold in the trainer should use `ê_⊥ = ê − (ê·û)û`, not "
            f"`ê − (ê·â)â`. At the live-like ρ=0.5 cell, ê⊥û locks "
            f"(slider {pole_slider['cos_intended']:+.3f}, leak "
            f"{pole_slider['leak_ratio']:+.3f}); ê⊥â turns hold off "
            f"(slider {pole_odd['cos_intended']:+.3f}, leak "
            f"{pole_odd['leak_ratio']:+.3f}). Same-energy different-mix "
            f"captions are already unused-ê "
            f"(slider {same['cos_intended']:+.3f}, leak "
            f"{same['leak_ratio']:+.3f}). Do not revert to Hub or "
            f"short-û project."
        ),
        "",
        "## Geometry",
        "",
        "Opposite-energy leak (current yaml):",
        "",
        "```",
        "ê(ρ) = ρ û + √(1−ρ²) unused",
        "```",
        "",
        f"Mean pair-odd on this field has |â·û| = {odd_u:.3f}, so ρ=0.5",
        "makes ê a synonym of the poles. Same-energy different-mix leak",
        "is ê = unused (both captions loud; difference is mix).",
        "",
        "## Baselines (same energy poles)",
        "",
        "| policy | ê·û | ê·â | λ | ortho | slider cos | leak | c+ | perc% | loss | ±1 | verdict |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name in (
        "pair_odd_no_hold",
        "hub",
        "gated_project_0.50",
        "v9_unused_e",
        "same_energy_hold_l8",
        "energy_slider_e_raw_l8",
        "energy_slider_e_slider_l8",
        "pole_synonym_raw_l8",
        "pole_synonym_slider_l8",
        "pole_synonym_odd_l8",
    ):
        lines.append(_row_md(base[name]))
    lines += [
        "",
        "## Opposite-energy overlap × λ × ortho",
        "",
        "| cell | ê·û | ê·â | λ | ortho | slider cos | leak | c+ | perc% | loss | ±1 | verdict |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in sweep:
        if row["leak_mode"] != "opposite":
            continue
        lines.append(_row_md(row))
    lines += [
        "",
        "![slider-cos and leak vs ê·û at λ=8](lm-hold-overlap/sweep.png)",
        "",
        "![+1 residuals at ρ=0 / 0.5 / 1.0](lm-hold-overlap/residuals.png)",
        "",
        "## Same-energy different-mix leak",
        "",
        "| cell | ê·û | ê·â | λ | ortho | slider cos | leak | c+ | perc% | loss | ±1 | verdict |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in blob["same_energy"]:
        lines.append(_row_md(row))
    live_hist = " ".join(
        f"step {h['step']} loss {h['loss']:.2f} c+ {h['cos_teacher']:+.2f} "
        f"col {h['collapse']:+.2f} p% {h['perc']*100:.0f}"
        for h in live_raw["history"]
    )
    pole_hist = " ".join(
        f"step {h['step']} loss {h['loss']:.2f} c+ {h['cos_teacher']:+.2f} "
        f"col {h['collapse']:+.2f} p% {h['perc']*100:.0f}"
        for h in pole_raw["history"]
    )
    lines += [
        "",
        "## Live-log analogue",
        "",
        "Mikkel ~step 515/800: loss 0.16–0.30 (not ~0.01), c+ ~0 / −0.10,",
        "col −0.21, perc 132%. That is high-D hold punching a teacher-aligned",
        "ê until the residual is numerically ~0 (cosine undefined / noisy).",
        "2-D MSE / hold share a ½ factor, so `student_ê = teacher_ê / (1+λ)`",
        "stays parallel to the teacher when ê ∥ â — c+ stays high, perc",
        "→ λ/(1+λ). The slider-lock failure still shows up as leftover unused",
        "when ê ≈ û.",
        "",
        f"ê = û, λ=8 raw: {live_hist}",
        "",
        f"ê ≈ â, λ=8 raw: {pole_hist}",
        "",
        "## What each recipe does",
        "",
        "- `pair_odd_no_hold` / Hub: copies leftover mix. Slider component",
        "  of the pair is present, unused leak is not held.",
        "- `gated_project_0.50` (v12): teacher = `(a·û)û`. Locks this cell",
        "  and is leak-0, but it is the live-fragile path that kills gender",
        "  at |odd·û|/||odd|| = 0.20. Not the default.",
        "- `v9_unused_e`: current 2-D hold-ê. PASS. The wrong ê for live energy.",
        "- raw hold at high overlap: punches the slider / the teacher.",
        "- `ê_⊥û = ê − (ê·û)û`: hold cannot punch the slider name. At ρ<1 the",
        "  leftover unused is held and the slider locks. At ρ=1 hold is off",
        f"  (slider {slider_slider['cos_intended']:+.3f}, leak "
        f"{slider_slider['leak_ratio']:+.3f}) — ê had no unused left.",
        "- `ê_⊥â = ê − (ê·â)â`: at the live-like pole synonym, leftover is 0",
        "  and hold turns off. Leak stays. Per-row â is also not one vector",
        "  on Music 3; û is encoded once.",
        "- Same-energy different-mix leak captions: ê is already unused.",
        "  Current hold-ê locks without a trainer change. Opposite-energy",
        "  energy-v4 captions are the wrong ê.",
        "",
        "## How to run",
        "",
        "```bash",
        "PYTHONPATH=. python analysis/slider2d/run_lm_overlap.py --out docs/lm-hold-overlap",
        "PYTHONPATH=. pytest tests/test_lm_hold_overlap.py tests/test_lm_live_cells.py tests/test_lm_signature.py -q",
        "```",
        "",
        f"Seed `{blob['seed']}`, `{blob['steps']}` Adam steps.",
        "",
        "c+ vs slider-cos as a test, high-D leftover that zeros the residual,",
        "and leftover-only ê vs `pair_odd_sub_e` are",
        "[lm-live-signature.md](lm-live-signature.md).",
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
    sweep = overlap_sweep(steps=args.steps, seed=args.seed)
    same = overlap_sweep(
        overlaps=(0.0,),
        leak_modes=("same_energy",),
        steps=args.steps,
        seed=args.seed,
    )
    baselines = baseline_table(steps=args.steps, seed=args.seed)
    geo = [leak_geometry(rho) for rho in OVERLAPS]
    blob = {
        "steps": args.steps,
        "seed": args.seed,
        "overlaps": list(OVERLAPS),
        "lambdas": list(HOLD_LAMBDAS),
        "live_energy_aligns": list(LIVE_ENERGY_ALIGNS),
        "mean_odd_dot_u": float(mean_odd_unit()[0]),
        "leak_hold_weight": LEAK_HOLD_WEIGHT,
        "geometry": geo,
        "sweep": [compact(r) | {"delta_plus": r["delta_plus"], "delta_minus": r["delta_minus"]} for r in sweep],
        "same_energy": [compact(r) for r in same],
        "baselines": [compact(r) | {"delta_plus": r["delta_plus"], "delta_minus": r["delta_minus"]} for r in baselines],
    }
    # compact() dropped delta; restore from the live objects for plots.
    for src, dst in ((sweep, blob["sweep"]), (baselines, blob["baselines"])):
        by = {r["name"]: r for r in src}
        for row in dst:
            live = by[row["name"]]
            row["delta_plus"] = [float(live["delta_plus"][0]), float(live["delta_plus"][1])]
            row["delta_minus"] = [float(live["delta_minus"][0]), float(live["delta_minus"][1])]
    (out / "metrics.json").write_text(json.dumps(blob, indent=2) + "\n", encoding="utf-8")
    plot_sweep(blob["sweep"], out / "sweep.png")
    plot_residuals(blob["sweep"], out / "residuals.png")
    write_report(blob, out.parent / "lm-hold-overlap.md")
    print(f"mean â·û = {blob['mean_odd_dot_u']:.3f}")
    for row in baselines:
        print(
            f"  {row['name']:28s} pass={str(row['pass']):5s} "
            f"cos={row['cos_intended']:+.3f} leak={row['leak_ratio']:+.3f} "
            f"c+={row['cos_teacher']:+.3f} perc={row['perc']*100:.0f} "
            f"loss={row['loss']:.3f}"
        )
    print("== opposite λ=8 ==")
    for row in sweep:
        if abs(row["hold_weight"] - 8.0) > 1e-9:
            continue
        print(
            f"  ρ={row['overlap']:.1f} {row['ortho']:6s} "
            f"pass={str(row['pass']):5s} cos={row['cos_intended']:+.3f} "
            f"leak={row['leak_ratio']:+.3f} c+={row['cos_teacher']:+.3f} "
            f"perc={row['perc']*100:.0f} loss={row['loss']:.3f} "
            f"ê·â={row['e_dot_odd']:+.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
