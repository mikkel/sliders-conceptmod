#!/usr/bin/env python3
"""Score the live energy-v14 signature cells and write docs/lm-live-signature.md."""

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

from analysis.slider2d.overlap import score_overlap_policy
from analysis.slider2d.signature import (
    LEAK_HOLD_WEIGHT,
    SIG_DIM,
    SIG_STEPS,
    SignatureField,
    compact,
    linear_shrink_factor,
    measured_shrink_factor,
    signature_table,
)


DEFAULT_OUT = _REPO / "docs" / "lm-live-signature"

TABLE_ORDER = (
    "gender_like_linear",
    "gender_like_curved",
    "energy_no_hold_linear",
    "energy_no_hold_curved",
    "synonym_perp_l8_linear",
    "synonym_perp_l8_curved",
    "pinned_perp_l8_curved",
    "leftover_only_l0.3_curved",
    "leftover_only_l1_curved",
    "leftover_only_l8_curved",
    "sub_e_leftover_linear",
    "sub_e_leftover_curved",
    "sub_e_synonym_curved",
)


def _row_md(row: dict) -> str:
    lam = row["used_hold"] if not row["subtract_e"] else "sub"
    lam = f"{lam:g}" if isinstance(lam, float) else lam
    return (
        f"| `{row['name']}` | {row['student']} | {row['leak_kind']} | {lam} | "
        f"{row['cos_teacher']:+.3f} | {row['slider_cos']:+.3f} | {row['intended_cos']:+.3f} | "
        f"{row['collapse']:+.3f} | {row['collapse_late_max']:+.3f} | {row['perc']*100:.0f} | "
        f"{row['loss']:.4f} | {row['loss_max']:.2f} | {row['leak_ratio']:+.2f} | "
        f"{'yes' if row['looks_like_v12'] else 'no'} | "
        f"{'**BROKEN**' if row['polarity_broken'] else 'no'} |"
    )


def plot_history(rows: dict, path: Path) -> None:
    picks = {
        "energy_no_hold_curved": ("#1e8449", "no hold (v12 look)"),
        "synonym_perp_l8_curved": ("#c0392b", "synonym ê_⊥û, λ=8"),
        "leftover_only_l1_curved": ("#2471a3", "leftover-only ê, λ=1"),
        "sub_e_leftover_curved": ("#7d6608", "pair_odd_sub_e"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for name, (color, label) in picks.items():
        hist = rows[name]["history"]
        steps = [h["step"] for h in hist]
        axes[0].plot(steps, [h["collapse"] for h in hist], color=color, label=label, lw=1.4)
        axes[1].plot(steps, [h["cos_teacher"] for h in hist], color=color, label=label, lw=1.4)
        axes[2].semilogy(steps, [max(h["loss_train"], 1e-6) for h in hist], color=color, label=label, lw=1.4)
    axes[0].axhline(0.18, color="#888888", ls="--", lw=0.9, label="live v14 col +0.18")
    axes[0].set_ylabel("collapse cos(Δ+1, Δ−1)")
    axes[0].set_title("±1 polarity (curved student)")
    axes[1].axhline(0.31, color="#888888", ls="--", lw=0.9, label="live v14 c+ 0.31")
    axes[1].set_ylabel("trainer c+ (cos with pair-odd a)")
    axes[1].set_title("fit to teacher")
    axes[2].set_ylabel("loss (log)")
    axes[2].set_title("loss — the λ=8 fight explodes early")
    for ax in axes:
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_shrink(shrink: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    dims = sorted({s["dim"] for s in shrink})
    for lam, color in ((0.3, "#2471a3"), (1.0, "#1e8449"), (8.0, "#c0392b")):
        xs = [d for d in dims]
        ys = [linear_shrink_factor(d, lam) for d in dims]
        ax.loglog(xs, ys, "-", color=color, lw=1.2, label=f"λ={lam:g} analytic")
        pts = [s for s in shrink if abs(s["hold_weight"] - lam) < 1e-9]
        ax.loglog([s["dim"] for s in pts], [s["measured"] for s in pts], "o", color=color, ms=5)
    ax.axvline(2, color="#888888", ls=":", lw=0.9)
    ax.annotate("2-D fixture", (2.1, 0.3), fontsize=8, color="#555555")
    ax.axvline(SIG_DIM, color="#888888", ls=":", lw=0.9)
    ax.annotate("live-D", (SIG_DIM * 0.55, 0.3), fontsize=8, color="#555555")
    ax.set_xlabel("hidden dim D")
    ax.set_ylabel("kept fraction of the held component  s_e / t_e")
    ax.set_title("hold vs MSE: s_e = t_e / (1 + λ·D/2)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_report(blob: dict, path: Path) -> None:
    rows = {r["name"]: r for r in blob["table"]}
    ref2d = blob["ref_2d"]
    gender = rows["gender_like_linear"]
    energy = rows["energy_no_hold_linear"]
    syn_lin = rows["synonym_perp_l8_linear"]
    syn_cur = rows["synonym_perp_l8_curved"]
    pin = rows["pinned_perp_l8_curved"]
    l03 = rows["leftover_only_l0.3_curved"]
    l1 = rows["leftover_only_l1_curved"]
    l8 = rows["leftover_only_l8_curved"]
    sub = rows["sub_e_leftover_curved"]
    sub_syn = rows["sub_e_synonym_curved"]
    s2 = linear_shrink_factor(2, 8.0)
    sd = linear_shrink_factor(SIG_DIM, 8.0)
    lines = [
        "# Live energy-v14 signature: c+ vs slider-cos, live-D ê_⊥, ±1 polarity",
        "",
        "The overlap cells ([lm-hold-overlap.md](lm-hold-overlap.md)) are",
        "orthonormal 2-D: they show ê·û overlap and that holding ê_⊥û locks",
        "the slider at ρ=0.5. Live energy-v14 logged three things those cells",
        "cannot say: trainer c+ vs slider-cos as different columns, a hold",
        "that is λ·D/2 stiff against a tiny messy ê_⊥, and a ±1 polarity",
        "break (collapse **+0.18**, loss 278 by step 13 on the rewrite).",
        f"This cell set is D={SIG_DIM}, live steps ({SIG_STEPS}), same teacher",
        "and hold code paths (`lm_hidden_targets` / `lm_hold_dir` /",
        "`lm_axis_hold` / `lm_slider_loss`).",
        "",
        "CPU only. No Hub, no GPU, no Music 3 weights. The live default",
        "(`--lm_target v9`, λ=8 hold on declared ê_⊥û) is unchanged.",
        "",
        "## Verdict",
        "",
        (
            f"**c+ is fit-to-pair-odd, not slider lock.** Gender-like prints "
            f"c+ {gender['cos_teacher']:+.2f} / collapse {gender['collapse']:+.2f} / "
            f"perc {gender['perc']*100:.0f}% and leaky-energy-no-hold prints "
            f"c+ {energy['cos_teacher']:+.2f} / collapse {energy['collapse']:+.2f} / "
            f"perc {energy['perc']*100:.0f}% — the same v12 look, but energy "
            f"carries leak {energy['leak_ratio']:+.2f}. The 2-D working hold "
            f"(ρ=0.5, ê_⊥û, λ=8) locks the slider at {ref2d['cos_intended']:+.2f} "
            f"with leftover {ref2d['leak_ratio']:+.2f} while printing "
            f"c+ {ref2d['cos_teacher']:+.2f}, perc {ref2d['perc']*100:.0f}%, "
            f"loss {ref2d['loss']:.2f}. A working hold on energy will never "
            f"look like v12; expecting gender's 0.97 there reads success as failure."
        ),
        "",
        (
            f"**λ is not portable across D.** The held component of a linear "
            f"fit is `s_e = t_e/(1+λ·D/2)`: λ=8 keeps {s2:.3f} in 2-D and "
            f"{sd:.6f} at D={SIG_DIM} (~4000× annihilation); even λ=0.3 keeps "
            f"{linear_shrink_factor(SIG_DIM, 0.3):.4f}. And ê_⊥û is one "
            f"direction out of a (D−2)-dim leftover, so the same recipe that "
            f"locked 2-D leftover at {ref2d['leak_ratio']:+.2f} leaves "
            f"{syn_lin['leak_ratio']:+.2f} at live D with loss stuck "
            f"{syn_lin['loss']/max(energy['loss'],1e-9):.0f}× off the band — "
            f"violent *and* barely curative."
        ),
        "",
        (
            f"**The ±1 break needs curvature; then λ=8 reproduces it.** A "
            f"linear residual provably cannot break polarity (the loss "
            f"separates over odd/even weights; under the identical fight "
            f"collapse is {syn_lin['collapse']:+.3f} with ‖w_even‖ = "
            f"{syn_lin['w_even_norm']:.1e}). The curved student (adapter "
            f"through a saturating layer) under synonym-ê_⊥û λ=8 lands on "
            f"the live shape: c+ {syn_cur['cos_teacher']:+.2f} (live 0.31), "
            f"collapse late-max {syn_cur['collapse_late_max']:+.2f} (live "
            f"+0.18), perc {syn_cur['perc']*100:.0f}% (live 132%), loss "
            f"spiking {syn_cur['loss_spike']:.0f}× in the first steps (live: "
            f"278 by step 13). The medium-energy pin does not save it "
            f"(c+ {pin['cos_teacher']:+.2f}, still broken): ê_⊥ is still a "
            f"pole synonym."
        ),
        "",
        (
            f"**Canary vs teacher change.** Leftover-only ê + λ=1 trains "
            f"(collapse {l1['collapse']:+.2f}, loss {l1['loss']:.4f}) but "
            f"leak stays {l1['leak_ratio']:+.2f} — per-row wording that one "
            f"declared ê cannot name. λ=8 breaks ±1 **even with a genuinely unused ê** "
            f"(collapse {l8['collapse']:+.2f}, spike {l8['loss_spike']:.0f}×): "
            f"at live D the fight itself is the failure, not just the wrong ê. "
            f"`pair_odd_sub_e` (subtract ê_⊥û from the teacher) removes the "
            f"same component with no fight at all: collapse "
            f"{sub['collapse']:+.2f}, loss {sub['loss']:.4f}, max "
            f"{sub['loss_max']:.3f}, leak {sub['leak_ratio']:+.2f}. Use "
            f"leftover-only ê + λ=1 as the live canary; if leak stays big, "
            f"`pair_odd_sub_e` is the next PR — with ê_⊥û orthogonalization "
            f"kept, because subtracting a synonym ê punches the heard slider "
            f"(intended cos {sub_syn['intended_cos']:+.2f}). Do not wire "
            f"either as the default from this fixture alone."
        ),
        "",
        "## Geometry",
        "",
        f"D = {SIG_DIM}. The declared probe û spans only part of the heard",
        "loudness (`heard = cos40°·û + sin40°·dense-wording`); rows are the",
        "live aligns 0.48/0.48/0.68/0.68 against `heard`, with the leftover",
        "split into a shared genre/BPM/mix direction and per-row wording.",
        "Declared ê variants:",
        "",
        "- `opposite` — energy-v4 leak captions: 0.85·heard + wording",
        f"  (ê·û {rows['synonym_perp_l8_curved']['e_dot_u']:+.2f}, ê_⊥·â "
        f"{rows['synonym_perp_l8_curved']['e_perp_dot_ahat']:+.2f}: a synonym in disguise)",
        "- `pinned` — the \"medium energy\" rewrite: 0.62·heard + wording",
        f"  (ê·û {pin['e_dot_u']:+.2f}, ê_⊥·â {pin['e_perp_dot_ahat']:+.2f}: still a synonym)",
        "- `leftover_only` — genre+BPM wording only, no density/loudness",
        f"  words (ê·û {l1['e_dot_u']:+.2f}, ê_⊥·â {l1['e_perp_dot_ahat']:+.2f}:",
        "  it *is* the heard leak inside a, which is the point)",
        "",
        "Students: `linear` = the odd+even residual every other cell uses;",
        "`curved` = the same adapter applied through a fixed tanh layer",
        "around a non-zero operating point (a LoRA at −1 is −ΔW in weight",
        "space, not −Δh in hidden space).",
        "",
        "![shrink factor](lm-live-signature/shrink.png)",
        "",
        "## Signature table",
        "",
        "slider-cos is vs the declared probe û; intended-cos is vs the heard",
        "loudness (gender: the concept); leak is leftover-norm / |intended|;",
        "colL is the max collapse over the last half of training.",
        "",
        "| cell | student | ê | λ | c+ | slider | intended | col | colL | perc% | loss | loss_max | leak | v12 look | ±1 broken |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for name in TABLE_ORDER:
        lines.append(_row_md(rows[name]))
    lines += [
        "",
        "2-D reference (existing overlap cell, steps 200): "
        f"`pole_synonym_slider_l8` slider {ref2d['cos_intended']:+.3f}, "
        f"leftover {ref2d['leak_ratio']:+.3f}, c+ {ref2d['cos_teacher']:+.3f}, "
        f"perc {ref2d['perc']*100:.0f}%, loss {ref2d['loss']:.3f} — PASS "
        "leftover, FAIL \"looks like v12\".",
        "",
        "![histories](lm-live-signature/history.png)",
        "",
        "## What each cell says",
        "",
        "- `gender_like_*`: hold 0 (no ê declared — the field refuses an",
        "  invented one). The \"copied pair-odd\" look is *correct* here.",
        "- `energy_no_hold_*`: the same look with the heard leak on board —",
        "  v12 / Hub. c+ cannot tell these two apart; leak can.",
        "- `synonym_perp_l8_linear`: geometry alone at live D — leak barely",
        "  cured, c+ down, loss stuck, but collapse exactly −1 (separability).",
        "- `synonym_perp_l8_curved`: the live v14 signature, including the",
        "  early explosion and the ±1 break.",
        "- `pinned_perp_l8_curved`: the same-loudness caption pin does not",
        "  cancel dense/sparse↔loud/quiet; λ=8 still fights and still breaks.",
        "- `leftover_only_l*`: the proposed canary. λ≤1 trains and stays",
        "  bipolar; λ=8 breaks even with the right ê — the λ, not only the ê,",
        "  is wrong at live D.",
        "- `sub_e_*`: `pair_odd_sub_e` = subtract declared ê_⊥û from `a`.",
        "  Same leak cut, trainable loss, exact bipolarity, no spike. With a",
        "  synonym ê it punches the heard slider — the ⊥û step and the",
        "  leftover-only caption rule stay mandatory.",
        "",
        "## What still cannot be seen",
        "",
        "- The exact live magnitudes (loss 278, c+ 0.31 at step 515 of a",
        "  real Qwen run) — the fixture matches shape, not scale.",
        "- Whether Qwen's encodings make the *live* leftover-only captions",
        "  actually ⊥ heard loudness — that is a probe on the real encoder",
        "  (`probe_lm_axis_signal.py`-style), not a fixture question.",
        "- Render/listen quality. Fixture PASS is necessary, never sufficient.",
        "",
        "## How to run",
        "",
        "```bash",
        "PYTHONPATH=. python analysis/slider2d/run_lm_signature.py --out docs/lm-live-signature",
        "PYTHONPATH=. pytest tests/test_lm_signature.py -q",
        "```",
        "",
        f"Seed `{blob['seed']}`, `{blob['steps']}` Adam steps, D={SIG_DIM}.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--steps", type=int, default=SIG_STEPS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    table = signature_table(steps=args.steps, seed=args.seed)
    ref2d = score_overlap_policy(
        "pole_synonym_slider_l8",
        overlap=0.5,
        hold_weight=LEAK_HOLD_WEIGHT,
        ortho="slider",
        steps=200,
        seed=args.seed,
    )
    shrink = [
        {"dim": d, "hold_weight": lam, "analytic": linear_shrink_factor(d, lam),
         "measured": measured_shrink_factor(d, lam)}
        for d in (2, 16, 128, SIG_DIM)
        for lam in (0.3, 1.0, 8.0)
    ]
    field = SignatureField()
    blob = {
        "steps": args.steps,
        "seed": args.seed,
        "dim": SIG_DIM,
        "aligns": list(field.aligns),
        "heard_split_deg": field.heard_split_deg,
        "leak_hold_weight": LEAK_HOLD_WEIGHT,
        "shrink": shrink,
        "table": [compact(r) for r in table],
        "ref_2d": {
            k: (float(v) if isinstance(v, (int, float)) else v)
            for k, v in ref2d.items()
            if isinstance(v, (int, float, str, bool))
        },
    }
    (out / "metrics.json").write_text(json.dumps(blob, indent=2) + "\n", encoding="utf-8")
    rows = {r["name"]: r for r in blob["table"]}
    plot_history(rows, out / "history.png")
    plot_shrink(shrink, out / "shrink.png")
    write_report(blob, out.parent / "lm-live-signature.md")
    for name in TABLE_ORDER:
        r = rows[name]
        print(
            f"  {name:28s} c+={r['cos_teacher']:+.3f} col={r['collapse']:+.3f} "
            f"colL={r['collapse_late_max']:+.3f} perc={r['perc']*100:4.0f}% "
            f"loss={r['loss']:8.4f} lmax={r['loss_max']:7.2f} leak={r['leak_ratio']:+.2f} "
            f"broken={r['polarity_broken']}"
        )
    print(
        f"2d ref: slider={blob['ref_2d']['cos_intended']:+.3f} "
        f"leak={blob['ref_2d']['leak_ratio']:+.3f} c+={blob['ref_2d']['cos_teacher']:+.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
