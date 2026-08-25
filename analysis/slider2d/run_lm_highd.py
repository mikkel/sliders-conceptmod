#!/usr/bin/env python3
"""Score hold-ê in high-D: λ·D/2 stiffness, ê wording, and ±1 polarity."""

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

from analysis.slider2d.highd import (
    BEND_ENERGY,
    BEND_GENDER,
    COVER_GRID,
    DIM_GRID,
    HOLD_LAMBDAS,
    LIVE_COLLAPSE,
    LIVE_C_PLUS,
    LIVE_GATE_ALIGN,
    LIVE_GENDER_COLLAPSE,
    LIVE_GENDER_C_PLUS,
    LIVE_GENDER_LOSS,
    LIVE_SPIKE_LOSS,
    bend_for_collapse,
    bend_sweep,
    calibrate_bend,
    cell_table,
    compact,
    content_sweep,
    energy_field,
    hold_spike,
    lambda_dim_sweep,
    lambda_fit_sweep,
    leftover_only_e,
    live_v14_analogue,
    match_sweep,
    polarity_grid,
    synonym_cover,
    synonym_e,
)
from analysis.slider2d.overlap import score_overlap_policy
from conceptmod.textsliders.slider_targets import LEAK_HOLD_WEIGHT


DEFAULT_OUT = _REPO / "docs" / "lm-highd-leftover"


def _fmt_bool(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def _cell_md(row: dict) -> str:
    return (
        f"| `{row['name']}` | {row['e_label']} | {row['dim']} | "
        f"{row['hold_weight']:g} | {row['lambda_eff']:.0f} | {row['bend']:g} | "
        f"{row['hold_cover']:.2f} | {row['c_plus']:+.3f} | "
        f"{row['c_plus_predicted']:+.3f} | {row['c_plus_if_held']:.2f} | "
        f"{row['cos_short_u']:+.3f} | {row['cos_intended']:+.3f} | "
        f"{row['leftover_leak']:+.3f} | {row['leftover_kept']:.2f} | "
        f"{row['collapse']:+.3f} | {row['perc']*100:.0f} | {row['loss']:.3f} | "
        f"**{_fmt_bool(row['pass'])}** |"
    )


CELL_HEADER = (
    "| cell | ê | D | λ | λ·D/2 | bend | p | c+ | c+ pred | c+ ceil | "
    "cos û | cos concept | leak | leak kept | ±1 | perc% | loss | verdict |\n"
    "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"
)


def plot_lambda_dim(closed: list[dict], fitted: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4))
    dims = sorted({r["dim"] for r in closed})
    colors = plt.cm.viridis([i / max(len(dims) - 1, 1) for i in range(len(dims))])
    ceiling = closed[0]["c_plus_if_held"]
    for color, dim in zip(colors, dims):
        rows = [r for r in closed if r["dim"] == dim and r["hold_weight"] > 0]
        rows.sort(key=lambda r: r["hold_weight"])
        axes[0].plot(
            [r["hold_weight"] for r in rows],
            [r["c_plus_predicted"] for r in rows],
            "o-",
            color=color,
            label=f"D={dim}",
        )
    axes[0].axhline(ceiling, color="#c0392b", ls="--", lw=0.9, label="c+ ceiling √(1−p²)")
    axes[0].set_xscale("log")
    ticks = sorted({r["hold_weight"] for r in closed if r["hold_weight"] > 0})
    axes[0].set_xticks(ticks)
    axes[0].set_xticklabels([f"{t:g}" for t in ticks])
    axes[0].minorticks_off()
    axes[0].set_xlabel("λ (--hold_weight)")
    axes[0].set_ylabel("trainer c+")
    axes[0].set_title("same λ, different hidden width")
    for color, dim in zip(colors, dims):
        rows = [r for r in closed if r["dim"] == dim and r["hold_weight"] > 0]
        rows.sort(key=lambda r: r["lambda_eff"])
        axes[1].plot(
            [r["lambda_eff"] for r in rows],
            [r["c_plus_predicted"] for r in rows],
            "-",
            color=color,
            label=f"D={dim}",
        )
    fit_x = [r["lambda_eff"] for r in fitted if r["hold_weight"] > 0]
    fit_y = [r["c_plus"] for r in fitted if r["hold_weight"] > 0]
    axes[1].plot(fit_x, fit_y, "kx", ms=7, label="fitted")
    axes[1].axhline(ceiling, color="#c0392b", ls="--", lw=0.9)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("λ · D / 2  (the stiffness that acts)")
    axes[1].set_title("one curve once λ·D/2 is the x-axis")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
        ax.set_ylim(0.45, 1.05)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_wording(rows: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    xs = [r["hold_on_content"] for r in rows]
    ax.plot(xs, [r["cos_intended"] for r in rows], "o-", color="#1e8449", label="cos(d+, concept)")
    ax.plot(xs, [abs(r["leftover_leak"]) for r in rows], "s-", color="#c0392b", label="|leftover leak|")
    ax.plot(xs, [r["cos_short_u"] for r in rows], "D--", color="#7f8c8d", label="cos(d+, short û) — the probe")
    ax.axhline(0.90, color="#1e8449", ls=":", lw=0.8)
    ax.axhline(0.20, color="#c0392b", ls=":", lw=0.8)
    ax.set_xlabel("how much of ê_⊥ is concept content (0 = leftover-only ê, 1 = pole synonym)")
    ax.set_ylabel("value")
    ax.set_title(f"λ={LEAK_HOLD_WEIGHT:g} hold: ê's wording is the knob, not λ")
    ax.set_ylim(-0.05, 1.6)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_polarity(rows: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    held = [r for r in rows if r["used_hold"] > 0]
    clean = [r for r in rows if r["used_hold"] == 0]
    for group, color, label in (
        (clean, "#2471a3", "gender-like, no ê, no hold"),
        (held, "#c0392b", f"energy synonym ê, λ={LEAK_HOLD_WEIGHT:g}"),
    ):
        group = sorted(group, key=lambda r: r["bend"])
        ax.plot([r["bend"] for r in group], [r["collapse"] for r in group], "o-", color=color, label=label)
    ax.axhline(LIVE_COLLAPSE, color="#c0392b", ls="--", lw=0.9, label=f"live energy-v14 {LIVE_COLLAPSE:+.2f}")
    ax.axhline(LIVE_GENDER_COLLAPSE, color="#2471a3", ls="--", lw=0.9, label=f"live gender-v14 {LIVE_GENDER_COLLAPSE:+.2f}")
    ax.axvline(1.0, color="#888888", ls=":", lw=0.8)
    ax.annotate("even reply = odd reply", (1.02, -0.45), fontsize=8, color="#555555")
    ax.set_xlabel("bend — size of the even (non-mirror) ±1 reply")
    ax.set_ylabel("collapse  cos(d+, d−)")
    ax.set_title("±1 polarity is set by the stack, not by ê / λ / D")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_report(blob: dict, path: Path) -> None:
    cells = {r["name"]: r for r in blob["cells"]}
    gender = cells["gender_like_no_e"]
    flat = cells["energy_2d_synonym_l8"]
    no_hold = cells["energy_highd_pair_odd"]
    synonym = cells["energy_highd_synonym_l8"]
    tiny = cells["energy_highd_tiny_l8"]
    messy = cells["energy_highd_tiny_messy_l8"]
    pin = cells["energy_highd_medium_pin_l8"]
    left1 = cells["energy_highd_leftover_l1"]
    left8 = cells["energy_highd_leftover_l8"]
    sub_left = cells["energy_highd_sub_e_leftover"]
    sub_syn = cells["energy_highd_sub_e_synonym"]
    bent = cells["energy_bend_synonym_l8"]
    analogue = blob["analogue"]
    live_wide = [r for r in blob["lambda_dim"] if r["dim"] == 1024]
    p = blob["synonym_cover"]
    lines = [
        "# High-D hold-ê: stiffness, ê wording, and the ±1 break",
        "",
        "The overlap / rich / faithful cells are orthonormal 2-D: the short",
        "declared û *is* the whole concept, and the student is a free",
        "per-coordinate residual. This cell keeps the live loss",
        "(`lm_hidden_targets` + `lm_axis_hold` on `ê_⊥ = ê − (ê·û)û`) and",
        "adds the three things live energy-v14 has and 2-D does not: a",
        "hidden width, concept content off the short caption axis, and a",
        "±1 pair that is not a perfect mirror (two forward passes with the",
        "LoRA multiplier flipped are not odd in the multiplier).",
        "",
        "CPU only. No Hub, no GPU, no Music 3 weights. `--lm_target v9`",
        "default is untouched.",
        "",
        "## Verdict",
        "",
        (
            f"**λ=8 is not a portable number, and it was never the knob.** "
            f"`F.mse_loss` averages over the hidden width, `lm_axis_hold` "
            f"does not, so the fit keeps `a_ê/(1 + λ·D/2)`. λ=8 on a 2-D "
            f"cell is a stiffness of 8; on a D=1024 hidden state it is "
            f"{live_wide[-1]['lambda_eff']:.0f}. At that width λ ∈ "
            f"{{0.3, 1, 8}} all land on the same residual "
            f"(c+ {live_wide[1]['c_plus_predicted']:.3f} / "
            f"{live_wide[2]['c_plus_predicted']:.3f} / "
            f"{live_wide[3]['c_plus_predicted']:.3f}, ceiling "
            f"{live_wide[0]['c_plus_if_held']:.3f}). Dropping to "
            f"`--hold_weight 1` buys stiffness, not less leak."
        ),
        "",
        (
            f"**Trainer c+ cannot stay at 0.97 if the hold works.** With "
            f"`p = |â·ê̂_⊥|` the ceiling is `√(1−p²)`. Gender-like keeps no "
            f"ê, so p=0 and c+ {gender['c_plus']:+.3f} — that is "
            f"\"I copied pair-odd\", not \"the slider locked\". The energy "
            f"synonym ê has p={synonym['hold_cover']:.2f}, so a working hold "
            f"must print c+ {synonym['c_plus_predicted']:+.3f}. Live "
            f"energy-v14 logged c+ {LIVE_C_PLUS:.2f} against a caption-axis "
            f"gate of {LIVE_GATE_ALIGN:.2f}: that is the hold doing its job "
            f"on a synonym ê, not the hold failing."
        ),
        "",
        (
            f"**What actually failed live is ê's wording.** The hold eats one "
            f"direction. On the synonym ê that direction is "
            f"{synonym['hold_on_content']:.0%} concept content, so cos to the "
            f"short probe *rises* to {synonym['cos_short_u']:+.3f} while cos to "
            f"the concept falls to {synonym['cos_intended']:+.3f} and "
            f"{synonym['leftover_kept']:.0%} of the leak survives. A 2-D field "
            f"cannot show that split — there û and the concept are one axis "
            f"(same cell in 2-D: cos û {flat['cos_short_u']:+.3f}, cos concept "
            f"{flat['cos_intended']:+.3f}, leak {flat['leftover_leak']:+.3f}, "
            f"**PASS**)."
        ),
        "",
        (
            f"**The ±1 break is not geometry.** With a symmetric pair-odd "
            f"teacher and any residual linear in the slider scale, the pole "
            f"MSE splits into `|w_odd − a|² + |w_even|²` and the hold splits "
            f"the same way, so `w_even` never leaves 0 and `cos(d+, d−) = −1`. "
            f"{blob['polarity_rows']} cells with a *free* even parameter — "
            f"D up to {max(r['dim'] for r in blob['polarity'])}, λ up to "
            f"{max(r['hold_weight'] for r in blob['polarity']):.0f}, tiny and "
            f"messy ê_⊥ — all print collapse "
            f"{blob['polarity_max_collapse']:+.6f}. Live ±1 is two forward "
            f"passes with the LoRA multiplier flipped, through attention "
            f"softmax and SwiGLU MLPs; none of that is odd in the multiplier, "
            f"so the replies are only approximate mirrors. `bend` is the size "
            f"of the non-mirror part, and collapse "
            f"{LIVE_GENDER_COLLAPSE:+.2f} → bend "
            f"{bend_for_collapse(LIVE_GENDER_COLLAPSE):.2f}, collapse "
            f"{LIVE_COLLAPSE:+.2f} → bend {bend_for_collapse(LIVE_COLLAPSE):.2f}. "
            f"Live energy-v14's `+{LIVE_COLLAPSE:.2f}` says the even reply was "
            f"*larger* than the odd one (bend > 1) at whatever scale that run "
            f"drove the LoRA to. That is a fact about the stack, not about ê: "
            f"no ê and no λ in this fixture moves collapse at all."
        ),
        "",
        (
            f"**Recommendation.** Leftover-only ê is right on direction and is "
            f"the canary for *that*: at λ=1 it is bipolar "
            f"({left1['collapse']:+.3f}), trainable "
            f"(loss {left1['loss']:.3f}), keeps the concept "
            f"({left1['cos_intended']:+.3f}) and still leaves leak "
            f"{left1['leftover_leak']:+.3f} — λ=8 barely moves it "
            f"({left8['leftover_leak']:+.3f}), because the leftover a single "
            f"caption pair does not name survives any λ. It is *not* a canary "
            f"for the stiffness that broke ±1. `pair_odd_sub_e` on the same ê "
            f"lands on the same residual (c+ {sub_left['c_plus']:+.3f} vs "
            f"{left8['c_plus']:+.3f}, leak {sub_left['leftover_leak']:+.3f} vs "
            f"{left8['leftover_leak']:+.3f}) with loss "
            f"{sub_left['loss']:.3f} and no stiffness at all: it is the hold's "
            f"λ→∞ limit reached in one step. Ship leftover-only ê with "
            f"`--hold_weight 1` to test the wording; then move the axis into "
            f"the teacher (`pair_odd_sub_e`, subtracting **ê_⊥**, not raw ê) "
            f"as the PR."
        ),
        "",
        "## The live signature, side by side",
        "",
        "Two rows are calibrated, not predicted: gender-like takes the ±1",
        f"asymmetry implied by its own collapse log (bend {BEND_GENDER:g}), and",
        "the energy analogue searches the asymmetry — size and gain / rotation",
        "split — nearest the live `(c+, ±1)` pair. Everything else in both rows",
        "is the geometry.",
        "",
        "| number | live gender-v14 | fixture | live energy-v14 | fixture analogue |",
        "|---|---:|---:|---:|---:|",
        f"| caption-axis gate `\\|odd·û\\|/\\|\\|odd\\|\\|` | 1.00 (clean pair) | "
        f"{gender['gate_align']:.2f} | {LIVE_GATE_ALIGN:.2f} | "
        f"{analogue['gate_align']:.2f} |",
        f"| trainer c+ | {LIVE_GENDER_C_PLUS:.2f} | {gender['c_plus']:+.3f} | "
        f"{LIVE_C_PLUS:.2f} | {analogue['c_plus']:+.3f} |",
        f"| collapse | {LIVE_GENDER_COLLAPSE:+.2f} | {gender['collapse']:+.3f} | "
        f"{LIVE_COLLAPSE:+.2f} | {analogue['collapse']:+.3f} |",
        f"| loss | {LIVE_GENDER_LOSS:.3f} | {gender['loss']:.3f} | "
        f"not in the 0.02 band | {analogue['loss']:.3f} |",
        f"| hold λ | 0 (no ê) | 0 | {LEAK_HOLD_WEIGHT:g} | {LEAK_HOLD_WEIGHT:g} |",
        f"| ±1 asymmetry | — | {BEND_GENDER:g} | — | "
        f"{analogue['bend']:g} ({analogue['bend_parallel']:g} gain / "
        f"{1 - analogue['bend_parallel']:g} rotation) |",
        f"| ‖d+‖ / ‖d−‖ | — | {gender['norm_ratio']:.2f} | — | "
        f"{analogue['norm_ratio']:.2f} |",
        "",
        (
            f"The gender row lands all three of its live numbers. The energy "
            f"search lands the collapse at bend {analogue['bend']:g} and gets "
            f"c+ {analogue['c_plus']:+.3f} against a logged {LIVE_C_PLUS:.2f}, "
            f"with the whole rest of the row — gate, λ, ê, D — untouched. Both "
            f"live energy numbers are therefore consistent with one thing: an "
            f"even reply {analogue['bend']:g}× the odd reply, on top of a hold "
            f"that is otherwise doing exactly what the closed form says."
        ),
        "",
        (
            f"The search prefers a pure rotation "
            f"({analogue['bend_parallel']:g} gain), which predicts the two "
            f"poles move by the *same* amount: ‖d+‖/‖d−‖ = "
            f"{analogue['norm_ratio']:.2f}. That is checkable without a new "
            f"run — the trainer already writes `pperc` and `nperc` per step in "
            f"`<name>_train.jsonl`. A gain share raises c+ at the same "
            f"collapse and splits the two poles "
            f"(at bend {blob['analogue_gain_bend']:g}, "
            f"{blob['analogue_gain_parallel']:g} gain: c+ "
            f"{blob['analogue_gain_c_plus']:+.3f}, ‖d+‖/‖d−‖ "
            f"{blob['analogue_gain_ratio']:.2f}), so if energy-v14's two poles "
            f"moved by very different amounts the asymmetry is bigger than "
            f"{analogue['bend']:g} and part of it is gain. Read those two "
            f"columns before spending a run on ê."
        ),
        "",
        "## Field",
        "",
        "```",
        "dim 0      short û          the declared slider_positive / negative pair",
        "dim 1      concept off û    loudness the short pair misses (dense, slammed, 168)",
        "dim 2..    leftover         unused mix / BPM wording / genre / syntax",
        "",
        f"a = {blob['scale']:.2f} · unit({LIVE_GATE_ALIGN:.2f} û + "
        f"{blob['content']:.2f} concept + {blob['leftover']:.2f} leftover)",
        f"|a·û|/||a|| = {blob['gate_align']:.2f}      (live energy-v14 gate log {LIVE_GATE_ALIGN:.2f})",
        "",
        "ê = on_u · û + on_content · concept + on_leftover · leftover",
        "ê_⊥ = ê − (ê·û)û            what lm_axis_hold renormalizes and holds",
        "p  = |â · ê̂_⊥|              the share of the teacher the hold removes",
        "",
        "δ(s) = s·w + bend·(par·w + √(1−par²)·G w)      ±1 replies of a stack",
        "                            bend = size of the even reply, par = the",
        "                            gain share of it, G fixed orthogonal",
        "```",
        "",
        "## Live bullets, one row each",
        "",
        CELL_HEADER,
    ]
    for row in blob["cells"]:
        lines.append(_cell_md(row))
    lines += [
        "",
        "`p`, `c+ pred` and `perc` are closed form:",
        "",
        "```",
        "k    = 1 / (1 + λ·D/2)                     # hold_shrink",
        "c+   = (1 − (1−k)p²) / √(1 − p² + k²p²)",
        "perc = (1 − k)·p",
        "ceil = √(1 − p²)                           # k → 0, hold has done its job",
        "```",
        "",
        "Every linear row above matches it to three decimals, so the table is",
        "a geometry statement, not a training artifact.",
        "",
        "Two flags separate the confusions this table exists for:",
        "",
        (
            f"- `looks_like_v12` — c+ ≥ 0.90, i.e. the residual copied the "
            f"pair-odd whole. True on `gender_like_no_e` "
            f"({gender['c_plus']:+.3f}) and on `energy_highd_pair_odd` "
            f"({no_hold['c_plus']:+.3f}, leak kept "
            f"{no_hold['leftover_kept']:.2f} — v12 / Hub are not leak-free). "
            f"False on every working hold, including the 2-D **PASS** row "
            f"({flat['c_plus']:+.3f}, perc {flat['perc']*100:.0f}%). A hold "
            f"that leaves c+ at gender's 0.97 has not held anything."
        ),
        (
            f"- `hold_explains_c_plus` — measured c+ is within "
            f"{0.06:.2f} of the closed form. True for every linear row, false "
            f"for both bend rows (`energy_bend_synonym_l8` "
            f"{bent['c_plus']:+.3f} vs predicted "
            f"{bent['c_plus_predicted']:+.3f}). That is the discriminator: a "
            f"low c+ inside the closed form is the hold working; a low c+ "
            f"*below* it is the stack, not ê."
        ),
        "",
        "## λ is not portable across hidden width",
        "",
        f"Closed form at the synonym ê (`p = {p:.3f}`, ceiling "
        f"`{(1 - p * p) ** 0.5:.3f}`):",
        "",
        "| D | λ=0.3 | λ=1 | λ=8 | λ·D/2 at λ=8 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for dim in DIM_GRID:
        row = {r["hold_weight"]: r for r in blob["lambda_dim"] if r["dim"] == dim}
        lines.append(
            f"| {dim} | {row[0.3]['c_plus_predicted']:.3f} | "
            f"{row[1.0]['c_plus_predicted']:.3f} | {row[8.0]['c_plus_predicted']:.3f} | "
            f"{row[8.0]['lambda_eff']:.0f} |"
        )
    lines += [
        "",
        "![c+ vs λ and vs λ·D/2](lm-highd-leftover/lambda_dim.png)",
        "",
        "Fitted rows (same ê, Adam, seed 0):",
        "",
        "| cell | λ·D/2 | c+ | c+ pred | cos û | cos concept | leak kept | loss |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in blob["lambda_fit"]:
        lines.append(
            f"| `{row['name']}` | {row['lambda_eff']:.0f} | {row['c_plus']:+.3f} | "
            f"{row['c_plus_predicted']:+.3f} | {row['cos_short_u']:+.3f} | "
            f"{row['cos_intended']:+.3f} | {row['leftover_kept']:.2f} | {row['loss']:.3f} |"
        )
    lines += [
        "",
        (
            f"The un-normalized half is also where a live `loss 278` comes "
            f"from: while the fit still carries the teacher's ê component the "
            f"hold term is `λ·(a·ê̂)²`, so λ=8 needs only `a·ê̂ ≈ "
            f"{(LIVE_SPIKE_LOSS / LEAK_HOLD_WEIGHT) ** 0.5:.1f}` in hidden "
            f"units to print {LIVE_SPIKE_LOSS:.0f}. On this field the same "
            f"term is {hold_spike(p * blob['scale'], LEAK_HOLD_WEIGHT):.2f}. "
            f"Nothing semantic exploded; the hold is a sum where the pole "
            f"MSE is a mean."
        ),
        "",
        "## ê's wording is the knob",
        "",
        f"λ={LEAK_HOLD_WEIGHT:g} throughout; only what the leak captions *say* moves.",
        "",
        "| ê_⊥ content share | p | hold on content | hold on leftover | c+ | cos û | cos concept | leak | leak kept |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in blob["wording"]:
        lines.append(
            f"| {row['hold_on_content']:.2f} | {row['hold_cover']:.2f} | "
            f"{row['hold_on_content']:.2f} | {row['hold_on_leftover']:.2f} | "
            f"{row['c_plus']:+.3f} | {row['cos_short_u']:+.3f} | "
            f"{row['cos_intended']:+.3f} | {row['leftover_leak']:+.3f} | "
            f"{row['leftover_kept']:.2f} |"
        )
    lines += [
        "",
        "![concept vs leak against ê wording](lm-highd-leftover/wording.png)",
        "",
        "The probe cosine and the concept cosine move in *opposite*",
        "directions as ê becomes a synonym. Reading `cos(d+, û)` — or a",
        "2-D cell — says the slider locked while the concept is being eaten.",
        "",
        "## How completely does ê name the leak?",
        "",
        "One declared caption pair is one direction. Leftover the poles",
        "carry off that direction survives every λ.",
        "",
        "| ê names | p | λ=1 leak | λ=1 kept | λ=8 leak | λ=8 kept | λ=1 cos concept |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    by_match = {r["leftover_match"]: r for r in blob["match_l8"]}
    for row in blob["match_l1"]:
        eight = by_match[row["leftover_match"]]
        share = f"{row['leftover_match']:.2f}"
        lines.append(
            f"| {share} | {row['hold_cover']:.2f} | {row['leftover_leak']:+.3f} | "
            f"{row['leftover_kept']:.2f} | {eight['leftover_leak']:+.3f} | "
            f"{eight['leftover_kept']:.2f} | {row['cos_intended']:+.3f} |"
        )
    lines += [
        "",
        "## ±1 polarity",
        "",
        "| bend | energy synonym ê λ=8: c+ / ±1 / cos concept | gender-like: c+ / ±1 / loss | implied bend from ±1 |",
        "|---:|---|---|---:|",
    ]
    held = {r["bend"]: r for r in blob["bend"] if r["used_hold"] > 0}
    clean = {r["bend"]: r for r in blob["bend"] if r["used_hold"] == 0}
    for bend in sorted(held):
        h = held[bend]
        c = clean[bend]
        lines.append(
            f"| {bend:g} | {h['c_plus']:+.3f} / {h['collapse']:+.3f} / "
            f"{h['cos_intended']:+.3f} | {c['c_plus']:+.3f} / "
            f"{c['collapse']:+.3f} / {c['loss']:.3f} | "
            f"{bend_for_collapse(c['collapse']):.2f} |"
        )
    lines += [
        "",
        "![collapse vs bend](lm-highd-leftover/polarity.png)",
        "",
        (
            f"`bend = {BEND_GENDER:g}` reproduces live gender-v14 on the clean "
            f"pair: c+ {gender['c_plus']:+.3f} (live {LIVE_GENDER_C_PLUS:.2f}), "
            f"±1 {gender['collapse']:+.3f} (live {LIVE_GENDER_COLLAPSE:+.2f}), "
            f"loss {gender['loss']:.3f} (live {LIVE_GENDER_LOSS:.3f}). "
            f"`bend = {BEND_ENERGY:g}` is what live energy-v14's "
            f"±1 {LIVE_COLLAPSE:+.2f} implies, and it costs the whole slider "
            f"on this field (c+ {bent['c_plus']:+.3f}, cos concept "
            f"{bent['cos_intended']:+.3f}) regardless of ê. Polarity crosses "
            f"zero exactly when the even reply matches the odd reply "
            f"(bend = 1)."
        ),
        "",
        f"Free-even geometry grid: {blob['polarity_rows']} cells, max collapse "
        f"{blob['polarity_max_collapse']:+.6f}, max even norm "
        f"{blob['polarity_max_even']:.2e}. Geometry cannot break ±1.",
        "",
        "## Row notes",
        "",
        f"- `energy_highd_pair_odd` (no ê, no hold) copies the leak: leak "
        f"{no_hold['leftover_leak']:+.3f}, leak kept "
        f"{no_hold['leftover_kept']:.2f}, c+ {no_hold['c_plus']:+.3f}. "
        f"v12 / Hub are not leak-free on these poles.",
        f"- `energy_highd_tiny_l8` (‖ê_⊥‖ {tiny['hold_norm']:.2f}) is "
        f"row-for-row `energy_highd_synonym_l8` (‖ê_⊥‖ "
        f"{synonym['hold_norm']:.2f}): `lm_axis_hold` renormalizes, so a tiny "
        f"leftover is not a weak hold.",
        f"- `energy_highd_tiny_messy_l8` rotates that tiny leftover into junk. "
        f"p drops to {messy['hold_cover']:.2f} and the hold becomes a near "
        f"no-op — concept back to {messy['cos_intended']:+.3f}, leak still "
        f"{messy['leftover_leak']:+.3f}.",
        f"- `energy_highd_medium_pin_l8` is the live rewrite: ê·û falls from "
        f"{synonym['e_dot_u']:+.2f} to {pin['e_dot_u']:+.2f} and every other "
        f"number is unchanged (c+ {pin['c_plus']:+.3f} vs "
        f"{synonym['c_plus']:+.3f}). Pinning \"medium energy\" on both leak "
        f"captions does not move ê_⊥ while density and genre still read as the "
        f"poles.",
        f"- `energy_highd_sub_raw_e_leftover` subtracts raw ê instead of ê_⊥ "
        f"and takes û with it (cos û {cells['energy_highd_sub_raw_e_leftover']['cos_short_u']:+.3f} "
        f"vs {sub_left['cos_short_u']:+.3f}). `pair_odd_sub_e` has to subtract "
        f"ê_⊥, the same axis the hold uses.",
        f"- `energy_highd_sub_e_synonym` shows the teacher change is not a "
        f"wording fix: on a synonym ê it lands where the λ=8 hold lands "
        f"(cos concept {sub_syn['cos_intended']:+.3f} vs "
        f"{synonym['cos_intended']:+.3f}), only without the stiffness.",
        f"- `gender_like_no_e` leak {gender['leftover_leak']:+.3f} is bend "
        f"junk, not caption leak: those poles carry no leftover at all "
        f"(leak kept {gender['leftover_kept']:.2f}). It is what a "
        f"±1 {gender['collapse']:+.3f} reply costs off-axis.",
        "",
        "## What is now a cell, and what still is not",
        "",
        "Now a cell:",
        "",
        "- trainer c+ (cos with `a`) and slider-cos (cos with û) as separate",
        "  columns, plus the closed-form ceiling, so \"looks like v12\" and",
        "  \"hold is working\" are different verdicts.",
        "- a concept axis the short caption pair only partly names, so the",
        "  probe cosine and the concept cosine can disagree.",
        "- tiny ê_⊥ (norm 0.20 vs 0.60) — identical rows, because",
        "  `lm_axis_hold` renormalizes. Only the direction matters.",
        "- messy ê_⊥: rotating a tiny leftover into junk drops p and turns the",
        "  hold into a no-op in *both* directions.",
        "- the same-loudness pin: dropping the loud/quiet words moves ê·û from",
        f"  {synonym['e_dot_u']:+.2f} to {pin['e_dot_u']:+.2f} and leaves ê_⊥",
        f"  where it was, so the row is unchanged (c+ {pin['c_plus']:+.3f} vs",
        f"  {synonym['c_plus']:+.3f}).",
        "- λ·D/2 as the stiffness that acts, and λ's irrelevance at live width.",
        "- leftover-only ê × λ ∈ {0.3, 1, 8} against `pair_odd_sub_e`.",
        "",
        "Still not a cell:",
        "",
        "- **the ±1 break as a consequence of ê / λ / D.** It is not one. It",
        "  needs a non-mirror ±1 reply, which is a property of the stack.",
        "  `bend` is a knob here, not a prediction.",
        "- **the live 278 transient.** The scale is explained (`λ·(a·ê̂)²` with",
        "  a sum-vs-mean mismatch), but a fixed 2-D/high-D vector fit with",
        "  Adam descends monotonically; reproducing the overshoot needs the",
        "  LoRA parametrization and the live clip / AdamW schedule.",
        "- **whether Qwen actually hears \"genre + BPM, no density\" as unused.**",
        "  `leftover_match` is the assumption that decides the residual leak,",
        "  and only a live probe can measure it.",
        "",
        "## How to run",
        "",
        "```bash",
        "PYTHONPATH=. python analysis/slider2d/run_lm_highd.py --out docs/lm-highd-leftover",
        "PYTHONPATH=. pytest tests/test_lm_highd_leftover.py -q",
        "```",
        "",
        f"Seed `{blob['seed']}`, `{blob['steps']}` Adam steps, lr 0.08.",
        "",
        f"2-D cross-check: this harness on the flat field prints slider "
        f"{flat['cos_intended']:+.3f} / leak {flat['leftover_leak']:+.3f} / "
        f"c+ {flat['c_plus']:+.3f} / perc {flat['perc']*100:.0f}%, against "
        f"{blob['overlap_2d']['cos_intended']:+.3f} / "
        f"{blob['overlap_2d']['leak_ratio']:+.3f} / "
        f"{blob['overlap_2d']['cos_teacher']:+.3f} / "
        f"{blob['overlap_2d']['perc']*100:.0f}% for `pole_synonym_slider_l8` in "
        f"[lm-hold-overlap.md](lm-hold-overlap.md). Same physics, new columns.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    cells = cell_table(steps=args.steps, seed=args.seed)
    closed = lambda_dim_sweep()
    fitted = lambda_fit_sweep(steps=args.steps, seed=args.seed)
    wording = content_sweep(steps=args.steps, seed=args.seed)
    match_l1 = match_sweep(hold_weight=1.0, steps=args.steps, seed=args.seed)
    match_l8 = match_sweep(hold_weight=LEAK_HOLD_WEIGHT, steps=args.steps, seed=args.seed)
    bend = bend_sweep(steps=args.steps, seed=args.seed)
    polarity = polarity_grid(steps=200, seed=args.seed)
    analogue, analogue_grid = live_v14_analogue(seed=args.seed)
    pure_bend = calibrate_bend(LIVE_COLLAPSE, parallel=0.0, steps=args.steps, seed=args.seed)
    gain_side = min(
        (r for r in analogue_grid if r["bend_parallel"] >= 0.6),
        key=lambda r: abs(r["collapse"] - LIVE_COLLAPSE),
    )
    overlap_2d = score_overlap_policy(
        "pole_synonym_slider_l8",
        overlap=0.5,
        hold_weight=LEAK_HOLD_WEIGHT,
        ortho="slider",
        steps=200,
        seed=args.seed,
    )
    energy = energy_field()
    blob = {
        "steps": args.steps,
        "seed": args.seed,
        "dim": energy.dim,
        "scale": energy.scale,
        "content": energy.content,
        "leftover": energy.leftover,
        "gate_align": energy.gate_align(),
        "synonym_cover": synonym_cover(),
        "leftover_cover": next(
            r["hold_cover"] for r in cells if r["name"] == "energy_highd_leftover_l1"
        ),
        "bend_gender": BEND_GENDER,
        "bend_energy": BEND_ENERGY,
        "live": {
            "gate_align": LIVE_GATE_ALIGN,
            "c_plus": LIVE_C_PLUS,
            "collapse": LIVE_COLLAPSE,
            "gender_c_plus": LIVE_GENDER_C_PLUS,
            "gender_collapse": LIVE_GENDER_COLLAPSE,
            "gender_loss": LIVE_GENDER_LOSS,
            "spike_loss": LIVE_SPIKE_LOSS,
        },
        "cells": [compact(r) for r in cells],
        "analogue": compact(analogue),
        "analogue_grid": [
            {
                key: round(float(r[key]), 6)
                for key in (
                    "bend",
                    "bend_parallel",
                    "c_plus",
                    "collapse",
                    "norm_ratio",
                    "cos_short_u",
                    "cos_intended",
                    "leftover_leak",
                    "perc",
                    "loss",
                )
            }
            for r in analogue_grid
        ],
        "analogue_pure_rotation": pure_bend,
        "analogue_gain_bend": gain_side["bend"],
        "analogue_gain_parallel": gain_side["bend_parallel"],
        "analogue_gain_c_plus": gain_side["c_plus"],
        "analogue_gain_ratio": gain_side["norm_ratio"],
        "lambda_dim": closed,
        "lambda_fit": [compact(r) for r in fitted],
        "wording": [compact(r, history=False) for r in wording],
        "match_l1": [compact(r) for r in match_l1],
        "match_l8": [compact(r) for r in match_l8],
        "bend": [compact(r) for r in bend],
        "polarity": [compact(r, history=False) for r in polarity],
        "polarity_rows": len(polarity),
        "polarity_max_collapse": max(r["collapse"] for r in polarity),
        "polarity_max_even": max(r["even_norm"] for r in polarity),
        "overlap_2d": {
            "cos_intended": overlap_2d["cos_intended"],
            "leak_ratio": overlap_2d["leak_ratio"],
            "cos_teacher": overlap_2d["cos_teacher"],
            "perc": overlap_2d["perc"],
            "loss": overlap_2d["loss"],
        },
        "cover_grid": list(COVER_GRID),
        "lambdas": list(HOLD_LAMBDAS),
        "dims": list(DIM_GRID),
    }
    (out / "metrics.json").write_text(json.dumps(blob, indent=2) + "\n", encoding="utf-8")
    plot_lambda_dim(closed, blob["lambda_fit"], out / "lambda_dim.png")
    plot_wording(blob["wording"], out / "wording.png")
    plot_polarity(blob["bend"], out / "polarity.png")
    write_report(blob, out.parent / "lm-highd-leftover.md")

    for row in cells:
        print(
            f"  {row['name']:32s} p={row['hold_cover']:.2f} "
            f"c+={row['c_plus']:+.3f} pred={row['c_plus_predicted']:+.3f} "
            f"cosu={row['cos_short_u']:+.3f} cosi={row['cos_intended']:+.3f} "
            f"leak={row['leftover_leak']:+.3f} col={row['collapse']:+.3f} "
            f"loss={row['loss']:.3f} pass={row['pass']}"
        )
    print(
        f"free-even polarity grid: {blob['polarity_rows']} cells, "
        f"max collapse {blob['polarity_max_collapse']:+.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
