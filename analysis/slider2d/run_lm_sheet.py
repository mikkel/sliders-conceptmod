#!/usr/bin/env python3
"""Score the pair-odd midpoint against a sheet: is the target a caption?"""

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

from analysis.slider2d.sheet import (
    BETA_GRID,
    COMMON_GRID,
    GARBLE_MAX,
    LEAK_LOCK,
    LIVE_COLLAPSE_WARN,
    LIVE_PROBE_COS,
    LIVE_V4_COLLAPSE,
    LIVE_V4_C_PLUS,
    SHEET_LOCK,
    SWING_FLOOR,
    beta_sweep,
    common_share,
    common_sweep,
    first_above,
    flip_point,
    floatable,
    gender_cell,
    gender_like_field,
    leaky_cell,
    leaky_field,
    live_log_table,
    live_probe_table,
    null_space_table,
    teacher_sheet_table,
)
from analysis.slider2d.live_exam import live_exam_cell


DEFAULT_OUT = _REPO / "docs" / "lm-sheet-goodhart"
LIVE_BAND = (
    common_share(min(LIVE_PROBE_COS.values())),
    common_share(max(LIVE_PROBE_COS.values())),
)


def _fmt_bool(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


CELL_HEADER = (
    "| recipe | pole_mode | target | cos± | ±1 | on-sheet | kept | garble | "
    "argmax | swing | leak_tok | null | KL | verdict |\n"
    "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"
)


def _cell_md(row: dict) -> str:
    return (
        f"| `{row['name']}` | {row['pole_mode']} | `{row['teacher']}` | "
        f"{row['pair_odd_cos']:+.3f} | {row['collapse']:+.3f} | "
        f"{row['on_sheet']:.3f} | {row['on_sheet_kept']:.3f} | "
        f"{row['garble']:.3f} | {row['argmax_on_sheet']:.2f} | "
        f"{row['swing_kept']:.2f} | {row['leak_tok']:+.3f} | "
        f"{row['null_kept']:.2f} | {row['kl_pole']:.3f} | "
        f"**{_fmt_bool(row['pass'])}** |"
    )


TEACHER_HEADER = (
    "| target | is a caption | off-caption | c kept | on-sheet | garble | "
    "argmax | KL to pole | says at ±1 |\n"
    "|---|---|---:|---:|---:|---:|---:|---:|---|"
)


def _teacher_md(row: dict) -> str:
    return (
        f"| `{row['name']}` | {'yes' if row['is_caption'] else 'no'} | "
        f"{row['off_caption']:.3f} | {row['sheet_dir_kept']:.2f} | "
        f"{row['on_sheet']:.3f} | {row['garble']:.3f} | "
        f"{row['argmax_on_sheet']:.2f} | {row['kl_pole']:.3f} | "
        f"`{row['says']}` |"
    )


def plot_common(sweep: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    x = [r["common"] for r in sweep]
    for ax in axes:
        ax.axvspan(*LIVE_BAND, color="#f39c12", alpha=0.16, lw=0)
        ax.set_xlabel("common share  |c| / |a|")
    axes[0].plot(x, [r["hidden_on_sheet"] for r in sweep], "o-", color="#c0392b", label="hidden: on-sheet")
    axes[0].plot(x, [r["hidden_garble"] for r in sweep], "s--", color="#c0392b", label="hidden: off-sheet mass")
    axes[0].plot(x, [r["kl_on_sheet"] for r in sweep], "o-", color="#2471a3", label="semantic_kl: on-sheet")
    axes[0].plot(x, [r["kl_garble"] for r in sweep], "s--", color="#2471a3", label="semantic_kl: off-sheet mass")
    axes[0].axhline(GARBLE_MAX, color="#7f8c8d", ls=":", lw=0.9)
    axes[0].set_ylabel("next-token mass at ±1")
    axes[0].set_title("what the student says")
    axes[1].plot(x, [r["hidden_pair_odd_cos"] for r in sweep], "o-", color="#c0392b", label="hidden: cos(d+, a)")
    axes[1].plot(x, [r["hidden_collapse"] for r in sweep], "s--", color="#c0392b", label="hidden: cos(d+, d−)")
    axes[1].plot(x, [r["kl_pair_odd_cos"] for r in sweep], "o-", color="#2471a3", label="semantic_kl: cos(d+, a)")
    axes[1].plot(x, [r["kl_collapse"] for r in sweep], "s--", color="#2471a3", label="semantic_kl: cos(d+, d−)")
    axes[1].axhline(LIVE_V4_C_PLUS, color="#7f8c8d", ls=":", lw=0.9)
    axes[1].axhline(LIVE_V4_COLLAPSE, color="#7f8c8d", ls=":", lw=0.9)
    axes[1].set_ylabel("logged hidden geometry")
    axes[1].set_title("what the trainer logs (flat, for hidden)")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7.4)
    fig.suptitle(
        "the two logged numbers do not move; the sheet does. shaded = live v4 axes",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_beta(sweep: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    x = [r["common_beta"] for r in sweep]
    ax.plot(x, [r["on_sheet"] for r in sweep], "o-", color="#1e8449", label="on-sheet mass")
    ax.plot(x, [r["garble"] for r in sweep], "s-", color="#c0392b", label="off-sheet mass")
    ax.set_xlabel("--common_beta  (0 = live v9 midpoint, 1 = the raw caption)")
    ax.set_ylabel("next-token mass at ±1")
    ax.grid(alpha=0.25)
    twin = ax.twinx()
    twin.plot(x, [r["pair_odd_cos"] for r in sweep], "^--", color="#7f8c8d", label="cos(d+, a)")
    twin.plot(x, [r["collapse"] for r in sweep], "v--", color="#5d6d7e", label="cos(d+, d−)")
    twin.set_ylabel("logged hidden geometry")
    lines = ax.get_lines() + twin.get_lines()
    ax.legend(lines, [ln.get_label() for ln in lines], fontsize=7.6, loc="center right")
    ax.set_title("β walks the target onto the sheet and the log the other way")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_recipes(rows: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    # Label offsets alternate so the three stacked hidden rows stay readable.
    for i, row in enumerate(sorted(rows, key=lambda r: r["on_sheet_kept"])):
        ok = row["pass"]
        ax.scatter(
            row["pair_odd_cos"],
            row["on_sheet_kept"],
            s=84,
            marker="o" if ok else "X",
            color="#1e8449" if ok else "#c0392b",
            zorder=3,
        )
        right = row["pair_odd_cos"] < 0.85
        ax.annotate(
            row["name"],
            (row["pair_odd_cos"], row["on_sheet_kept"]),
            textcoords="offset points",
            xytext=(9 if right else -9, 4 + 9 * (i % 2)),
            ha="left" if right else "right",
            fontsize=7.6,
        )
    ax.axhline(SHEET_LOCK, color="#7f8c8d", ls=":", lw=0.9)
    ax.axvline(LIVE_V4_C_PLUS, color="#f39c12", ls="--", lw=0.9)
    ax.annotate(
        "live gender-lm-v4 log",
        (LIVE_V4_C_PLUS, 0.62),
        rotation=90,
        fontsize=7.2,
        color="#b9770e",
        ha="right",
    )
    ax.annotate("sheet gate", (0.5, SHEET_LOCK + 0.012), fontsize=7.2, color="#5d6d7e")
    ax.set_xlim(0.45, 1.09)
    ax.set_ylim(0.25, 1.08)
    ax.set_xlabel("cos(d+, a)  — the live log column, 'better' to the right")
    ax.set_ylabel("share of the caption's on-sheet mass kept")
    ax.set_title("the two axes are anti-correlated: that is the Goodhart")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_report(blob: dict, path: Path) -> None:
    gender = blob["gender_cell"]
    leaky = blob["leaky_cell"]
    v9 = next(r for r in leaky if r["name"] == "v9_hidden")
    sub_e = next(r for r in leaky if r["name"] == "v15_pair_odd_sub_e")
    kl_mid = next(r for r in leaky if r["name"] == "kl_on_midpoint")
    kl_sub = next(r for r in leaky if r["name"] == "v16_semantic_kl_sub_e")
    faith_sub = next(r for r in leaky if r["name"] == "faithful_sub_e")
    caption = next(r for r in blob["teachers"] if r["name"] == "caption")
    midpoint = next(r for r in blob["teachers"] if r["name"] == "pair_odd")
    exam = blob["live_exam"]
    lines = [
        "# The sheet cell: hidden-only fields cannot see lyric garble",
        "",
        "Generated by `analysis/slider2d/run_lm_sheet.py`. CPU only, no Hub, no",
        "GPU, no Music 3 weights. Does not change the live trainer default.",
        "",
        "Every other 2-D / high-D cell in this repo scores a residual against",
        "hidden geometry alone, so none of them can tell \"the student reached",
        "the teacher point\" from \"the student reached a point no caption",
        "occupies\". This cell adds the smallest structure that can: a frozen",
        "linear next-token readout over a tiny vocabulary, which turns each",
        "hidden state into a policy and gives the poles a **sheet** — the",
        "nucleus of what a real caption would say next, plus the written lyric.",
        "",
        "## The geometry, in one line",
        "",
        "With `a = ½(h+−h−)` and `c = ½(h++h−) − h0`, the poles are exactly",
        "`h± = h0 ± a + c`. So the live default target `t± = h0 ± a`",
        "(`--lm_target v9`) is the real pole **minus its whole common",
        "component**. `c` is what both pole captions say and the neutral",
        "skeleton does not: genre, BPM, mood, mix, instruments — the",
        "Structured-Caption specificity that makes the continuation a song.",
        "",
        "The live trainer already prints the size of `c`. For `c ⊥ a`,",
        "`cos(pos−neu, neg−neu) = (‖c‖²−‖a‖²)/(‖c‖²+‖a‖²)`, so the",
        "**common share** `‖c‖/‖a‖` is `√((1+cos)/(1−cos))`. Reading the v4",
        "probe table in MUSIC3.md that way:",
        "",
        "| axis | logged cos | common share | warned? |",
        "|---|---:|---:|---|",
    ]
    for row in blob["live_probe"]:
        lines.append(
            f"| {row['axis']} | {row['probe_cos']:+.2f} | "
            f"{row['common_share']:.3f} | {'yes' if row['warned'] else 'no'} |"
        )
    lines += [
        "",
        "Every shipped LM axis has a common component between",
        f"**{LIVE_BAND[0]:.2f}× and {LIVE_BAND[1]:.2f}×** the norm of the whole",
        "pair-odd axis, and v9 throws all of it away. `--common_beta` is the",
        "flag that would put it back, and `--lm_target v9` ignores it (the",
        "trainer prints a note and drops it).",
        "",
        "The only warning the trainer emits fires above cos",
        f"{LIVE_COLLAPSE_WARN:+.1f}, so it treats a *large* `c` as the problem",
        "(collapse inherited from the raw targets) and a cos near 0 as",
        "healthy. Near 0 is the case where the deleted piece is the same size",
        "as the signal. Only `live-lm-v5` was ever warned, and it also has the",
        "campaign's worst logged cos± and collapse.",
        "",
        "## The target points, with no optimizer in the loop",
        "",
        "Sheet numbers read straight off the hidden state each recipe aims at,",
        f"on the energy-like field (common share {blob['leaky_common']:.3f}, the",
        "live energy probe). `off-caption` is the distance from the target to",
        "the pole it claims to represent, in units of `‖a‖`.",
        "",
        TEACHER_HEADER,
    ]
    lines += [_teacher_md(r) for r in blob["teachers"]]
    lines += [
        "",
        f"The v9 target sits {midpoint['off_caption']:.2f}·`‖a‖` from the caption",
        "it claims to represent — the whole slider axis away — keeps none of",
        f"`c`, and hands {midpoint['garble']*100:.0f}% of its next-token mass to",
        "words on neither pole's sheet, against the caption's own",
        f"{caption['garble']*100:.1f}%. At ±1 it says an off-sheet token outright.",
        "`pair_odd_sub_e` (#20) is very slightly worse, because subtracting",
        "`ê_⊥` moves it further from the caption, not closer. No fitting is",
        "involved in that table: a *perfect* hidden fit lands here.",
        "",
        "## Fitted recipes",
        "",
        "One shared residual per recipe (`δ(σ) = σ·w_odd + |σ|·w_even`, the",
        "`train.py` LM student), three prompt rows with different written",
        "lyrics and pole strengths. Gates are on-sheet share, off-sheet mass,",
        "argmax, token-space leak and audible swing. `cos±` and `±1` are",
        "**logged, never scored** — that is the point of the cell.",
        "",
        "### Gender-like: clean pair, no declared ê, hold 0",
        "",
        CELL_HEADER,
    ]
    lines += [_cell_md(r) for r in gender]
    lines += [
        "",
        "### Energy-like: unused gender inside `a`, ê declared, invisible mix detail",
        "",
        CELL_HEADER,
    ]
    lines += [_cell_md(r) for r in leaky]
    lines += [
        "",
        "## What that table says",
        "",
        f"1. **The lock is real and it is misleading.** `v9_hidden` prints",
        f"   `cos(d+, a) = {v9['pair_odd_cos']:+.3f}` and",
        f"   `cos(d+, d−) = {v9['collapse']:+.3f}` — a flawless bipolar lock, the",
        f"   live gender-lm-v4 log rounded up — while keeping only",
        f"   {v9['on_sheet_kept']*100:.0f}% of the caption's on-sheet mass and",
        f"   putting {v9['garble']*100:.0f}% on words the song has no sheet for.",
        f"   Its argmax at ±1 is off the sheet on every row it can be. The",
        "   cell flags this as `misleading_lock`.",
        f"2. **The #20 leak fix does not touch it.** `pair_odd_sub_e` drives",
        f"   token-space leak to {sub_e['leak_tok']:+.3f} (from",
        f"   {v9['leak_tok']:+.3f}) and leaves off-sheet mass at",
        f"   {sub_e['garble']*100:.0f}%. Fixing which *unused axis* the target",
        "   carries cannot fix the fact that the target is not a caption.",
        f"3. **KL is not the fix on its own.** `kl_on_midpoint` — semantic KL",
        "   onto the *midpoint's* policy — is just as garbled",
        f"   ({kl_mid['garble']*100:.0f}% off-sheet, `cos(d+, a) = "
        f"{kl_mid['pair_odd_cos']:+.3f}`). The load-bearing change is the",
        "   target point.",
        f"4. **Two recipes pass, and they share the target, not the loss.**",
        f"   `faithful_sub_e` (hidden MSE onto ê-cleaned real poles) and",
        f"   `v16_semantic_kl_sub_e` both land on-sheet",
        f"   ({faith_sub['on_sheet_kept']*100:.0f}% and"
        f" {kl_sub['on_sheet_kept']*100:.0f}% of the ceiling) with token-space",
        f"   leak at {faith_sub['leak_tok']:+.3f} / {kl_sub['leak_tok']:+.3f}.",
        f"   Both print *worse* pair-odd numbers than v9",
        f"   ({faith_sub['pair_odd_cos']:+.3f} / {kl_sub['pair_odd_cos']:+.3f}"
        f" and ±1 at {faith_sub['collapse']:+.3f} / {kl_sub['collapse']:+.3f}).",
        f"5. **The swing is a second Goodhart.** Every hidden-MSE row delivers",
        f"   about {v9['swing_kept']*100:.0f}% of the caption's own concept-token",
        "   swing while claiming a perfect hidden-space cosine, because the",
        "   mass it should have spent on the axis words went off-sheet instead.",
        "",
        "## The mechanism, stated precisely",
        "",
        "A tiny vocabulary and one linear readout from the hidden state are",
        "enough. No curved student is needed, and the curved one is worse: the",
        "softmax is the only nonlinearity the argument uses.",
        "",
        "It is tempting to describe the midpoint as \"a blend of the two poles,",
        "so its softmax prefers some third token\". That is not what happens",
        "here, and the distinction is the whole result. `h0 + a` is not between",
        "`h+` and `h−` — it is `h+` with `c` removed, i.e. *outside* the region",
        "any caption occupies, along the one direction all captions share. The",
        "off-sheet token `garble_hi` wins there because it is further along the",
        "concept direction û than `slam` (1.30 vs 1.00) and anti-loaded on the",
        "shared direction ŝ (−1.50). Real captions never reach that far along û",
        "alone, because they always carry `c`; strip `c` and the most extreme",
        "word on the axis wins even though nothing on the sheet would say it.",
        "",
        "That is why the `common = 0` row passes: with a perfectly odd pair",
        "there is no `c` to strip, the midpoint *is* the caption, and v9 is",
        "exactly right. The failure is not \"MSE\", not \"symmetry\", and not",
        "\"blending\" — it is deleting the shared component of two real",
        "captions and then aiming at where they aren't.",
        "",
        "## Where semantic KL earns its keep: the readout's null space",
        "",
        "Hidden MSE has to match the pole on every dim, including the ones no",
        "token reads. Semantic KL cannot see them and leaves them at zero.",
        "",
        "| recipe | on-sheet | garble | invisible pole content copied | leak_tok | KL |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in blob["null_space"]:
        lines.append(
            f"| `{row['name']}` | {row['on_sheet']:.3f} | {row['garble']:.3f} | "
            f"{row['null_kept']:.2f} | {row['leak_tok']:+.3f} | {row['kl_pole']:.3f} |"
        )
    lines += [
        "",
        "On this fixture the invisible block is 2 dims of 8 and copying it is",
        "harmless — it cannot change a token either way. Live the readout's row",
        "space is small next to a 3584-wide hidden state, so that is where most",
        "of the MSE budget goes: pole content that cannot change what the model",
        "says. That, not the sheet, is the argument for KL over MSE once both",
        "are aiming at a real caption.",
        "",
        "## The flip point, and where live sits",
        "",
        "Sweeping the common share with everything else fixed. A share of 0 is",
        "a perfectly odd pair: the midpoint *is* the caption and v9 is exactly",
        "right. Off-sheet behaviour arrives in two stages. The argmax leaves",
        f"its own pole's sheet at share **{blob['flip_point']:.2f}** (implied",
        f"logged cos {blob['flip_probe_cos']:+.3f}) — the student starts singing",
        "another row's lyric. Mass leaves the song's vocabulary altogether at",
        f"share **{blob['garble_flip']:.2f}**. The nine live axes sit at",
        f"{LIVE_BAND[0]:.2f} … {LIVE_BAND[1]:.2f}, every one of them past both.",
        "",
        "| common share | implied cos | hidden on-sheet | hidden garble | hidden cos± | hidden ±1 | KL on-sheet | KL garble | KL cos± | KL ±1 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in blob["common_sweep"]:
        lines.append(
            f"| {row['common']:.2f} | {row['probe_cos']:+.3f} | "
            f"{row['hidden_on_sheet']:.3f} | {row['hidden_garble']:.3f} | "
            f"{row['hidden_pair_odd_cos']:+.3f} | {row['hidden_collapse']:+.3f} | "
            f"{row['kl_on_sheet']:.3f} | {row['kl_garble']:.3f} | "
            f"{row['kl_pair_odd_cos']:+.3f} | {row['kl_collapse']:+.3f} |"
        )
    lines += [
        "",
        "`hidden`'s two logged columns are *constant* across the whole sweep",
        "while its on-sheet mass falls by a factor of four. No hidden-geometry",
        "metric can see this axis, which is why no existing cell caught it.",
        "",
        "![common sweep](lm-sheet-goodhart/common.png)",
        "",
        "## `--common_beta`: the dial that already exists",
        "",
        "`lm_hidden_targets` with `target_mode=symmetric` and `common_beta=1`",
        "returns exactly `(pos, neg)` — the identity `h0 ± a + c = h±`. So β",
        "interpolates the hidden-MSE target from the v9 midpoint to the raw",
        "caption, and the ladder is monotone in both directions at once:",
        "",
        "| β | on-sheet | garble | argmax | cos(d+, a) | cos(d+, d−) | leak_tok | KL |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in blob["beta_sweep"]:
        lines.append(
            f"| {row['common_beta']:.2f} | {row['on_sheet']:.3f} | "
            f"{row['garble']:.3f} | {row['argmax_on_sheet']:.2f} | "
            f"{row['pair_odd_cos']:+.3f} | {row['collapse']:+.3f} | "
            f"{row['leak_tok']:+.3f} | {row['kl_pole']:.3f} |"
        )
    lines += [
        "",
        "β does not fix leak — it is the same `a` plus a shared term, so the",
        "unused-attribute component rides along untouched (that is what",
        "`ê`-cleaning is for). It fixes the sheet, and it is reachable today",
        "with `--lm_target symmetric --common_beta 1`. It is *not* reachable",
        "with `--lm_target v9`, which hard-codes β = 0.",
        "",
        "![beta ladder](lm-sheet-goodhart/beta.png)",
        "",
        "![recipes](lm-sheet-goodhart/recipes.png)",
        "",
        "## Reproducing the live log columns",
        "",
        "The free-even student prints an exact 1.000 / −1.000 on a symmetric",
        "pair-odd teacher, because that teacher has no even part to learn.",
        f"Live prints {LIVE_V4_C_PLUS:+.2f} / {LIVE_V4_COLLAPSE:+.2f}"
        " (gender-lm-v4) because a real stack's ±1 replies are not exact",
        "mirrors — `highd`'s `bend`. Swapping the student for that caricature",
        "reproduces the live pair, and it lands *further* off the sheet, not",
        "closer:",
        "",
        "| cell | student | cos± | ±1 | on-sheet | kept | garble | argmax |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in blob["live_log"]:
        lines.append(
            f"| {row['cell']} | `{row['student']}` | {row['pair_odd_cos']:+.3f} | "
            f"{row['collapse']:+.3f} | {row['on_sheet']:.3f} | "
            f"{row['on_sheet_kept']:.3f} | {row['garble']:.3f} | "
            f"{row['argmax_on_sheet']:.2f} |"
        )
    lines += [
        "",
        "So the campaign's best log and the off-sheet failure are the same fit.",
        "",
        "## Verdict",
        "",
        "- Hidden MSE onto `t± = h0 ± a` **does** go off-sheet while looking",
        "  locked, on both the clean and the leaky cell, at the live common",
        "  share, and the flag for it is `misleading_lock`.",
        "- Semantic KL onto a real caption's policy **does** stay on-sheet,",
        "  and its logged pair-odd cos and ±1 collapse are worse. Under a",
        "  caption target the ±1 collapse approaches",
        "  `cos(pos−neu, neg−neu)` itself, so a v16 run logging collapse near",
        "  −1 has *not* recovered the common component.",
        "- The load-bearing change is the **target point**, not the loss:",
        "  hidden MSE onto ê-cleaned real poles passes every gate here too.",
        "  KL's own contribution is that it ignores the readout's null space,",
        "  which live is most of the hidden width.",
        "- `pair_odd_cos` is not a success metric on any of these rows. It is",
        "  maximized by the recipe that garbles most.",
        "",
        "## 2026-08-25 live exam: pair type and continuation",
        "",
        "The original energy-like cell above is deliberately retained, but it",
        "means **unused gender inside a clean same-song pair**. It is not a",
        "stand-in for energy-v4, whose poles are two different tracks",
        "(pop-punk/BPM 168 and ambient/BPM 52). On that divergent pair the",
        "genre/BPM direction is intended pole identity, not leak. Subtracting",
        "it turns `faithful_sub_e` into a blend teacher near the midpoint.",
        "",
        "The second missing measurement is continuation. Live `semantic_kl`",
        "supervises one next-token policy at `<|audio_start|>`. The close-pair",
        "cell gives the poles nearly the same first policy while putting their",
        "sequence plan in a hidden direction exposed by later frozen heads.",
        "Thus KL can be tiny while the fitted hidden remains far from the pole;",
        "a greedy rollout then enters an absorbing off-sheet token. Hidden MSE",
        "onto the same raw poles is the control that copies that plan.",
        "",
        "| row | pair | live predicted | target | one-token KL | hidden-far | off-caption teacher | continuation KL | rollout match | rollout garble | verdict |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in exam:
        live = row["live_run"] or "next control"
        expected = f"{live} ({row['expected_listen']})" if row["expected_listen"] else live
        lines.append(
            f"| `{row['name']}` | {row['pair_kind']} | {expected} | "
            f"`{row['pole_mode']} / {row['target']}` | {row['one_token_kl']:.4f} | "
            f"{row['hidden_far']:.3f} | {row['off_caption_teacher']:.3f} | "
            f"{row['teacher_forced_kl']:.3f} | {row['rollout_match']:.2f} | "
            f"{row['rollout_garble']:.2f} | **{_fmt_bool(row['pass'])}** |"
        )
    lines += [
        "",
        "This reproduces the exam ordering: energy-v16 fails because ê removal",
        "removes the divergent tracks' identity; energy-v18 passes on the same",
        "pair with raw faithful poles; gender-v16 is explicitly flagged",
        "`KL-small / hidden-far / rollout-garble`. The result does not establish",
        "that either mechanism is the unique cause in Music 3—the cell verifies",
        "which missing measurements are sufficient to predict all three listens.",
        "",
        "The untrained control that clears the close-pair continuation gate is:",
        "",
        "```text",
        "gender: --lm_target faithful --pole_mode hidden",
        "rank 8 / alpha 8 / lr 5e-4 / 800 steps / seed 7 / --no-early_stop",
        "endreg 1.0 / hold 0",
        "```",
        "",
        "## What this cell still cannot see",
        "",
        "- **Real Qwen lyrics.** The vocabulary here is nine tokens and the",
        "  readout is one frozen linear map with a hand-written structure. It",
        "  says *that* a sheet exists and *that* deleting `c` leaves it; it",
        "  does not say which real tokens a Music 3 slider substitutes. The",
        f"  flip point at common share {blob['flip_point']:.2f} is a property of",
        "  this readout, not a measured live threshold — the falsifiable",
        "  prediction is the *ordering*, not the number.",
        "- **Real autoregressive dynamics.** The live-exam subcell adds four",
        "  frozen continuation heads and an absorbing greedy rollout, enough to",
        "  expose first-token KL's blind spot. It is not Qwen's full transition",
        "  stack, a hundred-frame generation, or the downstream flow transformer.",
        "- **Semantic-code geometry.** Live the readout is the semantic band of",
        "  `lm_head`, whose rows are not a hand-chosen basis. Whether real",
        "  semantic-code rows anti-align with the shared caption component the",
        "  way `garble_*` does here is exactly the measurement to run next,",
        "  and it needs no training: encode the v4 pole captions, build",
        "  `t± = h0 ± a`, and compare the two policies.",
        "- **Whether the live fix is the default.** `--pole_mode` and",
        "  `--lm_target faithful_sub_e` are wired; the default is still",
        "  `--lm_target v9` / `--pole_mode hidden`. See the note below.",
        "",
        "## The live path",
        "",
        "`--pole_mode semantic_kl` and `--lm_target faithful_sub_e` are now",
        "live flags. The default is still `--lm_target v9` and",
        "`--pole_mode hidden` (`lm_slider_loss`, hidden MSE). The v16 card is",
        "gender: `--lm_target faithful --pole_mode semantic_kl` (no leak_*);",
        "leaky: `--lm_target faithful_sub_e --pole_mode semantic_kl` (leftover",
        "ê from YAML, hold 0). That is the wiring the fixture asked for —",
        "ê-cleaned *real pole* hiddens, and the semantic band of `lm_head`",
        "via `lm_semantic_pole_loss` / `lm_semantic_kl` /",
        "`lm_next_token_logits` — not \"KL instead of MSE\" onto the midpoint.",
        "`pair_odd_sub_e` stays the midpoint-minus-ê teacher. Do not retune",
        "the pair-odd early-stop gates to KL; `--no-early_stop` is the",
        "train-card choice.",
        "",
        "The cheapest next measurement needs no training and no new code path:",
        "encode the v4 pole and neutral captions with",
        "`scripts/probe_lm_axis_signal.py`'s loader, build `t± = h0 ± a`, and",
        "compare `softmax` over the semantic band at `t+` against `h+`. If the",
        "live poles behave like this fixture, the two policies disagree about",
        "as much as the caption and the midpoint do here.",
        "",
        "## Related cells",
        "",
        "- [lm-v9-2d.md](lm-v9-2d.md) — the pair-odd teacher and hold-ê on the",
        "  orthonormal 2-D field, where a perfect lock is the success metric.",
        "- [lm-faithful-2d.md](lm-faithful-2d.md) — why raw-pole MSE (v6) was",
        "  abandoned: it leaks the unused axis. This cell says it was also the",
        "  only live target that was a caption.",
        "- [lm-highd-leftover.md](lm-highd-leftover.md) — `bend`, λ·D/2, and the",
        "  short-û-is-a-probe split. The student here borrows its `bend`.",
        "- [lm-live-cells.md](lm-live-cells.md) — the live gender / energy",
        "  policy tables these two fields are shaped after.",
        "",
        "## How to run",
        "",
        "```bash",
        "PYTHONPATH=. python analysis/slider2d/run_lm_sheet.py --out docs/lm-sheet-goodhart",
        "PYTHONPATH=. pytest tests/test_lm_sheet_goodhart.py tests/test_lm_live_exam.py -q",
        "```",
        "",
        f"Seed `{blob['seed']}`, `{blob['steps']}` Adam steps, "
        f"{blob['rows']} prompt rows, {blob['dim']} hidden dims, "
        f"{len(blob['vocab'])} tokens, nucleus `p = {blob['sheet_p']:g}`.",
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

    gender = [floatable(r) for r in gender_cell(steps=args.steps, seed=args.seed)]
    leaky = [floatable(r) for r in leaky_cell(steps=args.steps, seed=args.seed)]
    teachers = [floatable(r) for r in teacher_sheet_table()]
    sweep = common_sweep(steps=args.steps // 2, seed=args.seed)
    betas = beta_sweep(steps=args.steps // 2, seed=args.seed)
    nulls = [floatable(r) for r in null_space_table(steps=args.steps, seed=args.seed)]
    logs = [floatable(r) for r in live_log_table(steps=args.steps, seed=args.seed)]
    exam = [floatable(r) for r in live_exam_cell(steps=args.steps, seed=args.seed)]
    flip = flip_point(sweep)
    garble_flip = first_above(sweep, "hidden_garble", GARBLE_MAX)

    field = leaky_field()
    blob = {
        "steps": args.steps,
        "seed": args.seed,
        "dim": field.dim,
        "rows": field.rows,
        "vocab": list(field.readout().tokens),
        "sheet_p": field.sheet_p,
        "gain": field.gain,
        "gender_common": gender_like_field().common,
        "leaky_common": field.common,
        "live_band": list(LIVE_BAND),
        "flip_point": flip,
        "garble_flip": garble_flip,
        "flip_probe_cos": next(
            r["probe_cos"] for r in sweep if flip is not None and r["common"] == flip
        )
        if flip is not None
        else None,
        "gates": {
            "sheet_lock": SHEET_LOCK,
            "garble_max": GARBLE_MAX,
            "leak_lock": LEAK_LOCK,
            "swing_floor": SWING_FLOOR,
        },
        "live": {
            "probe_cos": LIVE_PROBE_COS,
            "collapse_warn": LIVE_COLLAPSE_WARN,
            "v4_c_plus": LIVE_V4_C_PLUS,
            "v4_collapse": LIVE_V4_COLLAPSE,
        },
        "live_probe": live_probe_table(),
        "gender_cell": gender,
        "leaky_cell": leaky,
        "teachers": teachers,
        "common_sweep": sweep,
        "beta_sweep": betas,
        "null_space": nulls,
        "live_log": logs,
        "live_exam": exam,
    }
    (out / "metrics.json").write_text(json.dumps(blob, indent=2) + "\n", encoding="utf-8")
    plot_common(sweep, out / "common.png")
    plot_beta(betas, out / "beta.png")
    plot_recipes(leaky, out / "recipes.png")
    write_report(blob, out.parent / "lm-sheet-goodhart.md")

    for row in leaky:
        print(
            f"  {row['name']:24s} cos={row['pair_odd_cos']:+.3f} "
            f"pm1={row['collapse']:+.3f} sheet={row['on_sheet_kept']:.3f} "
            f"garble={row['garble']:.3f} argmax={row['argmax_on_sheet']:.2f} "
            f"leak={row['leak_tok']:+.3f} swing={row['swing_kept']:.2f} "
            f"pass={row['pass']} misleading={row['misleading_lock']}"
        )
    print(f"flip point |c|/|a| = {flip}, live band {LIVE_BAND[0]:.2f}..{LIVE_BAND[1]:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
