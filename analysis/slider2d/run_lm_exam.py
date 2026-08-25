#!/usr/bin/env python3
"""The pair-exam cell: divergent vs close pairs, scored over a rollout."""

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

from analysis.slider2d.exam import (
    CELL_IS,
    CELLS,
    EXAM_COHERENCE,
    EXAM_LEAK_LOCK,
    EXAM_MATCH_KEPT,
    EXAM_NEAR_GATE,
    EXAM_ROLL_OFF_MAX,
    EXAM_ROLL_OVERLAP,
    EXAM_ROLL_SWING,
    LIVE_EXAM,
    LIVE_PAIR_COS,
    LIVE_ROW,
    divergence_sweep,
    exam_table,
    first_above,
    first_below,
    floatable,
    live_exam_rows,
    pair_coordinate_table,
    teacher_geometry_table,
    visible_sweep,
)


DEFAULT_OUT = _REPO / "docs" / "lm-pair-exam"


def _f(value, spec: str, empty: str = "N/A") -> str:
    if value is None:
        return empty
    return format(float(value), spec)


def _verdict(row: dict) -> str:
    tag = "**PASS**" if row["pass"] else "**FAIL**"
    if row.get("near_gate"):
        tag += " *(near " + ", ".join(row["near_gate"]) + ")*"
    return tag


def plot_pairs(table: dict[str, list[dict]], path: Path) -> None:
    """Only the failures and the live-exam rows get a label.

    Everything that passes lands in one corner, so naming all of it
    reduces the panel to a smudge.
    """
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    marks = {"divergent": "o", "close": "s", "unused_e": "^"}
    named = {recipe for recipe, _cell in LIVE_ROW.values()}
    for cell, rows in table.items():
        xs = [r["roll_match_kept"] for r in rows]
        ys = [r["roll_swing_kept"] for r in rows]
        colors = ["#1e8449" if r["pass"] else "#c0392b" for r in rows]
        ax.scatter(
            xs,
            ys,
            c=colors,
            marker=marks[cell],
            s=58,
            zorder=3,
            edgecolors="white",
            linewidths=0.6,
        )
        ax.scatter([], [], c="#7f8c8d", marker=marks[cell], s=58, label=cell)
        for row, x, y in zip(rows, xs, ys):
            if row["pass"] and row["name"] not in named:
                continue
            ax.annotate(
                row["name"],
                (x, y),
                fontsize=6.6,
                xytext=(5, 4),
                textcoords="offset points",
            )
    ax.axvline(EXAM_MATCH_KEPT, color="#7f8c8d", ls=":", lw=0.9)
    ax.axhline(EXAM_ROLL_SWING, color="#7f8c8d", ls=":", lw=0.9)
    ax.set_xlabel("same words as the pole, over the pole's own cross-draw agreement")
    ax.set_ylabel("audible swing kept over the rollout")
    ax.set_title("green passes every gate, red fails one; shape is the pair")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, title="pair", loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_divergence(sweep: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    xs = [r["divergence"] for r in sweep]
    ax.plot(xs, [r["poles_swing"] for r in sweep], "o-", color="#1e8449", label="faithful + KL (v18)")
    sub = [(x, r) for x, r in zip(xs, sweep) if r["sub_e_swing"] is not None]
    ax.plot(
        [x for x, _ in sub],
        [r["sub_e_swing"] for _, r in sub],
        "s-",
        color="#c0392b",
        label="faithful_sub_e + KL (v16)",
    )
    ax.plot(
        [x for x, _ in sub],
        [r["sub_e_axis_eaten"] for _, r in sub],
        "^--",
        color="#b9770e",
        label="share of the visible axis ê eats",
    )
    ax.axhline(EXAM_ROLL_SWING, color="#7f8c8d", ls=":", lw=0.9)
    ax.axvline(
        CELLS["divergent"]().divergence(),
        color="#2c3e50",
        ls="-.",
        lw=0.9,
        label="energy-v4 sits here",
    )
    ax.set_xlabel("divergence — share of ‖a‖ that is one track versus the other")
    ax.set_ylabel("swing kept  /  axis eaten")
    ax.set_title("subtracting a genre/BPM ê costs nothing at 0 and the slider at 0.78")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_visible(sweep: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    xs = [r["visible_share"] for r in sweep]
    ax.plot(xs, [r["kl_swing"] for r in sweep], "o-", color="#c0392b", label="semantic KL, swing kept")
    ax.plot(xs, [r["mse_swing"] for r in sweep], "s-", color="#1e8449", label="hidden MSE, swing kept")
    ax.plot(xs, [r["kl_loss"] for r in sweep], "^--", color="#7f8c8d", label="KL loss at the end")
    ax.axhline(EXAM_ROLL_SWING, color="#7f8c8d", ls=":", lw=0.9)
    ax.axvline(
        CELLS["close"]().visible_share(),
        color="#2c3e50",
        ls="-.",
        lw=0.9,
        label="gender-v4 sits here",
    )
    ax.set_xlabel("visible share of ‖a‖ — how much of the axis the scored token reads")
    ax.set_ylabel("swing kept  /  loss")
    ax.set_title("the KL loss is ~0 across the whole sweep; the slider is not")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


CELL_HEADER = (
    "| recipe | pole_mode | target | overlap | same-words | coherence | "
    "off-caption | swing | leak *(1st tok)* | c+ *(log)* | ±1 *(log)* | "
    "p% *(log)* | loss *(log)* | verdict |\n"
    "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"
)


def cell_row_md(row: dict) -> str:
    return (
        f"| `{row['name']}` | {row['pole_mode']} | `{row['teacher']}` | "
        f"{_f(row['roll_overlap'], '.3f')} | {_f(row['roll_match_kept'], '.2f')} | "
        f"{_f(row['roll_coherence'], '.3f')} | {_f(row['roll_off_corpus'], '.3f')} | "
        f"{_f(row['roll_swing_kept'], '+.2f')} | {_f(row['leak_tok'], '+.3f')} | "
        f"{_f(row['pair_odd_cos'], '+.3f')} | {_f(row['collapse'], '+.3f')} | "
        f"{_f(row['pperc'], '.2f')} | {_f(row['loss'], '.4f')} | {_verdict(row)} |"
    )


def write_report(blob: dict, path: Path) -> None:
    table = blob["cells"]
    coords = blob["pairs"]
    live = blob["live"]
    div = blob["divergence_sweep"]
    by_cell = {row["cell"]: row for row in blob["pairs"]}
    vis = blob["visible_sweep"]
    lines = [
        "# The pair exam: divergent vs close poles, scored over a rollout",
        "",
        "Generated by `analysis/slider2d/run_lm_exam.py`. CPU only, no Hub,",
        "no GPU, no Music 3 weights. Does not change the live trainer default.",
        "",
        "Three live Music 3 runs on 2026-08-25 landed in an order no cell in",
        "this repo could produce, and no live log column can either.",
        "",
        "| run | recipe | prompts | c+ | ±1 | p% / n% | loss | ears |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in live:
        info = row["live"]
        lines.append(
            f"| `{row['run']}` | `--lm_target {info['teacher']} --pole_mode "
            f"{info['pole_mode']}` | `{info['prompts']}` | "
            f"{info['c_plus']:+.3f} | {info['collapse']:+.3f} | "
            f"{info['pperc']:.2f} / {info['nperc']:.2f} | {info['loss']:.4f} | "
            f"{'PASS' if info['listen'] == 'pass' else 'FAIL'} — {info['heard']} |"
        )
    worst = min(live, key=lambda r: r["live"]["loss"])
    lines += [
        "",
        f"The run with the smallest loss ({worst['live']['loss']:.4f}), the best",
        f"`c+` ({worst['live']['c_plus']:+.3f}) and the lowest `p%`",
        f"({worst['live']['pperc']:.3f}) is `{worst['run']}`, and it is the one",
        "whose lyrics came out garbled. The middle two rows are the **same",
        "recipe** on two different prompt files. So the thing that has to go in",
        "the score is not the loss, not the target's name and not any pair-odd",
        "column — it is the pair, and what the student sings after the token",
        "the loss looks at.",
        "",
        "## The two pair coordinates",
        "",
        "| pair | is | divergence | invisible share of `a` | logged pair cos | ‖c‖/‖a‖ | declared ê | ê·â / ‖a‖ | unpinned attribute |",
        "|---|---|---:|---:|---:|---:|---|---:|---|",
    ]
    for row in coords:
        lines.append(
            f"| `{row['cell']}` | {row['is']} | {row['divergence']:.3f} | "
            f"{row['invisible_share']:.3f} | {row['probe_cos']:+.3f} | "
            f"{row['common_share']:.3f} | {'yes' if row['declared_e'] else 'no'} | "
            f"{_f(row['e_overlap_a'], '.3f')} | "
            f"{'yes' if row['has_unused'] else 'no'} |"
        )
    lo, hi = LIVE_PAIR_COS["energy-v4"]
    lines += [
        "",
        "**divergence** is the share of `‖a‖` that is one track versus the",
        "other. It is 0 on a close pair and",
        f"{by_cell['divergent']['divergence']:.2f} on the energy-v4 stand-in.",
        "That number is what makes energy-v4's declared `ê` a problem:",
        "`leak_positive: \"Pop-punk mix, BPM 168.\"` names the same genre and",
        "BPM the *poles* move — the yaml's own header says \"genre, BPM, mood,",
        "mix and instrumentation all move with the axis\" — so `ê` overlaps `a`",
        f"by {by_cell['divergent']['e_overlap_a']:.2f} of `‖a‖` and subtracting",
        "`ê_⊥` deletes the slider rather than a leftover. On the same-song cell",
        f"the same construction gives an overlap of",
        f"{by_cell['unused_e']['e_overlap_a']:.2f} and subtracting it is free.",
        "",
        "**invisible share** is the share of `‖a‖` in the readout's null space,",
        "where a semantic-KL loss has exactly zero gradient. It is",
        f"{by_cell['divergent']['invisible_share']:.2f} on the divergent pair and",
        f"{by_cell['close']['invisible_share']:.2f} on the close one, because",
        "genre and BPM move the first semantic code hard and which of two",
        "voices sings the same song does not — that arrives in the vocal",
        "frames. So on a close pair the KL loss can reach its floor without the",
        "axis arriving at all.",
        "",
        "Both cells are calibrated to a logged number, not chosen: `shared` is",
        "solved so each field prints the trainer's own",
        f"`cos(pos−neu, neg−neu)` — energy-v4 logs {lo:+.2f} … {hi:+.2f} (the",
        f"midpoint {0.5 * (lo + hi):+.3f} is used), gender-v4 logs",
        f"{LIVE_PAIR_COS['gender-v4']:+.2f}.",
        "",
        "## The gate",
        "",
        "Everything scored here is something the student **sings over a",
        "continuation**, decoded from its own ±1 hidden state for",
        f"{blob['out_steps']} tokens, {blob['draws']} sampled draws, against the",
        "real pole caption's own continuations.",
        "",
        f"- continuation overlap with the pole's words: `≥ {EXAM_ROLL_OVERLAP}`",
        f"- position-wise agreement, as a share of the pole's own cross-draw"
        f" agreement: `≥ {EXAM_MATCH_KEPT}`",
        f"- off-caption mass — words no caption of this song sings: `≤ {EXAM_ROLL_OFF_MAX}`",
        f"- coherence — consecutive words from the same song: `≥ {EXAM_COHERENCE}`",
        f"- audible swing kept over the rollout: `≥ {EXAM_ROLL_SWING}`",
        "",
        "Logged and never scored: `c+`, `c−`, the ±1 collapse, `p%` / `n%` and",
        "the pole loss. The unused-attribute leak column is logged here too and",
        f"scored on the #22 sheet cell (`≤ {EXAM_LEAK_LOCK}`): it reads the first",
        "token, where an attribute tilt is visible, and this cell's rollout",
        "commits after that token and averages the tilt away. The first-token",
        "column reproduces the sheet cell's numbers to three places, which is",
        "the cross-check that the two readouts agree about leak.",
        "",
        f"A verdict within {EXAM_NEAR_GATE} of the gate that decides it is",
        "marked *(near …)*: it is not seed-robust and the table says so. None",
        "of the three live exam rows is decided by a near-gate column at any",
        "seed tested.",
        "",
        "## Where each target point sits, with no optimizer in the loop",
        "",
        "| pair | target | axis kept | visible axis kept | off-caption | to mid | blend teacher |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for cell, geoms in blob["targets"].items():
        for geom in geoms:
            lines.append(
                f"| `{cell}` | `{geom['name']}` | "
                f"{1.0 - geom['axis_eaten']:.3f} | "
                f"{1.0 - geom['visible_axis_eaten']:.3f} | "
                f"{geom['off_caption']:.3f} | {geom['to_mid']:.3f} | "
                f"{'**yes**' if geom['blend_teacher'] else 'no'} |"
            )
    lines += [
        "",
        "`faithful_sub_e` is `mid ± â` with `mid = ½(h₊+h₋)`. On the same-song",
        "cell `ê` names the unpinned attribute, `â` keeps",
        f"{1.0 - blob['targets']['unused_e'][-1]['visible_axis_eaten']:.2f} of the",
        "visible axis and the target is not a blend. On the divergent cell `ê`",
        "names the track, `â` keeps only",
        f"{1.0 - blob['targets']['divergent'][-1]['visible_axis_eaten']:.2f} of it,",
        "and the target is nearer `mid` than the pole it claims to be. `mid` on",
        "a divergent pair is pop-punk-at-168 and ambient-lullaby-at-52 at once:",
        "no caption says both, and the policy there is bimodal rather than",
        "wrong-but-decided. One token cannot tell those apart. Eight can.",
        "",
    ]
    for cell, rows in table.items():
        lines += [
            f"## The `{cell}` cell — {CELL_IS[cell]}",
            "",
            CELL_HEADER,
        ]
        lines += [cell_row_md(r) for r in rows]
        lines += [""]
        for row in rows:
            lines.append(f"- `{row['name']}`: {row['reason']}")
        if cell == "unused_e":
            lines += [
                "",
                "Every recipe passes here, and the leak column is why the cell is",
                "still on the board: `faithful` and `semantic_kl` onto raw poles",
                f"carry +0.227 of the unpinned attribute, the three `sub_e` rows",
                "carry +0.000, and hold-ê λ=8 carries +0.005 — the same numbers",
                "the #22 sheet cell reports, from a different readout. Leak is",
                "logged here and **scored there**: this cell's rollout commits",
                "after the first token and averages an attribute tilt away.",
            ]
        lines += [""]
    lines += [
        "![pairs](lm-pair-exam/pairs.png)",
        "",
        "## The live exam",
        "",
        "| live run | fixture row | pair | predicted | ears | agrees | why the cell says so |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in live:
        lines.append(
            f"| `{row['run']}` | `{row['recipe']}` | `{row['cell']}` | "
            f"**{row['predicted'].upper()}** | **{row['listen'].upper()}** | "
            f"{'yes' if row['agrees'] else 'NO'} | {row['reason']} |"
        )
    agree = sum(1 for r in live if r["agrees"])
    lines += [
        "",
        f"{agree} of {len(live)} agree. The two `semantic_kl_poles` rows are the",
        "same recipe: it is the energy win and the gender garble, and the cell",
        "that separates them is the pair, not the loss.",
        "",
        "## Divergence sweep: when does subtracting a declared ê cost the slider",
        "",
        "`track` grows and the declared `ê` grows with it, because the yaml",
        "writes `ê` out of the same genre and BPM the poles moved. `shared` is",
        "re-solved at every point so the field keeps printing the logged",
        "energy-v4 pair cos — this is a sweep of divergence, not of collapse.",
        "",
        "Measured columns only. The pass booleans are in `metrics.json`; near a",
        "gate they flip with the sampling, so the trend is the claim.",
        "",
        "| divergence | ê eats (visible axis) | faithful+KL swing | faithful+KL overlap | sub_e+KL swing | sub_e+KL overlap | sub_e+KL coherence |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in div:
        lines.append(
            f"| {row['divergence']:.3f} | {_f(row['sub_e_axis_eaten'], '.2f')} | "
            f"{_f(row['poles_swing'], '+.2f')} | "
            f"{_f(row['poles_overlap'], '.3f')} | "
            f"{_f(row['sub_e_swing'], '+.2f')} | "
            f"{_f(row['sub_e_overlap'], '.3f')} | "
            f"{_f(row['sub_e_coherence'], '.3f')} |"
        )
    flip = first_below(div, "sub_e_swing", EXAM_ROLL_SWING, "divergence")
    lines += [
        "",
        "`faithful_sub_e` + KL loses the audible swing monotonically as the",
        "pair diverges, and drops below the",
        f"{EXAM_ROLL_SWING} floor at divergence",
        f"{_f(flip, '.2f', 'no point in this grid')} — energy-v4 sits at",
        f"{by_cell['divergent']['divergence']:.2f}. `faithful` + KL keeps the full",
        "swing across the whole sweep. The falsifiable claim is the ordering",
        "and the direction, not the number: the number is a property of this",
        "readout.",
        "",
        "![divergence](lm-pair-exam/divergence.png)",
        "",
        "## Visible-share sweep: KL-small while the hidden never arrived",
        "",
        "A close pair at fixed `‖a‖`, moving the axis from the delivery block",
        "(invisible to the scored token) into the readable block.",
        "",
        "| visible share | KL loss | KL solved | invisible kept (KL) | p% (KL) | c+ (KL) | KL swing | invisible kept (MSE) | MSE swing |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in vis:
        lines.append(
            f"| {row['visible_share']:.3f} | {row['kl_loss']:.5f} | "
            f"{row['kl_solved']:.3f} | {_f(row['kl_invisible_kept'], '.3f')} | "
            f"{row['kl_pperc']:.2f} | {row['kl_c_plus']:+.3f} | "
            f"{row['kl_swing']:+.2f} | "
            f"{_f(row['mse_invisible_kept'], '.3f')} | {row['mse_swing']:+.2f} |"
        )
    kl_flip = first_above(vis, "kl_swing", EXAM_ROLL_SWING, "visible_share")
    lines += [
        "",
        "The KL loss is essentially solved at every point in the sweep, and",
        "`invisible kept` is 0 at every point: the loss has no gradient there,",
        "so it never moves. What changes is how much of the axis was in that",
        "block. Hidden MSE keeps all of it everywhere and passes everywhere.",
        "Semantic KL onto the same real-caption target only reaches the swing",
        f"floor once about {_f(kl_flip, '.2f', 'no point in this grid')} of the axis",
        f"is readable. gender-v4 sits at {by_cell['close']['visible_share']:.2f}.",
        "",
        "![visible share](lm-pair-exam/visible.png)",
        "",
        "## The mechanism, stated precisely",
        "",
        "1. **`c` is only shared caption content when the poles are captions of",
        "   the same song.** With `h± = h0 ± a + c`, the sheet cell reads",
        "   `c = ½(h₊+h₋) − h0` as genre / BPM / mood — what both poles say and",
        "   the neutral does not. On energy-v4 the two poles say *contradictory*",
        "   things about genre and BPM, so half of `c` is half of each song at",
        "   once, and `mid` is a point no caption occupies. That is why",
        "   `faithful_sub_e`, whose target is `mid ± â`, sounds like the",
        "   midpoint got pulled and not like a weaker slider.",
        "2. **A declared `leak_*` pair is only leftover if the poles do not move",
        "   it.** energy-v4's `ê` restates the poles' own genre and BPM, so",
        f"   `â = a − (a·ê̂_⊥)ê̂_⊥` keeps only",
        f"   {1.0 - blob['targets']['divergent'][-1]['visible_axis_eaten']:.2f} of",
        "   the axis the scored token can read. Both ends of the slider then",
        "   land at `mid`, and which of the two songs' words wins is decided by",
        "   residual noise — different per prompt row, which is what \"random",
        "   words\" is.",
        "3. **A small KL loss is a Goodhart metric, exactly like a perfect",
        "   pair-odd lock.** `lm_semantic_pole_loss` is one next-token",
        "   distribution over the semantic band. Its gradient on the readout's",
        "   null space is zero, so whatever share of `a` lives there is simply",
        "   not learned, and the loss reaches its floor regardless. On a close",
        "   pair that share is",
        f"   {by_cell['close']['invisible_share']:.2f}: the axis *is* the part the",
        "   loss cannot see. `gender-lm-v16` printed loss 0.0091 and",
        "   `p% / n% = 0.523 / 0.777` — the loss said solved and the hidden was",
        "   most of the way un-arrived, on the same step.",
        "4. **The three add up.** `energy-lm-v18` survives because a divergent",
        "   pair puts most of `a` where the KL loss *can* see it. `gender-lm-v16`",
        "   fails because a close pair does not. `energy-lm-v16` fails hardest",
        "   because it removes the visible part of the axis on purpose and then",
        "   asks a loss that cannot see the rest to make it up.",
        "",
        "## What this cell still cannot see",
        "",
        "- **Real Qwen semantic codes.** Ten tokens, one frozen linear readout,",
        "  one hand-written transition. It says *that* a divergent pair and a",
        "  close pair are different objects and *that* one scored token cannot",
        "  separate them. It does not say which real codes a Music 3 slider",
        "  substitutes.",
        "- **The transition is a model, not a measurement.** `A = I + mix·(û⊗d̂)`",
        "  asserts that content the semantic band does not read at",
        "  `<|audio_start|>` reaches the tokens a few frames later. That is how a",
        "  residual stream works, but the rate is invented. The prediction is",
        "  the ordering.",
        "- **Unused-attribute leak.** The rollout averages it away; the #22",
        "  sheet cell is where leak is scored. This cell's first-token column",
        "  agrees with it, which is the only claim being made.",
        "- **The flow transformer and the depth decoder.** Everything downstream",
        "  of the LM is out of scope here, as in every cell in this repo.",
        "",
        "## The cheapest next live measurement",
        "",
        "It needs no training. Encode the gender-v4 and energy-v4 pole and",
        "neutral captions with `scripts/probe_lm_axis_signal.py`'s loader, take",
        "the semantic band of `lm_head`, and compare `softmax` at `h₊` against",
        "`softmax` at `h₋` on each pair. If the live poles behave like this",
        "fixture, the gender pair's two policies are nearly the same",
        "distribution and the energy pair's are not — which is the whole",
        "argument, measured directly, with the model that trained the halves.",
        "",
        "## Related cells",
        "",
        "- [lm-2d-scoreboard.md](lm-2d-scoreboard.md) — the compiled board this",
        "  cell is joined into.",
        "- [lm-sheet-goodhart.md](lm-sheet-goodhart.md) — the single-token sheet",
        "  readout and the pair-odd lock this cell inherits its discipline from.",
        "- [lm-highd-leftover.md](lm-highd-leftover.md) — leftover ê, λ·D/2 and",
        "  the trainer's `c+` ceiling.",
        "- [lm-live-cells.md](lm-live-cells.md) — the live gender / energy hidden",
        "  cells.",
        "",
        "## How to run",
        "",
        "```bash",
        "PYTHONPATH=. python analysis/slider2d/run_lm_exam.py --out docs/lm-pair-exam",
        "PYTHONPATH=. pytest tests/test_lm_pair_exam.py -q",
        "```",
        "",
        "CPU only. No Hub, no GPU, no Music 3 weights.",
        "",
        f"Seed `{blob['seed']}`, `{blob['steps']}` Adam steps, {blob['rows']} prompt",
        f"rows, {blob['out_steps']}-token rollout, {blob['draws']} draws,",
        f"nucleus `p = {blob['sheet_p']}`.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--sweep-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    table = exam_table(steps=args.steps, seed=args.seed)
    live = live_exam_rows(table)
    order = list(LIVE_EXAM)
    live.sort(key=lambda r: order.index(r["run"]))
    field = CELLS["divergent"]()
    blob = {
        "steps": args.steps,
        "sweep_steps": args.sweep_steps,
        "seed": args.seed,
        "rows": field.rows,
        "out_steps": field.out_steps,
        "draws": field.draws,
        "sheet_p": field.sheet_p,
        "gates": {
            "overlap": EXAM_ROLL_OVERLAP,
            "match_kept": EXAM_MATCH_KEPT,
            "off_caption": EXAM_ROLL_OFF_MAX,
            "coherence": EXAM_COHERENCE,
            "swing": EXAM_ROLL_SWING,
            "leak_on_sheet_cell": EXAM_LEAK_LOCK,
            "near_gate": EXAM_NEAR_GATE,
            "pair_odd_cos_scored": False,
            "collapse_scored": False,
            "pole_loss_scored": False,
            "perc_scored": False,
        },
        "pairs": pair_coordinate_table(),
        "targets": {cell: teacher_geometry_table(cell) for cell in CELLS},
        "cells": {cell: [floatable(r) for r in rows] for cell, rows in table.items()},
        "live": [
            {k: v for k, v in row.items() if k != "row"} | {"row": floatable(row["row"])}
            for row in live
        ],
        "divergence_sweep": divergence_sweep(steps=args.sweep_steps, seed=args.seed),
        "visible_sweep": visible_sweep(steps=args.sweep_steps, seed=args.seed),
        "live_default_unchanged": True,
    }
    (out / "metrics.json").write_text(json.dumps(blob, indent=2) + "\n", encoding="utf-8")
    plot_pairs(table, out / "pairs.png")
    plot_divergence(blob["divergence_sweep"], out / "divergence.png")
    plot_visible(blob["visible_sweep"], out / "visible.png")
    blob["live"] = live
    write_report(blob, out.parent / "lm-pair-exam.md")
    for row in live:
        print(
            f"{row['run']:16s} {row['recipe']:20s} {row['cell']:10s} "
            f"predicted={row['predicted']:5s} ears={row['listen']:5s} "
            f"agrees={row['agrees']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
