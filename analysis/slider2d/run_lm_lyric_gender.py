#!/usr/bin/env python3
"""2D scoreboard: +1 lyric survival vs prefix gender, on both pair types."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from analysis.slider2d.lyric_gender import (
    ATTEND,
    AUDIO_CHANNEL,
    BASELINES,
    CAPTION_LEN,
    CAPTION_SHARE,
    CELL_IS,
    COMMON_STREAM,
    FIXTURE_ONLY,
    GENDER_MOVE_MIN,
    LYRIC_GENDER_CELLS,
    LYRIC_GENDER_RECIPES,
    LYRIC_LEN,
    PICK,
    lyric_gender_rank,
    lyric_gender_table,
    lyric_gender_verdict,
    sensitivity,
    sensitivity_verdict,
)
from analysis.slider2d.lyric_recall import LYRIC_RECALL_MIN
from analysis.slider2d.plus_exam import PLUS_COVER_MIN, PLUS_OFF_MAX
from analysis.slider2d.plus_neu_exam import PLUS_NEU_HOLD_MIN

DEFAULT_OUT = _REPO / "docs" / "lm-lyric-gender"

COLORS = {
    "faithful_plus_neu": "#c0392b",
    "faithful_plus_neu_prefix": "#2471a3",
    "faithful_plus_neu_prefix@0.25": "#5dade2",
    "faithful_plus_neu_orth": "#b9770e",
    "concept_prefix_teacher": "#7d3c98",
    "faithful_plus_neu_lyric": "#1e8449",
}


def _f(value, spec: str = ".3f", empty: str = "n/a") -> str:
    if value is None:
        return empty
    return format(float(value), spec)


def _cell(text: str) -> str:
    """One entry per prompt row, joined for a markdown cell.

    ` | ` is also the column separator, so rows join with ` / `; when
    every prompt row says the same thing the cell says it once.
    """
    parts = [part.strip() for part in str(text).split(" | ")]
    if parts and all(part == parts[0] for part in parts):
        return parts[0]
    return " / ".join(parts)


def _hit(flag) -> str:
    if flag is None:
        return "n/a"
    return "**HIT**" if flag else "MISS"


def _panel(ax, rows: list[dict], *, ykey: str, ylabel: str, ymin: float, title: str) -> None:
    want = Rectangle(
        (LYRIC_RECALL_MIN, ymin),
        1.0 - LYRIC_RECALL_MIN,
        1.0 - ymin,
        facecolor="#d5f5e3",
        edgecolor="#1e8449",
        lw=1.0,
        alpha=0.55,
        zorder=1,
    )
    ax.add_patch(want)
    # Recipes land on top of each other in the want-box, so draw them as
    # concentric rings largest-first: an overlap stays countable instead
    # of hiding rows behind the last one plotted.
    order = [c["name"] for c in LYRIC_GENDER_RECIPES]
    plotted = [r for r in rows if r[ykey] is not None]
    plotted.sort(key=lambda r: order.index(r["name"]))
    for index, row in enumerate(reversed(plotted)):
        color = COLORS.get(row["name"], "#7f8c8d")
        live = row["name"] not in FIXTURE_ONLY
        ax.scatter(
            [row["lyric_recall"]],
            [row[ykey]],
            s=90 + 90 * index,
            marker="o" if live else "D",
            facecolors=color if row["name"] == PICK else "none",
            edgecolors=color,
            linewidths=2.2,
            alpha=1.0 if live else 0.9,
            zorder=3 + index,
        )
    ax.axvline(LYRIC_RECALL_MIN, color="#7f8c8d", ls=":", lw=0.9)
    ax.axhline(ymin, color="#7f8c8d", ls=":", lw=0.9)
    ax.set_xlabel(f"lyric_recall @ +1  (gate {LYRIC_RECALL_MIN:g})")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9.5)
    ax.set_xlim(-0.06, 1.18)
    ax.set_ylim(-0.06, 1.18)
    ax.grid(alpha=0.25)


def plot_board(table: dict[str, list[dict]], path: Path) -> None:
    """The 2D scoreboard: one panel per pair type, both want-boxes drawn."""
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.6))
    _panel(
        axes[0],
        table["divergent"],
        ykey="cover",
        ylabel=f"cover of the + state  (gate {PLUS_COVER_MIN:g})",
        ymin=PLUS_COVER_MIN,
        title=(
            "grit-like / divergent — want lyric_recall AND cover\n"
            "(neu_hold also gated; it is 1.000 for every row here)"
        ),
    )
    _panel(
        axes[1],
        table["close"],
        ykey="gender_move",
        ylabel=f"gender_move in the prefix  (gate {GENDER_MOVE_MIN:g})",
        ymin=GENDER_MOVE_MIN,
        title=(
            "gender-like / close — want lyric_recall AND gender_move\n"
            "(Vocal Details at +1 may differ from neu; woman allowed)"
        ),
    )
    handles = [
        plt.Line2D(
            [],
            [],
            marker="D" if name in FIXTURE_ONLY else "o",
            ls="",
            markerfacecolor=color if name == PICK else "none",
            markeredgecolor=color,
            markeredgewidth=2.0,
            markersize=9,
            label=f"{name}{'  (fixture-only)' if name in FIXTURE_ONLY else ''}",
        )
        for name, color in COLORS.items()
    ]
    for ax in axes:
        ax.legend(handles=handles, fontsize=7.0, loc="lower left", framealpha=0.95)
    fig.suptitle(
        "UNI lyric/gender board — one recipe has to take both cells. "
        "Not exam_score, not leak_frac, not c+/p%, not the compiled bipolar board.",
        fontsize=10.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_sensitivity(rows: list[dict], path: Path) -> None:
    """Is the pick's win a property of the failure, or of one constant?

    Every setting lands on the same point in a scatter, so this is a
    strip: one row per knob value, the pick's two scored numbers as
    bars, the gate as a line, and the settings with no live failure in
    them greyed out because they get no vote.
    """
    height = 0.30 + 0.26 * len(rows)
    fig, ax = plt.subplots(figsize=(9.4, height))
    labels: list[str] = []
    for index, row in enumerate(rows):
        y = len(rows) - 1 - index
        split = bool(row["baselines_split"])
        grit_c = "#1e8449" if split else "#cfd8d3"
        gen_c = "#2471a3" if split else "#d6dee5"
        ax.barh(y + 0.19, row["pick_grit_lyric_recall"], height=0.34, color=grit_c)
        ax.barh(y - 0.19, row["pick_gender_move"], height=0.34, color=gen_c)
        star = " ★" if row["default"] else ""
        note = "" if split else "   (no live failure here)"
        labels.append(f"{row['knob']}={row['value']:g}{star}{note}")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(list(reversed(labels)), fontsize=7.2)
    ax.axvline(
        LYRIC_RECALL_MIN,
        color="#c0392b",
        ls="--",
        lw=1.2,
        label=f"gate {LYRIC_RECALL_MIN:g}",
    )
    ax.set_xlim(0.0, 1.06)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel("score  [0, 1]")
    handles = [
        plt.Line2D([], [], marker="s", ls="", color="#1e8449", label="grit lyric_recall @ +1"),
        plt.Line2D([], [], marker="s", ls="", color="#2471a3", label="gender_move in the prefix"),
        plt.Line2D([], [], color="#c0392b", ls="--", label=f"gate {LYRIC_RECALL_MIN:g}"),
    ]
    ax.legend(handles=handles, fontsize=7.4, loc="lower left", framealpha=0.95)
    ax.set_title(
        f"`{PICK}` re-scored at every fixture knob value\n"
        "coloured = the two baselines still split the way the live runs did\n"
        "grey = no live failure in that setting, so it gets no vote"
        "   ·   ★ = published default",
        fontsize=9.0,
    )
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_markdown(
    table: dict[str, list[dict]],
    verdict: dict,
    ranked: list[dict],
    sens: list[dict],
    sens_verdict: dict,
    path: Path,
    *,
    steps: int,
    seed: int,
) -> None:
    grit = {r["name"]: r for r in table["divergent"]}
    close = {r["name"]: r for r in table["close"]}
    pick_grit, pick_close = grit[PICK], close[PICK]
    uni_grit, uni_close = grit[BASELINES[0]], close[BASELINES[0]]
    pre_grit, pre_close = grit[BASELINES[1]], close[BASELINES[1]]
    one = "yes" if verdict["one_recipe_wins"] else "no"
    lines = [
        "# UNI lyric/gender board: one recipe, both failures",
        "",
        "Generated by `analysis/slider2d/run_lm_lyric_gender.py`. CPU only, no",
        "Hub, no GPU, no Music 3 weights, no Music 3 train. Does not change the",
        "live trainer default (`--lm_target v9` / `--pole_mode hidden`). This is",
        "a **separate scale**. It is not folded into the compiled bipolar board,",
        "and nothing here ranks on `exam_score`, `leak_frac`, `c+` or `p%`.",
        "",
        f"**Does one opt-in recipe take both pair types? `{one}` — "
        f"`--lm_target {PICK}`.**",
        "",
        "## 1. Diagnosis: two failures, one cause, opposite clamps",
        "",
        "Music 3 LM sliders, UNI (`faithful_plus_neu`): the student is",
        "`encode(neu tokens) + LoRA`, the teacher is `encode(pos tokens)` at the",
        "last real token, and the last real token is `<|audio_start|>`. The loss",
        "is a single point. The LoRA is not.",
        "",
        "A LoRA on attention fires at **every** position, and the continue token",
        "reads the prefix, so the last-token gradient arrives at every prefix",
        "token too. Two consequences, and they pull opposite ways:",
        "",
        "1. **Grit / distortion / joy (divergent).** `h+` is far from `h0`, so",
        "   the rewrite the LoRA has to produce is large, and the share of it",
        "   that lands on the lyric positions is large in absolute terms. The",
        "   written line stops being the written line: +1 sings the caption's",
        "   concept words. `+ REF` (pos caption, slider off) still sings the",
        "   line, so this is the LoRA's prefix rewrite, not the caption.",
        "2. **Gender (close).** `h+ − h0` is small and on-manifold, so nothing",
        "   shreds — but the concept *is* a prefix span. The + caption says",
        "   `One female lead singer. A woman is singing…`; the neutral says",
        "   `One lead singer`. Listen is neutral caption + LoRA, so the woman",
        "   can only exist if the LoRA rewrites that ungendered caption span.",
        "",
        "`faithful_plus_neu_prefix` holds the whole `encode(neu)` prefix. That",
        "fixes (1) by forbidding prefix rewrites — which is exactly what (2)",
        "needs. It is not a weight that is too high; it is a hold whose support",
        "covers the concept. Lowering the weight slides between the two failures",
        "instead of splitting them, and the board below shows that directly.",
        "",
        "The support is the variable, not the strength. The lyric span and the",
        "caption span are different token ranges of the same prompt, and only",
        "one of them is the thing we are protecting.",
        "",
        "## 2. The recipe",
        "",
        f"`--lm_target {PICK}` (opt-in; default stays `v9`).",
        "",
        "- last hidden @ +1 → raw `h+`. Unchanged from `faithful_plus_neu`.",
        "- last hidden @ scale 0 → `h0`. Unchanged.",
        "- **hidden at every position in `<|lyrics_start|> … <|lyrics_end|>`",
        "  @ +1 → the same positions of `encode(neu)`.** New, and the only",
        "  new term.",
        "- the caption span, including Vocal Details, is **not** held.",
        "- no minus teacher, no leftover-gate, no pair-odd, no `h0 ± a`.",
        "",
        "The span is resolved from the tokenizer (`lm_lyric_span_mask`), the",
        "written lyric tokens plus the closing marker whose hidden summarizes",
        "them, and it **fails closed**: a tokenizer that cannot name the markers,",
        "or a row without a well-formed span, raises rather than quietly",
        "degrading into `faithful_plus_neu` under a different flag name.",
        "",
        "The term is `lm_masked_hidden_mse`, a per-element mean, so weight 1.0",
        "puts it on the same scale as the last-token MSE. Worth stating plainly:",
        "the older whole-prefix hold uses `_masked_hidden_mse`, which sums over",
        "hidden width per position, so its effective weight is about",
        "`hidden_size` times the last-token MSE. That is a second, independent",
        "reason live prefix-hold froze the prefix hard — and it is left alone",
        "here, because it is the published #48 baseline and this board is",
        "supposed to compare against what actually ran. The 2D board normalizes",
        "both holds the same way, so the win below is not a weight artifact.",
        "",
        "```bash",
        "CUDA_VISIBLE_DEVICES=N python conceptmod/textsliders/train_lm_slider_music3.py \\",
        "  --name gender-lm-uni-lyric \\",
        "  --prompts_file conceptmod/textsliders/data/prompts-gender-v4.yaml \\",
        f"  --lm_target {PICK} --pole_mode hidden \\",
        "  --rank 8 --alpha 8 --lr 5e-4 --steps 800 --seed 7 \\",
        "  --no-early_stop --endreg_weight 1.0 --device 0",
        "```",
        "",
        "No v4 rewrite. Infer and listen with the yaml **neutral** caption +",
        "LoRA, not the + caption.",
        "",
        "## 3. Why the others lose",
        "",
        "| candidate | grit | gender | why it loses |",
        "|---|---|---|---|",
        "| `faithful_plus_neu` | lyric_recall "
        f"{_f(uni_grit['lyric_recall'])} {_hit(uni_grit['lyric_hit'])} | gender_move "
        f"{_f(uni_close['gender_move'])} {_hit(uni_close['gender_hit'])} | "
        "the failure itself: no hold anywhere, so the prefix rewrite is free |",
        "| `faithful_plus_neu_prefix` | lyric_recall "
        f"{_f(pre_grit['lyric_recall'])} {_hit(pre_grit['lyric_hit'])} | gender_move "
        f"{_f(pre_close['gender_move'])} {_hit(pre_close['gender_hit'])} | "
        "hold covers the caption span, so it pins the concept away |",
        "| `faithful_plus_neu_prefix@0.25` | lyric_recall "
        f"{_f(grit['faithful_plus_neu_prefix@0.25']['lyric_recall'])} "
        f"{_hit(grit['faithful_plus_neu_prefix@0.25']['lyric_hit'])} | gender_move "
        f"{_f(close['faithful_plus_neu_prefix@0.25']['gender_move'])} "
        f"{_hit(close['faithful_plus_neu_prefix@0.25']['gender_hit'])} | "
        "the same clamp, softer — a slide along one axis, not a split |",
        "| `faithful_plus_neu_orth` | cover "
        f"{_f(grit['faithful_plus_neu_orth']['cover'])} vs gate {PLUS_COVER_MIN:g} | gender_move "
        f"{_f(close['faithful_plus_neu_orth']['gender_move'])} "
        f"{_hit(close['faithful_plus_neu_orth']['gender_hit'])} | "
        "projects the *target*, not the LoRA: the prefix rewrite is untouched "
        "and the + state is no longer `h+`, so cover collapses |",
        "| `concept_prefix_teacher` | lyric_recall "
        f"{_f(grit['concept_prefix_teacher']['lyric_recall'])} "
        f"{_hit(grit['concept_prefix_teacher']['lyric_hit'])} | gender_move "
        f"{_f(close['concept_prefix_teacher']['gender_move'])} "
        f"{_hit(close['concept_prefix_teacher']['gender_hit'])} | "
        "wins on the fixture and cannot be built live — see below |",
        "",
        "**`lm_project_last_delta_off_lyric`.** It moves where the last-token",
        "teacher points, orthogonal to the lyric-hidden span. But the shred does",
        "not come from the *direction* of the last-token delta; it comes from",
        "the LoRA firing on lyric activations on its way to producing that",
        "delta. So the lyrics are saved for the wrong reason — the target is no",
        "longer `h+`, and grit cover falls to "
        f"{_f(grit['faithful_plus_neu_orth']['cover'])} against a "
        f"{PLUS_COVER_MIN:g} gate. This is the hypothesis this pass discards.",
        "",
        "**Concept-prefix teacher + lyric hold.** Teach the caption span toward",
        "`encode(pos)`'s caption span and hold the lyric span to `encode(neu)`.",
        "On this fixture it does take both cells (gender_move "
        f"{_f(close['concept_prefix_teacher']['gender_move'])}), and it is still",
        "not the pick, for one concrete reason: it needs a position-by-position",
        "correspondence between the two caption spans. The fixture has one",
        "because both spans are the same length by construction. The live yaml",
        "does not — `One female lead singer. A woman is singing…` and `One lead",
        "singer` are different token counts, so there is no position *i* to MSE",
        "against position *i*. A pooled variant is buildable, but it costs a",
        "second sequence teacher and a pooling choice, and this board says the",
        "extra teacher buys nothing: freeing the caption span already gets",
        f"gender_move {_f(pick_close['gender_move'])}, above the "
        f"{_f(close['concept_prefix_teacher']['gender_move'])} the explicit",
        "teacher reaches. It is on the board, marked fixture-only, rather than",
        "left unmentioned.",
        "",
        "## 4. The board",
        "",
        f"{steps} Adam steps, seed {seed}. The fit starts from a zero init and",
        "the cell geometry is seed-free, so these numbers are deterministic;",
        "the seed is recorded for form, not because it moves anything.",
        "",
        "Scored on **grit-like / divergent**: `lyric_recall` @ +1, `cover`,",
        f"`neu_hold` (gates {LYRIC_RECALL_MIN:g} / {PLUS_COVER_MIN:g} / "
        f"{PLUS_NEU_HOLD_MIN:g}).",
        "Scored on **gender-like / close**: `lyric_recall` @ +1 and",
        f"`gender_move` (gates {LYRIC_RECALL_MIN:g} / {GENDER_MOVE_MIN:g}).",
        "`cover` and `neu_hold` are logged on the close cell — the caption span",
        "is *supposed* to move there.",
        "",
        "`gender_move` is the share of the + caption's readable Vocal Details",
        "gender margin (`logit(female) − logit(male)` under the frozen readout)",
        "that the student put back into the **neutral** prefix at +1. 1.0 means",
        "the neutral prefix at +1 reads as female as the + caption does; 0.0",
        "means it still reads ungendered.",
        "",
    ]
    for cell in LYRIC_GENDER_CELLS:
        rows = sorted(table[cell], key=lambda r: (0 if r["hit"] else 1, -r["lyric_recall"]))
        lines += [
            f"### `{cell}` — {CELL_IS[cell]}",
            "",
            "| recipe | live flag | lyric_recall @ +1 | gender_move | cover | neu_hold "
            "| off-caption | hit |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
        for row in rows:
            lines.append(
                "| `{name}` | {live} | {rec} | {gm} | {cov} | {hold} | {off} | {hit} |".format(
                    name=row["name"],
                    live="yes" if row["live_flag"] else "fixture-only",
                    rec=_f(row["lyric_recall"]),
                    gm=_f(row["gender_move"]),
                    cov=_f(row["cover"]),
                    hold=_f(row["neu_hold"]),
                    off=_f(row["off_caption"]),
                    hit=_hit(row["hit"]),
                )
            )
        lines.append("")
    lines += [
        "![the 2D board on both pair types](lm-lyric-gender/lyric-gender-board.png)",
        "",
        "### Combined rank — both cells or nothing",
        "",
        "| rank | recipe | takes both | grit lyric_recall | grit cover "
        "| gender lyric_recall | gender_move | what it is |",
        "|---:|---|---|---:|---:|---:|---:|---|",
    ]
    for row in ranked:
        lines.append(
            "| {rank} | `{name}` | {both} | {gr} | {gc} | {cr} | {gm} | {is_} |".format(
                rank=row["rank"],
                name=row["name"],
                both="**yes**" if row["both"] else "—",
                gr=_f(row["grit_lyric_recall"]),
                gc=_f(row["grit_cover"]),
                cr=_f(row["gender_lyric_recall"]),
                gm=_f(row["gender_move"]),
                is_=row["is"],
            )
        )
    lines += [
        "",
        "### What the prefix actually sings",
        "",
        "The sung line is the greedy next token at each **lyric** position, read",
        "against the yaml lyric sheet — the one existing column #48 found that",
        "flags grit shred and keeps gender. Vocal Details is the gender word the",
        "caption position reads, `—` when it reads neither.",
        "",
        "| cell | recipe | sung line @ +1 | Vocal Details @ +1 | Vocal Details, + REF |",
        "|---|---|---|---|---|",
    ]
    for cell in LYRIC_GENDER_CELLS:
        for row in table[cell]:
            lines.append(
                "| `{cell}` | `{name}` | `{sings}` | {plus} | {ref} |".format(
                    cell=cell,
                    name=row["name"],
                    sings=_cell(row["sings_lyric"]),
                    plus=_cell(row["reads_vocal_plus"]),
                    ref=_cell(row["reads_vocal_ref_plus"]),
                )
            )
    lines += [
        "",
        "The `+ REF` column is the pos caption with the slider off. It sings the",
        f"line (`lyric_recall {_f(uni_grit['lyric_recall_ref_plus'])}` on grit)",
        "and it reads female on the close cell, which is the point: the caption",
        "is not what breaks the lyrics, and the gender is genuinely in the",
        "prefix rather than at the continue token.",
        "",
        "## 5. Is this a knife-edge?",
        "",
        "The fixture has knobs, so the board is re-run at every value of every",
        "one of them, and the sweep is published rather than tuned against.",
        "A setting where the two baselines stop splitting the way the live runs",
        "did is a setting with no live failure in it, so it gets no vote.",
        "",
        f"- settings swept: **{sens_verdict['settings']}**",
        "- settings that still reproduce the live split "
        f"(`faithful_plus_neu` shreds grit and moves gender, "
        f"`faithful_plus_neu_prefix` the reverse): "
        f"**{sens_verdict['settings_reproducing_the_live_split']}**",
        f"- of those, `{PICK}` takes both cells in "
        f"**{sens_verdict['pick_wins_both_where_split_reproduces']}**",
        f"- counterexamples: **{sens_verdict['counterexamples'] or 'none'}**",
        "",
        "| knob | value | live split reproduces | pick takes both | pick grit "
        "lyric_recall | pick gender_move |",
        "|---|---:|---|---|---:|---:|",
    ]
    for row in sens:
        lines.append(
            "| `{knob}`{star} | {value:g} | {split} | {wins} | {rec} | {gm} |".format(
                knob=row["knob"],
                star=" (default)" if row["default"] else "",
                value=row["value"],
                split="yes" if row["baselines_split"] else "no live failure here",
                wins="**yes**" if row["pick_wins_both"] else "no",
                rec=_f(row["pick_grit_lyric_recall"]),
                gm=_f(row["pick_gender_move"]),
            )
        )
    lines += [
        "",
        "![the pick re-scored at every knob value](lm-lyric-gender/lyric-gender-sensitivity.png)",
        "",
        "The settings that drop out of the split are the ones that give",
        "`<|audio_start|>` a large private channel or a weak read of the prefix.",
        "In that regime the continue token can be moved without touching the",
        "prompt at all — so nothing shreds, nothing needs a hold, and there is",
        "no problem to solve. The live evidence says we are not in that regime:",
        "UNI shreds, and holding the prefix costs last-token accuracy.",
        "",
        "## 6. The fixture",
        "",
        "Each row is a sequence, assembled the way `_assemble` assembles one:",
        "",
        "```",
        f"[ caption span (Vocal Details) x{CAPTION_LEN} ][ lyric span x{LYRIC_LEN} ][ <|audio_start|> ]",
        "```",
        "",
        "The `<|audio_start|>` base hidden is `h0` itself, so `cover`,",
        "`neu_hold` and off-caption read exactly the vectors the plus+neu exam",
        "reads. The student is a **shared linear map** on each position's own",
        "activation — `δ(σ, x) = (σ·W_odd + |σ|·W_even + W_zero) x` — not a free",
        "vector per position. That matters: whether a hold on one span is cheap",
        "has to *fall out* of whether that span's activations are separable, not",
        "be assumed.",
        "",
        "Three modelling choices carry the result, all of them named constants:",
        "",
        f"- `COMMON_STREAM = {COMMON_STREAM:g}` — the residual-stream component",
        "  every position shares. Real LM hiddens at different positions are far",
        "  from orthogonal, and that shared channel is what lets one LoRA weight",
        "  rewrite every position at once. Without it the map could specialize",
        "  per span for free, no last-token loss would ever touch the lyrics,",
        "  and the fixture could not reproduce the shred that started this.",
        f"- `AUDIO_CHANNEL = {AUDIO_CHANNEL:g}` — the continue token's own input",
        "  channel. Small, because the `<|audio_start|>` hidden is mostly a read",
        "  of the prompt. This is the knob the sweep is most sensitive to, and",
        "  it is sensitive in the honest direction: make it big and the live",
        "  failure disappears along with the need for a fix.",
        f"- `CAPTION_SHARE = {CAPTION_SHARE:g}` — how much of `h+ − h0` the +",
        "  caption states in its Vocal Details span rather than at the continue",
        "  token. This is what makes `gender_move` well-defined at all.",
        "",
        f"`ATTEND = {ATTEND:g}` is how much of the prefix rewrite the continue",
        "token re-reads; that coupling is why a last-token-only loss puts mass",
        "on the lyric span in the first place.",
        "",
        "The close cell is `close_field(unused=-0.55)`: gender-v4 with the moved",
        "Vocal Details attribute placed on the axis `male` / `female` actually",
        "read, negative because `female` reads `−ĝ`. `close_field` alone puts",
        "all of a close pair's motion in `delivery`, a zero column, so nothing",
        "could score whether the woman arrived. `shared` is still solved from",
        "the logged gender-v4 pair cos.",
        "",
        "## 7. Limits",
        "",
        "- There is still no 2D metric for slight +1 lyric *smear*. This board",
        "  scores whether the written line survives as the argmax, not whether",
        "  it is delivered cleanly. Listen and `scripts/blindspot_whisper.py`",
        "  `lyric_recall` remain the real exam.",
        "- `gender_move` is a readout margin in a fixture, not an F0 estimate.",
        "  It says the neutral prefix reads female at +1; it does not promise a",
        "  live median F0.",
        "- No Music 3 train, no GPU, no Hub, no weights on this branch. The live",
        "  flag is wired and unit-tested; it has not been run on real audio.",
        "",
        "## Related cells (not this scale)",
        "",
        "- [lm-lyric-recall.md](lm-lyric-recall.md) — which existing metric flags "
        "grit shred and keeps gender (#48).",
        "- [lm-plus-neu-exam.md](lm-plus-neu-exam.md) — last-hidden cover / neu_hold / off-caption.",
        "- [lm-plus-exam.md](lm-plus-exam.md) — plus-only cover / off-caption.",
        "- [lm-pair-exam.md](lm-pair-exam.md) — bipolar continuation gates.",
        "- [lm-2d-scoreboard.md](lm-2d-scoreboard.md) — compiled bipolar board, "
        "not updated and not folded into.",
        "",
        "## How to run",
        "",
        "```bash",
        "PYTHONPATH=. python analysis/slider2d/run_lm_lyric_gender.py --out docs/lm-lyric-gender",
        "PYTHONPATH=. pytest tests/test_lm_lyric_gender.py tests/test_lm_lyric_recall.py \\",
        "  tests/test_lm_plus_neu_exam.py tests/test_lm_trainer_v9.py -q",
        "```",
        "",
        f"CPU only. No Hub, no GPU, no Music 3 weights. Seed `{seed}`, `{steps}` "
        "Adam steps.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--sensitivity_steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    table = lyric_gender_table(steps=args.steps, seed=args.seed)
    verdict = lyric_gender_verdict(table)
    ranked = lyric_gender_rank(table)
    sens = sensitivity(steps=args.sensitivity_steps, seed=args.seed)
    sens_v = sensitivity_verdict(sens)
    blob = {
        "scale": "uni-lyric-gender",
        "not_the_bipolar_board": True,
        "pick": PICK,
        "live_flag": f"--lm_target {PICK}",
        "scored": {
            "divergent": ["lyric_recall", "cover", "neu_hold"],
            "close": ["lyric_recall", "gender_move"],
        },
        "not_scored": ["exam_score", "leak_frac", "c_plus", "pperc", "pair_odd_cos"],
        "gates": {
            "lyric_recall_min": LYRIC_RECALL_MIN,
            "gender_move_min": GENDER_MOVE_MIN,
            "cover_min": PLUS_COVER_MIN,
            "neu_hold_min": PLUS_NEU_HOLD_MIN,
            "off_caption_max": PLUS_OFF_MAX,
        },
        "fixture": {
            "caption_len": CAPTION_LEN,
            "lyric_len": LYRIC_LEN,
            "attend": ATTEND,
            "common_stream": COMMON_STREAM,
            "audio_channel": AUDIO_CHANNEL,
            "caption_share": CAPTION_SHARE,
        },
        "recipes": [c["name"] for c in LYRIC_GENDER_RECIPES],
        "fixture_only": sorted(FIXTURE_ONLY),
        "cells": list(LYRIC_GENDER_CELLS),
        "steps": args.steps,
        "seed": args.seed,
        "verdict": verdict,
        "rank": ranked,
        "sensitivity": {"verdict": sens_v, "rows": sens},
        "rows": table,
    }
    (args.out / "metrics.json").write_text(
        json.dumps(blob, indent=2, default=str) + "\n", encoding="utf-8"
    )
    plot_board(table, args.out / "lyric-gender-board.png")
    plot_sensitivity(sens, args.out / "lyric-gender-sensitivity.png")
    write_markdown(
        table,
        verdict,
        ranked,
        sens,
        sens_v,
        _REPO / "docs" / "lm-lyric-gender.md",
        steps=args.steps,
        seed=args.seed,
    )
    print(
        f"wrote {args.out}  one_recipe_wins={verdict['one_recipe_wins']} "
        f"live_winners={verdict['live_winners']} robust={sens_v['robust']}"
    )


if __name__ == "__main__":
    main()
