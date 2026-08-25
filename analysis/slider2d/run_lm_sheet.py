#!/usr/bin/env python3
"""Score the v15 midpoint Goodhart and the v16 semantic-KL sheet fix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from analysis.slider2d.sheet import (
    STEPS,
    TOKEN_NAMES,
    cell_table,
    compact,
    energy_sheet,
    field_geometry,
    gender_sheet,
    policy_at,
    readout_weight,
)
from conceptmod.textsliders.slider_targets import lm_policy_logits


DEFAULT_OUT = _REPO / "docs" / "lm-lyric-garble"


def _by_name(rows: list[dict]) -> dict[str, dict]:
    return {r["name"]: r for r in rows}


def _fmt_bool(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def _sheet_cell(row: dict) -> str:
    return (
        f"| `{row['name']}` | {row['pole_mode']} | `{row['teacher']}` | "
        f"`{row['student']}` | {row['c_plus']:+.3f} | "
        f"{row['c_plus_pair_odd']:+.3f} | "
        f"{row['collapse']:+.3f} | {row['on_sheet_mass']:.2f} | "
        f"{row['ood_mass']:.2f} | {row['ood_rate']:.2f} | "
        f"{row['argmax_plus']} / {row['argmax_minus']} | "
        f"{'yes' if row['looks_locked'] else 'no'} | "
        f"{'yes' if row['goodhart'] else 'no'} | "
        f"**{_fmt_bool(row['pass'])}** |"
    )


CELL_HEADER = (
    "| cell | pole | teacher | student | teacher cos | pair-odd cos | ±1 | "
    "on-sheet | ood mass | ood rate | argmax ± | locked look | Goodhart | verdict |\n"
    "|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---|"
)


def plot_plane(path: Path) -> None:
    """Hidden plane colored by argmax. Midpoint is off-sheet; real poles are not."""
    field = gender_sheet()
    weight = readout_weight(2)
    xs = torch.linspace(-1.8, 1.8, 180)
    ys = torch.linspace(-0.4, 1.6, 140)
    xx, yy = torch.meshgrid(xs, ys, indexing="xy")
    hidden = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
    logits = lm_policy_logits(hidden, weight)
    argmax = torch.argmax(logits, dim=-1).reshape(xx.shape).numpy()
    colors = {
        0: "#1a5276",
        1: "#6c3483",
        2: "#922b21",
        3: "#b03a2e",
        4: "#1e8449",
        5: "#b9770e",
    }
    cmap_list = [colors[i] for i in range(6)]
    from matplotlib.colors import ListedColormap

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.pcolormesh(
        xs.numpy(),
        ys.numpy(),
        argmax,
        cmap=ListedColormap(cmap_list),
        vmin=0,
        vmax=5,
        shading="auto",
        alpha=0.88,
    )
    pos, neg, neu = field.poles()
    t_plus, t_minus = field.midpoints()
    marks = (
        (float(pos[0]), float(pos[1]), "h+", "#f4f6f7", "encode(pos)"),
        (float(neg[0]), float(neg[1]), "h−", "#f4f6f7", "encode(neg)"),
        (float(t_plus[0]), float(t_plus[1]), "t+", "#f9e79f", "h0+a midpoint"),
        (float(t_minus[0]), float(t_minus[1]), "t−", "#f9e79f", "h0−a midpoint"),
        (float(neu[0]), float(neu[1]), "h0", "#d5d8dc", "neutral"),
    )
    for x, y, text, face, _label in marks:
        ax.scatter([x], [y], s=46, c=face, edgecolors="#1c2833", zorder=3)
        ax.annotate(text, (x, y), textcoords="offset points", xytext=(6, 5), fontsize=8)
    ax.axhline(0.0, color="#1c2833", lw=0.5, ls=":")
    ax.axvline(0.0, color="#1c2833", lw=0.5, ls=":")
    ax.set_xlabel("odd  (pair-odd / synthetic midpoint)")
    ax.set_ylabel("even  (real captions only)")
    ax.set_title("argmax of a linear readout: midpoint is not a caption")
    handles = [
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=colors[i], markersize=8, label=TOKEN_NAMES[i])
        for i in range(5)
    ]
    ax.legend(handles=handles, loc="upper right", framealpha=0.92, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_compare(rows: list[dict], path: Path) -> None:
    want = [
        "gender_hidden_odd_even",
        "gender_kl_odd",
        "gender_kl_odd_even",
        "energy_hidden_sub_e",
        "energy_kl_sub_e",
    ]
    by = _by_name(rows)
    labels = [name.replace("_", "\n") for name in want]
    sheet = [by[n]["on_sheet_mass"] for n in want]
    cplus = [by[n]["c_plus_pair_odd"] for n in want]
    ood = [by[n]["ood_mass"] for n in want]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    x = range(len(want))
    w = 0.25
    ax.bar([i - w for i in x], sheet, width=w, color="#1e8449", label="on-sheet mass")
    ax.bar(list(x), cplus, width=w, color="#1a5276", label="pair-odd cos")
    ax.bar([i + w for i in x], ood, width=w, color="#922b21", label="ood mass")
    ax.axhline(0.90, color="#1a5276", ls=":", lw=0.8)
    ax.set_xticks(list(x), labels, fontsize=8)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("mass / cosine")
    ax.set_title("pair-odd cos looks locked under hidden; sheet mass does not")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_markdown(rows: list[dict], out: Path) -> str:
    by = _by_name(rows)
    g_h = by["gender_hidden_odd_even"]
    g_odd = by["gender_hidden_odd"]
    g_kl = by["gender_kl_odd_even"]
    g_kl_odd = by["gender_kl_odd"]
    g_faith = by["gender_hidden_faithful_odd_even"]
    e_raw = by["energy_hidden_pair_odd"]
    e_sub = by["energy_hidden_sub_e"]
    e_kl = by["energy_kl_sub_e"]
    e_kl_raw = by["energy_kl_pair_odd"]
    geo_g = field_geometry(gender_sheet())
    geo_e = field_geometry(energy_sheet())
    p_mid = policy_at(gender_sheet().midpoints()[0], gender_sheet())
    p_cap = policy_at(gender_sheet().poles()[0], gender_sheet())

    gender_rows = "\n".join(
        _sheet_cell(by[n])
        for n in (
            "gender_hidden_odd",
            "gender_hidden_odd_even",
            "gender_hidden_faithful_odd_even",
            "gender_kl_odd",
            "gender_kl_odd_even",
        )
    )
    energy_rows = "\n".join(
        _sheet_cell(by[n])
        for n in (
            "energy_hidden_pair_odd",
            "energy_hidden_sub_e",
            "energy_kl_pair_odd",
            "energy_kl_sub_e",
        )
    )

    md = f"""# Lyric-garble Goodhart: hidden midpoint vs semantic KL

v15 pole term is hidden MSE onto the pair-odd midpoint
`t± = h0 ± ½(h+ − h−)`. That point is **not a real caption**. Hitting
it is what made gender-v15 start singing words that are not on the
sheet, while pair-odd cos / collapse printed the locked look
(~0.96 / −0.95). Existing 2-D cells treat high pair-odd cos as
success. That is the Goodhart.

v16 `--pole_mode semantic_kl` fits the next-token policy of a real
caption hidden. Gender: `KL(encode(pos) || LoRA@+1)` and the minus
pole; no leak_*, hold_ê=0. Leaky axes (`pair_odd_sub_e`): same KL,
ê-cleaned real poles, still hold_ê=0. Pair-odd cos / collapse are
logged only and look worse. That is expected.

This field invents the smallest extra structure that can see
off-sheet singing: a tiny vocab and a linear readout. CPU only. No
Hub, no GPU, no Music 3 weights. `--lm_target v9` / hold math is
untouched. The committed trainer still applies hidden MSE — this
cell scores the two pole terms without changing the live default.

## Verdict

**Hidden MSE onto the midpoint goes off-sheet while looking locked.
Semantic KL stays on-sheet if the student can hold the shared even.**

Gender hidden (`odd_even`): pair-odd cos {g_h['c_plus_pair_odd']:+.3f},
collapse {g_h['collapse']:+.3f}, argmax `{g_h['argmax_plus']}` /
`{g_h['argmax_minus']}`, on-sheet mass {g_h['on_sheet_mass']:.2f}.
Gender KL (`odd_even`): pair-odd cos {g_kl['c_plus_pair_odd']:+.3f},
collapse {g_kl['collapse']:+.3f}, on-sheet mass
{g_kl['on_sheet_mass']:.2f}, argmax `{g_kl['argmax_plus']}` /
`{g_kl['argmax_minus']}`. Worse-looking lock, on the sheet.

A linear readout is enough to *create* the third token. Softmax of
the hidden-space blend is not a blend of the two caption policies:
midpoint argmax `{geo_g['mid_argmax']}`, caption argmax
`{geo_g['caption_argmax']}` (sheet mass {p_cap['on_sheet_mass']:.2f}
vs midpoint {p_mid['on_sheet_mass']:.2f}). A linear *student* that
is odd in the slider scale cannot *stay* on-sheet under KL — no
bias, argmax is constant along a ray, and the whole pair-odd ray is
off-sheet (`gender_kl_odd` still sings `{g_kl_odd['argmax_plus']}` /
`{g_kl_odd['argmax_minus']}`). The `odd_even` residual
(`s·w_odd + |s|·w_even`) is the curved student that can hold the
shared even. Hypothesis: linear readout enough to leave the sheet
under MSE — **kept**. Linear odd student enough to stay on-sheet
under KL — **discarded**.

Do not call v15 leak-free or lock-healthy because pair-odd cos is
{g_h['c_plus_pair_odd']:.2f}. Do not revert a later live default to
hidden because this cell exists.

## Field

```
dim 0      pair-odd           a = ½(h+ − h−); t± = h0 ± a live here
dim 1      shared even        real captions only; midpoint drops it
dim 2      leftover ê         unused mix / BPM (energy cells)

h+ = (odd, even, leftover)     encode(pos) — on-sheet argmax pos
h− = (−odd, even, −leftover)   encode(neg) — on-sheet argmax neg
t+ = (odd, 0, leftover)        synthetic midpoint — off-sheet
â  = (odd, 0, 0)               ê-cleaned midpoint — still off-sheet
h+_clean = (odd, even, 0)      ê-cleaned real pole — on-sheet

vocab  pos, neg, ood+, ood−, lyric, leak
sheet  pos / neg / lyric
off    ood± (midpoint garble) and leak (unused-attr singing)
```

Linear readout (no bias). Midpoint vs caption is an argmax flip, not
a large hidden cosine:
cos(h+, t+) = {geo_g['cos_caption_to_mid']:.3f} on gender,
{geo_e['cos_caption_to_mid']:.3f} on energy. The live 0.96 lock is
the student-to-teacher cosine after hidden MSE, not this number.

## Table

### Gender-like (no ê, hold 0)

{CELL_HEADER}
{gender_rows}

### Energy-like leftover + `pair_odd_sub_e` (hold 0)

{CELL_HEADER}
{energy_rows}

![argmax regions](lm-lyric-garble/sheet.png)

![pair-odd cos vs on-sheet mass](lm-lyric-garble/compare.png)

## What each row does

- `gender_hidden_odd` / `gender_hidden_odd_even`: v15. Student hits
  `t+`. Even capacity does not help — the teacher has no even, so
  `w_even → 0`. Pair-odd cos {g_odd['c_plus_pair_odd']:+.3f} /
  {g_h['c_plus_pair_odd']:+.3f}, collapse {g_h['collapse']:+.3f},
  argmax `{g_h['argmax_plus']}` / `{g_h['argmax_minus']}`. The
  early-stop look (`c+ ≥ 0.90`, collapse `≤ −0.85`) fires. Off-sheet.
- `gender_hidden_faithful_odd_even`: control. Hidden MSE onto the
  *real* captions stays on-sheet (argmax `{g_faith['argmax_plus']}` /
  `{g_faith['argmax_minus']}`). The garble is the midpoint, not
  hidden MSE in general.
- `gender_kl_odd`: v16 loss, live-like odd LoRA. Shrinks along the
  pair-odd ray (temperature down, argmax unchanged). Still
  `{g_kl_odd['argmax_plus']}` / `{g_kl_odd['argmax_minus']}`.
  Semantic KL does not save a student that cannot leave the ray.
- `gender_kl_odd_even`: v16. Lands on encode(pos)/encode(neg).
  Pair-odd cos {g_kl['c_plus_pair_odd']:+.3f}, collapse
  {g_kl['collapse']:+.3f} — worse than v15, on-sheet.
- `energy_hidden_pair_odd`: raw pair-odd midpoint keeps leftover.
  Argmax `{e_raw['argmax_plus']}` / `{e_raw['argmax_minus']}`,
  leftover leak {e_raw['leftover_leak']:+.3f}. Off-sheet. v12 / Hub
  / v15 with no ê are not leak-free on these poles.
- `energy_hidden_sub_e`: ê-cleaned midpoint. Leftover gone
  ({e_sub['leftover']:.3f}), still argmax `{e_sub['argmax_plus']}` /
  `{e_sub['argmax_minus']}`. Cleaning ê does not put the midpoint
  back on the sheet.
- `energy_kl_pair_odd`: KL onto raw encode(pos). On-sheet
  (`{e_kl_raw['argmax_plus']}` / `{e_kl_raw['argmax_minus']}`) and
  still carries leftover {e_kl_raw['leftover']:.3f} — the unused
  attr is in the real captions.
- `energy_kl_sub_e`: v16 leaky path. KL onto ê-cleaned real poles,
  hold 0. On-sheet (`{e_kl['argmax_plus']}` / `{e_kl['argmax_minus']}`),
  leftover {e_kl['leftover']:.3f}, pair-odd cos
  {e_kl['c_plus_pair_odd']:+.3f}.

## Hypothesis, proved or discarded

| claim | result |
|---|---|
| Tiny vocab + linear readout is enough to make the midpoint prefer a third token | **proved** (mid `{geo_g['mid_argmax']}` ≠ caption `{geo_g['caption_argmax']}`) |
| Hidden MSE onto `t±` moves the student off the sheet while pair-odd cos looks locked | **proved** (`goodhart` on both gender hidden rows) |
| Semantic KL onto encode(pos)/encode(neg) stays on-sheet | **proved** for `odd_even`; **discarded** for odd-linear |
| A linear odd student can leave the sheet under MSE | **proved** (it goes to the midpoint) |
| A linear odd student can stay on-sheet under KL | **discarded** (ray-invariant argmax) |
| `pair_odd_sub_e` hidden is on-sheet because ê is gone | **discarded** (cleaned midpoint still `{geo_e['cleaned_mid_argmax']}`) |

## Live trainer

`train_lm_slider_music3.py` on `main` @ 38aeeed still applies
`lm_slider_loss` (hidden MSE) inside `lm_train_loss`. There is no
`--pole_mode` flag. `lm_semantic_kl` / `lm_e_cleaned_captions` live
in `slider_targets.py` as the CPU extract this fixture scores. This
PR does not wire the flag and does not change the live default.
Wire it only when a later change *is* the trainer change; do not
revert a later `semantic_kl` default to hidden because pair-odd cos
looks worse.

`--early_cos 0.97 --early_collapse -0.95` would stop the hidden
gender rows as done. That gate is blind to the sheet.

## What this field cannot see

- **Real Qwen lyrics.** The vocab is six tokens. Off-sheet here is
  `ood±` / `leak`, not a sung line from the written lyric sheet.
- **AR sampling.** Policy is one next-token softmax of a linear
  head. No teacher-forced composition, no `<|audio_end|>`, no
  endreg / planreg interaction.
- **Whether live LoRA's even reply is large enough.** Gender-v14
  collapse −0.95 implies bend ≈ 0.16. This readout flips argmax at
  even = 0.75, so 0.16 would still be off-sheet. A real Qwen head
  might flip with a smaller even — only a live probe can measure
  that.
- **Hidden width / λ·D/2 / ê wording.** That is
  [lm-highd-leftover.md](lm-highd-leftover.md).

## How to run

```bash
PYTHONPATH=. python analysis/slider2d/run_lm_sheet.py --out docs/lm-lyric-garble
PYTHONPATH=. pytest tests/test_lm_lyric_garble.py -q
```

CPU only. No Hub, no GPU, no Music 3 weights.

Seed `0`, `{STEPS}` Adam steps, lr 0.08.
"""
    return md


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    rows = cell_table(steps=args.steps, seed=args.seed)
    blob = {
        "rows": [compact(r) for r in rows],
        "gender_geometry": field_geometry(gender_sheet()),
        "energy_geometry": field_geometry(energy_sheet()),
    }
    (out / "metrics.json").write_text(json.dumps(blob, indent=2) + "\n")
    plot_plane(out / "sheet.png")
    plot_compare(rows, out / "compare.png")
    doc = Path(str(out) + ".md") if out.name != "lm-lyric-garble.md" else out
    if out.name == "lm-lyric-garble":
        doc = out.parent / "lm-lyric-garble.md"
    doc.write_text(write_markdown(rows, out))
    print(f"wrote {doc}")
    for row in rows:
        print(
            f"{row['name']:32s} tcos {row['c_plus']:+.3f} "
            f"odd {row['c_plus_pair_odd']:+.3f} "
            f"col {row['collapse']:+.3f} sheet {row['on_sheet_mass']:.2f} "
            f"argmax {row['argmax_plus']}/{row['argmax_minus']} "
            f"goodhart {row['goodhart']}"
        )


if __name__ == "__main__":
    main()
