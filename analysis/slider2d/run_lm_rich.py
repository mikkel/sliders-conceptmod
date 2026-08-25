#!/usr/bin/env python3
"""Sweep richer poles vs leak on the existing CPU 2-D / 4-D fixture."""

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

from analysis.slider2d.faithful import floatable
from analysis.slider2d.rich import (
    ALIGN_GRID,
    HOLD_LAMBDAS,
    RichField,
    align_sweep,
    beats_v9_on_rich,
    energy_mismatch_table,
    field2d_baselines,
    misaligned_e_cell,
    onaxis_slider_sweep,
    partial_pin_table,
    slider_richness_sweep,
    teacher_variant_table,
    unused_richness_sweep,
)


DEFAULT_OUT = _REPO / "docs" / "lm-rich-2d"


def _by_name(rows: list[dict]) -> dict[str, dict]:
    return {r["name"]: r for r in rows}


def plot_teachers(rows: list[dict], path: Path) -> None:
    want = _by_name(rows)
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    ax.axhline(0, color="#dddddd", lw=0.6)
    ax.axvline(0, color="#dddddd", lw=0.6)
    styles = {
        "faithful": ("#922b21", "faithful (raw rich poles)"),
        "pair_odd": ("#b9770e", "pair-odd"),
        "v9": ("#1e8449", "v9 hold-ê λ=8"),
        "pair_odd_sub_e": ("#1a5276", "pair-odd − ê"),
        "project_short": ("#6c3483", "project short û"),
        "project_rich": ("#148f77", "project rich û"),
    }
    for name, (color, label) in styles.items():
        row = want[name]
        intended = (row["proj_slider"] ** 2 + row["proj_rich"] ** 2) ** 0.5
        unused = (row["proj_gender"] ** 2 + row["proj_bpm"] ** 2) ** 0.5
        ax.arrow(
            0, 0, intended, unused,
            color=color, width=0.015, length_includes_head=True, head_width=0.08, alpha=0.95,
        )
        ax.annotate(f"{label}\nrich={row['rich_kept']:.2f}", (intended + 0.03, unused + 0.04), fontsize=7, color=color)
    ax.set_xlabel("intended  (short û² + slider-rich²)½")
    ax.set_ylabel("unused  (gender² + BPM²)½")
    ax.set_title("Rich leaky poles: leftover unused vs intended")
    ax.set_aspect("equal")
    ax.set_xlim(-0.2, 2.0)
    ax.set_ylim(-0.1, 2.2)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_richness_sweeps(slider_rows: list[dict], unused_g: list[dict], unused_b: list[dict], onaxis: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.8))
    colors = {
        "pair_odd": "#b9770e",
        "v9": "#1e8449",
        "pair_odd_sub_e": "#1a5276",
        "project_short": "#6c3483",
        "project_rich": "#148f77",
        "pair_odd_sub_all": "#212f3d",
    }

    def _draw(ax, rows, xlabel, title):
        recipes = sorted({r["recipe"] for r in rows})
        for recipe in recipes:
            pts = [r for r in rows if r["recipe"] == recipe]
            pts = sorted(pts, key=lambda r: r["sweep_x"])
            ax.plot(
                [r["sweep_x"] for r in pts],
                [abs(r["leak_ratio"]) for r in pts],
                "o-",
                color=colors.get(recipe, "#555555"),
                label=recipe,
                ms=4,
            )
        ax.axhline(0.20, color="#888888", ls=":", lw=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("|leak| unused / slider")
        ax.set_title(title)
        ax.set_ylim(-0.05, 2.4)

    _draw(axes[0], slider_rows, "off-axis slider richness r", "Slider synonyms, unused fixed")
    _draw(axes[1], unused_g, "odd unused gender", "Unused gender, slider fixed")
    _draw(axes[2], unused_b, "odd unused BPM", "Unused BPM, slider+ê fixed")
    axes[2].legend(fontsize=6, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for recipe in ("pair_odd", "v9", "pair_odd_sub_e", "project_short"):
        pts = sorted([r for r in onaxis if r["recipe"] == recipe], key=lambda r: r["sweep_x"])
        if not pts:
            continue
        ax.plot(
            [r["sweep_x"] for r in pts],
            [abs(r["leak_ratio"]) for r in pts],
            "o-",
            color=colors.get(recipe, "#555555"),
            label=recipe,
            ms=4,
        )
    ax.axhline(0.20, color="#888888", ls=":", lw=0.8)
    ax.set_xlabel("on-axis slider scale (Field2D x)")
    ax.set_ylabel("|leak|")
    ax.set_title("On-axis synonyms (existing energetic×gender field)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path.with_name("onaxis.png"), dpi=140)
    plt.close(fig)


def plot_align(rows: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    colors = {
        "pair_odd": "#b9770e",
        "v9": "#1e8449",
        "project_short": "#6c3483",
        "project_rich": "#148f77",
    }
    plotted = [r for r in rows if r.get("recipe") in colors and r.get("sweep") == "align" and "mismatch" not in r["name"]]
    for recipe, color in colors.items():
        pts = sorted([r for r in plotted if r["recipe"] == recipe], key=lambda r: r["sweep_x"])
        if not pts:
            continue
        ax.plot([r["sweep_x"] for r in pts], [r["rich_kept"] for r in pts], "o-", color=color, label=f"{recipe} rich-kept", ms=4)
        ax.plot([r["sweep_x"] for r in pts], [abs(r["leak_ratio"]) for r in pts], "s--", color=color, alpha=0.55, label=f"{recipe} |leak|", ms=4)
    ax.axvline(0.20, color="#922b21", ls=":", lw=0.8, label="gender-v1 0.20")
    ax.axvline(0.48, color="#888888", ls=":", lw=0.6)
    ax.axvline(0.68, color="#888888", ls=":", lw=0.6)
    ax.axvline(0.95, color="#888888", ls=":", lw=0.6)
    ax.set_xlabel("requested |odd·û_short| / ||odd||")
    ax.set_ylabel("rich kept  /  |leak|")
    ax.set_title("Align sweep: short û vs rich û (r=0.8)")
    ax.set_ylim(-0.05, 1.25)
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _fmt_axis(row: dict, key: str = "leak") -> str:
    return f"**{row['axis'][key]}**"


def write_report(blob: dict, path: Path) -> None:
    teachers = blob["teachers"]
    by = {r["name"]: r for r in teachers}
    v9 = by["v9"]
    sub = by["pair_odd_sub_e"]
    sub_all = by["pair_odd_sub_all"]
    proj_s = by["project_short"]
    proj_r = by["project_rich"]
    faith = by["faithful"]
    odd = by["pair_odd"]
    baselines = {r["name"]: r for r in blob["field2d_baselines"]}
    energy = blob["energy_mismatch"]
    pins = blob["partial_pin"]
    mis = blob["misaligned_e"]
    slider_rows = blob["slider_richness"]
    onaxis = blob["onaxis_slider"]
    unused_g = blob["unused_gender"]
    unused_b = blob["unused_bpm"]
    aligns = blob["align"]
    seed1 = blob.get("seed1_teachers") or []

    def _slider_at(rich: float, recipe: str) -> dict:
        hits = [r for r in slider_rows if abs(r["sweep_x"] - rich) < 1e-6 and r["recipe"] == recipe]
        return hits[0]

    def _onaxis_at(scale: float, recipe: str) -> dict:
        hits = [r for r in onaxis if abs(r["sweep_x"] - scale) < 1e-6 and r["recipe"] == recipe]
        return hits[0]

    def _unused_at(rows: list[dict], x: float, recipe: str) -> dict:
        hits = [r for r in rows if abs(r["sweep_x"] - x) < 1e-6 and r["recipe"] == recipe]
        return hits[0]

    def _pin(label: str, recipe: str) -> dict:
        hits = [r for r in pins if r["pin"] == label and r["recipe"] == recipe]
        return hits[0]

    def _align(a: float, recipe: str) -> dict:
        hits = [r for r in aligns if abs(r.get("sweep_x", -1) - a) < 1e-6 and r["recipe"] == recipe]
        return hits[0]

    winners = [r for r in teachers if beats_v9_on_rich(r, v9) and r["name"] != "v9"]
    # Hard-subtract of oracle ê is the hard limit of hold-ê, not a new live recipe
    # unless it also survives a wrong ê and does not revive gender-v1 project.
    wire = False
    if winners:
        # Only wire if a winner is not just "subtract the oracle axis we already hold".
        names = {r["name"] for r in winners}
        interesting = names - {"pair_odd_sub_e", "faithful_sub_e", "pair_odd_sub_all", "v9_hold_all", "project_rich"}
        wire = bool(interesting)

    if abs(proj_s["rich_kept"]) < 0.25 and abs(sub["leak_gender"]) < 0.05 and sub["rich_kept"] >= 0.70:
        yes = (
            "**Yes, if the extra words lie on the intended axis — or we drop only unused ê.** "
            f"Off-axis slider adjectives are kept by pair-odd − ê "
            f"(rich kept {sub['rich_kept']:.2f}, gender leak {sub['leak_gender']:+.3f}) "
            f"and by current v9 hold-ê (rich kept {v9['rich_kept']:.2f}, leftover leak {v9['leak_ratio']:+.3f}). "
            f"Project onto short û zeros that richness (rich kept {proj_s['rich_kept']:.2f}) — "
            "the gender-v1 kill in slider-adjective clothing. "
            "Unused mix/BPM/gender words are not richness; they *are* leak. "
        )
    else:
        yes = (
            "**Mostly no for unused words, yes for slider synonyms on-axis.** "
            "See the tables. "
        )
    if wire:
        verdict = yes + "A new teacher beat v9 by enough to consider wiring; see winners below. Live default was not changed in this PR pending a Hub check."
    else:
        verdict = (
            yes
            + f"`project_rich` is leak {proj_r['leak_ratio']:+.3f} and rich-kept {proj_r['rich_kept']:.2f} "
            "here because û is the oracle intended span (hypothesis 4) — not a live default; "
            "short û at 0.20 still kills. "
            f"`pair_odd_sub_all` / `v9_hold_all` (leak {sub_all['leak_ratio']:+.3f} / "
            f"{by['v9_hold_all']['leak_ratio']:+.3f}) is the same recipe with every unused axis declared. "
            f"Current `--lm_target v9` leftover leak {v9['leak_ratio']:+.3f} on this cell is undeclared BPM "
            f"(gender leftover {v9['leak_gender']:+.3f}); richness stays {v9['rich_kept']:.2f}. "
            "Do not change the live default. ê is a YAML caption pair, not an oracle."
        )

    geo = RichField()
    lines = [
        "# Richer poles without hosing leak",
        "",
        "Same CPU fixture as [lm-faithful-2d.md](lm-faithful-2d.md),",
        "[lm-v9-2d.md](lm-v9-2d.md), [lm-live-cells.md](lm-live-cells.md).",
        "Question: can Music 3-style structured poles keep slider detail",
        "(mix/timbre/genre adjectives that belong to the slider) without",
        "unused mix / BPM / gender riding along inside `h±`?",
        "",
        "Faithful+attributes (leak 0, pole cos 1) *cleans* the poles by",
        "pinning unused gender. This cell is the opposite direction:",
        "keep or add slider-detail, drop only unused dimensions.",
        "",
        "CPU only. No Hub, no GPU, no Music 3 weights. Live `--lm_target v9`",
        "is unchanged.",
        "",
        "## Verdict",
        "",
        verdict,
        "",
        f"Default rich leaky poles: short û `{geo.slider:.2f}`, off-axis slider",
        f"richness `{geo.rich:.2f}`, odd gender `{geo.odd_gender:.2f}`",
        f"(even gender `{geo.even_gender:.2f}`), odd BPM `{geo.odd_bpm:.2f}`.",
        f"Short-û align {geo.align_short():.3f}; rich-û align {geo.align_rich():.3f}.",
        "",
        "## Already scored baselines (energetic×gender Field2D, 250 Adam, seed 0)",
        "",
        "Reused, not the only table. New subtract-ê rows are the hard v9.",
        "",
        "| method | leak | ±1 cos | pole cos |",
        "|---|---:|---:|---:|",
    ]
    for name in (
        "lm_faithful_raw",
        "lm_faithful_hold_l8",
        "lm_faithful_attrs",
        "lm_symmetric",
        "lm_v9",
        "lm_v9_project",
        "lm_faithful_sub_e",
        "lm_odd_sub_e",
    ):
        r = baselines[name]
        lines.append(
            f"| `{name}` | {r['leak_ratio']:+.3f} | {r['cos_plus_minus']:+.3f} | {r['pole_cos_plus']:.3f} |"
        )
    lines += [
        "",
        "## Teacher variants on the same rich leaky poles",
        "",
        "| method | slider | leak | ±1 | rich | slider cos | leak | ±1 cos | rich kept | gender leak | BPM leak |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    order = [
        "faithful",
        "pair_odd",
        "faithful_sub_e",
        "pair_odd_sub_e",
        "pair_odd_sub_all",
        "hold_e_l1",
        "hold_e_l4",
        "hold_e_l8",
        "hold_e_l16",
        "hold_e_l32",
        "v9",
        "v9_hold_all",
        "project_short",
        "project_rich",
        "project_odd",
    ]
    for name in order:
        r = by[name]
        lines.append(
            f"| `{name}` | {_fmt_axis(r, 'slider')} | {_fmt_axis(r, 'leak')} | {_fmt_axis(r, 'collapse')} | "
            f"{_fmt_axis(r, 'rich')} | {r['cos_intended']:.3f} | {r['leak_ratio']:+.3f} | "
            f"{r['cos_plus_minus']:+.3f} | {r['rich_kept']:.2f} | {r['leak_gender']:+.3f} | {r['leak_bpm']:+.3f} |"
        )
    if seed1:
        s1 = {r["name"]: r for r in seed1}
        lines += [
            "",
            f"Seed 1 (same 250 steps), v9 leak {s1['v9']['leak_ratio']:+.3f} rich {s1['v9']['rich_kept']:.2f}; "
            f"pair-odd − ê leak {s1['pair_odd_sub_e']['leak_ratio']:+.3f} rich {s1['pair_odd_sub_e']['rich_kept']:.2f}; "
            f"project short rich {s1['project_short']['rich_kept']:.2f}.",
        ]
    lines += [
        "",
        "![teacher residuals, slider vs unused](lm-rich-2d/teachers.png)",
        "",
        "## Slider richness at fixed unused",
        "",
        "Off-axis structured adjectives (dim 3). Unused gender/BPM held at the",
        "default. Slider synonyms that *lie on* û are the on-axis Field2D sweep.",
        "",
        "| r | recipe | leak | rich kept | cos intended | short align |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for rich in (0.0, 0.8, 2.0):
        for recipe in ("pair_odd", "v9", "pair_odd_sub_e", "project_short", "project_rich"):
            r = _slider_at(rich, recipe)
            lines.append(
                f"| {rich:.1f} | `{recipe}` | {r['leak_ratio']:+.3f} | {r['rich_kept']:.2f} | "
                f"{r['cos_intended']:.3f} | {r['align_short']:.3f} |"
            )
    r0 = _onaxis_at(1.0, "v9")
    r2 = _onaxis_at(2.0, "v9")
    r3 = _onaxis_at(3.0, "pair_odd")
    lines += [
        "",
        f"On-axis Field2D: extra energetic/calm scale 1 → 2 drops v9 leak",
        f"{r0['leak_ratio']:+.3f} → {r2['leak_ratio']:+.3f} (same unused, more slider).",
        f"Pair-odd at scale 3 is leak {r3['leak_ratio']:+.3f}. On-axis slider words are free.",
        "",
        "![off-axis slider / unused sweeps](lm-rich-2d/sweeps.png)",
        "",
        "![on-axis slider scale on Field2D](lm-rich-2d/onaxis.png)",
        "",
        "## Unused richness at fixed slider",
        "",
        "Extra gender (declared ê) vs extra BPM (undeclared unused). Slider",
        f"richness stays `{geo.rich:.2f}`.",
        "",
        "| unused | axis | pair-odd leak | v9 leak | −ê leak | −all leak | v9 rich |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for which, rows in (("gender", unused_g), ("bpm", unused_b)):
        for x in (0.0, 0.6, 1.5):
            po = _unused_at(rows, x, "pair_odd")
            v = _unused_at(rows, x, "v9")
            se = _unused_at(rows, x, "pair_odd_sub_e")
            sa = _unused_at(rows, x, "pair_odd_sub_all")
            lines.append(
                f"| {x:.1f} | {which} | {po['leak_ratio']:+.3f} | {v['leak_ratio']:+.3f} | "
                f"{se['leak_ratio']:+.3f} | {sa['leak_ratio']:+.3f} | {v['rich_kept']:.2f} |"
            )
    lines += [
        "",
        "v9 / −ê only see declared gender. Extra BPM is leftover leak unless",
        "it is also declared (hold-all / −all) or pinned.",
        "",
        "## Partial pin (leave slider adjectives rich)",
        "",
        "| pin | recipe | leak | rich kept | gender leak | BPM leak |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for pin in ("free", "pin_gender", "pin_bpm", "pin_both"):
        for recipe in ("faithful", "pair_odd", "v9"):
            r = _pin(pin, recipe)
            lines.append(
                f"| `{pin}` | `{recipe}` | {r['leak_ratio']:+.3f} | {r['rich_kept']:.2f} | "
                f"{r['leak_gender']:+.3f} | {r['leak_bpm']:+.3f} |"
            )
    lines += [
        "",
        "Pinning unused gender (attributes-style) or unused BPM zeros that",
        "axis in `h± − h0`. Slider richness stays. Pin-both + faithful is",
        "the data fix: clean rich poles, leak 0. Pin-gender alone leaves BPM.",
        "",
        "## Align sweep: rich û vs short û",
        "",
        "Already-odd poles. Requested short-û aligns 0.20 / 0.48 / 0.68 / 0.95.",
        f"Off-axis richness r={geo.rich:.2f} caps short align at",
        f"`s/sqrt(s²+r²) = {geo.slider / (geo.slider ** 2 + geo.rich ** 2) ** 0.5:.3f}`",
        "when unused is zero — that is why structured poles vs two-word û",
        "print middling 0.48–0.68 even before unused mix.",
        "",
        "| align | recipe | realized short | leak | rich kept | cos intended |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for a in ALIGN_GRID:
        for recipe in ("pair_odd", "v9", "project_short", "project_rich"):
            r = _align(a, recipe)
            lines.append(
                f"| {a:.2f} | `{recipe}` | {r['align_short']:.3f} | {r['leak_ratio']:+.3f} | "
                f"{r['rich_kept']:.2f} | {r['cos_intended']:.3f} |"
            )
    killed = next(r for r in aligns if r["name"] == "mismatch_rich_project_short")
    kept = next(r for r in aligns if r["name"] == "mismatch_rich_pair_odd")
    lines += [
        "",
        f"Clean rich pair + tilted short û at 0.20 (gender-v1): project-short",
        f"rich kept {killed['rich_kept']:.2f}, cos intended {killed['cos_intended']:.3f},",
        f"strength-like norm {killed['norm_plus']:.3f}. Pair-odd on the same",
        f"clean pair: leak {kept['leak_ratio']:+.3f}, rich kept {kept['rich_kept']:.2f}.",
        "Project onto short names still kills the singer / the extra adjectives.",
        "",
        "![align sweep](lm-rich-2d/align.png)",
        "",
        "## Energy-like + mismatch (do not overfit even∥ê)",
        "",
        "On energy-like, poles are already odd. Faithful ≡ pair-odd.",
        "Hard-subtract ê is the hard limit of current v9.",
        "Mismatch is a clean pair: ê is not in the poles; subtract is a no-op.",
        "Project short û still fails (gender-v1).",
        "",
        "| cell | method | leak | ±1 cos | pass |",
        "|---|---|---:|---:|---|",
        f"| energy-like | faithful λ=0 | {energy['energy_faithful']['leak_ratio']:+.3f} | {energy['energy_faithful']['cos_plus_minus']:+.3f} | {energy['energy_faithful']['pass']} |",
        f"| energy-like | pair-odd λ=0 | {energy['energy_odd']['leak_ratio']:+.3f} | {energy['energy_odd']['cos_plus_minus']:+.3f} | {energy['energy_odd']['pass']} |",
        f"| energy-like | v9 hold-ê λ=8 | {energy['energy_v9']['leak_ratio']:+.3f} | {energy['energy_v9']['cos_plus_minus']:+.3f} | {energy['energy_v9']['pass']} |",
        f"| energy-like | pair-odd − ê | {energy['energy_sub_e']['leak_ratio']:+.3f} | {energy['energy_sub_e']['cos_plus_minus']:+.3f} | {energy['energy_sub_e']['pass']} |",
        f"| mismatch | faithful | {energy['mismatch_faithful']['leak_ratio']:+.3f} | {energy['mismatch_faithful']['cos_plus_minus']:+.3f} | {energy['mismatch_faithful']['pass']} |",
        f"| mismatch | pair-odd | {energy['mismatch_pair_odd']['leak_ratio']:+.3f} | {energy['mismatch_pair_odd']['cos_plus_minus']:+.3f} | {energy['mismatch_pair_odd']['pass']} |",
        f"| mismatch | project short û | {energy['mismatch_project_short']['leak_ratio']:+.3f} | {energy['mismatch_project_short']['cos_plus_minus']:+.3f} | {energy['mismatch_project_short']['pass']} |",
        f"| mismatch | subtract junk | {energy['mismatch_sub_junk']['leak_ratio']:+.3f} | {energy['mismatch_sub_junk']['cos_plus_minus']:+.3f} | {energy['mismatch_sub_junk']['pass']} |",
        "",
        "## Why a wrong ê should not become the live default",
        "",
        "Hard-subtract of *oracle* ê looks like a win (gender leak 0, richness kept).",
        "Live ê is `leak_positive` / `leak_negative` captions, not this basis vector.",
        f"Tilt ê by mixing 0.25 of û: subtract slider_kept {mis['sub_wrong']['slider_kept']:.2f}",
        f"(true ê {mis['sub_true']['slider_kept']:.2f}); hold-ê λ=8 slider_kept",
        f"{mis['hold_wrong']['slider_kept']:.2f} (true {mis['hold_true']['slider_kept']:.2f}).",
        "Hold is the soft version. Wiring hard-subtract would make a bad ê caption",
        "eat slider — the gender-v1 failure mode with a different name.",
        "",
        "## Geometry",
        "",
        "```",
        "h± = h0 ± s û_short ± r ê_rich + even_unused + odd_unused",
        "leak            = unused sitting inside h±",
        "slider richness = r ê_rich  (⊥ short û, still intended)",
        "on-axis richness= extra s   (energetic/calm synonyms on û)",
        "faithful        t± = h±                         # copies unused",
        "pair-odd        t± = h0 ± (h+−h−)/2             # drops even unused",
        "pair-odd − ê    t± = h0 ± (a − (a·ê)ê)          # drops declared unused, keeps r",
        "v9 hold-ê       L += λ ((h±−h0)·ê)²             # leftover ê / (1+λ)",
        "project short û â = (a·û)û                      # drops r and unused",
        "project rich û  â = (a·û_rich)û_rich            # keeps span{û, r}, drops unused",
        "project pair-odd â = a                          # identity cheat; unused stays",
        "```",
        "",
        "Recipe that works on rich leaky captions without rewriting them:",
        "**pair-odd + hold declared ê (current v9)**, or pin unused axes in the",
        "captions and leave slider adjectives. Declare every unused axis you",
        "care about (`leak_positive` / `leak_negative`); undeclared BPM is leak.",
        "Do not project onto two-word û.",
        "",
        "## What this field cannot see",
        "",
        "- Real Music 3 hidden geometry and noisy ê from actual leak captions.",
        "- AR endreg / planreg / semantic-KL.",
        "- Multi-row yaml averaging that is not parallel.",
        "",
        "## How to run",
        "",
        "```bash",
        "PYTHONPATH=. python analysis/slider2d/run_lm_rich.py --out docs/lm-rich-2d",
        "PYTHONPATH=. pytest tests/test_lm_rich_2d.py tests/test_lm_faithful_2d.py -q",
        "```",
        "",
        "CPU only. No Hub, no GPU, no Music 3 weights.",
        "",
        f"Seed `{blob['seed']}`, `{blob['steps']}` Adam steps"
        + (f"; seed 1 rerun of the teacher table." if seed1 else "."),
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--extra-seed", type=int, default=1)
    args = parser.parse_args(argv)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    teachers = teacher_variant_table(steps=args.steps, seed=args.seed)
    seed1 = teacher_variant_table(steps=args.steps, seed=args.extra_seed)
    slider_rows = slider_richness_sweep(steps=args.steps, seed=args.seed)
    onaxis = onaxis_slider_sweep(steps=args.steps, seed=args.seed)
    unused_g = unused_richness_sweep(which="gender", steps=args.steps, seed=args.seed)
    unused_b = unused_richness_sweep(which="bpm", steps=args.steps, seed=args.seed)
    pins = partial_pin_table(steps=args.steps, seed=args.seed)
    aligns = align_sweep(steps=args.steps, seed=args.seed)
    energy = energy_mismatch_table(steps=args.steps, seed=args.seed)
    baselines = field2d_baselines(steps=args.steps, seed=args.seed)
    mis = misaligned_e_cell(steps=args.steps, seed=args.seed)

    blob = {
        "steps": args.steps,
        "seed": args.seed,
        "extra_seed": args.extra_seed,
        "teachers": [floatable(r) for r in teachers],
        "seed1_teachers": [floatable(r) for r in seed1],
        "slider_richness": [floatable(r) for r in slider_rows],
        "onaxis_slider": [floatable(r) for r in onaxis],
        "unused_gender": [floatable(r) for r in unused_g],
        "unused_bpm": [floatable(r) for r in unused_b],
        "partial_pin": [floatable(r) for r in pins],
        "align": [floatable(r) for r in aligns],
        "energy_mismatch": {k: floatable(v) for k, v in energy.items()},
        "field2d_baselines": [floatable(r) for r in baselines],
        "misaligned_e": {k: floatable(v) for k, v in mis.items()},
    }
    (out / "metrics.json").write_text(json.dumps(blob, indent=2) + "\n", encoding="utf-8")
    plot_teachers(teachers, out / "teachers.png")
    plot_richness_sweeps(slider_rows, unused_g, unused_b, onaxis, out / "sweeps.png")
    plot_align(aligns, out / "align.png")
    write_report(blob, out.parent / "lm-rich-2d.md")

    by = _by_name(teachers)
    for name in ("faithful", "pair_odd", "v9", "pair_odd_sub_e", "pair_odd_sub_all", "project_short", "project_rich", "project_odd"):
        r = by[name]
        print(
            f"{name:20s} leak={r['leak_ratio']:+.3f} rich={r['rich_kept']:.2f} "
            f"g={r['leak_gender']:+.3f} bpm={r['leak_bpm']:+.3f} "
            f"cos={r['cos_intended']:.3f} ±1={r['cos_plus_minus']:+.3f}"
        )
    print(
        f"energy v9/sub leak={energy['energy_v9']['leak_ratio']:+.3f}/"
        f"{energy['energy_sub_e']['leak_ratio']:+.3f} "
        f"mismatch project pass={energy['mismatch_project_short']['pass']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
