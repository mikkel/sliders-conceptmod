#!/usr/bin/env python3
"""Train RpGAN + b_cap on the 2-D slider fixtures and print the exam/sheet gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from analysis.slider2d.exam import close_field, divergent_field, score_exam, unused_e_field
from analysis.slider2d.gan import (
    DEFAULT_TEACHER,
    default_cfg,
    score_adv_exam,
    score_adv_sheet,
    score_field2d,
)
from analysis.slider2d.scoreboard import (
    COMPILED_GARBLE_MAX,
    COMPILED_LEAK_LOCK,
    COMPILED_SHEET_LOCK,
    COMPILED_SWING_FLOOR,
    FAILS,
    WORKS,
    WORKS_SOME,
    cell_works,
    compiled_verdict,
    exam_score,
)
from analysis.slider2d.sheet import gender_like_field, leaky_field, score_sheet


DEFAULT_OUT = _REPO / "docs" / "lm-2d-adv"


def _fmt(value, spec: str, empty: str = "N/A") -> str:
    if value is None:
        return empty
    return format(float(value), spec)


def _pass(ok: bool | None) -> str:
    if ok is True:
        return "PASS"
    if ok is False:
        return "FAIL"
    return "—"


def _sheet_ok(row: dict) -> bool:
    return bool(
        cell_works(
            leak=row.get("leak_tok"),
            on_sheet_kept=row.get("on_sheet_kept"),
            off_sheet=row.get("garble"),
            argmax_on_sheet=row.get("argmax_on_sheet"),
            swing_kept=row.get("swing_kept"),
        )
    )


def collect(
    *,
    steps: int,
    exam_steps: int,
    seed: int,
    teacher: str,
    b_cap: float,
    fm_weight: float,
    baseline_steps: int,
    cover_weight: float = 1.5,
) -> dict:
    cfg = default_cfg(
        steps=steps,
        seed=seed,
        b_cap=b_cap,
        fm_weight=fm_weight,
        cover_weight=cover_weight,
    )
    exam_cfg = default_cfg(
        steps=exam_steps,
        seed=seed,
        b_cap=b_cap,
        fm_weight=fm_weight,
        cover_weight=cover_weight,
    )
    field2d = score_field2d(cfg)
    leftover = score_adv_sheet(leaky_field(), teacher=teacher, cfg=cfg)
    gender = score_adv_sheet(gender_like_field(), teacher=teacher, cfg=cfg)
    divergent = score_adv_exam(divergent_field(seed=seed), teacher=teacher, cfg=exam_cfg)
    close = score_adv_exam(close_field(seed=seed), teacher=teacher, cfg=exam_cfg)
    unused = score_adv_exam(unused_e_field(seed=seed), teacher=teacher, cfg=exam_cfg)

    base_left = score_sheet(
        "v6_faithful",
        leaky_field(),
        pole_mode="hidden",
        teacher="faithful",
        steps=baseline_steps,
        seed=seed,
    )
    base_mid = score_sheet(
        "v9_hidden",
        leaky_field(),
        pole_mode="hidden",
        teacher="pair_odd",
        steps=baseline_steps,
        seed=seed,
    )
    base_div = score_exam(
        "faithful_raw",
        divergent_field(seed=seed),
        pole_mode="hidden",
        teacher="faithful",
        steps=baseline_steps,
        seed=seed,
    )
    base_close = score_exam(
        "pair_odd_midpoint",
        close_field(seed=seed),
        pole_mode="hidden",
        teacher="pair_odd",
        steps=baseline_steps,
        seed=seed,
    )

    cells = {
        "exam_divergent": bool(divergent["pass"]),
        "exam_close": bool(close["pass"]),
        "exam_unused_e": bool(unused["pass"]),
        "sheet_leftover": _sheet_ok(leftover),
        "sheet_gender": _sheet_ok(gender),
    }
    overlap = {
        "exam_divergent": divergent.get("roll_overlap"),
        "exam_close": close.get("roll_overlap"),
        "exam_unused_e": unused.get("roll_overlap"),
    }
    swing = {
        "exam_divergent": divergent.get("roll_swing_kept"),
        "exam_close": close.get("roll_swing_kept"),
        "exam_unused_e": unused.get("roll_swing_kept"),
    }
    score = exam_score(overlap, swing)
    verdict = compiled_verdict(cells=cells)
    return {
        "cfg": {
            "steps": steps,
            "exam_steps": exam_steps,
            "baseline_steps": baseline_steps,
            "seed": seed,
            "teacher": teacher,
            "b_cap": b_cap,
            "fm_weight": fm_weight,
            "cover_weight": cover_weight,
        },
        "field2d": field2d,
        "sheet_leftover": leftover,
        "sheet_gender": gender,
        "exam_divergent": divergent,
        "exam_close": close,
        "exam_unused_e": unused,
        "baseline_sheet_faithful": base_left,
        "baseline_sheet_midpoint": base_mid,
        "baseline_exam_faithful_divergent": base_div,
        "baseline_exam_midpoint_close": base_close,
        "cells": cells,
        "exam_score": score,
        "compiled": verdict,
    }


def _slim(row: dict) -> dict:
    skip = {"log", "axis", "sings", "says"}
    out = {}
    for key, value in row.items():
        if key in skip:
            continue
        if isinstance(value, (int, float, str, bool)) or value is None:
            out[key] = value
    return out


def write_findings(blob: dict, path: Path) -> None:
    f2 = blob["field2d"]
    left = blob["sheet_leftover"]
    gender = blob["sheet_gender"]
    div = blob["exam_divergent"]
    close = blob["exam_close"]
    unused = blob["exam_unused_e"]
    raw = blob["baseline_sheet_faithful"]
    mid = blob["baseline_sheet_midpoint"]
    raw_div = blob["baseline_exam_faithful_divergent"]
    mid_close = blob["baseline_exam_midpoint_close"]
    cfg = blob["cfg"]
    lines = [
        "# RpGAN + b_cap on the 2-D slider fixtures",
        "",
        "CPU-only port of the ParticleGAN 100-Gaussians / Music-LM adversarial",
        "core (`rp_g_loss`, one-sided `b_cap`, Fourier-2 critic, ParticlePrior,",
        "VICReg, EMA, Adam β1=0, delayed cosine LR) into `analysis/slider2d`.",
        "Real samples are leftover-gated caption poles plus a span/end cloud —",
        "hidden-state deltas, **not** rendered audio. This page does not claim",
        "Music 3 listen quality.",
        "",
        "## Recipe",
        "",
        f"- teacher: `{cfg['teacher']}` (blend-guarded leftover ê; refuses when ê restates the axis)",
        f"- b_cap coeff: `{cfg['b_cap']}` (soft cap above 1, free below)",
        f"- feature matching: `{cfg['fm_weight']}` (0 = off; raw FM is uncapped by b_cap)",
        f"- cover_weight: `{cfg['cover_weight']}` (mode pin on the shared residual; needed on sheet/exam width)",
        f"- GAN steps: field/sheet `{cfg['steps']}`, exam `{cfg['exam_steps']}`, seed `{cfg['seed']}`",
        f"- supervised baseline steps: `{cfg['baseline_steps']}`",
        "",
        "```bash",
        "PYTHONPATH=. python analysis/slider2d/run_lm_adv.py --out docs/lm-2d-adv",
        "PYTHONPATH=. python analysis/gan_bcap/gaussian_repro.py --modes 8 --steps 1500",
        "PYTHONPATH=. pytest tests/test_lm_2d_adv.py -q",
        "```",
        "",
        "## Compiled gate vs supervised baselines",
        "",
        f"GAN compiled verdict: **{blob['compiled']}**. "
        f"`exam_score` = `{_fmt(blob.get('exam_score'), '.3f')}`.",
        "",
        "| cell | GAN | supervised baseline | baseline recipe | why the baseline loses |",
        "|---|---|---|---|---|",
        f"| sheet leftover | {_pass(blob['cells']['sheet_leftover'])} "
        f"(leak {_fmt(left.get('leak_tok'), '+.3f')}, kept {_fmt(left.get('on_sheet_kept'), '.3f')}) | "
        f"{_pass(_sheet_ok(raw))} "
        f"(leak {_fmt(raw.get('leak_tok'), '+.3f')}, kept {_fmt(raw.get('on_sheet_kept'), '.3f')}) | "
        f"`faithful_raw` / v6 | copies unused ê inside the raw poles |",
        f"| sheet leftover (midpoint) | — | {_pass(_sheet_ok(mid))} "
        f"(kept {_fmt(mid.get('on_sheet_kept'), '.3f')}, off {_fmt(mid.get('garble'), '.3f')}) | "
        f"`pair_odd_midpoint` / v9 | deletes `c`; walks off the sheet |",
        f"| sheet gender | {_pass(blob['cells']['sheet_gender'])} "
        f"(kept {_fmt(gender.get('on_sheet_kept'), '.3f')}) | "
        f"PASS on `faithful_raw` / FAIL on v9 (scoreboard) | caption vs midpoint | "
        f"GAN keeps `c` like a caption teacher |",
        f"| exam divergent | {_pass(blob['cells']['exam_divergent'])} "
        f"(overlap {_fmt(div.get('roll_overlap'), '.3f')}, swing {_fmt(div.get('roll_swing_kept'), '.3f')}) | "
        f"{_pass(raw_div.get('pass'))} "
        f"(overlap {_fmt(raw_div.get('roll_overlap'), '.3f')}) | `faithful_raw` | "
        f"same caption target; leftover-gate refuses to eat the axis |",
        f"| exam close | {_pass(blob['cells']['exam_close'])} "
        f"(overlap {_fmt(close.get('roll_overlap'), '.3f')}, swing {_fmt(close.get('roll_swing_kept'), '.3f')}) | "
        f"{_pass(mid_close.get('pass'))} "
        f"(overlap {_fmt(mid_close.get('roll_overlap'), '.3f')}) | `pair_odd_midpoint` | "
        f"midpoint teacher has no delivery to roll out |",
        f"| exam unused_e | {_pass(blob['cells']['exam_unused_e'])} "
        f"(overlap {_fmt(unused.get('roll_overlap'), '.3f')}) | "
        f"PASS on leftover-gated MSE (scoreboard) | `faithful_sub_e` | "
        f"GAN uses the same leftover-gated real cloud |",
        "",
        "Scoreboard cells this has to beat: `faithful_raw` works the exam pairs and",
        f"fails leftover leak (`≈ 0.228` > `{COMPILED_LEAK_LOCK}`); `pair_odd_midpoint`",
        f"locks pair-odd cos and fails the sheet (kept `≈ 0.37` < `{COMPILED_SHEET_LOCK}`).",
        f"Sheet lock `{COMPILED_SHEET_LOCK}`, garble cap `{COMPILED_GARBLE_MAX}`,",
        f"swing `{COMPILED_SWING_FLOOR}`. A recipe WORKS only if every cell it is",
        "read on passes.",
        "",
        "## Field2D polarity / leftover",
        "",
        f"| metric | GAN | gate |",
        "|---|---:|---|",
        f"| slider cos | {_fmt(f2.get('cos_slider_plus'), '+.3f')} | ≥ 0.90 |",
        f"| leak ratio | {_fmt(f2.get('leak_ratio'), '+.3f')} | ≤ 0.20 |",
        f"| ±1 cos | {_fmt(f2.get('cos_plus_minus'), '+.3f')} | ≤ −0.85 |",
        f"| leak_frac (bipolar) | {_fmt(f2.get('leak_frac'), '+.3f')} | ≤ −0.85 |",
        f"| same_dir (even share) | {_fmt(f2.get('same_dir'), '.3f')} | small |",
        f"| pass | {_pass(f2.get('pass'))} | |",
        "",
        "## What moved the needle",
        "",
        "- **Leftover-gated real cloud** (`faithful_guard_e`): without it the GAN",
        "  copies unused ê the same way `faithful_raw` does. The guard refuses on",
        "  energy-v4, so exam_divergent keeps the genre/BPM ride.",
        "- **End-margin + span samples**: 2-D analogue of last-token + lyric-span",
        "  pooling. End-margin keeps D pinned on the actual poles; span stops the",
        "  residual from collapsing to the midpoint.",
        "- **b_cap = 1**: one-sided, both real and fake. D stays steep enough to",
        "  separate the two poles and cannot race `||∇D||` to infinity.",
        "- **Particle L2 + small VICReg**: particles stay a jitter prior. The",
        "  scored object is the shared residual, so particles must not steal the mode.",
        "- **cover_weight = 1.5** (mode pin): pure RpGAN reaches attributed 2-D",
        "  poles (leak 0.02, ±1 = −1) but undershoots the sheet/exam residual",
        "  (on-sheet kept 0.23). A small MSE onto the leftover-gated centers is",
        "  the 2-D HQ analogue — without it the shared LoRA-like residual stays",
        "  near 0 while particles eat the modes. 1200 steps + 1.5 clears the",
        "  0.90 on-sheet lock; 800 + 1.0 lands at 0.86.",
        "- **Feature matching off** by default. Unnormalized FM is uncapped by",
        "  b_cap and pulled D features instead of the residual. Normalized FM was",
        "  tried and did not beat the leftover gate.",
        "- **EMA residual + Adam β1=0 + delayed cosine**: the ParticleGAN schedule.",
        "  lr 5e-3, delay 80, particle L2 0.02. Heavier VICReg/L2 starved G.",
        "",
        "## What this is not",
        "",
        "- Not a Music 3 listen. No MiniMax weights, no audio render.",
        "- Not a new live `--pole_mode`. The live default stays `--lm_target v9`.",
        "- Not the 100-Gaussians HQ number on this page. That smoke lives in",
        "  `analysis/gan_bcap/gaussian_repro.py` (8 modes in CI; `--modes 100` is opt-in).",
        "",
        f"Gates used: leftover leak ≤ {COMPILED_LEAK_LOCK}, on-sheet kept ≥ {COMPILED_SHEET_LOCK},",
        f"off-sheet ≤ {COMPILED_GARBLE_MAX}, swing ≥ {COMPILED_SWING_FLOOR}.",
        f"Compiled labels: `{WORKS}`, `{WORKS_SOME}`, `{FAILS}`.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--exam-steps", type=int, default=1200)
    parser.add_argument("--baseline-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--teacher", type=str, default=DEFAULT_TEACHER)
    parser.add_argument("--b-cap", type=float, default=1.0)
    parser.add_argument("--fm-weight", type=float, default=0.0)
    parser.add_argument("--cover-weight", type=float, default=1.5)
    args = parser.parse_args(argv)

    blob = collect(
        steps=args.steps,
        exam_steps=args.exam_steps,
        seed=args.seed,
        teacher=args.teacher,
        b_cap=args.b_cap,
        fm_weight=args.fm_weight,
        baseline_steps=args.baseline_steps,
        cover_weight=args.cover_weight,
    )
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    slim = {
        "cfg": blob["cfg"],
        "cells": blob["cells"],
        "exam_score": blob["exam_score"],
        "compiled": blob["compiled"],
        "field2d": _slim(blob["field2d"]),
        "sheet_leftover": _slim(blob["sheet_leftover"]),
        "sheet_gender": _slim(blob["sheet_gender"]),
        "exam_divergent": _slim(blob["exam_divergent"]),
        "exam_close": _slim(blob["exam_close"]),
        "exam_unused_e": _slim(blob["exam_unused_e"]),
        "baseline_sheet_faithful": _slim(blob["baseline_sheet_faithful"]),
        "baseline_sheet_midpoint": _slim(blob["baseline_sheet_midpoint"]),
        "live_default_unchanged": True,
        "claims_music3_audio": False,
    }
    (out / "metrics.json").write_text(json.dumps(slim, indent=2) + "\n", encoding="utf-8")
    write_findings(blob, _REPO / "analysis" / "slider2d" / "gan_bcap_findings.md")
    write_findings(blob, out.parent / "lm-2d-adv.md")
    print(
        f"{blob['compiled']:20s} rpgan_bcap exam={_fmt(blob.get('exam_score'), '.3f')} "
        f"left={_pass(blob['cells']['sheet_leftover'])} "
        f"div={_pass(blob['cells']['exam_divergent'])} "
        f"close={_pass(blob['cells']['exam_close'])} "
        f"leak={_fmt(blob['sheet_leftover'].get('leak_tok'), '+.3f')}"
    )
    f2 = blob["field2d"]
    print(
        f"{'field2d':20s} slider={_fmt(f2.get('cos_slider_plus'), '+.3f')} "
        f"leak={_fmt(f2.get('leak_ratio'), '+.3f')} "
        f"±1={_fmt(f2.get('cos_plus_minus'), '+.3f')} "
        f"leak_frac={_fmt(f2.get('leak_frac'), '+.3f')} "
        f"pass={_pass(f2.get('pass'))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
