"""Plus+neu exam: cover, off-caption, neu_hold. Separate scale.

CPU geometry only. Does not change the live trainer default.
Does not fold into exam_score, leak_frac, or the compiled bipolar board.
Does not change faithful_plus (still plus-only).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from analysis.slider2d.exam import close_field, divergent_field, unused_e_field
from analysis.slider2d.plus_exam import PLUS_COVER_MIN, PLUS_OFF_MAX
from analysis.slider2d.plus_neu_exam import (
    PLUS_NEU_HOLD_MIN,
    PLUS_NEU_RECIPES,
    drift_from_neu,
    neu_bags,
    neu_hold,
    plus_neu_exam_table,
    plus_neu_helps,
    plus_neu_rank,
    plus_neu_teacher,
)
from conceptmod.textsliders.slider_targets import (
    UNUSED_E_OVERLAP_MAX,
    lm_e_is_unused,
    lm_faithful_plus,
    lm_faithful_plus_neu,
    lm_faithful_sub_e,
    lm_plus_loss,
    lm_plus_neu_loss,
)
from conceptmod.textsliders.train_lm_slider_music3 import parse_args


STEPS = 400
_CACHE: dict[str, dict] = {}


def table(seed: int = 0) -> dict[str, list[dict]]:
    key = f"table{seed}"
    if key not in _CACHE:
        _CACHE[key] = plus_neu_exam_table(steps=STEPS, seed=seed)
    return _CACHE[key]


def by_name(cell: str, seed: int = 0) -> dict[str, dict]:
    return {row["name"]: row for row in table(seed)[cell]}


def test_live_default_is_still_v9_hidden():
    args = parse_args(["--prompts_file", "prompts.yaml"])
    assert args.lm_target == "v9"
    assert args.pole_mode == "hidden"


def test_faithful_plus_neu_teacher_is_raw_pos_not_leftover_gated():
    field = unused_e_field()
    pos, neg, neu = field.poles(0)
    e = field.declared_e()
    unused, overlap = lm_e_is_unused(pos, neg, e, slider_dir=field.short_u())
    assert unused is True
    assert float(overlap) < UNUSED_E_OVERLAP_MAX
    raw = lm_faithful_plus_neu(pos, neg, neu, e, slider_dir=field.short_u())
    gated = lm_faithful_plus(pos, neg, neu, e, slider_dir=field.short_u())
    want_gated, _ = lm_faithful_sub_e(pos, neg, neu, e, slider_dir=field.short_u())
    assert torch.allclose(raw, pos)
    assert torch.allclose(gated, want_gated, atol=1e-6)
    assert not torch.allclose(raw, gated, atol=1e-4)


def test_plus_neu_loss_is_mse_plus_and_mse_zero_only():
    pred_plus = torch.tensor([1.0, 0.0])
    tgt_plus = torch.tensor([1.0, 0.0])
    pred_zero = torch.tensor([0.0, 0.0])
    tgt_zero = torch.tensor([0.0, 0.0])
    other = torch.tensor([0.0, 9.0])
    base = lm_plus_neu_loss(pred_plus, tgt_plus, pred_zero, tgt_zero)
    moved_minus = lm_plus_neu_loss(pred_plus, tgt_plus, pred_zero, tgt_zero)
    assert float(base) == pytest.approx(0.0, abs=1e-8)
    assert float(moved_minus) == pytest.approx(0.0, abs=1e-8)
    moved_zero = lm_plus_neu_loss(pred_plus, tgt_plus, pred_zero + 1.0, tgt_zero)
    moved_plus = lm_plus_neu_loss(pred_plus + 1.0, tgt_plus, pred_zero, tgt_zero)
    assert float(moved_zero) > 0.0
    assert float(moved_plus) > 0.0
    plus_only = lm_plus_loss(pred_plus, tgt_plus)
    assert float(plus_only) == pytest.approx(0.0, abs=1e-8)
    _ = other


def test_neu_hold_is_in_unit_interval_and_hits_at_neu():
    neu = torch.tensor([0.0, 0.0])
    pos = torch.tensor([1.0, 0.0])
    mid = torch.tensor([0.5, 0.0])
    assert neu_hold(1.0, 0.0) == pytest.approx(1.0)
    assert neu_hold(0.9, 0.2) == pytest.approx(0.8)
    assert 0.0 <= neu_hold(1.2, -0.1) <= 1.0
    assert drift_from_neu(neu, neu, pos, mid) == pytest.approx(0.0, abs=1e-6)
    assert drift_from_neu(pos, neu, pos, mid) == pytest.approx(1.0, abs=1e-6)
    assert drift_from_neu(mid, neu, pos, mid) == pytest.approx(1.0, abs=1e-6)


def test_neu_bags_include_lyrics_field():
    field = divergent_field()
    bag = neu_bags(field)
    head = field.readout()
    assert head.index("lyric0") in bag
    assert head.index("lyric1") in bag
    assert head.index("lyric2") in bag
    # The neu caption also sings sheet tokens (ŝ). That is on-song, not
    # a − track teacher. Minus-unique track words are not required to
    # be absent from this bag.


def test_scored_columns_are_cover_off_caption_neu_hold():
    row = by_name("divergent")["faithful_plus_neu"]
    assert "cover" in row and "off_caption" in row and "neu_hold" in row
    assert "leak_frac" not in row
    assert "exam_score" not in row
    assert "collapse" not in row
    assert "pair_odd_cos" not in row
    assert row["canary"]["scored"] is False


def test_plus_neu_exam_does_not_import_the_bipolar_board():
    src = Path("analysis/slider2d/plus_neu_exam.py").read_text()
    assert "from analysis.slider2d.scoreboard" not in src
    assert "from analysis.slider2d.scoreboard import exam_score" not in src
    board = Path("analysis/slider2d/scoreboard.py").read_text()
    assert "faithful_plus_neu" not in board
    assert "plus_neu_exam" not in board


def test_faithful_plus_neu_hits_all_three_on_required_pairs():
    for cell in ("divergent", "close"):
        row = by_name(cell)["faithful_plus_neu"]
        assert row["hit"] is True
        assert row["cover"] >= PLUS_COVER_MIN
        assert row["off_caption"] <= PLUS_OFF_MAX
        assert row["neu_hold"] >= PLUS_NEU_HOLD_MIN
        assert row["plus_neu"] is True
        assert row["plus_only"] is False


def test_faithful_plus_is_still_plus_only_and_loses_neu_hold():
    for cell in ("divergent", "close"):
        plus = by_name(cell)["faithful_plus"]
        uni = by_name(cell)["faithful_plus_neu"]
        assert plus["plus_only"] is True
        assert plus["plus_neu"] is False
        assert uni["neu_hold"] > plus["neu_hold"] + 1e-6


def test_rank_puts_in_box_first_then_neu_hold():
    ranked = plus_neu_rank(table())
    names = [r["name"] for r in ranked]
    assert names[0] == "faithful_plus_neu"
    assert ranked[0]["in_box"] is True
    for earlier, later in zip(ranked, ranked[1:]):
        if earlier["in_box"] != later["in_box"]:
            assert earlier["in_box"] is True
            continue
        if abs(earlier["neu_hold"] - later["neu_hold"]) > 1e-9:
            assert earlier["neu_hold"] >= later["neu_hold"]
            continue
        if abs(earlier["cover"] - later["cover"]) > 1e-9:
            assert earlier["cover"] >= later["cover"]
            continue
        assert earlier["off_caption"] <= later["off_caption"] + 1e-9


def test_recipes_include_the_five_comparisons():
    names = [c["name"] for c in PLUS_NEU_RECIPES]
    assert names == [
        "faithful_plus_neu",
        "faithful_plus",
        "leftover_gate_bipolar",
        "faithful_even_blend",
        "pair_odd_midpoint",
    ]


def test_plus_neu_teacher_matches_lm_faithful_plus_neu():
    field = unused_e_field()
    plus, canary_minus = plus_neu_teacher(
        field, 0, teacher="faithful_plus_neu", leak_dir=field.declared_e()
    )
    pos, neg, neu = field.poles(0)
    assert torch.allclose(
        plus,
        lm_faithful_plus_neu(
            pos, neg, neu, field.declared_e(), slider_dir=field.short_u()
        ),
    )
    assert torch.allclose(plus, pos)
    assert torch.allclose(canary_minus, neg)


def test_plus_neu_helps_is_yes_on_hold_and_keeps_cover():
    verdict = plus_neu_helps(table())
    assert verdict["beats_plus_on_neu_hold"] is True
    assert verdict["keeps_plus_cover"] is True
    assert all(verdict["uni_hits_required"])
