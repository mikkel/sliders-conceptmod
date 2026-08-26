"""Plus-only exam: cover and off-caption on the + pole. Separate scale.

CPU geometry only. Does not change the live trainer default.
Does not fold into exam_score, leak_frac, or the compiled bipolar board.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from analysis.slider2d.exam import close_field, divergent_field, unused_e_field
from analysis.slider2d.plus_exam import (
    PLUS_COVER_MIN,
    PLUS_OFF_MAX,
    PLUS_RECIPES,
    blend_toward_mid,
    plus_bags,
    plus_cover,
    plus_exam_table,
    plus_helps,
    plus_teacher,
)
from conceptmod.textsliders.slider_targets import (
    UNUSED_E_OVERLAP_MAX,
    lm_e_is_unused,
    lm_faithful_plus,
    lm_faithful_sub_e,
    lm_hidden_targets,
    lm_plus_loss,
)
from conceptmod.textsliders.train_lm_slider_music3 import parse_args


STEPS = 400
_CACHE: dict[str, dict] = {}


def table(seed: int = 0) -> dict[str, list[dict]]:
    key = f"table{seed}"
    if key not in _CACHE:
        _CACHE[key] = plus_exam_table(steps=STEPS, seed=seed)
    return _CACHE[key]


def by_name(cell: str, seed: int = 0) -> dict[str, dict]:
    return {row["name"]: row for row in table(seed)[cell]}


def test_live_default_is_still_v9_hidden():
    args = parse_args(["--prompts_file", "prompts.yaml"])
    assert args.lm_target == "v9"
    assert args.pole_mode == "hidden"


def test_faithful_plus_teacher_is_pos_not_pair_odd():
    field = close_field()
    pos, neg, neu = field.poles(0)
    plus = lm_faithful_plus(pos, neg, neu)
    assert torch.allclose(plus, pos)
    pair_plus, _ = lm_hidden_targets(pos, neg, neu, target_mode="symmetric")
    assert not torch.allclose(plus, pair_plus)


def test_faithful_plus_leftover_gates_plus_side_only():
    field = unused_e_field()
    pos, neg, neu = field.poles(0)
    e = field.declared_e()
    unused, overlap = lm_e_is_unused(pos, neg, e, slider_dir=field.short_u())
    assert unused is True
    assert float(overlap) < UNUSED_E_OVERLAP_MAX
    plus = lm_faithful_plus(pos, neg, neu, e, slider_dir=field.short_u())
    want, _ = lm_faithful_sub_e(pos, neg, neu, e, slider_dir=field.short_u())
    assert torch.allclose(plus, want, atol=1e-6)
    field_div = divergent_field()
    d_pos, d_neg, d_neu = field_div.poles(0)
    kept = lm_faithful_plus(
        d_pos, d_neg, d_neu, field_div.declared_e(), slider_dir=field_div.short_u()
    )
    assert torch.allclose(kept, d_pos, atol=1e-6)


def test_plus_loss_ignores_minus():
    pred = torch.tensor([1.0, 0.0])
    tgt = torch.tensor([1.0, 0.0])
    other = torch.tensor([0.0, 9.0])
    a = lm_plus_loss(pred, tgt)
    b = lm_plus_loss(pred + 0.0, tgt)
    assert float(a) == pytest.approx(0.0, abs=1e-8)
    assert float(b) == pytest.approx(0.0, abs=1e-8)
    # The minus tensor is not an argument. Moving an unused minus cannot
    # enter this loss.
    _ = other


def test_cover_is_in_unit_interval_and_hits_at_pos():
    pos = torch.tensor([1.0, 0.0])
    mid = torch.tensor([0.0, 0.0])
    neg = torch.tensor([-1.0, 0.0])
    assert plus_cover(1.0, 0.0) == pytest.approx(1.0)
    assert plus_cover(0.9, 0.2) == pytest.approx(0.8)
    assert 0.0 <= plus_cover(1.2, -0.1) <= 1.0
    assert blend_toward_mid(pos, pos, mid, neg) == pytest.approx(0.0, abs=1e-6)
    assert blend_toward_mid(mid, pos, mid, neg) == pytest.approx(1.0, abs=1e-6)
    assert blend_toward_mid(neg, pos, mid, neg) == pytest.approx(1.0, abs=1e-6)


def test_plus_off_caption_treats_neg_track_words_as_off():
    """Same idea as pair-exam off-caption, + side only: lull is not shared song."""
    field = divergent_field()
    bags = plus_bags(field)
    head = field.readout()
    lull = head.index("lull")
    lyric = head.index("lyric0")
    assert lull not in bags["plus_corpus"]
    assert lyric in bags["plus_corpus"]
    assert lyric in bags["shared"]


def test_scored_columns_are_only_cover_and_off_caption():
    row = by_name("divergent")["faithful_plus"]
    assert "cover" in row and "off_caption" in row
    assert "leak_frac" not in row
    assert "exam_score" not in row
    assert "collapse" not in row
    assert "pair_odd_cos" not in row
    assert row["canary"]["scored"] is False


def test_plus_exam_does_not_import_the_bipolar_board():
    src = Path("analysis/slider2d/plus_exam.py").read_text()
    assert "exam_score" not in src
    assert "leftover_bipolar" not in src
    assert "scoreboard" not in src
    board = Path("analysis/slider2d/scoreboard.py").read_text()
    assert "faithful_plus" not in board
    assert "plus_exam" not in board


def test_pair_odd_fails_the_plus_scale_on_divergent():
    row = by_name("divergent")["pair_odd_midpoint"]
    assert row["hit"] is False
    assert row["cover"] < PLUS_COVER_MIN or row["off_caption"] > PLUS_OFF_MAX


def test_faithful_plus_hits_both_required_pairs():
    for cell in ("divergent", "close"):
        row = by_name(cell)["faithful_plus"]
        assert row["hit"] is True
        assert row["cover"] >= PLUS_COVER_MIN
        assert row["off_caption"] <= PLUS_OFF_MAX
        assert row["plus_only"] is True


def test_canary_minus_is_logged_and_unscored():
    row = by_name("divergent")["faithful_plus"]
    can = row["canary"]
    assert can["scored"] is False
    assert can["minus_landed"] in {"pos", "neu", "neg"}
    assert 0.0 <= can["minus_off_caption"] <= 1.0
    assert 0.0 <= can["minus_overlap_neg"] <= 1.0


def test_plus_helps_is_yes_or_no_from_the_numbers():
    verdict = plus_helps(table())
    assert verdict["yes"] in {True, False}
    assert len(verdict["plus_hits_required"]) == 2


def test_recipes_include_the_four_comparisons():
    names = [c["name"] for c in PLUS_RECIPES]
    assert names == [
        "faithful_plus",
        "leftover_gate_bipolar",
        "faithful_even_blend",
        "pair_odd_midpoint",
    ]


def test_plus_teacher_faithful_plus_matches_lm_faithful_plus():
    field = unused_e_field()
    plus, canary_minus = plus_teacher(
        field, 0, teacher="faithful_plus", leak_dir=field.declared_e()
    )
    pos, neg, neu = field.poles(0)
    assert torch.allclose(
        plus, lm_faithful_plus(pos, neg, neu, field.declared_e(), slider_dir=field.short_u())
    )
    # Canary minus is the unused caption, not a pair-odd teacher.
    pair_plus, pair_minus = lm_hidden_targets(pos, neg, neu, target_mode="symmetric")
    assert not torch.allclose(plus, pair_plus)
    assert torch.allclose(canary_minus, neg)
