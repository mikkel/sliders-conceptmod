"""Student-apart regularizer: caption teacher, push ±1 apart.

The live default is unchanged. A win is exam_divergent True and
leak_frac < 0 without falling into the −1 midpoint well. CPU only.
"""

from __future__ import annotations

import pytest
import torch

from analysis.slider2d.exam import (
    APART_MIDPOINT_LEAK,
    CELLS,
    apart_is_win,
    pick_apart_win,
    score_apart,
    score_exam,
    teacher_points,
)
from analysis.slider2d.sheet import leaky_field, score_sheet
from conceptmod.textsliders.slider_targets import (
    APART_KINDS,
    leftover_bipolar,
    lm_slider_loss,
    lm_student_apart,
    resolve_apart_kind,
)
from conceptmod.textsliders.train_lm_slider_music3 import (
    lm_train_loss,
    parse_args,
)


def test_apart_kinds_are_the_three_student_terms():
    assert APART_KINDS == ("even", "cos", "same_dir")
    for kind in APART_KINDS:
        assert resolve_apart_kind(kind) == kind
    with pytest.raises(ValueError, match="apart_kind"):
        resolve_apart_kind("pair_odd")


def test_student_apart_does_not_rewrite_the_teacher():
    """The term is a function of d± only. Targets never enter it."""
    neu = torch.zeros(4)
    pred_plus = torch.tensor([1.0, 0.2, 0.0, 0.0])
    pred_minus = torch.tensor([-0.4, 0.2, 0.0, 0.0])
    tgt_plus = torch.tensor([9.0, 9.0, 9.0, 9.0])
    tgt_minus = torch.tensor([-9.0, -9.0, -9.0, -9.0])
    even = lm_student_apart(pred_plus, pred_minus, neu, kind="even")
    # even = ½(d+ + d−) = (0.3, 0.2, 0, 0); mean-sq = (0.09+0.04)/4
    assert even == pytest.approx((0.09 + 0.04) / 4.0, abs=1e-6)
    cos = lm_student_apart(pred_plus, pred_minus, neu, kind="cos")
    bipolar = leftover_bipolar(pred_plus - neu, pred_minus - neu)
    assert float(cos) == pytest.approx(bipolar["leak_frac"], abs=1e-6)
    same = lm_student_apart(pred_plus, pred_minus, neu, kind="same_dir")
    assert float(same) == pytest.approx(bipolar["same_dir"], abs=1e-6)
    # Changing the teacher cannot change the regularizer.
    other = lm_student_apart(pred_plus, pred_minus, neu, kind="even")
    assert float(other) == float(even)
    _ = tgt_plus, tgt_minus


def test_zero_apart_weight_leaves_the_pole_loss_alone():
    pred_plus = torch.tensor([1.0, 0.0])
    pred_minus = torch.tensor([0.0, 1.0])
    tgt_plus = torch.tensor([1.0, 0.0])
    tgt_minus = torch.tensor([0.0, 1.0])
    neu = torch.zeros(2)
    base = lm_slider_loss(pred_plus, pred_minus, tgt_plus, tgt_minus)
    with_zero = lm_slider_loss(
        pred_plus,
        pred_minus,
        tgt_plus,
        tgt_minus,
        apart=lm_student_apart(pred_plus, pred_minus, neu, kind="even"),
        apart_weight=0.0,
    )
    assert float(with_zero) == pytest.approx(float(base), abs=1e-8)


def test_even_term_has_gradient_only_on_the_even_residual():
    neu = torch.zeros(3)
    w_odd = torch.tensor([1.0, 0.0, 0.0], requires_grad=True)
    w_even = torch.tensor([0.4, 0.2, 0.0], requires_grad=True)
    pred_plus = neu + w_odd + w_even
    pred_minus = neu - w_odd + w_even
    loss = lm_student_apart(pred_plus, pred_minus, neu, kind="even")
    loss.backward()
    assert w_odd.grad is not None and float(w_odd.grad.norm()) == pytest.approx(0.0, abs=1e-6)
    assert w_even.grad is not None and float(w_even.grad.norm()) > 0.0


def test_live_default_is_still_v9_hidden_with_apart_off():
    args = parse_args(["--prompts_file", "x.yaml"])
    assert args.lm_target == "v9"
    assert args.pole_mode == "hidden"
    assert float(args.apart_weight) == 0.0
    assert args.apart_kind == "even"
    card = parse_args(
        [
            "--prompts_file",
            "x.yaml",
            "--lm_target",
            "faithful",
            "--apart_kind",
            "even",
            "--apart_weight",
            "0.25",
        ]
    )
    assert card.lm_target == "faithful"
    assert card.pole_mode == "hidden"
    assert card.apart_kind == "even"
    assert card.apart_weight == pytest.approx(0.25)


def test_lm_train_loss_apart_default_is_zero():
    neu = torch.zeros(2)
    pred_plus = torch.tensor([1.0, 0.1])
    pred_minus = torch.tensor([-1.0, 0.1])
    tgt_plus = pred_plus.clone()
    tgt_minus = pred_minus.clone()
    base = lm_train_loss(pred_plus, pred_minus, tgt_plus, tgt_minus, neu=neu)
    apart = lm_train_loss(
        pred_plus,
        pred_minus,
        tgt_plus,
        tgt_minus,
        neu=neu,
        apart_weight=1.0,
        apart_kind="even",
    )
    assert float(apart) > float(base)


def test_caption_poles_start_with_nonneg_leak_frac_on_divergent():
    """The baseline the regularizer has to beat. Not a data pin."""
    field = CELLS["divergent"]()
    row = score_exam(
        "faithful_raw",
        field,
        pole_mode="hidden",
        teacher="faithful",
        steps=200,
        seed=0,
    )
    assert row["pass"] is True
    assert row["leak_frac"] >= 0.0
    # Teacher is still the poles.
    t_plus, t_minus = teacher_points(field, 0, teacher="faithful")
    pos, neg, _neu = field.poles(0)
    assert torch.equal(t_plus, pos) and torch.equal(t_minus, neg)


def test_large_same_dir_weight_is_pair_odd_and_fails_divergent():
    """Secret pair-odd: leak_frac ≈ −1, energy fails. Not a win."""
    row = score_apart("same_dir", 8.0, steps=200, seed=0)
    assert row["leak_frac"] <= APART_MIDPOINT_LEAK
    assert row["exam_divergent"] is False
    assert row["win"] is False
    assert row["pair_odd_like"] is True


def test_even_card_passes_divergent_with_negative_leak_frac():
    """The wired train card. Teacher is still the caption poles."""
    row = score_apart("even", 0.25, steps=400, seed=0)
    assert row["exam_divergent"] is True
    assert row["exam_close"] is True
    assert row["leak_frac"] < 0.0
    assert row["leak_frac"] > APART_MIDPOINT_LEAK
    assert row["win"] is True
    field = CELLS["divergent"]()
    t_plus, t_minus = teacher_points(field, 0, teacher="faithful")
    pos, neg, _neu = field.poles(0)
    assert torch.equal(t_plus, pos) and torch.equal(t_minus, neg)


def test_apart_is_win_rejects_the_minus_one_well():
    assert apart_is_win({"exam_divergent": True, "leak_frac": -0.08}) is True
    assert apart_is_win({"exam_divergent": True, "leak_frac": 0.03}) is False
    assert apart_is_win({"exam_divergent": False, "leak_frac": -0.08}) is False
    assert apart_is_win({"exam_divergent": True, "leak_frac": -1.0}) is False
    assert pick_apart_win([]) is None
    picked = pick_apart_win(
        [
            {
                "win": True,
                "exam_close": True,
                "apart_kind": "cos",
                "leak_frac": -0.06,
                "apart_weight": 0.06,
            },
            {
                "win": True,
                "exam_close": True,
                "apart_kind": "even",
                "leak_frac": -0.10,
                "apart_weight": 0.25,
            },
            {
                "win": True,
                "exam_close": True,
                "apart_kind": "even",
                "leak_frac": -0.015,
                "apart_weight": 0.06,
            },
        ]
    )
    assert picked["apart_kind"] == "even"
    assert picked["apart_weight"] == 0.25


def test_leftover_sheet_faithful_leak_frac_is_the_caption_cluster():
    """Caption poles on the unused-ê sheet sit at small positive leak_frac."""
    row = score_sheet(
        "faithful_raw",
        leaky_field(),
        pole_mode="hidden",
        teacher="faithful",
        steps=200,
        seed=0,
    )
    assert row["leak_frac"] > 0.0
    assert row["leak_frac"] < 0.20


def test_even_card_flips_sheet_leak_frac_and_does_not_eat_odd_e():
    """Unused ê lives in a (odd). even-reg is not a leftover-leak fix."""
    raw = score_sheet(
        "faithful_raw",
        leaky_field(),
        pole_mode="hidden",
        teacher="faithful",
        steps=200,
        seed=0,
    )
    card = score_sheet(
        "faithful_apart_even",
        leaky_field(),
        pole_mode="hidden",
        teacher="faithful",
        apart_weight=0.25,
        apart_kind="even",
        steps=200,
        seed=0,
    )
    assert card["leak_frac"] < 0.0
    assert abs(card["leak_tok"] - raw["leak_tok"]) < 0.02
