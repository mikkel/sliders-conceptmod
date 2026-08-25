"""Gender-v1 mismatch cell: clean pair-odd vs a short declared û.

The energetic×gender suite sets û from pole names. That is not this
check. These tests fail if someone replaces this cell with
``û = energetic/calm``.
"""

from __future__ import annotations

import pytest
import torch

from analysis.slider2d.field import E_SLIDER, Field2D
from analysis.slider2d.mismatch import (
    LIVE_GENDER_V1_ALIGN,
    POLE_NEG,
    POLE_POS,
    PROJECT_ALIGN_RECOMMENDED,
    PROJECT_ALIGN_SLIDER_KNEE,
    SHORT_NEG,
    SHORT_POS,
    MismatchField2D,
    knee_from_sweep,
    leak_cell_align,
    mismatch_all_right,
    mismatch_pairs,
    mismatch_verdicts,
    pair_slider_dir_hides_mismatch,
    score_against_odd,
    score_leak_policy,
    score_mismatch_policy,
    sweep_project_hold_align,
    train_mismatch,
)
from analysis.slider2d.train import music3_pairs, pair_slider_dir, train_lm
from conceptmod.textsliders.slider_targets import (
    lm_hidden_targets,
    lm_odd_align,
    lm_project_odd_axis,
    lm_should_project_odd,
)


_MISMATCH = None
_LEAK = None
_SWEEP = None


def mismatch_results():
    global _MISMATCH
    if _MISMATCH is None:
        _MISMATCH = {
            r["name"]: r
            for r in (
                score_mismatch_policy(
                    "pair_symmetric", project_odd=False, hold_weight=0.0, use_short_u=False
                ),
                score_mismatch_policy(
                    "always_project_hold", project_odd=True, hold_weight=1.0, use_short_u=True
                ),
                score_mismatch_policy(
                    "gated_align",
                    project_odd=True,
                    hold_weight=1.0,
                    use_short_u=True,
                    project_align_min=PROJECT_ALIGN_RECOMMENDED,
                ),
            )
        }
    return _MISMATCH


def leak_results():
    global _LEAK
    if _LEAK is None:
        _LEAK = {
            r["name"]: r
            for r in (
                score_leak_policy("pair_symmetric", project_odd=False, hold_weight=0.0),
                score_leak_policy("always_project_hold", project_odd=True, hold_weight=1.0),
                score_leak_policy(
                    "gated_align",
                    project_odd=True,
                    hold_weight=1.0,
                    project_align_min=PROJECT_ALIGN_RECOMMENDED,
                ),
            )
        }
    return _LEAK


def sweep_rows():
    global _SWEEP
    if _SWEEP is None:
        _SWEEP = sweep_project_hold_align()
    return _SWEEP


def test_declared_u_is_not_the_pole_names():
    field = MismatchField2D()
    assert field.pole_names == (POLE_POS, POLE_NEG)
    assert field.declared_captions == (SHORT_POS, SHORT_NEG)
    assert field.pole_names != field.declared_captions
    assert "energetic" not in field.pole_names
    assert "calm" not in field.pole_names
    pair = mismatch_pairs()[0]
    assert pair.positive.name == POLE_POS
    assert pair.negative.name == POLE_NEG


def test_pair_odd_is_already_the_concept():
    field = MismatchField2D()
    pos, neg, neu = field.rich_pair()
    odd = (pos - neg) / 2.0
    assert float(odd @ field.junk) == pytest.approx(0.0, abs=1e-6)
    assert float(odd @ field.concept) == pytest.approx(float(odd.norm()), abs=1e-6)
    plus, minus = lm_hidden_targets(pos, neg, neu, target_mode="symmetric")
    assert torch.allclose(plus - neu, odd, atol=1e-6)


def test_short_u_alignment_matches_live_gender_v1():
    field = MismatchField2D()
    pos, neg, _neu = field.rich_pair()
    align = float(lm_odd_align(pos, neg, field.declared_u))
    assert align == pytest.approx(LIVE_GENDER_V1_ALIGN, abs=1e-6)
    assert 0.18 <= align <= 0.22


def test_pole_name_u_is_identity_and_hides_the_failure():
    """û from pole names is the concept. Project+hold then looks like a pass.

    That is the old energetic×gender fixture. This cell is not that check.
    """
    field = MismatchField2D()
    pos, neg, neu = field.rich_pair()
    odd = (pos - neg) / 2.0
    pair = mismatch_pairs()[0]
    pole_u = pair_slider_dir(pair)
    assert torch.allclose(pole_u, E_SLIDER)
    assert torch.allclose(pole_u, field.concept)
    assert pair_slider_dir_hides_mismatch()
    plus, minus = lm_project_odd_axis(pos, neg, neu, pole_u)
    assert torch.allclose(plus, neu + odd, atol=1e-6)
    # The actual declared û is tilted. Projecting it is *not* identity.
    plus_short, _ = lm_project_odd_axis(pos, neg, neu, field.declared_u)
    projected = plus_short - neu
    assert float(projected.norm() / odd.norm()) == pytest.approx(LIVE_GENDER_V1_ALIGN, abs=1e-5)
    assert abs(float((projected / projected.norm()) @ (odd / odd.norm()))) == pytest.approx(
        LIVE_GENDER_V1_ALIGN, abs=1e-5
    )


def test_old_leak_pairs_are_not_this_cell():
    leak_pair = music3_pairs(False)[0]
    assert leak_pair.positive.name == "energetic"
    assert leak_pair.negative.name == "calm"
    assert torch.allclose(pair_slider_dir(leak_pair), E_SLIDER)
    mismatch = mismatch_pairs()[0]
    assert mismatch.positive.name != leak_pair.positive.name
    assert mismatch.negative.name != leak_pair.negative.name


def test_current_v9_fails_mismatch_cell():
    r = mismatch_results()["always_project_hold"]
    ax = r["axis"]
    assert r["odd_align"] == pytest.approx(LIVE_GENDER_V1_ALIGN, abs=1e-5)
    assert r["cos_concept"] == pytest.approx(LIVE_GENDER_V1_ALIGN, abs=0.05)
    assert r["strength"] < 0.30
    assert r["norm_plus"] < 0.40
    assert abs(r["leak_ratio"]) > 0.20
    assert ax["slider"] == "needs_help"
    assert ax["strength"] == "needs_help"
    assert ax["leak"] == "needs_help"
    assert r["pass"] is False


def test_pair_symmetric_passes_mismatch_cell():
    r = mismatch_results()["pair_symmetric"]
    assert r["cos_concept"] > 0.90
    assert r["strength"] > 0.80
    assert abs(r["leak_ratio"]) <= 0.20
    assert r["cos_plus_minus"] <= -0.85
    assert r["pass"] is True


def test_hold_eats_the_concept_when_u_is_tilted():
    field = MismatchField2D()
    residual = train_mismatch(
        field, project_odd=True, hold_weight=1.0, slider_dir=field.declared_u
    )
    odd = field.odd()
    metrics = score_against_odd(residual, odd, junk=field.junk)
    # Only align² of the pair remains on the true axis after project+hold.
    assert metrics["concept_kept"] == pytest.approx(LIVE_GENDER_V1_ALIGN ** 2, abs=0.05)
    assert metrics["concept_kept"] < 0.10


def test_leak_cell_v9_still_leak0():
    r = leak_results()["always_project_hold"]
    assert r["axis"]["slider"] == "right"
    assert r["axis"]["leak"] == "right"
    assert r["axis"]["collapse"] == "right"
    assert abs(r["leak_ratio"]) < 0.05
    assert r["cos_slider_plus"] > 0.90
    assert r["pass"] is True


def test_leak_cell_pair_symmetric_still_leaks():
    r = leak_results()["pair_symmetric"]
    assert abs(r["leak_ratio"]) > 0.20
    assert r["pass"] is False


def test_gated_policy_right_on_both_cells():
    mm = mismatch_results()["gated_align"]
    lk = leak_results()["gated_align"]
    assert mm["odd_align"] < PROJECT_ALIGN_RECOMMENDED
    assert lk["odd_align"] > PROJECT_ALIGN_RECOMMENDED
    assert mm["pass"] is True
    assert lk["pass"] is True
    assert abs(lk["leak_ratio"]) < 0.05
    assert mm["cos_concept"] > 0.90
    assert mm["strength"] > 0.80


def test_always_project_is_not_right_on_both_cells():
    mm = mismatch_results()["always_project_hold"]
    lk = leak_results()["always_project_hold"]
    assert lk["pass"] is True
    assert mm["pass"] is False


def test_pair_symmetric_is_not_right_on_both_cells():
    mm = mismatch_results()["pair_symmetric"]
    lk = leak_results()["pair_symmetric"]
    assert mm["pass"] is True
    assert lk["pass"] is False


def test_knee_is_above_live_gender_and_below_leak_cell():
    rows = sweep_rows()
    slider_knee = knee_from_sweep(rows, "slider")
    strength_knee = knee_from_sweep(rows, "strength")
    leak_knee = knee_from_sweep(rows, "leak")
    live = LIVE_GENDER_V1_ALIGN
    leak_align = leak_cell_align()
    assert slider_knee == pytest.approx(PROJECT_ALIGN_SLIDER_KNEE, abs=0.06)
    assert strength_knee == pytest.approx(0.50, abs=0.06)
    assert leak_knee >= 0.85
    assert live < strength_knee <= slider_knee <= leak_knee
    assert live < PROJECT_ALIGN_RECOMMENDED < leak_align
    # At the live gender number, project+hold is still a fail.
    live_row = next(r for r in rows if abs(r["align"] - live) < 1e-9)
    assert live_row["pass"] is False
    assert mismatch_verdicts(live_row)["slider"] == "needs_help"


def test_should_project_none_is_always_on():
    field = MismatchField2D()
    pos, neg, _ = field.rich_pair()
    should, align = lm_should_project_odd(pos, neg, field.declared_u, None)
    assert should is True
    assert float(align) == pytest.approx(LIVE_GENDER_V1_ALIGN, abs=1e-6)
    should_lo, _ = lm_should_project_odd(pos, neg, field.declared_u, 0.50)
    should_hi, _ = lm_should_project_odd(pos, neg, field.declared_u, 0.10)
    assert should_lo is False
    assert should_hi is True


def test_leak_cell_align_is_high():
    assert leak_cell_align() > 0.90
    field = Field2D()
    pos, neg = field.embed("energetic"), field.embed("calm")
    assert float(lm_odd_align(pos, neg, E_SLIDER)) > 0.90
