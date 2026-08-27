"""Live Music 3 failure signatures on the CPU 2-D / high-D fixture."""

from __future__ import annotations

import pytest

from analysis.slider2d.live_signature import (
    ENERGY_CPLUS_HOLD_WORKING,
    ENERGY_CPLUS_V12_LIKE,
    ENERGY_LEAK_LOCK,
    ENERGY_LOSS_FAIL,
    ENERGY_PERC_FAIL,
    ENERGY_SLIDER_LOCK,
    GENDER_COLLAPSE_MAX,
    GENDER_CPLUS_MIN,
    GenderLiveField2D,
    HighDLeakField,
    LEFTOVER_CANARY_LAM,
    LEFTOVER_CANARY_LEAK_MIN,
    hold_e_perp_norm,
    score_energy_perp_synonym,
    score_gender_pair_odd,
    score_high_d_tiny_eperp,
    score_leftover_only,
    score_pair_odd_sub_e,
    score_synonym_pin,
    signature_table,
    verdict_table,
)
from analysis.slider2d.overlap import leak_axis
from analysis.slider2d.energy import EnergyLiveField2D
from conceptmod.textsliders.slider_targets import LEAK_HOLD_WEIGHT, lm_hold_dir


_TABLE = None


def table():
    global _TABLE
    if _TABLE is None:
        _TABLE = signature_table(steps=200, seed=0)
    return _TABLE


def by_name():
    return {r["name"]: r for r in table()}


def test_gender_pair_odd_looks_like_v12_copy():
    row = by_name()["gender_pair_odd"]
    assert row["hold_weight"] == 0.0
    assert row["cos_teacher"] >= GENDER_CPLUS_MIN
    assert row["cos_slider"] >= GENDER_CPLUS_MIN
    assert row["collapse"] <= GENDER_COLLAPSE_MAX
    assert row["v12_looking"] is True
    assert row["hold_working_not_v12"] is False


def test_energy_perp_synonym_passes_leftover_fails_v12_look():
    row = by_name()["energy_pole_synonym_perp"]
    assert row["cos_slider"] >= ENERGY_SLIDER_LOCK
    assert abs(row["leak"]) <= ENERGY_LEAK_LOCK
    assert row["cos_teacher"] < ENERGY_CPLUS_V12_LIKE
    assert row["cos_teacher"] == pytest.approx(0.70, abs=0.08)
    assert row["perc"] >= ENERGY_PERC_FAIL
    assert row["loss"] >= ENERGY_LOSS_FAIL
    assert row["collapse"] <= GENDER_COLLAPSE_MAX
    assert row["hold_working_not_v12"] is True
    assert row["v12_looking"] is False


def test_c_plus_and_slider_cos_are_first_class_columns():
    for row in table():
        assert "cos_slider" in row
        assert "cos_teacher" in row
        assert "collapse" in row
        assert "hold_e_perp_norm" in row


def test_high_d_tiny_eperp_shrinks_residual():
    row = by_name()["high_d_tiny_eperp"]
    assert row["dims"] >= 32
    assert row["hold_e_perp_norm"] < 0.15
    assert row["cos_slider"] >= ENERGY_SLIDER_LOCK
    assert row["cos_teacher"] < by_name()["gender_pair_odd"]["cos_teacher"]
    assert row["cos_teacher"] <= ENERGY_CPLUS_HOLD_WORKING + 0.05
    # Symmetric odd_even fixture: polarity stays antipodal (live +0.18 not here).
    assert row["collapse"] <= GENDER_COLLAPSE_MAX


def test_synonym_pin_raw_hold_still_fights():
    row = by_name()["synonym_pin_medium_energy"]
    field = EnergyLiveField2D()
    raw = leak_axis(0.95, slider=field.intended, unused=field.unused)
    assert hold_e_perp_norm(raw, slider_dir=field.intended) < 0.35
    assert row["loss"] >= ENERGY_LOSS_FAIL
    assert row["cos_slider"] < ENERGY_SLIDER_LOCK
    assert row["v12_looking"] is False


def test_leftover_only_lambda_sweep():
    l03 = by_name()["leftover_only_l0.3"]
    l1 = by_name()["leftover_only_l1"]
    l8 = by_name()["leftover_only_l8"]
    assert abs(l03["leak"]) > LEFTOVER_CANARY_LEAK_MIN
    assert l1["collapse"] <= GENDER_COLLAPSE_MAX
    assert l1["loss"] < 1.0
    assert abs(l1["leak"]) > LEFTOVER_CANARY_LEAK_MIN
    assert abs(l8["leak"]) <= ENERGY_LEAK_LOCK
    assert l8["cos_slider"] >= ENERGY_SLIDER_LOCK


def test_leftover_only_l1_is_canary_before_sub_e():
    l1 = by_name()["leftover_only_l1"]
    sub = by_name()["pair_odd_sub_e"]
    assert l1["hold_weight"] == LEFTOVER_CANARY_LAM
    assert abs(l1["leak"]) > abs(sub["leak"])
    assert sub["cos_teacher"] < ENERGY_CPLUS_V12_LIKE
    assert abs(sub["leak"]) <= ENERGY_LEAK_LOCK


def test_pair_odd_sub_e_zeros_unused_not_live_default():
    row = by_name()["pair_odd_sub_e"]
    assert row["hold_weight"] == 0.0
    assert abs(row["leak"]) <= ENERGY_LEAK_LOCK
    assert row["cos_slider"] >= ENERGY_SLIDER_LOCK


def test_gender_live_field_has_no_leak_axis():
    field = GenderLiveField2D()
    pos, neg = field.embed(field.POLE_POS), field.embed(field.POLE_NEG)
    odd = (pos - neg) / 2.0
    assert float(odd @ field.intended) / odd.norm() == pytest.approx(1.0, abs=1e-6)


def test_high_d_field_leak_e_is_mostly_slider():
    field = HighDLeakField()
    assert float(field.leak_e @ field.intended) > 0.95
    held = lm_hold_dir(field.leak_e, slider_dir=field.intended, mode="slider")
    assert held is not None
    assert float(held.norm()) < 0.15


def test_verdict_table_covers_live_bullets():
    verdicts = verdict_table(table())
    fixtures = {v["fixture"] for v in verdicts}
    assert "gender_pair_odd" in fixtures
    assert "energy_pole_synonym_perp" in fixtures
    assert "high_d_tiny_eperp" in fixtures
    assert "leftover_only_l1" in fixtures
    high_d = next(v for v in verdicts if v["fixture"] == "high_d_tiny_eperp")
    assert high_d["visible"] == "partial"
