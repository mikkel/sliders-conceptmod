"""Faithful / v6 cannot stay faithful and kill leak on the leaky 2-D pair."""

from __future__ import annotations

import pytest
import torch

from analysis.slider2d.faithful import (
    HOLD_LAMBDAS,
    POLE_COS_MIN,
    leak_field_table,
    leak_teacher,
    hold_e_shrink,
    score_energy_faithful,
    score_energy_odd,
    score_leak_lm,
    score_mismatch_faithful,
    teacher_geometry,
)
from analysis.slider2d.field import E_ATTR, Field2D
from conceptmod.textsliders.slider_targets import lm_hidden_targets


_TABLE = None
_GEO = None


def table():
    global _TABLE
    if _TABLE is None:
        _TABLE = {r["name"]: r for r in leak_field_table(steps=200, seed=0)}
    return _TABLE


def geo():
    global _GEO
    if _GEO is None:
        _GEO = teacher_geometry(*leak_teacher(Field2D()))
    return _GEO


def test_faithful_teacher_is_raw_poles():
    pos, neg, neu = leak_teacher()
    plus, minus = lm_hidden_targets(pos, neg, neu, target_mode="faithful")
    assert torch.allclose(plus, pos)
    assert torch.allclose(minus, neg)


def test_common_beta_does_not_change_faithful_teacher():
    pos, neg, neu = leak_teacher()
    a, b = lm_hidden_targets(pos, neg, neu, target_mode="faithful", common_beta=1.0)
    c, d = lm_hidden_targets(pos, neg, neu, target_mode="faithful", common_beta=0.0)
    assert torch.allclose(a, c)
    assert torch.allclose(b, d)


def test_ungated_even_mode_is_parallel_to_leak_axis():
    g = geo()
    assert g["even_is_parallel_e"]
    assert abs(g["teacher_even_e"]) > 0.50
    assert abs(g["teacher_odd_e"]) > 0.20
    assert g["teacher_leak"] > 1.0
    assert g["r_raw"] > -0.50


def test_raw_faithful_copies_poles_and_leaks():
    raw = table()["lm_faithful_raw"]
    assert raw["teacher_faithful"]
    assert raw["faithful_fit"]
    assert raw["axis"]["leak"] == "needs_help"
    assert raw["axis"]["collapse"] == "needs_help"
    assert raw["leak_ratio"] == pytest.approx(geo()["teacher_leak"], abs=0.05)
    assert raw["pole_cos_plus"] > POLE_COS_MIN
    assert raw["e_copied_frac"] == pytest.approx(1.0, abs=0.05)
    assert not raw["wins_while_faithful"]


def test_hold_shrinks_teacher_e_not_a_new_teacher():
    g = geo()
    h8 = table()["lm_faithful_hold_l8"]
    h32 = table()["lm_faithful_hold_l32"]
    assert h8["teacher_faithful"] and h32["teacher_faithful"]
    assert h8["student_e_plus"] == pytest.approx(hold_e_shrink(g["teacher_e_plus"], 8.0), abs=0.05)
    assert h32["student_e_plus"] == pytest.approx(hold_e_shrink(g["teacher_e_plus"], 32.0), abs=0.05)
    assert h8["leak_ratio"] == pytest.approx(hold_e_shrink(g["teacher_leak"], 8.0), abs=0.04)
    assert h8["e_copied_frac"] == pytest.approx(1.0 / 9.0, abs=0.05)
    assert not h8["faithful_fit"]
    assert not h32["faithful_fit"]


def test_faithful_hold_fights_harder_than_pair_odd_hold():
    h8 = table()["lm_faithful_hold_l8"]
    v9 = table()["lm_v9"]
    # Same λ, same ê. Faithful still asks for even ê, so leftover leak is larger.
    assert abs(h8["leak_ratio"]) > abs(v9["leak_ratio"]) + 0.05
    assert abs(v9["leak_ratio"]) <= 0.20
    assert v9["axis"]["slider"] == v9["axis"]["leak"] == v9["axis"]["collapse"] == "right"
    assert not v9["teacher_faithful"]


def test_no_leaky_faithful_knob_wins_while_remaining_faithful():
    rows = table()
    leaky = [
        name
        for name, row in rows.items()
        if row["teacher_faithful"] and "attrs" not in name
    ]
    assert leaky
    assert all(not rows[name]["wins_while_faithful"] for name in leaky)
    # Passing the leak gate (high λ) means the poles were not copied.
    assert rows["lm_faithful_hold_l32"]["axis"]["leak"] == "right"
    assert rows["lm_faithful_hold_l32"]["pole_cos_plus"] < POLE_COS_MIN


def test_attributes_are_the_faithful_data_fix():
    attrs = table()["lm_faithful_attrs"]
    assert attrs["teacher_faithful"]
    assert attrs["faithful_fit"]
    assert attrs["wins_while_faithful"]
    assert attrs["axis"]["slider"] == attrs["axis"]["leak"] == attrs["axis"]["collapse"] == "right"
    assert abs(attrs["leak_ratio"]) < 0.05
    assert attrs["cos_plus_minus"] < -0.85
    assert attrs["e_copied_frac"] == pytest.approx(1.0, abs=0.05)


def test_hub_anchor_on_faithful_still_leaks():
    hub = table()["lm_faithful_hub"]
    raw = table()["lm_faithful_raw"]
    assert hub["teacher_faithful"]
    assert abs(hub["leak_ratio"]) > 0.20
    # Anchor is even blend-back, not an ê kill. Leak stays in the raw-pole ballpark.
    assert abs(hub["leak_ratio"]) == pytest.approx(abs(raw["leak_ratio"]), abs=0.35)


def test_nonfaithful_comparables_match_known_geometry():
    rows = table()
    assert not rows["lm_symmetric"]["teacher_faithful"]
    assert not rows["lm_v9_hub"]["teacher_faithful"]
    assert not rows["lm_v9_project"]["teacher_faithful"]
    assert rows["lm_symmetric"]["axis"]["collapse"] == "right"
    assert rows["lm_symmetric"]["axis"]["leak"] == "needs_help"
    assert abs(rows["lm_v9_hub"]["leak_ratio"]) > 0.20
    assert abs(rows["lm_v9_project"]["leak_ratio"]) < 0.05
    assert abs(rows["lm_v9"]["leak_ratio"]) < 0.05


def test_leaky_pole_cannot_be_copied_and_leak_right():
    """Any residual with leak ≤ 0.20 is far from the faithful +1 pole."""
    g = geo()
    teacher = torch.tensor([g["teacher_slider_plus"], g["teacher_e_plus"]])
    student = torch.tensor([g["teacher_slider_plus"], 0.20 * abs(g["teacher_slider_plus"])])
    cos = float(
        torch.nn.functional.cosine_similarity(teacher.unsqueeze(0), student.unsqueeze(0))
    )
    assert g["teacher_leak"] > 1.0
    assert cos < POLE_COS_MIN


def test_energy_faithful_equals_pair_odd_when_poles_are_already_odd():
    faith = score_energy_faithful(hold_weight=0.0, steps=120, seed=0)
    odd = score_energy_odd(hold_weight=0.0, steps=120, seed=0)
    assert faith["leak_ratio"] == pytest.approx(odd["leak_ratio"], abs=0.05)
    assert faith["cos_plus_minus"] == pytest.approx(odd["cos_plus_minus"], abs=0.05)
    assert abs(faith["leak_ratio"]) > 0.20
    fh = score_energy_faithful(hold_weight=8.0, steps=120, seed=0)
    oh = score_energy_odd(hold_weight=8.0, steps=120, seed=0)
    assert fh["leak_ratio"] == pytest.approx(oh["leak_ratio"], abs=0.08)
    assert abs(fh["leak_ratio"]) <= 0.20


def test_mismatch_clean_pair_faithful_already_works():
    row = score_mismatch_faithful(steps=120, seed=0)
    assert row["pass"]
    assert abs(row["leak_ratio"]) < 0.05
    assert row["cos_plus_minus"] < -0.85
    assert row["faithful_fit"]


def test_hold_grid_covers_the_asked_lambdas():
    assert HOLD_LAMBDAS == (0.0, 1.0, 4.0, 8.0, 16.0, 32.0, 64.0)
    raw = score_leak_lm("probe", target_mode="faithful", hold_weight=0.0, steps=40, seed=0)
    assert raw["name"] == "probe"
