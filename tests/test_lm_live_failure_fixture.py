"""Translate the live energy-v14 failure into CPU fixture numbers."""

from __future__ import annotations

import pytest

from analysis.slider2d.live_failure import (
    HIGH_DIM,
    TINY_LEFTOVER,
    fixture_table,
    high_dim_synonym,
)


def rows() -> dict[str, dict]:
    return {row["name"]: row for row in fixture_table()}


def test_trainer_c_plus_and_slider_cos_are_distinct_first_class_metrics():
    table = rows()
    gender = table["gender_pair_odd_no_hold"]
    held = table["energy_2d_hold_l8"]
    for row in table.values():
        assert "trainer_c_plus" in row
        assert "slider_cos" in row
    assert gender["trainer_c_plus"] >= 0.97
    assert gender["slider_cos"] >= 0.97
    assert gender["collapse"] <= -0.90
    # A working hold should not look like the copied-pair baseline.
    assert held["slider_cos"] >= 0.98
    assert held["trainer_c_plus"] == pytest.approx(0.70, abs=0.02)
    assert held["perc"] == pytest.approx(0.72, abs=0.02)
    assert held["leftover"] == pytest.approx(0.15, abs=0.02)
    assert held["loss"] > 0.80


def test_high_dim_scalar_hold_can_select_non_bipolar_shared_response():
    row = rows()["highd_synonym_hold_l8"]
    assert row["dim"] == HIGH_DIM
    assert row["hold_norm"] == pytest.approx(TINY_LEFTOVER, rel=1e-4)
    assert row["effective_factor"] == pytest.approx(257.0)
    assert row["trainer_c_plus"] == pytest.approx(0.31, abs=0.02)
    assert row["slider_cos"] == pytest.approx(0.37, abs=0.02)
    assert row["collapse"] == pytest.approx(+0.18, abs=0.03)
    assert row["perc"] > 0.90
    assert row["loss"] > 2.0


def test_same_loudness_pin_does_not_remove_density_synonym_residual():
    synonym = rows()["highd_synonym_hold_l8"]
    pinned = rows()["highd_same_loudness_pin_l8"]
    assert pinned["raw_e_dot_u"] < synonym["raw_e_dot_u"]
    assert pinned["hold_norm"] > synonym["hold_norm"]
    # Orthogonalization followed by unit normalization leaves the same
    # density/genre/syntax residual and therefore the same bad optimum.
    for metric in (
        "trainer_c_plus",
        "slider_cos",
        "collapse",
        "leftover",
        "perc",
        "loss",
    ):
        assert pinned[metric] == pytest.approx(synonym[metric], abs=1e-5)


def test_leftover_only_weight_sweep_stays_bipolar_but_trades_fit_for_leak():
    table = rows()
    weak = table["leftover_only_hold_l0.3"]
    canary = table["leftover_only_hold_l1"]
    hard = table["leftover_only_hold_l8"]
    assert weak["leftover"] > canary["leftover"] > hard["leftover"]
    assert weak["loss"] < canary["loss"] < hard["loss"]
    assert all(row["collapse"] <= -0.999 for row in (weak, canary, hard))
    assert canary["trainer_c_plus"] == pytest.approx(0.94, abs=0.02)
    assert canary["loss"] == pytest.approx(0.48, abs=0.03)
    assert canary["leftover"] == pytest.approx(0.70, abs=0.03)
    assert hard["leftover"] <= 0.20


def test_pair_odd_sub_e_is_a_teacher_change_not_a_better_pair_odd_fit():
    table = rows()
    held = table["leftover_only_hold_l8"]
    sub = table["pair_odd_sub_e"]
    assert sub["teacher"] == "pair_odd_sub_e"
    assert sub["leftover"] == pytest.approx(0.0, abs=1e-7)
    assert sub["collapse"] <= -0.999
    assert sub["loss"] == pytest.approx(0.0, abs=1e-7)
    assert sub["pair_odd_mse"] > held["pair_odd_mse"]
    assert sub["trainer_c_plus"] == pytest.approx(0.58, abs=0.01)


def test_dimension_alone_is_not_claimed_to_break_polarity():
    # With the direct odd residual, even the hard λ=8 row is exactly bipolar.
    # The +0.18 row additionally declares its shared sign-asymmetric response
    # mode; that local-Jacobian assumption is the causal fixture ingredient.
    table = rows()
    assert table["energy_2d_hold_l8"]["collapse"] <= -0.999
    assert table["highd_synonym_hold_l8"]["collapse"] > 0.0
    assert high_dim_synonym("repeat", hold_weight=8.0)["collapse"] == pytest.approx(
        table["highd_synonym_hold_l8"]["collapse"], abs=1e-7
    )
