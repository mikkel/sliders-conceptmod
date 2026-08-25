"""Live Music 3 hold-failure signatures on the CPU fixture.

Analysis / test / docs only. Default --lm_target v9 must stay v9.
"""

from __future__ import annotations

import pytest
import torch

from analysis.slider2d.mismatch import MismatchField2D
from analysis.slider2d.signature import (
    HIGH_D,
    LIVE_CAPTION_AXIS,
    HighDEnergyField,
    hold_working_not_v12,
    looks_like_v12,
    score_gender_like,
    score_highd,
    score_leftover_lambda,
    score_ortho_perp_not_v12,
    score_pair_odd_sub_e,
    short_u_at_caption_axis,
    shrink_factor,
    signature_table,
)
from conceptmod.textsliders.slider_targets import LEAK_HOLD_WEIGHT, lm_hold_dir, lm_unit
from conceptmod.textsliders.train_lm_slider_music3 import (
    parse_args,
    resolve_lm_loss_weights,
    resolve_lm_recipe,
)


_TABLE = None


def table():
    global _TABLE
    if _TABLE is None:
        _TABLE = {r["name"]: r for r in signature_table(steps=200, seed=0)}
    return _TABLE


def test_live_default_is_still_v9():
    args = parse_args(["--prompts_file", "prompts.yaml"])
    assert args.lm_target == "v9"
    assert resolve_lm_recipe(lm_target="v9", symmetric=True) == "v9"
    hold, _ = resolve_lm_loss_weights(
        "v9", hold_weight=None, anchor_weight=None, leak_declared=True
    )
    assert hold == pytest.approx(LEAK_HOLD_WEIGHT)


def test_gender_like_is_copied_pair_odd_without_junk_e():
    row = table()["gender_like_hold0"]
    assert row["hold_weight"] == 0.0
    assert row["leak_kind"] == "none"
    assert row["cos_teacher"] > 0.95
    assert row["cos_slider"] > 0.95
    assert row["collapse"] <= -0.90
    assert row["loss"] < 0.05
    assert looks_like_v12(row)
    assert not hold_working_not_v12(row)
    # Do not invent junk ê on the gender-like cell.
    field = MismatchField2D()
    assert torch.allclose(field.odd()[1:], torch.zeros_like(field.odd()[1:]), atol=1e-6)


def test_ortho_perp_pass_leftover_fail_v12_looking():
    row = table()["energy_perp_l8"]
    assert row["cos_slider"] > 0.90
    assert abs(row["leak_ratio"]) <= 0.20
    assert row["cos_teacher"] == pytest.approx(0.70, abs=0.08)
    assert row["perc"] == pytest.approx(0.72, abs=0.08)
    assert row["loss"] > 0.50
    assert row["collapse"] <= -0.85
    assert row["pass"] is True
    assert hold_working_not_v12(row)
    assert not looks_like_v12(row)
    copied = table()["pair_odd_no_hold"]
    assert looks_like_v12(copied)
    assert copied["cos_teacher"] > 0.95
    assert copied["loss"] < 0.05
    assert abs(copied["leak_ratio"]) > 1.0


def test_highd_raw_synonym_matches_closed_form_and_undefines_polarity():
    row = table()["highd_synonym_raw_l8"]
    closed = shrink_factor(HIGH_D, LEAK_HOLD_WEIGHT)
    assert row["strength"] == pytest.approx(closed, abs=0.01)
    assert row["cos_teacher"] > 0.95
    assert row["perc"] > 0.90
    assert row["loss"] > 0.05
    assert row["polarity_undefined"] is True
    assert row["collapse_live"] == pytest.approx(0.0, abs=1e-6)
    # Linear odd residual of εa and −εa is still antipodal.
    assert row["collapse"] <= -0.85
    assert not looks_like_v12(row)


def test_synonym_pin_caption_axis_still_fights_teacher():
    row = table()["highd_pin_caption037_l8"]
    assert row["caption_axis"] == pytest.approx(LIVE_CAPTION_AXIS, abs=0.02)
    assert abs(row["e_dot_a"]) > 0.95
    assert abs(row.get("hold_dot_a", 0.0)) > 0.80
    assert row["cos_teacher"] < 0.50
    assert row["perc"] > 0.80
    assert row["loss"] > 0.05
    assert not looks_like_v12(row)
    assert not hold_working_not_v12(row)


def test_tiny_unused_perp_is_unit_normalized_leftover_hold():
    tiny = table()["highd_tiny_unused_slider_l8"]
    unused = table()["highd_unused_slider_l8"]
    assert tiny["cos_slider"] > 0.90
    assert unused["cos_slider"] > 0.90
    assert abs(tiny["leak_ratio"] - unused["leak_ratio"]) < 0.05
    # Tiny leftover after ortho is not a weaker hold — unit-normalize restores it.


def test_leftover_only_lambda_sweep_vs_pair_odd_sub_e():
    weak = table()["leftover_only_l0.3"]
    canary = table()["leftover_only_l1"]
    locked = table()["leftover_only_l8"]
    sub = table()["pair_odd_sub_e"]
    assert canary["collapse"] <= -0.85
    assert canary["cos_teacher"] > 0.90
    assert canary["loss"] < 1.0
    assert abs(canary["leak_ratio"]) > 0.20
    assert not looks_like_v12(canary)
    assert locked["collapse"] <= -0.85
    assert abs(locked["leak_ratio"]) <= 0.20
    assert hold_working_not_v12(locked)
    assert abs(sub["leak_ratio"]) <= 0.05
    assert sub["cos_slider"] > 0.90
    assert sub["collapse"] <= -0.85
    # Teacher changed: c+ vs full pair-odd is leftover align, not ~0.97.
    assert sub["cos_teacher"] == pytest.approx(0.58, abs=0.05)
    assert not hold_working_not_v12(sub)
    assert abs(canary["leak_ratio"]) > abs(sub["leak_ratio"])
    assert abs(weak["leak_ratio"]) > abs(canary["leak_ratio"])


def test_caption_axis_construction():
    field = HighDEnergyField()
    e = lm_unit(field.odd())
    u = short_u_at_caption_axis(e, axis=LIVE_CAPTION_AXIS, junk=field.extra(2))
    assert float(e @ u) == pytest.approx(LIVE_CAPTION_AXIS, abs=1e-5)
    held = lm_hold_dir(e, slider_dir=u, mode="slider")
    assert held is not None
    assert abs(float(lm_unit(held) @ u)) < 1e-5
    assert abs(float(lm_unit(held) @ e)) > 0.90


def test_shrink_factor_2d_vs_highd():
    assert shrink_factor(2, 8.0) == pytest.approx(1.0 / 9.0)
    assert shrink_factor(32, 8.0) == pytest.approx(1.0 / 129.0)
    assert shrink_factor(32, 8.0) < 0.02


def test_direct_scorers_agree_with_table():
    gender = score_gender_like(steps=80, seed=0)
    assert gender["cos_teacher"] > 0.90
    perp = score_ortho_perp_not_v12(steps=80, seed=0)
    assert perp["cos_slider"] > 0.80
    raw = score_highd(
        "raw", leak_kind="synonym_a", hold_weight=8.0, ortho="raw", steps=80, seed=0
    )
    assert raw["polarity_undefined"] is True
    leftover = score_leftover_lambda(1.0, steps=80, seed=0)
    assert leftover["collapse"] <= -0.85
    sub = score_pair_odd_sub_e(steps=80, seed=0)
    assert abs(sub["leak_ratio"]) < 0.05
