"""Hold-ê when ê overlaps the slider — live energy, not unused gender.

Existing live cells PASS because ê = unused. Energy-v4 leak captions
are energy. This cell makes ê·û a knob.
"""

from __future__ import annotations

import pytest
import torch

from analysis.slider2d.energy import EnergyLiveField2D, energy_row_aligns
from analysis.slider2d.field import E_ATTR, E_SLIDER
from analysis.slider2d.overlap import (
    leak_axis,
    leak_geometry,
    mean_odd_unit,
    score_overlap_policy,
)
from conceptmod.textsliders.slider_targets import (
    LEAK_HOLD_WEIGHT,
    lm_hold_dir,
    lm_unit,
)


def test_leak_axis_overlap_is_e_dot_u():
    for rho in (0.0, 0.3, 0.5, 0.7, 0.9, 1.0):
        axis = leak_axis(rho)
        assert float(axis @ E_SLIDER) == pytest.approx(rho, abs=1e-6)
        assert float(axis.norm()) == pytest.approx(1.0, abs=1e-6)
    same = leak_axis(0.7, leak_mode="same_energy")
    assert torch.allclose(same, E_ATTR)
    assert float(same @ E_SLIDER) == pytest.approx(0.0, abs=1e-6)


def test_pole_synonym_is_overlap_near_mean_align():
    field = EnergyLiveField2D()
    geo = leak_geometry(0.5, field=field, leak_mode="opposite")
    odd = mean_odd_unit(field)
    aligns = energy_row_aligns(field)
    assert sum(aligns) / len(aligns) == pytest.approx(0.58, abs=1e-6)
    assert geo["e_dot_odd"] == pytest.approx(1.0, abs=0.02)
    assert float(odd @ E_SLIDER) == pytest.approx(0.58, abs=0.02)
    assert not torch.allclose(lm_unit(field.odd(0)), field.declared_u, atol=1e-4)


def test_hold_dir_slider_drops_u_keeps_unused():
    unused = leak_axis(0.0)
    slider = leak_axis(1.0)
    mix = leak_axis(0.5)
    assert torch.allclose(lm_hold_dir(unused, slider_dir=E_SLIDER, mode="slider"), unused)
    assert lm_hold_dir(slider, slider_dir=E_SLIDER, mode="slider") is None
    held = lm_hold_dir(mix, slider_dir=E_SLIDER, mode="slider")
    assert held is not None
    assert abs(float(lm_unit(held) @ E_SLIDER)) < 1e-5
    assert float(lm_unit(held) @ E_ATTR) == pytest.approx(1.0, abs=1e-5)


def test_hold_dir_odd_is_noop_on_this_plane():
    """ê_⊥â ⊥ â, so the pair-odd teacher already has hold=0. Not a fix."""
    field = EnergyLiveField2D()
    odd = mean_odd_unit(field)
    for rho in (0.0, 0.5, 1.0):
        raw = leak_axis(rho)
        held = lm_hold_dir(raw, odd_dir=odd, mode="odd")
        if held is None:
            assert abs(float(lm_unit(raw) @ odd)) == pytest.approx(1.0, abs=1e-3)
            continue
        assert abs(float(lm_unit(held) @ odd)) < 1e-5


def test_unused_e_raw_hold_still_locks():
    row = score_overlap_policy(
        "unused", overlap=0.0, hold_weight=LEAK_HOLD_WEIGHT, ortho="raw", steps=200, seed=0
    )
    assert row["pass"] is True
    assert row["cos_intended"] > 0.90
    assert abs(row["leak_ratio"]) <= 0.20
    assert row["slider_collapsed"] is False


def test_slider_e_raw_hold_collapses_the_slider():
    """Live energy: ê ≈ loud. Hold punches û. Slider half does not lock."""
    row = score_overlap_policy(
        "slider_e", overlap=1.0, hold_weight=LEAK_HOLD_WEIGHT, ortho="raw", steps=200, seed=0
    )
    assert row["pass"] is False
    assert row["cos_intended"] < 0.30
    assert abs(row["leak_ratio"]) > 1.0
    assert row["loss"] > 0.20
    assert row["history"][0]["perc"] == pytest.approx(1.0, abs=0.05)
    assert row["history"][-1]["loss"] > 0.20
    # Pair-odd no-hold on the same poles is the ~0.02 locked-teacher band.
    base = score_overlap_policy("no_hold", overlap=1.0, hold_weight=0.0, steps=200, seed=0)
    assert base["loss"] < 0.05
    assert base["perc"] < 0.20
    assert row["loss"] > 5.0 * base["loss"]


def test_pole_synonym_raw_hold_is_the_v13_miss():
    """ê·û=0.5 is ê≈â on this field (energy-v4 opposite-energy leak)."""
    row = score_overlap_policy(
        "pole", overlap=0.5, hold_weight=LEAK_HOLD_WEIGHT, ortho="raw", steps=200, seed=0
    )
    assert row["e_dot_odd"] == pytest.approx(1.0, abs=0.02)
    assert row["pass"] is False
    assert row["perc"] > 0.80
    assert row["loss"] > 1.0
    # 2-D shrink stays parallel to the teacher (c+ high). Live high-D
    # zeros that residual and logs c+ ~0; perc / loss still miss here.
    assert row["cos_teacher"] > 0.50
    assert row["history"][-1]["perc"] > 0.80


def test_slider_ortho_locks_live_like_overlap():
    """The named trainer rule: hold ê_⊥ = ê − (ê·û)û."""
    for rho in (0.0, 0.3, 0.5, 0.7, 0.9):
        row = score_overlap_policy(
            f"perp_{rho}",
            overlap=rho,
            hold_weight=LEAK_HOLD_WEIGHT,
            ortho="slider",
            steps=200,
            seed=0,
        )
        assert row["pass"] is True, (rho, row["cos_intended"], row["leak_ratio"])
        assert abs(row["hold_dot_u"]) < 1e-5


def test_slider_ortho_cannot_invent_unused_when_e_is_u():
    row = score_overlap_policy(
        "perp_1", overlap=1.0, hold_weight=LEAK_HOLD_WEIGHT, ortho="slider", steps=200, seed=0
    )
    assert row["hold_off"] is True
    assert row["pass"] is False
    assert abs(row["leak_ratio"]) > 1.0
    assert row["cos_intended"] == pytest.approx(0.58, abs=0.05)


def test_odd_ortho_never_kills_leak():
    for rho in (0.0, 0.5, 1.0):
        row = score_overlap_policy(
            f"odd_{rho}",
            overlap=rho,
            hold_weight=LEAK_HOLD_WEIGHT,
            ortho="odd",
            steps=200,
            seed=0,
        )
        assert row["pass"] is False
        assert abs(row["leak_ratio"]) > 1.0


def test_same_energy_leak_is_unused_e():
    row = score_overlap_policy(
        "same",
        overlap=0.9,
        hold_weight=LEAK_HOLD_WEIGHT,
        leak_mode="same_energy",
        ortho="raw",
        steps=200,
        seed=0,
    )
    assert row["e_dot_u"] == pytest.approx(0.0, abs=1e-6)
    assert row["pass"] is True
    assert abs(row["leak_ratio"]) <= 0.20


def test_gated_project_locks_but_is_not_the_default():
    row = score_overlap_policy(
        "v12",
        overlap=1.0,
        hold_weight=1.0,
        project_odd=True,
        project_align_min=0.50,
        steps=200,
        seed=0,
    )
    assert row["pass"] is True
    assert abs(row["leak_ratio"]) < 0.05
    assert row["cos_intended"] > 0.90
    # Teacher is (a·û)û, so c+ vs pair-odd is the leftover align, not ~1.
    assert row["cos_teacher"] < 0.70


def test_soft_lambda_is_not_a_high_overlap_fix():
    weak = score_overlap_policy(
        "soft", overlap=0.7, hold_weight=1.0, ortho="raw", steps=200, seed=0
    )
    assert weak["pass"] is False
    assert weak["cos_intended"] < 0.50
    assert abs(weak["leak_ratio"]) > 0.20
