"""Live energy-v14 signature on the CPU fixture.

Three live facts the existing 2-D cells could not show:

1. trainer c+ (cos with pair-odd ``a``) is not slider lock. Gender's
   0.97 c+ and leaky-energy-no-hold's 0.95 c+ look identical; the
   working hold prints c+ ~0.70 and a non-locked loss.
2. live is not 2-D: the held component of a linear fit is
   ``t_e/(1+λD/2)``, and ê_⊥ is one direction out of a huge leftover —
   λ=8 at live D is a ~4000× annihilation that barely cures the leak.
3. the ±1 polarity break (live collapse +0.18): provably invisible to
   any linear residual; the curved student reproduces it.
"""

from __future__ import annotations

import pytest
import torch

from analysis.slider2d.overlap import score_overlap_policy
from analysis.slider2d.signature import (
    LEAK_HOLD_WEIGHT,
    SIG_DIM,
    SignatureField,
    assert_signature_geometry,
    linear_shrink_factor,
    measured_shrink_factor,
    signature_table,
    train_signature,
)


@pytest.fixture(scope="module")
def table():
    rows = signature_table(seed=0)
    return {r["name"]: r for r in rows}


def test_signature_geometry_is_live_not_cheat():
    assert_signature_geometry()
    field = SignatureField()
    # Probe aligns are the live 0.48/0.68 shrunk by the heard split, never ~1.
    geo = field.leak_geometry("opposite")
    assert geo["e_perp_dot_ahat"] > 0.60  # synonym in disguise
    assert geo["e_dot_u"] > 0.5
    pinned = field.leak_geometry("pinned")
    # The medium-energy pin shrinks ê·û but ê_⊥ still overlaps the pair-odd.
    assert pinned["e_dot_u"] < geo["e_dot_u"]
    assert pinned["e_perp_dot_ahat"] > 0.45
    left = field.leak_geometry("leftover_only")
    assert abs(left["e_dot_u"]) < 1e-6
    assert left["e_perp_dot_ahat"] > 0.5  # it is the heard leak inside a


def test_gender_like_field_refuses_junk_e():
    gender = SignatureField(gender_like=True)
    with pytest.raises(ValueError):
        gender.leak_axis("opposite")
    assert gender.leak_axis("none") is None


def test_hold_stiffness_scales_with_dim():
    """MSE is a mean over D; the hold is a unit-ê dot. s_e = t_e/(1+λD/2)."""
    assert linear_shrink_factor(2, 8.0) == pytest.approx(1.0 / 9.0)
    assert linear_shrink_factor(SIG_DIM, 8.0) == pytest.approx(1.0 / 4097.0)
    assert measured_shrink_factor(2, 8.0) == pytest.approx(1.0 / 9.0, abs=1e-4)
    assert measured_shrink_factor(SIG_DIM, 8.0) == pytest.approx(1.0 / 4097.0, abs=1e-4)
    # λ=1 and even λ=0.3 still annihilate the held direction at live D.
    assert measured_shrink_factor(SIG_DIM, 1.0) < 0.01
    assert measured_shrink_factor(SIG_DIM, 0.3) < 0.01


def test_gender_and_leaky_no_hold_look_identical(table):
    """The v12-looking train: c+ high, collapse ~-1, tiny loss — on BOTH.

    Gender's look is right (a is the singer). Energy's identical look
    carries the heard leak. c+ cannot tell them apart.
    """
    gender = table["gender_like_linear"]
    energy = table["energy_no_hold_linear"]
    for row in (gender, energy):
        assert row["looks_like_v12"] is True
        assert row["cos_teacher"] >= 0.94
        assert row["collapse"] <= -0.99
        assert row["perc"] <= 0.40
    assert abs(gender["leak_ratio"]) < 0.05
    assert energy["leak_ratio"] > 1.0  # same look, heard leak on board
    # Slider-cos vs the declared probe: high c+ does not mean û lock.
    assert gender["slider_cos"] == pytest.approx(0.20, abs=0.02)
    assert energy["slider_cos"] < 0.60


def test_gender_curved_matches_live_healthy_band(table):
    row = table["gender_like_curved"]
    assert row["cos_teacher"] >= 0.95
    assert row["collapse"] <= -0.90
    assert row["loss"] < 0.01
    assert abs(row["leak_ratio"]) <= 0.20


def test_working_hold_2d_does_not_look_like_v12():
    """The existing 2-D PASS (ρ=0.5, ê_⊥û, λ=8) is c+ ~0.70 / perc ~72%.

    Hold is *supposed* to make fit-to-pair-odd worse. Reading trainer c+
    as slider lock calls this run broken; slider lock is cos with û.
    """
    row = score_overlap_policy(
        "pole_synonym_slider_l8",
        overlap=0.5,
        hold_weight=LEAK_HOLD_WEIGHT,
        ortho="slider",
        steps=200,
        seed=0,
    )
    assert row["pass"] is True
    assert row["cos_intended"] > 0.95           # slider locked
    assert abs(row["leak_ratio"]) <= 0.20        # leftover held
    assert row["cos_teacher"] == pytest.approx(0.70, abs=0.05)
    assert row["perc"] == pytest.approx(0.72, abs=0.05)
    assert row["loss"] > 0.5                     # never the 0.02 band
    assert row["cos_teacher"] < 0.90             # will never look like v12


def test_highd_e_perp_hold_barely_cures_the_leak(table):
    """2-D locked leftover at 0.154 because the leftover was 1-D. At live D
    the same recipe kills one direction of a (D-2)-dim leftover: leak stays."""
    row = table["synonym_perp_l8_linear"]
    assert row["leak_ratio"] > 0.8               # nothing like the 2-D 0.154
    assert row["cos_teacher"] < 0.75
    assert row["perc"] > 0.6
    no_hold = table["energy_no_hold_linear"]
    assert row["loss"] > 4.0 * no_hold["loss"]   # loss stuck off the band


def test_linear_student_can_never_break_polarity(table):
    """Separability: with symmetric targets the even weight gets zero
    gradient — under the identical λ=8 fight, collapse is -1 to machine
    precision. The live +0.18 is invisible to every linear-residual cell."""
    for name in ("synonym_perp_l8_linear", "energy_no_hold_linear", "gender_like_linear"):
        row = table[name]
        assert row["w_even_norm"] < 1e-8
        assert row["collapse"] == pytest.approx(-1.0, abs=1e-4)
        assert row["collapse_late_max"] == pytest.approx(-1.0, abs=1e-4)
        assert row["polarity_broken"] is False


def test_curved_synonym_fight_breaks_polarity(table):
    """The live v14 shape: collapse not -1 (late max ~+0.23 vs live +0.18),
    c+ low, perc ~100%, loss exploding early (live: 278 by step 13)."""
    row = table["synonym_perp_l8_curved"]
    assert row["polarity_broken"] is True
    assert row["collapse_late_max"] > 0.0
    assert row["collapse"] > -0.70
    assert row["cos_teacher"] < 0.60
    assert row["perc"] > 0.70
    assert row["loss_spike"] > 100.0
    # Same field, same fight, linear student: unbroken. Curvature is the
    # missing ingredient, not the geometry.
    assert table["synonym_perp_l8_linear"]["polarity_broken"] is False


def test_medium_energy_pin_does_not_save_lambda8(table):
    """Same-loudness rewrite: ê·û drops but ê_⊥ is still a pole synonym.
    λ=8 still fights a and still breaks ±1 (live: loss 278 by step 13)."""
    row = table["pinned_perp_l8_curved"]
    assert row["polarity_broken"] is True
    assert row["cos_teacher"] < 0.75
    assert row["loss_max"] > 1.0


def test_leftover_only_lambda1_is_trainable(table):
    """The proposed canary: leftover-only ê (no density/loudness words) at
    λ=1 keeps ±1 bipolar and a trainable loss; leak survives the live band."""
    row = table["leftover_only_l1_curved"]
    assert row["polarity_broken"] is False
    assert row["collapse"] <= -0.85
    assert row["loss"] < 0.01
    assert row["leak_ratio"] > 0.20              # leak does not fully clear
    assert row["leak_ratio"] < 0.6
    soft = table["leftover_only_l0.3_curved"]
    assert soft["polarity_broken"] is False
    assert soft["collapse"] <= -0.95


def test_leftover_only_lambda8_still_breaks_at_live_dim(table):
    """λ=8 is too stiff at live D even when ê is genuinely unused: the
    unit-normalized hold is λ·D/2 ≈ 4000× the MSE and the fight itself
    breaks ±1. λ is not portable across D."""
    row = table["leftover_only_l8_curved"]
    assert row["polarity_broken"] is True
    assert row["loss_max"] > 1.0


def test_pair_odd_sub_e_removes_the_same_leak_without_the_fight(table):
    """Teacher change: subtract ê_⊥û from a. Same leak reduction as the
    hold, loss in the trainable band, ±1 exactly bipolar, no spike."""
    sub = table["sub_e_leftover_curved"]
    hold = table["leftover_only_l1_curved"]
    assert sub["polarity_broken"] is False
    assert sub["collapse"] <= -0.99
    assert sub["collapse_late_max"] <= -0.99
    assert sub["loss"] < 0.01
    assert sub["loss_max"] < 0.05                # no explosion at any step
    assert abs(sub["leak_ratio"] - hold["leak_ratio"]) < 0.10
    assert sub["intended_cos"] > 0.90            # heard slider intact
    lin = table["sub_e_leftover_linear"]
    assert lin["collapse"] == pytest.approx(-1.0, abs=1e-4)
    assert lin["loss_max"] < 0.05


def test_sub_e_with_synonym_e_punches_the_slider(table):
    """sub_e is only as good as ê: the energy-v4 opposite-energy captions
    (even after ⊥û) subtract heard loudness out of the teacher."""
    row = table["sub_e_synonym_curved"]
    assert row["intended_cos"] < 0.60            # heard slider punched
    assert row["leak_ratio"] > 1.0


def test_history_carries_the_live_log_columns(table):
    row = table["synonym_perp_l8_curved"]
    hist = row["history"]
    assert hist[0]["step"] == 0
    for key in ("loss", "cos_teacher", "collapse", "perc", "slider_cos"):
        assert key in hist[0]
    # The explosion is early, like the live step-13 blowup.
    early = [h for h in hist if 0 < h["step"] <= 12]
    assert max(h["loss_train"] for h in early) > 1.0


def test_default_hold_weight_unchanged():
    """This is analysis only: the live default stays λ=8 on declared ê."""
    assert LEAK_HOLD_WEIGHT == 8.0


def test_train_signature_seed_variation_keeps_verdicts():
    """The polarity verdicts are not a seed fluke (spot check one more seed)."""
    energy = SignatureField()
    broken = train_signature(
        "syn_seed1", field=energy, student="curved", leak="opposite",
        hold_weight=8.0, seed=1,
    )
    assert broken["polarity_broken"] is True
    ok = train_signature(
        "sub_seed1", field=energy, student="curved", leak="leftover_only",
        subtract_e=True, seed=1,
    )
    assert ok["polarity_broken"] is False
    assert ok["collapse"] <= -0.99
