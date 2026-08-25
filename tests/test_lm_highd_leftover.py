"""High-D hold-ê: stiffness, ê wording, trainer c+ vs slider-cos, ±1.

Live energy-v14 held ê_⊥û at λ=8 and logged trainer c+ 0.31, collapse
+0.18, caption-axis gate 0.37. These cells split that into the parts a
2-D fixture can carry (c+ / perc / leak / loss, all closed form in
``p`` and ``λ·D/2``) and the part it cannot (±1 polarity, which needs a
non-mirror ±1 reply).
"""

from __future__ import annotations

import pytest
import torch

from analysis.slider2d.faithful import hold_e_shrink
from analysis.slider2d.highd import (
    BEND_ENERGY,
    BEND_GENDER,
    HOLD_EXPLAINS_TOL,
    LIVE_COLLAPSE,
    LIVE_C_PLUS,
    LIVE_GATE_ALIGN,
    LIVE_GENDER_COLLAPSE,
    LIVE_GENDER_C_PLUS,
    LIVE_GENDER_LOSS,
    LIVE_SPIKE_LOSS,
    HighDLeakField,
    bend_for_collapse,
    bend_sweep,
    calibrate_bend,
    cell_table,
    energy_2d_field,
    energy_field,
    gender_like_field,
    hold_cover,
    hold_direction,
    hold_predictions,
    hold_shrink,
    hold_spike,
    hold_split,
    lambda_dim_sweep,
    lambda_eff,
    lambda_fit_sweep,
    leak_axis,
    leftover_only_e,
    live_v14_analogue,
    match_sweep,
    medium_pin_e,
    polarity_grid,
    score_highd,
    synonym_cover,
    synonym_e,
    teacher_poles,
)
from analysis.slider2d.overlap import score_overlap_policy
from conceptmod.textsliders.slider_targets import (
    LEAK_HOLD_WEIGHT,
    lm_axis_hold,
    lm_unit,
)


STEPS = 400


@pytest.fixture(scope="module")
def cells():
    return {row["name"]: row for row in cell_table(steps=STEPS, seed=0)}


def test_hold_shrink_is_the_2d_formula_at_dim_2():
    for lam in (0.0, 0.3, 1.0, 8.0, 32.0):
        assert hold_shrink(lam, 2) == pytest.approx(hold_e_shrink(1.0, lam), abs=1e-12)
        assert lambda_eff(lam, 2) == pytest.approx(lam)
    # The live width is where the two disagree: mse_loss averages, the hold does not.
    assert hold_shrink(8.0, 1024) == pytest.approx(1.0 / (1.0 + 4096.0), abs=1e-9)
    assert lambda_eff(8.0, 1024) == pytest.approx(4096.0)


def test_hold_shrink_rejects_bad_args():
    with pytest.raises(ValueError):
        hold_shrink(-1.0, 8)
    with pytest.raises(ValueError):
        hold_shrink(1.0, 0)


def test_field_gate_matches_the_live_log():
    field = energy_field()
    assert field.gate_align() == pytest.approx(LIVE_GATE_ALIGN, abs=0.01)
    a = field.odd()
    u = field.short_u()
    # The concept is not the probe: most of a sits off the declared pair.
    assert abs(float(lm_unit(a) @ u)) == pytest.approx(LIVE_GATE_ALIGN, abs=0.01)
    assert float(lm_unit(a) @ field.intended()) > 0.70
    assert not torch.allclose(field.intended(), u)


def test_field_rejects_geometry_it_cannot_hold():
    with pytest.raises(ValueError):
        HighDLeakField(dim=1)
    with pytest.raises(ValueError):
        HighDLeakField(dim=2, content=0.5)
    with pytest.raises(ValueError):
        HighDLeakField(dim=2, content=0.0, leftover=0.0, align=0.0)


def test_closed_form_predicts_every_linear_fit(cells):
    linear = [
        row
        for row in cells.values()
        if row["bend"] == 0.0 and row["teacher"] == "pair_odd"
    ]
    assert len(linear) >= 8
    for row in linear:
        assert row["c_plus"] == pytest.approx(row["c_plus_predicted"], abs=0.01), row["name"]
        assert row["perc"] == pytest.approx(row["perc_predicted"], abs=0.02), row["name"]
        assert row["hold_explains_c_plus"] is True, row["name"]


def test_lambda_is_not_portable_across_width():
    rows = {(r["dim"], r["hold_weight"]): r for r in lambda_dim_sweep()}
    ceiling = rows[(2, 8.0)]["c_plus_if_held"]
    # 2-D: λ=8 leaves a quarter of the way to the ceiling to go.
    assert rows[(2, 8.0)]["c_plus_predicted"] > ceiling + 0.10
    assert rows[(2, 8.0)]["lambda_eff"] == pytest.approx(8.0)
    # Live width: every λ in the story lands on the ceiling.
    for lam in (0.3, 1.0, 8.0):
        wide = rows[(1024, lam)]
        assert wide["lambda_eff"] > 100.0
        assert wide["c_plus_predicted"] == pytest.approx(ceiling, abs=0.01)
    spread = max(rows[(1024, lam)]["c_plus_predicted"] for lam in (0.3, 1.0, 8.0)) - min(
        rows[(1024, lam)]["c_plus_predicted"] for lam in (0.3, 1.0, 8.0)
    )
    assert spread < 0.01


def test_fitted_rows_follow_lambda_times_dim():
    rows = {r["name"]: r for r in lambda_fit_sweep(steps=STEPS, seed=0)}
    # λ=1 at D=64 and λ=8 at D=8 are the same λ·D/2 = 32, so the same row.
    wide_soft = rows["d64_l1"]
    narrow_hard = rows["d8_l8"]
    assert wide_soft["lambda_eff"] == pytest.approx(narrow_hard["lambda_eff"])
    assert wide_soft["c_plus"] == pytest.approx(narrow_hard["c_plus"], abs=0.01)
    assert wide_soft["cos_intended"] == pytest.approx(narrow_hard["cos_intended"], abs=0.01)
    assert wide_soft["leftover_kept"] == pytest.approx(narrow_hard["leftover_kept"], abs=0.02)


def test_gender_like_is_the_copied_pair_odd_look(cells):
    row = cells["gender_like_no_e"]
    assert row["hold_off"] is True
    assert row["used_hold"] == 0.0
    assert row["c_plus"] == pytest.approx(LIVE_GENDER_C_PLUS, abs=0.03)
    assert row["collapse"] == pytest.approx(LIVE_GENDER_COLLAPSE, abs=0.03)
    assert row["loss"] == pytest.approx(LIVE_GENDER_LOSS, abs=0.01)
    assert row["cos_intended"] > 0.90
    assert row["looks_like_v12"] is True
    assert row["pass"] is True
    # No junk ê on the clean pair: the poles carry no leftover to hold.
    assert row["leftover_teacher"] == pytest.approx(0.0, abs=1e-9)
    assert row["hold_cover"] == 0.0


def test_gender_c_plus_is_not_a_slider_lock_and_energy_cannot_reach_it(cells):
    gender = cells["gender_like_no_e"]
    synonym = cells["energy_highd_synonym_l8"]
    assert gender["c_plus"] > 0.90
    assert synonym["c_plus_if_held"] < 0.60
    # A working hold *must* drop c+ below the ceiling of the ê it holds.
    assert synonym["c_plus"] < gender["c_plus"] - 0.30
    assert synonym["hold_explains_c_plus"] is True
    assert synonym["looks_like_v12"] is False


def test_2d_row_reproduces_the_existing_overlap_cell(cells):
    row = cells["energy_2d_synonym_l8"]
    old = score_overlap_policy(
        "pole_synonym_slider_l8",
        overlap=0.5,
        hold_weight=LEAK_HOLD_WEIGHT,
        ortho="slider",
        steps=200,
        seed=0,
    )
    assert row["cos_intended"] == pytest.approx(old["cos_intended"], abs=0.01)
    assert row["leftover_leak"] == pytest.approx(old["leak_ratio"], abs=0.01)
    assert row["c_plus"] == pytest.approx(old["cos_teacher"], abs=0.01)
    assert row["perc"] == pytest.approx(old["perc"], abs=0.02)
    assert row["collapse"] == pytest.approx(-1.0, abs=1e-3)
    # The live-story shape: leftover PASS, and it still does not look like v12.
    assert row["pass"] is True
    assert row["c_plus"] == pytest.approx(0.70, abs=0.03)
    assert row["perc"] == pytest.approx(0.72, abs=0.03)
    assert row["looks_like_v12"] is False


def test_2d_cannot_split_probe_from_concept(cells):
    flat = cells["energy_2d_synonym_l8"]
    high = cells["energy_highd_synonym_l8"]
    assert flat["cos_short_u"] == pytest.approx(flat["cos_intended"], abs=1e-6)
    # High-D: the probe goes up while the concept is eaten.
    assert high["cos_short_u"] > 0.60
    assert high["cos_intended"] < 0.30
    assert high["cos_short_u"] > high["cos_intended"] + 0.40


def test_synonym_hold_eats_concept_and_keeps_leak(cells):
    row = cells["energy_highd_synonym_l8"]
    base = cells["energy_highd_pair_odd"]
    assert row["hold_on_content"] > 0.85
    assert row["cos_intended"] < base["cos_intended"]
    assert row["leftover_kept"] > 0.40
    assert abs(row["leftover_leak"]) > 1.0
    assert row["loss"] > 0.05
    assert row["pass"] is False


def test_pair_odd_no_hold_is_not_leak_free(cells):
    row = cells["energy_highd_pair_odd"]
    assert row["c_plus"] == pytest.approx(1.0, abs=1e-3)
    assert row["leftover_kept"] == pytest.approx(1.0, abs=0.01)
    assert abs(row["leftover_leak"]) > 0.20
    assert row["looks_like_v12"] is True
    assert row["pass"] is False


def test_tiny_leftover_is_not_a_weak_hold(cells):
    small = cells["energy_highd_tiny_l8"]
    big = cells["energy_highd_synonym_l8"]
    assert small["hold_norm"] < 0.25
    assert big["hold_norm"] > 0.55
    assert small["hold_cover"] == pytest.approx(big["hold_cover"], abs=1e-4)
    for key in ("c_plus", "cos_short_u", "cos_intended", "leftover_leak", "loss"):
        assert small[key] == pytest.approx(big[key], abs=1e-3), key


def test_messy_tiny_leftover_turns_the_hold_off(cells):
    messy = cells["energy_highd_tiny_messy_l8"]
    clean = cells["energy_highd_tiny_l8"]
    assert messy["hold_cover"] < clean["hold_cover"] - 0.40
    assert messy["cos_intended"] > clean["cos_intended"]
    assert abs(messy["leftover_leak"]) > 0.20
    assert messy["pass"] is False


def test_medium_energy_pin_changes_nothing(cells):
    pin = cells["energy_highd_medium_pin_l8"]
    synonym = cells["energy_highd_synonym_l8"]
    assert pin["e_dot_u"] < synonym["e_dot_u"] - 0.40
    assert pin["hold_cover"] == pytest.approx(synonym["hold_cover"], abs=0.02)
    assert pin["c_plus"] == pytest.approx(synonym["c_plus"], abs=0.02)
    assert pin["cos_intended"] == pytest.approx(synonym["cos_intended"], abs=0.02)
    assert pin["pass"] is False


def test_leftover_only_e_keeps_the_concept_at_every_lambda(cells):
    rows = [cells[f"energy_highd_leftover_l{lam}"] for lam in ("0.3", "1", "8")]
    for row in rows:
        assert row["hold_on_content"] == pytest.approx(0.0, abs=1e-6)
        assert row["collapse"] == pytest.approx(-1.0, abs=1e-3)
        assert row["axis"]["collapse"] == "right"
        assert row["loss"] < 0.20
    soft, mid, hard = rows
    assert mid["cos_intended"] > 0.85
    assert hard["cos_intended"] > 0.85
    # λ=1 is the canary for the wording, not for the leak: λ=8 barely moves it.
    assert abs(mid["leftover_leak"]) > 0.20
    assert abs(hard["leftover_leak"]) == pytest.approx(abs(mid["leftover_leak"]), abs=0.05)
    assert soft["leftover_kept"] > mid["leftover_kept"] > hard["leftover_kept"]


def test_leak_left_over_is_set_by_the_caption_not_by_lambda():
    soft = {r["leftover_match"]: r for r in match_sweep(hold_weight=1.0, steps=STEPS)}
    hard = {r["leftover_match"]: r for r in match_sweep(hold_weight=8.0, steps=STEPS)}
    for match in soft:
        assert soft[match]["leftover_kept"] == pytest.approx(
            hard[match]["leftover_kept"], abs=0.20
        )
    # Naming more of the leak is what removes it.
    assert soft[0.5]["leftover_kept"] > soft[0.85]["leftover_kept"] > soft[1.0]["leftover_kept"]
    assert abs(soft[0.5]["leftover_leak"]) > 0.20
    assert abs(soft[1.0]["leftover_leak"]) <= 0.20
    assert soft[1.0]["pass"] is True


def test_wording_knee_is_where_e_starts_saying_the_concept():
    field = energy_field()
    quarter = leak_axis(field, on_u=0.05, on_content=0.25, on_leftover=(1 - 0.25 ** 2) ** 0.5)
    most = leak_axis(field, on_u=0.05, on_content=0.90, on_leftover=(1 - 0.90 ** 2) ** 0.5)
    safe = score_highd("quarter", field, leak_dir=quarter, hold_weight=LEAK_HOLD_WEIGHT, steps=STEPS)
    bad = score_highd("most", field, leak_dir=most, hold_weight=LEAK_HOLD_WEIGHT, steps=STEPS)
    assert safe["cos_intended"] > 0.90
    assert abs(safe["leftover_leak"]) <= 0.20
    assert safe["pass"] is True
    assert bad["cos_intended"] < 0.30
    assert abs(bad["leftover_leak"]) > 1.0
    assert bad["pass"] is False


def test_sub_e_is_the_hold_limit_without_the_stiffness(cells):
    hold = cells["energy_highd_leftover_l8"]
    sub = cells["energy_highd_sub_e_leftover"]
    assert sub["shrink"] == 0.0
    assert sub["c_plus"] == pytest.approx(hold["c_plus"], abs=0.03)
    assert sub["cos_intended"] == pytest.approx(hold["cos_intended"], abs=0.02)
    assert sub["leftover_leak"] == pytest.approx(hold["leftover_leak"], abs=0.02)
    assert sub["loss"] < 1e-4 < hold["loss"]


def test_sub_raw_e_takes_the_slider_with_it(cells):
    perp = cells["energy_highd_sub_e_leftover"]
    raw = cells["energy_highd_sub_raw_e_leftover"]
    assert raw["cos_short_u"] < perp["cos_short_u"]
    assert raw["c_plus"] < perp["c_plus"]


def test_sub_e_does_not_fix_a_synonym_e(cells):
    sub = cells["energy_highd_sub_e_synonym"]
    hold = cells["energy_highd_synonym_l8"]
    assert sub["cos_intended"] < 0.30
    assert sub["cos_intended"] == pytest.approx(hold["cos_intended"], abs=0.06)
    assert abs(sub["leftover_leak"]) > 1.0
    assert sub["pass"] is False


def test_teacher_modes_and_bad_names():
    field = energy_field()
    axis = leftover_only_e(field)
    plus, minus = teacher_poles(field, teacher="pair_odd")
    a = field.odd()
    assert torch.allclose(plus, a, atol=1e-6)
    assert torch.allclose(minus, -a, atol=1e-6)
    held = hold_direction(field, axis)
    sub_plus, _sub_minus = teacher_poles(field, teacher="pair_odd_sub_e", leak_dir=axis)
    assert abs(float(sub_plus @ lm_unit(held))) < 1e-5
    with pytest.raises(ValueError):
        teacher_poles(field, teacher="pair_odd_sub_e")
    with pytest.raises(ValueError):
        teacher_poles(field, teacher="nonsense", leak_dir=axis)


def test_geometry_can_never_break_polarity():
    rows = polarity_grid(steps=200, seed=0)
    assert len(rows) >= 40
    for row in rows:
        assert row["collapse"] == pytest.approx(-1.0, abs=1e-4), row["name"]
        assert row["even_norm"] == pytest.approx(0.0, abs=1e-9), row["name"]
        assert row["bend"] == 0.0


def test_polarity_break_needs_a_non_mirror_reply(cells):
    bent = cells["energy_bend_synonym_l8"]
    linear = cells["energy_highd_synonym_l8"]
    assert linear["collapse"] == pytest.approx(-1.0, abs=1e-3)
    assert bent["collapse"] > 0.0
    assert bent["bend"] == BEND_ENERGY
    # The break is not the hold's doing: c+ falls below what the hold explains.
    assert bent["hold_explains_c_plus"] is False
    assert bent["c_plus"] < bent["c_plus_predicted"] - HOLD_EXPLAINS_TOL
    # And it survives a good ê at low λ, so ê is not the cause either.
    other = cells["energy_bend_leftover_l1"]
    assert other["collapse"] > 0.0
    assert other["hold_explains_c_plus"] is False


def test_bend_crosses_zero_when_even_matches_odd():
    rows = bend_sweep(steps=STEPS, seed=0)
    clean = {r["bend"]: r for r in rows if r["used_hold"] == 0.0}
    assert clean[0.0]["collapse"] == pytest.approx(-1.0, abs=1e-3)
    assert clean[1.0]["collapse"] == pytest.approx(0.0, abs=0.02)
    assert clean[1.5]["collapse"] > 0.0
    for bend, row in clean.items():
        if bend == 0.0:
            continue
        # Exact only for G w ⊥ w; the fitted w is only generically orthogonal.
        assert bend_for_collapse(row["collapse"]) == pytest.approx(bend, rel=0.15)


def test_live_v14_analogue_lands_both_live_numbers():
    best, grid = live_v14_analogue(steps=200, seed=0)
    assert len(grid) >= 100
    assert best["gate_align"] == pytest.approx(LIVE_GATE_ALIGN, abs=0.01)
    assert best["used_hold"] == LEAK_HOLD_WEIGHT
    assert best["collapse"] == pytest.approx(LIVE_COLLAPSE, abs=0.02)
    assert best["c_plus"] == pytest.approx(LIVE_C_PLUS, abs=0.10)
    assert best["loss"] > 0.02
    # A pure rotation of the size the collapse log implies is the best fit.
    assert best["bend_parallel"] <= 0.2
    assert best["bend"] == pytest.approx(bend_for_collapse(LIVE_COLLAPSE), abs=0.3)
    assert best["norm_ratio"] == pytest.approx(1.0, abs=0.25)
    # The hold's closed form does not explain the row: that is the point.
    assert best["hold_explains_c_plus"] is False


def test_pure_rotation_bend_is_exact_and_gain_raises_c_plus():
    bend = calibrate_bend(LIVE_COLLAPSE, parallel=0.0, steps=200, seed=0)
    assert bend == pytest.approx(bend_for_collapse(LIVE_COLLAPSE), abs=0.02)
    spin = score_highd(
        "spin",
        HighDLeakField(dim=8, align=LIVE_GATE_ALIGN, bend=bend, bend_parallel=0.0),
        leak_dir=synonym_e(energy_field()),
        hold_weight=LEAK_HOLD_WEIGHT,
        steps=200,
    )
    gain = score_highd(
        "gain",
        HighDLeakField(dim=8, align=LIVE_GATE_ALIGN, bend=bend, bend_parallel=0.65),
        leak_dir=synonym_e(energy_field()),
        hold_weight=LEAK_HOLD_WEIGHT,
        steps=200,
    )
    assert spin["norm_ratio"] == pytest.approx(1.0, abs=1e-3)
    assert spin["collapse"] == pytest.approx(LIVE_COLLAPSE, abs=0.02)
    # Same asymmetry size, but a gain share splits the poles and lifts c+.
    assert gain["norm_ratio"] > 1.5
    assert gain["c_plus"] > spin["c_plus"]
    assert gain["collapse"] > spin["collapse"] - 0.05


def test_pure_gain_keeps_poles_antipodal():
    field = HighDLeakField(dim=8, align=LIVE_GATE_ALIGN, bend=0.5, bend_parallel=1.0)
    row = score_highd("pull", field, steps=200)
    assert row["norm_ratio"] > 1.5
    assert row["collapse"] == pytest.approx(-1.0, abs=1e-3)
    assert row["perc"] > 0.0


def test_live_collapse_logs_imply_these_bends():
    assert bend_for_collapse(LIVE_GENDER_COLLAPSE) == pytest.approx(BEND_GENDER, abs=0.03)
    assert bend_for_collapse(LIVE_COLLAPSE) == pytest.approx(BEND_ENERGY, abs=0.03)
    assert bend_for_collapse(0.0) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        bend_for_collapse(1.0)


def test_live_c_plus_is_what_a_working_synonym_hold_prints():
    field = energy_field(dim=1024)
    p = hold_cover(field, hold_direction(field, synonym_e(field)))
    pred = hold_predictions(p, LEAK_HOLD_WEIGHT, 1024)
    # Live logged c+ 0.31 with a caption-axis gate of 0.37.
    assert pred["c_plus_predicted"] == pytest.approx(pred["c_plus_if_held"], abs=0.01)
    assert pred["c_plus_predicted"] < LIVE_GENDER_C_PLUS - 0.30
    assert abs(pred["c_plus_predicted"] - LIVE_C_PLUS) < 0.30
    assert pred["c_plus_predicted"] > LIVE_C_PLUS


def test_spike_is_the_unnormalized_half_of_the_loss():
    # λ·(a·ê̂)² — live printed 278 at λ=8, i.e. a·ê̂ ≈ 5.9 hidden units.
    implied = (LIVE_SPIKE_LOSS / LEAK_HOLD_WEIGHT) ** 0.5
    assert hold_spike(implied, LEAK_HOLD_WEIGHT) == pytest.approx(LIVE_SPIKE_LOSS, abs=1.0)
    field = energy_field()
    a = field.odd()
    held = hold_direction(field, synonym_e(field))
    along = abs(float(a @ lm_unit(held)))
    assert hold_spike(along, LEAK_HOLD_WEIGHT) < 20.0
    # The same term computed by the trainer helper, at the teacher itself.
    neu = torch.zeros(field.dim)
    by_helper = float(lm_axis_hold(neu + a, neu - a, neu, held))
    assert LEAK_HOLD_WEIGHT * by_helper == pytest.approx(
        hold_spike(along, LEAK_HOLD_WEIGHT), rel=1e-4
    )


def test_hold_split_names_what_the_hold_eats():
    field = energy_field()
    synonym = hold_split(field, hold_direction(field, synonym_e(field)))
    leftover = hold_split(field, hold_direction(field, leftover_only_e(field)))
    pin = hold_split(field, hold_direction(field, medium_pin_e(field)))
    assert synonym["hold_on_u"] < 1e-5
    assert synonym["hold_on_content"] > 0.85
    assert leftover["hold_on_content"] < 1e-6
    assert leftover["hold_on_leftover"] == pytest.approx(1.0, abs=1e-5)
    assert pin["hold_on_content"] > 0.85
    assert hold_split(field, None)["hold_on_content"] == 0.0


def test_declared_e_never_becomes_the_slider_name():
    field = energy_field()
    for axis in (synonym_e(field), medium_pin_e(field), leftover_only_e(field)):
        assert float(axis.norm()) == pytest.approx(1.0, abs=1e-6)
        assert not torch.allclose(axis, field.short_u(), atol=1e-3)
        held = hold_direction(field, axis)
        assert held is not None
        assert abs(float(lm_unit(held) @ field.short_u())) < 1e-5
    # ê = û exactly leaves nothing to hold, as in the 2-D cell.
    assert hold_direction(field, field.short_u()) is None


def test_gender_field_declares_no_leftover_to_hold():
    field = gender_like_field()
    assert field.gate_align() == pytest.approx(1.0, abs=1e-6)
    assert torch.allclose(field.intended(), field.short_u())
    with pytest.raises(ValueError):
        leak_axis(field, on_u=0.5, on_content=0.5, on_leftover=0.0)


def test_synonym_cover_is_set_by_wording_not_width():
    base = synonym_cover()
    for dim in (4, 8, 16, 64):
        field = energy_field(dim=dim)
        assert hold_cover(field, hold_direction(field, synonym_e(field))) == pytest.approx(
            base, abs=0.02
        )


def test_2d_energy_field_is_the_flat_blind_spot():
    field = energy_2d_field()
    assert field.dim == 2
    assert field.content == 0.0
    assert torch.allclose(field.intended(), field.short_u())
    assert field.gate_align() == pytest.approx(0.58, abs=0.01)
