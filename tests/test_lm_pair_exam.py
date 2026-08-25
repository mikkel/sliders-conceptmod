"""The pair-exam cell: does it reproduce the 2026-08-25 live exam.

Three live runs, one of them a pass and two failures, two of them the same
recipe on different prompt files. The cell has to order them, and it has to
do that without reading any of the columns the live logs made look healthy.
Nothing here changes the live trainer default.
"""

from __future__ import annotations

import pytest
import torch

from analysis.slider2d.exam import (
    CELLS,
    EXAM_COHERENCE,
    EXAM_MATCH_KEPT,
    EXAM_ROLL_OFF_MAX,
    EXAM_ROLL_OVERLAP,
    EXAM_ROLL_SWING,
    LIVE_EXAM,
    LIVE_PAIR_COS,
    LIVE_ROW,
    close_field,
    divergence_sweep,
    divergent_field,
    exam_cell,
    exam_table,
    exam_verdicts,
    factorial_table,
    first_above,
    first_below,
    guard_is_uniform,
    guard_table,
    hold_direction,
    live_exam_rows,
    near_gate,
    rollouts,
    score_exam,
    shared_from_probe_cos,
    target_geometry,
    teacher_rollouts,
    teacher_self_match,
    unused_e_field,
    visible_sweep,
)
from analysis.slider2d.sheet import gender_cell, leaky_cell
from conceptmod.textsliders.slider_targets import (
    lm_blend_guard,
    lm_blind_projector,
    lm_faithful_guard_e,
    lm_faithful_sub_e,
    lm_unit,
)
from conceptmod.textsliders.train_lm_slider_music3 import parse_args


STEPS = 400
_CACHE: dict[str, dict] = {}


def table(seed: int = 0) -> dict[str, list[dict]]:
    key = f"table{seed}"
    if key not in _CACHE:
        _CACHE[key] = exam_table(steps=STEPS, seed=seed)
    return _CACHE[key]


def cell(name: str, seed: int = 0) -> dict[str, dict]:
    return {row["name"]: row for row in table(seed)[name]}


# -- the live exam -------------------------------------------------------


def test_the_cell_reproduces_all_three_live_listens():
    rows = live_exam_rows(table())
    assert {r["run"] for r in rows} == set(LIVE_EXAM)
    for row in rows:
        assert row["agrees"], (
            f"{row['run']}: cell says {row['predicted']}, ears said "
            f"{row['listen']} ({row['reason']})"
        )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_live_exam_verdicts_are_seed_robust(seed):
    for row in live_exam_rows(table(seed)):
        assert row["agrees"], f"seed {seed}: {row['run']} flipped"


def test_energy_v18_and_gender_v16_are_the_same_recipe_on_two_pairs():
    """The reason the board has to be recipe × pair and not recipe."""
    energy = LIVE_EXAM["energy-lm-v18"]
    gender = LIVE_EXAM["gender-lm-v16"]
    assert (energy["teacher"], energy["pole_mode"]) == (
        gender["teacher"],
        gender["pole_mode"],
    )
    assert energy["listen"] == "pass" and gender["listen"] == "fail"
    assert LIVE_ROW["energy-lm-v18"][0] == LIVE_ROW["gender-lm-v16"][0]
    assert LIVE_ROW["energy-lm-v18"][1] != LIVE_ROW["gender-lm-v16"][1]
    divergent = cell("divergent")["semantic_kl_poles"]
    close = cell("close")["semantic_kl_poles"]
    assert divergent["pass"] is True
    assert close["pass"] is False


def test_the_ranked_live_rows_put_the_win_above_both_failures():
    rows = {r["run"]: r for r in live_exam_rows(table())}
    win = rows["energy-lm-v18"]["row"]
    for run in ("energy-lm-v16", "gender-lm-v16"):
        lost = rows[run]["row"]
        assert win["roll_overlap"] > lost["roll_overlap"]
        assert win["roll_swing_kept"] > lost["roll_swing_kept"]
        assert win["roll_match_kept"] > lost["roll_match_kept"]


def test_no_live_log_column_orders_the_three_runs():
    """Why the gate cannot read the loss, c+, the collapse or p%."""
    gender = LIVE_EXAM["gender-lm-v16"]
    others = [LIVE_EXAM["energy-lm-v16"], LIVE_EXAM["energy-lm-v18"]]
    assert gender["listen"] == "fail"
    assert all(gender["loss"] < o["loss"] for o in others)
    assert all(gender["c_plus"] > o["c_plus"] for o in others)
    assert all(gender["pperc"] < o["pperc"] for o in others)


def test_the_gate_does_not_read_any_goodhart_column():
    scored = set(exam_verdicts(cell("divergent")["faithful_raw"]))
    assert scored == {"continuation", "same_words", "off_caption", "coherence", "swing"}
    forged = dict(cell("close")["semantic_kl_poles"])
    forged.update({"pair_odd_cos": 1.0, "collapse": -1.0, "loss": 0.0, "pperc": 0.0})
    assert exam_verdicts(forged) == cell("close")["semantic_kl_poles"]["axis"]


def test_verdicts_that_turn_on_a_near_gate_column_are_flagged():
    row = dict(cell("divergent")["semantic_kl_poles"])
    assert "swing" not in row["near_gate"]
    row["roll_swing_kept"] = EXAM_ROLL_SWING + 0.01
    row["axis"] = exam_verdicts(row)
    assert "swing" in near_gate(row)


# -- the pair coordinates ------------------------------------------------


def test_each_field_prints_the_pair_cos_its_yaml_logged():
    lo, hi = LIVE_PAIR_COS["energy-v4"]
    assert divergent_field().probe_cos() == pytest.approx(0.5 * (lo + hi), abs=1e-6)
    assert unused_e_field().probe_cos() == pytest.approx(0.5 * (lo + hi), abs=1e-6)
    assert close_field().probe_cos() == pytest.approx(
        LIVE_PAIR_COS["gender-v4"], abs=1e-6
    )


def test_shared_is_solved_and_not_chosen():
    for cos in (-0.15, -0.08, 0.0, 0.015, 0.3):
        shared = shared_from_probe_cos(cos, track=2.0, odd_norm=1.14)
        field = divergent_field(shared=shared, probe_cos_target=None)
        assert field.probe_cos() == pytest.approx(cos, abs=1e-4)
    # A divergent pair cannot print an arbitrarily negative pair cos: the
    # track halves sit in both ``a`` and ``c``, so the collapse the trainer
    # logs is bounded away from −1 by the divergence itself. That is a
    # property of the construction, not a limit of the solver.
    with pytest.raises(ValueError):
        shared_from_probe_cos(-0.5, track=2.0, odd_norm=1.14)
    assert shared_from_probe_cos(-0.5, track=0.0, odd_norm=1.14) > 0.0


def test_divergence_separates_the_two_pairs():
    assert close_field().divergence() == pytest.approx(0.0)
    assert divergent_field().divergence() > 0.7
    assert unused_e_field().divergence() == pytest.approx(0.0)


def test_the_close_pair_hides_its_axis_from_the_scored_token():
    """Hypothesis 3, as a field property rather than a fit property."""
    close = close_field()
    divergent = divergent_field()
    assert close.invisible_share() > 0.9
    assert divergent.invisible_share() < 0.5
    head = close.readout()
    assert 4 in head.null_dims()
    for dim in (0, 1, 2, 3, 5):
        assert dim not in head.null_dims()


def test_c_is_a_blend_of_two_songs_only_when_the_poles_are_two_songs():
    """Hypothesis 1: what ``½(h₊+h₋) − h0`` actually contains."""
    divergent = divergent_field()
    c = divergent.common_vec(0)
    on_plus = float(c @ divergent.plus_track())
    on_minus = float(c @ divergent.minus_track())
    assert on_plus > 0.5 and on_minus > 0.5
    assert on_plus == pytest.approx(on_minus)
    close = close_field()
    c_close = close.common_vec(0)
    assert float(c_close @ close.plus_track()) == pytest.approx(0.0)
    assert float(c_close @ close.minus_track()) == pytest.approx(0.0)
    assert float(c_close @ close.sheet_dir()) > 0.5


def test_energy_v4_declares_a_leak_axis_that_is_most_of_the_slider():
    divergent = divergent_field()
    a = divergent.odd(0)
    held = hold_direction(divergent, divergent.declared_e())
    overlap = abs(float(a @ lm_unit(held))) / float(a.norm())
    assert overlap > 0.7
    unused = unused_e_field()
    held_unused = hold_direction(unused, unused.declared_e())
    unused_overlap = abs(float(unused.odd(0) @ lm_unit(held_unused))) / float(
        unused.odd(0).norm()
    )
    assert unused_overlap < 0.5
    assert close_field().declared_e() is None


# -- the target points, with no optimizer -------------------------------


def test_faithful_sub_e_is_a_blend_teacher_only_on_the_divergent_pair():
    divergent = divergent_field()
    geom = target_geometry(
        divergent, teacher="faithful_sub_e", leak_dir=divergent.declared_e()
    )
    assert geom["blend_teacher"] is True
    assert geom["visible_axis_eaten"] > 0.3
    unused = unused_e_field()
    same = target_geometry(
        unused, teacher="faithful_sub_e", leak_dir=unused.declared_e()
    )
    assert same["blend_teacher"] is False
    assert same["visible_axis_eaten"] < 0.15


def test_the_target_points_come_from_the_live_functions():
    divergent = divergent_field()
    pos, neg, neu = divergent.poles(0)
    want = lm_faithful_sub_e(
        pos, neg, neu, divergent.declared_e(), slider_dir=divergent.short_u()
    )
    from analysis.slider2d.exam import teacher_points

    got = teacher_points(
        divergent, 0, teacher="faithful_sub_e", leak_dir=divergent.declared_e()
    )
    assert torch.allclose(got[0], want[0])
    assert torch.allclose(got[1], want[1])


# -- the rollout ---------------------------------------------------------


def test_a_real_pole_repeats_itself_and_the_blend_does_not():
    divergent = divergent_field()
    head = divergent.readout()
    pos, neg, _neu = divergent.poles(0)
    plus, _p = rollouts(divergent, pos, head, row=0, sign=1.0)
    assert all(len(seq) == divergent.out_steps for seq in plus)
    mid = 0.5 * (pos + neg)
    blend, _b = rollouts(divergent, mid, head, row=0, sign=1.0)
    pole_words = {frozenset(seq) for seq in plus}
    blend_words = {frozenset(seq) for seq in blend}
    assert len(pole_words) <= len(blend_words)
    assert head.index("punk") in set().union(*[set(s) for s in plus])


def test_the_rollout_ceiling_is_measured_and_not_assumed_to_be_one():
    for name in CELLS:
        field = CELLS[name]()
        head = field.readout()
        plus, minus, corpus = teacher_rollouts(field, head)
        self_match = teacher_self_match(plus, minus)
        assert 0.5 < self_match <= 1.0
        assert head.index(f"lyric{0}") in corpus
    assert (
        teacher_self_match(*teacher_rollouts(divergent_field(), divergent_field().readout())[:2])
        < teacher_self_match(*teacher_rollouts(unused_e_field(), unused_e_field().readout())[:2])
    )


def test_one_token_cannot_see_what_eight_can():
    """The blend's first-token policy is on the union sheet; the run is not."""
    divergent = divergent_field()
    head = divergent.readout()
    pos, neg, _neu = divergent.poles(0)
    mid = 0.5 * (pos + neg)
    policy = head.policy(mid)
    punk, lull = head.index("punk"), head.index("lull")
    assert float(policy[punk]) == pytest.approx(float(policy[lull]), abs=1e-6)
    assert float(policy[punk]) + float(policy[lull]) > 0.5
    row = cell("divergent")["semantic_kl_sub_e"]
    assert row["roll_coherence"] < 1.0
    assert row["roll_overlap"] < EXAM_ROLL_OVERLAP


# -- the diagnosis flags -------------------------------------------------


def test_kl_leaves_the_invisible_block_at_zero_and_mse_does_not():
    close = close_field()
    kl = score_exam(
        "kl", close, pole_mode="semantic_kl", teacher="faithful", steps=STEPS
    )
    mse = score_exam(
        "mse", close, pole_mode="hidden", teacher="faithful", steps=STEPS
    )
    assert kl["invisible_kept"] == pytest.approx(0.0, abs=1e-3)
    assert mse["invisible_kept"] == pytest.approx(1.0, abs=1e-2)
    assert kl["loss_solved"] > 0.9
    assert kl["kl_small_hidden_far"] is True
    assert mse["kl_small_hidden_far"] is False
    assert kl["pass"] is False
    assert mse["pass"] is True


def test_a_solved_loss_is_not_evidence_the_slider_works():
    close = cell("close")
    kl = close["semantic_kl_poles"]
    mse = close["faithful_raw"]
    assert kl["loss"] < mse["loss"]
    assert kl["loss_solved"] > 0.9
    assert kl["roll_swing_kept"] < mse["roll_swing_kept"]
    assert kl["pass"] is False and mse["pass"] is True


# -- the cells -----------------------------------------------------------


def test_the_next_untrained_winner_is_faithful_on_hidden_mse():
    """The card the board now points at: gender, faithful, hidden."""
    for name in CELLS:
        row = cell(name)["faithful_raw"]
        assert row["pass"] is True, f"{name}: {row['reason']}"
    close = cell("close")["faithful_raw"]
    assert close["roll_overlap"] >= EXAM_ROLL_OVERLAP
    assert close["roll_match_kept"] >= EXAM_MATCH_KEPT
    assert close["roll_coherence"] >= EXAM_COHERENCE
    assert close["roll_off_corpus"] <= EXAM_ROLL_OFF_MAX
    assert close["roll_swing_kept"] >= EXAM_ROLL_SWING


def test_subtracting_e_is_free_on_a_same_song_pair_and_not_on_two_tracks():
    unused = cell("unused_e")
    divergent = cell("divergent")
    for name in ("pair_odd_sub_e", "faithful_sub_e", "semantic_kl_sub_e"):
        assert unused[name]["pass"] is True, unused[name]["reason"]
    assert divergent["semantic_kl_sub_e"]["pass"] is False
    assert divergent["faithful_sub_e"]["pass"] is False


def test_the_exam_agrees_with_the_sheet_cell_about_leak():
    """Same numbers, different readout: the two cells are consistent."""
    sheet = {row["name"]: row for row in leaky_cell(steps=STEPS)}
    unused = cell("unused_e")
    pairs = (
        ("faithful_raw", "v6_faithful"),
        ("semantic_kl_poles", "v16_semantic_kl"),
        ("pair_odd_sub_e", "v15_pair_odd_sub_e"),
        ("semantic_kl_sub_e", "v16_semantic_kl_sub_e"),
    )
    for exam_name, sheet_name in pairs:
        assert unused[exam_name]["leak_tok"] == pytest.approx(
            sheet[sheet_name]["leak_tok"], abs=0.01
        )
    assert unused["hold_e_perp_l8"]["leak_tok"] == pytest.approx(
        sheet["v9_hold_e"]["leak_tok"], abs=0.01
    )


def test_the_rollout_averages_leak_away_which_is_why_the_sheet_scores_it():
    unused = cell("unused_e")["faithful_raw"]
    assert abs(unused["leak_tok"]) > 0.2
    assert abs(unused["leak_roll"]) < abs(unused["leak_tok"])
    assert "leak" not in unused["axis"]


# -- the two new techniques ----------------------------------------------


def test_the_guard_decides_e_from_the_captions_alone_and_agrees_per_row():
    rows = guard_table()
    assert guard_is_uniform(rows) is True
    by_cell = {}
    for row in rows:
        by_cell.setdefault(row["cell"], []).append(row)
    # energy-v4: ê restates the axis, the ê-cleaned target drifts nearer the
    # midpoint than the caption, and the guard keeps the caption.
    for row in by_cell["divergent"]:
        assert row["admits"] is False
        assert row["to_pole"] > row["to_mid"]
        assert row["teacher_used"] == "faithful"
    # A real leftover: the subtraction is taken and costs little axis.
    for row in by_cell["unused_e"]:
        assert row["admits"] is True
        assert row["to_pole"] < row["to_mid"]
        assert row["axis_eaten"] < 0.15
        assert row["teacher_used"] == "faithful_sub_e"
    # gender-v4 declares no ê, so there is nothing to decide.
    for row in by_cell["close"]:
        assert row["declared_e"] is False
        assert row["admits"] is None
        assert row["teacher_used"] == "faithful"


def test_faithful_guard_e_tops_both_live_pairs_and_keeps_the_leak_fix():
    for name in CELLS:
        row = cell(name)[f"faithful_guard_e"]
        assert row["pass"] is True, f"{name}: {row['reason']}"
    for pair in ("divergent", "close"):
        row = cell(pair)["faithful_guard_e"]
        assert min(row["roll_overlap"], row["roll_swing_kept"]) == pytest.approx(
            1.0, abs=1e-6
        )
    # On a divergent pair the guard refuses, so the fit is the one the raw
    # caption teacher makes — that is what "safe on both pair types" means,
    # and it is the difference from ``faithful_sub_e``, which fails here.
    guarded = cell("divergent")["faithful_guard_e"]
    assert guarded["roll_swing_kept"] == pytest.approx(
        cell("divergent")["faithful_raw"]["roll_swing_kept"], abs=1e-6
    )
    assert cell("divergent")["faithful_sub_e"]["pass"] is False
    # On the leftover cell it is the ê-cleaned fit, so the leak the sheet
    # charges ``faithful`` for is gone with no caption rewritten.
    sheet = {row["name"]: row for row in leaky_cell(steps=STEPS)}
    assert abs(sheet["v6_faithful"]["leak_tok"]) > 0.2
    assert abs(sheet["faithful_guard_e"]["leak_tok"]) < 0.01
    assert sheet["faithful_guard_e"]["pass"] is True
    assert sheet["v6_faithful"]["pass"] is False
    # And on a clean pair it is the caption teacher, unchanged.
    clean = {row["name"]: row for row in gender_cell(steps=STEPS)}
    assert clean["faithful_guard_e"]["pass"] is True
    assert clean["faithful_guard_e"]["on_sheet_kept"] == pytest.approx(
        clean["v6_faithful"]["on_sheet_kept"], abs=1e-3
    )


def test_dual_band_makes_the_close_pair_arrive_without_leaving_the_kl():
    """What the live gender-v16 garble was missing, as one added term."""
    close = cell("close")
    kl = close["semantic_kl_poles"]
    dual = close["dual_band_poles"]
    assert kl["pass"] is False and dual["pass"] is True
    assert kl["invisible_kept"] == pytest.approx(0.0, abs=1e-3)
    assert dual["invisible_kept"] == pytest.approx(1.0, abs=1e-2)
    assert kl["kl_small_hidden_far"] is True
    assert dual["kl_small_hidden_far"] is False
    assert dual["roll_swing_kept"] > 2 * kl["roll_swing_kept"]
    # It does not cost the divergent pair anything: that one was already a
    # live pass under plain KL.
    assert cell("divergent")["dual_band_poles"]["pass"] is True
    assert cell("divergent")["semantic_kl_poles"]["pass"] is True


def test_the_new_loss_needs_a_caption_target_and_says_so():
    """The control. Pinning the blind band does not rescue the midpoint."""
    for pair in ("divergent", "close"):
        row = cell(pair)["dual_band_midpoint"]
        assert row["invisible_kept"] == pytest.approx(1.0, abs=1e-2)
        assert row["is_caption"] is False
    assert cell("divergent")["dual_band_midpoint"]["pass"] is False
    assert cell("divergent")["dual_band_poles"]["pass"] is True


def test_the_factorial_shows_both_halves_are_necessary():
    rows = {(r["target"], r["pole_mode"]): r for r in factorial_table(steps=200)}
    caption_kl = rows[("caption", "semantic_kl")]
    caption_dual = rows[("caption", "dual_band")]
    caption_mse = rows[("caption", "hidden")]
    midpoint_dual = rows[("midpoint", "dual_band")]
    # Blind band off, caption on: the close pair does not arrive.
    assert caption_kl["close_pass"] is False
    assert caption_kl["divergent_pass"] is True
    # Blind band on, caption off: the divergent pair walks off its own words.
    assert midpoint_dual["divergent_pass"] is False
    # Both on: both pairs, by two different losses.
    assert caption_dual["both"] is True
    assert caption_mse["both"] is True
    assert caption_dual["exam_score"] > caption_kl["exam_score"]
    assert caption_dual["exam_score"] > midpoint_dual["exam_score"]


def test_the_exam_readout_really_has_a_band_the_kl_cannot_see():
    """Otherwise ``dual_band`` would be ``semantic_kl`` with extra words."""
    for name in CELLS:
        field = CELLS[name]()
        blind = lm_blind_projector(field.readout().weight)
        assert blind is not None, name
        assert float(blind.trace()) == pytest.approx(1.0, abs=1e-3)
        delivery = field.delivery_dir()
        assert torch.allclose(blind @ delivery, delivery, atol=1e-5)


def test_the_guard_is_the_live_function_and_not_a_second_rule():
    divergent = divergent_field()
    pos, neg, neu = divergent.poles(0)
    e = divergent.declared_e()
    from analysis.slider2d.exam import teacher_points

    got = teacher_points(divergent, 0, teacher="faithful_guard_e", leak_dir=e)
    want = lm_faithful_guard_e(pos, neg, neu, e, slider_dir=divergent.short_u())
    assert torch.allclose(got[0], want[0]) and torch.allclose(got[1], want[1])
    geom = target_geometry(divergent, teacher="faithful_sub_e", leak_dir=e)
    guard = lm_blend_guard(
        *lm_faithful_sub_e(pos, neg, neu, e, slider_dir=divergent.short_u()), pos, neg
    )
    assert geom["guard_admits"] is bool(guard["admissible"])
    assert geom["guard_admits"] is False


# -- the sweeps ----------------------------------------------------------


def test_swing_falls_as_the_pair_diverges_and_faithful_does_not():
    sweep = divergence_sweep(steps=150)
    subs = [r["sub_e_swing"] for r in sweep if r["sub_e_swing"] is not None]
    assert subs[0] > subs[-1]
    assert subs[-1] < EXAM_ROLL_SWING
    assert all(r["poles_swing"] >= EXAM_ROLL_SWING for r in sweep)
    flip = first_below(sweep, "sub_e_swing", EXAM_ROLL_SWING, "divergence")
    assert flip is not None
    assert flip < divergent_field().divergence()


def test_the_kl_loss_is_solved_across_the_whole_visible_sweep():
    sweep = visible_sweep(steps=150)
    assert all(r["kl_solved"] > 0.9 for r in sweep)
    assert all(r["kl_invisible_kept"] == pytest.approx(0.0, abs=1e-3) for r in sweep)
    assert all(r["mse_invisible_kept"] == pytest.approx(1.0, abs=1e-2) for r in sweep)
    assert sweep[0]["kl_swing"] < sweep[-1]["kl_swing"]
    reach = first_above(sweep, "kl_swing", EXAM_ROLL_SWING, "visible_share")
    assert reach is not None
    assert reach > close_field().visible_share()


def test_dual_band_holds_the_swing_where_the_kl_loses_it():
    """The sweep is the mechanism claim: the blind band is what was missing."""
    sweep = visible_sweep(steps=150)
    assert all(r["dual_invisible_kept"] == pytest.approx(1.0, abs=1e-2) for r in sweep)
    assert all(r["dual_swing"] >= EXAM_ROLL_SWING for r in sweep)
    assert all(r["dual_pass"] for r in sweep)
    # At the invisible end the KL has lost the axis and the dual band has not.
    assert sweep[0]["kl_swing"] < EXAM_ROLL_SWING
    assert sweep[0]["dual_swing"] > 2 * sweep[0]["kl_swing"]
    # At the readable end they agree, because there is nothing blind left to
    # add. A recipe that only differed at one end of this sweep would be a
    # coincidence rather than a mechanism.
    assert sweep[-1]["dual_swing"] == pytest.approx(sweep[-1]["kl_swing"], abs=0.1)


# -- guardrails ----------------------------------------------------------


def test_the_cell_is_cpu_only_and_touches_no_weights():
    row = exam_cell("close", steps=20)[0]
    assert row["cell"] == "close"
    field = close_field()
    assert field.readout().weight.device.type == "cpu"
    assert field.dim < 32


def test_the_live_trainer_default_is_untouched():
    args = parse_args(["--prompts", "x.yaml"])
    assert args.lm_target == "v9"
    assert args.pole_mode == "hidden"
    assert args.common_beta == 0.0
