"""Can anything be odd enough for ``leak_frac < 0`` on a divergent pair.

The criterion is ``exam_divergent`` True and ``leak_frac < 0`` together.
These tests pin the two things ``docs/lm-odd-leak-frac.md`` claims: that
``leak_frac`` is an exact function of the teacher's even and odd halves
rather than something a loss discovers, and that the recipe wired live
clears the gate for a reason the page states and does not clear the ones
it says it does not. Nothing here changes the live trainer default.
"""

from __future__ import annotations

import pytest
import torch

from analysis.slider2d.exam import CELLS, fit_exam, score_exam, teacher_points
from analysis.slider2d.odd_search import (
    LEAK_FRAC_WIN,
    WIN_CELL,
    algebra_check,
    by_candidate,
    candidates,
    frontier,
    gain_sweep,
    gain_window,
    hits,
    leftover_saturation,
    pair_budget,
    row_cos_sweep,
    score_candidate,
    sheet_leak,
    strength_invariance,
    worst_margin,
)
from conceptmod.textsliders.slider_targets import (
    leftover_bipolar,
    lm_blend_guard,
    lm_common_agree,
    lm_faithful_gain,
    lm_pair_even_odd,
    teacher_leak_frac,
)
from conceptmod.textsliders.train_lm_slider_music3 import (
    LM_RECIPES,
    lm_train_targets,
    resolve_gain_scale,
    resolve_lm_recipe,
)


STEPS = 400
_CACHE: dict[str, object] = {}


def cand(name: str) -> object:
    found = next((c for c in candidates() if c.name == name), None)
    assert found is not None, f"{name!r} is not a candidate"
    return found


def scored(name: str, cell: str = WIN_CELL, seed: int = 0) -> dict:
    key = f"{name}/{cell}/{seed}"
    if key not in _CACHE:
        _CACHE[key] = score_candidate(cand(name), cell, seed=seed, steps=STEPS)
    return _CACHE[key]


# -- the algebra the page reasons with ------------------------------------


def test_teacher_leak_frac_is_the_pair_cosine_on_a_caption_teacher():
    """β = γ = 1 ⇒ ``leak_frac`` is ``cos(h₊−h0, h₋−h0)``, which the trainer logs."""
    for name in ("divergent", "close", "unused_e"):
        field = CELLS[name]()
        pos, neg, neu = field.poles(0)
        assert teacher_leak_frac(pos, neg, neu) == pytest.approx(
            field.probe_cos(0), abs=1e-5
        )


def test_fitted_leak_frac_lands_on_the_four_caption_prediction():
    """No optimizer needed: the target pair already names the number."""
    for row in algebra_check():
        assert row["fitted"] == pytest.approx(row["closed_form"], abs=2e-5)
        assert row["predicted"] == pytest.approx(row["closed_form"], abs=1e-6)


def test_the_track_cancels_out_of_leak_frac_on_a_divergent_pair():
    """Both halves carry ``½·track²``; only shared vs flipping decides the sign."""
    budget = pair_budget()
    assert budget["even_sq"] == pytest.approx(
        budget["track_in_both"] + budget["shared_sq"], abs=1e-4
    )
    assert budget["odd_sq"] == pytest.approx(
        budget["track_in_both"] + budget["flipping_sq"], abs=1e-4
    )
    # And the wall really is only a few percent wide.
    assert 1.0 < budget["break_even_gain"] < 1.05


def test_leak_frac_is_invariant_to_inference_strength():
    """A teacher gain and turning the slider up are different objects."""
    rows = strength_invariance()
    base = rows[0]["leak_frac"]
    for row in rows:
        assert row["leak_frac"] == pytest.approx(base, abs=1e-5)
    # ...because the student itself did scale, and the cosine did not.
    assert rows[-1]["odd_norm"] > 5.0 * rows[0]["odd_norm"]


def test_the_caption_teacher_straddles_the_criterion_across_energy_v4_rows():
    """The sign belongs to the prompt row, not to the recipe."""
    sweep = row_cos_sweep(steps=200)
    assert all(
        row["leak_frac"] == pytest.approx(row["probe_cos"], abs=1e-3) for row in sweep
    )
    assert any(row["wins"] for row in sweep)
    assert any(not row["wins"] for row in sweep)


# -- the teacher functions themselves -------------------------------------


def test_faithful_gain_at_one_is_the_caption_pair():
    field = CELLS[WIN_CELL]()
    pos, neg, neu = field.poles(0)
    plus, minus = lm_faithful_gain(pos, neg, neu, target_scale=1.0)
    assert torch.equal(plus, pos)
    assert torch.equal(minus, neg)


def test_faithful_gain_never_touches_the_even_half():
    """``‖even‖`` is the same at every γ — the gain buys the denominator."""
    field = CELLS[WIN_CELL]()
    pos, neg, neu = field.poles(0)
    common, axis = lm_pair_even_odd(pos, neg, neu)
    for gain in (1.0, 1.5, 2.0, 4.0):
        plus, minus = lm_faithful_gain(pos, neg, neu, target_scale=gain)
        got_c, got_a = lm_pair_even_odd(plus, minus, neu)
        assert float((got_c - common).norm()) == pytest.approx(0.0, abs=1e-6)
        assert float(got_a.norm()) == pytest.approx(gain * float(axis.norm()), abs=1e-5)


def test_faithful_gain_is_never_a_blend_teacher():
    """``lm_blend_guard`` admits it at every gain, structurally."""
    for name in ("divergent", "close", "unused_e"):
        field = CELLS[name]()
        pos, neg, neu = field.poles(0)
        for gain in (1.05, 1.5, 2.0, 4.0, 10.0):
            plus, minus = lm_faithful_gain(pos, neg, neu, target_scale=gain)
            assert lm_blend_guard(plus, minus, pos, neg)["admissible"]


def test_common_agree_is_the_common_term_on_a_close_pair():
    """One song: everything in ``c`` is stated by both poles, so nothing goes."""
    field = CELLS["close"]()
    pos, neg, neu = field.poles(0)
    common, _axis = lm_pair_even_odd(pos, neg, neu)
    agreed = lm_common_agree(pos, neg, neu)
    assert float((agreed - common).norm()) == pytest.approx(0.0, abs=1e-6)


def test_common_agree_drops_the_blend_on_a_divergent_pair():
    """Two songs: the half of each track sitting in ``c`` is not stated by both."""
    field = CELLS[WIN_CELL]()
    pos, neg, neu = field.poles(0)
    common, _axis = lm_pair_even_odd(pos, neg, neu)
    agreed = lm_common_agree(pos, neg, neu)
    assert float(agreed @ field.plus_track()) == pytest.approx(0.0, abs=1e-6)
    assert float(agreed @ field.minus_track()) == pytest.approx(0.0, abs=1e-6)
    # The real shared specificity survives at full strength.
    assert float(agreed @ field.sheet_dir()) == pytest.approx(
        float(common @ field.sheet_dir()), abs=1e-6
    )
    # It is always a shrink, never an overshoot.
    assert float(agreed.norm()) < float(common.norm())


def test_common_agree_never_exceeds_the_common_term_coordinatewise():
    torch.manual_seed(0)
    for _ in range(20):
        pos, neg, neu = (torch.randn(32) for _ in range(3))
        common, _axis = lm_pair_even_odd(pos, neg, neu)
        agreed = lm_common_agree(pos, neg, neu)
        assert torch.all(agreed.abs() <= common.abs() + 1e-6)


# -- what the exam charges ------------------------------------------------


def test_the_hit_set_is_not_empty_and_faithful_gain_is_in_it():
    row = scored("faithful_gain_1.5")
    assert row["pass"]
    assert row["leak_frac"] < LEAK_FRAC_WIN
    assert row["wins"]


def test_the_hit_holds_on_the_boards_own_fixture_too():
    """The compiled board reads ``leak_frac`` off the #22 sheet cell."""
    board = sheet_leak(cand("faithful_gain_1.5"), steps=STEPS)
    assert board is not None
    assert board["board_leak_frac"] < LEAK_FRAC_WIN
    assert board["gender_leak_frac"] < LEAK_FRAC_WIN
    assert board["gender_pass"]


def test_the_hit_survives_every_seed():
    for seed in range(4):
        assert scored("faithful_gain_1.5", seed=seed)["pass"], seed


def test_the_hit_passes_the_close_and_unused_cells_too():
    for cell in ("close", "unused_e"):
        row = scored("faithful_gain_1.5", cell=cell)
        assert row["pass"], cell
        assert row["leak_frac"] < LEAK_FRAC_WIN, cell


def test_deleting_the_blend_fails_the_divergent_exam():
    """The useful negative: on two tracks the blend inside ``c`` is load-bearing."""
    agree = scored("faithful_common_agree")
    trap = scored("pair_odd_midpoint")
    assert agree["leak_frac"] < LEAK_FRAC_WIN
    assert not agree["pass"]
    # Same gate, same failure, as the named midpoint trap.
    assert agree["axis"]["same_words"] == "needs_help"
    assert trap["axis"]["same_words"] == "needs_help"
    assert agree["roll_match_kept"] == pytest.approx(trap["roll_match_kept"], abs=0.05)
    # And on one song it is the caption pair, so it cannot be a midpoint dressed up.
    assert scored("faithful_common_agree", cell="close")["pass"]


def test_the_blind_band_is_the_expensive_direction():
    """At matched ``leak_frac`` the blind-only gain is the seed-fragile one."""
    whole = [scored("faithful_gain_1.5", seed=s) for s in range(4)]
    blind = [scored("blind_gain_3", seed=s) for s in range(4)]
    assert blind[0]["leak_frac"] == pytest.approx(whole[0]["leak_frac"], abs=0.05)
    assert all(row["pass"] for row in whole)
    assert not all(row["pass"] for row in blind)
    assert min(r["worst_margin"] for r in whole) > min(
        r["worst_margin"] for r in blind
    )


def test_the_blind_band_buys_less_before_the_exam_goes():
    """Inside its own all-seed window it reaches a third of the leak_frac."""
    whole = gain_window(gain_sweep(steps=200, seeds=(0, 1, 2), blind=False))
    blind = gain_window(gain_sweep(steps=200, seeds=(0, 1, 2), blind=True))
    assert whole["best"]["leak_frac"] < blind["best"]["leak_frac"] < LEAK_FRAC_WIN


def test_semantic_kl_cannot_deliver_a_blind_band_over_drive():
    """Zero gradient there means the target's extra axis never arrives."""
    asked = scored("blind_gain_3_semantic_kl")
    plain = scored("semantic_kl_poles")
    assert asked["leak_frac"] == pytest.approx(plain["leak_frac"], abs=1e-4)
    # The target really did ask for it; the loss just could not hear.
    assert asked["target_leak_frac"] < plain["target_leak_frac"] - 0.2


def test_the_gain_window_is_bounded_on_both_sides():
    sweep = gain_sweep(steps=200, seeds=(0, 1, 2))
    window = gain_window(sweep)
    assert window["low"] is not None and window["low"] > 1.0
    assert window["high"] is not None and window["high"] < max(
        r["gain"] for r in sweep
    )
    # Below the window the sign has not flipped; above it the ±1 ends are
    # driven past any caption and the continuation garbles.
    assert sweep[0]["leak_frac"] > LEAK_FRAC_WIN
    assert sweep[-1]["roll_off_corpus"] > 0.2


# -- what a hit does not mean ---------------------------------------------


def test_the_leftover_column_improves_for_a_reason_that_is_not_real():
    """``leak_tok`` falls with γ; the hidden leftover ratio does not move."""
    rows = leftover_saturation()
    assert rows[0]["leak_tok"] > rows[-1]["leak_tok"] * 10.0
    base = rows[0]["hidden_ratio"]
    for row in rows:
        assert row["hidden_ratio"] == pytest.approx(base, abs=1e-5)
    # The hidden state carries *more* of the attribute, not less.
    assert rows[-1]["hidden_unused"] > rows[0]["hidden_unused"]


def test_the_sheet_leftover_pass_is_that_same_artifact():
    """γ walks the #22 leak lock without removing anything, and says so."""
    base = sheet_leak(cand("faithful_raw"), steps=STEPS)
    gain = sheet_leak(cand("faithful_gain_1.5"), steps=STEPS)
    assert not base["leftover_pass"]
    assert gain["leftover_pass"]
    assert gain["leftover_leak_tok"] < base["leftover_leak_tok"]
    # The one column on that cell the gain does not improve.
    assert gain["leftover_garble"] >= base["leftover_garble"]


def test_only_one_divergent_gate_moves_across_the_gain_family():
    """A win in this region rests on ``same_words``, and the page says so."""
    for name in ("faithful_gain_1.1", "faithful_gain_1.5", "faithful_gain_2"):
        row = scored(name)
        assert row["roll_overlap"] == pytest.approx(1.0, abs=1e-6)
        assert row["roll_off_corpus"] == pytest.approx(0.0, abs=1e-6)
        assert row["roll_coherence"] == pytest.approx(1.0, abs=1e-6)


def test_frontier_is_reported_either_way():
    summary = by_candidate(
        [
            score_candidate(c, WIN_CELL, seed=seed, steps=200)
            for c in candidates()
            for seed in (0, 1)
        ]
    )
    front = frontier(summary)
    assert front["divergent_passers"] > 0
    assert front["leak_frac_negative"] > 0
    assert front["best_leak_frac_among_divergent_passers"] is not None
    assert front["best_divergent_among_leak_frac_negative"] is not None
    assert hits(summary)


# -- the live wire --------------------------------------------------------


def test_live_default_is_unchanged():
    from conceptmod.textsliders.train_lm_slider_music3 import parse_args

    args = parse_args(["--prompts_file", "prompts/prompts-energy-v4.yaml"])
    assert args.lm_target == "v9"
    assert args.pole_mode == "hidden"
    assert args.target_scale == 1.0


def test_faithful_gain_is_a_live_recipe_that_needs_its_gain():
    assert "faithful_gain" in LM_RECIPES
    assert resolve_lm_recipe(lm_target="faithful_gain", symmetric=True) == "faithful_gain"
    assert resolve_gain_scale("faithful_gain", 1.5) == 1.5
    with pytest.raises(ValueError, match="target_scale"):
        resolve_gain_scale("faithful_gain", 1.0)
    with pytest.raises(ValueError):
        resolve_gain_scale("faithful_gain", -1.0)
    # Every other recipe keeps the flag it already had.
    assert resolve_gain_scale("v9", 1.0) == 1.0


def test_live_targets_match_the_fixture_teacher():
    field = CELLS[WIN_CELL]()
    pos, neg, neu = field.poles(0)
    live_plus, live_minus, anchor_plus, anchor_minus = lm_train_targets(
        pos, neg, neu, recipe="faithful_gain", target_scale=1.5
    )
    exam_plus, exam_minus = teacher_points(
        field, 0, teacher="faithful_gain", target_scale=1.5
    )
    assert torch.allclose(live_plus, exam_plus, atol=1e-6)
    assert torch.allclose(live_minus, exam_minus, atol=1e-6)
    assert anchor_plus is None and anchor_minus is None


def test_live_targets_realize_the_predicted_leak_frac():
    """The live function and the fixture agree on the number, per row."""
    field = CELLS[WIN_CELL]()
    for row in range(field.rows):
        pos, neg, neu = field.poles(row)
        plus, minus, _a, _b = lm_train_targets(
            pos, neg, neu, recipe="faithful_gain", target_scale=1.5
        )
        assert teacher_leak_frac(plus, minus, neu) < LEAK_FRAC_WIN
    residual, _room, _final = fit_exam(
        field, pole_mode="hidden", teacher="faithful_gain", target_scale=1.5, steps=STEPS
    )
    bipolar = leftover_bipolar(residual.delta(1.0), residual.delta(-1.0))
    assert bipolar["leak_frac"] < LEAK_FRAC_WIN


def test_worst_margin_agrees_with_the_exam_verdict():
    for name in ("faithful_gain_1.5", "pair_odd_midpoint", "faithful_common_agree"):
        row = scored(name)
        assert (worst_margin(row) > 0.0) == bool(row["pass"]), name
