"""Compiled 2-D / high-D / sheet / pair-exam gate.

Re-runs the fixtures rather than reading
``docs/lm-2d-scoreboard/metrics.json``. Every column the live logs made
look healthy — pair-odd cos, the ±1 collapse, the pole loss, ``p%``,
``leak_frac`` / ``same_dir``, off-caption — is accepted by the gate
helpers and ignored. Nothing here changes the live trainer default.
"""

from __future__ import annotations

import pytest

from analysis.slider2d.exam import LIVE_EXAM, LIVE_ROW
from analysis.slider2d.scoreboard import (
    BIPOLAR_MIRROR_BAND,
    BIPOLAR_MIRROR_FLOOR,
    CELL_ORDER,
    COMPILED_GARBLE_MAX,
    COMPILED_LEAK_LOCK,
    COMPILED_SHEET_LOCK,
    COMPILED_SWING_FLOOR,
    EXAM_ALIAS,
    FAILS,
    HIGH_LEAK_RECIPES,
    JOINT_MARK_RECIPES,
    RACE_RECIPES,
    SAME_DIR_BAND,
    UNSCORED,
    WORKS,
    WORKS_SOME,
    bipolar_from,
    cell_works,
    collect_scoreboard,
    compile_sheet_row,
    compiled_verdict,
    exam_cells_for,
    exam_score,
    failing_cells,
    gates_blob,
    in_want_leak_band,
    joint_axes,
    joint_chart_rows,
    joint_hit,
    leak_band,
    leak_frac_chart_rows,
    leak_ok,
    live_exam_report,
    predicts_for,
    recipe_off_caption,
    sheet_ok,
    sort_rows,
    two_of_three,
)
from analysis.slider2d.sheet import leaky_cell
from conceptmod.textsliders.train_lm_slider_music3 import (
    LM_RECIPES,
    POLE_MODES,
    parse_args,
    resolve_pole_mode,
)


STEPS = 400
_CACHE: dict[str, object] = {}


def leftover() -> dict[str, dict]:
    if "leftover" not in _CACHE:
        _CACHE["leftover"] = {row["name"]: row for row in leaky_cell(steps=STEPS)}
    return _CACHE["leftover"]


def compiled(name: str) -> dict:
    return compile_sheet_row(leftover()[name])


def board() -> list[dict]:
    if "board" not in _CACHE:
        _CACHE["board"] = collect_scoreboard(
            sheet_steps=STEPS, other_steps=200, exam_steps=STEPS
        )
    return _CACHE["board"]


def by_id() -> dict[str, dict]:
    return {row["id"]: row for row in board()}


# -- the gate itself -----------------------------------------------------


def test_no_goodhart_column_is_an_input_to_the_compiled_score():
    """A perfect lock, a solved loss and a small p% cannot flip fail → works."""
    failing = dict.fromkeys(CELL_ORDER, False)
    passing = {**failing, "exam_divergent": True}
    assert (
        compiled_verdict(cells=failing, pair_odd_cos=1.0, loss=0.0, pperc=0.0) == FAILS
    )
    assert compiled_verdict(cells=passing, pair_odd_cos=0.1) == WORKS_SOME
    assert (
        compiled_verdict(cells=failing, leak_frac=-1.0, same_dir=0.0) == FAILS
    )
    assert compiled_verdict(cells=passing, leak_frac=0.9, same_dir=1.0) == WORKS_SOME
    locked = cell_works(
        leak=0.0,
        on_sheet_kept=0.34,
        off_sheet=0.41,
        argmax_on_sheet=0.0,
        swing_kept=0.27,
        pair_odd_cos=1.0,
        collapse=-1.0,
        leak_frac=-1.0,
        same_dir=0.0,
    )
    unlocked = cell_works(
        leak=0.0,
        on_sheet_kept=0.34,
        off_sheet=0.41,
        argmax_on_sheet=0.0,
        swing_kept=0.27,
        pair_odd_cos=0.2,
        collapse=0.2,
        leak_frac=0.9,
        same_dir=1.0,
    )
    assert locked is False
    assert unlocked is False


def test_works_requires_every_pair_it_is_read_on():
    all_pass = {name: True for name in CELL_ORDER}
    assert compiled_verdict(cells=all_pass) == WORKS
    partial = {**all_pass, "exam_close": False}
    assert compiled_verdict(cells=partial) == WORKS_SOME
    assert failing_cells(partial) == ["exam_close"]
    sparse = dict.fromkeys(CELL_ORDER, None)
    assert compiled_verdict(cells=sparse) == UNSCORED
    sparse["exam_divergent"] = True
    assert compiled_verdict(cells=sparse) == WORKS


def test_the_sheet_gate_numbers_are_unchanged():
    assert leak_ok(0.0) is True
    assert leak_ok(COMPILED_LEAK_LOCK) is True
    assert leak_ok(COMPILED_LEAK_LOCK + 0.01) is False
    assert (
        sheet_ok(
            on_sheet_kept=COMPILED_SHEET_LOCK,
            off_sheet=COMPILED_GARBLE_MAX,
            argmax_on_sheet=1.0,
            swing_kept=COMPILED_SWING_FLOOR,
        )
        is True
    )
    assert (
        sheet_ok(on_sheet_kept=0.88, off_sheet=0.01, argmax_on_sheet=1.0, swing_kept=1.0)
        is False
    )
    assert cell_works(leak=0.0, intended_cos=0.99, rich_kept=1.0) is True
    assert cell_works(leak=0.4, intended_cos=0.99) is False


def test_exam_score_is_min_overlap_and_swing_on_live_pairs_only():
    """unused_e cannot inflate the number; a missing pair is skipped, not 1.0."""
    poles = exam_score(
        {"exam_divergent": 1.0, "exam_close": 0.90625, "exam_unused_e": 1.0},
        {"exam_divergent": 1.0, "exam_close": 0.38728, "exam_unused_e": 0.90},
    )
    sub = exam_score(
        {"exam_divergent": 0.552, "exam_unused_e": 0.99},
        {"exam_divergent": 0.104, "exam_unused_e": 0.99},
    )
    faithful = exam_score(
        {"exam_divergent": 1.0, "exam_close": 1.0, "exam_unused_e": 1.0},
        {"exam_divergent": 1.0, "exam_close": 1.0, "exam_unused_e": 1.0},
    )
    assert faithful == 1.0
    assert poles == 0.38728
    assert sub == 0.104
    assert poles > sub
    assert exam_score({}, {}) is None
    assert exam_score({"exam_unused_e": 0.05}, {"exam_unused_e": 0.05}) is None
    assert exam_score(
        {"exam_divergent": 0.97},
        {"exam_divergent": 0.97},
    ) == 0.97


def test_recipe_off_caption_is_max_on_live_pairs_and_not_a_free_zero():
    """unused_e cannot hide smear; a missing pair is skipped, not 0.0."""
    worst, pair = recipe_off_caption(
        {"exam_divergent": 0.016, "exam_close": 0.0, "exam_unused_e": 0.90}
    )
    assert worst == pytest.approx(0.016)
    assert pair == "exam_divergent"
    close_only, close_pair = recipe_off_caption({"exam_close": 0.04})
    assert close_only == pytest.approx(0.04)
    assert close_pair == "exam_close"
    assert recipe_off_caption({}) == (None, None)
    assert recipe_off_caption({"exam_unused_e": 0.90}) == (None, None)
    # A midpoint reading of 0.016 must not collapse to "clean".
    mid, _ = recipe_off_caption({"exam_divergent": 0.016, "exam_close": 0.0})
    assert mid > 0.0


def test_sort_is_exam_score_descending_nulls_last():
    def row(rid, score, verdict=WORKS_SOME):
        return {"id": rid, "compiled": verdict, "exam_score": score, "cells": {}}

    rows = sort_rows(
        [
            row("semantic_kl_sub_e", 0.10),
            row("semantic_kl_poles", 0.39),
            row("faithful_raw", 1.0),
            row("faithful_attrs", 1.0, WORKS),
            row("hub", None, FAILS),
        ]
    )
    assert [r["id"] for r in rows] == [
        "faithful_attrs",
        "faithful_raw",
        "semantic_kl_poles",
        "semantic_kl_sub_e",
        "hub",
    ]


# -- the live exam -------------------------------------------------------


def test_the_board_reproduces_all_three_live_listens():
    rows = live_exam_report(exam_steps=STEPS)
    assert [r["run"] for r in rows] == list(LIVE_EXAM)
    for row in rows:
        assert row["agrees"], (
            f"{row['run']}: board says {row['predicted']}, ears said "
            f"{row['listen']} ({row['reason']})"
        )


def test_the_energy_win_outranks_the_energy_garble():
    """The #23 ordering this board replaces had these the other way round."""
    rows = board()
    order = [r["id"] for r in rows]
    win, garble = "semantic_kl_poles", "semantic_kl_sub_e"
    assert order.index(win) < order.index(garble)
    assert by_id()[win]["cells"]["exam_divergent"] is True
    assert by_id()[garble]["cells"]["exam_divergent"] is False


def test_exam_score_sorts_the_three_live_exam_rows():
    """energy-v18's recipe above energy-v16's; faithful_raw/attrs outrank both."""
    rows = board()
    order = [r["id"] for r in rows]
    ids = by_id()
    for name in (
        "faithful_attrs",
        "faithful_raw",
        "hidden_beta1",
        "semantic_kl_poles",
        "semantic_kl_sub_e",
    ):
        assert ids[name]["exam_score"] is not None
    assert ids["hub"]["exam_score"] is None
    assert ids["project_short_u"]["exam_score"] is None
    assert order.index("faithful_attrs") < order.index("semantic_kl_poles")
    assert order.index("faithful_raw") < order.index("semantic_kl_poles")
    assert order.index("hidden_beta1") < order.index("semantic_kl_poles")
    assert order.index("semantic_kl_poles") < order.index("semantic_kl_sub_e")
    # The live energy-v16 fail is the sub_e row; unused_e must not rescue it.
    assert ids["semantic_kl_sub_e"]["exam_overlap"]["exam_unused_e"] > 0.9
    assert ids["semantic_kl_sub_e"]["exam_score"] < 0.5
    assert ids["semantic_kl_poles"]["exam_score"] > ids["semantic_kl_sub_e"]["exam_score"]
    scored = [i for i, r in enumerate(rows) if r["exam_score"] is not None]
    nulls = [i for i, r in enumerate(rows) if r["exam_score"] is None]
    assert scored and nulls and max(scored) < min(nulls)


def test_every_live_run_is_marked_on_the_row_that_predicted_it():
    rows = by_id()
    for run, (recipe, cell) in LIVE_ROW.items():
        assert rows[recipe]["predicts"][f"exam_{cell}"] == run
    assert predicts_for("semantic_kl_poles") == {
        "exam_divergent": "energy-lm-v18",
        "exam_close": "gender-lm-v16",
    }
    # An alias must not claim a live run it did not train.
    for alias in EXAM_ALIAS:
        assert predicts_for(alias) == {}


def test_the_same_recipe_splits_across_the_two_pairs():
    row = by_id()["semantic_kl_poles"]
    assert row["cells"]["exam_divergent"] is True
    assert row["cells"]["exam_close"] is False
    assert row["compiled"] == WORKS_SOME
    assert "KL-small / hidden-far" in row["exam_flags"].values()


def test_the_untrained_winner_is_faithful_on_hidden_mse():
    """The next live card: `--lm_target faithful --pole_mode hidden`."""
    row = by_id()["faithful_raw"]
    assert row["predicts"] == {}
    for cell in ("exam_divergent", "exam_close", "exam_unused_e"):
        assert row["cells"][cell] is True
    # Its only failing cell is the unused-ê sheet, which gender-v4 has no
    # `leak_*` to trip.
    assert failing_cells(row["cells"]) == ["sheet_leftover"]


def test_exam_readings_are_never_invented_for_a_recipe_the_cell_cannot_express():
    for row in board():
        found = exam_cells_for(row["id"], None)["cells"]
        assert all(v is None for v in found.values())
    assert by_id()["hub"]["cells"]["exam_divergent"] is None
    assert by_id()["project_rich_u"]["cells"]["exam_close"] is None


# -- the sheet cell still says what it said ------------------------------


def test_hidden_midpoint_still_fails_the_sheet():
    row = leftover()["v9_hidden"]
    compiled_row = compiled("v9_hidden")
    assert row["pair_odd_cos"] >= 0.99
    assert row["collapse"] <= -0.99
    assert row["on_sheet_kept"] < COMPILED_SHEET_LOCK
    assert row["garble"] > COMPILED_GARBLE_MAX
    assert compiled_row["cells"]["sheet_leftover"] is False


def test_the_e_cleaned_pole_targets_still_pass_the_sheet():
    for name in ("faithful_sub_e", "v16_semantic_kl_sub_e"):
        row = leftover()[name]
        compiled_row = compiled(name)
        assert abs(row["leak_tok"]) <= COMPILED_LEAK_LOCK
        assert row["on_sheet_kept"] >= COMPILED_SHEET_LOCK
        assert row["garble"] <= COMPILED_GARBLE_MAX
        assert row["argmax_on_sheet"] == 1.0
        assert row["swing_kept"] >= COMPILED_SWING_FLOOR
        assert compiled_row["cells"]["sheet_leftover"] is True
        assert row["pair_odd_cos"] < leftover()["v9_hidden"]["pair_odd_cos"]


def test_passing_the_sheet_is_no_longer_enough_to_work():
    """#23 called `semantic_kl_sub_e` a winner. The divergent pair does not."""
    assert compiled("v16_semantic_kl_sub_e")["cells"]["sheet_leftover"] is True
    assert by_id()["semantic_kl_sub_e"]["cells"]["exam_divergent"] is False
    assert by_id()["semantic_kl_sub_e"]["compiled"] != WORKS


def test_pair_odd_sub_e_still_fails_the_sheet():
    row = leftover()["v15_pair_odd_sub_e"]
    compiled_row = compiled("v15_pair_odd_sub_e")
    assert abs(row["leak_tok"]) <= COMPILED_LEAK_LOCK
    assert row["on_sheet_kept"] < COMPILED_SHEET_LOCK
    assert row["garble"] > COMPILED_GARBLE_MAX
    assert compiled_row["cells"]["sheet_leftover"] is False


def test_a_caption_target_without_e_cleaning_still_leaks_on_the_sheet():
    """The cell the board no longer uses as the energy stand-in."""
    for name in ("v6_faithful", "v16_semantic_kl"):
        row = leftover()[name]
        compiled_row = compiled(name)
        assert row["on_sheet_kept"] >= COMPILED_SHEET_LOCK
        assert abs(row["leak_tok"]) > COMPILED_LEAK_LOCK
        assert compiled_row["cells"]["sheet_leftover"] is False


# -- live default --------------------------------------------------------


def test_the_live_default_is_still_v9_on_hidden_mse():
    args = parse_args(["--prompts", "x.yaml"])
    assert args.lm_target == "v9"
    assert args.pole_mode == "hidden"
    assert args.common_beta == 0.0
    assert "v9" in LM_RECIPES
    assert "semantic_kl" in POLE_MODES
    assert "faithful_sub_e_if_unused" in LM_RECIPES
    assert "faithful_even_blend" in LM_RECIPES
    assert "faithful_guard_e" in LM_RECIPES
    assert "semantic_kl_null" in POLE_MODES
    assert "hidden_kl" in POLE_MODES
    assert "dual_band" in POLE_MODES
    assert "unrolled_kl" not in POLE_MODES
    assert resolve_pole_mode("hidden") == "hidden"


def test_hybrid_pole_mode_aliases_resolve_to_semantic_kl_null():
    """#29 / #32 / #33 are one hybrid. Old flags must not fork."""
    assert resolve_pole_mode("semantic_kl_null") == "semantic_kl_null"
    assert resolve_pole_mode("semantic_kl_plus_hidden") == "semantic_kl_null"
    assert resolve_pole_mode("semantic_kl_pin") == "semantic_kl_null"
    pin = parse_args(
        ["--prompts", "x.yaml", "--lm_target", "faithful", "--pole_mode", "semantic_kl_pin"]
    )
    plus = parse_args(
        [
            "--prompts",
            "x.yaml",
            "--lm_target",
            "faithful",
            "--pole_mode",
            "semantic_kl_plus_hidden",
        ]
    )
    assert pin.pole_mode == "semantic_kl_pin"
    assert plus.pole_mode == "semantic_kl_plus_hidden"
    assert resolve_pole_mode(pin.pole_mode) == "semantic_kl_null"
    assert resolve_pole_mode(plus.pole_mode) == "semantic_kl_null"


def test_combined_race_recipes_are_on_the_board_next_to_baselines():
    ids = by_id()
    for name in RACE_RECIPES:
        assert name in ids, name
        assert ids[name]["exam_score"] is not None
        assert ids[name]["cells"]["exam_divergent"] is not None
        assert ids[name]["cells"]["exam_close"] is not None
    for name in RACE_RECIPES - {"dual_band_midpoint"}:
        assert ids[name]["cells"]["exam_divergent"] is True, name
        assert ids[name]["cells"]["exam_close"] is True, name
    for baseline in (
        "faithful_attrs",
        "faithful_raw",
        "faithful_even_blend",
        "hidden_beta1",
        "gender_like_no_e",
        "pair_odd_midpoint",
        "hold_e_perp_l8",
        "pair_odd_sub_e",
        "faithful_sub_e",
        "semantic_kl_midpoint",
        "semantic_kl_poles",
        "semantic_kl_sub_e",
        "hub",
        "project_short_u",
        "hold_e_raw_l1",
    ):
        assert baseline in ids
    assert "semantic_kl_plus_hidden" not in ids
    assert "semantic_kl_pin" not in ids


def test_the_gated_leftover_row_hits_exam_score_one_on_both_live_pairs():
    """Not a rename of faithful_raw: leftover ê is cleaned, energy poles stay."""
    row = by_id()["faithful_sub_e_if_unused"]
    raw = by_id()["faithful_raw"]
    assert row["exam_score"] == pytest.approx(raw["exam_score"], abs=1e-6)
    assert row["exam_score"] >= 0.99
    assert row["cells"]["exam_divergent"] is True
    assert row["cells"]["exam_close"] is True
    assert row["cells"]["exam_unused_e"] is True
    assert row["cells"]["sheet_leftover"] is True
    assert row["cells"]["sheet_gender"] is True
    assert row["compiled"] == WORKS
    assert failing_cells(row["cells"]) == []
    assert row["predicts"] == {}
    assert raw["cells"]["sheet_leftover"] is False
    assert row["id"] != "faithful_raw"
    assert row["exam_score"] > by_id()["faithful_sub_e"]["exam_score"]


def test_semantic_kl_null_tops_both_exam_pairs():
    """KL + null-space pin: caption poles, delivery arrives, both pair types."""
    row = by_id()["semantic_kl_null"]
    assert row["exam_score"] is not None
    assert row["exam_score"] >= 0.99
    assert row["cells"]["exam_divergent"] is True
    assert row["cells"]["exam_close"] is True
    poles = by_id()["semantic_kl_poles"]
    assert row["exam_score"] > poles["exam_score"]
    assert poles["cells"]["exam_close"] is False


def test_new_hidden_kl_recipe_scores_one_on_close_and_divergent_pairs():
    row = by_id()["hidden_kl_poles"]
    raw = by_id()["faithful_raw"]
    assert row["exam_score"] is not None
    assert row["exam_score"] >= 0.99
    assert row["exam_score"] == pytest.approx(raw["exam_score"], abs=1e-3)
    assert row["cells"]["exam_divergent"] is True
    assert row["cells"]["exam_close"] is True
    assert row["exam_overlap"]["exam_divergent"] >= 0.99
    assert row["exam_overlap"]["exam_close"] >= 0.99
    assert row["exam_swing"]["exam_divergent"] >= 0.99
    assert row["exam_swing"]["exam_close"] >= 0.99
    assert row["id"] not in EXAM_ALIAS


def test_unrolled_kl_has_a_real_exam_reading_and_is_not_live():
    row = by_id()["unrolled_kl"]
    assert row["exam_score"] is not None
    assert row["exam_score"] >= 0.95
    assert row["cells"]["exam_divergent"] is True
    assert row["cells"]["exam_close"] is True
    assert "unrolled_kl" not in POLE_MODES


def test_faithful_guard_e_has_a_real_pair_exam_reading():
    row = by_id()["faithful_guard_e"]
    raw = by_id()["faithful_raw"]
    assert row["exam_score"] is not None
    assert row["exam_score"] >= 0.99
    assert row["exam_score"] == pytest.approx(raw["exam_score"], abs=1e-3)
    assert row["cells"]["exam_divergent"] is True
    assert row["cells"]["exam_close"] is True
    assert row["cells"]["sheet_leftover"] is True
    assert row["cells"]["sheet_gender"] is True
    assert row["id"] != "faithful_sub_e_if_unused"


def test_dual_band_rows_have_real_pair_exam_readings():
    dual = by_id()["dual_band_poles"]
    guarded = by_id()["dual_band_guard_e"]
    poles = by_id()["semantic_kl_poles"]
    assert dual["exam_score"] is not None
    assert guarded["exam_score"] is not None
    assert dual["exam_score"] >= 0.95
    assert guarded["exam_score"] >= 0.95
    assert dual["cells"]["exam_divergent"] is True
    assert dual["cells"]["exam_close"] is True
    assert guarded["cells"]["exam_divergent"] is True
    assert guarded["cells"]["exam_close"] is True
    assert dual["exam_score"] > poles["exam_score"]
    assert guarded["exam_score"] > poles["exam_score"]


def test_dual_band_midpoint_fails_the_divergent_pair():
    control = by_id()["dual_band_midpoint"]
    assert control["exam_score"] is not None
    assert control["cells"]["exam_divergent"] is False
    assert control["cells"]["exam_close"] is not None
    assert by_id()["dual_band_poles"]["cells"]["exam_divergent"] is True


def test_off_caption_is_logged_on_pair_exam_rows_and_midpoint_is_not_a_free_zero():
    """Lyric smear is present on every pair-exam row; pair-odd is not 0."""
    ids = by_id()
    for name in (
        "faithful_sub_e_if_unused",
        "faithful_even_blend",
        "faithful_raw",
        "pair_odd_midpoint",
        "dual_band_poles",
        "dual_band_guard_e",
        "dual_band_midpoint",
        "hidden_beta1",
    ):
        row = ids[name]
        assert row["exam_off_caption"], name
        assert any(cell in row["exam_off_caption"] for cell in ("exam_divergent", "exam_close")), name
        assert row["off_caption"] is not None, name
        assert row["off_caption_pair"] in row["exam_off_caption"], name
        assert row["exam_same_words"], name
        assert row["exam_coherence"], name
        # exam_score stays min(overlap, swing); smear is a sibling log.
        assert row["exam_score"] == pytest.approx(
            exam_score(row["exam_overlap"], row["exam_swing"]), abs=1e-9
        )
    mid = ids["pair_odd_midpoint"]
    assert mid["off_caption"] > 0.0
    assert mid["exam_off_caption"]["exam_divergent"] > 0.0
    # A recipe the pair-exam cell cannot express is null, not a free 0.
    assert ids["hub"]["off_caption"] is None
    assert ids["hub"]["exam_off_caption"] == {}
    assert ids["project_short_u"]["off_caption"] is None
    # The compiled gate still ignores smear as an extra column.
    failing = dict.fromkeys(CELL_ORDER, False)
    assert compiled_verdict(cells=failing) == FAILS
    assert ids["faithful_even_blend"]["cells"]["exam_divergent"] is not None
    assert ids["faithful_even_blend"]["id"] != "gate_odd_even_blend_s50"


def test_leftover_bipolar_is_wired_into_every_board_row():
    """leak_frac / same_dir come from leftover_bipolar on the fitted student."""
    gates = gates_blob()
    assert gates["leak_frac_scored"] is False
    assert gates["same_dir_scored"] is False
    assert gates["off_caption_scored"] is False
    assert gates["bipolar_mirror_floor"] == BIPOLAR_MIRROR_FLOOR
    for row in board():
        assert row.get("leak_frac") is not None, row["id"]
        assert row.get("same_dir") is not None, row["id"]
        assert -1.05 <= float(row["leak_frac"]) <= 1.05
        assert 0.0 <= float(row["same_dir"]) <= 1.0
    sheet_row = leftover()["v9_hidden"]
    leak_frac, same_dir = bipolar_from(sheet_row)
    assert leak_frac == pytest.approx(sheet_row["leak_frac"], abs=1e-9)
    assert same_dir == pytest.approx(sheet_row["same_dir"], abs=1e-9)
    assert "leak_frac" in leftover()["v6_faithful"]
    assert leftover()["v6_faithful"]["leak_frac"] > BIPOLAR_MIRROR_FLOOR


def test_pair_odd_midpoint_is_bipolar_and_faithful_is_not():
    """Clean pair-odd lands near leak_frac −1; caption poles keep even motion."""
    odd = by_id()["pair_odd_midpoint"]
    faithful = by_id()["faithful_raw"]
    gated = by_id()["faithful_sub_e_if_unused"]
    assert odd["leak_frac"] == pytest.approx(-1.0, abs=0.05)
    assert odd["same_dir"] <= 0.05
    assert faithful["leak_frac"] > BIPOLAR_MIRROR_FLOOR
    assert gated["leak_frac"] > BIPOLAR_MIRROR_FLOOR
    assert abs(faithful["leak_frac"] - (-1.0)) > 0.4
    assert abs(gated["leak_frac"] - odd["leak_frac"]) > 0.4
    # leftover-gate clears unused ê and does not clear caption-pole even motion.
    assert abs(gated["leftover_leak"]) <= COMPILED_LEAK_LOCK
    assert abs(faithful["leftover_leak"]) > COMPILED_LEAK_LOCK
    assert gated["compiled"] == WORKS
    race = [by_id()[name] for name in sorted(RACE_RECIPES)]
    for row in race:
        if row["id"] == "dual_band_midpoint":
            continue
        assert row["leak_frac"] > BIPOLAR_MIRROR_FLOOR, row["id"]


def test_exam_score_does_not_see_leak_frac_or_off_caption():
    overlap = {"exam_divergent": 1.0, "exam_close": 0.5}
    swing = {"exam_divergent": 1.0, "exam_close": 0.5}
    assert exam_score(overlap, swing) == 0.5
    assert exam_score.__code__.co_varnames[: exam_score.__code__.co_argcount] == (
        "overlap",
        "swing",
    )


def test_leak_frac_chart_keeps_race_and_high_leak_and_sorts_same_dir_on_top():
    """Informative mix: leftover-gate next to hub / pair-odd, not a winner table."""
    chart = leak_frac_chart_rows(board())
    ids = [r["id"] for r in chart]
    for name in RACE_RECIPES | HIGH_LEAK_RECIPES:
        assert name in ids, name
    assert "faithful_even_blend" in ids
    assert ids == sorted(ids, key=lambda n: float(by_id()[n]["leak_frac"]))
    assert float(by_id()[ids[-1]]["leak_frac"]) >= float(by_id()[ids[0]]["leak_frac"])
    # same-dir / caption-pole at the top of the figure (last in barh order)
    assert leak_band(by_id()[ids[-1]]["leak_frac"]) == SAME_DIR_BAND
    assert leak_band(by_id()["pair_odd_midpoint"]["leak_frac"]) == BIPOLAR_MIRROR_BAND
    assert leak_band(by_id()["hub"]["leak_frac"]) == BIPOLAR_MIRROR_BAND
    missing = leak_frac_chart_rows(
        [{"id": "no_reading", "leak_frac": None}, by_id()["hub"]]
    )
    assert [r["id"] for r in missing] == ["hub"]
    # leftover-gate: unused ê gone, leak_frac still not bipolar
    gated = by_id()["faithful_sub_e_if_unused"]
    assert abs(gated["leftover_leak"]) <= COMPILED_LEAK_LOCK
    assert leak_band(gated["leak_frac"]) != BIPOLAR_MIRROR_BAND


def test_joint_overlay_keeps_high_leak_cousins_and_does_not_score():
    """Want-box is the three logged measurements; none enter the compiled gate."""
    chart = joint_chart_rows(board())
    ids = {r["id"] for r in chart}
    for name in RACE_RECIPES | HIGH_LEAK_RECIPES | JOINT_MARK_RECIPES:
        assert name in ids, name
    assert "faithful_even_blend" in ids
    assert "hidden_beta1" in ids
    # High-leak cousins without a pair-exam still appear so the overlay
    # is not a pile of 0.994s.
    assert "hub" in ids
    assert by_id()["hub"]["exam_score"] is None
    assert "caption_odd_margin" not in ids
    # Positive same-dir is a miss; ≈ −1 that fails divergent is Goodhart.
    assert in_want_leak_band(0.105) is False
    assert in_want_leak_band(0.0) is True
    assert in_want_leak_band(-0.40) is True
    assert in_want_leak_band(-0.80) is False
    assert in_want_leak_band(-1.0) is False
    odd = by_id()["pair_odd_midpoint"]
    gated = by_id()["faithful_sub_e_if_unused"]
    raw = by_id()["faithful_raw"]
    blend = by_id()["faithful_even_blend"]
    assert joint_hit(odd) is False
    assert leak_band(odd["leak_frac"]) == BIPOLAR_MIRROR_BAND
    assert odd["cells"]["exam_divergent"] is False
    # leftover-gate / v21 / faithful_raw pass both live pairs and sit
    # smear-clean, but leak_frac is same-dir (outside the want band).
    for row in (gated, raw, blend):
        axes = joint_axes(row)
        assert axes["exam"] is True, row["id"]
        assert axes["off_caption"] is True, row["id"]
        assert axes["leak_frac"] is False, row["id"]
        assert two_of_three(row) is True, row["id"]
        assert joint_hit(row) is False, row["id"]
    # A missing off-caption is not a free zero.
    assert joint_axes(by_id()["hub"])["off_caption"] is False
    failing = dict.fromkeys(CELL_ORDER, False)
    assert compiled_verdict(cells=failing, leak_frac=-0.4) == FAILS
    # The data-fix row is the only recipe in (−0.80, 0] that also
    # passes both live pairs with smear 0. leftover-gate / v21 stay
    # same-dir. Midpoint cousins stay at the bipolar lock.
    attrs = by_id()["faithful_attrs"]
    assert in_want_leak_band(attrs["leak_frac"]) is True
    assert joint_hit(attrs) is True
    assert [r["id"] for r in board() if joint_hit(r)] == ["faithful_attrs"]
