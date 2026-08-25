"""Compiled 2-D / high-D / sheet / pair-exam gate.

Re-runs the fixtures rather than reading
``docs/lm-2d-scoreboard/metrics.json``. Every column the live logs made
look healthy — pair-odd cos, the ±1 collapse, the pole loss, ``p%`` — is
accepted by the gate helpers and ignored. Nothing here changes the live
trainer default.
"""

from __future__ import annotations

import pytest

from analysis.slider2d.exam import LIVE_EXAM, LIVE_ROW
from analysis.slider2d.scoreboard import (
    CELL_ORDER,
    COMPILED_GARBLE_MAX,
    COMPILED_LEAK_LOCK,
    COMPILED_SHEET_LOCK,
    COMPILED_SWING_FLOOR,
    EXAM_ALIAS,
    FAILS,
    UNSCORED,
    WORKS,
    WORKS_SOME,
    cell_works,
    collect_scoreboard,
    compile_sheet_row,
    compiled_verdict,
    exam_cells_for,
    exam_score,
    failing_cells,
    leak_ok,
    live_exam_report,
    predicts_for,
    sheet_ok,
    sort_rows,
)
from analysis.slider2d.sheet import leaky_cell
from conceptmod.textsliders.train_lm_slider_music3 import (
    LM_RECIPES,
    POLE_MODES,
    parse_args,
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
    locked = cell_works(
        leak=0.0,
        on_sheet_kept=0.34,
        off_sheet=0.41,
        argmax_on_sheet=0.0,
        swing_kept=0.27,
        pair_odd_cos=1.0,
        collapse=-1.0,
    )
    unlocked = cell_works(
        leak=0.0,
        on_sheet_kept=0.34,
        off_sheet=0.41,
        argmax_on_sheet=0.0,
        swing_kept=0.27,
        pair_odd_cos=0.2,
        collapse=0.2,
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
    assert "semantic_kl_pin" in POLE_MODES
    pin = parse_args(
        ["--prompts", "x.yaml", "--lm_target", "faithful", "--pole_mode", "semantic_kl_pin"]
    )
    assert pin.lm_target == "faithful"
    assert pin.pole_mode == "semantic_kl_pin"
    assert pin.null_pin_weight == 1.0


def test_the_new_caption_plus_pin_recipes_top_exam_score():
    """Not a rename of faithful_raw: different loss, same exam_score band."""
    ids = by_id()
    for name in ("semantic_kl_pin", "unrolled_kl"):
        row = ids[name]
        assert row["exam_score"] is not None
        assert row["exam_score"] >= 0.95
        assert row["cells"]["exam_divergent"] is True
        assert row["cells"]["exam_close"] is True
        assert row["predicts"] == {}
    order = [r["id"] for r in board()]
    assert order.index("semantic_kl_pin") < order.index("semantic_kl_poles")
    assert order.index("unrolled_kl") < order.index("semantic_kl_poles")
