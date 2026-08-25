"""Compiled 2-D / high-D / sheet gate.

Re-runs the leftover sheet cell rather than reading
``docs/lm-2d-scoreboard/metrics.json``. Pair-odd cos is accepted by the
gate helpers and ignored. Nothing here changes the live trainer default.
"""

from __future__ import annotations

import pytest

from analysis.slider2d.scoreboard import (
    COMPILED_GARBLE_MAX,
    COMPILED_LEAK_LOCK,
    COMPILED_SHEET_LOCK,
    COMPILED_SWING_FLOOR,
    FAILS,
    WORKS,
    WORKS_GENDER_ONLY,
    cell_works,
    collect_scoreboard,
    compile_sheet_row,
    compiled_verdict,
    leak_ok,
    sheet_ok,
    sort_rows,
)
from analysis.slider2d.sheet import leaky_cell
from conceptmod.textsliders.train_lm_slider_music3 import (
    LM_RECIPES,
    parse_args,
)


STEPS = 400
_CACHE: dict[str, dict] = {}


def leftover() -> dict[str, dict]:
    if "leftover" not in _CACHE:
        _CACHE["leftover"] = {row["name"]: row for row in leaky_cell(steps=STEPS)}
    return _CACHE["leftover"]


def compiled(name: str) -> dict:
    return compile_sheet_row(leftover()[name])


def board() -> list[dict]:
    if "board" not in _CACHE:
        _CACHE["board"] = collect_scoreboard(
            sheet_steps=STEPS,
            other_steps=120,
            seed=0,
        )
    return _CACHE["board"]


# -- the gate itself -----------------------------------------------------


def test_pair_odd_cos_is_not_an_input_to_the_compiled_score():
    """A perfect lock cannot flip fail → works."""
    assert (
        compiled_verdict(leftover_works=False, gender_works=False, pair_odd_cos=1.0)
        == FAILS
    )
    assert (
        compiled_verdict(leftover_works=True, gender_works=False, pair_odd_cos=0.1)
        == WORKS
    )
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


def test_works_requires_small_leak_and_an_intact_sheet():
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
        sheet_ok(
            on_sheet_kept=0.88,
            off_sheet=0.01,
            argmax_on_sheet=1.0,
            swing_kept=1.0,
        )
        is False
    )
    assert cell_works(leak=0.0, intended_cos=0.99, rich_kept=1.0) is True
    assert cell_works(leak=0.4, intended_cos=0.99) is False
    assert (
        cell_works(
            leak=0.0,
            on_sheet_kept=0.94,
            off_sheet=0.001,
            argmax_on_sheet=1.0,
            swing_kept=1.1,
        )
        is True
    )


def test_gender_only_is_the_join_when_leftover_fails():
    assert (
        compiled_verdict(leftover_works=False, gender_works=True, pair_odd_cos=0.99)
        == WORKS_GENDER_ONLY
    )
    assert compiled_verdict(leftover_works=None, gender_works=True) == WORKS_GENDER_ONLY
    assert compiled_verdict(leftover_works=None, gender_works=False) == FAILS


def test_sort_is_verdict_then_leak_then_on_sheet():
    rows = sort_rows(
        [
            {"id": "b", "compiled": FAILS, "leftover_leak": 0.0, "on_sheet": 0.9},
            {"id": "a", "compiled": WORKS, "leftover_leak": 0.1, "on_sheet": 0.8},
            {"id": "c", "compiled": WORKS, "leftover_leak": 0.0, "on_sheet": 0.7},
            {"id": "d", "compiled": WORKS, "leftover_leak": 0.0, "on_sheet": 0.95},
            {"id": "e", "compiled": WORKS_GENDER_ONLY, "leftover_leak": 0.2, "on_sheet": 0.9},
        ]
    )
    assert [r["id"] for r in rows] == ["d", "c", "a", "e", "b"]


def test_live_exam_rows_match_and_rank_the_three_listens():
    by_id = {row["id"]: row for row in board()}
    energy_v16 = by_id["energy_v16_semantic_kl_sub_e"]
    energy_v18 = by_id["energy_v18_semantic_kl_faithful"]
    gender_v16 = by_id["gender_v16_semantic_kl_faithful"]
    assert energy_v16["live_run"] == "energy-lm-v16"
    assert energy_v16["expected_listen"] == "FAIL"
    assert energy_v16["compiled"] == FAILS
    assert energy_v18["live_run"] == "energy-lm-v18"
    assert energy_v18["expected_listen"] == "PASS"
    assert energy_v18["compiled"] == WORKS
    assert gender_v16["live_run"] == "gender-lm-v16"
    assert gender_v16["expected_listen"] == "FAIL"
    assert gender_v16["hidden_far_while_kl_small"] is True
    assert gender_v16["compiled"] == FAILS
    ranks = {row["id"]: i for i, row in enumerate(board())}
    assert ranks[energy_v18["id"]] < ranks[energy_v16["id"]]
    assert ranks[energy_v18["id"]] < ranks[gender_v16["id"]]


def test_old_unused_e_sheet_is_retained_but_not_the_energy_proxy():
    by_id = {row["id"]: row for row in board()}
    old = by_id["semantic_kl_sub_e_unused_e"]
    live = by_id["energy_v16_semantic_kl_sub_e"]
    assert old["live_run"] is None
    assert "unused-ê same-song" in old["label"]
    assert live["pair_kind"] == "divergent"
    assert live["live_run"] == "energy-lm-v16"
    assert live["compiled"] == FAILS


def test_hidden_faithful_close_pair_outranks_gender_semantic_kl():
    by_id = {row["id"]: row for row in board()}
    hidden = by_id["gender_hidden_faithful_next"]
    semantic = by_id["gender_v16_semantic_kl_faithful"]
    assert hidden["compiled"] == WORKS_GENDER_ONLY
    assert hidden["rollout_match"] > semantic["rollout_match"]
    assert hidden["rollout_garble"] < semantic["rollout_garble"]
    ranks = {row["id"]: i for i, row in enumerate(board())}
    assert ranks[hidden["id"]] < ranks[semantic["id"]]


# -- the four locked recipes --------------------------------------------


def test_hidden_midpoint_fails_the_compiled_gate():
    row = leftover()["v9_hidden"]
    compiled_row = compiled("v9_hidden")
    assert row["pair_odd_cos"] >= 0.99
    assert row["collapse"] <= -0.99
    assert row["on_sheet_kept"] < COMPILED_SHEET_LOCK
    assert row["garble"] > COMPILED_GARBLE_MAX
    assert compiled_row["compiled"] == FAILS
    assert compiled_row["leftover_works"] is False


def test_faithful_sub_e_passes_the_compiled_gate():
    row = leftover()["faithful_sub_e"]
    compiled_row = compiled("faithful_sub_e")
    assert abs(row["leak_tok"]) <= COMPILED_LEAK_LOCK
    assert row["on_sheet_kept"] >= COMPILED_SHEET_LOCK
    assert row["garble"] <= COMPILED_GARBLE_MAX
    assert row["argmax_on_sheet"] == 1.0
    assert row["swing_kept"] >= COMPILED_SWING_FLOOR
    assert compiled_row["compiled"] == WORKS
    assert row["pair_odd_cos"] < leftover()["v9_hidden"]["pair_odd_cos"]


def test_semantic_kl_sub_e_passes_the_compiled_gate():
    row = leftover()["v16_semantic_kl_sub_e"]
    compiled_row = compiled("v16_semantic_kl_sub_e")
    assert abs(row["leak_tok"]) <= COMPILED_LEAK_LOCK
    assert row["on_sheet_kept"] >= COMPILED_SHEET_LOCK
    assert row["garble"] <= COMPILED_GARBLE_MAX
    assert row["argmax_on_sheet"] == 1.0
    assert row["swing_kept"] >= COMPILED_SWING_FLOOR
    assert compiled_row["compiled"] == WORKS
    assert row["pair_odd_cos"] < leftover()["v9_hidden"]["pair_odd_cos"]


def test_pair_odd_sub_e_fails_the_sheet():
    """#20 zeros leftover leak and stays off-sheet."""
    row = leftover()["v15_pair_odd_sub_e"]
    compiled_row = compiled("v15_pair_odd_sub_e")
    assert abs(row["leak_tok"]) <= COMPILED_LEAK_LOCK
    assert row["on_sheet_kept"] < COMPILED_SHEET_LOCK
    assert row["garble"] > COMPILED_GARBLE_MAX
    assert compiled_row["compiled"] == FAILS
    assert compiled_row["leftover_works"] is False


def test_a_caption_target_without_e_cleaning_still_fails_leftover():
    """On-sheet is not enough: unused gender still leaks."""
    for name in ("v6_faithful", "v16_semantic_kl"):
        row = leftover()[name]
        compiled_row = compiled(name)
        assert row["on_sheet_kept"] >= COMPILED_SHEET_LOCK
        assert abs(row["leak_tok"]) > COMPILED_LEAK_LOCK
        assert compiled_row["leftover_works"] is False
        assert compiled_row["compiled"] != WORKS


# -- live default --------------------------------------------------------


def test_the_live_default_is_still_v9():
    args = parse_args(["--prompts", "x.yaml"])
    assert args.lm_target == "v9"
    assert args.pole_mode == "hidden"
    assert args.common_beta == 0.0
    assert "v9" in LM_RECIPES
