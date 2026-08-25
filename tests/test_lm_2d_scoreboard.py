"""Compiled 2-D / high-D / sheet gate, scored against the live exam.

Re-runs the exam cells rather than reading
``docs/lm-2d-scoreboard/metrics.json``. Pair-odd cos is accepted by the
gate helpers and ignored. Unused-gender leftover is not the energy
stand-in. Nothing here changes the live trainer default.
"""

from __future__ import annotations

import pytest

from analysis.slider2d.exam import close_cell, divergent_cell, live_exam_rows
from analysis.slider2d.scoreboard import (
    COMPILED_GARBLE_MAX,
    COMPILED_LEAK_LOCK,
    COMPILED_SHEET_LOCK,
    COMPILED_SWING_FLOOR,
    FAILS,
    WORKS,
    WORKS_ENERGY_ONLY,
    WORKS_GENDER_ONLY,
    cell_works,
    compile_exam_row,
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


STEPS = 300
_CACHE: dict[str, dict] = {}


def leftover() -> dict[str, dict]:
    if "leftover" not in _CACHE:
        _CACHE["leftover"] = {row["name"]: row for row in leaky_cell(steps=STEPS)}
    return _CACHE["leftover"]


def divergent() -> dict[str, dict]:
    if "divergent" not in _CACHE:
        _CACHE["divergent"] = {row["name"]: row for row in divergent_cell(steps=STEPS)}
    return _CACHE["divergent"]


def close() -> dict[str, dict]:
    if "close" not in _CACHE:
        _CACHE["close"] = {row["name"]: row for row in close_cell(steps=STEPS)}
    return _CACHE["close"]


# -- the gate itself -----------------------------------------------------


def test_pair_odd_cos_is_not_an_input_to_the_compiled_score():
    """A perfect lock cannot flip fail → works."""
    assert (
        compiled_verdict(energy_works=False, gender_works=False, pair_odd_cos=1.0)
        == FAILS
    )
    assert (
        compiled_verdict(energy_works=True, gender_works=True, pair_odd_cos=0.1)
        == WORKS
    )
    assert (
        compiled_verdict(energy_works=True, gender_works=False, pair_odd_cos=0.1)
        == WORKS_ENERGY_ONLY
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


def test_unused_gender_leftover_is_not_an_energy_pass():
    """#23's leftover pass cannot promote a two-track fail."""
    assert (
        compiled_verdict(
            energy_works=False,
            gender_works=False,
            leftover_works=True,
            pair_odd_cos=0.5,
        )
        == FAILS
    )


def test_works_requires_small_leak_helpers_still_exist():
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


def test_energy_only_is_the_join_when_close_fails():
    assert (
        compiled_verdict(energy_works=True, gender_works=False, pair_odd_cos=0.99)
        == WORKS_ENERGY_ONLY
    )
    assert compiled_verdict(energy_works=False, gender_works=True) == WORKS_GENDER_ONLY
    assert compiled_verdict(energy_works=None, gender_works=False) == FAILS


def test_sort_is_verdict_then_leak_then_on_sheet():
    rows = sort_rows(
        [
            {"id": "b", "compiled": FAILS, "leftover_leak": 0.0, "on_sheet": 0.9},
            {"id": "a", "compiled": WORKS, "leftover_leak": 0.1, "on_sheet": 0.8},
            {"id": "c", "compiled": WORKS, "leftover_leak": 0.0, "on_sheet": 0.7},
            {"id": "d", "compiled": WORKS, "leftover_leak": 0.0, "on_sheet": 0.95},
            {"id": "e", "compiled": WORKS_GENDER_ONLY, "leftover_leak": 0.2, "on_sheet": 0.9},
            {"id": "f", "compiled": WORKS_ENERGY_ONLY, "leftover_leak": 0.0, "on_sheet": 0.9},
        ]
    )
    assert [r["id"] for r in rows] == ["d", "c", "a", "f", "e", "b"]


# -- the live exam rows --------------------------------------------------


def test_energy_v16_fails_the_compiled_exam_gate():
    energy = divergent()["v16_semantic_kl_sub_e"]
    compiled_row = compile_exam_row(energy)
    assert energy["pass"] is False
    assert compiled_row["compiled"] == FAILS
    assert compiled_row["energy_works"] is False


def test_energy_v18_passes_divergent_and_does_not_outrank_a_singer():
    energy = divergent()["v16_semantic_kl"]
    gender = close()["v16_semantic_kl"]
    compiled_row = compile_exam_row(energy, gender)
    assert energy["pass"] is True
    assert gender["pass"] is False
    assert compiled_row["compiled"] == WORKS_ENERGY_ONLY
    singer = compile_exam_row(divergent()["v6_faithful"], close()["v6_faithful"])
    assert singer["compiled"] == WORKS
    order = sort_rows([compiled_row, singer])
    assert order[0]["id"] == "v6_faithful"


def test_gender_v16_is_flagged_and_does_not_outrank_hidden_faithful():
    gender = close()["v16_semantic_kl"]
    compiled_row = compile_exam_row(divergent()["v16_semantic_kl"], gender)
    assert gender["kl_small_hidden_far"] is True
    assert compiled_row["compiled"] != WORKS
    assert compile_exam_row(divergent()["v6_faithful"], close()["v6_faithful"])[
        "compiled"
    ] == WORKS


def test_faithful_sub_e_no_longer_passes_the_energy_stand_in():
    """#23 locked this as WORKS on unused-gender leftover. Live energy-v16 garbled."""
    unused = leftover()["faithful_sub_e"]
    energy = divergent()["faithful_sub_e"]
    assert unused["pass"] is True
    assert energy["pass"] is False
    assert compile_exam_row(energy)["compiled"] == FAILS


def test_the_three_exam_rows_are_marked():
    rows = live_exam_rows(steps=STEPS)
    assert [r["live"] for r in rows] == [
        "energy-lm-v16",
        "energy-lm-v18",
        "gender-lm-v16",
    ]
    assert all(r["listen_match"] for r in rows)


# -- live default --------------------------------------------------------


def test_the_live_default_is_still_v9_hidden():
    args = parse_args(["--prompts", "x.yaml"])
    assert args.lm_target == "v9"
    assert args.pole_mode == "hidden"
    assert args.common_beta == 0.0
    assert "v9" in LM_RECIPES
