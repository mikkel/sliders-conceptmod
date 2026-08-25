"""Live-exam sheet: divergent two-track + close-pair rollout.

Re-runs the exam cells rather than reading metrics.json. CPU only.
Does not change the live trainer default.
"""

from __future__ import annotations

import pytest

from analysis.slider2d.exam import (
    LIVE_EXAM,
    close_cell,
    close_pair_field,
    divergent_cell,
    divergent_field,
    divergent_teacher_table,
    live_exam_rows,
)
from analysis.slider2d.sheet import GARBLE_MAX, HIDDEN_FAR, KL_SMALL, SHEET_LOCK
from conceptmod.textsliders.train_lm_slider_music3 import LM_RECIPES, parse_args


STEPS = 300
_CACHE: dict[str, list] = {}


def divergent() -> dict[str, dict]:
    if "divergent" not in _CACHE:
        _CACHE["divergent"] = divergent_cell(steps=STEPS)
    return {row["name"]: row for row in _CACHE["divergent"]}


def close() -> dict[str, dict]:
    if "close" not in _CACHE:
        _CACHE["close"] = close_cell(steps=STEPS)
    return {row["name"]: row for row in _CACHE["close"]}


def exam() -> list[dict]:
    if "exam" not in _CACHE:
        _CACHE["exam"] = live_exam_rows(steps=STEPS)
    return _CACHE["exam"]


def test_divergent_e_is_most_of_the_pole_gap():
    field = divergent_field()
    assert field.e_share(0) >= 0.90
    assert -0.15 <= field.probe_cos(0) <= 0.15


def test_faithful_sub_e_on_two_tracks_is_a_third_song():
    teachers = {row["name"]: row for row in divergent_teacher_table()}
    blend = teachers["faithful_sub_e"]
    caption = teachers["caption"]
    assert blend["off_caption"] >= 0.80
    assert "blend" in blend["says"]
    assert "blend" not in caption["says"]
    assert caption["off_caption"] == pytest.approx(0.0, abs=1e-6)


def test_energy_v16_faithful_sub_e_kl_fails_the_divergent_cell():
    row = divergent()["v16_semantic_kl_sub_e"]
    assert row["pass"] is False
    assert "blend" in row["says"]
    assert row["off_caption"] >= 0.80
    assert row["on_sheet_kept"] < SHEET_LOCK


def test_energy_v18_faithful_kl_passes_the_divergent_cell():
    row = divergent()["v16_semantic_kl"]
    assert row["pass"] is True
    assert "punk" in row["says"] and "lullaby" in row["says"]
    assert "blend" not in row["says"]
    assert row["on_sheet_kept"] >= SHEET_LOCK
    assert row["garble"] <= GARBLE_MAX
    # Genre/BPM *are* the poles — unused-gender leak_tok is not a fail.
    assert abs(row["leak_tok"]) <= 1e-6


def test_hidden_faithful_also_passes_divergent():
    assert divergent()["v6_faithful"]["pass"] is True


def test_gender_v16_kl_is_small_hidden_far_and_rollout_garbles():
    row = close()["v16_semantic_kl"]
    assert row["pass"] is False
    assert row["kl_small_hidden_far"] is True
    assert row["kl_pole"] <= KL_SMALL
    assert row["pperc"] >= HIDDEN_FAR
    assert row["on_sheet_kept"] >= SHEET_LOCK
    assert row["rollout_on_sheet_kept"] < SHEET_LOCK or row["rollout_garble"] > GARBLE_MAX
    assert "garble" in row["rollout_says"]


def test_hidden_faithful_sings_the_close_pair():
    """The untrained gender card: hidden MSE onto raw poles."""
    row = close()["v6_faithful"]
    assert row["pass"] is True
    assert row["kl_small_hidden_far"] is False
    assert row["rollout_on_sheet_kept"] >= SHEET_LOCK
    assert "verse_f" in row["rollout_says"] and "verse_m" in row["rollout_says"]
    assert close()["v16_semantic_kl"]["pass"] is False


def test_the_three_live_exam_rows_match_the_listens():
    rows = {row["live"]: row for row in exam()}
    assert set(rows) == {spec["live"] for spec in LIVE_EXAM}
    assert rows["energy-lm-v16"]["predicted"].startswith("FAIL")
    assert rows["energy-lm-v16"]["listen"] == "FAIL"
    assert rows["energy-lm-v18"]["predicted"] == "PASS"
    assert rows["energy-lm-v18"]["listen"] == "PASS"
    assert rows["gender-lm-v16"]["predicted"].startswith("FAIL")
    assert rows["gender-lm-v16"]["listen"] == "FAIL"
    assert all(row["listen_match"] for row in rows.values())


def test_the_live_default_is_still_v9_hidden():
    args = parse_args(["--prompts", "x.yaml"])
    assert args.lm_target == "v9"
    assert args.pole_mode == "hidden"
    assert "v9" in LM_RECIPES
