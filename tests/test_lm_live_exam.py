"""The CPU sheet/live exam introduced after the 2026-08-25 listens."""

from __future__ import annotations

import pytest

from analysis.slider2d.live_exam import (
    HIDDEN_FAR_MIN,
    ONE_TOKEN_KL_SMALL,
    ROLLOUT_MATCH_MIN,
    close_pair_field,
    divergent_pair_field,
    live_exam_cell,
    score_exam,
)


STEPS = 400
_CACHE: dict[str, dict] | None = None


def rows() -> dict[str, dict]:
    global _CACHE
    if _CACHE is None:
        _CACHE = {row["name"]: row for row in live_exam_cell(steps=STEPS)}
    return _CACHE


def test_divergent_pair_matches_the_live_energy_geometry():
    field = divergent_pair_field()
    pos, neg, neu = field.poles()
    assert field.pair_kind == "divergent"
    assert field.pair_kind != close_pair_field().pair_kind
    # Common and odd are equally sized, putting the logged pair cosine near
    # the energy-v4 band (-0.11 .. +0.14).
    assert field.common().norm() == pytest.approx(field.odd().norm(), rel=1e-6)
    dot_cos = float((pos - neu) @ (neg - neu)) / float(
        (pos - neu).norm() * (neg - neu).norm()
    )
    assert -0.11 <= dot_cos <= 0.14


def test_energy_v16_sub_e_fails_by_removing_intended_track_identity():
    row = rows()["energy_v16_semantic_kl_sub_e"]
    assert row["live_run"] == "energy-lm-v16"
    assert row["expected_listen"] == "FAIL"
    assert row["pair_kind"] == "divergent"
    assert row["off_caption_teacher"] > 0.5
    assert row["rollout_match"] < ROLLOUT_MATCH_MIN
    assert row["rollout_garble"] > 0.0
    assert row["pass"] is False


def test_energy_v18_raw_faithful_poles_pass_the_same_divergent_pair():
    row = rows()["energy_v18_semantic_kl_faithful"]
    assert row["live_run"] == "energy-lm-v18"
    assert row["expected_listen"] == "PASS"
    assert row["pair_kind"] == "divergent"
    assert row["off_caption_teacher"] == pytest.approx(0.0, abs=1e-7)
    assert row["rollout_match"] >= ROLLOUT_MATCH_MIN
    assert row["rollout_garble"] == 0.0
    assert row["pass"] is True


def test_close_pair_exposes_kl_small_hidden_far_rollout_garble():
    row = rows()["gender_v16_semantic_kl_faithful"]
    assert row["live_run"] == "gender-lm-v16"
    assert row["expected_listen"] == "FAIL"
    assert row["pair_kind"] == "close"
    assert row["one_token_kl"] <= ONE_TOKEN_KL_SMALL
    assert row["hidden_far"] >= HIDDEN_FAR_MIN
    assert row["hidden_far_while_kl_small"] is True
    assert row["rollout_match"] < ROLLOUT_MATCH_MIN
    assert row["rollout_garble"] > 0.0
    assert row["pass"] is False


def test_hidden_faithful_is_the_close_pair_control_that_would_have_sung():
    hidden = rows()["gender_hidden_faithful_next"]
    kl = rows()["gender_v16_semantic_kl_faithful"]
    assert hidden["pole_mode"] == "hidden"
    assert hidden["target"] == "faithful"
    assert hidden["hidden_far"] < 0.05
    assert hidden["teacher_forced_kl"] < kl["teacher_forced_kl"]
    assert hidden["rollout_match"] > kl["rollout_match"]
    assert hidden["rollout_garble"] == 0.0
    assert hidden["pass"] is True


def test_pair_odd_cos_and_collapse_do_not_determine_the_exam_verdict():
    """The exam score is continuation-based, not a relabeled lock gate."""
    failed = rows()["gender_v16_semantic_kl_faithful"]
    passed = rows()["gender_hidden_faithful_next"]
    assert -1.0 <= failed["pair_odd_cos"] <= 1.0
    assert -1.0 <= failed["collapse"] <= 1.0
    assert failed["pass"] is False
    assert passed["pass"] is True
    assert failed["rollout_match"] < ROLLOUT_MATCH_MIN <= passed["rollout_match"]


def test_unknown_exam_modes_are_rejected():
    field = close_pair_field()
    with pytest.raises(ValueError):
        score_exam(
            "bad",
            field,
            pole_mode="cosine",
            target="faithful",
            live_run=None,
            expected_listen=None,
            steps=1,
        )
    with pytest.raises(ValueError):
        score_exam(
            "bad",
            field,
            pole_mode="hidden",
            target="midpoint",
            live_run=None,
            expected_listen=None,
            steps=1,
        )

