"""UNI 2D expect leaderboard: one sortable number, both pair types.

CPU only. Does not change the live trainer default.
Does not fold into the compiled bipolar exam_score board.
Does not re-implement lyric / roles / orth flags.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from analysis.slider2d.expect import (
    EXPECT_GATE,
    MISS_WEIGHT,
    TIE_WEIGHT,
    expect_rank,
    expect_score,
    expect_table,
    expect_values_are_unique,
    expect_verdict,
)
from analysis.slider2d.lyric_gender import (
    EXPECT_RECIPES,
    REQUIRED_RECIPES,
    build_sequence,
    gender_cell,
    gender_move,
    lyric_hiddens,
)
from analysis.slider2d.plus_exam import PLUS_COVER_MIN
from analysis.slider2d.plus_neu_exam import PLUS_NEU_HOLD_MIN
from conceptmod.textsliders.slider_targets import (
    lm_plus_neu_loss,
    lm_plus_neu_lyric_loss,
    lm_plus_neu_orth_loss,
    lm_plus_neu_prefix_loss,
    lm_plus_neu_roles_loss,
)
from conceptmod.textsliders.train_lm_slider_music3 import parse_args


STEPS = 400
_CACHE: dict[str, dict] = {}


def table(seed: int = 0) -> dict[str, list[dict]]:
    key = f"table{seed}"
    if key not in _CACHE:
        _CACHE[key] = expect_table(steps=STEPS, seed=seed)
    return _CACHE[key]


def ranked(seed: int = 0) -> list[dict]:
    return expect_rank(table(seed))


def by_name(seed: int = 0) -> dict[str, dict]:
    return {row["name"]: row for row in ranked(seed)}


def test_live_default_is_still_v9_hidden():
    args = parse_args(["--prompts_file", "prompts.yaml"])
    assert args.lm_target == "v9"
    assert args.pole_mode == "hidden"


def test_required_live_recipes_are_on_the_board():
    names = [c["name"] for c in EXPECT_RECIPES]
    for required in REQUIRED_RECIPES:
        assert required in names
    assert "faithful_plus_neu_prefix@0.25" in names


def test_expect_uses_live_losses_not_new_flags():
    src = Path("analysis/slider2d/lyric_gender.py").read_text()
    assert "lm_plus_neu_loss" in src
    assert "lm_plus_neu_prefix_loss" in src
    assert "lm_plus_neu_lyric_loss" in src
    assert "lm_plus_neu_roles_loss" in src
    assert "lm_plus_neu_orth_loss" in src
    assert "lm_plus_neu_span_loss" not in src
    trainer = Path("conceptmod/textsliders/train_lm_slider_music3.py").read_text()
    assert 'default="v9"' in trainer or "default='v9'" in trainer
    last = torch.zeros(4)
    neu = torch.zeros(4)
    lyric = torch.zeros(3, 4)
    prefix = torch.zeros(5, 4)
    concept = torch.zeros(2, 4)
    assert float(lm_plus_neu_loss(last, last, neu, neu)) == pytest.approx(0.0)
    assert float(lm_plus_neu_prefix_loss(last, last, neu, neu, prefix, prefix, prefix_weight=0.25)) == pytest.approx(0.0)
    assert float(lm_plus_neu_lyric_loss(last, last, neu, neu, lyric, lyric)) == pytest.approx(0.0)
    assert float(lm_plus_neu_roles_loss(last, last, neu, neu, lyric, lyric, concept, concept)) == pytest.approx(0.0)
    assert float(lm_plus_neu_orth_loss(last, last, neu, neu, lyric, lyric, fail_closed=True)) == pytest.approx(0.0)


def test_expect_formula_ranks_cover_collapse_below_kept_cover():
    kept = expect_score(
        grit_lyric_recall=1.0,
        gender_move=1.0,
        grit_cover=0.980,
        gender_cover=0.981,
        neu_hold=1.0,
        grit_off_caption=0.0,
        gender_off_caption=0.0,
    )
    collapsed = expect_score(
        grit_lyric_recall=1.0,
        gender_move=0.872,
        grit_cover=0.483,
        gender_cover=0.870,
        neu_hold=1.0,
        grit_off_caption=0.0,
        gender_off_caption=0.0,
    )
    assert kept["bottleneck"] == pytest.approx(0.980)
    assert collapsed["bottleneck"] == pytest.approx(0.483)
    assert kept["expect"] > collapsed["expect"]
    assert collapsed["cover_gate"] is False


def test_expect_penalizes_missing_lyric_or_gender_gates():
    hit = expect_score(
        grit_lyric_recall=1.0,
        gender_move=1.0,
        grit_cover=0.98,
        gender_cover=0.98,
        neu_hold=1.0,
    )
    shred = expect_score(
        grit_lyric_recall=0.333,
        gender_move=0.872,
        grit_cover=1.0,
        gender_cover=1.0,
        neu_hold=1.0,
    )
    pinned = expect_score(
        grit_lyric_recall=1.0,
        gender_move=0.0,
        grit_cover=0.986,
        gender_cover=0.986,
        neu_hold=1.0,
    )
    assert shred["expect"] < hit["expect"]
    assert pinned["expect"] < hit["expect"]
    assert shred["miss"] == pytest.approx(MISS_WEIGHT * (EXPECT_GATE - 0.333))
    assert pinned["miss"] == pytest.approx(MISS_WEIGHT * EXPECT_GATE)
    assert shred["lyric_gate"] is False
    assert pinned["gender_gate"] is False


def test_expect_tie_break_splits_two_hits():
    a = expect_score(
        grit_lyric_recall=1.0,
        gender_move=1.0,
        grit_cover=0.990,
        gender_cover=0.990,
        neu_hold=1.0,
        grit_off_caption=0.00,
        gender_off_caption=0.00,
    )
    b = expect_score(
        grit_lyric_recall=1.0,
        gender_move=1.0,
        grit_cover=0.960,
        gender_cover=0.960,
        neu_hold=1.0,
        grit_off_caption=0.04,
        gender_off_caption=0.04,
    )
    assert a["bottleneck"] == pytest.approx(b["bottleneck"])
    assert a["expect"] != b["expect"]
    assert a["expect"] > b["expect"]
    assert abs(a["expect"] - b["expect"]) < 0.05
    assert TIE_WEIGHT < MISS_WEIGHT


def test_gender_move_is_readable_on_close_and_absent_on_grit():
    close = gender_cell(seed=0)
    seq = build_sequence(close, 0)
    assert gender_move(close, seq, seq.base[:-1]) == pytest.approx(0.0, abs=1e-6)
    moved = seq.base[:-1].clone()
    moved[:2] = seq.ref_caption
    assert gender_move(close, seq, moved) == pytest.approx(1.0, abs=1e-5)
    from analysis.slider2d.exam import divergent_field

    grit = divergent_field(seed=0)
    grit_seq = build_sequence(grit, 0)
    assert gender_move(grit, grit_seq, grit_seq.base[:-1]) is None
    assert lyric_hiddens(grit_seq.base[:-1]).shape[0] == 4


def test_not_folded_into_bipolar_board():
    board = Path("analysis/slider2d/scoreboard.py").read_text()
    expect_src = Path("analysis/slider2d/expect.py").read_text()
    cell = Path("analysis/slider2d/lyric_gender.py").read_text()
    runner = Path("analysis/slider2d/run_lm_expect.py").read_text()
    assert "from analysis.slider2d.scoreboard" not in expect_src
    assert "from analysis.slider2d.scoreboard" not in cell
    assert "from analysis.slider2d.scoreboard" not in runner
    assert "faithful_plus_neu_lyric" not in board
    assert "faithful_plus_neu_roles" not in board
    assert "faithful_plus_neu_orth" not in board
    assert "expect" not in board
    assert "not_the_bipolar_board" in runner
    assert "not_scored" in expect_src
    assert '["exam_score", "leak_frac", "c+", "p%"]' in runner or '"exam_score"' in runner


def test_scored_board_has_distinct_expect_and_required_recipes():
    rows = ranked()
    names = [r["name"] for r in rows]
    for required in REQUIRED_RECIPES:
        assert required in names
    assert expect_values_are_unique(rows)
    values = [r["expect"] for r in rows]
    assert values == sorted(values, reverse=True)
    for row in rows:
        assert "exam_score" not in row
        assert "leak_frac" not in row
        assert "c+" not in row
        assert "p%" not in row
        for key in (
            "expect",
            "grit_lyric_recall",
            "gender_move",
            "cover",
            "neu_hold",
        ):
            assert key in row


def test_baselines_split_and_winner_takes_both_gates():
    rows = by_name()
    uni = rows["faithful_plus_neu"]
    prefix = rows["faithful_plus_neu_prefix"]
    assert uni["grit_lyric_recall"] < EXPECT_GATE
    assert prefix["gender_move"] < EXPECT_GATE
    winner = ranked()[0]
    assert winner["lyric_gate"] is True
    assert winner["gender_gate"] is True
    assert winner["cover_gate"] is True
    assert winner["grit_lyric_recall"] >= EXPECT_GATE
    assert winner["gender_move"] >= EXPECT_GATE
    assert winner["cover"] >= PLUS_COVER_MIN
    assert winner["neu_hold"] >= PLUS_NEU_HOLD_MIN


def test_cover_collapse_cannot_outrank_kept_cover_on_the_live_table():
    rows = by_name()
    covers = {name: rows[name]["cover"] for name in REQUIRED_RECIPES}
    expects = {name: rows[name]["expect"] for name in REQUIRED_RECIPES}
    low_cover = min(covers, key=covers.get)
    high_cover = max(covers, key=covers.get)
    if covers[low_cover] + 1e-6 < PLUS_COVER_MIN <= covers[high_cover]:
        assert expects[low_cover] < expects[high_cover]


def test_published_page_leads_with_the_sorted_list():
    page = Path("docs/lm-uni-expect.md")
    if not page.exists():
        pytest.skip("page is generated by run_lm_expect.py")
    text = page.read_text()
    assert text.lstrip().startswith("# UNI 2D expect leaderboard")
    assert "## Sorted expect" in text
    assert "exam_score" in text
    assert "not" in text.lower()
    first_list = text.split("## Table", 1)[0]
    for required in REQUIRED_RECIPES:
        assert f"`{required}`" in first_list
    verdict = expect_verdict(table())
    assert f"`{verdict['winner']}`" in first_list
    assert "lm-2d-scoreboard.md" in text
