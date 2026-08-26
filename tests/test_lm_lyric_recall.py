"""+1 lyric_recall: yaml lyrics on a continuation. Separate scale.

CPU geometry only. Does not change the live trainer default.
Does not fold into exam_score, leak_frac, or the compiled bipolar board.
Does not change faithful_plus_neu or faithful_plus.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from analysis.slider2d.exam import close_field, divergent_field
from analysis.slider2d.plus_exam import PLUS_COVER_MIN, PLUS_OFF_MAX
from analysis.slider2d.plus_neu_exam import PLUS_NEU_HOLD_MIN
from analysis.slider2d.lyric_recall import (
    ATTEND,
    LYRIC_RECALL_MIN,
    LYRIC_RECIPES,
    SequenceResidual,
    encode_sequence,
    last_hidden_cannot_see_transplant,
    lyric_bag,
    lyric_embeds,
    lyric_exam_table,
    lyric_rank,
    lyric_recall,
    lyric_verdict,
    off_lyric,
    ref_plus_sequence,
    sung_line,
)
from conceptmod.textsliders.slider_targets import (
    lm_faithful_plus_neu,
    lm_faithful_plus_neu_prefix,
    lm_plus_neu_loss,
    lm_plus_neu_prefix_loss,
)
from conceptmod.textsliders.train_lm_slider_music3 import parse_args


STEPS = 400
_CACHE: dict[str, dict] = {}


def table(seed: int = 0) -> dict[str, list[dict]]:
    key = f"table{seed}"
    if key not in _CACHE:
        _CACHE[key] = lyric_exam_table(steps=STEPS, seed=seed)
    return _CACHE[key]


def by_name(cell: str, seed: int = 0) -> dict[str, dict]:
    return {row["name"]: row for row in table(seed)[cell]}


def test_live_default_is_still_v9_hidden():
    args = parse_args(["--prompts_file", "prompts.yaml"])
    assert args.lm_target == "v9"
    assert args.pole_mode == "hidden"


def test_prefix_target_is_wired_and_not_the_default():
    args = parse_args(
        ["--prompts_file", "prompts.yaml", "--lm_target", "faithful_plus_neu_prefix"]
    )
    assert args.lm_target == "faithful_plus_neu_prefix"
    assert args.pole_mode == "hidden"
    bare = parse_args(["--prompts_file", "prompts.yaml"])
    assert bare.lm_target == "v9"


def test_help_lists_prefix_target_and_keeps_v9_hidden_default():
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with pytest.raises(SystemExit):
        with redirect_stdout(buf):
            parse_args(["--help"])
    help_text = buf.getvalue()
    assert "faithful_plus_neu_prefix" in help_text
    assert "faithful_plus_neu" in help_text
    assert "v9" in help_text
    bare = parse_args(["--prompts_file", "prompts.yaml"])
    assert bare.lm_target == "v9"
    assert bare.pole_mode == "hidden"


def test_prefix_teacher_is_still_raw_h_plus():
    field = divergent_field()
    pos, neg, neu = field.poles(0)
    raw = lm_faithful_plus_neu(pos, neg, neu, None)
    prefix = lm_faithful_plus_neu_prefix(pos, neg, neu, None)
    assert torch.allclose(raw, pos)
    assert torch.allclose(prefix, pos)


def test_prefix_loss_is_last_plus_zero_plus_prefix():
    last_p = torch.tensor([1.0, 0.0])
    last_0 = torch.tensor([0.0, 0.0])
    pref = torch.zeros(3, 2)
    base = lm_plus_neu_prefix_loss(last_p, last_p, last_0, last_0, pref, pref)
    uni = lm_plus_neu_loss(last_p, last_p, last_0, last_0)
    assert float(base) == pytest.approx(0.0, abs=1e-8)
    assert float(uni) == pytest.approx(0.0, abs=1e-8)
    moved_prefix = lm_plus_neu_prefix_loss(
        last_p, last_p, last_0, last_0, pref + 1.0, pref
    )
    moved_minus_ignored = lm_plus_neu_prefix_loss(
        last_p, last_p, last_0, last_0, pref, pref
    )
    assert float(moved_prefix) > 0.0
    assert float(moved_minus_ignored) == pytest.approx(0.0, abs=1e-8)


def test_lyric_recall_is_yaml_lyrics_not_plus_caption():
    field = divergent_field()
    pref, _last = ref_plus_sequence(field, 0)
    bag = lyric_bag(field, 0)
    head = field.readout()
    assert head.index("lyric0") in bag
    assert head.index("punk") not in bag
    assert head.index("slam") not in bag
    assert lyric_recall(field, pref, 0) == pytest.approx(1.0)
    assert off_lyric(field, pref, 0) == pytest.approx(0.0)
    assert sung_line(field, pref) == [head.index("lyric0")] * len(pref)


def test_scored_columns_are_lyric_recall_cover_neu_hold():
    row = by_name("divergent")["faithful_plus_neu"]
    assert "lyric_recall" in row and "cover" in row and "neu_hold" in row
    assert "lyric_recall_zero" in row and "lyric_recall_ref_plus" in row
    assert "leak_frac" not in row
    assert "exam_score" not in row
    assert "pair_odd_cos" not in row
    assert row["canary"]["scored"] is False


def test_lyric_exam_does_not_import_the_bipolar_board():
    src = Path("analysis/slider2d/lyric_recall.py").read_text()
    assert "from analysis.slider2d.scoreboard" not in src
    board = Path("analysis/slider2d/scoreboard.py").read_text()
    assert "lyric_recall" not in board
    assert "faithful_plus_neu_prefix" not in board
    plus_neu = Path("analysis/slider2d/plus_neu_exam.py").read_text()
    assert "faithful_plus_neu_prefix" not in plus_neu


def test_last_hidden_fixture_cannot_see_prefix_rewrite():
    for ctor in (divergent_field, close_field):
        field = ctor()
        blind = last_hidden_cannot_see_transplant(field)
        # Last-hidden continuation of h+ is the + caption rollout (concept
        # words). It is not the yaml line, so a last-hidden lyric score
        # cannot tell UNI from prefix-hold.
        assert blind["last_hidden_lyric_share"] == pytest.approx(0.0)


def test_faithful_plus_neu_hits_old_box_and_misses_plus1_lyrics():
    uni_div = by_name("divergent")["faithful_plus_neu"]
    uni_close = by_name("close")["faithful_plus_neu"]
    assert uni_div["old_box"] is True
    assert uni_close["old_box"] is True
    assert uni_div["cover"] >= PLUS_COVER_MIN
    assert uni_div["neu_hold"] >= PLUS_NEU_HOLD_MIN
    assert uni_div["off_caption"] <= PLUS_OFF_MAX
    assert uni_div["lyric_recall"] < LYRIC_RECALL_MIN
    assert uni_div["lyric_recall_ref_plus"] >= LYRIC_RECALL_MIN
    assert uni_div["last_token_transplant"] is True
    # Close / gender-like keeps the yaml line under last-token UNI.
    assert uni_close["lyric_recall"] >= LYRIC_RECALL_MIN
    assert "punk" in uni_div["sings_lyric"]
    assert "lyric0" in uni_div["sings_ref_plus"]


def test_prefix_hold_lifts_lyric_recall_and_keeps_cover():
    for cell in ("divergent", "close"):
        uni = by_name(cell)["faithful_plus_neu"]
        prefix = by_name(cell)["faithful_plus_neu_prefix"]
        assert prefix["lyric_recall"] >= LYRIC_RECALL_MIN
        assert prefix["cover"] >= PLUS_COVER_MIN
        assert prefix["neu_hold"] >= PLUS_NEU_HOLD_MIN
        assert prefix["hit"] is True
        assert prefix["lyric_recall"] + 1e-6 >= uni["lyric_recall"]
        assert prefix["cover"] + 1e-6 >= min(uni["cover"], PLUS_COVER_MIN)
        assert "lyric" in prefix["sings_lyric"]


def test_rank_is_lyric_recall_then_cover():
    ranked = lyric_rank(table())
    names = [r["name"] for r in ranked]
    assert names[0] == "faithful_plus_neu_prefix"
    assert ranked[0]["in_box"] is True
    for earlier, later in zip(ranked, ranked[1:]):
        if abs(earlier["lyric_recall"] - later["lyric_recall"]) > 1e-9:
            assert earlier["lyric_recall"] >= later["lyric_recall"]
            continue
        if abs(earlier["cover"] - later["cover"]) > 1e-9:
            assert earlier["cover"] >= later["cover"]


def test_recipes_are_the_five_comparisons():
    names = [c["name"] for c in LYRIC_RECIPES]
    assert names == [
        "faithful_plus_neu",
        "faithful_plus",
        "faithful_plus_neu_prefix",
        "leftover_gate_bipolar",
        "pair_odd_midpoint",
    ]


def test_verdict_replicates_transplant_and_prefix_fixes_it():
    verdict = lyric_verdict(table())
    assert verdict["replicated_last_token_transplant"] is True
    assert verdict["prefix_lifts_lyric_recall"] is True
    assert verdict["prefix_keeps_cover"] is True
    assert verdict["prefix_hits_required"] is True
    assert verdict["uni_lyric_recall"][0] < LYRIC_RECALL_MIN
    assert verdict["uni_ref_plus"][0] >= LYRIC_RECALL_MIN


def test_sequence_last_hidden_is_causal_in_prefix():
    field = divergent_field()
    residual = SequenceResidual.create(field)
    with torch.no_grad():
        residual.p.add_(0.5 * field.plus_track())
    pref, last = encode_sequence(field, residual, 0, 1.0)
    pref_clean, last_clean = encode_sequence(
        field, SequenceResidual.create(field), 0, 1.0
    )
    assert not torch.allclose(last, last_clean, atol=1e-5)
    assert float(ATTEND) > 0.0
    assert pref.shape[0] == lyric_embeds(field, 0).shape[0]
