"""Role-split UNI: lyrics → encode(neu), Vocal Details → encode(pos).

CPU geometry only. Does not change the live trainer default.
Does not fold into exam_score, leak_frac, or the compiled bipolar board.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from analysis.slider2d.exam import close_field, divergent_field
from analysis.slider2d.lyric_recall import LYRIC_RECALL_MIN, lyric_embeds
from analysis.slider2d.plus_exam import PLUS_COVER_MIN
from analysis.slider2d.plus_neu_exam import PLUS_NEU_HOLD_MIN
from analysis.slider2d.roles import (
    GENDER_MOVE_MIN,
    ROLE_RECIPES,
    RoleResidual,
    concept_embeds_neu,
    concept_embeds_pos,
    encode_roles,
    gender_move_score,
    require_role_spans,
    role_exam_table,
    role_rank,
    role_verdict,
)
from conceptmod.textsliders.slider_targets import (
    RoleSpanError,
    lm_faithful_plus_neu,
    lm_faithful_plus_neu_prefix,
    lm_faithful_plus_neu_roles,
    lm_plus_neu_loss,
    lm_plus_neu_prefix_loss,
    lm_plus_neu_roles_loss,
    lm_project_last_delta_off_lyric,
    lm_role_span_bounds,
    lm_span_mse,
)
from conceptmod.textsliders.train_lm_slider_music3 import (
    PLUS_NEU_PREFIX_RECIPES,
    PLUS_NEU_RECIPES,
    PLUS_NEU_ROLES_RECIPES,
    parse_args,
)


STEPS = 400
_CACHE: dict[str, dict] = {}


def table(seed: int = 0) -> dict[str, list[dict]]:
    key = f"table{seed}"
    if key not in _CACHE:
        _CACHE[key] = role_exam_table(steps=STEPS, seed=seed)
    return _CACHE[key]


def by_name(cell: str, seed: int = 0) -> dict[str, dict]:
    return {row["name"]: row for row in table(seed)[cell]}


def test_live_default_is_still_v9_hidden():
    args = parse_args(["--prompts_file", "prompts.yaml"])
    assert args.lm_target == "v9"
    assert args.pole_mode == "hidden"


def test_roles_target_is_wired_and_not_the_default():
    args = parse_args(
        ["--prompts_file", "prompts.yaml", "--lm_target", "faithful_plus_neu_roles"]
    )
    assert args.lm_target == "faithful_plus_neu_roles"
    assert args.pole_mode == "hidden"
    assert "faithful_plus_neu_roles" in PLUS_NEU_RECIPES
    assert PLUS_NEU_ROLES_RECIPES == frozenset({"faithful_plus_neu_roles"})
    assert PLUS_NEU_PREFIX_RECIPES == frozenset({"faithful_plus_neu_prefix"})
    bare = parse_args(["--prompts_file", "prompts.yaml"])
    assert bare.lm_target == "v9"


def test_help_lists_roles_and_keeps_v9_hidden_default():
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with pytest.raises(SystemExit):
        with redirect_stdout(buf):
            parse_args(["--help"])
    help_text = buf.getvalue()
    assert "faithful_plus_neu_roles" in help_text
    assert "faithful_plus_neu_prefix" in help_text
    assert "faithful_plus_neu" in help_text
    assert "v9" in help_text
    bare = parse_args(["--prompts_file", "prompts.yaml"])
    assert bare.lm_target == "v9"
    assert bare.pole_mode == "hidden"


def test_existing_uni_teachers_are_unchanged():
    field = divergent_field()
    pos, neg, neu = field.poles(0)
    raw = lm_faithful_plus_neu(pos, neg, neu, None)
    prefix = lm_faithful_plus_neu_prefix(pos, neg, neu, None)
    roles = lm_faithful_plus_neu_roles(pos, neg, neu, None)
    leftover = torch.ones_like(pos)
    still = lm_faithful_plus_neu_roles(pos, neg, neu, leftover, slider_dir=pos)
    assert torch.allclose(raw, pos)
    assert torch.allclose(prefix, pos)
    assert torch.allclose(roles, pos)
    assert torch.allclose(still, pos)


def test_roles_loss_is_last_plus_zero_plus_lyric_plus_concept():
    last_p = torch.tensor([1.0, 0.0])
    last_0 = torch.tensor([0.0, 0.0])
    lyric = torch.zeros(4, 2)
    concept = torch.zeros(3, 2)
    base = lm_plus_neu_roles_loss(
        last_p, last_p, last_0, last_0, lyric, lyric, concept, concept
    )
    uni = lm_plus_neu_loss(last_p, last_p, last_0, last_0)
    assert float(base) == pytest.approx(0.0, abs=1e-8)
    assert float(uni) == pytest.approx(0.0, abs=1e-8)
    moved_lyric = lm_plus_neu_roles_loss(
        last_p, last_p, last_0, last_0, lyric + 1.0, lyric, concept, concept
    )
    moved_concept = lm_plus_neu_roles_loss(
        last_p, last_p, last_0, last_0, lyric, lyric, concept + 1.0, concept
    )
    moved_minus_ignored = lm_plus_neu_roles_loss(
        last_p, last_p, last_0, last_0, lyric, lyric, concept, concept
    )
    assert float(moved_lyric) > 0.0
    assert float(moved_concept) > 0.0
    assert float(moved_minus_ignored) == pytest.approx(0.0, abs=1e-8)
    prefix = lm_plus_neu_prefix_loss(
        last_p, last_p, last_0, last_0, torch.cat([concept, lyric], 0), torch.cat([concept, lyric], 0)
    )
    assert float(prefix) == pytest.approx(0.0, abs=1e-8)


def test_span_mse_pools_when_lengths_differ():
    neu = torch.zeros(2, 3)
    pos = torch.ones(5, 3)
    assert float(lm_span_mse(neu, neu)) == pytest.approx(0.0, abs=1e-8)
    assert float(lm_span_mse(neu, pos)) == pytest.approx(1.0, abs=1e-6)
    with pytest.raises(RoleSpanError, match="empty"):
        lm_span_mse(torch.zeros(0, 3), pos)


def test_role_spans_prefer_vocal_details_and_fail_closed():
    # ids: cap_start=1 GM=10 vd_head=20,21 vd=30,31 arr=40 arr_body=41 cap_end=2
    #      lyr_start=3 lyrics=50,51 lyr_end=4 audio=99
    ids = [8, 1, 10, 20, 21, 30, 31, 40, 41, 2, 3, 50, 51, 4, 9, 99]
    spans = lm_role_span_bounds(
        ids,
        lyrics_start_id=3,
        lyrics_end_id=4,
        caption_start_id=1,
        caption_end_id=2,
        vocal_details_ids=(20, 21),
        arrangement_ids=(40,),
    )
    assert spans["lyric"] == (11, 13)
    assert spans["concept"] == (5, 7)
    assert spans["source"] == "vocal_details"
    fallback = lm_role_span_bounds(
        ids,
        lyrics_start_id=3,
        lyrics_end_id=4,
        caption_start_id=1,
        caption_end_id=2,
        vocal_details_ids=(77, 78),
    )
    assert fallback["source"] == "caption"
    assert fallback["concept"] == (2, 9)
    with pytest.raises(RoleSpanError, match="lyrics span is empty"):
        lm_role_span_bounds(
            [1, 10, 2, 3, 4, 99],
            lyrics_start_id=3,
            lyrics_end_id=4,
            caption_start_id=1,
            caption_end_id=2,
        )
    with pytest.raises(RoleSpanError, match="lyrics span markers"):
        lm_role_span_bounds(
            [1, 10, 2, 99],
            lyrics_start_id=3,
            lyrics_end_id=4,
            caption_start_id=1,
            caption_end_id=2,
        )
    with pytest.raises(RoleSpanError, match="empty"):
        require_role_spans({"lyric": (0, 0), "concept": (1, 4)})


def test_not_last_delta_off_lyric_and_not_whole_prefix():
    src = Path("conceptmod/textsliders/slider_targets.py").read_text()
    roles = Path("analysis/slider2d/roles.py").read_text()
    assert "lm_plus_neu_roles_loss" in src
    assert "lm_project_last_delta_off_lyric" in src
    assert "lm_project_last_delta_off_lyric" not in roles
    delta = torch.ones(4)
    span = torch.ones(3, 4)
    projected = lm_project_last_delta_off_lyric(delta, span)
    assert projected.shape == delta.shape
    trainer = Path("conceptmod/textsliders/train_lm_slider_music3.py").read_text()
    assert "PLUS_NEU_PREFIX_RECIPES = frozenset({\"faithful_plus_neu_prefix\"})" in trainer
    assert "PLUS_NEU_ROLES_RECIPES = frozenset({\"faithful_plus_neu_roles\"})" in trainer


def test_scored_columns_are_the_required_rank_keys():
    row = by_name("divergent")["faithful_plus_neu_roles"]
    for key in ("lyric_recall", "cover", "neu_hold", "gender_move"):
        assert key in row
    assert "exam_score" not in row
    assert "leak_frac" not in row
    assert "c+" not in row
    assert "p%" not in row


def test_role_exam_does_not_import_the_bipolar_board():
    src = Path("analysis/slider2d/roles.py").read_text()
    assert "from analysis.slider2d.scoreboard" not in src
    board = Path("analysis/slider2d/scoreboard.py").read_text()
    assert "faithful_plus_neu_roles" not in board
    lyric = Path("analysis/slider2d/lyric_recall.py").read_text()
    assert "faithful_plus_neu_roles" not in lyric


def test_uni_shreds_grit_lyrics_and_prefix_kills_gender():
    uni = by_name("divergent")["faithful_plus_neu"]
    prefix = by_name("close")["faithful_plus_neu_prefix"]
    prefix_grit = by_name("divergent")["faithful_plus_neu_prefix"]
    assert uni["lyric_recall"] < LYRIC_RECALL_MIN
    assert "punk" in uni["sings_lyric"]
    assert prefix_grit["lyric_recall"] >= LYRIC_RECALL_MIN
    assert prefix["gender_move"] < GENDER_MOVE_MIN
    assert "lead" in prefix["sings_concept"]
    assert "woman" not in prefix["sings_concept"]


def test_roles_hits_grit_lyric_and_gender_move():
    grit = by_name("divergent")["faithful_plus_neu_roles"]
    close = by_name("close")["faithful_plus_neu_roles"]
    assert grit["lyric_recall"] >= LYRIC_RECALL_MIN
    assert close["gender_move"] >= GENDER_MOVE_MIN
    assert grit["cover"] >= PLUS_COVER_MIN
    assert grit["neu_hold"] >= PLUS_NEU_HOLD_MIN
    assert close["cover"] >= PLUS_COVER_MIN
    assert close["neu_hold"] >= PLUS_NEU_HOLD_MIN
    assert "lyric" in grit["sings_lyric"]
    assert "punk" not in grit["sings_lyric"]
    assert "woman" in close["sings_concept"]


def test_rank_keys_and_want_box():
    ranked = role_rank(table())
    names = [r["name"] for r in ranked]
    assert names[0] == "faithful_plus_neu_roles"
    assert ranked[0]["in_box"] is True
    for earlier, later in zip(ranked, ranked[1:]):
        if abs(earlier["lyric_recall"] - later["lyric_recall"]) > 1e-9:
            assert earlier["lyric_recall"] >= later["lyric_recall"]
            continue
        if abs(earlier["cover"] - later["cover"]) > 1e-9:
            assert earlier["cover"] >= later["cover"]
            continue
        if abs(earlier["neu_hold"] - later["neu_hold"]) > 1e-9:
            assert earlier["neu_hold"] >= later["neu_hold"]
            continue
        if abs(earlier["gender_move"] - later["gender_move"]) > 1e-9:
            assert earlier["gender_move"] >= later["gender_move"]
    verdict = role_verdict(table())
    assert verdict["want_box"] is True
    assert verdict["uni_grit_lyric"] is False
    assert verdict["prefix_gender_move"] is False
    assert verdict["roles_grit_lyric"] is True
    assert verdict["roles_gender_move"] is True


def test_recipes_are_the_three_uni_family_members():
    names = [c["name"] for c in ROLE_RECIPES]
    assert names == [
        "faithful_plus_neu",
        "faithful_plus_neu_prefix",
        "faithful_plus_neu_roles",
    ]


def test_gender_move_is_vocal_details_not_last_token():
    field = close_field()
    neu = concept_embeds_neu(field)
    pos = concept_embeds_pos(field, 0)
    assert gender_move_score(neu, neu, pos) == pytest.approx(0.0, abs=1e-6)
    assert gender_move_score(pos, neu, pos) == pytest.approx(1.0, abs=1e-6)
    residual = RoleResidual.create(field)
    _con, lyr, last = encode_roles(field, residual, 0, 0.0)
    assert lyr.shape[0] == lyric_embeds(field, 0).shape[0]
    assert last.shape == field.poles(0)[2].shape
