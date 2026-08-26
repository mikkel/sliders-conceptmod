"""UNI lyric/gender board + the `faithful_plus_neu_lyric` live wiring.

CPU geometry only. Does not change the live trainer default. Does not
fold into `exam_score`, `leak_frac`, or the compiled bipolar board.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from analysis.slider2d.exam import EXAM_ROLL_OVERLAP
from analysis.slider2d.lyric_gender import (
    BASELINES,
    CAPTION_LEN,
    FIXTURE_ONLY,
    GENDER_MOVE_MIN,
    LYRIC_GENDER_RECIPES,
    LYRIC_LEN,
    PICK,
    SpanLoRA,
    build_sequence,
    caption_hidden,
    encode_sequence,
    gender_cell,
    gender_margin,
    gender_move,
    gender_word,
    lyric_gender_rank,
    lyric_gender_table,
    lyric_gender_verdict,
    lyric_hiddens,
    sensitivity,
    sensitivity_verdict,
)
from analysis.slider2d.lyric_recall import LYRIC_RECALL_MIN
from analysis.slider2d.plus_exam import PLUS_COVER_MIN
from analysis.slider2d.plus_neu_exam import PLUS_NEU_HOLD_MIN
from conceptmod.textsliders.slider_targets import (
    lm_faithful_plus_neu,
    lm_faithful_plus_neu_lyric,
    lm_lyric_span_mask,
    lm_masked_hidden_mse,
    lm_plus_neu_loss,
    lm_plus_neu_prefix_loss,
    lm_plus_neu_span_loss,
)
from conceptmod.textsliders.train_lm_slider_music3 import (
    LM_RECIPES,
    LYRIC_HOLD_WEIGHT,
    PLUS_NEU_HOLD_RECIPES,
    PLUS_NEU_LYRIC_RECIPES,
    PLUS_NEU_PREFIX_RECIPES,
    PLUS_NEU_RECIPES,
    lm_train_loss,
    lm_train_targets,
    parse_args,
    resolve_lm_recipe,
)

STEPS = 400
_CACHE: dict[str, object] = {}


def table() -> dict[str, list[dict]]:
    if "table" not in _CACHE:
        _CACHE["table"] = lyric_gender_table(steps=STEPS, seed=0)
    return _CACHE["table"]  # type: ignore[return-value]


def by_name(cell: str) -> dict[str, dict]:
    return {row["name"]: row for row in table()[cell]}


# -- live wiring ---------------------------------------------------------


def test_live_default_is_still_v9_hidden():
    args = parse_args(["--prompts_file", "prompts.yaml"])
    assert args.lm_target == "v9"
    assert args.pole_mode == "hidden"


def test_lyric_target_is_opt_in_and_not_the_default():
    args = parse_args(
        ["--prompts_file", "prompts.yaml", "--lm_target", "faithful_plus_neu_lyric"]
    )
    assert args.lm_target == "faithful_plus_neu_lyric"
    assert args.pole_mode == "hidden"
    assert resolve_lm_recipe(lm_target="faithful_plus_neu_lyric", symmetric=True) == (
        "faithful_plus_neu_lyric"
    )
    assert "faithful_plus_neu_lyric" in LM_RECIPES
    assert parse_args(["--prompts_file", "prompts.yaml"]).lm_target == "v9"


def test_lyric_recipe_is_uni_and_separate_from_prefix_hold():
    assert "faithful_plus_neu_lyric" in PLUS_NEU_RECIPES
    assert PLUS_NEU_LYRIC_RECIPES == frozenset({"faithful_plus_neu_lyric"})
    # #48's prefix recipe is untouched: the two holds are separate sets.
    assert PLUS_NEU_PREFIX_RECIPES == frozenset({"faithful_plus_neu_prefix"})
    assert not PLUS_NEU_LYRIC_RECIPES & PLUS_NEU_PREFIX_RECIPES
    assert PLUS_NEU_HOLD_RECIPES == PLUS_NEU_PREFIX_RECIPES | PLUS_NEU_LYRIC_RECIPES


def test_help_lists_the_lyric_target_and_keeps_v9_hidden_default():
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with pytest.raises(SystemExit):
        with redirect_stdout(buf):
            parse_args(["--help"])
    help_text = buf.getvalue()
    assert "faithful_plus_neu_lyric" in help_text
    assert "lyrics_start" in help_text
    bare = parse_args(["--prompts_file", "prompts.yaml"])
    assert bare.lm_target == "v9"
    assert bare.pole_mode == "hidden"


def test_lyric_teacher_is_still_raw_h_plus():
    pos = torch.tensor([1.0, 2.0, 3.0])
    neg = torch.tensor([-1.0, 0.0, 1.0])
    neu = torch.zeros(3)
    assert torch.allclose(lm_faithful_plus_neu_lyric(pos, neg, neu, None), pos)
    assert torch.allclose(
        lm_faithful_plus_neu_lyric(pos, neg, neu, None),
        lm_faithful_plus_neu(pos, neg, neu, None),
    )
    plus, minus, anc_p, anc_m = lm_train_targets(
        pos, neg, neu, recipe="faithful_plus_neu_lyric"
    )
    assert torch.allclose(plus, pos)
    # Minus is a canary reference only, never a teacher.
    assert torch.allclose(minus, neg)
    assert anc_p is None and anc_m is None


def test_lyric_span_mask_covers_lyrics_not_caption_or_audio_start():
    #            im  cap  word  /cap  lyr  la   lb   /lyr  im_end  audio
    ids = torch.tensor([[0, 1, 7, 2, 3, 8, 9, 4, 5, 6]])
    mask = torch.ones_like(ids)
    span = lm_lyric_span_mask(ids, mask, lyrics_start_id=3, lyrics_end_id=4)
    assert span.tolist() == [[0, 0, 0, 0, 0, 1, 1, 1, 0, 0]]
    # The caption span and <|audio_start|> are outside the hold. That is
    # the whole difference from faithful_plus_neu_prefix.
    assert span[0, 2].item() == 0
    assert span[0, -1].item() == 0


def test_lyric_span_mask_respects_padding_and_fails_closed():
    ids = torch.tensor([[0, 3, 8, 4, 6, 0], [0, 3, 8, 9, 4, 6]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]])
    span = lm_lyric_span_mask(ids, mask, lyrics_start_id=3, lyrics_end_id=4)
    assert span.tolist() == [[0, 0, 1, 1, 0, 0], [0, 0, 1, 1, 1, 0]]
    with pytest.raises(ValueError):
        lm_lyric_span_mask(ids, mask, lyrics_start_id=3, lyrics_end_id=99)
    with pytest.raises(ValueError):
        lm_lyric_span_mask(ids, mask, lyrics_start_id=99, lyrics_end_id=4)


def test_span_loss_is_last_plus_zero_plus_the_masked_span():
    last_p, last_0 = torch.tensor([1.0, 0.0]), torch.zeros(2)
    pred = torch.zeros(1, 3, 2)
    tgt = torch.zeros(1, 3, 2)
    mask = torch.tensor([[0, 1, 0]])
    assert float(
        lm_plus_neu_span_loss(last_p, last_p, last_0, last_0, pred, tgt, mask)
    ) == pytest.approx(0.0, abs=1e-8)
    # Only the masked position is held: moving an unmasked one is free.
    off_span = pred.clone()
    off_span[0, 0] = 5.0
    assert float(
        lm_plus_neu_span_loss(last_p, last_p, last_0, last_0, off_span, tgt, mask)
    ) == pytest.approx(0.0, abs=1e-8)
    in_span = pred.clone()
    in_span[0, 1] = 1.0
    held = lm_plus_neu_span_loss(last_p, last_p, last_0, last_0, in_span, tgt, mask)
    assert float(held) > 0.0
    # Weight scales the span term and nothing else.
    heavy = lm_plus_neu_span_loss(
        last_p, last_p, last_0, last_0, in_span, tgt, mask, span_weight=3.0
    )
    assert float(heavy) == pytest.approx(3.0 * float(held), rel=1e-6)


def test_span_loss_with_a_whole_prefix_mask_is_the_prefix_hold():
    """The two recipes differ in support, not in formula."""
    torch.manual_seed(0)
    last_p, last_0 = torch.randn(4), torch.randn(4)
    tgt_p, tgt_0 = torch.randn(4), torch.randn(4)
    pred = torch.randn(1, 5, 4)
    tgt = torch.randn(1, 5, 4)
    whole = torch.ones(1, 5, dtype=torch.long)
    span = lm_plus_neu_span_loss(last_p, tgt_p, last_0, tgt_0, pred, tgt, whole)
    prefix = lm_plus_neu_prefix_loss(last_p, tgt_p, last_0, tgt_0, pred, tgt)
    assert float(span) == pytest.approx(float(prefix), rel=1e-6)


def test_masked_hidden_mse_is_a_per_element_mean():
    pred = torch.zeros(1, 3, 4)
    tgt = torch.ones(1, 3, 4)
    mask = torch.tensor([[1, 1, 0]])
    assert float(lm_masked_hidden_mse(pred, tgt, mask)) == pytest.approx(1.0)


def test_lm_train_loss_routes_the_lyric_hold_and_refuses_both_holds():
    last_p, last_0 = torch.zeros(3), torch.zeros(3)
    pred = torch.zeros(1, 4, 3)
    tgt = torch.zeros(1, 4, 3)
    mask = torch.tensor([[0, 1, 1, 0]])
    moved = pred.clone()
    moved[0, 1] = 2.0
    loss = lm_train_loss(
        last_p,
        last_p,
        last_p,
        last_p,
        plus_neu=True,
        pred_zero=last_0,
        tgt_zero=last_0,
        plus_neu_lyric=True,
        pred_plus_prefix=moved,
        tgt_neu_prefix=tgt,
        lyric_mask=mask,
    )
    assert float(loss) > 0.0
    with pytest.raises(ValueError):
        lm_train_loss(
            last_p,
            last_p,
            last_p,
            last_p,
            plus_neu=True,
            pred_zero=last_0,
            tgt_zero=last_0,
            plus_neu_lyric=True,
            plus_neu_prefix=True,
            pred_plus_prefix=moved,
            tgt_neu_prefix=tgt,
            lyric_mask=mask,
        )
    with pytest.raises(ValueError):
        lm_train_loss(
            last_p,
            last_p,
            last_p,
            last_p,
            plus_neu=True,
            pred_zero=last_0,
            tgt_zero=last_0,
            plus_neu_lyric=True,
            pred_plus_prefix=moved,
            tgt_neu_prefix=tgt,
        )


def test_trainer_wires_the_lyric_mask_and_the_sidecar_records_it():
    src = Path("conceptmod/textsliders/train_lm_slider_music3.py").read_text()
    assert "plus_neu_lyric=recipe in PLUS_NEU_LYRIC_RECIPES" in src
    assert 'lyric_mask=data.get("neu_lyric_mask")' in src
    assert '"plus_neu_lyric": recipe in PLUS_NEU_LYRIC_RECIPES' in src
    # Both holds need the full hidden sequence, not just the last token.
    assert "if recipe in PLUS_NEU_HOLD_RECIPES:" in src
    assert float(LYRIC_HOLD_WEIGHT) == 1.0


def test_plus_neu_listen_warnings_cover_the_lyric_recipe():
    for path in (
        "conceptmod/textsliders/infer_music3.py",
        "conceptmod/textsliders/generate_listen.py",
    ):
        assert "faithful_plus_neu_lyric" in Path(path).read_text()


# -- the fixture ---------------------------------------------------------


def test_gates_reuse_the_existing_continuation_floor():
    assert LYRIC_RECALL_MIN == pytest.approx(EXAM_ROLL_OVERLAP)
    assert GENDER_MOVE_MIN == pytest.approx(EXAM_ROLL_OVERLAP)


def test_sequence_has_a_caption_span_a_lyric_span_and_audio_start():
    field = gender_cell()
    seq = build_sequence(field, 0)
    assert seq.base.shape[0] == CAPTION_LEN + LYRIC_LEN + 1
    assert int(seq.caption_mask.sum()) == CAPTION_LEN
    assert int(seq.lyric_mask.sum()) == LYRIC_LEN
    assert int(seq.prefix_mask().sum()) == CAPTION_LEN + LYRIC_LEN
    # The two spans do not overlap, and neither touches <|audio_start|>.
    assert int((seq.caption_mask & seq.lyric_mask).sum()) == 0
    assert seq.caption_mask[0, -1].item() == 0
    assert seq.lyric_mask[0, -1].item() == 0
    # The continue token's base hidden is h0 itself, so cover / neu_hold
    # read exactly the vectors the plus+neu exam reads.
    assert torch.allclose(seq.base[-1], field.poles(0)[2])


def test_the_continue_token_has_a_channel_no_prefix_token_has():
    seq = build_sequence(gender_cell(), 0)
    private = seq.acts[:, -1]
    assert float(private[-1]) > 0.0
    assert float(private[:-1].abs().max()) == pytest.approx(0.0)
    # Every position shares the residual-stream channel.
    shared = seq.acts[:, -2]
    assert float(shared.min()) == pytest.approx(float(shared.max()))
    assert float(shared.min()) > 0.0


def test_a_prefix_rewrite_reaches_the_last_hidden():
    field = gender_cell()
    seq = build_sequence(field, 0)
    lora = SpanLoRA.create(field.dim, field.dim + 2)
    with torch.no_grad():
        lora.odd[0, CAPTION_LEN] = 0.0
        lora.odd[:, field.dim] = 0.5
    prefix, last = encode_sequence(seq, lora, 1.0)
    clean_prefix, clean_last = encode_sequence(
        seq, SpanLoRA.create(field.dim, field.dim + 2), 1.0
    )
    assert not torch.allclose(prefix, clean_prefix, atol=1e-6)
    assert not torch.allclose(last, clean_last, atol=1e-6)
    assert lyric_hiddens(seq, prefix).shape[0] == LYRIC_LEN


def test_neutral_vocal_details_is_ungendered_and_plus_ref_is_the_woman():
    field = gender_cell()
    seq = build_sequence(field, 0)
    assert gender_word(field, seq.base[0]) == "—"
    assert gender_margin(field, seq.base[0]) == pytest.approx(0.0, abs=1e-6)
    assert gender_word(field, seq.ref_caption) == "female"
    assert gender_margin(field, seq.ref_caption) > 0.0


def test_gender_move_is_zero_at_neu_and_one_at_the_plus_caption():
    field = gender_cell()
    seq = build_sequence(field, 0)
    base_prefix = seq.base[:-1]
    assert gender_move(field, seq, base_prefix) == pytest.approx(0.0)
    moved = base_prefix.clone()
    moved[:CAPTION_LEN] = seq.ref_caption
    assert gender_move(field, seq, moved) == pytest.approx(1.0)
    assert gender_word(field, caption_hidden(moved)) == "female"


def test_gender_move_is_undefined_where_the_pair_moves_no_readable_attribute():
    from analysis.slider2d.exam import divergent_field

    field = divergent_field()
    seq = build_sequence(field, 0)
    assert gender_move(field, seq, seq.base[:-1]) is None
    assert by_name("divergent")[PICK]["gender_move"] is None


# -- the board -----------------------------------------------------------


def test_uni_baseline_shreds_grit_lyrics_and_still_moves_gender():
    grit = by_name("divergent")[BASELINES[0]]
    close = by_name("close")[BASELINES[0]]
    assert grit["lyric_recall"] < LYRIC_RECALL_MIN
    assert grit["cover"] >= PLUS_COVER_MIN
    assert grit["neu_hold"] >= PLUS_NEU_HOLD_MIN
    # + REF (pos caption, slider off) still sings: the caption is not
    # what breaks the line, the LoRA's prefix rewrite is.
    assert grit["lyric_recall_ref_plus"] >= LYRIC_RECALL_MIN
    assert "punk" in grit["sings_lyric"]
    assert close["lyric_recall"] >= LYRIC_RECALL_MIN
    assert close["gender_move"] >= GENDER_MOVE_MIN
    assert grit["hit"] is False and close["hit"] is True


def test_prefix_hold_baseline_saves_grit_lyrics_and_kills_the_woman():
    grit = by_name("divergent")[BASELINES[1]]
    close = by_name("close")[BASELINES[1]]
    assert grit["lyric_recall"] >= LYRIC_RECALL_MIN
    assert grit["hit"] is True
    assert close["gender_move"] < GENDER_MOVE_MIN
    assert close["reads_vocal_plus"].startswith("—")
    assert close["hit"] is False


def test_a_softer_prefix_clamp_is_the_same_fight_not_a_split():
    soft_grit = by_name("divergent")["faithful_plus_neu_prefix@0.25"]
    soft_close = by_name("close")["faithful_plus_neu_prefix@0.25"]
    assert soft_grit["hold_weight"] == pytest.approx(0.25)
    assert soft_grit["lyric_recall"] >= LYRIC_RECALL_MIN
    assert soft_close["gender_move"] < GENDER_MOVE_MIN


def test_projecting_the_last_delta_off_the_lyric_span_loses_cover():
    grit = by_name("divergent")["faithful_plus_neu_orth"]
    # Lyrics survive, but for the wrong reason: the target is no longer
    # h+, so the + state stops covering the + caption.
    assert grit["lyric_recall"] >= LYRIC_RECALL_MIN
    assert grit["cover"] < PLUS_COVER_MIN
    assert grit["hit"] is False


def test_the_pick_takes_both_cells():
    grit = by_name("divergent")[PICK]
    close = by_name("close")[PICK]
    assert grit["lyric_recall"] >= LYRIC_RECALL_MIN
    assert grit["cover"] >= PLUS_COVER_MIN
    assert grit["neu_hold"] >= PLUS_NEU_HOLD_MIN
    assert grit["hit"] is True
    assert close["lyric_recall"] >= LYRIC_RECALL_MIN
    assert close["gender_move"] >= GENDER_MOVE_MIN
    assert close["reads_vocal_plus"].startswith("female")
    assert close["hit"] is True


def test_the_pick_is_the_only_live_recipe_that_takes_both():
    verdict = lyric_gender_verdict(table())
    assert verdict["baselines_split_as_reported"] is True
    assert verdict["one_recipe_wins"] is True
    assert verdict["live_winners"] == [PICK]
    assert lyric_gender_rank(table())[0]["name"] == PICK


def test_the_concept_prefix_teacher_is_marked_fixture_only():
    assert FIXTURE_ONLY == frozenset({"concept_prefix_teacher"})
    row = by_name("close")["concept_prefix_teacher"]
    assert row["live_flag"] is False
    # It is on the board because it does work here — and it is not the
    # pick because the live caption spans have no position alignment.
    assert row["gender_move"] >= GENDER_MOVE_MIN
    assert "concept_prefix_teacher" not in LM_RECIPES


def test_close_cell_scores_lyrics_and_gender_not_cover():
    """The caption span is supposed to move on a close pair."""
    close = by_name("close")[PICK]
    assert close["hit"] == bool(
        close["lyric_recall"] >= LYRIC_RECALL_MIN
        and close["gender_move"] >= GENDER_MOVE_MIN
    )
    frozen = by_name("close")[BASELINES[1]]
    # Prefix-hold clears cover and neu_hold and still misses, because
    # this cell is not scored on them.
    assert frozen["cover"] >= PLUS_COVER_MIN
    assert frozen["neu_hold"] >= PLUS_NEU_HOLD_MIN
    assert frozen["hit"] is False


def test_recipes_are_the_five_candidates_plus_the_pick():
    assert [c["name"] for c in LYRIC_GENDER_RECIPES] == [
        "faithful_plus_neu",
        "faithful_plus_neu_prefix",
        "faithful_plus_neu_prefix@0.25",
        "faithful_plus_neu_orth",
        "concept_prefix_teacher",
        "faithful_plus_neu_lyric",
    ]
    assert PICK == "faithful_plus_neu_lyric"


def test_board_does_not_rank_on_exam_score_leak_frac_c_plus_or_pperc():
    row = by_name("close")[PICK]
    for banned in ("exam_score", "leak_frac", "c_plus", "cos_pos", "pperc", "pair_odd_cos"):
        assert banned not in row


def test_board_is_not_folded_into_the_compiled_bipolar_board():
    src = Path("analysis/slider2d/lyric_gender.py").read_text()
    assert "from analysis.slider2d.scoreboard" not in src
    board = Path("analysis/slider2d/scoreboard.py").read_text()
    assert "lyric_gender" not in board
    assert "faithful_plus_neu_lyric" not in board
    assert "faithful_plus_neu_lyric" not in Path(
        "analysis/slider2d/plus_neu_exam.py"
    ).read_text()


def test_the_win_survives_every_fixture_knob_that_reproduces_the_failure():
    rows = sensitivity(steps=120, seed=0)
    verdict = sensitivity_verdict(rows)
    assert verdict["settings_reproducing_the_live_split"] > 0
    assert verdict["counterexamples"] == []
    assert verdict["robust"] is True
