"""Lyric-token UNI hold: span location and prefix vs lyric MSE.

CPU only. Does not change the live trainer default.
Does not fold into the compiled bipolar exam_score board.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from conceptmod.textsliders.slider_targets import (
    lm_faithful_plus_neu,
    lm_faithful_plus_neu_lyric,
    lm_faithful_plus_neu_prefix,
    lm_plus_neu_loss,
    lm_plus_neu_lyric_loss,
    lm_plus_neu_prefix_loss,
)
from conceptmod.textsliders.train_lm_slider_music3 import (
    _AUDIO_START,
    _LYRICS_END,
    _LYRICS_START,
    _assert_last_token_is_audio_start,
    _assert_lyric_span,
    _lyric_token_mask,
    _split_prefix_last,
    lm_train_loss,
    parse_args,
)


class _Tok:
    ids = {
        _AUDIO_START: 99,
        _LYRICS_START: 4,
        _LYRICS_END: 5,
        "<|im_start|>": 1,
        "<|caption_start|>": 2,
        "<|caption_end|>": 3,
        "<|im_end|>": 6,
    }

    def convert_tokens_to_ids(self, token: str):
        return self.ids.get(token, -1)


# Caption tokens: Vocal Details gender phrase differs between neu / pos.
# Lyrics sit between lyrics_start / lyrics_end.
_NEU_IDS = [1, 2, 10, 11, 12, 13, 14, 3, 4, 20, 21, 5, 6, 99]
_POS_IDS = [1, 2, 10, 11, 12, 15, 13, 14, 16, 3, 4, 20, 21, 5, 6, 99]
_WOMAN_POS = 8  # "woman" in pos caption (Vocal Details)
_LEAD_POS = 5  # "lead" in neu Vocal Details
_LYRIC_NEU = (9, 11)  # feel, air on neu
_LYRIC_POS = (11, 13)  # feel, air on pos


def test_live_default_is_still_v9_hidden():
    args = parse_args(["--prompts_file", "prompts.yaml"])
    assert args.lm_target == "v9"
    assert args.pole_mode == "hidden"


def test_lyric_span_is_yaml_lyrics_not_vocal_details():
    tok = _Tok()
    neu = torch.tensor([_NEU_IDS])
    pos = torch.tensor([_POS_IDS])
    neu_mask = torch.ones_like(neu)
    pos_mask = torch.ones_like(pos)
    _assert_last_token_is_audio_start(neu, neu_mask, tok, where="neu")
    _assert_last_token_is_audio_start(pos, pos_mask, tok, where="pos")
    found = _assert_lyric_span(
        neu, neu_mask, pos, pos_mask, tok, "I can feel it", where="gender row"
    )
    assert found[0].tolist() == [0] * 9 + [1, 1] + [0] * 3
    pos_found = _lyric_token_mask(
        pos, pos_mask, tok, "I can feel it", where="pos lyrics"
    )
    assert pos_found[0, _LYRIC_POS[0] : _LYRIC_POS[1]].tolist() == [1, 1]
    assert int(pos_found[0, _WOMAN_POS]) == 0
    assert int(found[0, _LEAD_POS]) == 0
    last, _hidden, prefix_mask = _split_prefix_last(
        torch.zeros(1, len(_NEU_IDS), 2), neu_mask
    )
    assert int(prefix_mask[0, _LEAD_POS]) == 1
    assert int(prefix_mask[0, -1]) == 0
    assert last.shape[-1] == 2


def test_empty_or_missing_lyric_span_fails_closed():
    tok = _Tok()
    neu = torch.tensor([_NEU_IDS])
    mask = torch.ones_like(neu)
    with pytest.raises(RuntimeError, match="empty lyrics"):
        _lyric_token_mask(neu, mask, tok, "   ", where="empty")
    with pytest.raises(RuntimeError, match="empty lyrics"):
        _lyric_token_mask(neu, mask, tok, "", where="empty")
    no_span = torch.tensor([[1, 2, 10, 11, 3, 6, 99]])
    no_mask = torch.ones_like(no_span)
    with pytest.raises(RuntimeError, match="span not found"):
        _lyric_token_mask(no_span, no_mask, tok, "feel air", where="missing")
    pos = torch.tensor([_POS_IDS])
    pos_mask = torch.ones_like(pos)
    with pytest.raises(RuntimeError, match="differ between neu and pos"):
        _assert_lyric_span(
            neu,
            mask,
            torch.tensor([[1, 2, 10, 3, 4, 30, 31, 5, 6, 99]]),
            torch.ones(1, 10, dtype=torch.long),
            tok,
            "feel air",
            where="mismatch",
        )
    _ = pos_mask


def test_lyric_hold_mse_skips_vocal_details_gender_phrase():
    last_p = torch.tensor([1.0, 0.0])
    last_0 = torch.tensor([0.0, 0.0])
    hidden = torch.zeros(1, 6, 2)
    lyric_mask = torch.tensor([[0, 0, 0, 1, 1, 0]])
    prefix_mask = torch.tensor([[1, 1, 1, 1, 1, 0]])
    moved_vd = hidden.clone()
    moved_vd[:, 2] = 4.0
    moved_ly = hidden.clone()
    moved_ly[:, 3] = 4.0
    lyric_base = lm_train_loss(
        last_p,
        last_p,
        last_p,
        last_p,
        plus_neu=True,
        plus_neu_prefix=True,
        pred_zero=last_0,
        tgt_zero=last_0,
        pred_plus_prefix=hidden,
        tgt_neu_prefix=hidden,
        prefix_mask=lyric_mask,
        pole_mode="hidden",
    )
    lyric_vd = lm_train_loss(
        last_p,
        last_p,
        last_p,
        last_p,
        plus_neu=True,
        plus_neu_prefix=True,
        pred_zero=last_0,
        tgt_zero=last_0,
        pred_plus_prefix=moved_vd,
        tgt_neu_prefix=hidden,
        prefix_mask=lyric_mask,
        pole_mode="hidden",
    )
    lyric_ly = lm_train_loss(
        last_p,
        last_p,
        last_p,
        last_p,
        plus_neu=True,
        plus_neu_prefix=True,
        pred_zero=last_0,
        tgt_zero=last_0,
        pred_plus_prefix=moved_ly,
        tgt_neu_prefix=hidden,
        prefix_mask=lyric_mask,
        pole_mode="hidden",
    )
    assert float(lyric_vd) == pytest.approx(float(lyric_base), abs=1e-8)
    assert float(lyric_ly) > float(lyric_base)
    prefix_base = lm_train_loss(
        last_p,
        last_p,
        last_p,
        last_p,
        plus_neu=True,
        plus_neu_prefix=True,
        pred_zero=last_0,
        tgt_zero=last_0,
        pred_plus_prefix=hidden,
        tgt_neu_prefix=hidden,
        prefix_mask=prefix_mask,
        pole_mode="hidden",
    )
    prefix_vd = lm_train_loss(
        last_p,
        last_p,
        last_p,
        last_p,
        plus_neu=True,
        plus_neu_prefix=True,
        pred_zero=last_0,
        tgt_zero=last_0,
        pred_plus_prefix=moved_vd,
        tgt_neu_prefix=hidden,
        prefix_mask=prefix_mask,
        pole_mode="hidden",
    )
    assert float(prefix_vd) > float(prefix_base)
    sliced = lm_plus_neu_lyric_loss(
        last_p, last_p, last_0, last_0, moved_ly[:, 3:5], hidden[:, 3:5]
    )
    assert float(sliced) > 0.0


def test_prefix_hold_still_holds_the_whole_prefix():
    last_p = torch.tensor([1.0, 0.0])
    last_0 = torch.tensor([0.0, 0.0])
    pref = torch.zeros(4, 2)
    base = lm_plus_neu_prefix_loss(last_p, last_p, last_0, last_0, pref, pref)
    moved_first = pref.clone()
    moved_first[0] = 3.0
    moved_last_prefix = pref.clone()
    moved_last_prefix[-1] = 3.0
    assert float(lm_plus_neu_prefix_loss(last_p, last_p, last_0, last_0, moved_first, pref)) > 0.0
    assert float(
        lm_plus_neu_prefix_loss(last_p, last_p, last_0, last_0, moved_last_prefix, pref)
    ) > 0.0
    assert float(base) == pytest.approx(0.0, abs=1e-8)


def test_last_token_teachers_are_still_raw_h_plus_and_h0():
    pos = torch.tensor([2.0, 0.0])
    neg = torch.tensor([0.0, 2.0])
    neu = torch.tensor([0.0, 0.0])
    assert torch.allclose(lm_faithful_plus_neu(pos, neg, neu, None), pos)
    assert torch.allclose(lm_faithful_plus_neu_prefix(pos, neg, neu, None), pos)
    assert torch.allclose(lm_faithful_plus_neu_lyric(pos, neg, neu, None), pos)
    pred_plus = pos + 0.1
    pred_zero = neu + 0.2
    uni = lm_plus_neu_loss(pred_plus, pos, pred_zero, neu)
    lyric = lm_plus_neu_lyric_loss(
        pred_plus, pos, pred_zero, neu, torch.zeros(2, 2), torch.zeros(2, 2)
    )
    assert float(lyric) == pytest.approx(float(uni), abs=1e-8)


def test_not_folded_into_bipolar_board():
    board = Path("analysis/slider2d/scoreboard.py").read_text()
    assert "faithful_plus_neu_lyric" not in board
    assert "gender_move" not in board
