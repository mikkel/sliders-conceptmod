"""Krea image UNI slider: CPU geometry + dummy trainer.

No Hub, no GPU, no Krea weights. Does not change Music 3 defaults.
Does not add Anima / ZiT / H3.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from conceptmod.textsliders.slider_targets import (
    KREA_DEFAULT_RANK,
    KREA_DEFAULT_RESOLUTION,
    KREA_RAW_CFG,
    KREA_RAW_MODEL,
    KREA_RAW_STEPS,
    KREA_TURBO_CFG,
    KREA_TURBO_STEPS,
    expand_attributes_krea,
    krea_cfg_compose,
    krea_cfg_direction,
    krea_concept_words,
    krea_hold_unused_embeds,
    krea_looks_turbo,
    krea_minus_canary,
    krea_plus_neu_loss,
    krea_plus_neu_teachers,
    krea_sample_card,
    krea_unused_hold_loss,
    krea_unused_hold_mask,
    krea_word_tokens,
)
from conceptmod.textsliders.train_lora_krea import (
    DummyKreaBackend,
    assert_krea_only,
    krea_step_loss,
    load_prompts,
    parse_args,
    train,
    unused_words_for,
)
from conceptmod.textsliders.train_lm_slider_music3 import parse_args as parse_lm


ROOT = Path(__file__).resolve().parents[1]
KREA_YAML = ROOT / "conceptmod/textsliders/data/prompts-krea.yaml"
KREA_TRAINER = ROOT / "conceptmod/textsliders/train_lora_krea.py"


def test_music3_lm_default_is_still_v9_hidden():
    args = parse_lm(["--prompts_file", "prompts.yaml"])
    assert args.lm_target == "v9"
    assert args.pole_mode == "hidden"


def test_music3_tf_defaults_are_unchanged():
    src = (ROOT / "conceptmod/textsliders/train_lora_music3.py").read_text(
        encoding="utf-8"
    )
    assert 'default=8)' in src or 'default=8,' in src
    assert 'default="nmse"' in src
    assert 'default="full"' in src
    assert '"--dummy"' in src


def test_krea_trainer_is_opt_in_not_the_music3_default():
    bare = parse_args([])
    assert bare.recipe == "uni"
    assert bare.rank == KREA_DEFAULT_RANK
    assert bare.resolution == KREA_DEFAULT_RESOLUTION
    assert bare.model_id == KREA_RAW_MODEL
    assert bare.dummy is False
    src = KREA_TRAINER.read_text(encoding="utf-8")
    assert "lyric-hold" in src
    assert "not music 3 lyric-hold" in src.lower()
    assert "_FOREIGN_BACKENDS" in src
    assert "from conceptmod.backends.anima" not in src
    assert "from conceptmod.backends.zimage" not in src
    assert 'load_backend("anima"' not in src
    assert 'load_backend("h3"' not in src


def test_help_lists_uni_and_raw_card():
    with pytest.raises(SystemExit):
        parse_args(["--help"])
    src = KREA_TRAINER.read_text(encoding="utf-8")
    assert "krea/Krea-2-Raw" in src
    assert "28" in src
    assert "4.5" in src
    assert "turbo" in src.lower()


def test_sample_card_raw_vs_turbo():
    raw = krea_sample_card("krea/Krea-2-Raw")
    assert raw["variant"] == "raw"
    assert raw["sample_steps"] == KREA_RAW_STEPS == 28
    assert raw["sample_guidance"] == KREA_RAW_CFG == 4.5
    turbo = krea_sample_card("/comfy/Krea-2-Turbo.safetensors")
    assert turbo["variant"] == "turbo"
    assert turbo["sample_steps"] == KREA_TURBO_STEPS == 8
    assert turbo["sample_guidance"] == KREA_TURBO_CFG == 0.0
    assert krea_looks_turbo("krea/Krea-2-Raw") is False
    assert krea_looks_turbo("local-turbo.safetensors") is True


def test_cfg_direction_is_cond_minus_empty():
    v_c = torch.tensor([3.0, 1.0])
    v_u = torch.tensor([1.0, 0.5])
    assert torch.allclose(krea_cfg_direction(v_c, v_u), torch.tensor([2.0, 0.5]))


def test_cfg_compose_matches_krea_convention():
    v_c = torch.tensor([2.0, 0.0])
    v_u = torch.tensor([0.0, 1.0])
    raw = krea_cfg_compose(v_c, v_u, 4.5)
    assert torch.allclose(raw, v_c + 4.5 * (v_c - v_u))
    turbo = krea_cfg_compose(v_c, v_u, 0.0)
    assert torch.allclose(turbo, v_c)


def test_uni_teachers_are_pos_and_neu_no_minus():
    v_pos = torch.tensor([1.0, 0.0])
    v_neu = torch.tensor([0.0, 0.0])
    v_uncond = torch.tensor([-1.0, 0.0])
    v_neg = torch.tensor([0.0, 1.0])
    plus, zero = krea_plus_neu_teachers(v_pos, v_neu, v_uncond, guidance=0.0)
    assert torch.allclose(plus, v_pos)
    assert torch.allclose(zero, v_neu)
    plus_g, zero_g = krea_plus_neu_teachers(v_pos, v_neu, v_uncond, guidance=4.5)
    assert torch.allclose(plus_g, krea_cfg_compose(v_pos, v_uncond, 4.5))
    assert torch.allclose(zero_g, v_neu)
    canary = krea_minus_canary(v_neg, v_uncond)
    assert torch.allclose(canary, v_neg - v_uncond)
    assert not torch.allclose(plus, v_neg)
    assert not torch.allclose(zero, v_neg)


def test_plus_neu_loss_is_mse_plus_and_mse_zero_only():
    pred_plus = torch.tensor([1.0, 0.0])
    tgt_plus = torch.tensor([1.0, 0.0])
    pred_zero = torch.tensor([0.0, 0.0])
    tgt_zero = torch.tensor([0.0, 0.0])
    assert float(krea_plus_neu_loss(pred_plus, tgt_plus, pred_zero, tgt_zero)) == pytest.approx(
        0.0, abs=1e-8
    )
    miss_minus = krea_plus_neu_loss(pred_plus, tgt_plus, pred_zero, tgt_zero)
    extra = F.mse_loss(torch.tensor([0.0, 1.0]), torch.tensor([0.0, -1.0]))
    assert float(miss_minus) < float(extra)


def test_concept_words_are_not_held():
    pos = "a photo of an old male person"
    neu = "a photo of a male person"
    concept = krea_concept_words(pos, neu)
    assert "old" in concept
    assert "an" in concept
    assert "male" not in concept
    assert "person" not in concept
    pos_toks = krea_word_tokens(pos)
    neu_toks = krea_word_tokens(neu)
    mask = krea_unused_hold_mask(pos_toks, neu_toks, unused_words=["male", "female"])
    by_tok = dict(zip(pos_toks, mask))
    assert by_tok["old"] is False
    assert by_tok["an"] is False
    assert by_tok["male"] is True
    assert by_tok["person"] is True
    assert by_tok["photo"] is True


def test_unused_hold_copies_neu_and_leaves_concept():
    pos_tokens = ["male", "old", "person"]
    neu_tokens = ["male", "person"]
    pos = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.5]])
    neu = torch.tensor([[9.0, 9.0], [8.0, 8.0]])
    mask = krea_unused_hold_mask(pos_tokens, neu_tokens, unused_words=["male"])
    held = krea_hold_unused_embeds(pos, neu, pos_tokens, neu_tokens, mask)
    assert torch.allclose(held[0], neu[0])  # male → encode(neu)
    assert torch.allclose(held[1], pos[1])  # old stays
    assert torch.allclose(held[2], neu[1])  # person → encode(neu)
    loss = krea_unused_hold_loss(pos, neu, pos_tokens, neu_tokens, mask)
    want = F.mse_loss(pos[[0, 2]], torch.stack([neu[0], neu[1]]))
    assert float(loss) == pytest.approx(float(want), abs=1e-6)


def test_attributes_pin_unused_on_pos_and_neu():
    row = {
        "target": "a person",
        "positive": "an old person",
        "neutral": "a person",
        "negative": "a young person",
        "attributes": ["male", "female"],
    }
    rows = expand_attributes_krea(row)
    assert len(rows) == 2
    assert rows[0]["positive"].startswith("male ")
    assert rows[0]["neutral"].startswith("male ")
    assert rows[1]["positive"].startswith("female ")
    assert "old" in rows[0]["positive"]
    assert "old" not in rows[0]["neutral"]


def test_yaml_is_pos_neu_with_unused_attrs_pinned():
    prompts, meta = load_prompts(KREA_YAML)
    assert meta.plus_label == "Old"
    assert len(prompts) == 4  # 2 rows × 2 attributes
    for prompt in prompts:
        assert prompt.positive
        assert prompt.neutral
        unused = unused_words_for(prompt)
        assert "male" in unused or "female" in unused
        assert any(w in prompt.positive.split() for w in unused)
        assert any(w in prompt.neutral.split() for w in unused)
        concept = krea_concept_words(prompt.positive, prompt.neutral)
        assert "old" in concept


def test_refuses_foreign_backends():
    with pytest.raises(ValueError, match="Krea-only"):
        assert_krea_only("anima/foo")
    with pytest.raises(ValueError, match="Krea-only"):
        assert_krea_only("local-zit.safetensors")
    with pytest.raises(ValueError, match="Krea-only"):
        assert_krea_only("h3-turbo.safetensors")
    assert_krea_only("krea/Krea-2-Raw")
    assert_krea_only("/comfy/Krea-2-Turbo.safetensors")


def test_dummy_step_has_no_minus_teacher(tmp_path: Path):
    backend = DummyKreaBackend(dim=8, rank=2, seed=0)
    from conceptmod.textsliders.train_lora_krea import KreaSliderPrompt

    prompt = KreaSliderPrompt(
        target="a person",
        positive="an old person",
        neutral="a person",
        negative="a young person",
        attributes=["male"],
    )
    z = torch.zeros(1, 8)
    loss, stats = krea_step_loss(backend, prompt, z, guidance=4.5)
    assert stats["minus_teacher"] == 0.0
    assert "canary_minus_norm" in stats
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in backend.trainable_parameters() if p.grad is not None]
    assert grads


def test_dummy_train_writes_sidecar_without_hub(tmp_path: Path):
    args = parse_args(
        [
            "--dummy",
            "--name",
            "krea-age-dummy",
            "--prompts_file",
            str(KREA_YAML),
            "--save_dir",
            str(tmp_path),
            "--steps",
            "8",
            "--seed",
            "7",
        ]
    )
    assert args.dummy is True
    sidecar = train(args)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["kind"] == "krea"
    assert payload["recipe"] == "uni"
    assert payload["dummy"] is True
    assert payload["minus_teacher"] is False
    assert payload["minus_canary"] is True
    assert payload["token_hold"] == "unused_to_neu"
    assert payload["lyric_hold"] is False
    assert payload["rank"] == 16
    assert payload["resolution"] == 512
    assert payload["model_id"] == KREA_RAW_MODEL
    assert payload["variant"] == "raw"
    assert payload["sample_steps"] == 28
    assert payload["sample_guidance"] == 4.5
    assert payload["official"] == "train LoRAs on Raw, run on Turbo"
    log = tmp_path / "krea-age-dummy_train.jsonl"
    assert log.exists()
    lines = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2


def test_live_loader_does_not_run_in_dummy():
    src = KREA_TRAINER.read_text(encoding="utf-8")
    assert "Krea2Pipeline.from_pretrained" not in src
    assert "hf_hub_download" not in src
    assert "_load_live_backend" in src
    assert "CI uses --dummy" in src
