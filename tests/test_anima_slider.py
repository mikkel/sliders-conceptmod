"""Anima image-slider: UNI + unused-token hold. CPU only, no Hub, no GPU."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from conceptmod.textsliders.anima_fake import (
    LORA_TARGETS as FAKE_LORA_TARGETS,
    FakeAnimaBackend,
    write_plus_alignment,
)
from conceptmod.textsliders.anima_slider import (
    DEFAULT_CFG,
    DEFAULT_MODEL_ID,
    DEFAULT_RANK,
    DEFAULT_RESOLUTION,
    DEFAULT_SAMPLE_STEPS,
    LORA_TARGETS,
    align_unused_positions,
    anima_cfg_delta,
    anima_uni_loss,
    anima_uni_teachers,
    anima_unused_hold_loss,
    concept_tokens,
    expand_attributes_anima,
    live_train_card,
    live_train_command,
    load_anima_prompts,
    minus_canary_cosine,
    row_token_plan,
    splice_unused_embeds,
    unused_token_mask,
    unused_vocab,
    word_tokens,
)
from conceptmod.textsliders.train_lora_anima import (
    load_live_backend,
    parse_args,
    train_dummy,
)

REPO = Path(__file__).resolve().parents[1]
PROMPTS = REPO / "conceptmod/textsliders/data/prompts-anima.yaml"
CANARY = REPO / "conceptmod/textsliders/data/prompts-anima-canary.yaml"


def test_music3_defaults_stay_v9_hidden():
    """Anima is opt-in. Music 3 trainer default is unchanged."""
    from conceptmod.textsliders.train_lm_slider_music3 import parse_args as lm_parse

    args = lm_parse(["--prompts_file", "prompts.yaml"])
    assert args.lm_target == "v9"
    assert args.pole_mode == "hidden"


def test_anima_cli_defaults_match_live_card():
    args = parse_args([])
    assert args.model_id == DEFAULT_MODEL_ID
    assert args.rank == DEFAULT_RANK == 16
    assert args.resolution == DEFAULT_RESOLUTION == 768
    assert args.sample_steps == DEFAULT_SAMPLE_STEPS == 40
    assert args.cfg == DEFAULT_CFG == 4.0
    assert args.dummy is False
    assert args.allow_hub is False
    card = live_train_card()
    assert card["model_id"] == DEFAULT_MODEL_ID
    assert card["lora"]["rank"] == 16
    assert card["lora"]["targets"] == list(LORA_TARGETS)
    assert card["lora"]["train_text_conditioner"] is False
    assert card["resolution"] == 768
    assert card["sample_steps"] == 40
    assert card["cfg"] == 4.0
    assert card["device"] == "cuda:0"
    assert card["music3_default_untouched"] == {
        "lm_target": "v9",
        "pole_mode": "hidden",
    }
    cmd = live_train_command()
    assert "circlestone-labs/Anima-Base-v1.0-Diffusers" in cmd
    assert "--rank 16" in cmd
    assert "--resolution 768" in cmd
    assert "--sample_steps 40" in cmd
    assert "--cfg 4" in cmd
    assert "HF_HUB_OFFLINE=1" in cmd


def test_cfg_delta_is_v_cond_minus_v_empty():
    v_c = torch.tensor([1.5, 0.25])
    v_u = torch.tensor([0.5, 0.05])
    assert torch.allclose(anima_cfg_delta(v_c, v_u), v_c - v_u)


def test_uni_loss_is_plus_and_zero_only():
    plus = torch.tensor([1.0, 0.0])
    zero = torch.tensor([0.0, 1.0])
    base = anima_uni_loss(plus, plus, zero, zero)
    assert float(base) == pytest.approx(0.0, abs=1e-8)
    moved_minus = anima_uni_loss(
        plus,
        plus,
        zero,
        zero,
        student_minus=torch.tensor([9.0, 9.0]),
        teacher_minus=torch.tensor([-9.0, -9.0]),
    )
    assert float(moved_minus) == pytest.approx(float(base), abs=1e-8)
    moved_plus = anima_uni_loss(plus + 1, plus, zero, zero)
    assert float(moved_plus) > float(base)
    teachers = anima_uni_teachers(plus, zero, torch.zeros_like(plus), v_neg=-plus)
    assert teachers["minus"] is None
    assert torch.allclose(teachers["plus"], plus)
    assert torch.allclose(teachers["zero"], zero)


def test_unused_mask_holds_subject_pins_not_concept():
    unused = unused_vocab(
        "a woman sitting on a chair",
        "a woman sitting on a chair",
        attributes=["indoor", "portrait"],
    )
    pos = word_tokens("a smiling woman sitting on a chair")
    assert "smiling" in concept_tokens("a smiling woman sitting on a chair", unused)
    assert "smiling" not in unused
    mask = unused_token_mask(pos, unused)
    assert pos[mask.index(False)] == "smiling"
    assert mask.count(False) == 1
    for tok, held in zip(pos, mask):
        if tok == "smiling":
            assert held is False
        else:
            assert held is True
    pairs = align_unused_positions(
        pos, word_tokens("a woman sitting on a chair"), unused
    )
    held_tokens = [pos[i] for i, _ in pairs]
    assert "smiling" not in held_tokens
    assert "woman" in held_tokens
    assert "chair" in held_tokens


def test_token_hold_mse_skips_concept_words():
    pos = torch.zeros(1, 6, 2)
    neu = torch.zeros(1, 5, 2)
    pos[0, 1] = 4.0  # smiling
    pos[0, 2] = 1.0  # woman
    neu[0, 1] = 0.0  # woman at neu
    pairs = [(0, 0), (2, 1), (3, 2), (4, 3), (5, 4)]  # skip concept index 1
    base = anima_unused_hold_loss(pos, neu, pairs=pairs)
    moved_concept = pos.clone()
    moved_concept[0, 1] = 99.0
    assert float(anima_unused_hold_loss(moved_concept, neu, pairs=pairs)) == pytest.approx(
        float(base), abs=1e-8
    )
    moved_unused = pos.clone()
    moved_unused[0, 2] = 8.0
    assert float(anima_unused_hold_loss(moved_unused, neu, pairs=pairs)) > float(base)


def test_splice_keeps_concept_copies_unused():
    pos = torch.tensor([[[1.0, 0.0], [0.0, 9.0], [3.0, 0.0]]])
    neu = torch.tensor([[[1.0, 0.0], [2.0, 0.0]]])
    held = splice_unused_embeds(pos, neu, pairs=[(0, 0), (2, 1)])
    assert torch.allclose(held[0, 1], pos[0, 1])  # concept word untouched
    assert torch.allclose(held[0, 2], neu[0, 1])


def test_yaml_expands_attributes_and_omits_minus():
    rows, meta = load_anima_prompts(PROMPTS)
    assert meta.plus_label == "smiling"
    assert all(not row.has_minus_canary for row in rows)
    assert {row.attributes[0] for row in rows} == {"indoor", "portrait"}
    assert all(row.positive.startswith(row.attributes[0]) for row in rows)
    plan = row_token_plan(rows[0])
    assert "smiling" in plan["concept"]
    assert "indoor" in plan["unused"] or "portrait" in plan["unused"]
    assert "smiling" not in plan["unused"]


def test_canary_yaml_does_not_create_minus_teacher():
    rows, _meta = load_anima_prompts(CANARY)
    assert rows[0].has_minus_canary
    v_pos = torch.tensor([1.0, 0.0])
    v_neu = torch.tensor([0.0, 0.0])
    v_u = torch.tensor([0.0, 0.0])
    v_neg = torch.tensor([-1.0, 0.0])
    teachers = anima_uni_teachers(v_pos, v_neu, v_u, v_neg)
    assert teachers["minus"] is None
    cos = minus_canary_cosine(v_pos, v_neg, v_u)
    assert float(cos) < 0.0


def test_expand_attributes_pins_unused():
    expanded = expand_attributes_anima(
        {
            "target": "a woman",
            "positive": "a smiling woman",
            "neutral": "a woman",
            "attributes": ["indoor"],
        }
    )
    assert expanded[0]["positive"] == "indoor a smiling woman"
    assert expanded[0]["pins"] == ["indoor"]


def test_fake_lora_targets_and_frozen_conditioner():
    backend = FakeAnimaBackend(device="cpu", rank=16, seed=0)
    names = backend.named_trainable()
    assert names
    assert not any("text_conditioner" in n for n in names)
    joined = " ".join(names)
    for target in LORA_TARGETS:
        stem = target.replace(".0", "")
        assert stem in joined or target in joined
    cond = backend.transformer.text_conditioner
    assert all(not p.requires_grad for p in cond.parameters()) or list(cond.parameters()) == []
    with backend.disable_adapter():
        for lora in backend.loras:
            assert lora.multiplier == 0.0
    assert FAKE_LORA_TARGETS == LORA_TARGETS


def test_dummy_train_fits_uni_without_hub(tmp_path):
    args = parse_args(
        [
            "--dummy",
            "--steps",
            "40",
            "--device",
            "cpu",
            "--name",
            "anima-unit",
            "--prompts_file",
            str(PROMPTS),
            "--save_dir",
            str(tmp_path),
            "--rank",
            "8",
            "--seed",
            "0",
            "--lr",
            "5e-2",
        ]
    )
    sidecar = train_dummy(args)
    assert sidecar["dummy"] is True
    assert sidecar["model_id"] == DEFAULT_MODEL_ID
    assert sidecar["lora_targets"] == list(LORA_TARGETS)
    assert sidecar["frozen_modules"] == ["text_conditioner"]
    assert sidecar["minus_canary"] is False
    assert sidecar["music3_default_untouched"]["lm_target"] == "v9"
    assert sidecar["loss_last"] is not None
    assert sidecar["loss_last"] < 0.05
    side_file = tmp_path / "anima-unit_dummy_last.json"
    assert side_file.is_file()
    written = json.loads(side_file.read_text())
    assert written["sample_steps"] == 40
    assert written["cfg"] == 4.0


def test_dummy_uni_moves_minimal_pair():
    """Student +1 on neu should chase frozen CFG(pos) on a separable pair."""
    backend = FakeAnimaBackend(device="cpu", rank=8, seed=0)
    neu, pos = "a woman", "a smiling woman"
    before = write_plus_alignment(backend, neu, pos, seed=0)
    params = backend.trainable_parameters()
    opt = torch.optim.AdamW(params, lr=5e-2)
    g = torch.Generator().manual_seed(0)
    z = torch.randn((1, *backend.latent_shape), generator=g)
    t = torch.tensor([500.0])
    with torch.no_grad():
        t_plus = anima_cfg_delta(
            backend.predict_v(pos, z, t, frozen=True),
            backend.predict_v("", z, t, frozen=True),
        )
        t_zero = anima_cfg_delta(
            backend.predict_v(neu, z, t, frozen=True),
            backend.predict_v("", z, t, frozen=True),
        )
    for _ in range(60):
        s_plus = anima_cfg_delta(
            backend.predict_v(neu, z, t, frozen=False, scale=1.0),
            backend.predict_v("", z, t, frozen=False, scale=1.0),
        )
        s_zero = anima_cfg_delta(
            backend.predict_v(neu, z, t, frozen=False, scale=0.0),
            backend.predict_v("", z, t, frozen=False, scale=0.0),
        )
        loss = anima_uni_loss(s_plus, t_plus, s_zero, t_zero)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    after = write_plus_alignment(backend, neu, pos, seed=0)
    assert backend.lora_B_norm() > 0.0
    assert after > before
    assert after > 0.3


def test_canary_dummy_logs_minus_without_teaching(tmp_path):
    args = parse_args(
        [
            "--dummy",
            "--steps",
            "4",
            "--device",
            "cpu",
            "--name",
            "anima-canary",
            "--prompts_file",
            str(CANARY),
            "--save_dir",
            str(tmp_path),
            "--rank",
            "4",
        ]
    )
    sidecar = train_dummy(args)
    assert sidecar["minus_canary"] is True
    assert sidecar["canary_cos_last"] is not None


def test_live_loader_stays_offline(monkeypatch):
    args = parse_args(["--device", "cpu"])
    assert args.allow_hub is False

    def _boom(*_a, **_k):
        raise AssertionError("live loader must not import / download in this test")

    monkeypatch.setattr(
        "conceptmod.textsliders.train_lora_anima.ModularPipeline", _boom, raising=False
    )
    with pytest.raises(RuntimeError, match="dummy"):
        load_live_backend(args, torch.device("cpu"))


def test_print_card_does_not_train():
    from conceptmod.textsliders.train_lora_anima import train

    out = train(parse_args(["--print_card"]))
    assert out["model_id"] == DEFAULT_MODEL_ID
    assert out["lora"]["rank"] == 16
