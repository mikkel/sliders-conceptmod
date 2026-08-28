"""Anima image-slider: UNI + unused-token hold. CPU only, no Hub, no GPU."""

from __future__ import annotations

import importlib.util
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
    CONDITIONER_LORA_TARGETS,
    DEFAULT_CFG,
    DEFAULT_CONCEPT_WORDS,
    DEFAULT_CONTROL_PROMPT,
    DEFAULT_LM_TARGET,
    DEFAULT_LORA_TARGETS,
    DEFAULT_LR,
    DEFAULT_MODEL_ID,
    DEFAULT_RANK,
    DEFAULT_RESOLUTION,
    DEFAULT_SAMPLE_EVERY,
    DEFAULT_SAMPLE_SCALES,
    DEFAULT_SAMPLE_SEED,
    DEFAULT_SAMPLE_STEPS,
    DEFAULT_CONCEPT_TARGET,
    DEFAULT_EMBED_IDENTITY_WEIGHT,
    DEFAULT_EMBED_WEIGHT,
    DEFAULT_STRUCT_WEIGHT,
    DEFAULT_TEACHER,
    DEFAULT_TEACHER_GAP_BOOST,
    DEFAULT_TEACHER_STRENGTH,
    DEFAULT_TRAJ_IDENTITY_WEIGHT,
    DEFAULT_TRAJ_STEPS,
    LORA_TARGETS,
    MAN_NEU,
    MAN_PLUS,
    WOMAN_NEU,
    WOMAN_PLUS,
    AnimaSampleGateError,
    align_unused_positions,
    anima_boost_teacher,
    anima_cfg_delta,
    anima_direct_loss,
    anima_direct_teachers,
    anima_embed_delta_cosine,
    anima_embed_mse,
    anima_embed_struct_loss,
    anima_embed_struct_requires_conditioner,
    anima_fake_crop_code,
    anima_flow_euler_step,
    anima_flow_invert,
    anima_flow_sigmas,
    anima_short_trajectory,
    anima_teacher_crop_gap,
    anima_teacher_expr_gap,
    anima_teacher_pair,
    anima_teacher_start_index,
    anima_trajectory_loss,
    anima_uni_loss,
    anima_uni_teachers,
    anima_unused_hold_loss,
    assert_sample_gate,
    concept_tokens,
    expand_attributes_anima,
    infer_sample_prompts,
    live_train_card,
    live_train_command,
    load_anima_prompts,
    looks_like_rgb_noise,
    minus_canary_cosine,
    embed_struct_smoke_command,
    parse_concept_words,
    resolve_anima_concept_target,
    resolve_anima_lm_target,
    resolve_anima_lora_targets,
    resolve_anima_teacher,
    resolve_anima_train_recipe,
    same_crop_smoke_command,
    row_token_plan,
    splice_unused_embeds,
    stock_teacher_smoke_captions,
    turbo_preview_card,
    turbo_preview_sample_command,
    unused_token_mask,
    unused_vocab,
    word_tokens,
    TURBO_COMFY_REPO,
    TURBO_DIFFUSERS_OUTPUT,
    TURBO_LICENSE,
    TURBO_PREVIEW_ONLY,
    TURBO_SAMPLE_CFG,
    TURBO_SAMPLE_STEPS,
    TURBO_TRANSFORMER_FILE,
)
from conceptmod.textsliders.train_lora_anima import (
    _call_modular_pipe,
    _cycle_row,
    _sample_zt,
    apply_anima_guider_cfg,
    emit_inprocess_samples,
    freeze_anima_conditioner,
    load_live_backend,
    parse_args,
    peft_adapter_scale,
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
    assert args.lr == DEFAULT_LR == 1e-4
    assert args.lm_target == DEFAULT_LM_TARGET == "trajectory"
    assert args.teacher is None
    assert resolve_anima_train_recipe(args.lm_target, args.teacher).teacher == DEFAULT_TEACHER == "caption"
    assert args.traj_steps == DEFAULT_TRAJ_STEPS == 4
    assert args.traj_identity_weight == DEFAULT_TRAJ_IDENTITY_WEIGHT == 0.25
    assert args.teacher_gap_boost == DEFAULT_TEACHER_GAP_BOOST == 1.0
    assert args.teacher_strength == DEFAULT_TEACHER_STRENGTH == 0.5
    assert args.concept_target is None
    assert args.embed_weight == DEFAULT_EMBED_WEIGHT == 1.0
    assert args.embed_identity_weight == DEFAULT_EMBED_IDENTITY_WEIGHT == 1.0
    assert args.struct_weight == DEFAULT_STRUCT_WEIGHT == 1.0
    assert args.control_prompt == DEFAULT_CONTROL_PROMPT
    assert args.sample_every == DEFAULT_SAMPLE_EVERY == 100
    assert args.sample_seed == DEFAULT_SAMPLE_SEED == 42
    assert args.sample_mode == "peft_pipe"
    assert args.dummy is False
    assert args.allow_hub is False
    assert args.lora_targets == DEFAULT_LORA_TARGETS == "conditioner"
    card = live_train_card()
    assert card["model_id"] == DEFAULT_MODEL_ID
    assert card["lora"]["rank"] == 16
    assert card["lora"]["lora_targets"] == "conditioner"
    assert card["lora"]["targets"] == list(CONDITIONER_LORA_TARGETS)
    assert card["lora"]["train_text_conditioner"] is True
    assert card["lora"]["train_dit"] is False
    assert card["lora"]["train_text_encoder"] is False
    assert card["lora"]["frozen_modules"] == ["text_encoder"]
    assert "AnimaTextConditioner" in card["lora"]["text_encoder_note"]
    assert card["resolution"] == 768
    assert card["sample_steps"] == 40
    assert card["cfg"] == 4.0
    assert card["lr"] == 1e-4
    assert card["control_prompt"] == "a bowl of fruit on a table"
    assert card["sample_scales"] == list(DEFAULT_SAMPLE_SCALES)
    assert card["sample_seed"] == 42
    assert card["lm_target"] == "trajectory"
    assert card["teacher"] == "caption"
    assert card["teachers"] == ["caption", "same_crop"]
    assert card["teacher_strength"] == 0.5
    assert "full-body" in card["caption_teacher_failure"]
    assert "invert" in card["same_crop_teacher"]
    assert "--teacher same_crop" in card["same_crop_smoke_4090"]
    assert "--steps 8" in card["same_crop_smoke_4090"]
    assert "embed_struct" in card["lm_targets"]
    assert card["concept_target"] == DEFAULT_CONCEPT_TARGET == "caption"
    assert card["concept_targets"] == ["caption", "same_crop"]
    assert "E_θ(neu)" in card["embed_struct_split"]
    assert "teacher_strength" in card["embed_struct_split"]
    assert "--lm_target embed_struct" in card["embed_struct_smoke_4090"]
    assert "--teacher_strength" not in card["embed_struct_smoke_4090"]
    assert card["embed_struct_vs"]["same_crop"]
    assert "any yaml" in card["embed_struct_vs"]["embed_struct"]
    assert card["traj_steps"] == 4
    assert card["teacher_gap_boost"] == 1.0
    assert "predict_v" in card["traj_loop"]
    assert "MSE(x_student, x_plus)" in card["traj_loss"]
    assert "0.99993" in card["one_step_failure"]
    assert card["sample_every"] == 100
    assert "PEFT" in card["sample_gate"]
    assert card["sample_mode"] == "peft_pipe"
    assert "train_faithful" in card["sample_modes"]
    assert card["stock_teacher_smoke"]["woman"]["neu"] == WOMAN_NEU
    assert card["stock_teacher_smoke"]["woman"]["plus"] == WOMAN_PLUS
    assert card["device"] == "cuda:0"
    assert card["music3_default_untouched"] == {
        "lm_target": "v9",
        "pole_mode": "hidden",
    }
    assert card["turbo"] == "preview_only"
    assert args.print_turbo_preview is False
    cmd = live_train_command()
    assert "circlestone-labs/Anima-Base-v1.0-Diffusers" in cmd
    assert "--rank 16" in cmd
    assert "--resolution 768" in cmd
    assert "--sample_steps 40" in cmd
    assert "--cfg 4" in cmd
    assert "--lr 0.0001" in cmd
    assert "--lm_target trajectory" in cmd
    assert "--traj_steps 4" in cmd
    assert "--sample_every 100" in cmd
    assert "--lora_targets conditioner" in cmd
    assert "HF_HUB_OFFLINE=1" in cmd
    assert "--lora_targets conditioner" in card["smile_retrain_4090"]
    assert "--resolution 512" in card["smile_retrain_4090"]


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


def test_yaml_pins_attributes_without_prefixing():
    rows, meta = load_anima_prompts(PROMPTS)
    assert meta.plus_label == "smiling"
    assert meta.concept_words == DEFAULT_CONCEPT_WORDS
    assert parse_concept_words(meta.concept_words) == [
        "smiling",
        "smile",
        "happy",
        "joyful",
        "teeth",
    ]
    assert len(rows) == 2
    assert all(not row.has_minus_canary for row in rows)
    assert rows[0].attributes == ["indoor", "portrait"]
    assert rows[1].attributes == ["indoor", "portrait"]
    assert rows[0].infer_prompt == rows[0].neutral == WOMAN_NEU
    assert rows[1].infer_prompt == rows[1].neutral == MAN_NEU
    assert all(not row.infer_prompt.startswith("indoor") for row in rows)
    assert all(not row.infer_prompt.startswith("portrait") for row in rows)
    assert all(not row.positive.startswith("indoor") for row in rows)
    assert all(not row.neutral.startswith("indoor") for row in rows)
    assert all("neutral expression, closed mouth" in row.neutral for row in rows)
    assert all("neutral expression, closed mouth" in row.target for row in rows)
    assert all("big smile showing teeth" in row.positive for row in rows)
    assert all("happy joyful expression" in row.positive for row in rows)
    assert all("smile" not in row.neutral for row in rows)
    assert all("teeth" not in row.neutral for row in rows)
    plan = row_token_plan(rows[0])
    for word in ("smile", "happy", "joyful", "teeth"):
        assert word in plan["concept"]
        assert word not in plan["unused"]
    assert "smiling" not in plan["unused"]
    assert "indoor" in plan["unused"]
    assert "portrait" in plan["unused"]
    assert "woman" in plan["unused"]
    assert "chair" in plan["unused"]


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
    assert expanded[0]["positive"] == "a smiling woman"
    assert expanded[0]["target"] == "a woman"
    assert expanded[0]["neutral"] == "a woman"
    assert expanded[0]["pins"] == ["indoor"]
    assert expanded[0]["attributes"] == ["indoor"]


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


def test_lora_targets_resolver_and_cli():
    spec = resolve_anima_lora_targets()
    assert spec.label == "conditioner"
    assert spec.train_conditioner is True
    assert spec.train_dit is False
    assert spec.frozen_modules == ("text_encoder",)
    assert spec.active_attn_targets == list(CONDITIONER_LORA_TARGETS)
    dit = resolve_anima_lora_targets("dit")
    assert dit.train_dit is True
    assert dit.train_conditioner is False
    assert "text_conditioner" in dit.frozen_modules
    joint = resolve_anima_lora_targets("dit+conditioner")
    assert joint.train_dit is True
    assert joint.train_conditioner is True
    assert joint.frozen_modules == ("text_encoder",)
    alias = resolve_anima_lora_targets("text_conditioner")
    assert alias.label == "conditioner"
    with pytest.raises(ValueError, match="text_encoder"):
        resolve_anima_lora_targets("text_encoder")
    with pytest.raises(ValueError, match="lora_targets"):
        resolve_anima_lora_targets("v9")
    args = parse_args(["--lora_targets", "dit"])
    assert args.lora_targets == "dit"
    joint_args = parse_args(["--lora_targets", "dit+conditioner"])
    assert joint_args.lora_targets == "dit+conditioner"


def test_dummy_conditioner_trainable_when_enabled():
    backend = FakeAnimaBackend(device="cpu", rank=4, seed=0, lora_targets="conditioner")
    names = backend.named_trainable()
    assert names
    assert any("text_conditioner" in n for n in names)
    joined = " ".join(names)
    for target in CONDITIONER_LORA_TARGETS:
        assert target in joined
    assert not any("to_q" in n or "to_k" in n for n in names)
    assert backend.lora_spec.train_conditioner is True
    params = backend.trainable_parameters()
    assert params
    assert all(p.requires_grad for p in params)


def test_dummy_conditioner_frozen_when_disabled():
    backend = FakeAnimaBackend(device="cpu", rank=4, seed=0, lora_targets="dit")
    names = backend.named_trainable()
    assert names
    assert not any("text_conditioner" in n for n in names)
    cond = backend.transformer.text_conditioner
    assert all(not p.requires_grad for p in cond.parameters())
    from conceptmod.textsliders.train_lora_anima import _assert_lora_train_state

    _assert_lora_train_state(backend, resolve_anima_lora_targets("dit"))
    with pytest.raises(RuntimeError, match="trainable"):
        _assert_lora_train_state(backend, resolve_anima_lora_targets("conditioner"))


def test_dummy_train_conditioner_only(tmp_path):
    args = parse_args(
        [
            "--dummy",
            "--steps",
            "6",
            "--device",
            "cpu",
            "--name",
            "anima-cond",
            "--prompts_file",
            str(PROMPTS),
            "--save_dir",
            str(tmp_path),
            "--rank",
            "4",
            "--lora_targets",
            "conditioner",
        ]
    )
    sidecar = train_dummy(args)
    assert sidecar["lora_targets"] == "conditioner"
    assert sidecar["train_text_conditioner"] is True
    assert sidecar["train_dit"] is False
    assert sidecar["conditioner_lora_targets"] == list(CONDITIONER_LORA_TARGETS)
    assert sidecar["dit_lora_targets"] == []
    assert sidecar["frozen_modules"] == ["text_encoder"]
    assert sidecar["adapted_modules"] == ["text_conditioner"]
    assert sidecar["loss_last"] is not None
    assert sidecar["music3_default_untouched"]["lm_target"] == "v9"


def test_dummy_train_dit_plus_conditioner(tmp_path):
    args = parse_args(
        [
            "--dummy",
            "--steps",
            "4",
            "--device",
            "cpu",
            "--name",
            "anima-joint",
            "--prompts_file",
            str(PROMPTS),
            "--save_dir",
            str(tmp_path),
            "--rank",
            "4",
            "--lora_targets",
            "dit+conditioner",
        ]
    )
    sidecar = train_dummy(args)
    assert sidecar["lora_targets"] == "dit+conditioner"
    assert sidecar["train_dit"] is True
    assert sidecar["train_text_conditioner"] is True
    assert sidecar["dit_lora_targets"] == list(LORA_TARGETS)
    assert sidecar["conditioner_lora_targets"] == list(CONDITIONER_LORA_TARGETS)
    assert sidecar["adapted_modules"] == ["transformer", "text_conditioner"]
    assert sidecar["frozen_modules"] == ["text_encoder"]


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
            "--lora_targets",
            "dit",
        ]
    )
    sidecar = train_dummy(args)
    assert sidecar["dummy"] is True
    assert sidecar["model_id"] == DEFAULT_MODEL_ID
    assert sidecar["lora_targets"] == "dit"
    assert sidecar["train_text_conditioner"] is False
    assert sidecar["dit_lora_targets"] == list(LORA_TARGETS)
    assert sidecar["frozen_modules"] == ["text_encoder", "text_conditioner"]
    assert sidecar["minus_canary"] is False
    assert sidecar["music3_default_untouched"]["lm_target"] == "v9"
    assert sidecar["lm_target"] == "trajectory"
    assert sidecar["traj_steps"] == 4
    assert sidecar["traj_loss"] == "MSE(x_student, x_plus) + λ_id*MSE(x_zero, x_neu)"
    assert sidecar["loss_last"] is not None
    assert sidecar["loss_last"] < 0.05
    assert sidecar["sample_grid"]["method"] == "peft_pipe_prompt"
    assert sidecar["sample_grid"]["n"] == 12
    assert WOMAN_NEU in sidecar["train_infer_prompts"]
    assert MAN_NEU in sidecar["train_infer_prompts"]
    assert sidecar["train_infer_prompts"][0] == sidecar["sample_infer_prompts"][0]
    assert sidecar["sample_infer_prompts"][:2] == [WOMAN_NEU, MAN_NEU]
    side_file = tmp_path / "anima-unit_dummy_last.json"
    assert side_file.is_file()
    written = json.loads(side_file.read_text())
    assert written["sample_steps"] == 40
    assert written["cfg"] == 4.0
    samples = tmp_path / "samples"
    meta = json.loads((samples / "step0040_meta.json").read_text())
    assert meta["seed"] == 42
    assert meta["scales"] == [0.0, 0.25, 0.5, 1.0]
    prompts = {row["prompt"] for row in meta["samples"]}
    assert WOMAN_NEU in prompts
    assert MAN_NEU in prompts
    assert "a bowl of fruit on a table" in prompts
    assert all(not row["looks_like_noise"] for row in meta["samples"])
    assert all(row.get("cfg_via") == "guider.config.guidance_scale" for row in meta["samples"])
    assert len(list(samples.glob("*.png"))) == 12


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
    assert out["lora"]["lora_targets"] == "conditioner"
    assert out["lora"]["train_text_conditioner"] is True
    assert out["lora"]["train_text_encoder"] is False
    assert out["lr"] == 1e-4
    assert out["lm_target"] == "trajectory"
    assert out["teacher"] == "caption"
    assert out["traj_steps"] == 4
    assert "MSE(x_student, x_plus)" in out["traj_loss"]
    same = train(parse_args(["--print_card", "--lm_target", "same_crop"]))
    assert same["lm_target"] == "same_crop"
    assert same["teacher"] == "same_crop"
    assert "invert" in same["recipe"]
    dit = train(parse_args(["--print_card", "--lora_targets", "dit"]))
    assert dit["lora"]["lora_targets"] == "dit"
    assert dit["lora"]["train_text_conditioner"] is False
    assert dit["lora"]["train_dit"] is True
    assert "text_conditioner" in dit["lora"]["frozen_modules"]
    split = train(parse_args(["--print_card", "--lm_target", "embed_struct"]))
    assert split["lm_target"] == "embed_struct"
    assert split["concept_target"] == "caption"
    assert "E_θ(neu)" in split["embed_struct_loss"]
    assert "--lm_target embed_struct" in split["embed_struct_smoke_4090"]


def test_sample_zt_draws_on_cpu_then_moves():
    backend = FakeAnimaBackend(device="cpu", rank=4, seed=0)
    z, t = _sample_zt(backend, seed=7, step=3)
    assert z.device == backend.device
    assert t.device == backend.device
    assert tuple(z.shape[1:]) == backend.latent_shape


def test_freeze_conditioner_does_not_need_pipe_named_parameters():
    class _NotAModule:
        def __init__(self):
            self.text_conditioner = torch.nn.Linear(2, 2)
            self.transformer = torch.nn.Linear(2, 2)
            for param in self.text_conditioner.parameters():
                param.requires_grad_(True)

    pipe = _NotAModule()
    assert not hasattr(pipe, "named_parameters")
    freeze_anima_conditioner(pipe)
    assert all(not param.requires_grad for param in pipe.text_conditioner.parameters())
    assert all(param.requires_grad for param in pipe.transformer.parameters())


def test_freeze_conditioner_keeps_lora_params_when_training():
    class _Cond(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = torch.nn.Linear(2, 2)
            self.lora_A = torch.nn.Linear(2, 1, bias=False)

    class _NotAModule:
        def __init__(self):
            self.text_conditioner = _Cond()
            self.transformer = torch.nn.Linear(2, 2)

    pipe = _NotAModule()
    freeze_anima_conditioner(pipe, train_conditioner=True)
    assert all(not p.requires_grad for p in pipe.text_conditioner.base.parameters())
    assert all(p.requires_grad for p in pipe.text_conditioner.lora_A.parameters())
    assert all(p.requires_grad for p in pipe.transformer.parameters())


def test_infer_sample_prompts_are_neu_plus_fruit_bowl():
    rows, _meta = load_anima_prompts(PROMPTS)
    prompts = infer_sample_prompts(rows)
    assert prompts == [
        WOMAN_NEU,
        MAN_NEU,
        "a bowl of fruit on a table",
    ]
    assert prompts[0] == rows[0].infer_prompt == rows[0].neutral
    assert prompts[1] == rows[1].infer_prompt == rows[1].neutral
    assert all("smile" not in p for p in prompts)
    assert all("teeth" not in p for p in prompts)
    assert all("happy" not in p for p in prompts)


def test_train_infer_prompt_equals_pipe_prompt_no_strip(tmp_path):
    """rows[0].infer_prompt / neutral is the exact pipe(prompt=...) string."""
    rows, _meta = load_anima_prompts(PROMPTS)
    assert rows[0].infer_prompt == rows[0].neutral == WOMAN_NEU
    assert rows[1].infer_prompt == rows[1].neutral == MAN_NEU
    sample = infer_sample_prompts(rows)
    assert sample[0] == rows[0].infer_prompt
    assert sample[1] == rows[1].infer_prompt
    backend = FakeAnimaBackend(device="cpu", rank=4, seed=0)
    args = parse_args(
        [
            "--dummy",
            "--device",
            "cpu",
            "--save_dir",
            str(tmp_path),
            "--resolution",
            "32",
        ]
    )
    records = emit_inprocess_samples(
        backend, args, tmp_path, step=1, rows=rows, dummy=True
    )
    pipe_prompts = [row["prompt"] for row in records]
    assert rows[0].infer_prompt in pipe_prompts
    assert rows[0].neutral in pipe_prompts
    assert backend.pipe.prompts_seen[0] == rows[0].infer_prompt
    assert set(backend.pipe.prompts_seen) == {
        rows[0].infer_prompt,
        rows[1].infer_prompt,
        DEFAULT_CONTROL_PROMPT,
    }
    assert "indoor" not in rows[0].infer_prompt
    assert not any(p.startswith("indoor ") for p in backend.pipe.prompts_seen)
    assert not any(p.startswith("portrait ") for p in backend.pipe.prompts_seen)


def test_rgb_noise_gate_catches_static_and_passes_ramps():
    rng = torch.Generator().manual_seed(0)
    noise = torch.randint(0, 256, (64, 64, 3), generator=rng, dtype=torch.uint8).numpy()
    assert looks_like_rgb_noise(noise)
    ys = torch.linspace(40, 180, 64).unsqueeze(1).expand(64, 64)
    ramp = torch.stack([ys, ys.T, 0.5 * ys + 0.5 * ys.T], dim=-1).numpy()
    assert not looks_like_rgb_noise(ramp)
    with pytest.raises(AnimaSampleGateError, match="base pipeline"):
        assert_sample_gate(
            [{"scale": 0.0, "mean": 122.0, "std": 75.0, "looks_like_noise": True}]
        )
    with pytest.raises(AnimaSampleGateError, match="adapter is broken"):
        assert_sample_gate(
            [
                {"scale": 0.0, "mean": 90.0, "std": 40.0, "looks_like_noise": False},
                {"scale": 0.25, "mean": 122.0, "std": 75.0, "looks_like_noise": True},
            ]
        )
    assert_sample_gate(
        [
            {"scale": 0.0, "mean": 90.0, "std": 40.0, "looks_like_noise": False},
            {"scale": 0.25, "mean": 95.0, "std": 42.0, "looks_like_noise": False},
        ]
    )


def test_peft_adapter_scale_uses_disable_not_merge():
    backend = FakeAnimaBackend(device="cpu", rank=4, seed=0)
    backend.set_lora_scale(1.0)
    with peft_adapter_scale(backend, 0.0):
        assert all(lora.multiplier == 0.0 for lora in backend.loras)
    assert all(lora.multiplier == 1.0 for lora in backend.loras)
    with peft_adapter_scale(backend, 0.25):
        assert all(abs(lora.multiplier - 0.25) < 1e-8 for lora in backend.loras)
    assert all(lora.multiplier == 1.0 for lora in backend.loras)
    out = backend.pipe(prompt="a woman sitting on a chair", height=32, width=32)
    assert out.images
    assert not looks_like_rgb_noise(out.images[0])


def test_stock_teacher_smoke_is_v3_closed_mouth_vs_teeth():
    smoke = stock_teacher_smoke_captions()
    assert smoke["woman"]["neu"] == WOMAN_NEU
    assert smoke["woman"]["plus"] == WOMAN_PLUS
    assert smoke["man"]["neu"] == MAN_NEU
    assert smoke["man"]["plus"] == MAN_PLUS
    assert smoke["concept_words"] == DEFAULT_CONCEPT_WORDS
    assert "closed mouth" in smoke["woman"]["neu"]
    assert "teeth" in smoke["woman"]["plus"]
    assert "CFG teacher" in smoke["note"]


def test_modular_pipe_sets_guider_config_cfg():
    backend = FakeAnimaBackend(device="cpu", rank=4, seed=0)
    pipe = backend.pipe
    pipe.guider.config.guidance_scale = 1.0
    pipe.guider.guidance_scale = 1.0
    images = _call_modular_pipe(
        pipe,
        WOMAN_NEU,
        steps=2,
        height=32,
        width=32,
        cfg=4.0,
        seed=0,
        device=backend.device,
    )
    assert images
    assert pipe.last_guidance_scale == 4.0
    # Restore after the call so later samples do not leak CFG.
    assert pipe.guider.config.guidance_scale == 1.0
    assert pipe.guider.guidance_scale == 1.0


def test_modular_pipe_ignores_top_level_guidance_scale():
    backend = FakeAnimaBackend(device="cpu", rank=4, seed=0)
    pipe = backend.pipe
    pipe.guider.config.guidance_scale = 1.0
    with pytest.warns(UserWarning, match="guidance_scale"):
        pipe(prompt=WOMAN_NEU, height=16, width=16, guidance_scale=4.0)
    assert pipe.last_guidance_scale == 1.0


def test_modular_cfg_fails_closed_without_guider():
    class _NoGuider:
        def __call__(self, **_kwargs):
            raise AssertionError("must not sample at CFG 1")

    with pytest.raises(RuntimeError, match="guider"):
        apply_anima_guider_cfg(_NoGuider(), 4.0)
    with pytest.raises(RuntimeError, match="CFG 4"):
        _call_modular_pipe(
            _NoGuider(),
            WOMAN_NEU,
            steps=1,
            height=8,
            width=8,
            cfg=4.0,
            seed=0,
            device="cpu",
        )


def test_v3_unused_hold_skips_declared_concept_words():
    unused = unused_vocab(
        WOMAN_NEU,
        WOMAN_NEU,
        attributes=["indoor", "portrait"],
        concept_words=DEFAULT_CONCEPT_WORDS,
    )
    pos = word_tokens(WOMAN_PLUS)
    concept = concept_tokens(WOMAN_PLUS, unused)
    for word in ("smile", "happy", "joyful", "teeth"):
        assert word in concept
        assert word not in unused
    assert "smiling" not in unused
    assert "woman" in unused
    assert "chair" in unused
    assert "indoor" in unused
    assert "mouth" in unused
    mask = unused_token_mask(pos, unused)
    held = [tok for tok, keep in zip(pos, mask) if keep]
    skipped = [tok for tok, keep in zip(pos, mask) if not keep]
    assert "smile" in skipped
    assert "teeth" in skipped
    assert "woman" in held
    assert "chair" in held


def test_cycle_training_rows_woman_then_man():
    rows, _meta = load_anima_prompts(PROMPTS)
    plans = [row_token_plan(row) for row in rows]
    first, _ = _cycle_row(rows, plans, 0)
    second, _ = _cycle_row(rows, plans, 1)
    third, _ = _cycle_row(rows, plans, 2)
    assert first.infer_prompt == WOMAN_NEU
    assert second.infer_prompt == MAN_NEU
    assert third.infer_prompt == WOMAN_NEU


def test_anima_lm_target_default_trajectory_rejects_unknown():
    assert resolve_anima_lm_target() == "trajectory"
    assert resolve_anima_lm_target("direct") == "direct"
    assert resolve_anima_lm_target("cfg_delta") == "cfg_delta"
    assert resolve_anima_lm_target("trajectory") == "trajectory"
    assert resolve_anima_lm_target("same_crop") == "same_crop"
    assert resolve_anima_lm_target("embed_struct") == "embed_struct"
    assert resolve_anima_lm_target("conditioner_embed") == "embed_struct"
    assert resolve_anima_lm_target("embed_structure") == "embed_struct"
    with pytest.raises(ValueError, match="lm_target"):
        resolve_anima_lm_target("v9")
    args = parse_args(["--lm_target", "cfg_delta"])
    assert args.lm_target == "cfg_delta"
    traj = parse_args(["--lm_target", "trajectory", "--traj_steps", "8"])
    assert traj.lm_target == "trajectory"
    assert traj.traj_steps == 8
    locked = parse_args(["--lm_target", "same_crop", "--teacher", "same_crop"])
    assert locked.lm_target == "same_crop"
    assert locked.teacher == "same_crop"
    embed = parse_args(["--lm_target", "conditioner_embed"])
    assert embed.lm_target == "conditioner_embed"
    assert resolve_anima_lm_target(embed.lm_target) == "embed_struct"


def test_direct_loss_is_raw_velocity_mse():
    plus = torch.tensor([1.0, 0.0])
    neu = torch.tensor([0.0, 1.0])
    teachers = anima_direct_teachers(plus, neu)
    assert teachers["minus"] is None
    assert torch.allclose(teachers["plus"], plus)
    assert torch.allclose(teachers["zero"], neu)
    base = anima_direct_loss(plus, plus, neu, neu)
    assert float(base) == pytest.approx(0.0, abs=1e-8)
    moved = anima_direct_loss(plus + 1, plus, neu, neu)
    assert float(moved) > float(base)


def test_direct_moves_minimal_pair():
    """Neu+adapter should chase frozen v(pos); scale 0 stays neu."""
    backend = FakeAnimaBackend(device="cpu", rank=8, seed=0)
    neu, pos = "a woman", "a smiling woman"
    before = write_plus_alignment(backend, neu, pos, seed=0)
    params = backend.trainable_parameters()
    opt = torch.optim.AdamW(params, lr=5e-2)
    g = torch.Generator().manual_seed(0)
    z = torch.randn((1, *backend.latent_shape), generator=g)
    t = torch.tensor([500.0])
    with torch.no_grad():
        t_plus = backend.predict_v(pos, z, t, frozen=True)
        t_zero = backend.predict_v(neu, z, t, frozen=True)
    for _ in range(60):
        s_plus = backend.predict_v(neu, z, t, frozen=False, scale=1.0)
        s_zero = backend.predict_v(neu, z, t, frozen=False, scale=0.0)
        loss = anima_direct_loss(s_plus, t_plus, s_zero, t_zero)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    after = write_plus_alignment(backend, neu, pos, seed=0)
    assert backend.lora_B_norm() > 0.0
    assert after > before
    assert after > 0.3


def test_dummy_train_cfg_delta_still_works(tmp_path):
    args = parse_args(
        [
            "--dummy",
            "--steps",
            "8",
            "--device",
            "cpu",
            "--name",
            "anima-cfg-delta",
            "--prompts_file",
            str(PROMPTS),
            "--save_dir",
            str(tmp_path),
            "--rank",
            "4",
            "--lm_target",
            "cfg_delta",
        ]
    )
    sidecar = train_dummy(args)
    assert sidecar["lm_target"] == "cfg_delta"
    assert WOMAN_NEU in sidecar["train_infer_prompts"]
    assert MAN_NEU in sidecar["train_infer_prompts"]
    assert sidecar["music3_default_untouched"]["lm_target"] == "v9"


def test_dummy_train_direct_still_works(tmp_path):
    args = parse_args(
        [
            "--dummy",
            "--steps",
            "8",
            "--device",
            "cpu",
            "--name",
            "anima-direct",
            "--prompts_file",
            str(PROMPTS),
            "--save_dir",
            str(tmp_path),
            "--rank",
            "4",
            "--lm_target",
            "direct",
        ]
    )
    sidecar = train_dummy(args)
    assert sidecar["lm_target"] == "direct"
    assert sidecar["teacher_gap_boost"] == 1.0
    assert sidecar["music3_default_untouched"]["lm_target"] == "v9"


def test_teacher_gap_boost_is_off_at_one():
    v_pos = torch.tensor([2.0, 0.0])
    v_neu = torch.tensor([0.0, 1.0])
    assert torch.allclose(anima_boost_teacher(v_pos, v_neu, 1.0), v_pos)
    assert torch.allclose(anima_boost_teacher(v_pos, v_neu, 0.5), v_pos)
    boosted = anima_boost_teacher(v_pos, v_neu, 3.0)
    assert torch.allclose(boosted, v_neu + 3.0 * (v_pos - v_neu))
    args = parse_args(["--teacher_gap_boost", "4"])
    assert args.teacher_gap_boost == 4.0


def test_flow_euler_step_matches_flowmatch_scheduler():
    sample = torch.tensor([1.0, -2.0])
    vel = torch.tensor([0.5, 0.25])
    out = anima_flow_euler_step(sample, vel, 1.0, 0.25)
    assert torch.allclose(out, sample + (0.25 - 1.0) * vel)
    sigmas = anima_flow_sigmas(4)
    assert sigmas.numel() == 5
    assert float(sigmas[0]) == pytest.approx(1.0)
    assert float(sigmas[-2]) == pytest.approx(0.25)
    assert float(sigmas[-1]) == pytest.approx(0.0)


def test_trajectory_loss_is_student_vs_plus_plus_identity():
    plus = torch.tensor([1.0, 0.0])
    student = torch.tensor([1.0, 0.0])
    neu = torch.tensor([0.0, 1.0])
    zero = torch.tensor([0.0, 1.0])
    base = anima_trajectory_loss(student, plus, zero, neu, identity_weight=0.25)
    assert float(base) == pytest.approx(0.0, abs=1e-8)
    moved = anima_trajectory_loss(student + 1, plus, zero, neu, identity_weight=0.25)
    assert float(moved) > float(base)
    ident = anima_trajectory_loss(student, plus, zero + 2, neu, identity_weight=0.25)
    assert float(ident) > float(base)
    no_id = anima_trajectory_loss(student, plus, zero + 2, neu, identity_weight=0.0)
    assert float(no_id) == pytest.approx(0.0, abs=1e-8)


def test_dummy_trajectory_moves_minimal_pair():
    """Neu+adapter short traj should chase frozen plus traj from the same z_T."""
    backend = FakeAnimaBackend(device="cpu", rank=8, seed=0)
    neu, pos = "a woman", "a smiling woman"
    g = torch.Generator().manual_seed(0)
    z = torch.randn((1, *backend.latent_shape), generator=g)
    with torch.no_grad():
        x_plus = anima_short_trajectory(backend, pos, z, num_steps=4, frozen=True)
        x_before = anima_short_trajectory(
            backend, neu, z, num_steps=4, frozen=False, scale=1.0
        )
    before = float(anima_trajectory_loss(x_before, x_plus, identity_weight=0.0))
    params = backend.trainable_parameters()
    opt = torch.optim.AdamW(params, lr=5e-2)
    for _ in range(40):
        x_student = anima_short_trajectory(
            backend, neu, z, num_steps=4, frozen=False, scale=1.0
        )
        with torch.no_grad():
            x_plus = anima_short_trajectory(backend, pos, z, num_steps=4, frozen=True)
            x_neu = anima_short_trajectory(backend, neu, z, num_steps=4, frozen=True)
            x_zero = anima_short_trajectory(
                backend, neu, z, num_steps=4, frozen=False, scale=0.0
            )
        loss = anima_trajectory_loss(
            x_student, x_plus, x_zero, x_neu, identity_weight=0.25
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        x_after = anima_short_trajectory(
            backend, neu, z, num_steps=4, frozen=False, scale=1.0
        )
        x_plus = anima_short_trajectory(backend, pos, z, num_steps=4, frozen=True)
        x_zero = anima_short_trajectory(
            backend, neu, z, num_steps=4, frozen=False, scale=0.0
        )
        x_neu = anima_short_trajectory(backend, neu, z, num_steps=4, frozen=True)
    after = float(anima_trajectory_loss(x_after, x_plus, identity_weight=0.0))
    ident = float(anima_trajectory_loss(x_zero, x_neu, identity_weight=0.0))
    assert backend.lora_B_norm() > 0.0
    assert after < before
    assert ident < 1e-6


def test_dummy_train_trajectory_prints_target(tmp_path):
    args = parse_args(
        [
            "--dummy",
            "--steps",
            "6",
            "--device",
            "cpu",
            "--name",
            "anima-traj",
            "--prompts_file",
            str(PROMPTS),
            "--save_dir",
            str(tmp_path),
            "--rank",
            "4",
            "--lm_target",
            "trajectory",
            "--traj_steps",
            "4",
        ]
    )
    sidecar = train_dummy(args)
    assert sidecar["lm_target"] == "trajectory"
    assert sidecar["teacher"] == "caption"
    assert sidecar["traj_steps"] == 4
    assert "Euler" in sidecar["traj_loop"]
    assert sidecar["sample_grid"]["scales"] == [0.0, 0.25, 0.5, 1.0]
    assert sidecar["music3_default_untouched"]["lm_target"] == "v9"


def test_same_crop_teacher_resolver_and_cli():
    assert resolve_anima_teacher() == "caption"
    assert resolve_anima_teacher("same_crop") == "same_crop"
    assert resolve_anima_teacher("img2img") == "same_crop"
    assert resolve_anima_teacher("invert") == "same_crop"
    with pytest.raises(ValueError, match="teacher"):
        resolve_anima_teacher("v9")
    implied = resolve_anima_train_recipe("same_crop")
    assert implied.loss_kind == "trajectory"
    assert implied.teacher == "same_crop"
    assert implied.teacher_strength == 0.5
    explicit = resolve_anima_train_recipe("trajectory", "same_crop", 0.35)
    assert explicit.lm_target == "trajectory"
    assert explicit.teacher == "same_crop"
    assert explicit.teacher_strength == 0.35
    with pytest.raises(ValueError, match="same_crop"):
        resolve_anima_train_recipe("direct", "same_crop")
    with pytest.raises(ValueError, match="same_crop"):
        resolve_anima_train_recipe("same_crop", "caption")
    args = parse_args(["--teacher", "same_crop", "--teacher_strength", "0.35"])
    assert args.teacher == "same_crop"
    assert args.teacher_strength == 0.35
    smoke = same_crop_smoke_command()
    assert "--lm_target same_crop" in smoke
    assert "--teacher same_crop" in smoke
    assert "--steps 8" in smoke
    assert "--resolution 512" in smoke


def test_flow_invert_is_linear_and_start_index_picks_sigma():
    x0 = torch.tensor([2.0, 0.0])
    noise = torch.tensor([0.0, 4.0])
    mid = anima_flow_invert(x0, noise, 0.5)
    assert torch.allclose(mid, torch.tensor([1.0, 2.0]))
    assert torch.allclose(anima_flow_invert(x0, noise, 1.0), noise)
    assert torch.allclose(anima_flow_invert(x0, noise, 0.0), x0)
    sigmas = anima_flow_sigmas(4)
    assert anima_teacher_start_index(sigmas, 1.0) == 0
    assert anima_teacher_start_index(sigmas, 0.5) == 2
    assert anima_teacher_start_index(sigmas, 0.35) == 3
    assert anima_teacher_start_index(sigmas, 0.25) == 3


def test_same_crop_teacher_shares_neu_crop_in_fake_backend():
    """Caption plus from z_T jumps crop; invert plus keeps neu crop."""
    backend = FakeAnimaBackend(device="cpu", rank=4, seed=0, lora_targets="dit")
    neu, pos = WOMAN_NEU, WOMAN_PLUS
    g = torch.Generator().manual_seed(3)
    z = torch.randn((1, *backend.latent_shape), generator=g)
    caption = anima_teacher_pair(
        backend, neu, pos, z, num_steps=4, teacher="caption"
    )
    locked = anima_teacher_pair(
        backend, neu, pos, z, num_steps=4, teacher="same_crop", strength=0.5
    )
    assert caption.shared_crop is False
    assert locked.shared_crop is True
    assert locked.z_mid is not None
    assert locked.start_index == 2
    assert locked.start_sigma == pytest.approx(0.5)
    cap_crop = anima_teacher_crop_gap(caption.x_neu, caption.x_plus)
    lock_crop = anima_teacher_crop_gap(locked.x_neu, locked.x_plus)
    lock_expr = anima_teacher_expr_gap(locked.x_neu, locked.x_plus)
    cap_expr = anima_teacher_expr_gap(caption.x_neu, caption.x_plus)
    assert cap_crop > lock_crop
    assert cap_crop > 0.05
    assert lock_crop < 0.5 * cap_crop
    assert lock_expr > 1e-3
    assert cap_expr > 1e-3
    assert torch.allclose(caption.x_neu, locked.x_neu)
    # Invert is latent-locked to the same z_T.
    assert locked.z_mid.shape == z.shape
    crop_neu = float(anima_fake_crop_code(locked.x_neu))
    crop_plus = float(anima_fake_crop_code(locked.x_plus))
    assert abs(crop_plus - crop_neu) < abs(
        float(anima_fake_crop_code(caption.x_plus)) - crop_neu
    )


def test_dummy_train_same_crop_runs(tmp_path):
    args = parse_args(
        [
            "--dummy",
            "--steps",
            "6",
            "--device",
            "cpu",
            "--name",
            "anima-same-crop",
            "--prompts_file",
            str(PROMPTS),
            "--save_dir",
            str(tmp_path),
            "--rank",
            "4",
            "--lm_target",
            "same_crop",
            "--teacher",
            "same_crop",
            "--teacher_strength",
            "0.5",
            "--traj_steps",
            "4",
        ]
    )
    sidecar = train_dummy(args)
    assert sidecar["lm_target"] == "same_crop"
    assert sidecar["teacher"] == "same_crop"
    assert sidecar["teacher_strength"] == 0.5
    assert "invert" in sidecar["recipe"]
    assert sidecar["loss_last"] is not None
    assert sidecar["music3_default_untouched"]["lm_target"] == "v9"
    assert sidecar["sample_grid"]["n"] == 12


def test_embed_struct_resolver_and_cli():
    assert resolve_anima_concept_target() == "caption"
    assert resolve_anima_concept_target("plus") == "caption"
    assert resolve_anima_concept_target("same_crop") == "same_crop"
    assert resolve_anima_concept_target("invert") == "same_crop"
    with pytest.raises(ValueError, match="concept_target"):
        resolve_anima_concept_target("v9")
    implied = resolve_anima_train_recipe("embed_struct")
    assert implied.loss_kind == "embed_struct"
    assert implied.concept_target == "caption"
    assert implied.teacher == "caption"
    blend = resolve_anima_train_recipe("embed_struct", "same_crop")
    assert blend.loss_kind == "embed_struct"
    assert blend.concept_target == "same_crop"
    explicit = resolve_anima_train_recipe(
        "embed_struct", "same_crop", concept_target="caption"
    )
    assert explicit.concept_target == "caption"
    alias = resolve_anima_train_recipe("conditioner_embed")
    assert alias.lm_target == "embed_struct"
    args = parse_args(["--lm_target", "embed_struct", "--concept_target", "caption"])
    assert resolve_anima_lm_target(args.lm_target) == "embed_struct"
    assert args.concept_target == "caption"
    smoke = embed_struct_smoke_command()
    assert "--lm_target embed_struct" in smoke
    assert "--lora_targets conditioner" in smoke
    assert "--teacher_strength" not in smoke
    assert "--steps 8" in smoke
    with pytest.raises(ValueError, match="conditioner"):
        anima_embed_struct_requires_conditioner(resolve_anima_lora_targets("dit"))
    anima_embed_struct_requires_conditioner(resolve_anima_lora_targets("conditioner"))


def test_embed_struct_loss_is_concept_plus_structure():
    plus = torch.tensor([[[2.0, 0.0], [0.0, 1.0]]])
    student = plus.clone()
    neu = torch.zeros_like(plus)
    zero = neu.clone()
    x_plus = torch.tensor([4.0, 0.0])
    x_neu = torch.tensor([0.0, 1.0])
    x_student = x_neu.clone()
    x_zero = x_neu.clone()
    base = anima_embed_struct_loss(
        student, plus, zero, neu, x_student, x_neu, x_zero
    )
    assert float(base) == pytest.approx(0.0, abs=1e-8)
    moved_e = anima_embed_struct_loss(
        student + 1, plus, zero, neu, x_student, x_neu, x_zero
    )
    assert float(moved_e) > float(base)
    moved_x = anima_embed_struct_loss(
        student, plus, zero, neu, x_student + 1, x_neu, x_zero
    )
    assert float(moved_x) > float(base)
    # Different caption lengths pool instead of crashing.
    short = torch.ones(1, 2, 3)
    long = torch.ones(1, 5, 3)
    assert float(anima_embed_mse(short, long)) == pytest.approx(0.0, abs=1e-8)
    # Teacher embeds must be stopgrad.
    e_s = torch.zeros(1, 2, 2, requires_grad=True)
    e_p = torch.ones(1, 2, 2, requires_grad=True)
    loss = anima_embed_struct_loss(e_s, e_p.detach())
    loss.backward()
    assert e_s.grad is not None
    assert e_p.grad is None


def test_embed_struct_moves_embeds_without_chasing_plus_crop():
    """Neu+adapter embeds chase plus; student traj crop stays near neu."""
    backend = FakeAnimaBackend(device="cpu", rank=8, seed=0, lora_targets="conditioner")
    neu, pos = "a woman", "a smiling woman"
    g = torch.Generator().manual_seed(0)
    z = torch.randn((1, *backend.latent_shape), generator=g)

    def _cos():
        e_s, _ = backend.encode_text(neu)
        with backend.disable_adapter():
            e_n, _ = backend.encode_text(neu)
            e_p, _ = backend.encode_text(pos)
        return float(anima_embed_delta_cosine(e_s, e_n, e_p))

    before = _cos()
    params = backend.trainable_parameters()
    opt = torch.optim.AdamW(params, lr=5e-2)
    for _ in range(40):
        e_student, _ = backend.encode_text(neu)
        with torch.no_grad(), backend.disable_adapter():
            e_plus, _ = backend.encode_text(pos)
            e_neu, _ = backend.encode_text(neu)
            x_neu = anima_short_trajectory(
                backend, neu, z, num_steps=4, frozen=True
            )
        x_student = anima_short_trajectory(
            backend, neu, z, num_steps=4, frozen=False, scale=1.0
        )
        loss = anima_embed_struct_loss(
            e_student,
            e_plus.detach(),
            None,
            e_neu.detach(),
            x_student,
            x_neu,
            embed_identity_weight=0.0,
            identity_weight=0.0,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    after = _cos()
    assert backend.lora_B_norm() > 0.0
    assert after > before
    with torch.no_grad():
        x_s = anima_short_trajectory(
            backend, neu, z, num_steps=4, frozen=False, scale=1.0
        )
        x_n = anima_short_trajectory(backend, neu, z, num_steps=4, frozen=True)
        x_p = anima_short_trajectory(backend, pos, z, num_steps=4, frozen=True)
    crop_student = anima_teacher_crop_gap(x_n, x_s)
    crop_plus = anima_teacher_crop_gap(x_n, x_p)
    assert crop_plus > 1e-4
    assert crop_student < crop_plus


def test_dummy_train_embed_struct_runs(tmp_path):
    args = parse_args(
        [
            "--dummy",
            "--steps",
            "6",
            "--device",
            "cpu",
            "--name",
            "anima-embed-struct",
            "--prompts_file",
            str(PROMPTS),
            "--save_dir",
            str(tmp_path),
            "--rank",
            "4",
            "--lm_target",
            "embed_struct",
            "--lora_targets",
            "conditioner",
            "--traj_steps",
            "4",
        ]
    )
    sidecar = train_dummy(args)
    assert sidecar["lm_target"] == "embed_struct"
    assert sidecar["concept_target"] == "caption"
    assert sidecar["teacher"] == "caption"
    assert sidecar["embed_weight"] == 1.0
    assert sidecar["struct_weight"] == 1.0
    assert "E_θ(neu)" in sidecar["embed_struct_loss"]
    assert "teacher_strength" in sidecar["why_general"]
    assert sidecar["loss_last"] is not None
    assert sidecar["music3_default_untouched"]["lm_target"] == "v9"
    assert sidecar["sample_grid"]["n"] == 12
    assert sidecar["train_text_conditioner"] is True


def test_dummy_train_embed_struct_rejects_dit_only(tmp_path):
    args = parse_args(
        [
            "--dummy",
            "--steps",
            "2",
            "--device",
            "cpu",
            "--name",
            "anima-embed-dit",
            "--prompts_file",
            str(PROMPTS),
            "--save_dir",
            str(tmp_path),
            "--rank",
            "4",
            "--lm_target",
            "embed_struct",
            "--lora_targets",
            "dit",
        ]
    )
    with pytest.raises(ValueError, match="conditioner"):
        train_dummy(args)


def test_turbo_is_preview_only_train_stays_base():
    card = turbo_preview_card()
    assert TURBO_PREVIEW_ONLY is True
    assert card["role"] == "preview_only"
    assert card["train_on"] == DEFAULT_MODEL_ID
    assert card["train_on"] == "circlestone-labs/Anima-Base-v1.0-Diffusers"
    assert card["comfy_repo"] == TURBO_COMFY_REPO
    assert card["transformer_file"] == TURBO_TRANSFORMER_FILE
    assert "anima-turbo-v1.1.safetensors" in card["transformer_file"]
    assert card["vae_class"] == "AutoencoderKLQwenImage"
    assert card["convert_splits"]["llm_adapter"] == "AnimaTextConditioner"
    assert card["convert_splits"]["rest"] == "CosmosTransformer3DModel"
    assert "--save_pipeline" in card["convert_flags"]
    assert "--dtype bf16" in card["convert_flags"]
    assert card["output"] == TURBO_DIFFUSERS_OUTPUT
    assert card["sample_cfg"] == TURBO_SAMPLE_CFG == 1.0
    assert card["sample_steps"] == TURBO_SAMPLE_STEPS == 10
    assert card["sample_steps_range"] == [8, 12]
    assert "Non-Commercial" in card["license"]
    assert card["license"] == TURBO_LICENSE
    assert "Anima-1.0-Turbo-Diffusers" in card["ignore_community"]
    assert "preview" in card["why"].lower()
    cmd = turbo_preview_sample_command()
    assert "--cfg 1" in cmd
    assert "--sample_steps 10" in cmd
    assert TURBO_DIFFUSERS_OUTPUT in cmd
    live = live_train_card()
    assert live["model_id"] == DEFAULT_MODEL_ID
    assert live["cfg"] == 4.0
    assert live["sample_steps"] == 40
    assert live["turbo"] == "preview_only"
    live_cmd = live_train_command()
    assert "circlestone-labs/Anima-Base-v1.0-Diffusers" in live_cmd
    assert "--cfg 4" in live_cmd
    assert "--sample_steps 40" in live_cmd


def test_print_turbo_preview_does_not_train():
    from conceptmod.textsliders.train_lora_anima import train

    out = train(parse_args(["--print_turbo_preview"]))
    assert out["role"] == "preview_only"
    assert out["train_on"] == DEFAULT_MODEL_ID
    assert parse_args([]).model_id == DEFAULT_MODEL_ID
    assert parse_args([]).cfg == 4.0
    assert parse_args([]).sample_steps == 40


def test_explicit_cfg_1_is_allowed_on_sample_path():
    """Turbo preview sample uses --cfg 1. That is an explicit request."""
    backend = FakeAnimaBackend(device="cpu", rank=4, seed=0)
    pipe = backend.pipe
    pipe.guider.config.guidance_scale = 4.0
    pipe.guider.guidance_scale = 4.0
    restore = apply_anima_guider_cfg(pipe, 1.0)
    assert pipe.guider.config.guidance_scale == 1.0
    restore()
    images = _call_modular_pipe(
        pipe,
        WOMAN_NEU,
        steps=2,
        height=32,
        width=32,
        cfg=1.0,
        seed=0,
        device=backend.device,
    )
    assert images
    assert pipe.last_guidance_scale == 1.0


def _load_turbo_convert():
    path = REPO / "scripts" / "convert_anima_turbo_diffusers.py"
    spec = importlib.util.spec_from_file_location("convert_anima_turbo_diffusers", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_convert_anima_turbo_helper_is_preview_only(tmp_path):
    convert = _load_turbo_convert()
    convert_recipe = convert.convert_recipe
    copy_tokenizers = convert.copy_tokenizers
    official_convert_argv = convert.official_convert_argv
    convert_parse = convert.parse_args
    write_preview_readme = convert.write_preview_readme
    resolve_tokenizer_dirs = convert.resolve_tokenizer_dirs

    args = convert_parse([])
    assert args.output.name == TURBO_DIFFUSERS_OUTPUT
    assert args.base_diffusers == DEFAULT_MODEL_ID
    assert args.dtype == "bf16"
    recipe = convert_recipe()
    assert recipe["role"] == "preview_only"
    assert recipe["train_on"] == DEFAULT_MODEL_ID
    assert recipe["turbo_is_not_a_train_target"] is True
    assert recipe["save_pipeline"] is True
    assert recipe["dtype"] == "bf16"
    names = [Path(row["filename"]).name for row in recipe["hub_files"]]
    assert names == [
        "anima-turbo-v1.1.safetensors",
        "qwen_3_06b_base.safetensors",
        "qwen_image_vae.safetensors",
    ]
    assert all(row["repo"] == "circlestone-labs/Anima" for row in recipe["hub_files"])
    argv = recipe["official_argv"]
    assert "--save_pipeline" in argv
    assert argv[argv.index("--dtype") + 1] == "bf16"
    assert "--cfg 1" in recipe["sample_command"]
    assert "--sample_steps 10" in recipe["sample_command"]
    built = official_convert_argv(
        convert_script=tmp_path / "convert_anima_to_diffusers.py",
        transformer=tmp_path / "anima-turbo-v1.1.safetensors",
        text_encoder=tmp_path / "qwen_3_06b_base.safetensors",
        vae=tmp_path / "qwen_image_vae.safetensors",
        qwen_tokenizer=tmp_path / "tokenizer",
        t5_tokenizer=tmp_path / "t5_tokenizer",
        output=tmp_path / TURBO_DIFFUSERS_OUTPUT,
    )
    assert "--save_pipeline" in built
    assert built[built.index("--dtype") + 1] == "bf16"
    readme = write_preview_readme(tmp_path / "out")
    text = readme.read_text()
    assert "preview only" in text.lower()
    assert "Do not train" in text
    assert DEFAULT_MODEL_ID in text
    assert "Non-Commercial" in text
    qwen = tmp_path / "src_tok"
    t5 = tmp_path / "src_t5"
    qwen.mkdir()
    t5.mkdir()
    (qwen / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (t5 / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    dest = tmp_path / "converted"
    dest.mkdir()
    copy_tokenizers(qwen, t5, dest)
    assert (dest / "tokenizer" / "tokenizer_config.json").is_file()
    assert (dest / "t5_tokenizer" / "tokenizer_config.json").is_file()
    local_base = tmp_path / "Anima-Base-v1.0-Diffusers"
    (local_base / "tokenizer").mkdir(parents=True)
    (local_base / "t5_tokenizer").mkdir(parents=True)
    (local_base / "tokenizer" / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (local_base / "t5_tokenizer" / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    got_qwen, got_t5 = resolve_tokenizer_dirs(str(local_base), tmp_path / "cache")
    assert got_qwen == local_base / "tokenizer"
    assert got_t5 == local_base / "t5_tokenizer"
    dry = convert.main(["--dry-run", "--output", str(tmp_path / "Anima-Turbo-v1.1-Diffusers")])
    assert dry == 0
    printed = convert.main(["--print-recipe"])
    assert printed == 0


def test_sync_peft_updates_stale_components_and_blocks():
    """Bare setattr leaves a captured components dict / block ref stale."""
    from types import SimpleNamespace

    from conceptmod.textsliders.anima_peft_sync import (
        assert_peft_modules_synced,
        sync_peft_into_modular_pipeline,
    )

    class _StalePipe:
        def __init__(self):
            self.transformer = torch.nn.Linear(2, 2)
            self.text_conditioner = torch.nn.Linear(3, 3)
            stale = torch.nn.Linear(3, 3)
            self.components = {
                "transformer": self.transformer,
                "text_conditioner": stale,
            }
            inner = SimpleNamespace(text_conditioner=stale, sub_blocks={})
            self._blocks = SimpleNamespace(
                text_conditioner=stale,
                sub_blocks={"text_conditioning": inner},
            )

    pipe = _StalePipe()
    peft = torch.nn.Linear(3, 3)
    stale_id = id(pipe.components["text_conditioner"])
    assert stale_id != id(peft)
    report = sync_peft_into_modular_pipeline(pipe, text_conditioner=peft)
    assert report["ok"] is True
    assert pipe.text_conditioner is peft
    assert pipe.components["text_conditioner"] is peft
    assert pipe._blocks.text_conditioner is peft
    assert pipe._blocks.sub_blocks["text_conditioning"].text_conditioner is peft
    assert_peft_modules_synced(pipe, text_conditioner=peft)


def test_after_attach_sample_and_encode_share_conditioner():
    backend = FakeAnimaBackend(device="cpu", rank=4, seed=0, lora_targets="conditioner")
    from conceptmod.textsliders.anima_peft_sync import (
        assert_sample_train_conditioner_shared,
        sample_conditioner_module,
        train_conditioner_module,
    )
    from conceptmod.textsliders.train_lora_anima import sync_backend_peft_modules

    sync_backend_peft_modules(backend)
    shared = assert_sample_train_conditioner_shared(backend)
    assert sample_conditioner_module(backend) is train_conditioner_module(backend)
    assert sample_conditioner_module(backend) is backend.pipe.text_conditioner
    assert sample_conditioner_module(backend) is backend.transformer.text_conditioner
    assert backend.pipe.components["text_conditioner"] is backend.pipe.text_conditioner
    assert shared["id"] == id(backend.pipe.text_conditioner)
    backend.pipe("a woman", height=8, width=8)
    assert backend.pipe.last_embeds is not None
    train_e, _ = backend.encode_text("a woman")
    assert torch.allclose(train_e, backend.pipe.last_embeds)


def test_scale_nonzero_changes_fake_conditioner_output():
    backend = FakeAnimaBackend(device="cpu", rank=4, seed=0, lora_targets="conditioner")
    for lora in backend.loras:
        with torch.no_grad():
            lora.up.weight.add_(0.4)
            lora.down.weight.add_(0.2)
    neu = "a woman sitting on a chair"
    backend.set_lora_scale(0.0)
    e0, _ = backend.encode_text(neu)
    pipe0 = backend.pipe.encode_prompt_embeds(neu)
    backend.set_lora_scale(1.0)
    e1, _ = backend.encode_text(neu)
    pipe1 = backend.pipe.encode_prompt_embeds(neu)
    assert not torch.allclose(e0, e1)
    assert not torch.allclose(pipe0, pipe1)
    assert torch.allclose(e1, pipe1)


def _load_embed_diag():
    path = REPO / "scripts" / "diag_anima_conditioner_embed.py"
    spec = importlib.util.spec_from_file_location("diag_anima_conditioner_embed", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_embed_diag_pass_fail_on_dummy():
    from conceptmod.textsliders.anima_peft_sync import (
        SAMPLE_TRAIN_MISMATCH,
        compare_sample_train_embeds,
        measure_conditioner_embed_deltas,
    )
    from types import SimpleNamespace

    args = SimpleNamespace(rank=4, seed=0, neu=WOMAN_NEU, plus=WOMAN_PLUS)
    report = _load_embed_diag().run_dummy(args)
    assert report["verdict"] == "PASS"
    assert report["mismatch_flag"] is None
    assert report["shared_conditioner_id"] == report["train_conditioner_id"]

    backend = FakeAnimaBackend(device="cpu", rank=4, seed=1, lora_targets="conditioner")

    def _fresh(prompt, scale):
        del scale
        embeds, _ = backend.encode_text(prompt)
        return embeds

    fresh = measure_conditioner_embed_deltas(_fresh, neu=WOMAN_NEU, plus=WOMAN_PLUS)
    assert fresh["verdict"] == "FAIL"
    assert "near-zero" in fresh["reason"]
    mismatch = compare_sample_train_embeds(
        torch.zeros(1, 4), torch.ones(1, 4)
    )
    assert mismatch["flag"] == SAMPLE_TRAIN_MISMATCH


def test_sample_mode_train_faithful_uses_encode_text(tmp_path):
    rows, _meta = load_anima_prompts(PROMPTS)
    backend = FakeAnimaBackend(device="cpu", rank=4, seed=0, lora_targets="conditioner")
    args = parse_args(
        [
            "--dummy",
            "--device",
            "cpu",
            "--save_dir",
            str(tmp_path),
            "--sample_mode",
            "train_faithful",
            "--resolution",
            "32",
        ]
    )
    assert args.sample_mode == "train_faithful"
    records = emit_inprocess_samples(
        backend, args, tmp_path, step=1, rows=rows, dummy=True
    )
    assert all(row["method"] == "train_faithful_encode_text" for row in records)
    assert all(row["sample_mode"] == "train_faithful" for row in records)
    assert getattr(backend.pipe, "last_train_embeds", None) is not None


def test_peft_modules_skip_transformer_when_train_dit_false():
    from conceptmod.textsliders.train_lora_anima import _peft_modules

    backend = FakeAnimaBackend(device="cpu", rank=4, seed=0, lora_targets="conditioner")
    # Dummy exposes backend-level scale APIs, so _peft_modules returns
    # [backend] — not the transformer. Live conditioner-only uses only
    # pipe.text_conditioner (see _LiveCond).
    modules = _peft_modules(backend)
    assert modules == [backend]
    assert backend.lora_spec.train_dit is False

    class _LiveCond:
        lora_spec = resolve_anima_lora_targets("conditioner")

        def __init__(self):
            self.pipe = type(
                "P",
                (),
                {
                    "transformer": object(),
                    "text_conditioner": object(),
                },
            )()

    live = _LiveCond()
    live_modules = _peft_modules(live)
    assert live_modules == [live.pipe.text_conditioner]
    assert live.pipe.transformer not in live_modules


def test_sample_train_mismatch_when_pipe_conditioner_stale():
    from conceptmod.textsliders.anima_peft_sync import (
        SAMPLE_TRAIN_MISMATCH,
        compare_sample_train_embeds,
    )

    backend = FakeAnimaBackend(device="cpu", rank=4, seed=0, lora_targets="conditioner")
    for lora in backend.loras:
        with torch.no_grad():
            lora.up.weight.add_(0.5)
    stale = type(backend.pipe.text_conditioner)(8)
    backend.pipe.text_conditioner = stale
    train_e, _ = backend.encode_text("a smiling woman")
    sample_e = backend.pipe.encode_prompt_embeds("a smiling woman")
    cmp = compare_sample_train_embeds(train_e, sample_e)
    assert cmp["match"] is False
    assert cmp["flag"] == SAMPLE_TRAIN_MISMATCH
