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
    DEFAULT_CONCEPT_WORDS,
    DEFAULT_CONTROL_PROMPT,
    DEFAULT_LM_TARGET,
    DEFAULT_LR,
    DEFAULT_MODEL_ID,
    DEFAULT_RANK,
    DEFAULT_RESOLUTION,
    DEFAULT_SAMPLE_EVERY,
    DEFAULT_SAMPLE_SCALES,
    DEFAULT_SAMPLE_SEED,
    DEFAULT_SAMPLE_STEPS,
    LORA_TARGETS,
    MAN_NEU,
    MAN_PLUS,
    WOMAN_NEU,
    WOMAN_PLUS,
    AnimaSampleGateError,
    align_unused_positions,
    anima_cfg_delta,
    anima_direct_loss,
    anima_direct_teachers,
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
    parse_concept_words,
    resolve_anima_lm_target,
    row_token_plan,
    splice_unused_embeds,
    stock_teacher_smoke_captions,
    unused_token_mask,
    unused_vocab,
    word_tokens,
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
    assert args.lm_target == DEFAULT_LM_TARGET == "direct"
    assert args.control_prompt == DEFAULT_CONTROL_PROMPT
    assert args.sample_every == DEFAULT_SAMPLE_EVERY == 100
    assert args.sample_seed == DEFAULT_SAMPLE_SEED == 42
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
    assert card["lr"] == 1e-4
    assert card["control_prompt"] == "a bowl of fruit on a table"
    assert card["sample_scales"] == list(DEFAULT_SAMPLE_SCALES)
    assert card["sample_seed"] == 42
    assert card["lm_target"] == "direct"
    assert card["sample_every"] == 100
    assert "PEFT" in card["sample_gate"]
    assert card["stock_teacher_smoke"]["woman"]["neu"] == WOMAN_NEU
    assert card["stock_teacher_smoke"]["woman"]["plus"] == WOMAN_PLUS
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
    assert "--lr 0.0001" in cmd
    assert "--lm_target direct" in cmd
    assert "--sample_every 100" in cmd
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
    assert sidecar["lm_target"] == "direct"
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
    assert out["lr"] == 1e-4


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


def test_anima_lm_target_default_direct_rejects_unknown():
    assert resolve_anima_lm_target() == "direct"
    assert resolve_anima_lm_target("cfg_delta") == "cfg_delta"
    with pytest.raises(ValueError, match="lm_target"):
        resolve_anima_lm_target("v9")
    args = parse_args(["--lm_target", "cfg_delta"])
    assert args.lm_target == "cfg_delta"


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
