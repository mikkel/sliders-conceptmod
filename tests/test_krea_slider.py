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
    KREA_CONTROL_PROMPT,
    KREA_DEFAULT_LORA_TARGETS,
    KREA_DEFAULT_RANK,
    KREA_DEFAULT_RESOLUTION,
    KREA_DUMMY_EMBED_LAYERS,
    KREA_DUMMY_EMBED_SEQ,
    KREA_EMBED_COSINE_WEIGHT,
    KREA_EMBED_EARLY_LAYERS,
    KREA_EMBED_EARLY_WEIGHT,
    KREA_EMBED_LATE_LAYER_START,
    KREA_EMBED_LATE_WEIGHT,
    KREA_EMBED_MID_WEIGHT,
    KREA_EMBED_REL_L2_WEIGHT,
    KREA_EMBED_SAMPLE_CFG,
    KREA_HOLD_WEIGHT,
    KREA_LM_TARGET_DEFAULT,
    KREA_LORA_TARGETS,
    KREA_LORA_TARGET_CHOICES,
    KREA_ORACLE_EMBED_COS,
    KREA_ORACLE_MASK_AB_SHOTS,
    KREA_ORACLE_SHOTS,
    KREA_TE_DIT_MASK_DEFAULT,
    KREA_RAW_CFG,
    KREA_RAW_MODEL,
    KREA_RAW_STEPS,
    KREA_RECIPE_DEFAULT,
    KREA_SAMPLE_SCALES,
    KREA_SMILE_HOLD_WEIGHT,
    KREA_TURBO_CFG,
    KREA_TURBO_STEPS,
    KREA_TE_LORA_TARGETS,
    apply_continuous_lora_scale,
    expand_attributes_krea,
    force_krea_embed_lora_targets,
    krea_cfg_compose,
    krea_cfg_direction,
    krea_cfg_uncond_te_frozen,
    krea_ones_attention_mask,
    krea_resolve_dit_encoder_mask,
    krea_te_sample_use_ones_mask,
    krea_concept_words,
    krea_embed_as_stacked,
    krea_embed_cosine,
    krea_embed_layer_weights,
    krea_embed_max_abs,
    krea_embed_mse,
    krea_embed_rel_l2,
    krea_embed_requires_te,
    krea_embed_train_stats,
    krea_embed_uni_loss,
    krea_hold_unused_embeds,
    krea_looks_turbo,
    krea_minus_canary,
    krea_plus_neu_loss,
    krea_plus_neu_teachers,
    krea_sample_card,
    krea_token_rows,
    krea_unused_hold_loss,
    krea_unused_hold_mask,
    krea_word_tokens,
    resolve_krea_lm_target,
    resolve_krea_lora_targets,
    resolve_krea_sample_guidance,
)
from conceptmod.textsliders.train_lora_krea import (
    DummyKreaBackend,
    DummyKreaTE,
    assert_krea_only,
    infer_oracle_pairs,
    infer_sample_prompts,
    krea_embed_step_loss,
    krea_step_loss,
    load_prompts,
    parse_args,
    resolve_krea_card,
    train,
    unused_words_for,
)
from conceptmod.textsliders.train_lm_slider_music3 import parse_args as parse_lm


ROOT = Path(__file__).resolve().parents[1]
KREA_YAML = ROOT / "conceptmod/textsliders/data/prompts-krea.yaml"
KREA_HAPPY_YAML = ROOT / "conceptmod/textsliders/data/prompts-krea-happy.yaml"
KREA_DETAILED_YAML = ROOT / "conceptmod/textsliders/data/prompts-krea-detailed.yaml"
KREA_TRAINER = ROOT / "conceptmod/textsliders/train_lora_krea.py"
KREA_LIVE = ROOT / "conceptmod/textsliders/krea_live.py"


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
    assert bare.allow_hub is False
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
    assert meta.bare_captions is False
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


def test_happy_yaml_is_bare_smile_not_age():
    prompts, meta = load_prompts(KREA_HAPPY_YAML)
    assert meta.plus_label == "Happy"
    assert meta.minus_label == "Sad"
    assert meta.bare_captions is True
    assert meta.control_prompt == KREA_CONTROL_PROMPT
    assert "smile" in meta.concept_words
    assert "teeth" in meta.concept_words
    assert "happy" in meta.concept_words
    assert "old" not in meta.concept_words
    assert len(prompts) == 2  # bare: one copy per row, no gender prefix
    for prompt in prompts:
        assert prompt.positive
        assert prompt.neutral
        assert not prompt.positive.startswith("male ")
        assert not prompt.positive.startswith("female ")
        assert "teeth" in prompt.positive
        assert "joyful" in prompt.positive
        assert "closed mouth" in prompt.neutral
        assert "smile" not in prompt.neutral
        unused = unused_words_for(prompt)
        assert "male" in unused and "female" in unused
        concept = krea_concept_words(prompt.positive, prompt.neutral)
        assert "smile" in concept or "teeth" in concept
        assert "old" not in concept
        assert prompt.negative  # canary only
    sample = infer_sample_prompts(prompts, meta.control_prompt)
    assert meta.control_prompt in sample
    assert all("teeth" not in cap for cap in sample if cap != meta.control_prompt)


def test_detailed_yaml_is_bare_not_smile():
    prompts, meta = load_prompts(KREA_DETAILED_YAML)
    assert meta.plus_label == "Detailed"
    assert meta.minus_label == "Simple"
    assert meta.bare_captions is True
    assert meta.recommended_range == [0.0, 2.0]
    assert meta.control_prompt == KREA_CONTROL_PROMPT
    assert "detailed" in meta.concept_words
    assert "intricate" in meta.concept_words
    assert "texture" in meta.concept_words
    assert "smile" not in meta.concept_words
    assert "teeth" not in meta.concept_words
    assert len(prompts) == 3  # bare: one copy per scene row
    for prompt in prompts:
        assert prompt.positive
        assert prompt.neutral
        assert not prompt.positive.startswith("male ")
        assert not prompt.positive.startswith("female ")
        assert "detailed" in prompt.positive
        assert "intricate" in prompt.positive
        assert "simple" in prompt.neutral
        assert "flat" in prompt.neutral
        assert "detailed" not in prompt.neutral
        assert "intricate" not in prompt.neutral
        unused = unused_words_for(prompt)
        assert "male" in unused and "female" in unused
        concept = krea_concept_words(prompt.positive, prompt.neutral)
        assert "detailed" in concept or "intricate" in concept
        assert "smile" not in concept
        assert prompt.negative  # canary only
        assert "washed-out" in prompt.negative or "featureless" in prompt.negative
    sample = infer_sample_prompts(prompts, meta.control_prompt)
    assert meta.control_prompt in sample
    assert all(
        "detailed" not in cap for cap in sample if cap != meta.control_prompt
    )


def test_bare_captions_do_not_prefix_attributes():
    row = {
        "target": "a person, closed mouth",
        "positive": "a person, big smile showing teeth",
        "neutral": "a person, closed mouth",
        "negative": "a sad person",
        "attributes": ["male", "female"],
    }
    prefixed = expand_attributes_krea(row, prefix=True)
    assert len(prefixed) == 2
    assert prefixed[0]["positive"].startswith("male ")
    bare = expand_attributes_krea(row, prefix=False)
    assert len(bare) == 1
    assert bare[0]["positive"] == row["positive"]
    assert bare[0]["attributes"] == ["male", "female"]


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
    assert payload["allow_hub"] is False
    assert payload["encoder_lora"] is False
    assert payload["dit_lora_only"] is True
    assert payload["lora_targets"] == "dit"
    assert payload["dit_lora"] is True
    assert payload["te_lora"] is False
    assert payload["dit_lora_path"] == "krea-age-dummy_lora"
    assert payload["te_lora_path"] is None
    assert payload["hold_weight"] == KREA_HOLD_WEIGHT
    assert payload["te_parking"] is True
    assert payload["sample_grid"]["gate"] == "smile-first"
    assert payload["sample_grid"]["crop_purity"] is False
    assert payload["sample_grid"]["scales"] == list(KREA_SAMPLE_SCALES)
    log = tmp_path / "krea-age-dummy_train.jsonl"
    assert log.exists()
    lines = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
    samples = tmp_path / "samples"
    pngs = list(samples.glob("*.png"))
    assert pngs
    meta = json.loads((samples / "final_meta.json").read_text(encoding="utf-8"))
    assert meta["gate"] == "smile-first"
    assert "a bowl of fruit on a table" in meta["prompts"]


def test_dummy_happy_train_writes_smile_grid(tmp_path: Path):
    args = parse_args(
        [
            "--dummy",
            "--name",
            "smile-krea-dummy",
            "--prompts_file",
            str(KREA_HAPPY_YAML),
            "--save_dir",
            str(tmp_path),
            "--steps",
            "8",
            "--seed",
            "7",
        ]
    )
    sidecar = train(args)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["plus_label"] == "Happy"
    assert payload["bare_captions"] is True
    assert payload["control_prompt"] == KREA_CONTROL_PROMPT
    assert "teeth" in payload["concept_words"]
    pngs = list((tmp_path / "samples").glob("*.png"))
    # 2 neu captions + fruit × 4 scales
    assert len(pngs) == 12
    names = " ".join(p.name for p in pngs)
    assert "fruit" in names
    assert "scale0" in names
    assert "scale1" in names


def test_dummy_does_not_import_krea_live(tmp_path: Path):
    import sys

    sys.modules.pop("conceptmod.textsliders.krea_live", None)
    sys.modules.pop("conceptmod.textsliders.krea_weights", None)
    args = parse_args(
        [
            "--dummy",
            "--name",
            "krea-no-live",
            "--prompts_file",
            str(KREA_HAPPY_YAML),
            "--save_dir",
            str(tmp_path),
        ]
    )
    train(args)
    assert "conceptmod.textsliders.krea_live" not in sys.modules
    assert "conceptmod.textsliders.krea_weights" not in sys.modules


def test_live_loader_does_not_run_in_dummy():
    src = KREA_TRAINER.read_text(encoding="utf-8")
    assert "Krea2Pipeline.from_pretrained" not in src
    assert "hf_hub_download" not in src
    assert "_load_live_backend" in src
    assert "CI uses --dummy" in src
    assert "allow_hub" in src
    live = KREA_LIVE.read_text(encoding="utf-8")
    assert "Krea2Pipeline.from_pretrained" in live
    assert "to_q" in live
    assert "q_proj" in live
    assert "park" in live.lower()
    assert "encoder_lora" in live
    assert "lora_targets" in live
    assert "apply_continuous_lora_scale" in live
    assert "allow_hub" in live
    assert "same_crop" not in live
    assert "embed_struct" not in live
    assert "frozen TE when encoder_lora" in live
    assert "reuse hidden states" in live
    assert "load_te_adapter" in live
    assert "ones" in live and "smile slots" in live


def test_hold_math_token_aligns_4d_krea_embeds():
    pos_tokens = ["male", "old", "person"]
    neu_tokens = ["male", "person"]
    pos = torch.zeros(1, 3, 2, 4)
    neu = torch.zeros(1, 2, 2, 4)
    pos[0, 0] = 1.0
    pos[0, 1] = 2.0
    pos[0, 2] = 3.0
    neu[0, 0] = 9.0
    neu[0, 1] = 8.0
    mask = krea_unused_hold_mask(pos_tokens, neu_tokens, unused_words=["male"])
    held = krea_hold_unused_embeds(pos, neu, pos_tokens, neu_tokens, mask)
    rows = krea_token_rows(held)
    assert rows.shape[0] == 3
    assert torch.allclose(held[0, 0], neu[0, 0])
    assert torch.allclose(held[0, 1], pos[0, 1])
    assert torch.allclose(held[0, 2], neu[0, 1])


class _FakePeftLoraLinear(torch.nn.Module):
    """Duck-typed PEFT LoraLayer: ``scaling`` multiplies the delta."""

    def __init__(self, dim: int = 4, rank: int = 2):
        super().__init__()
        self.base = torch.nn.Linear(dim, dim, bias=False)
        self.lora_A = torch.nn.ModuleDict(
            {"default": torch.nn.Linear(dim, rank, bias=False)}
        )
        self.lora_B = torch.nn.ModuleDict(
            {"default": torch.nn.Linear(rank, dim, bias=False)}
        )
        self.scaling = {"default": 1.0}
        torch.nn.init.eye_(self.base.weight)
        torch.nn.init.ones_(self.lora_A["default"].weight)
        torch.nn.init.ones_(self.lora_B["default"].weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.lora_B["default"](self.lora_A["default"](x))
        return self.base(x) + delta * self.scaling["default"]


class _NoOpSetAdapterScale(_FakePeftLoraLinear):
    """PEFT-shaped wrapper whose ``set_adapter_scale`` does nothing."""

    def set_adapter_scale(self, *args, **kwargs):
        return None


def test_krea_lora_targets_resolver_and_cli():
    spec = resolve_krea_lora_targets()
    assert spec.label == KREA_DEFAULT_LORA_TARGETS == "dit"
    assert spec.train_dit is True
    assert spec.train_te is False
    assert spec.dit_lora_only is True
    assert spec.te_parking is True
    assert spec.dit_lora_targets == list(KREA_LORA_TARGETS)
    assert spec.te_lora_targets == []
    te = resolve_krea_lora_targets("text_encoder")
    assert te.label == "te"
    assert te.train_te is True
    assert te.train_dit is False
    assert te.te_lora_targets == list(KREA_TE_LORA_TARGETS)
    assert te.frozen_modules == ("transformer",)
    joint = resolve_krea_lora_targets("dit+te")
    assert joint.train_dit is True
    assert joint.train_te is True
    assert joint.adapted_module_names == ["transformer", "text_encoder"]
    assert resolve_krea_lora_targets("dit+text_encoder").label == "dit+te"
    with pytest.raises(ValueError, match="lora_targets"):
        resolve_krea_lora_targets("conditioner")
    args = parse_args(["--lora_targets", "dit+te", "--hold_weight", "0.1"])
    assert args.lora_targets == "dit+te"
    assert args.hold_weight == pytest.approx(KREA_SMILE_HOLD_WEIGHT)
    bare = parse_args([])
    assert bare.lora_targets == "dit"
    assert bare.hold_weight == KREA_HOLD_WEIGHT
    assert KREA_LORA_TARGET_CHOICES == ("dit", "te", "dit+te")


def test_continuous_lora_scale_changes_nonzero_mock_output():
    """Mid-scales must move the LoRA delta — not just 0 vs 1.

    Live smile-krea grids at 0.25 / 0.5 / 1.0 were byte-identical because
    PEFT ``set_adapter_scale`` no-op'd. This writes ``scaling`` the way
    dummy ``dit.scale`` already multiplies the delta.
    """
    layer = _NoOpSetAdapterScale()
    x = torch.ones(2, 4)
    layer.set_adapter_scale(0.25)
    noop_a = layer(x).detach().clone()
    layer.set_adapter_scale(0.5)
    noop_b = layer(x).detach().clone()
    assert torch.allclose(noop_a, noop_b)

    outs = {}
    for scale in (0.25, 0.5, 1.0):
        n = apply_continuous_lora_scale(layer, scale)
        assert n >= 1
        outs[scale] = layer(x).detach().clone()
    assert not torch.allclose(outs[0.25], outs[0.5], atol=1e-6)
    assert not torch.allclose(outs[0.5], outs[1.0], atol=1e-6)
    apply_continuous_lora_scale(layer, 0.0)
    out0 = layer(x)
    apply_continuous_lora_scale(layer, 1.0)
    # Linear in the delta: 0.5 is halfway between 0 and 1.
    mid = out0 + 0.5 * (outs[1.0] - out0)
    assert torch.allclose(outs[0.5], mid, atol=1e-5)


def test_dummy_nonzero_scales_change_velocity():
    backend = DummyKreaBackend(dim=8, rank=2, seed=0)
    with torch.no_grad():
        backend.dit.lora_up.weight.fill_(0.35)
    z = torch.ones(1, 8)
    v0 = backend.predict_v("a person", z, scale=0.0)
    v25 = backend.predict_v("a person", z, scale=0.25)
    v50 = backend.predict_v("a person", z, scale=0.5)
    v100 = backend.predict_v("a person", z, scale=1.0)
    assert not torch.allclose(v0, v25, atol=1e-6)
    assert not torch.allclose(v25, v50, atol=1e-6)
    assert not torch.allclose(v50, v100, atol=1e-6)
    expected_mid = v0 + 0.5 * (v100 - v0)
    assert torch.allclose(v50, expected_mid, atol=1e-5)


def test_dummy_te_lora_is_trainable_and_scaled():
    backend = DummyKreaBackend(dim=8, rank=2, seed=1, lora_targets="dit+te")
    assert backend.encoder_lora is True
    assert isinstance(backend.te, DummyKreaTE)
    names = " ".join(n for n, _p in backend.te.named_parameters() if _p.requires_grad)
    assert "lora_down" in names
    assert all(t in dict(backend.te.named_modules()) for t in KREA_TE_LORA_TARGETS)
    with torch.no_grad():
        backend.te.lora_up.weight.fill_(0.4)
    z = torch.ones(1, 8)
    v25 = backend.predict_v("a person", z, scale=0.25)
    v50 = backend.predict_v("a person", z, scale=0.5)
    assert not torch.allclose(v25, v50, atol=1e-6)
    backend.set_adapter_scale(1.0)
    pos, _ = backend.encode_text("an old person")
    neu, _ = backend.encode_text("a person", frozen=True)
    assert pos.requires_grad
    assert not neu.requires_grad or neu.grad_fn is None


def test_dummy_te_hold_has_grad(tmp_path: Path):
    backend = DummyKreaBackend(dim=8, rank=2, seed=2, lora_targets="te")
    with torch.no_grad():
        backend.te.lora_up.weight.fill_(0.2)
    from conceptmod.textsliders.train_lora_krea import KreaSliderPrompt

    prompt = KreaSliderPrompt(
        target="a person",
        positive="an old person",
        neutral="a person",
        negative="a young person",
        attributes=["male"],
    )
    z = torch.zeros(1, 8)
    loss, stats = krea_step_loss(backend, prompt, z, guidance=0.0, hold_weight=0.1)
    assert torch.isfinite(loss)
    loss.backward()
    te_grads = [p.grad for p in backend.te.lora_down.parameters() if p.grad is not None]
    assert te_grads
    dit_grads = [p.grad for p in backend.dit.lora_down.parameters() if p.grad is not None]
    assert not dit_grads


def test_dummy_smile_v2_dit_plus_te(tmp_path: Path):
    args = parse_args(
        [
            "--dummy",
            "--name",
            "smile-krea-v2-dummy",
            "--prompts_file",
            str(KREA_HAPPY_YAML),
            "--save_dir",
            str(tmp_path),
            "--lora_targets",
            "dit+te",
            "--hold_weight",
            "0.1",
            "--steps",
            "8",
            "--seed",
            "7",
        ]
    )
    sidecar = train(args)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["lora_targets"] == "dit+te"
    assert payload["dit_lora"] is True
    assert payload["te_lora"] is True
    assert payload["encoder_lora"] is True
    assert payload["dit_lora_only"] is False
    assert payload["te_parking"] is False
    assert payload["hold_weight"] == pytest.approx(KREA_SMILE_HOLD_WEIGHT)
    assert payload["dit_lora_path"] == "smile-krea-v2-dummy_lora/dit_lora"
    assert payload["te_lora_path"] == "smile-krea-v2-dummy_lora/te_lora"
    assert payload["te_lora_targets"] == list(KREA_TE_LORA_TARGETS)
    assert payload["plus_label"] == "Happy"
    pngs = list((tmp_path / "samples").glob("*.png"))
    assert len(pngs) == 12


def test_docs_list_smile_krea_v2_card():
    docs = (ROOT / "docs/krea-slider.md").read_text(encoding="utf-8")
    assert "--name smile-krea-v2" in docs
    assert "--lora_targets dit+te" in docs
    assert "--hold_weight 0.1" in docs
    assert "--allow_hub" in docs
    assert "--steps 500" in docs
    assert "same_crop" in docs and "embed_struct" in docs
    trainer = KREA_TRAINER.read_text(encoding="utf-8")
    assert "--lora_targets" in trainer
    assert "smile-krea-v2" in trainer or "hold_weight 0.1" in trainer


def test_docs_list_smile_krea_v3_embed_card():
    docs = (ROOT / "docs/krea-slider.md").read_text(encoding="utf-8")
    assert "--name smile-krea-v3" in docs
    assert "--lora_targets te --lm_target embed" in docs
    assert "cos≈0.9999" in docs
    assert "cos≈0.67" in docs
    assert "[1, 512, 12, 2560]" in docs or "[1,512,12,2560]" in docs
    assert "--recipe embed_uni" in docs
    assert "same_crop" in docs and "embed_struct" in docs
    trainer = KREA_TRAINER.read_text(encoding="utf-8")
    assert "--lm_target" in trainer
    assert "embed" in trainer
    live = KREA_LIVE.read_text(encoding="utf-8")
    assert "lm_target embed" in live
    assert "structure lock" in live


def test_lm_target_resolver_and_cli():
    assert resolve_krea_lm_target() == KREA_LM_TARGET_DEFAULT == "v"
    assert resolve_krea_lm_target("velocity") == "v"
    assert resolve_krea_lm_target("embed") == "embed"
    assert resolve_krea_lm_target("embed_uni") == "embed"
    assert resolve_krea_lm_target(recipe="embed_uni") == "embed"
    assert resolve_krea_lm_target("embed", recipe="uni") == "embed"
    # argparse default ``v`` + explicit ``--recipe embed_uni`` → embed
    assert resolve_krea_lm_target("v", recipe="embed_uni") == "embed"
    with pytest.raises(ValueError, match="lm_target"):
        resolve_krea_lm_target("v9")
    bare = parse_args([])
    assert bare.lm_target == "v"
    assert bare.recipe == KREA_RECIPE_DEFAULT == "uni"
    embed = parse_args(["--lm_target", "embed", "--lora_targets", "te"])
    assert embed.lm_target == "embed"
    assert embed.embed_cosine_weight == pytest.approx(KREA_EMBED_COSINE_WEIGHT)
    assert embed.embed_cosine_weight == pytest.approx(0.0)
    assert embed.embed_rel_l2_weight == pytest.approx(KREA_EMBED_REL_L2_WEIGHT)
    assert embed.embed_rel_l2_weight == pytest.approx(1.0)
    assert embed.embed_late_weight == pytest.approx(KREA_EMBED_LATE_WEIGHT)
    assert embed.embed_late_weight == pytest.approx(2.0)
    assert embed.embed_late_layer_start == KREA_EMBED_LATE_LAYER_START == 6
    bare_flags = parse_args([])
    assert bare_flags.embed_cosine_weight == pytest.approx(0.0)
    assert bare_flags.embed_rel_l2_weight == pytest.approx(1.0)
    assert bare_flags.embed_late_weight == pytest.approx(2.0)
    assert bare_flags.embed_late_layer_start == 6
    detail_v2 = parse_args(
        [
            "--lm_target",
            "embed",
            "--lora_targets",
            "te",
            "--embed_rel_l2_weight",
            "2.0",
            "--embed_late_weight",
            "4.0",
        ]
    )
    assert detail_v2.embed_late_weight == pytest.approx(4.0)
    assert detail_v2.embed_rel_l2_weight == pytest.approx(2.0)
    assert detail_v2.embed_late_layer_start == 6
    alias = parse_args(["--recipe", "embed_uni"])
    assert resolve_krea_lm_target(alias.lm_target, alias.recipe) == "embed"
    forced = force_krea_embed_lora_targets("dit+te", lm_target="embed")
    assert forced.label == "te"
    assert forced.train_dit is False
    assert forced.train_te is True
    krea_embed_requires_te(forced)
    with pytest.raises(ValueError, match="TE-only"):
        krea_embed_requires_te(resolve_krea_lora_targets("dit"))
    with pytest.raises(ValueError, match="TE-only"):
        krea_embed_requires_te(resolve_krea_lora_targets("dit+te"))


def test_embed_uni_loss_mse_and_cosine():
    pred = torch.zeros(1, 4, 4, 8)
    tgt = torch.zeros(1, 4, 4, 8)
    pred[..., 2:, :] = 1.0
    tgt[..., 2:, :] = 1.0
    assert float(krea_embed_mse(pred, tgt)) == pytest.approx(0.0, abs=1e-8)
    assert float(krea_embed_uni_loss(pred, tgt, cosine_weight=0.0)) == pytest.approx(
        0.0, abs=1e-8
    )
    miss = pred.clone()
    miss[..., 2:, :] = 0.0
    early_miss = pred.clone()
    early_miss[..., :2, :] = 1.0
    late_loss = float(krea_embed_mse(miss, tgt))
    early_loss = float(krea_embed_mse(early_miss, tgt))
    assert late_loss > early_loss
    weights = krea_embed_layer_weights(4)
    assert float(weights[0]) < float(weights[2])
    assert int(weights.numel()) == 4
    stacked = krea_embed_as_stacked(torch.ones(5, 8))
    assert tuple(stacked.shape) == (1, 5, 1, 8)
    combo = krea_embed_uni_loss(miss, tgt, cosine_weight=1.0)
    mse_only = krea_embed_uni_loss(miss, tgt, cosine_weight=0.0)
    assert float(combo) > float(mse_only)
    ones = torch.ones(1, 4, 4, 8)
    assert float(krea_embed_cosine(ones, ones)) == pytest.approx(1.0, abs=1e-5)
    assert KREA_EMBED_COSINE_WEIGHT == pytest.approx(0.0)
    assert KREA_EMBED_REL_L2_WEIGHT == pytest.approx(1.0)


def test_embed_uni_loss_high_cos_wrong_magnitude_stays_high():
    """Default loss (MSE + rel-L2, cosine 0) catches scale drift."""
    tgt = torch.ones(1, 4, 12, 8) * 10.0
    pred = tgt * 0.01
    cos = float(krea_embed_cosine(pred, tgt))
    assert cos > 0.99
    assert float(1.0 - krea_embed_cosine(pred, tgt)) < 0.01
    default = float(krea_embed_uni_loss(pred, tgt))
    mse_only = float(krea_embed_uni_loss(pred, tgt, cosine_weight=0.0, rel_l2_weight=0.0))
    rel = float(krea_embed_rel_l2(pred, tgt))
    cos_term = float(
        krea_embed_uni_loss(pred, tgt, cosine_weight=1.0, rel_l2_weight=0.0)
    )
    assert default > 0.5
    assert rel > 0.5
    assert default > mse_only
    assert abs(cos_term - mse_only) < 1e-4
    assert float(krea_embed_max_abs(pred, tgt)) == pytest.approx(9.9, abs=1e-5)
    assert float(krea_embed_uni_loss(tgt, tgt)) == pytest.approx(0.0, abs=1e-8)
    logged = krea_embed_train_stats(pred, tgt)
    assert logged["embed_max_abs"] == pytest.approx(9.9, abs=1e-5)
    assert logged["embed_late_l2"] > 0.0
    assert "embed_l2_l6" in logged
    assert "embed_l2_l11" in logged


def test_embed_late_layers_weigh_more_than_mid_and_early():
    weights = krea_embed_layer_weights(12)
    assert int(weights.numel()) == 12
    assert float(weights[0]) == pytest.approx(KREA_EMBED_EARLY_WEIGHT)
    assert float(weights[1]) == pytest.approx(KREA_EMBED_EARLY_WEIGHT)
    assert float(weights[2]) == pytest.approx(KREA_EMBED_MID_WEIGHT)
    assert float(weights[5]) == pytest.approx(KREA_EMBED_MID_WEIGHT)
    assert float(weights[6]) == pytest.approx(KREA_EMBED_LATE_WEIGHT)
    assert float(weights[11]) == pytest.approx(KREA_EMBED_LATE_WEIGHT)
    assert KREA_EMBED_LATE_LAYER_START == 6
    assert float(weights[0]) < float(weights[2]) < float(weights[6])

    tgt = torch.ones(1, 2, 12, 4)

    def miss_layer(index: int) -> torch.Tensor:
        pred = tgt.clone()
        pred[:, :, index, :] = 0.0
        return pred

    late = float(krea_embed_mse(miss_layer(8), tgt))
    mid = float(krea_embed_mse(miss_layer(4), tgt))
    early = float(krea_embed_mse(miss_layer(0), tgt))
    assert late > mid > early
    dummy_w = krea_embed_layer_weights(4)
    assert float(dummy_w[0]) < float(dummy_w[2])
    assert float(dummy_w[2]) == pytest.approx(KREA_EMBED_MID_WEIGHT)


def test_higher_embed_late_weight_increases_late_layer_loss():
    """Heavier late_weight raises the late-layer share of embed MSE."""
    tgt = torch.ones(1, 2, 12, 4)
    late_miss = tgt.clone()
    late_miss[:, :, 6:, :] = 0.0
    mid_miss = tgt.clone()
    mid_miss[:, :, 3, :] = 0.0
    w2 = krea_embed_layer_weights(12, late_weight=2.0)
    w4 = krea_embed_layer_weights(12, late_weight=4.0)
    assert float(w4[6]) == pytest.approx(4.0)
    assert float(w2[6]) == pytest.approx(KREA_EMBED_LATE_WEIGHT)
    late2 = float(krea_embed_mse(late_miss, tgt, layer_weights=w2))
    late4 = float(krea_embed_mse(late_miss, tgt, layer_weights=w4))
    mid2 = float(krea_embed_mse(mid_miss, tgt, layer_weights=w2))
    mid4 = float(krea_embed_mse(mid_miss, tgt, layer_weights=w4))
    assert late4 > late2
    # Mid band stays 1.0; a larger late denom shrinks a mid-only miss.
    assert mid4 < mid2
    uni2 = float(krea_embed_uni_loss(late_miss, tgt, layer_weights=w2, cosine_weight=0.0))
    uni4 = float(krea_embed_uni_loss(late_miss, tgt, layer_weights=w4, cosine_weight=0.0))
    assert uni4 > uni2

    # Dummy 4-layer stack: late_start is past the stack unless lowered.
    dummy_tgt = torch.ones(1, 8, 4, 8)
    dummy_late = dummy_tgt.clone()
    dummy_late[:, :, 2:, :] = 0.0
    dw2 = krea_embed_layer_weights(4, late_layer_start=2, late_weight=2.0)
    dw4 = krea_embed_layer_weights(4, late_layer_start=2, late_weight=4.0)
    assert float(dw4[2]) == pytest.approx(4.0)
    assert float(krea_embed_mse(dummy_late, dummy_tgt, layer_weights=dw4)) > float(
        krea_embed_mse(dummy_late, dummy_tgt, layer_weights=dw2)
    )

    # Same-caption dummy: only LoRA Δ remains, and late (2+) gain is larger.
    backend = DummyKreaBackend(dim=8, rank=2, seed=3, lora_targets="te")
    with torch.no_grad():
        backend.te.lora_up.weight.fill_(0.35)
    from conceptmod.textsliders.train_lora_krea import KreaSliderPrompt

    prompt = KreaSliderPrompt(
        target="a person",
        positive="a person",
        neutral="a person",
        negative="a sad person",
        attributes=["male", "female"],
    )
    low, _ = krea_embed_step_loss(
        backend, prompt, hold_weight=0.0, late_weight=2.0, late_layer_start=2
    )
    high, _ = krea_embed_step_loss(
        backend, prompt, hold_weight=0.0, late_weight=4.0, late_layer_start=2
    )
    assert float(high) > float(low)


def test_dummy_te_emits_stacked_embeds():
    backend = DummyKreaBackend(dim=8, rank=2, seed=0, lora_targets="te")
    assert backend.te.n_layers == KREA_DUMMY_EMBED_LAYERS
    backend.set_adapter_scale(1.0)
    neu, tokens = backend.encode_text("a person")
    assert neu.dim() == 4
    assert tuple(neu.shape) == (1, KREA_DUMMY_EMBED_SEQ, KREA_DUMMY_EMBED_LAYERS, 8)
    assert tokens
    frozen, _ = backend.encode_text("a person", frozen=True)
    assert frozen.shape == neu.shape
    assert not frozen.requires_grad
    with torch.no_grad():
        backend.te.lora_up.weight.fill_(0.35)
    backend.set_adapter_scale(0.0)
    e0, _ = backend.encode_text("a person")
    backend.set_adapter_scale(1.0)
    e1, _ = backend.encode_text("a person")
    assert not torch.allclose(e0, e1, atol=1e-6)
    mid = e0 + 0.5 * (e1 - e0)
    backend.set_adapter_scale(0.5)
    e50, _ = backend.encode_text("a person")
    assert torch.allclose(e50, mid, atol=1e-5)


def test_dummy_embed_step_has_te_grad_no_dit_and_no_velocity():
    backend = DummyKreaBackend(dim=8, rank=2, seed=3, lora_targets="te")
    with torch.no_grad():
        backend.te.lora_up.weight.fill_(0.25)
    from conceptmod.textsliders.train_lora_krea import KreaSliderPrompt

    prompt = KreaSliderPrompt(
        target="a person, closed mouth",
        positive="a person, big smile showing teeth",
        neutral="a person, closed mouth",
        negative="a sad person",
        attributes=["male", "female"],
    )
    z = torch.zeros(1, 8)
    loss, stats = krea_step_loss(
        backend, prompt, z, guidance=4.5, hold_weight=0.1, lm_target="embed"
    )
    assert stats["lm_target"] == "embed"
    assert stats["minus_teacher"] == 0.0
    assert "embed_mse" in stats
    assert "embed_cos" in stats
    assert "embed_max_abs" in stats
    assert "embed_late_l2" in stats
    assert "embed_rel_l2" in stats
    assert "embed_l2_l0" in stats
    assert "cfg_dir_norm" not in stats
    assert torch.isfinite(loss)
    loss.backward()
    te_grads = [p.grad for p in backend.te.lora_down.parameters() if p.grad is not None]
    assert te_grads
    dit_grads = [p.grad for p in backend.dit.lora_down.parameters() if p.grad is not None]
    assert not dit_grads
    direct, direct_stats = krea_embed_step_loss(backend, prompt, hold_weight=0.1)
    assert direct_stats["lm_target"] == "embed"
    assert torch.isfinite(direct)


def test_dummy_smile_v3_embed_train(tmp_path: Path):
    args = parse_args(
        [
            "--dummy",
            "--name",
            "smile-krea-v3-dummy",
            "--prompts_file",
            str(KREA_HAPPY_YAML),
            "--save_dir",
            str(tmp_path),
            "--lora_targets",
            "te",
            "--lm_target",
            "embed",
            "--hold_weight",
            "0.1",
            "--steps",
            "8",
            "--seed",
            "7",
        ]
    )
    sidecar = train(args)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["recipe"] == "embed_uni"
    assert payload["lm_target"] == "embed"
    assert payload["dit_velocity_supervised"] is False
    assert payload["lora_targets"] == "te"
    assert payload["dit_lora"] is False
    assert payload["te_lora"] is True
    assert payload["encoder_lora"] is True
    assert payload["te_parking"] is False
    assert payload["hold_weight"] == pytest.approx(KREA_SMILE_HOLD_WEIGHT)
    assert payload["embed_cosine_weight"] == pytest.approx(KREA_EMBED_COSINE_WEIGHT)
    assert payload["embed_cosine_weight"] == pytest.approx(0.0)
    assert payload["embed_rel_l2_weight"] == pytest.approx(KREA_EMBED_REL_L2_WEIGHT)
    assert payload["embed_late_weight"] == pytest.approx(KREA_EMBED_LATE_WEIGHT)
    assert payload["embed_late_weight"] == pytest.approx(2.0)
    assert payload["embed_late_layer_start"] == KREA_EMBED_LATE_LAYER_START == 6
    assert payload["te_lora_path"] == "smile-krea-v3-dummy_lora/te_lora"
    assert payload["dit_lora_path"] is None
    assert payload["plus_label"] == "Happy"
    assert payload["sample_grid"]["gate"] == "smile-first"
    assert payload["sample_grid"]["crop_purity"] is False
    assert payload["sample_guidance"] == pytest.approx(KREA_EMBED_SAMPLE_CFG)
    assert payload["cfg_uncond_te_frozen"] is True
    assert payload["encode_once"] is True
    assert payload["oracle_grid"]["shots"] == list(KREA_ORACLE_SHOTS)
    log = tmp_path / "smile-krea-v3-dummy_train.jsonl"
    lines = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
    assert lines[0]["lm_target"] == "embed"
    assert "embed_mse" in lines[0]
    assert "embed_max_abs" in lines[0]
    assert "embed_late_l2" in lines[0]
    assert "embed_l2_l0" in lines[0]
    assert "cfg_dir_norm" not in lines[0]
    pngs = list((tmp_path / "samples").glob("*.png"))
    assert len(pngs) == 12
    oracle_pngs = list((tmp_path / "samples" / "oracle").glob("*.png"))
    assert len(oracle_pngs) == 6
    oracle_meta = json.loads(
        (tmp_path / "samples" / "oracle" / "oracle_meta.json").read_text(encoding="utf-8")
    )
    assert oracle_meta["kind"] == "oracle_apply_audit"
    assert "embed_cos" in oracle_meta["pairs"][0]


def test_dummy_embed_uni_recipe_alias(tmp_path: Path):
    args = parse_args(
        [
            "--dummy",
            "--name",
            "smile-embed-alias",
            "--prompts_file",
            str(KREA_HAPPY_YAML),
            "--save_dir",
            str(tmp_path),
            "--recipe",
            "embed_uni",
            "--hold_weight",
            "0.1",
        ]
    )
    sidecar = train(args)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["lm_target"] == "embed"
    assert payload["lora_targets"] == "te"
    assert payload["dit_lora"] is False


def test_velocity_path_still_default_after_embed_flag():
    bare = parse_args([])
    assert bare.lm_target == "v"
    assert bare.recipe == "uni"
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
    assert "cfg_dir_norm" in stats
    assert stats.get("lm_target") != "embed"
    assert torch.isfinite(loss)
    assert KREA_EMBED_EARLY_LAYERS == 2
    assert KREA_EMBED_LATE_LAYER_START == 6


def test_docs_list_detail_krea_v1_card():
    docs = (ROOT / "docs/krea-slider.md").read_text(encoding="utf-8")
    assert "--name detail-krea-v1" in docs
    assert "prompts-krea-detailed.yaml" in docs
    assert "--lora_targets te --lm_target embed" in docs
    assert "--embed_cosine_weight 0" in docs
    assert "--te_dit_mask auto" in docs
    assert "--steps 800" in docs
    assert "not embed_cos" in docs or "not `embed_cos`" in docs
    assert "frozen-plus oracle" in docs or "frozen-plus" in docs


def test_docs_list_detail_krea_v2_card():
    docs = (ROOT / "docs/krea-slider.md").read_text(encoding="utf-8")
    assert "--name detail-krea-v2" in docs
    assert "prompts-krea-detailed.yaml" in docs
    assert "--embed_late_weight 4.0" in docs
    assert "--embed_rel_l2_weight 2.0" in docs
    assert "--rank 32" in docs
    assert "--steps 1600" in docs
    assert "landscape" in docs
    assert "room" in docs or "interior" in docs
    assert "object" in docs
    assert "GATE FAIL" in docs
    assert "not embed_cos" in docs or "not `embed_cos`" in docs
    trainer = KREA_TRAINER.read_text(encoding="utf-8")
    assert "--embed_late_weight" in trainer
    assert "--embed_late_layer_start" in trainer
    assert "KREA_EMBED_LATE_WEIGHT" in trainer


def test_docs_list_smile_krea_v4_card():
    docs = (ROOT / "docs/krea-slider.md").read_text(encoding="utf-8")
    assert "--name smile-krea-v4" in docs
    assert "--sample_guidance 0" in docs
    assert "--embed_cosine_weight 0" in docs
    assert "--te_dit_mask auto" in docs
    assert "--steps 800" in docs
    assert "max_abs" in docs
    assert "cosine≠magnitude" in docs or "cosine hid" in docs
    assert "teeth vs oracle" in docs
    assert "not embed_cos" in docs or "not `embed_cos`" in docs
    assert "oracle_plus_frozen" in docs
    assert "student_neu_scale1" in docs
    assert "frozen TE" in docs
    assert "--load_te_lora" in docs
    assert "same_crop" in docs and "embed_struct" in docs
    trainer = KREA_TRAINER.read_text(encoding="utf-8")
    assert "--load_te_lora" in trainer
    assert "oracle_plus_frozen" in trainer
    live = KREA_LIVE.read_text(encoding="utf-8")
    assert "frozen TE when encoder_lora" in live
    assert "reuse hidden states" in live
    assert "ones" in live


def test_docs_list_smile_krea_v5_ones_mask():
    docs = (ROOT / "docs/krea-slider.md").read_text(encoding="utf-8")
    assert "smile-krea-v5-resmoke" in docs
    assert "mask truncated" in docs.lower() or "hid the smile slots" in docs or "smile slots" in docs
    assert "--te_dit_mask" in docs
    assert "student_neu_scale1_tokmask" in docs
    assert "same_crop" in docs and "embed_struct" in docs
    trainer = KREA_TRAINER.read_text(encoding="utf-8")
    assert "--te_dit_mask" in trainer
    assert "ones" in trainer
    live = KREA_LIVE.read_text(encoding="utf-8")
    assert "max_sequence_length" in live
    assert "smile slots" in live


def test_embed_defaults_sample_guidance_zero():
    assert resolve_krea_sample_guidance(
        None, model_id=KREA_RAW_MODEL, lm_target="embed"
    ) == pytest.approx(KREA_EMBED_SAMPLE_CFG)
    assert resolve_krea_sample_guidance(
        None, model_id=KREA_RAW_MODEL, lm_target="v"
    ) == pytest.approx(KREA_RAW_CFG)
    assert resolve_krea_sample_guidance(
        4.5, model_id=KREA_RAW_MODEL, lm_target="embed"
    ) == pytest.approx(4.5)
    embed_card = resolve_krea_card(
        KREA_RAW_MODEL, None, None, lm_target="embed"
    )
    assert embed_card["sample_guidance"] == pytest.approx(KREA_EMBED_SAMPLE_CFG)
    vel_card = resolve_krea_card(KREA_RAW_MODEL, None, None, lm_target="v")
    assert vel_card["sample_guidance"] == pytest.approx(KREA_RAW_CFG)
    turbo = resolve_krea_sample_guidance(
        None, model_id="/comfy/Krea-2-Turbo.safetensors", lm_target="v"
    )
    assert turbo == pytest.approx(KREA_TURBO_CFG)


def test_cfg_uncond_uses_frozen_te_when_encoder_lora():
    assert krea_cfg_uncond_te_frozen(True) is True
    assert krea_cfg_uncond_te_frozen(False) is False
    backend = DummyKreaBackend(dim=8, rank=2, seed=4, lora_targets="te")
    with torch.no_grad():
        backend.te.lora_up.weight.fill_(0.3)
    calls: list[tuple[str, bool]] = []
    orig = backend.encode_text

    def spy(prompt: str, *, frozen: bool = False):
        calls.append((prompt, frozen))
        return orig(prompt, frozen=frozen)

    backend.encode_text = spy  # type: ignore[method-assign]
    z = torch.ones(1, 8)
    vel = backend.cfg_predict_v("a person", z, scale=1.0, guidance=4.5)
    assert torch.isfinite(vel).all()
    uncond = [frozen for prompt, frozen in calls if prompt == ""]
    assert uncond
    assert all(frozen is True for frozen in uncond)
    cond = [frozen for prompt, frozen in calls if prompt == "a person"]
    assert any(frozen is False for frozen in cond)

    calls.clear()
    dit = DummyKreaBackend(dim=8, rank=2, seed=4, lora_targets="dit")
    dit_orig = dit.encode_text

    def dit_spy(prompt: str, *, frozen: bool = False):
        calls.append((prompt, frozen))
        return dit_orig(prompt, frozen=frozen)

    dit.encode_text = dit_spy  # type: ignore[method-assign]
    dit.cfg_predict_v("a person", z, scale=1.0, guidance=4.5)
    dit_uncond = [frozen for prompt, frozen in calls if prompt == ""]
    assert dit_uncond
    assert all(frozen is False for frozen in dit_uncond)


def test_generate_encodes_cfg_pair_once():
    backend = DummyKreaBackend(dim=8, rank=2, seed=5, lora_targets="te")
    calls: list[str] = []
    orig = backend.encode_text

    def spy(prompt: str, *, frozen: bool = False):
        calls.append(prompt)
        return orig(prompt, frozen=frozen)

    backend.encode_text = spy  # type: ignore[method-assign]
    backend.generate("a person", seed=0, scale=1.0, guidance=4.5, num_steps=8)
    assert calls.count("a person") == 1
    assert calls.count("") == 1


def test_embed_hold_does_not_adapt_encode_plus():
    backend = DummyKreaBackend(dim=8, rank=2, seed=6, lora_targets="te")
    with torch.no_grad():
        backend.te.lora_up.weight.fill_(0.3)
    from conceptmod.textsliders.train_lora_krea import KreaSliderPrompt

    prompt = KreaSliderPrompt(
        target="a person, closed mouth",
        positive="a person, big smile showing teeth",
        neutral="a person, closed mouth",
        negative="a sad person",
        attributes=["male", "female"],
    )
    calls: list[dict[str, object]] = []
    orig = backend.encode_text

    def spy(caption: str, *, frozen: bool = False):
        calls.append({"prompt": caption, "frozen": frozen})
        return orig(caption, frozen=frozen)

    backend.encode_text = spy  # type: ignore[method-assign]
    loss, stats = krea_embed_step_loss(backend, prompt, hold_weight=0.1)
    assert torch.isfinite(loss)
    assert stats["lm_target"] == "embed"
    plus = [c for c in calls if c["prompt"] == prompt.positive]
    assert plus
    assert all(c["frozen"] is True for c in plus)
    neu_adapted = [
        c for c in calls if c["prompt"] == prompt.neutral and c["frozen"] is False
    ]
    assert neu_adapted


def test_dummy_oracle_grid_and_load_te_lora_resmoke(tmp_path: Path):
    prompts, _meta = load_prompts(KREA_HAPPY_YAML)
    pairs = infer_oracle_pairs(prompts)
    assert len(pairs) == 2
    assert all("closed mouth" in neu for neu, _plus in pairs)
    assert all("teeth" in plus for _neu, plus in pairs)

    adapter = tmp_path / "saved_te_lora"
    adapter.mkdir()
    args = parse_args(
        [
            "--dummy",
            "--name",
            "smile-krea-v4-resmoke",
            "--prompts_file",
            str(KREA_HAPPY_YAML),
            "--save_dir",
            str(tmp_path),
            "--lora_targets",
            "te",
            "--lm_target",
            "embed",
            "--hold_weight",
            "0.1",
            "--load_te_lora",
            str(adapter),
        ]
    )
    sidecar = train(args)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["skipped_train"] is True
    assert payload["load_te_lora"] == str(adapter)
    assert payload["lm_target"] == "embed"
    assert payload["sample_guidance"] == pytest.approx(KREA_EMBED_SAMPLE_CFG)
    log = tmp_path / "smile-krea-v4-resmoke_train.jsonl"
    assert not log.exists() or log.read_text(encoding="utf-8").strip() == ""
    oracle_dir = tmp_path / "samples" / "oracle"
    names = " ".join(p.name for p in oracle_dir.glob("*.png"))
    assert "oracle_plus_frozen" in names
    assert "student_neu_scale1" in names
    assert "neu_scale0" in names
    assert "student_neu_scale1_onesmask" in names
    assert "student_neu_scale1_tokmask" in names
    assert "student_neu_scale1_plusmask" in names
    assert len(list(oracle_dir.glob("*.png"))) == 12
    meta = json.loads((oracle_dir / "oracle_meta.json").read_text(encoding="utf-8"))
    assert meta["embed_cos_threshold"] == pytest.approx(KREA_ORACLE_EMBED_COS)
    assert "apply bug" in meta["readout"]
    assert "ones-mask" in meta["readout"] or "smile slots" in meta["readout"]
    assert meta["mask_ab"] == list(KREA_ORACLE_MASK_AB_SHOTS)
    assert all("embed_cos" in pair for pair in meta["pairs"])
    assert payload["te_dit_mask"] == KREA_TE_DIT_MASK_DEFAULT
    assert payload["te_dit_ones_mask"] is True


def test_te_scale_gt0_dit_mask_is_ones_not_neu_span():
    """When TE scale>0 the mask passed to DiT is all-ones (longer than neu)."""
    neu = "a person, closed mouth"
    plus = "a person, big smile showing teeth, happy joyful expression"
    neu_n = len(krea_word_tokens(neu))
    plus_n = len(krea_word_tokens(plus))
    assert plus_n > neu_n
    assert neu_n < KREA_DUMMY_EMBED_SEQ

    tok = torch.zeros(1, KREA_DUMMY_EMBED_SEQ, dtype=torch.long)
    tok[0, :neu_n] = 1
    embeds = torch.randn(1, KREA_DUMMY_EMBED_SEQ, KREA_DUMMY_EMBED_LAYERS, 8)
    ones = krea_resolve_dit_encoder_mask(
        tok, embeds, encoder_lora=True, scale=1.0
    )
    assert ones is not None
    assert torch.equal(ones, torch.ones_like(tok))
    assert int(ones.sum()) > neu_n
    assert int(ones.sum()) == KREA_DUMMY_EMBED_SEQ
    assert torch.equal(
        krea_ones_attention_mask(embeds, like=tok), torch.ones_like(tok)
    )

    frozen = krea_resolve_dit_encoder_mask(
        tok, embeds, encoder_lora=True, scale=0.0
    )
    assert torch.equal(frozen, tok)
    frozen_te = krea_resolve_dit_encoder_mask(
        tok, embeds, encoder_lora=True, scale=1.0, frozen=True
    )
    assert torch.equal(frozen_te, tok)
    dit_only = krea_resolve_dit_encoder_mask(
        tok, embeds, encoder_lora=False, scale=1.0
    )
    assert torch.equal(dit_only, tok)
    old = krea_resolve_dit_encoder_mask(
        tok, embeds, encoder_lora=True, scale=1.0, te_dit_mask="tokenizer"
    )
    assert torch.equal(old, tok)
    transplant = torch.zeros_like(tok)
    transplant[0, : min(plus_n, KREA_DUMMY_EMBED_SEQ)] = 1
    plus_mask = krea_resolve_dit_encoder_mask(
        tok,
        embeds,
        encoder_lora=True,
        scale=1.0,
        te_dit_mask="transplant",
        transplant_mask=transplant,
    )
    assert torch.equal(plus_mask, transplant)
    assert krea_te_sample_use_ones_mask(True, 1.0) is True
    assert krea_te_sample_use_ones_mask(True, 0.0) is False
    assert krea_te_sample_use_ones_mask(False, 1.0) is False

    backend = DummyKreaBackend(dim=8, rank=2, seed=9, lora_targets="te")
    z = torch.ones(1, 8)
    backend.predict_v(neu, z, scale=1.0)
    mask1 = backend.last_dit_mask
    assert mask1 is not None
    assert torch.all(mask1 == 1)
    assert int(mask1.sum()) > neu_n
    assert mask1.shape[-1] == KREA_DUMMY_EMBED_SEQ

    backend.predict_v(neu, z, scale=0.0)
    mask0 = backend.last_dit_mask
    assert mask0 is not None
    assert int(mask0.sum()) == neu_n
    assert not torch.all(mask0 == 1)

    backend.generate(neu, seed=0, scale=1.0, guidance=0.0)
    gen_mask = backend.last_dit_mask
    assert gen_mask is not None
    assert torch.all(gen_mask == 1)

    backend.generate(neu, seed=0, scale=1.0, te_dit_mask="tokenizer")
    old_gen = backend.last_dit_mask
    assert old_gen is not None
    assert int(old_gen.sum()) == neu_n

    dit = DummyKreaBackend(dim=8, rank=2, seed=9, lora_targets="dit")
    dit.predict_v(neu, z, scale=1.0)
    dit_mask = dit.last_dit_mask
    assert dit_mask is not None
    assert int(dit_mask.sum()) == neu_n
