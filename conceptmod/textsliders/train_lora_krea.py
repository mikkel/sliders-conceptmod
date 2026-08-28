#!/usr/bin/env python3
"""Opt-in Krea image concept-slider trainer.

UNI analog (not Music 3 lyric-hold):

- student +1 → + concept prompt velocity
- student scale 0 → neutral prompt velocity
- no minus teacher (canary only)
- unused prompt tokens hold to encode(neu); concept words are not held

Velocity-space CFG geometry from conceptmod when it maps:

    v(z, t, c) − v(z, t, '')

Official card: train LoRAs on ``krea/Krea-2-Raw`` (28 steps, CFG 4.5),
rank 16, 512 px. Run on Turbo (local ComfyUI ``.safetensors``, 8 steps,
CFG 0). Default Music 3 trainers are unchanged.

``--dummy`` never loads Hub weights or a 12B transformer. CI uses that.
Live load is offline-safe unless ``--allow_hub`` (Anima pattern).
``--lora_targets dit`` (default) parks the frozen text encoder after
encode. ``te`` / ``dit+te`` keep Qwen3-VL on GPU so encode+backward
stay coherent. Happy/smile yaml: ``data/prompts-krea-happy.yaml``.
Smile v2 card: ``--lora_targets dit+te --hold_weight 0.1``.
Smile v3/v4/v5: ``--lora_targets te --lm_target embed`` (TE-only stacked
embeds; DiT stays base). Live gap: DiT v neu/plus cos≈0.9999;
TE ``[1,512,12,2560]`` cos≈0.67. Do not teach v-space on that path.
v4 retrain: MSE + rel-L2 (``--embed_cosine_weight`` default 0);
cosine hid a residual magnitude gap (cos≈0.9959, max_abs≈147).
v4: CFG uncond uses frozen TE; encode once per generate; embed
sample guidance defaults to 0; oracle apply-audit grid.
v5: TE scale>0 uses an all-ones DiT attention mask so UNI-matched
rows past neu's token span are actually attended.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
import yaml
from tqdm.auto import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from conceptmod.textsliders.slider_targets import (
    KREA_CONTROL_PROMPT,
    KREA_DEFAULT_LORA_TARGETS,
    KREA_DEFAULT_RANK,
    KREA_DEFAULT_RESOLUTION,
    KREA_DUMMY_EMBED_LAYERS,
    KREA_DUMMY_EMBED_SEQ,
    KREA_EMBED_COSINE_WEIGHT,
    KREA_EMBED_REL_L2_WEIGHT,
    KREA_EMBED_SAMPLE_CFG,
    KREA_HOLD_WEIGHT,
    KREA_LM_TARGET_CHOICES,
    KREA_LM_TARGET_DEFAULT,
    KREA_LORA_TARGET_CHOICES,
    KREA_ORACLE_EMBED_COS,
    KREA_ORACLE_MASK_AB_SHOTS,
    KREA_ORACLE_SHOTS,
    KREA_TE_DIT_MASK_CHOICES,
    KREA_TE_DIT_MASK_DEFAULT,
    KREA_RAW_CFG,
    KREA_RAW_MODEL,
    KREA_RECIPE_CHOICES,
    KREA_RECIPE_DEFAULT,
    KREA_SAMPLE_SCALES,
    KREA_SMILE_HOLD_WEIGHT,
    expand_attributes_krea,
    force_krea_embed_lora_targets,
    krea_cfg_compose,
    krea_cfg_direction,
    krea_cfg_uncond_te_frozen,
    krea_concept_words,
    krea_embed_cosine,
    krea_embed_mse,
    krea_embed_requires_te,
    krea_embed_train_stats,
    krea_embed_uni_loss,
    krea_hold_unused_embeds,
    krea_minus_canary,
    krea_oracle_readout,
    krea_resolve_dit_encoder_mask,
    krea_plus_neu_loss,
    krea_plus_neu_teachers,
    krea_sample_card,
    krea_unused_hold_loss,
    krea_unused_hold_mask,
    krea_word_tokens,
    resolve_krea_lm_target,
    resolve_krea_lora_targets,
    resolve_krea_sample_guidance,
)

DEFAULT_CONFIG = Path(__file__).resolve().parent / "data" / "config-krea.yaml"
DEFAULT_PROMPTS = Path(__file__).resolve().parent / "data" / "prompts-krea.yaml"
DEFAULT_SAVE_DIR = Path("models/krea-slider")

# Banned in this trainer. Anima / ZiT / H3 are a separate PR.
_FOREIGN_BACKENDS = ("anima", "zit", "h3", "z-image", "zimage")


@dataclass
class KreaSliderPrompt:
    target: str
    positive: str
    neutral: str
    negative: str = ""
    attributes: list[str] = field(default_factory=list)
    action: str = "enhance"
    guidance_scale: float = KREA_RAW_CFG
    batch_size: int = 1
    unconditional: str = ""


@dataclass
class PromptsMeta:
    plus_label: str = ""
    minus_label: str = ""
    recommended_range: list[float] = field(default_factory=lambda: [-2.0, 2.0])
    concept_words: str = ""
    control_prompt: str = KREA_CONTROL_PROMPT
    bare_captions: bool = False


def _load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_prompts(path: Path) -> tuple[list[KreaSliderPrompt], PromptsMeta]:
    raw = _load_yaml(path)
    meta = PromptsMeta()
    prefix_attributes = True
    if isinstance(raw, dict):
        meta.plus_label = str(raw.get("plus_label") or "")
        meta.minus_label = str(raw.get("minus_label") or "")
        meta.concept_words = str(raw.get("concept_words") or "")
        meta.control_prompt = str(raw.get("control_prompt") or KREA_CONTROL_PROMPT)
        bare = bool(raw.get("bare_captions"))
        if "prefix_attributes" in raw:
            prefix_attributes = bool(raw.get("prefix_attributes"))
        elif bare:
            prefix_attributes = False
        meta.bare_captions = not prefix_attributes
        rng = raw.get("recommended_range")
        if isinstance(rng, (list, tuple)) and len(rng) == 2:
            meta.recommended_range = [float(rng[0]), float(rng[1])]
        raw = raw.get("rows")
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"prompts file is empty: {path}")
    prompts: list[KreaSliderPrompt] = []
    for item in raw:
        if not isinstance(item, dict) or "positive" not in item:
            raise ValueError(f"each prompt must be a mapping with positive: {item!r}")
        for row in expand_attributes_krea(item, prefix=prefix_attributes):
            target = str(row.get("target") or row.get("neutral") or row["positive"])
            prompts.append(
                KreaSliderPrompt(
                    target=target,
                    positive=str(row["positive"]),
                    neutral=str(row.get("neutral") or target),
                    negative=str(row.get("negative") or ""),
                    attributes=[str(a) for a in (row.get("attributes") or item.get("attributes") or [])],
                    action=str(row.get("action") or "enhance"),
                    guidance_scale=float(row.get("guidance_scale", KREA_RAW_CFG)),
                    batch_size=int(row.get("batch_size", 1)),
                    unconditional=str(row.get("unconditional") or ""),
                )
            )
    return prompts, meta


def unused_words_for(prompt: KreaSliderPrompt) -> list[str]:
    words: list[str] = []
    for attr in prompt.attributes:
        words.extend(krea_word_tokens(attr))
    return words


def resolve_krea_card(
    model_id: str,
    sample_steps: int | None,
    sample_guidance: float | None,
    *,
    lm_target: str | None = None,
    recipe: str | None = None,
) -> dict[str, float | int | str]:
    card = krea_sample_card(model_id)
    if sample_steps is not None:
        card["sample_steps"] = int(sample_steps)
    resolved_lm = resolve_krea_lm_target(lm_target, recipe)
    card["sample_guidance"] = resolve_krea_sample_guidance(
        sample_guidance,
        model_id=model_id,
        lm_target=resolved_lm,
        recipe=recipe,
    )
    if (
        sample_guidance is None
        and resolved_lm == "embed"
        and abs(float(card["sample_guidance"]) - KREA_EMBED_SAMPLE_CFG) < 1e-12
    ):
        card["sample_guidance_default"] = "embed_te_only"
    return card


class DummyKreaEncode:
    """Deterministic word-table encode. No tokenizer, no Hub."""

    def __init__(self, dim: int = 8, seed: int = 0):
        self.dim = dim
        self._table: dict[str, torch.Tensor] = {}
        self._rng = torch.Generator().manual_seed(int(seed) + 17)

    def _vec(self, token: str) -> torch.Tensor:
        if token not in self._table:
            self._table[token] = torch.randn(self.dim, generator=self._rng)
        return self._table[token]

    def encode(self, prompt: str) -> tuple[torch.Tensor, list[str]]:
        tokens = krea_word_tokens(prompt) or [""]
        embeds = torch.stack([self._vec(tok) for tok in tokens], dim=0)
        return embeds, tokens


class DummyKreaDiT(nn.Module):
    """Tiny velocity head + multiplier LoRA. CPU only."""

    def __init__(self, dim: int = 8, rank: int = 2):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)
        self.lora_down = nn.Linear(dim, rank, bias=False)
        self.lora_up = nn.Linear(rank, dim, bias=False)
        nn.init.zeros_(self.lora_up.weight)
        self.scale = 0.0

    def forward(self, z: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        flat = text.reshape(-1, text.shape[-1])
        pooled = flat.mean(dim=0, keepdim=True).expand_as(z)
        base = self.proj(z + pooled)
        delta = self.lora_up(self.lora_down(z + pooled))
        return base + float(self.scale) * delta


class DummyKreaTE(nn.Module):
    """Fake Qwen3-VL attn + LoRA delta. Names match live TE targets.

    Live ``get_text_hidden_states`` is ``[B, T, 12, 2560]``. Dummy
    stacks ``KREA_DUMMY_EMBED_LAYERS`` (4) and pads to
    ``KREA_DUMMY_EMBED_SEQ`` so ``--lm_target embed`` can score the
    same 4D layout. Early layers (0–1) get a smaller LoRA gain so
    mid/late (2+) carry the concept Δ, matching the live diag.
    """

    def __init__(
        self,
        dim: int = 8,
        rank: int = 2,
        n_layers: int = KREA_DUMMY_EMBED_LAYERS,
        seq_pad: int = KREA_DUMMY_EMBED_SEQ,
        stacked: bool = True,
    ):
        super().__init__()
        self.n_layers = int(n_layers)
        self.seq_pad = int(seq_pad)
        self.stacked = bool(stacked)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        for layer in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            nn.init.eye_(layer.weight)
            layer.weight.requires_grad_(False)
        self.lora_down = nn.Linear(dim, rank, bias=False)
        self.lora_up = nn.Linear(rank, dim, bias=False)
        nn.init.zeros_(self.lora_up.weight)
        self.scale = 1.0

    def _pad_tokens(self, embeds: torch.Tensor) -> torch.Tensor:
        tokens, dim = embeds.shape
        pad_to = self.seq_pad
        if tokens == pad_to:
            return embeds
        if tokens < pad_to:
            return torch.cat(
                [embeds, embeds.new_zeros(pad_to - tokens, dim)], dim=0
            )
        return embeds[:pad_to]

    def forward(self, embeds: torch.Tensor) -> torch.Tensor:
        hidden = self.o_proj(self.v_proj(embeds))
        delta = self.lora_up(self.lora_down(embeds))
        if not self.stacked:
            return hidden + float(self.scale) * delta
        padded = self._pad_tokens(hidden)
        delta_pad = self._pad_tokens(delta)
        layers = []
        for index in range(self.n_layers):
            # Live diag: early layers stay close (cos≈0.91); mid/late move.
            gain = 0.25 if index < 2 else 1.0
            layers.append(padded + float(self.scale) * gain * delta_pad)
        return torch.stack(layers, dim=1).unsqueeze(0)


class DummyKreaBackend:
    """Mock Krea velocity backend. Never downloads weights."""

    def __init__(
        self,
        dim: int = 8,
        rank: int = 2,
        seed: int = 0,
        lora_targets: str = KREA_DEFAULT_LORA_TARGETS,
    ):
        self.dim = dim
        self.lora_spec = resolve_krea_lora_targets(lora_targets)
        self.encode_table = DummyKreaEncode(dim=dim, seed=seed)
        self.dit = DummyKreaDiT(dim=dim, rank=rank)
        self.te = (
            DummyKreaTE(dim=dim, rank=rank, stacked=True)
            if self.lora_spec.train_te
            else None
        )
        self.encoder_lora = self.lora_spec.train_te
        if not self.lora_spec.train_dit:
            for param in list(self.dit.lora_down.parameters()) + list(
                self.dit.lora_up.parameters()
            ):
                param.requires_grad_(False)
        self.latent_shape = (dim,)
        self._timestep: torch.Tensor | None = None
        self.loaded_te_lora: str | None = None
        self.max_sequence_length = (
            int(self.te.seq_pad) if self.te is not None else int(KREA_DUMMY_EMBED_SEQ)
        )
        self._last_mask: torch.Tensor | None = None
        self.last_dit_mask: torch.Tensor | None = None
        self._last_cond_mask: torch.Tensor | None = None
        self._last_uncond_mask: torch.Tensor | None = None

    def begin_step(self) -> None:
        self._timestep = torch.zeros(1)

    def sample_latents(self, device: torch.device | None = None) -> torch.Tensor:
        dest = device or torch.device("cpu")
        return torch.randn(1, self.dim, device=dest)

    def _tokenizer_mask(self, tokens: list[str], embeds: torch.Tensor) -> torch.Tensor:
        """1s on real tokens, 0s on dummy pad — analog of the live neu span."""
        if embeds.dim() >= 3:
            batch, seq = int(embeds.shape[0]), int(embeds.shape[1])
        elif embeds.dim() == 2:
            batch, seq = 1, int(embeds.shape[0])
        else:
            batch, seq = 1, max(len(tokens), 1)
        seq = max(seq, int(self.max_sequence_length) if self.te is not None else seq)
        if embeds.dim() >= 3:
            seq = int(embeds.shape[1])
        mask = torch.zeros(batch, seq, dtype=torch.long)
        n = min(len(tokens), seq)
        if n:
            mask[:, :n] = 1
        return mask

    def encode_text(
        self, prompt: str, *, frozen: bool = False
    ) -> tuple[torch.Tensor, list[str]]:
        embeds, tokens = self.encode_table.encode(prompt)
        if self.te is None:
            self._last_mask = self._tokenizer_mask(tokens, embeds)
            return embeds, tokens
        if frozen:
            prev = float(self.te.scale)
            self.te.scale = 0.0
            try:
                embeds = self.te(embeds)
            finally:
                self.te.scale = prev
            self._last_mask = self._tokenizer_mask(tokens, embeds)
            return embeds.detach(), tokens
        embeds = self.te(embeds)
        self._last_mask = self._tokenizer_mask(tokens, embeds)
        return embeds, tokens

    def set_adapter_scale(self, scale: float) -> None:
        self.dit.scale = float(scale) if self.lora_spec.train_dit else 0.0
        if self.te is not None:
            self.te.scale = float(scale)

    def encode_cfg_pair(
        self,
        prompt: str,
        *,
        scale: float = 1.0,
        te_dit_mask: str = KREA_TE_DIT_MASK_DEFAULT,
        mask_prompt: str | None = None,
    ) -> tuple[tuple[torch.Tensor, list[str]], tuple[torch.Tensor, list[str]]]:
        """Encode cond at ``scale``; empty uncond with frozen TE if encoder_lora."""
        cond_frozen = abs(float(scale)) < 1e-12
        transplant = None
        if mask_prompt:
            _emb, tok = self.encode_text(str(mask_prompt), frozen=True)
            transplant = self._tokenizer_mask(tok, _emb)
        self.set_adapter_scale(0.0 if cond_frozen else float(scale))
        cond = self.encode_text(prompt, frozen=cond_frozen)
        cond_tok = self._last_mask
        self._last_cond_mask = krea_resolve_dit_encoder_mask(
            cond_tok,
            cond[0],
            encoder_lora=bool(self.encoder_lora),
            scale=float(scale),
            frozen=cond_frozen,
            te_dit_mask=te_dit_mask,
            transplant_mask=transplant,
            max_sequence_length=self.max_sequence_length,
        )
        if krea_cfg_uncond_te_frozen(self.encoder_lora):
            uncond = self.encode_text("", frozen=True)
        else:
            uncond = self.encode_text("", frozen=cond_frozen)
        self._last_uncond_mask = self._last_mask
        return cond, uncond

    def cfg_predict_v(
        self,
        prompt: str,
        z: torch.Tensor,
        *,
        scale: float = 1.0,
        guidance: float = 0.0,
        te_dit_mask: str = KREA_TE_DIT_MASK_DEFAULT,
        mask_prompt: str | None = None,
    ) -> torch.Tensor:
        """Dummy CFG: cond at ``scale``, uncond frozen TE when encoder_lora."""
        (cond, _), (uncond, _) = self.encode_cfg_pair(
            prompt, scale=scale, te_dit_mask=te_dit_mask, mask_prompt=mask_prompt
        )
        self.set_adapter_scale(scale)
        v = self._forward(
            z,
            cond,
            self._last_cond_mask,
            scale=float(scale),
            frozen=abs(float(scale)) < 1e-12,
            te_dit_mask=te_dit_mask,
            mask_resolved=True,
        )
        if guidance and float(guidance) > 0.0 and prompt != "":
            v_u = self._forward(
                z,
                uncond,
                self._last_uncond_mask,
                scale=0.0,
                frozen=True,
                mask_resolved=True,
            )
            return krea_cfg_compose(v, v_u, guidance)
        return v

    def load_te_adapter(self, path: str | Path) -> None:
        """Dummy resmoke: record the path. No Hub / PEFT load."""
        self.loaded_te_lora = str(path)

    def _forward(
        self,
        z: torch.Tensor,
        text: torch.Tensor,
        mask: torch.Tensor | None,
        *,
        scale: float | None = None,
        frozen: bool = False,
        te_dit_mask: str = KREA_TE_DIT_MASK_DEFAULT,
        transplant_mask: torch.Tensor | None = None,
        mask_resolved: bool = False,
    ) -> torch.Tensor:
        if scale is None:
            if self.te is not None:
                scale = float(self.te.scale)
            else:
                scale = float(self.dit.scale)
        if mask_resolved:
            resolved = mask
        else:
            resolved = krea_resolve_dit_encoder_mask(
                mask,
                text,
                encoder_lora=bool(self.encoder_lora),
                scale=float(scale),
                frozen=bool(frozen),
                te_dit_mask=te_dit_mask,
                transplant_mask=transplant_mask,
                max_sequence_length=self.max_sequence_length,
            )
        self.last_dit_mask = None if resolved is None else resolved.detach().clone()
        return self.dit(z, text)

    def predict_v(
        self,
        prompt: str,
        z: torch.Tensor,
        *,
        scale: float = 0.0,
        pin_unused: bool = False,
        neu_prompt: str | None = None,
        unused_words: Sequence[str] | None = None,
        te_dit_mask: str = KREA_TE_DIT_MASK_DEFAULT,
        mask_prompt: str | None = None,
    ) -> torch.Tensor:
        self.set_adapter_scale(scale)
        frozen = abs(float(scale)) < 1e-12
        transplant = None
        if mask_prompt:
            _emb, tok = self.encode_text(str(mask_prompt), frozen=True)
            transplant = self._tokenizer_mask(tok, _emb)
        embeds, tokens = self.encode_text(prompt)
        tok_mask = self._last_mask
        if pin_unused and neu_prompt is not None:
            neu_embeds, neu_tokens = self.encode_text(neu_prompt, frozen=True)
            hold = krea_unused_hold_mask(tokens, neu_tokens, unused_words)
            embeds = krea_hold_unused_embeds(
                embeds, neu_embeds, tokens, neu_tokens, hold
            )
        return self._forward(
            z,
            embeds,
            tok_mask,
            scale=float(scale),
            frozen=frozen,
            te_dit_mask=te_dit_mask,
            transplant_mask=transplant,
        )

    def trainable_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        if self.lora_spec.train_dit:
            params.extend(self.dit.lora_down.parameters())
            params.extend(self.dit.lora_up.parameters())
        if self.te is not None:
            params.extend(self.te.lora_down.parameters())
            params.extend(self.te.lora_up.parameters())
        if not params:
            raise RuntimeError("Krea dummy LoRA attached but no trainable parameters")
        return params

    def generate(
        self,
        prompt: str,
        seed: int = 0,
        num_steps: int | None = None,
        guidance: float | None = None,
        scale: float = 1.0,
        height: int = 64,
        width: int = 64,
        te_dit_mask: str = KREA_TE_DIT_MASK_DEFAULT,
        mask_prompt: str | None = None,
    ):
        """Structured RGB ramp so ``--dummy`` can write a smile-first grid."""
        del num_steps
        # Encode cond + frozen uncond once (mirrors live generate).
        (cond, _), _uncond = self.encode_cfg_pair(
            prompt,
            scale=float(scale),
            te_dit_mask=te_dit_mask,
            mask_prompt=mask_prompt,
        )
        z = torch.zeros(1, self.dim)
        self._forward(
            z,
            cond,
            self._last_cond_mask,
            scale=float(scale),
            frozen=abs(float(scale)) < 1e-12,
            te_dit_mask=te_dit_mask,
            mask_resolved=True,
        )
        del guidance
        import numpy as np
        from PIL import Image

        h = max(8, int(height))
        w = max(8, int(width))
        rng = np.random.default_rng((abs(int(seed)) + (hash(prompt) % 997)) % (2**31))
        ys = np.linspace(48.0, 176.0, h, dtype=np.float64)
        xs = np.linspace(36.0, 168.0, w, dtype=np.float64)
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        tint = 12.0 * float(scale)
        blob = 18.0 * np.sin(yy / 18.0) * np.cos(xx / 22.0)
        noise = 4.0 * rng.standard_normal((h, w))
        img = np.stack(
            [yy + tint + blob, xx + 0.4 * tint, 0.45 * yy + 0.45 * xx + 16.0 + 0.6 * tint + noise],
            axis=-1,
        )
        return Image.fromarray(np.clip(img, 0.0, 255.0).astype("uint8"), mode="RGB")


def assert_krea_only(model_id: str) -> None:
    lowered = str(model_id).lower()
    for name in _FOREIGN_BACKENDS:
        if name in lowered:
            raise ValueError(
                f"this trainer is Krea-only; refused foreign backend {name!r}"
            )


def krea_embed_step_loss(
    backend: DummyKreaBackend,
    prompt: KreaSliderPrompt,
    *,
    hold_weight: float = KREA_SMILE_HOLD_WEIGHT,
    cosine_weight: float = KREA_EMBED_COSINE_WEIGHT,
    rel_l2_weight: float = KREA_EMBED_REL_L2_WEIGHT,
) -> tuple[torch.Tensor, dict[str, float]]:
    """TE-only embed UNI. Does not teach DiT velocity.

    Student: adapted TE encodes **neu** (scale +1).
    Teacher: stopgrad frozen TE encodes **pos** (disable_adapter / scale 0).
    Loss is layer-weighted MSE + relative L2 (optional 1−cos) on the
    stacked ``[B, seq, layers, dim]`` hidden states. Cosine default is
    0 — live high-cos still had a magnitude gap. Unused-token /
    attribute hold stays light. No Anima structure lock.
    """
    if not getattr(backend, "encoder_lora", False):
        raise ValueError(
            "--lm_target embed needs a TE adapter; pass --lora_targets te"
        )
    unused = unused_words_for(prompt)
    if hasattr(backend, "set_adapter_scale"):
        backend.set_adapter_scale(1.0)
    student, _neu_tokens = backend.encode_text(prompt.neutral)
    try:
        teacher, _pos_tokens = backend.encode_text(prompt.positive, frozen=True)
    except TypeError:
        teacher, _pos_tokens = backend.encode_text(prompt.positive)
    teacher = teacher.detach()
    embed_mse = krea_embed_mse(student, teacher)
    embed_cos = krea_embed_cosine(student, teacher)
    loss = krea_embed_uni_loss(
        student,
        teacher,
        cosine_weight=float(cosine_weight),
        rel_l2_weight=float(rel_l2_weight),
    )

    # Hold unused tokens frozen-pos vs frozen-neu. Do not encode plus
    # with the student adapter — that trains hold through Eθ(pos)
    # instead of the UNI student Eθ(neu).
    try:
        pos_embeds, pos_tokens = backend.encode_text(prompt.positive, frozen=True)
        neu_embeds, neu_tokens = backend.encode_text(prompt.neutral, frozen=True)
    except TypeError:
        if hasattr(backend, "set_adapter_scale"):
            backend.set_adapter_scale(0.0)
        pos_embeds, pos_tokens = backend.encode_text(prompt.positive)
        neu_embeds, neu_tokens = backend.encode_text(prompt.neutral)
        if hasattr(backend, "set_adapter_scale"):
            backend.set_adapter_scale(1.0)
    unused_mask = krea_unused_hold_mask(pos_tokens, neu_tokens, unused)
    hold = krea_unused_hold_loss(
        pos_embeds, neu_embeds, pos_tokens, neu_tokens, unused_mask
    )
    if float(hold_weight) > 0.0:
        loss = loss + float(hold_weight) * hold

    stats = {
        "loss": float(loss.detach()),
        "hold": float(hold.detach()),
        "embed_mse": float(embed_mse.detach()),
        "embed_cos": float(embed_cos.detach()),
        "lm_target": "embed",
        "minus_teacher": 0.0,
        "concept_words": ",".join(
            sorted(krea_concept_words(prompt.positive, prompt.neutral))
        ),
    }
    stats.update(krea_embed_train_stats(student, teacher))
    return loss, stats


def krea_step_loss(
    backend: DummyKreaBackend,
    prompt: KreaSliderPrompt,
    z: torch.Tensor,
    *,
    guidance: float,
    hold_weight: float = KREA_HOLD_WEIGHT,
    lm_target: str = KREA_LM_TARGET_DEFAULT,
    recipe: str = KREA_RECIPE_DEFAULT,
    embed_cosine_weight: float = KREA_EMBED_COSINE_WEIGHT,
    embed_rel_l2_weight: float = KREA_EMBED_REL_L2_WEIGHT,
) -> tuple[torch.Tensor, dict[str, float]]:
    """One UNI step. Embed path never scores DiT velocity."""
    resolved = resolve_krea_lm_target(lm_target, recipe)
    if resolved == "embed":
        return krea_embed_step_loss(
            backend,
            prompt,
            hold_weight=hold_weight,
            cosine_weight=float(embed_cosine_weight),
            rel_l2_weight=float(embed_rel_l2_weight),
        )
    unused = unused_words_for(prompt)
    with torch.no_grad():
        v_pos = backend.predict_v(prompt.positive, z, scale=0.0)
        v_neu = backend.predict_v(prompt.neutral, z, scale=0.0)
        v_uncond = backend.predict_v(prompt.unconditional, z, scale=0.0)
        v_neg = (
            backend.predict_v(prompt.negative, z, scale=0.0)
            if prompt.negative
            else None
        )
    tgt_plus, tgt_zero = krea_plus_neu_teachers(
        v_pos, v_neu, v_uncond, guidance=guidance
    )
    pred_plus = backend.predict_v(
        prompt.positive,
        z,
        scale=1.0,
        pin_unused=True,
        neu_prompt=prompt.neutral,
        unused_words=unused,
    )
    pred_zero = backend.predict_v(prompt.neutral, z, scale=0.0)
    loss = krea_plus_neu_loss(pred_plus, tgt_plus, pred_zero, tgt_zero)

    # Hold unused student TE tokens to frozen encode(neu). When TE is
    # frozen this is a constant (logged only). When TE trains, scale-1
    # pos embeds must keep a graph; neu is the adapter-off reference.
    if getattr(backend, "encoder_lora", False) and hasattr(backend, "set_adapter_scale"):
        backend.set_adapter_scale(1.0)
    pos_embeds, pos_tokens = backend.encode_text(prompt.positive)
    try:
        neu_embeds, neu_tokens = backend.encode_text(prompt.neutral, frozen=True)
    except TypeError:
        neu_embeds, neu_tokens = backend.encode_text(prompt.neutral)
    unused_mask = krea_unused_hold_mask(pos_tokens, neu_tokens, unused)
    hold = krea_unused_hold_loss(
        pos_embeds, neu_embeds, pos_tokens, neu_tokens, unused_mask
    )
    # The pin is applied on the +1 student condition. The hold term scores
    # unused pos-token embeds against encode(neu) so a free encoder cannot
    # move unused attributes. Concept words are excluded by the mask.
    if float(hold_weight) > 0.0:
        loss = loss + float(hold_weight) * hold

    stats = {
        "loss": float(loss.detach()),
        "hold": float(hold.detach()),
        "cfg_dir_norm": float(krea_cfg_direction(v_pos, v_uncond).norm().detach()),
        "concept_words": ",".join(sorted(krea_concept_words(prompt.positive, prompt.neutral))),
        "minus_teacher": 0.0,
    }
    if v_neg is not None:
        canary = krea_minus_canary(v_neg, v_uncond)
        stats["canary_minus_norm"] = float(canary.norm().detach())
    return loss, stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", type=str, default="krea-age")
    parser.add_argument("--rank", type=int, default=KREA_DEFAULT_RANK)
    parser.add_argument("--resolution", type=int, default=KREA_DEFAULT_RESOLUTION)
    parser.add_argument("--model_id", type=str, default=KREA_RAW_MODEL)
    parser.add_argument("--prompts_file", type=str, default=str(DEFAULT_PROMPTS))
    parser.add_argument("--config_file", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--save_dir", type=str, default=str(DEFAULT_SAVE_DIR))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--sample_steps", type=int, default=None)
    parser.add_argument(
        "--sample_guidance",
        type=float,
        default=None,
        help="sample CFG. Velocity UNI defaults to Raw 4.5 / Turbo 0. "
        f"--lm_target embed defaults to {KREA_EMBED_SAMPLE_CFG:g} "
        "(Raw CFG 4.5 fights TE-only). Explicit value always wins.",
    )
    parser.add_argument(
        "--hold_weight",
        type=float,
        default=KREA_HOLD_WEIGHT,
        help="unused-token hold → encode(neu). Age yaml keeps 1.0. "
        f"Smile/happy: pass {KREA_SMILE_HOLD_WEIGHT} so hold does not "
        "dominate the logged loss (live smile-krea: hold≈7.31 of ≈7.35).",
    )
    parser.add_argument(
        "--lora_targets",
        type=str,
        default=KREA_DEFAULT_LORA_TARGETS,
        help="dit (default: transformer to_q/k/v/out), te / text_encoder "
        "(Qwen3-VL q_proj/k_proj/v_proj/o_proj), or dit+te. "
        f"Choices: {', '.join(KREA_LORA_TARGET_CHOICES)} plus aliases.",
    )
    parser.add_argument(
        "--lm_target",
        type=str,
        default=KREA_LM_TARGET_DEFAULT,
        help="v (default): velocity UNI on DiT (+1 → v(pos), 0 → v(neu)). "
        "embed: TE-only stacked-embed UNI — student E_θ(neu) → "
        "stopgrad E_frozen(pos) on [B,seq,layers,dim]. Live gap: "
        "DiT v neu/plus cos≈0.9999 (useless); TE embeds cos≈0.67. "
        f"Choices/aliases: {', '.join(KREA_LM_TARGET_CHOICES)}.",
    )
    parser.add_argument(
        "--embed_cosine_weight",
        type=float,
        default=KREA_EMBED_COSINE_WEIGHT,
        help="embed UNI: weight on (1−cos) added to layer-weighted MSE. "
        "Default 0 (MSE + rel-L2 only). Cosine hid the live magnitude "
        "gap (cos≈0.9959, max_abs≈147). Ignored when --lm_target v.",
    )
    parser.add_argument(
        "--embed_rel_l2_weight",
        type=float,
        default=KREA_EMBED_REL_L2_WEIGHT,
        help="embed UNI: weight on mean_L ||s−t||² / (||t||²+eps) so "
        "layer scale cannot drift. Default 1. 0 disables. Ignored "
        "when --lm_target v.",
    )
    parser.add_argument(
        "--recipe",
        choices=list(KREA_RECIPE_CHOICES),
        default=KREA_RECIPE_DEFAULT,
        help="uni: velocity +1 → + concept, 0 → neu, no minus teacher, "
        "unused-token hold. embed_uni: alias for --lm_target embed "
        "(TE-only; DiT frozen). Not Music 3 lyric-hold.",
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="tiny CPU backend, 2 steps, never loads Krea / Hub weights",
    )
    parser.add_argument(
        "--allow_hub",
        action="store_true",
        help="permit a Hub download of krea/Krea-2-Raw (off; CI must not set this)",
    )
    parser.add_argument(
        "--control_prompt",
        type=str,
        default=None,
        help="verify-only fruit-bowl caption; never a teacher",
    )
    parser.add_argument(
        "--sample_seed",
        type=int,
        default=42,
        help="seed for the end-of-train smile-first scale grid",
    )
    parser.add_argument(
        "--load_te_lora",
        type=str,
        default=None,
        help="load a saved te_lora and emit sample + oracle grids without "
        "training (resmoke / apply audit). Dummy records the path only.",
    )
    parser.add_argument(
        "--te_dit_mask",
        type=str,
        default=KREA_TE_DIT_MASK_DEFAULT,
        choices=list(KREA_TE_DIT_MASK_CHOICES),
        help="DiT encoder mask when TE LoRA scale > 0. auto/ones attend "
        "the full padded stack (v5; UNI-matched smile slots past neu "
        "length). tokenizer is the old neu span. transplant uses "
        "--mask_prompt / plus caption at oracle time. Scale 0 / frozen "
        "TE always keep the real tokenizer mask.",
    )
    return parser.parse_args(argv)


def infer_sample_prompts(
    prompts: Sequence[KreaSliderPrompt],
    control_prompt: str = KREA_CONTROL_PROMPT,
) -> list[str]:
    """Neu / infer captions plus the fruit-bowl control. Never the + concept."""
    seen: list[str] = []
    for prompt in prompts:
        caption = (prompt.neutral or prompt.target or "").strip()
        if caption and caption not in seen:
            seen.append(caption)
    control = str(control_prompt or "").strip()
    if control and control not in seen:
        seen.append(control)
    return seen


def infer_oracle_pairs(
    prompts: Sequence[KreaSliderPrompt],
) -> list[tuple[str, str]]:
    """Unique (neu, plus) pairs. Fruit-bowl control is not a pair."""
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for prompt in prompts:
        neu = (prompt.neutral or prompt.target or "").strip()
        plus = (prompt.positive or "").strip()
        if not neu or not plus or neu in seen:
            continue
        seen.add(neu)
        pairs.append((neu, plus))
    return pairs


def sample_embed_cos(backend, neu: str, plus: str) -> float:
    """cos(Eθ(neu)@1, E_frozen(pos)) at sample time."""
    if hasattr(backend, "set_adapter_scale"):
        backend.set_adapter_scale(1.0)
    student, _ = backend.encode_text(neu)
    try:
        teacher, _ = backend.encode_text(plus, frozen=True)
    except TypeError:
        if hasattr(backend, "set_adapter_scale"):
            backend.set_adapter_scale(0.0)
        teacher, _ = backend.encode_text(plus)
        if hasattr(backend, "set_adapter_scale"):
            backend.set_adapter_scale(1.0)
    return float(krea_embed_cosine(student, teacher).detach())


def emit_oracle_grid(
    backend,
    args: argparse.Namespace,
    save_dir: Path,
    prompts: Sequence[KreaSliderPrompt],
    *,
    dummy: bool,
    card: dict[str, float | int | str],
) -> list[dict[str, Any]]:
    """Apply-audit grid: frozen plus vs student neu@1 vs neu@0.

    Per neu prompt:
    - ``oracle_plus_frozen`` = generate(plus, scale=0 / TE disabled)
    - ``student_neu_scale1`` = generate(neu, scale=1)
    - ``neu_scale0`` = generate(neu, scale=0)

    Logs ``embed_cos`` = cos(Eθ(neu)@1, E_frozen(pos)). If oracle has
    teeth and student does not despite cos>0.95 → remaining apply bug
    (v5: neu tokenizer mask may still hide UNI-matched smile slots —
    default student uses ones-mask). If oracle also lacks teeth →
    caption teacher is weak in pixels.
    """
    pairs = infer_oracle_pairs(prompts)
    out_dir = Path(save_dir) / "samples" / "oracle"
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = 2 if dummy else int(card["sample_steps"])
    guidance = float(card["sample_guidance"])
    seed = int(getattr(args, "sample_seed", 42))
    height = 64 if dummy else int(args.resolution)
    width = height
    te_dit_mask = str(getattr(args, "te_dit_mask", KREA_TE_DIT_MASK_DEFAULT))
    records: list[dict[str, Any]] = []
    pair_meta: list[dict[str, Any]] = []
    shots = (
        ("oracle_plus_frozen", "plus", 0.0),
        ("student_neu_scale1", "neu", 1.0),
        ("neu_scale0", "neu", 0.0),
    )
    assert tuple(tag for tag, _src, _s in shots) == KREA_ORACLE_SHOTS
    emit_mask_ab = bool(getattr(args, "load_te_lora", None))

    def _save_shot(
        *,
        neu: str,
        plus: str,
        tag: str,
        caption: str,
        scale: float,
        embed_cos: float,
        te_mask: str,
        mask_prompt: str | None,
    ) -> None:
        gen_kwargs: dict[str, Any] = {
            "seed": seed,
            "num_steps": steps,
            "guidance": guidance,
            "scale": float(scale),
            "height": height,
            "width": width,
        }
        try:
            image = backend.generate(
                caption,
                te_dit_mask=te_mask,
                mask_prompt=mask_prompt,
                **gen_kwargs,
            )
        except TypeError:
            image = backend.generate(caption, **gen_kwargs)
        name = f"{_slug(neu)}_{tag}.png"
        path = out_dir / name
        image.save(path)
        records.append(
            {
                "tag": tag,
                "prompt": caption,
                "neutral": neu,
                "positive": plus,
                "scale": float(scale),
                "path": f"oracle/{name}",
                "seed": seed,
                "sample_steps": steps,
                "cfg": guidance,
                "embed_cos": embed_cos,
                "te_dit_mask": te_mask,
                "mask_prompt": mask_prompt,
                "height": int(getattr(image, "height", height)),
                "width": int(getattr(image, "width", width)),
            }
        )

    for neu, plus in pairs:
        embed_cos = sample_embed_cos(backend, neu, plus)
        pair_meta.append(
            {
                "neutral": neu,
                "positive": plus,
                "embed_cos": embed_cos,
                "embed_cos_vs_threshold": embed_cos - float(KREA_ORACLE_EMBED_COS),
            }
        )
        for tag, source, scale in shots:
            caption = plus if source == "plus" else neu
            shot_mask = te_dit_mask
            shot_prompt = plus if (source == "neu" and float(scale) > 0 and te_dit_mask == "transplant") else None
            _save_shot(
                neu=neu,
                plus=plus,
                tag=tag,
                caption=caption,
                scale=float(scale),
                embed_cos=embed_cos,
                te_mask=shot_mask,
                mask_prompt=shot_prompt,
            )
        if emit_mask_ab:
            assert KREA_ORACLE_MASK_AB_SHOTS == (
                "student_neu_scale1_onesmask",
                "student_neu_scale1_tokmask",
                "student_neu_scale1_plusmask",
            )
            for tag, te_mask, mask_prompt in (
                ("student_neu_scale1_onesmask", "ones", None),
                ("student_neu_scale1_tokmask", "tokenizer", None),
                ("student_neu_scale1_plusmask", "transplant", plus),
            ):
                _save_shot(
                    neu=neu,
                    plus=plus,
                    tag=tag,
                    caption=neu,
                    scale=1.0,
                    embed_cos=embed_cos,
                    te_mask=te_mask,
                    mask_prompt=mask_prompt,
                )
    meta_path = out_dir / "oracle_meta.json"
    payload = {
        "kind": "oracle_apply_audit",
        "dummy": bool(dummy),
        "seed": seed,
        "cfg": guidance,
        "shots": list(KREA_ORACLE_SHOTS),
        "te_dit_mask": te_dit_mask,
        "mask_ab": list(KREA_ORACLE_MASK_AB_SHOTS) if emit_mask_ab else [],
        "embed_cos_threshold": float(KREA_ORACLE_EMBED_COS),
        "readout": krea_oracle_readout(),
        "pairs": pair_meta,
        "samples": records,
    }
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if pairs and not records:
        raise RuntimeError("Krea oracle apply-audit grid is empty")
    return records


def emit_inprocess_samples(
    backend,
    args: argparse.Namespace,
    save_dir: Path,
    prompts: Sequence[KreaSliderPrompt],
    *,
    dummy: bool,
    control_prompt: str,
    card: dict[str, float | int | str],
) -> list[dict[str, Any]]:
    """Smile-first scale grid: 0 / 0.25 / 0.5 / 1.0 on neu + fruit.

    Writes PNGs under ``save_dir/samples``. No crop-purity / same_crop
    metric — entanglement is accepted. Dummy writes structured ramps.
    """
    sample_prompts = infer_sample_prompts(prompts, control_prompt)
    scales = list(KREA_SAMPLE_SCALES)
    out_dir = Path(save_dir) / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = 2 if dummy else int(card["sample_steps"])
    guidance = float(card["sample_guidance"])
    seed = int(getattr(args, "sample_seed", 42))
    height = 64 if dummy else int(args.resolution)
    width = height
    te_dit_mask = str(getattr(args, "te_dit_mask", KREA_TE_DIT_MASK_DEFAULT))
    records: list[dict[str, Any]] = []
    for prompt in sample_prompts:
        for scale in scales:
            gen_kwargs = dict(
                seed=seed,
                num_steps=steps,
                guidance=guidance,
                scale=float(scale),
                height=height,
                width=width,
            )
            try:
                image = backend.generate(
                    prompt, te_dit_mask=te_dit_mask, **gen_kwargs
                )
            except TypeError:
                image = backend.generate(prompt, **gen_kwargs)
            slug = _slug(prompt)
            scale_tag = f"{scale:g}".replace("-", "m")
            name = f"final_{slug}_scale{scale_tag}.png"
            path = out_dir / name
            image.save(path)
            records.append(
                {
                    "prompt": prompt,
                    "scale": float(scale),
                    "path": str(path.name),
                    "seed": seed,
                    "sample_steps": steps,
                    "cfg": guidance,
                    "height": int(getattr(image, "height", height)),
                    "width": int(getattr(image, "width", width)),
                    "control": prompt == control_prompt,
                }
            )
    meta_path = out_dir / "final_meta.json"
    payload = {
        "dummy": bool(dummy),
        "seed": seed,
        "scales": scales,
        "prompts": sample_prompts,
        "gate": "smile-first",
        "crop_purity": False,
        "samples": records,
    }
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not records:
        raise RuntimeError("in-process Krea sample grid is empty")
    return records


def _slug(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "-" for ch in text.lower())
    return "-".join(part for part in cleaned.split("-") if part)[:48] or "prompt"


def _sample_z(backend, device: torch.device) -> torch.Tensor:
    if hasattr(backend, "sample_latents"):
        return backend.sample_latents(device)
    return torch.randn(1, backend.dim, device=device)


def train(args: argparse.Namespace) -> Path:
    assert_krea_only(args.model_id)
    prompts_path = Path(args.prompts_file)
    if not prompts_path.is_absolute():
        candidate = _REPO_ROOT / prompts_path
        prompts_path = candidate if candidate.exists() else Path(args.prompts_file)
    prompts, meta = load_prompts(prompts_path)
    lm_target = resolve_krea_lm_target(
        getattr(args, "lm_target", KREA_LM_TARGET_DEFAULT),
        getattr(args, "recipe", KREA_RECIPE_DEFAULT),
    )
    card = resolve_krea_card(
        args.model_id,
        args.sample_steps,
        args.sample_guidance,
        lm_target=lm_target,
        recipe=getattr(args, "recipe", KREA_RECIPE_DEFAULT),
    )
    if getattr(args, "sample_guidance", None) is None:
        args.sample_guidance = float(card["sample_guidance"])
    requested_targets = str(getattr(args, "lora_targets", KREA_DEFAULT_LORA_TARGETS))
    lora_spec = force_krea_embed_lora_targets(
        requested_targets, lm_target=lm_target, recipe=getattr(args, "recipe", None)
    )
    if lm_target == "embed":
        krea_embed_requires_te(lora_spec)
        if requested_targets not in ("te", "text_encoder", "encoder"):
            print(
                f"note: --lm_target embed forces --lora_targets te "
                f"(was {requested_targets!r}); DiT stays frozen"
            )
        args.lora_targets = lora_spec.label
        args.lm_target = lm_target
    steps = int(args.steps)
    if args.dummy:
        steps = min(steps, 2)
        device = torch.device("cpu")
        backend = DummyKreaBackend(
            dim=8,
            rank=min(2, int(args.rank)),
            seed=int(args.seed),
            lora_targets=lora_spec.label,
        )
    else:
        device = torch.device(f"cuda:{int(args.device)}" if torch.cuda.is_available() else "cpu")
        backend = _load_live_backend(args, device)

    skip_train = bool(getattr(args, "load_te_lora", None))
    if skip_train:
        if hasattr(backend, "load_te_adapter"):
            backend.load_te_adapter(args.load_te_lora)
        steps = 0

    torch.manual_seed(int(args.seed))
    params = backend.trainable_parameters()
    opt = torch.optim.AdamW(params, lr=float(args.lr))
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / f"{args.name}_train.jsonl"
    last_stats: dict[str, float] = {}
    guidance = float(card["sample_guidance"])
    control_prompt = str(
        getattr(args, "control_prompt", None) or meta.control_prompt or KREA_CONTROL_PROMPT
    )

    print(
        f"train krea slider name={args.name} recipe={args.recipe} "
        f"lm_target={lm_target} "
        f"rank={args.rank} res={args.resolution} model={args.model_id} "
        f"variant={card['variant']} sample_steps={card['sample_steps']} "
        f"cfg={card['sample_guidance']} dummy={bool(args.dummy)} "
        f"allow_hub={bool(getattr(args, 'allow_hub', False))} "
        f"lora_targets={lora_spec.label} hold_weight={float(args.hold_weight)} "
        f"minus_teacher=off unused_hold=on control={control_prompt!r}"
    )
    if lm_target == "embed":
        print(
            f"embed UNI cosine_weight={float(getattr(args, 'embed_cosine_weight', KREA_EMBED_COSINE_WEIGHT)):g} "
            f"rel_l2_weight={float(getattr(args, 'embed_rel_l2_weight', KREA_EMBED_REL_L2_WEIGHT)):g} "
            "(default MSE + rel-L2; cosine hid live magnitude)"
        )
    if meta.bare_captions and abs(float(args.hold_weight) - KREA_HOLD_WEIGHT) < 1e-12:
        print(
            f"note: happy/smile yaml with default --hold_weight {KREA_HOLD_WEIGHT:g}; "
            f"smile-krea-v2 uses --hold_weight {KREA_SMILE_HOLD_WEIGHT:g} so "
            "hold does not dominate the logged loss"
        )

    progress = tqdm(range(steps), desc="train-krea")
    for step in progress:
        prompt = prompts[step % len(prompts)]
        if hasattr(backend, "begin_step"):
            backend.begin_step()
        z = _sample_z(backend, device)
        loss, stats = krea_step_loss(
            backend,
            prompt,
            z,
            guidance=guidance,
            hold_weight=float(args.hold_weight),
            lm_target=lm_target,
            recipe=str(getattr(args, "recipe", KREA_RECIPE_DEFAULT)),
            embed_cosine_weight=float(
                getattr(args, "embed_cosine_weight", KREA_EMBED_COSINE_WEIGHT)
            ),
            embed_rel_l2_weight=float(
                getattr(args, "embed_rel_l2_weight", KREA_EMBED_REL_L2_WEIGHT)
            ),
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last_stats = stats
        progress.set_postfix({"loss": f"{stats['loss']:.4f}"})
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"step": step, **stats}) + "\n")

    samples = emit_inprocess_samples(
        backend,
        args,
        save_dir,
        prompts,
        dummy=bool(args.dummy),
        control_prompt=control_prompt,
        card=card,
    )
    oracle = emit_oracle_grid(
        backend,
        args,
        save_dir,
        prompts,
        dummy=bool(args.dummy),
        card=card,
    )
    adapter_dir = save_dir / f"{args.name}_lora"
    if hasattr(backend, "save_trained") and not args.dummy and not skip_train:
        backend.save_trained(adapter_dir)
    dit_lora_path = None
    te_lora_path = None
    if lora_spec.train_dit and lora_spec.train_te:
        dit_lora_path = f"{args.name}_lora/dit_lora"
        te_lora_path = f"{args.name}_lora/te_lora"
    elif lora_spec.train_dit:
        dit_lora_path = f"{args.name}_lora"
    elif lora_spec.train_te:
        te_lora_path = f"{args.name}_lora/te_lora"

    sidecar = {
        "kind": "krea",
        "recipe": "embed_uni" if lm_target == "embed" else "uni",
        "lm_target": lm_target,
        "embed_cosine_weight": float(
            getattr(args, "embed_cosine_weight", KREA_EMBED_COSINE_WEIGHT)
        )
        if lm_target == "embed"
        else None,
        "embed_rel_l2_weight": float(
            getattr(args, "embed_rel_l2_weight", KREA_EMBED_REL_L2_WEIGHT)
        )
        if lm_target == "embed"
        else None,
        "dit_velocity_supervised": lm_target != "embed",
        "name": args.name,
        "model_id": args.model_id,
        "rank": int(args.rank),
        "resolution": int(args.resolution),
        "official": "train LoRAs on Raw, run on Turbo",
        "minus_teacher": False,
        "minus_canary": True,
        "token_hold": "unused_to_neu",
        "lyric_hold": False,
        "dummy": bool(args.dummy),
        "allow_hub": bool(getattr(args, "allow_hub", False)),
        "lora_targets": lora_spec.label,
        "dit_lora": lora_spec.train_dit,
        "te_lora": lora_spec.train_te,
        "dit_lora_path": dit_lora_path,
        "te_lora_path": te_lora_path,
        "dit_lora_targets": lora_spec.dit_lora_targets,
        "te_lora_targets": lora_spec.te_lora_targets,
        "adapted_modules": lora_spec.adapted_module_names,
        "frozen_modules": list(lora_spec.frozen_modules),
        "encoder_lora": lora_spec.encoder_lora,
        "dit_lora_only": lora_spec.dit_lora_only,
        "te_parking": lora_spec.te_parking,
        "hold_weight": float(args.hold_weight),
        "plus_label": meta.plus_label,
        "minus_label": meta.minus_label,
        "concept_words": meta.concept_words,
        "control_prompt": control_prompt,
        "bare_captions": bool(meta.bare_captions),
        "recommended_range": meta.recommended_range,
        "sample_grid": {
            "scales": list(KREA_SAMPLE_SCALES),
            "gate": "smile-first",
            "crop_purity": False,
            "count": len(samples),
            "dir": "samples",
        },
        "oracle_grid": {
            "dir": "samples/oracle",
            "shots": list(KREA_ORACLE_SHOTS),
            "embed_cos_threshold": float(KREA_ORACLE_EMBED_COS),
            "count": len(oracle),
            "readout": krea_oracle_readout(),
        },
        "cfg_uncond_te_frozen": bool(lora_spec.encoder_lora),
        "encode_once": True,
        "te_dit_mask": str(getattr(args, "te_dit_mask", KREA_TE_DIT_MASK_DEFAULT)),
        "te_dit_ones_mask": bool(lora_spec.encoder_lora),
        "load_te_lora": getattr(args, "load_te_lora", None),
        "skipped_train": bool(skip_train),
        "last": last_stats,
        **card,
    }
    sidecar_path = save_dir / f"{args.name}_last.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    print(f"wrote {sidecar_path}")
    return sidecar_path


def _load_live_backend(args: argparse.Namespace, device: torch.device):
    """Lazy live loader. Tests never call this. No Anima / ZiT / H3.

    CI uses --dummy. This import is the only path that may touch Hub,
    and only when ``--allow_hub`` is set (Anima/Sana offline-safe default).
    """
    from conceptmod.textsliders.krea_live import load_live_krea_backend

    return load_live_krea_backend(args, device)


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
