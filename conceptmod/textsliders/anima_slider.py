"""Anima image-slider geometry: UNI + unused-token hold.

Image analog of Music 3 UNI (not lyric-hold — images have no lyrics /
``<|audio_start|>``):

- student +1 fits the + concept prompt (CFG / trajectory teacher)
- student scale 0 fits the neutral prompt
- live smile uses ``--lm_target trajectory`` (1-step v-space gap is tiny)
- caption-only plus teacher jumps crop (full-body→close-up). Opt-in
  ``--teacher same_crop`` / ``--lm_target same_crop`` inverts the frozen
  neu traj and denoises plus from mid-σ so expression moves without zoom
- no minus teacher unless the yaml still declares one as a canary
- hold unused prompt tokens / attributes (subject, composition, pins)
  to encode(neu); do **not** hold the concept words
- attributes are unused pins only — never prefixed onto captions
  (train infer/neu == sample ``pipe(prompt=...)``)

Velocity-space CFG is conceptmod's ``v(z, t, c) − v(z, t, '')``.

CPU-pure. No Hub, no GPU, no Anima weights. Does not change the Music 3
trainer default (``--lm_target v9`` / ``--pole_mode hidden``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml

DEFAULT_MODEL_ID = "circlestone-labs/Anima-Base-v1.0-Diffusers"
DEFAULT_RANK = 16
DEFAULT_ALPHA = 16.0
DEFAULT_RESOLUTION = 768
DEFAULT_SAMPLE_STEPS = 40
DEFAULT_CFG = 4.0
DEFAULT_HOLD_WEIGHT = 1.0
# DiT LoRA: 1e-2 blew up the prior RunPod adapter (loss looked fine,
# any nonzero scale collapsed denoise to RGB noise). Overridable.
DEFAULT_LR = 1e-4
DEFAULT_CONTROL_PROMPT = "a bowl of fruit on a table"
DEFAULT_SAMPLE_SCALES = (0.0, 0.25, 0.5, 1.0)
DEFAULT_SAMPLE_SEED = 42
# Live mid-train PEFT grid. End-of-train gate always runs.
DEFAULT_SAMPLE_EVERY = 100
# peft_pipe: ModularPipeline pipe(prompt=...) after sync_peft (default).
# train_faithful: backend.encode_text + transformer denoise (same as loss).
ANIMA_SAMPLE_MODES = ("peft_pipe", "train_faithful")
DEFAULT_SAMPLE_MODE = "peft_pipe"
# v4 1-step direct / cfg_delta cannot carry Anima expression (v-space
# pos/neu gap is microscopic). Live default is short-trajectory MSE.
# same_crop is trajectory loss + invert/img2img plus teacher (crop lock).
ANIMA_LM_TARGETS = ("direct", "cfg_delta", "trajectory", "same_crop")
DEFAULT_LM_TARGET = "trajectory"
DEFAULT_TRAJ_STEPS = 4
DEFAULT_TRAJ_IDENTITY_WEIGHT = 0.25
# Off: values <= 1 leave the 1-step teacher unamplified.
DEFAULT_TEACHER_GAP_BOOST = 1.0
# caption: denoise plus from the same z_T (already true) — stock still
# zooms when the plus caption says teeth. same_crop: invert x_neu and
# continue under plus at --teacher_strength so crop stays.
ANIMA_TEACHERS = ("caption", "same_crop")
DEFAULT_TEACHER = "caption"
DEFAULT_TEACHER_STRENGTH = 0.5
# Dummy DiT: concept tokens write expression on channel 1 always, and
# crop/zoom on channel 2 only at high σ (early denoise). That is the
# stock "teeth → close-up" analog the same-crop teacher is meant to skip.
FAKE_EXPR_CHANNEL = 1
FAKE_CROP_CHANNEL = 2
# Turbo v1.1 is preview-only. Train stays on Base. Official CircleStone
# card: CFG 1, 8–12 steps. This repo's sample helper uses 10.
TURBO_PREVIEW_ONLY = True
TURBO_COMFY_REPO = "circlestone-labs/Anima"
TURBO_TRANSFORMER_FILE = "split_files/diffusion_models/anima-turbo-v1.1.safetensors"
TURBO_TEXT_ENCODER_FILE = "split_files/text_encoders/qwen_3_06b_base.safetensors"
TURBO_VAE_FILE = "split_files/vae/qwen_image_vae.safetensors"
TURBO_DIFFUSERS_OUTPUT = "Anima-Turbo-v1.1-Diffusers"
TURBO_SAMPLE_CFG = 1.0
TURBO_SAMPLE_STEPS = 10
TURBO_SAMPLE_STEPS_RANGE = (8, 12)
TURBO_CONVERT_DTYPE = "bf16"
TURBO_LICENSE = "CircleStone Labs Non-Commercial (NC)"
TURBO_CONVERT_SCRIPT = (
    "https://raw.githubusercontent.com/huggingface/diffusers/main/"
    "scripts/convert_anima_to_diffusers.py"
)
TURBO_CONVERT_COSMOS_SCRIPT = (
    "https://raw.githubusercontent.com/huggingface/diffusers/main/"
    "scripts/convert_cosmos_to_diffusers.py"
)
# Community Anima-1.0-Turbo-Diffusers is v1.0 only and the wrong VAE
# class. Do not train or convert from it.
TURBO_IGNORE_COMMUNITY = "Anima-1.0-Turbo-Diffusers"
# DiT attn names (CosmosTransformer3DModel). Conditioner uses q/k/v/o_proj.
LORA_TARGETS = ("to_q", "to_k", "to_v", "to_out.0")
DIT_LORA_TARGETS = LORA_TARGETS
CONDITIONER_LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
ANIMA_LORA_TARGET_CHOICES = ("dit", "conditioner", "dit+conditioner")
# Smile default-on path: PEFT on AnimaTextConditioner (~269M), not Qwen3.
# dit is the old transformer-only recipe (v1–v5). Not silent: --print_card
# and docs/anima-slider.md state which modules are trained.
DEFAULT_LORA_TARGETS = "conditioner"
# Legacy dit-only freeze list. Prefer AnimaLoraSpec.frozen_modules.
FROZEN_MODULES = ("text_conditioner",)
_TOKEN_RE = re.compile(r"[a-z0-9]+")
# Uniform uint8 RGB static is ~127.5 / 73.9. The dead RunPod LoRA
# printed ~122 / 75. Structured images are spatially correlated.
_NOISE_MEAN_LO = 110.0
_NOISE_MEAN_HI = 145.0
_NOISE_STD_LO = 68.0
_NOISE_STD_HI = 82.0
_NOISE_CORR_MAX = 0.15


# v3 captions: closed-mouth neu vs hard-smile plus. Stock Anima already
# soft-smiles on a bare "a woman sitting on a chair", so UNI plus= vs
# that neu had almost no teacher gap. + is CFG teacher only; student +1
# stays on neu/infer (#62 analog).
WOMAN_NEU = "a woman sitting on a chair, neutral expression, closed mouth"
WOMAN_PLUS = (
    "a woman sitting on a chair, big smile showing teeth, happy joyful expression"
)
MAN_NEU = "a man reading at a table, neutral expression, closed mouth"
MAN_PLUS = (
    "a man reading at a table, big smile showing teeth, happy joyful expression"
)
DEFAULT_CONCEPT_WORDS = "smiling, smile, happy, joyful, teeth"


@dataclass
class AnimaSliderRow:
    target: str
    positive: str
    neutral: str
    negative: str = ""
    attributes: list[str] = field(default_factory=list)
    action: str = "enhance"
    guidance_scale: float = DEFAULT_CFG
    resolution: int = DEFAULT_RESOLUTION
    batch_size: int = 1
    pins: list[str] = field(default_factory=list)
    concept_words: str = ""

    @property
    def has_minus_canary(self) -> bool:
        return bool(str(self.negative).strip())

    @property
    def infer_prompt(self) -> str:
        """Slider inference prompt: target, else neu."""
        return self.target.strip() or self.neutral


@dataclass
class AnimaPromptsMeta:
    plus_label: str = ""
    minus_label: str = ""
    recommended_range: list[float] = field(default_factory=lambda: [-2.0, 2.0])
    concept_words: str = ""


def word_tokens(text: str) -> list[str]:
    """Whitespace / alnum tokenizer. Images have no lyric special tokens."""
    return _TOKEN_RE.findall((text or "").lower())


def parse_concept_words(raw: str | Iterable[str] | None) -> list[str]:
    """Split declared concept_words. Never held to encode(neu)."""
    if raw is None:
        return []
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.replace(",", " ").split() if p.strip()]
    else:
        parts = [str(p).strip() for p in raw if str(p).strip()]
    return word_tokens(" ".join(parts))


def unused_vocab(
    target: str,
    neutral: str,
    attributes: Iterable[str] | None = None,
    extra_pins: Iterable[str] | None = None,
    concept_words: str | Iterable[str] | None = None,
) -> set[str]:
    """Subject, composition, and pinned attributes — never concept words."""
    vocab = set(word_tokens(target)) | set(word_tokens(neutral))
    for item in list(attributes or []) + list(extra_pins or []):
        vocab.update(word_tokens(str(item)))
    for tok in parse_concept_words(concept_words):
        vocab.discard(tok)
    return vocab


def concept_tokens(positive: str, unused: set[str]) -> list[str]:
    """Tokens in the + prompt that are not unused / pinned."""
    return [tok for tok in word_tokens(positive) if tok not in unused]


def unused_token_mask(tokens: Sequence[str], unused: set[str]) -> list[bool]:
    return [tok in unused for tok in tokens]


def align_unused_positions(
    pos_tokens: Sequence[str],
    neu_tokens: Sequence[str],
    unused: set[str],
) -> list[tuple[int, int]]:
    """Pair unused + tokens with the next matching unused neu token.

    Fail closed (empty) if the + prompt has no unused tokens — the hold
    would otherwise have nothing to pin.
    """
    pairs: list[tuple[int, int]] = []
    neu_idx = 0
    for pos_i, tok in enumerate(pos_tokens):
        if tok not in unused:
            continue
        found = None
        for j in range(neu_idx, len(neu_tokens)):
            if neu_tokens[j] == tok:
                found = j
                break
        if found is None:
            for j, neu_tok in enumerate(neu_tokens):
                if neu_tok == tok:
                    found = j
                    break
        if found is None:
            continue
        pairs.append((pos_i, found))
        neu_idx = found + 1
    return pairs


def splice_unused_embeds(
    pos_embeds: torch.Tensor,
    neu_embeds: torch.Tensor,
    pairs: Sequence[tuple[int, int]],
) -> torch.Tensor:
    """Copy encode(neu) into unused + positions. Concept words stay."""
    held = pos_embeds.clone()
    for pos_i, neu_i in pairs:
        held[..., pos_i, :] = neu_embeds[..., neu_i, :]
    return held


def anima_cfg_delta(v_cond: torch.Tensor, v_uncond: torch.Tensor) -> torch.Tensor:
    """conceptmod velocity-space CFG direction: ``v(z,t,c) − v(z,t,'')``."""
    return v_cond - v_uncond


def anima_uni_teachers(
    v_pos: torch.Tensor,
    v_neu: torch.Tensor,
    v_uncond: torch.Tensor,
    v_neg: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | None]:
    """UNI teachers. Minus is a canary only — never a teacher."""
    del v_neg
    return {
        "plus": anima_cfg_delta(v_pos, v_uncond),
        "zero": anima_cfg_delta(v_neu, v_uncond),
        "minus": None,
    }


def anima_uni_loss(
    student_plus: torch.Tensor,
    teacher_plus: torch.Tensor,
    student_zero: torch.Tensor,
    teacher_zero: torch.Tensor,
    student_minus: torch.Tensor | None = None,
    teacher_minus: torch.Tensor | None = None,
) -> torch.Tensor:
    """``MSE(+) + MSE(0)``. Minus tensors are accepted as a canary and ignored."""
    del student_minus, teacher_minus
    return F.mse_loss(student_plus, teacher_plus) + F.mse_loss(
        student_zero, teacher_zero
    )


def anima_direct_teachers(
    v_pos: torch.Tensor,
    v_neu: torch.Tensor,
) -> dict[str, torch.Tensor | None]:
    """Raw-velocity teachers. Plus is frozen ``v(pos)``; zero is frozen ``v(neu)``."""
    return {"plus": v_pos, "zero": v_neu, "minus": None}


def anima_direct_loss(
    student_plus: torch.Tensor,
    teacher_plus: torch.Tensor,
    student_zero: torch.Tensor,
    teacher_zero: torch.Tensor,
) -> torch.Tensor:
    """``MSE(v(neu, adapter), v(pos, frozen)) + MSE(v(neu, scale 0), v(neu, frozen))``.

    Scale 0 matches neu. Neu+LoRA matches plus velocity. Clearer for
    expression transfer than CFG-delta UNI. Still 1-step: Anima's
    frozen ``v(pos)`` ≈ ``v(neu)`` so this cannot carry a smile.
    """
    return F.mse_loss(student_plus, teacher_plus) + F.mse_loss(
        student_zero, teacher_zero
    )


def anima_boost_teacher(
    v_pos: torch.Tensor,
    v_neu: torch.Tensor,
    boost: float,
) -> torch.Tensor:
    """CFG-amplified 1-step teacher: ``v_neu + boost * (v_pos − v_neu)``.

    ``boost <= 1`` is off (returns ``v_pos``). For ``direct`` / ``cfg_delta``
    only — not a substitute for ``trajectory``.
    """
    scale = float(boost)
    if scale <= 1.0 + 1e-8:
        return v_pos
    return v_neu + scale * (v_pos - v_neu)


def anima_flow_sigmas(
    num_steps: int,
    device=None,
    dtype=None,
) -> torch.Tensor:
    """Flow-match Euler schedule: ``linspace(1, 1/K, K)`` plus terminal 0.

    Matches ``FlowMatchEulerDiscreteScheduler`` custom-sigma usage
    (Music 3 ``rollout_states`` / Anima ModularPipeline). Length ``K+1``.
    """
    steps = int(num_steps)
    if steps < 1:
        raise ValueError(f"traj_steps must be >= 1, got {num_steps!r}")
    vals = torch.linspace(1.0, 1.0 / steps, steps, device=device, dtype=dtype)
    zero = torch.zeros(1, device=vals.device, dtype=vals.dtype)
    return torch.cat([vals, zero], dim=0)


def anima_flow_euler_step(
    sample: torch.Tensor,
    velocity: torch.Tensor,
    sigma: torch.Tensor | float,
    sigma_next: torch.Tensor | float,
) -> torch.Tensor:
    """``FlowMatchEulerDiscreteScheduler.step``: ``x + (σ_next − σ) * v``."""
    return sample + (sigma_next - sigma) * velocity


def anima_num_train_timesteps(backend) -> int:
    """Infer-noise level. Live reads the scheduler; dummy is 1000."""
    n = getattr(backend, "num_train_timesteps", None)
    if n:
        return int(n)
    pipe = getattr(backend, "pipe", None)
    sched = getattr(pipe, "scheduler", None)
    cfg = getattr(sched, "config", None)
    n = getattr(cfg, "num_train_timesteps", None)
    return int(n) if n else 1000


def anima_short_trajectory(
    backend,
    prompt: str,
    z_t: torch.Tensor,
    *,
    num_steps: int,
    frozen: bool = False,
    scale: float | None = None,
    start_index: int = 0,
) -> torch.Tensor:
    """K-step flow Euler over ``predict_v``. Not ModularPipeline denoise.

    ``z_t ~ N(0, I)`` at infer noise (σ=1). Same ``z_t`` + schedule for
    teacher and student. Thin loop so the student path keeps grad;
    ``pipe(...)`` is inference-only and has no grad through the DiT.

    ``start_index`` skips early σ (same-crop invert starts mid-schedule).
    ``z_t`` must already be the state at ``sigmas[start_index]``.
    """
    steps = int(num_steps)
    if steps < 1:
        raise ValueError(f"traj_steps must be >= 1, got {num_steps!r}")
    begin = int(start_index)
    if begin < 0 or begin >= steps:
        raise ValueError(f"start_index must be in [0, {steps}), got {start_index!r}")
    n_train = anima_num_train_timesteps(backend)
    sigmas = anima_flow_sigmas(steps, device=z_t.device, dtype=torch.float32)
    x = z_t
    for i in range(begin, steps):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        t = (sigma * float(n_train)).reshape(1).to(device=z_t.device)
        v = backend.predict_v(prompt, x, t, frozen=frozen, scale=scale)
        x = anima_flow_euler_step(x, v, sigma, sigma_next)
    return x


def anima_flow_invert(
    x0: torch.Tensor,
    noise: torch.Tensor,
    sigma: torch.Tensor | float,
) -> torch.Tensor:
    """Flow-match re-noise: ``x_σ = (1−σ)·x0 + σ·ε``.

    ``σ=1`` is the infer-noise latent; ``σ=0`` is the clean traj end.
    Same-crop uses the training ``z_T`` as ``ε`` so invert is latent-locked.
    """
    if torch.is_tensor(sigma):
        sig = sigma.to(device=x0.device, dtype=x0.dtype)
        while sig.ndim < x0.ndim:
            sig = sig.reshape(sig.shape + (1,))
    else:
        sig = x0.new_tensor(float(sigma))
    return (1.0 - sig) * x0 + sig * noise


def anima_teacher_start_index(
    sigmas: torch.Tensor,
    strength: float,
) -> int:
    """First denoise index whose σ is ``<= strength`` (terminal 0 excluded).

    ``strength=1`` → index 0 (caption-like, from infer noise).
    ``strength=0.5`` on K=4 (σ=1, 0.75, 0.5, 0.25, 0) → index 2.
    Always leaves at least one plus step.
    """
    scale = float(strength)
    if scale <= 0.0:
        raise ValueError(f"teacher_strength must be > 0, got {strength!r}")
    if sigmas.numel() < 2:
        raise ValueError("sigmas must be linspace(1,1/K,K) ∪ {0}")
    n_denoise = int(sigmas.numel()) - 1
    if scale >= 1.0 - 1e-8:
        return 0
    for i in range(n_denoise):
        if float(sigmas[i]) <= scale + 1e-8:
            return i
    return n_denoise - 1


def anima_latent_channel_mean(x: torch.Tensor, channel: int) -> torch.Tensor:
    """Mean of one latent channel. Dummy CHW or live CTHW."""
    if x.ndim < 3:
        raise ValueError(f"latent must be BCHW or BCTHW, got {tuple(x.shape)}")
    if x.shape[1] <= int(channel):
        raise ValueError(f"latent channel {channel} missing in {tuple(x.shape)}")
    plane = x[:, int(channel)]
    return plane.reshape(plane.shape[0], -1).mean(dim=-1)


def anima_fake_crop_code(x: torch.Tensor) -> torch.Tensor:
    """Dummy crop/zoom: reserved site on channel 2 (high-σ write)."""
    if x.ndim == 5:
        return x[:, FAKE_CROP_CHANNEL, 0, 0, 0]
    if x.ndim == 4:
        return x[:, FAKE_CROP_CHANNEL, 0, 0]
    raise ValueError(f"latent must be BCHW or BCTHW, got {tuple(x.shape)}")


def anima_fake_expr_code(x: torch.Tensor) -> torch.Tensor:
    """Dummy expression: original text_out concept pixel (flat index 1)."""
    if x.ndim == 5:
        return x[:, 0, 0, 0, 1]
    if x.ndim == 4:
        return x[:, 0, 0, 1]
    raise ValueError(f"latent must be BCHW or BCTHW, got {tuple(x.shape)}")


def anima_teacher_crop_gap(x_neu: torch.Tensor, x_plus: torch.Tensor) -> float:
    """|crop(plus) − crop(neu)| on the dummy crop channel."""
    return float((anima_fake_crop_code(x_plus) - anima_fake_crop_code(x_neu)).abs().mean())


def anima_teacher_expr_gap(x_neu: torch.Tensor, x_plus: torch.Tensor) -> float:
    """|expr(plus) − expr(neu)| on the dummy expression channel."""
    return float((anima_fake_expr_code(x_plus) - anima_fake_expr_code(x_neu)).abs().mean())


@dataclass(frozen=True)
class AnimaTeacherPair:
    """Frozen neu / plus short trajectories from one ``z_T``."""

    x_neu: torch.Tensor
    x_plus: torch.Tensor
    z_t: torch.Tensor
    teacher: str
    strength: float
    start_index: int
    start_sigma: float
    z_mid: torch.Tensor | None = None
    shared_crop: bool = False


def anima_same_crop_plus(
    backend,
    plus_prompt: str,
    x_neu: torch.Tensor,
    z_t: torch.Tensor,
    *,
    num_steps: int,
    strength: float = DEFAULT_TEACHER_STRENGTH,
    frozen: bool = True,
    scale: float | None = None,
) -> tuple[torch.Tensor, int, float, torch.Tensor]:
    """Img2img / invert: re-noise ``x_neu`` with ``z_T``, denoise under plus.

    Crop is committed by the frozen neu traj. Plus only runs the late
    σ ≤ strength steps, so teeth do not rebuild a close-up from noise.
    No ModularPipeline / Comfy — same ``predict_v`` Euler as trajectory.
    """
    steps = int(num_steps)
    sigmas = anima_flow_sigmas(steps, device=z_t.device, dtype=torch.float32)
    start_i = anima_teacher_start_index(sigmas, strength)
    start_sigma = sigmas[start_i]
    z_mid = anima_flow_invert(x_neu, z_t, start_sigma)
    x_plus = anima_short_trajectory(
        backend,
        plus_prompt,
        z_mid,
        num_steps=steps,
        frozen=frozen,
        scale=scale,
        start_index=start_i,
    )
    return x_plus, start_i, float(start_sigma), z_mid


def anima_teacher_pair(
    backend,
    neu_prompt: str,
    plus_prompt: str,
    z_t: torch.Tensor,
    *,
    num_steps: int,
    teacher: str | None = None,
    strength: float = DEFAULT_TEACHER_STRENGTH,
    frozen: bool = True,
) -> AnimaTeacherPair:
    """Build frozen neu + plus teachers. ``same_crop`` shares neu crop."""
    recipe = resolve_anima_teacher(teacher)
    x_neu = anima_short_trajectory(
        backend, neu_prompt, z_t, num_steps=num_steps, frozen=frozen
    )
    if recipe == "same_crop":
        x_plus, start_i, start_sigma, z_mid = anima_same_crop_plus(
            backend,
            plus_prompt,
            x_neu,
            z_t,
            num_steps=num_steps,
            strength=strength,
            frozen=frozen,
        )
        return AnimaTeacherPair(
            x_neu=x_neu,
            x_plus=x_plus,
            z_t=z_t,
            teacher=recipe,
            strength=float(strength),
            start_index=start_i,
            start_sigma=start_sigma,
            z_mid=z_mid,
            shared_crop=True,
        )
    x_plus = anima_short_trajectory(
        backend, plus_prompt, z_t, num_steps=num_steps, frozen=frozen
    )
    return AnimaTeacherPair(
        x_neu=x_neu,
        x_plus=x_plus,
        z_t=z_t,
        teacher=recipe,
        strength=1.0,
        start_index=0,
        start_sigma=1.0,
        z_mid=None,
        shared_crop=False,
    )


def anima_trajectory_loss(
    x_student: torch.Tensor,
    x_plus: torch.Tensor,
    x_zero: torch.Tensor | None = None,
    x_neu: torch.Tensor | None = None,
    identity_weight: float = DEFAULT_TRAJ_IDENTITY_WEIGHT,
) -> torch.Tensor:
    """``MSE(x_student, x_plus) + λ_id * MSE(x_zero, x_neu)``.

    ``x_plus`` is the frozen plus short trajectory (no grad).
    ``x_student`` is adapter-on, neu/infer, same ``z_T``.
    Identity term is optional / light so scale 0 stays the neu trajectory.
    """
    loss = F.mse_loss(x_student, x_plus)
    weight = float(identity_weight)
    if weight > 0.0 and x_zero is not None and x_neu is not None:
        loss = loss + weight * F.mse_loss(x_zero, x_neu)
    return loss


def resolve_anima_lm_target(lm_target: str | None = None) -> str:
    recipe = str(lm_target or DEFAULT_LM_TARGET).strip().lower()
    if recipe not in ANIMA_LM_TARGETS:
        raise ValueError(
            f"anima lm_target must be one of {ANIMA_LM_TARGETS}, got {lm_target!r}"
        )
    return recipe


def resolve_anima_teacher(teacher: str | None = None) -> str:
    raw = str(teacher if teacher is not None else DEFAULT_TEACHER).strip().lower()
    aliases = {
        "img2img": "same_crop",
        "invert": "same_crop",
        "caption_only": "caption",
    }
    raw = aliases.get(raw, raw)
    if raw not in ANIMA_TEACHERS:
        raise ValueError(
            f"anima teacher must be one of {ANIMA_TEACHERS}, got {teacher!r}"
        )
    return raw


@dataclass(frozen=True)
class AnimaTrainRecipe:
    """Resolved loss + teacher. ``same_crop`` is trajectory loss + invert."""

    lm_target: str
    loss_kind: str
    teacher: str
    teacher_strength: float


def resolve_anima_train_recipe(
    lm_target: str | None = None,
    teacher: str | None = None,
    teacher_strength: float | None = None,
) -> AnimaTrainRecipe:
    """``--lm_target same_crop`` aliases trajectory + invert teacher.

    ``--teacher same_crop`` on ``trajectory`` is the same path.
    ``direct`` / ``cfg_delta`` cannot take a same-crop teacher.
    """
    lm = resolve_anima_lm_target(lm_target)
    explicit = teacher is not None and str(teacher).strip() != ""
    teach = resolve_anima_teacher(teacher if explicit else None)
    if lm == "same_crop":
        if explicit and teach != "same_crop":
            raise ValueError(
                "--lm_target same_crop requires --teacher same_crop "
                f"(got {teacher!r})"
            )
        teach = "same_crop"
        loss = "trajectory"
    else:
        if teach == "same_crop" and lm in ("direct", "cfg_delta"):
            raise ValueError(
                "same_crop teacher requires --lm_target trajectory or same_crop, "
                f"got {lm!r}"
            )
        loss = lm
    strength = float(
        DEFAULT_TEACHER_STRENGTH if teacher_strength is None else teacher_strength
    )
    if teach == "same_crop" and not (0.0 < strength <= 1.0):
        raise ValueError(
            f"teacher_strength must be in (0, 1], got {teacher_strength!r}"
        )
    return AnimaTrainRecipe(
        lm_target=lm,
        loss_kind=loss,
        teacher=teach,
        teacher_strength=strength,
    )


@dataclass(frozen=True)
class AnimaLoraSpec:
    """Which Anima modules receive PEFT LoRA.

    Smile default is ``conditioner`` (AnimaTextConditioner). Qwen3
    ``text_encoder`` is never adapted here: 28 layers / ~1.2GB, VRAM-heavy,
    and the caption-level smile already lives in the 6-layer ~269M
    conditioner that maps Qwen hidden states + T5 ids to Cosmos embeds.
    """

    label: str
    train_dit: bool
    train_conditioner: bool

    @property
    def train_text_conditioner(self) -> bool:
        return self.train_conditioner

    @property
    def frozen_modules(self) -> tuple[str, ...]:
        frozen = ["text_encoder"]
        if not self.train_conditioner:
            frozen.append("text_conditioner")
        return tuple(frozen)

    @property
    def active_attn_targets(self) -> list[str]:
        names: list[str] = []
        if self.train_dit:
            names.extend(DIT_LORA_TARGETS)
        if self.train_conditioner:
            names.extend(CONDITIONER_LORA_TARGETS)
        return names

    @property
    def adapted_module_names(self) -> list[str]:
        names: list[str] = []
        if self.train_dit:
            names.append("transformer")
        if self.train_conditioner:
            names.append("text_conditioner")
        return names


def resolve_anima_lora_targets(lora_targets: str | None = None) -> AnimaLoraSpec:
    """Parse ``dit`` / ``conditioner`` / ``dit+conditioner``.

    Aliases: ``transformer`` → ``dit``, ``text_conditioner`` / ``cond`` →
    ``conditioner``. Qwen ``text_encoder`` is rejected (not a smile target).
    """
    raw = str(lora_targets if lora_targets is not None else DEFAULT_LORA_TARGETS)
    label = raw.strip().lower().replace(" ", "")
    aliases = {
        "transformer": "dit",
        "text_conditioner": "conditioner",
        "cond": "conditioner",
        "dit+text_conditioner": "dit+conditioner",
        "transformer+conditioner": "dit+conditioner",
        "transformer+text_conditioner": "dit+conditioner",
    }
    label = aliases.get(label, label)
    if label not in ANIMA_LORA_TARGET_CHOICES:
        raise ValueError(
            f"anima lora_targets must be one of {ANIMA_LORA_TARGET_CHOICES}, "
            f"got {lora_targets!r}. Qwen text_encoder is not adapted "
            "(28-layer / ~1.2GB); use conditioner for caption-level smile."
        )
    parts = set(label.split("+"))
    return AnimaLoraSpec(
        label=label,
        train_dit="dit" in parts,
        train_conditioner="conditioner" in parts,
    )


def anima_unused_hold_loss(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    pairs: Sequence[tuple[int, int]] | None = None,
    pred_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Masked MSE of unused positions to encode(neu). Concept words skipped.

    ``pred`` / ``tgt`` are ``(..., T, D)``. ``pairs`` maps pred unused
    indices onto tgt unused indices. A boolean ``pred_mask`` over the
    last token axis is also accepted (same-length sequences).
    """
    if pairs is not None:
        if not pairs:
            return pred.reshape(-1)[:1].new_zeros(())
        pred_idx = [p for p, _ in pairs]
        tgt_idx = [n for _, n in pairs]
        return F.mse_loss(pred[..., pred_idx, :], tgt[..., tgt_idx, :])
    if pred_mask is None:
        raise ValueError("anima_unused_hold_loss needs pairs or pred_mask")
    mask = pred_mask.to(dtype=pred.dtype)
    while mask.ndim < pred.ndim:
        mask = mask.unsqueeze(-1)
    denom = mask.sum().clamp_min(1.0)
    return ((pred - tgt).pow(2) * mask).sum() / denom


def minus_canary_cosine(
    student_plus: torch.Tensor,
    v_neg: torch.Tensor,
    v_uncond: torch.Tensor,
) -> torch.Tensor:
    """Log-only: how aligned student +1 is with a declared minus pole."""
    minus = anima_cfg_delta(v_neg, v_uncond).flatten().unsqueeze(0)
    plus = student_plus.flatten().unsqueeze(0)
    return F.cosine_similarity(plus, minus, dim=1, eps=1e-6).mean()


def expand_attributes_anima(row: dict) -> list[dict]:
    """Pin attributes for unused-token hold. Do **not** prefix captions.

    Prefixing (``indoor a woman sitting…``) was the v3 train/sample
    mismatch: live train used the prefixed string, in-process sample
    stripped it back to the bare infer/neu caption. Train and sample
    now share the yaml captions as-is. Attributes stay on the row as
    unused pins only (bookkeeping / hold vocab).
    """
    item = dict(row)
    attributes = [
        str(a).strip() for a in (row.get("attributes") or []) if str(a).strip()
    ]
    pins = [str(p).strip() for p in (row.get("pins") or []) if str(p).strip()]
    for attr in attributes:
        if attr not in pins:
            pins.append(attr)
    item["pins"] = pins
    item["attributes"] = attributes
    return [item]


def _as_row(item: dict) -> AnimaSliderRow:
    if "target" not in item and "positive" not in item:
        raise ValueError(f"anima prompt row needs target or positive: {item!r}")
    target = str(item.get("target") or item.get("neutral") or "")
    positive = str(item.get("positive") or target)
    neutral = str(item.get("neutral") or target)
    attributes = [str(a).strip() for a in (item.get("attributes") or []) if str(a).strip()]
    pins = [str(p).strip() for p in (item.get("pins") or []) if str(p).strip()]
    for attr in attributes:
        if attr not in pins:
            pins.append(attr)
    return AnimaSliderRow(
        target=target,
        positive=positive,
        neutral=neutral,
        negative=str(item.get("negative") or ""),
        attributes=attributes,
        action=str(item.get("action") or "enhance"),
        guidance_scale=float(item.get("guidance_scale", DEFAULT_CFG)),
        resolution=int(item.get("resolution", DEFAULT_RESOLUTION)),
        batch_size=int(item.get("batch_size", 1)),
        pins=pins,
        concept_words=str(item.get("concept_words") or ""),
    )


def load_anima_prompts(path: Path | str) -> tuple[list[AnimaSliderRow], AnimaPromptsMeta]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    meta = AnimaPromptsMeta()
    if isinstance(raw, dict):
        meta.plus_label = str(raw.get("plus_label") or "")
        meta.minus_label = str(raw.get("minus_label") or "")
        rng = raw.get("recommended_range")
        if isinstance(rng, (list, tuple)) and len(rng) == 2:
            meta.recommended_range = [float(rng[0]), float(rng[1])]
        meta.concept_words = str(raw.get("concept_words") or "")
        raw = raw.get("rows", raw.get("prompts"))
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"anima prompts file is empty: {path}")
    rows: list[AnimaSliderRow] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError(f"each anima prompt must be a mapping: {item!r}")
        row_concept = str(item.get("concept_words") or meta.concept_words)
        for expanded in expand_attributes_anima(item):
            expanded.setdefault("concept_words", row_concept)
            rows.append(_as_row(expanded))
    return rows, meta


def row_token_plan(row: AnimaSliderRow) -> dict[str, Any]:
    unused = unused_vocab(
        row.target,
        row.neutral,
        row.attributes,
        row.pins,
        concept_words=row.concept_words,
    )
    pos_tokens = word_tokens(row.positive)
    neu_tokens = word_tokens(row.neutral)
    return {
        "unused": unused,
        "concept": concept_tokens(row.positive, unused),
        "pos_tokens": pos_tokens,
        "neu_tokens": neu_tokens,
        "pos_hold_mask": unused_token_mask(pos_tokens, unused),
        "pairs": align_unused_positions(pos_tokens, neu_tokens, unused),
    }


class AnimaSampleGateError(RuntimeError):
    """In-process PEFT sample failed the RGB-noise gate."""


def infer_sample_prompts(
    rows: Sequence[AnimaSliderRow],
    control_prompt: str = DEFAULT_CONTROL_PROMPT,
) -> list[str]:
    """Infer/neu captions only, plus the fruit-bowl control. Never the + concept.

    Returns ``row.infer_prompt`` (equals ``row.neutral`` on the smile
    card) unchanged. No attribute-prefix strip — train and sample share
    the same yaml strings.
    """
    seen: list[str] = []
    for row in rows:
        prompt = (row.infer_prompt or row.neutral or "").strip()
        if prompt and prompt not in seen:
            seen.append(prompt)
    control = str(control_prompt or "").strip()
    if control and control not in seen:
        seen.append(control)
    return seen


def image_mean_std(arr: np.ndarray) -> tuple[float, float]:
    """uint8 / float HWC (or CHW) image stats used by the sample gate."""
    pixels = np.asarray(arr, dtype=np.float64)
    if pixels.size == 0:
        raise ValueError("empty image for mean/std")
    return float(pixels.mean()), float(pixels.std())


def looks_like_rgb_noise(arr: np.ndarray) -> bool:
    """True for TV-static RGB (~122/75): high std, mid mean, no spatial corr."""
    pixels = np.asarray(arr)
    if pixels.ndim == 3 and pixels.shape[0] in (1, 3) and pixels.shape[-1] not in (1, 3):
        pixels = np.transpose(pixels, (1, 2, 0))
    mean, std = image_mean_std(pixels)
    if not (_NOISE_MEAN_LO <= mean <= _NOISE_MEAN_HI and _NOISE_STD_LO <= std <= _NOISE_STD_HI):
        return False
    gray = pixels.astype(np.float64)
    if gray.ndim == 3:
        gray = gray.mean(axis=-1)
    if gray.shape[1] < 2:
        return True
    left = gray[:, :-1].ravel()
    right = gray[:, 1:].ravel()
    left = left - left.mean()
    right = right - right.mean()
    denom = float(left.std() * right.std())
    if denom < 1e-6:
        return True
    corr = float((left * right).mean() / denom)
    return abs(corr) < _NOISE_CORR_MAX


def assert_sample_gate(records: Sequence[dict[str, Any]]) -> None:
    """Fail the train job if scale 0 is noise, or 0.25 is noise while 0 is fine."""
    if not records:
        raise AnimaSampleGateError("in-process PEFT sample grid is empty")

    def _is_scale(row: dict[str, Any], target: float) -> bool:
        return abs(float(row.get("scale", 1e9)) - target) < 1e-6

    scale0 = [row for row in records if _is_scale(row, 0.0)]
    scale025 = [row for row in records if _is_scale(row, 0.25)]
    if not scale0:
        raise AnimaSampleGateError("in-process sample grid missing scale 0.0")
    if any(bool(row.get("looks_like_noise")) for row in scale0):
        bad = next(row for row in scale0 if row.get("looks_like_noise"))
        raise AnimaSampleGateError(
            "scale-0 sample looks like RGB noise "
            f"(mean={bad.get('mean')}/std={bad.get('std')}, ~122/75); "
            "base pipeline is broken"
        )
    if scale025 and any(bool(row.get("looks_like_noise")) for row in scale025):
        bad = next(row for row in scale025 if row.get("looks_like_noise"))
        raise AnimaSampleGateError(
            "scale 0.25 sample looks like RGB noise while scale 0 is fine "
            f"(mean={bad.get('mean')}/std={bad.get('std')}); "
            "adapter is broken — do not ship this LoRA"
        )


def stock_teacher_smoke_captions() -> dict[str, Any]:
    """Same-seed stock Anima neu vs plus. No Hub. Docs / smoke only.

    Before trusting a slider, stock neu must look clearly less smiley
    than stock plus. If both already grin, UNI has almost no teacher gap.
    """
    return {
        "woman": {"neu": WOMAN_NEU, "plus": WOMAN_PLUS},
        "man": {"neu": MAN_NEU, "plus": MAN_PLUS},
        "concept_words": DEFAULT_CONCEPT_WORDS,
        "same_seed": True,
        "cfg_via": "guider.config.guidance_scale",
        "note": (
            "stock neu must look clearly less smiley than stock plus "
            "(same seed). + is CFG teacher only; student +1 stays on "
            "neu/infer (#62 analog). Do not pass guidance_scale= to "
            "ModularPipeline — it is ignored."
        ),
    }


def anima_recipe_label(
    lm_target: str | None = None,
    teacher: str | None = None,
) -> str:
    resolved = resolve_anima_train_recipe(lm_target, teacher)
    if resolved.teacher == "same_crop":
        return (
            "same_crop invert teacher + trajectory MSE + unused_token_hold "
            "(plus from re-noised neu, not caption-from-z_T)"
        )
    if resolved.loss_kind == "trajectory":
        return "trajectory short FlowMatch Euler + unused_token_hold"
    if resolved.loss_kind == "direct":
        return "direct velocity UNI + unused_token_hold"
    return "cfg_delta UNI + unused_token_hold"


def same_crop_smoke_command(
    *,
    name: str = "smile-anima-same-crop-smoke",
    resolution: int = 512,
    steps: int = 8,
    sample_steps: int = 8,
    traj_steps: int = DEFAULT_TRAJ_STEPS,
    teacher_strength: float = DEFAULT_TEACHER_STRENGTH,
    save_dir: str = "models/smile-anima-same-crop-smoke",
) -> str:
    """Short 4090/L40S smoke — not a 500-step train."""
    return (
        "HF_HUB_OFFLINE=1 python conceptmod/textsliders/train_lora_anima.py \\\n"
        f"  --name {name} \\\n"
        "  --prompts_file conceptmod/textsliders/data/prompts-anima.yaml \\\n"
        "  --model_id circlestone-labs/Anima-Base-v1.0-Diffusers \\\n"
        f"  --lora_targets conditioner --rank 16 --resolution {int(resolution)} "
        f"--sample_steps {int(sample_steps)} --cfg 4 \\\n"
        f"  --lr 1e-4 --lm_target same_crop --teacher same_crop "
        f"--teacher_strength {float(teacher_strength)} \\\n"
        f"  --traj_steps {int(traj_steps)} --steps {int(steps)} "
        "--sample_every 0 \\\n"
        f"  --device cuda:0 --save_dir {save_dir}"
    )


def live_train_card(
    *,
    name: str = "smile-anima",
    prompts_file: str = "conceptmod/textsliders/data/prompts-anima.yaml",
    model_id: str = DEFAULT_MODEL_ID,
    rank: int = DEFAULT_RANK,
    resolution: int = DEFAULT_RESOLUTION,
    sample_steps: int = DEFAULT_SAMPLE_STEPS,
    cfg: float = DEFAULT_CFG,
    device: str = "cuda:0",
    lr: float = DEFAULT_LR,
    control_prompt: str = DEFAULT_CONTROL_PROMPT,
    lm_target: str = DEFAULT_LM_TARGET,
    sample_every: int = DEFAULT_SAMPLE_EVERY,
    traj_steps: int = DEFAULT_TRAJ_STEPS,
    traj_identity_weight: float = DEFAULT_TRAJ_IDENTITY_WEIGHT,
    teacher_gap_boost: float = DEFAULT_TEACHER_GAP_BOOST,
    teacher: str = DEFAULT_TEACHER,
    teacher_strength: float = DEFAULT_TEACHER_STRENGTH,
    lora_targets: str = DEFAULT_LORA_TARGETS,
) -> dict[str, Any]:
    """Documented live train card. CI never downloads these weights."""
    resolved = resolve_anima_train_recipe(lm_target, teacher, teacher_strength)
    recipe = resolved.lm_target
    spec = resolve_anima_lora_targets(lora_targets)
    return {
        "name": name,
        "model_id": model_id,
        "arch": "2B Cosmos-Predict2 DiT, Qwen3+T5, Qwen-Image VAE",
        "lora": {
            "lora_targets": spec.label,
            "rank": rank,
            "alpha": float(rank),
            "targets": spec.active_attn_targets,
            "dit_targets": list(DIT_LORA_TARGETS),
            "conditioner_targets": list(CONDITIONER_LORA_TARGETS),
            "train_dit": spec.train_dit,
            "train_text_conditioner": spec.train_conditioner,
            "train_text_encoder": False,
            "adapted_modules": spec.adapted_module_names,
            "frozen_modules": list(spec.frozen_modules),
            "frozen_ref": "base modules with PEFT adapters disabled",
            "save": (
                "{name}_conditioner_lora when conditioner is on; "
                "{name}_lora when dit is on"
            ),
            "text_encoder_note": (
                "Qwen3 text_encoder (28-layer, ~1.2GB) is not adapted. "
                "Caption-level smile lives in AnimaTextConditioner "
                "(6-layer, ~269M, q_proj/k_proj/v_proj/o_proj)."
            ),
            "vram_note": (
                "4090 ~24GB: smile retrain with --lora_targets conditioner "
                "--resolution 512 --rank 16. dit-only 768 traj_steps=4 was "
                "~23GB. dit+conditioner needs rank 8 or 512."
            ),
        },
        "resolution": resolution,
        "sample_steps": sample_steps,
        "cfg": cfg,
        "lr": lr,
        "device": device,
        "prompts_file": prompts_file,
        "control_prompt": control_prompt,
        "sample_scales": list(DEFAULT_SAMPLE_SCALES),
        "sample_seed": DEFAULT_SAMPLE_SEED,
        "sample_every": int(sample_every),
        "lm_target": recipe,
        "traj_steps": int(traj_steps),
        "traj_identity_weight": float(traj_identity_weight),
        "teacher_gap_boost": float(teacher_gap_boost),
        "teacher": resolved.teacher,
        "teachers": list(ANIMA_TEACHERS),
        "teacher_strength": float(resolved.teacher_strength),
        "caption_teacher_failure": (
            "caption-only plus from the same z_T still jumps crop: stock "
            "Anima goes full-body → close-up portrait when the caption "
            "goes closed-mouth → teeth. Pixel gate then only 'passes' "
            "smile with identity/crop collapse. Wiring (#70) is cleared "
            "(peft_pipe≡train_faithful, no SAMPLE_TRAIN_MISMATCH); this "
            "is formulation."
        ),
        "same_crop_teacher": (
            "invert frozen neu traj: x_σ=(1−σ)·x_neu+σ·z_T, then Euler "
            "plus from σ=teacher_strength. Crop is committed by neu; "
            "plus only edits late σ. No Comfy — same predict_v loop."
        ),
        "same_crop_smoke_4090": same_crop_smoke_command(
            traj_steps=traj_steps,
            teacher_strength=resolved.teacher_strength,
        ),
        "traj_loop": (
            "thin FlowMatch Euler over predict_v: σ=linspace(1,1/K,K)+0, "
            "x ← x+(σ_next−σ)*v. Not ModularPipeline denoise (no grad "
            "through pipe). Matches Anima FlowMatchEulerDiscreteScheduler.step."
        ),
        "traj_loss": (
            "MSE(x_student, x_plus) + λ_id*MSE(x_zero, x_neu); "
            "caption: x_plus=Euler_K(frozen, plus, z_T); "
            "same_crop: x_plus=Euler_from_σ(frozen, plus, invert(x_neu,z_T)); "
            "x_student=Euler_K(adapter, neu/infer, z_T); "
            "x_neu=Euler_K(frozen, neu, z_T); "
            "x_zero=Euler_K(scale=0, neu/infer, z_T); "
            "z_T~N(0,I)"
        ),
        "one_step_failure": (
            "v4 seed-42 velocity diagnostic: cos(v(pos,frozen), "
            "v(neu,frozen))≈0.99993 (MSE≈0.00037). Stock images differ "
            "(closed mouth vs toothy smile) but 1-step v-space gap is "
            "microscopic. Adapter Δ was 3–7× larger with δ-cos≈0.28. "
            "Need multi-step/trajectory, not another to_v 200-step 1-step retry."
        ),
        "sample_mode": DEFAULT_SAMPLE_MODE,
        "sample_modes": list(ANIMA_SAMPLE_MODES),
        "sample_gate": (
            "in-process PEFT disable_adapter / set_adapter_scale. "
            "Default peft_pipe: pipe(prompt=...) after "
            "sync_peft_into_modular_pipeline so sample and encode_text "
            "share the same conditioner/transformer objects. "
            "train_faithful: backend.encode_text + transformer (loss path). "
            "Fail if scale 0 is RGB noise or if scale 0.25 is noise while 0 is fine"
        ),
        "stock_teacher_smoke": stock_teacher_smoke_captions(),
        "train_sample_prompts": "same bare infer/neu captions; attributes are pins only",
        "recipe": anima_recipe_label(recipe, resolved.teacher),
        "music3_default_untouched": {"lm_target": "v9", "pole_mode": "hidden"},
        "turbo": "preview_only",
        "smile_retrain_4090": live_train_command(
            resolution=512,
            lora_targets="conditioner",
            rank=rank,
            lr=lr,
            traj_steps=traj_steps,
        ),
    }


def live_train_command(
    *,
    name: str = "smile-anima",
    prompts_file: str = "conceptmod/textsliders/data/prompts-anima.yaml",
    model_id: str = DEFAULT_MODEL_ID,
    rank: int = DEFAULT_RANK,
    resolution: int = DEFAULT_RESOLUTION,
    sample_steps: int = DEFAULT_SAMPLE_STEPS,
    cfg: float = DEFAULT_CFG,
    device: str = "cuda:0",
    save_dir: str = "models/smile-anima",
    lr: float = DEFAULT_LR,
    lm_target: str = DEFAULT_LM_TARGET,
    sample_every: int = DEFAULT_SAMPLE_EVERY,
    traj_steps: int = DEFAULT_TRAJ_STEPS,
    teacher_gap_boost: float = DEFAULT_TEACHER_GAP_BOOST,
    lora_targets: str = DEFAULT_LORA_TARGETS,
) -> str:
    recipe = resolve_anima_lm_target(lm_target)
    spec = resolve_anima_lora_targets(lora_targets)
    return (
        "HF_HUB_OFFLINE=1 python conceptmod/textsliders/train_lora_anima.py \\\n"
        f"  --name {name} \\\n"
        f"  --prompts_file {prompts_file} \\\n"
        f"  --model_id {model_id} \\\n"
        f"  --lora_targets {spec.label} --rank {rank} "
        f"--resolution {resolution} "
        f"--sample_steps {sample_steps} --cfg {cfg} \\\n"
        f"  --lr {lr} --lm_target {recipe} --traj_steps {int(traj_steps)} \\\n"
        f"  --teacher_gap_boost {float(teacher_gap_boost)} "
        f"--sample_every {int(sample_every)} \\\n"
        f"  --device {device} --save_dir {save_dir}"
    )


def turbo_preview_card() -> dict[str, Any]:
    """Anima-Turbo v1.1 is a convert + stock-smoke path, not a train target."""
    return {
        "role": "preview_only",
        "train_on": DEFAULT_MODEL_ID,
        "comfy_repo": TURBO_COMFY_REPO,
        "transformer_file": TURBO_TRANSFORMER_FILE,
        "text_encoder_file": TURBO_TEXT_ENCODER_FILE,
        "vae_file": TURBO_VAE_FILE,
        "vae_class": "AutoencoderKLQwenImage",
        "convert_script": "huggingface/diffusers scripts/convert_anima_to_diffusers.py",
        "convert_splits": {
            "llm_adapter": "AnimaTextConditioner",
            "rest": "CosmosTransformer3DModel",
        },
        "convert_flags": ["--save_pipeline", f"--dtype {TURBO_CONVERT_DTYPE}"],
        "output": TURBO_DIFFUSERS_OUTPUT,
        "sample_cfg": TURBO_SAMPLE_CFG,
        "sample_steps": TURBO_SAMPLE_STEPS,
        "sample_steps_range": list(TURBO_SAMPLE_STEPS_RANGE),
        "license": TURBO_LICENSE,
        "ignore_community": TURBO_IGNORE_COMMUNITY,
        "why": (
            "smile-anima-v5 on Base failed (tiny v-space teacher gap). "
            "Turbo is faster stock/preview smoke, not the next train "
            "unless smoke shows a larger closed-mouth vs teeth gap."
        ),
        "helper": "scripts/convert_anima_turbo_diffusers.py",
    }


def turbo_preview_sample_command(
    *,
    model_id: str = TURBO_DIFFUSERS_OUTPUT,
    sample_steps: int = TURBO_SAMPLE_STEPS,
    cfg: float = TURBO_SAMPLE_CFG,
) -> str:
    """Sample-path only. Does not change the Base train recipe."""
    return (
        "HF_HUB_OFFLINE=1 python conceptmod/textsliders/train_lora_anima.py \\\n"
        "  --name smile-anima-turbo-preview \\\n"
        "  --prompts_file conceptmod/textsliders/data/prompts-anima.yaml \\\n"
        f"  --model_id {model_id} \\\n"
        f"  --rank 16 --resolution 768 --sample_steps {int(sample_steps)} "
        f"--cfg {cfg:g} \\\n"
        "  --lr 1e-4 --lm_target trajectory --traj_steps 4 \\\n"
        "  --teacher_gap_boost 1 --sample_every 100 \\\n"
        "  --device cuda:0 --save_dir models/smile-anima-turbo-preview"
    )
