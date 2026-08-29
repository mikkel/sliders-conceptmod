"""LTX-2.5 backend: dummy video-attn LoRA + live LTX2Pipeline.

Live hub id is ``Lightricks/LTX-2.5-Diffusers``. Default DiT is the
distilled checkpoint in ``transformer/`` (``model_index.json``).
``transformer_full/`` is SFT and is **excluded** from the first download.

Dummy never imports that loader and never hits the Hub.

Hold is **PRE-connector**. Gemma 4 emits per-token hiddens; current
``LTX2Pipeline._get_gemma_prompt_embeds`` stacks every hidden layer and
``flatten(2, 3)`` (concat) only. Mean-center/scale is **inside**
``LTX2TextConnectors.forward`` (``per_layer_masked_mean_norm`` on the
LTX-2.0 path; ``per_token_rms_norm`` when ``per_modality_projections``).
Then ``text_proj_in`` and separate video/audio 1-D connectors **replace
padding with learnable registers** and run RoPE blocks, so T is no
longer 1:1 with prompt tokens. Token-id hold exists only before
``pipe.connectors(...)``.

Verified against current diffusers ``LTX2*`` (main, not a release):

* Attention class ``LTX2Attention``: ``to_q`` / ``to_k`` / ``to_v`` /
  ``to_out.0`` (``to_out.1`` is Dropout).
* Video-only v1 LoRA: ``attn1`` (self) and ``attn2`` (text cross). Do
  **not** wrap ``audio_attn*``, ``audio_to_video_attn``,
  ``video_to_audio_attn``, AdaLN, or FFN — a smile slider must not
  rewrite foley. A naive ``"to_q"`` / ``.endswith(".attn1")`` matches
  ``audio_attn1`` and is too broad.
* Transformer already has ``PeftAdapterMixin``. Live prefers PEFT, but
  ``set_adapter_scale`` no-ops (Krea #74): write
  ``LoraLayer.scaling = (alpha/r) * scale`` via
  ``apply_continuous_lora_scale``. LoRA-up is ``N(0, 0.02)``, not zeros.
* Forward returns ``AudioVisualModelOutput(sample, audio_sample)`` =
  flow velocity. Pipeline: ``x0 = x_t − sigma * v``.
* Distilled sample: ``sigmas=DISTILLED_SIGMA_VALUES``, guidance 1.0,
  STG/modality 0. Do **not** pass ``num_inference_steps``. Prompt
  enhancer OFF. Conv VAE ``pipe.vae`` (skip ``diffusion_decoder``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from conceptmod.textsliders.ltx25_uni import apply_unused_hold
from conceptmod.textsliders.slider_targets import apply_continuous_lora_scale

DEFAULT_MODEL = "Lightricks/LTX-2.5-Diffusers"
DEFAULT_TRANSFORMER_SUBFOLDER = "transformer"
FULL_TRANSFORMER_SUBFOLDER = "transformer_full"
# Distilled 8-sigma schedule from current diffusers.pipelines.ltx2.utils.
# Do not invent a linear num_inference_steps stand-in for distilled.
DISTILLED_SIGMA_VALUES = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875]
LTX_FPS = 24.0
LTX_FRAME_MOD = 8
LTX_FRAME_BIAS = 1
LTX_CANVAS_MULTIPLE = 32
DEFAULT_NUM_FRAMES = 49
DEFAULT_SAMPLE_HEIGHT = 544
DEFAULT_SAMPLE_WIDTH = 960
DEFAULT_SAMPLE_SCALES = (0.0, 0.5, 1.0)
DEFAULT_TRAIN_NUM_FRAMES = 9
DEFAULT_TRAIN_HEIGHT = 32
DEFAULT_TRAIN_WIDTH = 32
VIDEO_IN_CHANNELS = 128
LORA_ATTN_CLASS = "LTX2Attention"
LORA_VIDEO_HOSTS = ("attn1", "attn2")
LORA_LINEAR_NAMES = ("to_q", "to_k", "to_v", "to_out.0")
DEFAULT_LORA_UP_INIT_STD = 0.02
DEFAULT_ENCODER_DEVICE = "cpu"
FREEZE_LIST = (
    "text_encoder",  # LTX-specific Gemma 4 12B (gemma4_unified), not vanilla Gemma 4
    "connectors",
    "vae",
    "audio_vae",
    "vocoder",
    "tokenizer",
    "processor",
    "prompt_enhancer",
    "duration_head",
    "diffusion_decoder",
)
IGNORE_ON_FIRST_DOWNLOAD = ("transformer_full/*", "transformer_full/**")
TEXT_ENCODER_KIND = "gemma4_unified"


class ArchitectureMismatch(RuntimeError):
    """Raised when a conceptmod DiT helper does not exist on LTX-2.5."""


def distilled_sigmas() -> list[float]:
    """Live schedule. Import from current diffusers when present."""
    try:
        from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES as live

        return list(live)
    except Exception:
        return list(DISTILLED_SIGMA_VALUES)


def gemma4_tokenize_text(text: str) -> str:
    """Gemma-4 prepends a leading space. Hold tokenize matches that."""
    prompt = str(text)
    if prompt and not prompt.startswith(" "):
        return " " + prompt
    return prompt


class DummyTokenizer:
    """Whitespace tokenizer. Not the live Gemma-4 tokenizer."""

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}
        self.padding_side = "left"
        self.pad_token = None

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        ids = []
        for word in gemma4_tokenize_text(text).split():
            key = word.lower()
            if key not in self._vocab:
                self._vocab[key] = len(self._vocab) + 1
            ids.append(self._vocab[key])
        return ids or [1]

    def __call__(self, text, return_tensors=None, add_special_tokens=False, **_k):
        raw = text if isinstance(text, list) else [text]
        rows = [self.encode(t, add_special_tokens=add_special_tokens) for t in raw]
        n = max(len(r) for r in rows)
        ids = [r + [0] * (n - len(r)) for r in rows]
        mask = [[1] * len(r) + [0] * (n - len(r)) for r in rows]
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(mask, dtype=torch.long),
            }
        return {"input_ids": ids, "attention_mask": mask}


class LTX2Attention(nn.Module):
    """Tiny stand-in whose Linear names match ``LTX2Attention``.

    Live modules: ``to_q`` / ``to_k`` / ``to_v`` / ``to_out.0``.
    ``to_out.1`` is Dropout and is not a LoRA target.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.to_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_out = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.Identity(),
        ])

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states=None, **_k) -> torch.Tensor:
        ctx = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        if ctx.shape[1] != hidden_states.shape[1]:
            ctx = F.adaptive_avg_pool1d(
                ctx.transpose(1, 2), hidden_states.shape[1]
            ).transpose(1, 2)
        q = self.to_q(hidden_states)
        k = self.to_k(ctx)
        v = self.to_v(ctx)
        scale = hidden_states.shape[-1] ** -0.5
        attn = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
        return self.to_out[0](attn @ v)


class DummyLTX2Block(nn.Module):
    """Video attn1/attn2 plus audio / a2v / v2a that LoRA must skip."""

    def __init__(self, video_dim: int, audio_dim: int) -> None:
        super().__init__()
        self.attn1 = LTX2Attention(video_dim)
        self.attn2 = LTX2Attention(video_dim)
        self.audio_attn1 = LTX2Attention(audio_dim)
        self.audio_attn2 = LTX2Attention(audio_dim)
        self.audio_to_video_attn = LTX2Attention(video_dim)
        self.video_to_audio_attn = LTX2Attention(audio_dim)
        self.ff = nn.Linear(video_dim, video_dim)
        self.audio_ff = nn.Linear(audio_dim, audio_dim)
        self.scale_shift_table = nn.Parameter(torch.zeros(2, video_dim))

    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        audio_encoder_hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states + self.attn1(hidden_states)
        hidden_states = hidden_states + self.attn2(
            hidden_states, encoder_hidden_states=encoder_hidden_states,
        )
        audio_hidden_states = audio_hidden_states + self.audio_attn1(audio_hidden_states)
        audio_hidden_states = audio_hidden_states + self.audio_attn2(
            audio_hidden_states, encoder_hidden_states=audio_encoder_hidden_states,
        )
        hidden_states = hidden_states + self.ff(hidden_states)
        audio_hidden_states = audio_hidden_states + self.audio_ff(audio_hidden_states)
        return hidden_states, audio_hidden_states


class DummyLTX2Connectors(nn.Module):
    """Registers replace padding so T is no longer 1:1 with prompt tokens.

    Mirrors live ``LTX2TextConnectors``: ``text_proj_in``, then video/audio
    1-D paths that **replace left-pad with learnable registers** and mix
    the sequence (live connectors run RoPE 1-D blocks). Frozen. Hold
    applied **after** this module cannot align by token id — dummy tests
    fail if pack holds post-connector.
    """

    def __init__(self, dim: int = 8, n_registers: int = 2) -> None:
        super().__init__()
        self.text_proj_in = nn.Linear(dim, dim, bias=False)
        self.video_mix = nn.Linear(dim, dim, bias=False)
        self.audio_mix = nn.Linear(dim, dim, bias=False)
        self.video_registers = nn.Parameter(torch.randn(n_registers, dim) * 0.02)
        self.audio_registers = nn.Parameter(torch.randn(n_registers, dim) * 0.02)
        self.n_registers = int(n_registers)
        self.requires_grad_(False)
        self.eval()

    def _layout_and_mix(
        self,
        proj: torch.Tensor,
        registers: torch.Tensor,
        mix: nn.Linear,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Left-pad with registers, mix so row i is not token i, T' != T."""
        bsz = proj.shape[0]
        regs = registers.to(device=proj.device, dtype=proj.dtype)
        # Live: replace padding slots, front-align valid tokens, registers in tail.
        # Dummy has no pad on train encode — prepend registers then mix.
        laid_out = torch.cat((regs.unsqueeze(0).expand(bsz, -1, -1), proj), dim=1)
        ctx = laid_out.mean(dim=1, keepdim=True).expand_as(laid_out)
        mixed = mix(laid_out + 0.35 * ctx)
        if attention_mask is None:
            mask = torch.ones(mixed.shape[:2], device=proj.device, dtype=torch.long)
        else:
            extra = torch.ones(
                bsz, self.n_registers, device=proj.device, dtype=attention_mask.dtype,
            )
            mask = torch.cat((extra, attention_mask.to(proj.device)), dim=1)
        return mixed, mask

    def forward(
        self,
        text_encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        padding_side: str = "left",
    ):
        del padding_side
        proj = self.text_proj_in(text_encoder_hidden_states)
        video, mask = self._layout_and_mix(
            proj, self.video_registers, self.video_mix, attention_mask,
        )
        audio, _ = self._layout_and_mix(
            proj, self.audio_registers, self.audio_mix, attention_mask,
        )
        return video, audio, mask


# Survey / test alias.
LTX25DummyConnectors = DummyLTX2Connectors


class DummyLTX2Transformer(nn.Module):
    """Tiny flow-velocity teacher. Mirrors the live forward contract.

    ``forward`` returns video / audio velocity. Pack channels are
    ``proj_in.in_features`` (128), not inner_dim.
    """

    def __init__(
        self,
        hidden_size: int = 16,
        text_dim: int = 8,
        video_dim: int = VIDEO_IN_CHANNELS,
        audio_dim: int = 8,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.text_dim = text_dim
        self.video_dim = video_dim
        self.audio_dim = audio_dim
        self.in_channels = video_dim
        self.audio_in_channels = audio_dim
        self.proj_in = nn.Linear(video_dim, hidden_size)
        self.audio_proj_in = nn.Linear(audio_dim, hidden_size)
        self.caption_projection = nn.Linear(text_dim, hidden_size)
        self.audio_caption_projection = nn.Linear(text_dim, hidden_size)
        self.time_embedder = nn.Linear(1, hidden_size)
        self.transformer_blocks = nn.ModuleList([DummyLTX2Block(hidden_size, hidden_size)])
        self.norm_out = nn.Identity()
        self.audio_norm_out = nn.Identity()
        self.proj_out = nn.Linear(hidden_size, video_dim)
        self.audio_proj_out = nn.Linear(hidden_size, audio_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        audio_encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        audio_timestep: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        audio_encoder_attention_mask: torch.Tensor | None = None,
        num_frames: int | None = None,
        height: int | None = None,
        width: int | None = None,
        fps: float = 24.0,
        isolate_modalities: bool = False,
        spatio_temporal_guidance_blocks: list[int] | None = None,
        return_dict: bool = True,
        **_k,
    ):
        del encoder_attention_mask, audio_encoder_attention_mask
        del num_frames, height, width, fps, isolate_modalities
        del spatio_temporal_guidance_blocks
        video = self.proj_in(hidden_states)
        audio = self.audio_proj_in(audio_hidden_states)
        enc_v = self.caption_projection(encoder_hidden_states)
        enc_a = self.audio_caption_projection(audio_encoder_hidden_states)
        video = video + enc_v.mean(dim=1, keepdim=True)
        audio = audio + enc_a.mean(dim=1, keepdim=True)
        temb = self.time_embedder(timestep.reshape(-1, 1)[:1].to(video.dtype))
        video = video + temb.unsqueeze(1)
        if audio_timestep is not None:
            audio = audio + self.time_embedder(audio_timestep.reshape(-1, 1)[:1].to(audio.dtype)).unsqueeze(1)
        for block in self.transformer_blocks:
            video, audio = block(video, audio, enc_v, enc_a)
        output = self.proj_out(self.norm_out(video))
        audio_output = self.audio_proj_out(self.audio_norm_out(audio))
        if return_dict:
            return DummyVelocity(sample=output, audio_sample=audio_output)
        return output, audio_output


@dataclass
class DummyVelocity:
    sample: torch.Tensor
    audio_sample: torch.Tensor


@dataclass
class EncodedText:
    """PRE-connector token-aligned features. T == len(token_ids)."""

    embeds: torch.Tensor
    token_ids: list[int]
    attention_mask: torch.Tensor | None = None
    hold_stage: str = "pre_connector"


@dataclass
class PackedLayout:
    hidden_states: torch.Tensor
    audio_hidden_states: torch.Tensor
    encoder_hidden_states: torch.Tensor
    audio_encoder_hidden_states: torch.Tensor
    timestep: torch.Tensor
    audio_timestep: torch.Tensor
    encoder_attention_mask: torch.Tensor | None
    audio_encoder_attention_mask: torch.Tensor | None
    num_frames: int
    height: int
    width: int
    pre_connector_hidden: torch.Tensor
    n_prompt_tokens: int
    hold_stage: str = "pre_connector"


def is_video_attn_linear(module_name: str) -> bool:
    """True for ``attn1`` / ``attn2`` qkv/out only.

    Rejects ``audio_attn1`` (would match ``.endswith('.attn1')``),
    ``audio_to_video_attn``, ``video_to_audio_attn``, AdaLN, FFN.
    """
    parts = [p for p in str(module_name).split(".") if p]
    if any(
        p.startswith("audio") or p in ("audio_to_video_attn", "video_to_audio_attn")
        for p in parts
    ):
        return False
    if not any(p in LORA_VIDEO_HOSTS for p in parts):
        return False
    if parts[-1] in ("to_q", "to_k", "to_v"):
        return True
    if len(parts) >= 2 and parts[-2] == "to_out" and parts[-1] == "0":
        return True
    return False


def video_attn_lora_targets(transformer: nn.Module) -> list[str]:
    names = []
    for name, module in transformer.named_modules():
        if isinstance(module, nn.Linear) and is_video_attn_linear(name):
            names.append(name)
    return names


class _AttnLoRA(nn.Module):
    """LoRA on one Linear. Dummy / PEFT-fallback share this."""

    def __init__(
        self,
        name: str,
        module: nn.Linear,
        rank: int,
        alpha: float,
        up_init_std: float = DEFAULT_LORA_UP_INIT_STD,
    ) -> None:
        super().__init__()
        self.lora_name = name
        self.rank = rank
        self.scale = float(alpha) / float(rank)
        self.multiplier = 1.0
        host_kwargs: dict[str, Any] = {}
        if hasattr(module, "weight"):
            host_kwargs["device"] = module.weight.device
            host_kwargs["dtype"] = module.weight.dtype
        self.lora_down = nn.Linear(module.in_features, rank, bias=False, **host_kwargs)
        self.lora_up = nn.Linear(rank, module.out_features, bias=False, **host_kwargs)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5 ** 0.5)
        if float(up_init_std) > 0:
            nn.init.normal_(self.lora_up.weight, mean=0.0, std=float(up_init_std))
        else:
            nn.init.zeros_(self.lora_up.weight)
        self.org_forward = module.forward
        module.forward = self.forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.lora_down.weight
        x_lora = x.to(device=weight.device, dtype=weight.dtype)
        delta = self.lora_up(self.lora_down(x_lora)).to(device=x.device, dtype=x.dtype)
        return self.org_forward(x) + delta * (self.multiplier * self.scale)


class AttnLoRANetwork(nn.Module):
    """Wrap video ``attn1`` / ``attn2`` ``to_q/k/v/to_out.0`` only."""

    def __init__(
        self,
        transformer: nn.Module,
        rank: int,
        alpha: float,
        up_init_std: float = DEFAULT_LORA_UP_INIT_STD,
    ) -> None:
        super().__init__()
        self.lora_scale = 1.0
        self.alpha = float(alpha)
        self.rank = int(rank)
        self.up_init_std = float(up_init_std)
        self.loras: list[_AttnLoRA] = []
        for name, module in transformer.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if not is_video_attn_linear(name):
                continue
            lora_name = f"lora_ltx-{name}".replace(".", "-")
            lora = _AttnLoRA(lora_name, module, rank, alpha, up_init_std=self.up_init_std)
            self.loras.append(lora)
            self.add_module(lora_name, lora)

    def set_lora_slider(self, scale: float) -> None:
        self.lora_scale = float(scale)
        for lora in self.loras:
            lora.multiplier = float(scale)

    def __enter__(self):
        for lora in self.loras:
            lora.multiplier = 1.0 * self.lora_scale
        return self

    def __exit__(self, *exc):
        for lora in self.loras:
            lora.multiplier = 0.0
        return False

    def save_weights(self, file: str, dtype=None) -> None:
        from safetensors.torch import save_file

        state = {k: v.detach().cpu() for k, v in self.state_dict().items()}
        if dtype is not None:
            state = {k: v.to(dtype) for k, v in state.items()}
        save_file(state, file)

    def load_weights(self, file: str) -> None:
        from safetensors.torch import load_file

        state = load_file(file)
        keys = [k for k in state if str(k).startswith("lora_ltx-")]
        if not keys:
            raise ValueError(
                f"{file} has no lora_ltx-* keys. LTX-2.5 sliders save a "
                "custom AttnLoRANetwork (or PEFT adapter with lora_ltx sidecar)."
            )
        missing, unexpected = self.load_state_dict(state, strict=False)
        missing_lora = [k for k in missing if str(k).startswith("lora_ltx-")]
        if missing_lora:
            raise ValueError(f"{file} is missing LoRA keys: {missing_lora[:8]}")
        _ = unexpected


class PeftLoRANetwork:
    """Live PEFT wrapper. Writes ``LoraLayer.scaling`` (set_adapter_scale no-ops)."""

    def __init__(self, transformer: nn.Module, rank: int, alpha: float) -> None:
        self.transformer = transformer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.lora_scale = 1.0
        self.loras: list[Any] = []
        self._apply_scale(1.0)

    def _apply_scale(self, scale: float) -> None:
        apply_continuous_lora_scale(self.transformer, float(scale))

    def set_lora_slider(self, scale: float) -> None:
        self.lora_scale = float(scale)
        self._apply_scale(self.lora_scale)

    def __enter__(self):
        self._apply_scale(self.lora_scale)
        return self

    def __exit__(self, *exc):
        self._apply_scale(0.0)
        return False

    def parameters(self):
        return (p for p in self.transformer.parameters() if p.requires_grad)

    def state_dict(self):
        return {
            k: v for k, v in self.transformer.state_dict().items()
            if "lora_" in k
        }

    def save_weights(self, file: str, dtype=None) -> None:
        from safetensors.torch import save_file

        state = {k: v.detach().cpu() for k, v in self.state_dict().items()}
        if dtype is not None:
            state = {k: v.to(dtype) for k, v in state.items()}
        save_file(state, file)

    def load_weights(self, file: str) -> None:
        from safetensors.torch import load_file

        state = load_file(file)
        missing, unexpected = self.transformer.load_state_dict(state, strict=False)
        _ = missing, unexpected


class DummyEncoder(nn.Module):
    """Frozen stand-in for LTX Gemma 4 token features (PRE-connector)."""

    def __init__(self, vocab: int = 128, dim: int = 8) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.mix = nn.Linear(dim, dim, bias=False)
        self.requires_grad_(False)
        self.eval()

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        raw = self.embed(token_ids)
        ctx = raw.mean(dim=1, keepdim=True).expand_as(raw)
        return self.mix(raw + 0.15 * ctx)


class LTX25Backend:
    """Opt-in LTX-2.5 slider host. Dummy is the CI path."""

    def __init__(
        self,
        *,
        device: str = "cpu",
        encoder_device: str | None = DEFAULT_ENCODER_DEVICE,
        model_id: str = DEFAULT_MODEL,
        transformer_subfolder: str = DEFAULT_TRANSFORMER_SUBFOLDER,
        lora_rank: int = 8,
        lora_alpha: float = 8.0,
        lora_up_init_std: float = DEFAULT_LORA_UP_INIT_STD,
        dummy: bool = False,
    ) -> None:
        self.dummy = bool(dummy)
        self.device = torch.device(device if not dummy else "cpu")
        if dummy:
            self.encoder_device = torch.device("cpu")
        else:
            self.encoder_device = torch.device(encoder_device or DEFAULT_ENCODER_DEVICE)
        self.model_id = model_id
        self.transformer_subfolder = transformer_subfolder
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.lora_up_init_std = float(lora_up_init_std)
        self.guidance = 1.0
        self.stg_scale = 0.0
        self.modality_scale = 0.0
        self.prompt_enhancer_enabled = False
        self.pipe: Any = None
        self.tokenizer: Any
        self.encoder: Any
        self.connectors: Any
        self.transformer: nn.Module
        self.network: Any = None
        self.vae = None
        self.audio_vae = None
        self.vocoder = None
        self._peft = False
        if self.dummy:
            self._init_dummy()
        else:
            self._init_live()

    def _init_dummy(self) -> None:
        self.tokenizer = DummyTokenizer()
        self.encoder = DummyEncoder()
        self.connectors = DummyLTX2Connectors()
        self.transformer = DummyLTX2Transformer()
        self.vae = _FrozenStub("vae")
        self.audio_vae = _FrozenStub("audio_vae")
        self.vocoder = _FrozenStub("vocoder")
        self.network = AttnLoRANetwork(
            self.transformer,
            rank=self.lora_rank,
            alpha=self.lora_alpha,
            up_init_std=self.lora_up_init_std,
        )
        self.pipe = _DummyPipe(
            tokenizer=self.tokenizer,
            text_encoder=self.encoder,
            connectors=self.connectors,
            transformer=self.transformer,
            vae=self.vae,
            audio_vae=self.audio_vae,
            vocoder=self.vocoder,
        )

    def _init_live(self) -> None:
        self.pipe = _load_ltx25_pipeline(
            self.model_id,
            device=str(self.device),
            encoder_device=str(self.encoder_device),
            transformer_subfolder=self.transformer_subfolder,
        )
        self.tokenizer = self.pipe.tokenizer
        self.encoder = self.pipe.text_encoder
        self.connectors = self.pipe.connectors
        self.transformer = self.pipe.transformer
        self.vae = getattr(self.pipe, "vae", None)
        self.audio_vae = getattr(self.pipe, "audio_vae", None)
        self.vocoder = getattr(self.pipe, "vocoder", None)
        for name in FREEZE_LIST:
            mod = getattr(self.pipe, name, None)
            if mod is None:
                continue
            if hasattr(mod, "requires_grad_"):
                mod.requires_grad_(False)
            if hasattr(mod, "eval"):
                mod.eval()
        self.transformer.requires_grad_(False)
        self.transformer.eval()
        self.network = self._attach_lora(self.transformer)

    def _attach_lora(self, transformer: nn.Module):
        targets = video_attn_lora_targets(transformer)
        if not targets:
            # Host class names still LTX2Attention; collect via walk.
            network = AttnLoRANetwork(
                transformer,
                rank=self.lora_rank,
                alpha=self.lora_alpha,
                up_init_std=self.lora_up_init_std,
            )
            network.to(self.device)
            host_dtype = _module_param_dtype(transformer)
            if host_dtype is not None:
                network.to(dtype=host_dtype)
            return network
        try:
            from peft import LoraConfig, get_peft_model

            config = LoraConfig(
                r=self.lora_rank,
                lora_alpha=int(self.lora_alpha),
                target_modules=targets,
                bias="none",
            )
            wrapped = get_peft_model(transformer, config)
            _init_peft_lora_up(wrapped, std=self.lora_up_init_std)
            wrapped.to(self.device)
            self.transformer = wrapped
            if self.pipe is not None:
                self.pipe.transformer = wrapped
            self._peft = True
            return PeftLoRANetwork(wrapped, self.lora_rank, self.lora_alpha)
        except Exception:
            network = AttnLoRANetwork(
                transformer,
                rank=self.lora_rank,
                alpha=self.lora_alpha,
                up_init_std=self.lora_up_init_std,
            )
            network.to(self.device)
            host_dtype = _module_param_dtype(transformer)
            if host_dtype is not None:
                network.to(dtype=host_dtype)
            return network

    def lora_module_names(self) -> list[str]:
        if self.network is None:
            return []
        if getattr(self.network, "loras", None):
            names = []
            for lora in self.network.loras:
                names.append(getattr(lora, "lora_name", str(lora)))
            if names:
                return names
        return video_attn_lora_targets(self.transformer)

    def trainable_parameters(self) -> list[nn.Parameter]:
        if self.network is None:
            return []
        if hasattr(self.network, "parameters"):
            return [p for p in self.network.parameters() if p.requires_grad]
        return [p for p in self.transformer.parameters() if p.requires_grad]

    def encode_text(self, prompt: str, frozen: bool = True) -> EncodedText:
        """Token-aligned PRE-connector features. ``add_special_tokens=False``."""
        if self.dummy:
            ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            if not ids:
                ids = [1]
            tensor = torch.tensor([ids], dtype=torch.long)
            with torch.no_grad() if frozen else torch.enable_grad():
                embeds = self.encoder(tensor)
            mask = torch.ones(1, len(ids), dtype=torch.long)
            return EncodedText(
                embeds=embeds, token_ids=[int(x) for x in ids],
                attention_mask=mask, hold_stage="pre_connector",
            )
        return self._live_encode_text(prompt)

    def _live_encode_text(self, prompt: str) -> EncodedText:
        """Gemma 4 12B token features, then stop — connectors run at pack().

        Tokenize ``add_special_tokens=False`` (survey / hold alignment).
        Gemma-4 prepends a leading space. Current diffusers
        ``_get_gemma_prompt_embeds`` uses ``add_special_tokens=True`` and
        max-length pad for sample; train hold needs 1:1 token rows.
        Feature pack matches the live stack+flatten (concat layers).
        Mean-center/scale is **not** in ``pipeline_ltx2.py`` — it runs
        inside ``LTX2TextConnectors.forward`` after this return
        (``per_layer_masked_mean_norm`` / ``per_token_rms_norm``).
        Stay on ``encoder_device``; do not move TE features onto the DiT.
        """
        tok = self.tokenizer
        text = gemma4_tokenize_text(prompt)
        if hasattr(tok, "padding_side"):
            tok.padding_side = "left"
        batch = tok(
            text,
            return_tensors="pt",
            add_special_tokens=False,
            padding=False,
            truncation=True,
        )
        ids = batch["input_ids"]
        mask = batch.get("attention_mask")
        raw_ids = ids.tolist()[0] if hasattr(ids, "tolist") else list(ids)
        enc_dev = self.encoder_device
        encoder = self.encoder
        with torch.no_grad():
            out = encoder(
                input_ids=ids.to(enc_dev),
                attention_mask=None if mask is None else mask.to(enc_dev),
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = out.hidden_states
            stacked = torch.stack(hidden_states, dim=-1)
            embeds = stacked.flatten(2, 3)
        if mask is not None and mask.device != embeds.device:
            mask = mask.to(embeds.device)
        return EncodedText(
            embeds=embeds,
            token_ids=[int(x) for x in raw_ids],
            attention_mask=mask,
            hold_stage="pre_connector",
        )

    def _run_connectors(
        self,
        token_hidden: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        connectors = self.connectors
        if connectors is None:
            return token_hidden, token_hidden, attention_mask
        conn_dev = token_hidden.device
        try:
            conn_dev = next(connectors.parameters()).device
        except StopIteration:
            pass
        hidden = token_hidden.to(conn_dev)
        mask_in = None if attention_mask is None else attention_mask.to(conn_dev)
        if mask_in is None:
            mask_in = torch.ones(hidden.shape[:2], device=hidden.device)
        try:
            out = connectors(
                hidden,
                mask_in,
                padding_side=getattr(self.tokenizer, "padding_side", "left"),
            )
        except TypeError:
            out = connectors(hidden, mask_in)
        if isinstance(out, tuple):
            video_e, audio_e = out[0], out[1]
            mask = out[2] if len(out) > 2 else attention_mask
        else:
            video_e, audio_e, mask = out, out, attention_mask
        return video_e, audio_e, mask

    def pack_t2v(
        self,
        text: EncodedText,
        *,
        video_latents: torch.Tensor | None = None,
        audio_latents: torch.Tensor | None = None,
        hold_neu: EncodedText | None = None,
        hold_mask: torch.Tensor | None = None,
        num_frames: int = DEFAULT_TRAIN_NUM_FRAMES,
        height: int = DEFAULT_TRAIN_HEIGHT,
        width: int = DEFAULT_TRAIN_WIDTH,
    ) -> PackedLayout:
        """Hold PRE-connector, then connectors, then fake video/audio pack."""
        enc = text.embeds
        if hold_neu is not None and hold_mask is not None:
            enc = apply_unused_hold(
                enc, hold_neu.embeds, text.token_ids, hold_neu.token_ids, hold_mask,
            )
        n_tokens = enc.shape[1]
        video_e, audio_e, conn_mask = self._run_connectors(enc, text.attention_mask)
        if video_e.shape[1] == n_tokens and self.dummy:
            raise RuntimeError(
                "dummy connectors did not add registers; hold-after-connector "
                "tests cannot distinguish PRE-connector hold"
            )
        video_dim = ltx_pack_feature_dim(self.transformer, kind="video")
        audio_dim = ltx_pack_feature_dim(self.transformer, kind="audio")
        n_video = max(1, ltx_latent_tokens(num_frames, height, width))
        if video_latents is None:
            video_latents = torch.randn(enc.shape[0], n_video, video_dim)
        if audio_latents is None:
            audio_latents = torch.randn(enc.shape[0], 2, audio_dim)
        # Live timestep is already * timestep_scale_multiplier; dummy uses 0.5.
        timestep = torch.tensor([[0.5]], dtype=torch.float32).expand(enc.shape[0], video_latents.shape[1])
        audio_timestep = torch.tensor([[0.5]], dtype=torch.float32).expand(
            enc.shape[0], audio_latents.shape[1],
        )
        return _layout_to_device(
            PackedLayout(
                hidden_states=video_latents,
                audio_hidden_states=audio_latents,
                encoder_hidden_states=video_e,
                audio_encoder_hidden_states=audio_e,
                timestep=timestep,
                audio_timestep=audio_timestep,
                encoder_attention_mask=conn_mask,
                audio_encoder_attention_mask=conn_mask,
                num_frames=int(num_frames),
                height=int(height),
                width=int(width),
                pre_connector_hidden=enc,
                n_prompt_tokens=int(n_tokens),
                hold_stage="pre_connector",
            ),
            self.device,
        )

    # H3-shaped alias used by tests that copy the pack name.
    pack_t2va = pack_t2v

    def forward_velocity(self, packed: PackedLayout, *, scale: float) -> DummyVelocity:
        """Actual LTX2 transformer forward. Not conceptmod ``predict_v``."""
        if self.network is None:
            raise RuntimeError("LoRA was not attached")
        self.network.set_lora_slider(float(scale))
        kwargs = dict(
            hidden_states=packed.hidden_states,
            audio_hidden_states=packed.audio_hidden_states,
            encoder_hidden_states=packed.encoder_hidden_states,
            audio_encoder_hidden_states=packed.audio_encoder_hidden_states,
            timestep=packed.timestep,
            audio_timestep=packed.audio_timestep,
            encoder_attention_mask=packed.encoder_attention_mask,
            audio_encoder_attention_mask=packed.audio_encoder_attention_mask,
            num_frames=max(1, (int(packed.num_frames) - 1) // LTX_FRAME_MOD + 1),
            height=max(1, int(packed.height) // LTX_CANVAS_MULTIPLE),
            width=max(1, int(packed.width) // LTX_CANVAS_MULTIPLE),
            fps=LTX_FPS,
            isolate_modalities=False,
            spatio_temporal_guidance_blocks=None,
            return_dict=True,
        )
        with self.network:
            out = self.transformer(**kwargs)
        if isinstance(out, tuple):
            return DummyVelocity(sample=out[0], audio_sample=out[1])
        return DummyVelocity(sample=out.sample, audio_sample=out.audio_sample)

    def predict_v(self, *args, **kwargs):
        raise ArchitectureMismatch(
            "LTX-2.5 is not a conceptmod v-pred DiT helper. Use "
            "forward_velocity on hidden_states + audio_hidden_states "
            "(LTX2VideoTransformer3DModel returns flow velocity; "
            "LTX2Pipeline.convert_velocity_to_x0 does x0 = x_t - sigma * v). "
            "Do not fake predict_v."
        )

    def save_trained(self, path: str) -> None:
        if self.network is None:
            return
        self.network.save_weights(path + ".safetensors", dtype=torch.float32)

    def load_trained(self, path: str) -> str:
        if self.network is None:
            raise RuntimeError("LoRA was not attached")
        resolved = resolve_ltx_lora_path(path)
        self.network.load_weights(str(resolved))
        if hasattr(self.network, "to"):
            self.network.to(self.device)
        return str(resolved)

    def generate_t2v(
        self,
        prompt: str,
        *,
        scale: float = 1.0,
        num_frames: int | None = None,
        height: int | None = None,
        width: int | None = None,
        seed: int = 7,
    ) -> dict[str, Any]:
        """Short t2v clip at LoRA ``scale``. Distilled card, conv VAE.

        Infer scale 1 uses the **neu** caption (caller passes neu).
        Distilled: explicit ``sigmas=DISTILLED_SIGMA_VALUES``,
        ``guidance_scale=1.0``, STG/modality 0. Does **not** pass
        ``num_inference_steps``. Prompt enhancer OFF.
        """
        if self.network is None:
            raise RuntimeError("LoRA was not attached")
        frames = int(num_frames or DEFAULT_NUM_FRAMES)
        h = int(height or DEFAULT_SAMPLE_HEIGHT)
        w = int(width or DEFAULT_SAMPLE_WIDTH)
        frames = ltx_num_frames(frames)
        h, w = ltx_canvas_hw(h, w)
        self.network.set_lora_slider(float(scale))
        with self.network:
            if self.dummy:
                out = self._dummy_generate_t2v(prompt, scale=scale, seed=seed)
            else:
                out = self._live_generate_t2v(
                    prompt, num_frames=frames, height=h, width=w, seed=seed,
                )
        out.setdefault("prompt", prompt)
        out.setdefault("scale", float(scale))
        out.setdefault("num_frames", frames)
        out.setdefault("height", h)
        out.setdefault("width", w)
        out.setdefault("guidance", 1.0)
        out.setdefault("stg_scale", 0.0)
        out.setdefault("modality_scale", 0.0)
        out.setdefault("sigmas", distilled_sigmas())
        out.setdefault("prompt_enhancer", False)
        out.setdefault("decoder", "conv_vae")
        out.setdefault("seed", int(seed))
        return out

    def _dummy_generate_t2v(self, prompt: str, *, scale: float, seed: int) -> dict[str, Any]:
        g = torch.Generator().manual_seed(int(seed) + int(round(float(scale) * 100)))
        frames = torch.zeros(2, 8, 8, 3)
        tint = (sum(ord(c) for c in prompt) % 180) / 255.0
        frames[..., 0] = min(1.0, tint + 0.15 * float(scale))
        frames[..., 1] = 0.25 + 0.2 * float(scale)
        frames[..., 2] = 0.4
        frames = frames + 0.02 * torch.rand(frames.shape, generator=g)
        frames = (frames.clamp(0, 1) * 255).to(torch.uint8)
        return {
            "videos": [frames],
            "audio": [torch.zeros(2, 32)],
            "sampling_rate": 16000,
            "dummy": True,
            "num_frames": 2,
            "height": 8,
            "width": 8,
        }

    def _live_generate_t2v(
        self,
        prompt: str,
        *,
        num_frames: int,
        height: int,
        width: int,
        seed: int,
    ) -> dict[str, Any]:
        pipe = self.pipe
        if pipe is None or not callable(pipe):
            raise RuntimeError("live t2v sample needs LTX2Pipeline.__call__")
        gen_device = self.device if self.device.type == "cuda" else "cpu"
        generator = torch.Generator(device=str(gen_device) if gen_device != "cpu" else "cpu")
        generator.manual_seed(int(seed))
        sigmas = distilled_sigmas()
        kwargs = dict(
            prompt=prompt,
            height=int(height),
            width=int(width),
            num_frames=int(num_frames),
            frame_rate=LTX_FPS,
            sigmas=sigmas,
            guidance_scale=1.0,
            audio_guidance_scale=1.0,
            stg_scale=0.0,
            audio_stg_scale=0.0,
            modality_scale=0.0,
            audio_modality_scale=0.0,
            enable_prompt_enhancement=False,
            generator=generator,
            output_type="np",
            return_dict=False,
        )
        # Distilled: never pass num_inference_steps (generic linear schedule).
        results = pipe(**kwargs)
        if isinstance(results, (list, tuple)):
            video = results[0] if results else None
            audio = results[1] if len(results) > 1 else None
        else:
            video = getattr(results, "frames", None) or getattr(results, "videos", None)
            audio = getattr(results, "audio", None)
        return {
            "videos": video,
            "audio": audio,
            "sampling_rate": getattr(getattr(pipe, "vocoder", None), "config", None)
            and getattr(pipe.vocoder.config, "output_sampling_rate", 16000) or 16000,
            "dummy": False,
        }


class _FrozenStub(nn.Module):
    def __init__(self, name: str) -> None:
        super().__init__()
        self._name = name
        self.dummy_param = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.requires_grad_(False)
        self.eval()


class _DummyPipe:
    def __init__(self, **mods: Any) -> None:
        for key, value in mods.items():
            setattr(self, key, value)

    def __call__(self, *args, **kwargs):
        raise RuntimeError("dummy pipe is not a live LTX2Pipeline; use generate_t2v")


def ltx_pack_feature_dim(transformer: nn.Module, *, kind: str) -> int:
    """Channel dim for a fake pack: ``proj_in.in_features`` (128), not inner_dim."""
    if kind == "video":
        proj = getattr(transformer, "proj_in", None)
        if proj is not None and hasattr(proj, "in_features"):
            return int(proj.in_features)
        return int(getattr(transformer, "video_dim", None) or getattr(transformer, "in_channels", VIDEO_IN_CHANNELS))
    if kind == "audio":
        proj = getattr(transformer, "audio_proj_in", None)
        if proj is not None and hasattr(proj, "in_features"):
            return int(proj.in_features)
        return int(getattr(transformer, "audio_dim", None) or getattr(transformer, "audio_in_channels", 8))
    raise ValueError(f"kind must be video or audio, got {kind!r}")


def ltx_latent_tokens(num_frames: int, height: int, width: int) -> int:
    lf = max(1, (int(num_frames) - 1) // LTX_FRAME_MOD + 1)
    lh = max(1, int(height) // LTX_CANVAS_MULTIPLE)
    lw = max(1, int(width) // LTX_CANVAS_MULTIPLE)
    return lf * lh * lw


def ltx_num_frames(num_frames: int) -> int:
    """Snap up to ``8k + 1``."""
    raw = max(1, int(num_frames))
    n = 0
    frames = LTX_FRAME_BIAS
    while frames < raw:
        n += 1
        frames = LTX_FRAME_MOD * n + LTX_FRAME_BIAS
    return int(frames)


def ltx_canvas_hw(height: int, width: int) -> tuple[int, int]:
    h = max(LTX_CANVAS_MULTIPLE, (int(height) // LTX_CANVAS_MULTIPLE) * LTX_CANVAS_MULTIPLE)
    w = max(LTX_CANVAS_MULTIPLE, (int(width) // LTX_CANVAS_MULTIPLE) * LTX_CANVAS_MULTIPLE)
    return h, w


def resolve_ltx_lora_path(path: str):
    from pathlib import Path

    p = Path(path)
    candidates = []
    if p.is_file():
        return p
    if p.suffix == ".safetensors":
        candidates.append(p)
    candidates.append(Path(str(p) + ".safetensors"))
    if p.is_dir():
        candidates.extend(sorted(p.glob("*_lora.safetensors")))
        candidates.extend(sorted(p.glob("*.safetensors")))
    for cand in candidates:
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        f"no LTX-2.5 LoRA safetensors under {path} "
        "(expected {name}_lora.safetensors from save_trained)"
    )


def _module_param_dtype(module: nn.Module):
    for param in module.parameters():
        return param.dtype
    return None


def _init_peft_lora_up(module: nn.Module, *, std: float) -> None:
    """PEFT defaults lora_B / up to zeros — UNI identity. Draw N(0, std)."""
    if float(std) <= 0:
        return
    for child in module.modules():
        lora_b = getattr(child, "lora_B", None)
        if isinstance(lora_b, nn.ModuleDict):
            for key in lora_b:
                if hasattr(lora_b[key], "weight"):
                    nn.init.normal_(lora_b[key].weight, mean=0.0, std=float(std))
        elif lora_b is not None and hasattr(lora_b, "weight"):
            nn.init.normal_(lora_b.weight, mean=0.0, std=float(std))


def _layout_to_device(packed: PackedLayout, device) -> PackedLayout:
    fields = {}
    for name in PackedLayout.__dataclass_fields__:
        value = getattr(packed, name)
        fields[name] = value.to(device) if isinstance(value, torch.Tensor) else value
    return PackedLayout(**fields)


def place_ltx25_pipeline(
    pipe: Any,
    *,
    device: str,
    encoder_device: str | None = DEFAULT_ENCODER_DEVICE,
) -> Any:
    """Place TE on ``encoder_device`` (default CPU) and DiT on ``device``.

    Does **not** call blanket ``pipe.to`` — that would put Gemma 4 12B
    and the 22B DiT on the same GPU.
    """
    enc = encoder_device or DEFAULT_ENCODER_DEVICE
    for name in ("transformer",):
        mod = getattr(pipe, name, None)
        if mod is not None and hasattr(mod, "to"):
            mod.to(device)
    for name in ("vae",):
        mod = getattr(pipe, name, None)
        if mod is not None and hasattr(mod, "to"):
            mod.to(device)
    for name in ("text_encoder", "connectors", "prompt_enhancer"):
        mod = getattr(pipe, name, None)
        if mod is not None and hasattr(mod, "to"):
            mod.to(enc)
    for name in ("audio_vae", "vocoder", "duration_head"):
        mod = getattr(pipe, name, None)
        if mod is not None and hasattr(mod, "to"):
            try:
                mod.to(enc)
            except Exception:
                pass
    return pipe


def _load_ltx25_pipeline(
    model_id: str,
    *,
    device: str = "cuda:0",
    encoder_device: str = DEFAULT_ENCODER_DEVICE,
    transformer_subfolder: str = DEFAULT_TRANSFORMER_SUBFOLDER,
):
    """Live path. Dummy never calls this. Do not download LTX-2.5 in CI.

    First download excludes ``transformer_full/``. Distilled
    ``transformer/`` is what ``model_index.json`` points at.
    """
    from diffusers import LTX2Pipeline

    kwargs: dict[str, Any] = {"dtype": torch.bfloat16}
    if transformer_subfolder == DEFAULT_TRANSFORMER_SUBFOLDER:
        kwargs["ignore_patterns"] = list(IGNORE_ON_FIRST_DOWNLOAD)
    try:
        pipe = LTX2Pipeline.from_pretrained(model_id, **kwargs)
    except TypeError:
        kwargs.pop("ignore_patterns", None)
        try:
            pipe = LTX2Pipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        except TypeError:
            pipe = LTX2Pipeline.from_pretrained(model_id)
    if transformer_subfolder == FULL_TRANSFORMER_SUBFOLDER:
        from diffusers import LTX2VideoTransformer3DModel

        pipe.transformer = LTX2VideoTransformer3DModel.from_pretrained(
            model_id, subfolder=FULL_TRANSFORMER_SUBFOLDER, dtype=torch.bfloat16,
        )
    place_ltx25_pipeline(pipe, device=device, encoder_device=encoder_device)
    return pipe
