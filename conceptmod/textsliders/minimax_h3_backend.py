"""MiniMax-H3 backend: dummy packed-sequence Omni-Transformer, live ModularPipeline.

Live hub id is ``MiniMaxAI/MiniMax-H3`` (FL2VA / t2va). Dummy never imports
that loader and never hits the Hub.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from conceptmod.textsliders.minimax_h3_uni import apply_unused_hold

DEFAULT_MODEL = "MiniMaxAI/MiniMax-H3"
DEFAULT_VARIANT = "FL2VA"
DEFAULT_WORKFLOW = "t2va"
DEFAULT_TASK_INDEX = "FL2VA/model_index.json"
REF2VA_TASK_INDEX = "Ref2VA/model_index.json"
# Live t2va sample defaults that fit one B200/B300 (~135 GB resident).
# H3 snaps frames to 17*n+5; 5.0s @ 24 fps → 124 frames (~5.17s).
# Official generate window is 5–15s; 4s is below the floor.
H3_FPS = 24.0
H3_MIN_DURATION = 5.0
H3_MAX_DURATION = 15.0
H3_FRAME_MOD = 17
H3_FRAME_BIAS = 5
H3_CANVAS_MULTIPLE = 32
DEFAULT_SAMPLE_SCALES = (0.0, 1.0)
DEFAULT_SAMPLE_DURATION = 5.0
DEFAULT_SAMPLE_ASPECT = "16:9"
# Diffusers ModularPipeline attn names on MiniMaxH3Transformer3DModel.
LORA_ATTN_CLASS = "MiniMaxH3Attention"
LORA_LINEAR_NAMES = ("to_q", "to_k", "to_v", "to_out.0")
# Original FL2VA MiniMaxH3DiTModel checkpoint uses a fused QKV. Live load is
# ModularPipeline / MiniMaxH3Transformer3DModel, which splits those into to_q/k/v.
FL2VA_NATIVE_ATTN_NAMES = ("qkv_proj", "out_proj")
FREEZE_LIST = (
    "text_encoder",  # H3-Encoder = full Qwen3-VL-32B, layer-50 hidden
    "visual_vae",
    "audio_vae",
    "processor",
    "tokenizer",
)
HOSTED_NOT_IN_WEIGHTS = ("H3-Context-IR", "H3-Regenerate-2K")


class ArchitectureMismatch(RuntimeError):
    """Raised when a conceptmod DiT helper does not exist on MiniMax-H3."""


class DummyTokenizer:
    """Whitespace tokenizer with a stable word vocab. Not the H3 Qwen tokenizer."""

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = []
        for word in str(text).split():
            key = word.lower()
            if key not in self._vocab:
                self._vocab[key] = len(self._vocab) + 1
            ids.append(self._vocab[key])
        return ids or [1]


class MiniMaxH3Attention(nn.Module):
    """Tiny stand-in whose Linear names match ``MiniMaxH3Transformer3DModel``.

    Live modules: ``to_q`` / ``to_k`` / ``to_v`` / ``to_out.0``. AdaLN lives
    beside this class, not inside it.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.to_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(hidden_size, hidden_size, bias=False)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        q = self.to_q(hidden_states)
        k = self.to_k(hidden_states)
        v = self.to_v(hidden_states)
        scale = hidden_states.shape[-1] ** -0.5
        attn = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
        return self.to_out[0](attn @ v)


class DummyOmniBlock(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.attn = MiniMaxH3Attention(hidden_size)
        # Named like the live checkpoint. Never wrapped by attn LoRA.
        self.adaln_proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        shift = self.adaln_proj(F.silu(temb))
        return hidden_states + self.attn(hidden_states + shift.unsqueeze(1))


class DummyOmniTransformer(nn.Module):
    """Packed-sequence velocity teacher. Mirrors the live forward contract.

    ``forward`` returns data-pointing video / audio velocity on a tiny packed
    layout (text, video, audio rows + MM-RoPE ``(t, h, w)``). It is not a
    conceptmod ``predict_v`` wrapper.
    """

    def __init__(
        self,
        hidden_size: int = 16,
        text_dim: int = 8,
        video_dim: int = 4,
        audio_dim: int = 4,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.text_dim = text_dim
        self.video_dim = video_dim
        self.audio_dim = audio_dim
        self.context_embedder = nn.Linear(text_dim, hidden_size)
        self.proj_in = nn.Linear(video_dim, hidden_size)
        self.audio_proj_in = nn.Linear(audio_dim, hidden_size)
        self.time_embedder = nn.Linear(1, hidden_size)
        self.transformer_blocks = nn.ModuleList([DummyOmniBlock(hidden_size)])
        self.proj_out = nn.Linear(hidden_size, video_dim)
        self.audio_proj_out = nn.Linear(hidden_size, audio_dim)
        # 3-axis MM-RoPE stand-in: (t, h, w) → hidden bias.
        self.mm_rope = nn.Linear(3, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        timestep_indices: torch.Tensor,
        token_tags: torch.Tensor,
        position_ids: torch.Tensor,
        video_indices: torch.Tensor,
        audio_indices: torch.Tensor,
        text_indices: torch.Tensor,
        attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
    ):
        if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
            raise ValueError(f"position_ids must be (seq, 3), got {tuple(position_ids.shape)}")
        text_embeds = self.context_embedder(encoder_hidden_states)
        video_embeds = self.proj_in(hidden_states)
        audio_embeds = self.audio_proj_in(audio_hidden_states)
        seq_len = int(position_ids.shape[0])
        packed = text_embeds.new_zeros((text_embeds.shape[0], seq_len, text_embeds.shape[-1]))
        packed = packed.index_copy(1, text_indices, text_embeds)
        packed = packed.index_copy(1, video_indices, video_embeds)
        packed = packed.index_copy(1, audio_indices, audio_embeds)
        packed = packed + self.mm_rope(position_ids.to(packed.dtype)).unsqueeze(0)
        temb = self.time_embedder(timestep.reshape(-1, 1).to(packed.dtype))
        if temb.shape[0] == 1 and packed.shape[0] > 1:
            temb = temb.expand(packed.shape[0], -1)
        elif temb.shape[0] != packed.shape[0]:
            temb = temb[:1].expand(packed.shape[0], -1)
        for block in self.transformer_blocks:
            packed = block(packed, temb)
        video_output = self.proj_out(packed.index_select(1, video_indices))
        audio_output = self.audio_proj_out(packed.index_select(1, audio_indices))
        if return_dict:
            return DummyVelocity(sample=video_output, audio_sample=audio_output)
        return video_output, audio_output


@dataclass
class DummyVelocity:
    sample: torch.Tensor
    audio_sample: torch.Tensor


@dataclass
class EncodedText:
    embeds: torch.Tensor
    token_ids: list[int]


@dataclass
class PackedLayout:
    hidden_states: torch.Tensor
    audio_hidden_states: torch.Tensor
    encoder_hidden_states: torch.Tensor
    timestep: torch.Tensor
    timestep_indices: torch.Tensor
    token_tags: torch.Tensor
    position_ids: torch.Tensor
    video_indices: torch.Tensor
    audio_indices: torch.Tensor
    text_indices: torch.Tensor


class _AttnLoRA(nn.Module):
    """LoRA on one Linear. Dummy / live share this; does not import lora.py."""

    def __init__(self, name: str, module: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.lora_name = name
        self.rank = rank
        self.scale = float(alpha) / float(rank)
        self.multiplier = 1.0
        self.lora_down = nn.Linear(module.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, module.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_up.weight)
        self.org_forward = module.forward
        module.forward = self.forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.lora_up(self.lora_down(x))
        return self.org_forward(x) + delta * (self.multiplier * self.scale)


class AttnLoRANetwork(nn.Module):
    """Wrap ``MiniMaxH3Attention`` ``to_q/to_k/to_v/to_out.0`` only. Skip AdaLN."""

    def __init__(self, transformer: nn.Module, rank: int, alpha: float) -> None:
        super().__init__()
        self.lora_scale = 1.0
        self.loras: list[_AttnLoRA] = []
        for name, module in transformer.named_modules():
            if module.__class__.__name__ != LORA_ATTN_CLASS:
                continue
            for child_name, child in module.named_modules():
                if not isinstance(child, nn.Linear):
                    continue
                if child_name not in ("to_q", "to_k", "to_v", "to_out.0"):
                    continue
                lora_name = f"lora_h3-{name}-{child_name}".replace(".", "-")
                lora = _AttnLoRA(lora_name, child, rank, alpha)
                self.loras.append(lora)
                self.add_module(lora_name, lora)

    def set_lora_slider(self, scale: float) -> None:
        self.lora_scale = float(scale)

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
        """Load custom ``lora_h3-…`` keys. Not PEFT ``adapter_model``."""
        from safetensors.torch import load_file

        state = load_file(file)
        h3_keys = [k for k in state if str(k).startswith("lora_h3-")]
        if not h3_keys:
            raise ValueError(
                f"{file} has no lora_h3-* keys. MiniMax-H3 sliders save a "
                "custom AttnLoRANetwork, not PEFT adapter_model.safetensors."
            )
        missing, unexpected = self.load_state_dict(state, strict=False)
        missing_lora = [k for k in missing if str(k).startswith("lora_h3-")]
        if missing_lora:
            raise ValueError(f"{file} is missing LoRA keys: {missing_lora[:8]}")
        _ = unexpected


class DummyEncoder(nn.Module):
    """Frozen stand-in for H3-Encoder (live: Qwen3-VL-32B layer-50 hidden)."""

    def __init__(self, vocab: int = 128, dim: int = 8) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.mix = nn.Linear(dim, dim, bias=False)
        self.requires_grad_(False)
        self.eval()

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        raw = self.embed(token_ids)
        # Tiny context mix so unused tokens differ across plus vs neu captions
        # until apply_unused_hold copies encode(neu) onto those rows.
        ctx = raw.mean(dim=1, keepdim=True).expand_as(raw)
        return self.mix(raw + 0.15 * ctx)


class MiniMaxH3Backend:
    """Opt-in MiniMax-H3 slider host. Dummy is the CI path."""

    def __init__(
        self,
        *,
        device: str = "cpu",
        encoder_device: str | None = None,
        model_id: str = DEFAULT_MODEL,
        variant: str = DEFAULT_VARIANT,
        workflow: str = DEFAULT_WORKFLOW,
        short_side: int = 768,
        lora_rank: int = 8,
        lora_alpha: float = 8.0,
        dummy: bool = False,
    ) -> None:
        self.device = torch.device(device if not dummy else "cpu")
        # Dual-GPU escape hatch: encoder (Qwen3-VL-32B) on a second card.
        # Dummy stays CPU. When unset, encoder shares ``device`` (B200/B300).
        if dummy:
            self.encoder_device = torch.device("cpu")
        elif encoder_device:
            self.encoder_device = torch.device(encoder_device)
        else:
            self.encoder_device = self.device
        self.model_id = model_id
        self.variant = variant
        self.workflow = workflow
        self.short_side = int(short_side)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.dummy = bool(dummy)
        self.guidance = 0.0  # CFG-distilled
        self.pipe: Any = None
        self.tokenizer: Any
        self.encoder: Any
        self.transformer: nn.Module
        self.network: AttnLoRANetwork | None = None
        self.frozen: nn.Module | None
        self.visual_vae = None
        self.audio_vae = None
        self.processor = None
        if self.dummy:
            self._init_dummy()
        else:
            self._init_live()

    def _init_dummy(self) -> None:
        self.tokenizer = DummyTokenizer()
        self.encoder = DummyEncoder()
        self.transformer = DummyOmniTransformer()
        self.visual_vae = _FrozenStub("visual_vae")
        self.audio_vae = _FrozenStub("audio_vae")
        self.processor = _FrozenStub("processor")
        self.frozen = None  # LoRA off == teacher
        self.network = self._attach_lora(self.transformer, perturb=True)
        self.pipe = _DummyPipe(
            tokenizer=self.tokenizer,
            text_encoder=self.encoder,
            transformer=self.transformer,
            visual_vae=self.visual_vae,
            audio_vae=self.audio_vae,
            processor=self.processor,
        )

    def _init_live(self) -> None:
        self.pipe = _load_minimax_h3_modular(
            self.model_id,
            workflow=self.workflow,
            device=str(self.device),
            encoder_device=str(self.encoder_device),
        )
        self.tokenizer = self.pipe.tokenizer
        self.encoder = self.pipe.text_encoder
        self.transformer = self.pipe.transformer
        self.visual_vae = getattr(self.pipe, "vae", None) or getattr(self.pipe, "visual_vae", None)
        self.audio_vae = getattr(self.pipe, "audio_vae", None)
        self.processor = getattr(self.pipe, "processor", None)
        for name in FREEZE_LIST:
            mod = getattr(self.pipe, name, None) if name != "visual_vae" else self.visual_vae
            if name == "text_encoder":
                mod = self.encoder
            if mod is None:
                continue
            if hasattr(mod, "requires_grad_"):
                mod.requires_grad_(False)
            if hasattr(mod, "eval"):
                mod.eval()
        self.transformer.requires_grad_(False)
        self.transformer.eval()
        self.frozen = None
        self.network = self._attach_lora(self.transformer, perturb=False)

    def _attach_lora(self, transformer: nn.Module, *, perturb: bool) -> AttnLoRANetwork:
        network = AttnLoRANetwork(transformer, rank=self.lora_rank, alpha=self.lora_alpha)
        if perturb:
            # Zero-init up would make UNI identity at step 0. A small dummy
            # offset gives the smoke a nonzero first loss that SGD can drop.
            for lora in network.loras:
                nn.init.normal_(lora.lora_up.weight, mean=0.0, std=0.25)
        network.to(self.device)
        return network

    def lora_module_names(self) -> list[str]:
        if self.network is None:
            return []
        return [lora.lora_name for lora in self.network.loras]

    def trainable_parameters(self) -> list[nn.Parameter]:
        if self.network is None:
            return []
        return [p for p in self.network.parameters() if p.requires_grad]

    def encode_text(self, prompt: str, frozen: bool = True) -> EncodedText:
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if not ids:
            ids = [1]
        if self.dummy:
            tensor = torch.tensor([ids], dtype=torch.long)
            with torch.no_grad() if frozen else torch.enable_grad():
                embeds = self.encoder(tensor)
            return EncodedText(embeds=embeds, token_ids=[int(x) for x in ids])
        return self._live_encode_text(prompt)

    def _live_encode_text(self, prompt: str) -> EncodedText:
        """H3-Encoder: Qwen3-VL-32B hidden states from layer 50, H3 tokenizer."""
        tok = self.tokenizer
        batch = tok(prompt, return_tensors="pt")
        ids = batch["input_ids"]
        if hasattr(ids, "tolist"):
            raw_ids = ids.tolist()[0]
        else:
            raw_ids = list(ids)
        encoder = self.encoder
        enc_dev = self.encoder_device
        with torch.no_grad():
            if hasattr(encoder, "model"):
                out = encoder.model(
                    input_ids=ids.to(enc_dev),
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden = out.hidden_states[50]
            else:
                out = encoder(
                    input_ids=ids.to(enc_dev),
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden = out.hidden_states[50]
        # Hidden states are small; park them on the transformer/VAE device.
        if hidden.device != self.device:
            hidden = hidden.to(self.device)
        return EncodedText(embeds=hidden, token_ids=[int(x) for x in raw_ids])

    def pack_t2va(
        self,
        text: EncodedText,
        *,
        video_latents: torch.Tensor | None = None,
        audio_latents: torch.Tensor | None = None,
        hold_neu: EncodedText | None = None,
        hold_mask: torch.Tensor | None = None,
    ) -> PackedLayout:
        """Build a t2va packed layout. Dummy uses a tiny fake sequence."""
        enc = text.embeds
        if hold_neu is not None and hold_mask is not None:
            enc = apply_unused_hold(
                enc, hold_neu.embeds, text.token_ids, hold_neu.token_ids, hold_mask,
            )
        n_text = enc.shape[1]
        video_dim = int(getattr(self.transformer, "video_dim", None) or getattr(self.transformer, "in_channels", 24))
        audio_dim = int(getattr(self.transformer, "audio_dim", None) or getattr(self.transformer, "audio_in_channels", 32))
        if video_latents is None:
            video_latents = torch.randn(enc.shape[0], 2, video_dim)
        if audio_latents is None:
            audio_latents = torch.randn(enc.shape[0], 2, audio_dim)
        n_video = video_latents.shape[1]
        n_audio = audio_latents.shape[1]
        # Packed order: text, then video, then audio (t2va, no keyframes).
        text_indices = torch.arange(n_text, dtype=torch.long)
        video_indices = torch.arange(n_text, n_text + n_video, dtype=torch.long)
        audio_indices = torch.arange(n_text + n_video, n_text + n_video + n_audio, dtype=torch.long)
        seq = n_text + n_video + n_audio
        token_tags = torch.cat((
            torch.ones(n_text, dtype=torch.long),       # 1 = text
            torch.zeros(n_video, dtype=torch.long),     # 0 = video
            torch.full((n_audio,), 2, dtype=torch.long),  # 2 = audio
        ))
        # MM-RoPE (t, h, w). Text rows sit at the origin; video walks a 1x2
        # spatial strip; audio walks time only (stereo channels share t).
        pos = torch.zeros(seq, 3, dtype=torch.float32)
        for i in range(n_video):
            pos[int(video_indices[i])] = torch.tensor([0.0, 0.0, float(i)])
        for i in range(n_audio):
            pos[int(audio_indices[i])] = torch.tensor([float(i), 0.0, 0.0])
        timestep = torch.tensor([0.5], dtype=torch.float32)
        timestep_indices = torch.zeros(seq, dtype=torch.long)
        return _layout_to_device(
            PackedLayout(
                hidden_states=video_latents,
                audio_hidden_states=audio_latents,
                encoder_hidden_states=enc,
                timestep=timestep,
                timestep_indices=timestep_indices,
                token_tags=token_tags,
                position_ids=pos,
                video_indices=video_indices,
                audio_indices=audio_indices,
                text_indices=text_indices,
            ),
            self.device,
        )

    def forward_velocity(self, packed: PackedLayout, *, scale: float) -> DummyVelocity:
        """Actual Omni-Transformer forward. Not conceptmod ``predict_v``."""
        if self.network is None:
            raise RuntimeError("LoRA was not attached")
        # Always enter the network: LoRAModule starts at multiplier=1. Scale 0
        # must zero the adapter; skipping __enter__ would leak the init scale.
        self.network.set_lora_slider(float(scale))
        kwargs = dict(
            hidden_states=packed.hidden_states,
            audio_hidden_states=packed.audio_hidden_states,
            encoder_hidden_states=packed.encoder_hidden_states,
            timestep=packed.timestep,
            timestep_indices=packed.timestep_indices,
            token_tags=packed.token_tags,
            position_ids=packed.position_ids,
            video_indices=packed.video_indices,
            audio_indices=packed.audio_indices,
            text_indices=packed.text_indices,
            return_dict=True,
        )
        with self.network:
            out = self.transformer(**kwargs)
        if isinstance(out, tuple):
            return DummyVelocity(sample=out[0], audio_sample=out[1])
        return DummyVelocity(sample=out.sample, audio_sample=out.audio_sample)

    def predict_v(self, *args, **kwargs):
        raise ArchitectureMismatch(
            "MiniMax-H3 is not a conceptmod v-pred DiT helper. Use "
            "forward_velocity on the packed multimodal sequence "
            "(MiniMaxH3Transformer3DModel returns data-pointing velocity "
            "x0 = x_t + sigma * v via MiniMaxH3Scheduler). Do not fake predict_v."
        )

    def save_trained(self, path: str) -> None:
        if self.network is None:
            return
        self.network.save_weights(path + ".safetensors", dtype=torch.float32)

    def load_trained(self, path: str) -> str:
        """Load a directory or ``.safetensors`` written by ``save_trained``."""
        if self.network is None:
            raise RuntimeError("LoRA was not attached")
        resolved = resolve_h3_lora_path(path)
        self.network.load_weights(str(resolved))
        self.network.to(self.device)
        return str(resolved)

    def generate_t2va(
        self,
        prompt: str,
        *,
        scale: float = 1.0,
        num_frames: int | None = None,
        height: int | None = None,
        width: int | None = None,
        duration: float = DEFAULT_SAMPLE_DURATION,
        fps: float = H3_FPS,
        short_side: int | None = None,
        seed: int = 7,
    ) -> dict[str, Any]:
        """Short t2va clip at LoRA ``scale``. Guidance stays 0 (CFG-distilled).

        Dummy writes tiny synthetic frames (no Hub, no codec). Live calls
        ModularPipeline ``pipe(prompt=..., num_frames=...)`` and does **not**
        pass ``guidance_scale``.
        """
        if self.network is None:
            raise RuntimeError("LoRA was not attached")
        side = int(short_side or self.short_side)
        h, w = h3_canvas_hw(side)
        if height is not None:
            h = int(height)
        if width is not None:
            w = int(width)
        frames = int(num_frames or h3_num_frames(duration, fps))
        self.network.set_lora_slider(float(scale))
        with self.network:
            if self.dummy:
                out = self._dummy_generate_t2va(prompt, scale=scale, seed=seed)
            else:
                out = self._live_generate_t2va(
                    prompt, num_frames=frames, height=h, width=w, seed=seed,
                )
        out.setdefault("prompt", prompt)
        out.setdefault("scale", float(scale))
        out.setdefault("num_frames", frames)
        out.setdefault("height", h)
        out.setdefault("width", w)
        out.setdefault("duration", float(duration))
        out.setdefault("fps", float(fps))
        out.setdefault("guidance", 0.0)
        out.setdefault("seed", int(seed))
        return out

    def _dummy_generate_t2va(self, prompt: str, *, scale: float, seed: int) -> dict[str, Any]:
        """CPU stand-in: 2×8×8 RGB ramps that vary with prompt + scale."""
        g = torch.Generator().manual_seed(int(seed) + int(round(float(scale) * 100)))
        frames = torch.zeros(2, 8, 8, 3)
        tint = (sum(ord(c) for c in prompt) % 180) / 255.0
        frames[..., 0] = min(1.0, tint + 0.15 * float(scale))
        frames[..., 1] = 0.25 + 0.2 * float(scale)
        frames[..., 2] = 0.4
        frames = frames + 0.02 * torch.rand(frames.shape, generator=g)
        frames = (frames.clamp(0, 1) * 255).to(torch.uint8)
        audio = torch.zeros(2, 32)
        return {
            "videos": [frames],
            "audio": [audio],
            "sampling_rate": 32000,
            "dummy": True,
            "num_frames": 2,
            "height": 8,
            "width": 8,
        }

    def _live_generate_t2va(
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
            raise RuntimeError("live t2va sample needs ModularPipeline.__call__")
        gen_device = self.device if self.device.type == "cuda" else "cpu"
        generator = torch.Generator(device=str(gen_device) if gen_device != "cpu" else "cpu")
        generator.manual_seed(int(seed))
        kwargs = dict(
            prompt=prompt,
            num_frames=int(num_frames),
            height=int(height),
            width=int(width),
            generator=generator,
            output=["videos", "audio", "sampling_rate"],
        )
        # CFG-distilled: never pass guidance_scale / negative_prompt.
        results = pipe(**kwargs)
        if not isinstance(results, dict):
            results = {
                "videos": getattr(results, "videos", None),
                "audio": getattr(results, "audio", None),
                "sampling_rate": getattr(results, "sampling_rate", 32000),
            }
        return {
            "videos": results.get("videos"),
            "audio": results.get("audio"),
            "sampling_rate": results.get("sampling_rate", 32000),
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
        raise RuntimeError("dummy pipe is not a live ModularPipeline; use generate_t2va")


def same_device(a, b) -> bool:
    da, db = torch.device(a), torch.device(b)
    if da.type != db.type:
        return False
    if da.type == "cuda":
        ia = 0 if da.index is None else da.index
        ib = 0 if db.index is None else db.index
        return ia == ib
    return True


def h3_num_frames(duration: float, fps: float = H3_FPS) -> int:
    """Snap ``duration * fps`` up to the next ``17*n + 5`` H3 can decode."""
    raw = max(1, int(round(float(duration) * float(fps))))
    n = 0
    frames = H3_FRAME_BIAS
    while frames < raw:
        n += 1
        frames = H3_FRAME_MOD * n + H3_FRAME_BIAS
    return int(frames)


def h3_canvas_hw(short_side: int, aspect: str = DEFAULT_SAMPLE_ASPECT) -> tuple[int, int]:
    """Landscape canvas: short side is height. Both axes snap to 32."""
    try:
        w_r, h_r = (int(x) for x in str(aspect).split(":"))
    except ValueError:
        w_r, h_r = 16, 9
    height = int(short_side)
    width = int(round(float(short_side) * float(w_r) / float(h_r)))
    height = max(H3_CANVAS_MULTIPLE, (height // H3_CANVAS_MULTIPLE) * H3_CANVAS_MULTIPLE)
    width = max(H3_CANVAS_MULTIPLE, (width // H3_CANVAS_MULTIPLE) * H3_CANVAS_MULTIPLE)
    return height, width


def resolve_h3_lora_path(path: str):
    """Accept a dir, a ``.safetensors`` file, or the stem ``save_trained`` writes."""
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
        f"no MiniMax-H3 LoRA safetensors under {path} "
        "(expected {name}_lora.safetensors from save_trained)"
    )


def _layout_to_device(packed: PackedLayout, device) -> PackedLayout:
    fields = {}
    for name in PackedLayout.__dataclass_fields__:
        value = getattr(packed, name)
        fields[name] = value.to(device) if isinstance(value, torch.Tensor) else value
    return PackedLayout(**fields)


def place_minimax_h3_pipeline(
    pipe: Any,
    *,
    device: str,
    encoder_device: str | None = None,
) -> Any:
    """Place a live ModularPipeline.

    Default (``encoder_device`` unset or the same as ``device``): blanket
    ``pipe.to(device)`` — one B200/B300 holds transformer + Qwen3-VL + VAEs
    (~135 GB bf16).

    Dual-GPU (``encoder_device`` differs): transformer + visual/audio VAEs
    stay on ``device``; the encoder goes to ``encoder_device``. Does **not**
    call blanket ``pipe.to``.
    """
    enc = encoder_device or device
    if same_device(enc, device):
        if hasattr(pipe, "to"):
            try:
                pipe.to(device)
            except Exception:
                pass
        return pipe
    for name in ("transformer", "vae", "visual_vae", "audio_vae"):
        mod = getattr(pipe, name, None)
        if mod is not None and hasattr(mod, "to"):
            mod.to(device)
    text_encoder = getattr(pipe, "text_encoder", None)
    if text_encoder is not None and hasattr(text_encoder, "to"):
        text_encoder.to(enc)
    return pipe


def _load_minimax_h3_modular(
    model_id: str,
    *,
    workflow: str = DEFAULT_WORKFLOW,
    device: str = "cuda:0",
    encoder_device: str | None = None,
):
    """Live path. Dummy never calls this. Do not download MiniMax-H3 in CI."""
    from diffusers import ModularPipeline

    pipe = ModularPipeline.from_pretrained(model_id, workflow=workflow)
    pipe.load_components(workflow=workflow, dtype=torch.bfloat16)
    place_minimax_h3_pipeline(pipe, device=device, encoder_device=encoder_device)
    return pipe
