"""Live Krea 2 velocity backend for ``train_lora_krea.py``.

Lazy-imported from ``_load_live_backend`` only. ``--dummy`` must never
import this module (no Hub, no 12B, no peft).

Patterns follow mikkel/conceptmod ``conceptmod/backends/krea.py``:

- ``Krea2Pipeline.from_pretrained(..., dtype=bfloat16)`` (falls back to
  ``torch_dtype=`` on older Diffusers)
- LoRA-only on DiT attn ``to_q/to_k/to_v/to_out.0`` via peft (rank 16)
- frozen ref = adapter disabled (a second 12B copy will not fit)
- Park VAE on CPU; move text encoder onto GPU only for encode, then
  park it before DiT backward so a 48GB card can train
- Raw: ~28 steps, CFG 4.5; Turbo (local ``.safetensors`` or hub id
  containing ``turbo``): 8 steps, CFG 0
- Official: train LoRAs on Raw, run on Turbo
- DiT LoRA only — text encoder stays frozen. No encoder LoRA.

The public methods match ``DummyKreaBackend`` so ``krea_step_loss`` is
unchanged: ``encode_text``, ``predict_v``, ``trainable_parameters``,
plus ``dim`` / ``latent_shape`` / ``generate``.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn

from conceptmod.textsliders.slider_targets import (
    KREA_DEFAULT_RANK,
    KREA_DEFAULT_RESOLUTION,
    KREA_LORA_TARGETS,
    KREA_RAW_CFG,
    KREA_RAW_MODEL,
    KREA_RAW_STEPS,
    KREA_TURBO_CFG,
    KREA_TURBO_STEPS,
    krea_hold_unused_embeds,
    krea_looks_turbo,
    krea_unused_hold_mask,
    krea_word_tokens,
)

_TEXT_CACHE_MAX = 16
_SKELETON_MODEL = KREA_RAW_MODEL


def _calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 6400,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    try:
        from diffusers.pipelines.krea2.pipeline_krea2 import calculate_shift

        return float(
            calculate_shift(
                image_seq_len, base_seq_len, max_seq_len, base_shift, max_shift
            )
        )
    except Exception:
        slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        intercept = base_shift - slope * base_seq_len
        return float(image_seq_len * slope + intercept)


def _from_pretrained(cls, model_id: str, **kwargs):
    """Krea docs use ``dtype=``; some Diffusers builds still want ``torch_dtype=``."""
    try:
        return cls.from_pretrained(model_id, **kwargs)
    except TypeError as exc:
        if "dtype" not in kwargs:
            raise
        fallback = dict(kwargs)
        fallback["torch_dtype"] = fallback.pop("dtype")
        try:
            return cls.from_pretrained(model_id, **fallback)
        except Exception as retry:
            raise retry from exc


def _load_pipeline(model_id: str, *, allow_hub: bool):
    from diffusers import Krea2Pipeline

    local_files_only = not bool(allow_hub)
    kwargs = {"dtype": torch.bfloat16, "local_files_only": local_files_only}
    local = None
    try:
        from conceptmod.textsliders.krea_weights import (
            SKELETON_MODEL,
            load_comfy_krea_transformer,
            looks_turbo,
            resolve_local_transformer,
        )

        local = resolve_local_transformer(model_id)
    except Exception:
        local = None
        SKELETON_MODEL = _SKELETON_MODEL
        looks_turbo = lambda path: krea_looks_turbo(str(path))  # noqa: E731
        load_comfy_krea_transformer = None

    if local is None:
        return _from_pretrained(Krea2Pipeline, model_id, **kwargs)

    print(f"krea local transformer: {local}")
    if load_comfy_krea_transformer is None:
        raise RuntimeError(
            f"local Krea .safetensors {local} needs safetensors + "
            "diffusers.Krea2Transformer2DModel to remap ComfyUI / NVFP4 keys"
        )
    transformer = load_comfy_krea_transformer(local, skeleton=SKELETON_MODEL)
    pipe = _from_pretrained(
        Krea2Pipeline,
        SKELETON_MODEL,
        transformer=transformer,
        **kwargs,
    )
    if looks_turbo(local) and not bool(getattr(pipe.config, "is_distilled", False)):
        pipe.register_to_config(is_distilled=True)
        print("krea: local file looks like turbo; is_distilled=True")
    return pipe


def _set_adapter_scale(module, scale: float) -> bool:
    if not hasattr(module, "set_adapter_scale"):
        return False
    try:
        module.set_adapter_scale(float(scale))
        return True
    except TypeError:
        try:
            module.set_adapter_scale("default", float(scale))
            return True
        except Exception:
            return False


class LiveKreaBackend:
    """12B Krea DiT + parked Qwen3-VL. Dummy-compatible UNI surface."""

    def __init__(
        self,
        device: torch.device,
        model_id: str = KREA_RAW_MODEL,
        resolution: int = KREA_DEFAULT_RESOLUTION,
        rank: int = KREA_DEFAULT_RANK,
        sample_steps: int | None = None,
        sample_guidance: float | None = None,
        allow_hub: bool = False,
    ):
        if not torch.cuda.is_available() or device.type != "cuda":
            raise RuntimeError(
                "live Krea train needs CUDA (~48GB, A6000/A100) so the "
                "12B DiT and parked 4B text encoder fit. CI uses --dummy."
            )
        self.device = device
        self.resolution = int(resolution)
        self.model_id = str(model_id)
        self.allow_hub = bool(allow_hub)
        self.encoder_lora = False
        self.pipe = _load_pipeline(self.model_id, allow_hub=self.allow_hub)
        self.pipe.vae.to("cpu")
        self.pipe.text_encoder.to(self.device)
        self.pipe.transformer.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        print(
            f"krea transformer+text_encoder on {self.device}; vae parked on cpu"
        )

        self.is_distilled = bool(getattr(self.pipe.config, "is_distilled", False))
        if krea_looks_turbo(self.model_id):
            self.is_distilled = True
        if sample_steps is None:
            sample_steps = KREA_TURBO_STEPS if self.is_distilled else KREA_RAW_STEPS
        if sample_guidance is None:
            sample_guidance = KREA_TURBO_CFG if self.is_distilled else KREA_RAW_CFG
        self.generate_steps = int(sample_steps)
        self.generate_guidance = float(sample_guidance)

        from peft import LoraConfig, get_peft_model

        config = LoraConfig(
            r=int(rank),
            lora_alpha=int(rank),
            target_modules=list(KREA_LORA_TARGETS),
        )
        self.pipe.transformer = get_peft_model(self.pipe.transformer, config)
        self.pipe.transformer.to(self.device)
        for param in self.pipe.transformer.parameters():
            if param.requires_grad:
                param.data = param.data.float().to(self.device)
        self.transformer = self.pipe.transformer
        self.transformer.eval()
        base = self.transformer.get_base_model()
        if hasattr(base, "enable_gradient_checkpointing"):
            base.enable_gradient_checkpointing()

        self.pipe.vae.to("cpu")
        torch.cuda.empty_cache()
        self._park_text_encoder()

        self.patch_size = int(getattr(self.pipe, "patch_size", 2))
        self.vae_scale_factor = int(getattr(self.pipe, "vae_scale_factor", 8))
        spatial = self.resolution // self.vae_scale_factor
        packed = spatial // self.patch_size
        self.spatial_hw = (spatial, spatial)
        self.grid_hw = (packed, packed)
        in_channels = int(base.config.in_channels)
        self.latent_channels = in_channels // (self.patch_size ** 2)
        self.latent_shape = (packed * packed, in_channels)
        self.dim = in_channels
        self.compute_dtype = torch.bfloat16
        self.max_sequence_length = 512
        self._text_cache: OrderedDict[str, tuple[torch.Tensor, torch.Tensor | None, list[str]]] = (
            OrderedDict()
        )
        self._timestep: torch.Tensor | None = None
        self._lora_scale = 1.0

    def begin_step(self) -> None:
        """Sample one timestep shared by every predict_v in this UNI step."""
        num = int(getattr(self.pipe.scheduler.config, "num_train_timesteps", 1000))
        self._timestep = torch.randint(0, max(num, 1), (1,), device=self.device)

    def sample_latents(self, device: torch.device | None = None) -> torch.Tensor:
        dest = device or self.device
        return torch.randn((1, *self.latent_shape), device=dest, dtype=torch.float32)

    def _park_text_encoder(self) -> None:
        """Drop the 4B Qwen3-VL off GPU before the 12B DiT backward."""
        if self.encoder_lora:
            return
        try:
            first = next(self.pipe.text_encoder.parameters())
        except StopIteration:
            return
        if first.device.type == "cuda":
            self.pipe.text_encoder.to("cpu")
            torch.cuda.empty_cache()

    def _ensure_text_encoder(self) -> None:
        self.pipe.text_encoder.to(self.device)

    def _prompt_tokens(self, prompt: str) -> list[str]:
        tokenizer = getattr(self.pipe, "tokenizer", None)
        if tokenizer is None:
            return krea_word_tokens(prompt) or [""]
        try:
            encoded = tokenizer(
                prompt,
                padding="max_length",
                max_length=self.max_sequence_length,
                truncation=True,
                add_special_tokens=True,
            )
            ids = encoded["input_ids"]
            if ids and isinstance(ids[0], (list, tuple)):
                ids = ids[0]
            tokens: list[str] = []
            for tid in ids:
                piece = tokenizer.decode([int(tid)], skip_special_tokens=False)
                tokens.append(str(piece).strip().lower())
            return tokens
        except Exception:
            return krea_word_tokens(prompt) or [""]

    def _encode_raw(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor | None, list[str]]:
        self._ensure_text_encoder()
        embeds, mask = self.pipe.get_text_hidden_states(
            prompt, self.max_sequence_length, self.device
        )
        tokens = self._prompt_tokens(prompt)
        return embeds, mask, tokens

    def encode_text(self, prompt: str) -> tuple[torch.Tensor, list[str]]:
        """Encode then park the text encoder. Returns (embeds, tokens)."""
        key = str(prompt)
        if key not in self._text_cache:
            try:
                embeds, mask, tokens = self._encode_raw(prompt)
            finally:
                self._park_text_encoder()
            self._text_cache[key] = (
                embeds.detach().to("cpu"),
                None if mask is None else mask.detach().to("cpu"),
                tokens,
            )
            self._text_cache.move_to_end(key)
            while len(self._text_cache) > _TEXT_CACHE_MAX:
                self._text_cache.popitem(last=False)
        else:
            self._text_cache.move_to_end(key)
        embeds, mask, tokens = self._text_cache[key]
        self._last_mask = None if mask is None else mask.to(self.device)
        return embeds.to(self.device), tokens

    def _position_ids(self, text_seq_len: int):
        gh, gw = self.grid_hw
        return self.pipe.prepare_position_ids(text_seq_len, gh, gw, self.device)

    def _forward(self, z: torch.Tensor, text: torch.Tensor, mask: torch.Tensor | None):
        dtype = self.compute_dtype
        timestep = self._timestep
        if timestep is None:
            timestep = torch.zeros(1, device=self.device)
        t = timestep.expand(z.shape[0]).to(device=self.device, dtype=dtype)
        num = float(getattr(self.pipe.scheduler.config, "num_train_timesteps", 1000) or 1000)
        t = t / num
        if text.dim() == 3 and z.dim() == 3:
            # (T, F) dummy-style should not happen on live; keep a batch dim.
            text = text.unsqueeze(0)
        pos = self._position_ids(int(text.shape[1]))
        enc_mask = None if mask is None else mask.to(self.device)
        out = self.transformer(
            hidden_states=z.to(device=self.device, dtype=dtype),
            encoder_hidden_states=text.to(device=self.device, dtype=dtype),
            timestep=t,
            position_ids=pos,
            encoder_attention_mask=enc_mask,
            return_dict=False,
        )[0]
        return out.float()

    @contextmanager
    def _scale_ctx(self, scale: float, *, frozen: bool):
        module = self.transformer
        prev = self._lora_scale
        self._lora_scale = 0.0 if frozen or float(scale) == 0.0 else float(scale)
        if frozen or float(scale) == 0.0:
            disable = getattr(module, "disable_adapter", None)
            if callable(disable):
                try:
                    with torch.no_grad(), disable():
                        yield
                finally:
                    self._lora_scale = prev
                return
            _set_adapter_scale(module, 0.0)
            try:
                with torch.no_grad():
                    yield
            finally:
                _set_adapter_scale(module, 1.0)
                self._lora_scale = prev
            return
        _set_adapter_scale(module, float(scale))
        try:
            yield
        finally:
            _set_adapter_scale(module, 1.0)
            self._lora_scale = prev

    def set_adapter_scale(self, scale: float) -> None:
        _set_adapter_scale(self.transformer, float(scale))
        self._lora_scale = float(scale)

    def disable_adapter(self):
        return self.transformer.disable_adapter()

    def predict_v(
        self,
        prompt: str,
        z: torch.Tensor,
        *,
        scale: float = 0.0,
        pin_unused: bool = False,
        neu_prompt: str | None = None,
        unused_words: Sequence[str] | None = None,
    ) -> torch.Tensor:
        frozen = float(scale) == 0.0
        with self._scale_ctx(scale, frozen=frozen):
            embeds, tokens = self.encode_text(prompt)
            mask = getattr(self, "_last_mask", None)
            if pin_unused and neu_prompt is not None:
                neu_embeds, neu_tokens = self.encode_text(neu_prompt)
                hold_mask = krea_unused_hold_mask(tokens, neu_tokens, unused_words)
                embeds = krea_hold_unused_embeds(
                    embeds, neu_embeds, tokens, neu_tokens, hold_mask
                )
            return self._forward(z, embeds, mask)

    def trainable_parameters(self) -> list[nn.Parameter]:
        self.transformer.train()
        params = [p for p in self.transformer.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("Krea DiT LoRA attached but no trainable parameters")
        return params

    def save_trained(self, path: str | Path) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        self.transformer.save_pretrained(str(path))

    def _fresh_scheduler(self, num_steps: int):
        import numpy as np

        sched = self.pipe.scheduler.from_config(self.pipe.scheduler.config)
        sigmas = np.linspace(1.0, 1.0 / max(int(num_steps), 1), int(num_steps))
        image_seq_len = self.latent_shape[0]
        if self.is_distilled:
            mu = 1.15
        else:
            mu = _calculate_shift(
                image_seq_len,
                sched.config.get("base_image_seq_len", 256),
                sched.config.get("max_image_seq_len", 6400),
                sched.config.get("base_shift", 0.5),
                sched.config.get("max_shift", 1.15),
            )
        sched.set_timesteps(sigmas=sigmas, device=self.device, mu=mu)
        if hasattr(sched, "set_begin_index"):
            sched.set_begin_index(0)
        return sched

    def _predict_current(self, prompt: str, z: torch.Tensor) -> torch.Tensor:
        """Forward at whatever adapter scale the caller already set."""
        embeds, _tokens = self.encode_text(prompt)
        mask = getattr(self, "_last_mask", None)
        return self._forward(z, embeds, mask)

    def _cfg(self, prompt: str, z: torch.Tensor, timestep: torch.Tensor, guidance: float):
        prev = self._timestep
        self._timestep = timestep
        try:
            v = self._predict_current(prompt, z)
            if guidance and float(guidance) > 0 and prompt != "":
                v_u = self._predict_current("", z)
                v = v + float(guidance) * (v - v_u)
            return v
        finally:
            self._timestep = prev

    def _unpack_latents(self, z: torch.Tensor) -> torch.Tensor:
        unpack = getattr(self.pipe, "_unpack_latents", None)
        if not callable(unpack):
            raise RuntimeError("Krea2Pipeline has no _unpack_latents")
        try:
            return unpack(z, self.resolution, self.resolution)
        except TypeError:
            return unpack(
                z, self.resolution, self.resolution, self.vae_scale_factor
            )

    def decode(self, z: torch.Tensor):
        self.pipe.vae.to(self.device)
        vae = self.pipe.vae
        latents = self._unpack_latents(z).to(vae.dtype)
        mean = torch.tensor(
            vae.config.latents_mean, device=latents.device, dtype=latents.dtype
        ).view(1, vae.config.z_dim, 1, 1, 1)
        std = 1.0 / torch.tensor(
            vae.config.latents_std, device=latents.device, dtype=latents.dtype
        ).view(1, vae.config.z_dim, 1, 1, 1)
        latents = latents / std + mean
        img = vae.decode(latents, return_dict=False)[0][:, :, 0]
        return img.float()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        seed: int = 0,
        num_steps: int | None = None,
        guidance: float | None = None,
        scale: float = 1.0,
        height: int | None = None,
        width: int | None = None,
    ):
        """Smile-first sample. Parks TE + VAE around the 12B denoise."""
        from PIL import Image

        del height, width
        num_steps = int(num_steps or self.generate_steps)
        guidance = self.generate_guidance if guidance is None else float(guidance)
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        sched = self._fresh_scheduler(num_steps)
        z = torch.randn(
            (1, *self.latent_shape),
            generator=generator,
            device=self.device,
            dtype=torch.float32,
        )
        try:
            with self._scale_ctx(float(scale), frozen=float(scale) == 0.0):
                for t in sched.timesteps:
                    v = self._cfg(prompt, z, t, guidance)
                    z = sched.step(v, t, z, return_dict=False)[0]
            img = self.decode(z)
        finally:
            self.pipe.vae.to("cpu")
            self._park_text_encoder()
            torch.cuda.empty_cache()
        img = ((img.clamp(-1, 1) + 1) / 2 * 255).round().byte()
        arr = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(arr)


def load_live_krea_backend(args: Any, device: torch.device) -> LiveKreaBackend:
    """Hub / local loader. Never called from ``--dummy``."""
    allow_hub = bool(getattr(args, "allow_hub", False))
    if allow_hub:
        os.environ["HF_HUB_OFFLINE"] = "0"
    else:
        os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        from diffusers import Krea2Pipeline  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "live Krea needs diffusers + peft. Use --dummy for CPU tests."
        ) from exc
    try:
        backend = LiveKreaBackend(
            device=device,
            model_id=str(args.model_id),
            resolution=int(args.resolution),
            rank=int(args.rank),
            sample_steps=getattr(args, "sample_steps", None),
            sample_guidance=getattr(args, "sample_guidance", None),
            allow_hub=allow_hub,
        )
    except Exception as exc:
        raise RuntimeError(
            f"live Krea weights not available for {args.model_id!r} "
            f"(local_files_only={not allow_hub}). "
            "Pass --allow_hub to download krea/Krea-2-Raw (gated; accept "
            "the card, ~48GB GPU, TE parked on CPU). CI uses --dummy."
        ) from exc
    return backend
