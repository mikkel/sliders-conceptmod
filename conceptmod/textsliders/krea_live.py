"""Live Krea 2 velocity backend for ``train_lora_krea.py``.

Lazy-imported from ``_load_live_backend`` only. ``--dummy`` must never
import this module (no Hub, no 12B, no peft).

Patterns follow mikkel/conceptmod ``conceptmod/backends/krea.py``:

- ``Krea2Pipeline.from_pretrained(..., dtype=bfloat16)`` (falls back to
  ``torch_dtype=`` on older Diffusers)
- LoRA on DiT attn ``to_q/to_k/to_v/to_out.0`` via peft (rank 16)
  and optionally Qwen3-VL ``q_proj/k_proj/v_proj/o_proj``
  (``--lora_targets dit`` / ``te`` / ``dit+te``)
- frozen ref = adapter disabled (a second 12B copy will not fit)
- Park VAE on CPU. Frozen TE: encode on GPU, then park before DiT
  backward so a 48GB card can train. Trained TE: stay on GPU so
  encode+backward stay coherent
- Raw: ~28 steps, CFG 4.5; Turbo (local ``.safetensors`` or hub id
  containing ``turbo``): 8 steps, CFG 0
- Official: train LoRAs on Raw, run on Turbo
- Continuous adapter scale writes LoRA ``scaling`` (PEFT
  ``set_adapter_scale`` often no-ops, which made 0.25/0.5/1.0 grids
  byte-identical)

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
    KREA_DEFAULT_LORA_TARGETS,
    KREA_DEFAULT_RANK,
    KREA_DEFAULT_RESOLUTION,
    KREA_LORA_TARGETS,
    KREA_RAW_CFG,
    KREA_RAW_MODEL,
    KREA_RAW_STEPS,
    KREA_TE_LORA_TARGETS,
    KREA_TURBO_CFG,
    KREA_TURBO_STEPS,
    apply_continuous_lora_scale,
    krea_hold_unused_embeds,
    krea_looks_turbo,
    krea_unused_hold_mask,
    krea_word_tokens,
    resolve_krea_lora_targets,
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


def _enable_adapter_layers(module) -> None:
    for name in ("enable_adapter_layers", "enable_adapters", "enable_adapter"):
        fn = getattr(module, name, None)
        if callable(fn):
            try:
                fn()
                return
            except Exception:
                continue


def _try_peft_set_adapter_scale(module, scale: float) -> bool:
    """Last-resort PEFT APIs. Often no-ops — prefer ``apply_continuous_lora_scale``."""
    if hasattr(module, "set_adapter_scale"):
        for args in ((float(scale),), ({"default": float(scale)},), ("default", float(scale))):
            try:
                module.set_adapter_scale(*args)
                return True
            except TypeError:
                continue
            except Exception:
                continue
    if hasattr(module, "set_adapters"):
        for args, kwargs in (
            (("default",), {"weights": float(scale)}),
            ((["default"],), {"weights": [float(scale)]}),
        ):
            try:
                module.set_adapters(*args, **kwargs)
                return True
            except Exception:
                continue
    return False


def set_module_lora_scale(module, scale: float) -> int:
    """Write a real LoRA delta multiplier. Fail closed if nothing moved."""
    n = apply_continuous_lora_scale(module, float(scale))
    if n == 0:
        _try_peft_set_adapter_scale(module, float(scale))
        n = apply_continuous_lora_scale(module, float(scale))
    return n


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
        lora_targets: str = KREA_DEFAULT_LORA_TARGETS,
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
        self.lora_spec = resolve_krea_lora_targets(lora_targets)
        self.encoder_lora = self.lora_spec.train_te
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

        self.transformer = self.pipe.transformer
        if self.lora_spec.train_dit:
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
        base = (
            self.transformer.get_base_model()
            if hasattr(self.transformer, "get_base_model")
            else self.transformer
        )
        if hasattr(base, "enable_gradient_checkpointing"):
            base.enable_gradient_checkpointing()

        self._text_cache: OrderedDict[
            tuple[str, str], tuple[torch.Tensor, torch.Tensor | None, list[str]]
        ] = OrderedDict()
        self._timestep: torch.Tensor | None = None
        self._lora_scale = 1.0

        if self.lora_spec.train_te:
            self._attach_encoder_lora(int(rank))
        else:
            self._park_text_encoder()

        self.pipe.vae.to("cpu")
        torch.cuda.empty_cache()

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

    def _attach_encoder_lora(self, rank: int) -> None:
        """conceptmod ``attach_encoder_lora``: Qwen3-VL attn q/k/v/o."""
        from peft import LoraConfig, get_peft_model

        if any(
            hasattr(child, "lora_A") for child in self.pipe.text_encoder.modules()
        ):
            raise RuntimeError("Krea text-encoder LoRA already attached")
        config = LoraConfig(
            r=int(rank),
            lora_alpha=int(rank),
            target_modules=list(KREA_TE_LORA_TARGETS),
        )
        self.pipe.text_encoder = get_peft_model(self.pipe.text_encoder, config)
        self.pipe.text_encoder.to(self.device)
        for _name, param in self.pipe.text_encoder.named_parameters():
            if param.requires_grad:
                param.data = param.data.float().to(self.device)
        self.encoder_lora = True
        self._text_cache.clear()
        print(
            f"krea text-encoder LoRA on {self.device} "
            f"targets={list(KREA_TE_LORA_TARGETS)} rank={int(rank)}; "
            "TE stays resident (no park-after-encode)"
        )

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

    def _encode_with_te_disabled(
        self, prompt: str
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[str]]:
        te = self.pipe.text_encoder
        disable = getattr(te, "disable_adapter", None)
        if callable(disable):
            with disable():
                return self._encode_raw(prompt)
        prev = float(self._lora_scale)
        set_module_lora_scale(te, 0.0)
        try:
            return self._encode_raw(prompt)
        finally:
            set_module_lora_scale(te, prev if prev > 0.0 else 1.0)

    def _remember_text(
        self,
        key: tuple[str, str],
        embeds: torch.Tensor,
        mask: torch.Tensor | None,
        tokens: list[str],
    ) -> None:
        self._text_cache[key] = (
            embeds.detach().to("cpu"),
            None if mask is None else mask.detach().to("cpu"),
            tokens,
        )
        self._text_cache.move_to_end(key)
        while len(self._text_cache) > _TEXT_CACHE_MAX:
            self._text_cache.popitem(last=False)

    def encode_text(
        self, prompt: str, *, frozen: bool = False
    ) -> tuple[torch.Tensor, list[str]]:
        """Encode. Frozen TE parks after. Trained TE stays for backward."""
        use_te_lora = bool(self.encoder_lora) and not frozen
        cacheable = not (use_te_lora and torch.is_grad_enabled())
        if self.encoder_lora and frozen:
            cache_tag = "frozen"
        elif use_te_lora:
            cache_tag = f"{float(self._lora_scale):.6f}"
        else:
            cache_tag = "base"
        key = (str(prompt), cache_tag)
        if cacheable and key in self._text_cache:
            self._text_cache.move_to_end(key)
            embeds, mask, tokens = self._text_cache[key]
            self._last_mask = None if mask is None else mask.to(self.device)
            return embeds.to(self.device), tokens
        try:
            if self.encoder_lora and frozen:
                embeds, mask, tokens = self._encode_with_te_disabled(prompt)
            else:
                embeds, mask, tokens = self._encode_raw(prompt)
        finally:
            if not self.encoder_lora:
                self._park_text_encoder()
        if cacheable:
            self._remember_text(key, embeds, mask, tokens)
            embeds, mask, tokens = self._text_cache[key]
            self._last_mask = None if mask is None else mask.to(self.device)
            return embeds.to(self.device), tokens
        self._last_mask = None if mask is None else mask.to(self.device)
        return embeds, tokens

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

    def _lora_modules(self) -> list:
        modules: list = []
        if self.lora_spec.train_dit:
            modules.append(self.transformer)
        if self.lora_spec.train_te:
            modules.append(self.pipe.text_encoder)
        return modules

    @contextmanager
    def _scale_ctx(self, scale: float, *, frozen: bool):
        modules = self._lora_modules()
        prev = self._lora_scale
        target = 0.0 if frozen or abs(float(scale)) < 1e-12 else float(scale)
        self._lora_scale = target
        if target == 0.0:
            exits: list = []
            try:
                for module in modules:
                    disable = getattr(module, "disable_adapter", None)
                    if callable(disable):
                        ctx = disable()
                        if hasattr(ctx, "__enter__"):
                            ctx.__enter__()
                            exits.append(ctx)
                            continue
                    n = set_module_lora_scale(module, 0.0)
                    if n == 0:
                        raise RuntimeError(
                            "cannot disable Krea LoRA for scale 0 "
                            f"on {type(module).__name__}"
                        )
                with torch.no_grad():
                    yield
            finally:
                for ctx in reversed(exits):
                    ctx.__exit__(None, None, None)
                for module in modules:
                    set_module_lora_scale(module, 1.0)
                self._lora_scale = prev
            return
        for module in modules:
            _enable_adapter_layers(module)
            n = set_module_lora_scale(module, target)
            if n == 0:
                raise RuntimeError(
                    f"cannot apply continuous LoRA scale {target} on "
                    f"{type(module).__name__}; need LoraLayer.scaling "
                    "(PEFT set_adapter_scale is not trusted)"
                )
        try:
            yield
        finally:
            for module in modules:
                set_module_lora_scale(module, 1.0)
            self._lora_scale = prev

    def set_adapter_scale(self, scale: float) -> None:
        target = float(scale)
        self._lora_scale = target
        for module in self._lora_modules():
            if abs(target) >= 1e-12:
                _enable_adapter_layers(module)
            set_module_lora_scale(module, target)

    def disable_adapter(self):
        if self.lora_spec.train_dit and hasattr(self.transformer, "disable_adapter"):
            return self.transformer.disable_adapter()
        if self.lora_spec.train_te and hasattr(self.pipe.text_encoder, "disable_adapter"):
            return self.pipe.text_encoder.disable_adapter()
        raise RuntimeError("no PEFT disable_adapter on adapted Krea modules")

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
                neu_embeds, neu_tokens = self.encode_text(neu_prompt, frozen=True)
                hold_mask = krea_unused_hold_mask(tokens, neu_tokens, unused_words)
                embeds = krea_hold_unused_embeds(
                    embeds, neu_embeds, tokens, neu_tokens, hold_mask
                )
            return self._forward(z, embeds, mask)

    def trainable_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        if self.lora_spec.train_dit:
            self.transformer.train()
            params.extend(p for p in self.transformer.parameters() if p.requires_grad)
        if self.lora_spec.train_te:
            self.pipe.text_encoder.train()
            params.extend(
                p for p in self.pipe.text_encoder.parameters() if p.requires_grad
            )
        if not params:
            raise RuntimeError(
                f"Krea LoRA ({self.lora_spec.label}) attached but no trainable parameters"
            )
        return params

    def save_trained(self, path: str | Path) -> None:
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        spec = self.lora_spec
        if spec.train_dit and spec.train_te:
            dit_dir = root / "dit_lora"
            te_dir = root / "te_lora"
            dit_dir.mkdir(parents=True, exist_ok=True)
            te_dir.mkdir(parents=True, exist_ok=True)
            self.transformer.save_pretrained(str(dit_dir))
            self.pipe.text_encoder.save_pretrained(str(te_dir))
        elif spec.train_dit:
            self.transformer.save_pretrained(str(root))
        elif spec.train_te:
            te_dir = root / "te_lora"
            te_dir.mkdir(parents=True, exist_ok=True)
            self.pipe.text_encoder.save_pretrained(str(te_dir))
        else:
            raise RuntimeError("nothing to save: lora_targets trained neither dit nor te")

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
        """Smile-first sample. Parks frozen TE + VAE around the 12B denoise."""
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
        was_training = [module.training for module in self._lora_modules()]
        for module in self._lora_modules():
            module.eval()
        try:
            with self._scale_ctx(float(scale), frozen=float(scale) == 0.0):
                for t in sched.timesteps:
                    v = self._cfg(prompt, z, t, guidance)
                    z = sched.step(v, t, z, return_dict=False)[0]
            img = self.decode(z)
        finally:
            for module, flag in zip(self._lora_modules(), was_training):
                module.train(flag)
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
            lora_targets=str(getattr(args, "lora_targets", KREA_DEFAULT_LORA_TARGETS)),
        )
    except Exception as exc:
        raise RuntimeError(
            f"live Krea weights not available for {args.model_id!r} "
            f"(local_files_only={not allow_hub}). "
            "Pass --allow_hub to download krea/Krea-2-Raw (gated; accept "
            "the card, ~48GB GPU; frozen TE parks on CPU, --lora_targets "
            "te/dit+te keeps TE resident). CI uses --dummy."
        ) from exc
    return backend
