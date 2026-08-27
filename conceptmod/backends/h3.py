"""MiniMax-H3 image-slider backend (flow-matching DiT, LoRA-only).

Resolved hub id: ``MiniMaxAI/MiniMax-H3``.

Mikkel's "H3" is MiniMax-H3 (Slack: weight release, "testing h3",
Civitai buzz for 15s video, MiniMax-H3 × Z-Image fuse). It is **not**
in mikkel/conceptmod (that repo has sana, zimage, anima, krea, qwen,
cpu, klein).

Candidates checked before picking this id:

* ``tencent/HunyuanImage-3.0`` — native multimodal MoE
  (``AutoModelForCausalLM`` / ``hunyuan_image_3_moe``). Autoregressive,
  transformers library, **no** public flow-matching diffusers
  checkpoint.
* ``hunyuanvideo-community/HunyuanImage-2.1-Diffusers`` — real
  flow-matching DiT with a public ``HunyuanImagePipeline``, but it is
  HunyuanImage **2.1**, and Mikkel never calls it H3.
* ``Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers`` — older HunyuanDiT,
  not H3.

Pick: ``MiniMaxAI/MiniMax-H3``. Official public diffusers checkpoint
(``MiniMaxH3ModularPipeline`` / ``MiniMaxH3Pipeline``,
``MiniMaxH3Transformer3DModel`` / ``MiniMaxH3DiTModel``,
``MiniMaxH3Scheduler`` with flow-match ``shift=12.0``). 33B Omni-DiT
so a second frozen copy will not fit — LoRA-only, frozen = adapter
off.

Image sliders use the FL2VA family (t2va / first-frame) and a T=1
visual latent. Audio is not trained here.

No Hub download unless a live train explicitly constructs this class
without ``pipe=`` / ``dummy=True``.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from conceptmod.backends.base import Backend, TextEmbeds, require_cuda

DEFAULT_MODEL = "MiniMaxAI/MiniMax-H3"
DEFAULT_SUBFOLDER = "FL2VA"
DEFAULT_RESOLUTION = 768
# H3-VisualVAE: 24 latent channels, 16× spatial (README). T=1 image path.
DEFAULT_LATENT_CHANNELS = 24
DEFAULT_VAE_SPATIAL = 16
LORA_TARGETS = ("to_q", "to_k", "to_v", "to_out.0", "q_proj", "k_proj", "v_proj", "o_proj")


class DummyTokenizer:
    """Whitespace vocab for CPU tests. No Hub."""

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {"<pad>": 0}

    def _ids(self, text: str) -> list[int]:
        ids: list[int] = []
        for word in str(text).lower().split():
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)
            ids.append(self.vocab[word])
        if not ids:
            ids = [0]
        return ids

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return self._ids(text)

    def __call__(self, text: str, return_tensors: str | None = None, **_kw):
        ids = self._ids(text)
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return {"input_ids": ids}


class DummyH3Transformer(nn.Module):
    """Tiny flow-matching stand-in: ``v = z + embed + scale * LoRA(z)``."""

    def __init__(self, channels: int = 4, hidden: int = 8):
        super().__init__()
        self.in_channels = channels
        self.hidden = hidden
        self.text_table = nn.Embedding(64, hidden)
        self.text_table.weight.requires_grad_(False)
        self.lora = nn.Linear(channels, channels, bias=False)
        nn.init.zeros_(self.lora.weight)
        self.lora_delta = nn.Parameter(torch.zeros(channels))
        self._adapter_scale = 1.0

    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        ids = token_ids.clamp(min=0, max=self.text_table.num_embeddings - 1)
        return self.text_table(ids)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        timestep: torch.Tensor | None = None,
        return_dict: bool = False,
        **_kw,
    ):
        del timestep, return_dict
        z = hidden_states
        if encoder_hidden_states is None:
            text = torch.zeros(z.shape[0], self.hidden, device=z.device, dtype=z.dtype)
        else:
            text = encoder_hidden_states.float().mean(dim=1)
        add = text[:, : z.shape[1]]
        if add.shape[1] < z.shape[1]:
            add = F.pad(add, (0, z.shape[1] - add.shape[1]))
        v = z + add[:, :, None, None]
        if self._adapter_scale:
            b, c, h, w = z.shape
            mapped = self.lora(z.permute(0, 2, 3, 1).reshape(-1, c))
            mapped = mapped.view(b, h, w, c).permute(0, 3, 1, 2)
            shift = self.lora_delta.to(device=z.device, dtype=z.dtype)
            v = v + float(self._adapter_scale) * (mapped + shift[None, :, None, None])
        return (v,)

    @contextmanager
    def disable_adapter(self):
        prev = self._adapter_scale
        self._adapter_scale = 0.0
        try:
            yield
        finally:
            self._adapter_scale = prev

    def set_adapter_scale(self, scale: float) -> None:
        self._adapter_scale = float(scale)


class DummyScheduler:
    def __init__(self, num_steps: int = 8, device: str = "cpu"):
        self.timesteps = torch.linspace(1000.0, 1.0, num_steps, device=device)

    def from_config(self, _config=None):
        return DummyScheduler(num_steps=len(self.timesteps), device=str(self.timesteps.device))

    def set_timesteps(self, num_steps: int, device=None):
        dev = device if device is not None else self.timesteps.device
        self.timesteps = torch.linspace(1000.0, 1.0, num_steps, device=dev)
        return self

    def step(self, v, t, z, return_dict=False):
        del t, return_dict
        return (z + 0.1 * v,)


class DummyH3Pipe:
    """In-memory pipe so ``--dummy`` / tests never touch the Hub."""

    def __init__(self, channels: int = 4, hidden: int = 8):
        self.tokenizer = DummyTokenizer()
        self.transformer = DummyH3Transformer(channels=channels, hidden=hidden)
        self.scheduler = DummyScheduler()
        self.text_encoder = None
        self.vae = None
        self.vae_scale_factor = 16

    def encode_prompt(self, prompt: str, device: str = "cpu", **_kw):
        ids = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long, device=device)
        embeds = self.transformer.encode(ids.to(device=device))
        mask = torch.ones(ids.shape, dtype=torch.bool, device=device)
        return embeds, mask


class H3Backend(Backend):
    """LoRA-only MiniMax-H3 backend. Frozen reference = adapter disabled."""

    def __init__(
        self,
        device: str = "cpu",
        model_id: str = DEFAULT_MODEL,
        resolution: int = DEFAULT_RESOLUTION,
        lora_rank: int | None = 8,
        generate_steps: int = 8,
        generate_guidance: float = 1.0,
        dummy: bool = False,
        pipe=None,
        subfolder: str = DEFAULT_SUBFOLDER,
    ):
        self.model_id = model_id
        self.subfolder = subfolder
        self.resolution = int(resolution)
        self.generate_steps = int(generate_steps)
        self.generate_guidance = float(generate_guidance)
        self.encoder_lora = False
        self._text_cache: dict[tuple[str, bool], TextEmbeds] = {}

        if dummy or pipe is not None:
            self.device = str(device)
            self.pipe = pipe if pipe is not None else DummyH3Pipe()
            self.lora_rank = 8 if lora_rank is None else int(lora_rank)
            self.transformer = self.pipe.transformer
            self.frozen = None
            channels = getattr(self.transformer, "in_channels", 4)
            spatial = max(4, self.resolution // (DEFAULT_VAE_SPATIAL * 4))
            self.latent_shape = (channels, spatial, spatial)
            return

        self.device = str(require_cuda(device))
        if lora_rank is None:
            lora_rank = 8
            print("h3 backend is LoRA-only; defaulting to rank 8")
        self.lora_rank = int(lora_rank)
        self.pipe = _load_h3_pipeline(model_id, subfolder=subfolder)
        self.pipe.set_progress_bar_config(disable=True)
        self._move_live_modules()
        self._attach_lora()
        self.frozen = None
        channels = _live_in_channels(self.transformer)
        spatial = max(4, self.resolution // DEFAULT_VAE_SPATIAL)
        self.latent_shape = (channels, spatial, spatial)
        if hasattr(self.transformer, "enable_gradient_checkpointing"):
            self.transformer.enable_gradient_checkpointing()
        elif hasattr(self.transformer, "get_base_model"):
            base = self.transformer.get_base_model()
            if hasattr(base, "enable_gradient_checkpointing"):
                base.enable_gradient_checkpointing()

    def training_defaults(self) -> dict:
        return {"sample_steps": 8, "sample_guidance": 1.0}

    # ---------------- text ----------------

    def _encode_raw(self, prompt: str) -> TextEmbeds:
        if hasattr(self.pipe, "encode_prompt"):
            out = self.pipe.encode_prompt(prompt, device=self.device)
            if isinstance(out, tuple):
                embeds = out[0]
                mask = out[1] if len(out) > 1 else None
            else:
                embeds, mask = out, None
            return TextEmbeds(embeds, mask)
        tokenizer = getattr(self.pipe, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("H3 pipe has no encode_prompt or tokenizer")
        batch = tokenizer(prompt, return_tensors="pt")
        ids = batch["input_ids"].to(self.device)
        encoder = getattr(self.pipe, "text_encoder", None)
        if encoder is None:
            embeds = F.one_hot(ids, num_classes=int(ids.max().item()) + 1).float()
        else:
            embeds = encoder(ids).last_hidden_state
        return TextEmbeds(embeds, torch.ones_like(ids, dtype=torch.bool))

    @torch.no_grad()
    def encode_text(self, prompt: str, frozen: bool = False) -> TextEmbeds:
        if not self.encoder_lora:
            frozen = False
        key = (prompt, frozen)
        if key not in self._text_cache:
            if self.encoder_lora and frozen:
                with self.pipe.text_encoder.disable_adapter():
                    self._text_cache[key] = self._encode_raw(prompt)
            else:
                self._text_cache[key] = self._encode_raw(prompt)
        return self._text_cache[key]

    def encode_text_grad(self, prompt: str) -> TextEmbeds:
        return self._encode_raw(prompt)

    def attach_encoder_lora(self, rank: int = 8):
        from peft import LoraConfig, get_peft_model

        assert not self.encoder_lora, "encoder LoRA already attached"
        encoder = getattr(self.pipe, "text_encoder", None)
        if encoder is None:
            raise RuntimeError("H3 dummy/live pipe has no text_encoder to wrap")
        config = LoraConfig(
            r=rank,
            lora_alpha=rank,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        self.pipe.text_encoder = get_peft_model(encoder, config)
        for p in self.pipe.text_encoder.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        self.encoder_lora = True
        self._text_cache.clear()
        return [p for p in self.pipe.text_encoder.parameters() if p.requires_grad]

    # ---------------- velocity ----------------

    def _forward(self, model, z, timestep, text: TextEmbeds) -> torch.Tensor:
        t = timestep.expand(z.shape[0]).to(device=z.device, dtype=torch.float32)
        embeds = text.embeds.to(device=z.device)
        mask = None if text.mask is None else text.mask.to(device=z.device)
        kwargs = {
            "hidden_states": z,
            "encoder_hidden_states": embeds,
            "timestep": t,
            "return_dict": False,
        }
        if mask is not None:
            kwargs["encoder_attention_mask"] = mask
        try:
            out = model(**kwargs)
        except TypeError:
            out = model(z, t, embeds, return_dict=False)
        pred = out[0] if isinstance(out, (tuple, list)) else getattr(out, "sample", out)
        if pred.dim() == 5:
            pred = pred[:, :, 0]
        return pred.float()

    def predict_v(self, prompt, z, timestep, frozen):
        text = self.encode_text(prompt, frozen=frozen)
        if frozen:
            with torch.no_grad(), self._adapter_disabled():
                return self._forward(self.transformer, z, timestep, text)
        return self._forward(self.transformer, z, timestep, text)

    def predict_v_scaled(self, prompt, z, timestep, scale: float):
        """Student velocity at a LoRA multiplier. ``0`` is adapter-off."""
        if abs(float(scale)) < 1e-8:
            return self.predict_v(prompt, z, timestep, frozen=True)
        with self._lora_scale(float(scale)):
            return self.predict_v(prompt, z, timestep, frozen=False)

    def _adapter_disabled(self):
        if hasattr(self.transformer, "disable_adapter"):
            return self.transformer.disable_adapter()
        return _null_cm()

    @contextmanager
    def _lora_scale(self, scale: float):
        tr = self.transformer
        if hasattr(tr, "set_adapter_scale"):
            prev = getattr(tr, "_adapter_scale", 1.0)
            tr.set_adapter_scale(scale)
            try:
                yield
            finally:
                tr.set_adapter_scale(prev)
            return
        if hasattr(tr, "set_adapter"):
            try:
                tr.set_adapter(tr.active_adapter if hasattr(tr, "active_adapter") else "default")
            except Exception:
                pass
        yield

    # ---------------- sampling ----------------

    def _fresh_scheduler(self, num_steps: int):
        sched = self.pipe.scheduler
        if hasattr(sched, "from_config") and hasattr(sched, "config"):
            sched = sched.from_config(sched.config)
        if hasattr(sched, "set_timesteps"):
            try:
                sched.set_timesteps(num_steps, device=self.device)
            except TypeError:
                sched.set_timesteps(num_steps)
        return sched

    def _noise(self, generator):
        return torch.randn(
            (1, *self.latent_shape),
            generator=generator,
            device=self.device,
            dtype=torch.float32,
        )

    def _cfg(self, prompt, z, t, guidance, frozen):
        v = self.predict_v(prompt, z, t, frozen=frozen)
        if guidance and guidance != 1.0 and prompt != "":
            v_u = self.predict_v("", z, t, frozen=frozen)
            v = v_u + float(guidance) * (v - v_u)
        return v

    @torch.no_grad()
    def partial_denoise(self, prompt, stop_index, num_steps, guidance, generator):
        sched = self._fresh_scheduler(num_steps)
        z = self._noise(generator)
        for i, t in enumerate(sched.timesteps):
            if i >= stop_index:
                return z, t
            v = self._cfg(prompt, z, t, guidance, frozen=False)
            z = sched.step(v, t, z, return_dict=False)[0]
        return z, sched.timesteps[-1]

    def decode(self, z, grad=False):
        vae = getattr(self.pipe, "vae", None)
        if vae is None:
            # Dummy: map C×H×W → 3×H×W in [-1, 1]
            rgb = z[:, :3]
            if rgb.shape[1] < 3:
                rgb = F.pad(rgb, (0, 0, 0, 0, 0, 3 - rgb.shape[1]))
            return rgb.float().tanh()
        with torch.set_grad_enabled(grad):
            img = vae.decode(z.to(vae.dtype), return_dict=False)[0]
            if img.dim() == 5:
                img = img[:, :, 0]
        return img.float()

    @torch.no_grad()
    def generate(self, prompt, seed, num_steps=None, guidance=None, frozen=False):
        num_steps = num_steps or self.generate_steps
        guidance = self.generate_guidance if guidance is None else guidance
        g = torch.Generator(device=self.device).manual_seed(int(seed))
        sched = self._fresh_scheduler(num_steps)
        z = self._noise(g)
        for t in sched.timesteps:
            v = self._cfg(prompt, z, t, guidance, frozen=frozen)
            z = sched.step(v, t, z, return_dict=False)[0]
        img = self.decode(z, grad=False)
        img = ((img.clamp(-1, 1) + 1) / 2 * 255).round().byte()
        arr = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        try:
            from PIL import Image

            return Image.fromarray(arr)
        except ImportError:
            class _ArrayImage:
                def __init__(self, pixels):
                    self.size = (int(pixels.shape[1]), int(pixels.shape[0]))

            return _ArrayImage(arr)

    # ---------------- training ----------------

    def trainable_parameters(self, train_method: str = "lora"):
        self.transformer.train()
        params = [p for p in self.transformer.parameters() if p.requires_grad]
        if not params:
            extras = []
            if hasattr(self.transformer, "lora"):
                self.transformer.lora.weight.requires_grad_(True)
                extras.append(self.transformer.lora.weight)
            if hasattr(self.transformer, "lora_delta"):
                self.transformer.lora_delta.requires_grad_(True)
                extras.append(self.transformer.lora_delta)
            params = extras
        if train_method not in (None, "lora", "full"):
            print(f"h3 backend is LoRA-only; ignoring train_method={train_method!r}")
        assert params, "H3 LoRA selected no parameters"
        return params

    def save_trained(self, path: str) -> None:
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(self.transformer, "save_pretrained"):
            self.transformer.save_pretrained(str(dest))
            return
        torch.save(
            {k: v.detach().cpu() for k, v in self.transformer.state_dict().items() if v.requires_grad},
            dest if dest.suffix else dest.with_suffix(".pt"),
        )

    def _move_live_modules(self) -> None:
        for name in ("transformer", "text_encoder", "tokenizer"):
            mod = getattr(self.pipe, name, None)
            if isinstance(mod, nn.Module):
                mod.to(self.device)
        vae = getattr(self.pipe, "vae", None)
        if isinstance(vae, nn.Module):
            vae.to("cpu")
        self.transformer = self.pipe.transformer

    def _attach_lora(self) -> None:
        from peft import LoraConfig, get_peft_model

        config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_rank,
            target_modules=list(LORA_TARGETS),
        )
        self.pipe.transformer = get_peft_model(self.pipe.transformer, config)
        self.pipe.transformer.to(self.device)
        for p in self.pipe.transformer.parameters():
            if p.requires_grad:
                p.data = p.data.float().to(self.device)
        self.transformer = self.pipe.transformer


def _load_h3_pipeline(model_id: str, subfolder: str = DEFAULT_SUBFOLDER):
    """Live load only. Tests never call this."""
    try:
        from diffusers import MiniMaxH3Pipeline

        return MiniMaxH3Pipeline.from_pretrained(
            model_id, subfolder=subfolder, torch_dtype=torch.bfloat16,
        )
    except Exception:
        pass
    from diffusers import ModularPipeline

    return ModularPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)


def _live_in_channels(transformer) -> int:
    model = transformer.get_base_model() if hasattr(transformer, "get_base_model") else transformer
    cfg = getattr(model, "config", None)
    if cfg is not None and hasattr(cfg, "in_channels"):
        return int(cfg.in_channels)
    return DEFAULT_LATENT_CHANNELS


@contextmanager
def _null_cm():
    yield
