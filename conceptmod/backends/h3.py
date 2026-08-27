"""HunyuanImage-3.0 (H3) adapter — AR/MoE, not a flow-matching DiT.

Resolved hub id: ``tencent/HunyuanImage-3.0``.

This is **not** MiniMax-H3 and **not** HunyuanImage-2.1. Official H3 is
an ~80B native multimodal MoE (``AutoModelForCausalLM`` /
``hunyuan_image_3_moe``, 13B active). Autoregressive. Transformers
library. There is **no** public flow-matching diffusers checkpoint.

conceptmod backends (sana / zimage / anima / krea / qwen / cpu / klein)
are velocity-space DiTs: ``v(z, t, text)`` + Euler. H3 is a different
stack. This adapter is honest about that:

* ``encode_text`` is native (tokenizer + hidden states).
* UNI image sliders train on encode / last hidden, not fake velocity.
* ``predict_v`` / Euler sample raise ``ArchitectureMismatch``.
* LoRA-only: a second 80B copy will not fit. Frozen = adapter off.

Cheaper live checkpoint (still AR/MoE, 8-step Instruct): 
``tencent/HunyuanImage-3.0-Instruct-Distil``. Default remains the base
``tencent/HunyuanImage-3.0``.

No Hub download unless a live train constructs this class without
``dummy=True`` / ``pipe=``.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn as nn

from conceptmod.backends.base import Backend, TextEmbeds, require_cuda

DEFAULT_MODEL = "tencent/HunyuanImage-3.0"
DISTIL_MODEL = "tencent/HunyuanImage-3.0-Instruct-Distil"
DEFAULT_RESOLUTION = 1024
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "to_q", "to_k", "to_v", "to_out.0")


class ArchitectureMismatch(RuntimeError):
    """H3 is AR/MoE. Do not treat it as a conceptmod velocity DiT."""


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


class DummyH3LM(nn.Module):
    """Tiny causal stand-in: last hidden = mean(embeds) + scale · LoRA.

    Unused token rows stay on the frozen embedding so UNI can hold them
    to ``encode(neu)`` without a lyric-span recipe.
    """

    def __init__(self, hidden: int = 8, vocab: int = 64):
        super().__init__()
        self.hidden = hidden
        self.embed = nn.Embedding(vocab, hidden)
        self.embed.weight.requires_grad_(False)
        self.lora_delta = nn.Parameter(torch.zeros(hidden))
        self._adapter_scale = 1.0

    def forward_hidden(self, token_ids: torch.Tensor) -> torch.Tensor:
        ids = token_ids.clamp(min=0, max=self.embed.num_embeddings - 1)
        h = self.embed(ids).float()
        pooled = h.mean(dim=1)
        last = pooled + float(self._adapter_scale) * self.lora_delta
        out = h.clone()
        out[:, -1] = last
        return out

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


class DummyH3Pipe:
    def __init__(self, hidden: int = 8):
        self.tokenizer = DummyTokenizer()
        self.language_model = DummyH3LM(hidden=hidden)
        self.text_encoder = None
        self.transformer = self.language_model
        self.vae = None


class H3Backend(Backend):
    """LoRA-only HunyuanImage-3.0 adapter. Frozen = adapter disabled."""

    def __init__(
        self,
        device: str = "cpu",
        model_id: str = DEFAULT_MODEL,
        resolution: int = DEFAULT_RESOLUTION,
        lora_rank: int | None = 8,
        dummy: bool = False,
        pipe=None,
        **_ignored,
    ):
        self.model_id = model_id
        self.resolution = int(resolution)
        self.encoder_lora = True  # UNI is encode-side on this AR model
        self._text_cache: dict[tuple[str, bool, float], TextEmbeds] = {}
        self.latent_shape = (4, 8, 8)  # unused; H3 is not a velocity DiT
        self.frozen = None

        if dummy or pipe is not None:
            self.device = str(device)
            self.pipe = pipe if pipe is not None else DummyH3Pipe()
            self.lora_rank = 8 if lora_rank is None else int(lora_rank)
            self.transformer = self.pipe.language_model
            return

        self.device = str(require_cuda(device))
        if lora_rank is None:
            lora_rank = 8
            print("h3 backend is LoRA-only; defaulting to rank 8")
        self.lora_rank = int(lora_rank)
        self.pipe = _load_h3_ar(model_id)
        self._attach_lora()

    def training_defaults(self) -> dict:
        return {"sample_steps": 0, "sample_guidance": 0.0, "stack": "autoregressive_moe"}

    def _ids(self, prompt: str) -> torch.Tensor:
        tok = self.pipe.tokenizer
        return torch.tensor([tok.encode(prompt, add_special_tokens=False) or [0]],
                            dtype=torch.long, device=self.device)

    def _encode_raw(self, prompt: str, scale: float | None = None) -> TextEmbeds:
        ids = self._ids(prompt)
        lm = self.transformer
        if scale is not None and hasattr(lm, "set_adapter_scale"):
            with _lora_scale(lm, scale):
                hidden = _hidden(lm, ids)
        else:
            hidden = _hidden(lm, ids)
        mask = torch.ones(ids.shape, dtype=torch.bool, device=ids.device)
        return TextEmbeds(hidden, mask)

    @torch.no_grad()
    def encode_text(self, prompt: str, frozen: bool = False) -> TextEmbeds:
        key = (prompt, bool(frozen), 0.0 if frozen else 1.0)
        if key not in self._text_cache:
            if frozen:
                with self._adapter_disabled():
                    self._text_cache[key] = self._encode_raw(prompt)
            else:
                self._text_cache[key] = self._encode_raw(prompt)
        return self._text_cache[key]

    def encode_text_grad(self, prompt: str, scale: float = 1.0) -> TextEmbeds:
        return self._encode_raw(prompt, scale=scale)

    def encode_scaled(self, prompt: str, scale: float) -> TextEmbeds:
        if abs(float(scale)) < 1e-8:
            return self.encode_text(prompt, frozen=True)
        return self.encode_text_grad(prompt, scale=float(scale))

    def _adapter_disabled(self):
        if hasattr(self.transformer, "disable_adapter"):
            return self.transformer.disable_adapter()
        return _null_cm()

    def predict_v(self, prompt, z, timestep, frozen):
        del prompt, z, timestep, frozen
        raise ArchitectureMismatch(
            "tencent/HunyuanImage-3.0 is an autoregressive MoE "
            "(AutoModelForCausalLM / hunyuan_image_3_moe), not a "
            "flow-matching DiT. conceptmod sana/zimage/anima/krea "
            "velocity Euler does not apply. Train UNI on encode_text "
            "/ last hidden, not predict_v."
        )

    def partial_denoise(self, prompt, stop_index, num_steps, guidance, generator):
        del prompt, stop_index, num_steps, guidance, generator
        raise ArchitectureMismatch(
            "H3 has no Euler / flow-matching sampler. "
            "Live generate is AR token sampling on the MoE LM."
        )

    @torch.no_grad()
    def generate(self, prompt, seed, num_steps=None, guidance=None, frozen=False):
        del num_steps, guidance
        # Dummy: render a gray tile from the last hidden so CI can call
        # generate without claiming a velocity sampler exists.
        text = self.encode_text(prompt, frozen=frozen)
        last = text.embeds[0, -1]
        val = last.mean().tanh()
        arr = (val.detach().cpu() * torch.ones(16, 16, 3) * 127 + 128).byte().numpy()

        class _ArrayImage:
            def __init__(self, pixels):
                self.size = (int(pixels.shape[1]), int(pixels.shape[0]))

        try:
            from PIL import Image

            return Image.fromarray(arr)
        except ImportError:
            return _ArrayImage(arr)

    def trainable_parameters(self, train_method: str = "lora"):
        self.transformer.train()
        params = [p for p in self.transformer.parameters() if p.requires_grad]
        if not params and hasattr(self.transformer, "lora_delta"):
            self.transformer.lora_delta.requires_grad_(True)
            params = [self.transformer.lora_delta]
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
            {k: v.detach().cpu() for k, v in self.transformer.state_dict().items()
             if getattr(v, "requires_grad", False) or "lora" in k},
            dest if dest.suffix else dest.with_suffix(".pt"),
        )

    def _attach_lora(self) -> None:
        from peft import LoraConfig, get_peft_model

        config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_rank,
            target_modules=list(LORA_TARGETS),
        )
        self.pipe.language_model = get_peft_model(self.pipe.language_model, config)
        self.pipe.language_model.to(self.device)
        for p in self.pipe.language_model.parameters():
            if p.requires_grad:
                p.data = p.data.float().to(self.device)
        self.transformer = self.pipe.language_model


def _load_h3_ar(model_id: str):
    """Live AR/MoE load only. Tests never call this."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    lm = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    lm.eval()

    class _LivePipe:
        tokenizer = tok
        language_model = lm

    return _LivePipe()


def _hidden(lm, ids: torch.Tensor) -> torch.Tensor:
    if hasattr(lm, "forward_hidden"):
        return lm.forward_hidden(ids)
    out = lm(ids, output_hidden_states=True)
    states = out.hidden_states[-1] if getattr(out, "hidden_states", None) else out.last_hidden_state
    return states.float()


@contextmanager
def _lora_scale(lm, scale: float):
    if hasattr(lm, "set_adapter_scale"):
        prev = getattr(lm, "_adapter_scale", 1.0)
        lm.set_adapter_scale(scale)
        try:
            yield
        finally:
            lm.set_adapter_scale(prev)
        return
    yield


@contextmanager
def _null_cm():
    yield
