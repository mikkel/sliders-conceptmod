#!/usr/bin/env python3
"""Opt-in Anima image-slider trainer (UNI + unused-token hold).

Live card (documented by ``scripts/smoke_anima_slider.py``):

    model  circlestone-labs/Anima-Base-v1.0-Diffusers
    arch   2B Cosmos-Predict2 DiT, Qwen3+T5, Qwen-Image VAE
    lora   rank 16 on attn to_q / to_k / to_v / to_out.0
    res    768   sample 40 steps   CFG 4   lr 1e-4
    frozen base transformer with adapter disabled
    never  train text_conditioner
    sample in-process PEFT pipe(prompt=...) at 0 / 0.25 / 0.5 / 1.0

Default Music 3 trainer stays ``--lm_target v9`` / ``--pole_mode hidden``.
This file does not rewrite Music 3 yamls.

CI / this repo: pass ``--dummy``. Dummy never downloads Anima weights.
Live load is ``local_files_only`` / ``HF_HUB_OFFLINE=1`` unless
``--allow_hub`` is set.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
from tqdm.auto import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from conceptmod.textsliders.anima_fake import FakeAnimaBackend, write_plus_alignment
from conceptmod.textsliders.anima_slider import (
    DEFAULT_CFG,
    DEFAULT_CONTROL_PROMPT,
    DEFAULT_HOLD_WEIGHT,
    DEFAULT_LR,
    DEFAULT_MODEL_ID,
    DEFAULT_RANK,
    DEFAULT_RESOLUTION,
    DEFAULT_SAMPLE_SCALES,
    DEFAULT_SAMPLE_SEED,
    DEFAULT_SAMPLE_STEPS,
    FROZEN_MODULES,
    LORA_TARGETS,
    anima_cfg_delta,
    anima_uni_loss,
    anima_uni_teachers,
    anima_unused_hold_loss,
    assert_sample_gate,
    image_mean_std,
    infer_sample_prompts,
    live_train_card,
    live_train_command,
    load_anima_prompts,
    looks_like_rgb_noise,
    minus_canary_cosine,
    row_token_plan,
)

DEFAULT_PROMPTS = Path(__file__).resolve().parent / "data" / "prompts-anima.yaml"
DEFAULT_SAVE_DIR = Path("models/anima-slider")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", type=str, default="smile-anima")
    parser.add_argument("--prompts_file", type=str, default=str(DEFAULT_PROMPTS))
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--rank", type=int, default=DEFAULT_RANK)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--sample_steps", type=int, default=DEFAULT_SAMPLE_STEPS)
    parser.add_argument("--cfg", type=float, default=DEFAULT_CFG)
    parser.add_argument("--hold_weight", type=float, default=DEFAULT_HOLD_WEIGHT)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument(
        "--control_prompt",
        type=str,
        default=DEFAULT_CONTROL_PROMPT,
        help="verify-only fruit-bowl caption; never a teacher",
    )
    parser.add_argument(
        "--sample_every",
        type=int,
        default=0,
        help="in-process PEFT scale grid every N steps (0 = end of train only)",
    )
    parser.add_argument(
        "--sample_first_n",
        type=int,
        default=0,
        help="also emit the PEFT scale grid after each of the first N steps",
    )
    parser.add_argument(
        "--sample_seed",
        type=int,
        default=DEFAULT_SAMPLE_SEED,
        help="seed for in-process pipe(prompt=...) samples (default 42)",
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="tiny CPU DiT, never loads Anima / never hits the Hub",
    )
    parser.add_argument(
        "--allow_hub",
        action="store_true",
        help="permit a Hub download (off; CI must not set this)",
    )
    parser.add_argument(
        "--print_card",
        action="store_true",
        help="print the live train card and exit",
    )
    return parser.parse_args(argv)


def _device(arg: str, dummy: bool) -> torch.device:
    if dummy:
        if arg in ("cpu", "dummy"):
            return torch.device("cpu")
        if arg.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
    if arg.isdigit():
        return torch.device(f"cuda:{arg}" if torch.cuda.is_available() else "cpu")
    dev = torch.device(arg)
    if dev.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return dev


def _sample_zt(backend, seed: int, step: int):
    """CPU generator + CUDA latent is illegal. Draw on CPU, then ``.to(device)``."""
    g = torch.Generator(device="cpu").manual_seed(int(seed) * 1009 + step)
    z = torch.randn((1, *backend.latent_shape), generator=g, device="cpu")
    z = z.to(device=backend.device)
    t = torch.tensor([float(100 + (step * 37) % 800)], device=backend.device)
    return z, t


def freeze_anima_conditioner(pipe) -> None:
    """ModularPipeline is not an ``nn.Module``. Freeze via components.

    Do **not** call ``pipe.named_parameters()`` — that AttributeError'd
    on the live RunPod box. Walk ``pipe.text_conditioner`` and
    ``pipe.transformer`` instead.
    """
    cond = getattr(pipe, "text_conditioner", None)
    transformer = getattr(pipe, "transformer", None)
    modules = [cond, transformer]
    if transformer is not None:
        modules.append(getattr(transformer, "text_conditioner", None))
        get_base = getattr(transformer, "get_base_model", None)
        if callable(get_base):
            try:
                modules.append(get_base())
            except Exception:
                pass
    seen: set[int] = set()
    for module in modules:
        if module is None or not hasattr(module, "named_parameters"):
            continue
        ident = id(module)
        if ident in seen:
            continue
        seen.add(ident)
        for name, param in module.named_parameters():
            if module is cond or "text_conditioner" in name:
                param.requires_grad_(False)


def _assert_not_training_conditioner(backend) -> None:
    cond = getattr(getattr(backend, "transformer", None), "text_conditioner", None)
    if cond is None:
        cond = getattr(getattr(backend, "pipe", None), "text_conditioner", None)
    if cond is None:
        return
    for name, param in cond.named_parameters():
        if param.requires_grad:
            raise RuntimeError(
                f"text_conditioner.{name} is trainable; Anima LoRA is "
                "transformer-only (CircleStone)"
            )


def _peft_module(backend):
    """PEFT lives on ``pipe.transformer`` live; dummy exposes backend APIs."""
    if hasattr(backend, "set_lora_scale") or hasattr(backend, "disable_adapter"):
        return backend
    pipe = getattr(backend, "pipe", None)
    if pipe is not None and getattr(pipe, "transformer", None) is not None:
        return pipe.transformer
    return getattr(backend, "transformer", backend)


def _try_set_adapter_scale(module, scale: float) -> bool:
    """PEFT ``set_adapter_scale`` / adapter weights — never weight-merge."""
    if hasattr(module, "set_adapter_scale"):
        for args in ((scale,), ({"default": float(scale)},), ("default", float(scale))):
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
    if hasattr(module, "set_lora_scale"):
        module.set_lora_scale(float(scale))
        return True
    return False


def _enable_adapter(module) -> None:
    for name in ("enable_adapter_layers", "enable_adapters", "enable_adapter"):
        fn = getattr(module, name, None)
        if callable(fn):
            try:
                fn()
                return
            except Exception:
                continue


@contextmanager
def peft_adapter_scale(backend, scale: float):
    """In-process PEFT scale. Scale 0 uses ``disable_adapter`` when present."""
    module = _peft_module(backend)
    target = float(scale)
    if abs(target) < 1e-12:
        disable = getattr(module, "disable_adapter", None)
        if callable(disable):
            ctx = disable()
            if hasattr(ctx, "__enter__"):
                with ctx:
                    yield
                return
        if _try_set_adapter_scale(module, 0.0):
            try:
                yield
            finally:
                _try_set_adapter_scale(module, 1.0)
            return
        raise RuntimeError("cannot disable PEFT adapter for scale 0.0 sample")
    _enable_adapter(module)
    if not _try_set_adapter_scale(module, target):
        raise RuntimeError(
            f"cannot set PEFT adapter scale to {target}: need "
            "disable_adapter / set_adapter_scale / adapter weights "
            "(not a post-hoc W += scale*(α/r)*(B@A) merge)"
        )
    try:
        yield
    finally:
        _try_set_adapter_scale(module, 1.0)


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")
    return slug[:48] or "prompt"


def _as_uint8_hwc(image) -> np.ndarray:
    if hasattr(image, "convert"):
        image = image.convert("RGB")
        return np.asarray(image, dtype=np.uint8)
    arr = np.asarray(image)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        max_v = float(arr.max()) if arr.size else 0.0
        if max_v <= 1.0 + 1e-5:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return arr


def _extract_images(result) -> list[np.ndarray]:
    if result is None:
        return []
    images = getattr(result, "images", None)
    if images is None and isinstance(result, dict):
        images = result.get("images")
    if images is None:
        images = result
    if isinstance(images, (list, tuple)):
        return [_as_uint8_hwc(item) for item in images]
    return [_as_uint8_hwc(images)]


def _cpu_noise(shape, seed: int, device) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn(shape, generator=g, device="cpu").to(device=device)


def _call_modular_pipe(
    pipe,
    prompt: str,
    *,
    steps: int,
    height: int,
    width: int,
    cfg: float,
    seed: int,
    device,
    latent_shape=None,
):
    """Same ``pipe(prompt=...)`` path as live infer. PEFT stays on transformer."""
    dummy_h = int(height)
    dummy_w = int(width)
    # Dummy stay cheap; live uses the train resolution.
    if type(pipe).__name__ == "FakeAnimaModularPipe":
        dummy_h = min(dummy_h, 64)
        dummy_w = min(dummy_w, 64)
    guider = getattr(pipe, "guider", None)
    prev_cfg = None
    if guider is not None and hasattr(guider, "guidance_scale"):
        prev_cfg = guider.guidance_scale
        guider.guidance_scale = float(cfg)
    kwargs: dict[str, Any] = {
        "prompt": prompt,
        "height": dummy_h,
        "width": dummy_w,
        "num_inference_steps": int(steps),
        "output_type": "np",
    }
    if latent_shape is not None:
        kwargs["latents"] = _cpu_noise((1, *tuple(latent_shape)), seed, device)
    else:
        kwargs["generator"] = torch.Generator(device="cpu").manual_seed(int(seed))
    try:
        try:
            result = pipe(guidance_scale=float(cfg), **kwargs)
        except TypeError:
            result = pipe(**kwargs)
    finally:
        if prev_cfg is not None:
            guider.guidance_scale = prev_cfg
    return _extract_images(result)


def emit_inprocess_samples(
    backend,
    args: argparse.Namespace,
    save_dir: Path,
    *,
    step: int,
    rows,
    dummy: bool,
) -> list[dict[str, Any]]:
    """PEFT scale grid through ``pipe(prompt=...)``. Not a velocity dump."""
    pipe = getattr(backend, "pipe", None)
    if pipe is None or not callable(pipe):
        raise RuntimeError(
            "in-process Anima sample needs backend.pipe(prompt=...) "
            "with PEFT still attached to pipe.transformer"
        )
    prompts = infer_sample_prompts(rows, getattr(args, "control_prompt", DEFAULT_CONTROL_PROMPT))
    scales = list(DEFAULT_SAMPLE_SCALES)
    height = int(args.resolution)
    width = int(args.resolution)
    steps = 2 if dummy else int(args.sample_steps)
    seed = int(getattr(args, "sample_seed", DEFAULT_SAMPLE_SEED))
    cfg = float(args.cfg)
    out_dir = Path(save_dir) / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    latent_shape = getattr(backend, "latent_shape", None) if not dummy else None
    records: list[dict[str, Any]] = []
    for prompt in prompts:
        for scale in scales:
            with torch.no_grad(), peft_adapter_scale(backend, scale):
                images = _call_modular_pipe(
                    pipe,
                    prompt,
                    steps=steps,
                    height=height,
                    width=width,
                    cfg=cfg,
                    seed=seed,
                    device=backend.device,
                    latent_shape=latent_shape,
                )
            if not images:
                raise RuntimeError(f"pipe(prompt={prompt!r}) returned no images")
            arr = images[0]
            mean, std = image_mean_std(arr)
            noisy = looks_like_rgb_noise(arr)
            slug = _slug(prompt)
            scale_tag = f"{scale:g}".replace("-", "m")
            name = f"step{int(step):04d}_{slug}_scale{scale_tag}.png"
            path = out_dir / name
            from PIL import Image

            Image.fromarray(arr, mode="RGB").save(path)
            records.append(
                {
                    "step": int(step),
                    "prompt": prompt,
                    "scale": float(scale),
                    "mean": mean,
                    "std": std,
                    "looks_like_noise": bool(noisy),
                    "path": str(path.name),
                    "seed": seed,
                    "sample_steps": steps,
                    "cfg": cfg,
                    "height": int(arr.shape[0]),
                    "width": int(arr.shape[1]),
                    "method": "peft_pipe_prompt",
                }
            )
    meta_path = out_dir / f"step{int(step):04d}_meta.json"
    payload = {
        "step": int(step),
        "dummy": bool(dummy),
        "seed": seed,
        "scales": scales,
        "prompts": prompts,
        "samples": records,
    }
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    assert_sample_gate(records)
    return records


def _should_sample(step: int, args: argparse.Namespace, *, last: bool) -> bool:
    if last:
        return True
    every = int(getattr(args, "sample_every", 0) or 0)
    first_n = int(getattr(args, "sample_first_n", 0) or 0)
    idx = int(step) + 1
    if first_n and idx <= first_n:
        return True
    if every > 0 and idx % every == 0:
        return True
    return False


def train_dummy(args: argparse.Namespace) -> dict:
    device = _device(str(args.device), dummy=True)
    rows, meta = load_anima_prompts(args.prompts_file)
    if not rows:
        raise ValueError("no anima prompt rows")
    rank = int(args.rank)
    backend = FakeAnimaBackend(device=str(device), rank=rank, seed=int(args.seed))
    _assert_not_training_conditioner(backend)
    params = backend.trainable_parameters()
    names = backend.named_trainable()
    if any("text_conditioner" in n for n in names):
        raise RuntimeError("dummy LoRA attached to text_conditioner")
    opt = torch.optim.AdamW(params, lr=float(args.lr))
    history: list[float] = []
    canary: list[float] = []
    row = rows[0]
    plan = row_token_plan(row)
    infer = row.infer_prompt
    plus_before = write_plus_alignment(backend, infer, row.positive, seed=int(args.seed))

    save_dir = Path(args.save_dir or DEFAULT_SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    sample_records: list[dict] = []
    total = int(args.steps)
    pbar = tqdm(range(total), disable=total < 3)
    for step in pbar:
        z, t = _sample_zt(backend, int(args.seed), step)
        loss, canary_v = _uni_step(
            backend, row, plan, z, t, float(args.hold_weight)
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        history.append(float(loss.detach()))
        if canary_v is not None:
            canary.append(canary_v)
        pbar.set_postfix(loss=f"{history[-1]:.4f}")
        if _should_sample(step, args, last=False):
            sample_records = emit_inprocess_samples(
                backend, args, save_dir, step=step + 1, rows=rows, dummy=True
            )

    plus_after = write_plus_alignment(backend, infer, row.positive, seed=int(args.seed))
    sample_records = emit_inprocess_samples(
        backend, args, save_dir, step=total, rows=rows, dummy=True
    )
    sidecar = {
        "name": args.name,
        "dummy": True,
        "model_id": args.model_id,
        "rank": rank,
        "resolution": int(args.resolution),
        "sample_steps": int(args.sample_steps),
        "cfg": float(args.cfg),
        "lr": float(args.lr),
        "device": str(device),
        "lora_targets": list(LORA_TARGETS),
        "frozen_modules": list(FROZEN_MODULES),
        "recipe": "uni_plus_neu + unused_token_hold",
        "plus_label": meta.plus_label,
        "minus_canary": bool(row.has_minus_canary),
        "canary_cos_last": canary[-1] if canary else None,
        "plus_align_before": plus_before,
        "plus_align_after": plus_after,
        "loss_last": history[-1] if history else None,
        "steps": int(args.steps),
        "prompts_file": str(args.prompts_file),
        "control_prompt": str(args.control_prompt),
        "sample_grid": {
            "n": len(sample_records),
            "scales": list(DEFAULT_SAMPLE_SCALES),
            "seed": int(args.sample_seed),
            "method": "peft_pipe_prompt",
        },
        "music3_default_untouched": {"lm_target": "v9", "pole_mode": "hidden"},
    }
    side_path = save_dir / f"{args.name}_dummy_last.json"
    side_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    print(json.dumps(sidecar, indent=2))
    return sidecar


def _refuse_hub_download() -> None:
    if os.environ.get("HF_HUB_OFFLINE", "1") in ("0", "false", "False"):
        return
    # Default path: never call from_pretrained without local_files_only.


class LiveAnimaBackend:
    """Thin wrapper around a local Anima ModularPipeline.

    Adapted from conceptmod's Anima backend (encode / velocity / disable
    adapter) without bringing the DSL trainer. LoRA is PEFT on
    ``to_q/to_k/to_v/to_out.0`` only. ``text_conditioner`` stays frozen.
    """

    def __init__(self, pipe, device: torch.device, resolution: int):
        self.pipe = pipe
        self.device = device
        self.resolution = resolution
        self.compute_dtype = torch.bfloat16
        self.max_sequence_length = 512
        base = pipe.transformer.get_base_model()
        scale = pipe.vae_scale_factor
        h = resolution // scale
        self.latent_shape = (base.config.in_channels, 1, h, h)
        self._padding_mask = torch.zeros(
            1, 1, resolution, resolution, device=device, dtype=torch.bfloat16
        )
        self._text_cache: dict[str, torch.Tensor] = {}

    def encode_text(self, prompt: str) -> tuple[torch.Tensor, list[str]]:
        if prompt not in self._text_cache:
            self._text_cache[prompt] = self._encode_raw(prompt)
        embeds = self._text_cache[prompt]
        return embeds, []

    def _encode_raw(self, prompt: str) -> torch.Tensor:
        prompts = [prompt]
        cond_dtype = self.pipe.text_conditioner.dtype
        tok = self.pipe.tokenizer(
            prompts,
            padding="longest",
            max_length=self.max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        ids = tok.input_ids.to(self.device)
        mask = tok.attention_mask.to(self.device)
        if ids.shape[-1] == 0:
            ids = ids.new_zeros((ids.shape[0], 1))
            mask = mask.new_zeros((mask.shape[0], 1))
        qwen = self.pipe.text_encoder(
            input_ids=ids, attention_mask=mask, output_hidden_states=False
        ).last_hidden_state
        qwen = qwen * mask.to(qwen.dtype).unsqueeze(-1)
        t5 = self.pipe.t5_tokenizer(
            prompts,
            padding="longest",
            max_length=self.max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        cond = self.pipe.text_conditioner(
            source_hidden_states=qwen.to(device=self.device, dtype=cond_dtype),
            target_input_ids=t5.input_ids.to(self.device),
            target_attention_mask=t5.attention_mask.to(self.device),
            source_attention_mask=mask,
        )
        return cond.float().to(self.device)

    def _forward(self, z, timestep, embeds: torch.Tensor):
        dtype = self.compute_dtype
        t = timestep.expand(z.shape[0]).to(device=self.device, dtype=dtype)
        t = t / self.pipe.scheduler.config.num_train_timesteps
        hidden = z.to(device=self.device, dtype=dtype)
        if hidden.ndim == 4:
            hidden = hidden.unsqueeze(2)
        out = self.pipe.transformer(
            hidden_states=hidden,
            timestep=t,
            encoder_hidden_states=embeds.to(device=self.device, dtype=dtype),
            padding_mask=self._padding_mask.to(device=self.device, dtype=dtype),
            return_dict=False,
        )[0]
        if out.ndim == 4:
            out = out.unsqueeze(2)
        return out.float()

    def predict_v(self, prompt, z, timestep, frozen=False, scale=None):
        embeds, _ = self.encode_text(prompt)
        if frozen or scale == 0.0:
            with torch.no_grad(), self.pipe.transformer.disable_adapter():
                return self._forward(z, timestep, embeds)
        return self._forward(z, timestep, embeds)

    def text_features(self, prompt, frozen=False, scale=None):
        embeds, tokens = self.encode_text(prompt)
        return embeds, tokens

    def trainable_parameters(self):
        self.pipe.transformer.train()
        params = [p for p in self.pipe.transformer.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("Anima PEFT returned no trainable params")
        return params

    def named_trainable(self):
        return [
            n
            for n, p in self.pipe.transformer.named_parameters()
            if p.requires_grad
        ]


def load_live_backend(args: argparse.Namespace, device: torch.device):
    """Load Anima locally. Never downloads unless ``--allow_hub``."""
    if not args.allow_hub:
        os.environ["HF_HUB_OFFLINE"] = "1"
        _refuse_hub_download()
    else:
        os.environ["HF_HUB_OFFLINE"] = "0"
    try:
        from diffusers import ModularPipeline
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise RuntimeError(
            "live Anima needs diffusers + peft. Use --dummy for CPU tests."
        ) from exc

    kwargs = {"local_files_only": not bool(args.allow_hub)}
    try:
        pipe = ModularPipeline.from_pretrained(args.model_id, **kwargs)
        pipe.load_components(dtype=torch.bfloat16)
        pipe.to(str(device))
    except Exception as exc:
        raise RuntimeError(
            f"live Anima weights not available for {args.model_id!r} "
            f"(local_files_only={not args.allow_hub}). "
            "CI must use --dummy. See scripts/smoke_anima_slider.py."
        ) from exc

    config = LoraConfig(
        r=int(args.rank),
        lora_alpha=int(args.alpha or args.rank),
        target_modules=list(LORA_TARGETS),
    )
    pipe.transformer = get_peft_model(pipe.transformer, config)
    pipe.transformer.to(device)
    freeze_anima_conditioner(pipe)
    return LiveAnimaBackend(pipe, device, resolution=int(args.resolution))


def _uni_step(backend, row, plan, z, t, hold_weight: float):
    infer = row.infer_prompt
    with torch.no_grad():
        v_pos = backend.predict_v(row.positive, z, t, frozen=True)
        v_neu = backend.predict_v(row.neutral, z, t, frozen=True)
        v_null = backend.predict_v("", z, t, frozen=True)
        v_neg = (
            backend.predict_v(row.negative, z, t, frozen=True)
            if row.has_minus_canary
            else None
        )
    teachers = anima_uni_teachers(v_pos, v_neu, v_null, v_neg)
    s_plus = anima_cfg_delta(
        backend.predict_v(infer, z, t, frozen=False, scale=1.0),
        backend.predict_v("", z, t, frozen=False, scale=1.0),
    )
    s_zero = anima_cfg_delta(
        backend.predict_v(infer, z, t, frozen=False, scale=0.0),
        backend.predict_v("", z, t, frozen=False, scale=0.0),
    )
    loss = anima_uni_loss(s_plus, teachers["plus"], s_zero, teachers["zero"])
    feat_s, tokens_s = backend.text_features(row.positive, frozen=False, scale=1.0)
    feat_n, tokens_n = backend.text_features(row.neutral, frozen=True)
    # Dummy tokenizer matches ``row_token_plan`` word ids. Live Qwen/T5
    # ids do not — skip the index hold there (encoder is frozen anyway).
    if tokens_s and tokens_n and plan["pairs"]:
        loss = loss + hold_weight * anima_unused_hold_loss(
            feat_s, feat_n, pairs=plan["pairs"]
        )
    canary = None
    if v_neg is not None:
        canary = float(minus_canary_cosine(s_plus.detach(), v_neg, v_null))
    return loss, canary


def train_live(args: argparse.Namespace) -> dict:
    device = _device(str(args.device), dummy=False)
    if device.type != "cuda":
        raise RuntimeError(
            "live Anima needs CUDA. Pass --dummy for the CPU fake, or "
            "--device cuda:0 with local weights."
        )
    backend = load_live_backend(args, device)
    if any("text_conditioner" in n for n in backend.named_trainable()):
        raise RuntimeError("text_conditioner must stay frozen")
    rows, meta = load_anima_prompts(args.prompts_file)
    row = rows[0]
    plan = row_token_plan(row)
    opt = torch.optim.AdamW(backend.trainable_parameters(), lr=float(args.lr))
    history: list[float] = []
    canary: list[float] = []
    save_dir = Path(args.save_dir or DEFAULT_SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    sample_records: list[dict] = []
    total = int(args.steps)
    for step in tqdm(range(total)):
        z, t = _sample_zt(backend, int(args.seed), step)
        loss, canary_v = _uni_step(
            backend, row, plan, z, t, float(args.hold_weight)
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        history.append(float(loss.detach()))
        if canary_v is not None:
            canary.append(canary_v)
        if _should_sample(step, args, last=False):
            sample_records = emit_inprocess_samples(
                backend, args, save_dir, step=step + 1, rows=rows, dummy=False
            )
    backend.pipe.transformer.save_pretrained(str(save_dir / f"{args.name}_lora"))
    sample_records = emit_inprocess_samples(
        backend, args, save_dir, step=total, rows=rows, dummy=False
    )
    sidecar = {
        "name": args.name,
        "dummy": False,
        "model_id": args.model_id,
        "rank": int(args.rank),
        "resolution": int(args.resolution),
        "sample_steps": int(args.sample_steps),
        "cfg": float(args.cfg),
        "lr": float(args.lr),
        "device": str(device),
        "lora_targets": list(LORA_TARGETS),
        "frozen_modules": list(FROZEN_MODULES),
        "recipe": "uni_plus_neu + unused_token_hold",
        "plus_label": meta.plus_label,
        "minus_canary": bool(row.has_minus_canary),
        "canary_cos_last": canary[-1] if canary else None,
        "loss_last": history[-1] if history else None,
        "steps": int(args.steps),
        "prompts_file": str(args.prompts_file),
        "control_prompt": str(args.control_prompt),
        "sample_grid": {
            "n": len(sample_records),
            "scales": list(DEFAULT_SAMPLE_SCALES),
            "seed": int(args.sample_seed),
            "method": "peft_pipe_prompt",
        },
        "music3_default_untouched": {"lm_target": "v9", "pole_mode": "hidden"},
    }
    (save_dir / f"{args.name}_last.json").write_text(
        json.dumps(sidecar, indent=2), encoding="utf-8"
    )
    print(json.dumps(sidecar, indent=2))
    return sidecar


def train(args: argparse.Namespace) -> dict:
    if args.print_card:
        card = live_train_card(
            name=args.name,
            prompts_file=args.prompts_file,
            model_id=args.model_id,
            rank=int(args.rank),
            resolution=int(args.resolution),
            sample_steps=int(args.sample_steps),
            cfg=float(args.cfg),
            device=str(args.device),
            lr=float(args.lr),
            control_prompt=str(args.control_prompt),
        )
        print(json.dumps(card, indent=2))
        print()
        print(live_train_command(
            name=args.name,
            prompts_file=args.prompts_file,
            model_id=args.model_id,
            rank=int(args.rank),
            resolution=int(args.resolution),
            sample_steps=int(args.sample_steps),
            cfg=float(args.cfg),
            device=str(args.device),
            lr=float(args.lr),
        ))
        return card
    if args.dummy:
        return train_dummy(args)
    return train_live(args)


def main(argv: list[str] | None = None) -> dict:
    return train(parse_args(argv))


if __name__ == "__main__":
    main()
