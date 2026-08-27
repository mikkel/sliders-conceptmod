"""CPU fake Anima: tiny flow DiT with the live LoRA contract.

No Hub, no GPU, no 2B weights. Same geometry the live trainer feeds:

    v(z, t, c) − v(z, t, '')

DiT attention linears are named ``to_q`` / ``to_k`` / ``to_v`` / ``to_out.0``.
``text_conditioner`` uses live AnimaTextConditioner names
``q_proj`` / ``k_proj`` / ``v_proj`` / ``o_proj`` so PEFT can attach there
when ``--lora_targets conditioner`` (smile default). Frozen ref = adapter
disabled (scale 0).
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from conceptmod.textsliders.anima_slider import (
    CONDITIONER_LORA_TARGETS,
    DEFAULT_CFG,
    DEFAULT_RANK,
    DIT_LORA_TARGETS,
    LORA_TARGETS,
    resolve_anima_lora_targets,
    word_tokens,
)

LATENT_CHANNELS = 4
LATENT_HW = 8
TEXT_DIM = 8
Z_DIM = LATENT_CHANNELS * LATENT_HW * LATENT_HW
MAX_TOKENS = 24


class AnimaFakeAttention(nn.Module):
    """Minimal attn whose Linear names match the live Anima LoRA targets."""

    def __init__(self, dim: int = TEXT_DIM):
        super().__init__()
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim, bias=False)])
        for layer in (self.to_q, self.to_k, self.to_v, self.to_out[0]):
            nn.init.eye_(layer.weight)

    def project_text(self, embeds: torch.Tensor) -> torch.Tensor:
        """Per-token text features after ``to_v`` / ``to_out.0`` (hold site)."""
        return self.to_out[0](self.to_v(embeds))

    def forward(self, hidden: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        q = self.to_q(hidden)
        k = self.to_k(context)
        v = self.to_v(context)
        scale = hidden.shape[-1] ** -0.5
        attn = torch.softmax(q @ k.transpose(-1, -2) * scale, dim=-1)
        return self.to_out[0](attn @ v)


class AnimaFakeConditioner(nn.Module):
    """Tiny stand-in for ``AnimaTextConditioner`` attn names.

    Live conditioner uses ``q_proj`` / ``k_proj`` / ``v_proj`` / ``o_proj``
    (not DiT ``to_q`` / ``to_k``). Identity-init so scale 0 matches the
    class-table embeds; LoRA on these linears can move caption features.
    """

    def __init__(self, dim: int = TEXT_DIM):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        for layer in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            nn.init.eye_(layer.weight)

    def forward(self, embeds: torch.Tensor) -> torch.Tensor:
        return self.o_proj(self.v_proj(embeds))


class AnimaFakeDiT(nn.Module):
    """``v = proj_out(attn(proj_in(z, t), text))``.

    Unused words live on axis 0 of the embedding table; concept words on
    axis 1. CFG geometry is therefore in ``v(c) − v('')``.
    """

    def __init__(self, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.proj_in = nn.Linear(Z_DIM + 1, TEXT_DIM)
        self.attn = AnimaFakeAttention(TEXT_DIM)
        self.proj_out = nn.Linear(TEXT_DIM, Z_DIM)
        # Fixed class path: unused vs concept stay linearly separable so
        # UNI has a teacher the attn LoRA can actually chase.
        self.text_out = nn.Linear(TEXT_DIM, Z_DIM, bias=False)
        # Live AnimaTextConditioner stand-in. Frozen unless --lora_targets
        # includes conditioner.
        self.text_conditioner = AnimaFakeConditioner(TEXT_DIM)
        nn.init.normal_(self.proj_in.weight, std=0.02, generator=g)
        nn.init.zeros_(self.proj_in.bias)
        nn.init.normal_(self.proj_out.weight, std=0.02, generator=g)
        nn.init.zeros_(self.proj_out.bias)
        with torch.no_grad():
            self.text_out.weight.zero_()
            self.text_out.weight[0, 0] = 1.0
            self.text_out.weight[1, 1] = 2.5
        table = torch.zeros(64, TEXT_DIM)
        # unused / subject cluster
        table[1, 0] = 1.0
        # concept cluster (stronger so pos ⟂ unused is visible)
        table[2, 1] = 1.0
        self.register_buffer("class_table", table)
        self._token_ids: dict[str, int] = {}

    def token_id(self, word: str) -> int:
        key = word.lower()
        if key in self._token_ids:
            return self._token_ids[key]
        # Concept-ish words sit on class 2; everything else on class 1.
        conceptish = {
            "smiling",
            "smile",
            "grinning",
            "happy",
            "joyful",
            "teeth",
            "old",
            "young",
            "red",
            "blue",
        }
        idx = 2 if key in conceptish else 1
        if key in ("",):
            idx = 0
        self._token_ids[key] = idx
        return idx

    def encode_tokens(self, prompt: str) -> tuple[torch.Tensor, list[str]]:
        tokens = word_tokens(prompt)
        if not tokens:
            tokens = [""]
        tokens = tokens[:MAX_TOKENS]
        ids = torch.tensor([self.token_id(tok) for tok in tokens], dtype=torch.long)
        embeds = self.class_table[ids].unsqueeze(0)
        embeds = self.text_conditioner(embeds)
        return embeds, tokens

    def forward(
        self, z: torch.Tensor, timestep: torch.Tensor, embeds: torch.Tensor
    ) -> torch.Tensor:
        b = z.shape[0]
        z_flat = z.reshape(b, -1)
        t = timestep.to(dtype=z.dtype, device=z.device).reshape(b, 1) / 1000.0
        hidden = self.proj_in(torch.cat([z_flat, t], dim=-1)).unsqueeze(1)
        cond = embeds.to(dtype=z.dtype, device=z.device)
        if cond.ndim == 2:
            cond = cond.unsqueeze(1)
        h = self.attn(hidden, cond).mean(dim=1)
        v = self.proj_out(h) + self.text_out(cond.mean(dim=1))
        return v.reshape_as(z)


class LoRALinear(nn.Module):
    """Drop-in LoRA around a Linear. B is zero-init (identity at step 0)."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.down = nn.Linear(base.in_features, rank, bias=False)
        self.up = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)
        self.scale = float(alpha) / float(rank)
        self.multiplier = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.up(self.down(x))
        return self.base(x) + self.multiplier * self.scale * delta


def _module_by_path(root: nn.Module, path: str) -> tuple[nn.Module, str]:
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]  # type: ignore[index]
        else:
            parent = getattr(parent, part)
    return parent, parts[-1]


def attach_anima_lora(
    root: nn.Module,
    rank: int,
    alpha: float,
    targets: tuple[str, ...] | list[str] | None = None,
) -> list[LoRALinear]:
    """Wrap named Linears. Default DiT ``to_q`` / ``to_k`` / ``to_v`` / ``to_out.0``."""
    target_names = tuple(targets) if targets is not None else tuple(DIT_LORA_TARGETS)
    wrapped: list[LoRALinear] = []
    named = dict(root.named_modules())
    for target in target_names:
        matches = [name for name in named if name == target or name.endswith("." + target)]
        if not matches:
            raise RuntimeError(f"Anima LoRA target {target!r} not found on {type(root).__name__}")
        for name in matches:
            parent, attr = _module_by_path(root, name)
            base = getattr(parent, attr) if not attr.isdigit() else parent[int(attr)]
            if not isinstance(base, nn.Linear):
                raise RuntimeError(f"{name} is {type(base)}, expected Linear")
            lora = LoRALinear(base, rank=rank, alpha=alpha)
            if attr.isdigit():
                parent[int(attr)] = lora  # type: ignore[index]
            else:
                setattr(parent, attr, lora)
            wrapped.append(lora)
    return wrapped


class FakeAnimaBackend:
    """In-repo Anima stand-in. ``--dummy`` on the trainer."""

    def __init__(
        self,
        device: str = "cpu",
        rank: int = DEFAULT_RANK,
        seed: int = 0,
        resolution: int = 64,
        lora_targets: str = "dit",
    ):
        del resolution
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")
        self.rank = rank
        self.lora_spec = resolve_anima_lora_targets(lora_targets)
        self.latent_shape = (LATENT_CHANNELS, LATENT_HW, LATENT_HW)
        self.num_train_timesteps = 1000
        torch.manual_seed(seed)
        self.transformer = AnimaFakeDiT(seed=seed).to(self.device)
        self.loras: list[LoRALinear] = []
        if self.lora_spec.train_dit:
            self.loras.extend(
                attach_anima_lora(
                    self.transformer, rank=rank, alpha=float(rank), targets=DIT_LORA_TARGETS
                )
            )
        if self.lora_spec.train_conditioner:
            self.loras.extend(
                attach_anima_lora(
                    self.transformer.text_conditioner,
                    rank=rank,
                    alpha=float(rank),
                    targets=CONDITIONER_LORA_TARGETS,
                )
            )
        else:
            for param in self.transformer.text_conditioner.parameters():
                param.requires_grad_(False)
        self.transformer.to(self.device)
        self._text_cache: dict[str, tuple[torch.Tensor, list[str]]] = {}
        self.set_lora_scale(1.0)
        # Same call path as live infer: pipe(prompt=...) with LoRA still
        # attached. Conditioner is the same object as transformer.text_conditioner.
        self.pipe = FakeAnimaModularPipe(self)

    def set_lora_scale(self, scale: float) -> None:
        for lora in self.loras:
            lora.multiplier = float(scale)

    def set_adapter_scale(self, scale: float) -> None:
        """PEFT-shaped alias used by the in-process sample grid."""
        self.set_lora_scale(scale)

    @contextmanager
    def disable_adapter(self):
        prev = [lora.multiplier for lora in self.loras]
        self.set_lora_scale(0.0)
        try:
            yield
        finally:
            for lora, value in zip(self.loras, prev):
                lora.multiplier = value

    def encode_text(self, prompt: str) -> tuple[torch.Tensor, list[str]]:
        # Conditioner LoRA changes embeds; a cached tensor cannot be reused
        # across steps (stale graph) or adapter scales.
        if self.lora_spec.train_conditioner:
            embeds, tokens = self.transformer.encode_tokens(prompt)
            return embeds.to(self.device), tokens
        key = prompt
        if key not in self._text_cache:
            embeds, tokens = self.transformer.encode_tokens(prompt)
            self._text_cache[key] = (embeds.to(self.device), tokens)
        embeds, tokens = self._text_cache[key]
        return embeds.to(self.device), tokens

    def _forward(self, z: torch.Tensor, timestep: torch.Tensor, embeds: torch.Tensor):
        if not torch.is_tensor(timestep):
            timestep = torch.tensor(timestep, device=self.device)
        t = timestep.reshape(-1).to(device=self.device, dtype=torch.float32)
        if t.numel() == 1:
            t = t.expand(z.shape[0])
        return self.transformer(z.to(self.device), t, embeds.to(self.device)).float()

    def predict_v(
        self,
        prompt: str,
        z: torch.Tensor,
        timestep: torch.Tensor,
        frozen: bool = False,
        scale: float | None = None,
    ) -> torch.Tensor:
        # Encode under the same adapter scale so conditioner LoRA (when
        # attached) is disabled for frozen / scale-0 teachers.
        if frozen or scale == 0.0:
            with torch.no_grad(), self.disable_adapter():
                embeds, _tokens = self.encode_text(prompt)
                return self._forward(z, timestep, embeds)
        if scale is not None:
            prev = [lora.multiplier for lora in self.loras]
            self.set_lora_scale(scale)
            try:
                embeds, _tokens = self.encode_text(prompt)
                return self._forward(z, timestep, embeds)
            finally:
                for lora, value in zip(self.loras, prev):
                    lora.multiplier = value
        embeds, _tokens = self.encode_text(prompt)
        return self._forward(z, timestep, embeds)

    def text_features(
        self, prompt: str, frozen: bool = False, scale: float | None = None
    ) -> tuple[torch.Tensor, list[str]]:
        attn = self.transformer.attn

        def _run(embeds):
            return attn.project_text(embeds.to(self.device))

        if frozen or scale == 0.0:
            with torch.no_grad(), self.disable_adapter():
                embeds, tokens = self.encode_text(prompt)
                return _run(embeds), tokens
        if scale is not None:
            prev = [lora.multiplier for lora in self.loras]
            self.set_lora_scale(scale)
            try:
                embeds, tokens = self.encode_text(prompt)
                return _run(embeds), tokens
            finally:
                for lora, value in zip(self.loras, prev):
                    lora.multiplier = value
        embeds, tokens = self.encode_text(prompt)
        return _run(embeds), tokens

    def trainable_parameters(self) -> list[nn.Parameter]:
        params = []
        for lora in self.loras:
            params.extend([lora.down.weight, lora.up.weight])
        return params

    def lora_B_norm(self) -> float:
        total = 0.0
        for lora in self.loras:
            total += lora.up.weight.detach().float().pow(2).sum().item()
        return total**0.5

    def named_trainable(self) -> list[str]:
        names = []
        for name, param in self.transformer.named_parameters():
            if param.requires_grad and ("down.weight" in name or "up.weight" in name):
                names.append(name)
        return names


class FakeAnimaModularPipe:
    """Dummy ``ModularPipeline``: ``pipe(prompt=...)`` with PEFT on transformer.

    Images are structured (low-frequency ramps), never RGB TV-static, so
    ``--dummy`` still passes the in-process sample gate without Hub weights.
    """

    def __init__(self, backend: FakeAnimaBackend):
        self.backend = backend
        self.transformer = backend.transformer
        self.text_conditioner = backend.transformer.text_conditioner
        # Same shape as live ClassifierFreeGuidance: config.guidance_scale
        # is the real field. A top-level pipe(guidance_scale=) is ignored.
        self.guider = SimpleNamespace(
            config=SimpleNamespace(guidance_scale=DEFAULT_CFG),
            guidance_scale=DEFAULT_CFG,
        )
        # Same FlowMatch Euler contract as live Anima (thin loop, not pipe).
        self.scheduler = SimpleNamespace(
            config=SimpleNamespace(num_train_timesteps=1000)
        )
        self.last_guidance_scale = float(DEFAULT_CFG)
        self.last_prompt = ""
        self.prompts_seen: list[str] = []

    def __call__(
        self,
        prompt: str = "",
        height: int = 64,
        width: int = 64,
        num_inference_steps: int = 8,
        generator=None,
        output_type: str = "pil",
        **kwargs,
    ):
        del num_inference_steps
        self.last_prompt = str(prompt)
        self.prompts_seen.append(str(prompt))
        if "guidance_scale" in kwargs:
            warnings.warn(
                "Unexpected input 'guidance_scale' … ignored",
                stacklevel=2,
            )
        config = getattr(self.guider, "config", None)
        cfg = getattr(config, "guidance_scale", None)
        if cfg is None:
            cfg = getattr(self.guider, "guidance_scale", DEFAULT_CFG)
        self.last_guidance_scale = float(cfg)
        h = max(8, int(height))
        w = max(8, int(width))
        scale = 1.0
        if self.backend.loras:
            scale = float(self.backend.loras[0].multiplier)
        seed = 0
        if generator is not None and hasattr(generator, "initial_seed"):
            try:
                seed = int(generator.initial_seed())
            except Exception:
                seed = 0
        arr = _structured_dummy_image(prompt, scale=scale, height=h, width=w, seed=seed)
        if output_type == "np":
            images = [arr.astype(np.float32) / 255.0]
        elif output_type == "pt":
            images = [torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0]
        else:
            from PIL import Image

            images = [Image.fromarray(arr, mode="RGB")]
        return SimpleNamespace(images=images)


def _structured_dummy_image(
    prompt: str, *, scale: float, height: int, width: int, seed: int
) -> np.ndarray:
    """Low-frequency RGB ramp. Mean/std are image-like, spatially correlated."""
    rng = np.random.default_rng(abs(int(seed)) % (2**31) + (hash(prompt) % 997))
    ys = np.linspace(48.0, 176.0, height, dtype=np.float64)
    xs = np.linspace(36.0, 168.0, width, dtype=np.float64)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    tint = 12.0 * float(scale)
    blob = 18.0 * np.sin(yy / 18.0) * np.cos(xx / 22.0)
    noise = 4.0 * rng.standard_normal((height, width))
    r = yy + tint + blob
    g = xx + 0.4 * tint
    b = 0.45 * yy + 0.45 * xx + 16.0 + 0.6 * tint + noise
    img = np.stack([r, g, b], axis=-1)
    return np.clip(img, 0.0, 255.0).astype(np.uint8)


def write_plus_alignment(
    backend: FakeAnimaBackend,
    neu: str,
    pos: str,
    seed: int = 0,
) -> float:
    """Cosine of student(+1, neu) CFG vs frozen(pos) CFG."""
    from conceptmod.textsliders.anima_slider import anima_cfg_delta

    g = torch.Generator(device="cpu").manual_seed(seed + 3)
    z = torch.randn((1, *backend.latent_shape), generator=g, device=backend.device)
    t = torch.tensor([500.0], device=backend.device)
    with torch.no_grad():
        v_pos = backend.predict_v(pos, z, t, frozen=True)
        v_null = backend.predict_v("", z, t, frozen=True)
        v_s = backend.predict_v(neu, z, t, frozen=False, scale=1.0)
        v_s_null = backend.predict_v("", z, t, frozen=False, scale=1.0)
    d_t = anima_cfg_delta(v_pos, v_null).flatten().unsqueeze(0)
    d_s = anima_cfg_delta(v_s, v_s_null).flatten().unsqueeze(0)
    return F.cosine_similarity(d_s, d_t, dim=1, eps=1e-6).item()
