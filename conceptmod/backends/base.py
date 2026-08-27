"""Backend protocol copied from mikkel/conceptmod (encode / velocity / sample).

conceptmod's DiT backends implement all of these. H3 is AR/MoE, so this
fork keeps the same method names: ``encode_text`` is native, while
``predict_v`` / ``partial_denoise`` raise ``ArchitectureMismatch``.
Live load requires CUDA. Tests inject CPU mocks and never load Hub weights.
"""

from __future__ import annotations

import abc

import torch


def require_cuda(device: str) -> torch.device:
    """Live backends run on one CUDA device. CPU is not a fallback."""
    dev = torch.device(device)
    if dev.type != "cuda":
        raise ValueError(f"backends require a CUDA device, got {device!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return dev


class TextEmbeds:
    """Prompt embeddings plus attention mask (mask may be None)."""

    def __init__(self, embeds: torch.Tensor, mask: torch.Tensor | None):
        self.embeds = embeds
        self.mask = mask


class Backend(abc.ABC):
    device: str
    latent_shape: tuple  # (C, H, W) for one sample

    @abc.abstractmethod
    def encode_text(self, prompt: str) -> TextEmbeds:
        """Encode a prompt (cached). '' is the unconditional prompt."""

    @abc.abstractmethod
    def predict_v(
        self,
        prompt: str,
        z: torch.Tensor,
        timestep: torch.Tensor,
        frozen: bool,
    ) -> torch.Tensor:
        """Velocity prediction for latents ``z`` at a scheduler timestep."""

    @abc.abstractmethod
    def partial_denoise(
        self,
        prompt: str,
        stop_index: int,
        num_steps: int,
        guidance: float,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Euler steps from noise down to ``timesteps[stop_index]``.

        Uses the trained model (no grad). Returns ``(z_t, timestep_t)``.
        """

    @abc.abstractmethod
    def generate(
        self,
        prompt: str,
        seed: int,
        num_steps: int,
        guidance: float,
        frozen: bool = False,
    ):
        """Full generation → PIL.Image. ``frozen=True`` is the reference."""

    @abc.abstractmethod
    def trainable_parameters(self, train_method: str) -> list[torch.nn.Parameter]:
        """Parameter group to finetune (LoRA-only for H3)."""

    @abc.abstractmethod
    def save_trained(self, path: str) -> None:
        """Save the trained weights (LoRA adapter)."""

    def training_defaults(self) -> dict:
        return {}
