"""Image-slider backends in this fork.

conceptmod today ships sana / zimage / anima / krea / qwen / cpu / klein.
H3 is not one of those. This package registers **only** H3 so Anima / Krea
/ ZiT stay on their own PRs.
"""

from conceptmod.backends.base import Backend, TextEmbeds
from conceptmod.backends.h3 import DEFAULT_MODEL, H3Backend

BACKENDS = ("h3",)


def load_backend(name: str, device: str, **kwargs) -> Backend:
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    if name == "h3":
        return H3Backend(device=device, **kwargs)
    raise ValueError(
        f"unknown backend {name!r}; this fork registers {BACKENDS} only "
        "(Anima / Krea / ZiT are separate PRs)"
    )


__all__ = [
    "BACKENDS",
    "Backend",
    "DEFAULT_MODEL",
    "H3Backend",
    "TextEmbeds",
    "load_backend",
]
