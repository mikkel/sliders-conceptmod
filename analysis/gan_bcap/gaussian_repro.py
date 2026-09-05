"""Compact 2-D Gaussian-mixture smoke for the RpGAN + b_cap core.

The cited ParticleGAN 100-Gaussians run (seed 1234, 7k updates, b_cap=1)
covers 100/100 modes at HQ 0.987. This file is the same game on a CPU
budget: 8 modes by default (tests), 100 modes via ``--modes 100``.

Does not claim Music 3 audio quality. CPU only.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn

from analysis.slider2d.adv import (
    EMA,
    Fourier2MLP,
    ParticlePrior,
    cap_penalty,
    delayed_cosine,
    input_grad,
    rp_d_loss,
    rp_g_loss,
    vicreg_loss,
)


def mixture_means(n_modes: int, radius: float = 2.0) -> torch.Tensor:
    angles = torch.linspace(0.0, 2.0 * math.pi, int(n_modes) + 1)[:-1]
    return torch.stack([torch.cos(angles), torch.sin(angles)], dim=1) * float(radius)


def sample_mixture(means: torch.Tensor, n: int, sigma: float) -> torch.Tensor:
    idx = torch.randint(0, means.shape[0], (int(n),))
    return means[idx] + float(sigma) * torch.randn(int(n), means.shape[1])


class TinyGen(nn.Module):
    """G: particle → 2-D point. Identity-plus-MLP so particles can sit on modes."""

    def __init__(self, dim: int = 2, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.net(z)


def hq_and_cover(
    samples: torch.Tensor,
    means: torch.Tensor,
    sigma: float,
    radius: float = 3.0,
) -> dict[str, float]:
    """HQ = fraction of samples within ``radius * sigma`` of some mode."""
    d = torch.cdist(samples, means)
    nearest, which = d.min(dim=1)
    hq = nearest <= float(radius) * float(sigma)
    covered = torch.zeros(means.shape[0], dtype=torch.bool)
    covered[which[hq]] = True
    return {
        "hq": float(hq.float().mean()),
        "modes": int(covered.sum()),
        "n_modes": int(means.shape[0]),
        "cover": float(covered.float().mean()),
    }


def train_gaussians(
    *,
    n_modes: int = 8,
    steps: int = 1500,
    seed: int = 1234,
    b_cap: float = 1.0,
    sigma: float = 0.05,
    batch: int = 64,
    n_particles: int | None = None,
    lr: float = 2.0e-3,
) -> dict:
    torch.manual_seed(int(seed))
    means = mixture_means(n_modes)
    n_particles = int(n_particles or max(n_modes, 2 * n_modes))
    prior = ParticlePrior(n_particles, 2, init_std=0.4)
    gen = TinyGen()
    critic = Fourier2MLP(2, n_rand=24, hidden=64, seed=seed)
    g_params = list(gen.parameters()) + list(prior.parameters())
    opt_g = torch.optim.Adam(g_params, lr=lr, betas=(0.0, 0.99))
    opt_d = torch.optim.Adam(critic.parameters(), lr=lr, betas=(0.0, 0.99))
    ema_g = EMA(list(gen.parameters()), decay=0.995)
    ema_p = EMA(list(prior.parameters()), decay=0.995)
    delay = max(50, int(0.12 * steps))

    for step in range(int(steps)):
        scale = delayed_cosine(step, total=steps, delay=delay, min_ratio=0.05)
        for opt in (opt_g, opt_d):
            for group in opt.param_groups:
                group["lr"] = float(lr) * scale
        real = sample_mixture(means, batch, sigma)
        z = prior.sample(batch, jitter=0.02)
        fake = gen(z).detach()
        real_g = real.detach().requires_grad_(True)
        fake_g = fake.detach().requires_grad_(True)
        d_loss = rp_d_loss(critic(real_g), critic(fake_g)) + cap_penalty(
            input_grad(critic, real_g),
            input_grad(critic, fake_g),
            coeff=b_cap,
        )
        opt_d.zero_grad()
        d_loss.backward()
        opt_d.step()

        z = prior.sample(batch, jitter=0.02)
        fake = gen(z)
        g_loss = rp_g_loss(critic(real.detach()), critic(fake))
        g_loss = g_loss + 0.05 * vicreg_loss(prior.particles, std_target=0.15)
        opt_g.zero_grad()
        g_loss.backward()
        opt_g.step()
        ema_g.update(list(gen.parameters()))
        ema_p.update(list(prior.parameters()))

    ema_g.copy_to(list(gen.parameters()))
    ema_p.copy_to(list(prior.parameters()))
    with torch.no_grad():
        fake = gen(prior.sample(2000, jitter=0.02))
        metrics = hq_and_cover(fake, means, sigma)
    metrics.update(
        {
            "n_modes_req": int(n_modes),
            "steps": int(steps),
            "seed": int(seed),
            "b_cap": float(b_cap),
            "sigma": float(sigma),
            "d_loss": float(d_loss.detach()),
            "g_loss": float(g_loss.detach()),
        }
    )
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--b-cap", type=float, default=1.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    row = train_gaussians(
        n_modes=args.modes, steps=args.steps, seed=args.seed, b_cap=args.b_cap
    )
    print(
        f"modes={row['modes']}/{row['n_modes']} cover={row['cover']:.3f} "
        f"hq={row['hq']:.5f} b_cap={row['b_cap']:g} steps={row['steps']}"
    )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            "\n".join(f"{k}={v}" for k, v in row.items()) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
