# RpGAN + b_cap on the 2-D slider fixtures

CPU-only port of the ParticleGAN 100-Gaussians / Music-LM adversarial
core (`rp_g_loss`, one-sided `b_cap`, Fourier-2 critic, ParticlePrior,
VICReg, EMA, Adam β1=0, delayed cosine LR) into `analysis/slider2d`.
Real samples are leftover-gated caption poles plus a span/end cloud —
hidden-state deltas, **not** rendered audio. This page does not claim
Music 3 listen quality.

## Recipe

- teacher: `faithful_guard_e` (blend-guarded leftover ê; refuses when ê restates the axis)
- b_cap coeff: `1.0` (soft cap above 1, free below)
- feature matching: `0.0` (0 = off; raw FM is uncapped by b_cap)
- cover_weight: `1.5` (mode pin on the shared residual; needed on sheet/exam width)
- GAN steps: field/sheet `1200`, exam `1200`, seed `0`
- supervised baseline steps: `400`

```bash
PYTHONPATH=. python analysis/slider2d/run_lm_adv.py --out docs/lm-2d-adv
PYTHONPATH=. python analysis/gan_bcap/gaussian_repro.py --modes 8 --steps 1500
PYTHONPATH=. pytest tests/test_lm_2d_adv.py -q
```

## Compiled gate vs supervised baselines

GAN compiled verdict: **works**. `exam_score` = `0.968`.

| cell | GAN | supervised baseline | baseline recipe | why the baseline loses |
|---|---|---|---|---|
| sheet leftover | PASS (leak -0.000, kept 0.931) | FAIL (leak +0.228, kept 0.993) | `faithful_raw` / v6 | copies unused ê inside the raw poles |
| sheet leftover (midpoint) | — | FAIL (kept 0.367, off 0.406) | `pair_odd_midpoint` / v9 | deletes `c`; walks off the sheet |
| sheet gender | PASS (kept 0.995) | PASS on `faithful_raw` / FAIL on v9 (scoreboard) | caption vs midpoint | GAN keeps `c` like a caption teacher |
| exam divergent | PASS (overlap 1.000, swing 1.000) | PASS (overlap 1.000) | `faithful_raw` | same caption target; leftover-gate refuses to eat the axis |
| exam close | PASS (overlap 1.000, swing 0.968) | PASS (overlap 1.000) | `pair_odd_midpoint` | midpoint teacher has no delivery to roll out |
| exam unused_e | PASS (overlap 0.984) | PASS on leftover-gated MSE (scoreboard) | `faithful_sub_e` | GAN uses the same leftover-gated real cloud |

Scoreboard cells this has to beat: `faithful_raw` works the exam pairs and
fails leftover leak (`≈ 0.228` > `0.2`); `pair_odd_midpoint`
locks pair-odd cos and fails the sheet (kept `≈ 0.37` < `0.9`).
Sheet lock `0.9`, garble cap `0.05`,
swing `0.6`. A recipe WORKS only if every cell it is
read on passes.

## Field2D polarity / leftover

| metric | GAN | gate |
|---|---:|---|
| slider cos | +1.000 | ≥ 0.90 |
| leak ratio | +0.001 | ≤ 0.20 |
| ±1 cos | -1.000 | ≤ −0.85 |
| leak_frac (bipolar) | -1.000 | ≤ −0.85 |
| same_dir (even share) | 0.000 | small |
| pass | PASS | |

## What moved the needle

- **Leftover-gated real cloud** (`faithful_guard_e`): without it the GAN
  copies unused ê the same way `faithful_raw` does. The guard refuses on
  energy-v4, so exam_divergent keeps the genre/BPM ride.
- **End-margin + span samples**: 2-D analogue of last-token + lyric-span
  pooling. End-margin keeps D pinned on the actual poles; span stops the
  residual from collapsing to the midpoint.
- **b_cap = 1**: one-sided, both real and fake. D stays steep enough to
  separate the two poles and cannot race `||∇D||` to infinity.
- **Particle L2 + small VICReg**: particles stay a jitter prior. The
  scored object is the shared residual, so particles must not steal the mode.
- **cover_weight = 1.5** (mode pin): pure RpGAN reaches attributed 2-D
  poles (leak 0.02, ±1 = −1) but undershoots the sheet/exam residual
  (on-sheet kept 0.23). A small MSE onto the leftover-gated centers is
  the 2-D HQ analogue — without it the shared LoRA-like residual stays
  near 0 while particles eat the modes. 1200 steps + 1.5 clears the
  0.90 on-sheet lock; 800 + 1.0 lands at 0.86.
- **Feature matching off** by default. Unnormalized FM is uncapped by
  b_cap and pulled D features instead of the residual. Normalized FM was
  tried and did not beat the leftover gate.
- **EMA residual + Adam β1=0 + delayed cosine**: the ParticleGAN schedule.
  lr 5e-3, delay 80, particle L2 0.02. Heavier VICReg/L2 starved G.

## What this is not

- Not a Music 3 listen. No MiniMax weights, no audio render.
- Not a new live `--pole_mode`. The live default stays `--lm_target v9`.
- Not the 100-Gaussians HQ number on this page. That smoke lives in
  `analysis/gan_bcap/gaussian_repro.py` (8 modes in CI; `--modes 100` is opt-in).

Gates used: leftover leak ≤ 0.2, on-sheet kept ≥ 0.9,
off-sheet ≤ 0.05, swing ≥ 0.6.
Compiled labels: `works`, `works-on-some-pairs`, `fails`.

