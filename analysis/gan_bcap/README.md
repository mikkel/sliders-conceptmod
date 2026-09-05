# Gaussian / b_cap toys

The 2-D slider wiring and diagnostic table live in
[`../slider2d/gan_bcap_findings.md`](../slider2d/gan_bcap_findings.md).

This folder is the ParticleGAN 100-Gaussians smoke for the same
`RpGAN + b_cap` core (`gaussian_repro.py`). CI runs 8 modes; `--modes 100`
is opt-in and is **not** a Music 3 listen.

```bash
PYTHONPATH=. python analysis/gan_bcap/gaussian_repro.py --modes 8 --steps 1500
```
