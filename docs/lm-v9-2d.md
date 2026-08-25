# LM v9 formulation on the 2-D CPU field

Same geometry as [2d-analysis.md](2d-analysis.md): energetic ↔ calm
versus unused male ↔ female. This scores the **loss / target math**
from the Hub v9 sidecar (`ntc-ai/minimax-music3-concept-sliders`,
energy-lm-v9), not Hub weights and not caption BPM.

The live `train_lm_slider_music3.py` in this tree still only has
`--symmetric` / `--common_beta`. v9 terms (`target_mode`,
`leakage_floor`, `anchor_weight`, `anchor_autocal`) are a CPU
stand-in of the published Hub README formulas, in
`slider_targets.py`. Endreg / planreg / semantic-KL poles are
AR-only and are not on this field.

## Verdict

**v9 does not fix the 2-D attribute leak.** Slider axis
**right**, unused-gender leak **needs_help**,
±1 collapse **right**. Same pattern as
`--symmetric` alone. `leakage_floor` only sizes the *even-mode*
kappa that `--symmetric` already cancelled. It would **not** have
stopped unused-gender leak without `--attributes`.

| method | slider | leak | ±1 collapse | slider cos | leak ratio | ±1 cos |
|---|---|---|---|---:|---:|---:|
| `lm_raw` | **needs_help** | **needs_help** | **needs_help** | 0.509 | 1.692 | 0.253 |
| `lm_symmetric` | **right** | **needs_help** | **right** | 0.946 | 0.342 | -1.000 |
| `lm_symmetric_floor` | **right** | **needs_help** | **right** | 0.946 | 0.342 | -1.000 |
| `lm_v9` | **right** | **needs_help** | **right** | 0.929 | 0.397 | -0.995 |
| `lm_raw_attrs` | **right** | **right** | **right** | 1.000 | 0.000 | -1.000 |
| `m3_nmse_axis` | **right** | **needs_help** | **right** | 0.949 | 0.333 | -1.000 |

![v9 vs raw / symmetric residuals](lm-v9-2d/compare.png)

![Learned residuals on the teacher field](lm-v9-2d/quiver.png)

![Leak vs slider cosine](lm-v9-2d/scatter.png)

## What each flag does here

- `lm_raw` / v6 `target_mode: faithful`: poles are raw `pos`/`neg`.
  ±1 cos=0.253 (even-mode collapse).
  Leak 1.692 — both poles sit above quiet `song`
  *and* `energetic` already leaks male.
- `lm_symmetric` / v4 polarity: `tgt(±1) = neu ± (pos−neg)/2`.
  Collapse **right** (-1.000). Leak
  0.342 remains — it lives in the odd teacher.
- `lm_symmetric_floor`: `leakage_floor=-0.9` with `anchor_weight=0`.
  Identical to symmetric (leak 0.342,
  ±1 -1.000). The floor is inert
  without an anchor term.
- `lm_v9`: symmetric pole MSE + `anchor_weight=0.3` toward
  `(1−κ)(neu ± a) + κ·raw`, κ autocalibrated so a perfect blend
  fit stays at collapse ≤ −0.9. On this pair r=0.253,
  κ=0.177, so the blend is still ~82%
  symmetric. Leak 0.397 — same leaked odd axis.
- `lm_raw_attrs`: `--attributes male,female` zeros the unused
  gender axis (leak 0.000). That is still
  the only fix this field can see.
- TF `nmse`+`axis`: leak 0.333, already
  measured in [tf-leak.md](tf-leak.md). Default TF is not the v9 question.

## Why leakage_floor cannot kill unused-gender leak

Hub v9:

```
a = (h+ − h−) / 2
t± = h0 ± a                         # pole target, still pos−neg
r = cos(h+ − h0, h− − h0)
ρ² = (1 − r) / (1 + r)
κ = √(ρ² · (1 + floor) / (1 − floor))   # clamped to [0, 1]
anchor± = (1 − κ)(h0 ± a) + κ h±
L = MSE(h(±1), t±) + 0.3 · MSE(h(±1), anchor±)
```

Ungated pair on this field: r = 0.253 (even-mode
collapse), odd leak = 0.342. `leakage_floor` solves
for how much even mode may be blended *back in*. The unused-gender
component is already inside `a`. No amount of κ projects it out.
`--attributes` is still the paper disentangle; v9 is a collapse
governor on top of `--symmetric`.

## What this field cannot see

- AR endreg / planreg / `pole_mode: semantic_kl` / `collapse_weight`
  (v6 had those; v9 turns planreg and collapse off).
- Real Music 3 hidden geometry, Hub weights, v7 prompt yamls.
- Caption-BPM leak (that is a TF pair fact; see tf-leak.md).

## How to run

```bash
PYTHONPATH=. python analysis/slider2d/run_lm_v9.py --out docs/lm-v9-2d
PYTHONPATH=. pytest tests/test_lm_v9_2d.py tests/test_2d_slider_geometry.py -q
```

CPU only. No Hub, no GPU, no Music 3 weights.

Seed `0`, `250` Adam steps.

