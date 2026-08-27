# Live Music 3 hold failure on a CPU fixture

Analysis/test/docs only. The live `--lm_target v9` default remains
full pair-odd; this fixture does not wire a teacher or trainer change.

## Verdict

The orthonormal 2-D PASS was real but easy to misread: at λ=8 it has slider-cos 0.988 while trainer c+ is only 0.699, perc is 72%, and loss is 0.849. It locks û and suppresses leftover ê; it is not a v12-looking copy of pair-odd.

A D=64 local-Jacobian analogue now reaches the live shape: a tiny ||ê_⊥û||=0.020 is normalized before hold, the effective scalar-vs-MSE factor is 257, c+=0.311, slider-cos=0.369, collapse=+0.182, and loss=2.634. Dimension makes the hold hard; the polarity break additionally requires a shared sign-asymmetric response mode. Dimension alone cannot move an exactly odd residual away from collapse −1.

The same-loudness pin is unchanged after orthogonalization: reducing the raw slider coefficient does not remove the normalized density/genre/syntax residual. This reproduces why adding “medium energy” does not prove ê became leftover-only.

Leftover-only ê at λ=1 is still the canary, not a result to wire: c+=0.943, collapse=-1.000, loss=0.478, leftover=0.702. λ=8 stays bipolar and reaches leftover 0.156 in the direct field. `pair_odd_sub_e` is leak-free but has c+=0.580 against the original pair-odd: it succeeds by changing the teacher. Try the leftover-only captions + λ=1 live first; if the normalized high-D hold still breaks polarity, compare the subtract-ê teacher in a separate PR.

## Real fixture numbers

| cell | D | teacher | λ | trainer c+ | slider cos | collapse | leftover | perc% | loss |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `gender_pair_odd_no_hold` | 2 | pair_odd | 0 | +1.000 | +1.000 | -1.000 | +0.000 | 0 | 0.000 |
| `energy_2d_hold_l8` | 2 | pair_odd | 8 | +0.699 | +0.988 | -1.000 | +0.156 | 72 | 0.849 |
| `highd_synonym_hold_l8` | 64 | pair_odd | 8 | +0.311 | +0.369 | +0.182 | +0.323 | 95 | 2.634 |
| `highd_same_loudness_pin_l8` | 64 | pair_odd | 8 | +0.311 | +0.369 | +0.182 | +0.323 | 95 | 2.634 |
| `leftover_only_hold_l0.3` | 2 | pair_odd | 0.3 | +0.992 | +0.679 | -1.000 | +1.080 | 19 | 0.221 |
| `leftover_only_hold_l1` | 2 | pair_odd | 1 | +0.943 | +0.818 | -1.000 | +0.702 | 41 | 0.478 |
| `leftover_only_hold_l8` | 2 | pair_odd | 8 | +0.699 | +0.988 | -1.000 | +0.156 | 72 | 0.849 |
| `pair_odd_sub_e` | 2 | pair_odd_sub_e | 0 | +0.580 | +1.000 | -1.000 | +0.000 | 81 | 0.000 |

`loss` is the actual fixture objective. For `pair_odd_sub_e` it is
zero against the changed teacher; `pair_odd_mse` in `metrics.json`
keeps the cost against the original full pair-odd visible.

## What is now visible

- Gender-like/no-ê copies pair-odd: high c+, high intended-axis cosine,
  bipolar ±1. No junk ê is invented for this row.
- The old D=2 λ=8 PASS exposes trainer c+ and slider-cos separately.
- A tiny normalized high-D ê_⊥ plus sign-asymmetric shared capacity can
  reproduce low c+, high loss, and collapse near the live +0.18.
- Same-loudness wording does not help while the residual density/genre
  direction is unchanged.
- Leftover-only λ∈{0.3,1,8} and pair-odd−ê are directly comparable.

## Still not visible

- This is not Qwen and does not prove its exact Jacobian or identify
  which attention/MLP parameters create the shared response mode.
- It does not embed the actual captions, model multi-token hidden states,
  or reproduce the step-13 loss explosion. The synonym and pin geometry
  are declared fixture inputs, not inferred language semantics.
- The fixture shows a possible mechanism for polarity break, not that
  high dimensionality by itself causes one.

## Geometry

The live pole loss uses per-element MSE, while hold is a scalar squared
projection. Along a unit held direction in D dimensions, the relative
normal-equation factor is:

```text
student_e = teacher_e / (1 + λ D / 2)
```

For D=2 this is `1+λ`, so shrink stays parallel and ±1 remains exactly
opposite. The D=64 cell adds one two-parameter local response:
`p = x a + y b+`, `m = −x a + y b−`. The b modes are hold-free,
share parameters, and have different sign-conditioned Jacobians. Hard
hold suppresses x and selects y, making polarity failure measurable.

## Run

```bash
PYTHONPATH=. python3 analysis/slider2d/run_lm_live_failure.py
PYTHONPATH=. python3 -m pytest tests/test_lm_live_failure_fixture.py -q
```

CPU only. No Hub, GPU, Music 3, or Qwen weights.
