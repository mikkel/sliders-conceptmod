# Live Music 3 failure signatures (CPU fixture)

Maps live energy-v14 / gender-v14 trainer logs onto orthonormal 2-D
and a minimal high-D cell. **c+** is trainer alignment with pair-odd
``a``; **slider-cos** is alignment with declared ``û``. They diverge
when hold is working on energy — do not read c+ as slider lock.

CPU only. No Hub, no GPU, no Music 3 weights. Does not change
``--lm_target v9`` default.

## Signature table

| cell | D | λ | ê·û | ‖ê_⊥‖ | slider-cos | c+ | collapse | leak | perc% | loss | v12-looking | hold≠v12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `gender_pair_odd` | 2 | 0 | 0.00 | 0.000 | +1.000 | +1.000 | -1.000 | +0.000 | 0 | 0.000 | yes | no |
| `energy_pole_synonym_perp` | 2 | 8 | 0.50 | 0.866 | +0.988 | +0.696 | -1.000 | +0.154 | 72 | 0.852 | no | yes |
| `high_d_tiny_eperp` | 64 | 8 | 1.00 | 0.100 | +0.998 | +0.585 | -1.000 | +0.008 | 81 | 0.030 | no | no |
| `synonym_pin_medium_energy` | 2 | 8 | 0.95 | 0.312 | -0.165 | +0.698 | -1.000 | +5.968 | 72 | 0.846 | no | no |
| `leftover_only_l0.3` | 2 | 0.3 | 0.00 | 1.000 | +0.684 | +0.984 | -1.000 | +1.068 | 21 | 0.237 | no | no |
| `leftover_only_l1` | 2 | 1 | 0.00 | 1.000 | +0.821 | +0.936 | -1.000 | +0.694 | 42 | 0.489 | no | no |
| `leftover_only_l8` | 2 | 8 | 0.00 | 1.000 | +0.988 | +0.696 | -1.000 | +0.154 | 72 | 0.852 | no | yes |
| `pair_odd_sub_e` | 2 | 0 | 0.00 | 0.000 | +1.000 | +0.580 | -1.000 | +0.000 | 81 | 0.956 | no | yes |

## Verdict: live bullet → fixture

| live bullet | fixture | visible | slider-cos | c+ | verdict |
|---|---|---:|---:|---:|---|
| gender-like copy pair-odd | `gender_pair_odd` | yes | +1.000 | +1.000 | 2-D cell; c+ high, collapse antipodal, hold λ=0 |
| energy ê synonym + ê_⊥û λ=8 | `energy_pole_synonym_perp` | yes | +0.988 | +0.696 | slider-cos high, c+ ~0.70, loss ~0.85 — PASS leftover, FAIL v12 look |
| high-D tiny ê_⊥ + λ=8 collapse +0.18 | `high_d_tiny_eperp` | partial | +0.998 | +0.585 | c+ drops vs 2-D; ê_⊥ residual shrinks. collapse -1.00 stays antipodal in fixture — live +0.18 not reproduced. |
| synonym pin (medium energy, density ∥ û) | `synonym_pin_medium_energy` | yes | -0.165 | +0.698 | raw hold still fights; ê_⊥ tiny at high ρ |
| leftover-only ê + λ=1 canary | `leftover_only_l1` | yes | +0.821 | +0.936 | leak +0.694, loss 0.489 — trainable; try live before pair_odd_sub_e |
| pair_odd_sub_e vs hold | `pair_odd_sub_e` | yes | +1.000 | +0.580 | leak-0 unused; teacher projects like short-û — next step if λ=1 canary fails live |

## Notes

- **Gender-like** copies full pair-odd (hold λ=0). c+ and slider-cos
  are both high — the look people misread as “slider locked”.
- **Energy ê_⊥û** at ρ=0.5 locks slider-cos and leftover leak but
  c+ ~0.70 and loss ~0.85 — hold working, not v12-looking.
- **High-D** shrinks the ê_⊥ residual; c+ floors near |odd·û|.
  Symmetric ``odd_even`` keeps collapse antipodal; live +0.18 is
  not reproduced on this fixture.
- **Synonym pin** (ρ=0.95, raw hold): ê_⊥ is tiny; λ=8 still fights.
- **Leftover-only ê + λ=1** is the canary before ``pair_odd_sub_e``.
- **pair_odd_sub_e** zeros unused leak but uses a projected teacher
  (c+ ~ align) — compare to hold, do not wire live without a cell win.

## How to run

```bash
PYTHONPATH=. python analysis/slider2d/run_lm_live_signature.py --out docs/lm-live-signature
PYTHONPATH=. pytest tests/test_lm_live_signature.py -q
```

Seed `0`, `200` Adam steps, high-D dim `64`.

