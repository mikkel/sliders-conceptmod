# Live 2-D cells: gender-like and energy-like

The old energetic×gender leak cell set û from the pole names, so
`|odd·û|/||odd|| ≈ 0.95` and project+hold looked leak-0. That hid
live energy the same way the first fixture hid gender.

Two CPU cells match the live Music 3 logs. One default loss has to
be right on both. Hub flags are not required.

CPU only. No Hub weights, no GPU, no Music 3 downloads.

## Cells

- **Gender-like:** clean pair, short û at `|odd·û|/||odd|| = 0.20`.
  Always-project+hold must FAIL (tiny `||d+||`, hold eats the singer).
  Pair-symmetric / Hub-odd must PASS.
- **Energy-like:** leaky pair (unused attr in `pos−neg`), short û **is**
  the intended axis, alignments `[0.48, 0.48, 0.68, 0.68]` on four rows.
  Per-row 0.50 splits the rows (mixed teacher) and must FAIL.
  Hub / symmetric-on-pair must still leak. A good loss is leak-low,
  keeps strength on the intended û, and uses the **same teacher on every row**.
- **Old leak-0 cell** (û = energetic/calm pole names) stays a regression
  only: project+hold must still be leak-0 there. It is not energy.

## Verdict

**`--lm_target v9` is now slider-level `|odd·û|/||odd||` ≥ 0.5.**
One mean over the slider, then project+hold on every row or
pair-symmetric on every row. Gender mean 0.20 → fallback (do not
regress the live woman/man save). Energy mean 0.58 → project all
four rows onto the short loud/calm û (no mixed teacher, leak-0).

Gender-like NEW: leak +0.000, `||d+||/||odd||` 1.000, cos 1.000, mixed False.
Energy-like NEW: leak +0.000, `||d+||/||odd||` 0.580, cos to û 1.000, mixed False.
Per-row 0.50 on energy: mixed=True, leak +0.605, cos 0.856.

Discarded: always-project (kills gender at 0.20), Hub leash
(still leaks unused attr), per-row 0.50 (mixed energy teacher),
soft per-row blend of pair-odd and `(a·û)û` (λ that keeps gender
still leaks energy; λ that kills energy leak eats gender).
û = pole-odd on the energy cell is identity project and still leaks —
cheat leak +1.828, align 1.000.

## Table (gender-like / energy-like × Hub / always-project / gated-0.50 / NEW)

### Gender-like (intended axis = clean pair-odd)

| policy | leak | ||d+||/||odd|| | cos intended | mixed-row | verdict |
|---|---:|---:|---:|---|---|
| `hub` | +0.000 | 1.000 | 1.000 | no | **PASS** |
| `always_project_hold` | +4.899 | 0.200 | 0.200 | no | **FAIL** |
| `gated_row_0.50` | +0.000 | 1.000 | 1.000 | no | **PASS** |
| `slider_align_0.50` | +0.000 | 1.000 | 1.000 | no | **PASS** |

### Energy-like (intended axis = short loud/calm û, not pole-odd)

| policy | leak | ||d+||/||odd|| | cos intended | mixed-row | verdict |
|---|---:|---:|---:|---|---|
| `hub` | +1.388 | 0.992 | 0.584 | no | **FAIL** |
| `always_project_hold` | +0.000 | 0.580 | 1.000 | no | **PASS** |
| `gated_row_0.50` | +0.605 | 0.678 | 0.856 | yes | **FAIL** |
| `slider_align_0.50` | +0.000 | 0.580 | 1.000 | no | **PASS** |

![energy-like residuals](lm-live-cells/energy.png)

## What each policy does

- `hub`: pair-odd + published floor/anchor. **PASS** gender (clean pair),
  **FAIL** energy (unused attr stays in `(pos−neg)/2`).
- `always_project_hold` / `--lm_target v9_always`: **FAIL** gender
  (strength 0.20, hold eats the singer), **PASS** energy (û is the axis).
- `gated_row_0.50` / live v12: **PASS** gender (fallback), **FAIL** energy
  as mixed-teacher (0.48 fallback, 0.68 project).
- `slider_align_0.50` / default `--lm_target v9`: **PASS** both.
  Same teacher on every row of one slider.

## Bare Music 3 LM train

No Hub flags. Default `--lm_target v9` is the slider-level recipe.
YAML already declares `slider_positive` / `slider_negative`.
Old always-project is `--lm_target v9_always`. Per-row v12 is
`--lm_target v9 --project_align_scope row --project_align_min 0.50`.

```bash
python conceptmod/textsliders/train_lm_slider_music3.py --prompts_file conceptmod/textsliders/data/prompts-gender-v4.yaml
python conceptmod/textsliders/train_lm_slider_music3.py --prompts_file conceptmod/textsliders/data/prompts-energy-v4.yaml
```

## How to run

```bash
PYTHONPATH=. python analysis/slider2d/run_lm_live.py --out docs/lm-live-cells
PYTHONPATH=. pytest tests/test_lm_live_cells.py tests/test_lm_v9_mismatch.py tests/test_lm_v9_2d.py tests/test_lm_trainer_v9.py -q
```

Seed `0`, `200` Adam steps.

