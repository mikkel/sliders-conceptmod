# Released Hub v9 sliders on the 2-D CPU field

Read-and-run on `mikkel/sliders-conceptmod` main (PRs #6 / #7).
Scores the **published Hugging Face v9 recipe**, not ntc-ai git and not
Music 3 LoRA weights. The 2-D field cannot load those weights.

Hub repo: [ntc-ai/minimax-music3-concept-sliders](https://huggingface.co/ntc-ai/minimax-music3-concept-sliders).
Download was JSON / YAML / MD only (`hf download … --include '*.json' --include '*.yaml' --include '*.md'`).
Zero safetensors.

Trainer defaults were not changed. `train_lm_slider_music3.py` was not
edited.

## Verdict

Every released `*-lm-v9` sidecar is the published Hub recipe
(`--symmetric` + `anchor_weight 0.3` + `anchor_autocal` +
`leakage_floor -0.9`). None used projected-odd / hold. All fourteen
map to the `lm_v9_hub` cell.

On the energetic×gender fixture that cell is **slider right, leak
needs_help (~0.40), collapse right**. `lm_v9` (project-odd + hold) is
the leak-0 cell; the released weights were not trained that way.

`lm_target` is absent from every sidecar (schema 3, before
`--lm_target`). Flags match current `--lm_target hub`, not `--lm_target
v9`.

## 2-D score (energetic × gender, CPU)

`PYTHONPATH=. python analysis/slider2d/run_lm_v9.py --out /tmp/hub-v9-2d`
(seed 0, 250 Adam steps). Same numbers as [lm-v9-2d.md](lm-v9-2d.md).

| method | slider | leak | collapse |
|---|---|---|---|
| `lm_symmetric` | **right** | **needs_help** (0.342) | **right** |
| `lm_v9_hub` | **right** | **needs_help** (0.397) | **right** |
| `lm_v9` | **right** | **right** (0.000) | **right** |

Full fixture (same run):

| method | slider | leak | collapse | slider cos | leak ratio | ±1 cos |
|---|---|---|---|---:|---:|---:|
| `lm_raw` | **needs_help** | **needs_help** | **needs_help** | 0.509 | 1.692 | 0.253 |
| `lm_symmetric` | **right** | **needs_help** | **right** | 0.946 | 0.342 | -1.000 |
| `lm_symmetric_floor` | **right** | **needs_help** | **right** | 0.946 | 0.342 | -1.000 |
| `lm_v9_hub` | **right** | **needs_help** | **right** | 0.929 | 0.397 | -0.995 |
| `lm_v9` | **right** | **right** | **right** | 1.000 | 0.000 | -1.000 |
| `lm_raw_attrs` | **right** | **right** | **right** | 1.000 | 0.000 | -1.000 |
| `m3_nmse_axis` | **right** | **needs_help** | **right** | 0.949 | 0.333 | -1.000 |

Pair geometry: r = 0.253, κ(`leakage_floor=-0.9`) = 0.177, odd leak =
0.342. Hub blend-back makes unused-gender leak slightly *worse* than
plain `--symmetric`.

`PYTHONPATH=. pytest tests/test_lm_v9_2d.py -q` → 16 passed.

## Released LM v9 → 2-D cell

Uniform across all fourteen. Absent keys: `lm_target`, `hold_weight`,
`project_odd`, `slider_positive`, `slider_negative`.

| slider | 2-D cell | symmetric | leakage_floor | anchor_weight | anchor_autocal | lm_target | prompts_file |
|---|---|---|---:|---:|---|---|---|
| breath-lm-v9 | `lm_v9_hub` | true | -0.9 | 0.3 | true | *(absent)* | `prompts-breath-v7.yaml` |
| distortion-lm-v9 | `lm_v9_hub` | true | -0.9 | 0.3 | true | *(absent)* | `prompts-distortion-v7.yaml` |
| energy-lm-v9 | `lm_v9_hub` | true | -0.9 | 0.3 | true | *(absent)* | `prompts-energy-v7.yaml` |
| gender-lm-v9 | `lm_v9_hub` | true | -0.9 | 0.3 | true | *(absent)* | `prompts-gender-v7.yaml` |
| grit-lm-v9 | `lm_v9_hub` | true | -0.9 | 0.3 | true | *(absent)* | `prompts-grit-v7.yaml` |
| hurt-lm-v9 | `lm_v9_hub` | true | -0.9 | 0.3 | true | *(absent)* | `prompts-hurt-v7.yaml` |
| joy-lm-v9 | `lm_v9_hub` | true | -0.9 | 0.3 | true | *(absent)* | `prompts-joy-v7.yaml` |
| rapslow-lm-v9 | `lm_v9_hub` | true | -0.9 | 0.3 | true | *(absent)* | `prompts-rapslow-v7.yaml` |
| rhyme-lm-v9 | `lm_v9_hub` | true | -0.9 | 0.3 | true | *(absent)* | `prompts-rhyme-v7.yaml` |
| sexy-lm-v9 | `lm_v9_hub` | true | -0.9 | 0.3 | true | *(absent)* | `prompts-sexy-v7.yaml` |
| tempo-lm-v9 | `lm_v9_hub` | true | -0.9 | 0.3 | true | *(absent)* | `prompts-tempo-v7.yaml` |
| tender-lm-v9 | `lm_v9_hub` | true | -0.9 | 0.3 | true | *(absent)* | `prompts-tender-v7.yaml` |
| triphop-lm-v9 | `lm_v9_hub` | true | -0.9 | 0.3 | true | *(absent)* | `prompts-triphop-v7.yaml` |
| yearn-lm-v9 | `lm_v9_hub` | true | -0.9 | 0.3 | true | *(absent)* | `prompts-yearn-v7.yaml` |

Catalog extras that are **not** v9 LM: `live-lm-v6b` (no v9 pair),
`dust-lm-v1-faithful-kl-pole03` (faithful / KL), `space-tf-v6`
(transformer). Not scored as Hub v9.

## Sidecar flag dump

Every `weights/<axis>-lm-v9/<axis>-lm-v9_last.json` on the Hub (2026-08-24
card). Shared train flags:

```
symmetric:          true
target_mode:        symmetric
leakage_floor:     -0.9
anchor_weight:      0.3
anchor_autocal:     true
common_beta:        0.0
lm_target:          <absent>
hold_weight:        <absent>
project_odd:        <absent>
slider_positive:    <absent>
slider_negative:    <absent>
schema:             3
kind:               language_model
rank / alpha:       8 / 8.0
steps / seed:       800 / 7
planreg_weight:     0.0
collapse_weight:    0.0
endreg.enabled:     true
endreg.weight:      1.0
```

Hub README “v9 objective” lists the same flags and still calls them
“trainer defaults”. On this repo after #7 the live default is
`--lm_target v9` (project odd + hold, κ=0). That is a card stale-note
only; this PR does not change trainer defaults.

Machine-readable dump: [hub-v9-released-2d/sidecars.json](hub-v9-released-2d/sidecars.json).
Fixture metrics: [hub-v9-released-2d/metrics.json](hub-v9-released-2d/metrics.json).

## How to run

```bash
PYTHONPATH=. python analysis/slider2d/run_lm_v9.py --out /tmp/hub-v9-2d
PYTHONPATH=. pytest tests/test_lm_v9_2d.py -q
hf download ntc-ai/minimax-music3-concept-sliders \
  --include '*.json' --include '*.yaml' --include '*.md' \
  --local-dir /tmp/hub-sliders
```

CPU only. No GPU. No Music 3 weights.
