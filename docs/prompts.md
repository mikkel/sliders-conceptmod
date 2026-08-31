# Prompt-file catalog

All files are under `conceptmod/textsliders/data/`. Axes and
constraints below are taken from the YAML headers and the trainer
that loads them — not from listen folklore.

Pass the path with `--prompts_file`. Do not prefix yaml `attributes`
onto captions unless that backend says so (Anima / Krea / LTX / H3
chiaro: attributes are unused-token pins only).

## Image / video (opt-in UNI)

| file | trainer | axis | notes |
|---|---|---|---|
| `prompts-krea-happy.yaml` | `train_lora_krea.py` | smile / happy | Bare captions. TE-only embed UNI card (`--lora_targets te --lm_target embed`). |
| `prompts-krea-detailed.yaml` | `train_lora_krea.py` | fine detail / texture | Same-subject as neu (no style jump). detail-krea-v3. |
| `prompts-krea.yaml` | `train_lora_krea.py` | age | plus = old, neu = person. Unused gender pinned. |
| `prompts-anima.yaml` | `train_lora_anima.py` | smile / happy | Bare captions. `concept_words: smiling, smile, happy, joyful, teeth`. |
| `prompts-anima-canary.yaml` | `train_lora_anima.py` | same, minus left in | Minus is a canary, not a teacher. |
| `prompts-sana.yaml` | `train_lora_sana.py` | happy | Train/infer +1 on **neu**. Fruit-bowl control. Stronger smile positives (weak "a happy person" did not move stock Sana). |
| `prompts-zimage.yaml` | `train_lora_zimage.py` | age | plus = old, neu = person. Infer +1 on neu + LoRA. |
| `prompts-ltx25-smile.yaml` | `train_lora_ltx25.py` | smile / happy | Bare captions. Same subject / clothes / framing / motion / light / sound. Hold PRE-connector. |
| `prompts-minimax-h3.yaml` | `train_lora_minimax_h3.py` | age | FL2VA / t2va. Negative is a canary. |
| `prompts-minimax-h3-chiaroscuro.yaml` | `train_lora_minimax_h3.py` | lighting | Same concrete subject on neu and plus (person / dining room / scholar jar). Default `--hold_mode non_concept`. Recommended live card is chiaro-v5 (`--hold_weight 3.0`, 2500 steps). |

Upstream SD / XL / SD3 / Flux / Cascade yamls (`prompts.yaml`,
`prompts-xl.yaml`, `prompts-sd3.yaml`, `prompts-flux.yaml`,
`prompts-cascade.yaml`) belong to the original Baulab trainers, not
the Music 3 or UNI cards above.

## Music 3 — shipped / current axes

v4 files are Structured Caption poles (Global Metadata / Vocal
Details / Arrangement). v3 files are the earlier flat
`Genre: … BPM: …` poles — off-distribution for the studio rewriter.
TF halves stay on the v2 / single-row files named in MUSIC3.md.

| axis | v2 / TF | v3 LM | v4 LM | leftover `ê` (v4 header) |
|---|---|---|---|---|
| energy (quiet ↔ loud) | `prompts-energy.yaml` | `prompts-energy-v3.yaml` | `prompts-energy-v4.yaml` | mix / BPM / genre restates `a` — see MUSIC3.md |
| distortion (clean ↔ metal) | `prompts-distortion.yaml` | `prompts-distortion-v3.yaml` | `prompts-distortion-v4.yaml` | unused leftover; `--lm_target pair_odd_sub_e` |
| tempo (slow ↔ fast) | `prompts-tempo.yaml` | `prompts-tempo-v3.yaml` | `prompts-tempo-v4.yaml` | leftover = genre; BPM is the slider |
| space (dry ↔ wet) | `prompts-space.yaml` | — | — | TF only in-repo |
| gender (male ↔ female) | `prompts-gender-tf.yaml` (failed TF) | `prompts-gender-v3.yaml` | `prompts-gender-v4.yaml` | clean pair; hold 0 |
| trip-hop ↔ pop | `prompts-triphop.yaml` / `prompts-triphop-v3-single.yaml` | `prompts-triphop-v3.yaml` | `prompts-triphop-v4.yaml` | leftover = BPM only; genre/dusty/glossy is the slider |
| rap ↔ slow sung | — | `prompts-rapslow-v3.yaml` | `prompts-rapslow-v4.yaml` | delivery; row 1 tempo-matched |
| rhyme | — | `prompts-rhyme-v3.yaml` | `prompts-rhyme-v4.yaml` | Vocal Details + arrangement |
| breath | — | `prompts-breath-v3.yaml` | `prompts-breath-v4.yaml` | air / inhales; needs whole-record gestalts |
| live (room bleed) | — | `prompts-live-v3.yaml` | `prompts-live-v4.yaml` | people / leakage; tempo and energy pinned |
| dust (dusty ↔ glossy) | `prompts-cand-dust-v1.yaml` | — | — | **only TF slider that passed all six render gates** |

`prompts-pop-v3-single.yaml` is `prompts-triphop-v3-single.yaml` with
poles swapped (unidirectional glossy-pop LoRA). `prompts-gender.yaml`
/ `prompts-music3.yaml` are older generic rows; prefer the axis files
above.

Leaky v4 axes train with `--lm_target pair_odd_sub_e` (or default
`v9` + declared `leak_positive` / `leak_negative`). Gender stays
`v9` with no `ê`.

## Music 3 — candidate pairs (pipeline onboard)

Certify with `slider_pipeline.py onboard` **before** training. These
are production-axis candidates at fixed genre/BPM/arrangement (except
energy, which puts level on the odd axis). Gate notes are in each
file header (revised 2026-08-21).

| file | intended axis | pipeline note |
|---|---|---|
| `prompts-cand-dust-v1.yaml` | dust / sheen | passing TF baseline (`dust-tf-v1`) |
| `prompts-cand-space-v1.yaml` | dry / cavernous | G0 first |
| `prompts-cand-energy-v1.yaml` | density / level | next energy pair after caption-BPM leak; do **not** add `--attributes` to TF |
| `prompts-cand-grit-v1.yaml` | grit / clean | G0 first |
| `prompts-cand-vintage-v1.yaml` | vintage / modern | `G-vintage` failed G0 at p=0.75 (SCORING.md) |

## Music 3 — probe / try cells

Not shipped. Used by `scripts/probe_lm_axis_signal.py` when hunting
a separable pole pair.

| prefix | what the headers test |
|---|---|
| `prompts-breath-try-*.yaml` | arrangement-swap vs minimal pair; caption shape; nameless copy; identity-as-who |
| `prompts-rhyme-try-*.yaml` | country AABB, hook-pop sprawl, artsong minus, mantra, overflow, recitative |

Keep artist names out of new Music 3 poles (MUSIC3.md).
