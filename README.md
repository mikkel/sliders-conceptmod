# Concept Sliders

**MiniMax Music 3 port:** see [MUSIC3.md](MUSIC3.md) (current trainer defaults,
shipped sliders, GPU pitfalls) and [slider_pipeline/README.md](slider_pipeline/README.md)
(paired recipe-comparison runbook). Listen sets live in `eval/listen/`. Use the
`minimax-music3` conda env and **do not** `pip install -r requirements.txt`.

**Krea image sliders (opt-in):** see [docs/krea-slider.md](docs/krea-slider.md).
UNI analog on `krea/Krea-2-Raw` (train LoRAs on Raw, run on Turbo). Rank 16,
512 px, Raw 28 steps CFG 4.5 / Turbo 8 steps CFG 0. Does not change the
Music 3 default. Anima / ZiT / H3 are not in this trainer.

**Sana 0.6B image sliders (opt-in cheap test backend):** see
[docs/sana-slider.md](docs/sana-slider.md). UNI analog on
`Efficient-Large-Model/Sana_600M_512px_diffusers`. Train **xattn**
(conceptmod 0.6B default) or `--lora RANK`, 512 px, 20 steps, CFG 4.5.
Fruit-bowl control: `a bowl of fruit on a table`. Does not change the
Music 3 default.

###  [Project Website](https://sliders.baulab.info) | [Arxiv Preprint](https://arxiv.org/pdf/2311.12092.pdf) | [Trained Sliders](https://sliders.baulab.info/weights/xl_sliders/) | [Colab Demo](https://colab.research.google.com/github/rohitgandikota/sliders/blob/main/demo_concept_sliders.ipynb) <br>
Official code implementation of "Concept Sliders: LoRA Adaptors for Precise Control in Diffusion Models"

<div align='center'>
<img src = 'images/main_figure.png'>
</div>

## Colab Demo
Try out our colab demo here [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rohitgandikota/sliders/blob/main/demo_concept_sliders.ipynb)

## Setup
To set up your python environment:
```
conda create -n sliders python=3.9
conda activate sliders

git  clone https://github.com/rohitgandikota/sliders.git
cd sliders
pip install -r requirements.txt
```
## Textual Concept Sliders
### Training SD-1.x and SD-2.x LoRa
To train an age slider - go to `train-scripts/textsliders/data/prompts.yaml` and edit the `target=person` and `positive=old person` and `unconditional=young person` (opposite of positive) and `neutral=person` and `action=enhance` with `guidance=4`. <br>
If you do not want your edit to be targetted to person replace it with any target you want (eg. dog) or if you need it global replace `person` with `""`  <br>
Finally, run the command:
```
python trainscripts/textsliders/train_lora.py --attributes 'male, female' --name 'ageslider' --rank 4 --alpha 1 --config_file 'trainscripts/textsliders/data/config.yaml'
```

`--attributes` argument is used to disentangle concepts from the slider. For instance age slider makes all old people male (so instead add the `"female, male"` attributes to allow disentanglement)


#### Evaluate 
To evaluate your trained models use the notebook `SD1-sliders-inference.ipynb`


### Training SD-XL
To train sliders for SD-XL, use the script `train_lora_xl.py`. The setup is same as SDv1.4

```
python trainscripts/textsliders/train_lora_xl.py --attributes 'male, female' --name 'agesliderXL' --rank 4 --alpha 1 --config_file 'trainscripts/textsliders/data/config-xl.yaml'
```

#### Evaluate 
To evaluate your trained models use the notebook `XL-sliders-inference.ipynb`


## Z-Image Turbo (ZiT) image sliders

Opt-in UNI analog on [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
(6B, LoRA 16, 768px, 8 steps, CFG 0). Train and infer both use the
neutral caption at +1; the + caption is the concept teacher only.
Positive / neutral yaml, unused attributes pinned, no minus teacher.
**Does not change Music 3 defaults.** Live train card:
[docs/zimage-slider.md](docs/zimage-slider.md).

```
CUDA_VISIBLE_DEVICES=N python conceptmod/textsliders/train_lora_zimage.py \
  --name age-zit --prompts_file conceptmod/textsliders/data/prompts-zimage.yaml \
  --rank 16 --alpha 16 --resolution 768 --sample_steps 8 --sample_guidance 0.0 \
  --steps 500 --seed 7 --device 0
```

## Sana 0.6B image sliders (cheap test backend)

Opt-in UNI analog on
[Efficient-Large-Model/Sana_600M_512px_diffusers](https://huggingface.co/Efficient-Large-Model/Sana_600M_512px_diffusers)
(0.6B, xattn or LoRA, 512px, 20 steps, CFG 4.5). Train and infer both
use the neutral caption at +1; the + caption is the CFG teacher only.
Positive / neutral yaml, unused attributes pinned, no minus teacher.
Fruit bowl is the control prompt. **Does not change Music 3 defaults.**
Live train card:
[docs/sana-slider.md](docs/sana-slider.md).

```
CUDA_VISIBLE_DEVICES=0 python conceptmod/textsliders/train_lora_sana.py \
  --name happy-sana --prompts_file conceptmod/textsliders/data/prompts-sana.yaml \
  --train_method xattn --resolution 512 --sample_steps 20 --sample_guidance 4.5 \
  --control_prompt "a bowl of fruit on a table" \
  --steps 500 --lr 2e-5 --seed 7 --device 0
```

## Visual Concept Sliders
### Training SD-1.x and SD-2.x LoRa
To train image based sliders, you need to create a ~4-6 pairs of image dataset (before/after edit for desired concept). Save the before images and after images separately. You can also create a dataset with varied intensity effect and save them differently. 

To train an image slider for eye size - go to `train-scripts/imagesliders/data/config.yaml` and edit the `target=eye` and `itive='eye'` and `unconditional=''` and `neutral=eye` and `action=enhance` with `guidance=4`. <br>
If you want the diffusion model to figure out the edit concept - leave `target, positive, unconditional, neutral` as `''`<br>
Finally, run the command:
```
python trainscripts/imagesliders/train_lora-scale.py --name 'eyeslider' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config.yaml' --folder_main 'datasets/eyesize/' --folders 'bigsize, smallsize' --scales '1, -1' 
```
For this to work - you need to store your before images in `smallsize` and after images in `bigsize`. The corresponding paired files in both the folders should have same names. Both these subfolders should be under `datasets/eyesize`. Feel free to make your own datasets in your own named conventions.
### Training SD-XL
To train image sliders for SD-XL, use the script `train-lora-scale-xl.py`. The setup is same as SDv1.4

```
python trainscripts/imagesliders/train_lora-scale-xl.py --name 'eyesliderXL' --rank 4 --alpha 1 --config_file 'trainscripts/imagesliders/data/config-xl.yaml' --folder_main '/share/u/rohit/imageXLdataset/eyesize_data/'
```

### Anima (opt-in yaml slider)

Flow-matching 2B DiT (`circlestone-labs/Anima-Base-v1.0-Diffusers`). Default `--lm_target trajectory` + `--teacher caption`: K-step FlowMatch Euler so neu+LoRA matches the frozen plus *trajectory* (1-step `direct` / `cfg_delta` cannot carry smile — v-space pos/neu gap is ~1e-4). Caption-only plus still jumps crop (full-body→close-up on closed-mouth→teeth); `--teacher same_crop` / `--lm_target same_crop` inverts the neu traj and denoises plus from mid-σ so expression moves without zoom. Train and sample share bare infer/neu captions (attributes are unused-token pins, not prefixes). Cycles woman + man. Default `--lora_targets conditioner` (AnimaTextConditioner `q_proj/k_proj/v_proj/o_proj`; Qwen3 `text_encoder` stays frozen). `--lora_targets dit` is the old transformer-only recipe. Rank 16, `--lr 1e-4`, `--traj_steps 4`, `--sample_every 100`. 4090 smile retrain: `--resolution 512`. In-process PEFT scale grid through `pipe(prompt=...)` after `sync_peft_into_modular_pipeline` (same conditioner object as `encode_text`). Conditioner embed diag: `scripts/diag_anima_conditioner_embed.py`. Same-crop dummy: `scripts/smoke_anima_same_crop_teacher.py`. Does not change Music 3 defaults (`--lm_target v9`). **Anima-Turbo v1.1 is preview-only** (CFG 1, 8–12 steps; convert helper `scripts/convert_anima_turbo_diffusers.py`); do not train on Turbo. Train card: `docs/anima-slider.md`. Dummy smoke: `scripts/smoke_anima_slider.py`.

```
HF_HUB_OFFLINE=1 python conceptmod/textsliders/train_lora_anima.py \
  --name smile-anima \
  --prompts_file conceptmod/textsliders/data/prompts-anima.yaml \
  --model_id circlestone-labs/Anima-Base-v1.0-Diffusers \
  --lora_targets conditioner --rank 16 --resolution 512 --sample_steps 40 --cfg 4 \
  --lr 1e-4 --lm_target trajectory --traj_steps 4 --sample_every 100 \
  --device cuda:0 --save_dir models/smile-anima
```

## Editing Real Images
Concept sliders can be used to edit real images. We use null inversion to edit the images - instead of prompt, we use sliders! <br>
Checkout - `demo_image_editing.ipynb` for mode details.


## Citing our work
The preprint can be cited as follows
```
@article{gandikota2023sliders,
  title={Concept Sliders: LoRA Adaptors for Precise Control in Diffusion Models},
  author={Rohit Gandikota and Joanna Materzy\'nska and Tingrui Zhou and Antonio Torralba and David Bau},
  journal={arXiv preprint arXiv:2311.12092},
  year={2023}
}
```
