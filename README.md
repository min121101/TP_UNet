# TP-UNet (anonymous release)

2D medical image segmentation with an optional **Temporal Prompt (TP)**: slice-level temporal metadata is encoded by Electra and fused into the segmentation network. This repository supports two benchmarks used in our paper — **LiTS (liver CT)** and **UWMGI (UW-Madison GI tract)**.

> **Dataset protocol** (download links, preprocessing, train/val split, temporal tagging rules) is specified in the paper (*Dataset* / *Implementation Details*). This README gives the matching commands and file layout for reproduction.

---

## Table of contents

1. [Environment](#environment)
2. [Pretrained language encoders](#pretrained-language-encoders)
3. [Dataset: LiTS](#dataset-lits)
4. [Dataset: UWMGI](#dataset-uwmgi)
5. [Training on LiTS](#training-on-lits)
6. [Training on UWMGI](#training-on-uwmgi)
7. [Model configs](#model-configs)
8. [Logging (Weights & Biases)](#logging-weights--biases)
9. [Repository layout](#repository-layout)
10. [Citation](#citation)

---

## Environment

```bash
conda create -n tpunet python=3.10 -y
conda activate tpunet
pip install -r requirements.txt
```

Secrets (do **not** commit `.env`):

```bash
cp .env.example .env
# Local only: replace Your_Key with real keys if you use online wandb / APIs
```

---

## Pretrained language encoders

| Model | Role | Setup |
|-------|------|--------|
| **Electra** | Temporal prompt + TP-UNet text branch | Download `google/electra-small-discriminator` into `model/electra-small-discriminator/` |
| **CLIP ViT-B/32** | Optional text encoder (UWMGI / TP-UNet-Clip) | Auto-downloaded on first `clip.load()` |

---

## Dataset: LiTS

### Acquisition

1. Register and download the **LiTS Challenge** abdominal CT data (liver / tumor segmentation) from the official challenge portal (see paper for the exact source and license).
2. You will receive 3D volumes (`*.nii` / `*.nii.gz`) and segmentation masks.

### Preprocessing (2D slices)

As described in the paper:

1. Convert 3D volumes to 2D PNG slices (axial direction). A reference script is provided:

   ```bash
   # Edit paths inside the script, then run:
   python dataset/LITS_3D_to_2D.py
   ```

2. Organize data in a flat or per-volume folder structure, e.g.:

   ```
   data/
   ├── 2D_img/          # e.g. volume-0/0.png, volume-0/1.png, ...
   └── 2D_label/        # matching masks
   ```

3. Build an index CSV (relative paths recommended):

   ```bash
   # Edit img_root / mask_root in dataset/create_dataset_csv.py
   python dataset/create_dataset_csv.py
   ```

   Template: `dataset/data_info.example.csv`.

### Index CSV (LiTS)

| Column | Description |
|--------|-------------|
| `image_path` | Path to 2D image (or `image_paths` for 2.5D `.npy` stacks) |
| `mask_path` | Path to mask |
| `case` | Volume / patient id (for grouped CV) |
| `tagging` | **Temporal prompt** — normalized slice index or relative position in volume (see paper) |
| `rle_len`, `empty` | Mask statistics for stratified split |
| `id` | Row id |

For **multimodal / TP** training, `multimodal_main_LITS.py` builds a text field automatically:

`describe = "This is a CT of the liver with a segmentation period of {tagging}."`

Set `use_multimodal_data: True` in the yaml for TP models.

### LiTS training settings (paper)

- Task: **binary** liver segmentation (`num_classes: 1` in yaml).
- Default input size: `192×192`.
- 5-fold **StratifiedGroupKFold** on `case` (implemented in `multimodal_main_LITS.py`).
- Entry script: **`multimodal_main_LITS.py`**.

---

## Dataset: UWMGI

### Acquisition

1. Download the **UW-Madison GI Tract Image Segmentation** dataset from Kaggle:

   ```bash
   pip install kaggle
   # Configure ~/.kaggle/kaggle.json with Your_Key (see Kaggle API docs)
   kaggle competitions download -c uw-madison-gi-tract-image-segmentation
   unzip uw-madison-gi-tract-image-segmentation.zip -d ./data/uwmgi_raw
   ```

2. Follow the **paper** for:
   - decoding RLE masks into PNG / multi-class labels;
   - constructing **2.5D** input stacks (`image_paths` / `mask_paths` as `.npy`);
   - assigning **`timestamp`** (temporal metadata for TP);
   - optional **`class`** text (large bowel / small bowel / stomach).

### Index CSV (UWMGI)

| Column | Description |
|--------|-------------|
| `image_path`, `mask_path` | 2D paths (used when `2.5D: False`) |
| `image_paths`, `mask_paths` | Stack paths (used when `2.5D: True`) |
| `timestamp` | Temporal metadata (mapped to `tagging` if `tagging` is absent) |
| `class` | Organ class → auto `describe` for multimodal training |
| `case`, `id`, `rle_len`, `empty` | Same role as LiTS |

Template: `dataset/uwmgi_data_info.example.csv`.

### UWMGI training settings (paper)

- Task: **3-class** GI tract segmentation (`num_classes: 3` in `config/multimodal_uwmgi_config/*.yaml`).
- Often trained with **`2.5D: True`** in yaml.
- Entry script: **`multimodal_main_UWMGI.py`**.
- Checkpoints: `model_weights_save/UWMGI/img_encoder_weights/`.

---

## Training on LiTS

```bash
cd /path/to/TP-UNet

# --- Monomodal baselines (no text encoder) ---
python multimodal_main_LITS.py \
  --cfg ./config/multimodal_config/skcdf.yaml \
  --csv ./dataset/your_lits_index.csv

python multimodal_main_LITS.py \
  --cfg ./config/multimodal_config/csc_pa.yaml \
  --csv ./dataset/your_lits_index.csv

# --- With Temporal Prompt (Electra on tagging) ---
python multimodal_main_LITS.py \
  --cfg ./config/multimodal_config/skcdf_tp.yaml \
  --csv ./dataset/your_lits_index.csv

python multimodal_main_LITS.py \
  --cfg ./config/multimodal_config/csc_pa_tp.yaml \
  --csv ./dataset/your_lits_index.csv

python multimodal_main_LITS.py \
  --cfg ./config/multimodal_config/electra.yaml \
  --csv ./dataset/your_lits_index.csv

# --- GPU ---
python multimodal_main_LITS.py --cfg ./config/multimodal_config/skcdf_tp.yaml --gpu 0
python multimodal_main_LITS.py --cfg ./config/multimodal_config/skcdf_tp.yaml --pick_gpu

# --- Debug (small subset) ---
python multimodal_main_LITS.py --cfg ./config/multimodal_config/skcdf.yaml --debug
```

**Outputs:** `model_weights_save/LITS/img_encoder_weights/{model_name}_fold{k}_best.pth`

Optional yaml keys: `cuda_visible_devices`, `path_rewrites` (list of `{from, to}` for legacy absolute paths).

---

## Training on UWMGI

```bash
cd /path/to/TP-UNet

# TP-UNet + CLIP text encoder
python multimodal_main_UWMGI.py \
  --cfg ./config/multimodal_uwmgi_config/clip.yaml \
  --csv ./dataset/your_uwmgi_index.csv \
  --gpu 0

# TP-UNet + Electra
python multimodal_main_UWMGI.py \
  --cfg ./config/multimodal_uwmgi_config/electra.yaml \
  --csv ./dataset/your_uwmgi_index.csv

# Other segmentation backbones (same dataloader / split)
python multimodal_main_UWMGI.py \
  --cfg ./config/multimodal_uwmgi_config/CrossUnet.yaml \
  --csv ./dataset/your_uwmgi_index.csv

python multimodal_main_UWMGI.py \
  --cfg ./config/multimodal_uwmgi_config/Unet.yaml \
  --csv ./dataset/your_uwmgi_index.csv
```

Ensure the yaml matches your setup:

| Key | Typical UWMGI value |
|-----|---------------------|
| `use_multimodal_data` | `True` |
| `2.5D` | `True` |
| `num_classes` | `3` |
| `img_size` | `[192, 192]` |

**Outputs:** `model_weights_save/UWMGI/img_encoder_weights/{model_name}_fold{k}_best.pth`

---

## Model configs

### LiTS — `config/multimodal_config/`

| File | `model_name` | TP |
|------|----------------|-----|
| `skcdf.yaml` | `SKCDF` | No |
| `skcdf_tp.yaml` | `SKCDF_TP` | Yes (Electra temporal only) |
| `csc_pa.yaml` | `CSC_PA` | No |
| `csc_pa_tp.yaml` | `CSC_PA_TP` | Yes |
| `electra.yaml` | `TP_UNet_Electra` | Yes |
| `clip.yaml` | `TP_UNet_Clip` | Yes |
| `CrossUnet.yaml`, `Unet.yaml`, … | Various | Per yaml |

### UWMGI — `config/multimodal_uwmgi_config/`

| File | Notes |
|------|--------|
| `electra.yaml` | TP-UNet + Electra, 3 classes, 2.5D |
| `clip.yaml` | TP-UNet + CLIP |
| `CrossUnet.yaml`, `TransUnet.yaml`, `Unet.yaml`, … | Baseline architectures |

Hyperparameters (`lr`, `epochs`, `train_bs`, …) should match the paper tables; edit yaml locally for ablations.

---

## Logging (Weights & Biases)

| Variable | Default / placeholder |
|----------|------------------------|
| `WANDB_API_KEY` | `Your_Key` in `.env.example` |
| `WANDB_PROJECT` | `TP-UNet` (LiTS) / `TP-UNet-UWMGI` |
| `WANDB_ENTITY` | `Your_Entity` |
| `WANDB_MODE` | `offline` (no key required) |

```bash
export WANDB_MODE=online   # only after setting a real API key locally
```

---

## Repository layout

```
TP-UNet/
├── multimodal_main_LITS.py      # LiTS training
├── multimodal_main_UWMGI.py     # UWMGI training
├── multimodal_training.py       # LiTS train/val loop
├── multimodal_training_uwmgi.py # UWMGI train/val loop
├── config/
│   ├── multimodal_config/       # LiTS experiments
│   └── multimodal_uwmgi_config/ # UWMGI experiments
├── model/
│   ├── SKCDF_2D.py, CSC_PA_2D.py
│   └── electra-small-discriminator/  # download separately
├── dataloader/
├── dataset/
│   ├── data_info.example.csv
│   ├── uwmgi_data_info.example.csv
│   ├── create_dataset_csv.py
│   └── LITS_3D_to_2D.py
├── scripts/gpu_pick.py
├── .env.example
└── requirements.txt
```

---

## Citation

If you use this code, please cite our paper (title and BibTeX in the camera-ready).

## License

Research use only. Third-party submodules keep their original licenses.
