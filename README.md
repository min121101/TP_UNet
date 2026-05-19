# TP-UNet (anonymous release)

2D liver CT segmentation with optional **temporal prompt (TP)** via Electra-encoded temporal metadata. This repository is prepared for **double-blind review**: no credentials, no machine-specific paths, and no experiment logs are included.

## Models

| Config | `model_name` | Description |
|--------|--------------|-------------|
| `config/multimodal_config/skcdf.yaml` | `SKCDF` | 2D SKCDF baseline (no TP) |
| `config/multimodal_config/skcdf_tp.yaml` | `SKCDF_TP` | SKCDF + temporal prompt |
| `config/multimodal_config/csc_pa.yaml` | `CSC_PA` | 2D CSC-PA (DeepLabV3+ + prototype correlation) |
| `config/multimodal_config/csc_pa_tp.yaml` | `CSC_PA_TP` | CSC-PA + temporal prompt |
| `config/multimodal_config/electra.yaml` | `TP_UNet_Electra` | TP-UNet with Electra |

## Setup

```bash
conda create -n tpunet python=3.10 -y
conda activate tpunet
pip install -r requirements.txt
```

Copy secrets template (do **not** commit `.env`):

```bash
cp .env.example .env
# Edit .env: set WANDB_API_KEY=Your_Key (or your real key locally only)
```

Download **LiTS** 2D slices and build a CSV with columns `image_path`, `mask_path`, `tagging`, `case`, `rle_len`, etc. See `dataset/data_info.example.csv` and `dataset/create_dataset_csv.py`.

Place Electra weights under `model/electra-small-discriminator/` (Hugging Face: `google/electra-small-discriminator`).

## Training (LiTS)

```bash
cd /path/to/TP-UNet

# Baseline (example config)
python multimodal_main_LITS.py --cfg ./config/multimodal_config/skcdf.yaml --csv ./dataset/your_data_index.csv

# With temporal prompt
python multimodal_main_LITS.py --cfg ./config/multimodal_config/skcdf_tp.yaml --csv ./dataset/your_data_index.csv

# GPU selection (optional)
python multimodal_main_LITS.py --cfg ./config/multimodal_config/csc_pa_tp.yaml --gpu 0
python multimodal_main_LITS.py --cfg ./config/multimodal_config/csc_pa_tp.yaml --pick_gpu
```

Checkpoints are written to `model_weights_save/LITS/img_encoder_weights/` (gitignored).

## Weights & Biases

- Default mode: **offline** (`WANDB_MODE=offline`).
- Set `WANDB_API_KEY=Your_Key` in `.env` until you replace it with a real key on your machine.
- Optional: `WANDB_PROJECT`, `WANDB_ENTITY` in `.env` or yaml (`wandb_project`, `wandb_entity`).

## Repository layout

```
TP-UNet/
├── multimodal_main_LITS.py    # Main training entry (LiTS)
├── multimodal_training.py
├── config/multimodal_config/  # Experiment yaml files
├── model/                     # Networks (SKCDF_2D, CSC_PA_2D, UNet, …)
├── dataloader/
├── loss_function/
├── dataset/                   # CSV builders + example index
└── scripts/gpu_pick.py
```

## Citation

If you use this code, please cite the accompanying paper (details in the camera-ready version).

## License

Research use only. Third-party modules retain their original licenses (see subfolders under `model/`).
