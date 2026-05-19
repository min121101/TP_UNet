# Dataset preparation

1. Obtain the **LiTS** challenge data and export 2D slices (see `LITS_3D_to_2D.py`).
2. Build an index CSV with at least: `image_path`, `mask_path`, `case`, `tagging`, `rle_len`, `empty`.
3. Use **relative paths** from the repo root, e.g. `./data/2D_img/volume-0/0.png`.
4. Copy `data_info.example.csv` as a template or run `create_dataset_csv.py` after editing paths inside that script.

The full index file is **not** shipped with this release (`dataset/*.csv` is gitignored except `data_info.example.csv`).

Optional yaml `path_rewrites` (list of `{from, to}`) can rewrite legacy absolute paths when loading a CSV.
