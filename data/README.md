# Data Folder

This folder contains the dataset used for CT-LNS.

## Structure

- `raw`  
  Raw data files:
  - `.mhd` CT volumes  
  - `annotations.csv` containing nodule annotations

- `npy_preprocessed`  
  Preprocessed `.npy` files ready for training:
  - `_img.npy` → preprocessed CT volume  
  - `_mask.npy` → corresponding nodule mask

## Notes

- To generate the preprocessed `.npy` files from `raw/`, run the preprocessing scripts in `src/utils.py`.
- Ensure consistent folder paths in `config/config.yaml` before preprocessing.
