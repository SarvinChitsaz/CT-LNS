# Data Folder

This folder contains the dataset used for **CT-LNS**.

## Structure

- `raw`  
  Raw data files:  
  - `.mhd` CT volumes  
  - `annotations.csv` containing nodule annotations

- `npy_preprocessed`  
  Preprocessed `.npy` files ready for training:  
  - `_img.npy` → preprocessed CT volume  
  - `_mask.npy` → corresponding nodule mask

## Dataset Source

The pipeline uses annotated 3D CT scans from the **LUNA16 subset5 dataset**:

- **Dataset page**: [LUNA16 Challenge](https://luna16.grand-challenge.org)  
- **CT volumes**: [subset5.zip](https://zenodo.org/records/3723295/files/subset5.zip?download=1)  
- **Annotations**: [annotations.csv](https://zenodo.org/records/3723295/files/annotations.csv?download=1)

## Notes

- To generate the preprocessed `.npy` files from `raw/`, run the preprocessing scripts in `src/utils.py`.  
- Ensure consistent folder paths in `config/config.yaml` before preprocessing.
