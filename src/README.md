# Source Code 

This folder contains the main source code for **CT-LNS**, including data handling, model architecture, loss functions, training, evaluation, and visualization.

## Files Overview

- **`dataset.py`**  
  Defines the `NpyPatchDataset` class for loading 3D patch-based data.  
  - Extracts 3D patches from volumes  
  - Optional data augmentation (random flips)  
  - Supports oversampling for nodule balancing  

- **`model.py`**  
  Implements the 3D U-Net architecture.  
  - `DoubleConv`: Two consecutive 3D convolutions with BatchNorm and ReLU  
  - `UNet3D`: Full 3D U-Net with encoder-decoder structure and skip connections  

- **`losses.py`**  
  Implements loss functions and evaluation metrics for segmentation.  
  - `dice_loss`: Dice loss  
  - `combined_loss`: Cross-entropy + Dice loss  
  - `dice_coeff`: Dice coefficient  
  - `nodule_acc`: Nodule voxel accuracy  

- **`utils.py`**  
  Helper functions for preprocessing and data handling.  
  - `resample_image`: Resample CT/mask images to a new spacing  
  - `create_mask_from_annotations`: Generate 3D masks from annotation CSV  
  - `preprocess_and_save`: Preprocess `.mhd` files and save as `.npy`  

- **`train.py`**  
  Training routines.  
  - `run_epoch`: Run a single epoch (train or validation)  
  - Handles mixed precision and optimizer updates  
  - Updates learning rate scheduler  

- **eval.py**  
  Evaluation scripts.  
  - Load trained models from `Models/checkpoints`  
  - Evaluate using Dice coefficient and nodule accuracy  

- **visualize.py**  
  Visualization utilities.  
  - `visualize_ct_nodule`: Display CT slices with ground truth and predicted masks  
