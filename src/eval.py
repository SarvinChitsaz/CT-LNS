import os
import numpy as np
import matplotlib.pyplot as plt
import random

def visualize_ct_nodule(npy_folder, pred_thresh = 0.3):
    def overlay_mask(ct_slice, mask_slice, color = 'red', alpha = 0.4):
        ct_norm = (ct_slice - ct_slice.min()) / (ct_slice.max() - ct_slice.min())
        ct_rgb = np.stack([ct_norm] * 3, axis = -1)
        mask_rgb = np.zeros_like(ct_rgb)
        mask_float = mask_slice.astype(np.float32)
        if color == 'red': mask_rgb[...,0] = mask_float
        elif color == 'green': mask_rgb[...,1] = mask_float
        elif color == 'blue': mask_rgb[...,2] = mask_float
        return np.clip((1 - alpha) * ct_rgb + alpha * mask_rgb, 0, 1)

    volume_files = [f for f in sorted(os.listdir(npy_folder)) if f.endswith("_mask.npy")]
    volumes_with_nodule = [f for f in volume_files if np.load(os.path.join(npy_folder, f)).sum() > 0]
    f = random.choice(volumes_with_nodule)
    mask = np.load(os.path.join(npy_folder, f))
    img = np.load(os.path.join(npy_folder, f.replace("_mask.npy", "_img.npy")))
    slices_with_nodule = np.where(mask.sum(axis = (1,2)) > 0)[0]
    slice_idx = random.choice(slices_with_nodule)
    pred_prob = mask.astype(np.float32)
    pred_mask = pred_prob > pred_thresh
    fig, axes = plt.subplots(1, 3, figsize = (18,6))
    axes[0].imshow(img[slice_idx], cmap = 'gray')
    axes[0].set_title("CT Slice"); axes[0].axis('off')
    overlay_gt = overlay_mask(img[slice_idx], mask[slice_idx], color = 'red', alpha = 0.4)
    axes[1].imshow(overlay_gt); axes[1].set_title("Ground Truth Mask"); axes[1].axis('off')
    overlay_pred = overlay_mask(img[slice_idx], pred_mask[slice_idx], color = 'blue', alpha = 0.4)
    axes[2].imshow(overlay_pred); axes[2].set_title(f"Predicted Mask (thresh = {pred_thresh})"); axes[2].axis('off')
    plt.tight_layout()
    plt.show()
