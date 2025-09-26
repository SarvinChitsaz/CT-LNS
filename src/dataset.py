import numpy as np
import torch
from torch.utils.data import Dataset
import os

class NpyPatchDataset(Dataset):
    def __init__(self, folder, patch_size = (96,96,96), min_nodule_ratio = 0.0, oversample = 1, augment = False):
        self.folder = folder
        self.patch_size = patch_size
        self.min_nodule_ratio = min_nodule_ratio
        self.oversample = oversample
        self.augment = augment
        self.files = [f for f in os.listdir(folder) if f.endswith("_img.npy")]

    def __len__(self):
        return len(self.files) * self.oversample

    def __getitem__(self, idx):
        file_idx = idx % len(self.files)
        base = self.files[file_idx].replace("_img.npy", "")
        img = np.load(os.path.join(self.folder, base + "_img.npy"))
        msk = np.load(os.path.join(self.folder, base + "_mask.npy"))
        ps = self.patch_size
        x = np.random.randint(0, max(1, img.shape[0] - ps[0]))
        y = np.random.randint(0, max(1, img.shape[1] - ps[1]))
        z = np.random.randint(0, max(1, img.shape[2] - ps[2]))
        img_patch = img[x : x + ps[0], y : y + ps[1], z : z + ps[2]]
        msk_patch = msk[x : x + ps[0], y : y + ps[1], z : z + ps[2]]
        if self.augment and np.random.rand() < 0.5:
            img_patch = np.flip(img_patch, axis = 0).copy()
            msk_patch = np.flip(msk_patch, axis = 0).copy()
        img_patch = torch.from_numpy(img_patch).unsqueeze(0).float()
        msk_patch = torch.from_numpy(msk_patch).long()
        return img_patch, msk_patch
