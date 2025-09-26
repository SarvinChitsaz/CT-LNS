import torch
from torch.utils.data import DataLoader, random_split
from src.utils import preprocess_and_save
from src.dataset import NpyPatchDataset
from src.model import UNet3D
from src.train import run_epoch
from src.visualize import visualize_ct_nodule
from torch import optim
import os

mhd_folder = "data/raw"
annotations_file = "data/raw/annotations.csv"
npy_folder = "data/npy_preprocessed"
os.makedirs(npy_folder, exist_ok = True)

preprocess_and_save(mhd_folder, annotations_file, npy_folder)

dataset = NpyPatchDataset(npy_folder, patch_size = (96,96,96), augment = True)
val_ratio = 0.2
val_size = int(len(dataset) * val_ratio)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size = 2, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = 2, shuffle = False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet3D().to(device)
optimizer = optim.Adam(model.parameters(), lr = 1e-4)

num_epochs = 5
for epoch in range(1, num_epochs + 1):
    train_loss, train_dice, train_acc = run_epoch(train_loader, model, optimizer, device, phase = "Train")
    val_loss, val_dice, val_acc = run_epoch(val_loader, model, optimizer, device, phase = "Val")
    print(f"Epoch {epoch}: Train Dice = {train_dice:.4f}, Val Dice = {val_dice:.4f}")

visualize_ct_nodule(npy_folder, pred_thresh = 0.3)
