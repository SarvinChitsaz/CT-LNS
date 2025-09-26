import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import NpyPatchDataset
from model import UNet3D
from losses import combined_loss, dice_coeff, nodule_acc
from torch.utils.data import DataLoader, random_split

def run_epoch(loader, model, optimizer, device, phase = "Train"):
    model.train() if phase == "Train" else model.eval()
    epoch_loss, epoch_dice, epoch_acc = 0, 0, 0
    n_batches = len(loader)
    with torch.set_grad_enabled(phase == "Train"):
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = combined_loss(outputs, targets)
            if phase == "Train":
                loss.backward()
                optimizer.step()
            epoch_loss += loss.item()
            epoch_dice += dice_coeff(outputs, targets).item()
            epoch_acc += nodule_acc(outputs, targets).item()
    return epoch_loss / n_batches, epoch_dice / n_batches, epoch_acc / n_batches
