# Checkpoints

This folder is intended to store trained 3D U-Net model weights (.pth files).

**Tip:** After training, you can save models here for later evaluation:

```python
import torch
from src.model import UNet3D

model = UNet3D()
torch.save(model.state_dict(), "Models/checkpoints/unet_epoch_X.pth")
model.load_state_dict(torch.load("Models/checkpoints/unet_epoch_X.pth"))
model.eval()
