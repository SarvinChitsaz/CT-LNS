import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding = 1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace = True),
            nn.Conv3d(out_ch, out_ch, 3, padding = 1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace = True),
        )
    def forward(self, x):
        return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 2, base_ch = 16):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base_ch)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = DoubleConv(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = DoubleConv(base_ch * 2, base_ch * 4)
        self.up2 = nn.ConvTranspose3d(base_ch * 4, base_ch * 2, 2, stride = 2)
        self.dec2 = DoubleConv(base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose3d(base_ch * 2, base_ch, 2, stride = 2)
        self.dec1 = DoubleConv(base_ch * 2, base_ch)
        self.outc = nn.Conv3d(base_ch, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        d2 = self.up2(e3)
        d2 = torch.cat([d2, e2], dim = 1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim = 1)
        d1 = self.dec1(d1)
        return self.outc(d1)
