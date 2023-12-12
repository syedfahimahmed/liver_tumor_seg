import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        print("g shape: ", g.shape)
        g1 = self.W_g(g)
        print("g1 shape: ", g1.shape)
        x1 = self.W_x(x)
        print("x1 shape: ", x1.shape)
        psi = self.relu(g1 + x1)
        print("psi shape: ", psi.shape)
        psi = self.psi(psi)
        print("psi shape after sigmoid: ", psi.shape)
        return x * psi

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Adjusted to take 768 channels as input
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv with attention gate"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        
        # Adjust the attention gate to have F_int as half of F_l
        self.attention_gate = AttentionGate(F_g=in_channels // 2, F_l=out_channels, F_int=out_channels // 2)
        
        # After concatenation, the number of channels will be (in_channels // 2 + out_channels // 2)
        # Conv layer should then reduce it to the final desired out_channels
        if in_channels == 1024:
            self.conv = DoubleConv(1024, out_channels)
            
        elif in_channels == 512:
            self.conv = DoubleConv(512, out_channels)
        
        elif in_channels == 256:
            self.conv = DoubleConv(256, out_channels)
        
        elif in_channels == 128:
            self.conv = DoubleConv(128, out_channels)
        
        elif in_channels == 64:
            self.conv = DoubleConv(64, out_channels)

    def forward(self, x1, x2):
        print("x1 shape: ", x1.shape)
        x1 = self.up(x1)  # Upsample or transpose conv x1
        print("x1 shape after upsample: ", x1.shape)
        x2 = self.attention_gate(x2, x1)  # Apply attention gate
        print("x2 shape after attention gate: ", x2.shape)
        # Concatenate along the channels dimension
        x = torch.cat([x1, x2], dim=1)
        print("x shape after concatenation: ", x.shape)
        # Convolve to get back to the desired number of channels
        x = self.conv(x)
        print("x shape after conv: ", x.shape)
        return x

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        print("x4 shape: ", x4.shape)
        x5 = self.down4(x4)
        print("x5 shape: ", x5.shape)
        x = self.up1(x5, x4)
        print("x shape after up1: ", x.shape)
        x = self.up2(x, x3)
        print("x shape after up2: ", x.shape)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits