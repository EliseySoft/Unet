import torch
import torch.nn as nn
from torch.nn import Conv2d, BatchNorm2d, ReLU, MaxPool2d, ConvTranspose2d

from unet_blocks import DoubleConv, Down, Up


class Unet(nn.Module):
    def __init__(self, in_channels, number_classes) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.number_classes = number_classes

        # BLOCKS
        self.start = DoubleConv(in_channels=self.in_channels, out_channels=64)

        self.down1 = Down(in_channels=64, out_channels=128)
        self.down2 = Down(in_channels=128, out_channels=256)
        self.down3 = Down(in_channels=256, out_channels=512)
        self.down4 = Down(in_channels=512, out_channels=1024)

        self.up1 = Up(in_channels=1024, out_channels=512)
        self.up2 = Up(in_channels=512, out_channels=256)
        self.up3 = Up(in_channels=256, out_channels=128)
        self.up4 = Up(in_channels=128, out_channels=64)

        self.final = Conv2d(in_channels=64, out_channels=self.number_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # encoder stage
        # print("ENCODER STAGE...")
        x1 = self.start(x)  # [b, 64, 568, 568]
        # print(x1.shape)
        x2 = self.down1(x1) # [b, 128, 280, 280]
        # print(x2.shape)
        x3 = self.down2(x2) # [b, 256, 136, 136]
        # print(x3.shape)
        x4 = self.down3(x3) # [b, 512, 64, 64]
        # print(x4.shape)
        x5 = self.down4(x4) # [b, 1024, 28, 28]
        # print(x5.shape, end="\n\n")

        # decoder stage
        # print("DECODER STAGE...")
        x = self.up1(x5, x4) # [b, 512, 52, 52]
        # print(x.shape)
        x = self.up2(x, x3) # [b, 256, 100, 100]
        # print(x.shape)
        x = self.up3(x, x2) # [b, 128, 196, 196]
        # print(x.shape)
        x = self.up4(x, x1) # [b, 64, 388, 388]
        # print(x.shape)

        # final layer
        x = self.final(x)
        # print(F"FINAL SHAPE: {x.shape}")
        
        return self.sigmoid(x)


if __name__ == "__main__":
    model = Unet(in_channels = 3, number_classes=2)
    input = torch.randn([8, 3, 572, 572])
    output = model(input)
