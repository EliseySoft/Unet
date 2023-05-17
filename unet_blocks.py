import torch
import torch.nn as nn
from torch.nn import Conv2d, BatchNorm2d, ReLU, MaxPool2d, ConvTranspose2d


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        # Conv -> BatchNorm -> ReLU
        self.double_conv = nn.Sequential(
            Conv2d(in_channels, out_channels, kernel_size=3),
            BatchNorm2d(out_channels),
            ReLU(inplace=True),
            Conv2d(out_channels, out_channels, kernel_size=3),
            BatchNorm2d(out_channels),
            ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        # DownSample -> DoubleConv
        self.down = MaxPool2d(kernel_size=2, stride=2)
        self.double_conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x):
        x = self.down(x)
        x = self.double_conv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.double_conv = DoubleConv(in_channels=in_channels, out_channels=out_channels)

    @staticmethod
    def crop(inputs, new_shape: torch.Size):
        old_y, old_x = inputs.shape[2:]
        new_y, new_x = new_shape[2:]
        
        x1 = int(old_x / 2 - new_x / 2)
        y1 = int(old_y / 2 - new_y / 2)
        x2 = int(x1 + new_x)
        y2 = int(y1 + new_y)
        
        return inputs[:, :, y1:y2, x1: x2]

    def forward(self, x, encoder_features):
        x = self.up(x)
        encoder_features = self.crop(encoder_features, new_shape=x.shape)
        x = torch.cat((encoder_features, x), dim=1)
        x = self.double_conv(x)
        return x
