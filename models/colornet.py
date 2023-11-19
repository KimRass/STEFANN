# References:
    # https://github.com/prasunroy/stefann/blob/master/colornet.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import N_CLASSES


class Colornet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv11 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x, y):
        x = self.conv1(x)
        y = self.conv2(y)
        x = torch.cat([x, y], dim=1)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool1(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.maxpool2(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = F.interpolate(x, scale_factor=2)
        x = self.conv10(x)
        x = F.interpolate(x, scale_factor=2)
        x = self.conv11(x)
        x = self.conv12(x)
        # x = F.leaky_relu(x, negative_slope=0.2)
        return x


if __name__ == "__main__":
    colornet = Colornet()
    y = torch.randn(4, 3, 64, 64)
    x = torch.randn(4, 3, 64, 64)
    colornet(x, y).shape
