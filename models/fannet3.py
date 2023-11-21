# References:
    # https://github.com/KimRass/Pix2Pix/blob/main/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        upsample=False,
        drop=True,
        normalize=True,
        leaky=False,
    ):
        super().__init__()

        self.stride = stride
        self.upsample = upsample
        self.drop = drop
        self.normalize = normalize
        self.leaky = leaky

        if not upsample:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=stride,
                padding=1,
                bias=False if normalize else True,
            )
        else:
            self.conv = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=stride,
                padding=1,
                bias=False if normalize else True,
            )
        if normalize:
            self.norm = nn.InstanceNorm2d(out_channels, affine=False, track_running_stats=False)
        if drop:
            self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        if self.normalize:
            x = self.norm(x)
        if self.drop:
            x = self.dropout(x)
        if self.leaky:
            x = F.leaky_relu(x, 0.2)
        else:
            x = torch.relu(x)
        return x


def _init_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            m.weight.data.normal_(0, 0.02)


class CustomFANnet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, dim=64):
        super().__init__()

        self.layer1 = ConvBlock(
            in_channels, dim, upsample=False, drop=False, normalize=False, leaky=True,
        )
        self.layer2 = ConvBlock(dim, dim * 2, upsample=False, drop=False, normalize=True, leaky=True)
        self.layer3 = ConvBlock(dim * 2, dim * 4, upsample=False, drop=False, normalize=True, leaky=True)
        self.layer4 = ConvBlock(dim * 4, dim * 8, upsample=False, drop=False, normalize=True, leaky=True)
        self.layer5 = ConvBlock(dim * 8, dim * 8, upsample=False, drop=False, normalize=True, leaky=True)
        self.layer6 = ConvBlock(dim * 8, dim * 8, upsample=False, drop=False, normalize=False, leaky=True)

        self.layer7 = ConvBlock(dim * 8, dim * 8, upsample=True, drop=True, normalize=True, leaky=False)
        self.layer8 = ConvBlock(dim * 16, dim * 8, upsample=True, drop=True, normalize=True, leaky=False)
        self.layer9 = ConvBlock(dim * 16, dim * 4, upsample=True, drop=True, normalize=True, leaky=False)
        self.layer10 = ConvBlock(dim * 8, dim * 2, upsample=True, drop=False, normalize=True, leaky=False)
        self.layer11 = ConvBlock(dim * 4, dim, upsample=True, drop=False, normalize=True, leaky=False)
        self.layer12 = ConvBlock(dim * 2, dim * 2, upsample=True, drop=False, normalize=False, leaky=False)
        self.layer13 = nn.ConvTranspose2d(dim * 2, out_channels, kernel_size=4, stride=2, padding=1)

        _init_weights(self)

    def forward(self, x):
        x1 = self.layer1(x) # (b, 64, 32, 32)
        x2 = self.layer2(x1) # (b, 128, 16, 16)
        x3 = self.layer3(x2) # (b, 256, 8, 8)
        x4 = self.layer4(x3) # (b, 512, 4, 4)
        x5 = self.layer5(x4) # (b, 512, 2, 2)
        x6 = self.layer6(x5) # (b, 512, 1, 1)

        x = self.layer7(x6) # (b, 512, 2, 2)
        x = self.layer8(torch.cat([x5, x], dim=1)) # (b, 512, 4, 4)
        x = self.layer9(torch.cat([x4, x], dim=1)) # (b, 256, 8, 8)
        x = self.layer10(torch.cat([x3, x], dim=1)) # (b, 128, 16, 16)
        x = self.layer11(torch.cat([x2, x], dim=1)) # (b, 64, 32, 32)
        x = self.layer13(torch.cat([x1, x], dim=1)) # (b, 1, 64, 64)
        x = torch.tanh(x)
        return x


if __name__ == "__main__":
    fannet = CustomFANnet()
    # x = torch.randn(4, 1, 256, 256)
    x = torch.randn(4, 1, 64, 64)
    out = fannet(x)
    # out.shape
