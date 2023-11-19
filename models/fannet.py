# References:
    # https://github.com/prasunroy/stefann/blob/master/fannet.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import N_CLASSES

# "Our generative font adaptive neural network (FANnet) takes two different inputs – an image
# of the source character of size 64 × 64 and a one-hot encoding $v$ of length 26 of the target character."
class FANnet(nn.Module):
    def __init__(self, dim=512):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(dim * 8, dim)

        self.label_embed = nn.Embedding(N_CLASSES, dim)
        self.fc3 = nn.Linear(dim * 2, dim * 2)
        self.fc4 = nn.Linear(dim * 2, dim * 2)

        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(16, 1, kernel_size=3, padding=1)

    def forward(self, x, y):
        # "The input image passes through three convolution layers having 16, 16 and 1 filters respectively,
        # followed by flattening and a fully-connected (FC) layer 'FC1'."
        # The outputs of 'FC1' and 'FC2' give 512 dimensional latent representations of respective inputs.
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x = torch.relu(x)
        x = x.view(-1, 64 * 64)
        x = self.fc1(x)
        x = torch.relu(x)

        # "The encoded vector $v$ also passes through an FC layer 'FC2'."
        # The outputs of 'FC1' and 'FC2' give 512 dimensional latent representations of respective inputs.
        y = self.label_embed(y)
        x = torch.relu(x)

        # Outputs of 'FC1' and 'FC2' are concatenated and followed by two more FC layers, 'FC3' and 'FC4'
        # having 1024 neurons each."
        x = torch.cat([x, y], dim=1)
        x = self.fc3(x)
        x = torch.relu(x)
        x = F.dropout(x, p=0.5)
        x = self.fc4(x)
        x = torch.relu(x)

        # The expanding part of the network contains reshaping to a dimension 8 × 8 × 16 followed by three
        # 'up-conv' layers having 16, 16 and 1 filters respectively. Each 'up-conv' layer contains
        # an upsampling followed by a 2D convolution. All the convolution layers have kernel size 3 × 3
        # and ReLU activation."
        x = x.view(-1, 16, 8, 8)
        x = F.interpolate(x, scale_factor=2)
        x = self.conv4(x)
        x = torch.relu(x)
        x = F.interpolate(x, scale_factor=2)
        x = self.conv5(x)
        x = torch.relu(x)
        x = F.interpolate(x, scale_factor=2)
        x = self.conv6(x)
        x = torch.tanh(x)
        return x


if __name__ == "__main__":
    fannet = FANnet()
    x = torch.randn(4, 1, 64, 64)
    # y = torch.randn(4, N_CLASSES)
    y = torch.randint(0, 62, (4, ))
    out = fannet(x, y)
    # src_image.min(), src_image.max()
    out.min(), out.max()
    x.shape, out.shape
