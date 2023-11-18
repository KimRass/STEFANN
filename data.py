# Source: https://drive.google.com/drive/folders/1dOl4_yk2x-LTHwgKBykxHQpmqDvqlkab
# References:
    # https://github.com/prasunroy/stefann/blob/master/fannet.py

# "I is an image that has multiple text regions, and is the domain of a text region that requires modification. The region can be selected using any text detection algorithm [5, 20, 36]. Alternatively, a user can select the corner points of a polygon that bounds a word to define $\Ohm$."
# ". After selecting the text region, we apply the MSER algorithm [8] to detect the binary masks of individual characters present in the region . However, MSER alone cannot generate a sharp mask for most of the"
# "characters. Thus, we calculate the final binarized image Ic defined as Ic(p) = ( IM(p) J IB(p) if p 2 0 otherwise where IM is the binarized output of the MSER algorithm [8] when applied on I, IB is the binarized image of I and J denotes the element-wise product of matrices. The image Ic contains the binarized characters in the selected region . If the color of the source character is darker than its background, we apply inverse binarization on I to get IB."

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from itertools import product
from pathlib import Path
from PIL import Image

from utils import ascii_to_index, to_one_hot


class FANnetDataset(Dataset):
    def __init__(self, fannet_dir, split):
        super().__init__()

        self.img_path_pairs = list()
        for font_dir in (Path(fannet_dir)/split).glob("*"):
            img_paths = sorted(list(font_dir.glob("*.jpg")))
            self.img_path_pairs.extend(list(product(img_paths, img_paths)))

        self.transformer = T.Compose(
            [T.ToTensor(), T.Normalize(mean=0.5, std=0.5)],
        )

    def __len__(self):
        return len(self.img_path_pairs)

    def __getitem__(self, idx):
        src_img_path, trg_img_path = self.img_path_pairs[idx]
        src_image = Image.open(src_img_path).convert(mode="L")
        trg_image = Image.open(trg_img_path).convert(mode="L")
        src_image = self.transformer(src_image)
        trg_image = self.transformer(trg_image)

        ascii = int(trg_img_path.stem)
        one_hot = to_one_hot(ascii_to_index(ascii))
        return src_image, trg_image, one_hot


if __name__ == "__main__":
    fannet_dir = "/Users/jongbeomkim/Desktop/workspace/STEFANN/dataset/fannet"
    split = "test"
    ds = FANnetDataset(fannet_dir=fannet_dir, split=split)
    src_image, trg_image, one_hot = ds[20001]
    # src_image.show(), trg_image.show()
