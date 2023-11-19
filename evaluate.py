# References:
    # https://github.com/tzm-tora/Stroke-Based-Scene-Text-Erasing/blob/main/src/evaluate.py
    # https://cvnote.ddlee.cc/2019/09/12/psnr-ssim-python

import numpy as np
import cv2
from PIL import Image
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from pathlib import Path
from skimage import metrics
import torch
import cv2
import argparse
import matplotlib.pyplot as plt

from utils import get_config, ROOT, FANNET_DIR, image_to_grid, N_CLASSES, IDX2ASCII
from data import FANnetDataset
from models.fannet import FANnet


def get_args():
    parser = argparse.ArgumentParser()

    # parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--n_cpus", type=int, required=False, default=0)

    args = parser.parse_args()
    return args


def to_array(x):
    if isinstance(x, Image.Image):
        x = np.array(x)
    return x


def get_ssim(x, y):
    x = to_array(x)
    y = to_array(y)
    ssim = round(
        metrics.structural_similarity(y, x, data_range=255, channel_axis=2), 4,
    )
    return ssim


def get_ssim_using_pt(pred, gt, reduction="sum"):
    ssim = StructuralSimilarityIndexMeasure(data_range=2, reduction=reduction)
    return ssim(pred, gt)


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


if __name__ == "__main__":
    args = get_args()
    CONFIG = get_config(
        config_path=ROOT/"configs/fannet.yaml", args=args,
    )
    fannet = FANnet(dim=CONFIG["ARCHITECTURE"]["DIM"]).to(CONFIG["DEVICE"])
    state_dict = torch.load("/Users/jongbeomkim/Documents/fannet/fannet_epoch_5.pth", map_location=CONFIG["DEVICE"])
    fannet.load_state_dict(state_dict, strict=True)
    fannet.eval()

    test_ds = FANnetDataset(fannet_dir=FANNET_DIR, split="test")
    test_dl = DataLoader(
        test_ds,
        batch_size=N_CLASSES,
        shuffle=False,
        num_workers=CONFIG["N_CPUS"],
        pin_memory=True,
        drop_last=True,
    )

    cum_ssim = torch.empty(size=(N_CLASSES,))
    cnt = torch.ones(size=(N_CLASSES,))
    with torch.no_grad():
        for src_image, src_label, trg_image, trg_label in tqdm(test_dl):
            src_image = src_image.to(CONFIG["DEVICE"])
            src_label = src_label.to(CONFIG["DEVICE"])
            trg_image = trg_image.to(CONFIG["DEVICE"])
            trg_label = trg_label.to(CONFIG["DEVICE"])

            pred = fannet(src_image.detach(), trg_label.detach())

            ssim = get_ssim_using_pt(pred, trg_image)
            row = torch.unique(src_label).item()
            cum_ssim[row] += ssim
            cnt[row] += 1

    avg_by_src = cum_ssim / (cnt * N_CLASSES)
    print(avg_by_src)
    plt.plot(avg_by_src)
    plt.show()
