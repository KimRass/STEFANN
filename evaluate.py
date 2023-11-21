# References:
    # https://github.com/tzm-tora/Stroke-Based-Scene-Text-Erasing/blob/main/src/evaluate.py
    # https://cvnote.ddlee.cc/2019/09/12/psnr-ssim-python

import numpy as np
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

from utils import get_config, ROOT, N_CLASSES
from data import FANnetDataset
from models.fannet import FANnet


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
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


@torch.no_grad()
def evaluate(dl, fannet, metric, device):
    fannet.eval()

    cum_ssim = 0
    for src_image, _, trg_image, trg_label in tqdm(dl, desc=f"Validating...", leave=False):
        src_image = src_image.to(device)
        trg_image = trg_image.to(device)
        trg_label = trg_label.to(device)

        pred = fannet(src_image.detach(), trg_label.detach())
        ssim = metric(pred, trg_image)
        cum_ssim += ssim
    avg_ssim = cum_ssim / (len(dl) * dl.batch_size)

    fannet.train()
    return avg_ssim.item()


if __name__ == "__main__":
    args = get_args()
    CONFIG = get_config(
        config_path=ROOT/"configs/fannet.yaml", args=args,
    )
    fannet = FANnet(dim=CONFIG["ARCHITECTURE"]["DIM"]).to(CONFIG["DEVICE"])
    state_dict = torch.load(CONFIG["CKPT_PATH"], map_location=CONFIG["DEVICE"])
    fannet.load_state_dict(state_dict, strict=True)

    val_ds = FANnetDataset(fannet_dir=CONFIG["DATA_DIR"], split="valid")
    val_dl = DataLoader(
        val_ds,
        batch_size=N_CLASSES,
        shuffle=False,
        num_workers=CONFIG["N_CPUS"],
        pin_memory=False,
        drop_last=True,
    )

    metric = StructuralSimilarityIndexMeasure(reduction="sum").to(CONFIG["DEVICE"])
    avg_ssim = evaluate(dl=val_dl, fannet=fannet, metric=metric, device=CONFIG["DEVICE"])
    print(avg_ssim)

    # cum_ssim = torch.empty(size=(N_CLASSES,))
    # cnt = torch.ones(size=(N_CLASSES,))
    # with torch.no_grad():
    #     for src_image, src_label, trg_image, trg_label in tqdm(val_dl, position=0):
    #         src_image = src_image.to(CONFIG["DEVICE"])
    #         src_label = src_label.to(CONFIG["DEVICE"])
    #         trg_image = trg_image.to(CONFIG["DEVICE"])
    #         trg_label = trg_label.to(CONFIG["DEVICE"])

    #         pred = fannet(src_image.detach(), trg_label.detach())

    #         ssim = metric(pred, trg_image)
    #         row = torch.unique(src_label).item()
    #         cum_ssim[row] += ssim
    #         cnt[row] += 1

    # avg_by_src = cum_ssim / (cnt * N_CLASSES)
    # print(avg_by_src)
    # plt.plot(avg_by_src)
    # plt.show()
