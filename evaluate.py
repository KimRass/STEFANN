# References:
    # https://github.com/tzm-tora/Stroke-Based-Scene-Text-Erasing/blob/main/src/evaluate.py
    # https://cvnote.ddlee.cc/2019/09/12/psnr-ssim-python

from torch.utils.data import DataLoader
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
from skimage import metrics
import torch
import cv2
import argparse

from utils import get_config, ROOT, FANNET_DIR, image_to_grid
from data import FANnetDataset
from models.fannet import FANnet


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--n_cpus", type=int, required=False, default=0)

    args = parser.parse_args()
    return args


def get_ssim(x, y):
    ssim = round(
        metrics.structural_similarity(y, x, data_range=255, channel_axis=2), 4,
    )
    return ssim


if __name__ == "__main__":
    args = get_args()
    CONFIG = get_config(
        config_path=ROOT/"configs/fannet.yaml", args=args,
    )
    fannet = FANnet(dim=CONFIG["ARCHITECTURE"]["DIM"]).to(CONFIG["DEVICE"])
    state_dict = torch.load("/Users/jongbeomkim/Documents/fannet/fannet_epoch_3.pth", map_location=CONFIG["DEVICE"])
    fannet.load_state_dict(state_dict, strict=True)

    test_ds = FANnetDataset(fannet_dir=FANNET_DIR, split="test")
    test_dl = DataLoader(
        test_ds,
        batch_size=CONFIG["BATCH_SIZE"],
        shuffle=False,
        num_workers=CONFIG["N_CPUS"],
        pin_memory=True,
        drop_last=True,
    )

    for src_image, src_label, trg_image, trg_label in test_dl:
        src_image = src_image.to(CONFIG["DEVICE"])
        src_label = src_label.to(CONFIG["DEVICE"])
        trg_image = trg_image.to(CONFIG["DEVICE"])
        trg_label = trg_label.to(CONFIG["DEVICE"])

        pred = fannet(src_image, trg_label)
        pred_image = image_to_grid(pred, n_cols=CONFIG["BATCH_SIZE"])
        gt_image = image_to_grid(trg_image, n_cols=CONFIG["BATCH_SIZE"])

        ssim = get_ssim(np.array(pred_image), np.array(gt_image))
        # print(ssim)
        # print(torch.unique(src_label))
        print(src_label)
