import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF
import argparse
from pathlib import Path
import time
import math
from tqdm import tqdm

from utils import get_config, get_elapsed_time, save_model, ROOT, FANNET_DIR
from data import FANnetDataset
from models.fannet import FANnet


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--n_cpus", type=int, required=False, default=0)

    args = parser.parse_args()
    return args


def denorm(tensor):
    tensor /= 2
    tensor += 0.5
    return tensor


def image_to_grid(image, n_cols=0):
    if n_cols == 0:
        n_cols = int(image.shape[0] ** 0.5)
    tensor = image.clone().detach().cpu()
    tensor = denorm(tensor)
    grid = make_grid(tensor, nrow=n_cols, padding=1, pad_value=1)
    grid.clamp_(0, 1)
    grid = TF.to_pil_image(grid)
    return grid


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

    for src_image, trg_image, one_hot in test_dl:
        src_image = src_image.to(CONFIG["DEVICE"])
        trg_image = trg_image.to(CONFIG["DEVICE"])
        one_hot = one_hot.to(CONFIG["DEVICE"])

        pred = fannet(src_image, one_hot)
        pred_image = image_to_grid(pred)
        pred_image.show()
        gt_image = image_to_grid(trg_image)
        gt_image.show()
        break
