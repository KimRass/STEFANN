import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np

from utils import get_config, ROOT, FANNET_DIR, image_to_grid, modify_state_dict
from data import FANnetDataset
# from models.fannet import FANnet
from models.fannet2 import FANnet


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--n_cpus", type=int, required=False, default=0)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    CONFIG = get_config(
        config_path=ROOT/"configs/fannet.yaml", args=args,
    )
    fannet = FANnet(
        dim=CONFIG["ARCHITECTURE"]["DIM"], normalization=False,
    ).to(CONFIG["DEVICE"])
    state_dict = torch.load("/Users/jongbeomkim/Desktop/workspace/STEFANN/pretrained/instancenorm_0.5939.pth", map_location=CONFIG["DEVICE"])
    state_dict = modify_state_dict(state_dict)
    # fannet.load_state_dict(state_dict, strict=True)
    fannet.load_state_dict(state_dict, strict=False)

    test_ds = FANnetDataset(
        fannet_dir=FANNET_DIR, img_size=CONFIG["DATA"]["IMG_SIZE"], split="test",
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=CONFIG["BATCH_SIZE"],
        shuffle=False,
        num_workers=CONFIG["N_CPUS"],
        pin_memory=True,
        drop_last=True,
    )

    cnt = 0
    for src_image, src_label, trg_image, trg_label in test_dl:
        cnt += 1
        if cnt >= 2:
            break
        src_image = src_image.to(CONFIG["DEVICE"])
        src_label = src_label.to(CONFIG["DEVICE"])
        trg_image = trg_image.to(CONFIG["DEVICE"])
        trg_label = trg_label.to(CONFIG["DEVICE"])

        pred = fannet(src_image, trg_label)
        pred_image = image_to_grid(pred, n_cols=CONFIG["BATCH_SIZE"])
        gt_image = image_to_grid(trg_image, n_cols=CONFIG["BATCH_SIZE"])
        pred_image.show()
    gt_image.show()
