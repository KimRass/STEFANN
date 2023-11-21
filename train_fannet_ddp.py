# References:
    # https://github.com/pytorch/examples/blob/main/imagenet/main.py

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchmetrics.image import StructuralSimilarityIndexMeasure
import argparse
from pathlib import Path
import time
from tqdm import tqdm
import os

from utils import (
    ROOT,   
    get_config,
    get_elapsed_time,
    save_model,
    image_to_grid,
)
from data import FANnetDataset
from models.fannet import FANnet
from evaluate import evaluate


def get_args():
    parser = argparse.ArgumentParser()

    # parser.add_argument("--run_id", type=str, required=False)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--n_epochs", type=int, required=False, default=4)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--n_cpus", type=int, required=False, default=0)
    parser.add_argument("--ddp", action="store_true")

    parser.add_argument("--torch_compile", action="store_true", required=False)

    args = parser.parse_args()
    return args


def train_single_step(src_image, trg_image, trg_label, fannet, optim, scaler, crit, device):
    src_image = src_image.to(device)
    trg_image = trg_image.to(device)
    trg_label = trg_label.to(device)

    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
        enabled=True if device.type == "cuda" else False,
    ):
        pred = fannet(src_image, trg_label)
        loss = crit(pred, trg_image)
    optim.zero_grad()
    if device.type == "cuda":
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
    else:
        loss.backward()
        optim.step()
    return loss.item()


def main_worker(gpu, n_gpus_per_node, config):
    dist_url = "tcp://224.66.41.62:23456" # URL used to set up distributed training
    world_size = -1 # # of nodes
    rank = -1 # Node rank

    config["GPU"] = gpu

    if config["GPU"] is not None:
        print(f"""Using GPU: {config["GPU"]} for training""")

    if config["DISTRIBUTED"]:
        if dist_url == "env://" and rank == -1:
            rank = int(os.environ["RANK"])
        if config["MULTIPROCESSING_DISTRIBUTED"]:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            rank = rank * n_gpus_per_node + gpu
        dist.init_process_group(
            backend="nccl",
            init_method=dist_url,
            world_size=world_size,
            rank=rank,
        )

    train_ds = FANnetDataset(
        fannet_dir=config["DATA_DIR"], img_size=config["DATA"]["IMG_SIZE"], split="train",
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=config["BATCH_SIZE"],
        shuffle=True,
        num_workers=config["N_CPUS"],
        pin_memory=True,
        drop_last=True,
    )
    val_ds = FANnetDataset(
        fannet_dir=config["DATA_DIR"], img_size=config["DATA"]["IMG_SIZE"], split="valid",
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=config["BATCH_SIZE"],
        shuffle=False,
        num_workers=config["N_CPUS"],
        pin_memory=True,
        drop_last=True,
    )

    fannet = FANnet(dim=config["ARCHITECTURE"]["DIM"]).to(config["DEVICE"])
    if config["DDP"]:
        fannet = DDP(fannet)
        print(f"Using {torch.cuda.device_count()} GPUs")
    if config["TORCH_COMPILE"]:
        fannet = torch.compile(fannet)

    crit = nn.MSELoss(reduction="mean")
    metric = StructuralSimilarityIndexMeasure(reduction="sum").to(config["DEVICE"])

    optim = Adam(
        fannet.parameters(),
        lr=config["LR"],
        betas=(config["ADAM"]["BETA1"], config["ADAM"]["BETA2"]),
        eps=config["ADAM"]["EPS"],
    )

    scaler = GradScaler(enabled=True if config["DEVICE"].type == "cuda" else False)

    SAVE_DIR = ROOT/"samples_during_training"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    max_avg_ssim = 0
    prev_save_path = Path(".pth")
    for epoch in range(1, config["N_EPOCHS"] + 1):
        cum_loss = 0
        start_time = time.time()
        for src_image, _, trg_image, trg_label in tqdm(train_dl, desc=f"Epoch {epoch}", leave=False):
            loss = train_single_step(
                src_image=src_image,
                trg_image=trg_image,
                trg_label=trg_label,
                fannet=fannet,
                optim=optim,
                scaler=scaler,
                crit=crit,
                device=config["DEVICE"],
            )
            cum_loss += loss
        train_loss = cum_loss / len(train_dl)

        avg_ssim = evaluate(dl=val_dl, fannet=fannet, metric=metric, device=config["DEVICE"])
        if avg_ssim > max_avg_ssim:
            max_avg_ssim = avg_ssim

            cur_save_path = config["CKPTS_DIR"]/f"fannet_epoch_{epoch}.pth"
            save_model(model=fannet, save_path=cur_save_path)
            if prev_save_path.exists():
                prev_save_path.unlink()
            prev_save_path = cur_save_path

        msg = f"[ {get_elapsed_time(start_time)} ]"
        msg += f"""[ {epoch}/{config["N_EPOCHS"]} ]"""
        msg += f"[ Train loss: {train_loss:.4f} ]"
        msg += f"[ Avg. val. SSIM: {avg_ssim:.4f} ]"
        msg += f"[ Min. avg. val. SSIM: {max_avg_ssim:.4f} ]"
        print(msg)

        src_image = src_image[: 64].to(config["DEVICE"])
        trg_label = trg_label[: 64].to(config["DEVICE"])

        pred = fannet(src_image, trg_label)
        pred_image = image_to_grid(pred, n_cols=8)
        pred_image.save(SAVE_DIR/f"epoch_pred_{epoch}.jpg")

        trg_image = image_to_grid(trg_image[:, 64], n_cols=8)
        trg_image.save(SAVE_DIR/f"epoch_gt_{epoch}.jpg")



def main():
    args = get_args()
    CONFIG = get_config(
        config_path=ROOT/"configs/fannet.yaml", args=args,
    )

    if torch.cuda.is_available():
        n_gpus_per_node = torch.cuda.device_count()
    else:
        n_gpus_per_node = 1
    world_size *= world_size
    if CONFIG["DDP"]:
        mp.spawn(main_worker, nprocs=n_gpus_per_node, args=(n_gpus_per_node, args))
    else:
        main_worker(gpu=CONFIG["GPU"], n_gpus_per_node=n_gpus_per_node, config=CONFIG)


if __name__ == "__main__":
    main()
