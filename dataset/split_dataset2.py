from pathlib import Path
import random
import argparse
import shutil
from tqdm import tqdm

from utils import get_config, VAL_TEST_SIZE, ROOT


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--src_fannet_dir", type=str, required=True)

    args = parser.parse_args()
    return args


def split_dataset(src_fannet_dir, val_ratio=0.1):
    src_fannet_dir = Path(src_fannet_dir)
    save_dir = ROOT/"dataset/fannet_new"

    train_save_dir = save_dir/"train"
    val_save_dir = save_dir/"val"

    train_save_dir.mkdir(parents=True, exist_ok=True)
    val_save_dir.mkdir(parents=True, exist_ok=True)

    train_val_ls = list(src_fannet_dir.glob("*"))
    val_size = round(len(train_val_ls) * val_ratio)
    val_ls = random.sample(train_val_ls, val_size)
    for subdir in tqdm(train_val_ls):
        if subdir in val_ls:
            shutil.copytree(subdir, val_save_dir/subdir.name)
        else:
            shutil.copytree(subdir, train_save_dir/subdir.name)


if __name__ == "__main__":
    args = get_args()
    CONFIG = get_config(
        config_path=ROOT/"configs/fannet.yaml", args=args,
    )

    split_dataset(
        src_fannet_dir=CONFIG["SRC_FANNET_DIR"],
        val_ratio=0.1,
    )
