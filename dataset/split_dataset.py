from pathlib import Path
import random
import argparse
import shutil

from utils import get_config, VAL_TEST_SIZE, ROOT


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--src_fannet_dir", type=str, required=True)

    args = parser.parse_args()
    return args


def copy_dirs(src_fannet_dir, test_ratio=0.2):
    src_fannet_dir = Path(src_fannet_dir)

    test_size = round(VAL_TEST_SIZE * test_ratio)

    src_train_dir = src_fannet_dir/"train"
    src_val_test_dir = src_fannet_dir/"valid"

    trg_dir = ROOT/"dataset/fannet"
    trg_train_dir = trg_dir/"train"
    trg_val_dir = trg_dir/"val"
    trg_test_dir = trg_dir/"test"

    if not trg_train_dir.exists():
        shutil.copytree(src_train_dir, trg_train_dir)

    if not trg_val_dir.exists() and not trg_test_dir.exists():
        val_test_ls = list(src_val_test_dir.glob("*"))
        test_ls = random.sample(val_test_ls, test_size)
        for dir in val_test_ls:
            if dir in test_ls:
                shutil.copytree(dir, trg_test_dir/dir.name)
            else:
                shutil.copytree(dir, trg_val_dir/dir.name)


if __name__ == "__main__":
    args = get_args()
    CONFIG = get_config(
        config_path=ROOT/"configs/fannet.yaml", args=args,
    )

    copy_dirs(CONFIG["SRC_FANNET_DIR"])
