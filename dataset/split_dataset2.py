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


def split_dataset(src_fannet_dir, test_to_val_ratio=0.2):
    src_fannet_dir = "/Users/jongbeomkim/Downloads/fannet_new"
    save_dir = Path("/Users/jongbeomkim/Downloads/fannet_new_splitted")
    src_fannet_dir = Path(src_fannet_dir)

    train_save_dir = save_dir/"train"
    val_save_dir = save_dir/"val"

    train_save_dir.mkdir(parents=True, exist_ok=True)
    val_save_dir.mkdir(parents=True, exist_ok=True)

    train_val_ls = list(src_fannet_dir.glob("*"))
    val_size = round(len(train_val_ls) * 0.1)
    val_ls = random.sample(train_val_ls, val_size)
    for subdir in train_val_ls:
        if subdir in val_ls:
            shutil.copytree(subdir, val_save_dir/subdir.name)
        else:
            shutil.copytree(subdir, train_save_dir/subdir.name)



    test_size = round(VAL_TEST_SIZE * test_to_val_ratio)

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

    split_dataset(
        src_fannet_dir=CONFIG["SRC_FANNET_DIR"],
        test_to_val_ratio=CONFIG["DATA"]["TEST_TO_VAL_RATIO"],
    )
