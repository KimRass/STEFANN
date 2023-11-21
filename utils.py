import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF
from datetime import timedelta
from time import time
from PIL import Image
from pathlib import Path
import yaml
from collections import OrderedDict
import random
import numpy as np
import os
from copy import deepcopy
import re

ROOT = Path(__file__).resolve().parent
FANNET_DIR = ROOT/"dataset/fannet"

# CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
CHARS = "0123456789"
ASCII2IDX = {ord(char): idx for idx, char in enumerate(CHARS)}
IDX2ASCII = {v: k for k, v in ASCII2IDX.items()}
N_CLASSES = len(ASCII2IDX.values())

TRAIN_SIZE = 1015
VAL_TEST_SIZE = 300


def ascii_to_index(ascii):
    return ASCII2IDX[ascii]


def to_one_hot(gt):
    return F.one_hot(torch.tensor(gt), num_classes=N_CLASSES).float()


def _args_to_config(args, config):
    copied = deepcopy(config)
    for k, v in vars(args).items():
        copied[k.upper()] = v
    return copied


def get_config(config_path, args=None):
    config = load_config(config_path)
    if args is not None:
        config = _args_to_config(args=args, config=config)

    config["PARENT_DIR"] = Path(__file__).resolve().parent
    config["CKPTS_DIR"] = config["PARENT_DIR"]/"checkpoints"
    config["WANDB_CKPT_PATH"] = config["CKPTS_DIR"]/"checkpoint.tar"

    config["DEVICE"] = get_device()

    set_seed(config["SEED"])
    return config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def load_config(yaml_path):
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    return config


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def get_elapsed_time(start_time):
    return timedelta(seconds=round(time() - start_time))


def modify_state_dict(state_dict, keyword="_orig_mod."):
    new_state_dict = OrderedDict()
    for old_key in list(state_dict.keys()):
        if old_key and old_key.startswith(keyword):
            new_key = old_key[len(keyword):]
        else:
            new_key = old_key
        new_state_dict[new_key] = state_dict[old_key]
    return new_state_dict


def save_model(model, save_path):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(modify_state_dict(model.state_dict()), str(save_path))


def denorm_(tensor):
    tensor *= 0.5
    tensor += 0.5


def image_to_grid(image, n_cols=0):
    if n_cols == 0:
        n_cols = int(image.shape[0] ** 0.5)
    tensor = image.clone().detach().cpu()
    denorm_(tensor)
    grid = make_grid(tensor, nrow=n_cols, padding=1, pad_value=1)
    grid = TF.to_pil_image(grid)
    return grid


def modify_state_dict(state_dict, pattern=r"^module.|^_orig_mod."):
    new_state_dict = OrderedDict()
    for old_key, value in state_dict.items():
        new_key = re.sub(pattern=pattern, repl="", string=old_key)
        new_state_dict[new_key] = value
    return new_state_dict
