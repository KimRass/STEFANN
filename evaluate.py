# References:
    # https://github.com/tzm-tora/Stroke-Based-Scene-Text-Erasing/blob/main/src/evaluate.py
    # https://cvnote.ddlee.cc/2019/09/12/psnr-ssim-python

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from pathlib import Path
from skimage import metrics
import torch
import cv2


def _get_ssim(x, y):
    ssim = round(
        metrics.structural_similarity(y, x, data_range=255, channel_axis=2), 4,
    )
    return ssim
