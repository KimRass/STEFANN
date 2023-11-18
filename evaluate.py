# Reference: https://github.com/tzm-tora/Stroke-Based-Scene-Text-Erasing/blob/main/src/evaluate.py

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
