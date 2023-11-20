from PIL import Image, ImageFont, ImageDraw
from pathlib import Path
import numpy as np
from tqdm import tqdm

from utils import CHARS


def _get_canvas(img_size):
    return Image.new(mode="L", size=(img_size, img_size), color="rgb(0, 0, 0)")


CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
# CHARS = "PpJj"


ofl_dir = "/Users/jongbeomkim/Documents/fonts-main/ofl"
ofl_dir = Path(ofl_dir)
save_dir = "/Users/jongbeomkim/Downloads/fannet_new"
save_dir = Path(save_dir)

img_size = 64
font_size = 50
for font_path in tqdm(sorted(list(ofl_dir.glob("**/*.ttf")))):
    for char in CHARS:
        font = ImageFont.truetype(font=str(font_path), size=font_size)
        canvas = _get_canvas(img_size)
        draw = ImageDraw.Draw(canvas)
        try:
            draw.text(
                xy=(img_size // 2, round(img_size * 0.7)),
                text=char,
                fill="rgb(255, 255, 255)",
                font=font,
                align="center",
                anchor="ms",
                direction="ltr",
            )
        except OSError:
            break

        save_path = save_dir/f"{font_path.stem}/{ord(char)}.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(save_path)
