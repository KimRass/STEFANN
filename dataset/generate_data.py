from PIL import Image, ImageFont, ImageDraw
from pathlib import Path
from tqdm import tqdm
import argparse

from utils import CHARS


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ofl_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)

    args = parser.parse_args()
    return args


def _get_canvas(img_size):
    return Image.new(mode="L", size=(img_size, img_size), color="rgb(0, 0, 0)")


if __name__ == "__main__":
    args = get_args()

    IMG_SIZE = 128
    FONT_SIZE = IMG_SIZE * 0.9
    for font_path in tqdm(sorted(list(Path(args.ofl_dir).glob("**/*.ttf")))):
        for char in CHARS:
            try:
                font = ImageFont.truetype(font=str(font_path), size=FONT_SIZE)
                canvas = _get_canvas(IMG_SIZE)
                draw = ImageDraw.Draw(canvas)
                draw.text(
                    xy=(IMG_SIZE // 2, round(IMG_SIZE * 0.7)),
                    text=char,
                    fill="rgb(255, 255, 255)",
                    font=font,
                    align="center",
                    anchor="ms",
                    direction="ltr",
                )
            except OSError:
                break

            save_path = Path(args.save_dir)/f"{font_path.stem}/{ord(char)}.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(save_path)
