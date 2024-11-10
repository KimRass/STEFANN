from PIL import Image, ImageFont, ImageDraw
from pathlib import Path
from tqdm import tqdm
import argparse

from utils import CHARS


def draw_char_on_canvas(img_size, font_size, font_path):
    font = ImageFont.truetype(font=str(font_path), size=font_size)
    canvas = Image.new(mode="L", size=(img_size, img_size), color="rgb(0, 0, 0)")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        xy=(img_size // 2, round(img_size * 0.7)),
        text=char,
        fill="rgb(255, 255, 255)",
        font=font,
        align="center",
        anchor="ms",
        direction="ltr",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ofl_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--img_size", type=int, required=False, default=64)
    parser.add_argument("--font_size_ratio", type=float, required=False, default=0.9)
    args = parser.parse_args()

    font_size = round(args.img_size * args.font_size_ratio)
    for font_path in tqdm(sorted(list(Path(args.ofl_dir).glob("**/*.ttf")))):
        imgs = []
        for char in CHARS:
            try:
                img = draw_char_on_canvas(
                    img_size=args.img_size, font_size=font_size, font_path=font_path,
                )   
                imgs.append((char, img))
            except OSError:
                break
        else:
            for char, img in imgs:
                save_path = Path(args.save_dir)/f"{font_path.stem}/{ord(char)}.jpg"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                img.save(save_path)
