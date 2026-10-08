from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image, ImageFilter

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(PROJECT_DIR, "ui", "assets")
DEFAULT_SOURCE = os.path.join(ASSETS, "app_icon_source.jpg")

SIZE = 1024
SUPERSAMPLE = 4
SQUIRCLE_EXPONENT = 5.0
MAC_BODY = 824


def squircle_mask(size: int) -> Image.Image:
    big = size * SUPERSAMPLE
    coords = (np.arange(big, dtype=np.float64) + 0.5) / big * 2.0 - 1.0
    x = np.abs(coords)[None, :]
    y = np.abs(coords)[:, None]
    inside = (x ** SQUIRCLE_EXPONENT + y ** SQUIRCLE_EXPONENT) <= 1.0
    mask = Image.fromarray((inside * 255).astype(np.uint8), "L")
    return mask.resize((size, size), Image.LANCZOS)


def squircle_icon(source: Image.Image, size: int) -> Image.Image:
    art = source.convert("RGB").resize((size, size), Image.LANCZOS)
    icon = art.convert("RGBA")
    icon.putalpha(squircle_mask(size))
    return icon


def mac_icon(source: Image.Image) -> Image.Image:
    body = squircle_icon(source, MAC_BODY)
    canvas = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    offset = (SIZE - MAC_BODY) // 2

    shadow_alpha = Image.new("L", (SIZE, SIZE), 0)
    shadow_alpha.paste(body.getchannel("A").point(lambda a: a * 0.35), (offset, offset + 12))
    shadow_alpha = shadow_alpha.filter(ImageFilter.GaussianBlur(14))
    shadow = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 255))
    shadow.putalpha(shadow_alpha)

    canvas.alpha_composite(shadow)
    canvas.alpha_composite(body, (offset, offset))
    return canvas


def main() -> None:
    src_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SOURCE
    with Image.open(src_path) as src:
        w, h = src.size
        side = min(w, h)
        source = src.crop(((w - side) // 2, (h - side) // 2, (w + side) // 2, (h + side) // 2)).copy()

    os.makedirs(os.path.join(ASSETS, "mac"), exist_ok=True)

    full = squircle_icon(source, SIZE)
    full.save(os.path.join(ASSETS, "app_icon.png"))

    ico_sizes = [16, 24, 32, 48, 64, 128, 256]
    frames = [squircle_icon(source, s) for s in ico_sizes]
    frames[-1].save(os.path.join(ASSETS, "app_icon.ico"), format="ICO",
                    sizes=[(s, s) for s in ico_sizes], append_images=frames[:-1])

    mac_icon(source).save(os.path.join(ASSETS, "mac", "app_icon.png"))
    print("Wrote ui/assets/app_icon.png, ui/assets/app_icon.ico, ui/assets/mac/app_icon.png")


if __name__ == "__main__":
    main()
