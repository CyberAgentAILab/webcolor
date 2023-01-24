from io import BytesIO
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from cairosvg import svg2png
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def convert_text(
    text_list: List[str],
    has_text_from_pseudo: bool,
) -> Optional[np.ndarray]:
    num_text = len(text_list)
    if num_text == 0:
        return None

    text = " ".join(text_list)
    num_words = len(text.split())
    methods = [
        "isalpha",
        "isascii",
        "isdecimal",
        "islower",
        "isnumeric",
        "isprintable",
        "isspace",
        "istitle",
        "isupper",
    ]
    flags = [getattr(text, method)() for method in methods]
    flags.append(has_text_from_pseudo)

    feat = np.asarray([num_text, num_words] + flags, dtype=np.float32)

    return feat


def convert_image(img_path: Union[str, Path]) -> Optional[np.ndarray]:
    """You need to install an extra package with `poetry install -E image`."""

    img_path = Path(img_path)
    if not img_path.exists():
        return None

    is_svg = img_path.suffix == ".svg"
    if is_svg:
        try:
            try:
                b_png = svg2png(url=str(img_path))
            except ValueError:
                # ValueError: The SVG size is undefined
                b_png = svg2png(url=str(img_path), output_width=300, output_height=150)
            img = Image.open(BytesIO(b_png))
        except Exception:
            return None
    else:
        try:
            img = Image.open(img_path)
        except Exception:
            return None

    try:
        img.load()
    except OSError:
        # OSError: image file is truncated (0 bytes not processed)
        return None

    W, H = img.size
    C = len(img.getbands())
    ratio = float(W) / H
    feat = np.asarray([is_svg, W, H, C, ratio], dtype=np.float32)

    if C != 4:
        img = img.convert("RGBA")

    img = np.asarray(img) / 255.0
    rgba_mean = img.mean((0, 1))
    rgba_std = img.std((0, 1))

    feat = np.concatenate([feat, rgba_mean, rgba_std], dtype=np.float32)

    return feat
