from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

NUM_COLOR_BINS = 8
DIM_COLOR = 2  # RGB bin index + alpha bin index
DIM_TEXT = 12
DIM_IMAGE = 13

# valid tags from https://developer.mozilla.org/en-US/docs/Web/HTML/Element
HTML_TAGS = Path(__file__).with_name("html_tags.txt").read_text().split("\n")
TAG_TO_IDX = dict((tag, idx) for idx, tag in enumerate(HTML_TAGS))
IDX_TO_TAG = dict((idx, tag) for idx, tag in enumerate(HTML_TAGS))


def discretize_to_eight_bins(x: int) -> Tuple[int, float]:
    """Discretize the value from 0 to 255 into 8 bins and return the bin index
    and the residual."""
    min_value = 0
    max_value = 255
    bin_size = 32
    # bins = array([ 32,  64,  96, 128, 160, 192, 224, 256])
    bins = np.arange(min_value, max_value, bin_size) + bin_size
    idx = int(np.digitize(x, bins, right=False))

    # compute the residual as a proportion in the bin
    res = x % bin_size / (bin_size - 1)

    return idx, res


def convert_color(color_str: Optional[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert the color string into two feature vectors: one representing the
    RGB and alpha bin indices and the other the color residuals."""
    if color_str is None:
        color_dis = torch.zeros(DIM_COLOR, dtype=torch.long)
        color_res = torch.zeros(len("rgba"), dtype=torch.float)

    elif color_str.startswith("rgb"):
        residuals = []
        start = color_str.find("(") + 1
        values = [eval(v) for v in color_str[start:-1].split(", ")]
        if color_str.startswith("rgb("):
            values.append(1)

        # discretize RGB value
        rgb = values[:3]
        octal = "0o"
        for x in rgb:
            idx, res = discretize_to_eight_bins(x)
            octal += str(idx)
            residuals.append(res)
        idx_rgb = eval(octal)

        # discretize alpha value
        alpha = values[3]
        assert 0.0 <= alpha <= 1.0
        idx_alpha, res = discretize_to_eight_bins(round(alpha * 255))
        residuals.append(res)

        color_dis = torch.tensor([idx_rgb, idx_alpha], dtype=torch.long)
        color_res = torch.tensor(residuals, dtype=torch.float)

    else:
        raise NotImplementedError(color_str)

    return color_dis, color_res


def revert_color(color_dis: torch.Tensor, color_res: torch.Tensor) -> str:
    """Revert the two feature vectors to a color string in RGBA format."""
    min_value = 0
    max_value = 255
    bin_size = 32
    # bins = tensor([  0,  32,  64,  96, 128, 160, 192, 224])
    bins = torch.arange(min_value, max_value, bin_size)

    idx_rgb, idx_a = color_dis.tolist()
    idx_r, idx_g, idx_b = [int(i) for i in format(idx_rgb, "03o")]
    rgba = bins[[idx_r, idx_g, idx_b, idx_a]] + (bin_size - 1) * color_res.cpu()
    rgb = rgba[:3].long().tolist()
    alpha = round(rgba[3].item() / max_value, 4)

    color_str = f"rgba{tuple(rgb) + (alpha,)}"

    return color_str


def convert_order(sibling_order: int) -> torch.Tensor:
    """Convert a sibling order to a tensor."""
    return torch.tensor(sibling_order, dtype=torch.long)


def convert_tag(html_tag: str) -> torch.Tensor:
    """Convert an HTML tag to a tensor."""
    idx = TAG_TO_IDX.get(html_tag, len(TAG_TO_IDX))
    return torch.tensor(idx, dtype=torch.long)


def convert_text(text_feat: Optional[np.ndarray]) -> torch.Tensor:
    """Convert the input text feature to a tensor or return an all-zeros tensor if None.
    The actual conversion process can be found in `converter_reference.py`."""
    if text_feat is None:
        return torch.zeros(DIM_TEXT, dtype=torch.float)
    else:
        return torch.asarray(text_feat, dtype=torch.float)


def convert_image(img_feat: Optional[np.ndarray]) -> torch.Tensor:
    """Convert the input image feature to a tensor or return an all-zeros tensor if None.
    The actual conversion process can be found in `converter_reference.py`."""
    if img_feat is None:
        return torch.zeros(DIM_IMAGE, dtype=torch.float)
    else:
        return torch.asarray(img_feat, dtype=torch.float)
