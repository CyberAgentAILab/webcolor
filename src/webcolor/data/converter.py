from pathlib import Path
from typing import Optional

import numpy as np
import torch

DIM_COLOR = 2  # RGB bin index + alpha bin index
DIM_TEXT = 12
DIM_IMAGE = 13

# valid tags from https://developer.mozilla.org/en-US/docs/Web/HTML/Element
HTML_TAGS = Path(__file__).with_name("html_tags.txt").read_text().split("\n")
TAG_TO_IDX = dict((tag, idx) for idx, tag in enumerate(HTML_TAGS))
IDX_TO_TAG = dict((idx, tag) for idx, tag in enumerate(HTML_TAGS))


def discretize_to_eight_bins(x: int) -> int:
    """Discretize a value from 0 to 255 into 8 bins and return the bin index."""
    min_value = 0
    max_value = 255
    bin_size = 32
    # bins = array([ 32,  64,  96, 128, 160, 192, 224, 256])
    bins = np.arange(min_value, max_value, bin_size) + bin_size
    idx = int(np.digitize(x, bins, right=False))
    return idx


def convert_color(color_str: Optional[str]) -> torch.Tensor:
    """Convert a color string into a feature vector consisting of RGB and alpha bin indices."""
    # TODO: Convert for color upsampler.
    if color_str is None:
        return torch.zeros(DIM_COLOR, dtype=torch.long)
    elif color_str.startswith("rgb"):
        start = color_str.find("(") + 1
        values = [eval(v) for v in color_str[start:-1].split(", ")]
        if color_str.startswith("rgb("):
            values.append(1)

        # discretize RGB value
        rgb = values[:3]
        octal = "0o"
        for x in rgb:
            idx = discretize_to_eight_bins(x)
            octal += str(idx)
        idx_rgb = eval(octal)

        # discretize alpha value
        alpha = values[3]
        assert 0.0 <= alpha <= 1.0
        idx_alpha = discretize_to_eight_bins(round(alpha * 255))

        return torch.tensor([idx_rgb, idx_alpha], dtype=torch.long)
    else:
        raise NotImplementedError(color_str)


def convert_order(sibling_order: int) -> torch.Tensor:
    """Convert a sibling order to a tensor."""
    return torch.tensor(sibling_order, dtype=torch.long)


def convert_tag(html_tag: str) -> torch.Tensor:
    """Convert an HTML tag to a tensor."""
    idx = TAG_TO_IDX.get(html_tag, len(TAG_TO_IDX))
    return torch.tensor(idx, dtype=torch.long)


def convert_text(text_feat: Optional[np.ndarray]) -> torch.Tensor:
    """Convert the input text feature to a tensor or return an all-zeros tensor if None."""
    # TODO: Put the actual conversion process somewhere.
    if text_feat is None:
        return torch.zeros(DIM_TEXT, dtype=torch.float)
    else:
        return torch.asarray(text_feat, dtype=torch.float)


def convert_image(img_feat: Optional[np.ndarray]) -> torch.Tensor:
    """Convert the input image feature to a tensor or return an all-zeros tensor if None."""
    # TODO: Put the actual conversion process somewhere.
    if img_feat is None:
        return torch.zeros(DIM_IMAGE, dtype=torch.float)
    else:
        return torch.asarray(img_feat, dtype=torch.float)
