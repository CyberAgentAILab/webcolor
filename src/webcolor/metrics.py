import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.image.fid import FrechetInceptionDistance

from webcolor.data.converter import NUM_COLOR_BINS


class FrechetColorDistance(FrechetInceptionDistance):
    def __init__(
        self,
        reset_real_features: bool = False,
    ) -> None:
        super().__init__(
            feature=ColorFeatureExtractor(),
            reset_real_features=reset_real_features,
            normalize=False,
        )


class ColorFeatureExtractor(nn.Module):
    def __init__(self, ignore_idx: int = -1) -> None:
        super().__init__()
        self.num_features = NUM_COLOR_BINS**3
        self.ignore_idx = ignore_idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.size()) == 2:
            x = x.clone()
            x[x == self.ignore_idx] = self.num_features
            freq = F.one_hot(x, self.num_features + 1)[:, :, :-1]
            hist = freq.sum(dim=1) / freq.sum(dim=(1, 2)).unsqueeze(-1)
            return hist.nan_to_num()  # type: ignore
        elif len(x.size()) == 4:
            # TODO: pixel-level FCD
            return torch.zeros(self.num_features)
        else:
            raise NotImplementedError(x.size())
