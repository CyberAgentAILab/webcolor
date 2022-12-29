from typing import Dict

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from webcolor.data.converter import HTML_TAGS, NUM_COLOR_BINS
from webcolor.models.base import BaseGenerator, BaseModel


class Stats(BaseGenerator):  # type: ignore
    freq_text_rgb: torch.Tensor
    freq_text_alpha: torch.Tensor
    freq_bg_rgb: torch.Tensor
    freq_bg_alpha: torch.Tensor

    def __init__(self, sampling: bool) -> None:
        super(BaseModel, self).__init__()
        self.sampling = sampling

        nb = NUM_COLOR_BINS
        freq_rgb = torch.zeros(len(HTML_TAGS) + 1, nb**3).long()
        freq_alpha = torch.zeros(len(HTML_TAGS) + 1, nb).long()

        self.register_buffer("freq_text_rgb", freq_rgb)
        self.register_buffer("freq_text_alpha", freq_alpha)
        self.register_buffer("freq_bg_rgb", freq_rgb.clone())
        self.register_buffer("freq_bg_alpha", freq_alpha.clone())

        # To avoid ValueError in initializing the optimizer
        self._dummy = nn.Parameter(torch.rand(1))

    def forward(
        self,
        g: dgl.DGLGraph,
        batch_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        tag = g.ndata["tag"].unsqueeze(1)
        text_mask = g.ndata["has_text"]

        # record frequencies of rgb bins
        D = self.freq_text_rgb.size(1)
        _tag = tag.expand(-1, D)
        one_hot = F.one_hot(g.ndata["text_color"][:, 0], D)
        self.freq_text_rgb.scatter_add_(0, _tag[text_mask], one_hot[text_mask])
        one_hot = F.one_hot(g.ndata["bg_color"][:, 0], D)
        self.freq_bg_rgb.scatter_add_(0, _tag, one_hot)

        # record frequencies of alpha bins
        D = self.freq_text_alpha.size(1)
        _tag = tag.expand(-1, D)
        one_hot = F.one_hot(g.ndata["text_color"][:, 1], D)
        self.freq_text_alpha.scatter_add_(0, _tag[text_mask], one_hot[text_mask])
        one_hot = F.one_hot(g.ndata["bg_color"][:, 1], D)
        self.freq_bg_alpha.scatter_add_(0, _tag, one_hot)

        return dict(dummy_loss=self._dummy)

    def generate(
        self,
        g: dgl.DGLGraph,
        batch_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        tag = g.ndata["tag"].unsqueeze(1)
        self.fill_zero_freq()

        # format rgb frequencies
        D = self.freq_text_rgb.size(1)
        _tag = tag.expand(-1, D)
        freq_text_rgb = self.freq_text_rgb.gather(0, _tag).float()
        freq_bg_rgb = self.freq_bg_rgb.gather(0, _tag).float()

        # format alpha frequencies
        D = self.freq_text_alpha.size(1)
        _tag = tag.expand(-1, D)
        freq_text_alpha = self.freq_text_alpha.gather(0, _tag).float()
        freq_bg_alpha = self.freq_bg_alpha.gather(0, _tag).float()

        if self.sampling:
            pred_text_rgb = torch.multinomial(freq_text_rgb, 1).squeeze(1)
            pred_text_alpha = torch.multinomial(freq_text_alpha, 1).squeeze(1)
            pred_bg_rgb = torch.multinomial(freq_bg_rgb, 1).squeeze(1)
            pred_bg_alpha = torch.multinomial(freq_bg_alpha, 1).squeeze(1)
        else:
            pred_text_rgb = freq_text_rgb.argmax(1)
            pred_text_alpha = freq_text_alpha.argmax(1)
            pred_bg_rgb = freq_bg_rgb.argmax(1)
            pred_bg_alpha = freq_bg_alpha.argmax(1)

        return {
            "pred_text_rgb": pred_text_rgb,
            "pred_text_alpha": pred_text_alpha,
            "pred_bg_rgb": pred_bg_rgb,
            "pred_bg_alpha": pred_bg_alpha,
        }

    def fill_zero_freq(self) -> None:
        def _fill_zero_freq(freq: torch.Tensor) -> None:
            mask = freq.sum(1).eq(0)
            freq_global = torch.sum(freq[~mask], dim=0, keepdim=True)
            freq[mask] = freq_global.expand_as(freq[mask])

        _fill_zero_freq(self.freq_text_rgb)
        _fill_zero_freq(self.freq_text_alpha)
        _fill_zero_freq(self.freq_bg_rgb)
        _fill_zero_freq(self.freq_bg_alpha)
