import math
from typing import Union

import dgl
import torch
import torch.nn as nn


def to_dense_batch(
    x: torch.Tensor,
    batch_mask: torch.Tensor,
    fill_value: Union[int, float] = 0,
) -> torch.Tensor:
    """This function is similar to PyG's ``to_dense_batch``."""
    B, M = batch_mask.size()
    if len(x.size()) == 1:
        size = [B, M]
    elif len(x.size()) == 2:
        D = x.size(-1)
        size = [B, M, D]
    else:
        raise NotImplementedError
    out = x.new_full(size, fill_value)
    out[batch_mask] = x
    return out


def make_batch_mask(g: dgl.DGLGraph) -> torch.Tensor:
    num_nodes = g.batch_num_nodes()
    _arange = torch.arange(num_nodes.max(), device=g.device)
    batch_mask: torch.Tensor = _arange.unsqueeze(0) < num_nodes.unsqueeze(-1)
    return batch_mask


class PositionalEncoding(nn.Module):
    """Based on https://pytorch.org/tutorials/beginner/transformer_tutorial.html"""

    pe: torch.Tensor

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)  # type: ignore
