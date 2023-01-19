from typing import Any, Dict

import dgl
import torch
import torch.nn as nn

from webcolor.data.dataset import MAX_NODE_SIZE
from webcolor.models.base import BaseGenerator
from webcolor.models.utils import PositionalEncoding, to_dense_batch


class NARTransformer(BaseGenerator):  # type: ignore
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        norm_first: bool,
        **kwargs: Any,
    ):
        super().__init__(
            d_model=d_model,
            has_style_encoder=False,
            **kwargs,
        )

        # Transformer modules
        self.transformer = nn.TransformerEncoder(  # type: ignore
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                batch_first=True,
                norm_first=norm_first,
            ),
            num_layers=num_layers,
        )
        self.positional_encoding = PositionalEncoding(d_model, max_len=MAX_NODE_SIZE)

    def forward(
        self,
        g: dgl.DGLGraph,
        batch_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # encode
        x_con = self.encode_content(g)

        # make an input sequence for Transformer encoder
        src = self.positional_encoding(to_dense_batch(x_con, batch_mask))

        # make a causal mask for Transformer encoder
        src_key_padding_mask = ~batch_mask

        # Transformer encoder
        x = self.transformer(src, src_key_padding_mask=src_key_padding_mask)

        # revert to original batch format
        x = x[batch_mask]

        # decode
        out: Dict[str, torch.Tensor] = self.decode_style(x)

        return out

    def generate(
        self,
        g: dgl.DGLGraph,
        batch_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # compute logit
        key_to_logit = self(g, batch_mask)

        # convert logit to prediction
        out = {
            key.replace("logit", "pred"): logit.argmax(-1)
            for key, logit in key_to_logit.items()
        }

        return out
