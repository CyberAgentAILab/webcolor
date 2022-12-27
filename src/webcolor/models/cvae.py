from typing import Any, Dict, Tuple

import dgl
import torch
import torch.nn as nn

from webcolor.data.dataset import MAX_NODE_SIZE
from webcolor.models.base_generator import BaseGenerator
from webcolor.models.utils import PositionalEncoding, to_dense_batch


class CVAE(BaseGenerator):  # type: ignore
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
            has_style_encoder=True,
            **kwargs,
        )

        # Transformer modules
        self.fc_input = nn.Linear(d_model * 2, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            norm_first=norm_first,
        )
        self.positional_encoding = PositionalEncoding(d_model, max_len=MAX_NODE_SIZE)

        # VAE modules
        self.fc_vae = nn.Linear(d_model, d_model * 2)
        self.fc_mu = nn.Linear(d_model * 2, d_model)
        self.fc_logvar = nn.Linear(d_model * 2, d_model)

    def forward(
        self,
        g: dgl.DGLGraph,
        batch_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # encode
        x_con = self.encode_content(g)
        x_sty = self.encode_style(**g.ndata)
        mu, logvar = self.encode(x_con, x_sty, batch_mask)

        # reparameterization trick
        z = self.reparameterize(mu, logvar)

        # decode
        x = self.decode(x_con, z, batch_mask)
        out: Dict[str, torch.Tensor] = self.decode_style(x)

        out["mu"] = mu
        out["logvar"] = logvar

        return out

    def generate(
        self,
        g: dgl.DGLGraph,
        batch_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # encode
        x_con = self.encode_content(g)

        # sample latent vectors from a normal distribution
        z = torch.randn_like(x_con)

        # decode
        x = self.decode(x_con, z, batch_mask)
        out: Dict[str, torch.Tensor] = self.decode_style(x)

        return out

    def encode(
        self,
        x_con: torch.Tensor,
        x_sty: torch.Tensor,
        batch_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # make an input sequence for Transformer encoder
        x = self.fc_input(torch.cat([x_con, x_sty], dim=-1))
        src = self.positional_encoding(to_dense_batch(x, batch_mask))

        # make a causal mask for Transformer encoder
        src_key_padding_mask = ~batch_mask

        # Transformer encoder
        memory = self.transformer.encoder(
            src, src_key_padding_mask=src_key_padding_mask
        )

        # VAE ops
        x = self.fc_vae(memory[batch_mask])
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(
        self,
        x_con: torch.Tensor,
        z: torch.Tensor,
        batch_mask: torch.Tensor,
    ) -> torch.Tensor:
        # make input sequences for Transformer decoder
        tgt = self.positional_encoding(to_dense_batch(x_con, batch_mask))
        memory = self.positional_encoding(to_dense_batch(z, batch_mask))

        # make causal masks for Transformer decoder
        tgt_key_padding_mask = ~batch_mask
        memory_key_padding_mask = tgt_key_padding_mask

        # Transformer decoder
        x = self.transformer.decoder(
            tgt,
            memory,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        # revert to original batch format
        x = x[batch_mask]

        return x  # type:ignore
