from collections import defaultdict
from typing import Any, Dict, List

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from webcolor.data.dataset import MAX_NODE_SIZE
from webcolor.models.base import BaseGenerator
from webcolor.models.utils import PositionalEncoding, to_dense_batch


class ARTransformer(BaseGenerator):  # type: ignore
    tgt_mask: torch.Tensor

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        norm_first: bool,
        top_k: int,
        top_p: float,
        **kwargs: Any,
    ):
        super().__init__(
            d_model=d_model,
            has_style_encoder=True,
            **kwargs,
        )

        # Transformer modules
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
        self.bos = nn.Parameter(torch.rand(1, 1, d_model))
        tgt_mask = self.transformer.generate_square_subsequent_mask(MAX_NODE_SIZE)
        self.register_buffer("tgt_mask", tgt_mask)

        # decoding parameters
        self.top_k = top_k
        self.top_p = top_p

    def forward(
        self,
        g: dgl.DGLGraph,
        batch_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # encode
        x_con = self.encode_content(g)
        x_sty = self.encode_style(**g.ndata)

        # make input sequences for Transformer
        x_con_dense = to_dense_batch(x_con, batch_mask)
        src = self.positional_encoding(x_con_dense)

        x_sty_dense = to_dense_batch(x_sty, batch_mask)
        bos = self.bos.expand(g.batch_size, -1, -1)
        x_sty_shifted = torch.cat([bos, x_sty_dense[:, :-1]], dim=1)
        tgt = self.positional_encoding(x_sty_shifted)

        # make causal masks for Transformer
        src_key_padding_mask = ~batch_mask
        tgt_key_padding_mask = src_key_padding_mask
        memory_key_padding_mask = src_key_padding_mask

        # Transformer
        L = tgt.size(1)
        x = self.transformer(
            src,
            tgt,
            tgt_mask=self.tgt_mask[:L, :L],
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

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
        # encode
        x_con = self.encode_content(g)

        # make causal masks for Transformer
        src_key_padding_mask = ~batch_mask
        tgt_key_padding_mask = src_key_padding_mask
        memory_key_padding_mask = src_key_padding_mask

        # Transformer encoder
        x_con_dense = to_dense_batch(x_con, batch_mask)
        src = self.positional_encoding(x_con_dense)
        memory = self.transformer.encoder(
            src, src_key_padding_mask=src_key_padding_mask
        )

        # autoregressive decoding
        L_max = batch_mask.size(1)
        tgt_wo_pos = self.bos.expand(g.batch_size, -1, -1)
        pred_all: Dict[str, List[torch.Tensor]] = defaultdict(list)
        for i in range(L_max):
            if i > 0:
                x_sty_dense = self._encode_last_prediction(g, batch_mask, pred_all)
                tgt_wo_pos = torch.cat([tgt_wo_pos, x_sty_dense], dim=1)
            tgt = self.positional_encoding(tgt_wo_pos)

            # Transformer decoder
            L = tgt.size(1)
            x = self.transformer.decoder(
                tgt,
                memory,
                tgt_mask=self.tgt_mask[:L, :L],
                tgt_key_padding_mask=tgt_key_padding_mask[:, :L],
                memory_key_padding_mask=memory_key_padding_mask,
            )

            # use only last prediction
            x = x[:, -1]

            # decode
            key_to_logit = self.decode_style(x)
            for key, logit in key_to_logit.items():
                key = key.replace("logit", "pred")
                pred = top_k_top_p_filtering(logit, self.top_k, self.top_p)
                pred_all[key].append(pred)

        # format prediction
        out = {
            key: torch.stack(pred_list, dim=1)[batch_mask]
            for key, pred_list in pred_all.items()
        }

        return out

    def _encode_last_prediction(
        self,
        g: dgl.DGLGraph,
        batch_mask: torch.Tensor,
        pred_all: Dict[str, List[torch.Tensor]],
    ) -> torch.Tensor:
        pred_text_rgb = pred_all["pred_text_rgb"][-1]
        pred_text_alpha = pred_all["pred_text_alpha"][-1]
        text_color = torch.stack([pred_text_rgb, pred_text_alpha], dim=1)

        pred_bg_rgb = pred_all["pred_bg_rgb"][-1]
        pred_bg_alpha = pred_all["pred_bg_alpha"][-1]
        bg_color = torch.stack([pred_bg_rgb, pred_bg_alpha], dim=1)

        L = len(pred_all["pred_text_rgb"])
        _batch_mask = batch_mask[:, L - 1]
        text_color = text_color[_batch_mask]
        bg_color = bg_color[_batch_mask]

        has_text = to_dense_batch(g.ndata["has_text"], batch_mask)
        has_text = has_text[:, L - 1][_batch_mask]

        x_sty = self.encode_style(text_color, bg_color, has_text)
        x_sty_dense = to_dense_batch(x_sty, _batch_mask.unsqueeze(1))

        return x_sty_dense  # type: ignore


# based on https://gist.github.com/bsantraigi/5752667525d88d375207f099bd78818b
def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 0.0,
    filter_value: float = -float("Inf"),
) -> torch.Tensor:
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (vocabulary size)
        top_k >0: keep only top k tokens with highest probability (top-k filtering).
        top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    Basic outline taken from https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 2  # [BATCH_SIZE, VOCAB_SIZE]
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k, dim=1)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)

    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove = sorted_indices_to_remove.roll(1, -1)
    sorted_indices_to_remove[..., 0] = False

    # Replace logits to be removed with -inf in the sorted_logits
    sorted_logits[sorted_indices_to_remove] = filter_value
    # Then reverse the sorting process by mapping back sorted_logits to their original position
    logits = torch.gather(sorted_logits, 1, sorted_indices.argsort(-1))

    pred_token = torch.multinomial(F.softmax(logits, -1), 1)  # [BATCH_SIZE, 1]
    return pred_token.squeeze(1)
