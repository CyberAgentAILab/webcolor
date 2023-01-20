from typing import Callable

import dgl
import dgl.function as fn
import dgl.udf
import torch
import torch.nn as nn

from webcolor.data.converter import DIM_IMAGE, DIM_TEXT, HTML_TAGS
from webcolor.data.dataset import MAX_NODE_SIZE


class ContentEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        disable_message_passing: bool,
        disable_residual: bool,
    ):
        super().__init__()

        # embed
        self.embed_order = nn.Embedding(MAX_NODE_SIZE, d_model)
        self.embed_tag = nn.Embedding(len(HTML_TAGS) + 1, d_model)  # +1 for <UNK>
        self.embed_text = nn.Linear(DIM_TEXT, d_model)
        self.embed_img = nn.Linear(DIM_IMAGE, d_model)
        self.embed_bgimg = nn.Linear(DIM_IMAGE, d_model)

        # bottom-up
        self.h_leaf = nn.Parameter(torch.rand(1, d_model))
        self.mlp_up = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # top-down
        self.h_root = nn.Parameter(torch.rand(1, d_model))
        self.mlp_down = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # ablation flags
        assert not disable_message_passing or not disable_residual
        self.disable_message_passing = disable_message_passing
        self.disable_residual = disable_residual

    def embed_all(self, g: dgl.DGLGraph) -> torch.Tensor:
        """Embed all features and take the maximum."""
        order = self.embed_order(g.ndata.pop("order").squeeze(-1))
        tag = self.embed_tag(g.ndata.pop("tag").squeeze(-1))
        text = self.embed_text(g.ndata.pop("text"))
        img = self.embed_img(g.ndata.pop("img"))
        bgimg = self.embed_bgimg(g.ndata.pop("bgimg"))
        feat = [order, tag, text, img, bgimg]
        return torch.stack(feat).max(0).values  # type: ignore

    def forward(self, g: dgl.DGLGraph) -> torch.Tensor:
        N = g.num_nodes()
        h = self.embed_all(g)

        if self.disable_message_passing:
            return h

        g.ndata["h"] = h
        g.ndata["u"] = self.h_leaf.expand(N, -1)

        # bottom-up message
        dgl.prop_nodes_topo(
            g,
            message_func=fn.copy_src("u", "m"),
            reduce_func=fn_max("m", "u"),
            apply_node_func=self.nfunc_up,
        )

        _g = g.reverse(copy_ndata=True)
        _g.ndata["d"] = self.h_root.expand(N, -1)

        # top-down message
        dgl.prop_nodes_topo(
            _g,
            message_func=fn.copy_src("d", "m"),
            reduce_func=fn_max("m", "d"),
            apply_node_func=self.nfunc_down,
        )

        x = _g.ndata.pop("d")

        # residual
        if not self.disable_residual:
            x += h

        del g.ndata["h"], g.ndata["u"]

        return x  # type: ignore

    def nfunc_up(self, nodes: dgl.udf.NodeBatch) -> dict:
        """Merge element features and intermediate features coming from children."""
        hu = [nodes.data[k] for k in "hu"]
        u = self.mlp_up(torch.cat(hu, dim=-1))
        return {"u": u}

    def nfunc_down(self, nodes: dgl.udf.NodeBatch) -> dict:
        """Merge previous features and intermediate features coming from parents."""
        ud = [nodes.data[k] for k in "ud"]
        d = self.mlp_up(torch.cat(ud, dim=-1))
        return {"d": d}


def fn_max(k: str, v: str) -> Callable:
    # NOTE: fn.max function seems to fail in backpropagation.
    def udf(nodes: dgl.udf.NodeBatch) -> dict:
        return {v: nodes.mailbox.pop(k).max(1).values}

    return udf
