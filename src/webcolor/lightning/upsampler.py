from typing import Dict, Tuple

import dgl
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from webcolor.models.base import BaseUpsampler
from webcolor.models.utils import make_batch_mask


class LitBaseUpsampler(LightningModule):
    def __init__(self, model_name: str, model: BaseUpsampler):
        super().__init__()
        self.model_name = model_name
        self.model = model

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the default optimizer.
        The optimizer can be overridden by command line arguments, such as
        ``--optimizer SGD --optimizer.lr 0.1``.
        """
        return torch.optim.AdamW(self.model.parameters(), lr=1e-4)

    def forward(
        self,
        g: dgl.DGLGraph,
        batch_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return self.model(g, batch_mask)  # type: ignore

    def generate(
        self,
        g: dgl.DGLGraph,
        batch_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return self.model.generate(g, batch_mask)  # type: ignore

    def training_step(self, batch: dgl.DGLGraph, batch_idx: int) -> torch.Tensor:
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch: dgl.DGLGraph, batch_idx: int) -> None:
        self._common_step(batch, batch_idx, "val")

    def test_step(self, batch: dgl.DGLGraph, batch_idx: int) -> None:
        self._common_step(batch, batch_idx, "test")

    def prepare_batch(self, batch: dgl.DGLGraph) -> Tuple[dgl.DGLGraph, torch.Tensor]:
        batch_mask = make_batch_mask(batch)
        return batch, batch_mask

    def _common_step(
        self,
        batch: dgl.DGLGraph,
        batch_idx: int,
        stage: str,
    ) -> torch.Tensor:
        g, batch_mask = self.prepare_batch(batch)
        out = self(g, batch_mask)

        # format prediction
        key_to_pred_target = self._format_prediction(g, out)

        # compute loss
        loss_dict = {
            f"upsampler_loss_{key}": F.mse_loss(logit, target)
            for key, (logit, target) in key_to_pred_target.items()
        }
        loss = sum(loss_dict.values())

        # log values
        B = g.batch_size
        self.log(f"{stage}/upsampler_loss", loss, batch_size=B)
        for key, value in loss_dict.items():
            self.log(f"{stage}/{key}", value, batch_size=B)

        return loss  # type: ignore

    def _format_prediction(
        self, g: dgl.DGLGraph, out: Dict[str, torch.Tensor]
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        # compute text color only for elements having text
        text_mask = g.ndata["has_text"]
        text_res = g.ndata["text_color_res"][text_mask]
        pred_text_res = out["pred_text_res"][text_mask]
        return {
            "text": (pred_text_res, text_res),
            "bg": (out["pred_bg_res"], g.ndata["bg_color_res"]),
        }


class Upsampler(LitBaseUpsampler):
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        norm_first: bool = True,
        disable_message_passing: bool = False,
        disable_residual: bool = False,
    ):
        """
        Transformer-based color upsampler.

        Args:
            d_model: Base dimension size
            nhead: Number of attention heads in the Transformer
            num_layers: Number of encoder/decoder layers in the Transformer
            dim_feedforward: Dimension of the feedforward network in the Transformer
            norm_first: Perform the layer normalization on each layer before other operations
            disable_message_passing: Disable message passing in the content encoder
            disable_residual: Disable residual connection in the content encoder
        """
        from webcolor.models.color_upsampler import ColorUpsampler as model_cls

        super().__init__(
            "Upsampler",
            model_cls(
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                norm_first=norm_first,
                disable_message_passing=disable_message_passing,
                disable_residual=disable_residual,
            ),
        )
