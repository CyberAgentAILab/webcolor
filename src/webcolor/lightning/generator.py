from typing import Dict, Tuple

import dgl
import torch
import torch.nn.functional as F
import torchmetrics
from pytorch_lightning import LightningModule
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from torchmetrics.functional.classification import multiclass_accuracy

from webcolor.data.converter import NUM_COLOR_BINS
from webcolor.models.base import BaseGenerator
from webcolor.models.utils import make_batch_mask


class LitBaseGenerator(LightningModule):
    def __init__(self, model_name: str, model: BaseGenerator):
        super().__init__()
        self.model_name = model_name
        self.model = model
        self.test_metrics: Dict[str, torchmetrics.Metric] = dict()

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

    def _prepare_batch(self, batch: dgl.DGLGraph) -> Tuple[dgl.DGLGraph, torch.Tensor]:
        batch_mask = make_batch_mask(batch)
        return batch, batch_mask

    def _common_step(
        self,
        batch: dgl.DGLGraph,
        batch_idx: int,
        stage: str,
    ) -> torch.Tensor:
        g, batch_mask = self._prepare_batch(batch)
        out = self(g, batch_mask)

        # format prediction
        key_to_logit_target = self._format_prediction(g, out)

        # compute loss
        loss_dict = {
            f"loss_{key}": F.cross_entropy(logit, target)
            for key, (logit, target) in key_to_logit_target.items()
        }
        loss = sum(loss_dict.values())

        if self.model_name == "CVAE":
            mu, logvar = out["mu"], out["logvar"]
            _sum = torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1)
            loss_dict["loss_kl"] = torch.mean(-0.5 * _sum, dim=0)
            loss += self.coeff_kl * loss_dict["loss_kl"]

        # compute metrics
        nb = NUM_COLOR_BINS
        acc_dict = {
            f"acc_{key}": multiclass_accuracy(
                logit,
                target,
                num_classes=nb**3 if key == "rgb" else nb,
                average="micro",
            )
            for key, (logit, target) in key_to_logit_target.items()
        }

        # log values
        B = g.batch_size
        self.log(f"{stage}/loss", loss, batch_size=B)
        for key, value in loss_dict.items():
            self.log(f"{stage}/{key}", value, batch_size=B)
        for key, value in acc_dict.items():
            self.log(f"{stage}/{key}", value, on_step=False, on_epoch=True, batch_size=B)  # type: ignore

        return loss  # type: ignore

    def _format_prediction(
        self, g: dgl.DGLGraph, out: Dict[str, torch.Tensor]
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        # compute text color only for elements having text
        text_mask = g.ndata["has_text"]
        text_color = g.ndata["text_color"][text_mask]
        logit_text_rgb = out["logit_text_rgb"][text_mask]
        logit_text_alpha = out["logit_text_alpha"][text_mask]

        # concatenate text and bg colors
        logit_rgb = torch.cat([logit_text_rgb, out["logit_bg_rgb"]])
        logit_alpha = torch.cat([logit_text_alpha, out["logit_bg_alpha"]])
        target_rgb = torch.cat([text_color[:, 0], g.ndata["bg_color"][:, 0]])
        target_alpha = torch.cat([text_color[:, 1], g.ndata["bg_color"][:, 1]])

        return {
            "rgb": (logit_rgb, target_rgb),
            "alpha": (logit_alpha, target_alpha),
        }

    def on_test_start(self) -> None:
        nb = NUM_COLOR_BINS
        keys = ["rgb", "alpha"]

        # Accuracy
        self.test_metrics.update(
            {
                f"acc_{key}": MulticlassAccuracy(
                    num_classes=nb**3 if key == "rgb" else nb,
                    average="micro",
                )
                for key in keys
            }
        )

        # Macro F-score
        self.test_metrics.update(
            {
                f"f1_{key}": MulticlassF1Score(
                    num_classes=nb**3 if key == "rgb" else nb,
                    average="macro",
                )
                for key in keys
            }
        )

        # TODO: Frechet Color Distance, Contrast violation

    def test_step(self, batch: dgl.DGLGraph, batch_idx: int) -> None:
        g, batch_mask = self._prepare_batch(batch)
        out = self.model.generate(g, batch_mask)

        # format prediction
        key_to_logit_target = self._format_prediction(g, out)

        # update all metrics
        for metric_name, metric in self.test_metrics.items():
            key = metric_name.split("_")[-1]
            metric.update(*key_to_logit_target[key])

    def on_test_end(self) -> None:
        # compute all metrics
        for metric_name, metric in self.test_metrics.items():
            print(metric_name, metric.compute().item())
            metric.reset()


class CVAE(LitBaseGenerator):
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        norm_first: bool = True,
        coeff_kl: float = 0.1,
        disable_message_passing: bool = False,
        disable_residual: bool = False,
    ):
        """
        Conditional Variational Autoencoder.

        Args:
            d_model: Base dimension size
            nhead: Number of attention heads in the Transformer
            num_layers: Number of encoder/decoder layers in the Transformer
            dim_feedforward: Dimension of the feedforward network in the Transformer
            norm_first: Perform the layer normalization on each layer before other operations
            coeff_kl: Coefficient of the KL term
            disable_message_passing: Disable message passing in the content encoder
            disable_residual: Disable residual connection in the content encoder
        """
        from webcolor.models.cvae import CVAE as model_cls

        self.coeff_kl = coeff_kl
        super().__init__(
            "CVAE",
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


class NAR(LitBaseGenerator):
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
        Non-autoregressive Transformer.

        Args:
            d_model: Base dimension size
            nhead: Number of attention heads in the Transformer
            num_layers: Number of encoder/decoder layers in the Transformer
            dim_feedforward: Dimension of the feedforward network in the Transformer
            norm_first: Perform the layer normalization on each layer before other operations
            disable_message_passing: Disable message passing in the content encoder
            disable_residual: Disable residual connection in the content encoder
        """
        from webcolor.models.nar import NARTransformer as model_cls

        super().__init__(
            "NAR",
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
