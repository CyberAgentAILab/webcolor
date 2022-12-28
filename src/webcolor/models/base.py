from typing import Any, Dict

import dgl
import torch
import torch.nn as nn

from webcolor.models.content_module import ContentEncoder
from webcolor.models.style_module import StyleDecoder, StyleEncoder


class BaseModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        has_style_encoder: bool,
        pred_residual: bool,
        disable_message_passing: bool = False,
        disable_residual: bool = False,
    ):
        super().__init__()

        # encoder
        self.content_encoder = ContentEncoder(
            d_model,
            disable_message_passing=disable_message_passing,
            disable_residual=disable_residual,
        )
        if has_style_encoder:
            self.style_encoder = StyleEncoder(d_model)
        else:
            self.style_encoder = None

        # decoder
        self.style_decoder = StyleDecoder(d_model, pred_residual)

    def encode_content(self, g: dgl.DGLGraph) -> torch.Tensor:
        return self.content_encoder(g)  # type: ignore

    def encode_style(
        self,
        text_color: torch.Tensor,
        bg_color: torch.Tensor,
        has_text: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        assert self.style_encoder, "This model does not have the style encoder."
        return self.style_encoder(text_color, bg_color, has_text)  # type: ignore

    def decode_style(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.style_decoder(x)  # type: ignore

    def forward(
        self, g: dgl.DGLGraph, batch_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward method for loss calculation (training)."""
        raise NotImplementedError

    def generate(
        self, g: dgl.DGLGraph, batch_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Generate colors for testing."""
        raise NotImplementedError


class BaseGenerator(BaseModel):
    def __init__(self, **kwargs: Any):
        super().__init__(pred_residual=False, **kwargs)


class BaseUpsampler(BaseModel):
    def __init__(self, **kwargs: Any):
        super().__init__(pred_residual=True, has_style_encoder=True, **kwargs)
