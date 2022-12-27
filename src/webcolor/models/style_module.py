from typing import Dict

import torch
import torch.nn as nn

from webcolor.data.converter import NUM_COLOR_BINS


class StyleEncoder(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()

        nb = NUM_COLOR_BINS
        self.embed_rgb = nn.Embedding(nb**3, d_model)
        self.embed_alpha = nn.Embedding(nb, d_model)
        self.fc1 = nn.Linear(d_model * 2, d_model)
        self.fc2 = nn.Linear(d_model * 2, d_model)
        self.non_text_emb = nn.Parameter(torch.rand(1, d_model))

    def forward(
        self,
        text_color: torch.Tensor,
        bg_color: torch.Tensor,
        has_text: torch.Tensor,
    ) -> torch.Tensor:
        rgb = torch.stack([text_color[:, 0], bg_color[:, 0]])
        alpha = torch.stack([text_color[:, 1], bg_color[:, 1]])

        # merge rgb and alpha embeddings
        x_rgb = self.embed_rgb(rgb)
        x_alpha = self.embed_alpha(alpha)
        x = torch.cat([x_rgb, x_alpha], dim=-1)
        x_rgba = torch.relu(self.fc1(x))

        # replace x_text for non-text elements with special embedding
        x_text, x_bg = x_rgba
        non_text_emb = self.non_text_emb.expand_as(x_text)
        x_text = torch.where(has_text.unsqueeze(1), x_text, non_text_emb)

        # merge text_color and bg_color embeddings
        x = torch.cat([x_text, x_bg], dim=-1)
        x = self.fc2(x)

        return x


class StyleDecoder(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()

        nb = NUM_COLOR_BINS
        self.fc_text = nn.Linear(d_model, d_model)
        self.fc_bg = nn.Linear(d_model, d_model)
        self.fc_rgb = nn.Linear(d_model, nb**3)
        self.fc_alpha = nn.Linear(d_model, nb)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x_text = torch.relu(self.fc_text(x))
        x_bg = torch.relu(self.fc_bg(x))
        x = torch.stack([x_text, x_bg])

        logit_text_rgb, logit_bg_rgb = self.fc_rgb(x)
        logit_text_alpha, logit_bg_alpha = self.fc_alpha(x)

        return {
            "logit_text_rgb": logit_text_rgb,
            "logit_text_alpha": logit_text_alpha,
            "logit_bg_rgb": logit_bg_rgb,
            "logit_bg_alpha": logit_bg_alpha,
        }
