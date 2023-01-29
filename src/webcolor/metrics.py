import json
import subprocess
from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Metric
from torchmetrics.image.fid import FrechetInceptionDistance

import webcolor.utils as utils
from webcolor.data.converter import NUM_COLOR_BINS


class FrechetColorDistance(FrechetInceptionDistance):
    def __init__(
        self,
        reset_real_features: bool = False,
    ) -> None:
        super().__init__(
            feature=ColorFeatureExtractor(),
            reset_real_features=reset_real_features,
            normalize=False,
        )


class ColorFeatureExtractor(nn.Module):
    def __init__(self, ignore_idx: int = -1) -> None:
        super().__init__()
        self.num_features = NUM_COLOR_BINS**3
        self.ignore_idx = ignore_idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.size()) == 2:
            x = x.clone()
            x[x == self.ignore_idx] = self.num_features
            freq = F.one_hot(x, self.num_features + 1)[:, :, :-1]
            hist = freq.sum(dim=1) / freq.sum(dim=(1, 2)).unsqueeze(-1)
            return hist.nan_to_num()  # type: ignore
        elif len(x.size()) == 4:
            # To inform `num_features` in the `__init__` function of
            # `FrechetInceptionDistance``, using a dummy batched image as input.
            return torch.zeros(self.num_features)
        else:
            raise NotImplementedError(x.size())


class ContrastViolation(Metric):
    violating_elements: torch.Tensor
    violating_pages: torch.Tensor
    total_pages: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        zero = torch.tensor(0)
        self.add_state("violating_pages", default=zero.clone(), dist_reduce_fx="sum")
        self.add_state("violating_elements", default=zero.clone(), dist_reduce_fx="sum")
        self.add_state("total_pages", default=zero.clone(), dist_reduce_fx="sum")

    def update(self, x: Union[dict, str, Path]) -> None:
        result = x if isinstance(x, dict) else self.get_lighthouse_result(x)
        if result is not None:
            res_contrast = result["audits"]["color-contrast"]
            if res_contrast["score"] is not None:
                self.violating_pages += 1 - res_contrast["score"]
                self.violating_elements += len(res_contrast["details"]["items"])
                self.total_pages += 1

    def compute(self) -> Dict[str, torch.Tensor]:
        return {
            "% pages": self.violating_pages.float() / self.total_pages * 100,
            "# elements": self.violating_elements.float() / self.total_pages,
        }

    @staticmethod
    def get_lighthouse_result(
        html_path: Union[str, Path],
        timeout: int = 60,
        trials: int = 3,
    ) -> Optional[dict]:
        html_path = Path(html_path)
        httpd, base_url = utils.ServeDirectoryWithHTTP(str(html_path.parent))
        url = f"{base_url}/{html_path.name}"

        chrome_flags = " ".join(
            [
                "--headless",
                "--disable-web-security",
                "--allow-running-insecure-content",
                "--no-sandbox",
                "--disable-infobars",
                "--hide-scrollbars",
                "--disable-dev-shm-usage",
                f'--user-agent="{utils.UA}"',
                f'--window-size="{utils.WIDTH},{utils.HEIGHT}"',
            ]
        )

        cmd = [
            "lighthouse",
            url,
            "--quiet",
            "--only-audits=color-contrast",
            "--output=json",
            f"--chrome-flags='{chrome_flags}'",
        ]

        result = None
        for _ in range(trials):
            try:
                out = subprocess.check_output(
                    " ".join(cmd),
                    shell=True,
                    stderr=subprocess.DEVNULL,
                    timeout=timeout,
                )
                result = json.loads(out.decode())
                break

            except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                pass

        httpd.shutdown()

        return result
