import subprocess
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path
from typing import List, Union

import dgl
import requests
import torch

from webcolor.data.converter import revert_color

BASE_URL = "https://storage.googleapis.com/ailab-public/webcolor/dataset/processed/"
UA = "Mozilla/5.0 (iPhone; CPU iPhone OS 11_0 like Mac OS X) AppleWebKit/604.1.38 (KHTML, like Gecko) Version/11.0 Mobile/15A372 Safari/604.1"
WIDTH = 375
HEIGHT = 812


def save_image(
    g: dgl.DGLGraph,
    data_id: str,
    out_path: Union[str, Path],
) -> None:
    out_path = Path(out_path)
    zip_url = BASE_URL + f"{data_id}.zip"
    with tempfile.TemporaryDirectory() as _tmpdir:
        tmpdir = Path(_tmpdir)
        selector_path = tmpdir / "selectors.txt"
        html_path = tmpdir / "user.html"
        css_path = tmpdir / "user.css"

        response = requests.get(zip_url)
        with zipfile.ZipFile(BytesIO(response.content)) as f:
            f.extractall(_tmpdir)

        selectors = selector_path.read_text().split("\n")

        for i, _g in enumerate(dgl.unbatch(g)):
            if i == 0:
                _out_path = out_path
            else:
                name = out_path.stem + f"_{i-1}" + out_path.suffix
                _out_path = out_path.with_name(name)

            css_text = convert_to_css_text(
                selectors,
                _g.ndata["has_text"],
                _g.ndata["text_color"],
                _g.ndata["text_color_res"],
                _g.ndata["bg_color"],
                _g.ndata["bg_color_res"],
            )
            css_path.write_text(css_text)
            screenshot(html_path, _out_path)


def convert_to_css_text(
    selectors: List[str],
    has_text: torch.Tensor,
    text_color: torch.Tensor,
    text_color_res: torch.Tensor,
    bg_color: torch.Tensor,
    bg_color_res: torch.Tensor,
) -> str:
    css_text = ""
    for i, selector in enumerate(selectors):
        style = {
            "background-color": revert_color(bg_color[i], bg_color_res[i]),
        }
        if has_text[i]:
            style["color"] = revert_color(text_color[i], text_color_res[i])

        content = " ".join(f"{k}: {v};" for k, v in style.items())
        css_text += f"{selector} {{ {content} }}\n"
    return css_text


def screenshot(
    html_path: Union[str, Path],
    out_path: Union[str, Path],
    timeout: int = 60,
    trials: int = 3,
) -> None:
    html_path, out_path = Path(html_path), Path(out_path)
    out_path.parent.mkdir(exist_ok=True, parents=True)

    cmd = [
        "google-chrome",
        "--headless",
        "--disable-web-security",
        "--allow-running-insecure-content",
        "--no-sandbox",
        "--disable-infobars",
        "--hide-scrollbars",
        "--disable-dev-shm-usage",
        f'--user-agent="{UA}"',
        f'--window-size="{WIDTH},{HEIGHT}"',
        f'--screenshot="{str(out_path.resolve())}"',
    ]

    for _ in range(trials):
        try:
            subprocess.run(
                " ".join(cmd + [str(html_path)]),
                shell=True,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout,
            )
            break

        except subprocess.TimeoutExpired:
            pass
    else:
        raise RuntimeError("Failed to take a screenshot.")
