import http.server
import subprocess
import tempfile
import zipfile
from functools import partial
from io import BytesIO
from os.path import abspath
from pathlib import Path
from threading import Thread
from typing import Dict, Union

import dgl
import requests

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
    with tempfile.TemporaryDirectory() as _tmpdir:
        tmpdir = Path(_tmpdir)
        paths = download_web_page(data_id, tmpdir)
        for i, _g in enumerate(dgl.unbatch(g)):
            if i == 0:
                _out_path = out_path
            else:
                name = out_path.stem + f"_{i-1}" + out_path.suffix
                _out_path = out_path.with_name(name)

            update_css(_g, paths["selector"], paths["css"])
            screenshot(paths["html"], _out_path)


def download_web_page(
    data_id: str,
    out_dir: Union[str, Path],
) -> Dict[str, Path]:
    """Download a processed web page specified by `data_id` to `out_dir`."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    zip_url = BASE_URL + f"{data_id}.zip"
    response = requests.get(zip_url)
    with zipfile.ZipFile(BytesIO(response.content)) as f:
        f.extractall(out_dir)

    return {
        "selector": out_dir / "selectors.txt",
        "html": out_dir / "user.html",
        "css": out_dir / "user.css",
    }


def update_css(
    g: dgl.DGLGraph,
    selector_path: Path,
    css_path: Path,
) -> None:
    # translate graph data to css text
    css_text = ""
    selectors = selector_path.read_text().split("\n")
    for i, selector in enumerate(selectors):
        color_dis = g.ndata["bg_color"][i]
        color_res = g.ndata["bg_color_res"][i]
        style = {"background-color": revert_color(color_dis, color_res)}

        if g.ndata["has_text"][i]:
            color_dis = g.ndata["text_color"][i]
            color_res = g.ndata["text_color_res"][i]
            style["color"] = revert_color(color_dis, color_res)

        content = " ".join(f"{k}: {v};" for k, v in style.items())
        css_text += f"{selector} {{ {content} }}\n"

    css_path.write_text(css_text)


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


def ServeDirectoryWithHTTP(directory: str = ".") -> tuple:
    # based on https://gist.github.com/kwk/5387c0e8d629d09f93665169879ccb86

    class NoLogHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args, **kwargs):  # type: ignore
            pass

    hostname = "localhost"
    directory = abspath(directory)
    handler = partial(NoLogHandler, directory=directory)
    httpd = http.server.HTTPServer((hostname, 0), handler, False)
    # Block only for 0.5 seconds max
    httpd.timeout = 0.5
    # Allow for reusing the address
    # HTTPServer sets this as well but I wanted to make this more obvious.
    httpd.allow_reuse_address = True

    httpd.server_bind()

    address = "http://%s:%d" % (httpd.server_name, httpd.server_port)

    httpd.server_activate()

    def serve_forever(httpd: http.server.HTTPServer) -> None:
        with httpd:  # to make sure httpd.server_close is called
            httpd.serve_forever()

    thread = Thread(target=serve_forever, args=(httpd,))
    thread.setDaemon(True)
    thread.start()

    return httpd, address
