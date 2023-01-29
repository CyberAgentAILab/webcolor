import argparse
import multiprocessing as mp
import tempfile
from pathlib import Path
from typing import Tuple

import dgl
import numpy as np
import torch
import torch.hub
from PIL import Image
from pytorch_lightning import seed_everything
from tqdm import tqdm

import webcolor.lightning.generator as lit_generator
from webcolor.data.converter import NUM_COLOR_BINS
from webcolor.data.dataset import WebColorDataset
from webcolor.lightning.generator import LitBaseGenerator
from webcolor.lightning.upsampler import Upsampler
from webcolor.metrics import ContrastViolation, FrechetColorDistance
from webcolor.utils import download_web_page, screenshot, update_css

# To save downloaded ckpt files under `checkpoints`.
torch.hub.set_dir(".")  # type: ignore


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    # load dataset and models
    dataset = WebColorDataset("test")
    generator, upsampler = load_models(args)

    # load to device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generator = generator.to(device)
    upsampler = upsampler.to(device)

    # initialize metrics
    pixel_fcd = FrechetColorDistance().to(device)
    contrast = ContrastViolation().to(device)

    results = []
    with tempfile.TemporaryDirectory() as _tmpdir:
        tmpdir = Path(_tmpdir)
        loop = tqdm(dataset, dynamic_ncols=True, desc="inference")
        for idx, batch in enumerate(loop):
            # prepare for inference
            compute_real_for_fcd = batch[1]
            data_id = dataset.data_ids[idx]
            g, batch_mask = generator.prepare_batch(batch[0].to(device))

            # main inference
            h = generate_color_style(g, batch_mask, generator, upsampler)

            # download web page
            out_dir = tmpdir / data_id
            img_path = out_dir / "screenshot.png"
            paths = download_web_page(data_id, out_dir)

            # update css and take screenshot
            if compute_real_for_fcd:
                update_css(g, paths["selector"], paths["css"])
            else:
                update_css(h, paths["selector"], paths["css"])
            screenshot(paths["html"], img_path)
            if compute_real_for_fcd:
                update_css(h, paths["selector"], paths["css"])

            # collect paths
            results.append(
                {
                    "html_path": paths["html"],
                    "img_path": img_path,
                    "real": compute_real_for_fcd,
                }
            )

        # run lighthouse with multiple processes
        with mp.Pool(args.num_workers) as p:
            for result in tqdm(
                p.imap_unordered(add_lighthouse_result, results),
                total=len(results),
                dynamic_ncols=True,
                desc="lighthouse",
            ):
                # update metrics
                img_path = result["img_path"]
                x = load_image_as_discrete_colors(img_path).to(device)
                pixel_fcd.update(x.unsqueeze(0), real=result["real"])
                contrast.update(result["lighthouse"])

    # compute metrics
    score = pixel_fcd.compute().item()
    print(f"Pixel-FCD {score*1e3:.3f} x 1e-3")

    print("Contrast violation:")
    result = contrast.compute()
    for key, value in result.items():
        print(f"\t{key} {value.item():.3f}")


def load_models(args: argparse.Namespace) -> Tuple[LitBaseGenerator, Upsampler]:
    _cls = getattr(lit_generator, args.model)
    if args.model == "Stats":
        generator = _cls.load_from_checkpoint(args.ckpt_path, sampling=args.sampling)
    elif args.model == "AR":
        generator = _cls.load_from_checkpoint(args.ckpt_path, top_p=args.top_p)
    else:
        generator = _cls.load_from_checkpoint(args.ckpt_path)
    upsampler = Upsampler.load_from_checkpoint(args.upsampler_path)
    generator, upsampler = generator.eval(), upsampler.eval()  # type: ignore
    return generator, upsampler


def generate_color_style(
    g: dgl.DGLGraph,
    batch_mask: torch.Tensor,
    generator: LitBaseGenerator,
    upsampler: Upsampler,
) -> dgl.DGLGraph:
    g, h = g.clone(), g.clone()

    # generator
    with torch.no_grad():
        pred = generator.generate(g, batch_mask)
    text_color = torch.stack([pred["pred_text_rgb"], pred["pred_text_alpha"]], dim=1)
    bg_color = torch.stack([pred["pred_bg_rgb"], pred["pred_bg_alpha"]], dim=1)
    h.ndata["text_color"] = text_color
    h.ndata["bg_color"] = bg_color

    # upsampler
    with torch.no_grad():
        pred = upsampler.generate(h, batch_mask)
    h.ndata["text_color_res"] = pred["pred_text_res"]
    h.ndata["bg_color_res"] = pred["pred_bg_res"]

    return h


def add_lighthouse_result(result: dict) -> dict:
    result["lighthouse"] = ContrastViolation.get_lighthouse_result(result["html_path"])
    return result


def load_image_as_discrete_colors(img_path: Path) -> torch.Tensor:
    nb = NUM_COLOR_BINS
    min_value = 0
    max_value = 255
    bin_size = 32
    x = np.asarray(Image.open(img_path).convert("RGB"))
    bins = np.arange(min_value, max_value, bin_size) + bin_size
    idx = np.digitize(x, bins, right=False).reshape(-1, 3)
    idx_rgb = sum(nb**j * idx[:, i] for i, j in zip(range(3), reversed(range(3))))
    return torch.from_numpy(idx_rgb)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # General
    parser.add_argument(
        "--seed",
        type=int,
        help="manual seed",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="number of workers",
    )

    # Models
    parser.add_argument(
        "--model",
        type=str,
        choices=["CVAE", "NAR", "AR", "Stats"],
        help="model name",
        required=True,
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        help="checkpoint path",
        required=True,
    )
    parser.add_argument(
        "--upsampler_path",
        type=str,
        help="checkpoint path for Upsampler",
        required=True,
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.0,
        help="`top_p` parameter for AR model",
    )
    parser.add_argument(
        "--sampling",
        type=bool,
        default=False,
        help="`sampling` parameter for Stats model",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()
