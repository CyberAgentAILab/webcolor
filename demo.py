import argparse
import random
from pathlib import Path
from typing import Tuple

import dgl
import torch.hub
from pytorch_lightning import seed_everything

import webcolor.lightning.generator as lit_generator
from webcolor.data.dataset import WebColorDataset
from webcolor.lightning.generator import LitBaseGenerator
from webcolor.lightning.upsampler import Upsampler
from webcolor.visualize import save_image

# To save downloaded ckpt files under `checkpoints`.
torch.hub.set_dir(".")  # type: ignore


def main() -> None:
    args = parse_args()

    out_path = Path(args.out_path)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # load data and models
    batch, data_id = load_data(args)
    generator, upsampler = load_models(args)

    # load to device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch = batch.to(device)
    generator = generator.to(device)
    upsampler = upsampler.to(device)

    # prepare for inference
    seed_everything(args.seed)
    g, batch_mask = generator.prepare_batch(batch)
    h = g.clone()

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

    # save images
    if args.save_gt:
        _g = dgl.unbatch(g)[0]
        name = out_path.stem + "_gt" + out_path.suffix
        save_image(_g, data_id, out_path.with_name(name))
    save_image(h, data_id, out_path)


def load_data(args: argparse.Namespace) -> Tuple[dgl.DGLGraph, str]:
    dataset = WebColorDataset("test")
    if args.target.isdecimal():
        idx = int(args.target)
    elif args.target == "random":
        idx = random.randrange(len(dataset))
    else:
        idx = dataset.data_ids.index(args.target)

    batch = dgl.batch([dataset[idx][0]] * args.num_save)
    data_id = dataset.data_ids[idx]
    print(f'Target: {{ index: {idx}, data_id: "{data_id}" }}')

    return batch, data_id


def load_models(args: argparse.Namespace) -> Tuple[LitBaseGenerator, Upsampler]:
    _cls = getattr(lit_generator, args.model)
    if args.model == "Stats":
        generator = _cls.load_from_checkpoint(args.ckpt_path, sampling=args.sampling)
    elif args.model == "AR":
        generator = _cls.load_from_checkpoint(args.ckpt_path, top_p=args.top_p)
    else:
        generator = _cls.load_from_checkpoint(args.ckpt_path)
    upsampler = Upsampler.load_from_checkpoint(args.upsampler_path)
    generator, upsampler = generator.eval(), upsampler.eval()
    return generator, upsampler


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

    # Data and save
    parser.add_argument(
        "--target",
        type=str,
        default="test_GB_www.warehouse.co.uk_12679",
        help='index or data_id to identify the data in the test set, or the keyword "random"',
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="output/screenshot.png",
        help="output path",
    )
    parser.add_argument(
        "--num_save",
        type=int,
        default=1,
        help="number of screenshots to save",
    )
    parser.add_argument(
        "--save_gt",
        action="store_true",
        help="save ground-truth",
    )

    args = parser.parse_args()

    if (
        args.model == "NAR"
        or (args.model == "Stats" and not args.sampling)
        or (args.model == "AR" and args.top_p == 0.0)
    ):
        if args.num_save > 1:
            print("Set num_save to 1 due to deterministic model")
            args.num_save = 1

    return args


if __name__ == "__main__":
    main()
