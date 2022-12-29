import sys

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.cli import LightningCLI

import webcolor.lightning.generator  # noqa: F401
import webcolor.lightning.upsampler  # noqa: F401
from webcolor.lightning.datamodule import WebColorDataModule
from webcolor.lightning.generator import LitBaseGenerator
from webcolor.lightning.upsampler import LitBaseUpsampler


def cli_generator(subcommand: str, model_name: str) -> None:
    model_ckpt_kwargs_list = []
    if subcommand == "fit":
        kwargs = dict(filename="last", monitor="step", mode="max")
        model_ckpt_kwargs_list.append(kwargs)

        if model_name != "Stats":
            kwargs = dict(filename="best", monitor="val/acc_rgb", mode="max")
            model_ckpt_kwargs_list.append(kwargs)

    callbacks = [ModelCheckpoint(**kwargs) for kwargs in model_ckpt_kwargs_list]  # type: ignore

    LightningCLI(
        model_class=LitBaseGenerator,
        datamodule_class=WebColorDataModule,
        subclass_mode_model=True,
        trainer_defaults=dict(max_steps=int(1e5), callbacks=callbacks),
    )


def cli_upsampler(subcommand: str) -> None:
    model_ckpt_kwargs_list = []
    if subcommand == "fit":
        model_ckpt_kwargs_list += [
            dict(filename="last", monitor="step", mode="max"),
            dict(filename="best", monitor="val/upsampler_loss", mode="min"),
        ]

    callbacks = [ModelCheckpoint(**kwargs) for kwargs in model_ckpt_kwargs_list]  # type: ignore

    LightningCLI(
        model_class=LitBaseUpsampler,
        datamodule_class=WebColorDataModule,
        subclass_mode_model=True,
        trainer_defaults=dict(max_steps=int(1e5), callbacks=callbacks),
    )


if __name__ == "__main__":
    msg = "The argument ``--model MODEL_NAME`` is required."
    assert "--model" in sys.argv, msg

    idx_model = sys.argv.index("--model") + 1
    assert idx_model < len(sys.argv), msg

    subcommand = sys.argv[1]
    model_name = sys.argv[idx_model]
    if model_name in {"Upsampler"}:
        cli_upsampler(subcommand)
    else:
        cli_generator(subcommand, model_name)
