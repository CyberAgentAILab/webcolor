import sys

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.cli import LightningCLI

import webcolor.lightning.generator  # noqa: F401
import webcolor.lightning.upsampler  # noqa: F401
from webcolor.lightning.datamodule import WebColorDataModule
from webcolor.lightning.generator import LitBaseGenerator
from webcolor.lightning.upsampler import LitBaseUpsampler


def cli_generator() -> None:
    callbacks = [
        ModelCheckpoint(filename="last", monitor="step", mode="max"),
        ModelCheckpoint(filename="best", monitor="val/acc_rgb", mode="max"),
    ]
    LightningCLI(
        model_class=LitBaseGenerator,
        datamodule_class=WebColorDataModule,
        subclass_mode_model=True,
        trainer_defaults=dict(max_steps=int(1e5), callbacks=callbacks),
    )


def cli_upsampler() -> None:
    callbacks = [
        ModelCheckpoint(filename="last", monitor="step", mode="max"),
        ModelCheckpoint(filename="best", monitor="val/upsampler_loss", mode="min"),
    ]
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

    if sys.argv[idx_model] in {"Upsampler"}:
        cli_upsampler()
    else:
        cli_generator()
