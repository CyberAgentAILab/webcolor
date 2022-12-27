from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.cli import LightningCLI

import webcolor.models.lightning  # noqa: F401
from webcolor.data.lightning import GeneratorDataModule
from webcolor.models.lightning import LitBaseGenerator


def cli_main() -> None:
    callbacks = [
        ModelCheckpoint(filename="last", monitor="step", mode="max"),
        ModelCheckpoint(filename="best", monitor="val/acc_rgb", mode="max"),
    ]
    LightningCLI(
        model_class=LitBaseGenerator,
        datamodule_class=GeneratorDataModule,
        subclass_mode_model=True,
        trainer_defaults=dict(max_steps=int(1e5), callbacks=callbacks),
    )


if __name__ == "__main__":
    cli_main()
