import dgl
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from .dataset import WebColorDataset


class BaseDataModule(LightningDataModule):
    def __init__(
        self,
        train_dataset: WebColorDataset,
        val_dataset: WebColorDataset,
        test_dataset: WebColorDataset,
        batch_size: int,
        num_workers: int,
    ) -> None:
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self) -> DataLoader:
        return dgl.dataloading.GraphDataLoader(  # type: ignore
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return dgl.dataloading.GraphDataLoader(  # type: ignore
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return dgl.dataloading.GraphDataLoader(  # type: ignore
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class GeneratorDataModule(BaseDataModule):
    def __init__(self, batch_size: int = 32, num_workers: int = 4) -> None:
        super().__init__(
            train_dataset=WebColorDataset("train"),
            val_dataset=WebColorDataset("val"),
            test_dataset=WebColorDataset("test"),
            batch_size=batch_size,
            num_workers=num_workers,
        )


class UpsamplerDataModule(BaseDataModule):
    def __init__(self, batch_size: int = 32, num_workers: int = 4) -> None:
        # TODO: implement this
        raise NotImplementedError
