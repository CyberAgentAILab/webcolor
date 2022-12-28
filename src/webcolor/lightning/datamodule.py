import dgl
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from webcolor.data.dataset import WebColorDataset


class WebColorDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = WebColorDataset("train")
        self.val_dataset = WebColorDataset("val")
        self.test_dataset = WebColorDataset("test")

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
