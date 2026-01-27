from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.datasets.norway import NorwayALS  # you create this dataset class

class NorwayDataModule(LightningDataModule):
    def __init__(self, paths, dataloader, **kwargs):
        super().__init__()
        self.paths = paths
        self.dataloader_cfg = dataloader
        self.kwargs = kwargs

    def setup(self, stage=None):
        root = self.paths.data_dir
        self.train_set = NorwayALS(root=str(root), stage="train", **self.kwargs)
        self.val_set   = NorwayALS(root=str(root), stage="val",   **self.kwargs)

    def train_dataloader(self):
        return DataLoader(self.train_set, **self.dataloader_cfg)

    def val_dataloader(self):
        return DataLoader(self.val_set, **self.dataloader_cfg)
