import os.path as osp
import torch
import laspy
from typing import List

from src.datasets import BaseDataset
from src.data import Data
from src.datasets.norway_config import TILES, ID2TRAINID, CLASS_NAMES, CLASS_COLORS, NOR_NUM_CLASSES

__all__ = ["NorwayALS", "MiniNorwayALS"]

def read_norway_tile(filepath: str, remap: bool = True) -> Data:
    data = Data()
    las = laspy.read(filepath)

    # XYZ -> pos (med offset som DALES gjør)
    pos = torch.stack([torch.tensor(las[ax]) for ax in ["X", "Y", "Z"]], dim=-1)
    pos = pos * las.header.scale
    pos_offset = pos[0]
    data.pos = (pos - pos_offset).float()
    data.pos_offset = pos_offset

    # Semantic labels: bruk LAS classification (bytt hvis du har annet felt)
    y = torch.as_tensor(las["classification"], dtype=torch.long)
    if remap:
        data.y = torch.from_numpy(ID2TRAINID)[y]
    else:
        data.y = y

    return data


class NorwayALS(BaseDataset):
    @property
    def class_names(self) -> List[str]:
        return CLASS_NAMES  # må være num_classes + 1 (siste = ignored)

    @property
    def num_classes(self) -> int:
        return NOR_NUM_CLASSES

    @property
    def class_colors(self) -> List[List[int]]:
        return CLASS_COLORS

    @property
    def all_base_cloud_ids(self) -> List[str]:
        return TILES

    def read_single_raw_cloud(self, raw_cloud_path: str) -> "Data":
        return read_norway_tile(raw_cloud_path, remap=True)

    @property
    def raw_file_structure(self) -> str:
        return f"""
    {self.root}/
        └── raw/
            └── {{train, val, test}}/
                └── {{tile_name}}.laz
            """

    def id_to_relative_raw_path(self, id: str) -> str:
        # Finn stage basert på split-lister
        if id in self.all_cloud_ids["train"]:
            stage = "train"
        elif id in self.all_cloud_ids["val"]:
            stage = "val"
        elif id in self.all_cloud_ids["test"]:
            stage = "test"
        else:
            raise ValueError(f"Unknown tile id '{id}'")
        return osp.join(stage, id + ".laz")


class MiniNorwayALS(NorwayALS):
    _NUM_MINI = 2

    @property
    def all_cloud_ids(self):
        return {k: v[:self._NUM_MINI] for k, v in super().all_cloud_ids.items()}
