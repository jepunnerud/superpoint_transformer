# src/datasets/norway_binary_config.py
import numpy as np

# IDs (uten .laz) – må matche filnavnene dine i raw/{train,val,test}/
TILES = {
    "train": [
        "32-1-468-145-43", 
        "32-1-468-145-44",
        ],
    "val":   [
        "32-1-468-145-42"
        ],
    "test":  [],
}

# mapping fra LAS classification (0-255) -> train id (0/1/2)
# 0..1 = gyldige, 2 = void/ignored
ID2TRAINID = np.full(256, 2, dtype=np.int64)

# eksempel: ground=2 i LAS -> 0, alt annet -> 1
ID2TRAINID[2] = 0
# alle andre blir 1 (hvis du vil)
mask = np.ones(256, dtype=bool)
mask[2] = False
ID2TRAINID[mask] = 1

CLASS_NAMES = ["ground", "not_ground", "ignored"]
CLASS_COLORS = [
    [140, 90, 60],   # ground
    [180, 180, 180], # not_ground
    [0, 0, 0],       # ignored
]

NOR_NUM_CLASSES = 2
