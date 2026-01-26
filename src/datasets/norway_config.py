import numpy as np

# 1) Splits: legg inn FILNAVN uten endelse (stem) hvis du vil
TILES = {
    "train": [
        "32-1-468-145-63",   # eksempel: filen 32-1-468-145-63.laz
        "32-1-468-145-64",
    ],
    "val": [
        "32-1-468-145-65",
    ],
    "test": []
}

# 2) Klasser (eksempel – bytt til dine)
NOR_NUM_CLASSES = 2 
CLASS_NAMES = ["Ground", "NotGround"]

# 3) Mapping fra LAS "classification" (0..255) -> train id
IGNORE = NOR_NUM_CLASSES
ID2TRAINID = np.full(256, 1, dtype=np.int64)  # default = NotGround
ID2TRAINID[2] = 0                             # Ground

# 4) Farger (valgfritt)
CLASS_COLORS = np.asarray([
    [243, 214, 171],  # ground
    [214,  66,  54],  # not ground
])