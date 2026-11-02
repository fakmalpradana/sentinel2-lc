"""Output classes and the label-consensus rule used to build training labels."""

import numpy as np

IGNORE = 255
NODATA = 0  # inference only: no clear Sentinel-2 observation

# id: (Bahasa Indonesia, English, RGB)
CLASSES: dict[int, tuple[str, str, tuple[int, int, int]]] = {
    0: ("Tidak teramati", "No data (cloud)", (0, 0, 0)),
    1: ("Air", "Water", (65, 155, 223)),
    2: ("Hutan & pohon", "Tree cover", (57, 125, 73)),
    3: ("Mangrove", "Mangrove", (0, 207, 117)),
    4: ("Perkebunan sawit", "Oil palm", (196, 160, 0)),
    5: ("Pertanian", "Cropland", (228, 150, 53)),
    6: ("Terbangun", "Built-up", (196, 40, 27)),
    7: ("Lahan terbuka", "Bare", (165, 155, 143)),
    8: ("Semak & rumput", "Shrub & grass", (183, 220, 120)),
    9: ("Lahan basah", "Herbaceous wetland", (0, 150, 160)),
}
N_CLASSES = len(CLASSES)  # model outputs logits for ids 1..9 at index id-1 (0 is never predicted)


def consensus(wc: np.ndarray, io: np.ndarray, op: np.ndarray) -> np.ndarray:
    """Training label where ESA WorldCover 2021, Esri/IO LULC 2021 and the oil palm map agree.

    wc: WorldCover codes, io: Esri/IO codes, op: oil palm extent (0 none, 1 industrial,
    2 smallholder). Disagreement becomes IGNORE so label noise never reaches the loss.
    """
    palm = op > 0
    out = np.full(wc.shape, IGNORE, np.uint8)
    rules = [
        (1, (wc == 80) & (io == 1)),
        (2, (wc == 10) & (io == 2) & ~palm),
        (3, (wc == 95) & np.isin(io, (2, 4))),
        (4, palm & np.isin(wc, (10, 20, 30, 40))),
        (5, (wc == 40) & (io == 5) & ~palm),
        (6, (wc == 50) & (io == 7)),
        (7, (wc == 60) & (io == 8)),
        (8, np.isin(wc, (20, 30)) & (io == 11) & ~palm),
        (9, (wc == 90) & (io == 4)),
    ]
    for cid, m in rules:
        out[m] = cid
    return out


if __name__ == "__main__":
    wc = np.array([80, 10, 10, 95, 40, 50, 60, 20, 90, 10, 40])
    io = np.array([1, 2, 2, 4, 5, 7, 8, 11, 4, 5, 5])
    op = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2])
    assert consensus(wc, io, op).tolist() == [1, 2, 4, 3, 5, 6, 7, 8, 9, IGNORE, 4]
    print("ok")
