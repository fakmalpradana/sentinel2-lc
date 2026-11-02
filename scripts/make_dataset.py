"""Sample 10.24 km chips across Indonesia: 2021 quarterly composites + consensus labels.

Chips are 1024 px because Sentinel-2 COGs use 1024 px blocks: a smaller window downloads
almost the same bytes. Training cuts each chip into 16 tiles of 256 px.

    uv run python scripts/make_dataset.py --n 300 --workers 8

Resumable: chip i is deterministic (seeded by i) and skipped if its file exists.
"""

import argparse
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from s2lc import imagery, labels

OUT = Path("data/chips")
# (minlon, minlat, maxlon, maxlat), sampling weight
REGIONS = {
    "sumatra": ((95.0, -6.0, 106.0, 5.9), 0.22),
    "jawa_bali": ((105.0, -8.9, 115.8, -5.8), 0.18),
    "kalimantan": ((108.5, -4.3, 119.0, 4.4), 0.20),
    "sulawesi": ((118.8, -5.8, 125.3, 1.9), 0.14),
    "nusa_tenggara": ((115.8, -10.5, 125.2, -8.0), 0.08),
    "maluku": ((124.0, -8.3, 135.0, 2.6), 0.06),
    "papua": ((130.8, -9.2, 141.0, -0.2), 0.12),
}
RARE = (3, 4, 6, 7, 8, 9)  # mangrove, oil palm, built-up, bare, shrub, wetland
YEAR = 2021
CHIP = 1024


def pick(rng: np.random.Generator) -> tuple[str, float, float]:
    names = list(REGIONS)
    w = np.array([REGIONS[n][1] for n in names])
    name = names[rng.choice(len(names), p=w / w.sum())]
    x0, y0, x1, y1 = REGIONS[name][0]
    return name, rng.uniform(x0, x1), rng.uniform(y0, y1)


def accept(lab: np.ndarray, rng: np.random.Generator) -> bool:
    known = lab != 255
    if known.mean() < 0.25 or (lab == 1).mean() > 0.9:
        return False
    if any((lab == c).mean() >= 0.03 for c in RARE):
        return True
    return rng.random() < 0.35


def make(i: int) -> str:
    rng = np.random.default_rng(i)
    for _ in range(40):
        region, lon, lat = pick(rng)
        gb = imagery.grid_for_point(lon, lat, size=CHIP)
        lab = labels.read_labels(gb)
        if accept(lab, rng):
            break
    else:
        return f"{i}: no chip accepted"
    for q in rng.choice([1, 2, 3, 4], size=2, replace=False):
        path = OUT / f"{i:05d}_q{q}.npz"
        if path.exists():
            continue
        c = imagery.composite(gb, YEAR, int(q), pool=4)
        np.savez_compressed(path, data=c.data.astype(np.float16), filled=c.filled, label=lab,
                            lon=lon, lat=lat, quarter=q, region=region,
                            n_scenes=len(c.scenes))
    return f"{i}: {region} {lon:.3f},{lat:.3f}"


def safe(i: int) -> str:
    if len(list(OUT.glob(f"{i:05d}_q*.npz"))) >= 2:
        return f"{i}: exists"
    for attempt in range(3):
        try:
            return make(i)
        except Exception:
            err = traceback.format_exc(limit=1).strip().splitlines()[-1]
            time.sleep(5 * (attempt + 1))
    return f"{i}: FAILED {err}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with Pool(a.workers) as pool:
        for k, msg in enumerate(pool.imap_unordered(safe, range(a.start, a.start + a.n)), 1):
            print(f"[{k}/{a.n} {time.time() - t0:.0f}s] {msg}", flush=True)
    sys.exit(0)
