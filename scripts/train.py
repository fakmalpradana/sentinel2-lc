"""Train the U-Net on data/chips.

    uv run python scripts/train.py --epochs 25
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from s2lc import imagery, model

CHIPS = Path("data/chips")
RUNS = Path("runs")


def split_of(lon: float, lat: float) -> str:
    """Spatial split by 1-degree cell, so neighbouring chips never straddle train/test."""
    h = (math.floor(lon) * 73856093 ^ math.floor(lat) * 19349663) % 100
    return "test" if h < 15 else "val" if h < 30 else "train"


def load_chips(split: str, tile: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """(N, 10, 256, 256) float16 reflectance, (N, 256, 256) uint8 labels (255 = ignore).

    1024 px chips are cut into 256 px tiles and cached as an on-disk memmap: the whole set
    does not fit in 16 GB next to the model.
    """
    files = []
    for f in sorted(CHIPS.glob("*.npz")):
        z = np.load(f)
        if split_of(float(z["lon"]), float(z["lat"])) == split:
            files.append(f)
    xp, yp = Path(f"data/cache/{split}_x.npy"), Path(f"data/cache/{split}_y.npy")
    names = Path(f"data/cache/{split}_files.txt")
    if names.exists() and names.read_text() == "\n".join(f.name for f in files):
        return np.load(xp, mmap_mode="r"), np.load(yp)
    xp.parent.mkdir(parents=True, exist_ok=True)
    n = (1024 // tile) ** 2
    x = np.lib.format.open_memmap(xp, "w+", np.float16, (len(files) * n, 10, tile, tile))
    y = np.zeros((len(files) * n, tile, tile), np.uint8)
    k = 0
    for f in files:
        z = np.load(f)
        d, lab = z["data"], z["label"]
        for i in range(0, d.shape[1], tile):
            for j in range(0, d.shape[2], tile):
                x[k], y[k] = d[:, i:i + tile, j:j + tile], lab[i:i + tile, j:j + tile]
                k += 1
    x.flush()
    np.save(yp, y)
    names.write_text("\n".join(f.name for f in files))
    return np.load(xp, mmap_mode="r"), y


def stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    idx = rng.choice(len(x), min(len(x), 300), replace=False)
    f = np.stack([imagery.features(x[i].astype(np.float32)) for i in idx])
    f = f.transpose(1, 0, 2, 3).reshape(13, -1)
    return np.nanmean(f, 1).astype(np.float32), np.nanstd(f, 1).astype(np.float32) + 1e-6


def gpu_features(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Torch twin of imagery.features + model.normalize, run on device per batch."""
    x = x.float()
    g, r, nir, swir = x[:, 1], x[:, 2], x[:, 6], x[:, 8]
    idx = torch.stack([(nir - r) / (nir + r + 1e-6), (g - nir) / (g + nir + 1e-6),
                       (swir - nir) / (swir + nir + 1e-6)], 1)
    f = (torch.cat([x, idx], 1) - mean[None, :, None, None]) / std[None, :, None, None]
    return torch.nan_to_num(f, nan=0.0)


def miou(conf: np.ndarray) -> float:
    iou = np.diag(conf) / (conf.sum(0) + conf.sum(1) - np.diag(conf) + 1e-9)
    present = conf.sum(1) > 0
    return float(iou[present].mean())


def train_unet(xtr, ytr, xva, yva, mean, std, epochs: int, bs: int) -> dict:
    dev = model.device()
    net = model.build_unet(pretrained=True).to(dev)
    m_t, s_t = torch.from_numpy(mean).to(dev), torch.from_numpy(std).to(dev)
    freq = np.bincount(ytr[ytr != 255].ravel(), minlength=10)[1:].astype(np.float64)
    # inverse-sqrt frequency; the 0.1% floor caps how far a rare class can be up-weighted
    w = 1 / np.sqrt(np.maximum(freq / freq.sum(), 1e-3))
    w /= w[freq > 0].mean()
    weight = torch.tensor(w, dtype=torch.float32, device=dev)
    print("class freq", freq.astype(int).tolist(), "weights", np.round(w, 2).tolist())
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    steps = epochs * math.ceil(len(xtr) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=steps, pct_start=0.05)
    best, best_ck = -1.0, None
    rng = np.random.default_rng(0)
    for ep in range(epochs):
        net.train()
        t0, tot = time.time(), 0.0
        for b in np.array_split(rng.permutation(len(xtr)), math.ceil(len(xtr) / bs)):
            x = torch.from_numpy(xtr[b]).to(dev)
            y = torch.from_numpy(ytr[b].astype(np.int64)).to(dev)
            y = torch.where(torch.isnan(x[:, 0].float()) | (y == 255), -1, y - 1)
            k = int(rng.integers(4))
            x, y = torch.rot90(x, k, (2, 3)), torch.rot90(y, k, (1, 2))
            if rng.random() < 0.5:
                x, y = x.flip(3), y.flip(2)
            loss = F.cross_entropy(net(gpu_features(x, m_t, s_t)), y, weight=weight,
                                   ignore_index=-1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(b)
        conf = confusion_unet(net, xva, yva, m_t, s_t, bs)
        score = miou(conf)
        oa = np.diag(conf).sum() / conf.sum()
        print(f"epoch {ep + 1}/{epochs} loss {tot / len(xtr):.4f} val mIoU {score:.4f} "
              f"OA {oa:.4f} {time.time() - t0:.0f}s", flush=True)
        if score > best:
            best = score
            best_ck = {"mean": mean, "std": std, "val_miou": score,
                       "state_dict": {k: v.detach().cpu().clone()
                                      for k, v in net.state_dict().items()}}
            torch.save(best_ck, RUNS / "unet.pt")
    return best_ck


@torch.no_grad()
def confusion_unet(net, x, y, m_t, s_t, bs) -> np.ndarray:
    net.eval()
    dev = m_t.device
    conf = np.zeros((9, 9), np.int64)
    for b in np.array_split(np.arange(len(x)), max(1, math.ceil(len(x) / bs))):
        xb = torch.from_numpy(x[b]).to(dev)
        pred = net(gpu_features(xb, m_t, s_t)).argmax(1).cpu().numpy()
        conf += confusion(pred + 1, y[b], np.isnan(x[b][:, 0].astype(np.float32)))
    return conf


def confusion(pred: np.ndarray, lab: np.ndarray, nan: np.ndarray) -> np.ndarray:
    """9x9 (rows = reference, cols = predicted) over labelled, observed pixels."""
    m = (lab != 255) & ~nan & (pred >= 1)
    return np.bincount((lab[m].astype(np.int64) - 1) * 9 + pred[m] - 1,
                       minlength=81).reshape(9, 9)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--bs", type=int, default=16)
    a = ap.parse_args()
    RUNS.mkdir(exist_ok=True)
    xtr, ytr = load_chips("train")
    xva, yva = load_chips("val")
    print("train", xtr.shape, "val", xva.shape, flush=True)
    mean, std = stats(xtr)
    train_unet(xtr, ytr, xva, yva, mean, std, a.epochs, a.bs)
    print("done", flush=True)
