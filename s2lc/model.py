"""U-Net (SSL4EO-S12 ResNet50 encoder) and LightGBM classifiers behind one `predict()`."""

import os
import urllib.request
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch

from s2lc.classes import NODATA

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = Path(os.environ.get("S2LC_WEIGHTS", ROOT / "data/weights/model.pt"))
WEIGHTS_URL = os.environ.get(
    "S2LC_WEIGHTS_URL",
    "https://github.com/fakmalpradana/sentinel2-lc/releases/download/v0.1.0/model.pt",
)
SSL4EO = ROOT / "data/weights/ssl4eo_s2_resnet50_moco.pth"
N_OUT = 9  # logits for class ids 1..9
# our 13 channels -> SSL4EO input order B1 B2 B3 B4 B5 B6 B7 B8 B8A B9 B10 B11 B12.
# Bands map 1:1; the three indices reuse the B1/B9/B10 filters as a starting point.
SSL4EO_CHANNELS = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 0, 9, 10]


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def build_unet(pretrained: bool = False) -> smp.Unet:
    net = smp.Unet("resnet50", encoder_weights=None, in_channels=13, classes=N_OUT)
    if pretrained:
        sd = torch.load(SSL4EO, map_location="cpu", weights_only=True)
        sd = {k: v for k, v in sd.items() if not k.startswith("fc.")}
        sd["conv1.weight"] = sd["conv1.weight"][:, SSL4EO_CHANNELS]
        net.encoder.load_state_dict(sd)
    return net


def normalize(feats: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Standardize per channel; NaN (no observation) becomes 0 = the channel mean."""
    x = (feats - mean[:, None, None]) / std[:, None, None]
    return np.nan_to_num(x, nan=0.0).astype(np.float32)


_cache: dict = {}


def load(path: Path = WEIGHTS) -> dict:
    if path not in _cache:
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(WEIGHTS_URL, path)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if ck["kind"] == "unet":
            net = build_unet()
            net.load_state_dict(ck["state_dict"])
            ck["net"] = net.to(device()).eval()
        else:
            import lightgbm as lgb
            ck["booster"] = lgb.Booster(model_str=ck["model_str"])
        _cache[path] = ck
    return _cache[path]


@torch.no_grad()
def _unet_probs(net: smp.Unet, x: np.ndarray, tile: int = 512, overlap: int = 64) -> np.ndarray:
    """Tiled inference with overlap; logits averaged where tiles overlap."""
    c, h, w = x.shape
    ph, pw = max(h, tile), max(w, tile)
    ph, pw = -(-ph // 32) * 32, -(-pw // 32) * 32  # U-Net needs multiples of 32
    xp = np.zeros((c, ph, pw), np.float32)
    xp[:, :h, :w] = x
    acc = np.zeros((N_OUT, ph, pw), np.float32)
    cnt = np.zeros((ph, pw), np.float32)
    step = tile - overlap
    ys = sorted({*range(0, ph - tile + 1, step), ph - tile})
    xs = sorted({*range(0, pw - tile + 1, step), pw - tile})
    dev = device()
    for y in ys:
        for x0 in xs:
            t = torch.from_numpy(xp[None, :, y:y + tile, x0:x0 + tile]).to(dev)
            p = torch.softmax(net(t), 1)[0].float().cpu().numpy()
            acc[:, y:y + tile, x0:x0 + tile] += p
            cnt[y:y + tile, x0:x0 + tile] += 1
    return (acc / cnt)[:, :h, :w]


def predict(feats: np.ndarray, ck: dict | None = None) -> np.ndarray:
    """(13, H, W) features with NaN -> (H, W) uint8 class ids; NODATA where unobserved."""
    ck = ck or load()
    valid = ~np.isnan(feats).any(0)
    x = normalize(feats, ck["mean"], ck["std"])
    if ck["kind"] == "unet":
        cls = _unet_probs(ck["net"], x).argmax(0) + 1
    else:
        p = ck["booster"].predict(x.reshape(13, -1).T)
        cls = (p.argmax(1) + 1).reshape(feats.shape[1:])
    return np.where(valid, cls, NODATA).astype(np.uint8)
