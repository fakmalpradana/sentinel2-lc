"""Run the pipeline on example AOIs and write side-by-side PNGs (true colour | land cover).

    uv run python scripts/preview.py 2025 3 examples/*.geojson
"""

import json
import sys
from pathlib import Path

import numpy as np
import rasterio

from s2lc import imagery, pipeline
from s2lc.classes import CLASSES


def png(path: Path, rgb: np.ndarray) -> None:
    h, w = rgb.shape[1:]
    with rasterio.open(path, "w", driver="PNG", width=w, height=h, count=3, dtype="uint8") as d:
        d.write(rgb)


if __name__ == "__main__":
    year, quarter = int(sys.argv[1]), int(sys.argv[2])
    out = Path("docs")
    out.mkdir(exist_ok=True)
    for f in sys.argv[3:]:
        name = Path(f).stem
        aoi = json.loads(Path(f).read_text())
        files, summary = pipeline.run(aoi, year, quarter, Path("out") / name)
        with rasterio.open(files[0]) as src:
            cls = src.read(1)
            gb = imagery.grid_for_bbox(pipeline.parse_aoi(aoi).bounds)
            assert gb.shape == cls.shape
        c = imagery.composite(gb, year, quarter)
        rgb = np.nan_to_num(c.data[[2, 1, 0]] / 0.3 * 255).clip(0, 255).astype(np.uint8)
        lut = np.zeros((256, 3), np.uint8)
        for k, v in CLASSES.items():
            lut[k] = v[2]
        lut[255] = (255, 255, 255)
        lc = lut[cls].transpose(2, 0, 1)
        gap = np.full((3, cls.shape[0], 8), 255, np.uint8)
        png(out / f"{name}_{year}q{quarter}.png", np.concatenate([rgb, gap, lc], 2))
        (out / f"{name}_{year}q{quarter}.json").write_text(json.dumps(summary, indent=1))
        top = sorted(summary["classes"], key=lambda c: -c["percent"])[:4]
        print(name, summary["area_km2"], "km2", [(c["name_en"], c["percent"]) for c in top])
