"""Read the 2021 reference maps onto a chip grid and combine them into consensus labels.

Training only. All three sources are CC BY 4.0 (see NOTICE).
"""

import math
from pathlib import Path

import numpy as np
import planetary_computer
import rasterio
from odc.geo.geobox import GeoBox
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from s2lc.classes import consensus

WC_URL = ("https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
          "ESA_WorldCover_10m_2021_v200_{tile}_Map.tif")
# Esri/IO tiles are one file per UTM zone + latitude band. Building the URL directly avoids
# the Planetary Computer STAC search, which rate-limits (429/503) under parallel load.
IO_URL = ("https://ai4edataeuwest.blob.core.windows.net/io-lulc/io-annual-lulc-v02/"
          "{tile}_20210101-20220101.tif")
OILPALM_VRT = Path(__file__).resolve().parent.parent / "data/raw/oilpalm.vrt"


def _warp(href: str, geobox: GeoBox) -> np.ndarray:
    h, w = geobox.shape
    with rasterio.open(href) as src, WarpedVRT(
        src, crs=geobox.crs.to_wkt(), transform=geobox.affine, width=w, height=h,
        resampling=Resampling.nearest,
    ) as vrt:  # no dst nodata: GDAL would bump valid 0s (oil palm "other") to 1
        return vrt.read(1)


def _mosaic(hrefs: list[str], geobox: GeoBox) -> np.ndarray:
    out = np.zeros(geobox.shape, np.uint8)
    for href in hrefs:
        try:
            a = _warp(href, geobox)
        except rasterio.errors.RasterioIOError:  # WorldCover has no file for all-sea tiles
            continue
        out = np.where(out == 0, a, out)
    return out


def worldcover_tiles(bbox: tuple[float, float, float, float]) -> list[str]:
    """3x3 degree tiles named by their lower-left corner, e.g. S09E105."""
    tiles = []
    for lat in range(math.floor(bbox[1] / 3) * 3, math.ceil(bbox[3] / 3) * 3, 3):
        for lon in range(math.floor(bbox[0] / 3) * 3, math.ceil(bbox[2] / 3) * 3, 3):
            ns = f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}"
            ew = f"{'E' if lon >= 0 else 'W'}{abs(lon):03d}"
            tiles.append(ns + ew)
    return tiles


def io_tiles(bbox: tuple[float, float, float, float]) -> list[str]:
    """UTM zone + MGRS latitude band of each bbox corner, e.g. 48M."""
    return sorted({f"{int((x + 180) // 6) + 1}{'CDEFGHJKLMNPQRSTUVWX'[int((y + 80) // 8)]}"
                   for x in bbox[0::2] for y in bbox[1::2]})


def read_labels(geobox: GeoBox) -> np.ndarray:
    bbox = tuple(geobox.geographic_extent.boundingbox)
    wc = _mosaic([WC_URL.format(tile=t) for t in worldcover_tiles(bbox)], geobox)
    io = _mosaic([planetary_computer.sign(IO_URL.format(tile=t)) for t in io_tiles(bbox)],
                 geobox)
    op = _warp(str(OILPALM_VRT), geobox)
    return consensus(wc, io, op)


if __name__ == "__main__":
    assert worldcover_tiles((106.9, -6.3, 107.0, -6.2)) == ["S09E105"]
    assert worldcover_tiles((104.5, -0.5, 105.5, 0.5)) == ["S03E102", "S03E105",
                                                           "N00E102", "N00E105"]
    assert io_tiles((106.9, -6.3, 107.0, -6.2)) == ["48M"]
    assert io_tiles((101.9, -0.1, 102.1, 0.1)) == ["47M", "47N", "48M", "48N"]
    print("ok")
