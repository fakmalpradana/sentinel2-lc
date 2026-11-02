"""Find Sentinel-2 L2A scenes for a quarter and build a cloud-masked median composite.

Training and inference both call `composite()`, so the model sees identical inputs.
"""

import warnings
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import odc.stac
from odc.geo.geobox import GeoBox
from pyproj import CRS, Transformer
from pystac_client import Client

STAC_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-c1-l2a"
BANDS = ["blue", "green", "red", "rededge1", "rededge2", "rededge3",
         "nir", "nir08", "swir16", "swir22"]  # B02 B03 B04 B05 B06 B07 B08 B8A B11 B12
CLEAR_SCL = (4, 5, 6)  # vegetation, not-vegetated, water; everything else is cloud/shadow/etc.
MAX_SCENES = 30
MAX_CLOUD = 80

_client: Client | None = None


@dataclass
class Composite:
    data: np.ndarray            # (10, H, W) float32 surface reflectance, NaN where no obs
    filled: np.ndarray          # (H, W) bool: value borrowed from the previous quarter
    geobox: GeoBox
    scenes: list[str] = field(default_factory=list)

    @property
    def valid(self) -> np.ndarray:
        return ~np.isnan(self.data[0])


def quarter_range(year: int, quarter: int) -> tuple[date, date]:
    if quarter not in (1, 2, 3, 4):
        raise ValueError("quarter must be 1..4")
    start = date(year, 3 * quarter - 2, 1)
    end = date(year + quarter // 4, (3 * quarter) % 12 + 1, 1)  # exclusive
    return start, end


def previous_quarter(year: int, quarter: int) -> tuple[int, int]:
    return (year, quarter - 1) if quarter > 1 else (year - 1, 4)


def utm_crs(lon: float, lat: float) -> CRS:
    return CRS.from_epsg((32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1)


def grid_for_bbox(bbox: tuple[float, float, float, float], res: int = 10) -> GeoBox:
    """10 m UTM grid covering a WGS84 bbox (minx, miny, maxx, maxy)."""
    crs = utm_crs((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    x0, y0, x1, y1 = Transformer.from_crs(4326, crs, always_xy=True).transform_bounds(*bbox)
    return GeoBox.from_bbox((x0, y0, x1, y1), crs=crs, resolution=res, tight=False)


def grid_for_point(lon: float, lat: float, size: int = 256, res: int = 10) -> GeoBox:
    crs = utm_crs(lon, lat)
    x, y = Transformer.from_crs(4326, crs, always_xy=True).transform(lon, lat)
    x, y = round(x / res) * res, round(y / res) * res
    half = size * res / 2
    return GeoBox.from_bbox((x - half, y - half, x + half, y + half), crs=crs, resolution=res)


def search(geobox: GeoBox, start: date, end: date) -> list:
    global _client
    _client = _client or Client.open(STAC_URL)
    bbox = geobox.geographic_extent.boundingbox
    items = _client.search(
        collections=[COLLECTION], bbox=tuple(bbox),
        datetime=f"{start.isoformat()}/{end.isoformat()}T00:00:00Z",
        query={"eo:cloud_cover": {"lt": MAX_CLOUD}},
    ).item_collection()
    items = [i for i in items if i.datetime.date() < end]
    return sorted(items, key=lambda i: i.properties["eo:cloud_cover"])[:MAX_SCENES]


def _median(items: list, geobox: GeoBox, pool: int) -> np.ndarray:
    """(10, H, W) median reflectance over clear pixels; NaN where never clear."""
    out = np.full((len(BANDS), *geobox.shape), np.nan, np.float32)
    if not items:
        return out
    rb = items[0].assets["red"].extra_fields["raster:bands"][0]
    scale, offset = rb.get("scale", 1e-4), rb.get("offset", 0.0)
    # ponytail: one scale/offset for the stack; C1 L2A is uniformly reprocessed (baseline 05.x,
    # offset -0.1). Mixing collections would need per-item scaling here.
    assert all(i.assets["red"].extra_fields["raster:bands"][0].get("offset", 0.0) == offset
               for i in items), "mixed BOA offsets in one stack"
    ds = odc.stac.load(items, bands=[*BANDS, "scl"], geobox=geobox, groupby="solar_day",
                       resampling={"scl": "nearest", "*": "bilinear"}, pool=pool)
    clear = np.isin(ds["scl"].values, CLEAR_SCL)
    for b, name in enumerate(BANDS):
        dn = ds[name].values
        v = np.where(clear & (dn > 0), dn * scale + offset, np.nan).astype(np.float32)
        if np.isfinite(v).any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns
                out[b] = np.nanmedian(v, axis=0)
    return out


def composite(geobox: GeoBox, year: int, quarter: int, gap_fill: bool = True,
              pool: int = 8) -> Composite:
    start, end = quarter_range(year, quarter)
    items = search(geobox, start, end)
    data = _median(items, geobox, pool)
    filled = np.zeros(geobox.shape, bool)
    scenes = [i.id for i in items]
    hole = np.isnan(data[0])
    if gap_fill and hole.any():
        py, pq = previous_quarter(year, quarter)
        pitems = search(geobox, *quarter_range(py, pq))
        prev = _median(pitems, geobox, pool)
        take = hole & ~np.isnan(prev[0])
        data[:, take] = prev[:, take]
        filled = take
        scenes += [i.id for i in pitems]
    return Composite(data, filled, geobox, scenes)


def features(data: np.ndarray) -> np.ndarray:
    """10 bands + NDVI, NDWI, NDBI = 13 channels; NaN stays NaN."""
    g, r, nir, swir = data[1], data[2], data[6], data[8]
    eps = 1e-6
    ndvi = (nir - r) / (nir + r + eps)
    ndwi = (g - nir) / (g + nir + eps)
    ndbi = (swir - nir) / (swir + nir + eps)
    return np.concatenate([data, np.stack([ndvi, ndwi, ndbi])]).astype(np.float32)


if __name__ == "__main__":
    assert quarter_range(2021, 1) == (date(2021, 1, 1), date(2021, 4, 1))
    assert quarter_range(2021, 4) == (date(2021, 10, 1), date(2022, 1, 1))
    assert previous_quarter(2021, 1) == (2020, 4)
    print("ok")
