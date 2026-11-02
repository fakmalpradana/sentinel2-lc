"""AOI + year + quarter -> land cover raster, polygons and summary. The merge seam:
landcover_detection's run_job() calls `run()` for model `sentinel2`."""

import json
import os
from collections.abc import Callable
from datetime import date
from pathlib import Path

import numpy as np
import rasterio
import shapely
from odc.geo.geobox import GeoBox, GeoboxTiles
from pyproj import Transformer
from rasterio import features
from shapely.geometry import mapping, shape
from shapely.ops import unary_union

from s2lc import imagery, model
from s2lc.classes import CLASSES, NODATA

INDONESIA_BBOX = (94.0, -11.5, 142.0, 6.5)
MAX_AOI_KM2 = float(os.environ.get("S2LC_MAX_AOI_KM2", 1000))
MIN_YEAR = 2019
SIEVE_PX = 50  # 0.5 ha minimum mapping unit at 10 m
BLOCK, HALO = 1024, 64
OUTSIDE = 255  # raster nodata: outside the AOI
IMAGE_BANDS = ["red", "green", "blue", "nir"]  # composite export, true colour + NIR
ATTRIBUTION = (
    "Contains modified Copernicus Sentinel data {year}. Model trained with labels derived "
    "from ESA WorldCover 2021 (CC BY 4.0), Esri/Impact Observatory LULC 2021 (CC BY 4.0) and "
    "Descals et al. 2024 global oil palm extent (CC BY 4.0)."
)


class AoiError(ValueError):
    pass


def parse_aoi(geojson: dict):
    """Feature / FeatureCollection / bare geometry -> one valid (Multi)Polygon in WGS84."""
    t = geojson.get("type")
    if t == "FeatureCollection":
        geoms = [shape(f["geometry"]) for f in geojson.get("features", [])]
    elif t == "Feature":
        geoms = [shape(geojson["geometry"])]
    else:
        geoms = [shape(geojson)]
    if not geoms:
        raise AoiError("AOI has no geometry")
    geom = unary_union(geoms)
    if geom.geom_type not in ("Polygon", "MultiPolygon") or geom.is_empty:
        raise AoiError("AOI must be a Polygon or MultiPolygon")
    if not geom.is_valid:
        raise AoiError("AOI geometry is invalid (self-intersection?)")
    x0, y0, x1, y1 = geom.bounds
    bx0, by0, bx1, by1 = INDONESIA_BBOX
    if x0 < bx0 or y0 < by0 or x1 > bx1 or y1 > by1:
        raise AoiError("AOI must lie inside Indonesia (lon 94..142, lat -11.5..6.5)")
    return geom


def check_period(year: int, quarter: int, today: date | None = None) -> None:
    if quarter not in (1, 2, 3, 4):
        raise AoiError("quarter must be 1, 2, 3 or 4")
    if year < MIN_YEAR:
        raise AoiError(f"year must be >= {MIN_YEAR}")
    if imagery.quarter_range(year, quarter)[1] > (today or date.today()):
        raise AoiError(f"Q{quarter} {year} is not complete yet")


def reproject(geom, src, dst):
    t = Transformer.from_crs(src, dst, always_xy=True)
    return shapely.transform(geom, lambda x, y: t.transform(x, y), interleaved=False)


def to_crs(geom, crs):
    return reproject(geom, 4326, crs)


def classify_grid(geobox: GeoBox, year: int, quarter: int, gap_fill: bool,
                  progress: Callable[[int], None] = lambda p: None):
    """Composite + predict block by block (with a halo so block edges keep context)."""
    cls = np.full(geobox.shape, NODATA, np.uint8)
    filled = np.zeros(geobox.shape, bool)
    image = np.zeros((len(IMAGE_BANDS), *geobox.shape), np.uint16)
    scenes: set[str] = set()
    tiles = GeoboxTiles(geobox, (BLOCK, BLOCK))
    idx = [(i, j) for i in range(tiles.shape[0]) for j in range(tiles.shape[1])]
    for k, (i, j) in enumerate(idx):
        sub = tiles[i, j]
        padded = sub.pad(HALO)
        c = imagery.composite(padded, year, quarter, gap_fill=gap_fill)
        pred = model.predict(imagery.features(c.data))
        h, w = sub.shape
        y0, x0 = i * BLOCK, j * BLOCK
        cls[y0:y0 + h, x0:x0 + w] = pred[HALO:HALO + h, HALO:HALO + w]
        filled[y0:y0 + h, x0:x0 + w] = c.filled[HALO:HALO + h, HALO:HALO + w]
        refl = c.data[[imagery.BANDS.index(b) for b in IMAGE_BANDS], HALO:HALO + h, HALO:HALO + w]
        image[:, y0:y0 + h, x0:x0 + w] = np.nan_to_num(refl * 10000, nan=0).clip(1, 65535)
        image[:, y0:y0 + h, x0:x0 + w][:, np.isnan(refl[0])] = 0
        scenes.update(c.scenes)
        progress(10 + int(80 * (k + 1) / len(idx)))
    return cls, filled, image, sorted(scenes)


def summarize(cls: np.ndarray, inside: np.ndarray, filled: np.ndarray, px_m2: float) -> dict:
    n = int(inside.sum())
    classes = []
    for cid, (name_id, name_en, _) in CLASSES.items():
        c = int(((cls == cid) & inside).sum())
        if c:
            classes.append({"id": cid, "name": name_id, "name_en": name_en,
                            "area_km2": round(c * px_m2 / 1e6, 4),
                            "percent": round(100 * c / n, 3)})
    nodata = int(((cls == NODATA) & inside).sum())
    return {
        "area_km2": round(n * px_m2 / 1e6, 4),
        "classes": classes,
        "clear_percent": round(100 * (1 - (nodata + (filled & inside).sum()) / n), 2),
        "filled_percent": round(100 * (filled & inside).sum() / n, 2),
        "nodata_percent": round(100 * nodata / n, 2),
    }


def write_cog(path: Path, cls: np.ndarray, geobox: GeoBox) -> None:
    h, w = cls.shape
    with rasterio.open(path, "w", driver="COG", width=w, height=h, count=1, dtype="uint8",
                       crs=geobox.crs.to_wkt(), transform=geobox.affine, nodata=OUTSIDE,
                       compress="deflate") as dst:
        dst.write(cls, 1)
        dst.write_colormap(1, {k: (*v[2], 255) for k, v in CLASSES.items()})


def write_image(path: Path, image: np.ndarray, geobox: GeoBox) -> None:
    """The composite the model saw: reflectance x 10000, 0 = no observation."""
    h, w = image.shape[1:]
    with rasterio.open(path, "w", driver="COG", width=w, height=h, count=len(image),
                       dtype="uint16", crs=geobox.crs.to_wkt(), transform=geobox.affine,
                       nodata=0, compress="deflate") as dst:
        dst.write(image)
        dst.descriptions = tuple(IMAGE_BANDS)


def write_geojson(path: Path, cls: np.ndarray, inside: np.ndarray, geobox: GeoBox) -> None:
    feats = []
    for geom, v in features.shapes(cls, mask=inside, transform=geobox.affine):
        g = shape(geom).simplify(5)  # metres, in UTM
        name_id, name_en, _ = CLASSES[int(v)]
        wgs = reproject(g, geobox.crs.to_wkt(), 4326)
        feats.append({"type": "Feature", "geometry": mapping(wgs),
                      "properties": {"class_id": int(v), "class": name_id,
                                     "class_en": name_en, "area_ha": round(g.area / 1e4, 3)}})
    path.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))


def run(aoi: dict, year: int, quarter: int, out_dir: Path,
        formats: tuple[str, ...] = ("geotiff", "geojson"), gap_fill: bool = True,
        progress: Callable[[int], None] = lambda p: None) -> tuple[list[Path], dict]:
    check_period(year, quarter)
    geom = parse_aoi(aoi)
    geobox = imagery.grid_for_bbox(geom.bounds)
    geom_utm = to_crs(geom, geobox.crs.to_wkt())
    if geom_utm.area / 1e6 > MAX_AOI_KM2:
        raise AoiError(f"AOI is {geom_utm.area / 1e6:.0f} km2; max is {MAX_AOI_KM2:.0f} km2")
    progress(5)

    cls, filled, image, scenes = classify_grid(geobox, year, quarter, gap_fill, progress)
    inside = features.geometry_mask([geom_utm], geobox.shape, geobox.affine, invert=True)
    cls = features.sieve(cls, SIEVE_PX, mask=inside)
    cls[~inside] = OUTSIDE

    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    if "geotiff" in formats:
        files.append(out_dir / "landcover.tif")
        write_cog(files[-1], cls, geobox)
    if "geojson" in formats:
        files.append(out_dir / "landcover.geojson")
        write_geojson(files[-1], cls, inside, geobox)
    if "image" in formats:
        files.append(out_dir / "composite.tif")
        write_image(files[-1], image, geobox)
    progress(98)

    res = abs(geobox.resolution.x)
    summary = summarize(cls, inside, filled, res * res)
    summary["imagery"] = {"source": "Sentinel-2 L2A (Copernicus)", "year": year,
                          "quarter": quarter, "scene_count": len(scenes), "scenes": scenes,
                          "gap_fill": gap_fill}
    summary["attribution"] = ATTRIBUTION.format(year=year)
    summary["crs"] = str(geobox.crs)
    return files, summary
