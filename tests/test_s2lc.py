from datetime import date

import numpy as np
import pytest
from odc.geo.geobox import GeoBox
from rasterio import features

from s2lc import imagery, pipeline
from s2lc.classes import IGNORE, consensus

SQUARE = {"type": "Polygon", "coordinates": [[[107.0, -6.3], [107.05, -6.3], [107.05, -6.25],
                                               [107.0, -6.25], [107.0, -6.3]]]}


def test_consensus_rules():
    wc = np.array([80, 10, 10, 95, 40, 50, 60, 20, 90, 10, 40])
    io = np.array([1, 2, 2, 4, 5, 7, 8, 11, 4, 5, 5])
    op = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2])
    assert consensus(wc, io, op).tolist() == [1, 2, 4, 3, 5, 6, 7, 8, 9, IGNORE, 4]


def test_quarters():
    assert imagery.quarter_range(2024, 4) == (date(2024, 10, 1), date(2025, 1, 1))
    pipeline.check_period(2024, 4, today=date(2025, 1, 1))
    with pytest.raises(pipeline.AoiError):
        pipeline.check_period(2025, 1, today=date(2025, 3, 31))  # not complete
    with pytest.raises(pipeline.AoiError):
        pipeline.check_period(2018, 4)


def test_aoi_validation():
    feat = {"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": SQUARE}]}
    assert pipeline.parse_aoi(feat).geom_type == "Polygon"
    with pytest.raises(pipeline.AoiError):  # a point has no area
        pipeline.parse_aoi({"type": "Point", "coordinates": [101.7, 3.1]})
    with pytest.raises(pipeline.AoiError):
        pipeline.parse_aoi({"type": "Polygon", "coordinates": [[[150, 0], [151, 0], [151, 1],
                                                               [150, 1], [150, 0]]]})
    bowtie = {"type": "Polygon", "coordinates": [[[107, -6], [107.1, -6.1], [107.1, -6],
                                                  [107, -6.1], [107, -6]]]}
    with pytest.raises(pipeline.AoiError):
        pipeline.parse_aoi(bowtie)


def test_area_and_summary_match_aoi():
    geom = pipeline.parse_aoi(SQUARE)
    gb: GeoBox = imagery.grid_for_bbox(geom.bounds)
    g = pipeline.to_crs(geom, gb.crs.to_wkt())
    inside = features.geometry_mask([g], gb.shape, gb.affine, invert=True)
    cls = np.where(np.arange(gb.shape[1])[None, :] < gb.shape[1] // 2, 2, 6).astype(np.uint8)
    cls = np.broadcast_to(cls, gb.shape).copy()
    s = pipeline.summarize(cls, inside, np.zeros(gb.shape, bool), 100.0)
    assert abs(s["area_km2"] - g.area / 1e6) / (g.area / 1e6) < 0.01
    assert abs(sum(c["percent"] for c in s["classes"]) - 100) < 0.1
    assert {c["id"] for c in s["classes"]} == {2, 6}
