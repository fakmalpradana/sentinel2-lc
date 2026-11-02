# sentinel2-lc

Quarterly land cover classification of Indonesia from Sentinel-2. The user gives only an
area of interest (GeoJSON, or a polygon drawn on the map) plus a year and quarter (Q1–Q4).
The service then:

1. finds Sentinel-2 L2A scenes for that quarter,
2. builds a cloud-masked median composite,
3. classifies it into 9 land cover classes,
4. crops the result to the AOI and returns a Cloud-Optimized GeoTIFF, GeoJSON polygons and
   a per-class area summary.

This is the experimental second path for NARAGA's `landcover_detection` service. The
existing path (`pandala`) needs an orthophoto. This one needs only an AOI. See
[Merging into landcover_detection](#merging-into-landcover_detection).

## Classes

| id | Kelas | Class |
|---|---|---|
| 1 | Air | Water |
| 2 | Hutan & pohon | Tree cover |
| 3 | Mangrove | Mangrove |
| 4 | Perkebunan sawit | Oil palm |
| 5 | Pertanian | Cropland |
| 6 | Terbangun | Built-up |
| 7 | Lahan terbuka | Bare |
| 8 | Semak & rumput | Shrub & grass |
| 9 | Lahan basah | Herbaceous wetland |
| 0 | Tidak teramati | No clear observation (cloud) |

Raster value 255 marks pixels outside the AOI.

## Quick start

```bash
uv sync
uv run uvicorn s2lc.api:app --port 8000
```

Open <http://localhost:8000> to draw an AOI. You can also call the API directly:

```bash
curl -X POST localhost:8000/v1/classify -H 'Content-Type: application/json' \
  -d "{\"aoi\": $(cat examples/bekasi.geojson), \"year\": 2025, \"quarter\": 3}"
curl localhost:8000/v1/jobs/<job_id>
curl -O localhost:8000/v1/jobs/<job_id>/files/landcover.tif
```

On first use the model weights are downloaded from the GitHub release (`S2LC_WEIGHTS_URL`)
to `data/weights/model.pt` (`S2LC_WEIGHTS`).

### API

| Method | Path | |
|---|---|---|
| POST | `/v1/classify` | `{aoi, year, quarter, output_formats?: ["geotiff","geojson","image"], gap_fill?: true}` → `202 {job_id}`; `422` on an invalid AOI or period |
| GET | `/v1/jobs/{id}` | `{status, progress_percent, result_summary, files, error_code}` |
| GET | `/v1/jobs/{id}/files/{name}` | `landcover.tif`, `landcover.geojson`, `composite.tif` (R,G,B,NIR reflectance ×10000) |

Input rules:
- The AOI must be a valid Polygon or MultiPolygon inside Indonesia (lon 94–142, lat −11.5–6.5).
- The AOI must be at most `S2LC_MAX_AOI_KM2` (default 1000 km²).
- The quarter must be complete, and the year must be 2019 or later.

`result_summary` follows the contract's `LandcoverDetectionResult`
(`area_km2`, `classes[{name, area_km2, percent}]`) plus these fields:
- `clear_percent`, `filled_percent` and `nodata_percent`.
- `imagery`: the scenes used.
- `attribution`: show it to end users.

## How it works

**Imagery.** Scenes come from Sentinel-2 Collection 1 L2A Cloud-Optimized GeoTIFFs on the
Element 84 Earth Search STAC (`sentinel-2-c1-l2a`, AWS Open Data, no account needed).
- The 16 least-cloudy scenes of the quarter are loaded at 10 m in the AOI's UTM zone.
- Pixels whose scene classification (SCL) is not vegetation, bare or water are masked.
- The per-band median is taken over the masked stack.
- If more than 1 % of pixels were never clear in the quarter, those pixels take the previous
  quarter's median. This is reported as `filled_percent`; turn it off with `gap_fill: false`.
- Pixels still without data become class 0.

**Model.** The model is a U-Net whose ResNet50 encoder is pretrained on Sentinel-2 by
SSL4EO-S12 (MoCo).
- Input: 13 channels (10 bands B02–B12 + NDVI, NDWI, NDBI).
- Inference is tiled at 512 px with 64 px overlap.
- A 0.5 ha sieve removes speckle.

**Training data.** 300 chips of 10.24 km × 10.24 km are sampled across Sumatra, Jawa-Bali,
Kalimantan, Sulawesi, Nusa Tenggara, Maluku and Papua. Rare classes are oversampled. Each
chip uses two random quarters of 2021, built with exactly the inference compositing code.
Labels come from the consensus of three 2021 reference maps:
- ESA WorldCover
- Esri/Impact Observatory LULC
- global oil palm extent

A pixel gets a label only where the maps agree. Disagreements are ignored in the loss, which
removes most label noise. Chips are split train/val/test by 1° grid cell, so test areas are
never adjacent to training areas.

```bash
uv run python scripts/make_dataset.py --n 300 --workers 8   # needs data/raw/oilpalm.vrt
uv run python scripts/train.py --epochs 25
uv run python scripts/evaluate.py runs/unet.pt
```

The oil palm layer (Zenodo 10.5281/zenodo.13379129, `GlobalOilPalm_OP-extent.zip`) is
unzipped to `data/raw/oilpalm/tif/` and mosaicked with
`gdalbuildvrt data/raw/oilpalm.vrt data/raw/oilpalm/tif/*.tif`.

## Accuracy

See [REPORT.md](REPORT.md). The numbers measure agreement with consensus reference labels on
held-out regions. They are **not** field-validated accuracy; that needs independent
Indonesian reference points.

## Licensing

The code is Apache-2.0. Every input is cleared for commercial use. Attribution is required:
see [NOTICE](NOTICE). The API returns the attribution string with every result.

| Input | License |
|---|---|
| Sentinel-2 data | Copernicus free, full and open |
| ESA WorldCover 2021, Esri/IO LULC 2021, Descals oil palm 2021 | CC BY 4.0 |
| SSL4EO-S12 weights | CC BY 4.0 |
| Python dependencies | BSD / MIT / Apache-2.0 |

Deliberately not used:
- Google Earth Engine and Dynamic World via GEE: commercial use needs a paid license.
- Sentinel Hub: paid.
- AGPL libraries.
- KLHK maps: the license for model training is unclear.
- The demo page uses OSM tiles. Replace them before production, because the OSM tile policy
  forbids heavy use.

## Merging into landcover_detection

`s2lc.pipeline.run(aoi, year, quarter, out_dir)` is the seam.

1. **naraga-contract.** Make these changes:
   - `LandcoverDetectionParams.model` becomes `[pandala, sentinel2]`.
   - `orthophoto` is required only for `pandala`.
   - `sentinel2` requires `period: {year, quarter}`.
   - `LandcoverDetectionResult` gains `imagery` and `attribution`.
   - Credits for `sentinel2` are estimated from AOI area.
2. **naraga-land-cover-detection.** `run_job()` branches on `params.model`. For `sentinel2`
   it downloads the AOI dataset from its signed URL and runs `pipeline.run` under
   `asyncio.to_thread`. It then PUTs the files to `output_upload_urls`. The service's outbox,
   sequencing and cancellation rules stay as they are.
3. **naraga-fe.** Add a source toggle (Orthophoto / Sentinel-2), an AOI draw tool and a
   year + Q1–Q4 picker.
