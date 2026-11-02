# sentinel2-lc — training report v0.1.0

Training ran on 2026-11-02, overnight, on a MacBook Air M4 (16 GB, MPS).

## Data

- **Chips:** 334 locations of 10.24 km × 10.24 km.
  - 300 are random, stratified by island group, with rare classes oversampled.
  - 34 are bare-land and savanna hotspots: mines, volcanoes, salt flats, Sumba, Komodo.
- **Imagery:** two random 2021 quarters per location, giving 668 quarterly composites. Each
  composite uses at most 16 scenes, SCL cloud masking, a per-band median, and a
  previous-quarter gap fill. This is the same code path as inference.
- **Labels:** consensus of ESA WorldCover 2021, Esri/IO LULC 2021 and the Descals 2021 oil palm
  extent. Pixels where the maps disagree are ignored.
- **Split:** by 1° grid cell. Train has 7,936 tiles (256 px), val 1,536 and test 1,920. No
  test cell touches a training cell.

## Model

- U-Net with a ResNet50 encoder, initialised from SSL4EO-S12 MoCo Sentinel-2 weights.
- Input: 13 channels (10 bands + NDVI, NDWI, NDBI).
- Training: 25 epochs, batch 16, AdamW + one-cycle schedule (max lr 3e-4), class-weighted
  cross-entropy, flip/rot90 augmentation.
- Time: about 9 minutes per epoch, about 3.7 hours in total.
- Checkpoint: the best val mIoU, which was the last epoch (val OA 0.947, mIoU 0.744).

Val mIoU by epoch: 0.51, 0.45, 0.67, 0.65, 0.69, 0.63, 0.60, 0.61, 0.71, 0.71, 0.70, 0.72, 0.73,
0.54, 0.74, 0.73, 0.70, 0.73, 0.71, 0.74, 0.74, 0.72, 0.74, 0.72, 0.74.

## Test results (held-out 1° cells)

**Overall accuracy 0.950, mIoU 0.737**, over 69.2 M labelled test pixels.

| Class | Precision | Recall | F1 | Test pixels |
|---|---|---|---|---|
| Water | 0.994 | 0.981 | 0.988 | 6,654,313 |
| Tree cover | 0.975 | 0.958 | 0.967 | 43,597,441 |
| Mangrove | 0.925 | 0.934 | 0.929 | 4,043,317 |
| Oil palm | 0.798 | 0.888 | 0.841 | 4,939,484 |
| Cropland | 0.954 | 0.943 | 0.949 | 4,449,297 |
| Built-up | 0.975 | 0.943 | 0.959 | 1,382,217 |
| Bare | 0.015 | 0.751 | 0.029 | 410 |
| Shrub & grass | 0.851 | 0.906 | 0.877 | 4,011,961 |
| Herbaceous wetland | 0.562 | 0.844 | 0.674 | 139,097 |

Excluding Bare, which has only 410 test pixels, the mean F1 of the other 8 classes is 0.898.
The full confusion matrix is in `runs/eval.md`.

### What the numbers mean

- **They measure agreement with consensus reference maps, not field accuracy.** The test
  pixels are the "easy" ones where three independent maps agree. True accuracy on all
  pixels, including mixed and transitional land, will be lower. Measuring it needs independent
  Indonesian reference points, for example a few hundred photo-interpreted samples per class.
- **Bare land is not measurable yet.** The test cells contain only 410 bare pixels. The model
  does predict bare land (recall 0.75), but the precision figure is meaningless at this
  support.
- **Oil palm vs tree cover is the main real confusion.** About 1.0 M tree pixels are predicted
  as oil palm, and 0.5 M oil palm pixels as tree cover. Mature closed-canopy palm and forest
  look alike in a single quarterly median.
- **Herbaceous wetland** is over-predicted (precision 0.56). It is confused with shrub/grass
  and mangrove.

## Visual samples (test chips, 2021)

`docs/samples/*.png`: true colour | consensus reference (grey = maps disagree) | prediction.

- Oil palm block boundaries and riverine forest closely follow the reference. The river is
  now kept.
- Mangrove and the large river in Papua are nearly identical to the reference.
- The savanna/tree mosaic in Sumba is captured.
- Settlements along the Jawa north coast are detected but too blobby. Aquaculture ponds are
  missed when the quarter is cloudy.

## Known limits

1. Labels come from 2021 maps, so the model inherits their definitions. For example,
   WorldCover calls rice paddies cropland whether flooded or not.
2. There is one quarter per prediction and no temporal smoothing, so classes can flicker
   between quarters.
3. Narrow linear features (small rivers, roads) are under 2 pixels wide at 10 m and get lost.
4. Persistent cloud (Q4/Q1, highlands) yields class 0 after gap fill.

## Island review AOIs, Q3 2026

Five areas of about 900 km² each, classified for Q3 2026 with this model. The AOI files are in
`examples/islands/`. Outputs per AOI (not committed, `out/islands_2026q3/<aoi>/`):
- `landcover.tif`: COG with an embedded palette.
- `landcover.geojson`: polygons. Load the style from `docs/landcover_polygons_style.qml`.
- `composite.tif`: R, G, B, NIR reflectance × 10000.
- `summary.json`
- `quicklook.png`

| AOI | Run time (M4, home network) | Clear | Main classes |
|---|---|---|---|
| Jawa: Karawang–Bekasi | 4.4 min | 100 % | Cropland 52 %, Built-up 38 %, Shrub/grass 5 %, Tree 4 % |
| Sumatra: Siak–Pelalawan | 4.0 min | 100 % | Oil palm 57 %, Tree 37 %, Shrub/grass 4 %, Built-up 2 % |
| Kalimantan: Mahakam delta | 6.9 min | 97 % | Water 43 %, Tree 26 %, Mangrove 21 %, Shrub/grass 4 %, Wetland 3 % |
| Sulawesi: Makassar–Maros | 4.4 min | 99.9 % | Water 24 %, Built-up 24 %, Cropland 23 %, Shrub/grass 16 %, Tree 11 % |
| Papua: Timika lowland | 7.3 min | 89 % | Tree 67 %, Mangrove 10 %, Shrub/grass 8 %, Water 5 %, Bare 4 % |

First visual read, before the user's QGIS review:
- **Rivers, coastlines, the city of Makassar, Timika town and the Siak plantation grid are
  well delineated.**
- **Mahakam:** the aquaculture ponds (tambak) east of the delta come out as Water, because
  the scheme has no aquaculture class. Nipa/old mangrove in the delta interior comes out as
  Tree cover.
- **Siak:** acacia pulpwood plantations are folded into Tree cover or Oil palm, because there
  is no pulpwood class. Harvested pulpwood blocks come out as Shrub/grass or Cropland.
- **Karawang:** Built-up is likely over-estimated. Tree-lined villages and roadside ribbons
  merge into large built-up patches.
- **Timika:** the Ajkwa tailings deposit is a mix of Bare, Shrub/grass and Water, which is
  partly right because parts of it are revegetated. About 11 % stays unobserved because of
  persistent cloud.
