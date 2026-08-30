# Data Directory

This project does not commit large raster files to the repository (see
`.gitignore`). Instead, this folder documents how to (re)generate the data.

## `data/raw/`

Populate this folder with GeoTIFFs exported from the Google Earth Engine
scripts in `gee/`:

- `Kurigram_S1_preflood_<event>.tif`, pre-flood VV/VH composite
- `Kurigram_S1_peakflood_<event>.tif`, peak-flood VV/VH (+ derived bands) composite
- `Kurigram_S1_postflood_<event>.tif`, post-flood VV/VH composite
- `Kurigram_S1_change_<event>.tif`, VV/VH temporal change bands
- `Kurigram_flood_label_<event>.tif`, reference/proxy flood label
- `Kurigram_permanent_water_mask.tif`, JRC-derived permanent water mask

Run `gee/sentinel1_preprocessing.js` and `gee/flood_dataset_export.js` in the
[GEE Code Editor](https://code.earthengine.google.com), which will export
these files to a Google Drive folder (`flood_deep_learning_bd/` by default).
Download them into `data/raw/`.

## `data/processed/`

Populated automatically by the data pipeline (patch extraction, feature
stacking), see `src/data/patch_dataset.py` and `src/features/sar_features.py`.
Nothing here should be created manually.

## Notes on data provenance

| Dataset | Provider | Resolution | Purpose |
|---|---|---|---|
| Sentinel-1 GRD (IW, VV+VH) | ESA Copernicus (via GEE `COPERNICUS/S1_GRD`) | 10 m | Primary SAR input |
| JRC Global Surface Water | EC Joint Research Centre (`JRC/GSW1_4/GlobalSurfaceWater`) | 30 m | Permanent water mask for reference labels & error analysis |
| SRTM DEM | USGS (`USGS/SRTMGL1_003`) | 30 m | Terrain context (optional) |
| CHIRPS Daily | UCSB Climate Hazards Group (`UCSB-CHG/CHIRPS/DAILY`) | ~5.5 km | Rainfall context for event selection (optional) |
| FAO GAUL admin boundaries | FAO (`FAO/GAUL/2015/level2`) | vector | Administrative overlay for exposure analysis |

All datasets are accessed through Google Earth Engine and are openly
licensed for research use; see each provider's terms for redistribution.
