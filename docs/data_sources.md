# Data Sources

## Primary Remote Sensing

### Sentinel-1 SAR (C-band)
- **Provider**: European Space Agency / Copernicus Programme
- **GEE asset**: `COPERNICUS/S1_GRD`
- **Mode**: Interferometric Wide (IW) swath
- **Polarizations**: VV, VH (dual-pol)
- **Spatial resolution**: 10 m (Ground Range Detected, terrain-corrected)
- **Temporal resolution**: ~6–12 day revisit over Bangladesh (varies by orbit availability)
- **Acquisition period used**: 2020 and 2022 monsoon flood events (see `configs/config.yaml`)
- **Preprocessing applied**: orbit-consistent filtering (descending pass), focal-median
  speckle filtering, dB-scale composites, AOI clipping
- **Source URL**: https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD
- **License**: Copernicus open data policy (free, full, open access)

## Supporting Datasets

### JRC Global Surface Water
- **Provider**: European Commission Joint Research Centre
- **GEE asset**: `JRC/GSW1_4/GlobalSurfaceWater`
- **Resolution**: 30 m
- **Variables used**: `occurrence` band (% of time water was detected, 1984–2021)
- **Purpose**: distinguishes permanent water bodies from flood water in reference
  labels and error analysis
- **Source URL**: https://developers.google.com/earth-engine/datasets/catalog/JRC_GSW1_4_GlobalSurfaceWater
- **License**: freely available for research/non-commercial use

### SRTM Digital Elevation Model
- **Provider**: USGS / NASA
- **GEE asset**: `USGS/SRTMGL1_003`
- **Resolution**: 30 m
- **Purpose**: terrain context; optional slope/elevation covariates for
  interpreting flood extent patterns (e.g. char land, floodplain topography)
- **Source URL**: https://developers.google.com/earth-engine/datasets/catalog/USGS_SRTMGL1_003
- **License**: public domain

### CHIRPS Daily Precipitation
- **Provider**: UC Santa Barbara Climate Hazards Group
- **GEE asset**: `UCSB-CHG/CHIRPS/DAILY`
- **Resolution**: ~5.5 km
- **Purpose**: contextual rainfall record used to justify flood-event date windows
- **Source URL**: https://developers.google.com/earth-engine/datasets/catalog/UCSB-CHG_CHIRPS_DAILY
- **License**: public domain (US Government work)

### FAO GAUL Administrative Boundaries
- **Provider**: Food and Agriculture Organization of the United Nations
- **GEE asset**: `FAO/GAUL/2015/level2`
- **Purpose**: district/upazila-level exposure analysis (Figure 15)
- **Source URL**: https://developers.google.com/earth-engine/datasets/catalog/FAO_GAUL_2015_level2
- **License**: free for use, FAO attribution requested

### Sentinel-2 (optional, supporting visual context)
- **Provider**: ESA / Copernicus
- **GEE asset**: `COPERNICUS/S2_SR_HARMONIZED`
- **Resolution**: 10–20 m (used bands)
- **Purpose**: optional true-color context imagery for study-area figures;
  not used as a model input due to persistent monsoon cloud cover during
  flood events, which is precisely why SAR is the primary sensor for this project
- **Source URL**: https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED
- **License**: Copernicus open data policy
