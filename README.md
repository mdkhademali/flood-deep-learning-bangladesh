# Deep Learning-Based Multi-Temporal Flood Extent Mapping Using Sentinel-1 SAR: A GeoAI Framework for Flood Dynamics Assessment in Bangladesh**

![Python](https://img.shields.io/badge/python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![Status](https://img.shields.io/badge/status-research--in--progress-yellow)

---

## Overview

This repository implements an end-to-end GeoAI pipeline for automatic flood
extent mapping from Sentinel-1 SAR imagery over **Kurigram District,
Bangladesh**, one of the country's most flood-prone districts, situated
along the Brahmaputra-Jamuna river system. The pipeline covers data
acquisition (Google Earth Engine), SAR preprocessing, a Random
Forest/XGBoost baseline, a U-Net deep learning segmentation model,
rigorous accuracy assessment, multi-temporal flood dynamics, and
GIS-based exposure analysis.

## Research Question

Can a U-Net semantic segmentation model trained on Sentinel-1 SAR
backscatter meaningfully outperform a pixel-wise machine-learning baseline
for automatic flood extent mapping, and how did flood extent evolve across
the 2020 and 2022 monsoon flood events in Kurigram District?

## Objectives

1. Build a reproducible, GEE-to-Python pipeline for Sentinel-1 SAR flood mapping.
2. Establish a Random Forest / XGBoost pixel-wise baseline.
3. Train and evaluate a U-Net segmentation model.
4. Compare baseline vs. deep learning performance with rigorous, imbalance-aware metrics.
5. Quantify multi-temporal flood dynamics (persistence, expansion, recession).
6. Assess flood exposure by land-use class and administrative unit.

## Key Contributions

- A fully modular, config-driven pipeline (`configs/config.yaml`), no hard-coded paths or hyperparameters.
- A **spatially separated** train/val/test split that avoids the spatial-autocorrelation leakage common in naive remote-sensing ML pipelines.
- An explicit scientific-integrity policy: no fabricated results (see below).
- CPU-compatible by default, with automatic GPU use when available, plus a `--debug` fast-verification mode.

## Study Area

**Kurigram District**, Rangpur Division, Bangladesh (~2,245 km²), see
`docs/methodology.md` Section 3.1 for the full justification.

## Data Sources

See `docs/data_sources.md` for full provider/resolution/license details.
Primary input: Sentinel-1 GRD (VV+VH, 10 m). Supporting: JRC Global Surface
Water, SRTM DEM, CHIRPS, FAO GAUL boundaries.

## Methodology

Full methodology write-up: `docs/methodology.md`. Summary workflow:

```
GEE (Sentinel-1 acquisition + preprocessing)
        │
        ▼
Patch extraction + spatial train/val/test split
        │
        ├──► Feature engineering ──► Random Forest / XGBoost baseline
        │
        └──► Augmentation ──► U-Net training (BCE+Dice loss)
                    │
                    ▼
        Sliding-window inference on full scenes
                    │
                    ▼
        Multi-temporal flood maps (pre / peak / post)
                    │
                    ▼
        Flood dynamics + GIS exposure analysis
                    │
                    ▼
        Publication-quality figures + metrics
```

## Model Architecture

Standard encoder-decoder U-Net, 4 downsampling stages, batch norm, dropout
0.2 in the bottleneck, skip connections, combined BCE+Dice loss. See
`docs/methodology.md` Section 3.7 and `src/models/unet.py`.

## Training

```bash
# Fast pipeline smoke test (2 epochs, tiny subset)
python -m src.training.train_unet --debug \
    --image_path data/raw/Kurigram_S1_peakflood_2020_stack.tif \
    --label_path data/raw/Kurigram_flood_label_2020.tif

# Full U-Net training
python -m src.training.train_unet \
    --image_path data/raw/Kurigram_S1_peakflood_2020_stack.tif \
    --label_path data/raw/Kurigram_flood_label_2020.tif \
    --experiment B_vv_vh

# Baseline Random Forest / XGBoost
python -m src.training.train_baseline \
    --pre_vv data/raw/preflood_VV.tif --pre_vh data/raw/preflood_VH.tif \
    --peak_vv data/raw/peakflood_VV.tif --peak_vh data/raw/peakflood_VH.tif \
    --label data/raw/Kurigram_flood_label_2020.tif
```

## Evaluation

Metrics are computed with `src/evaluation/metrics.py` (accuracy, precision,
recall, F1, per-class IoU, mean IoU, Dice, Cohen's Kappa, accuracy alone is
intentionally not used as the headline metric due to flood/non-flood class
imbalance). Figures are generated with `src/evaluation/make_figures.py`.

## Results

| Model | Accuracy | Precision | Recall | F1 | Mean IoU | Dice | Kappa |
|---|---|---|---|---|---|---|---|
| Random Forest | *to be generated after running the pipeline* | | | | | | |
| XGBoost | *to be generated after running the pipeline* | | | | | | |
| U-Net | *to be generated after running the pipeline* | | | | | | |

The pipeline itself has been executed end-to-end on synthetic
placeholder rasters to verify correctness (all unit tests pass; see
`tests/`). No performance numbers are reported here because the model has
not yet been trained on real, exported Sentinel-1 data, per the project's
scientific-integrity policy, fabricated numbers are never substituted for
real results.

## Figures

15 publication-quality figures are specified and generated by
`src/evaluation/make_figures.py` and the mapping/exposure modules once real
data is available (study area map, workflow diagram, backscatter
comparison, training curves, confusion matrix, flood extent maps,
multi-temporal dynamics, exposure charts, etc.). All figures save at 300
DPI PNG + PDF to `outputs/figures/`.

## Reproducibility

1. Data provenance: `docs/data_sources.md`
2. Data preparation: `data/README.md`
3. Training: commands above, all hyperparameters in `configs/config.yaml`
4. Evaluation: `src/evaluation/metrics.py`, `src/evaluation/make_figures.py`
5. Mapping: `src/mapping/flood_mapping.py`
6. Random seed fixed at 42 throughout (`src/utils/common.set_seed`)
7. Run `pytest tests/` to verify the codebase (11 tests, all passing)

## Limitations

- Reference flood labels are a **SAR-threshold-derived proxy**, not
  field-validated ground truth (see `docs/methodology.md` 3.5). Reported
  metrics reflect agreement with this proxy, not absolute real-world accuracy.
- No economic-loss estimates are produced (would require validated asset data).
- SAR flood detection is subject to known confusions with wet soil, radar
  shadow/layover, and dense vegetation, see the Error Analysis workflow
  (Section 19 of the original project brief) for how these are investigated.
- Results in this README are placeholders pending a full training run on
  real exported Sentinel-1 data.

## Future Work

- Field-validated or very-high-resolution optical flood labels for a subset of dates.
- Ablation experiments (VV-only vs. VV+VH vs. VV+VH+derived features, configured in `configs/config.yaml` under `experiments`).
- Extension to additional flood-prone districts (e.g. Sirajganj, Jamalpur) for cross-region generalization testing.
- Integration of Sentinel-2 optical imagery for cloud-free windows.

## Citation

See `CITATION.cff`.

## Author

**Md Khadem Ali** Department of Geography and
Environment, National University, Bangladesh.
Portfolio: khademali.com · GitHub: [mdkhademali](https://github.com/mdkhademali)
