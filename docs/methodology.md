# Methodology

## 3.1 Study Area

Kurigram District, located in Rangpur Division in northern Bangladesh, was
selected as the study area. Kurigram lies along the Brahmaputra-Jamuna river
system and is characterized by extensive char (riverine sandbar) land, low
floodplain elevation, and a monsoon-dominated hydrological regime. The
district is among the most flood-affected in Bangladesh, experiencing
significant riverine flooding in most monsoon seasons, including the
well-documented 2020 and 2022 events used in this study. The area (~2,245
km²) is large enough to capture meaningful flood dynamics while remaining
computationally tractable for patch-based deep learning on limited hardware.
Sentinel-1 coverage over the district is dense and consistent across
ascending/descending orbits, supporting reliable multi-temporal composites.

## 3.2 Data Acquisition

Sentinel-1 IW GRD scenes (VV+VH, descending orbit) were acquired via Google
Earth Engine for three date windows per flood event: pre-flood, peak-flood,
and post-flood (see `configs/config.yaml` for exact windows). A single,
consistent orbit pass was enforced across all composites to minimize
incidence-angle-related backscatter variation unrelated to inundation.
Supporting datasets (JRC Global Surface Water, SRTM DEM, CHIRPS, FAO GAUL
boundaries) were acquired to support labeling, context, and exposure
analysis (see `docs/data_sources.md`).

## 3.3 SAR Preprocessing

Composites were built by taking the per-pixel median of all available
scenes within each date window, which suppresses residual speckle relative
to a single scene. A 3×3 focal-median filter was then applied as an
additional speckle-reduction step. All composites were clipped to the study
area and reprojected to UTM Zone 46N (EPSG:32646) for consistent area-based
calculations.

## 3.4 Feature Engineering

For the baseline machine-learning models, the following SAR-derived
features were computed per pixel: VV, VH, the linear-scale VV/VH ratio, the
dB-scale VV−VH difference, VV and VH temporal change (peak minus pre-flood),
and a GLCM-contrast texture measure computed from the VV band. These
features are documented in `src/features/sar_features.py` and are grounded
in established SAR flood-mapping literature, where low VV backscatter and
low VV/VH ratio are established indicators of smooth, specular-reflecting
open water surfaces.

## 3.5 Training Data Preparation

Reference flood labels were generated as a **proxy dataset**, not
ground-truthed observations: pixels with peak-flood VV backscatter below a
fixed threshold (−17 dB, a widely used SAR water-detection threshold) and
not already classified as permanent water by the JRC Global Surface Water
layer were labeled as flood. This is an explicit limitation of the project
(see README "Limitations"), a rigorous operational deployment would
validate these labels against field surveys or very-high-resolution optical
imagery from the same period.

Patches of 256×256 pixels (stride 128) were extracted from the full-scene
composites. **Crucially, the train/validation/test split was performed
spatially — by contiguous geographic row-blocks of the raster — rather than
by randomly shuffling individual patches or pixels.** Remote sensing data
exhibits strong spatial autocorrelation: neighboring pixels are highly
similar, so a naive random pixel/patch split would place near-duplicate
information in both the training and test sets, inflating reported accuracy
and giving a misleading picture of how the model would generalize to a
genuinely unseen area. The spatial block split (60% train / 20% validation /
20% test regions) avoids this leakage.

## 3.6 Baseline Machine Learning

Random Forest (300 trees, max depth 20, balanced class weights) and XGBoost
(300 estimators, max depth 8, learning rate 0.05) classifiers were trained
on the per-pixel feature stack described in 3.4, using the same spatial
split as the deep learning model. These baselines establish a reference
performance level and provide interpretable feature-importance rankings
(Figure 6) that help sanity-check what the more opaque U-Net model may be
learning.

## 3.7 U-Net Architecture

A standard encoder-decoder U-Net (`src/models/unet.py`) was implemented
with 4 downsampling stages, batch normalization, and dropout (0.2) in the
bottleneck. Skip connections concatenate encoder feature maps with the
corresponding decoder stage. The architecture is intentionally conservative, the project prioritizes methodological validity and reproducibility over
architectural novelty (Section 8 of the original brief). Output is a
single-channel logit map passed through a sigmoid for binary flood
probability.

## 3.8 Model Training

The model was trained with a combined Binary Cross-Entropy + Dice loss to
address the class imbalance inherent to flood mapping (flood pixels are
typically a minority class even during peak flooding). Training used the
Adam optimizer, a ReduceLROnPlateau learning-rate schedule, early stopping
(patience 12 epochs) on validation loss, and best-checkpoint saving. GPU use
is automatic when available (`torch.cuda.is_available()`); the pipeline runs
unmodified on CPU. A `--debug` flag runs a 2-epoch, small-subset smoke test
to verify the pipeline before committing to a full run.

## 3.9 Accuracy Assessment

Because flood pixels are typically a minority class, accuracy alone is not
used as the primary metric. Precision, recall, F1, IoU (per-class and mean),
Dice coefficient, and Cohen's Kappa are all reported (`src/evaluation/metrics.py`).
Results for the RF, XGBoost, and U-Net models are compared side by side
(Section 13 comparison table). Any metric that has not yet been generated by
an actual training run is explicitly labeled "to be generated after running
the pipeline", no numbers are fabricated (see Scientific Integrity, README).

## 3.10 Multi-Temporal Flood Analysis

The trained model is applied via sliding-window inference
(`src/mapping/flood_mapping.py`) to the pre-flood, peak-flood, and
post-flood composites for each flood event, producing three binary flood
maps per event. From these, flooded area (km²), percent of study area
flooded, flood persistence (flooded at both peak and post-flood dates),
recession (flooded at peak but not post-flood), and expansion (not flooded
pre-flood but flooded at peak) are computed.

## 3.11 Exposure Assessment

Flood extent rasters are vectorized and overlaid with land-use/land-cover
polygons and FAO GAUL administrative boundaries
(`src/mapping/exposure_analysis.py`) to compute exposed area per LULC class
and per administrative unit, and exposed road length. No economic loss
estimates are produced, since that would require validated asset-value data
that is out of scope for this project.

---

*References to the broader SAR flood-mapping literature (e.g. threshold-based
change detection, U-Net segmentation for flood extent) are described
conceptually above; specific citations should be added by the author from
the relevant published literature (e.g. Twele et al., Nemni et al., and
similar works on Sentinel-1 flood mapping) before submission for
publication, no fabricated citations are included here.*
