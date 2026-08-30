/**
 * flood_dataset_export.js
 * ---------------------------------------------------------------------------
 * Generates weak/reference flood labels for the Kurigram AOI by combining:
 *   (a) JRC Global Surface Water (permanent water mask)
 *   (b) A standard SAR change-detection threshold on peak-flood VV backscatter
 *       (Otsu-style split is approximated here with a fixed dB threshold,
 *        a widely used and defensible approach in SAR flood-mapping
 *        literature, see docs/methodology.md 3.5 for citations and caveats)
 *
 * These labels are a REFERENCE/PROXY dataset for model training, not
 * ground-truthed flood observations. This limitation is explicitly stated
 * in docs/methodology.md and README.md ("Limitations").
 * ---------------------------------------------------------------------------
 */

var aoi = ee.Geometry.Rectangle([89.55, 25.45, 89.95, 26.05]);

// Permanent water mask (occurrence > 50% over 1984-2021 record)
var gsw = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence');
var permanentWater = gsw.gt(50).unmask(0).clip(aoi);

// Peak-flood Sentinel-1 VV composite (see sentinel1_preprocessing.js)
var s1peak = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(aoi)
  .filterDate('2020-07-15', '2020-07-31')
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
  .select('VV')
  .median()
  .clip(aoi);

// Standard SAR flood threshold: VV backscatter below ~-17 dB indicates
// smooth (water) surfaces under typical incidence angles for Sentinel-1 IW.
var VV_THRESHOLD_DB = -17;
var openWaterCandidate = s1peak.lt(VV_THRESHOLD_DB);

// Flood = candidate open water that is NOT already permanent water
var floodLabel = openWaterCandidate.and(permanentWater.not()).rename('flood');
var referenceLabel = floodLabel.unmask(0).toByte();

Map.centerObject(aoi, 9);
Map.addLayer(referenceLabel, {min: 0, max: 1, palette: ['white', 'blue']}, 'Reference flood label');

// ---------------------------------------------------------------------------
// EXPORT LABEL RASTER
// ---------------------------------------------------------------------------
Export.image.toDrive({
  image: referenceLabel,
  description: 'Kurigram_flood_label_2020',
  folder: 'flood_deep_learning_bd',
  region: aoi,
  scale: 10,
  crs: 'EPSG:32646',
  maxPixels: 1e10
});

// ---------------------------------------------------------------------------
// EXPORT PERMANENT WATER MASK (used to disambiguate errors during evaluation)
// ---------------------------------------------------------------------------
Export.image.toDrive({
  image: permanentWater.toByte(),
  description: 'Kurigram_permanent_water_mask',
  folder: 'flood_deep_learning_bd',
  region: aoi,
  scale: 10,
  crs: 'EPSG:32646',
  maxPixels: 1e10
});
