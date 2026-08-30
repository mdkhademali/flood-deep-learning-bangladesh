/**
 * sentinel1_preprocessing.js
 * ---------------------------------------------------------------------------
 * Google Earth Engine script: Sentinel-1 GRD acquisition and preprocessing
 * for the Kurigram District (Bangladesh) flood mapping project.
 *
 * Steps:
 *   1. Define study area (Kurigram District bounding box)
 *   2. Load Sentinel-1 IW GRD collection, filter by date / orbit / bands
 *   3. Apply speckle filtering (refined Lee-style focal median filter)
 *   4. Convert to dB (already dB in COPERNICUS/S1_GRD) and clip to AOI
 *   5. Export pre-flood, peak-flood, post-flood composites to Google Drive
 *
 * Run this in the GEE Code Editor: https://code.earthengine.google.com
 * All datasets/bands referenced here exist in the GEE Data Catalog as of
 * this writing (COPERNICUS/S1_GRD, IW mode, VV+VH, log-scaled dB units).
 * ---------------------------------------------------------------------------
 */

// ---------------------------------------------------------------------------
// 1. STUDY AREA
// ---------------------------------------------------------------------------
var aoi = ee.Geometry.Rectangle([89.55, 25.45, 89.95, 26.05]); // Kurigram District bbox
Map.centerObject(aoi, 9);
Map.addLayer(aoi, {color: 'red'}, 'Kurigram AOI', false);

// ---------------------------------------------------------------------------
// 2. HELPER: speckle filtering (focal median, 3x3 kernel)
// ---------------------------------------------------------------------------
function speckleFilter(image) {
  var kernel = ee.Kernel.square({radius: 1, units: 'pixels'});
  return image.focal_median({kernel: kernel, iterations: 1}).copyProperties(image, image.propertyNames());
}

// ---------------------------------------------------------------------------
// 3. HELPER: load & preprocess a Sentinel-1 composite for a date window
// ---------------------------------------------------------------------------
function getS1Composite(startDate, endDate, aoiGeom) {
  var col = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(aoiGeom)
    .filterDate(startDate, endDate)
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    .select(['VV', 'VH']);

  var count = col.size();
  print('S1 scenes found for', startDate, '-', endDate, ':', count);

  // Median composite reduces speckle further when multiple scenes are available
  var composite = col.median().clip(aoiGeom);
  var filtered = speckleFilter(composite);
  return filtered;
}

// ---------------------------------------------------------------------------
// 4. BUILD COMPOSITES FOR ONE FLOOD EVENT (2020 monsoon flood, example)
//    Adjust the windows below to match configs/config.yaml flood_events.
// ---------------------------------------------------------------------------
var preFlood2020  = getS1Composite('2020-06-01', '2020-06-15', aoi);
var peakFlood2020 = getS1Composite('2020-07-15', '2020-07-31', aoi);
var postFlood2020 = getS1Composite('2020-09-01', '2020-09-15', aoi);

var vvVhVis = {bands: ['VV'], min: -25, max: 0};
Map.addLayer(preFlood2020,  vvVhVis, 'Pre-flood VV (2020)', false);
Map.addLayer(peakFlood2020, vvVhVis, 'Peak-flood VV (2020)', false);
Map.addLayer(postFlood2020, vvVhVis, 'Post-flood VV (2020)', false);

// ---------------------------------------------------------------------------
// 5. DERIVED FEATURES (VV/VH ratio, difference, change)
// ---------------------------------------------------------------------------
function addDerivedBands(image) {
  var ratio = image.select('VV').subtract(image.select('VH')).rename('VV_VH_diff');
  // Note: bands are already in dB (log scale), so ratio in linear scale ~ subtraction in dB
  var linVV = ee.Image(10).pow(image.select('VV').divide(10));
  var linVH = ee.Image(10).pow(image.select('VH').divide(10));
  var linRatio = linVV.divide(linVH).rename('VV_VH_ratio');
  return image.addBands([ratio, linRatio]);
}

var peakFlood2020Derived = addDerivedBands(peakFlood2020);

var changeVV = peakFlood2020.select('VV').subtract(preFlood2020.select('VV')).rename('VV_change');
var changeVH = peakFlood2020.select('VH').subtract(preFlood2020.select('VH')).rename('VH_change');

// ---------------------------------------------------------------------------
// 6. EXPORT TO GOOGLE DRIVE
// ---------------------------------------------------------------------------
Export.image.toDrive({
  image: preFlood2020.select(['VV', 'VH']),
  description: 'Kurigram_S1_preflood_2020',
  folder: 'flood_deep_learning_bd',
  region: aoi,
  scale: 10,
  crs: 'EPSG:32646',
  maxPixels: 1e10
});

Export.image.toDrive({
  image: peakFlood2020Derived,
  description: 'Kurigram_S1_peakflood_2020',
  folder: 'flood_deep_learning_bd',
  region: aoi,
  scale: 10,
  crs: 'EPSG:32646',
  maxPixels: 1e10
});

Export.image.toDrive({
  image: postFlood2020.select(['VV', 'VH']),
  description: 'Kurigram_S1_postflood_2020',
  folder: 'flood_deep_learning_bd',
  region: aoi,
  scale: 10,
  crs: 'EPSG:32646',
  maxPixels: 1e10
});

Export.image.toDrive({
  image: changeVV.addBands(changeVH),
  description: 'Kurigram_S1_change_2020',
  folder: 'flood_deep_learning_bd',
  region: aoi,
  scale: 10,
  crs: 'EPSG:32646',
  maxPixels: 1e10
});

// Repeat the block above for the 2022 flood event (2022-05-15/2022-08-31 windows)
// by re-calling getS1Composite() with the corresponding date ranges from
// configs/config.yaml, then exporting as 'Kurigram_S1_*_2022'.
