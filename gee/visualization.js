/**
 * visualization.js
 * ---------------------------------------------------------------------------
 * Quick-look visualization helpers for the GEE Code Editor. Not part of the
 * automated export pipeline, used interactively to sanity-check composites
 * and labels before export.
 * ---------------------------------------------------------------------------
 */

var aoi = ee.Geometry.Rectangle([89.55, 25.45, 89.95, 26.05]);
Map.centerObject(aoi, 9);

// RGB-style false color from VV/VH/VV-VH ratio — helps visually separate
// smooth water (low VV, low VH) from rough vegetated/urban surfaces
function s1FalseColor(image) {
  var vv = image.select('VV');
  var vh = image.select('VH');
  var ratio = vv.subtract(vh).rename('ratio');
  return ee.Image.cat([vv, vh, ratio]).rename(['R', 'G', 'B']);
}

var s1peak = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(aoi)
  .filterDate('2020-07-15', '2020-07-31')
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .select(['VV', 'VH'])
  .median()
  .clip(aoi);

Map.addLayer(s1FalseColor(s1peak), {min: -25, max: 5}, 'S1 False Color (peak flood)');

// Administrative boundary overlay for context
var admin = ee.FeatureCollection('FAO/GAUL/2015/level2')
  .filter(ee.Filter.eq('ADM2_NAME', 'Kurigram'));
Map.addLayer(admin.style({color: 'yellow', fillColor: '00000000', width: 2}), {}, 'Kurigram boundary');
