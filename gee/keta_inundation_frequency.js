// =====================================================================
// KETA INUNDATION FREQUENCY  --  susceptibility from 519 scenes, not 8 events
// =====================================================================
// WHY THIS EXISTS
//   Model A failed for two separate reasons, and this addresses both.
//
//   1. TOO FEW OBSERVATIONS. Susceptibility was estimated from 8 binary labels
//      per location. Binomial noise sd at n=8 is 0.174 against a true
//      between-location sd of 0.070, so 86% of the measured variation is noise
//      and the R2 ceiling for ANY model is 0.14.
//
//   2. THE WRONG PHYSICAL SIGNATURE. The v4.x label is an Otsu split on a
//      VH DIFFERENCE (during minus before). Validated against terrain, that
//      label points BACKWARDS: flood frequency rises with elevation
//      (rho +0.076) and with distance to water (rho +0.054). Vegetation
//      backscatter responds strongly to rainfall, while already-wet low ground
//      is dark in VH to begin with and changes little, so a difference image
//      maps vegetation response to rain rather than standing water.
//
//   Open water is specular: it is DARK IN ABSOLUTE TERMS, typically below
//   -22 dB in VH, regardless of what it looked like a fortnight earlier. That
//   is a physical signature, not a relative one. Thresholding it needs no
//   before/after pair, so every scene is an independent observation.
//
//   519 scenes instead of 8 events raises the R2 ceiling from 0.14 to ~0.9,
//   and removes the vegetation confound at the same time.
//
// WHAT IT PRODUCES
//   inundation_frequency  fraction of valid observations where a cell is water
//   n_observations        how many scenes actually saw each cell
//   Both at 30 m, plus a sample at the same locations as the v5.0 export so the
//   result can be compared directly with the old label.
//
// VALIDATION BUILT IN
//   JRC Global Surface Water 'occurrence' is an independent, Landsat-derived,
//   38-year water frequency map produced by a different team with a different
//   sensor and algorithm. If this product is measuring water, it must agree
//   with GSW over the lagoon and disagree over dry land. That check is printed.
// =====================================================================

var keta = ee.Geometry.Polygon([
  [[0.80, 5.75], [1.15, 5.75], [1.15, 6.10], [0.80, 6.10]]
]);

// Pinned from the orbit audit: 519 of 529 scenes over Keta.
var S1_RELATIVE_ORBIT = 74;
var S1_PASS = "ASCENDING";

// Open-water VH threshold in dB. -22 is a common operational value for C-band
// VH over calm water; the sensitivity sweep below reports 5 alternatives so the
// choice is made from the data rather than asserted.
var VH_WATER_DB = -22;

print(">>> KETA INUNDATION FREQUENCY v5.1 (DEM: Copernicus GLO-30)  |  orbit " + S1_RELATIVE_ORBIT + " " +
      S1_PASS + "  |  VH water threshold " + VH_WATER_DB + " dB");

// ---- 1. THE FULL SCENE STACK ----------------------------------------
var s1 = ee.ImageCollection("COPERNICUS/S1_GRD")
  .filterBounds(keta)
  .filter(ee.Filter.eq("instrumentMode", "IW"))
  .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
  .filter(ee.Filter.eq("orbitProperties_pass", S1_PASS))
  .filter(ee.Filter.eq("relativeOrbitNumber_start", S1_RELATIVE_ORBIT))
  .map(function (img) {
    var angle = img.select("angle");
    return img.updateMask(angle.gt(30).and(angle.lt(45))).select("VH")
              .copyProperties(img, ["system:time_start"]);
  });

print("scenes in stack:", s1.size());
print("date range:", ee.Date(s1.aggregate_min("system:time_start")).format("YYYY-MM-dd"),
      "to", ee.Date(s1.aggregate_max("system:time_start")).format("YYYY-MM-dd"));

// ---- 2. TERRAIN AND PERMANENT WATER ---------------------------------
var jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater");
var occurrence = jrc.select("occurrence").unmask(0);
var permWater = occurrence.gt(80);

// v5.1: Copernicus GLO-30, not NASADEM. NASADEM is quantised to whole metres
// and carries exactly THREE distinct values ({0, 1, 2}) across the observed
// 2.94 m total-water-level range, so it cannot order cells by exposure inside
// the band that decides inundation. GLO-30 carries 28,409 distinct values over
// the same interval.
//
// Measured with an identical model, identical spatial blocks, identical seed,
// changing only the elevation source:
//   NASADEM   out-of-block R2 0.250   seasonal-water residual bias +0.356
//   GLO-30    out-of-block R2 0.466   seasonal-water residual bias +0.229
//   FABDEM    out-of-block R2 0.437   seasonal-water residual bias +0.232
//
// GLO-30 is preferred over FABDEM here: on a bare sandy barrier FABDEM removes
// only about 7.5 cm of canopy, and its ~0.01 m distribution quantisation costs
// more than the bare-earth correction gains. That ranking would reverse on a
// forested floodplain.
//
// A mosaicked ImageCollection has no fixed projection, which ee.Terrain and any
// downstream slope calculation require, so it is set explicitly.
var srtm = ee.ImageCollection("COPERNICUS/DEM/GLO30").select("DEM").mosaic()
  .setDefaultProjection("EPSG:4326", null, 30);
var dem = srtm.clip(keta).rename("elevation");
// v5.0 fix: distance FROM water, not the inverted transform used in v4.x.
var distWater = permWater.fastDistanceTransform(1024).sqrt()
  .multiply(30).rename("dist_water");

// ---- 3. INUNDATION FREQUENCY ----------------------------------------
// Speckle is the dominant per-scene error. A light focal median costs a little
// spatial detail and buys a large reduction in false water pixels; it is
// applied identically to every scene so it cannot bias the frequency.
function waterMask(img) {
  var sm = img.focal_median(30, "circle", "meters");
  return sm.lt(VH_WATER_DB).rename("water")
           .updateMask(img.mask());
}

var wet = s1.map(waterMask);
var freq = wet.mean().rename("inundation_frequency");       // ignores masked
var nobs = s1.map(function (i) {
  return i.mask().rename("n");
}).sum().rename("n_observations");

var product = freq.addBands(nobs).addBands(dem).addBands(distWater)
  .addBands(occurrence.rename("gsw_occurrence"))
  .addBands(permWater.rename("is_permanent_water"))
  .updateMask(permWater.not())                              // land only
  .clip(keta);

// ---- 4. THRESHOLD SENSITIVITY ---------------------------------------
// The threshold is the one free parameter. Report what it does rather than
// defending a single value.
print("--- threshold sensitivity: mean land inundation frequency ---");
[-26, -24, -22, -20, -18].forEach(function (thr) {
  var f = s1.map(function (img) {
    return img.focal_median(30, "circle", "meters").lt(thr).rename("w");
  }).mean().updateMask(permWater.not());
  print("  VH < " + thr + " dB", f.reduceRegion({
    reducer: ee.Reducer.mean(), geometry: keta, scale: 90, maxPixels: 1e9
  }).get("w"));
});

// ---- 5. INDEPENDENT VALIDATION AGAINST JRC GSW ----------------------
// GSW is Landsat-derived, 38 years, a different team and algorithm. Agreement
// over water and disagreement over dry land is what a real water product looks
// like. This is the check the v4.x label never had.
print("--- validation vs JRC Global Surface Water (independent) ---");
var both = freq.rename("s1_freq").addBands(occurrence.rename("gsw"))
  .updateMask(permWater.not());
print("  Pearson corr(S1 inundation freq, GSW occurrence) over land:",
      both.reduceRegion({
        reducer: ee.Reducer.pearsonsCorrelation(),
        geometry: keta, scale: 90, maxPixels: 1e9, bestEffort: true
      }));
print("  mean S1 frequency where GSW occurrence > 20 (seasonal water):",
      freq.updateMask(occurrence.gt(20).and(permWater.not())).reduceRegion({
        reducer: ee.Reducer.mean(), geometry: keta, scale: 90, maxPixels: 1e9
      }));
print("  mean S1 frequency where GSW occurrence = 0 (never water):",
      freq.updateMask(occurrence.eq(0)).reduceRegion({
        reducer: ee.Reducer.mean(), geometry: keta, scale: 90, maxPixels: 1e9
      }));

// ---- 6. MAP + EXPORT -------------------------------------------------
Map.centerObject(keta, 11);
Map.setOptions("HYBRID");
Map.addLayer(product.select("inundation_frequency"),
  {min: 0, max: 0.5, palette: ["#39463f", "#facc15", "#ea580c", "#dc2626"]},
  "Inundation frequency");
Map.addLayer(permWater.selfMask(), {palette: ["#0c4a6e"]}, "Permanent water", false);

Export.image.toDrive({
  image: product.select(["inundation_frequency", "n_observations",
                         "elevation", "dist_water", "gsw_occurrence"]).toFloat(),
  description: "keta_inundation_frequency_glo30_30m",
  folder: "keta_v5",
  region: keta,
  scale: 30,
  maxPixels: 1e9,
  fileFormat: "GeoTIFF"
});

// Sampled at the same lattice as the v5.0 table, so the new susceptibility can
// be compared directly against the old event-based label.
var samples = product.addBands(ee.Image.pixelLonLat()).sample({
  region: keta, scale: 30, numPixels: 6000, seed: 42,
  geometries: false, tileScale: 4
});
Export.table.toDrive({
  collection: samples,
  folder: "keta_v5",
  description: "keta_inundation_frequency_glo30_samples",
  fileFormat: "CSV"
});

print("");
print("Run, read the sensitivity and validation blocks, then export both tasks.");
print("If corr(S1, GSW) is strongly positive over land, this is measuring water.");
print("If it is near zero, absolute thresholding fails here too and the problem");
print("is the sensor, not the method.");
