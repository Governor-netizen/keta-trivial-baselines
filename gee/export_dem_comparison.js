// =====================================================================
// DEM COMPARISON FOR KETA  --  is DEM error the binding constraint?
// =====================================================================
// WHY
//   The physics-constrained susceptibility model captures distance-to-water
//   almost perfectly (residuals flat to +/-0.013 across four distance decades)
//   but under-predicts by 0.31 to 0.38 wherever JRC GSW says seasonal water.
//   Those are shallow depressions and lagoon-margin wetlands whose relief lives
//   inside a 1-2 m band. NASADEM/SRTM carries roughly 6 m vertical error at
//   this site, several times the entire tidal range, so it cannot resolve them.
//
//   That is a testable claim, not a hunch: swap the DEM and see whether the
//   residual structure shrinks.
//
// THE CANDIDATES
//   NASADEM       SRTM reprocessed, 30 m. What the project uses now. ~6 m LE90.
//   COP GLO-30    Copernicus DEM, 30 m. TanDEM-X derived, ~2-4 m absolute.
//                 A DSM: it includes canopy and buildings.
//   FABDEM        Forest And Buildings removed Copernicus DEM. GLO-30 with
//                 canopy and structures stripped by an ML correction
//                 (Hawker et al. 2022). Bare earth, which is what an inundation
//                 model actually needs. Community asset, not an official GEE
//                 dataset, so availability is checked rather than assumed.
//
// OUTPUT
//   One 4-band GeoTIFF on the SAME grid as keta_inundation_frequency_30m.tif,
//   so the rasters align cell for cell and the comparison is like for like.
//
// HOW TO RUN
//   Paste, Run, read the availability and statistics blocks, then start the
//   export task. The local script make_dem_comparison.py does the analysis.
// =====================================================================

var keta = ee.Geometry.Polygon([
  [[0.80, 5.75], [1.15, 5.75], [1.15, 6.10], [0.80, 6.10]]
]);

print(">>> DEM COMPARISON FOR KETA");

// ---- NASADEM: the incumbent -----------------------------------------
var nasadem = ee.Image("NASA/NASADEM_HGT/001").select("elevation")
  .rename("dem_nasadem").toFloat();

// ---- Copernicus GLO-30: official GEE asset ---------------------------
var glo30 = ee.ImageCollection("COPERNICUS/DEM/GLO30").select("DEM")
  .mosaic().rename("dem_glo30").toFloat();

// ---- FABDEM: community asset, existence not guaranteed ---------------
// Wrapped so a missing or renamed asset reports itself instead of killing the
// script. If this prints UNAVAILABLE, use GLO-30: it is still a large step up
// from SRTM, it is just a surface model rather than bare earth.
var fabdem = null;
var fabdemOk = false;
try {
  fabdem = ee.ImageCollection("projects/sat-io/open-datasets/FABDEM")
    .mosaic().rename("dem_fabdem").toFloat();
  print("FABDEM asset resolved. Sample value at the Keta tide point:",
        fabdem.reduceRegion({
          reducer: ee.Reducer.first(),
          geometry: ee.Geometry.Point([0.97, 5.90]), scale: 30
        }));
  fabdemOk = true;
} catch (e) {
  print("FABDEM UNAVAILABLE ->", e);
  print("Falling back to GLO-30 only. Try the alternative id:");
  print("  projects/sat-io/open-datasets/FABDEM_V1-2");
}

var bands = fabdemOk
  ? nasadem.addBands(glo30).addBands(fabdem)
  : nasadem.addBands(glo30);

// ---- how much do they disagree? --------------------------------------
print("--- elevation statistics over the AOI (m) ---");
var stats = bands.reduceRegion({
  reducer: ee.Reducer.percentile([5, 25, 50, 75, 95])
    .combine(ee.Reducer.mean(), "", true),
  geometry: keta, scale: 90, maxPixels: 1e9, bestEffort: true
});
print(stats);

if (fabdemOk) {
  var diff = glo30.subtract(fabdem).rename("glo30_minus_fabdem");
  print("--- GLO30 minus FABDEM: how much canopy/building was removed? ---");
  print(diff.reduceRegion({
    reducer: ee.Reducer.percentile([5, 50, 95]).combine(ee.Reducer.mean(), "", true),
    geometry: keta, scale: 90, maxPixels: 1e9, bestEffort: true
  }));
  var dn = fabdem.subtract(nasadem).rename("fabdem_minus_nasadem");
  print("--- FABDEM minus NASADEM: the change that matters for flooding ---");
  print(dn.reduceRegion({
    reducer: ee.Reducer.percentile([5, 50, 95]).combine(ee.Reducer.stdDev(), "", true),
    geometry: keta, scale: 90, maxPixels: 1e9, bestEffort: true
  }));
}

// ---- fraction of land below the observed TWL p99 (1.73 m) ------------
// A DEM that cannot resolve the tidal band will place implausibly little or
// implausibly much land inside it.
print("--- fraction of AOI below 1.73 m (observed TWL p99) ---");
var names = fabdemOk ? ["dem_nasadem", "dem_glo30", "dem_fabdem"]
                     : ["dem_nasadem", "dem_glo30"];
names.forEach(function (n) {
  print("  " + n, bands.select(n).lt(1.73).reduceRegion({
    reducer: ee.Reducer.mean(), geometry: keta, scale: 90, maxPixels: 1e9
  }).get(n));
});

Map.centerObject(keta, 11);
Map.setOptions("HYBRID");
Map.addLayer(bands.select("dem_nasadem"), {min: -2, max: 12,
  palette: ["#08306b", "#4292c6", "#c6dbef", "#fdd0a2", "#d94801"]}, "NASADEM");
if (fabdemOk) {
  Map.addLayer(bands.select("dem_fabdem"), {min: -2, max: 12,
    palette: ["#08306b", "#4292c6", "#c6dbef", "#fdd0a2", "#d94801"]}, "FABDEM");
}

// Same region and scale as keta_inundation_frequency_30m.tif so the two
// rasters land on the same grid.
Export.image.toDrive({
  image: bands.clip(keta),
  description: "keta_dem_comparison_30m",
  folder: "keta_v5",
  region: keta,
  scale: 30,
  maxPixels: 1e9,
  fileFormat: "GeoTIFF"
});

print("");
print("Read the two difference blocks first. If FABDEM minus NASADEM has a");
print("large spread, the DEMs disagree about exactly the relief that decides");
print("inundation, and the swap is worth making. If they agree closely, DEM");
print("error is not the binding constraint and the residual is something else.");
