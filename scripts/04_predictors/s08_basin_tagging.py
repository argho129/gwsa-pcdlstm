# -*- coding: utf-8 -*-
r"""
================================================================================
BASIN TAGGING  -  assign every 0.5deg cell to a HydroBASINS level-3 basin
                  and flag the focused aquifers (incl. Bengal)
================================================================================
Thesis "Aquifer to Atmosphere" | Arpon, KUET

Produces a per-cell table (j,i,lat,lon -> HYBAS_ID, MAIN_BAS, focus_name) by
point-in-polygon join of cell centers against HydroBASINS level-3 polygons
(all 9 continents merged). Focused aquifers are identified by seed coordinates
inside them, then matched to whichever basin polygon contains that seed.

OUTPUT: OUTPUTS\09_ml_diagnostic\cell_basins.parquet
        + console report of focused-basin cell counts
================================================================================
"""
import os, glob, numpy as np, pandas as pd, geopandas as gpd
from shapely.geometry import Point

ROOT = r"D:\THESIS\Arpon_Thesis_Final"
HB   = os.path.join(ROOT, "DATA", "HydroBasins")
OUT  = os.path.join(ROOT, "OUTPUTS", "09_ml_diagnostic")
CAL  = os.path.join(ROOT, "OUTPUTS", "03_calibrated", "gwd_trend_calibrated_v2.nc")
os.makedirs(OUT, exist_ok=True)

def hr(t=""):
    print("\n"+"="*72)
    if t: print(t); print("="*72)

# focused aquifers: name -> seed lon/lat inside the basin
FOCUS = {
    "Bengal_GBM":      (90.4, 23.7),   # Ganges-Brahmaputra-Meghna (Dhaka)
    "Punjab_Indus":    (74.0, 31.0),   # Indus / Punjab
    "NorthChinaPlain": (116.0, 37.0),  # Hai/Yellow river plain
    "CentralValley":   (-120.5, 36.5), # California Central Valley
    "HighPlains":      (-101.0, 37.0), # Ogallala / High Plains
}

hr("1. LOAD + MERGE HydroBASINS level-3 (all continents)")
shps = sorted(glob.glob(os.path.join(HB, "**", "*lev03*.shp"), recursive=True))
print(f"  level-3 shapefiles: {len(shps)}")
gdfs = []
for s in shps:
    g = gpd.read_file(s)[["HYBAS_ID","MAIN_BAS","SUB_AREA","geometry"]]
    gdfs.append(g)
basins = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs="EPSG:4326")
print(f"  total level-3 basins: {len(basins)}")

hr("2. IDENTIFY FOCUSED BASINS via seed points")
focus_ids = {}   # name -> set of MAIN_BAS ids
for name,(lon,lat) in FOCUS.items():
    pt = Point(lon, lat)
    hit = basins[basins.contains(pt)]
    if len(hit):
        mb = hit.iloc[0]["MAIN_BAS"]
        # grab ALL level-3 polygons sharing that MAIN_BAS (the whole river system)
        ids = set(basins[basins["MAIN_BAS"]==mb]["HYBAS_ID"])
        focus_ids[name] = (mb, ids)
        area = basins[basins["MAIN_BAS"]==mb]["SUB_AREA"].sum()
        print(f"  {name:16s}: MAIN_BAS={mb}  ({len(ids)} sub-polys, ~{area:,.0f} km2)")
    else:
        print(f"  {name:16s}: NO basin found at seed ({lon},{lat}) - check coords")

hr("3. BUILD 0.5deg CELL CENTERS (land cells from calibrated file)")
import xarray as xr
cal = xr.open_dataset(CAL)
LAT = cal["lat"].values; LON = cal["lon"].values
land = np.isfinite(cal["trend_calibrated"].values)
cal.close()
nlat,nlon = land.shape
jj,ii = np.where(land)
lats = LAT[jj]; lons = LON[ii]
print(f"  land cells: {len(jj)}")
cells = gpd.GeoDataFrame(
    {"j":jj,"i":ii,"lat":lats,"lon":lons},
    geometry=[Point(x,y) for x,y in zip(lons,lats)], crs="EPSG:4326")

hr("4. SPATIAL JOIN cells -> basins (point in polygon)")
joined = gpd.sjoin(cells, basins[["HYBAS_ID","MAIN_BAS","geometry"]],
                   how="left", predicate="within")
# a cell could match >1 (shared borders); keep first
joined = joined[~joined.index.duplicated(keep="first")]
print(f"  cells matched to a basin: {joined['HYBAS_ID'].notna().sum()} / {len(cells)}")

# tag focus name
joined["focus_name"] = ""
for name,(mb,ids) in focus_ids.items():
    mask = joined["MAIN_BAS"]==mb
    joined.loc[mask,"focus_name"] = name
    print(f"  {name:16s}: {int(mask.sum())} cells")

hr("5. WRITE")
out = joined.drop(columns=["geometry","index_right"], errors="ignore")
out = pd.DataFrame(out)
outp = os.path.join(OUT, "cell_basins.parquet")
out.to_parquet(outp)
print(f"  written: {outp}")
print(f"  columns: {list(out.columns)}")
print(f"  focused cells total: {(out['focus_name']!='').sum()}")
hr("BASIN TAGGING COMPLETE")