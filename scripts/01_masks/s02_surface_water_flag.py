#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
PHASE 0.5b -- PERSIST SURFACE-WATER FLAG INTO THE RELIABILITY MASK
--------------------------------------------------------------------------------
Until now the WHYMAP waterbody flag was recomputed inline in every downstream
script (2.2, 3.0). That is fine for a trend field where water only ever gets
excluded, but the ENERGY step is different: misattributed surface-water
evaporation inflates the latent-heat headline ONE-DIRECTIONALLY. A load-bearing
exclusion must therefore be authoritative and stored ONCE, not re-derived
per script where a path typo or an all_touched flip could silently change it.

This script reads the existing reliability_mask.nc, rasterizes the WHYMAP
waterbodies to the SAME 0.5 deg grid and with the SAME all_touched=True
convention used in 3.0 (so the flag is bit-identical to what 3.0 produced),
adds a `surface_water` variable, and rewrites the mask in place (with a
timestamped backup). It does NOT touch permanent_ice or any other variable.

Idempotent: re-running overwrites surface_water with the same values and makes
a fresh backup. Non-destructive to every other layer.
================================================================================
"""
import os, shutil, datetime as dt
import numpy as np, pandas as pd
import xarray as xr

DATA     = r"D:\THESIS\Arpon_Thesis_Final\DATA"
OUT_ROOT = r"D:\THESIS\Arpon_Thesis_Final\OUTPUTS"
MASK_NC  = os.path.join(OUT_ROOT, "00_reliability_mask", "reliability_mask.nc")
WATER_SHP= os.path.join(DATA,"WHYMAP","WHYMAP_GWR","shp","whymap_waterbodies__v1_poly.shp")

GRID = 0.5
LAT = np.arange(-90+GRID/2, 90, GRID)
LON = np.arange(-180+GRID/2, 180, GRID)

LINE = "="*78
def _p(s=""): print(s)
def hdr(t): _p("\n"+LINE); _p(t); _p(LINE)

def rasterize_water(shp):
    """Same path and convention as 3.0: all_touched=True, flipud to lat-ascending."""
    import geopandas as gpd
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    g = gpd.read_file(shp)
    if g.crs is None: g = g.set_crs("EPSG:4326")
    elif g.crs.to_epsg() != 4326: g = g.to_crs("EPSG:4326")
    shapes = ((geom, 1) for geom in g.geometry if geom is not None)
    arr = rasterize(shapes, out_shape=(len(LAT), len(LON)),
                    transform=from_origin(-180, 90, GRID, GRID),
                    fill=0, all_touched=True, dtype="int32")
    return np.flipud(arr).astype(bool)

def main():
    hdr("PHASE 0.5b -- persist surface-water flag")

    if not os.path.exists(MASK_NC):
        raise FileNotFoundError(f"reliability_mask.nc not found at {MASK_NC}")

    ds = xr.open_dataset(MASK_NC)
    ds.load()          # read fully so we can close and rewrite the same path
    ds.close()
    _p(f"  mask variables present: {list(ds.data_vars)}")

    # grid sanity: the mask must be on the same 0.5 deg grid we rasterize to
    ny, nx = ds[list(ds.data_vars)[0]].shape
    if (ny, nx) != (len(LAT), len(LON)):
        raise ValueError(f"mask grid {ny}x{nx} != expected {len(LAT)}x{len(LON)}; "
                         "grids differ, aborting rather than silently regridding")
    _p(f"  grid check OK: {ny} x {nx}")

    water = rasterize_water(WATER_SHP)
    _p(f"  surface-water cells rasterized: {int(water.sum()):,}")

    # overlap diagnostics against existing layers (report, do not modify them)
    if "permanent_ice" in ds:
        ice = ds["permanent_ice"].values.astype(bool)
        _p(f"  overlap water & permanent_ice: {int((water & ice).sum()):,} "
           f"({100*(water & ice).sum()/max(water.sum(),1):.2f}% of water cells)")

    # attach; align to the mask's own coords to avoid any coordinate drift
    lat_name = "lat" if "lat" in ds.coords else list(ds.coords)[0]
    lon_name = "lon" if "lon" in ds.coords else list(ds.coords)[1]
    ds["surface_water"] = ((lat_name, lon_name), water.astype("int8"))
    ds["surface_water"].attrs.update({
        "long_name": "WHYMAP surface-water body flag (1=water, 0=land)",
        "source": os.path.basename(WATER_SHP),
        "rasterization": "all_touched=True, 0.5 deg, identical to phase 3.0",
        "purpose": "load-bearing exclusion for the energy step: misattributed "
                   "surface-water evaporation inflates latent heat one-directionally",
        "created": dt.datetime.now().isoformat(timespec="seconds")})

    # timestamped backup, then rewrite in place
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = MASK_NC.replace(".nc", f"_backup_{stamp}.nc")
    shutil.cop2 = shutil.copy2  # explicit
    shutil.copy2(MASK_NC, backup)
    _p(f"  backup written: {os.path.basename(backup)}")

    tmp = MASK_NC.replace(".nc", "_tmp.nc")
    ds.to_netcdf(tmp)
    ds.close()
    os.replace(tmp, MASK_NC)
    _p(f"  [ OK ]   surface_water persisted into {os.path.basename(MASK_NC)}")

    # verify round-trip
    chk = xr.open_dataset(MASK_NC); chk.load(); chk.close()
    assert "surface_water" in chk, "surface_water missing after write"
    assert int(chk["surface_water"].values.sum()) == int(water.sum()), \
        "surface_water round-trip mismatch"
    _p(f"  verified: surface_water reads back with {int(chk['surface_water'].values.sum()):,} cells")
    _p("\n  Downstream scripts should now read reliability_mask.nc['surface_water'] "
       "instead of rasterizing the shapefile inline.")

if __name__ == "__main__":
    main()