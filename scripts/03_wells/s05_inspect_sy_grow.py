#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2.0_Inspect_Sy_and_GROW.py — read specific-yield shapefile + GROW timeseries
structure before Phase 2. Read-only.
"""
import os, numpy as np, warnings
warnings.filterwarnings("ignore")
import geopandas as gpd, pandas as pd

DATA = r"D:\THESIS\Arpon_Thesis_Final\DATA"
SY_SHP = os.path.join(DATA, "specific yeild-上传", "geishuidu-min.shp")
GROW_DIR = os.path.join(DATA, "GROW")

LINE = "="*78
def hdr(t): print("\n"+LINE+"\n"+t+"\n"+LINE)

# ---------------- SPECIFIC YIELD ----------------
hdr("SPECIFIC YIELD SHAPEFILE")
if not os.path.exists(SY_SHP):
    print(f"  [FAIL] not found: {SY_SHP}")
else:
    sy = gpd.read_file(SY_SHP)
    print(f"  polygons : {len(sy)}")
    print(f"  CRS      : {sy.crs}")
    print(f"  columns  : {list(sy.columns)}")
    print(f"  bounds   : {sy.total_bounds}")
    for c in sy.columns:
        if c == sy.geometry.name: continue
        s = sy[c]
        print(f"\n  --- column '{c}'  dtype={s.dtype} ---")
        if np.issubdtype(s.dtype, np.number):
            print(f"      min={s.min():.6g}  max={s.max():.6g}  mean={s.mean():.6g}")
            print(f"      median={s.median():.6g}  n_null={s.isna().sum()}")
            print(f"      percentiles 5/25/50/75/95: "
                  f"{np.nanpercentile(s,5):.4g} {np.nanpercentile(s,25):.4g} "
                  f"{np.nanpercentile(s,50):.4g} {np.nanpercentile(s,75):.4g} "
                  f"{np.nanpercentile(s,95):.4g}")
            print(f"      if values are 0-1 => fraction; if 0-100 => percent; "
                  f"if 0-0.5 => classic Sy range")
        else:
            u = s.dropna().unique()
            print(f"      {len(u)} unique; first 12: {list(u[:12])}")

# ---------------- GROW TIMESERIES ----------------
hdr("GROW TIMESERIES (structure only)")
ts = os.path.join(GROW_DIR, "grow_timeseries.parquet")
if not os.path.exists(ts):
    print(f"  [FAIL] not found: {ts}")
else:
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(ts)
    print(f"  rows (total) : {pf.metadata.num_rows:,}")
    print(f"  row groups   : {pf.metadata.num_row_groups}")
    print(f"  columns      : {[c for c in pf.schema.names]}")
    head = next(pf.iter_batches(batch_size=5)).to_pandas()
    print("\n  first 5 rows:")
    print(head.to_string())
    for c in head.columns:
        print(f"    {c:<34} dtype={head[c].dtype}")

# ---------------- GROW ATTRIBUTES (key cols only) ----------------
hdr("GROW ATTRIBUTES (calibration-relevant columns)")
at = os.path.join(GROW_DIR, "grow_attributes.parquet")
if os.path.exists(at):
    df = pd.read_parquet(at, columns=[
        "GROW_ID","latitude","longitude","country","interval","length_years",
        "gap_fraction","plateaus","outliers_change_points","negative_signs_wtd",
        "trend_direction","trend_slope_m_year-1","reference_point",
        "aquifer_type_class","total_porosity_0-100_m_fraction"])
    print(f"  rows: {len(df):,}")
    print("\n  trend_slope_m_year-1 stats (ALL wells):")
    s = df["trend_slope_m_year-1"]
    print(f"     n_nonnull={s.notna().sum():,}  min={s.min():.4g}  max={s.max():.4g}")
    print(f"     p1={np.nanpercentile(s,1):.4g}  p50={np.nanpercentile(s,50):.4g}  "
          f"p99={np.nanpercentile(s,99):.4g}")
    print(f"     units m/yr; NEGATIVE should mean declining water table")
    print("\n  reference_point values:")
    print("    " + df["reference_point"].value_counts(dropna=False).to_string().replace("\n","\n    "))
    print("\n  aquifer_type_class values:")
    print("    " + df["aquifer_type_class"].value_counts(dropna=False).head(10).to_string().replace("\n","\n    "))
    print("\n  lat/lon extent:")
    print(f"     lat {df['latitude'].min():.2f}..{df['latitude'].max():.2f}   "
          f"lon {df['longitude'].min():.2f}..{df['longitude'].max():.2f}")
else:
    print(f"  [FAIL] not found: {at}")

print("\n"+LINE)
print("Inspection complete: Sy units and GROW schema reported above.")
print(LINE)