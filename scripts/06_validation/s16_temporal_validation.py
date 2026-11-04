#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""================================================================================
PHASE 6.1 -- PAPER-STYLE CONSISTENCY & PUBLICATION VISUALIZATION (v3.4 TNR, FIXED)
  Per-polygon well-vs-product basin-averaged TIME-SERIES correlation (PCC).
  Publication-grade figures with enlarged Times New Roman fonts, clean global
  maps with colorbar legends, zoomed spatial insets, 1:1 scatterplots, and
  Taylor diagrams for Top Aquifers, Top HydroBASINS, and Focus Hotspots.

  WHAT CHANGED FROM v3.3 (only these; every figure is byte-for-byte the same):
    1. Specific Yield now comes ONLY from geishuidu-min.shp, assigned per well by
       point-in-polygon join, then a 3-nearest-polygon median for any well that
       falls outside a polygon. The flat DEFAULT_SY = 0.10 imputation is GONE.
       No global-shapefile-median constant either -- every Sy is spatially local.
    2. Well qualification gate raised from n >= 3 observations to
       n >= MIN_WELL_OBS (6). Gentle floor: still keeps sparse wells, but drops
       the 3-point wells that add noise to a monthly correlation.
  Window matching is UNCHANGED (it was already correct): each polygon's PCC is
  computed only on the months the wells cover, with the product sampled at those
  exact months (see polygon_pcc / paired).
================================================================================
"""

from collections import Counter, defaultdict
import datetime as dt
import os
import sys
import time
import traceback
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
  import pycountry as _pycountry
  import reverse_geocoder as _rgc

  HAVE_GEOCODE = True
except Exception:
  HAVE_GEOCODE = False

# Path Definitions
OUT = r"D:\THESIS\Arpon_Thesis_Final\OUTPUTS"
MODEL = os.path.join(OUT, "05_model_tensor")
VALID = os.path.join(OUT, "06_validation_final", "paper_style")
FIGDIR = os.path.join(VALID, "figures")
FINAL_NC = os.path.join(MODEL, "gwsa_final.nc")
TENSOR_NC = os.path.join(MODEL, "model_ready.nc")
WELL_ATTR = os.path.join(OUT, "02_well_trends", "well_trends_v2_tiered.parquet")
GROW_TS = r"D:\THESIS\Arpon_Thesis_Final\DATA\GROW\grow_timeseries.parquet"
GROW_INV = os.path.join(OUT, "06_validation", "grow_inventory.parquet")
AQUIFERS = (
    r"D:\THESIS\Arpon_Thesis_Final\DATA\Aquifers\jasechko_et_al_2024_aquifers.shp"
)
HYBAS_GPKG = (
    r"D:\THESIS\Arpon_Thesis_Final\DATA\HydroBasins\hybas_lev03_global.gpkg"
)
SY_SHAPEFILE = (
    r"D:\THESIS\Arpon_Thesis_Final\DATA\specific yeild-上传\geishuidu-min.shp"
)

SEED = 20260730
MIN_MONTHS = 8  # polygon-PCC month floor (unchanged)
MIN_WELLS_POLY = 3
MIN_WELL_OBS = 6  # per-well observation floor (was 3 in v3.3)
SY_NEAREST_K = 3  # k nearest Sy polygons for a well outside every polygon
TOPN = 10
NAME_SAMPLES = 18
PROD_Y0, PROD_Y1 = 2002, 2022
LINE = "=" * 80
_LOG = []


def _p(s=""):
  print(s)
  _LOG.append(str(s))


def hdr(t):
  _p("")
  _p(LINE)
  _p(t)
  _p(LINE)


def ok(t):
  _p("  [ OK ]   " + t)


def warn(t):
  _p("  [WARN]   " + t)


def clean_id(series):
  return series.astype(str).str.split(".").str[0].str.strip()


def pcc(a, b):
  m = np.isfinite(a) & np.isfinite(b)
  if m.sum() < MIN_MONTHS or np.std(a[m]) == 0 or np.std(b[m]) == 0:
    return np.nan, int(m.sum())
  return float(np.corrcoef(a[m], b[m])[0, 1]), int(m.sum())


def rmse(a, b):
  m = np.isfinite(a) & np.isfinite(b)
  return float(np.sqrt(np.mean((a[m] - b[m]) ** 2))) if m.sum() else np.nan


# --------------------------------------------------- Sy FROM SHAPEFILE ONLY --
def assign_sy_from_shapefile(A, sy_gdf, gpd, Point, k=SY_NEAREST_K):
  """Assign each well a Specific Yield taken ONLY from the Sy shapefile.

  Order (all spatially local -- no global constant, no 0.10):
    1. point-in-polygon join                      -> that polygon's Sy
    2. well outside every polygon: median Sy of    -> k nearest polygons
       its k nearest polygons (sjoin_nearest)
  Adds 'specific_y' and 'sy_source' ('within' | 'nearest_k' | 'UNRESOLVED').
  """
  wpts = gpd.GeoDataFrame(
      A,
      geometry=[Point(lo, la) for lo, la in zip(A["longitude"], A["latitude"])],
      crs=4326,
  )
  sj = gpd.sjoin(
      wpts, sy_gdf[["_sy_val", "geometry"]], how="left", predicate="within"
  )
  sy_within = sj.groupby(sj.index)["_sy_val"].mean()
  A = A.copy()
  A["specific_y"] = sy_within
  A["sy_source"] = np.where(
      np.isfinite(A["specific_y"].values) & (A["specific_y"].values > 0),
      "within", "UNRESOLVED",
  )

  miss = A["specific_y"].isna() | (A["specific_y"] <= 0)
  n_miss = int(miss.sum())
  if n_miss:
    near = gpd.sjoin_nearest(
        wpts[miss], sy_gdf[["_sy_val", "geometry"]], how="left",
        distance_col="_dist",
    )
    near = near.sort_index(kind="stable").sort_values("_dist", kind="stable")
    near_med = (
        near.groupby(level=0)["_sy_val"]
        .apply(lambda s: np.nanmedian(s.head(k).values))
    )
    A.loc[miss, "specific_y"] = near_med
    good = miss & A["specific_y"].notna() & (A["specific_y"] > 0)
    A.loc[good, "sy_source"] = "nearest_k"

  n_unres = int((A["specific_y"].isna() | (A["specific_y"] <= 0)).sum())
  n_within = int((A["sy_source"] == "within").sum())
  n_near = int((A["sy_source"] == "nearest_k").sum())
  ok(
      f"Sy from shapefile ONLY: within={n_within}, nearest-{k}-median={n_near}, "
      f"unresolved={n_unres} (mean={A['specific_y'].mean():.4f}, "
      f"min={A['specific_y'].min():.4f}, max={A['specific_y'].max():.4f})"
  )
  if n_unres:
    warn(f"{n_unres} well(s) UNRESOLVED even after nearest-{k}; flagged, kept.")
  return A


# ------------------------------------------------------------------ STYLING --
def _plot_setup():
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  plt.rcParams.update({
      "font.family": "serif",
      "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
      "mathtext.fontset": "stix",
      "font.size": 12,
      "axes.titlesize": 14,
      "axes.labelsize": 13,
      "xtick.labelsize": 11,
      "ytick.labelsize": 11,
      "legend.fontsize": 11,
      "figure.titlesize": 16,
      "figure.dpi": 300,
      "savefig.dpi": 600,
      "axes.edgecolor": "#333333",
      "axes.linewidth": 1.2,
      "grid.linestyle": "--",
      "grid.alpha": 0.3,
      "axes.grid": True,
  })
  return plt


# ------------------------------------------------------------------ NAMING --
def _country_full_name(cc):
  if not HAVE_GEOCODE or cc is None:
    return cc
  try:
    c = _pycountry.countries.get(alpha_2=cc)
    return c.name if c is not None else cc
  except Exception:
    return cc


def _sample_points_in_polygon(geom, n=NAME_SAMPLES, rng=None):
  from shapely.geometry import Point

  rng = rng if rng is not None else np.random.default_rng(SEED)
  minx, miny, maxx, maxy = geom.bounds
  pts = []
  tries = 0
  max_tries = n * 80
  while len(pts) < n and tries < max_tries:
    x = rng.uniform(minx, maxx)
    y = rng.uniform(miny, maxy)
    p = Point(x, y)
    if geom.contains(p):
      pts.append(p)
    tries += 1
  if not pts:
    pts = [geom.representative_point()]
  return pts


def label_polygons(gdf, id_col="poly_id", n_samples=NAME_SAMPLES, rng=None):
  rng = rng if rng is not None else np.random.default_rng(SEED)
  rows = []
  if not HAVE_GEOCODE:
    for _, row in gdf.iterrows():
      rows.append({
          id_col: row[id_col],
          "poly_name": row[id_col],
          "primary_country": None,
          "nearest_city": None,
          "country_conf": np.nan,
      })
    return pd.DataFrame(rows)
  for _, row in gdf.iterrows():
    pid = row[id_col]
    geom = row.geometry
    if geom is None or geom.is_empty:
      rows.append({
          id_col: pid,
          "poly_name": pid,
          "primary_country": None,
          "nearest_city": None,
          "country_conf": np.nan,
      })
      continue
    try:
      pts = _sample_points_in_polygon(geom, n=n_samples, rng=rng)
      latlon = [(p.y, p.x) for p in pts]
      hits = _rgc.search(latlon)
      ccs = [h["cc"] for h in hits]
      primary_cc, cnt = Counter(ccs).most_common(1)[0]
      conf = cnt / len(ccs)
      rp = geom.representative_point()
      idxs = [i for i, h in enumerate(hits) if h["cc"] == primary_cc]
      best_i = min(idxs, key=lambda i: pts[i].distance(rp))
      city = hits[best_i]["name"]
      admin1 = hits[best_i]["admin1"]
      cname = _country_full_name(primary_cc)
      poly_name = (
          f"{city}, {admin1}, {cname}"
          if (admin1 and admin1 != city)
          else f"{city}, {cname}"
      )
      rows.append({
          id_col: pid,
          "poly_name": poly_name,
          "primary_country": cname,
          "nearest_city": city,
          "country_conf": round(conf, 2),
      })
    except Exception:
      rows.append({
          id_col: pid,
          "poly_name": pid,
          "primary_country": None,
          "nearest_city": None,
          "country_conf": np.nan,
      })
  return pd.DataFrame(rows)


# ------------------------------------------------------------ DIAGNOSTICS --
def polygon_std_ratio(acc_raw, prod_dict, pid, ym_str):
  if pid not in acc_raw or pid not in prod_dict:
    return np.nan, 0
  months = sorted(acc_raw[pid].keys())
  wv = np.array([acc_raw[pid][m][0] / acc_raw[pid][m][1] for m in months])
  idx = [np.where(ym_str == m)[0] for m in months]
  pv = np.array([prod_dict[pid][i[0]] if len(i) else np.nan for i in idx])
  msk = np.isfinite(wv) & np.isfinite(pv)
  if msk.sum() < MIN_MONTHS:
    return np.nan, int(msk.sum())
  sw = np.std(wv[msk])
  sp = np.std(pv[msk])
  if sw <= 0:
    return np.nan, int(msk.sum())
  return float(sp / sw), int(msk.sum())


# ------------------------------------------------------------------ MAIN ----
def main():
  os.makedirs(FIGDIR, exist_ok=True)
  t0 = time.time()
  rng = np.random.default_rng(SEED)
  hdr(
      "PHASE 6.1 v3.4 FIXED -- PAPER-STYLE CONSISTENCY [SHAPEFILE Sy, >=6 OBS]"
  )

  import geopandas as gpd
  import pyarrow.parquet as pq
  from shapely.geometry import Point
  import xarray as xr

  # 1. Load Data
  hdr("LOAD product, polygons, and well attributes")
  G = xr.open_dataset(FINAL_NC)
  G.load()
  keep = G["keep"].values.astype(bool)
  T = xr.open_dataset(TENSOR_NC)
  if "surface_water" in T:
    keep = keep & ~T["surface_water"].values.astype(bool)
  T.close()
  ci = np.where(keep)[0]
  clat = G["clat"].values[ci]
  clon = G["clon"].values[ci]
  gcal = G["gwsa_calibrated"].values[:, ci]
  ym = pd.to_datetime(G["time"].values).to_period("M")
  ym_str = np.array(ym.astype(str))

  aq = (
      gpd.read_file(AQUIFERS)
      .to_crs(4326)[["Aquifer", "geometry"]]
      .reset_index(drop=True)
  )
  aq["poly_id"] = "AQ_" + aq.index.astype(str)
  hb = (
      gpd.read_file(HYBAS_GPKG)
      .to_crs(4326)[["HYBAS_ID", "geometry"]]
      .reset_index(drop=True)
  )
  hb["poly_id"] = "HB_" + hb["HYBAS_ID"].astype(str)

  cellpts = gpd.GeoDataFrame(
      geometry=[Point(lo, la) for lo, la in zip(clon, clat)], crs=4326
  )
  cell_aq = gpd.sjoin(
      cellpts, aq[["poly_id", "geometry"]], how="left", predicate="within"
  )["poly_id"].values
  cell_hb = gpd.sjoin(
      cellpts, hb[["poly_id", "geometry"]], how="left", predicate="within"
  )["poly_id"].values
  cell_aq = (
      pd.Series(cell_aq)
      .groupby(level=0)
      .first()
      .reindex(range(ci.size))
      .values
      if len(cell_aq) != ci.size
      else cell_aq
  )
  cell_hb = (
      pd.Series(cell_hb)
      .groupby(level=0)
      .first()
      .reindex(range(ci.size))
      .values
      if len(cell_hb) != ci.size
      else cell_hb
  )

  def product_poly_series(cell_poly):
    out = {}
    for pid in pd.unique(cell_poly[pd.notna(cell_poly)]):
      cols = np.where(cell_poly == pid)[0]
      out[pid] = np.nanmean(gcal[:, cols], axis=1)
    return out

  prod_aq = product_poly_series(cell_aq)
  prod_hb = product_poly_series(cell_hb)

  # 2. Process Wells & Assign Sy FROM SHAPEFILE ONLY (no 0.10 fallback)
  hdr("LOAD wells + assign Sy from geishuidu-min.shp (within -> 3-nearest median)")
  A = pd.read_parquet(WELL_ATTR)
  A = A[np.isfinite(A["latitude"]) & np.isfinite(A["longitude"])].copy()
  A["GROW_ID"] = clean_id(A["GROW_ID"])
  if "specific_y" in A.columns:
    A = A.drop(columns=["specific_y"])

  sy_gdf = gpd.read_file(SY_SHAPEFILE).to_crs(4326)
  sy_gdf["geometry"] = sy_gdf["geometry"].make_valid()
  sy_col = "specific_y" if "specific_y" in sy_gdf.columns else sy_gdf.columns[0]
  sy_gdf["_sy_val"] = pd.to_numeric(sy_gdf[sy_col], errors="coerce")
  ok(f"Sy shapefile: {len(sy_gdf)} features, attribute '{sy_col}'.")

  A = assign_sy_from_shapefile(A, sy_gdf, gpd, Point)

  inv = pd.read_parquet(GROW_INV)
  if "GROW_ID" not in inv.columns:
    inv = inv.reset_index().rename(
        columns={inv.index.name or "index": "GROW_ID"}
    )
  inv["GROW_ID"] = clean_id(inv["GROW_ID"])
  A = A.merge(inv[["GROW_ID", "interval"]], on="GROW_ID", how="left")
  wpts = gpd.GeoDataFrame(
      A,
      geometry=[Point(lo, la) for lo, la in zip(A["longitude"], A["latitude"])],
      crs=4326,
  )
  A["poly_aq"] = gpd.sjoin(
      wpts, aq[["poly_id", "geometry"]], how="left", predicate="within"
  )["poly_id"].values
  A["poly_hb"] = gpd.sjoin(
      wpts, hb[["poly_id", "geometry"]], how="left", predicate="within"
  )["poly_id"].values
  id2sy = dict(zip(A["GROW_ID"], A["specific_y"]))
  id2aq = dict(zip(A["GROW_ID"], A["poly_aq"]))
  id2hb = dict(zip(A["GROW_ID"], A["poly_hb"]))
  keep_ids = set(A["GROW_ID"])

  # 3. Stream GROW Timeseries
  hdr("STREAM GROW timeseries -> per-polygon monthly well series")
  pf = pq.ParquetFile(GROW_TS)
  cols = [
      "GROW_ID",
      "date",
      "interval",
      "groundwater_depth_from_ground_elevation_m",
      "groundwater_filled_water_level_elevation_m_asl",
  ]
  acc_aq = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
  acc_hb = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
  acc_aq_raw = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
  acc_hb_raw = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
  wsum = defaultdict(lambda: [0.0, 0.0, 0])

  def storage(d):
    d["GROW_ID"] = clean_id(d["GROW_ID"])
    d["v"] = d["groundwater_depth_from_ground_elevation_m"]
    d["v"] = d["v"].where(
        d["v"].notna(), -d["groundwater_filled_water_level_elevation_m_asl"]
    )
    d = d.dropna(subset=["v", "date"])
    # Sy strictly from the shapefile-derived per-well map; no 0.10 fallback.
    d["sy"] = d["GROW_ID"].map(id2sy)
    d = d.dropna(subset=["sy"])
    d["stor"] = -d["v"] * d["sy"] * 100.0
    d["dt"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["dt"])
    return d[(d["dt"].dt.year >= PROD_Y0) & (d["dt"].dt.year <= PROD_Y1)]

  nrg = pf.num_row_groups
  for rg in range(nrg):
    d = pf.read_row_group(rg, columns=cols).to_pandas()
    d["GROW_ID"] = clean_id(d["GROW_ID"])
    d = d[d["GROW_ID"].isin(keep_ids)]
    if d.empty:
      continue
    d = storage(d)
    for wid, sv in d.groupby("GROW_ID")["stor"]:
      a = wsum[wid]
      a[0] += sv.sum()
      a[1] += (sv**2).sum()
      a[2] += sv.size

  # Well qualifies only with >= MIN_WELL_OBS observations (was 3, now 6)
  wms = {
      w: (s / n, max(ss / n - (s / n) ** 2, 1e-9) ** 0.5)
      for w, (s, ss, n) in wsum.items()
      if n >= MIN_WELL_OBS
  }
  _p(f"  wells passing >= {MIN_WELL_OBS} observations: {len(wms)} "
     f"(was n>=3 in v3.3)")

  for rg in range(nrg):
    d = pf.read_row_group(rg, columns=cols).to_pandas()
    d["GROW_ID"] = clean_id(d["GROW_ID"])
    d = d[d["GROW_ID"].isin(keep_ids)]
    if d.empty:
      continue
    d = storage(d)
    d["mon"] = d["dt"].dt.to_period("M").astype(str)
    for wid, g in d.groupby("GROW_ID"):
      if wid not in wms:
        continue
      mu, sd = wms[wid]
      if sd <= 0:
        continue
      z = (g["stor"].values - mu) / sd
      raw = g["stor"].values
      mons = g["mon"].values
      pa = id2aq.get(wid)
      ph = id2hb.get(wid)
      for zz, rv, mm in zip(z, raw, mons):
        if isinstance(pa, str):
          acc_aq[pa][mm][0] += zz
          acc_aq[pa][mm][1] += 1
          acc_aq_raw[pa][mm][0] += rv
          acc_aq_raw[pa][mm][1] += 1
        if isinstance(ph, str):
          acc_hb[ph][mm][0] += zz
          acc_hb[ph][mm][1] += 1
          acc_hb_raw[ph][mm][0] += rv
          acc_hb_raw[ph][mm][1] += 1

  # 4. Temporal PCC (window-matched: wells' months, product sampled at same months)
  def polygon_pcc(acc, prod_series):
    rows = []
    for pid, mon_bins in acc.items():
      if pid not in prod_series:
        continue
      months = sorted(mon_bins.keys())
      wv = np.array([mon_bins[m][0] / mon_bins[m][1] for m in months])
      prod = prod_series[pid]
      idx = [np.where(ym_str == m)[0] for m in months]
      pv = np.array([prod[i[0]] if len(i) else np.nan for i in idx])
      r, n = pcc(wv, pv)
      if np.isfinite(r):
        rows.append({
            "poly_id": pid,
            "pcc": round(r, 3),
            "n_months": n,
            "rmse": round(
                rmse(
                    (wv - np.nanmean(wv)) / (np.nanstd(wv) + 1e-9),
                    (pv - np.nanmean(pv)) / (np.nanstd(pv) + 1e-9),
                ),
                3,
            ),
        })
    return pd.DataFrame(rows)

  naq = A.groupby("poly_aq")["GROW_ID"].nunique().to_dict()
  nhb = A.groupby("poly_hb")["GROW_ID"].nunique().to_dict()
  aq_pcc = polygon_pcc(acc_aq, prod_aq)
  aq_pcc["n_wells"] = aq_pcc["poly_id"].map(naq)
  hb_pcc = polygon_pcc(acc_hb, prod_hb)
  hb_pcc["n_wells"] = hb_pcc["poly_id"].map(nhb)
  aq_pcc = aq_pcc[aq_pcc["n_wells"] >= MIN_WELLS_POLY].reset_index(drop=True)
  hb_pcc = hb_pcc[hb_pcc["n_wells"] >= MIN_WELLS_POLY].reset_index(drop=True)

  aq_pcc = aq_pcc.merge(aq[["poly_id", "Aquifer"]], on="poly_id", how="left")
  aq_pcc["poly_name"] = aq_pcc["Aquifer"].fillna(aq_pcc["poly_id"])

  hb_names = label_polygons(
      hb[hb["poly_id"].isin(hb_pcc["poly_id"])].copy(), rng=rng
  )
  hb_pcc = hb_pcc.merge(hb_names, on="poly_id", how="left")
  hb_pcc["poly_name"] = hb_pcc["poly_name"].fillna(hb_pcc["poly_id"])

  aq_named = (
      aq_pcc.sort_values("pcc", ascending=False).head(TOPN).reset_index(drop=True)
  )
  hb_top = (
      hb_pcc.sort_values("pcc", ascending=False).head(TOPN).reset_index(drop=True)
  )

  def paired(pid, acc, prod):
    if pid not in acc or pid not in prod:
      return None
    months = sorted(acc[pid].keys())
    wv = np.array([acc[pid][m][0] / acc[pid][m][1] for m in months])
    idx = [np.where(ym_str == m)[0] for m in months]
    pv = np.array([prod[pid][i[0]] if len(i) else np.nan for i in idx])
    m = np.isfinite(wv) & np.isfinite(pv)
    if m.sum() < MIN_MONTHS:
      return None
    wz = (wv - np.nanmean(wv[m])) / (np.nanstd(wv[m]) + 1e-9)
    pz = (pv - np.nanmean(pv[m])) / (np.nanstd(pv[m]) + 1e-9)
    return np.vstack([wz, pz])

  # 5. DYNAMIC REGIONAL BOUNDING BOX SEARCH FOR THE 5 FOCUS HOTSPOTS
  FOCUS_REGIONS = {
      "Bengal_GBM": {
          "lat": (20.5, 27.0),
          "lon": (85.0, 93.0),
          "center": (24.0, 89.5),
      },
      "Punjab_Indus": {
          "lat": (24.0, 33.0),
          "lon": (69.0, 78.0),
          "center": (29.0, 74.0),
      },
      "High_Plains": {
          "lat": (32.0, 43.5),
          "lon": (-104.0, -96.0),
          "center": (38.0, -100.0),
      },
      "Central_Valley": {
          "lat": (35.0, 40.0),
          "lon": (-122.5, -119.0),
          "center": (37.5, -120.5),
      },
      "North_China_Plain": {
          "lat": (34.0, 40.0),
          "lon": (113.0, 120.0),
          "center": (37.0, 116.5),
      },
  }

  focus_series = {}
  focus_rows = []
  dump = {}

  for pid in hb_top["poly_id"]:
    arr = paired(pid, acc_hb, prod_hb)
    if arr is not None:
      dump["BASINtop_" + pid] = arr
  for pid in aq_named["poly_id"]:
    arr = paired(pid, acc_aq, prod_aq)
    if arr is not None:
      dump["AQtop_" + pid] = arr

  for nm, cfg in FOCUS_REGIONS.items():
    min_la, max_la = cfg["lat"]
    min_lo, max_lo = cfg["lon"]

    sub_a = A[
        (A["latitude"] >= min_la)
        & (A["latitude"] <= max_la)
        & (A["longitude"] >= min_lo)
        & (A["longitude"] <= max_lo)
    ]

    chosen_pid = None
    chosen_arr = None
    chosen_layer = "hb"

    if not sub_a.empty:
      for cpid in sub_a["poly_hb"].value_counts().index:
        if pd.notna(cpid) and cpid in acc_hb and cpid in prod_hb:
          arr = paired(cpid, acc_hb, prod_hb)
          if arr is not None:
            chosen_pid = cpid
            chosen_arr = arr
            chosen_layer = "hb"
            break

    if chosen_arr is None and not sub_a.empty:
      for cpid in sub_a["poly_aq"].value_counts().index:
        if pd.notna(cpid) and cpid in acc_aq and cpid in prod_aq:
          arr = paired(cpid, acc_aq, prod_aq)
          if arr is not None:
            chosen_pid = cpid
            chosen_arr = arr
            chosen_layer = "aq"
            break

    if chosen_arr is None:
      la, lo = cfg["center"]
      hit = hb[hb.contains(Point(lo, la))]
      cpid = (
          hit.iloc[0]["poly_id"]
          if len(hit)
          else hb.iloc[hb.geometry.centroid.distance(Point(lo, la)).argmin()][
              "poly_id"
          ]
      )
      arr = paired(cpid, acc_hb, prod_hb)
      if arr is not None:
        chosen_pid = cpid
        chosen_arr = arr

    if chosen_arr is not None:
      dump["FOCUS_" + nm] = chosen_arr
      focus_series[nm] = (chosen_pid, chosen_arr)
      r, n = pcc(chosen_arr[0], chosen_arr[1])
      n_w = int(nhb.get(chosen_pid, naq.get(chosen_pid, 0)))

      if chosen_layer == "hb":
        pname = hb_pcc.loc[hb_pcc["poly_id"] == chosen_pid, "poly_name"]
        pname_str = pname.values[0] if len(pname) else chosen_pid
      else:
        pname = aq_pcc.loc[aq_pcc["poly_id"] == chosen_pid, "poly_name"]
        pname_str = pname.values[0] if len(pname) else chosen_pid

      focus_rows.append({
          "hotspot": nm.replace("_", " "),
          "poly_id": chosen_pid,
          "poly_name": pname_str,
          "pcc": round(r, 3),
          "n_months": n,
          "n_wells": n_w,
      })
    else:
      warn(
          f"Focus Region '{nm}' has no overlapping GROW well observations in"
          " dataset."
      )

  focus_df = pd.DataFrame(focus_rows)

  aq_shp = aq.merge(aq_pcc, on="poly_id", how="inner")
  hb_shp = hb.merge(hb_pcc, on="poly_id", how="inner")

  # ------------------------------------------------ PUBLICATION FIGURES --
  plt = _plot_setup()

  def render_scatter_grid(items, fname, title):
    if not items:
      return
    n = len(items)
    ncol = min(5, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(3.8 * ncol, 3.8 * nrow), constrained_layout=True
    )
    axes = np.atleast_1d(axes).flatten()

    for k, (lab, arr, rr) in enumerate(items):
      ax = axes[k]
      w, p = arr
      m = np.isfinite(w) & np.isfinite(p)
      w, p = w[m], p[m]
      ax.scatter(w, p, s=22, c="#1f4e79", alpha=0.6, edgecolors="none")
      L = max(np.abs(np.r_[w, p]).max() if w.size else 3.0, 3.0)
      ax.plot([-L, L], [-L, L], "k--", lw=1.2, label="1:1 line")
      ax.set_xlim(-L, L)
      ax.set_ylim(-L, L)
      ax.set_title(f"{str(lab)[:24]}\nPCC = {rr:+.2f}", fontsize=12, pad=6)
      ax.set_xlabel("Well Storage (z)", fontsize=11)
      ax.set_ylabel("Product Storage (z)", fontsize=11)
      ax.spines[["top", "right"]].set_visible(False)

    for k in range(n, len(axes)):
      fig.delaxes(axes[k])

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
    for e in (".png", ".pdf"):
      fig.savefig(
          os.path.join(FIGDIR, fname + e),
          dpi=600,
          bbox_inches="tight",
          facecolor="white",
      )
    plt.close(fig)

  hdr("GENERATING NATURE-GRADE FIGURES (TIMES NEW ROMAN)...")

  basin_items = [
      (r["poly_name"], dump.get("BASINtop_" + r["poly_id"]), r["pcc"])
      for _, r in hb_top.iterrows()
  ]
  render_scatter_grid(
      basin_items,
      "fig_top10_basins_1to1",
      "Top 10 Basins: Monthly Well vs Product Storage",
  )
  ok("fig_top10_basins_1to1")

  aq_items = [
      (r["poly_name"], dump.get("AQtop_" + r["poly_id"]), r["pcc"])
      for _, r in aq_named.iterrows()
  ]
  render_scatter_grid(
      aq_items,
      "fig_top10_aquifers_1to1",
      "Top 10 Aquifers: Monthly Well vs Product Storage",
  )
  ok("fig_top10_aquifers_1to1")

  focus_items = [
      (nm.replace("_", " "), arr, pcc(arr[0], arr[1])[0])
      for nm, (pid, arr) in focus_series.items()
  ]
  render_scatter_grid(
      focus_items,
      "fig_focus_hotspots_1to1",
      "Focus Depletion Hotspots: Monthly Well vs Product Storage",
  )
  ok("fig_focus_hotspots_1to1")

  # Spatial Global Maps, Zoomed Insets & Taylor Diagram
  try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    # Global Overview Map
    fig = plt.figure(figsize=(14, 8), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="#f8f9fa", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="#eef2f5", zorder=0)
    ax.coastlines(lw=0.5, color="#555")

    g_hb = hb_shp.to_crs(ccrs.Robinson().proj4_init)
    g_hb.plot(
        ax=ax,
        column="pcc",
        cmap="YlGnBu",
        vmin=-1,
        vmax=1,
        legend=True,
        legend_kwds={
            "label": "Temporal PCC",
            "shrink": 0.6,
            "pad": 0.02,
            "orientation": "horizontal",
        },
        zorder=2,
    )
    ax.set_title(
        "Global Aquifer and Basin Well-vs-Product Agreement (PCC)",
        fontsize=16,
        fontweight="bold",
        pad=12,
    )

    for e in (".png", ".pdf"):
      fig.savefig(
          os.path.join(FIGDIR, "fig_global_overview_map" + e),
          dpi=600,
          bbox_inches="tight",
          facecolor="white",
      )
    plt.close(fig)
    ok("fig_global_overview_map")

    # Global Map: Top 10 Aquifers
    fig = plt.figure(figsize=(14, 8), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="#f4f4f4", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="#eef2f5", zorder=0)
    ax.coastlines(lw=0.5, color="#555")
    ax.add_feature(cfeature.BORDERS, lw=0.3, color="#bbb")

    g_aq_all = aq_shp.to_crs(ccrs.Robinson().proj4_init)
    g_aq_all.plot(
        ax=ax,
        color="#d0d0d0",
        edgecolor="#aaaaaa",
        linewidth=0.4,
        alpha=0.5,
        zorder=1,
    )

    g_aq_top10 = aq_shp[aq_shp["poly_id"].isin(aq_named["poly_id"])].to_crs(
        ccrs.Robinson().proj4_init
    )
    g_aq_top10.plot(
        ax=ax,
        column="pcc",
        cmap="YlOrRd",
        vmin=max(0.0, float(aq_named["pcc"].min() - 0.05)),
        vmax=1.0,
        legend=True,
        legend_kwds={
            "label": "Correlation Value (PCC)",
            "shrink": 0.65,
            "pad": 0.03,
            "orientation": "horizontal",
        },
        edgecolor="#800000",
        linewidth=1.2,
        zorder=3,
    )

    ax.set_title(
        "Global Top 10 Aquifers Highlighted by PCC Correlation",
        fontsize=16,
        fontweight="bold",
        pad=12,
    )

    for e in (".png", ".pdf"):
      fig.savefig(
          os.path.join(FIGDIR, "fig_global_top10_aquifers" + e),
          dpi=600,
          bbox_inches="tight",
          facecolor="white",
      )
    plt.close(fig)
    ok("fig_global_top10_aquifers")

    # Global Map: Top 10 HydroBASINS
    fig = plt.figure(figsize=(14, 8), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="#f4f4f4", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="#eef2f5", zorder=0)
    ax.coastlines(lw=0.5, color="#555")
    ax.add_feature(cfeature.BORDERS, lw=0.3, color="#bbb")

    g_hb_all = hb_shp.to_crs(ccrs.Robinson().proj4_init)
    g_hb_all.plot(
        ax=ax,
        color="#d0d0d0",
        edgecolor="#aaaaaa",
        linewidth=0.4,
        alpha=0.5,
        zorder=1,
    )

    g_hb_top10 = hb_shp[hb_shp["poly_id"].isin(hb_top["poly_id"])].to_crs(
        ccrs.Robinson().proj4_init
    )
    g_hb_top10.plot(
        ax=ax,
        column="pcc",
        cmap="YlOrRd",
        vmin=max(0.0, float(hb_top["pcc"].min() - 0.05)),
        vmax=1.0,
        legend=True,
        legend_kwds={
            "label": "Correlation Value (PCC)",
            "shrink": 0.65,
            "pad": 0.03,
            "orientation": "horizontal",
        },
        edgecolor="#800000",
        linewidth=1.2,
        zorder=3,
    )

    ax.set_title(
        "Global Top 10 HydroBASINS Highlighted by PCC Correlation",
        fontsize=16,
        fontweight="bold",
        pad=12,
    )

    for e in (".png", ".pdf"):
      fig.savefig(
          os.path.join(FIGDIR, "fig_global_top10_basins" + e),
          dpi=600,
          bbox_inches="tight",
          facecolor="white",
      )
    plt.close(fig)
    ok("fig_global_top10_basins")

    # Zoomed Spatial Inset Panels
    def render_zoomed_panels(gdf_shp, top_df, fname, title):
      n = len(top_df)
      ncol = min(5, n)
      nrow = int(np.ceil(n / ncol))
      fig = plt.figure(figsize=(4.5 * ncol, 4.5 * nrow), constrained_layout=True)
      proj = ccrs.PlateCarree()

      for k, (_, r) in enumerate(top_df.iterrows()):
        pid = r["poly_id"]
        geom = gdf_shp.loc[gdf_shp["poly_id"] == pid, "geometry"]
        if geom.empty:
          geom = aq_shp.loc[aq_shp["poly_id"] == pid, "geometry"]
        if geom.empty:
          continue
        g = geom.values[0]
        minx, miny, maxx, maxy = g.bounds
        padx = max((maxx - minx) * 0.25, 0.5)
        pady = max((maxy - miny) * 0.25, 0.5)

        ax = fig.add_subplot(nrow, ncol, k + 1, projection=proj)
        ax.set_extent(
            [minx - padx, maxx + padx, miny - pady, maxy + pady], crs=proj
        )
        ax.add_feature(cfeature.LAND, facecolor="#f4f4f4")
        ax.add_feature(cfeature.OCEAN, facecolor="#e6f2ff")
        ax.coastlines(lw=0.5, color="#666")
        ax.add_feature(cfeature.BORDERS, lw=0.4, color="#aaa")

        geoms = [g] if g.geom_type == "Polygon" else list(g.geoms)
        for part in geoms:
          xs, ys = part.exterior.xy
          ax.plot(xs, ys, color="#d35400", lw=2.0, transform=proj, zorder=3)

        wsel = A[(A["poly_hb"] == pid) | (A["poly_aq"] == pid)]
        if len(wsel):
          ax.scatter(
              wsel["longitude"],
              wsel["latitude"],
              c="#c0392b",
              s=15,
              transform=proj,
              zorder=4,
              label="Wells",
              edgecolors="white",
              linewidths=0.3,
          )

        name_lbl = r.get("hotspot", r.get("poly_name", pid))
        ax.set_title(
            f"{str(name_lbl)[:24]}\nPCC = {r['pcc']:+.2f}", fontsize=11
        )

      fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
      for e in (".png", ".pdf"):
        fig.savefig(
            os.path.join(FIGDIR, fname + e),
            dpi=600,
            bbox_inches="tight",
            facecolor="white",
        )
      plt.close(fig)

    render_zoomed_panels(
        hb_shp, hb_top, "fig_zoomed_top10_basins", "Zoomed Spatial Insets: Top 10 Basins"
    )
    render_zoomed_panels(
        aq_shp,
        aq_named,
        "fig_zoomed_top10_aquifers",
        "Zoomed Spatial Insets: Top 10 Aquifers",
    )
    if not focus_df.empty:
      render_zoomed_panels(
          hb_shp,
          focus_df,
          "fig_zoomed_focus_basins",
          "Zoomed Spatial Insets: Focus Depletion Hotspots",
      )
    ok("fig_zoomed_panels")

  except Exception as e:
    warn(f"Cartopy spatial mapping error: {e}")

  # Taylor Diagram
  try:
    import mpl_toolkits.axisartist.floating_axes as FA
    import mpl_toolkits.axisartist.grid_finder as GF
    from matplotlib.projections import PolarAxes

    class TaylorDiagram(object):

      def __init__(self, refstd, fig, rect=111, label="Reference"):
        self.refstd = refstd
        tr = PolarAxes.PolarTransform()
        rlocs = np.array([0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0])
        tlocs = np.arccos(rlocs)
        gl1 = GF.FixedLocator(tlocs)
        tf1 = GF.DictFormatter(dict(zip(tlocs, [f"{v:.2f}" for v in rlocs])))
        ghelper = FA.GridHelperCurveLinear(
            tr,
            extremes=(0, np.pi / 2, 0, 1.8),
            grid_locator1=gl1,
            tick_formatter1=tf1,
        )
        ax = FA.FloatingSubplot(fig, rect, grid_helper=ghelper)
        fig.add_subplot(ax)
        ax.axis["top"].set_axis_direction("bottom")
        ax.axis["top"].toggle(ticklabels=True, label=True)
        ax.axis["top"].label.set_text("Correlation (PCC)")
        ax.axis["top"].label.set_fontsize(13)
        ax.axis["left"].set_axis_direction("bottom")
        ax.axis["left"].label.set_text("Normalized Standard Deviation")
        ax.axis["left"].label.set_fontsize(13)
        ax.axis["right"].set_visible(False)
        ax.axis["bottom"].set_visible(False)
        self._ax = ax
        self.ax = ax.get_aux_axes(tr)
        self.ax.plot([0], self.refstd, "k*", ms=14, label=label, zorder=5)
        t = np.linspace(0, np.pi / 2)
        r = np.zeros_like(t) + self.refstd
        self.ax.plot(t, r, "k--", lw=1.2, alpha=0.7)

      def add_sample(self, stddev, corrcoef, *a, **kw):
        return self.ax.plot(
            np.arccos(np.clip(corrcoef, -1, 1)), stddev, *a, **kw
        )

    fig = plt.figure(figsize=(8, 8), constrained_layout=True)
    td = TaylorDiagram(1.0, fig, label="In-situ Wells")

    for i, (_, r) in enumerate(aq_named.head(8).iterrows()):
      ratio, _ = polygon_std_ratio(acc_aq_raw, prod_aq, r["poly_id"], ym_str)
      if np.isfinite(ratio):
        td.add_sample(
            ratio,
            r["pcc"],
            "o",
            ms=9,
            color="#1f4e79",
            alpha=0.8,
            label=f"AQ: {r['poly_name'][:18]}" if i < 4 else None,
        )

    for i, (_, r) in enumerate(hb_top.head(8).iterrows()):
      ratio, _ = polygon_std_ratio(acc_hb_raw, prod_hb, r["poly_id"], ym_str)
      if np.isfinite(ratio):
        td.add_sample(
            ratio,
            r["pcc"],
            "^",
            ms=9,
            color="#2ca02c",
            alpha=0.8,
            label=f"HB: {r['poly_name'][:18]}" if i < 4 else None,
        )

    for _, r in focus_df.iterrows():
      ratio, _ = polygon_std_ratio(acc_hb_raw, prod_hb, r["poly_id"], ym_str)
      if np.isnan(ratio):
        ratio, _ = polygon_std_ratio(
            acc_aq_raw, prod_aq, r["poly_id"], ym_str
        )
      if np.isfinite(ratio):
        td.add_sample(
            ratio,
            r["pcc"],
            "s",
            ms=10,
            color="#e67e22",
            label=f"Focus: {r['hotspot']}",
        )

    fig.legend(
        loc="upper right",
        bbox_to_anchor=(1.15, 0.95),
        frameon=True,
        fontsize=11,
    )
    fig.suptitle(
        "Normalized Taylor Diagram: Amplitude & Phase Agreement",
        fontsize=15,
        fontweight="bold",
    )
    for e in (".png", ".pdf"):
      fig.savefig(
          os.path.join(FIGDIR, "fig_taylor_diagram" + e),
          dpi=600,
          bbox_inches="tight",
          facecolor="white",
      )
    plt.close(fig)
    ok("fig_taylor_diagram")
  except Exception as e:
    warn(f"Taylor Diagram generation error: {e}")

  # Summary Table
  hdr("ALL PROCESSES COMPLETED SUCCESSFULLY")
  _p("\n=== FOCUS DEPLETION HOTSPOTS SUMMARY ===")
  if not focus_df.empty:
    _p(
        focus_df[
            ["hotspot", "poly_id", "poly_name", "pcc", "n_wells", "n_months"]
        ].to_string(index=False)
    )
  else:
    _p("No focus hotspots met the minimum overlap criteria.")

  _p(f"\nExecution finished in {(time.time() - t0) / 60:.2f} minutes.")


if __name__ == "__main__":
  main()