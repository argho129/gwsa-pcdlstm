#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
PHASE 2.1 -- WELL TREND COMPUTATION (period-matched 2002-2022)
--------------------------------------------------------------------------------
[0] Diagnostic: where does the Phase 1 +/-100 cm cap bite?
[1] Chunked Sen's slope per well from grow_timeseries.parquet, 2002-2022 ONLY.
[2] Sign normalisation: NEGATIVE = declining water table = storage loss.
      depth-from-ground   : depth INCREASES as table falls -> slope_storage = -slope_depth
      water-level-elev    : elevation DECREASES as table falls -> slope_storage = +slope_elev
[3] Physical clip |slope| <= 5 m/yr (GROW contains values down to -268 m/yr).
[4] Specific yield join (Sun et al. shapefile, 'specific_y', fraction 0.01-0.18).
[5] Storage trend (cm/yr) = water-level trend (m/yr) x Sy x 100.
[6] Physical screen on storage trend: |storage trend| <= 25 cm/yr.
[7] Comparison set A: GROW's own full-record precomputed slopes, same treatment.
Writes -> OUTPUTS\02_well_trends\well_trends.parquet  (+ diagnostics + figures)
================================================================================
"""
import os, sys, warnings, traceback
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ CONFIG ----
SAMPLE_MODE = False      # True = 2 row groups only (fast check). False = full run.

DATA     = r"D:\THESIS\Arpon_Thesis_Final\DATA"
OUT_ROOT = r"D:\THESIS\Arpon_Thesis_Final\OUTPUTS"
OUT_DIR  = os.path.join(OUT_ROOT, "02_well_trends")
GWSA_NC  = os.path.join(OUT_ROOT, "01_gwsa_monthly", "gwsa_monthly.nc")
GROW_DIR = os.path.join(DATA, "GROW")
TS_PQ    = os.path.join(GROW_DIR, "grow_timeseries.parquet")
AT_PQ    = os.path.join(GROW_DIR, "grow_attributes.parquet")
SY_SHP   = os.path.join(DATA, "specific yeild-上传", "geishuidu-min.shp")

YR0, YR1     = 2002, 2022
MIN_OBS      = 8        # >=8 annual values in window (mirrors Sun et al. screening)
MAX_ABS_MYR  = 5.0      # physical clip on water-level trend, m/yr
CAP_CM       = 100.0    # Phase-1 Rodell cap, for the diagnostic
MAX_STORAGE_CM = 25.0  # physical clip on storage trend, cm/yr

DEPTH_COL = "groundwater_filled_depth_from_ground_elevation_m"
DEPTH_RAW = "groundwater_depth_from_ground_elevation_m"
ELEV_COL  = "groundwater_filled_water_level_elevation_m_asl"
ELEV_RAW  = "groundwater_water_level_elevation_m_asl"

LINE = "=" * 78
_LOG = []
def _p(s=""): print(s); _LOG.append(str(s))
def hdr(t): _p("\n"+LINE); _p(t); _p(LINE)
def ok(t): _p("  [ OK ]   "+t)
def warn(t): _p("  [WARN]   "+t)
def fail(t): _p("  [FAIL]   "+t)

# ---------------------------------------------------------------- Sen slope ---
def sen_slope(x, y):
    """Theil-Sen median of pairwise slopes. x,y are 1-D finite arrays."""
    n = x.size
    if n < 2: return np.nan
    # pairwise slopes, vectorised upper triangle
    xi = x[:, None]; xj = x[None, :]
    yi = y[:, None]; yj = y[None, :]
    dx = xj - xi
    with np.errstate(invalid="ignore", divide="ignore"):
        s = (yj - yi) / dx
    iu = np.triu_indices(n, k=1)
    s = s[iu]
    s = s[np.isfinite(s)]
    return float(np.median(s)) if s.size else np.nan

# ============================================================ STEP 0 ==========
def diagnose_cap():
    hdr("STEP 0 -- DIAGNOSTIC: where does the Phase-1 +/-100 cm cap bite?")
    try:
        import xarray as xr
    except Exception:
        warn("xarray missing; skipping cap diagnostic."); return
    if not os.path.exists(GWSA_NC):
        warn(f"not found: {GWSA_NC}"); return
    ds = xr.open_dataset(GWSA_NC)
    G  = ds["gwsa"].values
    LAT= ds["lat"].values; LON = ds["lon"].values
    dom= ds["domain"].values.astype(bool)
    ds.close()
    at_cap = np.isfinite(G) & (np.abs(G) >= CAP_CM - 1e-3)
    n_tot  = int(np.isfinite(G).sum())
    n_cap  = int(at_cap.sum())
    _p(f"  finite cell-months: {n_tot:,}   at-cap: {n_cap:,} ({100*n_cap/max(n_tot,1):.3f}%)")
    per_cell = at_cap.sum(axis=0)
    jj, ii = np.where(per_cell > 0)
    _p(f"  distinct cells ever capped: {len(jj):,}")
    if len(jj):
        lats = LAT[jj]
        for lo_b, hi_b, lbl in [(-90,-60,"S of 60S"), (-60,0,"60S-0"), (0,30,"0-30N"),
                                (30,50,"30-50N"), (50,60,"50-60N"), (60,90,"N of 60N")]:
            m = (lats >= lo_b) & (lats < hi_b)
            _p(f"     {lbl:<10}: {int(m.sum()):>6} capped cells "
               f"({100*m.sum()/len(jj):5.1f}%)")
        _p("  top-10 most-capped cells:")
        order = np.argsort(per_cell[jj, ii])[::-1][:10]
        for k in order:
            _p(f"     lat={LAT[jj[k]]:+7.2f} lon={LON[ii[k]]:+8.2f}  "
               f"{int(per_cell[jj[k], ii[k]]):>4} months capped")
        # hotspot check: are depletion zones being clipped?
        _p("  hotspot cap check (capped months at key cells):")
        for nm, la, lo in [("NW India",30.5,75.5),("N China Plain",37.0,116.0),
                           ("Arabia",24.0,45.0),("Central Valley",36.5,-120.0)]:
            j = int(np.argmin(np.abs(LAT-la))); i = int(np.argmin(np.abs(LON-lo)))
            _p(f"     {nm:<15}: {int(per_cell[j,i])} months at cap")
        _p("  0 capped months at the hotspots means the cap only suppresses artifacts.")
        _p("  Many capped months at the hotspots would mean the cap clips real signal.")

# ============================================================ STEP 1 ==========
def compute_well_trends():
    hdr("STEP 1 -- Sen's slope per well, 2002-2022 (chunked)")
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(TS_PQ)
    ngr = pf.metadata.num_row_groups
    use = 2 if SAMPLE_MODE else ngr
    _p(f"  row groups: {ngr}   processing: {use}  (SAMPLE_MODE={SAMPLE_MODE})")

    cols = ["GROW_ID","year",DEPTH_COL,DEPTH_RAW,ELEV_COL,ELEV_RAW]
    avail = set(pf.schema.names)
    cols = [c for c in cols if c in avail]
    _p(f"  reading columns: {cols}")

    carry = pd.DataFrame()      # rows of a well split across row-group boundary
    recs  = []
    for g in range(use):
        tb = pf.read_row_group(g, columns=cols).to_pandas()
        tb = tb[(tb["year"] >= YR0) & (tb["year"] <= YR1)]
        if tb.empty:
            continue
        tb = pd.concat([carry, tb], ignore_index=True) if len(carry) else tb
        # hold back the last GROW_ID (may continue into next row group)
        if g < use - 1:
            last = tb["GROW_ID"].iloc[-1]
            carry = tb[tb["GROW_ID"] == last]
            tb    = tb[tb["GROW_ID"] != last]
        else:
            carry = pd.DataFrame()
        if tb.empty:
            continue
        for wid, grp in tb.groupby("GROW_ID", sort=False):
            rec = _well_record(wid, grp)
            if rec: recs.append(rec)
        if (g+1) % 5 == 0 or g == use-1:
            _p(f"     row group {g+1}/{use}: wells with slope so far = {len(recs):,}")

    df = pd.DataFrame(recs)
    ok(f"wells with 2002-2022 Sen slope: {len(df):,}")
    return df

def _well_record(wid, grp):
    """Return dict with period-matched slope for one well, or None."""
    # prefer filled depth, else raw depth, else elevation
    src = None
    for c, kind in ((DEPTH_COL,"depth"),(DEPTH_RAW,"depth"),
                    (ELEV_COL,"elev"),(ELEV_RAW,"elev")):
        if c in grp.columns:
            v = grp[c].values.astype("float64")
            if np.isfinite(v).sum() >= MIN_OBS:
                src = (c, kind, v); break
    if src is None: return None
    col, kind, v = src
    yr = grp["year"].values.astype("float64")
    m  = np.isfinite(v) & np.isfinite(yr)
    if m.sum() < MIN_OBS: return None
    # collapse duplicate years (yearly data should be unique, but be safe)
    ser = pd.Series(v[m]).groupby(yr[m]).mean()
    xx  = ser.index.values.astype("float64"); yy = ser.values
    if xx.size < MIN_OBS: return None
    s = sen_slope(xx, yy)                 # m per year, in the units of `col`
    if not np.isfinite(s): return None
    # SIGN NORMALISATION -> negative = declining water table = storage loss
    slope_wl = -s if kind == "depth" else s
    return {"GROW_ID": wid, "n_obs": int(xx.size),
            "yr_first": float(xx.min()), "yr_last": float(xx.max()),
            "src_col": col, "src_kind": kind,
            "slope_wl_m_yr": float(slope_wl)}

# ============================================================ STEP 2 ==========
def attach_meta_sy(df):
    hdr("STEP 2 -- attach coordinates + specific yield")

    # Deduplication to catch potential row-group boundary overlaps
    n_raw = len(df)
    df = df.drop_duplicates(subset="GROW_ID", keep="first")
    if len(df) < n_raw:
        warn(f"deduplicated GROW_ID: {n_raw:,} -> {len(df):,} "
             f"({n_raw-len(df):,} duplicates from row-group boundaries)")

    at = pd.read_parquet(AT_PQ, columns=[
        "GROW_ID","latitude","longitude","country","interval","length_years",
        "gap_fraction","plateaus","outliers_change_points","negative_signs_wtd",
        "trend_slope_m_year-1","reference_point","aquifer_type_class"])
    df = df.merge(at, on="GROW_ID", how="left")
    _p(f"  merged attributes: {len(df):,} rows")

    # QC filter (same as Phase 0 gate)
    n0 = len(df)
    df = df[df["length_years"] >= 10]
    df = df[df["gap_fraction"] <= 0.10]
    df = df[df["interval"].astype(str).isin(["YS","MS"])]
    df = df[~df["plateaus"].astype(str).str.lower().isin(["true","1","1.0"])]
    df = df[~df["outliers_change_points"].astype(str).str.lower().isin(["true","1","1.0"])]
    df = df[~df["negative_signs_wtd"].astype(str).isin(["All","Some"])]
    ok(f"after QC filter: {len(df):,} / {n0:,}")

    # physical clip
    n1 = len(df)
    bad = df["slope_wl_m_yr"].abs() > MAX_ABS_MYR
    _p(f"  |slope| > {MAX_ABS_MYR} m/yr removed: {int(bad.sum()):,}")
    df = df[~bad]
    ok(f"after physical clip: {len(df):,} / {n1:,}")

    # specific yield spatial join
    import geopandas as gpd
    from shapely.geometry import Point
    sy = gpd.read_file(SY_SHP)[["specific_y","geometry"]]
    if sy.crs is None: sy = sy.set_crs("EPSG:4326")
    elif sy.crs.to_epsg() != 4326: sy = sy.to_crs("EPSG:4326")
    g = gpd.GeoDataFrame(df, geometry=[Point(a,b) for a,b in
                          zip(df["longitude"], df["latitude"])], crs="EPSG:4326")
    j = gpd.sjoin(g, sy, how="left", predicate="within").drop(columns=["index_right"])
    j = j[~j.index.duplicated(keep="first")]

    # Sy processing and imputation flag
    n_sy = j["specific_y"].notna().sum()
    j["sy_imputed"] = j["specific_y"].isna()          # flag BEFORE filling
    med_sy = float(sy["specific_y"].median())
    j["specific_y"] = j["specific_y"].fillna(med_sy)
    ok(f"wells with Sy from shapefile: {n_sy:,} / {len(j):,}")
    warn(f"Sy median-filled for {int(j['sy_imputed'].sum()):,} wells (Sy={med_sy:.3f})")

    return pd.DataFrame(j.drop(columns="geometry")), n_sy

# ============================================================ STEP 3 ==========
def to_storage(df):
    hdr("STEP 3 -- convert water-level trend -> storage trend (cm/yr)")
    # storage (cm/yr) = water-level trend (m/yr) * Sy * 100
    df["slope_storage_cm_yr"] = df["slope_wl_m_yr"] * df["specific_y"] * 100.0

    # Physical screen on implied storage trend
    n_pre = df["slope_storage_cm_yr"].notna().sum()
    over  = df["slope_storage_cm_yr"].abs() > MAX_STORAGE_CM
    _p(f"  physical screen |storage trend| > {MAX_STORAGE_CM} cm/yr: "
       f"{int(over.sum()):,} wells removed ({100*over.sum()/max(n_pre,1):.2f}%)")
    df = df[~over].copy()

    # comparison set A: GROW full-record precomputed slope, same treatment
    a = df["trend_slope_m_year-1"].astype("float64")
    a = a.where(a.abs() <= MAX_ABS_MYR)
    df["slopeA_storage_cm_yr"] = a * df["specific_y"] * 100.0

    s = df["slope_storage_cm_yr"]
    _p(f"  B (2002-2022 period-matched)  n={s.notna().sum():,}")
    _p(f"     min={s.min():.3f} p1={s.quantile(.01):.3f} p50={s.median():.3f} "
       f"p99={s.quantile(.99):.3f} max={s.max():.3f} cm/yr")
    sa = df["slopeA_storage_cm_yr"]
    _p(f"  A (GROW full-record)          n={sa.notna().sum():,}")
    _p(f"     min={sa.min():.3f} p1={sa.quantile(.01):.3f} p50={sa.median():.3f} "
       f"p99={sa.quantile(.99):.3f} max={sa.max():.3f} cm/yr")

    both = df[["slope_storage_cm_yr","slopeA_storage_cm_yr"]].dropna()
    if len(both) > 10:
        r = np.corrcoef(both.iloc[:,0], both.iloc[:,1])[0,1]
        _p(f"  A vs B correlation (n={len(both):,}): r = {r:.3f}")
        _p(f"  median B - A = {(both.iloc[:,0]-both.iloc[:,1]).median():+.4f} cm/yr")

    _p("  Sign check: declining wells are NEGATIVE.")
    frac_neg = 100*(s < 0).sum()/max(s.notna().sum(),1)
    _p(f"  fraction of wells with NEGATIVE (declining) storage trend: {frac_neg:.1f}%")
    return df

# ============================================================ MAIN ============
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    _p(LINE); _p(f"PHASE 2.1 -- WELL TRENDS  (SAMPLE_MODE={SAMPLE_MODE})"); _p(LINE)

    diagnose_cap()

    df = compute_well_trends()
    if df.empty:
        fail("no well slopes computed; check schema."); return
    df, n_sy = attach_meta_sy(df)
    df = to_storage(df)

    hdr("STEP 4 -- write")
    out = os.path.join(OUT_DIR, "well_trends_SAMPLE.parquet" if SAMPLE_MODE
                       else "well_trends.parquet")
    df.to_parquet(out, index=False)
    ok(f"written: {out}   rows={len(df):,}")
    _p(f"  columns: {list(df.columns)}")

    with open(os.path.join(OUT_DIR, "well_trends_diagnostics.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(_LOG))

    _p("\n"+LINE)
    if SAMPLE_MODE:
        _p("SAMPLE run (2 row groups): sign convention, Sy join and A-vs-B correlation reported above.")
        _p("Set SAMPLE_MODE=False for the full set.")
    else:
        _p("Full run complete.")
    _p(LINE)

if __name__ == "__main__":
    try: main()
    except Exception: traceback.print_exc()