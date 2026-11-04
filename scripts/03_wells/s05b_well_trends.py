#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
PHASE 2.1 v2 -- WELL TREND COMPUTATION (period-matched 2002-2022)
--------------------------------------------------------------------------------
Changes vs v1 (all traceable; v1 outputs are NOT overwritten):
  [1] MIN_OBS lowered 8 -> 3. Record length is RETAINED per well (n_obs) so any
      threshold can be applied downstream without re-reading 35.7M rows.
  [2] Per-well trend UNCERTAINTY carried: Theil-Sen 95% CI via the Kendall-based
      rank method (Sen 1968; Hollander & Wolfe), plus MAD of pairwise slopes.
      -> lets the threshold be set on trend RELIABILITY, not record length alone.
  [3] Contamination / applicability FLAGS carried forward for strata reporting:
      Sy imputed, aquifer type, confinement (where GROW reports it), reference
      point convention, and the well's own QC flags.
  [4] Screens unchanged and stated: |slope| <= 5 m/yr (physical), storage screen
      applied downstream so it can be varied without a re-run.
Sign convention: NEGATIVE = declining water table = storage loss.
    depth-from-ground : depth INCREASES as table falls -> slope_wl = -slope_depth
    water-level-elev  : elevation DECREASES as table falls -> slope_wl = +slope
Storage trend (cm/yr) = water-level trend (m/yr) x Sy x 100.
Writes -> OUTPUTS\02_well_trends\well_trends_v2.parquet  (+ diagnostics)
================================================================================
"""
import os, sys, warnings, traceback
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

SAMPLE_MODE = False        # True = 2 row groups (fast verify)

DATA     = r"D:\THESIS\Arpon_Thesis_Final\DATA"
OUT_ROOT = r"D:\THESIS\Arpon_Thesis_Final\OUTPUTS"
OUT_DIR  = os.path.join(OUT_ROOT, "02_well_trends")
FIG_DIR  = os.path.join(OUT_DIR, "figures")
GROW_DIR = os.path.join(DATA, "GROW")
TS_PQ    = os.path.join(GROW_DIR, "grow_timeseries.parquet")
AT_PQ    = os.path.join(GROW_DIR, "grow_attributes.parquet")
SY_SHP   = os.path.join(DATA, "specific yeild-上传", "geishuidu-min.shp")

YR0, YR1    = 2002, 2022
MIN_OBS     = 3            # lowered; thresholds applied downstream
MAX_ABS_MYR = 5.0          # physical screen on water-level trend

DEPTH_COL = "groundwater_filled_depth_from_ground_elevation_m"
DEPTH_RAW = "groundwater_depth_from_ground_elevation_m"
ELEV_COL  = "groundwater_filled_water_level_elevation_m_asl"
ELEV_RAW  = "groundwater_water_level_elevation_m_asl"

LINE = "="*78
_LOG = []
def _p(s=""): print(s); _LOG.append(str(s))
def hdr(t): _p("\n"+LINE); _p(t); _p(LINE)
def ok(t): _p("  [ OK ]   "+t)
def warn(t): _p("  [WARN]   "+t)
def fail(t): _p("  [FAIL]   "+t)

# ------------------------------------------------------------- Theil-Sen ------
def sen_with_ci(x, y, alpha=0.05):
    """
    Theil-Sen slope + distribution-free CI (Sen 1968).
    Returns (slope, lo, hi, mad_of_pairwise_slopes, n_pairs).
    CI uses the normal approximation to Kendall's S variance; exact for n>=10,
    conservative below. Reported so the threshold can be set on reliability.
    """
    n = x.size
    if n < 3: return (np.nan,)*4 + (0,)
    xi = x[:,None]; xj = x[None,:]
    yi = y[:,None]; yj = y[None,:]
    with np.errstate(invalid="ignore", divide="ignore"):
        s = (yj - yi) / (xj - xi)
    s = s[np.triu_indices(n,1)]
    s = np.sort(s[np.isfinite(s)])
    N = s.size
    if N == 0: return (np.nan,)*4 + (0,)
    slope = float(np.median(s))
    mad   = float(np.median(np.abs(s - slope)))
    # variance of Kendall's S (no tie correction; conservative)
    varS = n*(n-1)*(2*n+5)/18.0
    from math import sqrt
    try:
        from scipy.stats import norm
        z = norm.ppf(1-alpha/2)
    except Exception:
        z = 1.959963985
    C = z*sqrt(varS)
    lo_i = int(max(0, np.floor((N - C)/2) - 1))
    hi_i = int(min(N-1, np.ceil((N + C)/2)))
    return slope, float(s[lo_i]), float(s[hi_i]), mad, int(N)

# ============================================================ STEP 1 ==========
def compute_well_trends():
    hdr("STEP 1 -- Theil-Sen slope + CI per well, 2002-2022 (chunked)")
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(TS_PQ)
    ngr = pf.metadata.num_row_groups
    use = 2 if SAMPLE_MODE else ngr
    _p(f"  row groups: {ngr}  processing: {use}  (SAMPLE_MODE={SAMPLE_MODE})")
    _p(f"  MIN_OBS = {MIN_OBS} annual values inside {YR0}-{YR1}")

    want = ["GROW_ID","year",DEPTH_COL,DEPTH_RAW,ELEV_COL,ELEV_RAW]
    cols = [c for c in want if c in set(pf.schema.names)]
    _p(f"  columns read: {cols}")

    carry = pd.DataFrame(); recs = []
    for g in range(use):
        tb = pf.read_row_group(g, columns=cols).to_pandas()
        tb = tb[(tb["year"] >= YR0) & (tb["year"] <= YR1)]
        if tb.empty: continue
        if len(carry): tb = pd.concat([carry, tb], ignore_index=True)
        if g < use-1:
            last = tb["GROW_ID"].iloc[-1]
            carry = tb[tb["GROW_ID"] == last]
            tb    = tb[tb["GROW_ID"] != last]
        else:
            carry = pd.DataFrame()
        if tb.empty: continue
        for wid, grp in tb.groupby("GROW_ID", sort=False):
            r = _well_record(wid, grp)
            if r: recs.append(r)
        if (g+1) % 5 == 0 or g == use-1:
            _p(f"     row group {g+1}/{use}: wells so far = {len(recs):,}")
    df = pd.DataFrame(recs)
    ok(f"wells with a {YR0}-{YR1} Theil-Sen slope: {len(df):,}")
    if len(df):
        _p(f"  n_obs distribution: p10={df['n_obs'].quantile(.1):.0f} "
           f"p50={df['n_obs'].median():.0f} p90={df['n_obs'].quantile(.9):.0f} "
           f"max={df['n_obs'].max():.0f}")
        for t in (3,5,8,10,15,20):
            _p(f"     wells with n_obs >= {t:>2}: {int((df['n_obs']>=t).sum()):,}")
    return df

def _well_record(wid, grp):
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
    m = np.isfinite(v) & np.isfinite(yr)
    if m.sum() < MIN_OBS: return None
    ser = pd.Series(v[m]).groupby(yr[m]).mean()
    xx = ser.index.values.astype("float64"); yy = ser.values
    if xx.size < MIN_OBS: return None
    s, lo, hi, mad, npair = sen_with_ci(xx, yy)
    if not np.isfinite(s): return None
    flip = -1.0 if kind == "depth" else 1.0
    return {"GROW_ID": wid, "n_obs": int(xx.size),
            "yr_first": float(xx.min()), "yr_last": float(xx.max()),
            "yr_span": float(xx.max()-xx.min()+1),
            "src_col": col, "src_kind": kind,
            "slope_wl_m_yr": float(flip*s),
            "slope_ci_lo":   float(flip*hi if flip < 0 else lo),
            "slope_ci_hi":   float(flip*lo if flip < 0 else hi),
            "slope_mad":     float(mad),
            "n_pairs":       npair}

# ============================================================ STEP 2 ==========
def attach_meta_sy(df):
    hdr("STEP 2 -- attributes, QC strata, specific yield")
    n_raw = len(df)
    df = df.drop_duplicates(subset="GROW_ID", keep="first")
    if len(df) < n_raw:
        warn(f"deduplicated GROW_ID: {n_raw:,} -> {len(df):,}")

    at = pd.read_parquet(AT_PQ, columns=[
        "GROW_ID", "latitude", "longitude", "country", "interval", "length_years",
        "gap_fraction", "plateaus", "outliers_change_points", "negative_signs_wtd",
        "trend_slope_m_year-1", "reference_point", "aquifer_type_class",
        "confinement", "total_porosity_0-100_m_fraction", "koeppen_geiger_class",
        "main_landuse_class"])
    df = df.merge(at, on="GROW_ID", how="left")
    _p(f"  merged attributes: {len(df):,}")

    # QC flags kept as COLUMNS (strata), not applied as hard filters here
    df["qc_len10"]    = df["length_years"] >= 10
    df["qc_gap10"]    = df["gap_fraction"] <= 0.10
    df["qc_interval"] = df["interval"].astype(str).isin(["YS","MS"])
    df["qc_plateau"]  = ~df["plateaus"].astype(str).str.lower().isin(["true","1","1.0"])
    df["qc_outlier"]  = ~df["outliers_change_points"].astype(str).str.lower().isin(["true","1","1.0"])
    df["qc_negsign"]  = ~df["negative_signs_wtd"].astype(str).isin(["All","Some"])
    df["qc_all"] = (df["qc_len10"] & df["qc_gap10"] & df["qc_interval"] &
                    df["qc_plateau"] & df["qc_outlier"] & df["qc_negsign"])
    _p(f"  QC strata (kept as flags, not filters):")
    for c in ["qc_len10","qc_gap10","qc_interval","qc_plateau","qc_outlier",
              "qc_negsign","qc_all"]:
        _p(f"     {c:<14}: {int(df[c].sum()):>7,} pass  "
           f"({100*df[c].mean():5.1f}%)")

    # physical screen on water-level trend (stated, applied)
    bad = df["slope_wl_m_yr"].abs() > MAX_ABS_MYR
    _p(f"  physical screen |slope| > {MAX_ABS_MYR} m/yr: {int(bad.sum()):,} removed")
    df = df[~bad].copy()

    # confinement stratum -- Sy applies to UNCONFINED aquifers only
    conf = df["confinement"].astype(str).str.lower()
    df["is_confined"]   = conf.str.contains("confin") & ~conf.str.contains("uncon")
    df["conf_known"]    = ~conf.isin(["nan","none",""])
    _p(f"  confinement reported for {int(df['conf_known'].sum()):,} wells "
       f"({100*df['conf_known'].mean():.1f}%); flagged confined: "
       f"{int(df['is_confined'].sum()):,}")
    _p("  Sy is a DRAINAGE property of unconfined aquifers; confined wells")
    _p("      respond by elastic storativity (orders of magnitude smaller), so")
    _p("      applying Sy overestimates their storage change. Reported as stratum.")

    # specific yield join
    import geopandas as gpd
    from shapely.geometry import Point
    sy = gpd.read_file(SY_SHP)[["specific_y","geometry"]]
    if sy.crs is None: sy = sy.set_crs("EPSG:4326")
    elif sy.crs.to_epsg() != 4326: sy = sy.to_crs("EPSG:4326")
    g = gpd.GeoDataFrame(df, geometry=[Point(a,b) for a,b in
                         zip(df["longitude"], df["latitude"])], crs="EPSG:4326")
    j = gpd.sjoin(g, sy, how="left", predicate="within").drop(columns=["index_right"])
    j = j[~j.index.duplicated(keep="first")]
    n_sy = int(j["specific_y"].notna().sum())
    j["sy_imputed"] = j["specific_y"].isna()
    med_sy = float(sy["specific_y"].median())
    j["specific_y"] = j["specific_y"].fillna(med_sy)
    ok(f"Sy from shapefile: {n_sy:,} / {len(j):,}")
    warn(f"Sy median-filled ({med_sy:.3f}) for {int(j['sy_imputed'].sum()):,} wells "
         f"-- flagged in 'sy_imputed'")
    return pd.DataFrame(j.drop(columns="geometry"))

# ============================================================ STEP 3 ==========
def to_storage(df):
    hdr("STEP 3 -- storage trend + trend-reliability metric")
    df["slope_storage_cm_yr"] = df["slope_wl_m_yr"] * df["specific_y"] * 100.0
    df["storage_ci_lo"] = df["slope_ci_lo"] * df["specific_y"] * 100.0
    df["storage_ci_hi"] = df["slope_ci_hi"] * df["specific_y"] * 100.0
    df["ci_width_cm_yr"] = (df["storage_ci_hi"] - df["storage_ci_lo"]).abs()
    # reliability: |slope| relative to its own CI half-width. >1 = resolved.
    half = df["ci_width_cm_yr"]/2.0
    df["trend_snr"] = df["slope_storage_cm_yr"].abs() / half.replace(0, np.nan)
    df["trend_resolved"] = df["trend_snr"] >= 1.0

    a = df["trend_slope_m_year-1"].astype("float64")
    df["slopeA_storage_cm_yr"] = a.where(a.abs() <= MAX_ABS_MYR)*df["specific_y"]*100.0

    s = df["slope_storage_cm_yr"]
    _p(f"  B (period-matched)  n={s.notna().sum():,}")
    _p(f"     p1={s.quantile(.01):+.3f} p50={s.median():+.3f} "
       f"p99={s.quantile(.99):+.3f} cm/yr")
    _p(f"  CI width (cm/yr): p25={df['ci_width_cm_yr'].quantile(.25):.3f} "
       f"p50={df['ci_width_cm_yr'].median():.3f} "
       f"p75={df['ci_width_cm_yr'].quantile(.75):.3f}")
    _p(f"  trends RESOLVED (|slope| >= CI half-width): "
       f"{int(df['trend_resolved'].sum()):,} ({100*df['trend_resolved'].mean():.1f}%)")
    _p("  'resolved' is a reliability criterion independent of record length;")
    _p("      it is the defensible basis for a threshold (not maximising r).")
    _p("\n  resolved fraction by record length:")
    for t in (3,5,8,10,15,20):
        sub = df[df["n_obs"] >= t]
        if len(sub) < 50: continue
        _p(f"     n_obs >= {t:>2}: n={len(sub):>7,}  resolved="
           f"{100*sub['trend_resolved'].mean():5.1f}%  "
           f"median CI width={sub['ci_width_cm_yr'].median():.3f}")
    frac_neg = 100*(s < 0).mean()
    _p(f"\n  wells with NEGATIVE (declining) storage trend: {frac_neg:.1f}%")
    return df

# ============================================================ FIGURES =========
def figures(df):
    hdr("STEP 4 -- figures")
    os.makedirs(FIG_DIR, exist_ok=True)
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    rcParams["font.family"]="serif"
    rcParams["font.serif"]=["Times New Roman","DejaVu Serif"]
    rcParams["mathtext.fontset"]="stix"
    rcParams.update({"font.size":20,"axes.titlesize":24,"axes.labelsize":22,
                     "xtick.labelsize":18,"ytick.labelsize":18,
                     "legend.fontsize":17,"axes.linewidth":1.4,
                     "xtick.major.width":1.3,"ytick.major.width":1.3})
    def save(f,nm):
        for e in (".png",".pdf"):
            f.savefig(os.path.join(FIG_DIR,nm+e),dpi=600,bbox_inches="tight",
                      facecolor="white")
        plt.close(f); ok(nm)

    # A: record length vs trend reliability (the threshold argument)
    f,axes=plt.subplots(1,2,figsize=(18,7.5))
    ts=[3,4,5,6,8,10,12,15,20]
    res=[df[df["n_obs"]>=t]["trend_resolved"].mean()*100 for t in ts]
    nn =[int((df["n_obs"]>=t).sum()) for t in ts]
    axes[0].plot(ts,res,marker="o",ms=13,lw=3.2,color="#1f4e79")
    for t,r_,n_ in zip(ts,res,nn):
        axes[0].annotate(f"{n_:,}",(t,r_),textcoords="offset points",
                         xytext=(0,-28),ha="center",fontsize=14,color="0.35")
    axes[0].set_xlabel("Minimum record length (annual values)")
    axes[0].set_ylabel("Trends resolved (%)")
    axes[0].set_title("(a) Trend reliability vs record length",fontweight="bold")
    axes[0].grid(alpha=.25)
    cw=[df[df["n_obs"]>=t]["ci_width_cm_yr"].median() for t in ts]
    axes[1].plot(ts,cw,marker="s",ms=13,lw=3.2,color="#c0392b")
    axes[1].set_xlabel("Minimum record length (annual values)")
    axes[1].set_ylabel("Median CI width (cm yr$^{-1}$)")
    axes[1].set_title("(b) Trend uncertainty vs record length",fontweight="bold")
    axes[1].grid(alpha=.25)
    f.tight_layout(); save(f,"fig18_v2_length_vs_reliability")

    # B: storage trend distribution, resolved vs unresolved
    f,ax=plt.subplots(figsize=(13,8.5))
    b=np.linspace(-15,10,80)
    ax.hist(df.loc[df["trend_resolved"],"slope_storage_cm_yr"],bins=b,
            color="#1f4e79",alpha=.85,label="resolved",edgecolor="white",lw=.4)
    ax.hist(df.loc[~df["trend_resolved"],"slope_storage_cm_yr"],bins=b,
            color="#e59866",alpha=.75,label="unresolved",edgecolor="white",lw=.4)
    ax.axvline(0,color="0.3",lw=1.6)
    ax.set_xlabel("Well storage trend (cm yr$^{-1}$)"); ax.set_ylabel("Wells")
    ax.set_title("Storage Trends by Reliability",fontweight="bold")
    ax.legend(frameon=False); ax.grid(alpha=.25)
    save(f,"fig19_v2_resolved_distribution")

# ============================================================ MAIN ============
def main():
    os.makedirs(OUT_DIR,exist_ok=True)
    _p(LINE); _p(f"PHASE 2.1 v2 -- WELL TRENDS (SAMPLE_MODE={SAMPLE_MODE})"); _p(LINE)
    df = compute_well_trends()
    if df.empty: fail("no slopes computed."); return
    df = attach_meta_sy(df)
    df = to_storage(df)
    figures(df)
    out = os.path.join(OUT_DIR, "well_trends_v2_SAMPLE.parquet" if SAMPLE_MODE
                       else "well_trends_v2.parquet")
    df.to_parquet(out, index=False)
    ok(f"written: {out}  rows={len(df):,}")
    _p(f"  columns: {list(df.columns)}")
    with open(os.path.join(OUT_DIR,"well_trends_v2_diagnostics.txt"),"w",
              encoding="utf-8") as fh:
        fh.write("\n".join(_LOG))
    _p("\n"+LINE)
    _p("Run complete. The output carries n_obs, CI, SNR and QC strata so")
    _p("    downstream steps can vary every threshold without another 35.7M-row pass.")
    _p(LINE)

if __name__=="__main__":
    try: main()
    except Exception: traceback.print_exc()