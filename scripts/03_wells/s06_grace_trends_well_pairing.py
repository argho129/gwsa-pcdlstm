#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
PHASE 2.2 -- GRACE CELL TRENDS + WELL-TO-CELL PAIRING
--------------------------------------------------------------------------------
[1] Theil-Sen trend per cell on monthly GWSA (robust; handles NaN gaps natively).
[2] Match each well INDIVIDUALLY to its 0.5deg cell (never pixel-mean first),
    then take a robust WITHIN-CELL MEDIAN of well storage trends.
[3] Build the paired dataset: cells having BOTH a GRACE trend and >=1 well.
[4] Report the GRACE-vs-well correlation -- the assumption quantile mapping rests
    on, and which the carbon paper does not report.
Writes -> OUTPUTS\02_well_trends\paired_cells.parquet
          OUTPUTS\02_well_trends\grace_cell_trends.nc
================================================================================
"""
import os, warnings, traceback
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

OUT_ROOT = r"D:\THESIS\Arpon_Thesis_Final\OUTPUTS"
DATA     = r"D:\THESIS\Arpon_Thesis_Final\DATA"
OUT_DIR  = os.path.join(OUT_ROOT, "02_well_trends")
GWSA_NC  = os.path.join(OUT_ROOT, "01_gwsa_monthly", "gwsa_monthly.nc")
WELLS_PQ = os.path.join(OUT_DIR, "well_trends.parquet")
CTY      = os.path.join(DATA, "Global SHapefile", "World_Countries_Generalized.shp")

MIN_MONTHS = 60      # minimum valid months for a cell trend
GRID = 0.5

LINE = "="*78
_LOG = []
def _p(s=""): print(s); _LOG.append(str(s))
def hdr(t): _p("\n"+LINE); _p(t); _p(LINE)
def ok(t): _p("  [ OK ]   "+t)
def warn(t): _p("  [WARN]   "+t)
def fail(t): _p("  [FAIL]   "+t)

# ---------------------------------------------------------------- Sen slope ---
def sen_slope_ts(y, x):
    """Theil-Sen on a time series with NaNs already removed. Returns slope/step."""
    n = y.size
    if n < 2: return np.nan
    xi = x[:, None]; xj = x[None, :]
    yi = y[:, None]; yj = y[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        s = (yj - yi) / (xj - xi)
    iu = np.triu_indices(n, k=1)
    s = s[iu]; s = s[np.isfinite(s)]
    return float(np.median(s)) if s.size else np.nan

# ============================================================ STEP 1 ==========
def grace_cell_trends():
    hdr("STEP 1 -- Theil-Sen trend per cell on monthly GWSA")
    import xarray as xr
    ds = xr.open_dataset(GWSA_NC)
    G   = ds["gwsa"].values                      # (t, lat, lon) cm
    LAT = ds["lat"].values; LON = ds["lon"].values
    dom = ds["domain"].values.astype(bool)
    ds.close()
    T = G.shape[0]
    tmon = np.arange(T, dtype="float64")         # month index

    valid = np.sum(np.isfinite(G), axis=0)
    cand  = dom & (valid >= MIN_MONTHS)
    jj, ii = np.where(cand)
    _p(f"  domain cells: {int(dom.sum()):,}   with >= {MIN_MONTHS} months: {len(jj):,}")

    trend = np.full(G.shape[1:], np.nan, "float64")
    nobs  = np.zeros(G.shape[1:], "int16")
    for k, (j, i) in enumerate(zip(jj, ii)):
        s = G[:, j, i]; m = np.isfinite(s)
        trend[j, i] = sen_slope_ts(s[m], tmon[m]) * 12.0   # cm/month -> cm/yr
        nobs[j, i]  = int(m.sum())
        if (k+1) % 10000 == 0:
            _p(f"     {k+1:,} / {len(jj):,} cells")
    ok(f"GRACE Theil-Sen trends computed for {int(np.isfinite(trend).sum()):,} cells")
    t = trend[np.isfinite(trend)]
    _p(f"  trend stats (cm/yr): min={t.min():.3f} p1={np.percentile(t,1):.3f} "
       f"p50={np.median(t):.3f} p99={np.percentile(t,99):.3f} max={t.max():.3f}")
    _p(f"  fraction of cells with NEGATIVE trend: {100*(t<0).sum()/t.size:.1f}%")

    # sanity: hotspots must stay negative under Theil-Sen too
    _p("  hotspot check (Theil-Sen):")
    for nm, la, lo, exp in [("NW India",30.5,75.5,"NEG"),("N China Plain",37.0,116.0,"NEG"),
                            ("Arabia",24.0,45.0,"NEG"),("Amazon",-3.0,-60.0,"POS/~0")]:
        j = int(np.argmin(np.abs(LAT-la))); i = int(np.argmin(np.abs(LON-lo)))
        _p(f"     {nm:<15}: {trend[j,i]:+.3f} cm/yr  (expect {exp})")
    return trend, nobs, LAT, LON, dom

# ============================================================ STEP 2 ==========
def wells_to_cells(LAT, LON, dom):
    hdr("STEP 2 -- match wells individually to 0.5deg cells")
    w = pd.read_parquet(WELLS_PQ)
    _p(f"  wells loaded: {len(w):,}")
    w = w[np.isfinite(w["latitude"]) & np.isfinite(w["longitude"])]
    w = w[np.isfinite(w["slope_storage_cm_yr"])]
    _p(f"  with valid coords + storage trend: {len(w):,}")

    # cell index from coordinates (grid starts at -90 / -180, 0.5 deg)
    jw = np.floor((w["latitude"].values  + 90.0) / GRID).astype(int)
    iw = np.floor((w["longitude"].values + 180.0) / GRID).astype(int)
    good = (jw >= 0) & (jw < len(LAT)) & (iw >= 0) & (iw < len(LON))
    w = w[good].copy(); jw = jw[good]; iw = iw[good]
    w["j"] = jw; w["i"] = iw
    _p(f"  wells inside grid: {len(w):,}")

    in_dom = dom[jw, iw]
    _p(f"  wells inside analysis domain: {int(in_dom.sum()):,} "
       f"({100*in_dom.sum()/len(w):.1f}%)")
    _p(f"  wells OUTSIDE domain (ocean-masked / ice / beyond GRACE): "
       f"{int((~in_dom).sum()):,}")
    w = w[in_dom].copy()

    # robust within-cell aggregation -- MEDIAN, never mean
    agg = (w.groupby(["j","i"])
             .agg(well_median_cm_yr=("slope_storage_cm_yr","median"),
                  well_mean_cm_yr  =("slope_storage_cm_yr","mean"),
                  well_std_cm_yr   =("slope_storage_cm_yr","std"),
                  n_wells          =("slope_storage_cm_yr","size"),
                  n_sy_imputed     =("sy_imputed","sum"),
                  sy_median        =("specific_y","median"))
             .reset_index())
    ok(f"cells containing >=1 well: {len(agg):,}")
    _p(f"  well-count per cell: p50={agg['n_wells'].median():.0f} "
       f"p90={agg['n_wells'].quantile(.9):.0f} max={agg['n_wells'].max():.0f}")
    for thr in (1,2,3,5,10,25,50):
        _p(f"     cells with >= {thr:>3} wells: {int((agg['n_wells']>=thr).sum()):,}")
    return w, agg

# ============================================================ STEP 3 ==========
def pair_and_correlate(trend, nobs, agg, LAT, LON):
    hdr("STEP 3 -- paired dataset + GRACE-vs-well correlation")
    agg["grace_cm_yr"] = trend[agg["j"].values, agg["i"].values]
    agg["grace_nmonths"] = nobs[agg["j"].values, agg["i"].values]
    agg["lat"] = LAT[agg["j"].values]; agg["lon"] = LON[agg["i"].values]
    pair = agg[np.isfinite(agg["grace_cm_yr"])].copy()
    ok(f"PAIRED cells (GRACE trend AND >=1 well): {len(pair):,}")
    _p(f"  as % of domain cells with a GRACE trend: "
       f"{100*len(pair)/max(int(np.isfinite(trend).sum()),1):.2f}%")

    _p("\n  --- correlation of GRACE vs well trends (the QM assumption) ---")
    for thr in (1,3,5,10,25):
        sub = pair[pair["n_wells"] >= thr]
        if len(sub) < 30:
            _p(f"     >= {thr:>3} wells/cell: n={len(sub):,}  (too few)"); continue
        x = sub["grace_cm_yr"].values; y = sub["well_median_cm_yr"].values
        r  = np.corrcoef(x, y)[0,1]
        rs = pd.Series(x).corr(pd.Series(y), method="spearman")
        sign_agree = 100*np.mean(np.sign(x) == np.sign(y))
        _p(f"     >= {thr:>3} wells/cell: n={len(sub):>6,}  Pearson r={r:+.3f}  "
           f"Spearman={rs:+.3f}  sign agreement={sign_agree:.1f}%")
    _p("  r rising with wells/cell = point-vs-footprint noise, as expected.")
    _p("  If r stays ~0 at all thresholds, quantile mapping is NOT well founded.")

    _p("\n  --- magnitude comparison ---")
    for nm, col in (("GRACE","grace_cm_yr"), ("wells","well_median_cm_yr")):
        v = pair[col].values
        _p(f"     {nm:<6}: p1={np.percentile(v,1):+.3f} p50={np.median(v):+.3f} "
           f"p99={np.percentile(v,99):+.3f}  sd={v.std():.3f}")
    _p("  wells typically show LARGER spread than GRACE (mascon smoothing).")

    _p("\n  --- geographic spread of paired cells ---")
    for nm, la0, la1, lo0, lo1 in [
        ("N America",  15, 72, -170, -50), ("S America", -56, 15, -82, -34),
        ("Europe",     35, 72,  -12,  40), ("Africa",   -35, 37,  -18,  52),
        ("S Asia",      5, 40,   60, 100), ("E Asia",    18, 55,  100, 146),
        ("Australia", -45,-10,  112, 155), ("Rest",     -90, 90, -180, 180)]:
        m = ((pair["lat"]>=la0)&(pair["lat"]<=la1)&
             (pair["lon"]>=lo0)&(pair["lon"]<=lo1))
        _p(f"     {nm:<10}: {int(m.sum()):>6,} paired cells  "
           f"({100*m.sum()/len(pair):5.1f}%)")
    return pair

# ============================================================ FIGURES =========
def figures(trend, agg, pair, LAT, LON, dom):
    hdr("STEP 4 -- figures (full global extent)")
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import rcParams
        from matplotlib.colors import TwoSlopeNorm, LogNorm
        import geopandas as gpd
    except Exception as e:
        warn(f"plot deps: {e}"); return
    rcParams["font.family"]="serif"
    rcParams["font.serif"]=["Times New Roman","DejaVu Serif"]
    rcParams["mathtext.fontset"]="stix"
    rcParams.update({"font.size":16,"axes.titlesize":21,"axes.labelsize":18,
                     "xtick.labelsize":15,"ytick.labelsize":15,"axes.linewidth":1.1})
    cty = gpd.read_file(CTY)
    if cty.crs is None: cty=cty.set_crs("EPSG:4326")
    elif cty.crs.to_epsg()!=4326: cty=cty.to_crs("EPSG:4326")
    ext=[-180,180,-90,90]
    def base(ax):
        ax.set_facecolor("#eef3f7")
        cty.plot(ax=ax, facecolor="#f7f7f5", edgecolor="none", zorder=1)
        cty.boundary.plot(ax=ax, linewidth=0.38, color="0.22", zorder=5)
        ax.set_xlim(-180,180); ax.set_ylim(-58,84)
        ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")
        ax.set_xticks(range(-180,181,60)); ax.set_yticks(range(-40,81,20))

    # A: well count per cell (full extent -- sparsity IS the message)
    cnt = np.full(trend.shape, np.nan)
    cnt[agg["j"].values, agg["i"].values] = agg["n_wells"].values
    f,ax=plt.subplots(figsize=(17,8.2))
    im=ax.imshow(cnt,origin="lower",extent=ext,cmap="viridis",
                 norm=LogNorm(vmin=1,vmax=max(np.nanmax(cnt),10)),zorder=2)
    base(ax); ax.set_title("Monitoring-Well Density per 0.5° Cell, 2002–2022",
                           pad=14,fontweight="bold")
    cb=f.colorbar(im,ax=ax,shrink=0.80,pad=0.015,extend="max")
    cb.set_label("Wells per cell (log scale)",labelpad=10)
    f.savefig(os.path.join(OUT_DIR,"fig5_well_density.png"),dpi=600,
              bbox_inches="tight",facecolor="white"); plt.close(f); ok("fig5")

    # B: GRACE Theil-Sen trend (full extent)
    f,ax=plt.subplots(figsize=(17,8.2))
    v=float(np.nanpercentile(np.abs(trend),92))
    im=ax.imshow(trend,origin="lower",extent=ext,cmap="RdYlBu",
                 norm=TwoSlopeNorm(vmin=-v,vcenter=0,vmax=v),zorder=2)
    base(ax); ax.set_title("GRACE-Derived GWSA Trend (Theil–Sen), 2002–2022",
                           pad=14,fontweight="bold")
    cb=f.colorbar(im,ax=ax,shrink=0.80,pad=0.015,extend="both")
    cb.set_label("GWSA trend (cm yr$^{-1}$)",labelpad=10)
    f.savefig(os.path.join(OUT_DIR,"fig6_grace_theilsen_trend.png"),dpi=600,
              bbox_inches="tight",facecolor="white"); plt.close(f); ok("fig6")

    # C: GRACE vs wells scatter
    f,ax=plt.subplots(figsize=(9,8.5))
    sub=pair[pair["n_wells"]>=3]
    sc=ax.scatter(sub["grace_cm_yr"],sub["well_median_cm_yr"],
                  c=sub["n_wells"],cmap="viridis",norm=LogNorm(),
                  s=18,alpha=0.55,edgecolors="none")
    lim=np.nanpercentile(np.abs(np.r_[sub["grace_cm_yr"],sub["well_median_cm_yr"]]),99)
    ax.plot([-lim,lim],[-lim,lim],color="0.25",lw=1.2,ls="--",label="1:1")
    ax.axhline(0,color="0.7",lw=0.8); ax.axvline(0,color="0.7",lw=0.8)
    ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim)
    ax.set_xlabel("GRACE-derived trend (cm yr$^{-1}$)")
    ax.set_ylabel("Well-derived storage trend (cm yr$^{-1}$)")
    ax.set_title("Satellite vs In-Situ Storage Trends",pad=14,fontweight="bold")
    ax.legend(frameon=False,loc="upper left")
    cb=f.colorbar(sc,ax=ax,shrink=0.85,pad=0.02); cb.set_label("Wells per cell")
    f.savefig(os.path.join(OUT_DIR,"fig7_grace_vs_wells.png"),dpi=600,
              bbox_inches="tight",facecolor="white"); plt.close(f); ok("fig7")

# ============================================================ MAIN ============
def main():
    os.makedirs(OUT_DIR,exist_ok=True)
    _p(LINE); _p("PHASE 2.2 -- GRACE CELL TRENDS + WELL PAIRING"); _p(LINE)
    trend,nobs,LAT,LON,dom = grace_cell_trends()
    w,agg = wells_to_cells(LAT,LON,dom)
    pair  = pair_and_correlate(trend,nobs,agg,LAT,LON)

    hdr("STEP 5 -- write")
    pair.to_parquet(os.path.join(OUT_DIR,"paired_cells.parquet"),index=False)
    ok(f"paired_cells.parquet  rows={len(pair):,}")
    import xarray as xr
    xr.Dataset({"grace_trend_cm_yr":(("lat","lon"),trend.astype("float32")),
                "n_months":(("lat","lon"),nobs)},
               coords={"lat":LAT,"lon":LON},
               attrs={"method":"Theil-Sen on monthly GWSA, >=60 months",
                      "units":"cm yr-1"}
              ).to_netcdf(os.path.join(OUT_DIR,"grace_cell_trends.nc"))
    ok("grace_cell_trends.nc")

    figures(trend,agg,pair,LAT,LON,dom)
    with open(os.path.join(OUT_DIR,"pairing_diagnostics.txt"),"w",encoding="utf-8") as fh:
        fh.write("\n".join(_LOG))
    _p("\n"+LINE)
    _p("KEY NUMBERS: paired cell count, and GRACE-vs-well correlation by density.")
    _p("These determine whether quantile mapping is well founded.")
    _p(LINE)

if __name__=="__main__":
    try: main()
    except Exception: traceback.print_exc()