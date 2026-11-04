#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
PHASE 6 -- FINAL VALIDATION & BENCHMARK (publication build, v4)
  Three GWSA products, one well pipeline, apple-to-apple:
    ML-GWSA  gwsa_final.nc (calibrated, cm)  <- Recon. GWSA (This Study)
    WGAP     WaterGAP groundwstor (kg m-2 = mm -> /10 cm), baselined 2004-2009
    SEDA     GRACE-SeDA TWSA (mm -> /10 cm) minus identical OSA -> SeDA-implied GWSA
  Reference: GROW wells (Sy-scaled storage trends), independent, point-scale.

OUTPUT  ->  OUTPUTS\06_validation_final\
  benchmark_metrics_FINAL.csv        (keepable metrics + interproduct metrics)
  benchmark_summary_FINAL.txt
  figures\  fig_benchmark_bars_density, fig_sign_agreement_bars,
            fig_interproduct_comparison, fig_aquifer_scatters,
            fig_trend_map_*, fig_aquifer_trend_map_*, fig_basin_trend_map_*,
            fig_wells_density_map, fig_residual_map_*,
            fig_taylor_diagram, fig_metrics_heatmap, fig_zonal_profile,
            fig_density_vs_agreement
================================================================================
"""
import os, sys, time, warnings, traceback, datetime as dt
import numpy as np
import pandas as pd    # module-level (not just inside main()) so sign_discrepancy_diagnostic(),
                        # a module-level function, can see it too
warnings.filterwarnings("ignore")

OUT   = r"D:\THESIS\Arpon_Thesis_Final\OUTPUTS"
MODEL = os.path.join(OUT, "05_model_tensor")
VALID = os.path.join(OUT, "06_validation_final")
FIGDIR= os.path.join(VALID, "figures")
FINAL_NC  = os.path.join(MODEL, "gwsa_final.nc")
TENSOR_NC = os.path.join(MODEL, "model_ready.nc")
WELL_ATTR = os.path.join(OUT, "02_well_trends", "well_trends_v2_tiered.parquet")
GROW_INV  = os.path.join(OUT, "06_validation", "grow_inventory.parquet")
AQUIFERS  = r"D:\THESIS\Arpon_Thesis_Final\DATA\Aquifers\jasechko_et_al_2024_aquifers.shp"
HYBAS_GPKG= r"D:\THESIS\Arpon_Thesis_Final\DATA\HydroBasins\hybas_lev03_global.gpkg"
SEDA_NC   = r"D:\THESIS\Arpon_Thesis_Final\DATA\Water_Paper Validation data (TWSA)\GRACE-SeDA_v0_2002_2019.nc"
WGAP_NC   = r"D:\THESIS\Arpon_Thesis_Final\DATA\WaterGAP_Data\watergap_22d_WFDEI-GPCC_histsoc_groundwstor_monthly_1901_2016.nc4"
FIG_STYLE_DIR = r"D:\THESIS\Arpon_Thesis_Final\Codes\Into the Models"

SEED=20260730
YS_MIN_YR=8; MIN_WELLS_POLY=3; MIN_POLY_FOR_R=5; BOOT=1000
BASE_Y0,BASE_Y1=2004,2009
AREA_LARGE=200000.0
NLAT,NLON=360,720
DEFAULT_SY=0.10           # same fallback specific yield used in Phase 6.1 -- only matters for
                           # diagnostic/QC reporting here, since slope_storage_cm_yr already has
                           # Sy baked in upstream (Stage 02); this is a transparency check, not a recompute.
SIGN_DISCREPANCY_R_MAX=-0.75   # flag well-vs-product points at or below this for the sign-flip QC check


def clean_id(series):
    """Robust ID normalizer -- strips a trailing '.0' introduced when an ID column
    round-trips through a float dtype (e.g. parquet/Excel), then strips whitespace.
    Using plain .astype(str).str.strip() (as this script did before) leaves IDs like
    '10234.0' instead of '10234', which silently fails to join against IDs cleaned the
    other way -- this was quietly shrinking the matched well set. Mirrors Phase 6.1."""
    return series.astype(str).str.split(".").str[0].str.strip()

# Granular density tiers for benchmarking reliability
DENSITY_BINS = [3, 5, 10, 25, 50, np.inf]
DENSITY_LABELS = ["3-5", "5-10", "10-25", "25-50", ">50"]

PRODUCT_DISPLAY = {
    "MINE": "Recon. GWSA",
    "WGAP": "WaterGAP",
    "SEDA": "SeDA",
}
PRODUCT_COLOR = {
    "MINE": "#b2182b",      # red   -- Recon. GWSA
    "WGAP": "#2166ac",      # blue  -- WaterGAP
    "SEDA": "#1a9850",      # green -- GRACE-SeDA
}

LINE="="*78; _LOG=[]
def _p(s=""): print(s); _LOG.append(str(s))
def hdr(t): _p(""); _p(LINE); _p(t); _p(LINE)
def ok(t): _p("  [ OK ]   "+t)
def warn(t): _p("  [WARN]   "+t)

def rmse(s,o):
    m=np.isfinite(s)&np.isfinite(o); return float(np.sqrt(np.mean((s[m]-o[m])**2))) if m.sum() else np.nan
def bias(s,o):
    m=np.isfinite(s)&np.isfinite(o); return float(np.mean(s[m]-o[m])) if m.sum() else np.nan
def nse(s,o):
    m=np.isfinite(s)&np.isfinite(o)
    if m.sum()<5: return np.nan
    oo,ss=o[m],s[m]; d=np.sum((oo-oo.mean())**2); return float(1-np.sum((ss-oo)**2)/d) if d>0 else np.nan
def pear(x,y):
    m=np.isfinite(x)&np.isfinite(y)
    if m.sum()<MIN_POLY_FOR_R: return np.nan,int(m.sum())
    return float(np.corrcoef(x[m],y[m])[0,1]),int(m.sum())
def signm(x,y):
    m=np.isfinite(x)&np.isfinite(y)&(x!=0)&(y!=0)
    return (float((np.sign(x[m])==np.sign(y[m])).mean()),int(m.sum())) if m.sum() else (np.nan,0)
def boot_r(x,y,n=BOOT,seed=SEED):
    m=np.isfinite(x)&np.isfinite(y); x,y=x[m],y[m]
    if x.size<MIN_POLY_FOR_R: return (np.nan,np.nan,np.nan)
    rng=np.random.default_rng(seed); out=[]
    for _ in range(n):
        i=rng.integers(0,x.size,x.size)
        if np.std(x[i])>0 and np.std(y[i])>0: out.append(np.corrcoef(x[i],y[i])[0,1])
    out=np.array(out); return (float(np.nanmedian(out)),float(np.nanpercentile(out,2.5)),float(np.nanpercentile(out,97.5)))
def theil(y,x):
    m=np.isfinite(y)&np.isfinite(x)
    if m.sum()<24: return np.nan
    yy,xx=y[m],x[m]; n=yy.size
    if n>150:
        idx=np.random.default_rng(0).choice(n,150,replace=False); yy,xx=yy[idx],xx[idx]; n=150
    sl=[]
    for i in range(n):
        dv=xx[i+1:]-xx[i]; g=dv!=0; sl.append((yy[i+1:][g]-yy[i])/dv[g])
    s=np.concatenate(sl) if sl else np.array([np.nan]); return float(np.median(s))


def sign_discrepancy_diagnostic(df_poly, id_col, well_col, prod_cols, name_map=None,
                                 out_csv=None, label=""):
    """
    QC diagnostic only -- NEVER mutates the main results, just reports.

    This script correlates ONE well-trend vs ONE product-trend per polygon, pooled
    ACROSS MANY aquifers/basins (a spatial r), unlike Phase 6.1's per-polygon TEMPORAL
    PCC. So there's no single-polygon 'r' here to threshold at -0.75. What actually
    drags a spatial r negative (or flips its sign) is individual polygons where the
    well and product trends disagree in direction. For each such polygon this reports:
      - r_leave_this_one_out : pooled r with this one polygon dropped (is it an outlier
        single-handedly pulling r toward negative?)
      - r_if_well_sign_flipped : pooled r if just this polygon's WELL trend is negated.
        If that flip both lands the point back near the 1:1 line AND raises the pooled r,
        it is a concrete lead to go check that well/region for a genuine sign-convention
        bug (e.g. depth-below-surface vs water-level-elevation, or a stray unit sign flip)
        rather than assuming the disagreement is real hydrology.
    Nothing here changes benchmark_metrics_FINAL.csv -- it is a separate audit trail.
    """
    if id_col not in df_poly.columns or well_col not in df_poly.columns:
        return pd.DataFrame()
    w_all = df_poly[well_col].values.astype(float)
    rows = []
    for pcol in prod_cols:
        if pcol not in df_poly.columns: continue
        p_all = df_poly[pcol].values.astype(float)
        m_all = np.isfinite(w_all) & np.isfinite(p_all)
        if m_all.sum() < MIN_POLY_FOR_R: continue
        r_base, _ = pear(w_all, p_all)
        disagree = m_all & (w_all != 0) & (p_all != 0) & (np.sign(w_all) != np.sign(p_all))
        for i in np.where(disagree)[0]:
            wv, pv = w_all[i], p_all[i]
            keep = m_all.copy(); keep[i] = False
            r_loo, _ = pear(w_all[keep], p_all[keep])
            w_flip = w_all.copy(); w_flip[i] = -wv
            r_flip, _ = pear(w_flip[m_all], p_all[m_all])
            pid = df_poly[id_col].iloc[i]
            name = name_map.get(pid, pid) if name_map else pid
            rows.append({
                "poly_id": pid, "poly_name": name, "product": pcol,
                "well_trend": round(float(wv), 4), "product_trend": round(float(pv), 4),
                "r_all_polygons_pooled": round(r_base, 3) if np.isfinite(r_base) else np.nan,
                "r_leave_this_one_out": round(r_loo, 3) if np.isfinite(r_loo) else np.nan,
                "r_if_well_sign_flipped": round(r_flip, 3) if np.isfinite(r_flip) else np.nan,
                "flip_improves_pooled_r": bool(np.isfinite(r_flip) and np.isfinite(r_base) and r_flip > r_base),
            })
    out = pd.DataFrame(rows)
    if len(out):
        n_flag = int(out["flip_improves_pooled_r"].sum())
        warn(f"SIGN DISCREPANCY CHECK [{label}]: {len(out)} well/product sign disagreement(s) found; "
             f"flipping the well's sign would IMPROVE pooled r for {n_flag} of them -- "
             f"inspect those specific wells/polygons for a possible sign-convention error "
             f"before treating the disagreement as genuine hydrology.")
        if out_csv:
            out.to_csv(out_csv, index=False)
            ok(f"sign discrepancy diagnostic -> {out_csv}")
    else:
        ok(f"SIGN DISCREPANCY CHECK [{label}]: no well/product sign disagreements found.")
    return out


def main():
    os.makedirs(FIGDIR,exist_ok=True); t0=time.time()
    _p(LINE); _p("PHASE 6 FINAL -- MULTI-SCALE VALIDATION & BENCHMARK (v4)"); _p(f"run {dt.datetime.now().isoformat(timespec='seconds')}"); _p(LINE)
    import xarray as xr, pandas as pd
    try:
        import geopandas as gpd; from shapely.geometry import Point; HAVE_GEO=True
    except Exception as e:
        HAVE_GEO=False; warn(f"geopandas unavailable: {e}")

    # ---------------- load product / grid
    hdr("LOAD product / grid")
    G=xr.open_dataset(FINAL_NC); G.load()
    keep=G["keep"].values.astype(bool)
    _Tn=xr.open_dataset(TENSOR_NC)
    if "surface_water" in _Tn:
        _sw=_Tn["surface_water"].values.astype(bool); keep=keep & ~_sw
        _p(f"  excluded {int(_sw.sum())} surface-water cells")
    _Tn.close()
    ci=np.where(keep)[0]
    clat=G["clat"].values; clon=G["clon"].values
    hyb=G["hybas_l3"].values; hyb_ci=hyb[ci]
    ym=pd.to_datetime(G["time"].values).to_period("M")
    mine_trend=G["gwsa_trend"].values                 # full-record (2002-2022) trend -- kept for standalone maps only
    mine_ts_full=G["gwsa_calibrated"].values           # (time, cell), cm -- raw series, needed to re-window trends fairly
    osa=G["osa"].values if "osa" in G else None
    T=xr.open_dataset(TENSOR_NC); T.load()
    if osa is None and "osa" in T: osa=T["osa"].values
    def to_idx(la,lo): return np.round((la+89.75)/0.5).astype(int),np.round((lo+179.75)/0.5).astype(int)
    ai,aj=to_idx(clat[ci],clon[ci])
    _p(f"  cells={ci.size} months={ym.size}")

    # ---------------- WaterGAP
    hdr("LOAD WaterGAP (mm->cm, baseline 2004-2009)")
    wgap_trend=None
    wgap_ts_cell=None; wgap_common_ym=None    # retained monthly array for temporal validation
    wg_an_full=None; wm_full=None    # exposed for per-well window-matched trends later
    if os.path.exists(WGAP_NC):
        Rw=xr.open_dataset(WGAP_NC,decode_times=False)
        u=Rw["time"].attrs.get("units","months since 1901-01-01")
        ref=pd.Timestamp(u.replace("months since","").strip())
        wm=pd.period_range(ref.strftime("%Y-%m"),periods=Rw.sizes["time"],freq="M")
        wg=Rw["groundwstor"].values/10.0; wlat=Rw["lat"].values
        if wlat[0]>wlat[-1]: wg=wg[:,::-1,:]
        ws=list(wm.astype(str)); os_=list(ym.astype(str))
        common_wgap=[c for c in os_ if c in ws]
        wi_wgap=np.array([ws.index(c) for c in common_wgap])    # index into WaterGAP's own time axis (0..1391)
        oi_wgap=np.array([os_.index(c) for c in common_wgap])   # index into MINE's own time axis (0..248)
        base_sel=[k for k,c in enumerate(common_wgap) if BASE_Y0<=int(c[:4])<=BASE_Y1]
        wg_ov=wg[wi_wgap]; wg_an=wg_ov-np.nanmean(wg_ov[base_sel],0)[None]
        cell=wg_an[:,ai,aj]; tyr_wgap=np.arange(len(common_wgap))/12.0
        wgap_trend=np.array([theil(cell[:,k],tyr_wgap) for k in range(ci.size)])
        wgap_ts_cell=cell                                  # (n_overlap, ci) monthly cm -- temporal
        wgap_common_ym=pd.PeriodIndex(common_wgap,freq="M")
        _p(f"  overlap {len(common_wgap)} ({common_wgap[0]}..{common_wgap[-1]})"); ok("WaterGAP trend done.")

        # MINE re-windowed to WaterGAP's exact overlap -- fair pairwise comparison, no 20yr-vs-14yr mismatch
        mine_cell_wgap = mine_ts_full[:,ci][oi_wgap]
        mine_trend_wgap_matched = np.array([theil(mine_cell_wgap[:,k],tyr_wgap) for k in range(ci.size)])
        _p(f"  MINE trend re-matched to WaterGAP's {len(common_wgap)}-month window (was full 2002-2022 before).")

        # baseline against WaterGAP's OWN full record so any well's [t0,t1] can be sliced later, not just the overlap
        base_sel_full=[k for k,c in enumerate(ws) if BASE_Y0<=int(c[:4])<=BASE_Y1]
        wg_an_full = wg - np.nanmean(wg[base_sel_full],0)[None]
        wm_full = wm
    else: warn("WaterGAP missing.")

    # ---------------- SeDA-implied GWSA
    hdr("LOAD SeDA (mm->cm minus OSA)")
    warn("SeDA-implied GWSA subtracts THIS STUDY'S OWN 'osa' -- it is NOT a fully independent check.")
    warn("Treat SeDA as a GRACE-processing consistency check, not independent validation (see OSA-sharing diagnostic below).")
    seda_trend=None
    seda_ts_cell=None; seda_common_ym=None    # retained monthly array for temporal validation
    sda_full_cells=None; ss_full=None    # exposed for per-well window-matched trends later (cell-space, see below)
    seda_trend_wgap_matched=None
    if os.path.exists(SEDA_NC) and osa is not None:
        S=xr.open_dataset(SEDA_NC); S.load()
        sda=S["twsa"].transpose("time","latitude","longitude").values/10.0
        slat=S["latitude"].values
        if slat[0]>slat[-1]: sda=sda[:,::-1,:]
        st=pd.to_datetime(S["time"].values,unit="D",origin="1858-11-17").to_period("M")
        ss=list(st.astype(str)); os_=list(ym.astype(str))
        common_seda=[c for c in os_ if c in ss and 2002<=int(c[:4])<=2019]
        oi_seda=np.array([os_.index(c) for c in common_seda]); si_seda=np.array([ss.index(c) for c in common_seda])
        cell=sda[si_seda][:,ai,aj]-osa[oi_seda][:,ci]; tyr_seda=np.arange(len(common_seda))/12.0
        seda_trend=np.array([theil(cell[:,k],tyr_seda) for k in range(ci.size)])
        seda_ts_cell=cell                                  # (n_overlap, ci) monthly cm -- temporal
        seda_common_ym=pd.PeriodIndex(common_seda,freq="M")
        _p(f"  overlap {len(common_seda)}"); ok("SeDA-implied GWSA trend done.")

        # MINE re-windowed to SeDA's exact overlap
        mine_cell_seda = mine_ts_full[:,ci][oi_seda]
        mine_trend_seda_matched = np.array([theil(mine_cell_seda[:,k],tyr_seda) for k in range(ci.size)])
        _p(f"  MINE trend re-matched to SeDA's {len(common_seda)}-month window.")

        # SeDA re-windowed to WaterGAP's (shorter) overlap, for a fair WGAP-vs-SeDA pairing
        if wgap_trend is not None:
            common_wgap_seda=[c for c in common_wgap if c in ss]
            oi_ws=np.array([os_.index(c) for c in common_wgap_seda]); si_ws=np.array([ss.index(c) for c in common_wgap_seda])
            cell_ws=sda[si_ws][:,ai,aj]-osa[oi_ws][:,ci]; tyr_ws=np.arange(len(common_wgap_seda))/12.0
            seda_trend_wgap_matched=np.array([theil(cell_ws[:,k],tyr_ws) for k in range(ci.size)])
            _p(f"  SeDA trend also re-matched to WaterGAP's {len(common_wgap_seda)}-month window.")

        # full per-month SeDA-minus-OSA anomaly, wherever both exist -- for per-well slicing later
        # NOTE: built in CELL-space (like osa itself), not lat/lon grid-space, since osa is cell-indexed
        common_full=[c for c in ss if c in os_]
        oi_full=np.array([os_.index(c) for c in common_full]); si_full=np.array([ss.index(c) for c in common_full])
        sda_full_cells = sda[si_full][:,ai,aj]-osa[oi_full][:,ci]    # (n_common_full, n_kept_cells)
        ss_full = pd.PeriodIndex(common_full, freq="M")

        # ---- OSA-SHARING CIRCULARITY DIAGNOSTIC ----
        # If OSA's own trend correlates strongly with MINE's trend AND with SeDA-implied trend,
        # part of the MINE-SeDA agreement is coming from a shared OSA bias, not real GW signal.
        hdr("OSA-SHARING CIRCULARITY DIAGNOSTIC (MINE vs SeDA both use this study's OSA)")
        osa_ts_ci = osa[:,ci]                      # (time, cell), this study's OSA restricted to kept cells
        tyr_full = np.arange(osa_ts_ci.shape[0])/12.0
        osa_trend = np.array([theil(osa_ts_ci[:,k],tyr_full) for k in range(ci.size)])
        r_osa_mine,_  = pear(osa_trend, mine_trend[ci])
        r_osa_seda,_  = pear(osa_trend, seda_trend)
        r_mine_seda,_ = pear(mine_trend[ci], seda_trend)
        _p(f"  corr(OSA_trend, MINE_trend)          = {r_osa_mine:+.3f}")
        _p(f"  corr(OSA_trend, SeDA-implied_trend)   = {r_osa_seda:+.3f}")
        _p(f"  corr(MINE_trend, SeDA-implied_trend)  = {r_mine_seda:+.3f}   <- headline MINE-vs-SeDA agreement")
        if np.isfinite(r_osa_mine) and np.isfinite(r_osa_seda) and abs(r_osa_mine)>0.3 and abs(r_osa_seda)>0.3:
            warn(f"OSA trend correlates with BOTH MINE (r={r_osa_mine:+.2f}) and SeDA-implied (r={r_osa_seda:+.2f}).")
            warn("Some of the MINE-vs-SeDA agreement above is likely inflated by shared OSA, not independent skill.")
        else:
            ok("OSA trend does not strongly co-vary with both products -- MINE-vs-SeDA agreement is less likely to be an OSA artifact.")
    else: warn("SeDA/OSA missing.")

    # ---------------- wells
    hdr("LOAD wells")
    A=pd.read_parquet(WELL_ATTR)
    n0=len(A)
    A=A[np.isfinite(A["latitude"])&np.isfinite(A["longitude"])&np.isfinite(A["slope_storage_cm_yr"])].copy()
    _p(f"  wells with finite lat/lon/slope: {len(A)}/{n0}")

    # ---- Sy (specific yield) QC / transparency check ----
    # slope_storage_cm_yr already has Sy applied upstream (Stage 02, well_trends_v2_tiered), so we
    # do NOT recompute storage here. But we DO surface how many wells relied on a fallback/invalid
    # Sy, same as Phase 6.1's explicit DEFAULT_SY imputation -- this matters for uncertainty framing.
    if "specific_y" in A.columns:
        sy_raw=pd.to_numeric(A["specific_y"],errors="coerce")
        bad_sy=~np.isfinite(sy_raw)|(sy_raw<=0)
        A["sy_is_fallback"]=bad_sy.values
        n_bad=int(bad_sy.sum())
        if n_bad:
            warn(f"{n_bad}/{len(A)} wells ({100*n_bad/len(A):.1f}%) have missing/invalid specific_y "
                 f"(<=0 or NaN) -- their slope_storage_cm_yr upstream should have used the "
                 f"DEFAULT_SY={DEFAULT_SY} fallback. Flagged in A['sy_is_fallback'] for downstream QC.")
            # -- OPTIONAL cross-check against a spatial Sy zonation map, if configured below.
            # This is REPORTING ONLY: it cannot retroactively fix slope_storage_cm_yr (that value
            # is precomputed upstream in Stage 02 / Phase 6.1). It just tells you how many of the
            # flat-default wells sit inside a region where a better, spatially-appropriate Sy exists,
            # so you know whether re-running the upstream stage with a spatial Sy fill is worth it.
            SY_SHP=r"D:\THESIS\Arpon_Thesis_Final\DATA\specific yeild-上传\geishuidu-min.shp"
            SY_COL=None   # <-- set to the shapefile's Sy attribute column name to enable this check
            if SY_COL is not None and os.path.exists(SY_SHP) and HAVE_GEO:
                try:
                    syz=gpd.read_file(SY_SHP)
                    syz=syz.to_crs(4326) if syz.crs is not None else syz.set_crs(4326)
                    if SY_COL in syz.columns:
                        bad_pts=gpd.GeoDataFrame(A[bad_sy],geometry=[Point(lo,la) for lo,la in
                                    zip(A.loc[bad_sy,"longitude"],A.loc[bad_sy,"latitude"])],crs=4326)
                        covered=gpd.sjoin(bad_pts,syz[[SY_COL,"geometry"]],how="left",predicate="within")
                        n_covered=int(covered[SY_COL].notna().sum())
                        _p(f"  of those {n_bad} flat-default wells, {n_covered} fall inside the "
                           f"specific-yield map's coverage -- re-running the upstream Sy fill with "
                           f"that map (see phase6_1_specific_yield_spatial_fill_patch.py) would give "
                           f"them a spatially appropriate Sy instead of the flat {DEFAULT_SY}.")
                    else:
                        warn(f"SY_COL not found in {os.path.basename(SY_SHP)}; skipping spatial Sy cross-check.")
                except Exception as e:
                    warn(f"spatial Sy cross-check failed: {e}")
        else:
            ok("all wells have a finite, positive specific_y on file.")
    else:
        A["sy_is_fallback"]=False
        warn("no 'specific_y' column in well attribute table -- cannot audit Sy provenance here; "
             "trusting upstream Stage 02 to have applied the DEFAULT_SY fallback correctly.")

    # ---- CI-width floor diagnostic ----
    # np.maximum(...,1e-3) silently protects against divide-by-zero for missing/degenerate CI widths,
    # but a well hitting that floor gets an artificially huge weight -- worth knowing how many do.
    ciw=pd.to_numeric(A["ci_width_cm_yr"],errors="coerce")
    floored=~np.isfinite(ciw)|(ciw<=1e-3*3.92)
    n_floor=int(floored.sum())
    if n_floor:
        warn(f"{n_floor}/{len(A)} wells ({100*n_floor/len(A):.1f}%) have missing/near-zero "
             f"ci_width_cm_yr and hit the weight floor -- these get outsized influence in "
             f"weighted means (wmean/wm). Consider capping weights if this fraction is large.")
    A["w"]=1.0/np.maximum(A["ci_width_cm_yr"]/3.92,1e-3)**2

    inv=pd.read_parquet(GROW_INV)
    if "GROW_ID" not in inv.columns: inv=inv.reset_index().rename(columns={inv.index.name or "index":"GROW_ID"})
    # Robust ID cleaning (strip a stray '.0' from float round-tripping) instead of a bare
    # .astype(str).str.strip() -- the bare version was quietly failing to match IDs like
    # '10234.0' against '10234', shrinking the well set that gets interval/span_yr/t0/t1.
    A["GROW_ID"]=clean_id(A["GROW_ID"]); inv["GROW_ID"]=clean_id(inv["GROW_ID"])
    inv["t0"]=pd.to_datetime(inv["t0"]); inv["t1"]=pd.to_datetime(inv["t1"]); inv["span_yr"]=(inv["t1"]-inv["t0"]).dt.days/365.25
    A=A.drop(columns=[c for c in ["interval","span_yr","t0","t1","n"] if c in A.columns])
    n_before_inv=len(A)
    A=A.merge(inv[[c for c in ["GROW_ID","interval","span_yr","t0","t1"] if c in inv.columns]],on="GROW_ID",how="left")
    n_matched_inv=int(A["span_yr"].notna().sum()) if "span_yr" in A.columns else 0
    _p(f"  GROW_ID matched to inventory: {n_matched_inv}/{n_before_inv} "
       f"({100*n_matched_inv/max(n_before_inv,1):.1f}%)")
    if n_matched_inv < 0.9*n_before_inv:
        warn("less than 90% of wells matched the GROW inventory by GROW_ID -- check ID formatting "
             "upstream (well_trends_v2_tiered.parquet vs grow_inventory.parquet) if this seems low.")
    aiw,ajw=to_idx(A["latitude"].values,A["longitude"].values); A["jc"]=aiw*10000+ajw
    prod=pd.DataFrame({"jc":ai*10000+aj,"hyb":hyb_ci,"mine":mine_trend[ci],
                       "clat":clat[ci],"clon":clon[ci]})
    if wgap_trend is not None: prod["wgap"]=wgap_trend
    if seda_trend is not None: prod["seda"]=seda_trend
    if wgap_trend is not None: prod["mine_wgap_matched"]=mine_trend_wgap_matched
    if seda_trend is not None: prod["mine_seda_matched"]=mine_trend_seda_matched
    if seda_trend_wgap_matched is not None: prod["seda_wgap_matched"]=seda_trend_wgap_matched
    A=A.merge(prod[["jc","hyb"]],on="jc",how="left")
    ys=A[(A["interval"]=="YS")&(A["span_yr"]>=YS_MIN_YR)].copy()
    _p(f"  YS wells >=8yr: {len(ys)}")
    PRODUCTS=[("MINE","mine")]+([("WGAP","wgap")] if wgap_trend is not None else [])+([("SEDA","seda")] if seda_trend is not None else [])

    # Independence tier -- for reporting only, doesn't affect stats, but keep this straight in every table/figure/caption
    INDEPENDENCE_TIER = {
        "MINE": "This study",
        "WGAP": "Independent (physically-based, no GRACE)",
        "SEDA": "Quasi-independent -- shares GRACE source AND this study's own OSA subtraction; treat as a "
                "consistency check, not proof of accuracy",
    }
    for nm in [n for n,_ in PRODUCTS]:
        _p(f"  independence[{nm}] = {INDEPENDENCE_TIER.get(nm,'?')}")

    # ---------------------------------------------------------------
    # WINDOW-MATCHED PRODUCT TRENDS AT EACH WELL'S OWN [t0,t1]
    # ---------------------------------------------------------------
    hdr("COMPUTE PER-WELL WINDOW-MATCHED PRODUCT TRENDS (exact well t0-t1 per row)")
    MIN_MATCH_MONTHS = 24   # need >=2yr overlap with a product's own record to trust a matched trend

    jc_to_pos = {int(v): p for p, v in enumerate(ai*10000+aj)}   # jc -> position within kept-cell arrays

    def _theil_on_window(vals, periods, t0, t1):
        if pd.isna(t0) or pd.isna(t1): return np.nan
        t0p, t1p = pd.Period(t0, "M"), pd.Period(t1, "M")
        m = (periods >= t0p) & (periods <= t1p) & np.isfinite(vals)
        if m.sum() < MIN_MATCH_MONTHS: return np.nan
        v = vals[m]; tp = periods[m]
        tyr = (tp.to_timestamp() - tp.to_timestamp()[0]).days / 365.25
        return theil(v, tyr)

    n_ys = len(ys)
    mine_matched = np.full(n_ys, np.nan)
    wgap_matched = np.full(n_ys, np.nan)
    seda_matched = np.full(n_ys, np.nan)
    jc_arr = ys["jc"].values; t0_arr = ys["t0"].values; t1_arr = ys["t1"].values

    for i in range(n_ys):
        jcv = int(jc_arr[i]); aiw_i, ajw_i = jcv // 10000, jcv % 10000
        t0i, t1i = t0_arr[i], t1_arr[i]
        pos = jc_to_pos.get(jcv)
        if pos is not None:
            mine_matched[i] = _theil_on_window(mine_ts_full[:,ci[pos]], ym, t0i, t1i)
        if wg_an_full is not None and 0<=aiw_i<NLAT and 0<=ajw_i<NLON:
            wgap_matched[i] = _theil_on_window(wg_an_full[:,aiw_i,ajw_i], wm_full, t0i, t1i)
        if sda_full_cells is not None and pos is not None:
            seda_matched[i] = _theil_on_window(sda_full_cells[:,pos], ss_full, t0i, t1i)

    ys["mine_matched"]=mine_matched; ys["wgap_matched"]=wgap_matched; ys["seda_matched"]=seda_matched
    match_cols = {"MINE":"mine_matched","WGAP":"wgap_matched","SEDA":"seda_matched"}
    _p(f"  matched MINE : {np.isfinite(mine_matched).sum()}/{n_ys} wells (>= {MIN_MATCH_MONTHS}mo overlap with well's own record)")
    _p(f"  matched WGAP : {np.isfinite(wgap_matched).sum()}/{n_ys} wells")
    _p(f"  matched SEDA : {np.isfinite(seda_matched).sum()}/{n_ys} wells")

    rows=[]
    def rec(scale,nm,x,y,extra=None):
        r,_=pear(x,y); sm,_=signm(x,y); d=dict(scale=scale,product=nm,r=round(r,4),
            RMSE_cm_yr=round(rmse(y,x),4),NSE=round(nse(y,x),4),bias=round(bias(y,x),4),
            sign_pct=round(100*sm,1),n=int(np.isfinite(x*y).sum()))
        if extra: d.update(extra)
        rows.append(d); return d

    # ---------------- cell scale (each well compared to product trend over that well's OWN t0-t1)
    hdr("CELL scale vs wells (window-matched)")
    def wm(g):
        w=g["w"].values; v=g["slope_storage_cm_yr"].values; m=np.isfinite(w)&np.isfinite(v)&(w>0)
        return np.sum(w[m]*v[m])/np.sum(w[m]) if m.sum() else np.nan
    def wmean(g,col):
        w=g["w"].values; v=g[col].values; m=np.isfinite(w)&np.isfinite(v)&(w>0)
        return np.sum(w[m]*v[m])/np.sum(w[m]) if m.sum() else np.nan

    def _agg_cell(g):
        d={"well":wmean(g,"slope_storage_cm_yr"),"n":len(g)}
        for nm,mcol in match_cols.items():
            if nm in dict(PRODUCTS): d[dict(PRODUCTS)[nm]]=wmean(g,mcol)
        return pd.Series(d)

    CW=ys.groupby("jc").apply(_agg_cell,include_groups=False).reset_index()
    CW=CW[CW["n"]>=MIN_WELLS_POLY]
    _p(f"  {'prod':<7}{'r':<8}{'RMSE':<8}{'NSE':<9}{'sign%':<7} (cells={len(CW)})")
    for nm,c in PRODUCTS:
        if c not in CW.columns: continue
        d=rec("cell",nm,CW["well"].values,CW[c].values)
        _p(f"  {nm:<7}{d['r']:+.3f}  {d['RMSE_cm_yr']:<8.3f}{d['NSE']:<+9.3f}{d['sign_pct']:<7}")

    # ---------------- density bins
    hdr("BENCHMARKING BY WELL DENSITY TIER")
    CW["density_tier"] = pd.cut(CW["n"], bins=DENSITY_BINS, labels=DENSITY_LABELS, right=False)
    for lab in DENSITY_LABELS:
        sub_cw = CW[CW["density_tier"] == lab]
        for nm, c in PRODUCTS:
            rec(f"density_{lab}", nm, sub_cw["well"].values, sub_cw[c].values)
        r_str = " ".join(f"{nm}={pear(sub_cw['well'].values, sub_cw[c].values)[0]:+.2f}" for nm, c in PRODUCTS)
        _p(f"  Bin {lab:<8} (n_cells={len(sub_cw):<4}): {r_str}")

    # ---------------- aquifer scale
    hdr("AQUIFER scale vs wells (window-matched)")
    aqM=None
    if HAVE_GEO and os.path.exists(AQUIFERS):
        aq=gpd.read_file(AQUIFERS).to_crs(4326)
        gp=gpd.GeoDataFrame(ys,geometry=[Point(a,b) for a,b in zip(ys["longitude"],ys["latitude"])],crs=4326)
        jw=gpd.sjoin(gp,aq[["Aquifer","geometry"]],how="inner",predicate="within")

        def _agg_aq(g):
            d={"well":np.average(g["slope_storage_cm_yr"],weights=g["w"]) if g["w"].sum()>0 else g["slope_storage_cm_yr"].mean(),
               "n":len(g)}
            for nm,mcol in match_cols.items():
                d[f"{dict(PRODUCTS).get(nm,nm.lower())}_wm"]=wmean(g,mcol)
            return pd.Series(d)

        wl=jw.groupby("Aquifer").apply(_agg_aq,include_groups=False).reset_index()
        wl=wl[wl["n"]>=MIN_WELLS_POLY]

        # grid-based, PAIRWISE-window-matched product columns -- for inter-product-only comparisons (no wells involved)
        pgp=gpd.GeoDataFrame(prod,geometry=[Point(prod["clon"].iloc[k],prod["clat"].iloc[k]) for k in range(len(prod))],crs=4326)
        pj=gpd.sjoin(pgp,aq[["Aquifer","geometry"]],how="inner",predicate="within")
        agg_map={c:"mean" for _,c in PRODUCTS}
        for extra in ["mine_wgap_matched","mine_seda_matched","seda_wgap_matched"]:
            if extra in pj.columns: agg_map[extra]="mean"
        pr=pj.groupby("Aquifer").agg(agg_map).reset_index()

        aqM=wl.merge(pr,on="Aquifer",how="inner")
        _p(f"  {'prod':<7}{'r':<8}{'CI95':<16}{'RMSE':<8}{'NSE':<9}{'sign%':<7} (aquifers={len(aqM)})")
        for nm,c in PRODUCTS:
            wm_c=f"{c}_wm"
            if wm_c not in aqM.columns: continue
            rb,lo,hi=boot_r(aqM["well"].values,aqM[wm_c].values)
            d=rec("aquifer",nm,aqM["well"].values,aqM[wm_c].values,dict(CI_lo=round(lo,3),CI_hi=round(hi,3)))
            _p(f"  {nm:<7}{d['r']:+.3f}  [{lo:+.2f},{hi:+.2f}]   {d['RMSE_cm_yr']:<8.3f}{d['NSE']:<+9.3f}{d['sign_pct']:<7}")
        ok("aquifer done.")

        aq_prod_cols=[f"{c}_wm" for _,c in PRODUCTS if f"{c}_wm" in aqM.columns]
        sign_discrepancy_diagnostic(aqM,"Aquifer","well",aq_prod_cols,
            out_csv=os.path.join(VALID,"sign_discrepancy_aquifer.csv"),label="aquifer")

    # ---------------- basin scale
    hdr("BASIN scale (+>=200k km2) vs wells (window-matched)")
    yb=ys[ys["hyb"]>0]

    def _agg_bas(g):
        d={"well":wm(g),"n":len(g)}
        for nm,mcol in match_cols.items():
            d[f"{dict(PRODUCTS).get(nm,nm.lower())}_wm"]=wmean(g,mcol)
        return pd.Series(d)

    wlb=yb.groupby("hyb").apply(_agg_bas,include_groups=False).reset_index()
    wlb=wlb[wlb["n"]>=MIN_WELLS_POLY]

    agg_map={c:"mean" for _,c in PRODUCTS}
    for extra in ["mine_wgap_matched","mine_seda_matched","seda_wgap_matched"]:
        if extra in prod.columns: agg_map[extra]="mean"
    prb=prod.groupby("hyb").agg(agg_map).reset_index()

    Bb=wlb.merge(prb,on="hyb",how="inner")
    area_map={}
    if HAVE_GEO and os.path.exists(HYBAS_GPKG):
        hb=gpd.read_file(HYBAS_GPKG); ac=[c for c in hb.columns if c.upper()=="SUB_AREA"]
        if ac: area_map=dict(zip(hb["HYBAS_ID"].astype("int64"),hb[ac[0]].astype(float)))
    Bb["area"]=Bb["hyb"].map(area_map); big=Bb[Bb["area"]>=AREA_LARGE]
    _p(f"  {'prod':<7}{'r_all':<9}{'RMSE':<8}{'r>=200k':<9}{'n>=200k':<7} (basins={len(Bb)})")
    for nm,c in PRODUCTS:
        wm_c=f"{c}_wm"
        if wm_c not in Bb.columns: continue
        d=rec("basin",nm,Bb["well"].values,Bb[wm_c].values)
        rbig,nbig=pear(big["well"].values,big[wm_c].values)
        rec("basin_ge200k",nm,big["well"].values,big[wm_c].values)
        _p(f"  {nm:<7}{d['r']:+.3f}   {d['RMSE_cm_yr']:<8.3f}{rbig:+.3f}    {nbig:<7}")

    bas_prod_cols=[f"{c}_wm" for _,c in PRODUCTS if f"{c}_wm" in Bb.columns]
    sign_discrepancy_diagnostic(Bb,"hyb","well",bas_prod_cols,
        out_csv=os.path.join(VALID,"sign_discrepancy_basin.csv"),label="basin")

    # ---------------- HOTSPOT VALIDATION (cf. Wang et al. 2025 igGWSA) ----------------
    hdr("HOTSPOT VALIDATION -- depletion hotspots (cf. Wang et al. 2025)")
    HOTSPOTS={"North_China_Plain":(34,41,113,120),"Central_Valley":(35,40.5,-122.5,-118.5),
              "High_Plains":(32,43,-103.5,-96),"Ganges_Brahmaputra":(22,31,75,90),
              "Punjab_Indus":(27,33,70,77)}
    cw_la=(CW["jc"].values//10000)*0.5-89.75; cw_lo=(CW["jc"].values%10000)*0.5-179.75
    hot_rows=[]
    _p(f"  {'hotspot':<22}{'n':<6}" + "".join(f"{n:<9}" for n,_ in PRODUCTS))
    pooled=np.zeros(len(CW),bool)
    for hname,(la0,la1,lo0,lo1) in HOTSPOTS.items():
        sel=(cw_la>=la0)&(cw_la<=la1)&(cw_lo>=lo0)&(cw_lo<=lo1); pooled|=sel
        sub=CW[sel]
        if len(sub)<5: _p(f"  {hname:<22}{len(sub):<6}(too few)"); continue
        line=f"  {hname:<22}{len(sub):<6}"; row={"hotspot":hname,"n_cells":int(len(sub))}
        for nm,c in PRODUCTS:
            if c not in sub.columns: line+=f"{'--':<9}"; continue
            r,_=pear(sub["well"].values,sub[c].values); row[f"{nm}_r"]=round(r,3)
            line+=f"{r:+.3f}   "; rec(f"hotspot_{hname}",nm,sub["well"].values,sub[c].values)
        _p(line); hot_rows.append(row)
    ps=CW[pooled]
    _p(f"  {'POOLED':<22}{len(ps):<6}" + "".join(f"{pear(ps['well'].values,ps[c].values)[0]:+.3f}   " for _,c in PRODUCTS))
    for nm,c in PRODUCTS:
        if c in ps.columns: rec("hotspot_pooled",nm,ps["well"].values,ps[c].values)
    _p("  READ: agreement rises at hotspots (large signal, dense wells); cf. Wang 2025 PCC 0.89-0.99.")
    _p("  Our GLOBAL 515-aquifer validation is a harder, broader test than hotspot-only.")
    try: pd.DataFrame(hot_rows).to_csv(os.path.join(VALID,"hotspot_metrics_FINAL.csv"),index=False); ok("hotspot csv")
    except Exception as e: warn(f"hotspot csv: {e}")

    # ---------------- TEMPORAL VALIDATION -- basin-averaged time series ----------------
    hdr("TEMPORAL VALIDATION -- basin-averaged time series")
    def basin_monthly(ts):
        return {b: np.nanmean(ts[:,np.where(hyb_ci==b)[0]],axis=1) for b in np.unique(hyb_ci[hyb_ci>0])}
    prod_monthly={"MINE":basin_monthly(mine_ts_full[:,ci])}; prod_ym={"MINE":ym}
    if wgap_ts_cell is not None: prod_monthly["WGAP"]=basin_monthly(wgap_ts_cell); prod_ym["WGAP"]=wgap_common_ym
    if seda_ts_cell is not None: prod_monthly["SEDA"]=basin_monthly(seda_ts_cell); prod_ym["SEDA"]=seda_common_ym
    from itertools import combinations
    temporal_rows=[]
    _p("  (A) product-vs-product basin MONTHLY temporal r:")
    for p1,p2 in combinations(list(prod_monthly),2):
        ym1=list(prod_ym[p1].astype(str)); ym2set=set(prod_ym[p2].astype(str))
        common=[m for m in ym1 if m in ym2set]
        if len(common)<24: continue
        ym2l=list(prod_ym[p2].astype(str))
        i1=[ym1.index(m) for m in common]; i2=[ym2l.index(m) for m in common]; rs=[]
        for b in prod_monthly[p1]:
            if b not in prod_monthly[p2]: continue
            a=prod_monthly[p1][b][i1]; bb=prod_monthly[p2][b][i2]; m=np.isfinite(a)&np.isfinite(bb)
            if m.sum()>=24 and np.std(a[m])>0 and np.std(bb[m])>0: rs.append(np.corrcoef(a[m],bb[m])[0,1])
        rs=np.array(rs)
        if rs.size:
            _p(f"    {p1} vs {p2}: median temporal r={np.nanmedian(rs):+.3f} (basins={rs.size})")
            temporal_rows.append({"comparison":f"{p1}_vs_{p2}","type":"prod_monthly",
                                  "median_r":round(float(np.nanmedian(rs)),3),"n_basins":int(rs.size)})
    _p("  (C) seasonal phase (peak month) agreement, MINE vs others:")
    def peak_month(series,ymp):
        d=pd.DataFrame({"v":series,"m":ymp.month}); c=d.groupby("m")["v"].mean()
        return int(c.idxmax()) if c.notna().any() else np.nan
    for p2 in [k for k in prod_monthly if k!="MINE"]:
        ym1=list(ym.astype(str)); ym2l=list(prod_ym[p2].astype(str)); ym2set=set(ym2l)
        common=[m for m in ym1 if m in ym2set]
        if len(common)<24: continue
        i1=[ym1.index(m) for m in common]; i2=[ym2l.index(m) for m in common]
        ymp1=ym[i1]; ymp2=prod_ym[p2][i2]; dph=[]
        for b in prod_monthly["MINE"]:
            if b not in prod_monthly[p2]: continue
            pm1=peak_month(prod_monthly["MINE"][b][i1],ymp1); pm2=peak_month(prod_monthly[p2][b][i2],ymp2)
            if np.isfinite(pm1) and np.isfinite(pm2):
                dd=abs(pm1-pm2); dph.append(min(dd,12-dd))
        dph=np.array(dph)
        if dph.size:
            _p(f"    MINE vs {p2}: median phase diff={np.nanmedian(dph):.1f} mo (within 1mo: {100*np.mean(dph<=1):.0f}%, n={dph.size})")
            temporal_rows.append({"comparison":f"MINE_vs_{p2}","type":"phase",
                                  "median_phase_months":round(float(np.nanmedian(dph)),1),"n_basins":int(dph.size)})
    try: pd.DataFrame(temporal_rows).to_csv(os.path.join(VALID,"temporal_metrics_FINAL.csv"),index=False); ok("temporal csv")
    except Exception as e: warn(f"temporal csv: {e}")

    # ---------------- INTER-PRODUCT PAIRWISE COMPARISONS (each pair on its own matched common window)
    hdr("INTER-PRODUCT DIRECT COMPARISONS (window-matched per pair)")
    inter_rows=[]
    prod_cols = dict(PRODUCTS)
    # (name1, matched_col1, name2, matched_col2, window note, caveat)
    pair_defs = [
        ("MINE","mine_wgap_matched","WGAP","wgap",
         "MINE re-windowed to WaterGAP's overlap", ""),
        ("MINE","mine_seda_matched","SEDA","seda",
         "MINE re-windowed to SeDA's overlap",
         "CAUTION: MINE & SeDA share GRACE + this study's own OSA -- not independent, see diagnostic above"),
        ("WGAP","wgap","SEDA","seda_wgap_matched",
         "SeDA re-windowed to WaterGAP's overlap",
         "SeDA side still uses this study's own OSA"),
    ]
    for p1,c1,p2,c2,note,caveat in pair_defs:
        if p1 not in prod_cols or p2 not in prod_cols: continue
        if c1 not in prod.columns or c2 not in prod.columns: continue
        r_c,_=pear(prod[c1].values,prod[c2].values); sm_c,_=signm(prod[c1].values,prod[c2].values)
        r_a=sm_a=np.nan
        if aqM is not None and c1 in aqM.columns and c2 in aqM.columns:
            r_a,_=pear(aqM[c1].values,aqM[c2].values); sm_a,_=signm(aqM[c1].values,aqM[c2].values)
        r_b=sm_b=np.nan
        if Bb is not None and c1 in Bb.columns and c2 in Bb.columns:
            r_b,_=pear(Bb[c1].values,Bb[c2].values); sm_b,_=signm(Bb[c1].values,Bb[c2].values)

        _p(f"  {p1} vs {p2}  ({note})" + (f"  [{caveat}]" if caveat else "") + ":")
        _p(f"    Cell    : r={r_c:+.3f}, sign_match={sm_c*100:.1f}%")
        if aqM is not None: _p(f"    Aquifer : r={r_a:+.3f}, sign_match={sm_a*100:.1f}%")
        if Bb is not None:  _p(f"    Basin   : r={r_b:+.3f}, sign_match={sm_b*100:.1f}%")

        inter_rows.append({"pair":f"{p1}_vs_{p2}","window_note":note,"caveat":caveat,
                            "cell_r":round(r_c,3),"cell_sign_pct":round(sm_c*100,1),
                            "aquifer_r":round(r_a,3) if np.isfinite(r_a) else np.nan,
                            "aquifer_sign_pct":round(sm_a*100,1) if np.isfinite(sm_a) else np.nan,
                            "basin_r":round(r_b,3) if np.isfinite(r_b) else np.nan,
                            "basin_sign_pct":round(sm_b*100,1) if np.isfinite(sm_b) else np.nan})

    # ---------------- EXPORT table
    hdr("EXPORT metrics summary")
    df=pd.DataFrame(rows)
    csv=os.path.join(VALID,"benchmark_metrics_FINAL.csv"); df.to_csv(csv,index=False)
    if inter_rows:
        pd.DataFrame(inter_rows).to_csv(os.path.join(VALID,"interproduct_metrics_FINAL.csv"),index=False)
    ok(f"written {csv} ({len(df)} rows)")
    _p("")
    _p("  === BENCHMARK METRICS SUMMARY ===")
    show=df[df["scale"].isin(["cell","aquifer","basin","basin_ge200k"])]
    _p(show.to_string(index=False))

    # ---------------- FIGURES
    # ---------------- FIGURES
    try:
        _figs(prod,ci,clat,clon,keep,mine_trend,wgap_trend,seda_trend,hyb,
              CW,aqM,Bb,PRODUCTS,df,ys,
              hot_rows=hot_rows,temporal_rows=temporal_rows)
    except Exception as e:
        warn(f"figures: {e}"); traceback.print_exc()

    with open(os.path.join(VALID,"benchmark_summary_FINAL.txt"),"w",encoding="utf-8") as fh: fh.write("\n".join(_LOG))
    _p(f"\n  wall {(time.time()-t0)/60:.1f} min"); _p(LINE)


def _figs(prod,ci,clat,clon,keep,mine_tr,wgap_tr,seda_tr,hyb,CW,aqM,Bb,PRODUCTS,df,ys,
          hot_rows=None,temporal_rows=None):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, numpy as np
    sys.path.insert(0,FIG_STYLE_DIR); import fig_style as fs; fs.apply()

    DISP = PRODUCT_DISPLAY
    COL  = PRODUCT_COLOR
    names=[n for n,_ in PRODUCTS]

    SZ_TITLE, SZ_LABEL, SZ_TICK, SZ_ANNOT = 16, 14, 13, 12
    matplotlib.rcParams.update({"font.size": SZ_TICK})

    def grid(vals):
        g=np.full((NLAT,NLON),np.nan)
        jla=np.round((clat[ci]+89.75)/0.5).astype(int); jlo=np.round((clon[ci]+179.75)/0.5).astype(int)
        ok_=(jla>=0)&(jla<NLAT)&(jlo>=0)&(jlo<NLON); g[jla[ok_],jlo[ok_]]=vals[ok_]; return g

    trends={"MINE":mine_tr[ci]}
    if wgap_tr is not None: trends["WGAP"]=wgap_tr
    if seda_tr is not None: trends["SEDA"]=seda_tr
    lim=float(np.nanpercentile(np.abs(mine_tr[ci][np.isfinite(mine_tr[ci])]),97))

    # ================================================================
    # FIG 1: Cell-Scale GWSA Trend Maps
    # ================================================================
    PANELS = [
        ("MINE", "a", "fig_trend_map_a_mlgwsa", False),
        ("WGAP", "b", "fig_trend_map_b_wgap",   False),
        ("SEDA", "c", "fig_trend_map_c_seda",   True),
    ]
    for nm, letter, fname, colorbar_on in PANELS:
        if nm not in trends: continue
        fig=plt.figure(figsize=(7.4,4.0 if colorbar_on else 3.6))
        map_rect = (0.05,0.20,0.90,0.70) if colorbar_on else (0.05,0.10,0.90,0.80)
        ax=fs.global_map_axis(fig,map_rect,projection="robinson",basins=False)
        m=fs.pcolor_global_smooth(ax,grid(trends[nm]),NLAT,NLON,fs.CMAP_TREND,-lim,lim,upsample=3)
        ax.set_title(f"{DISP.get(nm,nm)} cell-scale GWSA trend",fontsize=SZ_TITLE)
        fs.panel_letter(ax,letter)
        if colorbar_on:
            fs.hcolorbar(fig,m,(0.28,0.075,0.44,0.020),"GWSA trend (cm yr$^{-1}$)")
        fs.save(fig,fname,figdir=FIGDIR); plt.close(fig)

    # ================================================================
    # FIG 1B: Aquifer-Scale Aggregated Trend Maps
    # ================================================================
    if aqM is not None:
        for nm, letter, fname, cbar_on in [("MINE","a","fig_aquifer_trend_map_a_mlgwsa",False),
                                          ("WGAP","b","fig_aquifer_trend_map_b_wgap",False),
                                          ("SEDA","c","fig_aquifer_trend_map_c_seda",True)]:
            if nm not in names: continue
            col_nm = dict(PRODUCTS)[nm]
            fig=plt.figure(figsize=(7.4,4.0 if cbar_on else 3.6))
            map_rect = (0.05,0.20,0.90,0.70) if cbar_on else (0.05,0.10,0.90,0.80)
            ax=fs.global_map_axis(fig,map_rect,projection="robinson",basins=False)
            m=fs.pcolor_global_smooth(ax,grid(prod[col_nm].values),NLAT,NLON,fs.CMAP_TREND,-lim,lim,upsample=3)
            ax.set_title(f"{DISP.get(nm,nm)} aquifer-aggregated trend",fontsize=SZ_TITLE)
            fs.panel_letter(ax,letter)
            if cbar_on:
                fs.hcolorbar(fig,m,(0.28,0.075,0.44,0.020),"GWSA trend (cm yr$^{-1}$)")
            fs.save(fig,fname,figdir=FIGDIR); plt.close(fig)

    # ================================================================
    # FIG 1C: Basin-Scale Aggregated Trend Maps
    # ================================================================
    if Bb is not None:
        for nm, letter, fname, cbar_on in [("MINE","a","fig_basin_trend_map_a_mlgwsa",False),
                                          ("WGAP","b","fig_basin_trend_map_b_wgap",False),
                                          ("SEDA","c","fig_basin_trend_map_c_seda",True)]:
            if nm not in names: continue
            col_nm = dict(PRODUCTS)[nm]
            b_dict = dict(zip(Bb["hyb"], Bb[col_nm]))
            b_vals = prod["hyb"].map(b_dict).values
            fig=plt.figure(figsize=(7.4,4.0 if cbar_on else 3.6))
            map_rect = (0.05,0.20,0.90,0.70) if cbar_on else (0.05,0.10,0.90,0.80)
            ax=fs.global_map_axis(fig,map_rect,projection="robinson",basins=False)
            m=fs.pcolor_global_smooth(ax,grid(b_vals),NLAT,NLON,fs.CMAP_TREND,-lim,lim,upsample=3)
            ax.set_title(f"{DISP.get(nm,nm)} basin-aggregated trend",fontsize=SZ_TITLE)
            fs.panel_letter(ax,letter)
            if cbar_on:
                fs.hcolorbar(fig,m,(0.28,0.075,0.44,0.020),"GWSA trend (cm yr$^{-1}$)")
            fs.save(fig,fname,figdir=FIGDIR); plt.close(fig)

    # ================================================================
    # FIG 2: Benchmark Bars Across Spatial Scales & Well Density Tiers
    # ================================================================
    fig=plt.figure(figsize=(11.0,5.2))
    W=0.8/len(names)

    # Panel a: Spatial Scale Performance
    ax1=fig.add_axes([0.09,0.16,0.38,0.58])
    scales=["cell","aquifer","basin","basin_ge200k"]
    all_vals_a = []
    for j,nm in enumerate(names):
        vals=[df[(df.scale==s)&(df["product"]==nm)]["r"].values for s in scales]
        vals=[v[0] if len(v) else np.nan for v in vals]
        x_bars = np.arange(len(scales)) + j * W
        ax1.bar(x_bars, vals, W, label=DISP.get(nm,nm), color=COL.get(nm,"#666666"))
        all_vals_a.extend([v for v in vals if np.isfinite(v)])

    ax1.set_xticks(np.arange(len(scales))+W*(len(names)-1)/2)
    ax1.set_xticklabels(["Cell","Aquifer","Basin","Basin\n$\\geq$200k km$^2$"],fontsize=SZ_TICK)
    ax1.set_ylabel("Pearson $r$ vs wells",fontsize=SZ_LABEL); ax1.axhline(0,color="k",lw=0.6)
    ax1.tick_params(axis="y",labelsize=SZ_TICK)
    ax1.set_title("Agreement with wells by scale",fontsize=SZ_TITLE,pad=12)
    ax1.spines[["top","right"]].set_visible(False); fs.panel_letter(ax1,"a")
    if all_vals_a:
        ax1.set_ylim(min(0, min(all_vals_a) - 0.12), max(0.4, max(all_vals_a) + 0.15))

    # Panel b: Well Density Tiers
    ax2=fig.add_axes([0.57,0.16,0.38,0.58])
    all_vals_b = []
    for j,nm in enumerate(names):
        vals=[df[(df.scale==f"density_{lab}")&(df["product"]==nm)]["r"].values for lab in DENSITY_LABELS]
        vals=[v[0] if len(v) else np.nan for v in vals]
        x_bars = np.arange(len(DENSITY_LABELS)) + j * W
        ax2.bar(x_bars, vals, W, label=DISP.get(nm,nm), color=COL.get(nm,"#666666"))
        all_vals_b.extend([v for v in vals if np.isfinite(v)])

    ax2.set_xticks(np.arange(len(DENSITY_LABELS))+W*(len(names)-1)/2)
    ax2.set_xticklabels(DENSITY_LABELS,fontsize=SZ_TICK)
    ax2.set_xlabel("Wells per cell",fontsize=SZ_LABEL)
    ax2.set_ylabel("Pearson $r$ vs wells",fontsize=SZ_LABEL); ax2.axhline(0,color="k",lw=0.6)
    ax2.tick_params(axis="y",labelsize=SZ_TICK)
    ax2.set_title("Agreement by well density tier",fontsize=SZ_TITLE,pad=12)
    ax2.spines[["top","right"]].set_visible(False); fs.panel_letter(ax2,"b")
    if all_vals_b:
        ax2.set_ylim(min(0, min(all_vals_b) - 0.12), max(0.4, max(all_vals_b) + 0.15))

    handles, labels_leg = ax1.get_legend_handles_labels()
    fig.legend(handles, labels_leg, frameon=False, fontsize=SZ_ANNOT, ncol=len(names),
               loc="upper center", bbox_to_anchor=(0.5, 0.98))

    fs.save(fig,"fig_benchmark_bars_density",figdir=FIGDIR); plt.close(fig)

    # ================================================================
    # FIG 2B: SIGN AGREEMENT BARS (%) ACROSS SCALES & DENSITIES
    # ================================================================
    fig=plt.figure(figsize=(11.0,5.2))

    # Panel a: Sign Agreement by Spatial Scale
    ax1=fig.add_axes([0.09,0.16,0.38,0.58])
    all_s_a = []
    for j,nm in enumerate(names):
        vals=[df[(df.scale==s)&(df["product"]==nm)]["sign_pct"].values for s in scales]
        vals=[v[0] if len(v) else np.nan for v in vals]
        x_bars = np.arange(len(scales)) + j * W
        ax1.bar(x_bars, vals, W, label=DISP.get(nm,nm), color=COL.get(nm,"#666666"))
        all_s_a.extend([v for v in vals if np.isfinite(v)])

    ax1.set_xticks(np.arange(len(scales))+W*(len(names)-1)/2)
    ax1.set_xticklabels(["Cell","Aquifer","Basin","Basin\n$\\geq$200k km$^2$"],fontsize=SZ_TICK)
    ax1.set_ylabel("Sign Match (%) with wells",fontsize=SZ_LABEL)
    ax1.tick_params(axis="y",labelsize=SZ_TICK)
    ax1.set_title("Directional Sign Match by Scale",fontsize=SZ_TITLE,pad=12)
    ax1.spines[["top","right"]].set_visible(False); fs.panel_letter(ax1,"a")
    if all_s_a: ax1.set_ylim(30, max(100, max(all_s_a) + 8))

    # Panel b: Sign Agreement by Well Density Tier
    ax2=fig.add_axes([0.57,0.16,0.38,0.58])
    all_s_b = []
    for j,nm in enumerate(names):
        vals=[df[(df.scale==f"density_{lab}")&(df["product"]==nm)]["sign_pct"].values for lab in DENSITY_LABELS]
        vals=[v[0] if len(v) else np.nan for v in vals]
        x_bars = np.arange(len(DENSITY_LABELS)) + j * W
        ax2.bar(x_bars, vals, W, label=DISP.get(nm,nm), color=COL.get(nm,"#666666"))
        all_s_b.extend([v for v in vals if np.isfinite(v)])

    ax2.set_xticks(np.arange(len(DENSITY_LABELS))+W*(len(names)-1)/2)
    ax2.set_xticklabels(DENSITY_LABELS,fontsize=SZ_TICK)
    ax2.set_xlabel("Wells per cell",fontsize=SZ_LABEL)
    ax2.set_ylabel("Sign Match (%) with wells",fontsize=SZ_LABEL)
    ax2.tick_params(axis="y",labelsize=SZ_TICK)
    ax2.set_title("Directional Sign Match by Well Density Tier",fontsize=SZ_TITLE,pad=12)
    ax2.spines[["top","right"]].set_visible(False); fs.panel_letter(ax2,"b")
    if all_s_b: ax2.set_ylim(30, max(100, max(all_s_b) + 8))

    handles, labels_leg = ax1.get_legend_handles_labels()
    fig.legend(handles, labels_leg, frameon=False, fontsize=SZ_ANNOT, ncol=len(names),
               loc="upper center", bbox_to_anchor=(0.5, 0.98))

    fs.save(fig,"fig_sign_agreement_bars",figdir=FIGDIR); plt.close(fig)

    # ================================================================
    # FIG 3: Aquifer Scatters
    # ================================================================
    if aqM is not None:
        fig=plt.figure(figsize=(9.6,3.4))
        for k,(nm,c) in enumerate(PRODUCTS):
            ax=fig.add_axes([0.075+k*0.325,0.22,0.27,0.64])
            x=aqM["well"].values; y=aqM[f"{c}_wm"].values; m=np.isfinite(x)&np.isfinite(y)
            L=np.nanpercentile(np.abs(np.concatenate([x[m],y[m]])),96)
            ax.scatter(x[m],y[m],s=12,c=COL.get(nm,"#666666"),alpha=0.55,linewidths=0,edgecolors="none")
            ax.plot([-L,L],[-L,L],"k--",lw=0.7,alpha=0.6)
            ax.set_xlim(-L,L); ax.set_ylim(-L,L); r=np.corrcoef(x[m],y[m])[0,1]
            sm, _ = signm(x[m], y[m])
            ax.set_title(f"{DISP.get(nm,nm)}\n($r$={r:+.2f}, sign={sm*100:.0f}%)",fontsize=SZ_TITLE-1)
            fs.panel_letter(ax,chr(97+k))
            ax.tick_params(labelsize=SZ_TICK)
            ax.set_xlabel("Well trend (cm yr$^{-1}$)",fontsize=SZ_LABEL)
            if k==0: ax.set_ylabel("Product trend (cm yr$^{-1}$)",fontsize=SZ_LABEL)
            ax.spines[["top","right"]].set_visible(False)
        fs.save(fig,"fig_aquifer_scatters",figdir=FIGDIR); plt.close(fig)

    # ================================================================
    # FIG 3B: DIRECT INTER-PRODUCT COMPARISON SCATTERS
    # ================================================================
    if aqM is not None and "wgap" in aqM.columns and "seda" in aqM.columns:
        fig=plt.figure(figsize=(9.6,3.4))
        inter_pairs = [("MINE", "WGAP", "mine", "wgap"),
                       ("MINE", "SEDA", "mine", "seda"),
                       ("WGAP", "SEDA", "wgap", "seda")]
        for k, (p1, p2, c1, c2) in enumerate(inter_pairs):
            ax=fig.add_axes([0.075+k*0.325,0.22,0.27,0.64])
            x=aqM[c2].values; y=aqM[c1].values; m=np.isfinite(x)&np.isfinite(y)
            L=np.nanpercentile(np.abs(np.concatenate([x[m],y[m]])),96)
            ax.scatter(x[m],y[m],s=14,c="#444444",alpha=0.60,linewidths=0,edgecolors="none")
            ax.plot([-L,L],[-L,L],"k--",lw=0.7,alpha=0.6)
            ax.set_xlim(-L,L); ax.set_ylim(-L,L)
            r, _ = pear(x[m], y[m])
            sm, _ = signm(x[m], y[m])
            ax.set_title(f"{DISP.get(p1,p1)} vs {DISP.get(p2,p2)}\n($r$={r:+.2f}, match={sm*100:.0f}%)",fontsize=SZ_TITLE-1)
            fs.panel_letter(ax,chr(97+k))
            ax.tick_params(labelsize=SZ_TICK)
            ax.set_xlabel(f"{DISP.get(p2,p2)} (cm yr$^{-1}$)",fontsize=SZ_LABEL)
            ax.set_ylabel(f"{DISP.get(p1,p1)} (cm yr$^{-1}$)",fontsize=SZ_LABEL)
            ax.spines[["top","right"]].set_visible(False)
        fs.save(fig,"fig_interproduct_comparison",figdir=FIGDIR); plt.close(fig)

    # ================================================================
    # FIG 4A: Well Density Map (Standalone)
    # ================================================================
    fig=plt.figure(figsize=(7.4,4.2))
    dens=np.full((NLAT,NLON),np.nan)
    jla=np.round((ys["latitude"].values+89.75)/0.5).astype(int); jlo=np.round((ys["longitude"].values+179.75)/0.5).astype(int)
    import collections; cnt=collections.Counter(zip(jla,jlo))
    for (a,b),v in cnt.items():
        if 0<=a<NLAT and 0<=b<NLON: dens[a,b]=v

    ax=fs.global_map_axis(fig,(0.05,0.20,0.90,0.70),projection="robinson",basins=False)
    m=fs.pcolor_global_smooth(ax,np.log10(dens),NLAT,NLON,fs.CMAP_SEQ_Y,0,2.5,upsample=1)
    ax.set_title("GROW well density (log$_{10}$ wells per 0.5$^\\circ$ cell)",fontsize=SZ_TITLE)
    fs.panel_letter(ax,"a")
    fs.hcolorbar(fig,m,(0.28,0.075,0.44,0.020),"log$_{10}$ wells per cell")
    fs.save(fig,"fig_wells_density_map",figdir=FIGDIR); plt.close(fig)

    # ================================================================
    # FIG 4B: Separate Product Residual Maps (|Product - Well|)
    # ================================================================
    jc_la=(CW["jc"].values//10000).astype(int); jc_lo=(CW["jc"].values%10000).astype(int)
    for nm, c in PRODUCTS:
        fig=plt.figure(figsize=(7.4,4.2))
        resid=np.full((NLAT,NLON),np.nan)
        rv=np.abs(CW[c].values-CW["well"].values)
        for a,b,v in zip(jc_la,jc_lo,rv):
            if 0<=a<NLAT and 0<=b<NLON: resid[a,b]=v
        ax=fs.global_map_axis(fig,(0.05,0.20,0.90,0.70),projection="robinson",basins=False)
        m=fs.pcolor_global_smooth(ax,resid,NLAT,NLON,fs.CMAP_MAG,0,3,upsample=1)
        ax.set_title(f"|{DISP.get(nm,nm)} $-$ well| trend residual",fontsize=SZ_TITLE)
        fs.panel_letter(ax, "a" if nm=="MINE" else ("b" if nm=="WGAP" else "c"))
        fs.hcolorbar(fig,m,(0.28,0.075,0.44,0.020),"|trend residual| (cm yr$^{-1}$)")
        fs.save(fig,f"fig_residual_map_{nm.lower()}",figdir=FIGDIR); plt.close(fig)

    # ================================================================
    # FIG 5: Taylor Diagram
    # ================================================================
    if aqM is not None:
        _fig_taylor(aqM,PRODUCTS,DISP,COL,fs,SZ_TITLE,SZ_LABEL,SZ_TICK,SZ_ANNOT)

    # ================================================================
    # FIG 6: Metrics Heatmap (White BG & High Contrast RMSE)
    # ================================================================
    _fig_metrics_heatmap(df,PRODUCTS,DISP,COL,fs,SZ_TITLE,SZ_LABEL,SZ_TICK,SZ_ANNOT)

    # ================================================================
    # FIG 7: Latitudinal Profile
    # ================================================================
    _fig_zonal_profile(prod,ys,PRODUCTS,DISP,COL,fs,SZ_TITLE,SZ_LABEL,SZ_TICK,SZ_ANNOT)

    # ================================================================
    # FIG 8: Density vs Agreement
    # ================================================================
    _fig_density_vs_agreement(CW,PRODUCTS,DISP,COL,fs,SZ_TITLE,SZ_LABEL,SZ_TICK,SZ_ANNOT)

    # ================================================================
    # FIG 9-11: Hotspots, Inter-product, Temporal
    # ================================================================
    try: _fig_hotspots(CW,PRODUCTS,fs,SZ_TITLE,SZ_LABEL,SZ_TICK,SZ_ANNOT)
    except Exception as e: warn(f"hotspot fig: {e}")
    try: _fig_interproduct(prod,aqM,Bb,PRODUCTS,fs,SZ_TITLE,SZ_LABEL,SZ_TICK,SZ_ANNOT)
    except Exception as e: warn(f"interproduct fig: {e}")
    try: _fig_temporal(temporal_rows,fs,SZ_TITLE,SZ_LABEL,SZ_TICK,SZ_ANNOT)
    except Exception as e: warn(f"temporal fig: {e}")


def _fig_taylor(aqM,PRODUCTS,DISP,COL,fs,SZ_TITLE=16,SZ_LABEL=14,SZ_TICK=13,SZ_ANNOT=12):
    import matplotlib.pyplot as plt, numpy as np
    fig=plt.figure(figsize=(6.8,6.4))
    ax=fig.add_axes([0.10,0.10,0.80,0.78],polar=True)
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    ax.set_thetamin(0); ax.set_thetamax(90)

    rmax = 2.0
    corr_ticks=[0,0.1,0.3,0.5,0.7,0.8,0.9,0.95,0.99,1.0]
    ax.set_thetagrids([np.degrees(np.arccos(c)) for c in corr_ticks],
                       labels=[f"{c:.2f}" for c in corr_ticks],fontsize=SZ_TICK)
    ax.set_rgrids(np.arange(0.5,rmax+0.01,0.5),
                  labels=[f"{v:.1f}" for v in np.arange(0.5,rmax+0.01,0.5)],
                  angle=90,fontsize=SZ_TICK)
    ax.set_rlim(0,rmax)
    ax.text(np.radians(45),rmax*1.15,"Correlation coefficient",
            ha="center",va="center",fontsize=SZ_LABEL,rotation=-45)
    ax.set_ylabel("Normalised standard deviation ($\\sigma_{model}/\\sigma_{well}$)",
                  fontsize=SZ_LABEL,labelpad=34)

    rmse_levels=np.arange(0.5,rmax+0.5,0.5)
    theta_grid=np.linspace(0,np.pi/2,400); r_grid=np.linspace(0,rmax,400)
    Th,Rg=np.meshgrid(theta_grid,r_grid); X=Rg*np.cos(Th); Y=Rg*np.sin(Th)
    D=np.sqrt((X-1.0)**2+Y**2)
    csA=ax.contour(Th,Rg,D,levels=rmse_levels,colors="grey",linestyles="--",linewidths=0.8,alpha=0.85)
    ax.clabel(csA,fmt="%.1f",fontsize=SZ_ANNOT,inline=True)

    ax.plot(0,1.0,marker="*",ms=20,mec="k",mfc="gold",zorder=5)
    ax.annotate("Wells\n(reference)",xy=(0,1.0),xytext=(np.radians(6),1.20),
                fontsize=SZ_ANNOT,ha="left")

    for nm,c in PRODUCTS:
        x=aqM["well"].values; y=aqM[f"{c}_wm"].values; m=np.isfinite(x)&np.isfinite(y)
        if m.sum()<4: continue
        r=np.corrcoef(x[m],y[m])[0,1]
        sd_ratio=np.nanstd(y[m])/np.nanstd(x[m])
        theta=np.arccos(np.clip(r,-1,1))
        ax.plot(theta,sd_ratio,marker="o",ms=12,mec="k",mew=0.9,
                mfc=COL.get(nm,"#666666"),zorder=6,label=DISP.get(nm,nm))

    ax.legend(loc="upper right",bbox_to_anchor=(1.32,1.08),frameon=False,fontsize=SZ_ANNOT)
    ax.set_title("Taylor diagram -- aquifer-scale trends vs independent wells",
                 fontsize=SZ_TITLE,pad=30)
    fs.save(fig,"fig_taylor_diagram",figdir=FIGDIR); plt.close(fig)


def _fig_metrics_heatmap(df,PRODUCTS,DISP,COL,fs,SZ_TITLE=16,SZ_LABEL=14,SZ_TICK=13,SZ_ANNOT=12):
    import matplotlib.pyplot as plt, numpy as np

    scales=["cell","aquifer","basin","basin_ge200k"]
    scale_lab=["Cell","Aquifer","Basin","Basin\n$\\geq$200k km$^2$"]
    names=[n for n,_ in PRODUCTS]

    Rmat=np.full((len(scales),len(names)),np.nan)
    RMSEmat=np.full((len(scales),len(names)),np.nan)
    for i,s in enumerate(scales):
        for j,nm in enumerate(names):
            sub=df[(df.scale==s)&(df["product"]==nm)]
            if len(sub):
                Rmat[i,j]=sub["r"].values[0]; RMSEmat[i,j]=sub["RMSE_cm_yr"].values[0]

    finite_r = Rmat[np.isfinite(Rmat)]
    vlim_r = float(max(0.15, np.nanmax(np.abs(finite_r)) if finite_r.size else 0.5))
    finite_rmse = RMSEmat[np.isfinite(RMSEmat)]
    vmin_rmse = 0.0
    vmax_rmse = float(max(0.2, np.nanmax(finite_rmse) if finite_rmse.size else 1.0))

    def _panel(ax, mat, cmap, vmin, vmax, letter, color_by):
        im=ax.imshow(mat,cmap=cmap,vmin=vmin,vmax=vmax,aspect="auto")
        ax.set_xticks(range(len(names)))
        xt=ax.set_xticklabels([DISP.get(n,n) for n in names],fontsize=SZ_LABEL,fontweight="semibold")
        for tick,nm in zip(xt,names): tick.set_color(COL.get(nm,"#333333"))
        ax.set_yticks(range(len(scales)))
        ax.set_yticklabels(scale_lab if letter=="a" else [], fontsize=SZ_LABEL)
        ax.tick_params(length=0)
        for i in range(len(scales)):
            for j in range(len(names)):
                if np.isfinite(Rmat[i,j]):
                    val = mat[i,j]
                    if color_by == "r":
                        frac = abs(val) / vmax
                    else:
                        frac = 1.0 - (val - vmin) / max(vmax - vmin, 1e-9)
                    txt_col="white" if frac > 0.50 else "black"
                    ax.text(j,i,f"$r$={Rmat[i,j]:+.2f}\nRMSE={RMSEmat[i,j]:.2f}",
                            ha="center",va="center",fontsize=SZ_ANNOT,color=txt_col,fontweight="medium")
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.set_xticks(np.arange(-.5,len(names),1),minor=True)
        ax.set_yticks(np.arange(-.5,len(scales),1),minor=True)
        ax.grid(which="minor",color="white",linewidth=2.4)
        ax.tick_params(which="minor",length=0)
        fs.panel_letter(ax,letter)
        return im

    fig=plt.figure(figsize=(4.0*len(names)+3.0,1.25*len(scales)+2.2))
    fig.patch.set_facecolor("white"); fig.patch.set_alpha(1.0)

    axL=fig.add_axes([0.16,0.22,0.34,0.62]); axL.set_facecolor("white")
    imL=_panel(axL,Rmat,"RdBu_r",-vlim_r,vlim_r,"a","r")
    axL.set_title("Coloured by $r$",fontsize=SZ_TITLE)

    axR=fig.add_axes([0.58,0.22,0.34,0.62]); axR.set_facecolor("white")
    imR=_panel(axR,RMSEmat,"YlGnBu_r",vmin_rmse,vmax_rmse,"b","rmse")
    axR.set_title("Coloured by RMSE (lower = better)",fontsize=SZ_TITLE)

    fig.suptitle("Validation summary vs independent wells",fontsize=SZ_TITLE+1,y=0.965)

    caxL=fig.add_axes([0.16,0.10,0.34,0.035]); caxL.set_facecolor("white")
    cbL=fig.colorbar(imL,cax=caxL,orientation="horizontal")
    cbL.set_label("Pearson $r$",fontsize=SZ_LABEL); cbL.ax.tick_params(labelsize=SZ_TICK)

    caxR=fig.add_axes([0.58,0.10,0.34,0.035]); caxR.set_facecolor("white")
    cbR=fig.colorbar(imR,cax=caxR,orientation="horizontal")
    cbR.set_label("RMSE (cm yr$^{-1}$)",fontsize=SZ_LABEL); cbR.ax.tick_params(labelsize=SZ_TICK)

    out_path=os.path.join(FIGDIR,"fig_metrics_heatmap.png")
    fig.savefig(out_path,dpi=300,transparent=False,facecolor="white",bbox_inches="tight")
    plt.close(fig)


def _fig_zonal_profile(prod,ys,PRODUCTS,DISP,COL,fs,SZ_TITLE=16,SZ_LABEL=14,SZ_TICK=13,SZ_ANNOT=12):
    import matplotlib.pyplot as plt, numpy as np, pandas as pd

    bins=np.arange(-60,85,5); lab=(bins[:-1]+bins[1:])/2
    prod=prod.copy(); prod["latbin"]=pd.cut(prod["clat"],bins=bins,labels=lab)
    zprod=prod.groupby("latbin",observed=True).agg({c:"mean" for _,c in PRODUCTS})

    ys2=ys.copy(); ys2["latbin"]=pd.cut(ys2["latitude"],bins=bins,labels=lab)
    zwell=ys2.groupby("latbin",observed=True)["slope_storage_cm_yr"].mean()
    zwell_n=ys2.groupby("latbin",observed=True)["slope_storage_cm_yr"].count()

    fig=plt.figure(figsize=(8.8,7.4))
    ax=fig.add_axes([0.38,0.10,0.40,0.80])
    handles=[]
    for nm,c in PRODUCTS:
        if c in zprod.columns:
            h,=ax.plot(zprod[c].values,zprod.index.astype(float),lw=2.2,
                       color=COL.get(nm,"#666666"),label=DISP.get(nm,nm))
            handles.append(h)
    h_well,=ax.plot(zwell.values,zwell.index.astype(float),lw=0,marker="o",ms=6,
                    mfc="none",mec="k",mew=1.3,label="GROW wells")
    handles.append(h_well)
    ax.axvline(0,color="k",lw=0.6); ax.set_ylim(-60,80)
    ax.set_ylabel("Latitude ($^\\circ$)",fontsize=SZ_LABEL)
    ax.set_xlabel("Zonal-mean GWSA trend (cm yr$^{-1}$)",fontsize=SZ_LABEL)
    ax.set_title("Latitudinal profile of\nGWSA trend agreement",fontsize=SZ_TITLE)
    ax.tick_params(labelsize=SZ_TICK)
    ax.spines[["top","right"]].set_visible(False); fs.panel_letter(ax,"a")

    labels=[h.get_label() for h in handles]
    fig.legend(handles,labels,loc="center left",bbox_to_anchor=(0.02,0.5),
               frameon=False,fontsize=SZ_ANNOT,title="Product",
               title_fontsize=SZ_LABEL,handlelength=2.2,labelspacing=1.1)

    ax2=fig.add_axes([0.82,0.10,0.15,0.80],sharey=ax)
    ax2.barh(zwell_n.index.astype(float),zwell_n.values,height=4.0,color="#999999")
    ax2.set_xlabel("Wells\nper band",fontsize=SZ_ANNOT)
    ax2.tick_params(labelleft=False,labelsize=SZ_TICK)
    ax2.spines[["top","right"]].set_visible(False); fs.panel_letter(ax2,"b")

    fs.save(fig,"fig_zonal_profile",figdir=FIGDIR); plt.close(fig)


def _fig_density_vs_agreement(CW,PRODUCTS,DISP,COL,fs,SZ_TITLE=16,SZ_LABEL=14,SZ_TICK=13,SZ_ANNOT=12):
    import matplotlib.pyplot as plt, numpy as np, pandas as pd

    CW = CW.copy()
    CW["density_tier"] = pd.cut(CW["n"], bins=DENSITY_BINS, labels=DENSITY_LABELS, right=False)

    fig = plt.figure(figsize=(8.2, 5.0))
    ax = fig.add_axes([0.14, 0.18, 0.80, 0.62])

    W = 0.8 / len(PRODUCTS)
    all_r = []

    for j, (nm, c) in enumerate(PRODUCTS):
        rvals = []
        for lab in DENSITY_LABELS:
            sub = CW[CW["density_tier"] == lab]
            if len(sub) >= MIN_POLY_FOR_R:
                x, y = sub["well"].values, sub[c].values
                m = np.isfinite(x) & np.isfinite(y)
                r_val = np.corrcoef(x[m], y[m])[0, 1] if m.sum() >= MIN_POLY_FOR_R else np.nan
            else:
                r_val = np.nan
            rvals.append(r_val)
            all_r.append(r_val)

        x_bars = np.arange(len(DENSITY_LABELS)) + j * W
        ax.bar(x_bars, rvals, W, label=DISP.get(nm, nm), color=COL.get(nm, "#666666"))

    ax.set_xticks(np.arange(len(DENSITY_LABELS)) + W * (len(PRODUCTS) - 1) / 2)
    ax.set_xticklabels(DENSITY_LABELS, fontsize=SZ_TICK)
    ax.set_xlabel("Wells per cell", fontsize=SZ_LABEL)
    ax.set_ylabel("Pearson $r$ vs wells\n(within density tier)", fontsize=SZ_LABEL)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title("Agreement Strength by Well Density Tier", fontsize=SZ_TITLE, pad=25)
    ax.tick_params(labelsize=SZ_TICK)

    valid_r = [r for r in all_r if np.isfinite(r)]
    min_r = min(valid_r) if valid_r else -0.2
    max_r = max(valid_r) if valid_r else 0.8
    ax.set_ylim(min(0, min_r - 0.12), max(0.5, max_r + 0.15))

    ax.legend(frameon=False, fontsize=SZ_ANNOT, ncol=len(PRODUCTS),
              loc="upper center", bbox_to_anchor=(0.5, 1.15))
    ax.spines[["top", "right"]].set_visible(False)

    fs.save(fig, "fig_density_vs_agreement", figdir=FIGDIR)
    plt.close(fig)


# ================== additional figure helpers ==================

def _fig_hotspots(CW,PRODUCTS,fs,SZ_TITLE=16,SZ_LABEL=14,SZ_TICK=13,SZ_ANNOT=12):
    import matplotlib.pyplot as plt, numpy as np, os
    HOTSPOTS={"N.China Plain":(34,41,113,120),"Central Valley":(35,40.5,-122.5,-118.5),
              "High Plains":(32,43,-103.5,-96),"Ganges-Brahm.":(22,31,75,90),
              "Punjab-Indus":(27,33,70,77)}
    COL={"MINE":"#1f4e79","WGAP":"#c0392b","SEDA":"#2ca02c"}
    names=[n for n,_ in PRODUCTS]
    cw_la=(CW["jc"].values//10000)*0.5-89.75; cw_lo=(CW["jc"].values%10000)*0.5-179.75
    labs=list(HOTSPOTS.keys()); W=0.8/max(len(names),1)
    fig=plt.figure(figsize=(9,3.6)); ax=fig.add_axes([0.08,0.26,0.9,0.62])
    for j,nm in enumerate(names):
        c=dict(PRODUCTS)[nm]; vals=[]
        for _,(la0,la1,lo0,lo1) in HOTSPOTS.items():
            sel=(cw_la>=la0)&(cw_la<=la1)&(cw_lo>=lo0)&(cw_lo<=lo1); sub=CW[sel]
            if len(sub)>=5 and c in sub.columns:
                x=sub["well"].values; y=sub[c].values; m=np.isfinite(x)&np.isfinite(y)
                vals.append(np.corrcoef(x[m],y[m])[0,1] if m.sum()>3 else np.nan)
            else: vals.append(np.nan)
        ax.bar(np.arange(len(labs))+j*W,vals,W,label=nm,color=COL.get(nm,"#888"))
    ax.axhspan(0.89,0.99,color="0.85",alpha=0.6,zorder=0,label="Wang 2025")
    ax.axhline(0,color="k",lw=0.5)
    ax.set_xticks(np.arange(len(labs))+W*(len(names)-1)/2); ax.set_xticklabels(labs,fontsize=SZ_TICK,rotation=15)
    ax.set_ylabel("Pearson r vs wells",fontsize=SZ_LABEL); ax.set_ylim(-0.3,1.05)
    ax.legend(frameon=False,fontsize=SZ_ANNOT,ncol=len(names)+1)
    ax.set_title("Agreement at depletion hotspots (cf. Wang et al. 2025)",fontsize=SZ_TITLE)
    ax.spines[["top","right"]].set_visible(False)
    for e in (".png",".pdf"): fig.savefig(os.path.join(FIGDIR,"fig_hotspots"+e),dpi=600,bbox_inches="tight",facecolor="white")
    plt.close(fig)


def _fig_interproduct(prod,aqM,Bb,PRODUCTS,fs,SZ_TITLE=16,SZ_LABEL=14,SZ_TICK=13,SZ_ANNOT=12):
    import matplotlib.pyplot as plt, numpy as np, os
    def pear(x,y):
        m=np.isfinite(x)&np.isfinite(y)
        return np.corrcoef(x[m],y[m])[0,1] if m.sum()>=8 else np.nan
    pairs=[("MINE","WGAP","mine_wgap_matched","wgap"),
           ("MINE","SEDA","mine_seda_matched","seda"),
           ("WGAP","SEDA","wgap","seda_wgap_matched")]
    have=dict(PRODUCTS); pairs=[p for p in pairs if p[0] in have and p[1] in have]
    if not pairs: return
    scales=["Cell","Aquifer","Basin"]; W=0.8/len(scales)
    COLS={"Cell":"#8c9fb0","Aquifer":"#1f4e79","Basin":"#0b2942"}
    fig=plt.figure(figsize=(8.6,3.8)); ax=fig.add_axes([0.09,0.26,0.88,0.62]); labels=[]
    for i,(p1,p2,c1,c2) in enumerate(pairs):
        labels.append(p1+"\nvs "+p2)
        rc = pear(prod[c1].values,prod[c2].values) if (c1 in prod and c2 in prod) else np.nan
        ra = pear(aqM[c1].values,aqM[c2].values) if (aqM is not None and c1 in aqM and c2 in aqM) else np.nan
        rb = pear(Bb[c1].values,Bb[c2].values) if (Bb is not None and c1 in Bb and c2 in Bb) else np.nan
        for j,(sc,rv) in enumerate(zip(scales,[rc,ra,rb])):
            ax.bar(i+j*W,rv,W,color=COLS[sc],label=sc if i==0 else "")
            if np.isfinite(rv):
                ax.annotate(f"{rv:+.2f}",(i+j*W,rv),fontsize=SZ_ANNOT,ha="center",
                            va="bottom" if rv>=0 else "top",xytext=(0,2),textcoords="offset points")
    ax.axhline(0,color="k",lw=0.5)
    ax.set_xticks(np.arange(len(pairs))+W); ax.set_xticklabels(labels,fontsize=SZ_TICK)
    ax.set_ylabel("Inter-product Pearson r",fontsize=SZ_LABEL); ax.set_ylim(-0.2,1.05)
    ax.legend(frameon=False,fontsize=SZ_ANNOT,ncol=3,title="scale")
    ax.set_title("Inter-product agreement across scales",fontsize=SZ_TITLE)
    ax.spines[["top","right"]].set_visible(False)
    for e in (".png",".pdf"): fig.savefig(os.path.join(FIGDIR,"fig_interproduct"+e),dpi=600,bbox_inches="tight",facecolor="white")
    plt.close(fig)


def _fig_temporal(temporal_rows,fs,SZ_TITLE=16,SZ_LABEL=14,SZ_TICK=13,SZ_ANNOT=12):
    import matplotlib.pyplot as plt, numpy as np, pandas as pd, os
    if not temporal_rows: return
    df=pd.DataFrame(temporal_rows)
    rdf=df[df["type"]=="prod_monthly"]; pdf=df[df["type"]=="phase"]
    fig=plt.figure(figsize=(8.4,3.4))
    ax=fig.add_axes([0.09,0.26,0.4,0.62])
    if len(rdf):
        ax.bar(range(len(rdf)),rdf["median_r"],color="#1f4e79")
        for i,(v,n) in enumerate(zip(rdf["median_r"],rdf["n_basins"])):
            ax.annotate(f"{v:+.2f}\nn={n}",(i,v),fontsize=SZ_ANNOT,ha="center",xytext=(0,3),textcoords="offset points")
        ax.set_xticks(range(len(rdf))); ax.set_xticklabels(rdf["comparison"],rotation=20,fontsize=SZ_TICK)
    ax.set_ylabel("Median basin temporal r",fontsize=SZ_LABEL); ax.set_ylim(0,1.05); ax.axhline(0,color="k",lw=0.5)
    ax.set_title("Monthly temporal agreement",fontsize=SZ_TITLE); ax.spines[["top","right"]].set_visible(False)
    ax=fig.add_axes([0.58,0.26,0.38,0.62])
    if len(pdf):
        ax.bar(range(len(pdf)),pdf["median_phase_months"],color="#2ca02c")
        for i,v in enumerate(pdf["median_phase_months"]):
            ax.annotate(f"{v:.1f}",(i,v),fontsize=SZ_ANNOT,ha="center",xytext=(0,3),textcoords="offset points")
        ax.set_xticks(range(len(pdf))); ax.set_xticklabels(pdf["comparison"],rotation=20,fontsize=SZ_TICK)
    ax.set_ylabel("Median seasonal phase diff (mo)",fontsize=SZ_LABEL)
    ax.set_title("Seasonal phase agreement",fontsize=SZ_TITLE); ax.spines[["top","right"]].set_visible(False)
    for e in (".png",".pdf"): fig.savefig(os.path.join(FIGDIR,"fig_temporal"+e),dpi=600,bbox_inches="tight",facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()