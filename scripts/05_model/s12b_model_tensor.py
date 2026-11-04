#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
PHASE 5.0 (v2) -- MODEL-READY TENSOR + EXACT-CLOSURE DIAGNOSTICS
--------------------------------------------------------------------------------
CHANGES VS v1 (driven by the Phase 5.0a gate passing at 7.6e-06 cm):
  * OSA (other-storage anomaly) is ingested and mapped to cells. The physics
    constraint is now EXACT and MONTHLY:
          d(GWSA_pred + OSA)/dt  ~=  (NetFlux - E)/10        [cm/month]
    replacing the low-frequency approximation used in v1.
  * STEP 8 is rewritten as a water-budget CLOSURE test on reconstructed TWSA,
    reported as Nash-Sutcliffe efficiency at cell and basin level. This is
    directly comparable to the published benchmark of Gou & Soja (2024,
    Nature Water), who report positive NSE over 83% of studied area for their
    downscaled product, 77% for GRACE and 75% for WGHM. Their comparison used
    HydroBasins level 4; this uses level 3, which must be stated when quoting.
  * Loss weighting is defined here, not in 5.1:
        - data term   : OBSERVED months only (is_filled == 0)
        - physics term: ALL months, including GRACE gaps
    The network therefore fills gaps by physics and learned dynamics rather
    than reproducing the Phase 2.0 STL interpolation. STL is retained purely
    as a baseline to compare against.
  * New gate: OSA extremes (GLDAS SWE accumulating without bound in permanent
    snow) must be confined to excluded cells. Leakage into the analysis domain
    would corrupt the physics term one-directionally.
  * P, ET, R are smoothed with the 1/4-1/2-1/4 filter before closure, matching
    the reference method, so the NSE numbers are like-for-like.

OUTPUT
  OUTPUTS\05_model_tensor\model_ready.nc
  OUTPUTS\05_model_tensor\phase5_0_diagnostics.txt
  OUTPUTS\05_model_tensor\fig5_*.png   (600 dpi, Times New Roman)

RUN: CPU only. No GPU, no torch at this stage.
================================================================================
"""

import os, warnings, traceback, datetime as dt
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="Degrees of freedom")

# ==============================================================================
# CONFIG
# ==============================================================================
DATA     = r"D:\THESIS\Arpon_Thesis_Final\DATA"
OUT_ROOT = r"D:\THESIS\Arpon_Thesis_Final\OUTPUTS"
OUT_DIR  = os.path.join(OUT_ROOT, "05_model_tensor")

# consolidated monthly dataset (output of s11_dataset_assembler.py)
CONSOLIDATED_NC =r"D:\THESIS\Arpon_Thesis_Final\OUTPUTS\10_model\data\dataset_full.nc"

OSA_NC    = os.path.join(OUT_DIR, "other_storage_anomaly.nc")   # from Phase 5.0a
MASK_NC   = os.path.join(OUT_ROOT, "00_reliability_mask", "reliability_mask.nc")
BASINS_L3 = os.path.join(DATA, "HydroBasins", "hybas_lev03_global.gpkg")
HYGEO_SHP = os.path.join(DATA, "WHYMAP", "WHYMAP_GWR", "shp", "whymap_gwr__v1_poly.shp")

# per-cell well trend product (output of s07_well_suitability.py)
WELL_CELL_NC = os.path.join(OUT_ROOT, "02_well_trends", "well_trends_by_cell.nc")

USE_LATLON  = False
ANCHOR_FRAC = 0.30
SEED        = 20260730

GRID   = 0.5
CAP_CM = 100.0
OSA_SANE_CM = 200.0      # |OSA| beyond this in-domain indicates SWE runaway

DYN_FEATS = ["NetFlux", "E", "t2m", "ndvi", "sm"]
STA_FEATS = ["elevation", "slope", "porosity", "logk_ferr",
             "aei", "aeigw", "urban_frac_mean"]

HYGEO_UNCONFINED = {1, 2}
HYGEO_LOCAL      = {3}
HYGEO_LOWPERM    = {8}

LINE = "=" * 78
_LOG = []
def _p(s=""):
    print(s); _LOG.append(str(s))
def hdr(t):
    _p(""); _p(LINE); _p(t); _p(LINE)
def ok(t):   _p("  [ OK ]   " + t)
def warn(t): _p("  [WARN]   " + t)
def fail(t): _p("  [FAIL]   " + t)


def need_file(path, what):
    if not os.path.exists(path):
        fail(f"{what} not found:\n           {path}")
        return False
    ok(f"{what}: {os.path.basename(path)}")
    return True


def vif_from_matrix(X, names):
    X = np.asarray(X, dtype="float64")
    n, p = X.shape
    out = {}
    for j in range(p):
        y = X[:, j]
        Z = np.column_stack([np.ones(n), np.delete(X, j, axis=1)])
        try:
            beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
            res = y - Z @ beta
            ss_res = float(np.sum(res ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            if ss_tot <= 0:
                out[names[j]] = np.inf; continue
            r2 = min(max(1.0 - ss_res / ss_tot, 0.0), 1.0 - 1e-12)
            out[names[j]] = 1.0 / (1.0 - r2)
        except Exception:
            out[names[j]] = np.inf
    return out


def per_cell_temporal_vif(dyn, names, max_cells=4000, seed=SEED):
    T, ncell, nf = dyn.shape
    rng = np.random.default_rng(seed)
    idx = rng.choice(ncell, size=min(max_cells, ncell), replace=False)
    acc = {nm: [] for nm in names}
    for c in idx:
        M = dyn[:, c, :]
        m = np.all(np.isfinite(M), axis=1)
        if m.sum() < 3 * nf:
            continue
        Mc = M[m]
        if np.any(np.std(Mc, axis=0) <= 0):
            continue
        v = vif_from_matrix(Mc, names)
        for nm in names:
            if np.isfinite(v[nm]):
                acc[nm].append(v[nm])
    return ({nm: (float(np.mean(a)) if a else np.nan) for nm, a in acc.items()},
            {nm: (float(np.median(a)) if a else np.nan) for nm, a in acc.items()},
            {nm: len(a) for nm, a in acc.items()})


def smooth_121(X):
    """1/4-1/2-1/4 temporal filter along axis 0, matching the reference method."""
    Y = np.full_like(X, np.nan)
    Y[1:-1] = 0.25 * X[:-2] + 0.5 * X[1:-1] + 0.25 * X[2:]
    return Y


def nse(sim, obs):
    m = np.isfinite(sim) & np.isfinite(obs)
    if m.sum() < 12:
        return np.nan
    o = obs[m]; s = sim[m]
    den = np.sum((o - np.mean(o)) ** 2)
    if den <= 0:
        return np.nan
    return float(1.0 - np.sum((s - o) ** 2) / den)


# ==============================================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    _p(LINE)
    _p("PHASE 5.0 (v2) -- MODEL-READY TENSOR + EXACT-CLOSURE DIAGNOSTICS")
    _p(f"run: {dt.datetime.now().isoformat(timespec='seconds')}   seed={SEED}")
    _p(f"USE_LATLON={USE_LATLON}   ANCHOR_FRAC={ANCHOR_FRAC}")
    _p(LINE)

    import xarray as xr

    # ------------------------------------------------------------------ STEP 1
    hdr("STEP 1 -- inputs")
    good = True
    good &= need_file(CONSOLIDATED_NC, "consolidated monthly dataset")
    good &= need_file(OSA_NC, "other-storage anomaly (Phase 5.0a)")
    good &= need_file(MASK_NC, "reliability mask")
    good &= need_file(BASINS_L3, "HydroBasins level 3")
    if not need_file(WELL_CELL_NC, "per-cell GROW well trends"):
        good = False
    if not os.path.exists(HYGEO_SHP):
        warn("WHYMAP HYGEO polygons absent -> aquifer-type gating skipped")
    if not good:
        fail("required inputs missing; fix CONFIG paths and re-run.")
        return

    ds = xr.open_dataset(CONSOLIDATED_NC); ds.load()
    ncell = int(ds.sizes["cell"]); T = int(ds.sizes["time"])
    _p(f"  cells={ncell}  months={T}")
    miss = [v for v in ["P", "E", "R", "t2m", "ndvi", "sm", "gwsa"] if v not in ds]
    if miss:
        fail(f"missing variables: {miss}"); return
    ok("required variables present")

    clat = np.asarray(ds["clat"].values, "float64")
    clon = np.asarray(ds["clon"].values, "float64")

    is_filled = (np.asarray(ds["is_filled"].values).astype(bool)
                 if "is_filled" in ds else np.zeros((T, ncell), bool))
    frac_filled = float(is_filled.mean())
    _p(f"  gap-filled cell-months (Phase 2.0 STL): {100*frac_filled:.2f}%")
    _p(f"  months with ANY filled cell: "
       f"{int(is_filled.any(axis=1).sum())}/{T}")
    _p("  -> data-fidelity loss will use OBSERVED months only; physics loss uses ALL.")

    # ------------------------------------------------------------------ STEP 2
    hdr("STEP 2 -- features + unit audit")
    P = np.asarray(ds["P"].values, "float32")
    E = np.asarray(ds["E"].values, "float32")
    R = np.asarray(ds["R"].values, "float32")
    NETFLUX = (P - R).astype("float32")
    gwsa = np.asarray(ds["gwsa"].values, "float32")

    _p("  UNIT AUDIT:")
    for nm, a, exp in [("P", P, "mm/month"), ("E", E, "mm/month"),
                       ("R", R, "mm/month"), ("NetFlux", NETFLUX, "mm/month"),
                       ("gwsa", gwsa, "cm")]:
        _p(f"     {nm:<8} mean={np.nanmean(a):9.3f} min={np.nanmin(a):9.3f} "
           f"max={np.nanmax(a):9.3f}   expect {exp}")
    _p("  -> mm->cm conversion (/10) applied in all flux-vs-storage comparisons.")

    dyn = np.stack([NETFLUX, E,
                    np.asarray(ds["t2m"].values, "float32"),
                    np.asarray(ds["ndvi"].values, "float32"),
                    np.asarray(ds["sm"].values, "float32")], axis=-1)
    ok(f"dynamic tensor {dyn.shape} ({dyn.nbytes/1e6:.0f} MB)")

    sta_names = list(STA_FEATS)
    sl = []
    for v in sta_names:
        if v not in ds:
            fail(f"static '{v}' missing"); return
        sl.append(np.asarray(ds[v].values, "float32"))
    if USE_LATLON:
        sta_names += ["lat_feat", "lon_feat"]
        sl += [clat.astype("float32"), clon.astype("float32")]
    sta = np.stack(sl, axis=-1)
    ok(f"static tensor {sta.shape} features={sta_names}")

    # ------------------------------------------------------------------ STEP 3
    hdr("STEP 3 -- ingest OSA and map to cells")
    ods = xr.open_dataset(OSA_NC); ods.load(); ods.close()
    LAT = np.arange(-90 + GRID / 2, 90, GRID)
    LON = np.arange(-180 + GRID / 2, 180, GRID)
    jj = np.clip(np.round((clat - LAT[0]) / GRID).astype(int), 0, len(LAT) - 1)
    ii = np.clip(np.round((clon - LON[0]) / GRID).astype(int), 0, len(LON) - 1)

    OSA_g = np.asarray(ods["osa"].values, "float32")           # (T,360,720)
    if OSA_g.shape[0] != T:
        fail(f"OSA has {OSA_g.shape[0]} months, dataset has {T}"); return
    osa = OSA_g[:, jj, ii].astype("float32")                   # (T, ncell)
    del OSA_g
    ok(f"OSA mapped to cells: {osa.shape}  finite={100*np.mean(np.isfinite(osa)):.2f}%")
    _p(f"  OSA (all cells) range: {np.nanmin(osa):+.2f} .. {np.nanmax(osa):+.2f} cm")

    # ------------------------------------------------------------------ STEP 4
    hdr("STEP 4 -- reliability mask")
    mds = xr.open_dataset(MASK_NC); mds.load(); mds.close()
    ice = (np.asarray(mds["permanent_ice"].values, bool)[jj, ii]
           if "permanent_ice" in mds else np.zeros(ncell, bool))
    water = (np.asarray(mds["surface_water"].values).astype(bool)[jj, ii]
             if "surface_water" in mds else np.zeros(ncell, bool))
    ok(f"permanent_ice cells: {int(ice.sum())}")
    ok(f"surface_water cells: {int(water.sum())}")

    valid_frac = np.mean(np.isfinite(gwsa), axis=0)
    keep = (~ice) & (valid_frac > 0.0)
    _p(f"  cells kept (ice excluded): {int(keep.sum())}/{ncell}")

    # --- OSA runaway gate -----------------------------------------------------
    osa_absmax_cell = np.nanmax(np.abs(osa), axis=0)
    leak = keep & np.isfinite(osa_absmax_cell) & (osa_absmax_cell > OSA_SANE_CM)
    _p("")
    _p(f"  OSA RUNAWAY CHECK (threshold |OSA| > {OSA_SANE_CM} cm in-domain):")
    _p(f"     in-domain cells exceeding: {int(leak.sum())}")
    if leak.sum():
        warn("SWE runaway leaking into analysis domain. Worst offenders:")
        wi = np.argsort(-np.where(leak, osa_absmax_cell, -np.inf))[:10]
        for c in wi:
            if not leak[c]:
                break
            _p(f"       cell {c:6d}  lat={clat[c]:7.2f} lon={clon[c]:8.2f}  "
               f"|OSA|max={osa_absmax_cell[c]:9.1f} cm")
        _p("     -> tighten the Phase 0 SWE threshold or add these to the ice mask")
        _p("        BEFORE trusting the physics term in these cells.")
    else:
        ok("no SWE runaway in the analysis domain; extremes confined to excluded cells")

    # ------------------------------------------------------------------ STEP 5
    hdr("STEP 5 -- WHYMAP HYGEO2 aquifer type")
    hygeo_code = np.full(ncell, -1, "int16")
    aq_type = np.zeros(ncell, "int8")
    if os.path.exists(HYGEO_SHP):
        try:
            import geopandas as gpd
            from shapely.geometry import Point
            g = gpd.read_file(HYGEO_SHP)
            g = g.set_crs("EPSG:4326") if g.crs is None else (
                g.to_crs("EPSG:4326") if g.crs.to_epsg() != 4326 else g)
            fld = next((c for c in ["HYGEO2", "hygeo2", "Hygeo2", "HYGEO", "hygeo"]
                        if c in g.columns), None)
            if fld is None:
                warn(f"no HYGEO2 column; columns={list(g.columns)[:15]}")
            else:
                pts = gpd.GeoDataFrame({"cell": np.arange(ncell)},
                                       geometry=[Point(a, b) for a, b in zip(clon, clat)],
                                       crs="EPSG:4326")
                j = gpd.sjoin(pts, g[[fld, "geometry"]], how="left", predicate="within")
                j = j.drop_duplicates(subset="cell").sort_values("cell")
                lead = np.full(ncell, -1, "int16")
                for i, v in enumerate(j[fld].to_numpy()):
                    try:
                        lead[i] = int(str(int(v))[0])
                    except Exception:
                        lead[i] = -1
                hygeo_code = lead
                aq_type = np.where(np.isin(lead, list(HYGEO_UNCONFINED)), 1,
                          np.where(np.isin(lead, list(HYGEO_LOCAL)), 2,
                          np.where(np.isin(lead, list(HYGEO_LOWPERM)), 3, 0))).astype("int8")
                for k, nm in [(1, "unconfined-dominated"), (2, "local/shallow"),
                              (3, "low-permeability"), (0, "unclassified")]:
                    _p(f"     aq_type={k} ({nm:<22}): {int((aq_type==k).sum()):6d} cells")
                ok(f"HYGEO2 attached from '{fld}'")
        except Exception as e:
            warn(f"HYGEO2 join failed: {e}")

    # ------------------------------------------------------------------ STEP 6
    hdr("STEP 6 -- HydroBasins level 3 + GROW wells")
    hyb = np.full(ncell, -1, "int64")
    try:
        import geopandas as gpd
        from shapely.geometry import Point
        gb = gpd.read_file(BASINS_L3)
        gb = gb.set_crs("EPSG:4326") if gb.crs is None else (
            gb.to_crs("EPSG:4326") if gb.crs.to_epsg() != 4326 else gb)
        idf = "HYBAS_ID" if "HYBAS_ID" in gb.columns else gb.columns[0]
        _p(f"  basin polygons: {len(gb)}  id field: {idf}")
        pts = gpd.GeoDataFrame({"cell": np.arange(ncell)},
                               geometry=[Point(a, b) for a, b in zip(clon, clat)],
                               crs="EPSG:4326")
        j = gpd.sjoin(pts, gb[[idf, "geometry"]], how="left", predicate="within")
        j = j.drop_duplicates(subset="cell").sort_values("cell")
        hyb = np.nan_to_num(j[idf].to_numpy(dtype="float64"), nan=-1).astype("int64")
        ok(f"assigned to {len(np.unique(hyb[hyb>0]))} basins; "
           f"unassigned {int((hyb<=0).sum())}")
        miss_c = np.where((hyb <= 0) & keep)[0]
        if miss_c.size:
            cen = gb.geometry.representative_point()
            bx = np.asarray(cen.x); by = np.asarray(cen.y)
            bid = gb[idf].to_numpy(dtype="int64")
            for c in miss_c:
                hyb[c] = bid[int(np.argmin((bx - clon[c])**2 + (by - clat[c])**2))]
            ok(f"nearest-basin fallback for {miss_c.size} coastal cells")
    except Exception as e:
        fail(f"basin join failed: {e}"); return

    n_wells = np.zeros(ncell, "int32")
    well_trend = np.full(ncell, np.nan, "float32")
    try:
        wds = xr.open_dataset(WELL_CELL_NC); wds.load(); wds.close()
        _p(f"  well file variables: {list(wds.data_vars)}")
        cv = next((v for v in wds.data_vars
                   if v.lower() in ("n_wells", "nwells", "well_count", "count")), None)
        tv = next((v for v in wds.data_vars
                   if "trend" in v.lower() or "slope" in v.lower()), None)
        if cv is None or tv is None:
            warn(f"cannot identify count/trend vars in {list(wds.data_vars)}; "
                 "edit STEP 6 to name them explicitly")
        else:
            ac = np.asarray(wds[cv].values); at = np.asarray(wds[tv].values, "float32")
            if ac.ndim == 2:
                n_wells = ac[jj, ii].astype("int32"); well_trend = at[jj, ii].astype("float32")
            elif ac.shape[0] == ncell:
                n_wells = ac.astype("int32"); well_trend = at.astype("float32")
            else:
                warn(f"unexpected well shape {ac.shape}")
            ok(f"cells >=1 well: {int((n_wells>0).sum())}  "
               f">=10: {int((n_wells>=10).sum())}  >=25: {int((n_wells>=25).sum())}")
    except Exception as e:
        warn(f"well ingest failed: {e}")

    # ------------------------------------------------------------------ STEP 7
    hdr("STEP 7 -- VIF under stated conventions")
    sub = np.where(keep)[0]
    _p("  (a) per-cell temporal VIF, final dynamic set")
    mv, mdv, nv = per_cell_temporal_vif(dyn[:, sub, :], DYN_FEATS)
    for nm in DYN_FEATS:
        _p(f"     {nm:<10} mean={mv[nm]:8.2f} median={mdv[nm]:8.2f} (n={nv[nm]})")
    _p("")
    _p("  (a2) same convention with P and R SEPARATE (reproduces fig04b panel a)")
    dpr = np.stack([P[:, sub], R[:, sub], E[:, sub],
                    np.asarray(ds['t2m'].values, 'float32')[:, sub],
                    np.asarray(ds['ndvi'].values, 'float32')[:, sub],
                    np.asarray(ds['sm'].values, 'float32')[:, sub]], axis=-1)
    m2, md2, _ = per_cell_temporal_vif(dpr, ["P", "R", "E", "t2m", "ndvi", "sm"])
    for nm in ["P", "R", "E", "t2m", "ndvi", "sm"]:
        _p(f"     {nm:<10} mean={m2[nm]:8.2f} median={md2[nm]:8.2f}")
    _p("     -> P and R are expected to exceed 10 here (consistent with fig04b).")
    del dpr
    _p("")
    _p("  (b) cross-cell VIF: statics + per-cell dynamic means")
    Xj = np.column_stack([np.nanmean(dyn[:, sub, :], axis=0), sta[sub, :]])
    nj = DYN_FEATS + sta_names
    fin = np.all(np.isfinite(Xj), axis=1)
    vj = vif_from_matrix(Xj[fin], nj)
    for nm in nj:
        _p(f"     {nm:<18} VIF={vj[nm]:8.2f}"
           + ("   <-- EXCEEDS 10" if np.isfinite(vj[nm]) and vj[nm] > 10 else ""))
    hi = [n for n in nj if np.isfinite(vj[n]) and vj[n] > 10]
    (warn(f"joint VIF>10: {hi}") if hi else ok("joint design matrix clean"))

    # ------------------------------------------------------------------ STEP 8
    hdr("STEP 8 -- EXACT WATER-BUDGET CLOSURE ON RECONSTRUCTED TWSA")
    _p("  TWSA_recon = GWSA + OSA  (identity verified in Phase 5.0a to 7.6e-06 cm)")
    _p("  TWSC_storage = d(TWSA_recon)/dt   [cm/month, centred difference]")
    _p("  TWSC_budget  = (NetFlux - E)/10   [cm/month, 1/4-1/2-1/4 smoothed]")
    _p("")

    twsa_r = gwsa + osa                                    # (T, ncell) cm
    dtw = np.full_like(twsa_r, np.nan)
    dtw[1:-1] = (twsa_r[2:] - twsa_r[:-2]) / 2.0
    bud = smooth_121(((NETFLUX - E) / 10.0).astype("float32"))

    obs = ~is_filled
    dtw_o = np.where(obs, dtw, np.nan)
    bud_o = np.where(obs, bud, np.nan)

    ns_all = np.full(ncell, np.nan, "float32")
    ns_obs = np.full(ncell, np.nan, "float32")
    rr_all = np.full(ncell, np.nan, "float32")
    for c in sub:
        ns_all[c] = nse(bud[:, c], dtw[:, c])
        ns_obs[c] = nse(bud_o[:, c], dtw_o[:, c])
        m = np.isfinite(dtw[:, c]) & np.isfinite(bud[:, c])
        if m.sum() >= 24 and np.std(dtw[m, c]) > 0 and np.std(bud[m, c]) > 0:
            rr_all[c] = np.corrcoef(dtw[m, c], bud[m, c])[0, 1]

    def rep(tag, arr):
        a = arr[np.isfinite(arr)]
        if a.size == 0:
            _p(f"     {tag:<34}: none"); return np.nan
        _p(f"     {tag:<34}: median={np.median(a):+.3f}  "
           f"frac>0={np.mean(a>0):.3f}  n={a.size}")
        return float(np.mean(a > 0))

    _p("  CELL-LEVEL:")
    f_all = rep("NSE, all months", ns_all)
    f_obs = rep("NSE, observed months only", ns_obs)
    rep("correlation, all months", rr_all)

    # basin-level closure (comparable in spirit to the reference Fig. 6)
    _p("")
    _p("  BASIN-LEVEL (HydroBasins L3; reference used L4 -- state this when quoting):")
    bas = np.unique(hyb[keep & (hyb > 0)])
    bns = []
    for b in bas:
        m = keep & (hyb == b)
        if m.sum() < 3:
            continue
        s = np.nanmean(dtw[:, m], axis=1)
        u = np.nanmean(bud[:, m], axis=1)
        v = nse(u, s)
        if np.isfinite(v):
            bns.append(v)
    bns = np.asarray(bns, "float32")
    if bns.size:
        _p(f"     basins evaluated                  : {bns.size}")
        _p(f"     median basin NSE                  : {np.median(bns):+.3f}")
        _p(f"     fraction of basins with NSE > 0   : {np.mean(bns>0):.3f}")
        _p("")
        _p("     Reference benchmark (Gou & Soja 2024, level-4 basins, ERA5-Land):")
        _p("       downscaled 0.83 | GRACE 0.77 | WGHM 0.75 (fraction positive NSE)")
        _p("     The pre-model baseline is above; Phase 5.1 is compared against it.")
        _p("     The improvement over this baseline measures the effect of the physics term.")
    else:
        warn("no basin closure computed")

    # ------------------------------------------------------------------ STEP 9
    hdr("STEP 9 -- basin-level 30/70 anchor/validation split")
    _p("  Wells are NOT split randomly (neighbouring wells share a GRACE pixel).")
    rng = np.random.default_rng(SEED)
    cell_ok = keep & (hyb > 0)
    wb = {int(b): int(n_wells[cell_ok & (hyb == b)].sum())
          for b in np.unique(hyb[cell_ok])}
    wbs = np.array([b for b, c in wb.items() if c > 0], dtype="int64")
    _p(f"  basins total={len(wb)}  well-bearing={len(wbs)}")

    anchor = np.array([], dtype="int64")
    if wbs.size >= 4:
        cnts = np.array([wb[int(b)] for b in wbs])
        srt = wbs[np.argsort(-cnts)]
        pick = []
        for s in np.array_split(srt, 4):
            k = max(1, int(round(ANCHOR_FRAC * len(s))))
            pick.append(rng.choice(s, size=min(k, len(s)), replace=False))
        anchor = np.concatenate(pick)
        ok(f"anchor basins: {anchor.size} ({anchor.size/wbs.size:.0%})")
    else:
        warn("too few well-bearing basins to split")

    is_anchor = np.isin(hyb, anchor) & cell_ok
    is_valid = (~np.isin(hyb, anchor)) & cell_ok & (n_wells > 0)
    tot = max(int(n_wells[cell_ok].sum()), 1)
    _p(f"  anchor  : {int((is_anchor & (n_wells>0)).sum())} cells, "
       f"{int(n_wells[is_anchor].sum())} wells "
       f"({100*n_wells[is_anchor].sum()/tot:.0f}%)")
    _p(f"  validate: {int(is_valid.sum())} cells, "
       f"{int(n_wells[is_valid].sum())} wells "
       f"({100*n_wells[is_valid].sum()/tot:.0f}%)   target 30/70")

    # ----------------------------------------------------------------- STEP 10
    hdr("STEP 10 -- standardisation on anchor basins only")
    fitm = is_anchor if is_anchor.sum() > 100 else cell_ok
    if not np.array_equal(fitm, is_anchor):
        warn("anchor set small; standardising on all kept cells (note in thesis)")
    dmu = np.nanmean(dyn[:, fitm, :], axis=(0, 1)).astype("float32")
    dsd = np.nanstd(dyn[:, fitm, :], axis=(0, 1)).astype("float32"); dsd[dsd <= 0] = 1
    smu = np.nanmean(sta[fitm, :], axis=0).astype("float32")
    ssd = np.nanstd(sta[fitm, :], axis=0).astype("float32"); ssd[ssd <= 0] = 1
    for nm, a, b in zip(DYN_FEATS, dmu, dsd):
        _p(f"     dyn {nm:<10} mu={a:12.4f} sd={b:12.4f}")
    for nm, a, b in zip(sta_names, smu, ssd):
        _p(f"     sta {nm:<10} mu={a:12.4f} sd={b:12.4f}")

    # ----------------------------------------------------------------- STEP 11
    hdr("STEP 11 -- SANITY GATES")
    gates = [
        ("Rodell cap respected",
         int(np.nansum(np.abs(gwsa) > CAP_CM + 1e-3)) == 0,
         f"{int(np.nansum(np.abs(gwsa) > CAP_CM + 1e-3))} exceed"),
        ("dynamic finite >99%",
         float(np.mean(np.isfinite(dyn[:, sub, :]))) > 0.99,
         f"{100*np.mean(np.isfinite(dyn[:, sub, :])):.2f}%"),
        ("static finite >99%",
         float(np.mean(np.isfinite(sta[sub, :]))) > 0.99,
         f"{100*np.mean(np.isfinite(sta[sub, :])):.2f}%"),
        ("OSA finite in-domain >99%",
         float(np.mean(np.isfinite(osa[:, sub]))) > 0.99,
         f"{100*np.mean(np.isfinite(osa[:, sub])):.2f}%"),
        ("no SWE runaway in domain", int(leak.sum()) == 0,
         f"{int(leak.sum())} cells > {OSA_SANE_CM} cm"),
        ("no ice cells retained", int((keep & ice).sum()) == 0,
         f"{int((keep & ice).sum())}"),
        ("anchor/validation disjoint", int((is_anchor & is_valid).sum()) == 0,
         f"{int((is_anchor & is_valid).sum())}"),
        ("no basin in both sets",
         len(set(np.unique(hyb[is_anchor]).tolist()) &
             set(np.unique(hyb[is_valid]).tolist())) == 0,
         f"{len(set(np.unique(hyb[is_anchor]).tolist()) & set(np.unique(hyb[is_valid]).tolist()))}"),
        ("baseline basin closure computed", bns.size > 0, f"{bns.size} basins"),
    ]
    for nm, p_, d_ in gates:
        _p(f"     [{'PASS' if p_ else 'FAIL'}]  {nm:<34} {d_}")
    nf = sum(1 for _, p_, _ in gates if not p_)
    warn(f"{nf} gate(s) failed") if nf else ok("all gates passed")

    # ----------------------------------------------------------------- STEP 12
    hdr("STEP 12 -- write model_ready.nc")
    out = xr.Dataset(
        {"dyn": (("time", "cell", "dfeat"), dyn.astype("float32")),
         "sta": (("cell", "sfeat"), sta.astype("float32")),
         "gwsa": (("time", "cell"), gwsa.astype("float32")),
         "osa": (("time", "cell"), osa.astype("float32")),
         "twsa_recon": (("time", "cell"), twsa_r.astype("float32")),
         "budget_cm": (("time", "cell"), bud.astype("float32")),
         "gwsa_sigma": (("time", "cell"),
                        np.asarray(ds["gwsa_sigma"].values, "float32")
                        if "gwsa_sigma" in ds else np.full_like(gwsa, np.nan)),
         "is_filled": (("time", "cell"), is_filled.astype("int8")),
         "w_data": (("time", "cell"), obs.astype("int8")),
         "keep": (("cell",), keep.astype("int8")),
         "surface_water": (("cell",), water.astype("int8")),
         "permanent_ice": (("cell",), ice.astype("int8")),
         "osa_runaway": (("cell",), leak.astype("int8")),
         "hybas_l3": (("cell",), hyb),
         "n_wells": (("cell",), n_wells),
         "well_trend_m_per_yr": (("cell",), well_trend),
         "hygeo_lead": (("cell",), hygeo_code),
         "aquifer_type": (("cell",), aq_type),
         "is_anchor": (("cell",), is_anchor.astype("int8")),
         "is_valid": (("cell",), is_valid.astype("int8")),
         "nse_baseline_all": (("cell",), ns_all),
         "nse_baseline_obs": (("cell",), ns_obs),
         "dyn_mu": (("dfeat",), dmu), "dyn_sd": (("dfeat",), dsd),
         "sta_mu": (("sfeat",), smu), "sta_sd": (("sfeat",), ssd),
         "clat": (("cell",), clat.astype("float32")),
         "clon": (("cell",), clon.astype("float32"))},
        coords={"time": ds["time"].values, "cell": np.arange(ncell),
                "dfeat": DYN_FEATS, "sfeat": sta_names},
        attrs={
            "title": "Phase 5.0 v2 model-ready tensor (Aquifer-to-Atmosphere)",
            "created": dt.datetime.now().isoformat(timespec="seconds"),
            "seed": SEED, "use_latlon": int(USE_LATLON),
            "dyn_feats": ",".join(DYN_FEATS), "sta_feats": ",".join(sta_names),
            "netflux_def": "NetFlux = P - R (mm/month); E separate",
            "identity": "TWSA_recon = GWSA + OSA (verified 7.6e-06 cm in Phase 5.0a)",
            "physics_constraint": ("d(GWSA_pred + OSA)/dt ~= (NetFlux - E)/10 "
                                   "cm/month, soft penalty, ALL months"),
            "data_term": "observed months only (w_data == 1); STL fills excluded",
            "baseline_basin_nse_frac_positive":
                float(np.mean(bns > 0)) if bns.size else np.nan,
            "split": "basin-level HydroBasins L3; wells never split randomly",
            "standardisation": "fit on anchor basins only"})
    enc = {v: {"zlib": True, "complevel": 4} for v in out.data_vars}
    pth = os.path.join(OUT_DIR, "model_ready.nc")
    out.to_netcdf(pth, encoding=enc)
    ok(f"written: {pth} ({os.path.getsize(pth)/1e6:.0f} MB)")

    try:
        make_figures(vj, nj, ns_all, ns_obs, bns, is_anchor, is_valid,
                     cell_ok, clat, clon, n_wells, aq_type, is_filled)
    except Exception as e:
        warn(f"figures failed: {e}")

    with open(os.path.join(OUT_DIR, "phase5_0_diagnostics.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(_LOG))

    _p("")
    _p(LINE)
    _p("Diagnostics reported above:")
    _p("    1. STEP 4  OSA runaway check")
    _p("    2. STEP 7  (a2) P and R VIF")
    _p("    3. STEP 8  basin NSE baseline")
    _p("    4. STEP 9  anchor/validation split, basins disjoint")
    _p("    5. STEP 11 sanity gates")
    _p(LINE)


def make_figures(vj, nj, ns_all, ns_obs, bns, is_anchor, is_valid,
                 cell_ok, clat, clon, n_wells, aq_type, is_filled):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    rcParams["font.family"] = "serif"
    rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    rcParams.update({"font.size": 13, "axes.titlesize": 15, "axes.labelsize": 14})

    f, ax = plt.subplots(figsize=(11, 5))
    vals = [vj[n] if np.isfinite(vj[n]) else 0 for n in nj]
    ax.bar(range(len(vals)), vals,
           color=["#c0392b" if v > 10 else "#2c7fb8" for v in vals],
           edgecolor="k", linewidth=0.4)
    ax.axhline(10, ls="--", color="k", lw=1.2, label="VIF = 10")
    ax.set_xticks(range(len(nj))); ax.set_xticklabels(nj, rotation=45, ha="right")
    ax.set_ylabel("VIF (cross-cell)"); ax.set_title("Joint collinearity")
    ax.legend()
    f.savefig(os.path.join(OUT_DIR, "fig5_1_joint_vif.png"),
              dpi=600, bbox_inches="tight"); plt.close(f)

    f, ax = plt.subplots(figsize=(9, 5))
    for a, lab, c in [(ns_all, "all months", "#e34a33"),
                      (ns_obs, "observed only", "#1f4e79")]:
        v = a[np.isfinite(a)]
        v = v[v > -2]
        if v.size:
            ax.hist(v, bins=70, histtype="step", lw=1.8, color=c, density=True,
                    label=f"{lab} (median {np.median(v):+.2f})")
    ax.axvline(0, color="0.5", lw=0.9)
    ax.set_xlabel("Water-budget closure NSE (cell level)")
    ax.set_ylabel("Density")
    ax.set_title("Pre-model closure baseline (clipped at NSE $>-2$)")
    ax.legend()
    f.savefig(os.path.join(OUT_DIR, "fig5_2_closure_nse.png"),
              dpi=600, bbox_inches="tight"); plt.close(f)

    if bns.size:
        f, ax = plt.subplots(figsize=(9, 5))
        v = bns[bns > -2]
        ax.hist(v, bins=40, color="#2c7fb8", edgecolor="k", linewidth=0.4)
        ax.axvline(0, color="k", ls="--", lw=1.2)
        ax.set_xlabel("Basin-level closure NSE (HydroBasins L3)")
        ax.set_ylabel("Basins")
        ax.set_title(f"Baseline basin closure: {100*np.mean(bns>0):.0f}% positive")
        f.savefig(os.path.join(OUT_DIR, "fig5_5_basin_nse.png"),
                  dpi=600, bbox_inches="tight"); plt.close(f)

    f, ax = plt.subplots(figsize=(14, 6.5))
    m = cell_ok & ~is_anchor & ~is_valid
    ax.scatter(clon[m], clat[m], s=0.6, c="0.82", marker="s", linewidths=0)
    ax.scatter(clon[is_valid], clat[is_valid], s=2.2, c="#1f4e79", marker="s",
               linewidths=0, label="validation basins (wells held out)")
    aw = is_anchor & (n_wells > 0)
    ax.scatter(clon[aw], clat[aw], s=2.2, c="#e34a33", marker="s",
               linewidths=0, label="anchor basins (wells in loss)")
    ax.set_xlim(-180, 180); ax.set_ylim(-60, 85)
    ax.set_xlabel("Longitude ($^\\circ$)"); ax.set_ylabel("Latitude ($^\\circ$)")
    ax.set_title("Basin-level anchor / validation split")
    ax.legend(loc="lower left", markerscale=6, framealpha=0.9)
    f.savefig(os.path.join(OUT_DIR, "fig5_3_split_map.png"),
              dpi=600, bbox_inches="tight"); plt.close(f)

    f, ax = plt.subplots(figsize=(13, 4.5))
    ff = is_filled.mean(axis=1)
    ax.fill_between(np.arange(ff.size), 0, 100 * ff, color="#e34a33", alpha=0.85)
    ax.set_ylabel("Cells gap-filled (%)"); ax.set_xlabel("Month index")
    ax.set_title("Phase 2.0 STL gap-fill coverage (data term excludes these)")
    f.savefig(os.path.join(OUT_DIR, "fig5_6_gap_coverage.png"),
              dpi=600, bbox_inches="tight"); plt.close(f)

    ok("figures written (fig5_1, fig5_2, fig5_3, fig5_5, fig5_6)")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()