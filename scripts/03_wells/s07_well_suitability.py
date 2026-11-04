#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
PHASE 2.3 v3 -- MONITORING-NETWORK SUITABILITY CLASSIFICATION (UPDATED)
--------------------------------------------------------------------------------
Classifies GROW wells by their fitness to validate a satellite-derived
groundwater storage trend, aggregates to 0.5 deg, and quantifies the
observational ceiling on any well-satellite agreement statistic.

TERMINOLOGY
  Median Absolute Deviation (MAD)
      Robust dispersion estimator, MAD = median(|x - median(x)|).
      Scaled by 1.4826 so that it is a consistent estimator of the standard
      deviation under normality (sigma_MAD).
  Spatial representativeness error
      Disagreement among point observations falling inside a single grid cell.
      Estimated here by UNWEIGHTED sigma_MAD across Class A wells within a cell,
      measuring physical spread rather than precision-weighted deviation.
  Significance filtering
      Retaining only estimates whose confidence interval excludes zero. Biases
      the retained sample toward large magnitudes ("winner's curse"). Diagnosed
      and avoided; see STEP 4.
  Inverse-variance weighting
      Combining estimates of differing precision with w = 1/sigma^2. Used for
      cell central estimates (STEP 5 & 6).
================================================================================
"""

import os, sys, warnings, traceback, datetime as dt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIG
# ==============================================================================
OUT_ROOT = r"D:\THESIS\Arpon_Thesis_Final\OUTPUTS"
OUT_DIR = os.path.join(OUT_ROOT, "02_well_trends")
FIG_DIR = os.path.join(OUT_DIR, "figures")
PARQUET = os.path.join(OUT_DIR, "well_trends_v2.parquet")

# directory containing fig_style.py  (REQUIRED for the figures)
FIG_STYLE_DIR = r"D:\THESIS\Arpon_Thesis_Final\Codes\Into the Models"

GRID = 0.5
LAT = np.arange(-90 + GRID / 2, 90, GRID)
LON = np.arange(-180 + GRID / 2, 180, GRID)
LAND_CELLS = 64000

MAD_SCALE = 1.4826  # MAD -> sigma under normality

CONFINEMENT_MODE = "presumed"  # "strict" | "presumed" | "any"

# PATCH A: Specific yield is a drainage parameter of porous granular media; it is not
# well defined in fractured rock or karst, where slope_wl x Sy is meaningless
# regardless of confinement. This is an Sy-APPLICABILITY screen, not a
# confinement determination -- confinement is reported for only 5.6% of wells
# and remains an explicit limitation.
SY_APPLICABLE = {"porous"}  # conservative default
SY_APPLICABLE_WIDE = {"porous", "porous/fractured"}  # sensitivity variant
USE_WIDE_AQUIFER_CLASS = False

A_MIN_OBS = 10
A_FIRST_BY = 2010
A_LAST_FROM = 2015
B_MIN_OBS = 5

W_FLOOR_CM = 0.05
W_CAP_RATIO = 50.0

CLASS_COLOR = {"A": "#08519c", "B": "#6baed6", "C": "#bdbdbd"}
CLASS_DESC = {"A": "anchor + magnitude",
              "B": "sign / pattern only",
              "C": "coverage only"}

LINE = "=" * 78
_LOG = []


def _p(s=""):
    print(s);
    _LOG.append(str(s))


def hdr(t):
    _p("");
    _p(LINE);
    _p(t);
    _p(LINE)


def ok(t):   _p("  [ OK ]   " + t)


def warn(t): _p("  [WARN]   " + t)


def fail(t): _p("  [FAIL]   " + t)


def col(df, name, default=None):
    if name in df.columns:
        return df[name]
    if default is None:
        return None
    return pd.Series([default] * len(df), index=df.index)


def weighted_median(v, w):
    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if m.sum() == 0:
        return np.nan
    v = np.asarray(v)[m];
    w = np.asarray(w)[m]
    o = np.argsort(v);
    v = v[o];
    w = w[o]
    cw = np.cumsum(w)
    if cw[-1] <= 0:
        return float(np.median(v))
    return float(v[np.searchsorted(cw, 0.5 * cw[-1])])


# ==============================================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    _p(LINE)
    _p("PHASE 2.3 v3 -- MONITORING-NETWORK SUITABILITY CLASSIFICATION")
    _p(f"run: {dt.datetime.now().isoformat(timespec='seconds')}")
    _p(f"CONFINEMENT_MODE = {CONFINEMENT_MODE}")
    _p(f"USE_WIDE_AQUIFER_CLASS = {USE_WIDE_AQUIFER_CLASS}")
    _p(LINE)

    # ------------------------------------------------------------------ STEP 1
    hdr("STEP 1 -- load Phase 2.1 well trends")
    if not os.path.exists(PARQUET):
        fail(f"not found: {PARQUET}");
        return
    df = pd.read_parquet(PARQUET)
    _p(f"  wells: {len(df):,}")
    need = ["latitude", "longitude", "slope_storage_cm_yr", "n_obs"]
    if [c for c in need if c not in df.columns]:
        fail(f"missing: {[c for c in need if c not in df.columns]}");
        return
    ok("required columns present")

    # ------------------------------------------------------------------ STEP 2
    hdr("STEP 2 -- confinement and aquifer-class inventory")
    conf_raw = col(df, "confinement", "").astype(str).str.strip().str.lower()
    _p("  confinement value counts:")
    for k, v in conf_raw.value_counts().head(15).items():
        _p(f"     {str(k)[:46]:<48}: {v:>8,}")

    aqc = col(df, "aquifer_type_class", "").astype(str).str.strip().str.lower()
    _p("")
    _p("  aquifer_type_class value counts:")
    for k, v in aqc.value_counts().head(15).items():
        _p(f"     {str(k)[:46]:<48}: {v:>8,}")

    NULLS = {"nan", "none", "", "unknown", "not reported", "na", "<na>"}
    conf_known = ~conf_raw.isin(NULLS)
    is_confined = conf_raw.str.contains("confin") & ~conf_raw.str.contains("uncon")
    unconf_explicit = conf_known & ~is_confined

    # PATCH B: Sy-applicability screen based on flow medium
    _ok_classes = SY_APPLICABLE_WIDE if USE_WIDE_AQUIFER_CLASS else SY_APPLICABLE
    phreatic = aqc.isin(_ok_classes)
    _p("")
    _p(f"  Sy-applicable aquifer classes: {sorted(_ok_classes)}")

    _p(f"  confinement reported              : {int(conf_known.sum()):>8,} "
       f"({100 * conf_known.mean():.1f}%)")
    _p(f"  explicitly confined               : {int(is_confined.sum()):>8,}")
    _p(f"  explicitly unconfined             : {int(unconf_explicit.sum()):>8,}")
    _p(f"  unreported + Sy-applicable class  : {int((~conf_known & phreatic).sum()):>8,}")

    if CONFINEMENT_MODE == "strict":
        unconf_ok = unconf_explicit
    elif CONFINEMENT_MODE == "presumed":
        unconf_ok = unconf_explicit | (~conf_known & phreatic)
    else:
        unconf_ok = ~is_confined
    _p(f"  -> mode '{CONFINEMENT_MODE}' admits {int(unconf_ok.sum()):,} wells")

    # ------------------------------------------------------------------ STEP 3
    hdr("STEP 3 -- classification funnel (pre-treatment screens only)")
    n0 = len(df)
    n_obs = col(df, "n_obs").astype(float)
    yr_first = col(df, "yr_first", np.nan).astype(float)
    yr_last = col(df, "yr_last", np.nan).astype(float)
    qc_all = col(df, "qc_all", True).astype(bool)
    sy_imp = col(df, "sy_imputed", True).astype(bool)
    strend = col(df, "slope_storage_cm_yr").astype(float)
    ciw = col(df, "ci_width_cm_yr", np.nan).astype(float)
    signif = col(df, "trend_resolved", False).astype(bool)  # 95% CI excludes 0

    has_trend = np.isfinite(strend)
    has_xy = (np.isfinite(col(df, "latitude").astype(float)) &
              np.isfinite(col(df, "longitude").astype(float)))

    def line(tag, mk):
        _p(f"     {tag:<50}: {int(mk.sum()):>8,}  ({100 * mk.sum() / max(n0, 1):5.1f}%)")

    m = has_trend & has_xy;
    line("wells with slope and coordinates", m)
    m = m & (n_obs >= A_MIN_OBS);
    line(f"+ >= {A_MIN_OBS} annual values", m)
    m = m & (yr_first <= A_FIRST_BY);
    line(f"+ record begins by {A_FIRST_BY}", m)
    m = m & (yr_last >= A_LAST_FROM);
    line(f"+ record extends to {A_LAST_FROM}+", m)
    m = m & qc_all;
    line("+ passes GROW quality control", m)
    m = m & (~sy_imp);
    line("+ measured specific yield", m)
    m = m & unconf_ok;
    line(f"+ unconfined/Sy-applicable ({CONFINEMENT_MODE})", m)
    cls_a = m.copy()
    _p("")
    _p("  Statistical significance of the trend is NOT a screen. Filtering on it")
    _p("  biases the retained sample toward large magnitudes (STEP 4).")

    cls_b = has_trend & has_xy & (n_obs >= B_MIN_OBS) & (~cls_a)
    cls_c = has_trend & has_xy & (~cls_a) & (~cls_b)

    _p("")
    _p("  CLASS TOTALS:")
    for c, mk in [("A", cls_a), ("B", cls_b), ("C", cls_c)]:
        _p(f"     Class {c} ({CLASS_DESC[c]:<20}): {int(mk.sum()):>8,}  "
           f"({100 * mk.sum() / max(n0, 1):5.1f}%)")
    df["wclass"] = np.where(cls_a, "A", np.where(cls_b, "B", "C"))

    # ------------------------------------------------------------------ STEP 4
    hdr("STEP 4 -- significance-filtering bias (diagnostic, not applied)")
    ia = df.index[cls_a]
    bias = np.nan
    if len(ia):
        ss = strend.loc[ia][signif.loc[ia]];
        ss = ss[np.isfinite(ss)]
        sn = strend.loc[ia][~signif.loc[ia]];
        sn = sn[np.isfinite(sn)]
        if ss.size and sn.size:
            _p(f"     significant (95% CI excludes 0) n={ss.size:>7,}  "
               f"median={ss.median():+.4f} cm/yr  neg={100 * (ss < 0).mean():.1f}%")
            _p(f"     not significant                 n={sn.size:>7,}  "
               f"median={sn.median():+.4f} cm/yr  neg={100 * (sn < 0).mean():.1f}%")
            bias = float(ss.median() - sn.median())
            _p(f"     median shift induced by filtering: {bias:+.4f} cm/yr")
            _p("")
            _p("  This is the quantitative justification for inverse-variance")
            _p("      weighting in place of significance filtering. Report it.")
        aa = strend.loc[ia];
        aa = aa[np.isfinite(aa)]
        _p(f"     Class A, unfiltered (used downstream): median="
           f"{aa.median():+.4f} cm/yr  neg={100 * (aa < 0).mean():.1f}%")

    _p("")
    _p("  class medians (large separation would indicate outcome-dependent screens):")
    for c in ["A", "B", "C"]:
        s = df.loc[df["wclass"] == c, "slope_storage_cm_yr"].astype(float)
        s = s[np.isfinite(s)]
        if s.size:
            _p(f"     Class {c}: n={s.size:>7,}  median={s.median():+.4f}  "
               f"p10={s.quantile(.10):+.4f}  p90={s.quantile(.90):+.4f}  "
               f"neg={100 * (s < 0).mean():.1f}%")

    # ------------------------------------------------------------------ STEP 5
    hdr("STEP 5 -- inverse-variance weights (Hedges & Olkin 1985)")
    cw = ciw.to_numpy().astype(float)
    sigma = np.where(np.isfinite(cw), np.maximum(cw, W_FLOOR_CM) / 3.92, np.nan)
    w = 1.0 / (sigma ** 2)
    fw = np.isfinite(w)
    if fw.sum() == 0:
        warn("no finite CI widths; equal weights used")
        w = np.ones(len(df))
    else:
        med = np.nanmedian(w[fw])
        w = np.where(fw, w, med)
        w = np.clip(w, med / W_CAP_RATIO, med * W_CAP_RATIO)
    _p("  sigma approximated as CI_width / 3.92 (95% normal interval)")
    _p(f"  w = 1/sigma^2, clipped to +/- {W_CAP_RATIO:.0f}x median")
    _p(f"  weight range: {np.nanmin(w):.4g} .. {np.nanmax(w):.4g}")

    # ------------------------------------------------------------------ STEP 6
    hdr("STEP 6 -- aggregation to 0.5 degree cells")
    lat = col(df, "latitude").astype(float).to_numpy()
    lon = col(df, "longitude").astype(float).to_numpy()
    jj = np.floor((lat + 90.0) / GRID).astype(int)
    ii = np.floor((lon + 180.0) / GRID).astype(int)
    valid = (jj >= 0) & (jj < len(LAT)) & (ii >= 0) & (ii < len(LON))

    st = strend.to_numpy()
    wl = col(df, "slope_wl_m_yr", np.nan).astype(float).to_numpy()
    wc = df["wclass"].to_numpy()
    sg = signif.to_numpy()

    ny, nx = len(LAT), len(LON)
    n_a = np.zeros((ny, nx), "int32")
    n_b = np.zeros((ny, nx), "int32")
    n_all = np.zeros((ny, nx), "int32")
    t_a = np.full((ny, nx), np.nan, "float32")
    t_b = np.full((ny, nx), np.nan, "float32")
    rep_err = np.full((ny, nx), np.nan, "float32")  # sigma_MAD (UNWEIGHTED)
    wdom = np.full((ny, nx), np.nan, "float32")  # weight dominance diagnostic
    wl_a = np.full((ny, nx), np.nan, "float32")
    sig_frac = np.full((ny, nx), np.nan, "float32")

    key = np.where(valid, jj.astype(np.int64) * nx + ii.astype(np.int64), -1)
    order = np.argsort(key, kind="stable")
    ks = key[order]
    uk = np.unique(ks[ks >= 0])
    a_i = np.searchsorted(ks, uk, side="left")
    b_i = np.searchsorted(ks, uk, side="right")

    for k, a, b in zip(uk, a_i, b_i):
        idx = order[a:b]
        j_, i_ = int(k // nx), int(k % nx)
        n_all[j_, i_] = idx.size
        ai = idx[wc[idx] == "A"]
        bi = idx[wc[idx] == "B"]
        n_a[j_, i_] = ai.size
        n_b[j_, i_] = bi.size
        if ai.size:
            c = weighted_median(st[ai], w[ai])
            t_a[j_, i_] = c
            wl_a[j_, i_] = weighted_median(wl[ai], w[ai])
            sig_frac[j_, i_] = float(np.mean(sg[ai]))

            # PATCH C: representativeness error is UNWEIGHTED: it measures physical
            # disagreement among wells, not precision-weighted disagreement.
            _v = st[ai][np.isfinite(st[ai])]
            if _v.size >= 3:
                rep_err[j_, i_] = MAD_SCALE * np.median(np.abs(_v - np.median(_v)))
            # weight concentration diagnostic
            _w = w[ai]
            wdom[j_, i_] = float(np.max(_w) / np.sum(_w)) if np.sum(_w) > 0 else np.nan

        if bi.size:
            t_b[j_, i_] = weighted_median(st[bi], w[bi])

    _p("  occupied cells by Class A well count:")
    for thr in (1, 3, 5, 10, 25, 50):
        _p(f"     >= {thr:>2} wells: {int((n_a >= thr).sum()):>6,} cells   "
           f"({100 * (n_a >= thr).sum() / LAND_CELLS:6.3f}% of land)")
    _p(f"  cells with >=1 Class B  : {int((n_b > 0).sum()):>6,}")
    _p(f"  cells with >=1 any well : {int((n_all > 0).sum()):>6,}")

    # ------------------------------------------------------------------ STEP 7
    hdr("STEP 7 -- spatial representativeness error")
    _p("  sigma_MAD = 1.4826 x UNWEIGHTED MAD of Class A well trends within a cell.")
    _p("  Interpreted as the point-to-pixel representativeness error: the spread")
    _p("  a satellite cell average must be reconciled against.")
    m3 = n_a >= 3
    ratio_med = np.nan
    if m3.sum():
        r = rep_err[m3]
        _p(f"  cells with >=3 Class A wells: {int(m3.sum()):,}")
        r_valid = r[np.isfinite(r)]
        if r_valid.size:
            _p(f"  unweighted sigma_MAD (cm/yr): p25={np.percentile(r_valid, 25):.4f}  "
               f"median={np.median(r_valid):.4f}  p75={np.percentile(r_valid, 75):.4f}")

        # PATCH D: Per-cell ratio calculation
        am = np.abs(t_a[m3])
        good = np.isfinite(r) & np.isfinite(am) & (am > 1e-6)
        if good.sum():
            pc = r[good] / am[good]
            ratio_med = float(np.median(pc))
            _p(f"  per-cell sigma_MAD/|trend|: p25={np.percentile(pc, 25):.2f}  "
               f"median={ratio_med:.2f}  p75={np.percentile(pc, 75):.2f}")
            _p(f"  cells where sigma_MAD >= |trend|: "
               f"{100 * np.mean(pc >= 1.0):.1f}%")
            _p("")
            if ratio_med >= 0.7:
                _p("  Within-cell spread is comparable to the signal. Any")
                _p("      well-satellite correlation is then bounded by the")
                _p("      network itself, not by model skill. This is the")
                _p("      strongest defence of a moderate correlation.")
            else:
                _p("  Within-cell spread is well below the signal; the")
                _p("      network can in principle resolve these trends.")
    else:
        warn("no cells with >=3 Class A wells")

    # ------------------------------------------------------------------ STEP 8
    hdr("STEP 8 -- write")
    import xarray as xr
    out = xr.Dataset(
        {"n_wells": (("lat", "lon"), n_a),
         "well_trend": (("lat", "lon"), t_a),
         "n_wells_A": (("lat", "lon"), n_a),
         "n_wells_B": (("lat", "lon"), n_b),
         "n_wells_all": (("lat", "lon"), n_all),
         "trend_A": (("lat", "lon"), t_a),
         "trend_B": (("lat", "lon"), t_b),
         "repr_error": (("lat", "lon"), rep_err),
         "weight_dominance": (("lat", "lon"), wdom),
         "wl_trend_A": (("lat", "lon"), wl_a),
         "signif_frac": (("lat", "lon"), sig_frac)},
        coords={"lat": LAT, "lon": LON},
        attrs={
            "title": "GROW well trends by 0.5 deg cell, suitability-classified",
            "created": dt.datetime.now().isoformat(timespec="seconds"),
            "source": os.path.basename(PARQUET),
            "units_trend": "cm/yr storage (water-level trend x Sy x 100)",
            "aggregation": "inverse-variance weighted median (Hedges & Olkin 1985)",
            "repr_error_def": ("spatial representativeness error, "
                               "1.4826 x unweighted MAD (Rousseeuw & Croux 1993)"),
            "selection_note": ("classes use pre-treatment screens only; trend "
                               "significance is a weight, never a filter"),
            "confinement_mode": CONFINEMENT_MODE,
            "use_wide_aquifer_class": str(USE_WIDE_AQUIFER_CLASS),
            "class_A_screens": (f"n_obs>={A_MIN_OBS}, first<={A_FIRST_BY}, "
                                f"last>={A_LAST_FROM}, qc_all, measured Sy, "
                                f"Sy_applicable_aquifer({sorted(_ok_classes)})"),
            "n_class_A": int(cls_a.sum()), "n_class_B": int(cls_b.sum()),
            "n_class_C": int(cls_c.sum()),
            "significance_filter_bias_cm_yr": float(bias) if np.isfinite(bias) else np.nan,
            "repr_error_to_signal_ratio": float(ratio_med) if np.isfinite(ratio_med) else np.nan})
    enc = {v: {"zlib": True, "complevel": 4} for v in out.data_vars}
    p = os.path.join(OUT_DIR, "well_trends_by_cell.nc")
    out.to_netcdf(p, encoding=enc)
    ok(f"written: {p}")
    df.to_parquet(os.path.join(OUT_DIR, "well_trends_v2_tiered.parquet"), index=False)
    ok("written: well_trends_v2_tiered.parquet")

    try:
        figures(df, lat, lon, n_a, t_a, rep_err, n_all,
                cls_a, cls_b, cls_c, strend, signif, bias)
    except Exception as e:
        warn(f"figures failed: {e}")
        traceback.print_exc()

    with open(os.path.join(OUT_DIR, "phase2_3_diagnostics.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(_LOG))

    _p("")
    _p(LINE)
    _p("Diagnostics reported above:")
    _p("    1. STEP 2 -- Class A count under the selected aquifer-class setting")
    _p("    2. STEP 6 -- weight_dominance written to well_trends_by_cell.nc")
    _p("    3. STEP 7 -- unweighted sigma_MAD quantiles")
    _p("    4. STEP 7 -- per-cell sigma_MAD / |trend| ratio quantiles")
    _p(LINE)


# ==============================================================================
# FIGURES -- cartopy Robinson, HydroBASINS background, via shared fig_style
# ==============================================================================
def figures(df, lat, lon, n_a, t_a, rep_err, n_all,
            cls_a, cls_b, cls_c, strend, signif, bias):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    sys.path.insert(0, FIG_STYLE_DIR)
    try:
        import fig_style as fs
    except Exception as e:
        fail(f"fig_style.py not importable from {FIG_STYLE_DIR}: {e}")
        fail("figures skipped -- set FIG_STYLE_DIR and re-run.")
        return
    import cartopy.crs as ccrs
    fs.apply()

    PC = ccrs.PlateCarree()

    def cell_centres(arr):
        j, i = np.where(np.isfinite(arr) if arr.dtype.kind == "f" else arr > 0)
        return LON[i], LAT[j], arr[j, i]

    def save(fig, nm):
        fs.save(fig, nm, figdir=FIG_DIR)

    # ---------------------------------------------------------------- FIGURE 1
    # Four-panel network overview
    fig = plt.figure(figsize=(7.4, 7.6))

    # (a) well locations by class
    ax = fs.global_map_axis(fig, (0.045, 0.700, 0.905, 0.265),
                            projection="robinson", basins=True)
    for c in ["C", "B", "A"]:
        m = (df["wclass"] == c).to_numpy()
        if m.sum():
            ax.scatter(lon[m], lat[m], s=(0.7 if c == "C" else 1.6),
                       c=CLASS_COLOR[c], marker="o", linewidths=0,
                       alpha=(0.35 if c == "C" else 0.85),
                       transform=PC, zorder=3 + "CBA".index(c),
                       label=f"Class {c} ({CLASS_DESC[c]}), n={int(m.sum()):,}")
    ax.set_title("Groundwater monitoring wells by validation suitability",
                 fontsize=fs.SZ_TITLE)
    fs.panel_letter(ax, "a")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, -0.30), ncol=3,
              frameon=False, fontsize=fs.SZ_ANNOT, markerscale=4,
              handletextpad=0.4, columnspacing=1.0)

    # (b) Class A density
    ax = fs.global_map_axis(fig, (0.045, 0.395, 0.435, 0.235),
                            projection="robinson", basins=True)
    x, y, v = cell_centres(n_a)
    if v.size:
        sc = ax.scatter(x, y, c=v, s=2.5 + 9.0 * np.log10(np.maximum(v, 1) + 1),
                        cmap="YlGnBu", norm=mcolors.LogNorm(vmin=1, vmax=max(2, v.max())),
                        marker="o", linewidths=0.15, edgecolors="#333333",
                        transform=PC, zorder=5)
        fs.hcolorbar(fig, sc, (0.075, 0.362, 0.375, 0.011),
                     "Class A wells per 0.5$^\\circ$ cell")
    ax.set_title("Class A network density", fontsize=fs.SZ_TITLE)
    fs.panel_letter(ax, "b")

    # (c) Class A cell trend
    ax = fs.global_map_axis(fig, (0.520, 0.395, 0.435, 0.235),
                            projection="robinson", basins=True)
    x, y, v = cell_centres(t_a)
    if v.size:
        lim = float(np.nanpercentile(np.abs(v), 95)) or 1.0
        sc = ax.scatter(x, y, c=v, s=6.5, cmap="RdBu", vmin=-lim, vmax=lim,
                        marker="o", linewidths=0.15, edgecolors="#333333",
                        transform=PC, zorder=5)
        fs.hcolorbar(fig, sc, (0.550, 0.362, 0.375, 0.011),
                     "Storage trend (cm yr$^{-1}$)")
    ax.set_title("Class A cell storage trend", fontsize=fs.SZ_TITLE)
    fs.panel_letter(ax, "c")

    # (d) representativeness error
    ax = fs.global_map_axis(fig, (0.045, 0.070, 0.435, 0.235),
                            projection="robinson", basins=True)
    x, y, v = cell_centres(rep_err)
    if v.size:
        vmax = float(np.nanpercentile(v, 95)) or 1.0
        sc = ax.scatter(x, y, c=v, s=6.5, cmap="magma_r", vmin=0, vmax=vmax,
                        marker="o", linewidths=0.15, edgecolors="#333333",
                        transform=PC, zorder=5)
        fs.hcolorbar(fig, sc, (0.075, 0.037, 0.375, 0.011),
                     "$\\sigma_{\\mathrm{MAD}}$ (cm yr$^{-1}$)")
    ax.set_title("Spatial representativeness error", fontsize=fs.SZ_TITLE)
    fs.panel_letter(ax, "d")

    # (e) class counts
    ax = fig.add_axes((0.575, 0.085, 0.360, 0.205))
    counts = [int(cls_a.sum()), int(cls_b.sum()), int(cls_c.sum())]
    bars = ax.barh(["C", "B", "A"], counts[::-1],
                   color=[CLASS_COLOR[c] for c in ["C", "B", "A"]],
                   edgecolor="#333333", linewidth=0.5, height=0.62)
    tot = max(sum(counts), 1)
    for r, v in zip(bars, counts[::-1]):
        ax.text(v + tot * 0.015, r.get_y() + r.get_height() / 2,
                f"{v:,} ({100 * v / tot:.1f}%)", va="center", fontsize=fs.SZ_ANNOT)
    ax.set_xlim(0, tot * 1.30)
    ax.set_xlabel("Wells", fontsize=fs.SZ_LABEL)
    ax.set_ylabel("Class", fontsize=fs.SZ_LABEL)
    ax.spines[["top", "right"]].set_visible(False)
    fs.panel_letter(ax, "e")
    save(fig, "fig2_3_1_network_suitability")

    # ---------------------------------------------------------------- FIGURE 2
    # Selection bias + class distributions
    fig = plt.figure(figsize=(7.4, 3.1))
    bins = np.linspace(-8, 6, 100)

    ax = fig.add_axes((0.075, 0.185, 0.395, 0.700))
    ia = df.index[cls_a]
    ss = strend.loc[ia][signif.loc[ia]];
    ss = ss[np.isfinite(ss)]
    sn = strend.loc[ia][~signif.loc[ia]];
    sn = sn[np.isfinite(sn)]
    if ss.size:
        ax.hist(ss, bins=bins, histtype="stepfilled", lw=1.1, density=True,
                color="#c0392b", alpha=0.22)
        ax.hist(ss, bins=bins, histtype="step", lw=1.3, density=True,
                color="#c0392b", label=f"95% CI excludes 0 (n={ss.size:,})")
    if sn.size:
        ax.hist(sn, bins=bins, histtype="stepfilled", lw=1.1, density=True,
                color="#1f4e79", alpha=0.22)
        ax.hist(sn, bins=bins, histtype="step", lw=1.3, density=True,
                color="#1f4e79", label=f"not significant (n={sn.size:,})")
    ax.axvline(0, color="#555555", lw=0.7)
    ax.set_xlabel("Storage trend (cm yr$^{-1}$)", fontsize=fs.SZ_LABEL)
    ax.set_ylabel("Density", fontsize=fs.SZ_LABEL)
    ax.set_title("Significance filtering shifts the sample", fontsize=fs.SZ_TITLE)
    ax.legend(frameon=False, fontsize=fs.SZ_ANNOT, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    if np.isfinite(bias):
        ax.text(0.97, 0.55, f"median shift\n{bias:+.3f} cm yr$^{{-1}}$",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=fs.SZ_ANNOT, color="#333333")
    fs.panel_letter(ax, "a")

    ax = fig.add_axes((0.585, 0.185, 0.395, 0.700))
    for c in ["C", "B", "A"]:
        s = df.loc[df["wclass"] == c, "slope_storage_cm_yr"].astype(float)
        s = s[np.isfinite(s)]
        if s.size:
            ax.hist(s, bins=bins, histtype="step", lw=1.4, density=True,
                    color=CLASS_COLOR[c],
                    label=f"Class {c} (med {s.median():+.2f})")
    ax.axvline(0, color="#555555", lw=0.7)
    ax.set_xlabel("Storage trend (cm yr$^{-1}$)", fontsize=fs.SZ_LABEL)
    ax.set_ylabel("Density", fontsize=fs.SZ_LABEL)
    ax.set_title("Trend distribution by suitability class", fontsize=fs.SZ_TITLE)
    ax.legend(frameon=False, fontsize=fs.SZ_ANNOT, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fs.panel_letter(ax, "b")
    save(fig, "fig2_3_2_selection_bias")

    # ---------------------------------------------------------------- FIGURE 3
    # Representativeness ceiling
    fig = plt.figure(figsize=(7.4, 3.1))

    ax = fig.add_axes((0.075, 0.185, 0.395, 0.700))
    m = (n_a >= 3) & np.isfinite(rep_err) & np.isfinite(t_a)
    if m.sum():
        xv = np.abs(t_a[m]);
        yv = rep_err[m]
        lim = float(np.nanpercentile(np.concatenate([xv, yv]), 97)) or 1.0
        ax.scatter(xv, yv, s=8, alpha=0.45, color="#1f4e79", linewidths=0)
        ax.plot([0, lim], [0, lim], ls="--", lw=1.0, color="#c0392b")
        ax.text(lim * 0.97, lim * 0.90,
                "$\\sigma_{\\mathrm{MAD}} = |$trend$|$", ha="right",
                fontsize=fs.SZ_ANNOT, color="#c0392b")
        ax.set_xlim(0, lim);
        ax.set_ylim(0, lim)
        ax.set_xlabel("$|$cell storage trend$|$ (cm yr$^{-1}$)", fontsize=fs.SZ_LABEL)
        ax.set_ylabel("$\\sigma_{\\mathrm{MAD}}$ (cm yr$^{-1}$)", fontsize=fs.SZ_LABEL)
        ax.set_title("Observational ceiling on agreement", fontsize=fs.SZ_TITLE)
    ax.spines[["top", "right"]].set_visible(False)
    fs.panel_letter(ax, "a")

    ax = fig.add_axes((0.585, 0.185, 0.395, 0.700))
    if m.sum():
        ax.scatter(n_a[m], rep_err[m], s=8, alpha=0.45,
                   color="#1f4e79", linewidths=0)
        ax.set_xscale("log")
        ax.set_xlabel("Class A wells in cell", fontsize=fs.SZ_LABEL)
        ax.set_ylabel("$\\sigma_{\\mathrm{MAD}}$ (cm yr$^{-1}$)", fontsize=fs.SZ_LABEL)
        ax.set_title("Does density reduce disagreement?", fontsize=fs.SZ_TITLE)
    ax.spines[["top", "right"]].set_visible(False)
    fs.panel_letter(ax, "b")
    save(fig, "fig2_3_3_representativeness_ceiling")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()