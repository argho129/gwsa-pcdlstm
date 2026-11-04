#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
PHASE 2.0 -- GAP-FILLED CONTINUOUS MONTHLY GWSA PRODUCT
--------------------------------------------------------------------------------
Runs after the monthly GWSA isolation (s03) and before the dataset assembly (s11).

METHOD -- seasonal-trend decomposition with local regression (STL-type), applied
per cell. Different from, and simpler than, the KNN + SARIMA approach of Sun et al.:
  y(t) = trend(t) + seasonal(month) + residual(t)
  trend    : LOWESS-type local linear fit on observed months (robust to the gap,
             since 20 yr of surrounding data anchor it)
  seasonal : robust monthly climatology of detrended residuals
  fill     : trend(t) + seasonal(month) for missing t
  sigma    : empirical, from an INJECTION test -- observed months are held out,
             reconstructed, and the error is measured as a function of distance
             to the nearest observation. Dimensionless scaling relative to sample SD.

OUTPUTS -> OUTPUTS\02_gapfill\
  gwsa_monthly_filled.nc   gwsa (filled), gwsa_sigma, is_filled, domain
  aquifer_series.csv       Jasechko-aggregated monthly series + flagged events
  gapfill_diagnostics.txt  + figures
================================================================================
"""
import os, warnings, traceback
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")

OUT_ROOT = r"D:\THESIS\Arpon_Thesis_Final\OUTPUTS"
DATA = r"D:\THESIS\Arpon_Thesis_Final\DATA"
OUT_DIR = os.path.join(OUT_ROOT, "02_gapfill")
FIG_DIR = os.path.join(OUT_DIR, "figures")
GWSA_NC = os.path.join(OUT_ROOT, "01_gwsa_monthly", "gwsa_monthly.nc")
AQ_SHP = os.path.join(DATA, "Aquifers", "jasechko_et_al_2024_aquifers.shp")
CTY = os.path.join(DATA, "Global SHapefile", "World_Countries_Generalized.shp")

GRID = 0.5
TREND_WINDOW = 61  # months in the local linear window (~5 yr each side)
MIN_OBS_CELL = 60  # months required to decompose a cell
N_INJECT = 2000  # cells sampled for the injection test
MIN_AQ_CELLS = 4  # aquifers smaller than this are not event-scanned
EVENT_SIGMA = 2.0  # departure threshold, in SD of the aquifer series

LINE = "=" * 78;
_LOG = []


def _p(s=""): print(s); _LOG.append(str(s))


def hdr(t): _p("\n" + LINE); _p(t); _p(LINE)


def ok(t): _p("  [ OK ]   " + t)


def warn(t): _p("  [WARN]   " + t)


# ------------------------------------------------------------- decomposition --
def local_linear_trend(y, t, win=TREND_WINDOW):
    """
    LOWESS-type local linear fit evaluated at every t, using observed points only.
    Tricube weights inside a symmetric window. Robust to a contiguous gap because
    the window still contains observations on at least one side.
    """
    n = t.size
    obs = np.isfinite(y)
    if obs.sum() < 10: return np.full(n, np.nan)
    to, yo = t[obs], y[obs]
    out = np.empty(n)
    half = win / 2.0
    for k in range(n):
        d = np.abs(to - t[k])
        m = d <= half
        if m.sum() < 6:  # widen until enough support
            idx = np.argsort(d)[:max(12, min(30, to.size))]
            m = np.zeros(to.size, bool);
            m[idx] = True
        dd = d[m];
        scale = dd.max() if dd.max() > 0 else 1.0
        w = (1 - (dd / scale) ** 3) ** 3
        X = np.c_[np.ones(m.sum()), to[m] - t[k]]
        W = w[:, None]
        try:
            beta = np.linalg.lstsq(X * W, yo[m] * w, rcond=None)[0]
            out[k] = beta[0]
        except Exception:
            out[k] = np.average(yo[m], weights=w)
    return out


def decompose_fill(y, months, t):
    """Return (filled series, trend, seasonal, residual sd)."""
    tr = local_linear_trend(y, t)
    det = y - tr
    seas = np.zeros(12)
    for mo in range(1, 13):
        v = det[(months == mo) & np.isfinite(det)]
        seas[mo - 1] = np.median(v) if v.size else 0.0
    seas -= seas.mean()  # seasonal sums to zero
    s_full = seas[months - 1]
    recon = tr + s_full
    resid = y - recon
    sd = float(np.nanstd(resid[np.isfinite(resid)])) if np.isfinite(resid).any() else np.nan
    filled = np.where(np.isfinite(y), y, recon)
    return filled, tr, s_full, sd


# ============================================================ STEP 1 ==========
def load():
    hdr("STEP 1 -- load Phase-1 GWSA")
    import xarray as xr
    ds = xr.open_dataset(GWSA_NC)
    G = ds["gwsa"].values.astype("float32")
    LAT = ds["lat"].values;
    LON = ds["lon"].values
    dom = ds["domain"].values.astype(bool)
    tim = pd.to_datetime(ds["time"].values)
    ds.close()
    T = G.shape[0]
    miss = np.isnan(G).all(axis=(1, 2))
    _p(f"  months {T}  ({tim[0]:%Y-%m} .. {tim[-1]:%Y-%m})")
    _p(f"  fully-missing months: {int(miss.sum())}")
    runs, cur = [], 0
    for m in miss:
        if m:
            cur += 1
        elif cur:
            runs.append(cur); cur = 0
    if cur: runs.append(cur)
    _p(f"  contiguous gap runs: {runs}")
    _p(f"  domain cells: {int(dom.sum()):,}")
    return G, LAT, LON, dom, tim


# ============================================================ STEP 2 ==========
def injection_test(G, dom, tim):
    hdr("STEP 2 -- INJECTION TEST: empirical fill uncertainty (not assumed)")
    T = G.shape[0];
    t = np.arange(T, dtype="float64");
    months = tim.month.values
    rng = np.random.default_rng(7)

    jj, ii = np.where(dom & (np.isfinite(G).sum(axis=0) >= MIN_OBS_CELL))
    sel = rng.choice(len(jj), size=min(N_INJECT, len(jj)), replace=False)
    _p(f"  cells sampled: {len(sel):,}")
    _p("  holding out OBSERVED months (gap months have no truth to test against);")
    _p("  error is reported vs distance to nearest retained observation.")

    recs = []
    for c, k in enumerate(sel):
        j, i = jj[k], ii[k]
        y = G[:, j, i].astype("float64")
        good = np.where(np.isfinite(y))[0]
        if good.size < MIN_OBS_CELL: continue
        # hold out a contiguous block of the same length as the real gap
        blk = min(11, max(3, good.size // 10))
        start = rng.integers(0, max(1, good.size - blk))
        hold = good[start:start + blk]
        ytr = y.copy();
        ytr[hold] = np.nan
        filled, _, _, sd = decompose_fill(ytr, months, t)
        for pos, idx in enumerate(hold):
            d = int(min(abs(idx - hold[0] + 1), abs(hold[-1] - idx + 1)))
            recs.append({"err": filled[idx] - y[idx], "dist": d, "resid_sd": sd})
        if (c + 1) % 500 == 0: _p(f"     {c + 1}/{len(sel)} cells")
    inj = pd.DataFrame(recs)
    if inj.empty:
        warn("injection produced no records.");
        return None, None

    e = inj["err"]
    ref_sd = float(np.nanmedian(inj["resid_sd"]))
    _p(f"\n  reference residual sd across injected cells: {ref_sd:.3f} cm")
    _p(f"  reconstruction error over {len(inj):,} held-out cell-months:")
    _p(f"     bias={e.mean():+.4f}  MAE={e.abs().mean():.4f}  "
       f"RMSE={np.sqrt((e ** 2).mean()):.4f} cm")
    _p("\n  error vs distance from the nearest retained observation:")
    _p("   dist(mo)     n      MAE      RMSE     RobSD")
    tab = []
    for d in sorted(inj["dist"].unique())[:12]:
        s_err = inj[inj["dist"] == d]["err"]
        if len(s_err) < 30: continue
        mae = float(s_err.abs().mean())
        rmse = float(np.sqrt((s_err ** 2).mean()))
        rob = float(1.4826 * np.median(np.abs(s_err - np.median(s_err))))
        _p(f"   {d:>6}  {len(s_err):>7,}  {mae:7.4f}  {rmse:7.4f}  {rob:7.4f}")
        tab.append({"dist": int(d), "n": len(s_err), "mae": mae,
                    "rmse": rmse, "robust_sd": rob})
    _p("  Robust SD / RMSE(dist) is the uncertainty model applied to filled months.")
    return pd.DataFrame(tab), ref_sd


# ============================================================ STEP 3 ==========
def fill_all(G, dom, tim, unc_tab, ref_sd):
    hdr("STEP 3 -- decompose and fill every domain cell")
    _p(f"  for context: SD of observed monthly GWSA in domain = {np.nanstd(G[:, dom]):.3f} cm")

    T = G.shape[0];
    t = np.arange(T, dtype="float64");
    months = tim.month.values
    F = np.full(G.shape, np.nan, "float32")
    S = np.full(G.shape, np.nan, "float32")
    is_filled = np.isnan(G) & dom[None, :, :]

    # distance (in months) from each timestep to the nearest observed month
    obs_any = ~np.isnan(G).all(axis=(1, 2))
    oidx = np.where(obs_any)[0]
    dist = np.array([0 if obs_any[k] else int(np.min(np.abs(oidx - k)))
                     for k in range(T)])
    _p(f"  max distance to an observed month: {dist.max()}")

    # uncertainty model: robust_sd(dist), extrapolated flat beyond tested range
    if unc_tab is not None and len(unc_tab):
        metric_col = "robust_sd" if "robust_sd" in unc_tab.columns else "rmse"
        dm = unc_tab.set_index("dist")[metric_col].to_dict()
        maxd = max(dm);
        base = dm[maxd]
        sig_of_dist = np.array([dm.get(int(d), base) if d > 0 else 0.0
                                for d in dist])
    else:
        sig_of_dist = np.where(dist > 0, 1.0, 0.0);
        warn("fallback sigma = 1 cm")

    jj, ii = np.where(dom)
    n_ok = 0
    for k, (j, i) in enumerate(zip(jj, ii)):
        y = G[:, j, i].astype("float64")
        if np.isfinite(y).sum() < MIN_OBS_CELL:
            F[:, j, i] = y;
            S[:, j, i] = np.nan
            continue
        filled, tr, se, sd = decompose_fill(y, months, t)
        F[:, j, i] = filled

        # Dimensionless scaling relative to reference residual SD
        scale = (sd / ref_sd) if (np.isfinite(sd) and ref_sd is not None and ref_sd > 0) else 1.0
        scale = float(np.clip(scale, 0.3, 3.0))  # bound relative scaling
        s = sig_of_dist * scale
        s[np.isfinite(y)] = 0.0
        S[:, j, i] = s
        n_ok += 1
        if (k + 1) % 10000 == 0: _p(f"     {k + 1:,}/{len(jj):,} cells")

    ok(f"cells decomposed and filled: {n_ok:,}")
    nf = int(np.isfinite(F[:, dom]).sum());
    nt = int(F[:, dom].size)
    _p(f"  completeness in domain: {100 * nf / nt:.2f}%")
    fm = S[is_filled];
    fm = fm[np.isfinite(fm)]
    if fm.size:
        _p(f"  sigma on FILLED months (cm): p50={np.median(fm):.3f} "
           f"p90={np.percentile(fm, 90):.3f} max={fm.max():.3f}")
    return F, S, is_filled


# ============================================================ STEP 4 ==========
def trend_effect(G, F, dom):
    hdr("STEP 4 -- effect of filling on trends")
    T = G.shape[0];
    t = np.arange(T, dtype="float64")

    def sen(y, x):
        n = y.size
        if n < 3: return np.nan
        with np.errstate(invalid="ignore", divide="ignore"):
            s = (y[None, :] - y[:, None]) / (x[None, :] - x[:, None])
        s = s[np.triu_indices(n, 1)];
        s = s[np.isfinite(s)]
        return float(np.median(s)) if s.size else np.nan

    jj, ii = np.where(dom & (np.isfinite(G).sum(axis=0) >= MIN_OBS_CELL))
    rng = np.random.default_rng(11)
    sel = rng.choice(len(jj), size=min(4000, len(jj)), replace=False)
    d = []
    for k in sel:
        j, i = jj[k], ii[k]
        a = G[:, j, i];
        m = np.isfinite(a)
        b = F[:, j, i];
        mb = np.isfinite(b)
        if m.sum() < MIN_OBS_CELL or mb.sum() < MIN_OBS_CELL: continue
        d.append(sen(b[mb].astype("float64"), t[mb]) * 12 -
                 sen(a[m].astype("float64"), t[m]) * 12)
    d = np.array(d);
    d = d[np.isfinite(d)]
    if d.size:
        _p(f"  cells tested: {d.size:,}")
        _p(f"  median |trend change|: {np.median(np.abs(d)):.5f} cm/yr")
        _p(f"  p95    |trend change|: {np.percentile(np.abs(d), 95):.5f} cm/yr")
        _p(f"  mean signed change   : {d.mean():+.5f} cm/yr")
        _p("  Trend change caused by filling is summarised above.")
    return d


# ============================================================ STEP 5 ==========
def aquifer_series(F, S, is_filled, LAT, LON, dom, tim):
    hdr("STEP 5 -- Jasechko aquifer series + event flags")
    try:
        import geopandas as gpd
        from rasterio.features import rasterize
        from rasterio.transform import from_origin
    except Exception as e:
        warn(f"aquifer step skipped: {e}");
        return None
    if not os.path.exists(AQ_SHP):
        warn(f"aquifer shapefile not found: {AQ_SHP}");
        return None
    aq = gpd.read_file(AQ_SHP)
    if aq.crs is None:
        aq = aq.set_crs("EPSG:4326")
    elif aq.crs.to_epsg() != 4326:
        aq = aq.to_crs("EPSG:4326")
    aq = aq.reset_index(drop=True);
    aq["aq_id"] = np.arange(1, len(aq) + 1)
    lab = rasterize(((g, k) for g, k in zip(aq.geometry, aq["aq_id"])),
                    out_shape=(len(LAT), len(LON)),
                    transform=from_origin(-180, 90, GRID, GRID),
                    fill=0, all_touched=False, dtype="int32")
    lab = np.flipud(lab)
    _p(f"  aquifers rasterised: {len(aq):,}")

    rows, ev = [], []
    for k, name in zip(aq["aq_id"], aq["Aquifer"]):
        m = (lab == k) & dom
        n = int(m.sum())
        if n < MIN_AQ_CELLS: continue
        ser = np.nanmean(F[:, m], axis=1)
        sig = np.sqrt(np.nanmean(S[:, m] ** 2, axis=1)) / max(np.sqrt(n), 1)
        fillfrac = np.nanmean(is_filled[:, m], axis=1)
        for tt in range(len(tim)):
            rows.append({"aq_id": int(k), "aquifer": name, "n_cells": n,
                         "date": tim[tt], "gwsa_cm": ser[tt],
                         "sigma_cm": sig[tt], "filled_frac": fillfrac[tt]})
        # event scan: departure from the cell's own detrended variability
        s = pd.Series(ser)
        dev = s - s.rolling(25, center=True, min_periods=9).median()
        sd = float(dev.std())
        if not np.isfinite(sd) or sd == 0: continue
        z = dev / sd
        for tt in np.where(np.abs(z) >= EVENT_SIGMA)[0]:
            ev.append({"aq_id": int(k), "aquifer": name, "n_cells": n,
                       "date": tim[tt], "z": float(z.iloc[tt]),
                       "gwsa_cm": float(ser[tt]),
                       "filled_frac": float(fillfrac[tt]),
                       "verified": bool(fillfrac[tt] < 0.5)})
    ser_df = pd.DataFrame(rows);
    ev_df = pd.DataFrame(ev)
    ok(f"aquifers with >= {MIN_AQ_CELLS} cells: {ser_df['aq_id'].nunique():,}")
    if len(ev_df):
        _p(f"  departures |z| >= {EVENT_SIGMA}: {len(ev_df):,}  "
           f"(verified, i.e. not gap-driven: {int(ev_df['verified'].sum()):,})")
        top = (ev_df[ev_df["verified"]]
               .assign(absz=lambda d: d["z"].abs())
               .sort_values("absz", ascending=False).head(12))
        _p("\n  largest VERIFIED departures:")
        for _, r in top.iterrows():
            _p(f"     {r['date']:%Y-%m}  z={r['z']:+5.2f}  "
               f"{r['gwsa_cm']:+7.2f} cm  {str(r['aquifer'])[:44]}")
        _p("  Events overlapping reconstructed months are marked unverified")
        _p("      and must not be interpreted as observed excursions.")
    return ser_df, ev_df


# ============================================================ FIGURES =========
def figures(G, F, S, is_filled, LAT, LON, dom, tim, unc_tab, dtr, ser_df):
    hdr("STEP 6 -- figures")
    os.makedirs(FIG_DIR, exist_ok=True)
    import matplotlib;
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    import geopandas as gpd
    rcParams["font.family"] = "serif"
    rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    rcParams["mathtext.fontset"] = "stix"
    rcParams.update({"font.size": 20, "axes.titlesize": 24, "axes.labelsize": 22,
                     "xtick.labelsize": 18, "ytick.labelsize": 18,
                     "legend.fontsize": 17, "axes.linewidth": 1.4})
    cty = gpd.read_file(CTY)
    if cty.crs is None:
        cty = cty.set_crs("EPSG:4326")
    elif cty.crs.to_epsg() != 4326:
        cty = cty.to_crs("EPSG:4326")

    def save(f, nm):
        for e in (".png", ".pdf"):
            f.savefig(os.path.join(FIG_DIR, nm + e), dpi=600,
                      bbox_inches="tight", facecolor="white")
        plt.close(f);
        ok(nm)

    # global mean series, observed vs filled, with sigma band
    gm_o = np.array([np.nanmean(G[k][dom]) for k in range(G.shape[0])])
    gm_f = np.array([np.nanmean(F[k][dom]) for k in range(F.shape[0])])
    gs = np.array([np.nanmean(S[k][dom]) for k in range(S.shape[0])])
    f, ax = plt.subplots(figsize=(16, 8))
    ax.fill_between(tim, gm_f - gs, gm_f + gs, color="#c0392b", alpha=.25,
                    label="reconstruction $\\pm1\\sigma$")
    ax.plot(tim, gm_f, color="#c0392b", lw=2.6, label="continuous (filled)")
    ax.plot(tim, gm_o, color="#1f4e79", lw=3.0, label="observed")
    for k in np.where(np.isnan(gm_o))[0]:
        ax.axvspan(tim[max(k - 1, 0)], tim[min(k + 1, len(tim) - 1)],
                   color="0.85", zorder=0)
    ax.axhline(0, color="0.6", lw=1.0)
    ax.set_xlabel("Year");
    ax.set_ylabel("Global-mean GWSA (cm)")
    ax.set_title("Continuous Monthly Groundwater Storage Anomaly",
                 fontweight="bold")
    ax.legend(frameon=False);
    ax.grid(alpha=.25)
    save(f, "fig24_continuous_series")

    # reconstruction error vs distance
    if unc_tab is not None and len(unc_tab):
        f, ax = plt.subplots(figsize=(12, 8))
        metric_col = "robust_sd" if "robust_sd" in unc_tab.columns else "rmse"
        ax.plot(unc_tab["dist"], unc_tab[metric_col], marker="o", ms=13, lw=3.2,
                color="#1f4e79", label="Robust Error SD" if metric_col == "robust_sd" else "RMSE")
        ax.plot(unc_tab["dist"], unc_tab["mae"], marker="s", ms=11, lw=2.6,
                ls="--", color="#c0392b", label="MAE")
        ax.set_xlabel("Months from nearest observation")
        ax.set_ylabel("Reconstruction error (cm)")
        ax.set_title("Empirical Fill Uncertainty", fontweight="bold")
        ax.legend(frameon=False);
        ax.grid(alpha=.25)
        save(f, "fig25_fill_uncertainty")

    # trend perturbation
    if dtr is not None and dtr.size:
        f, ax = plt.subplots(figsize=(12, 8))
        ax.hist(dtr, bins=70, color="#2c7fb8", edgecolor="white", lw=.5)
        ax.axvline(0, color="0.2", lw=2)
        ax.set_xlabel("Trend change from filling (cm yr$^{-1}$)")
        ax.set_ylabel("Cells")
        ax.set_title("Filling Barely Perturbs Trends", fontweight="bold")
        ax.grid(alpha=.25)
        save(f, "fig26_trend_perturbation")

    # map of mean filled-month uncertainty
    f, ax = plt.subplots(figsize=(17, 8.2))
    smap = np.where(dom, np.nanmean(np.where(is_filled, S, np.nan), axis=0), np.nan)
    ax.set_facecolor("#eef3f7")
    cty.plot(ax=ax, facecolor="#f7f7f5", edgecolor="none", zorder=1)
    im = ax.imshow(smap, origin="lower", extent=[-180, 180, -90, 90],
                   cmap="magma_r", zorder=2)
    cty.boundary.plot(ax=ax, linewidth=.38, color="0.22", zorder=5)
    ax.set_xlim(-180, 180);
    ax.set_ylim(-58, 84)
    ax.set_xlabel("Longitude (°)");
    ax.set_ylabel("Latitude (°)")
    ax.set_title("Reconstruction Uncertainty of Filled Months",
                 pad=14, fontweight="bold")
    cb = f.colorbar(im, ax=ax, shrink=.80, pad=.015, extend="max")
    cb.set_label("Mean $\\sigma$ on filled months (cm)", labelpad=10)
    save(f, "fig27_fill_uncertainty_map")


# ============================================================ MAIN ============
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    _p(LINE);
    _p("PHASE 2.0 -- GAP-FILLED CONTINUOUS MONTHLY GWSA");
    _p(LINE)
    G, LAT, LON, dom, tim = load()
    unc, ref_sd = injection_test(G, dom, tim)
    F, S, is_filled = fill_all(G, dom, tim, unc, ref_sd)
    dtr = trend_effect(G, F, dom)
    aq = aquifer_series(F, S, is_filled, LAT, LON, dom, tim)
    ser_df, ev_df = aq if aq else (None, None)
    figures(G, F, S, is_filled, LAT, LON, dom, tim, unc, dtr, ser_df)

    hdr("STEP 7 -- write")
    import xarray as xr
    xr.Dataset(
        {"gwsa": (("time", "lat", "lon"), F),
         "gwsa_sigma": (("time", "lat", "lon"), S),
         "is_filled": (("time", "lat", "lon"), is_filled.astype("int8")),
         "domain": (("lat", "lon"), dom.astype("int8"))},
        coords={"time": tim, "lat": LAT, "lon": LON},
        attrs={"title": "Continuous monthly GWSA (gap-filled)", "units": "cm",
               "method": "seasonal-trend decomposition with local linear regression; "
                         "seasonal = robust monthly climatology of detrended residuals",
               "uncertainty": "empirical, from injection test vs distance to nearest observation",
               "note": "sigma = 0 on observed months; filled months are reconstructions, "
                       "not measurements; filled values are not independent between cells"}
    ).to_netcdf(os.path.join(OUT_DIR, "gwsa_monthly_filled.nc"))
    ok("gwsa_monthly_filled.nc")
    if unc is not None: unc.to_csv(os.path.join(OUT_DIR, "fill_uncertainty.csv"), index=False)
    if ser_df is not None:
        ser_df.to_csv(os.path.join(OUT_DIR, "aquifer_series.csv"), index=False)
        ok("aquifer_series.csv")
    if ev_df is not None and len(ev_df):
        ev_df.to_csv(os.path.join(OUT_DIR, "aquifer_events.csv"), index=False)
        ok("aquifer_events.csv")
    with open(os.path.join(OUT_DIR, "gapfill_diagnostics.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(_LOG))
    _p("\n" + LINE)
    _p("Continuous product written: gwsa_monthly_filled.nc")
    _p(LINE)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()