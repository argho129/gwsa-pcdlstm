#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
PHASE 5.2 -- BIAS CORRECTION + FINAL CALIBRATED GWSA PRODUCT
--------------------------------------------------------------------------------
WHY THIS EXISTS
  Phase 5.1 reconstruction has strong spatial pattern skill (held-out r=0.78,
  NSE=0.60) but displays amplitude mis-scaling. The fix is per-cell variance-
  matching calibration fit exclusively on observed GRACE months: matching
  each cell's reconstructed distribution to the observed GRACE distribution.

WHAT IT DOES
  1. Diagnoses amplitude bias structure (uniform vs regional).
  2. Fits per-cell variance matching (slope a = std_obs/std_rec, intercept b)
     on observed months (>=24 months), with a global fallback for sparse cells.
  3. Re-evaluates held-out skill AFTER correction (KGE, NSE, gamma, r).
  4. Runs explicit trend-preservation diagnostics (trend sign preservation %,
     pattern correlation, and strong-trend stability).
  5. Computes final trend field using Theil-Sen with annual block-bootstrap CIs.
  5b. Quantifies effect of surface-water cell exclusion against pre-fix product.
  6. Exports the citable frozen product: OUTPUTS/05_model_tensor/gwsa_final.nc.

OUTPUT
  OUTPUTS\05_model_tensor\gwsa_final.nc
  OUTPUTS\05_model_tensor\phase5_2_diagnostics.txt
  OUTPUTS\05_model_tensor\figures\fig5_2_*.png
================================================================================
"""

import os, sys, time, warnings, traceback, datetime as dt
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

OUT_ROOT = r"D:\THESIS\Arpon_Thesis_Final\OUTPUTS"
OUT_DIR = os.path.join(OUT_ROOT, "05_model_tensor")
FIG_DIR = os.path.join(OUT_DIR, "figures")
RECON_NC = os.path.join(OUT_DIR, "gwsa_reconstructed.nc")
TENSOR = os.path.join(OUT_DIR, "model_ready.nc")
FIG_STYLE_DIR = r"D:\THESIS\Arpon_Thesis_Final\Codes\Into the Models"

SEED = 20260730
MIN_OBS_CAL = 24  # min observed months to fit a per-cell correction
BOOT = 200  # bootstrap iterations for trend CI
DRIFT_TOL = 0.5

LINE = "=" * 78
_LOG = []


def _p(s=""): print(s); _LOG.append(str(s))


def hdr(t): _p(""); _p(LINE); _p(t); _p(LINE)


def ok(t): _p("  [ OK ]   " + t)


def warn(t): _p("  [WARN]   " + t)


def fail(t): _p("  [FAIL]   " + t)


def kge(sim, obs):
    m = np.isfinite(sim) & np.isfinite(obs)
    if m.sum() < 12: return (np.nan,) * 4
    o, s = obs[m], sim[m]
    if o.std() == 0 or s.std() == 0: return (np.nan,) * 4
    r = np.corrcoef(s, o)[0, 1]
    beta = s.mean() / o.mean() if o.mean() != 0 else np.nan
    gamma = s.std() / o.std()
    k = 1 - np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2)
    return float(k), float(r), float(beta), float(gamma)


def nse(sim, obs):
    m = np.isfinite(sim) & np.isfinite(obs)
    if m.sum() < 12: return np.nan
    o, s = obs[m], sim[m];
    d = np.sum((o - o.mean()) ** 2)
    return float(1 - np.sum((s - o) ** 2) / d) if d > 0 else np.nan


def theilsen(y, x):
    m = np.isfinite(y)
    if m.sum() < 24: return np.nan
    yy, xx = y[m], x[m]
    n = yy.size
    if n > 200:  # subsample pairs for computational speed
        idx = np.random.default_rng(0).choice(n, 200, replace=False)
        yy, xx = yy[idx], xx[idx];
        n = 200
    sl = []
    for i in range(n):
        dv = xx[i + 1:] - xx[i];
        ok_ = dv != 0
        sl.append((yy[i + 1:][ok_] - yy[i]) / dv[ok_])
    s = np.concatenate(sl) if sl else np.array([np.nan])
    return float(np.median(s))


def ols_trend(y, tyr):
    m = np.isfinite(y)
    if m.sum() < 36: return np.nan
    x = tyr[m] - tyr[m].mean()
    yy = y[m] - y[m].mean()
    dd = np.sum(x * x)
    return float(np.sum(x * yy) / dd) if dd > 0 else np.nan


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    t0 = time.time()
    np.random.seed(SEED)
    _p(LINE);
    _p("PHASE 5.2 -- BIAS CORRECTION + FINAL CALIBRATED PRODUCT")
    _p(f"run: {dt.datetime.now().isoformat(timespec='seconds')}");
    _p(LINE)

    import xarray as xr

    # ------------------------------------------------------------------ STEP 1
    hdr("STEP 1 -- load reconstruction + tensor")
    for p, w in [(RECON_NC, "reconstruction"), (TENSOR, "tensor")]:
        if not os.path.exists(p): fail(f"missing {w}: {p}"); return
    rc = xr.open_dataset(RECON_NC);
    rc.load()
    tn = xr.open_dataset(TENSOR);
    tn.load()
    T = rc.sizes["time"];
    ncell = rc.sizes["cell"]
    rec = rc["gwsa_recon"].values.astype("float32")  # (T,cell)
    obs = rc["gwsa_obs"].values.astype("float32")
    wdat = rc["w_data"].values.astype(bool)
    role = rc["month_role"].values
    keep = rc["keep"].values.astype(bool)
    _sw = tn["surface_water"].values.astype(bool) if "surface_water" in tn else np.zeros_like(keep)
    keep = keep & ~_sw                       # exclude leaked surface-water cells (Phase 0.5b gap)
    hyb = rc["hybas_l3"].values
    clat = rc["clat"].values;
    clon = rc["clon"].values
    ci = np.where(keep)[0]
    _p(f"  excluded {int(_sw.sum())} surface-water cells; kept {ci.size}")
    _p(f"  cells={ncell} kept={ci.size} months={T}")
    _p(f"  reconstruction W_BASIN={rc.attrs.get('chosen_w_basin', '?')} "
       f"heldout_nse={rc.attrs.get('heldout_nse', '?')}")

    # ------------------------------------------------------------------ STEP 2
    hdr("STEP 2 -- diagnose amplitude bias (uniform vs regional)")
    ratios = []
    for k in ci:
        om = wdat[:, k]
        if om.sum() < MIN_OBS_CAL: continue
        so = obs[om, k].std();
        sr = rec[om, k].std()
        if so > 0: ratios.append(sr / so)
    ratios = np.array(ratios)
    _p(f"  cells with >= {MIN_OBS_CAL} obs: {ratios.size}")
    _p(f"  std ratio recon/obs: p25={np.percentile(ratios, 25):.2f} "
       f"median={np.median(ratios):.2f} p75={np.percentile(ratios, 75):.2f}")
    iqr = np.percentile(ratios, 75) - np.percentile(ratios, 25)
    if iqr < 0.5:
        _p("  -> bias is fairly UNIFORM; per-cell matching still applied (safe).")
    else:
        _p("  -> bias is REGIONAL; per-cell variance matching is REQUIRED.")

    # ------------------------------------------------------------------ STEP 3
    hdr("STEP 3 -- per-cell variance-matching correction (fit on observed only)")
    a_cell = np.full(ncell, np.nan, "float32")
    b_cell = np.full(ncell, np.nan, "float32")
    src = np.zeros(ncell, "int8")  # 0 none, 1 per-cell, 2 global-fallback

    # Global fallback params
    go = obs[wdat];
    gr = rec[wdat]
    mo = np.isfinite(go) & np.isfinite(gr)
    g_a = go[mo].std() / gr[mo].std() if gr[mo].std() > 0 else 1.0

    for k in ci:
        om = wdat[:, k] & (role == 1)
        if om.sum() >= MIN_OBS_CAL:
            so = obs[om, k].std();
            sr = rec[om, k].std()
            if sr > 0:
                a = so / sr
                a_cell[k] = a
                b_cell[k] = obs[om, k].mean() - a * rec[om, k].mean()
                src[k] = 1
        if src[k] == 0:  # fallback
            a_cell[k] = g_a
            b_cell[k] = (obs[om, k].mean() - g_a * rec[om, k].mean()) if om.sum() > 0 else 0.0
            src[k] = 2

    weak = sum(1 for k in ci if src[k] == 2 and wdat[:, k].sum() < 12)
    _p(f"  per-cell corrections: {int((src == 1).sum())}  "
       f"global-fallback: {int((src == 2).sum())}")
    _p(f"  of global-fallback cells, {weak} have <12 obs (intercept unstable; flagged)")
    _p(f"  correction slope a: median={np.nanmedian(a_cell[ci]):.2f} "
       f"(a<1 shrinks inflated amplitude, a>1 expands deflated amplitude)")

    corr = rec * a_cell[None, :] + b_cell[None, :]
    corr[:, ~keep] = np.nan

    # ------------------------------------------------------------------ STEP 4
    hdr("STEP 4 -- re-evaluate held-out skill AFTER correction")
    tmask = (role == 3)[:, None] & wdat

    def report(field, tag):
        s = field[tmask];
        o = obs[tmask];
        m = np.isfinite(s) & np.isfinite(o)
        k, r, beta, gamma = kge(s[m], o[m]);
        n = nse(s[m], o[m])
        rmse = float(np.sqrt(np.mean((s[m] - o[m]) ** 2)))
        _p(f"  [{tag}] RMSE={rmse:.3f} NSE={n:+.3f} KGE={k:+.3f} "
           f"(r={r:+.2f} beta={beta:+.2f} gamma={gamma:+.2f})")
        return dict(rmse=rmse, nse=n, kge=k, r=r, beta=beta, gamma=gamma)

    _p("  held-out GRACE months:")
    ev_before = report(rec, "before")
    ev_after = report(corr, "after ")
    _p("")
    if ev_after["kge"] > ev_before["kge"]:
        ok(f"KGE improved {ev_before['kge']:+.3f} -> {ev_after['kge']:+.3f}; "
           f"gamma (variance match) {ev_before['gamma']:.2f} -> {ev_after['gamma']:.2f}, "
           f"r={ev_after['r']:+.2f}")
    else:
        warn("KGE did not improve -- inspect before freezing the product.")

    # ------------------------------------------------------------------ STEP 4b
    hdr("STEP 4b -- CALIBRATION SENSITIVITY & EXPLICIT TREND DIAGNOSTICS")
    tyr = np.arange(T) / 12.0

    # 1. Sensitivity across MIN_OBS thresholds
    _p(f"  {'MIN_OBS':<9}{'per-cell':<10}{'KGE':<9}{'beta':<8}{'gamma':<8}{'trend r vs raw':<14}")
    sens_rows = []
    sample = ci[::max(1, ci.size // 3000)]
    raw_tr_sample = np.array([ols_trend(rec[:, k], tyr) for k in sample])

    for thr in (12, 18, 24, 36, 48):
        aC = np.full(ncell, np.nan, "float32")
        bC = np.full(ncell, np.nan, "float32")
        sC = np.zeros(ncell, "int8")
        for k in ci:
            om = wdat[:, k]
            if om.sum() >= thr and rec[om, k].std() > 0:
                a = obs[om, k].std() / rec[om, k].std()
                aC[k] = a;
                bC[k] = obs[om, k].mean() - a * rec[om, k].mean();
                sC[k] = 1
            if sC[k] == 0:
                aC[k] = g_a
                bC[k] = (obs[om, k].mean() - g_a * rec[om, k].mean()) if om.sum() > 0 else 0.0
                sC[k] = 2
        cor_temp = rec * aC[None, :] + bC[None, :]
        s_t = cor_temp[tmask];
        o_t = obs[tmask];
        m_t = np.isfinite(s_t) & np.isfinite(o_t)
        k_, r_, be_, ga_ = kge(s_t[m_t], o_t[m_t])
        cor_tr_sample = np.array([ols_trend(cor_temp[:, k], tyr) for k in sample])
        mm = np.isfinite(cor_tr_sample) & np.isfinite(raw_tr_sample)
        tr_r = float(np.corrcoef(cor_tr_sample[mm], raw_tr_sample[mm])[0, 1]) if mm.sum() > 10 else np.nan
        _p(f"  {thr:<9}{int((sC == 1).sum()):<10}{k_:<+9.3f}{be_:<8.2f}{ga_:<8.2f}{tr_r:<14.3f}")
        sens_rows.append((thr, int((sC == 1).sum()), k_, be_, ga_, tr_r))
    _p("")

    # 2. Detailed Trend Diagnostic Suite
    _p("  Running full trend-preservation diagnostics across all kept cells...")
    tr_raw = np.array([ols_trend(rec[:, k], tyr) for k in ci])
    tr_cor = np.array([ols_trend(corr[:, k], tyr) for k in ci])
    m_tr = np.isfinite(tr_raw) & np.isfinite(tr_cor)
    same = np.sign(tr_raw[m_tr]) == np.sign(tr_cor[m_tr])
    p_corr = np.corrcoef(tr_raw[m_tr], tr_cor[m_tr])[0, 1]
    big = np.isfinite(tr_raw) & (np.abs(tr_raw) > 0.5)
    big_same = (np.sign(tr_raw[big]) == np.sign(tr_cor[big])).mean() * 100.0 if big.sum() > 0 else 100.0

    _p(f"  cells compared                    : {m_tr.sum()}")
    _p(f"  trend SIGN preserved              : {100 * same.mean():.2f}%")
    _p(f"  pattern correlation (magnitude)   : {p_corr:.4f}")
    _p(f"  sign preserved for |raw trend|>0.5: {big_same:.2f}%")

    if same.mean() == 1.0:
        ok("Trend directional signs are 100% PRESERVED across all cells.")
    else:
        warn(f"Trend sign flipped in {100 * (1 - same.mean()):.2f}% of cells.")

    # ------------------------------------------------------------------ STEP 5
    hdr("STEP 5 -- final trend field (Theil-Sen) + bootstrap CI")
    trend = np.full(ncell, np.nan, "float32")
    ci_lo = np.full(ncell, np.nan, "float32")
    ci_hi = np.full(ncell, np.nan, "float32")
    rng = np.random.default_rng(SEED)
    _p(f"  Theil-Sen slope per cell, {BOOT}x block bootstrap CI...")
    done = 0
    for k in ci:
        y = corr[:, k]
        m = np.isfinite(y)
        if m.sum() < 36: continue
        trend[k] = theilsen(y, tyr)
        # block bootstrap over years for CI
        yr_idx = (np.arange(T) // 12)
        uyr = np.unique(yr_idx[m])
        if uyr.size < 6: continue
        bs = []
        for _ in range(BOOT):
            pick = rng.choice(uyr, uyr.size, replace=True)
            sel = np.concatenate([np.where(yr_idx == u)[0] for u in pick])
            sel = sel[np.isfinite(y[sel])]
            if sel.size < 24: continue
            bs.append(theilsen(y[sel], tyr[sel]))
        bs = np.array([v for v in bs if np.isfinite(v)])
        if bs.size >= 50:
            ci_lo[k] = np.percentile(bs, 2.5);
            ci_hi[k] = np.percentile(bs, 97.5)
        done += 1
        if done % 5000 == 0: _p(f"     {done} cells...")
    _p(f"  trends computed: {int(np.isfinite(trend).sum())}")
    sig = np.isfinite(ci_lo) & np.isfinite(ci_hi) & ((ci_lo > 0) | (ci_hi < 0))
    _p(f"  cells with CI excluding zero (significant): {int(sig.sum())}")
    _p(f"  global trend range: {np.nanpercentile(trend[ci], 2):.2f} .. "
       f"{np.nanpercentile(trend[ci], 98):.2f} cm/yr")

    # ---- quantify effect of excluding surface-water cells (defensible sentence) ----
    hdr("STEP 5b -- effect of surface-water exclusion vs pre-fix product")
    _prefix = os.path.join(OUT_DIR, "gwsa_final_PREFIX.nc")
    if os.path.exists(_prefix):
        try:
            _old = xr.open_dataset(_prefix); _to = _old["gwsa_trend"].values; _old.close()
            _k = keep & np.isfinite(_to) & np.isfinite(trend)
            _d = 100*np.abs(trend[_k]-_to[_k])/np.maximum(np.abs(_to[_k]),1e-6)
            _p(f"  cells compared: {int(_k.sum())}")
            _p(f"  median trend change vs pre-fix: {np.nanmedian(_d):.2f}%  p95: {np.nanpercentile(_d,95):.2f}%")
            _p(f"  cells changing >5%: {int(np.sum(_d>5))} ({100*np.mean(_d>5):.2f}%)")
            ok("surface-water exclusion effect quantified (expect median <1%).")
        except Exception as e:
            warn(f"pre-fix comparison skipped: {e}")
    else:
        warn(f"no pre-fix backup at {_prefix}; skipping comparison (back up BEFORE first re-run).")

    # ------------------------------------------------------------------ STEP 6
    hdr("STEP 6 -- write FINAL product")
    out = xr.Dataset(
        {"gwsa_calibrated": (("time", "cell"), corr.astype("float32")),
         "gwsa_recon_raw": (("time", "cell"), rec.astype("float32")),
         "gwsa_obs": (("time", "cell"), obs.astype("float32")),
         "gwsa_trend": (("cell",), trend),
         "trend_ci_lo": (("cell",), ci_lo),
         "trend_ci_hi": (("cell",), ci_hi),
         "trend_significant": (("cell",), sig.astype("int8")),
         "calib_slope_a": (("cell",), a_cell),
         "calib_intercept_b": (("cell",), b_cell),
         "calib_source": (("cell",), src),
         "month_role": (("time",), role),
         "w_data": (("time", "cell"), wdat.astype("int8")),
         "keep": (("cell",), keep.astype("int8")),
         "hybas_l3": (("cell",), hyb),
         "clat": (("cell",), clat.astype("float32")),
         "clon": (("cell",), clon.astype("float32"))},
        coords={"time": rc["time"].values, "cell": np.arange(ncell)},
        attrs={
            "title": "Aquifer-to-Atmosphere calibrated GWSA product (Phase 5.2)",
            "created": dt.datetime.now().isoformat(timespec="seconds"),
            "calibration": "per-cell variance matching on observed months; "
                           "global fallback where obs<{}".format(MIN_OBS_CAL),
            "heldout_kge_before": float(ev_before["kge"]),
            "heldout_kge_after": float(ev_after["kge"]),
            "heldout_nse_after": float(ev_after["nse"]),
            "heldout_beta_after": float(ev_after["beta"]),
            "heldout_gamma_after": float(ev_after["gamma"]),
            "heldout_rmse_after": float(ev_after["rmse"]),
            "trend_sign_preserved_pct": float(100 * same.mean()),
            "trend_pattern_corr": float(p_corr),
            "trend_method": "Theil-Sen; 95% block-bootstrap CI over years",
            "calib_source_codes": "0=none,1=per-cell,2=global-fallback",
            "note": "citable product; Phase 6 wells VALIDATE, do not modify this."})
    enc = {v: {"zlib": True, "complevel": 4} for v in out.data_vars}
    p = os.path.join(OUT_DIR, "gwsa_final.nc");
    out.to_netcdf(p, encoding=enc)
    ok(f"written: {p} ({os.path.getsize(p) / 1e6:.0f} MB)")

    try:
        figs(rec, corr, obs, role, wdat, ci, clat, clon, trend, ci_lo, ci_hi, sig,
             a_cell, src, ev_before, ev_after)
    except Exception as e:
        warn(f"figures failed: {e}");
        traceback.print_exc()

    with open(os.path.join(OUT_DIR, "phase5_2_diagnostics.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(_LOG))
    _p(f"\n  wall time: {(time.time() - t0) / 60:.1f} min")
    _p(LINE)
    _p("Run complete.")
    _p(LINE)


def figs(rec, corr, obs, role, wdat, ci, clat, clon, trend, ci_lo, ci_hi, sig, a_cell, src,
         ev_b, ev_a):
    import matplotlib;
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        sys.path.insert(0, FIG_STYLE_DIR);
        import fig_style as fs
        fs.apply();
        SZL, SZT, SZA = fs.SZ_LABEL + 1, fs.SZ_TITLE + 1, fs.SZ_ANNOT;
        HAVE = True
    except Exception:
        from matplotlib import rcParams
        rcParams["font.family"] = "serif";
        rcParams.update({"font.size": 11, "savefig.dpi": 600})
        SZL = SZT = SZA = 11;
        HAVE = False

    def save(f, nm):
        for e in (".png", ".pdf"):
            f.savefig(os.path.join(FIG_DIR, nm + e), dpi=600, bbox_inches="tight", facecolor="white")
        plt.close(f);
        ok(nm)

    tmask = (role == 3)[:, None] & wdat
    # FIG A: before/after calibration scatter + KGE bars
    fig = plt.figure(figsize=(11, 4))
    for i, (F, tag, ev) in enumerate([(rec, "before", ev_b), (corr, "after", ev_a)]):
        ax = fig.add_axes((0.06 + i * 0.34, 0.15, 0.27, 0.75))
        s = F[tmask];
        o = obs[tmask];
        m = np.isfinite(s) & np.isfinite(o)
        lim = np.nanpercentile(np.abs(np.concatenate([s[m], o[m]])), 99)
        ax.hexbin(o[m], s[m], gridsize=45, cmap="magma", mincnt=1, extent=[-lim, lim, -lim, lim])
        ax.plot([-lim, lim], [-lim, lim], "w--", lw=1)
        ax.set_xlabel("Observed (cm)", fontsize=SZL)
        if i == 0: ax.set_ylabel("Reconstructed (cm)", fontsize=SZL)
        ax.set_title(f"{tag}: KGE={ev['kge']:+.2f} $\\gamma$={ev['gamma']:.2f}", fontsize=SZT)
    ax = fig.add_axes((0.76, 0.15, 0.21, 0.75))
    labs = ["KGE", "NSE", "beta", "gamma"]
    bv = [ev_b["kge"], ev_b["nse"], ev_b["beta"], ev_b["gamma"]]
    av = [ev_a["kge"], ev_a["nse"], ev_a["beta"], ev_a["gamma"]]
    x = np.arange(4);
    ax.bar(x - 0.2, bv, 0.4, label="before", color="#c0392b")
    ax.bar(x + 0.2, av, 0.4, label="after", color="#1f4e79")
    ax.axhline(1, color="0.5", ls=":", lw=0.8);
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(x);
    ax.set_xticklabels(labs, rotation=30);
    ax.legend(frameon=False, fontsize=SZA)
    ax.set_title("Metrics", fontsize=SZT);
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig5_2_calibration")

    # FIG B: final trend map (Robinson)
    fig = plt.figure(figsize=(11, 5.5))
    if HAVE:
        import cartopy.crs as ccrs
        ax = fs.global_map_axis(fig, (0.04, 0.14, 0.92, 0.80), projection="robinson", basins=True)
        lim = np.nanpercentile(np.abs(trend[ci]), 97)
        sc = ax.scatter(clon[ci], clat[ci], c=trend[ci], cmap="RdBu", vmin=-lim, vmax=lim,
                        s=3.5, marker="s", linewidths=0, transform=ccrs.PlateCarree())
        fs.hcolorbar(fig, sc, (0.30, 0.10, 0.40, 0.014), "Calibrated GWSA trend (cm yr$^{-1}$)")
        ax.set_title("Final calibrated groundwater storage trend, 2002-2022", fontsize=SZT)
    else:
        ax = fig.add_subplot(111)
        lim = np.nanpercentile(np.abs(trend[ci]), 97)
        sc = ax.scatter(clon[ci], clat[ci], c=trend[ci], cmap="RdBu", vmin=-lim, vmax=lim, s=2, linewidths=0)
        plt.colorbar(sc, ax=ax, shrink=0.7, label="cm/yr")
        ax.set_xlim(-180, 180);
        ax.set_ylim(-60, 85)
    save(fig, "fig5_2_final_trend_map")

    # FIG C: significance + calibration provenance
    fig = plt.figure(figsize=(11, 5))
    ax = fig.add_axes((0.05, 0.12, 0.44, 0.80))
    if HAVE:
        import cartopy.crs as ccrs
        ax = fs.global_map_axis(fig, (0.03, 0.10, 0.45, 0.82), projection="robinson", basins=True)
        col = np.where(sig[ci], 1, 0)
        ax.scatter(clon[ci], clat[ci], c=col, cmap="RdGy_r", vmin=0, vmax=1, s=3, marker="s",
                   linewidths=0, transform=ccrs.PlateCarree())
        ax.set_title("Significant trends (CI excludes 0)", fontsize=SZT)
    ax2 = fig.add_axes((0.55, 0.15, 0.40, 0.72))
    ax2.hist(a_cell[ci][np.isfinite(a_cell[ci])], bins=50, color="#1f4e79", edgecolor="k", lw=0.3)
    ax2.axvline(1, color="#c0392b", ls="--", lw=1)
    ax2.set_xlabel("Calibration slope $a$ (std$_{obs}$/std$_{recon}$)", fontsize=SZL)
    ax2.set_ylabel("Cells", fontsize=SZL)
    ax2.set_title("Amplitude correction factor", fontsize=SZT)
    ax2.spines[["top", "right"]].set_visible(False)
    save(fig, "fig5_2_significance_provenance")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()