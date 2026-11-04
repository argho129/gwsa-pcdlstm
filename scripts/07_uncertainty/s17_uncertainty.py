#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
PHASE 7.0 -- UNCERTAINTY DECOMPOSITION (Spatially-Correlated Monte Carlo)
================================================================================
All components DERIVED FROM DATA (no assumed inflation factor).
Per-cell trend uncertainty (cm/yr) split into 3 independent components:
  (1) OBS    : propagate gwsa_sigma(time, cell) through trend estimation
  (2) MODEL  : held-out residual RMSE (data-derived) as monthly model-error sigma
  (3) SY     : specific-yield error (measured ~15%, imputed ~40% relative)

Monte Carlo: N=200 realizations, spatially-correlated and temporally-independent
Gaussian noise applied at fitted correlation length -> vectorized OLS trend refit ->
per-cell total sigma + component variance fractions.

Tiering: Well-constrained (well density >= 1 Class A) vs Physics-extrapolated.
================================================================================
"""

import os
import sys
import time
import warnings
import traceback
import datetime as dt
import numpy as np

warnings.filterwarnings("ignore")

# Path Configuration
OUT = r"D:\THESIS\Arpon_Thesis_Final\OUTPUTS"
MODEL = os.path.join(OUT, "05_model_tensor")
UNC = os.path.join(OUT, "07_uncertainty")
FIGDIR = os.path.join(UNC, "figures")
FINAL_NC = os.path.join(MODEL, "gwsa_final.nc")
TENSOR_NC = os.path.join(MODEL, "model_ready.nc")
WELL_NC = os.path.join(OUT, "02_well_trends", "well_trends_by_cell.nc")
FIG_STYLE_DIR = r"D:\THESIS\Arpon_Thesis_Final\Codes\Into the Models"

SEED = 20260730
N_MC = 200
HELDOUT_RMSE_CM = 7.05  # Phase 5.2 held-out model error (monthly, cm)
SY_ERR_MEASURED = 0.15  # relative sigma, measured Sy
SY_ERR_IMPUTED = 0.40   # relative sigma, proxy Sy
NLAT, NLON = 360, 720
LINE = "=" * 78
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


def ols_vec(pert, tyr):
    """
    Fast vectorized OLS slope calculation along time axis (axis 0).
    pert: shape (Tn, nk)
    tyr: shape (Tn,)
    Returns slope per cell: shape (nk,)
    """
    xx = tyr - tyr.mean()
    ss_xx = np.sum(xx ** 2)
    yy = pert - pert.mean(axis=0, keepdims=True)
    return (xx[:, None] * yy).sum(axis=0) / ss_xx


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    hdr("PHASE 7.0 -- UNCERTAINTY DECOMPOSITION")
    _p(f"Run date/time: {dt.datetime.now().isoformat(timespec='seconds')}")

    import xarray as xr
    import pandas as pd

    G = xr.open_dataset(FINAL_NC)
    G.load()
    T = xr.open_dataset(TENSOR_NC)
    T.load()

    keep = G["keep"].values.astype(bool)
    if "surface_water" in T:                                    # exclude leaked surface-water cells (Phase 0.5b gap)
        _sw = T["surface_water"].values.astype(bool)
        keep = keep & ~_sw
        _p(f"  excluded {int(_sw.sum())} surface-water cells; kept {int(keep.sum())}")
    ci = np.where(keep)[0]
    clat = G["clat"].values
    clon = G["clon"].values
    gcal = G["gwsa_calibrated"].values  # (time, cell)
    trend = G["gwsa_trend"].values
    tvals = pd.to_datetime(G["time"].values)
    tyr = (np.arange(tvals.size)) / 12.0
    sigma = T["gwsa_sigma"].values     # (time, cell)
    Tn = tvals.size
    nk = ci.size

    # ---- STEP 1: Observation Sigma Floor
    hdr("STEP 1 -- Observation Uncertainty Floor & Clean-up")
    sig_ci = sigma[:, ci]  # (time, nk)
    pos = sig_ci[sig_ci > 0]
    floor = float(np.nanpercentile(pos, 25)) if pos.size else 1.0
    sig_ci = np.where(sig_ci > 0, sig_ci, floor)
    _p(f"  gwsa_sigma > 0 median = {np.nanmedian(pos):.3f} cm, floor(p25) = {floor:.3f} cm applied to zeros")

    # ---- STEP 2: Spatial Variogram Analysis
    hdr("STEP 2 -- Spatial Sampling: Empirical Variogram -> Correlation Length")
    tr = trend[ci]
    la = clat[ci]
    lo = clon[ci]
    fin = np.isfinite(tr)
    trf, laf, lof = tr[fin], la[fin], lo[fin]

    ns = min(4000, trf.size)
    idx = rng.choice(trf.size, ns, replace=False)
    P = np.c_[lof[idx], laf[idx]]
    V = trf[idx]

    from scipy.spatial import cKDTree
    tree = cKDTree(P)
    bins = np.linspace(0, 20, 21)
    semiv = np.zeros(len(bins) - 1)
    cnt = np.zeros(len(bins) - 1)

    for i in range(ns):
        d_spatial, j_idx = tree.query(P[i], k=min(60, ns))
        dd = np.sqrt(((P[j_idx] - P[i]) ** 2).sum(1))
        dv = 0.5 * (V[j_idx] - V[i]) ** 2
        b = np.digitize(dd, bins) - 1
        for bb in range(len(bins) - 1):
            sel = (b == bb)
            semiv[bb] += dv[sel].sum()
            cnt[bb] += sel.sum()

    semiv = np.where(cnt > 0, semiv / np.maximum(cnt, 1), np.nan)
    centers = 0.5 * (bins[:-1] + bins[1:])
    sill = np.nanmedian(semiv[centers > 10])

    cross = np.where(semiv >= 0.95 * sill)[0]
    rng_deg = float(centers[cross[0]]) if cross.size else 5.0
    rng_deg = min(max(rng_deg, 1.0), 6.0)
    rng_km = rng_deg * 111.0

    cell_deg = 0.5
    n_per_corr = (rng_deg / cell_deg) ** 2
    infl = 1.0
    _p(f"  Variogram range ~ {rng_deg:.1f} deg (~{rng_km:.0f} km); sill = {sill:.3f}")
    _p(f"  Correlation area ~ {n_per_corr:.0f} cells (basin aggregation factor)")

    # ---- STEP 3: Specific Yield Error Assignment
    hdr("STEP 3 -- Specific Yield Error per Cell")
    W = xr.open_dataset(WELL_NC)
    W.load()
    wlat = W["lat"].values
    wlon = W["lon"].values
    nA = W["n_wells_A"].values

    def to_idx(la_, lo_):
        return np.abs(wlat[None, :] - la_[:, None]).argmin(1), np.abs(wlon[None, :] - lo_[:, None]).argmin(1)

    jla, jlo = to_idx(clat[ci], clon[ci])
    has_wells = nA[jla, jlo] >= 1
    sy_relerr = np.where(has_wells, SY_ERR_MEASURED, SY_ERR_IMPUTED)
    _p(f"  Cells with measured Sy (>=1 Class A well): {int(has_wells.sum())}; else imputed error {SY_ERR_IMPUTED}")

    # ---- STEP 4: Spatially-Correlated, Temporally-Independent Monte Carlo
    hdr(f"STEP 4 -- Vectorized Monte Carlo (N={N_MC}, Spatial Range={rng_deg:.1f} deg, Independent Months)")
    from scipy.ndimage import gaussian_filter

    ai = np.round((clat[ci] + 89.75) / 0.5).astype(int)
    aj = np.round((clon[ci] + 179.75) / 0.5).astype(int)
    sigma_blur = max(rng_deg / 0.5 / 2.0, 0.8)
    monthly_model = HELDOUT_RMSE_CM
    tr_obs = np.zeros((N_MC, nk), dtype="float32")

    for it in range(N_MC):
        # Construct 3D tensor: (Tn, NLAT, NLON)
        base = np.zeros((Tn, NLAT, NLON), dtype="float32")
        base[:, ai, aj] = rng.standard_normal((Tn, nk)).astype("float32")

        # 3D Gaussian smoothing: axis 0 (time) = 0 sigma; axis 1 & 2 (space) = sigma_blur
        cf = gaussian_filter(base, (0, sigma_blur, sigma_blur), mode="wrap")
        nz = cf[:, ai, aj]

        # Standardize noise spatially per month to unit variance
        nz_std = nz.std(axis=1, keepdims=True) + 1e-9
        nz = nz / nz_std

        # Apply perturbation independently across months
        pert = gcal[:, ci] + sig_ci * nz + monthly_model * nz

        # Vectorized linear regression across time axis
        tr_obs[it] = ols_vec(pert, tyr)

        if (it + 1) % 50 == 0 or it == 0:
            _p(f"    MC progress: {it + 1}/{N_MC} realizations done")

    sig_obsmodel = np.nanstd(tr_obs, axis=0) * infl
    _p(f"  Obs+Model trend sigma: median = {np.nanmedian(sig_obsmodel):.3f} cm/yr")

    # ---- STEP 5: Variance Component Decomposition
    hdr("STEP 5 -- Combine Component Variances -> Total Trend Sigma")
    sig_sy = np.abs(tr) * sy_relerr

    mean_var_obs = np.nanmean(sig_ci ** 2, axis=0)
    mean_var_model = monthly_model ** 2
    obs_var_share = mean_var_obs / (mean_var_obs + mean_var_model)

    sig_obs = sig_obsmodel * np.sqrt(obs_var_share)
    sig_model = sig_obsmodel * np.sqrt(1.0 - obs_var_share)

    var_total = sig_obs ** 2 + sig_model ** 2 + sig_sy ** 2
    sig_total = np.sqrt(var_total)

    frac_obs = (sig_obs ** 2) / var_total
    frac_model = (sig_model ** 2) / var_total
    frac_sy = (sig_sy ** 2) / var_total

    _p(f"  TOTAL trend sigma: p25={np.nanpercentile(sig_total, 25):.3f}, "
       f"median={np.nanmedian(sig_total):.3f}, p95={np.nanpercentile(sig_total, 95):.3f} cm/yr")
    _p(f"  Mean variance fractions: Obs={np.nanmean(frac_obs):.3f}, "
       f"Model={np.nanmean(frac_model):.3f}, Sy={np.nanmean(frac_sy):.3f}")

    dom = np.argmax(np.c_[frac_obs, frac_model, frac_sy], axis=1)
    for k, nmc in enumerate(["Obs", "Model", "Sy"]):
        _p(f"    Dominant = {nmc}: {int((dom == k).sum())} cells ({100 * np.mean(dom == k):.1f}%)")

    # ---- STEP 6: SNR & Tier Diagnostics
    hdr("STEP 6 -- SNR, Significance, and Tiering")
    snr = np.abs(tr) / np.maximum(sig_total, 1e-6)
    sig_after = snr >= 1.96
    tier = np.where(has_wells, 1, 0)

    _p(f"  Cells significant after full uncertainty (SNR >= 1.96): "
       f"{int(np.nansum(sig_after))} ({100 * np.nanmean(sig_after):.1f}%)")
    _p(f"  Well-constrained cells (Tier 1): {int(tier.sum())} ({100 * tier.mean():.1f}%); "
       f"Physics-extrapolated (Tier 0): {int((tier == 0).sum())}")
    _p(f"  Median SNR: Well-constrained = {np.nanmedian(snr[tier == 1]):.2f}, "
       f"Extrapolated = {np.nanmedian(snr[tier == 0]):.2f}")

    # ---- STEP 7: NetCDF Output Generation
    hdr("STEP 7 -- Export NetCDF Output")

    def full(v):
        g = np.full(keep.size, np.nan, dtype="float32")
        g[ci] = v
        return g

    ds = xr.Dataset(
        {
            "trend": (("cell",), full(tr)),
            "sigma_total": (("cell",), full(sig_total)),
            "sigma_obs": (("cell",), full(sig_obs)),
            "sigma_model": (("cell",), full(sig_model)),
            "sigma_sy": (("cell",), full(sig_sy)),
            "frac_obs": (("cell",), full(frac_obs)),
            "frac_model": (("cell",), full(frac_model)),
            "frac_sy": (("cell",), full(frac_sy)),
            "dominant": (("cell",), full(dom.astype("float32"))),
            "snr": (("cell",), full(snr)),
            "significant_full": (("cell",), full(sig_after.astype("float32"))),
            "tier": (("cell",), full(tier.astype("float32"))),
            "keep": (("cell",), keep.astype("int8")),
            "clat": (("cell",), clat.astype("float32")),
            "clon": (("cell",), clon.astype("float32")),
        },
        coords={"cell": np.arange(keep.size)},
        attrs={
            "created": dt.datetime.now().isoformat(timespec="seconds"),
            "method": "spatially-correlated Monte Carlo; independent monthly noise; vector OLS",
            "spatial_range_deg": rng_deg,
            "spatial_inflation": infl,
            "cells_per_corr_area": n_per_corr,
            "n_mc": N_MC,
            "heldout_rmse_cm": HELDOUT_RMSE_CM,
            "tier_codes": "1=well-constrained, 0=physics-extrapolated",
            "dominant_codes": "0=obs, 1=model, 2=Sy",
        },
    )

    out_nc_path = os.path.join(UNC, "gwsa_uncertainty.nc")
    ds.to_netcdf(out_nc_path)
    ok(f"Successfully written: {out_nc_path}")

    # ---- STEP 8: Generate Publication Figures
    hdr("STEP 8 -- Generating Figures")
    try:
        _figs(
            ci, clat, clon, keep, tr, sig_total, dom, snr, sig_after, tier,
            frac_obs, frac_model, frac_sy, sig_obs, sig_model, sig_sy,
            centers, semiv, sill, rng_deg
        )
    except Exception as e:
        warn(f"Figure generation error: {e}")
        traceback.print_exc()

    with open(os.path.join(UNC, "phase7_diagnostics.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(_LOG))

    _p(f"\n  Wall time: {(time.time() - t0) / 60:.2f} min")
    _p(LINE)


def _figs(
    ci, clat, clon, keep, tr, sig, dom, snr, sig_after, tier,
    fo, fm, fsy, sig_obs, sig_model, sig_sy,
    centers, semiv, sill, rng_deg
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, FIG_STYLE_DIR)
    import fig_style as fs
    fs.apply()

    def grid(v):
        g = np.full((NLAT, NLON), np.nan, dtype="float32")
        jla = np.round((clat[ci] + 89.75) / 0.5).astype(int)
        jlo = np.round((clon[ci] + 179.75) / 0.5).astype(int)
        ok_ = (jla >= 0) & (jla < NLAT) & (jlo >= 0) & (jlo < NLON)
        g[jla[ok_], jlo[ok_]] = v[ok_]
        return g

    # --------------------------------------------------------------------------
    # FIGURE 1: 4-Panel Global Uncertainty Overview
    # --------------------------------------------------------------------------
    fig1 = plt.figure(figsize=(7.6, 7.2))
    specs1 = [
        (r"Total Trend Uncertainty $\sigma_{\mathrm{total}}$ (cm yr$^{-1}$)", grid(sig), fs.CMAP_MAG, 0.0, float(np.nanpercentile(sig, 95)), "a"),
        (r"Signal-to-Noise Ratio $|\mathrm{trend}| / \sigma_{\mathrm{total}}$", grid(snr), fs.CMAP_SEQ_Y, 0.0, 3.0, "b"),
        (r"Dominant Uncertainty Component (0: Obs, 1: Model, 2: $\mathrm{S}_y$)", grid(dom), "viridis", 0, 2, "c"),
        (r"Significant After Full Uncertainty ($\mathrm{SNR} \geq 1.96$)", grid(sig_after), fs.CMAP_FRAC, 0, 1, "d"),
    ]
    for k, (ttl, g, cm, vmn, vmx, lt) in enumerate(specs1):
        ax = fs.global_map_axis(fig1, (0.04, 0.74 - 0.235 * k, 0.92, 0.21), projection="robinson")
        m = fs.pcolor_global_smooth(ax, g, NLAT, NLON, cm, vmn, vmx, upsample=3 if k != 2 else 1)
        ax.set_title(ttl, fontsize=fs.SZ_TITLE)
        fs.panel_letter(ax, lt)
        fs.hcolorbar(fig1, m, (0.35, 0.735 - 0.235 * k, 0.30, 0.008), "")
    fs.save(fig1, "fig7_01_uncertainty_maps", figdir=FIGDIR)

    # --------------------------------------------------------------------------
    # FIGURE 2: 3-Panel Component Breakdown Maps (Obs, Model, Sy)
    # --------------------------------------------------------------------------
    fig2 = plt.figure(figsize=(7.6, 5.6))
    specs2 = [
        (r"Observation Uncertainty $\sigma_{\mathrm{obs}}$ (cm yr$^{-1}$)", grid(sig_obs), fs.CMAP_MAG, 0.0, float(np.nanpercentile(sig_obs, 95)), "a"),
        (r"Model Error Uncertainty $\sigma_{\mathrm{model}}$ (cm yr$^{-1}$)", grid(sig_model), fs.CMAP_MAG, 0.0, float(np.nanpercentile(sig_model, 95)), "b"),
        (r"Specific Yield Uncertainty $\sigma_{\mathrm{S}_y}$ (cm yr$^{-1}$)", grid(sig_sy), fs.CMAP_MAG, 0.0, float(np.nanpercentile(sig_sy, 95)), "c"),
    ]
    for k, (ttl, g, cm, vmn, vmx, lt) in enumerate(specs2):
        ax = fs.global_map_axis(fig2, (0.04, 0.68 - 0.32 * k, 0.92, 0.28), projection="robinson")
        m = fs.pcolor_global_smooth(ax, g, NLAT, NLON, cm, vmn, vmx, upsample=3)
        ax.set_title(ttl, fontsize=fs.SZ_TITLE)
        fs.panel_letter(ax, lt)
        fs.hcolorbar(fig2, m, (0.35, 0.675 - 0.32 * k, 0.30, 0.010), "")
    fs.save(fig2, "fig7_02_components_breakdown_maps", figdir=FIGDIR)

    # --------------------------------------------------------------------------
    # FIGURE 3: Variance Fractions, Tiering & Density
    # --------------------------------------------------------------------------
    fig3, axes = plt.subplots(2, 2, figsize=(8.2, 6.2))

    # Panel a: Variance fraction distributions
    ax = axes[0, 0]
    ax.hist(
        [fo[np.isfinite(fo)], fm[np.isfinite(fm)], fsy[np.isfinite(fsy)]],
        bins=25,
        stacked=False,
        label=["Obs", "Model", r"$\mathrm{S}_y$"],
        color=["#1f4e79", "#c0392b", "#2ca02c"],
        alpha=0.85,
    )
    ax.set_xlabel("Variance Fraction", fontsize=fs.SZ_LABEL)
    ax.set_ylabel("Cells", fontsize=fs.SZ_LABEL)
    ax.legend(frameon=False, fontsize=fs.SZ_ANNOT)
    ax.set_title("Variance Component Fractions", fontsize=fs.SZ_TITLE)
    ax.spines[["top", "right"]].set_visible(False)
    fs.panel_letter(ax, "a")

    # Panel b: SNR by Tier
    ax = axes[0, 1]
    d1 = snr[tier == 1]
    d0 = snr[tier == 0]
    ax.hist(
        [d1[np.isfinite(d1)], d0[np.isfinite(d0)]],
        bins=25,
        label=["Well-constrained", "Physics-extrap."],
        color=["#1f4e79", "#bbbbbb"],
        alpha=0.85,
    )
    ax.axvline(1.96, color="k", ls="--", lw=0.9, label="SNR = 1.96")
    ax.set_xlim(0, 6)
    ax.set_xlabel("SNR", fontsize=fs.SZ_LABEL)
    ax.set_ylabel("Cells", fontsize=fs.SZ_LABEL)
    ax.legend(frameon=False, fontsize=fs.SZ_ANNOT)
    ax.set_title("Signal-to-Noise Ratio by Tier", fontsize=fs.SZ_TITLE)
    ax.spines[["top", "right"]].set_visible(False)
    fs.panel_letter(ax, "b")

    # Panel c: Bivariate hexbin (|trend| vs sigma_total)
    ax = axes[1, 0]
    valid_pts = np.isfinite(tr) & np.isfinite(sig)
    hb = ax.hexbin(
        np.abs(tr[valid_pts]),
        sig[valid_pts],
        gridsize=35,
        cmap="YlGnBu",
        mincnt=1,
        bins="log",
    )
    ax.plot([0, 3], [0, 3 / 1.96], color="#e74c3c", ls="--", lw=1.0, label="SNR = 1.96 threshold")
    ax.set_xlim(0, float(np.nanpercentile(np.abs(tr), 98)))
    ax.set_ylim(0, float(np.nanpercentile(sig, 98)))
    ax.set_xlabel(r"|GWSA Trend| (cm yr$^{-1}$)", fontsize=fs.SZ_LABEL)
    ax.set_ylabel(r"Total Uncertainty $\sigma$ (cm yr$^{-1}$)", fontsize=fs.SZ_LABEL)
    ax.set_title("Trend Magnitude vs Uncertainty", fontsize=fs.SZ_TITLE)
    ax.legend(frameon=False, loc="upper left", fontsize=fs.SZ_TICK)
    ax.spines[["top", "right"]].set_visible(False)
    cb = fig3.colorbar(hb, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label(r"$\log_{10}$ Cell Count", fontsize=fs.SZ_TICK - 0.5)
    fs.panel_letter(ax, "c")

    # Panel d: Dominant component breakdown by Tier
    ax = axes[1, 1]
    dom_t1 = [np.mean(dom[tier == 1] == k) * 100 for k in range(3)]
    dom_t0 = [np.mean(dom[tier == 0] == k) * 100 for k in range(3)]
    x_bar = np.arange(3)
    w_bar = 0.35
    ax.bar(x_bar - w_bar / 2, dom_t1, width=w_bar, label="Well-constrained", color="#1f4e79", alpha=0.85)
    ax.bar(x_bar + w_bar / 2, dom_t0, width=w_bar, label="Physics-extrap.", color="#bbbbbb", alpha=0.85)
    ax.set_xticks(x_bar)
    ax.set_xticklabels(["Obs", "Model", r"$\mathrm{S}_y$"], fontsize=fs.SZ_LABEL)
    ax.set_ylabel("Cells (%)", fontsize=fs.SZ_LABEL)
    ax.set_title("Dominant Error Component by Tier", fontsize=fs.SZ_TITLE)
    ax.legend(frameon=False, fontsize=fs.SZ_ANNOT)
    ax.spines[["top", "right"]].set_visible(False)
    fs.panel_letter(ax, "d")

    fig3.tight_layout()
    fs.save(fig3, "fig7_03_components_and_tier_diagnostics", figdir=FIGDIR)

    # --------------------------------------------------------------------------
    # FIGURE 4: Empirical Variogram & Spatial Correlation Model
    # --------------------------------------------------------------------------
    fig4, ax = plt.subplots(figsize=(5.4, 3.6))
    valid_v = np.isfinite(semiv)
    ax.plot(centers[valid_v], semiv[valid_v], "o-", color="#1f4e79", ms=4, lw=1.2, label="Empirical Semivariance")
    ax.axhline(sill, color="#c0392b", ls="--", lw=1.0, label=f"Sill ({sill:.3f})")
    ax.axvline(rng_deg, color="#2ca02c", ls=":", lw=1.2, label=rf"Correlation Range ({rng_deg:.1f}$^\circ$)")
    ax.set_xlabel("Spatial Distance (degrees)", fontsize=fs.SZ_LABEL)
    ax.set_ylabel(r"Semivariance $\gamma(h)$", fontsize=fs.SZ_LABEL)
    ax.set_title("Trend Spatial Empirical Variogram", fontsize=fs.SZ_TITLE)
    ax.grid(alpha=0.25, lw=0.4)
    ax.legend(frameon=False, fontsize=fs.SZ_ANNOT)
    ax.spines[["top", "right"]].set_visible(False)
    fs.panel_letter(ax, "a")
    fig4.tight_layout()
    fs.save(fig4, "fig7_04_spatial_variogram", figdir=FIGDIR)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()