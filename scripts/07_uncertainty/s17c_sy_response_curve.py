#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
PHASE 7.2 -- Sy RESPONSE CURVE + CORRELATION-AREA AGGREGATION
================================================================================
Post-process only. Reads the existing Phase 7.0 output (gwsa_uncertainty.nc).
No Monte Carlo re-run: Sy enters analytically, so everything below is arithmetic
on the already-saved per-cell fields (trend, sigma_obs, sigma_model, tier).

Produces three objects:
  (A) Dense cell-level response curve   -> shows the significance cliff & where
      the adopted imputed Sy error (0.40) sits relative to the 1/1.96 tipping pt.
  (B) Correlation-area aggregation      -> blocks sized to the variogram range
      (~6 deg). Significance re-evaluated at basin/correlation scale, which is
      the physically meaningful question and far less sensitive to Sy error.
  (C) A block-level significance map at the adopted 0.40 (conservative bound).

Aggregation uses TWO explicit, stated Sy-error correlation bounds per block:
  * CONSERVATIVE : Sy relative error fully spatially correlated within a block
                   -> no cancellation. sigma_sy_block = |T_block| * mean_relerr.
  * INDEPENDENT  : Sy relative error independent cell-to-cell within a block
                   -> averages down. sigma_sy_block from summed cell variances.
Obs+model error is kept fully correlated within a block in BOTH bounds (its
correlation length ~= the block size), so the only thing that differs between
the two curves is the Sy assumption -- clean isolation of the variable in
question. The truth is bracketed by these two curves.

Outputs (into the 07_uncertainty folder):
  sy_response_curve.csv                -> data behind the curve figure
  sy_block_significance_p040.csv       -> per-block table at adopted 0.40
  fig7_05_sy_response_curve.png/.pdf   -> Figure A.12a (data: Table A.1)
  fig7_06_block_significance_map.png/.pdf -> Figure A.12b
================================================================================
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# CONFIG -- edit only this block
# ----------------------------------------------------------------------------
UNC_NC = r"D:\THESIS\Arpon_Thesis_Final\OUTPUTS\07_uncertainty\gwsa_uncertainty.nc"
OUTDIR = r"D:\THESIS\Arpon_Thesis_Final\OUTPUTS\07_uncertainty"

SNR_THRESHOLD = 1.96          # significance threshold (matches Phase 7.0)
MEAS_ERR_FIXED = 0.15         # measured-Sy error (held fixed; sweep showed it is inert)
ADOPTED_IMPUTED = 0.40        # the value used in Phase 7.0
LIT_RANGE = (0.30, 0.45)      # literature range for proxy Sy error (shaded on plot)

# dense sweep of the imputed Sy relative error
IMPUTED_SWEEP = np.round(np.arange(0.15, 0.76, 0.01), 2)

# correlation-area block size (deg). Phase 7.0 variogram range ~= 6 deg.
BLOCK_DEG = 6.0
MIN_CELLS_PER_BLOCK = 5       # ignore tiny edge blocks
# ----------------------------------------------------------------------------

TIP = 1.0 / SNR_THRESHOLD      # ~0.51: imputed Sy error where Sy-dominated cells flip


# ----------------------------------------------------------------------------
# Load per-cell fields
# ----------------------------------------------------------------------------
def load_cells(path):
    ds = xr.open_dataset(path)
    ds.load()
    trend = np.asarray(ds["trend"].values, dtype="float64")
    sig_obs = np.asarray(ds["sigma_obs"].values, dtype="float64")
    sig_model = np.asarray(ds["sigma_model"].values, dtype="float64")
    tier = np.asarray(ds["tier"].values, dtype="float64")   # 1 = measured Sy
    clat = np.asarray(ds["clat"].values, dtype="float64")
    clon = np.asarray(ds["clon"].values, dtype="float64")
    ds.close()

    sig_om = np.sqrt(sig_obs ** 2 + sig_model ** 2)         # obs+model, Sy-independent
    valid = (np.isfinite(trend) & np.isfinite(sig_om)
             & np.isfinite(tier) & np.isfinite(clat) & np.isfinite(clon))

    df = pd.DataFrame({
        "trend": trend[valid],
        "sig_om": sig_om[valid],
        "measured": tier[valid] >= 0.5,
        "clat": clat[valid],
        "clon": clon[valid],
    })
    df["absT"] = df["trend"].abs()
    df["w"] = np.clip(np.cos(np.deg2rad(df["clat"])), 0.0, None)   # area weight
    return df


# ----------------------------------------------------------------------------
# (A) cell-level significance at a given imputed Sy error
# ----------------------------------------------------------------------------
def cell_significance(df, imp_err):
    relerr = np.where(df["measured"], MEAS_ERR_FIXED, imp_err)
    sig_sy = df["absT"].values * relerr
    sig_tot = np.sqrt(df["sig_om"].values ** 2 + sig_sy ** 2)
    snr = df["absT"].values / np.maximum(sig_tot, 1e-6)
    sig = snr >= SNR_THRESHOLD
    w = df["w"].values
    return {
        "pct_count": 100.0 * sig.mean(),
        "pct_area": 100.0 * (w[sig].sum() / w.sum()),
        "median_sigma": float(np.median(sig_tot)),
    }


# ----------------------------------------------------------------------------
# (B) correlation-area aggregation -- precompute block sums once
# ----------------------------------------------------------------------------
def build_blocks(df):
    bi = np.floor((df["clat"].values + 90.0) / BLOCK_DEG).astype(int)
    bj = np.floor((df["clon"].values + 180.0) / BLOCK_DEG).astype(int)
    df = df.copy()
    df["block"] = bi * 100000 + bj

    w = df["w"].values
    T = df["trend"].values
    absT = df["absT"].values
    som = df["sig_om"].values
    meas = df["measured"].values

    g = pd.DataFrame({
        "block": df["block"].values,
        "w": w,
        "wT": w * T,
        "wom": w * som,                       # obs+model: correlated within block
        "w_meas": np.where(meas, w, 0.0),      # for mean-relerr (conservative)
        "w_imp": np.where(~meas, w, 0.0),
        "wsqT2_meas": np.where(meas, (w * absT) ** 2, 0.0),  # for independent bound
        "wsqT2_imp": np.where(~meas, (w * absT) ** 2, 0.0),
        "wlat": w * df["clat"].values,
        "wlon": w * df["clon"].values,
    }).groupby("block").sum()

    n_cells = df.groupby("block").size()
    g["n_cells"] = n_cells
    g = g[g["n_cells"] >= MIN_CELLS_PER_BLOCK].copy()

    g["T_b"] = g["wT"] / g["w"]
    g["absT_b"] = g["T_b"].abs()
    g["sig_om_b"] = g["wom"] / g["w"]          # weighted-mean (fully correlated)
    g["lat_b"] = g["wlat"] / g["w"]
    g["lon_b"] = g["wlon"] / g["w"]
    return g


def block_significance(g, imp_err, mode):
    """mode: 'conservative' (Sy correlated) or 'independent' (Sy uncorrelated)."""
    if mode == "conservative":
        mean_relerr = (MEAS_ERR_FIXED * g["w_meas"] + imp_err * g["w_imp"]) / g["w"]
        sig_sy_b = g["absT_b"].values * mean_relerr.values
    elif mode == "independent":
        var_sy = (MEAS_ERR_FIXED ** 2 * g["wsqT2_meas"]
                  + imp_err ** 2 * g["wsqT2_imp"]) / (g["w"] ** 2)
        sig_sy_b = np.sqrt(var_sy.values)
    else:
        raise ValueError(mode)

    sig_tot_b = np.sqrt(g["sig_om_b"].values ** 2 + sig_sy_b ** 2)
    snr_b = g["absT_b"].values / np.maximum(sig_tot_b, 1e-6)
    sig = snr_b >= SNR_THRESHOLD
    w = g["w"].values
    return {
        "pct_count": 100.0 * sig.mean(),
        "pct_area": 100.0 * (w[sig].sum() / w.sum()),
        "snr_b": snr_b, "sig": sig, "sig_sy_b": sig_sy_b, "sig_tot_b": sig_tot_b,
    }


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    if not os.path.exists(UNC_NC):
        raise FileNotFoundError(UNC_NC)
    os.makedirs(OUTDIR, exist_ok=True)

    df = load_cells(UNC_NC)
    g = build_blocks(df)
    print(f"Loaded {len(df)} cells; formed {len(g)} correlation-area blocks "
          f"({BLOCK_DEG:.0f} deg), median {int(g['n_cells'].median())} cells/block.\n")

    # ---- dense sweep for the three curves
    rows = []
    for imp in IMPUTED_SWEEP:
        c = cell_significance(df, imp)
        bc = block_significance(g, imp, "conservative")
        bi = block_significance(g, imp, "independent")
        rows.append({
            "imputed_err": imp,
            "cell_pct_count": c["pct_count"],
            "cell_pct_area": c["pct_area"],
            "cell_median_sigma": c["median_sigma"],
            "block_cons_pct_area": bc["pct_area"],
            "block_cons_pct_count": bc["pct_count"],
            "block_indep_pct_area": bi["pct_area"],
            "block_indep_pct_count": bi["pct_count"],
        })
    curve = pd.DataFrame(rows)
    curve.to_csv(os.path.join(OUTDIR, "sy_response_curve.csv"), index=False)

    # ---- per-block table at the adopted 0.40 (conservative bound)
    bc0 = block_significance(g, ADOPTED_IMPUTED, "conservative")
    bi0 = block_significance(g, ADOPTED_IMPUTED, "independent")
    block_tbl = pd.DataFrame({
        "lat": g["lat_b"].values, "lon": g["lon_b"].values,
        "n_cells": g["n_cells"].values, "trend_cm_yr": g["T_b"].values,
        "sig_om_cm_yr": g["sig_om_b"].values,
        "sig_sy_cons": bc0["sig_sy_b"], "sig_total_cons": bc0["sig_tot_b"],
        "snr_cons": bc0["snr_b"], "significant_cons": bc0["sig"],
        "snr_indep": bi0["snr_b"], "significant_indep": bi0["sig"],
    }).sort_values("trend_cm_yr")
    block_tbl.to_csv(os.path.join(OUTDIR, "sy_block_significance_p040.csv"), index=False)

    # ---- key numbers at adopted 0.40
    at = curve.iloc[(curve["imputed_err"] - ADOPTED_IMPUTED).abs().argmin()]
    print("AT ADOPTED IMPUTED Sy ERROR = 0.40")
    print("-" * 60)
    print(f"  cell-level significant      : {at['cell_pct_count']:5.1f}% of cells "
          f"({at['cell_pct_area']:.1f}% of area)")
    print(f"  block conservative bound    : {at['block_cons_pct_area']:5.1f}% of area")
    print(f"  block independent bound     : {at['block_indep_pct_area']:5.1f}% of area")
    print(f"  tipping point (1/1.96)      : imputed Sy error ~ {TIP:.3f}\n")

    # ---- FIGURE 1: response curves
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.axvspan(LIT_RANGE[0], LIT_RANGE[1], color="#2ecc71", alpha=0.12,
               label=f"Literature range ({LIT_RANGE[0]:.2f}-{LIT_RANGE[1]:.2f})")
    ax.axvline(ADOPTED_IMPUTED, color="#2c3e50", lw=1.3, ls="-",
               label=f"Adopted ({ADOPTED_IMPUTED:.2f})")
    ax.axvline(TIP, color="#7f8c8d", lw=1.0, ls=":",
               label=f"Cell tipping point ({TIP:.2f})")
    ax.plot(curve["imputed_err"], curve["cell_pct_count"], color="#c0392b",
            lw=1.8, label="Cell level (0.5 deg)")
    ax.plot(curve["imputed_err"], curve["block_cons_pct_area"], color="#1f4e79",
            lw=1.8, label="Basin scale - Sy correlated (conservative)")
    ax.plot(curve["imputed_err"], curve["block_indep_pct_area"], color="#1f4e79",
            lw=1.8, ls="--", label="Basin scale - Sy independent")
    ax.set_xlabel("Imputed specific-yield relative error")
    ax.set_ylabel("Significant (%)   [cells: count; basins: area]")
    ax.set_ylim(0, 102)
    ax.set_title("Significance vs specific-yield error assumption")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUTDIR, f"fig7_05_sy_response_curve.{ext}"), dpi=300)
    plt.close(fig)

    # ---- FIGURE 2: block significance map at 0.40 (conservative)
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    sig = bc0["sig"]
    ax.scatter(g["lon_b"].values[~sig], g["lat_b"].values[~sig], s=14,
               c="#ddddcc"[:7], marker="s", label="Not significant", edgecolors="none")
    sc = ax.scatter(g["lon_b"].values[sig], g["lat_b"].values[sig], s=16,
                    c=g["T_b"].values[sig], cmap="RdBu", marker="s",
                    vmin=-np.nanpercentile(np.abs(g["T_b"]), 95),
                    vmax=np.nanpercentile(np.abs(g["T_b"]), 95),
                    label="Significant", edgecolors="none")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Correlation-area blocks significant at imputed Sy = {ADOPTED_IMPUTED:.2f} "
                 f"(conservative bound)")
    cb = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("Block trend (cm yr$^{-1}$)")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUTDIR, f"fig7_06_block_significance_map.{ext}"), dpi=300)
    plt.close(fig)

    print("Saved: sy_response_curve.csv, sy_block_significance_p040.csv,")
    print("       fig7_05_sy_response_curve.(png/pdf), fig7_06_block_significance_map.(png/pdf)")


if __name__ == "__main__":
    main()
