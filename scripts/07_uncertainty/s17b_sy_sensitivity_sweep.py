#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
PHASE 7.1 -- SPECIFIC-YIELD ERROR SENSITIVITY SWEEP  (post-process, no MC re-run)
================================================================================
Reads the EXISTING Phase 7.0 output (gwsa_uncertainty.nc) and re-derives the
total trend uncertainty for a grid of Sy relative-error assumptions.

Why no Monte Carlo re-run is needed
-----------------------------------
In Phase 7.0 the expensive spatially-correlated MC produces ONLY the obs+model
trend sigma. Specific yield enters analytically AFTER the MC:
        sig_sy      = |trend| * sy_relerr
        var_total   = sig_obs^2 + sig_model^2 + sig_sy^2
Since sigma_obs, sigma_model and trend are already saved per cell, the whole Sy
sweep is a few array operations. The (measured=0.15, imputed=0.40) row reproduces
the original Phase 7.0 numbers exactly -> built-in sanity check.

Output: a CSV table (sy_sensitivity_table.csv) and a console summary.
================================================================================
"""

import os
import numpy as np
import pandas as pd
import xarray as xr

# ----------------------------------------------------------------------------
# CONFIG  -- edit only this block
# ----------------------------------------------------------------------------
UNC_NC = r"D:\THESIS\Arpon_Thesis_Final\OUTPUTS\07_uncertainty\gwsa_uncertainty.nc"
OUT_CSV = r"D:\THESIS\Arpon_Thesis_Final\OUTPUTS\07_uncertainty\sy_sensitivity_table.csv"

SNR_THRESHOLD = 1.96          # significance threshold used in Phase 7.0
BASELINE = (0.15, 0.40)       # (measured, imputed) used in Phase 7.0

# Sweep grid: measured Sy error x imputed Sy error
MEASURED_GRID = [0.10, 0.15, 0.20]
IMPUTED_GRID = [0.25, 0.40, 0.55]
# extra stress rows (measured, imputed) appended on top of the full grid
STRESS_ROWS = [(0.15, 0.70), (0.20, 0.60)]
# ----------------------------------------------------------------------------


def load_fields(path):
    """Load the per-cell fields needed for the analytic Sy sweep."""
    ds = xr.open_dataset(path)
    ds.load()

    trend = np.asarray(ds["trend"].values, dtype="float64")
    sig_obs = np.asarray(ds["sigma_obs"].values, dtype="float64")
    sig_model = np.asarray(ds["sigma_model"].values, dtype="float64")
    tier = np.asarray(ds["tier"].values, dtype="float64")  # 1 = measured Sy, 0 = imputed Sy

    # obs+model variance is INDEPENDENT of Sy -> reconstruct and freeze it
    var_obsmodel = sig_obs ** 2 + sig_model ** 2

    # valid analysis cells: everything Phase 7.0 kept and could evaluate
    valid = np.isfinite(trend) & np.isfinite(var_obsmodel) & np.isfinite(tier)

    ds.close()
    return {
        "trend": trend[valid],
        "var_obsmodel": var_obsmodel[valid],
        "measured_mask": (tier[valid] >= 0.5),  # True where Sy is measured
        "n_cells": int(valid.sum()),
    }


def evaluate(fields, meas_err, imp_err):
    """Recompute total sigma / SNR / significance for one Sy-error scenario."""
    trend = fields["trend"]
    var_obsmodel = fields["var_obsmodel"]
    meas_mask = fields["measured_mask"]

    sy_relerr = np.where(meas_mask, meas_err, imp_err)
    var_sy = (np.abs(trend) * sy_relerr) ** 2

    var_total = var_obsmodel + var_sy
    sig_total = np.sqrt(var_total)
    snr = np.abs(trend) / np.maximum(sig_total, 1e-6)
    significant = snr >= SNR_THRESHOLD

    # dominant component fractions (obs+model treated as one block here)
    frac_sy = var_sy / np.maximum(var_total, 1e-12)
    sy_dominant = frac_sy > 0.5  # Sy carries >half the variance

    n = trend.size
    return {
        "meas_err": meas_err,
        "imp_err": imp_err,
        "median_sigma_cm_yr": float(np.median(sig_total)),
        "p95_sigma_cm_yr": float(np.percentile(sig_total, 95)),
        "n_significant": int(significant.sum()),
        "pct_significant": 100.0 * significant.mean(),
        "pct_sy_dominant": 100.0 * sy_dominant.mean(),
        "mean_frac_sy": float(np.mean(frac_sy)),
        "n_cells": n,
    }


def main():
    if not os.path.exists(UNC_NC):
        raise FileNotFoundError(f"Cannot find Phase 7.0 output:\n  {UNC_NC}")

    fields = load_fields(UNC_NC)
    print(f"Loaded {fields['n_cells']} analysis cells from:\n  {UNC_NC}\n")

    # build scenario list: full grid + stress rows, de-duplicated, baseline first
    scenarios = []
    for m in MEASURED_GRID:
        for i in IMPUTED_GRID:
            scenarios.append((m, i))
    scenarios += list(STRESS_ROWS)
    # de-dup while preserving order, baseline pulled to the front
    seen, ordered = set(), []
    for s in [BASELINE] + scenarios:
        if s not in seen:
            seen.add(s)
            ordered.append(s)

    rows = [evaluate(fields, m, i) for (m, i) in ordered]
    df = pd.DataFrame(rows)

    # deltas relative to the baseline row
    base = df.iloc[0]
    df["is_baseline"] = [(m == BASELINE[0] and i == BASELINE[1])
                         for m, i in zip(df["meas_err"], df["imp_err"])]
    df["d_pct_significant"] = df["pct_significant"] - base["pct_significant"]

    col_order = ["is_baseline", "meas_err", "imp_err", "median_sigma_cm_yr",
                 "p95_sigma_cm_yr", "n_significant", "pct_significant",
                 "d_pct_significant", "pct_sy_dominant", "mean_frac_sy", "n_cells"]
    df = df[col_order]

    df.to_csv(OUT_CSV, index=False)

    # console table
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    print("SPECIFIC-YIELD ERROR SENSITIVITY SWEEP")
    print("=" * 78)
    print(df.to_string(index=False))
    print("=" * 78)

    b = df[df["is_baseline"]].iloc[0]
    lo, hi = df["pct_significant"].min(), df["pct_significant"].max()
    print(f"\nBaseline (meas={BASELINE[0]}, imp={BASELINE[1]}): "
          f"median sigma = {b['median_sigma_cm_yr']:.3f} cm/yr, "
          f"significant = {b['pct_significant']:.1f}%")
    print(f"Across the FULL sweep, % significant ranges {lo:.1f}% -> {hi:.1f}% "
          f"(spread {hi - lo:.1f} pts).")
    print(f"\nSaved table: {OUT_CSV}")


if __name__ == "__main__":
    main()
