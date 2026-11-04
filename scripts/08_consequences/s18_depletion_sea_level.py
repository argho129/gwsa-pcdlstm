"""
Phase 9 - Depletion volume quantification + sea-level-equivalent
Arpon Sarkar, URP, KUET (ID 2017055)

Input : OUTPUTS\\08_final_product\\GWSA_KUET_v1_1D.nc
Output: OUTPUTS\\09_depletion_sle\\depletion_summary.json
        OUTPUTS\\09_depletion_sle\\depletion_cells.csv

METHOD
------
1. Isolate "depletion cells": keep==1, trend significant, gwsa_trend < 0.
2. Convert trend (cm/yr equivalent water height) -> volume (km3/yr == Gt/yr,
   since freshwater density ~1000 kg/m3) using latitude-weighted cell area.
3. Aggregate two ways:
     - GROSS depletion  = sum over depleting cells only (severity/extent metric)
     - NET storage change = sum over ALL valid cells, pos+neg (the physically
       correct quantity for a sea-level-equivalent budget, since recharging
       cells partly offset depleting ones)
4. Propagate uncertainty two ways:
     a. Naive quadrature (assumes independent cell errors) -> LOWER BOUND,
        will understate true uncertainty because GRACE trend errors are
        spatially correlated.
     b. Block-correlated: bin cells into ~6-deg blocks (matches the Phase 7
        MC spatial_range_deg), sum sigma linearly within a block (full
        correlation assumed), combine blocks in quadrature (independence
        assumed across blocks). Still approximate.
     c. BEST (not run here): if the 200-member MC trend ensemble is retained
        from Phase 7, recompute total volume for each realization and take
        the std/percentiles of that distribution directly - this preserves
        the true spatial correlation structure exactly as generated.
5. Convert to sea-level-equivalent (SLE) rate:
       1 mm SLE = ocean_area(m2) * 0.001 m / 1e9 m3/km3 = 361.8 Gt
   (ocean area = 361.8 x 10^6 km2, Cogley 2012 - the standard value used in
   GRACE-based groundwater-depletion / sea-level literature, e.g. Konikow
   2011, Wada et al. 2010.)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# ---------------------------------------------------------------- paths ---
IN_PATH = Path(r"D:\THESIS\Arpon_Thesis_Final\OUTPUTS\08_final_product\GWSA_KUET_v1_1D.nc")
OUT_DIR = Path(r"D:\THESIS\Arpon_Thesis_Final\OUTPUTS\09_depletion_sle")
OUT_DIR.mkdir(parents=True, exist_ok=True)

R_EARTH_KM = 6371.0
CELL_DEG = 0.5
OCEAN_AREA_KM2 = 361.8e6
GT_PER_MM_SLE = OCEAN_AREA_KM2 * 1e6 * 1e-3 / 1e9  # = 361.8

# --------------------------------------------------- flexible var lookup ---
# Phase 8 may have renamed fields to CF-compliant names (e.g. gwsa_calibrated
# -> gwsa). This tries the likely candidates so the script doesn't break on
# a naming mismatch - PRINT ds.data_vars once and confirm before trusting output.
def pick(ds, *names):
    for n in names:
        if n in ds.data_vars or n in ds.coords:
            return ds[n]
    raise KeyError(f"None of {names} found. Available: {list(ds.data_vars) + list(ds.coords)}")


ds = xr.open_dataset(IN_PATH)
print("Variables in file:", list(ds.data_vars))
print("Coords in file:", list(ds.coords))

trend = pick(ds, "gwsa_trend", "trend")
sigma_total = pick(ds, "sigma_total")
significant = pick(ds, "trend_significant", "significant_full")
clat = pick(ds, "clat", "lat")
clon = pick(ds, "clon", "lon")
tier = ds["tier"] if "tier" in ds.data_vars or "tier" in ds.coords else None

# Not all packaged files carry an explicit 'keep' flag - if Phase 8 already
# dropped invalid cells before writing, every cell in this file IS valid.
# Deriving validity from non-null trend/sigma works correctly either way
# (whether invalid cells were dropped entirely, or kept as NaN placeholders).
n_total = int(trend.sizes.get("cell", trend.size))
valid = trend.notnull() & sigma_total.notnull()
if "keep" in ds.data_vars or "keep" in ds.coords:
    keep = ds["keep"]
    valid = valid & (keep == 1)
    print("'keep' variable found - using it as an additional validity filter.")
else:
    print(f"No 'keep' variable in this file - treating non-null trend/sigma "
          f"cells as valid ({n_total} total cells in file).")

n_valid = int(valid.sum())
print(f"Valid cells: {n_valid} / {n_total}")

# ---------------------------------------------------- cell area (lat-wt) ---
lat_rad = np.deg2rad(clat)
cell_area_km2 = (R_EARTH_KM ** 2) * np.deg2rad(CELL_DEG) * np.deg2rad(CELL_DEG) * np.cos(lat_rad)

# cm/yr * km2 * 1e-5 = km3/yr (== Gt/yr for freshwater)
vol_trend = trend * cell_area_km2 * 1e-5          # signed, km3/yr per cell
vol_sigma = sigma_total * cell_area_km2 * 1e-5     # km3/yr per cell

# ------------------------------------------------------- depletion mask ---
depl_mask = valid & (significant == 1) & (trend < 0)
n_depl = int(depl_mask.sum())

gross_depletion_km3_yr = -float(vol_trend.where(depl_mask).sum())   # positive magnitude
net_storage_km3_yr = float(vol_trend.where(valid).sum())            # signed

# ----------------------------------------------- uncertainty: naive (lb) ---
sigma_naive_depl = float(np.sqrt((vol_sigma.where(depl_mask) ** 2).sum()))
sigma_naive_net = float(np.sqrt((vol_sigma.where(valid) ** 2).sum()))


# --------------------------------------- uncertainty: block-correlated -----
def block_correlated_sigma(mask, block_deg=6.0):
    sig = vol_sigma.where(mask).values
    la = clat.values
    lo = clon.values
    ok = ~np.isnan(sig)
    df = pd.DataFrame({
        "sigma": sig[ok],
        "lat_bin": (la[ok] // block_deg).astype(int),
        "lon_bin": (lo[ok] // block_deg).astype(int),
    })
    block_sums = df.groupby(["lat_bin", "lon_bin"])["sigma"].sum()  # full corr. within block
    return float(np.sqrt((block_sums ** 2).sum()))                 # independence across blocks


sigma_block_depl = block_correlated_sigma(depl_mask)
sigma_block_net = block_correlated_sigma(valid)


# ------------------------------------------------------- sea-level equiv ---
def to_sle(vol_km3_yr, sigma_km3_yr):
    return vol_km3_yr / GT_PER_MM_SLE, sigma_km3_yr / GT_PER_MM_SLE


sle_gross, sle_gross_sigma = to_sle(gross_depletion_km3_yr, sigma_block_depl)
sle_net, sle_net_sigma = to_sle(abs(net_storage_km3_yr), sigma_block_net)

# ------------------------------------------------------------- report ------
summary = {
    "n_valid_cells": n_valid,
    "n_depleting_cells": n_depl,
    "gross_depletion_km3_per_yr": round(gross_depletion_km3_yr, 2),
    "gross_depletion_sigma_naive_lowerbound": round(sigma_naive_depl, 2),
    "gross_depletion_sigma_block_correlated": round(sigma_block_depl, 2),
    "net_storage_change_km3_per_yr": round(net_storage_km3_yr, 2),
    "net_storage_sigma_naive_lowerbound": round(sigma_naive_net, 2),
    "net_storage_sigma_block_correlated": round(sigma_block_net, 2),
    "sea_level_equivalent_gross_mm_per_yr": round(sle_gross, 4),
    "sea_level_equivalent_gross_sigma_mm_per_yr": round(sle_gross_sigma, 4),
    "sea_level_equivalent_net_mm_per_yr": round(sle_net, 4),
    "sea_level_equivalent_net_sigma_mm_per_yr": round(sle_net_sigma, 4),
    "method_notes": (
        "Gross depletion = sum over cells with significant negative trend only "
        "(extent/severity of aquifer loss). Net storage change = sum over ALL "
        "valid cells, pos+neg (physically correct quantity for SLE budgets, "
        "since recharging cells partly offset depleting ones). Naive sigma "
        "assumes independent cell errors -> lower bound only. Block-correlated "
        "sigma bins cells at the same 6-deg correlation length used in the "
        "Phase 7 Monte Carlo and is a closer (still approximate) estimate. "
        "For the rigorous number, replay the saved n_mc=200 trend realizations "
        "from Phase 7, if retained, and take the empirical std/CI of total "
        "volume across the ensemble."
    ),
}

with open(OUT_DIR / "depletion_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))

# --------------------------------------------- save the depletion layer ---
depl_df = pd.DataFrame({
    "clat": clat.where(depl_mask).values,
    "clon": clon.where(depl_mask).values,
    "gwsa_trend_cm_yr": trend.where(depl_mask).values,
    "sigma_total_cm_yr": sigma_total.where(depl_mask).values,
    "vol_trend_km3_yr": vol_trend.where(depl_mask).values,
})
if tier is not None:
    depl_df["tier"] = tier.where(depl_mask).values

depl_df = depl_df.dropna(subset=["gwsa_trend_cm_yr"])
depl_df.to_csv(OUT_DIR / "depletion_cells.csv", index=False)
print(f"\nSaved depletion layer ({len(depl_df)} cells) -> {OUT_DIR / 'depletion_cells.csv'}")