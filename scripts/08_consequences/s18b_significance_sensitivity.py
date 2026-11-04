"""
Phase 9b - Sensitivity diagnostic: significance-filter effect on gross depletion
Arpon Sarkar, URP, KUET (ID 2017055)

Purpose: quantify how much of the difference between the gross depletion
estimate of this study and the well-based estimate of Sun et al. (2026)
(550 +/- 73 km3/yr) is explained by the trend_significant filter dropping
real-but-noisy depleting cells.

Input : OUTPUTS\\08_final_product\\GWSA_KUET_v1_1D.nc
"""

from pathlib import Path

import numpy as np
import xarray as xr

IN_PATH = Path(r"D:\THESIS\Arpon_Thesis_Final\OUTPUTS\08_final_product\GWSA_KUET_v1_1D.nc")
SUN_ET_AL_2026_KM3_YR = 550.0
SUN_ET_AL_2026_SIGMA = 73.0


def pick(ds, *names):
    for n in names:
        if n in ds.data_vars or n in ds.coords:
            return ds[n]
    raise KeyError(f"None of {names} found. Available: {list(ds.data_vars) + list(ds.coords)}")


ds = xr.open_dataset(IN_PATH)
trend = pick(ds, "gwsa_trend", "trend")
sigma_total = pick(ds, "sigma_total")
significant = pick(ds, "trend_significant", "significant_full")
clat = pick(ds, "clat", "lat")
clon = pick(ds, "clon", "lon")
tier = ds["tier"] if "tier" in ds.data_vars or "tier" in ds.coords else None

R_EARTH_KM = 6371.0
CELL_DEG = 0.5
lat_rad = np.deg2rad(clat)
cell_area_km2 = (R_EARTH_KM ** 2) * np.deg2rad(CELL_DEG) * np.deg2rad(CELL_DEG) * np.cos(lat_rad)
vol_trend = trend * cell_area_km2 * 1e-5  # km3/yr per cell

valid = trend.notnull() & sigma_total.notnull()

depl_mask_sig = valid & (significant == 1) & (trend < 0)
depl_mask_nosig = valid & (trend < 0)

gross_sig = -float(vol_trend.where(depl_mask_sig).sum())
gross_nosig = -float(vol_trend.where(depl_mask_nosig).sum())

print(f"Gross depletion, WITH significance filter:    {gross_sig:.2f} km3/yr  ({int(depl_mask_sig.sum())} cells)")
print(f"Gross depletion, WITHOUT significance filter: {gross_nosig:.2f} km3/yr  ({int(depl_mask_nosig.sum())} cells)")
print(f"Difference attributable to significance filtering: {gross_nosig - gross_sig:.2f} km3/yr")
print()
print(f"vs Sun et al. 2026 ({SUN_ET_AL_2026_KM3_YR} +/- {SUN_ET_AL_2026_SIGMA} km3/yr):")
print(f"  WITH filter:    {100 * gross_sig / SUN_ET_AL_2026_KM3_YR:.1f}% of their estimate")
print(f"  WITHOUT filter: {100 * gross_nosig / SUN_ET_AL_2026_KM3_YR:.1f}% of their estimate")

if tier is not None:
    print("\nBreakdown by tier (no-significance-filter mask):")
    tier_vals = tier.where(valid).values
    tier_ints = sorted(set(tier_vals[~np.isnan(tier_vals)].astype(int)))
    for t in tier_ints:
        mask_t = depl_mask_nosig & (tier == t)
        vol_t = -float(vol_trend.where(mask_t).sum())
        print(f"  tier={t}: {vol_t:.2f} km3/yr  ({int(mask_t.sum())} cells)")