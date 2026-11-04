           # -*- coding: utf-8 -*-
r"""
================================================================================
Data_Figures_Part2.py — Supplementary Publication Figures for dataset_full.nc
================================================================================
"Aquifer to Atmosphere" | Arpon, KUET
Style: Matched exactly to Fig 02 & Fig 09 style (smooth Robinson maps, no lat/lon ticks)

New Figures:
  16   Net Water Input & Flux Global Panels (P-R and P-E-R Means & Trends)
  17   Focus Basins Spatial Domain & Classification Map (Smooth Discrete Field)
  18   Dynamic Predictors Global Mean Panels (6-panel matched to Fig 09)
  19   Remaining Static & Uncertainty Fields (Slope, GWSA Uncertainty, NDVI Obs Count)

Run alongside Data_Figures.py using the same environment and fig_style.py.
================================================================================
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig_style as fs

fs.apply()

DS = r"D:\THESIS\Arpon_Thesis_Final\OUTPUTS\10_model\data\dataset_full.nc"
d = xr.open_dataset(DS)
TIME = pd.to_datetime(d["time"].values)
foc = d["focus_name"].values.astype(str)
BASINS = ["Bengal_GBM", "Punjab_Indus", "HighPlains", "CentralValley", "NorthChinaPlain"]


def hr(t): print("\n" + "=" * 66 + f"\n{t}\n" + "=" * 66)


# ============================================================ FIG 16: NET WATER INPUT (P-R) & FLUX PANELS
def fig16_net_flux_panels():
    hr("FIG 16  Net Water Input (P-R) and Net Availability (P-E-R) Means & Trends")
    P = d["P"].values
    E = d["E"].values
    R = d["R"].values

    P_minus_R = P - R
    P_minus_ER = P - E - R

    # Time array for trends
    tyr = (TIME - TIME[0]).days.values / 365.25
    A = np.vstack([tyr, np.ones_like(tyr)]).T

    # 1. Means
    pr_mean = np.nanmean(P_minus_R, axis=0)
    per_mean = np.nanmean(P_minus_ER, axis=0)

    # 2. Trends
    coef_pr, *_ = np.linalg.lstsq(A, P_minus_R, rcond=None)
    pr_trend = coef_pr[0]

    coef_per, *_ = np.linalg.lstsq(A, P_minus_ER, rcond=None)
    per_trend = coef_per[0]

    specs = [
        (pr_mean, "Net Moisture Input $P-R$ Mean (mm mo$^{-1}$)", "YlGnBu", 0, 150, False),
        (pr_trend, "$P-R$ Trend (mm mo$^{-1}$ yr$^{-1}$)", "BrBG", -2.0, 2.0, True),
        (per_mean, "Net Water Availability $P-E-R$ Mean (mm mo$^{-1}$)", "PuOr", -50, 50, False),
        (per_trend, "$P-E-R$ Trend (mm mo$^{-1}$ yr$^{-1}$)", "BrBG", -1.5, 1.5, True),
    ]

    fig = plt.figure(figsize=(7.6, 6.4))
    letters = "abcd"

    for k, (data_arr, lab, cmap, vmin, vmax, is_smooth) in enumerate(specs):
        row = k // 2; col = k % 2
        x0 = 0.05 + col * 0.50
        y0 = 0.55 - row * 0.44
        ax = fs.global_map_axis(fig, (x0, y0, 0.42, 0.34), projection="robinson")

        g = fs.cell_reduce_to_grid(d, data_arr)
        nlat, nlon = g.shape

        if is_smooth:
            m = fs.pcolor_global_smooth(ax, g, nlat, nlon, cmap, vmin, vmax, upsample=3)
        else:
            m = fs.pcolor_global(ax, g, cmap, vmin, vmax)

        fs.panel_letter(ax, letters[k])
        cax = fig.add_axes([x0 + 0.02, y0 - 0.03, 0.36, 0.012])
        cb = fig.colorbar(m, cax=cax, orientation="horizontal")
        cb.set_label(lab, fontsize=fs.SZ_CBAR - 0.5)
        cb.ax.tick_params(labelsize=fs.SZ_TICK - 0.5, width=0.4, length=1.8)
        cb.outline.set_linewidth(0.4)

    fs.save(fig, "fig16_net_flux_panels")


# ============================================================ FIG 17: FOCUS BASINS GEOGRAPHICAL EXTENT MAP
def fig17_focus_basins_map():
    hr("FIG 17  Focus Basins Geographical Distribution Map")

    # Construct discrete categorical raster for focus basins
    basin_ids = np.zeros(d.dims["cell"], dtype=float)
    for idx, b in enumerate(BASINS, start=1):
        basin_ids[foc == b] = idx

    basin_ids[basin_ids == 0] = np.nan
    g_basins = fs.cell_reduce_to_grid(d, basin_ids)

    # Discrete color palette for 5 basins
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
    cmap_b = mcolors.ListedColormap(colors)
    bounds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    norm_b = mcolors.BoundaryNorm(bounds, cmap_b.N)

    fig = plt.figure(figsize=(7.6, 4.8))
    ax = fs.global_map_axis(fig, (0.04, 0.18, 0.92, 0.76), projection="robinson")

    m = fs.pcolor_global(ax, g_basins, cmap_b, vmin=0.5, vmax=5.5)
    ax.set_title("Global Aquifer Focus Study Basins Domain", fontsize=fs.SZ_TITLE)
    fs.panel_letter(ax, "a")

    # Custom discrete legend below map
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(len(BASINS))]
    labels = [fs.BASIN_LABEL[b] for b in BASINS]
    ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.15),
              ncol=3, frameon=False, fontsize=fs.SZ_ANNOT)

    fs.save(fig, "fig17_focus_basins_map")


# ============================================================ FIG 18: DYNAMIC PREDICTORS GLOBAL MEAN PANELS
def fig18_dynamic_means():
    hr("FIG 18  Dynamic Predictors Global Mean Panels (6-panel)")
    specs = [
        ("P", "mean P (mm mo$^{-1}$)", "YlGnBu", 0, 200),
        ("E", "mean E (mm mo$^{-1}$)", "YlOrRd", 0, 120),
        ("R", "mean R (mm mo$^{-1}$)", "GnBu", 0, 80),
        ("t2m", "mean $T_{2m}$ (K)", "Spectral_r", 250, 305),
        ("ndvi", "mean NDVI (-)", "YlGn", 0.0, 0.8),
        ("sm", "mean Soil Moisture (-)", "Blues", 0.0, 0.5),
    ]

    fig = plt.figure(figsize=(7.6, 9.4))
    letters = "abcdef"

    for k, (var, lab, cmap, vmin, vmax) in enumerate(specs):
        if var not in d:
            continue
        v_mean = np.nanmean(d[var].values, axis=0)

        row = k // 2; col = k % 2
        x0 = 0.05 + col * 0.50
        y0 = 0.70 - row * 0.335
        ax = fs.global_map_axis(fig, (x0, y0, 0.42, 0.24), projection="robinson")

        g = fs.cell_reduce_to_grid(d, v_mean)
        m = fs.pcolor_global(ax, g, cmap, vmin, vmax)

        fs.panel_letter(ax, letters[k])
        cax = fig.add_axes([x0 + 0.02, y0 - 0.028, 0.36, 0.011])
        cb = fig.colorbar(m, cax=cax, orientation="horizontal")
        cb.set_label(lab, fontsize=fs.SZ_CBAR - 0.5)
        cb.ax.tick_params(labelsize=fs.SZ_TICK - 0.5, width=0.4, length=1.8)
        cb.outline.set_linewidth(0.4)

    fs.save(fig, "fig18_dynamic_means")


# ============================================================ FIG 19: REMAINING STATIC & UNCERTAINTY FIELDS
def fig19_slope_and_uncertainty():
    hr("FIG 19  Slope, GWSA Uncertainty, and NDVI Observation Ratio")

    # 1. Slope
    slope_c = d["slope"].values if "slope" in d else np.full(d.dims["cell"], np.nan)

    # 2. GWSA Uncertainty (Mean Sigma)
    gwsa_sigma_c = np.nanmean(d["gwsa_sigma"].values, axis=0) if "gwsa_sigma" in d else np.full(d.dims["cell"], np.nan)

    # 3. NDVI Observed Fraction
    ndvi_obs_frac = np.mean(d["ndvi_observed"].values, axis=0) if "ndvi_observed" in d else np.full(d.dims["cell"], np.nan)

    specs = [
        (slope_c, "Terrain Slope (degrees)", "magma", 0, 10, False),
        (gwsa_sigma_c, "GWSA Uncertainty $\\sigma$ (cm)", "Purples", 0, 15, False),
        (ndvi_obs_frac * 100, "NDVI Observed Months (%)", "YlGn", 70, 100, False),
    ]

    fig = plt.figure(figsize=(7.6, 9.4))
    letters = "abc"

    for k, (data_arr, lab, cmap, vmin, vmax, is_smooth) in enumerate(specs):
        row = k // 2; col = k % 2
        x0 = 0.05 + col * 0.50
        y0 = 0.70 - row * 0.335
        ax = fs.global_map_axis(fig, (x0, y0, 0.42, 0.24), projection="robinson")

        g = fs.cell_reduce_to_grid(d, data_arr)
        nlat, nlon = g.shape

        if is_smooth:
            m = fs.pcolor_global_smooth(ax, g, nlat, nlon, cmap, vmin, vmax, upsample=3)
        else:
            m = fs.pcolor_global(ax, g, cmap, vmin, vmax)

        fs.panel_letter(ax, letters[k])
        cax = fig.add_axes([x0 + 0.02, y0 - 0.028, 0.36, 0.011])
        cb = fig.colorbar(m, cax=cax, orientation="horizontal")
        cb.set_label(lab, fontsize=fs.SZ_CBAR - 0.5)
        cb.ax.tick_params(labelsize=fs.SZ_TICK - 0.5, width=0.4, length=1.8)
        cb.outline.set_linewidth(0.4)

    # Remove 4th blank panel position
    fig.add_axes([0.55, 0.365, 0.42, 0.24]).axis("off")

    fs.save(fig, "fig19_slope_uncertainty_fields")


# ============================================================ RUN ALL PART 2
if __name__ == "__main__":
    fig16_net_flux_panels()
    fig17_focus_basins_map()
    fig18_dynamic_means()
    fig19_slope_and_uncertainty()
    d.close()
    print("\nSUPPLEMENTARY FIGURES (16-19) GENERATED SUCCESSFULLY.")