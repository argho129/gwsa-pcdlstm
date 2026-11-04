# -*- coding: utf-8 -*-
r"""
================================================================================
Data_Figures.py  —  Publication figure set for dataset_full.nc
================================================================================
"Aquifer to Atmosphere" | Arpon, KUET
Style: Nature Water conventions via fig_style.py
Output: 17 figures (600 DPI PNG + vector PDF) into OUTPUTS\10_model\figures

Figures
  1    Data completeness matrix
  2    Static predictor global panels (6-panel)
  3    Dynamic climatology (seasonal P/E/R/sm per focus basin)
  4    Collinearity heatmaps (pairwise finite dynamic correlation)
  4b   VIF diagnostics + NetFlux (P-R) collinearity experiment
  5a   GWSA target field: Mean map (bilinear-smoothed)
  5b   GWSA target field: Trend map (smoothed + true-shape masked basin insets + callouts)
  6    Per-basin predictor distributions (violins)
  7    Water-balance residual map
  8    t2m + NDVI climatology per basin
  9    Dynamic predictor global trends (6-panel)
  10   Domain cell spatial density (Gaussian-smoothed)
  11   Orthographic true-globe dual-hemisphere trend field (bilinear-smoothed)
  12   Bivariate density: Target vs Static Predictors (with zoomed insets, sci-notation ticks)
  13   Bivariate density: Target vs Dynamic Predictors (with zoomed insets, sci-notation ticks)
  14   Relation diagnostics: Pearson vs Spearman vs Mutual Information (+ CSV table)
  15   Lagged cross-correlation of dynamic drivers vs GWSA, per basin (-12..+12 months)

Run once. Read-only on the dataset.
================================================================================
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

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


# ============================================================ FIG 1
def fig1_completeness():
    hr("FIG 1  data completeness")
    isf = d["is_filled"].values
    ndo = d["ndvi_observed"].values
    gwsa_obs = 1.0 - isf.mean(axis=1)
    ndvi_obs = ndo.mean(axis=1)

    fig = plt.figure(figsize=(7.2, 4.6))
    ax1 = fig.add_axes([0.10, 0.60, 0.86, 0.30])
    ax1.plot(TIME, gwsa_obs * 100, lw=0.9, color="#08519c", label="GWSA observed")
    ax1.plot(TIME, ndvi_obs * 100, lw=0.9, color="#238b45", label="NDVI observed")
    ax1.set_ylabel("cells observed (%)")
    ax1.set_ylim(70, 101)
    ax1.grid(alpha=0.25, linewidth=0.4)
    ax1.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.02, 1, 0.15),
               mode="expand", ncol=2, borderaxespad=0)
    # Custom offset: the legend already sits right above this axis (bbox_to_anchor
    # pushes it up to ~y=1.17), so the default "outside" panel-letter position
    # would collide with it. Push "a" further up, clear of the legend box.
    fs.panel_letter(ax1, "a", x=-0.02, y=1.22)
    ax1.margins(x=0.01)

    ax2 = fig.add_axes([0.10, 0.10, 0.86, 0.38])
    order = np.argsort(d["clat"].values)
    sub = order[::max(1, len(order) // 600)]
    M = isf[:, sub].T
    im = ax2.imshow(M, aspect="auto", cmap="Greys", vmin=0, vmax=1,
                    extent=[0, len(TIME), 0, len(sub)], interpolation="nearest")
    ax2.set_xlabel("month index (2002-04 .. 2022-12)")
    ax2.set_ylabel("cells (sorted by latitude)")
    fs.panel_letter(ax2, "b")
    cb = fig.colorbar(im, ax=ax2, fraction=0.025, pad=0.01, ticks=[0, 1])
    cb.ax.set_yticklabels(["obs", "fill"], fontsize=fs.SZ_TICK)
    fs.save(fig, "fig01_data_completeness")


# ============================================================ FIG 2
def fig2_static_panels():
    hr("FIG 2  static predictor global panels")
    specs = [
        ("logk_ferr", "log$_{10}$ permeability (m$^2$)", fs.CMAP_PERM, -16.5, -8),
        ("porosity", "porosity (-)", fs.CMAP_FRAC, 0, 0.27),
        ("aei", "irrigated area (%)", fs.CMAP_FRAC, 0, 60),
        ("aeigw", "GW-irrigated fraction (%)", fs.CMAP_FRAC, 0, 100),
        ("urban_frac_mean", "urban fraction (-)", fs.CMAP_FRAC, 0, 0.3),
        ("elevation", "elevation (m)", fs.CMAP_MAG, -200, 5000),
    ]
    fig = plt.figure(figsize=(7.6, 9.4))
    letters = "abcdef"
    for k, (var, lab, cmap, vmin, vmax) in enumerate(specs):
        row = k // 2; col = k % 2
        x0 = 0.05 + col * 0.50
        y0 = 0.70 - row * 0.335
        ax = fs.global_map_axis(fig, (x0, y0, 0.42, 0.24), projection="robinson")
        g = fs.grid_from_cells(d, var)
        m = fs.pcolor_global(ax, g, cmap, vmin, vmax)
        fs.panel_letter(ax, letters[k])
        cax = fig.add_axes([x0 + 0.02, y0 - 0.028, 0.36, 0.011])
        cb = fig.colorbar(m, cax=cax, orientation="horizontal")
        cb.set_label(lab, fontsize=fs.SZ_CBAR - 0.5)
        cb.ax.tick_params(labelsize=fs.SZ_TICK - 0.5, width=0.4, length=1.8)
        cb.outline.set_linewidth(0.4)
    fs.save(fig, "fig02_static_predictors")


# ============================================================ FIG 3
def fig3_climatology():
    hr("FIG 3  dynamic climatology per basin")
    dyn = [("P", "#3182bd", "P"), ("E", "#e6550d", "E"),
           ("R", "#31a354", "R"), ("sm", "#756bb1", "SM x100")]
    mon = TIME.month.values
    fig, axes = plt.subplots(1, 5, figsize=(9.4, 2.9), sharex=True)
    month_letters = list("JFMAMJJASOND")
    for bi, b in enumerate(BASINS):
        ax = axes[bi]; sel = foc == b
        for var, c, lbl in dyn:
            a = d[var].values[:, sel]
            series = np.nanmean(a, axis=1)
            clim = [np.nanmean(series[mon == m]) for m in range(1, 13)]
            if var == "sm": clim = np.array(clim) * 100
            ax.plot(range(1, 13), clim, color=c, lw=1.2, label=lbl)
        ax.set_title(fs.BASIN_LABEL[b], fontsize=fs.SZ_ANNOT)
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(month_letters, fontsize=fs.SZ_TICK - 1)
        ax.grid(alpha=0.25, linewidth=0.4)
        ax.margins(y=0.12)
        if bi == 0:
            ax.set_ylabel("flux (mm/mo)")
        fs.panel_letter(ax, "abcde"[bi])
    handles = [plt.Line2D([0], [0], color=c, lw=1.5) for _, c, _ in dyn]
    labels = [lbl for _, _, lbl in dyn]
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fs.save(fig, "fig03_climatology")


# ============================================================ FIG 4
def fig4_collinearity():
    hr("FIG 4  collinearity heatmaps")
    STAT = ["elevation", "slope", "aei", "aeigw", "urban_frac_mean", "logk_ferr", "porosity"]
    STAT = [v for v in STAT if v in d]
    DYN = ["P", "E", "R", "t2m", "ndvi", "sm"]
    DYN = [v for v in DYN if v in d]

    X = np.column_stack([d[v].values for v in STAT]).astype(float)
    ok_s = np.all(np.isfinite(X), axis=1)
    Cs = pd.DataFrame(np.corrcoef(X[ok_s].T), index=STAT, columns=STAT)

    n_dyn = len(DYN)
    dyn_data = {v: d[v].values for v in DYN}
    ncell = dyn_data[DYN[0]].shape[1]

    corr_sum = np.zeros((n_dyn, n_dyn))
    corr_cnt = np.zeros((n_dyn, n_dyn))

    for c in range(0, ncell, 5):
        cell_series = [dyn_data[v][:, c] for v in DYN]
        for i in range(n_dyn):
            for j in range(i, n_dyn):
                s1, s2 = cell_series[i], cell_series[j]
                valid = np.isfinite(s1) & np.isfinite(s2)
                if valid.sum() >= 12:
                    r = np.corrcoef(s1[valid], s2[valid])[0, 1]
                    if np.isfinite(r):
                        corr_sum[i, j] += r
                        corr_cnt[i, j] += 1
                        if i != j:
                            corr_sum[j, i] += r
                            corr_cnt[j, i] += 1

    Cd_arr = np.where(corr_cnt > 0, corr_sum / corr_cnt, np.nan)
    Cd = pd.DataFrame(Cd_arr, index=DYN, columns=DYN)

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.6), gridspec_kw={"width_ratios": [9, 6]})
    for a, (C, ttl, lt) in zip(ax, [(Cs, "Static predictors", "a"), (Cd, "Dynamic predictors", "b")]):
        im = a.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
        a.set_xticks(range(len(C))); a.set_xticklabels(C.columns, rotation=45, ha="right")
        a.set_yticks(range(len(C))); a.set_yticklabels(C.columns)
        a.set_title(ttl, fontsize=fs.SZ_TITLE)
        for ii in range(len(C)):
            for jj in range(len(C)):
                val = C.iloc[ii, jj]
                txt = f"{val:.2f}" if np.isfinite(val) else "NaN"
                a.text(jj, ii, txt, ha="center", va="center", fontsize=fs.SZ_ANNOT - 0.5,
                       color="white" if np.isfinite(val) and abs(val) > 0.6 else "black")
        fs.panel_letter(a, lt)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Pearson r")
    fs.save(fig, "fig04_collinearity")


# ============================================================ FIG 4b (NEW: VIF + NetFlux experiment)
def fig04b_vif_diagnostics():
    hr("FIG 4b  VIF diagnostics + NetFlux (P-R) collinearity experiment")
    DYN = ["P", "E", "R", "t2m", "ndvi", "sm"]
    DYN = [v for v in DYN if v in d]
    dyn_data = {v: d[v].values for v in DYN}

    # --- Panel a: baseline VIF (expect P & R to blow up — both track the same
    #     water-balance signal) ---
    means0, counts0 = fs.dynamic_vif_percell(dyn_data, DYN, cell_stride=5)

    # --- Panel b: replace P & R with NetFlux = P - R and recheck ---
    names1 = [v for v in DYN if v not in ("P", "R")] + ["NetFlux"]
    dyn1 = {k: dyn_data[k] for k in DYN if k not in ("P", "R")}
    dyn1["NetFlux"] = dyn_data["P"] - dyn_data["R"]
    means1, counts1 = fs.dynamic_vif_percell(dyn1, names1, cell_stride=5)

    # --- Panel c: if anything is still > 10, drop the worst offender and recheck ---
    finite_means1 = {k: v for k, v in means1.items() if np.isfinite(v)}
    worst = max(finite_means1, key=finite_means1.get) if finite_means1 else None
    if worst is not None and finite_means1[worst] > 10:
        names2 = [v for v in names1 if v != worst]
        dyn2 = {k: dyn1[k] for k in names2}
        means2, counts2 = fs.dynamic_vif_percell(dyn2, names2, cell_stride=5)
        dropped = worst
        title_c = f"Dropped worst offender ({worst})"
    else:
        names2, means2 = names1, means1
        dropped = None
        title_c = "No further drop needed (all ≤ 10)"

    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.3))
    panels = [
        (means0, DYN, "a", "Baseline VIF (P, R present)"),
        (means1, names1, "b", "P, R \u2192 NetFlux = P \u2212 R"),
        (means2, names2, "c", title_c),
    ]
    for ax, (means, names, letter, title) in zip(axes, panels):
        vals = [means[n] for n in names]
        colors = ["#e41a1c" if (np.isfinite(v) and v > 10) else "#377eb8" for v in vals]
        ax.bar(range(len(names)), vals, color=colors, edgecolor="#222222", linewidth=0.4)
        ax.axhline(10, color="black", lw=0.8, ls="--", label="VIF = 10 threshold")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=fs.SZ_TICK)
        ax.set_ylabel("VIF (per-cell mean)", fontsize=fs.SZ_LABEL)
        ax.set_title(title, fontsize=fs.SZ_ANNOT)
        ax.grid(axis="y", alpha=0.25, linewidth=0.4)
        fs.panel_letter(ax, letter)
    axes[0].legend(frameon=False, fontsize=fs.SZ_TICK - 1, loc="upper right")
    fig.tight_layout()
    fs.save(fig, "fig04b_vif_diagnostics")


# ============================================================ FIG 5a (SPLIT MEAN, SMOOTHED)
def fig5a_gwsa_mean():
    hr("FIG 5a  GWSA mean field (bilinear-smoothed)")
    gwsa = d["gwsa"].values
    mean_c = np.nanmean(gwsa, axis=0)
    gmean = fs.cell_reduce_to_grid(d, mean_c)
    nlat, nlon = gmean.shape

    fig = plt.figure(figsize=(7.2, 4.4))
    axa = fs.global_map_axis(fig, (0.04, 0.18, 0.92, 0.76), projection="robinson")
    ma = fs.pcolor_global_smooth(axa, gmean, nlat, nlon, fs.CMAP_ANOM, -20, 20, upsample=3)
    axa.set_title("Global Mean GWSA Anomaly (2002-2022)", fontsize=fs.SZ_TITLE)
    fs.panel_letter(axa, "a")
    fs.hcolorbar(fig, ma, [0.30, 0.10, 0.40, 0.014], "mean GWSA (cm)")
    fs.save(fig, "fig05a_gwsa_mean")


# ============================================================ FIG 5b (SPLIT TREND + INSETS, SMOOTHED)
def fig5b_gwsa_trend():
    hr("FIG 5b  GWSA trend field with basin insets (bilinear-smoothed)")
    gwsa = d["gwsa"].values
    tyr = (TIME - TIME[0]).days.values / 365.25
    A = np.vstack([tyr, np.ones_like(tyr)]).T
    coef, *_ = np.linalg.lstsq(A, gwsa, rcond=None)
    trend_c = coef[0]
    gtrend = fs.cell_reduce_to_grid(d, trend_c)
    nlat, nlon = gtrend.shape

    bgdf = fs.load_hydrobasins()
    basin_geoms = {}
    for b in BASINS:
        sel = foc == b
        geom, ids = fs.resolve_focus_basin_geometry(bgdf, d["clon"].values[sel], d["clat"].values[sel])
        basin_geoms[b] = geom

    fig = plt.figure(figsize=(8.2, 6.4))
    axb = fs.global_map_axis(fig, (0.05, 0.40, 0.90, 0.54), projection="robinson")
    mb = fs.pcolor_global_smooth(axb, gtrend, nlat, nlon, fs.CMAP_TREND, -2, 2, upsample=3)
    axb.set_title("Global GWSA Linear Trend (2002-2022)", fontsize=fs.SZ_TITLE)
    fs.panel_letter(axb, "a")
    fs.hcolorbar(fig, mb, [0.30, 0.38, 0.40, 0.014], "GWSA trend (cm yr$^{-1}$)")

    n_basins = len(BASINS)
    inset_w, inset_h = 0.16, 0.22
    total_w = 0.90
    gap = (total_w - n_basins * inset_w) / (n_basins - 1)
    x0 = 0.05

    for k, b in enumerate(BASINS):
        rect = [x0 + k * (inset_w + gap), 0.08, inset_w, inset_h]
        geom = basin_geoms[b]
        # Note: the insets themselves already use the smoothed, polygon-masked
        # basin field (fs.smooth_basin_inset_field) inside add_basin_inset_true_shape,
        # and the gridliner there now caps to ~3 ticks/axis with rotated lat labels.
        ax_inset = fs.add_basin_inset_true_shape(fig, rect, gtrend, b, fs.CMAP_TREND, -2, 2, geom)

        # Vertical basin label on left axis
        basin_clean_title = fs.BASIN_LABEL[b].replace(" ", "\n")
        ax_inset.set_ylabel(basin_clean_title, fontsize=fs.SZ_ANNOT - 0.5, labelpad=2, rotation=90)
        ax_inset.yaxis.set_label_position("left")

        if geom is not None:
            c_lon, c_lat = geom.centroid.x, geom.centroid.y
        else:
            lon0, lon1, lat0, lat1 = fs.BASIN_BOXES[b]
            c_lon, c_lat = (lon0 + lon1) / 2.0, (lat0 + lat1) / 2.0

        fs.connect_inset_to_map(fig, axb, ax_inset, c_lon, c_lat)

    fs.save(fig, "fig05b_gwsa_trend")


# ============================================================ FIG 6
def fig6_basin_violins():
    hr("FIG 6  per-basin predictor distributions")
    variables = [("logk_ferr", "log$_{10}$k (m$^2$)"),
                 ("porosity", "porosity"),
                 ("aei", "irrigated %"),
                 ("aeigw", "GW-irrig %"),
                 ("elevation", "elevation (m)")]
    fig, axes = plt.subplots(1, 5, figsize=(7.4, 2.6))
    xt = range(1, len(BASINS) + 1)
    labels = [fs.BASIN_LABEL[b].split()[0] for b in BASINS]
    for ax, (var, lab) in zip(axes, variables):
        data = [d[var].values[foc == b] for b in BASINS]
        data = [x[np.isfinite(x)] for x in data]
        parts = ax.violinplot(data, showmedians=True, widths=0.8)
        for pc in parts["bodies"]:
            pc.set_facecolor("#6baed6"); pc.set_edgecolor("#08519c"); pc.set_alpha(0.7)
        for key in ("cmedians", "cbars", "cmins", "cmaxes"):
            if key in parts: parts[key].set_color("#08306b"); parts[key].set_linewidth(0.8)
        ax.set_xticks(xt); ax.set_xticklabels(labels, rotation=90, fontsize=fs.SZ_ANNOT)
        ax.set_title(lab, fontsize=fs.SZ_ANNOT)
        ax.tick_params(labelsize=fs.SZ_TICK - 0.5)
        ax.grid(axis="y", alpha=0.25, linewidth=0.4)
    fig.tight_layout()
    fs.save(fig, "fig06_basin_distributions")


# ============================================================ FIG 7
def fig7_budget_residual():
    hr("FIG 7  water-balance residual map")
    P = d["P"].values; E = d["E"].values; R = d["R"].values
    gwsa = d["gwsa"].values
    isf = d["is_filled"].values.astype(bool)
    F = P - E - R
    dS = np.full_like(gwsa, np.nan)
    dS[1:-1] = (gwsa[2:] - gwsa[:-2]) / 2.0 * 10.0
    resid = F - dS
    mask = (~isf) & np.isfinite(resid)
    resid_masked = np.where(mask, resid, np.nan)
    b_c = np.nanmean(resid_masked, axis=0)

    gb = fs.cell_reduce_to_grid(d, b_c)
    fig = plt.figure(figsize=(7.2, 4.6))
    ax = fs.global_map_axis(fig, (0.04, 0.18, 0.92, 0.78), projection="robinson")
    m = fs.pcolor_global(ax, gb, "PuOr", -30, 30)
    fs.panel_letter(ax, "a")
    fs.hcolorbar(fig, m, [0.30, 0.10, 0.40, 0.014],
                 "mean budget residual $b_c$ = $\\langle$P$-$E$-$R$-$dS/dt$\\rangle$ (mm mo$^{-1}$)")
    fs.save(fig, "fig07_budget_residual")


# ============================================================ FIG 8
def fig8_t2m_ndvi_climatology():
    hr("FIG 8  t2m + NDVI climatology per basin")
    mon = TIME.month.values
    month_letters = list("JFMAMJJASOND")
    fig, axes = plt.subplots(2, 5, figsize=(9.4, 4.4), sharex=True)
    for bi, b in enumerate(BASINS):
        sel = foc == b
        t2m_series = np.nanmean(d["t2m"].values[:, sel], axis=1)
        ndvi_series = np.nanmean(d["ndvi"].values[:, sel], axis=1)
        t2m_clim = [np.nanmean(t2m_series[mon == m]) for m in range(1, 13)]
        ndvi_clim = [np.nanmean(ndvi_series[mon == m]) for m in range(1, 13)]
        ax0, ax1 = axes[0, bi], axes[1, bi]
        ax0.plot(range(1, 13), t2m_clim, color="#e6550d", lw=1.2)
        ax1.plot(range(1, 13), ndvi_clim, color="#238b45", lw=1.2)
        ax0.set_title(fs.BASIN_LABEL[b], fontsize=fs.SZ_ANNOT)
        for ax in (ax0, ax1):
            ax.set_xticks(range(1, 13))
            ax.set_xticklabels(month_letters, fontsize=fs.SZ_TICK - 1)
            ax.grid(alpha=0.25, linewidth=0.4)
            ax.margins(y=0.12)
        if bi == 0:
            ax0.set_ylabel("t2m (K)"); ax1.set_ylabel("NDVI (-)")
        fs.panel_letter(ax0, "abcde"[bi])
    fig.tight_layout()
    fs.save(fig, "fig08_t2m_ndvi_climatology")


# ============================================================ FIG 9 (DYNAMIC DRIVER TRENDS)
def fig9_dynamic_trends():
    hr("FIG 9  dynamic predictor global trends")
    specs = [
        ("P", "P trend (mm mo$^{-1}$ yr$^{-1}$)", "BrBG", -2.0, 2.0),
        ("E", "E trend (mm mo$^{-1}$ yr$^{-1}$)", "PuOr", -1.5, 1.5),
        ("R", "R trend (mm mo$^{-1}$ yr$^{-1}$)", "YlGnBu", -1.0, 1.0),
        ("t2m", "t2m trend (K yr$^{-1}$)", "RdBu_r", -0.1, 0.1),
        ("ndvi", "NDVI trend ($\times 10^{-3}$ yr$^{-1}$)", "PiYG", -5.0, 5.0),
        ("sm", "SM trend ($\times 10^{-3}$ yr$^{-1}$)", "BrBG", -5.0, 5.0),
    ]
    tyr = (TIME - TIME[0]).days.values / 365.25
    A = np.vstack([tyr, np.ones_like(tyr)]).T

    fig = plt.figure(figsize=(7.6, 9.4))
    letters = "abcdef"

    for k, (var, lab, cmap, vmin, vmax) in enumerate(specs):
        if var not in d:
            continue
        v_val = d[var].values
        coef, *_ = np.linalg.lstsq(A, v_val, rcond=None)
        trend_c = coef[0]

        if var in ["ndvi", "sm"]:
            trend_c = trend_c * 1000.0

        row = k // 2; col = k % 2
        x0 = 0.05 + col * 0.50
        y0 = 0.70 - row * 0.335
        ax = fs.global_map_axis(fig, (x0, y0, 0.42, 0.24), projection="robinson")
        g = fs.cell_reduce_to_grid(d, trend_c)
        m = fs.pcolor_global(ax, g, cmap, vmin, vmax)
        fs.panel_letter(ax, letters[k])
        cax = fig.add_axes([x0 + 0.02, y0 - 0.028, 0.36, 0.011])
        cb = fig.colorbar(m, cax=cax, orientation="horizontal")
        cb.set_label(lab, fontsize=fs.SZ_CBAR - 0.5)
        cb.ax.tick_params(labelsize=fs.SZ_TICK - 0.5, width=0.4, length=1.8)
        cb.outline.set_linewidth(0.4)

    fs.save(fig, "fig09_dynamic_trends")


# ============================================================ FIG 10
def fig10_domain_density():
    hr("FIG 10  Gaussian-smoothed domain cell density")
    nlat, nlon = fs._infer_grid_size(d)
    j = np.asarray(d["j"].values).astype(int)
    i = np.asarray(d["i"].values).astype(int)

    presence = np.zeros((nlat, nlon), dtype=float)
    presence[j, i] = 1.0

    density_smooth = gaussian_filter(presence, sigma=2.5, mode="constant", cval=0.0)
    density_masked = np.where(presence > 0, density_smooth, np.nan)

    fig = plt.figure(figsize=(7.2, 4.6))
    ax = fs.global_map_axis(fig, (0.04, 0.16, 0.92, 0.78), projection="robinson")
    m = fs.pcolor_global(ax, density_masked, fs.CMAP_SEQ_Y, 0, np.nanmax(density_masked))
    fs.panel_letter(ax, "a")
    total = int(presence.sum())
    ax.set_title(f"Domain Cell Spatial Density (total cells: {total:,})", fontsize=fs.SZ_TITLE)
    fs.hcolorbar(fig, m, [0.30, 0.10, 0.40, 0.014], "Gaussian-smoothed cell density field")
    fs.save(fig, "fig10_domain_density")


# ============================================================ FIG 11 (SMOOTHED GLOBES)
def fig11_orthographic_globes():
    hr("FIG 11  Orthographic true-globe dual-hemisphere views (bilinear-smoothed)")
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    gwsa = d["gwsa"].values
    tyr = (TIME - TIME[0]).days.values / 365.25
    A = np.vstack([tyr, np.ones_like(tyr)]).T
    coef, *_ = np.linalg.lstsq(A, gwsa, rcond=None)
    trend_c = coef[0]
    gtrend = fs.cell_reduce_to_grid(d, trend_c)
    nlat, nlon = gtrend.shape

    fig = plt.figure(figsize=(8.2, 4.2))

    # Western Hemisphere Orthographic
    ax1 = fig.add_axes([0.04, 0.12, 0.42, 0.80],
                       projection=ccrs.Orthographic(central_longitude=-80, central_latitude=15))
    ax1.add_feature(cfeature.LAND, facecolor="#f0f0f0", zorder=0)
    ax1.add_feature(cfeature.OCEAN, facecolor="#ffffff", zorder=0)
    fs.pcolor_global_smooth(ax1, gtrend, nlat, nlon, fs.CMAP_TREND, -2, 2, upsample=3)
    ax1.coastlines(linewidth=0.35, color="#444444", zorder=5)
    ax1.gridlines(crs=ccrs.PlateCarree(), linewidth=0.3, color="#888888", alpha=0.5, linestyle="--")
    fs.panel_letter(ax1, "a")
    ax1.set_title("Western Hemisphere", fontsize=fs.SZ_TITLE)

    # Eastern Hemisphere Orthographic
    ax2 = fig.add_axes([0.50, 0.12, 0.42, 0.80],
                       projection=ccrs.Orthographic(central_longitude=85, central_latitude=15))
    ax2.add_feature(cfeature.LAND, facecolor="#f0f0f0", zorder=0)
    ax2.add_feature(cfeature.OCEAN, facecolor="#ffffff", zorder=0)
    m2 = fs.pcolor_global_smooth(ax2, gtrend, nlat, nlon, fs.CMAP_TREND, -2, 2, upsample=3)
    ax2.coastlines(linewidth=0.35, color="#444444", zorder=5)
    ax2.gridlines(crs=ccrs.PlateCarree(), linewidth=0.3, color="#888888", alpha=0.5, linestyle="--")
    fs.panel_letter(ax2, "b")
    ax2.set_title("Eastern Hemisphere", fontsize=fs.SZ_TITLE)

    fs.hcolorbar(fig, m2, [0.30, 0.06, 0.40, 0.015], "GWSA trend (cm yr$^{-1}$)")
    fs.save(fig, "fig11_orthographic_globes")


# ============================================================ FIG 12 (FIXED COLORMAP + SCI TICKS)
def fig12_bivariate_static():
    hr("FIG 12  bivariate density: GWSA mean vs static predictors")
    STAT = ["logk_ferr", "porosity", "aei", "aeigw", "urban_frac_mean", "elevation", "slope"]
    STAT = [v for v in STAT if v in d]
    y = np.nanmean(d["gwsa"].values, axis=0)

    fig, axes = plt.subplots(2, 4, figsize=(7.5, 4.5))
    axes = axes.flatten()
    insets = ["porosity", "urban_frac_mean"]

    hb_last = None
    letters = "abcdefg"
    for k, var in enumerate(STAT):
        ax = axes[k]
        x = d[var].values
        valid = np.isfinite(x) & np.isfinite(y)
        xv, yv = x[valid], y[valid]

        # Native Matplotlib cmap "YlGnBu" replaces non-standard "crest"
        hb = ax.hexbin(xv, yv, gridsize=40, cmap="YlGnBu", mincnt=1, bins="log", edgecolors="none")
        hb_last = hb

        if len(xv) > 0:
            bins = np.linspace(np.percentile(xv, 1), np.percentile(xv, 99), 20)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            digitized = np.digitize(xv, bins)
            bin_medians = [np.median(yv[digitized == idx]) if np.sum(digitized == idx) > 0 else np.nan for idx in range(1, len(bins))]
            ax.plot(bin_centers, bin_medians, color="#d95f02", lw=1.2)

        ax.set_title(var, fontsize=fs.SZ_ANNOT)
        ax.set_xlabel(var, fontsize=fs.SZ_TICK)
        ax.set_ylabel("mean GWSA (cm)", fontsize=fs.SZ_TICK)
        ax.tick_params(labelsize=fs.SZ_TICK - 0.5)
        ax.grid(alpha=0.2, lw=0.3)
        # Scientific-notation ticks only kick in for genuinely long numbers
        # (e.g. logk_ferr ~1e-13). Mean-GWSA axis (-20..20) stays plain decimal,
        # since it's already short -- nothing to fix there.
        ax.ticklabel_format(style="sci", scilimits=(-2, 3), axis="both")
        fs.panel_letter(ax, letters[k])

        if var in insets and len(xv) > 0:
            ax_ins = inset_axes(ax, width="42%", height="42%", loc="upper left")
            xp1, xp2 = np.percentile(xv, [35, 65])
            yp1, yp2 = np.percentile(yv, [35, 65])
            ax_ins.hexbin(xv, yv, gridsize=25, cmap="YlGnBu", mincnt=1, bins="log", edgecolors="none")
            ax_ins.plot(bin_centers, bin_medians, color="#d95f02", lw=1.0)
            ax_ins.set_xlim(xp1, xp2)
            ax_ins.set_ylim(yp1, yp2)
            ax_ins.tick_params(labelsize=fs.SZ_TICK - 2)
            ax_ins.ticklabel_format(style="sci", scilimits=(-2, 3), axis="both")
            mark_inset(ax, ax_ins, loc1=2, loc2=4, fc="none", ec="#d95f02", lw=0.6, ls="--")

    axes[7].axis("off")
    if hb_last is not None:
        cax = inset_axes(axes[7], width="80%", height="14%", loc="center")
        cb = fig.colorbar(hb_last, cax=cax, orientation="horizontal")
        cb.set_label("cell density ($\log_{10}$ count)", fontsize=fs.SZ_TICK - 0.5)
        cb.ax.tick_params(labelsize=fs.SZ_TICK - 1)

    fig.tight_layout()
    fs.save(fig, "fig12_bivariate_static")


# ============================================================ FIG 13 (FIXED COLORMAP + SCI TICKS)
def fig13_bivariate_dynamic():
    hr("FIG 13  bivariate density: GWSA vs dynamic predictors")
    DYN = ["P", "E", "R", "t2m", "ndvi", "sm"]
    DYN = [v for v in DYN if v in d]
    y_all = d["gwsa"].values.ravel()

    fig, axes = plt.subplots(2, 3, figsize=(7.5, 4.8))
    axes = axes.flatten()
    insets = ["P", "R", "ndvi"]

    hb_last = None
    letters = "abcdef"
    for k, var in enumerate(DYN):
        ax = axes[k]
        x_all = d[var].values.ravel()

        valid = np.isfinite(x_all) & np.isfinite(y_all)
        xv, yv = x_all[valid], y_all[valid]
        if len(xv) > 100000:
            sub = np.random.choice(len(xv), size=100000, replace=False)
            xv, yv = xv[sub], yv[sub]

        # Native Matplotlib cmap "YlGnBu" replaces non-standard "mako"
        hb = ax.hexbin(xv, yv, gridsize=40, cmap="YlGnBu", mincnt=1, bins="log", edgecolors="none")
        hb_last = hb

        if len(xv) > 0:
            bins = np.linspace(np.percentile(xv, 1), np.percentile(xv, 99), 20)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            digitized = np.digitize(xv, bins)
            bin_medians = [np.median(yv[digitized == idx]) if np.sum(digitized == idx) > 0 else np.nan for idx in range(1, len(bins))]
            ax.plot(bin_centers, bin_medians, color="#e74c3c", lw=1.2)

        ax.set_title(var, fontsize=fs.SZ_ANNOT)
        ax.set_xlabel(var, fontsize=fs.SZ_TICK)
        ax.set_ylabel("GWSA (cm)", fontsize=fs.SZ_TICK)
        ax.tick_params(labelsize=fs.SZ_TICK - 0.5)
        ax.grid(alpha=0.2, lw=0.3)
        # Same sci-notation rule as fig12: only long numbers get reformatted.
        ax.ticklabel_format(style="sci", scilimits=(-2, 3), axis="both")
        fs.panel_letter(ax, letters[k])

        if var in insets and len(xv) > 0:
            ax_ins = inset_axes(ax, width="40%", height="40%", loc="lower right",
                                bbox_to_anchor=(0.02, 0.05, 0.95, 0.95), bbox_transform=ax.transAxes)
            xp1, xp2 = np.percentile(xv, [35, 65])
            yp1, yp2 = np.percentile(yv, [35, 65])
            ax_ins.hexbin(xv, yv, gridsize=25, cmap="YlGnBu", mincnt=1, bins="log", edgecolors="none")
            ax_ins.plot(bin_centers, bin_medians, color="#e74c3c", lw=1.0)
            ax_ins.set_xlim(xp1, xp2)
            ax_ins.set_ylim(yp1, yp2)
            ax_ins.tick_params(labelsize=fs.SZ_TICK - 2)
            ax_ins.ticklabel_format(style="sci", scilimits=(-2, 3), axis="both")
            mark_inset(ax, ax_ins, loc1=1, loc2=3, fc="none", ec="#e74c3c", lw=0.6, ls="--")

    plt.subplots_adjust(wspace=0.28, hspace=0.38, bottom=0.15)
    if hb_last is not None:
        cbar_ax = fig.add_axes([0.25, 0.04, 0.50, 0.02])
        cb = fig.colorbar(hb_last, cax=cbar_ax, orientation="horizontal")
        cb.set_label("sample density ($\log_{10}$ count)", fontsize=fs.SZ_TICK - 0.5)
        cb.ax.tick_params(labelsize=fs.SZ_TICK - 1)

    fs.save(fig, "fig13_bivariate_dynamic")


# ============================================================ FIG 14 (NEW: RELATION DIAGNOSTICS)
def fig14_relation_diagnostics():
    hr("FIG 14  relation diagnostics (Pearson / Spearman / Mutual Information)")
    STAT = ["logk_ferr", "porosity", "aei", "aeigw", "urban_frac_mean", "elevation", "slope"]
    STAT = [v for v in STAT if v in d]
    DYN = ["P", "E", "R", "t2m", "ndvi", "sm"]
    DYN = [v for v in DYN if v in d]

    y_mean = np.nanmean(d["gwsa"].values, axis=0)
    y_all = d["gwsa"].values.ravel()
    rng = np.random.default_rng(0)
    rows = []

    # Static predictors vs mean GWSA (one value per cell)
    for var in STAT:
        x = d[var].values
        valid = np.isfinite(x) & np.isfinite(y_mean)
        xv, yv = x[valid], y_mean[valid]
        if len(xv) < 10:
            continue
        pear = np.corrcoef(xv, yv)[0, 1]
        spear = spearmanr(xv, yv).correlation
        mi = fs.mutual_info_safe(xv, yv)
        rows.append({"variable": var, "kind": "static", "pearson": pear,
                     "spearman": spear, "mutual_info": mi, "n": len(xv)})

    # Dynamic predictors vs full GWSA time series (space x time, subsampled)
    for var in DYN:
        x_all = d[var].values.ravel()
        valid = np.isfinite(x_all) & np.isfinite(y_all)
        xv, yv = x_all[valid], y_all[valid]
        if len(xv) > 50000:
            sub = rng.choice(len(xv), size=50000, replace=False)
            xv, yv = xv[sub], yv[sub]
        if len(xv) < 10:
            continue
        pear = np.corrcoef(xv, yv)[0, 1]
        spear = spearmanr(xv, yv).correlation
        mi = fs.mutual_info_safe(xv, yv)
        rows.append({"variable": var, "kind": "dynamic", "pearson": pear,
                     "spearman": spear, "mutual_info": mi, "n": len(xv)})

    df = pd.DataFrame(rows)
    os.makedirs(fs.FIGDIR, exist_ok=True)
    csv_path = os.path.join(fs.FIGDIR, "fig14_relation_diagnostics_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"  saved: {csv_path}")

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    for ax, kind, letter in zip(axes, ["static", "dynamic"], "ab"):
        sub = df[df["kind"] == kind].reset_index(drop=True)
        n = len(sub)
        x = np.arange(n)
        w = 0.27
        ax.bar(x - w, sub["pearson"].abs(), width=w, color="#377eb8", label="|Pearson r|")
        ax.bar(x, sub["spearman"].abs(), width=w, color="#4daf4a", label="|Spearman \u03c1|")
        ax2 = ax.twinx()
        ax2.bar(x + w, sub["mutual_info"], width=w, color="#984ea3", label="Mutual info")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["variable"], rotation=45, ha="right", fontsize=fs.SZ_TICK)
        ax.set_ylabel("|correlation|", fontsize=fs.SZ_LABEL)
        ax2.set_ylabel("mutual information (nats)", fontsize=fs.SZ_LABEL)
        ax.set_title(f"{kind.capitalize()} predictors vs GWSA", fontsize=fs.SZ_ANNOT)
        ax.grid(axis="y", alpha=0.2, lw=0.3)
        fs.panel_letter(ax, letter)
        if kind == "static":
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=fs.SZ_TICK - 1, loc="upper right")
    fig.tight_layout()
    fs.save(fig, "fig14_relation_diagnostics")


# ============================================================ FIG 15 (NEW: LAGGED CORRELATION)
def fig15_lagged_correlation():
    hr("FIG 15  lagged cross-correlation of dynamic drivers vs GWSA, per basin")
    DYN = ["P", "E", "R", "t2m", "ndvi", "sm"]
    DYN = [v for v in DYN if v in d]
    lags = np.arange(-12, 13)
    gwsa = d["gwsa"].values

    def lagged_corr(x, y, lag):
        # positive lag: x leads y by `lag` months (e.g. precipitation showing
        # up in groundwater months later)
        if lag > 0:
            xs, ys = x[:-lag], y[lag:]
        elif lag < 0:
            xs, ys = x[-lag:], y[:lag]
        else:
            xs, ys = x, y
        valid = np.isfinite(xs) & np.isfinite(ys)
        if valid.sum() < 12:
            return np.nan
        return np.corrcoef(xs[valid], ys[valid])[0, 1]

    fig, axes = plt.subplots(1, len(BASINS), figsize=(9.8, 2.9), sharey=True)
    cmap_lines = plt.cm.tab10(np.linspace(0, 1, len(DYN)))

    for bi, b in enumerate(BASINS):
        ax = axes[bi]
        sel = foc == b
        gwsa_b = np.nanmean(gwsa[:, sel], axis=1)
        for vi, var in enumerate(DYN):
            x_b = np.nanmean(d[var].values[:, sel], axis=1)
            cc = np.array([lagged_corr(x_b, gwsa_b, lag) for lag in lags])
            ax.plot(lags, cc, lw=1.1, color=cmap_lines[vi], label=var)
        ax.axvline(0, color="black", lw=0.6, ls="--")
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_title(fs.BASIN_LABEL[b], fontsize=fs.SZ_ANNOT)
        ax.set_xlabel("lag (months)", fontsize=fs.SZ_TICK)
        ax.tick_params(labelsize=fs.SZ_TICK - 0.5)
        ax.grid(alpha=0.25, linewidth=0.4)
        fs.panel_letter(ax, "abcde"[bi])
    axes[0].set_ylabel("cross-correlation with GWSA", fontsize=fs.SZ_LABEL)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(DYN), frameon=False,
               bbox_to_anchor=(0.5, 1.10))
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fs.save(fig, "fig15_lagged_correlation")


# ============================================================ RUN ALL
if __name__ == "__main__":
    fig1_completeness()
    fig2_static_panels()
    fig3_climatology()
    fig4_collinearity()
    fig04b_vif_diagnostics()
    fig5a_gwsa_mean()
    fig5b_gwsa_trend()
    fig6_basin_violins()
    fig7_budget_residual()
    fig8_t2m_ndvi_climatology()
    fig9_dynamic_trends()
    fig10_domain_density()
    fig11_orthographic_globes()
    fig12_bivariate_static()
    fig13_bivariate_dynamic()
    fig14_relation_diagnostics()
    fig15_lagged_correlation()
    d.close()
    print("\nALL 17 FIGURES GENERATED SUCCESSFULLY.")