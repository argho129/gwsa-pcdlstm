"""
================================================================================
PHASE 11 - GLOBAL MAPS OF GWSA TREND, DEPLETION LAYER AND UNCERTAINTY
Arpon Sarkar, URP, KUET
================================================================================

WHAT THIS READS
  A single file: OUTPUTS\\08_final_product\\GWSA_KUET_v1_1D.nc
  Variables used: gwsa_trend, sigma_total, trend_significant (or
  significant_full), lat, lon, tier. Nothing else is required, so this script
  is independent of the exact contents of the gridded NetCDF.

  NOTE ON THE SIGNIFICANCE FLAG. The Phase 8 file carries BOTH
  'trend_significant' and 'significant_full'. They are not the same variable and
  will give different depletion-cell counts. SIG_VAR below makes the choice
  explicit rather than leaving it to lookup order; keep it consistent with what
  Phase 9 and Phase 13 used ('trend_significant') or the cell counts quoted in
  different chapters will not agree.

REGRIDDING
  The 1D (cell,) product is placed back onto its native 0.5 deg grid with the
  same convention Phase 8 used:
      row = round((clat + 89.75) / 0.5)      col = round((clon + 179.75) / 0.5)
  giving 360 x 720. The script VERIFIES that every valid cell landed on the grid
  and that no two cells collided, and reports both; a silent mismatch here would
  quietly drop data from every map.

FIGURES
  fig11_1_global_maps       four Robinson panels: trend, depletion, uncertainty, tier
  fig11_2_globe_views       two orthographic views of the depletion layer
  fig11_3_regional_hotspots labelled PlateCarree zooms of the major depletion centres
  All are written as 600 dpi PNG and vector PDF.

TYPOGRAPHY
  Times New Roman throughout, with a graceful fallback chain and a printed
  warning if it is unavailable. Mathtext uses STIX, which is metrically close to
  Times, so symbols and text match.

Requires: cartopy, matplotlib, numpy, xarray
    conda install -c conda-forge cartopy      (preferred on Windows)
================================================================================
"""

from pathlib import Path

import numpy as np
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import font_manager

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

# ------------------------------------------------------------------ config --
IN_PATH = Path(r"D:\THESIS\Arpon_Thesis_Final\OUTPUTS\08_final_product\GWSA_KUET_v1_1D.nc")
OUT_DIR = Path(r"D:\THESIS\Arpon_Thesis_Final\OUTPUTS\11_maps")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SIG_VAR = "trend_significant"      # or "significant_full" - see note above
GRID_RES = 0.5
N_LAT, N_LON = 360, 720
DPI = 600
SAVE_PDF = True

# Regional zooms: (name, lon_min, lon_max, lat_min, lat_max)
REGIONS = [
    ("North-west India and Indus basin", 65, 92, 18, 37),
    ("North China plain", 100, 125, 28, 45),
    ("US High Plains and Central Valley", -125, -93, 28, 47),
    ("Middle East and Caspian", 35, 65, 22, 45),
]

# ------------------------------------------------------------- typography ---
_avail = {f.name for f in font_manager.fontManager.ttflist}
_serif_chain = ["Times New Roman", "Nimbus Roman", "Liberation Serif",
                "DejaVu Serif", "serif"]
_chosen = next((f for f in _serif_chain if f in _avail), "serif")
if _chosen != "Times New Roman":
    print(f"[WARN] Times New Roman not found; falling back to '{_chosen}'. "
          f"Install the font or accept the substitute.")

FS_TICK, FS_LABEL, FS_TITLE, FS_PANEL = 9, 10, 11, 12
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": _serif_chain,
    "mathtext.fontset": "stix",          # Times-compatible symbols
    "font.size": FS_TICK,
    "axes.labelsize": FS_LABEL,
    "axes.titlesize": FS_TITLE,
    "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "savefig.dpi": DPI,
    "figure.dpi": 120,
    "pdf.fonttype": 42,                  # embed as TrueType, editable in Illustrator
    "ps.fonttype": 42,
})

LAND, OCEAN = "#f2f0ed", "#ffffff"
COAST, BORDER = "#5a5a5a", "#9a9a9a"


def save(fig, name):
    fig.savefig(OUT_DIR / f"{name}.png", dpi=DPI, bbox_inches="tight",
                facecolor="white")
    if SAVE_PDF:
        fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {name}.png" + (" / .pdf" if SAVE_PDF else ""))


def pick(ds, *names):
    for n in names:
        if n in ds.data_vars or n in ds.coords:
            return ds[n]
    raise KeyError(f"None of {names} found. Available: {list(ds.data_vars) + list(ds.coords)}")


# ------------------------------------------------------------------- load ---
print("=" * 70)
print("PHASE 11 -- global maps")
print("=" * 70)
ds = xr.open_dataset(IN_PATH)
print(f"  file: {IN_PATH.name}")
print(f"  variables: {list(ds.data_vars)}")

trend = pick(ds, "gwsa_trend", "trend")
sigma_total = pick(ds, "sigma_total")
if SIG_VAR not in ds.variables:
    raise KeyError(f"SIG_VAR '{SIG_VAR}' not in file; available: {list(ds.data_vars)}")
significant = ds[SIG_VAR]
clat = pick(ds, "clat", "lat")
clon = pick(ds, "clon", "lon")
tier = ds["tier"] if "tier" in ds.variables else None

# Report both significance flags so the choice is auditable.
for _sv in ("trend_significant", "significant_full"):
    if _sv in ds.variables:
        _n = int((np.asarray(ds[_sv].values) == 1).sum())
        print(f"  {_sv}: {_n} cells flagged" + ("   <-- used" if _sv == SIG_VAR else ""))

valid = (trend.notnull() & sigma_total.notnull()).values
n_cell = trend.sizes.get("cell", trend.size)
print(f"  valid cells: {int(valid.sum())} / {n_cell}")

clat_v = np.asarray(clat.values, dtype=float)
clon_v = np.asarray(clon.values, dtype=float)
trend_v = np.asarray(trend.values, dtype=float)
sigma_v = np.asarray(sigma_total.values, dtype=float)
sig_v = np.asarray(significant.values)


# ----------------------------------------------------------------- regrid ---
row_idx = np.round((clat_v + 89.75) / GRID_RES).astype(int)
col_idx = np.round((clon_v + 179.75) / GRID_RES).astype(int)
in_range = (row_idx >= 0) & (row_idx < N_LAT) & (col_idx >= 0) & (col_idx < N_LON)

# Verify the placement before trusting any map drawn from it.
_flat = row_idx[valid & in_range] * N_LON + col_idx[valid & in_range]
_uniq = np.unique(_flat).size
_dropped = int((valid & ~in_range).sum())
print(f"  regrid: {_flat.size} valid cells placed, {_uniq} unique grid points, "
      f"{_dropped} out of range")
if _dropped:
    print(f"  [WARN] {_dropped} valid cells fell outside the grid - check clat/clon.")
if _uniq != _flat.size:
    print(f"  [WARN] {_flat.size - _uniq} cells collided on the same grid point; "
          f"the later value wins. Check for duplicate coordinates.")


def to_grid(values_1d):
    g = np.full((N_LAT, N_LON), np.nan)
    ok = in_range & np.isfinite(values_1d)
    g[row_idx[ok], col_idx[ok]] = values_1d[ok]
    return g


trend_grid = to_grid(np.where(valid, trend_v, np.nan))
sigma_grid = to_grid(np.where(valid, sigma_v, np.nan))

depl_mask = valid & (sig_v == 1) & (trend_v < 0)
# Plot depletion as a POSITIVE magnitude: a sequential ramp from 0 reads far
# more clearly than a reversed diverging map anchored at zero.
depl_grid = to_grid(np.where(depl_mask, -trend_v, np.nan))
print(f"  depleting cells (significant, trend < 0): {int(depl_mask.sum())}")

tier_grid = to_grid(np.where(valid, np.asarray(tier.values, dtype=float), np.nan)) \
    if tier is not None else None

lat_edges = -90 + GRID_RES * np.arange(N_LAT + 1)
lon_edges = -180 + GRID_RES * np.arange(N_LON + 1)
lat_c = (lat_edges[:-1] + lat_edges[1:]) / 2
lon_c = (lon_edges[:-1] + lon_edges[1:]) / 2
LON2D, LAT2D = np.meshgrid(lon_c, lat_c)

trend_lim = float(np.nanpercentile(np.abs(trend_grid), 98))
depl_lim = float(np.nanpercentile(depl_grid, 98))
sigma_lim = float(np.nanpercentile(sigma_grid, 98))
print(f"  colour limits: trend +/-{trend_lim:.2f}, depletion 0..{depl_lim:.2f}, "
      f"sigma 0..{sigma_lim:.2f} cm/yr")


# ------------------------------------------------------------- map helpers --
def base_map(ax, global_extent=True):
    if global_extent:
        ax.set_global()
    ax.add_feature(cfeature.OCEAN, facecolor=OCEAN, zorder=0)
    ax.add_feature(cfeature.LAND, facecolor=LAND, zorder=0)
    ax.coastlines(linewidth=0.35, color=COAST, zorder=4)
    ax.add_feature(cfeature.BORDERS, linewidth=0.18, color=BORDER, zorder=4)
    ax.spines["geo"].set_linewidth(0.6)
    ax.spines["geo"].set_edgecolor("0.35")


def graticule(ax, labels=True, dlon=60, dlat=30):
    """Graticule with degree labels. Label support for curved projections varies
    by cartopy version, so fall back to an unlabelled graticule rather than
    failing the whole figure."""
    try:
        gl = ax.gridlines(draw_labels=labels, linewidth=0.25, color="0.55",
                          alpha=0.55, linestyle=(0, (3, 3)),
                          xlocs=np.arange(-180, 181, dlon),
                          ylocs=np.arange(-90, 91, dlat), zorder=3)
        if labels:
            gl.top_labels = gl.right_labels = False
            gl.xformatter, gl.yformatter = LongitudeFormatter(), LatitudeFormatter()
            gl.xlabel_style = {"size": FS_TICK, "color": "0.2"}
            gl.ylabel_style = {"size": FS_TICK, "color": "0.2"}
        return gl
    except Exception as exc:
        print(f"  [note] labelled graticule unavailable ({type(exc).__name__}); "
              f"drawing without labels")
        ax.gridlines(draw_labels=False, linewidth=0.25, color="0.55", alpha=0.55,
                     linestyle=(0, (3, 3)), zorder=3)
        return None


def panel_letter(ax, letter):
    ax.text(0.0, 1.045, letter, transform=ax.transAxes, fontsize=FS_PANEL,
            fontweight="bold", va="bottom", ha="left")


def add_cbar(fig, im, rect, label, ticks=None, extend="neither"):
    cax = fig.add_axes(rect)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend=extend)
    cb.set_label(label, fontsize=FS_LABEL)
    cb.ax.tick_params(labelsize=FS_TICK, width=0.6, length=2.5)
    cb.outline.set_linewidth(0.5)
    if ticks is not None:
        cb.set_ticks(ticks)
    return cb


# ============================================ FIG 1: four global panels ======
print("\n  building fig11_1_global_maps ...")
fig = plt.figure(figsize=(13.6, 8.2))
proj = ccrs.Robinson(central_longitude=0)

# axes rectangles: two rows, two columns, room beneath each for a colourbar
rects = [(0.035, 0.575, 0.435, 0.365), (0.530, 0.575, 0.435, 0.365),
         (0.035, 0.085, 0.435, 0.365), (0.530, 0.085, 0.435, 0.365)]
cbars = [(0.105, 0.532, 0.295, 0.014), (0.600, 0.532, 0.295, 0.014),
         (0.105, 0.042, 0.295, 0.014), (0.600, 0.042, 0.295, 0.014)]

# (a) trend
ax = fig.add_axes(rects[0], projection=proj)
base_map(ax)
graticule(ax)
im = ax.pcolormesh(LON2D, LAT2D, trend_grid, transform=ccrs.PlateCarree(),
                   cmap="RdBu", vmin=-trend_lim, vmax=trend_lim, shading="auto",
                   zorder=2, rasterized=True)
ax.set_title("Groundwater storage trend", fontsize=FS_TITLE)
panel_letter(ax, "a")
add_cbar(fig, im, cbars[0], "Trend (cm yr$^{-1}$)  |  blue, gain    red, loss",
         extend="both")

# (b) depletion layer
ax = fig.add_axes(rects[1], projection=proj)
base_map(ax)
graticule(ax)
im = ax.pcolormesh(LON2D, LAT2D, depl_grid, transform=ccrs.PlateCarree(),
                   cmap="YlOrRd", vmin=0, vmax=depl_lim, shading="auto",
                   zorder=2, rasterized=True)
ax.set_title(f"Depletion layer  ({int(depl_mask.sum()):,} significant cells)",
             fontsize=FS_TITLE)
panel_letter(ax, "b")
add_cbar(fig, im, cbars[1], "Depletion rate (cm yr$^{-1}$)", extend="max")

# (c) uncertainty
ax = fig.add_axes(rects[2], projection=proj)
base_map(ax)
graticule(ax)
im = ax.pcolormesh(LON2D, LAT2D, sigma_grid, transform=ccrs.PlateCarree(),
                   cmap="viridis", vmin=0, vmax=sigma_lim, shading="auto",
                   zorder=2, rasterized=True)
ax.set_title("Trend uncertainty", fontsize=FS_TITLE)
panel_letter(ax, "c")
add_cbar(fig, im, cbars[2], r"$\sigma_{\mathrm{total}}$ (cm yr$^{-1}$)", extend="max")

# (d) tier
if tier_grid is not None:
    ax = fig.add_axes(rects[3], projection=proj)
    base_map(ax)
    graticule(ax)
    cmap_tier = mcolors.ListedColormap(["#d95f02", "#1b9e77"])
    norm_tier = mcolors.BoundaryNorm([-0.5, 0.5, 1.5], cmap_tier.N)
    im = ax.pcolormesh(LON2D, LAT2D, tier_grid, transform=ccrs.PlateCarree(),
                       cmap=cmap_tier, norm=norm_tier, shading="auto",
                       zorder=2, rasterized=True)
    ax.set_title("Constraint tier", fontsize=FS_TITLE)
    panel_letter(ax, "d")
    cb = add_cbar(fig, im, cbars[3], "", ticks=[0, 1])
    cb.ax.set_xticklabels(["physics-extrapolated", "well-constrained"],
                          fontsize=FS_TICK)
save(fig, "fig11_1_global_maps")


# ============================================= FIG 2: orthographic globes ====
print("  building fig11_2_globe_views ...")
fig = plt.figure(figsize=(11.5, 6.0))
views = [((20, 80), "Asia and the Indian subcontinent"),
         ((10, -95), "The Americas")]
grects = [(0.045, 0.145, 0.42, 0.79), (0.520, 0.145, 0.42, 0.79)]
im = None
for i, (((lat0, lon0), ttl), rect) in enumerate(zip(views, grects)):
    ax = fig.add_axes(rect, projection=ccrs.Orthographic(central_longitude=lon0,
                                                         central_latitude=lat0))
    base_map(ax)
    graticule(ax, labels=False, dlon=30, dlat=15)
    im = ax.pcolormesh(LON2D, LAT2D, depl_grid, transform=ccrs.PlateCarree(),
                       cmap="YlOrRd", vmin=0, vmax=depl_lim, shading="auto",
                       zorder=2, rasterized=True)
    ax.set_title(ttl, fontsize=FS_TITLE)
    panel_letter(ax, "ab"[i])
add_cbar(fig, im, (0.335, 0.075, 0.33, 0.017),
         "Depletion rate (cm yr$^{-1}$)", extend="max")
save(fig, "fig11_2_globe_views")


# ============================================== FIG 3: regional hotspots =====
print("  building fig11_3_regional_hotspots ...")
ncol = 2
nrow = int(np.ceil(len(REGIONS) / ncol))
fig = plt.figure(figsize=(12.0, 4.6 * nrow))
im = None
for k, (name, lo0, lo1, la0, la1) in enumerate(REGIONS):
    r, c = divmod(k, ncol)
    w, h = 0.385, 0.66 / nrow
    x = 0.075 + c * 0.475
    y = 1.0 - (r + 1) / nrow + (0.30 / nrow)
    ax = fig.add_axes((x, y, w, h), projection=ccrs.PlateCarree())
    ax.set_extent([lo0, lo1, la0, la1], crs=ccrs.PlateCarree())
    base_map(ax, global_extent=False)
    im = ax.pcolormesh(LON2D, LAT2D, depl_grid, transform=ccrs.PlateCarree(),
                       cmap="YlOrRd", vmin=0, vmax=depl_lim, shading="auto",
                       zorder=2, rasterized=True)
    # PlateCarree supports proper axis ticks, so label the coordinates directly
    ax.set_xticks(np.arange(np.ceil(lo0 / 10) * 10, lo1 + 1, 10),
                  crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(np.ceil(la0 / 5) * 5, la1 + 1, 5), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(labelsize=FS_TICK, width=0.6, length=2.5)
    ax.gridlines(linewidth=0.25, color="0.55", alpha=0.5,
                 linestyle=(0, (3, 3)), zorder=3)
    ax.set_title(name, fontsize=FS_TITLE)
    panel_letter(ax, "abcdefgh"[k])
add_cbar(fig, im, (0.335, 0.035, 0.33, 0.014 / nrow * 2),
         "Depletion rate (cm yr$^{-1}$)", extend="max")
save(fig, "fig11_3_regional_hotspots")

print("\n  all figures written to", OUT_DIR)
print("=" * 70)
