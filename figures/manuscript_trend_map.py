"""
================================================================================
MANUSCRIPT FIGURE - CALIBRATED GLOBAL GROUNDWATER STORAGE TREND, 2002-2022
Arpon Sarkar, URP, KUET
================================================================================

PURPOSE
  One single publication-quality map of the post-calibration GWSA trend, with
  surface-water (waterbody) cells excluded. Nothing else. Intended to be dropped
  straight into the manuscript.

WHICH FILE IT READS, AND WHY
  Default: OUTPUTS\\08_final_product\\GWSA_KUET_v1_1D.nc
    This is the packaged product that Phases 9, 11 and 13 read, so the map
    agrees cell-for-cell with every number quoted elsewhere in the thesis.
  Alternative: OUTPUTS\\05_model_tensor\\gwsa_final.nc  (set SOURCE = "phase52")
    The direct Phase 5.2 output. Same calibrated field, one step earlier in the
    chain. Use it only if you are deliberately reporting the 5.2 product.
  Both are AFTER the per-cell variance-matching calibration, and in both the
  surface-water cells are already NaN. The script CONFIRMS the exclusion by
  counting non-finite cells and printing it, rather than assuming it.

SIGNIFICANCE
  DISPLAY = "significance_masked" (default) draws statistically significant
  cells in full colour and non-significant cells in a neutral grey, so the
  reader is never invited to interpret a trend the data cannot support.
  DISPLAY = "all" colours every valid cell regardless of significance.
  Both are defensible; the first is the stricter choice and is what most
  reviewers expect. State in the caption which one you used.

  The Phase 8 file carries TWO significance flags, 'trend_significant' and
  'significant_full'. They differ. SIG_VAR makes the choice explicit and the
  script prints both counts so the decision is auditable and consistent with
  Phase 9 / 13.

OUTPUT
  OUTPUTS\\11_maps\\fig_manuscript_trend_map.png   (600 dpi)
  OUTPUTS\\11_maps\\fig_manuscript_trend_map.pdf   (vector, fonts embedded)

Requires: cartopy, matplotlib, numpy, xarray
================================================================================
"""

from pathlib import Path

import numpy as np
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import TwoSlopeNorm

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

# ------------------------------------------------------------------ config --
ROOT = Path(r"D:\THESIS\Arpon_Thesis_Final\OUTPUTS")
SOURCE = "phase8"                 # "phase8" (recommended) or "phase52"
PATHS = {"phase8": ROOT / "08_final_product" / "GWSA_KUET_v1_1D.nc",
         "phase52": ROOT / "05_model_tensor" / "gwsa_final.nc"}
OUT_DIR = ROOT / "11_maps"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_NAME = "fig_manuscript_trend_map"

SIG_VAR = "trend_significant"     # or "significant_full"
DISPLAY = "significance_masked"   # or "all"
PERIOD_LABEL = "2002-2022"

GRID_RES = 0.5
N_LAT, N_LON = 360, 720
CLIP_PCTL = 98                    # colour saturation percentile
DPI = 600
FIGSIZE = (11.0, 6.0)             # single-column-friendly aspect for a global map
ANNOTATE = True                   # small stats line under the map

# ------------------------------------------------------------- typography ---
_avail = {f.name for f in font_manager.fontManager.ttflist}
_chain = ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif", "serif"]
_chosen = next((f for f in _chain if f in _avail), "serif")
if _chosen != "Times New Roman":
    print(f"[WARN] Times New Roman unavailable; using '{_chosen}'.")

FS_TICK, FS_LABEL, FS_TITLE = 10, 11, 13
matplotlib.rcParams.update({
    "font.family": "serif", "font.serif": _chain, "mathtext.fontset": "stix",
    "font.size": FS_TICK, "axes.labelsize": FS_LABEL, "axes.titlesize": FS_TITLE,
    "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
    "axes.linewidth": 0.6, "savefig.dpi": DPI, "figure.dpi": 120,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

LAND, OCEAN = "#f4f2ef", "#ffffff"
COAST, BORDER = "#4f4f4f", "#9c9c9c"
NONSIG = "#c9c9c9"                # neutral grey for non-significant cells

print("=" * 72)
print("MANUSCRIPT TREND MAP")
print("=" * 72)


def pick(ds, *names):
    for n in names:
        if n in ds.data_vars or n in ds.coords:
            return ds[n]
    raise KeyError(f"None of {names} found. Available: {list(ds.data_vars)}")


# ------------------------------------------------------------------- load ---
in_path = PATHS[SOURCE]
if not in_path.exists():
    raise FileNotFoundError(f"{in_path} not found. Check SOURCE.")
ds = xr.open_dataset(in_path)
print(f"  source   : {SOURCE}  ->  {in_path.name}")
print(f"  variables: {list(ds.data_vars)}")

trend = pick(ds, "gwsa_trend", "trend")
clat = pick(ds, "clat", "lat")
clon = pick(ds, "clon", "lon")

sig_name = SIG_VAR if SIG_VAR in ds.variables else None
for cand in ("trend_significant", "significant_full", "trend_significant"):
    if cand in ds.variables:
        n_flag = int((np.asarray(ds[cand].values) == 1).sum())
        print(f"  {cand}: {n_flag} cells flagged"
              + ("   <-- used" if cand == sig_name else ""))
if sig_name is None and DISPLAY == "significance_masked":
    raise KeyError(f"SIG_VAR '{SIG_VAR}' absent but DISPLAY requires it.")

tr = np.asarray(trend.values, dtype=float)
lat_v = np.asarray(clat.values, dtype=float)
lon_v = np.asarray(clon.values, dtype=float)
n_cell = tr.size

# ------------------------------------------- confirm the waterbody exclusion -
finite = np.isfinite(tr)
if "keep" in ds.variables:
    keep = np.asarray(ds["keep"].values).astype(bool)
    valid = finite & keep
    print(f"  'keep' present: {int(keep.sum())} kept")
else:
    valid = finite
    print("  no 'keep' variable; using finite trend values as the valid mask")
n_excluded = n_cell - int(valid.sum())
print(f"  valid cells: {int(valid.sum())} / {n_cell}   "
      f"({n_excluded} excluded, incl. surface water)")
if n_excluded == 0:
    print("  [WARN] nothing is excluded. Surface-water cells may still be present "
          "- verify this file is post-Phase-5.2.")

sig = (np.asarray(ds[sig_name].values) == 1) if sig_name else np.ones(n_cell, bool)
n_sig = int((valid & sig).sum())
print(f"  significant among valid: {n_sig} ({100 * n_sig / max(valid.sum(), 1):.1f}%)")

# ----------------------------------------------------------------- regrid ---
row = np.round((lat_v + 89.75) / GRID_RES).astype(int)
col = np.round((lon_v + 179.75) / GRID_RES).astype(int)
in_range = (row >= 0) & (row < N_LAT) & (col >= 0) & (col < N_LON)
_sel = valid & in_range
_flat = row[_sel] * N_LON + col[_sel]
print(f"  regrid: {_flat.size} placed, {np.unique(_flat).size} unique grid points, "
      f"{int((valid & ~in_range).sum())} out of range")
if np.unique(_flat).size != _flat.size:
    print("  [WARN] duplicate grid points detected; later values overwrite earlier.")


def to_grid(vals, mask):
    g = np.full((N_LAT, N_LON), np.nan)
    ok = mask & in_range & np.isfinite(vals)
    g[row[ok], col[ok]] = vals[ok]
    return g


if DISPLAY == "significance_masked":
    main_grid = to_grid(tr, valid & sig)
    grey_grid = to_grid(np.ones_like(tr), valid & ~sig)
else:
    main_grid = to_grid(tr, valid)
    grey_grid = None

lat_edges = -90 + GRID_RES * np.arange(N_LAT + 1)
lon_edges = -180 + GRID_RES * np.arange(N_LON + 1)
LON2D, LAT2D = np.meshgrid((lon_edges[:-1] + lon_edges[1:]) / 2,
                           (lat_edges[:-1] + lat_edges[1:]) / 2)

lim = float(np.nanpercentile(np.abs(main_grid), CLIP_PCTL))
lim = float(np.round(lim, 2)) or 1.0
print(f"  colour scale: +/-{lim:.2f} cm/yr ({CLIP_PCTL}th pctl of |trend|)")
tv = tr[valid]
print(f"  trend range (valid): {np.nanmin(tv):+.2f} .. {np.nanmax(tv):+.2f} cm/yr; "
      f"median {np.nanmedian(tv):+.3f}")

# ------------------------------------------------------------------- plot ---
fig = plt.figure(figsize=FIGSIZE)
ax = fig.add_axes((0.035, 0.155, 0.93, 0.79), projection=ccrs.Robinson())
ax.set_global()
ax.add_feature(cfeature.OCEAN, facecolor=OCEAN, zorder=0)
ax.add_feature(cfeature.LAND, facecolor=LAND, zorder=0)

# graticule with degree labels; degrade gracefully on older cartopy
try:
    gl = ax.gridlines(draw_labels=True, linewidth=0.25, color="0.55", alpha=0.5,
                      linestyle=(0, (3, 3)), zorder=3,
                      xlocs=np.arange(-180, 181, 60), ylocs=np.arange(-60, 91, 30))
    gl.top_labels = gl.right_labels = False
    gl.xformatter, gl.yformatter = LongitudeFormatter(), LatitudeFormatter()
    gl.xlabel_style = {"size": FS_TICK, "color": "0.15"}
    gl.ylabel_style = {"size": FS_TICK, "color": "0.15"}
except Exception as exc:
    print(f"  [note] labelled graticule unavailable ({type(exc).__name__}); "
          f"drawing unlabelled")
    ax.gridlines(draw_labels=False, linewidth=0.25, color="0.55", alpha=0.5,
                 linestyle=(0, (3, 3)), zorder=3)

# non-significant cells first, so significant colour sits on top
if grey_grid is not None:
    ax.pcolormesh(LON2D, LAT2D, grey_grid, transform=ccrs.PlateCarree(),
                  cmap=matplotlib.colors.ListedColormap([NONSIG]), vmin=0, vmax=1,
                  shading="auto", zorder=1, rasterized=True)

im = ax.pcolormesh(LON2D, LAT2D, main_grid, transform=ccrs.PlateCarree(),
                   cmap="RdBu", norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim),
                   shading="auto", zorder=2, rasterized=True)

ax.coastlines(linewidth=0.35, color=COAST, zorder=4)
ax.add_feature(cfeature.BORDERS, linewidth=0.18, color=BORDER, zorder=4)
ax.spines["geo"].set_linewidth(0.6)
ax.spines["geo"].set_edgecolor("0.35")
ax.set_title(f"Calibrated groundwater storage trend, {PERIOD_LABEL}",
             fontsize=FS_TITLE, pad=10)

cax = fig.add_axes((0.315, 0.105, 0.37, 0.020))
cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="both")
cb.set_label("Trend (cm yr$^{-1}$)", fontsize=FS_LABEL)
cb.ax.tick_params(labelsize=FS_TICK, width=0.6, length=2.5)
cb.outline.set_linewidth(0.5)

if DISPLAY == "significance_masked":
    lax = fig.add_axes((0.735, 0.104, 0.021, 0.022))
    lax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=NONSIG, edgecolor="0.4",
                                linewidth=0.4))
    lax.set_xticks([]); lax.set_yticks([]); lax.set_frame_on(False)
    fig.text(0.762, 0.115, "not significant", fontsize=FS_TICK, va="center",
             color="0.2")

if ANNOTATE:
    note = (f"{int(valid.sum()):,} cells  |  {n_sig:,} significant "
            f"({100 * n_sig / max(valid.sum(), 1):.0f}%)  |  "
            f"surface-water cells excluded  |  colour saturates at "
            f"$\\pm${lim:.2f} cm yr$^{{-1}}$")
    fig.text(0.5, 0.045, note, ha="center", fontsize=FS_TICK - 1, color="0.35")

fig.savefig(OUT_DIR / f"{OUT_NAME}.png", dpi=DPI, bbox_inches="tight",
            facecolor="white")
fig.savefig(OUT_DIR / f"{OUT_NAME}.pdf", bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"\n  wrote {OUT_DIR / (OUT_NAME + '.png')}")
print(f"  wrote {OUT_DIR / (OUT_NAME + '.pdf')}")
print("=" * 72)
print("  Caption skeleton:")
print(f"  Calibrated groundwater storage anomaly trend, {PERIOD_LABEL}, from the")
print("  physics-constrained reconstruction after per-cell variance-matching")
print("  calibration. Blue indicates storage gain, red storage loss. Cells whose")
print("  Theil-Sen confidence interval includes zero are shown in grey. Surface-")
print("  water cells are excluded. Colour saturates at the 98th percentile of")
print("  |trend|; extremes beyond that range are clipped.")
print("=" * 72)