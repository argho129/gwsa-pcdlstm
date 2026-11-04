# -*- coding: utf-8 -*-
r"""
================================================================================
fig_style.py  —  SHARED figure style for "Aquifer to Atmosphere" (Arpon, KUET)
================================================================================
Updated:
  - Main global maps use Robinson ("globe-like" oval) projection.
  - Smooth, polygon-masked HydroBASINS insets with bilinear interpolation.
  - Callout connectors (connect_inset_to_map) from map centroids to insets.
  - Multi-collinearity VIF computation (vif_from_columns / dynamic_vif_percell).
  - Robust version-safe Cartopy gridlines.
  - Panel letters now placed just outside the axes box (no overlap with data).
  - Inset gridlines capped to ~3 ticks per axis, with rotated (vertical) lat labels.
  - Bilinear-upsampling smoother for full global fields (fig5a/5b/11-style panels).
  - Relation diagnostics helper (mutual information) for non-linear predictor checks.
================================================================================
"""
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap

# ------------------------------------------------------------------ output dir
FIGDIR = r"D:\THESIS\Arpon_Thesis_Final\OUTPUTS\10_model\figures"
HYDROBASINS_GPKG = r"D:\THESIS\Arpon_Thesis_Final\DATA\HydroBasins\hybas_lev03_global.gpkg"

# ------------------------------------------------------------------ font chain
_SERIF_CHAIN = ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"]

def _resolve_serif():
    have = set(f.name for f in fm.fontManager.ttflist)
    for name in _SERIF_CHAIN:
        if name in have:
            return name
    return "serif"

SERIF = _resolve_serif()

# ------------------------------------------------------------------ type scale
SZ_TICK   = 7
SZ_LABEL  = 8
SZ_TITLE  = 8.5
SZ_PANEL  = 9
SZ_CBAR   = 7.5
SZ_ANNOT  = 6.5
SZ_GRID   = 5.5

# ------------------------------------------------------------------ colormaps
CMAP_ANOM  = "RdBu"
CMAP_TREND = "RdBu"
CMAP_MAG   = "viridis"
CMAP_SEQ_Y = "YlGnBu"
CMAP_PERM  = "cividis"
NODATA_GREY = "#d9d9d9"

CMAP_FRAC = LinearSegmentedColormap.from_list(
    "frac", ["#f7fbff", "#6baed6", "#08306b"])

def apply():
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": [SERIF],
        "mathtext.fontset": "dejavuserif",
        "axes.titlesize": SZ_TITLE,
        "axes.labelsize": SZ_LABEL,
        "xtick.labelsize": SZ_TICK,
        "ytick.labelsize": SZ_TICK,
        "legend.fontsize": SZ_TICK,
        "figure.titlesize": SZ_TITLE,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.edgecolor": "#333333",
        "savefig.dpi": 600,
        "figure.dpi": 120,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": True,
    })

# ------------------------------------------------------------ TRUE grid size
_GRID_SIZE_CACHE = {}

def _infer_grid_size(ds):
    """Infer the TRUE global grid size (nlat, nlon) from real cell spacing."""
    key = id(ds)
    if key in _GRID_SIZE_CACHE:
        return _GRID_SIZE_CACHE[key]
    j = np.asarray(ds["j"].values).astype(int)
    i = np.asarray(ds["i"].values).astype(int)
    clat = np.asarray(ds["clat"].values, dtype=float)
    clon = np.asarray(ds["clon"].values, dtype=float)
    j_lo, j_hi = int(j.min()), int(j.max())
    i_lo, i_hi = int(i.min()), int(i.max())
    clat_lo = clat[j == j_lo][0]; clat_hi = clat[j == j_hi][0]
    clon_lo = clon[i == i_lo][0]; clon_hi = clon[i == i_hi][0]
    dlat = (clat_hi - clat_lo) / (j_hi - j_lo)
    dlon = (clon_hi - clon_lo) / (i_hi - i_lo)
    nlat = int(round(180.0 / dlat))
    nlon = int(round(360.0 / dlon))
    _GRID_SIZE_CACHE[key] = (nlat, nlon)
    return nlat, nlon

# ------------------------------------------------------------ grid reconstruction
def grid_from_cells(ds, var, fill=np.nan):
    nlat, nlon = _infer_grid_size(ds)
    j = np.asarray(ds["j"].values).astype(int)
    i = np.asarray(ds["i"].values).astype(int)
    g = np.full((nlat, nlon), fill, dtype="float64")
    v = np.asarray(ds[var].values, dtype="float64")
    g[j, i] = v
    return g

def grid_from_cells_t(ds, var, tindex, fill=np.nan):
    nlat, nlon = _infer_grid_size(ds)
    j = np.asarray(ds["j"].values).astype(int)
    i = np.asarray(ds["i"].values).astype(int)
    g = np.full((nlat, nlon), fill, dtype="float64")
    v = np.asarray(ds[var].isel(time=tindex).values, dtype="float64")
    g[j, i] = v
    return g

def cell_reduce_to_grid(ds, var, reducer=np.nanmean, fill=np.nan):
    nlat, nlon = _infer_grid_size(ds)
    j = np.asarray(ds["j"].values).astype(int)
    i = np.asarray(ds["i"].values).astype(int)
    if isinstance(var, str):
        arr = np.asarray(ds[var].values, dtype="float64")
        if arr.ndim == 2:
            arr = reducer(arr, axis=0)
    else:
        arr = np.asarray(var, dtype="float64")
    g = np.full((nlat, nlon), fill, dtype="float64")
    g[j, i] = arr
    return g

# ------------------------------------------------------------ cartopy map helpers
def _lonlat_axes(nlat, nlon):
    lat = -90 + (np.arange(nlat) + 0.5) * (180.0 / nlat)
    lon = -180 + (np.arange(nlon) + 0.5) * (360.0 / nlon)
    return lat, lon

# ------------------------------------------------------------ HydroBASINS
_BASINS_GDF = None

def load_hydrobasins(path=HYDROBASINS_GPKG):
    global _BASINS_GDF
    if _BASINS_GDF is None:
        import geopandas as gpd
        _BASINS_GDF = gpd.read_file(path)
    return _BASINS_GDF

def add_hydrobasins_outline(ax, basins_gdf, edgecolor="#8a8a8a", linewidth=0.25, zorder=4):
    import cartopy.crs as ccrs
    ax.add_geometries(basins_gdf.geometry, crs=ccrs.PlateCarree(),
                      facecolor="none", edgecolor=edgecolor,
                      linewidth=linewidth, zorder=zorder)

def resolve_focus_basin_geometry(basins_gdf, cell_lons, cell_lats):
    """Union of every HydroBASINS level-3 polygon at least one focus cell sits inside."""
    import geopandas as gpd
    from shapely.geometry import Point
    pts = gpd.GeoDataFrame(
        geometry=[Point(lo, la) for lo, la in zip(cell_lons, cell_lats)],
        crs=basins_gdf.crs)
    try:
        joined = gpd.sjoin(pts, basins_gdf[["HYBAS_ID", "geometry"]],
                           how="left", predicate="within")
    except TypeError:
        joined = gpd.sjoin(pts, basins_gdf[["HYBAS_ID", "geometry"]],
                           how="left", op="within")
    ids = joined["HYBAS_ID"].dropna().unique()
    if len(ids) == 0:
        return None, []
    sub = basins_gdf[basins_gdf["HYBAS_ID"].isin(ids)]
    try:
        geom = sub.geometry.union_all()
    except AttributeError:
        geom = sub.unary_union
    return geom, list(ids)

DOMAIN_EXTENT = (-180, 180, -58, 84)

def global_map_axis(fig, rect=(0.0, 0.0, 1.0, 1.0), extent=DOMAIN_EXTENT,
                    projection="robinson", gridlines=True, basins=True):
    """Cartopy axis (Robinson oval by default), cropped to domain extent."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    proj = ccrs.Robinson() if projection == "robinson" else ccrs.PlateCarree()
    ax = fig.add_axes(rect, projection=proj)
    try:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    except Exception:
        ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="#f0f0f0", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
    ax.coastlines(linewidth=0.3, color="#555555", zorder=5)
    ax.spines["geo"].set_linewidth(0.5)
    if basins:
        try:
            add_hydrobasins_outline(ax, load_hydrobasins())
        except Exception as e:
            print(f"  [warn] HydroBASINS outline skipped: {e}")
    if gridlines:
        try:
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                              linewidth=0.3, color="#999999", alpha=0.5, linestyle="--")
            gl.top_labels = False; gl.right_labels = False
            gl.xlabel_style = {"size": SZ_TICK}; gl.ylabel_style = {"size": SZ_TICK}
        except Exception:
            ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                        linewidth=0.3, color="#999999", alpha=0.5, linestyle="--")
            print("  [note] version fallback: drawing unlabeled gridlines")
    return ax

def pcolor_global(ax, grid, cmap, vmin, vmax, nlat=None, nlon=None):
    import cartopy.crs as ccrs
    if nlat is None: nlat = grid.shape[0]
    if nlon is None: nlon = grid.shape[1]
    lat, lon = _lonlat_axes(nlat, nlon)
    LON, LAT = np.meshgrid(lon, lat)
    m = ax.pcolormesh(LON, LAT, np.ma.masked_invalid(grid),
                      transform=ccrs.PlateCarree(),
                      cmap=cmap, vmin=vmin, vmax=vmax, shading="auto",
                      rasterized=True)
    return m

# ------------------------------------------------------------ smooth global fields
def smooth_global_field(grid, nlat, nlon, upsample=3):
    """Bilinear-upsample a full (nlat,nlon) global grid for a smoother look.
       NaN (ocean/no-data) propagates naturally -- coastlines stay sharp-ish,
       interior fields look continuous instead of blocky 0.5deg cells."""
    from scipy.interpolate import RegularGridInterpolator
    lat, lon = _lonlat_axes(nlat, nlon)
    fine_lat = np.linspace(lat[0], lat[-1], (nlat - 1) * upsample + 1)
    fine_lon = np.linspace(lon[0], lon[-1], (nlon - 1) * upsample + 1)
    interp = RegularGridInterpolator((lat, lon), grid, method="linear",
                                     bounds_error=False, fill_value=np.nan)
    FLON, FLAT = np.meshgrid(fine_lon, fine_lat)
    fine_vals = interp(np.stack([FLAT.ravel(), FLON.ravel()], axis=-1)).reshape(FLAT.shape)
    return fine_lat, fine_lon, fine_vals

def pcolor_global_explicit(ax, lon, lat, grid, cmap, vmin, vmax):
    import cartopy.crs as ccrs
    LON, LAT = np.meshgrid(lon, lat)
    return ax.pcolormesh(LON, LAT, np.ma.masked_invalid(grid), transform=ccrs.PlateCarree(),
                         cmap=cmap, vmin=vmin, vmax=vmax, shading="auto", rasterized=True)

def pcolor_global_smooth(ax, grid, nlat, nlon, cmap, vmin, vmax, upsample=3):
    """Convenience wrapper: smooth_global_field + pcolor_global_explicit in one call."""
    flat, flon, fine = smooth_global_field(grid, nlat, nlon, upsample=upsample)
    return pcolor_global_explicit(ax, flon, flat, fine, cmap, vmin, vmax)

BASIN_BOXES = {
    "Bengal_GBM":      (73, 98, 21, 31),
    "Punjab_Indus":    (68, 80, 27, 34),
    "HighPlains":      (-105, -96, 32, 44),
    "CentralValley":   (-123, -118, 35, 41),
    "NorthChinaPlain": (112, 120, 34, 41),
}
BASIN_LABEL = {
    "Bengal_GBM": "Bengal (GBM)", "Punjab_Indus": "Punjab-Indus",
    "HighPlains": "High Plains", "CentralValley": "Central Valley",
    "NorthChinaPlain": "North China Plain",
}

# ------------------------------------------------------------ smooth, masked insets
def _polygon_contains(geom, LON, LAT):
    try:
        import shapely
        if int(shapely.__version__.split(".")[0]) >= 2:
            pts = shapely.points(LON.ravel(), LAT.ravel())
            return shapely.contains(geom, pts).reshape(LON.shape)
    except Exception:
        pass
    try:
        import shapely.vectorized as svec
        return svec.contains(geom, LON, LAT)
    except Exception:
        from shapely.geometry import Point
        inside = np.zeros(LON.shape, dtype=bool)
        for r in range(LON.shape[0]):
            for c in range(LON.shape[1]):
                inside[r, c] = geom.contains(Point(LON[r, c], LAT[r, c]))
        return inside

def smooth_basin_inset_field(grid, nlat, nlon, extent, geom, upsample=6):
    from scipy.interpolate import RegularGridInterpolator
    lat, lon = _lonlat_axes(nlat, nlon)
    lon0, lon1, lat0, lat1 = extent
    j0 = max(int(np.searchsorted(lat, lat0)) - 2, 0)
    j1 = min(int(np.searchsorted(lat, lat1)) + 2, nlat)
    i0 = max(int(np.searchsorted(lon, lon0)) - 2, 0)
    i1 = min(int(np.searchsorted(lon, lon1)) + 2, nlon)
    sub = grid[j0:j1, i0:i1]
    sub_lat = lat[j0:j1]; sub_lon = lon[i0:i1]
    fine_lat = np.linspace(sub_lat[0], sub_lat[-1], max(j1 - j0 - 1, 1) * upsample + 1)
    fine_lon = np.linspace(sub_lon[0], sub_lon[-1], max(i1 - i0 - 1, 1) * upsample + 1)
    interp = RegularGridInterpolator((sub_lat, sub_lon), sub, method="linear",
                                     bounds_error=False, fill_value=np.nan)
    FLON, FLAT = np.meshgrid(fine_lon, fine_lat)
    fine_vals = interp(np.stack([FLAT.ravel(), FLON.ravel()], axis=-1)).reshape(FLAT.shape)
    if geom is not None:
        inside = _polygon_contains(geom, FLON, FLAT)
        fine_vals = np.where(inside, fine_vals, np.nan)
    return FLON, FLAT, fine_vals

def add_basin_inset_true_shape(fig, rect, grid, basin_name, cmap, vmin, vmax,
                                geom, nlat=None, nlon=None, title=True,
                                pad_frac=0.12, upsample=6):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    if nlat is None: nlat = grid.shape[0]
    if nlon is None: nlon = grid.shape[1]
    ax = fig.add_axes(rect, projection=ccrs.PlateCarree())
    if geom is not None:
        minx, miny, maxx, maxy = geom.bounds
        padx = (maxx - minx) * pad_frac; pady = (maxy - miny) * pad_frac
        extent = [minx - padx, maxx + padx, miny - pady, maxy + pady]
    else:
        lon0, lon1, lat0, lat1 = BASIN_BOXES[basin_name]
        extent = [lon0, lon1, lat0, lat1]
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="#f5f5f5", zorder=0)
    ax.coastlines(linewidth=0.3, color="#555555", zorder=5)

    FLON, FLAT, fine_vals = smooth_basin_inset_field(grid, nlat, nlon, extent, geom,
                                                      upsample=upsample)
    ax.pcolormesh(FLON, FLAT, np.ma.masked_invalid(fine_vals),
                  transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax,
                  shading="auto", rasterized=True, zorder=3)
    if geom is not None:
        ax.add_geometries([geom], crs=ccrs.PlateCarree(), facecolor="none",
                          edgecolor="#111111", linewidth=0.9, zorder=6)

    # -- inset gridlines: capped to ~3 ticks/axis, lat labels rotated to run
    #    parallel to the (vertical) axis so they don't clutter small boxes --
    gl = ax.gridlines(draw_labels=True, linewidth=0.2, color="#aaaaaa", alpha=0.6)
    gl.top_labels = False; gl.right_labels = False
    gl.ylocator = mticker.MaxNLocator(nbins=3)
    gl.xlocator = mticker.MaxNLocator(nbins=3)
    gl.xlabel_style = {"size": SZ_GRID}
    gl.ylabel_style = {"size": SZ_GRID, "rotation": 90, "ha": "center", "va": "center"}

    for s in ax.spines.values():
        s.set_linewidth(0.5); s.set_edgecolor("#333333")
    if title:
        ax.set_title(BASIN_LABEL.get(basin_name, basin_name), fontsize=SZ_ANNOT, pad=2)
    return ax

def connect_inset_to_map(fig, map_ax, inset_ax, lon, lat, color="#444444"):
    """Thin dashed connector line + centroid dot linking inset back to geographic map."""
    import cartopy.crs as ccrs
    from matplotlib.patches import ConnectionPatch
    px, py = map_ax.projection.transform_point(lon, lat, ccrs.PlateCarree())
    map_ax.plot([px], [py], marker="o", ms=2.2, color=color,
               transform=map_ax.transData, zorder=7)
    con = ConnectionPatch(
        xyA=(px, py), coordsA=map_ax.transData,
        xyB=(0.5, 1.0), coordsB=inset_ax.transAxes,
        color=color, lw=0.4, linestyle="--", zorder=1)
    fig.add_artist(con)

# ------------------------------------------------------------ VIF (multicollinearity)
def vif_from_columns(cols_dict, min_obs=24):
    """Compute VIF_i = 1 / (1 - R^2_i) for each variable via OLS."""
    names = list(cols_dict.keys())
    X = np.column_stack([np.asarray(cols_dict[n], dtype=float) for n in names])
    ok = np.all(np.isfinite(X), axis=1)
    X = X[ok]
    n = X.shape[0]
    if n < max(min_obs, X.shape[1] + 2):
        return None, None, 0
    vifs = []
    for k in range(X.shape[1]):
        y = X[:, k]
        Xo = np.delete(X, k, axis=1)
        Xo1 = np.column_stack([Xo, np.ones(n)])
        coef, *_ = np.linalg.lstsq(Xo1, y, rcond=None)
        yhat = Xo1 @ coef
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        r2 = min(r2, 0.999999)
        vif = 1.0 / (1.0 - r2) if r2 < 0.999999 else np.inf
        vifs.append(vif)
    return names, vifs, n

def dynamic_vif_percell(dyn, names, cell_stride=5, min_obs=24):
    """Average per-cell VIF across domain cells."""
    ncell = dyn[names[0]].shape[1]
    acc = {v: [] for v in names}
    for c in range(0, ncell, cell_stride):
        cols = {v: dyn[v][:, c] for v in names}
        nm, vifs, n = vif_from_columns(cols, min_obs=min_obs)
        if nm is None:
            continue
        for v, val in zip(nm, vifs):
            if np.isfinite(val):
                acc[v].append(val)
    means = {v: (float(np.mean(acc[v])) if acc[v] else np.nan) for v in names}
    counts = {v: len(acc[v]) for v in names}
    return means, counts

# ------------------------------------------------------------ relation diagnostics
def mutual_info_safe(x, y, n_neighbors=3):
    """Mutual information between x and y. Tries sklearn (kNN estimator, good
       for continuous data); falls back to a binned histogram MI estimate if
       sklearn isn't available."""
    try:
        from sklearn.feature_selection import mutual_info_regression
        return float(mutual_info_regression(
            np.asarray(x).reshape(-1, 1), np.asarray(y),
            n_neighbors=n_neighbors, random_state=0)[0])
    except Exception:
        nbins = 20
        c_xy, _, _ = np.histogram2d(x, y, bins=nbins)
        pxy = c_xy / c_xy.sum()
        px = pxy.sum(axis=1, keepdims=True)
        py = pxy.sum(axis=0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where((pxy > 0) & (px > 0) & (py > 0), pxy / (px * py), 1.0)
            mi = np.nansum(pxy * np.log(ratio))
        return float(max(mi, 0.0))

# ------------------------------------------------------------ annotations
def panel_letter(ax, letter, outside=True, x=None, y=None, color="black"):
    """Bold lowercase panel label. outside=True (default) places it just
       above/left of the axes box, clear of data/titles. outside=False keeps
       the old inside-corner placement."""
    if x is None or y is None:
        x, y = (-0.02, 1.06) if outside else (0.01, 0.98)
    va = "bottom" if outside else "top"
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=SZ_PANEL,
            fontweight="bold", va=va, ha="left", color=color)

def hcolorbar(fig, mappable, rect, label):
    cax = fig.add_axes(rect)
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    cb.set_label(label, fontsize=SZ_CBAR)
    cb.ax.tick_params(labelsize=SZ_TICK, width=0.5, length=2)
    cb.outline.set_linewidth(0.5)
    return cb

# ------------------------------------------------------------ save (PNG + PDF)
def save(fig, name, figdir=FIGDIR, close=True):
    os.makedirs(figdir, exist_ok=True)
    png = os.path.join(figdir, name + ".png")
    pdf = os.path.join(figdir, name + ".pdf")
    fig.savefig(png, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    if close:
        plt.close(fig)
    print(f"  saved: {png}")
    print(f"  saved: {pdf}")


if __name__ == "__main__":
    apply()
    print("fig_style OK")
    print("  resolved serif font:", SERIF)
    print("  figdir:", FIGDIR)