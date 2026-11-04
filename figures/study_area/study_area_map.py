"""
study_area_map.py
Study area map + well-network coverage comparison for the global GWSA
reconstruction.

Figure layout
    (a) Global map: reconstruction domain, WHYMAP aquifer systems,
        HydroBASINS level-3 basin outlines, and both well networks drawn
        as coloured points (GROW and Jasechko et al. 2024). Point overlap
        renders as tonal density, so no separate density grid is needed.
    (b) Wells reporting per year for each network, with the analysis
        window shaded.
    (c) 1 deg grid cells containing at least one well, split into shared
        cells and cells unique to each network.

All furniture sits in reserved space: the legend has its own strip below
the map, the scale bar sits over open ocean, and graticule labels are
kept clear of both.

Usage
    python study_area_map.py

Requires: matplotlib, numpy, pandas, pyarrow, geopandas, cartopy, xarray.
"""

import os
import re
import glob
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from fig_style import apply_style, save_fig

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ======================================================================
# CONFIG
# ======================================================================

BASINS_FP   = r"D:\THESIS\Arpon_Thesis_Final\DATA\hydrobasins\hybas_lev03_global.gpkg"
GROW_FP     = r"D:\THESIS\Arpon_Thesis_Final\DATA\GROW\grow_timeseries.parquet"
JASECHKO_FP = r"D:\THESIS\FINAL\Data\GW_Level\AnnualDepthToGroundwater.csv"
DOMAIN_NC   = r"D:\THESIS\Arpon_Thesis_Final\OUTPUTS\08_final_product\GWSA_KUET_v1_1D.nc"

# This study's OWN water and permanent-ice cell masks, on the 0.5-deg grid
# (Phase 0.5 reliability mask). These REPLACE the WHYMAP/Natural Earth layers:
# the map now shows the exact cells this study identified, merged into clean
# outlines. surface_water and permanent_ice both live here as (lat, lon) grids.
RELIABILITY_NC = r"D:\THESIS\Arpon_Thesis_Final\OUTPUTS\00_reliability_mask\reliability_mask.nc"
MASK_GRID_DEG  = 0.5   # resolution of the reliability-mask cells

# Companion GROW file with one row per site and its coordinates.
# None = auto-search (it found grow_attributes.parquet last run).
GROW_META_FP = None

# Aquifer polygons. None = auto-search the roots below.
AQUIFER_FP = None
SEARCH_ROOTS = [r"D:\THESIS\Arpon_Thesis_Final\DATA",
                r"D:\THESIS\FINAL\Data",
                r"D:\THESIS"]

# WHYMAP carries water bodies and permanent ice as classes inside the
# same polygon layer as the aquifers. None = auto-detect which attribute
# field holds those classes (the script prints what it found and the
# class values it matched, so you can pin it here if the guess is wrong).
AQ_CLASS_FIELD = None
# If WHYMAP yields no water or ice class, fall back to Natural Earth
# (downloaded and cached by cartopy on first use; needs internet once).
NE_FALLBACK = True

OUTDIR = r"D:\THESIS\Arpon_Thesis_Final\OUTPUTS\figures"
STEM   = "fig01_study_area"

REBUILD_CACHE  = False       # True re-reads the parquet and the CSV
ANALYSIS_YEARS = (2002, 2022)
COVERAGE_RES   = 1.0         # degrees, coverage comparison in panel (c)
FIG_W_MM       = 190
PANEL_SCALE    = 1.0

# Well point styling. Size is in points^2; alpha below 1 lets overlapping
# points build up tone, which is what makes dense regions read as dense.
GROW_MS, GROW_ALPHA = 0.55, 0.40
JAS_MS,  JAS_ALPHA  = 0.55, 0.40
DRAW_ORDER = "grow_first"    # or "jasechko_first"

C_LAND, C_OCEAN, C_COAST = "#f7f5f1", "#ffffff", "#6f6f6f"
C_BASIN                  = "#4a6785"
C_AQ_FACE, C_AQ_EDGE     = "#bcd9c4", "#7fa98b"
C_WATER, C_WATER_EDGE    = "#a9c9e6", "#6f96bd"
C_ICE, C_ICE_EDGE        = "#eef3f8", "#b6c6d6"
C_DOMAIN                 = "#e4e2dd"
C_GROW, C_JAS            = "#8c1d4d", "#12798f"

# ======================================================================
# Column detection
# ======================================================================

LON_KEYS = ("lon", "long", "longitude", "lng", "x", "dec_long_va",
            "lon_dd", "well_lon", "site_lon", "longitude_dd")
LAT_KEYS = ("lat", "latitude", "y", "dec_lat_va", "lat_dd",
            "well_lat", "site_lat", "latitude_dd")
ID_KEYS  = ("grow_id", "site_id", "well_id", "id", "station_id", "site",
            "well", "station", "point_id", "well_code", "code", "name")
TIME_KEYS = ("year", "date", "time", "datetime", "obs_date",
             "measurement_date", "sample_date")

YEAR_RE = re.compile(r"^(18|19|20)\d{2}$")
YEAR_IN = re.compile(r"(18|19|20)\d{2}")


def _tokens(name):
    return [t for t in re.split(r"[^a-z0-9]+", str(name).lower()) if t]


def _pick(cols, keys, exclude=(), loose=False):
    """
    Exact name match, then whole-token match. Loose substring matching is
    opt-in: a substring search for "lat" also matches "plateaus", and "x"
    or "y" match almost anything, which silently returns the wrong column.
    """
    low = {str(c).lower().strip(): c for c in cols if c not in exclude}
    for k in keys:
        if k in low:
            return low[k]
    for k in keys:
        for lc, orig in low.items():
            if k in _tokens(lc):
                return orig
    if loose:
        for k in keys:
            if len(k) < 3:
                continue
            for lc, orig in low.items():
                if k in lc:
                    return orig
    return None


def _to_year(s):
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() > 0.9 and num.dropna().between(1800, 2100).mean() > 0.9:
        return num.astype("Int64")
    return pd.to_datetime(s, errors="coerce").dt.year.astype("Int64")


def _year_columns(cols):
    """Detect a wide layout where each year is its own column."""
    exact = [c for c in cols if YEAR_RE.match(str(c).strip())]
    if len(exact) >= 5:
        return exact, {c: int(str(c).strip()) for c in exact}
    embedded = [c for c in cols if YEAR_IN.search(str(c))]
    if len(embedded) >= 5:
        return embedded, {c: int(YEAR_IN.search(str(c)).group()) for c in embedded}
    return [], {}


# ======================================================================
# Vector layers
# ======================================================================

def _find_aquifers():
    if AQUIFER_FP and os.path.exists(AQUIFER_FP):
        return AQUIFER_FP
    pats = ["**/*whymap*.shp", "**/*WHYMAP*.shp", "**/*aquifer*.shp",
            "**/*Aquifer*.shp", "**/*whymap*.gpkg"]
    for root in SEARCH_ROOTS:
        if not os.path.isdir(root):
            continue
        for p in pats:
            hits = glob.glob(os.path.join(root, p), recursive=True)
            if hits:
                hits.sort(key=lambda f: -os.path.getsize(f))
                return hits[0]
    return None


def _read_vector(fp, what, keep_poly=True):
    import geopandas as gpd
    print(f"  [..]   reading {what} ...", flush=True)
    if not fp or not os.path.exists(fp):
        print(f"  [SKIP] {what}: not found ({fp})", flush=True)
        return None
    try:
        g = gpd.read_file(fp)
    except Exception as e:
        print(f"  [FAIL] {what}: {e}", flush=True)
        return None
    g = g.set_crs(4326) if g.crs is None else g.to_crs(4326)
    g = g[g.geometry.notna() & ~g.geometry.is_empty]
    if keep_poly:
        g = g[g.geom_type.isin(["Polygon", "MultiPolygon"])]
    print(f"  [ok]   {what}: {len(g)} features  <- {os.path.basename(fp)}",
          flush=True)
    return g if len(g) else None


# ----------------------------------------------------------------------
# Water bodies and permanent ice
# ----------------------------------------------------------------------

WATER_WORDS = ("lake", "water body", "waterbody", "water bodies", "reservoir",
               "lagoon", "inland water", "freshwater", "saltwater", "salt lake",
               "inland sea")
ICE_WORDS = ("ice sheet", "icesheet", "ice cap", "icecap", "ice shelf",
             "glacier", "glaciated", "continuous ice", "permanent ice",
             "perennial ice", "inland ice", "snow and ice")


def _classify(series, words):
    s = series.astype(str).str.lower().str.strip()
    return s.apply(lambda v: any(w in v for w in words))


def _split_whymap(g):
    """
    Water bodies and permanent ice are NO LONGER taken from WHYMAP. WHYMAP is
    still read only for the AQUIFER polygons; water/ice now come from this
    study's own 0.5-deg reliability masks (see _study_masks). This function is
    therefore reduced to a pass-through: it returns the aquifer layer unchanged
    and (None, None) for water/ice, which _study_masks then fills.

    Returns (aquifers, None, None).
    """
    return (g if (g is not None and len(g)) else None), None, None


def _mask_to_outline(mask2d, lat, lon, label, grid=MASK_GRID_DEG):
    """
    Turn a boolean (lat, lon) cell mask into MERGED polygon outlines: each True
    cell becomes a grid-square, all squares are dissolved with a single unary
    union so adjacent cells fuse into clean regions (no pixelated per-cell
    edges). Returns a one-row GeoDataFrame carrying the dissolved geometry, or
    None if the mask is empty.
    """
    import geopandas as gpd
    from shapely.geometry import box
    from shapely.ops import unary_union

    mask2d = np.asarray(mask2d).astype(bool)
    if not mask2d.any():
        print(f"         {label}: mask empty -> not drawn")
        return None
    jj, ii = np.where(mask2d)                 # row (lat) , col (lon) indices
    h = grid / 2.0
    cell_lat = np.asarray(lat)[jj]
    cell_lon = np.asarray(lon)[ii]
    boxes = [box(x - h, y - h, x + h, y + h) for x, y in zip(cell_lon, cell_lat)]
    merged = unary_union(boxes)               # dissolve adjacent cells
    gdf = gpd.GeoDataFrame({"kind": [label]}, geometry=[merged], crs=4326)
    print(f"         {label}: {int(mask2d.sum())} cells -> merged outline")
    return gdf


def _study_masks():
    """
    Load THIS STUDY's surface-water and permanent-ice cell masks from the
    Phase 0.5 reliability file and dissolve each into clean outlines. Both are
    on the same 0.5-deg (lat, lon) grid. Returns (water_gdf, ice_gdf); either
    may be None if its mask is empty or the file is missing.
    """
    import xarray as xr

    if not os.path.exists(RELIABILITY_NC):
        print(f"  [SKIP] study masks: reliability file not found "
              f"({RELIABILITY_NC}) -> no water/ice drawn")
        return None, None
    print(f"  [..]   reading study water/ice masks <- "
          f"{os.path.basename(RELIABILITY_NC)}", flush=True)
    ds = xr.open_dataset(RELIABILITY_NC)
    lat = ds["lat"].values
    lon = ds["lon"].values

    water = None
    if "surface_water" in ds:
        water = _mask_to_outline(ds["surface_water"].values, lat, lon,
                                 "surface water")
    else:
        print("         surface_water not in reliability file")

    ice = None
    if "permanent_ice" in ds:
        ice = _mask_to_outline(ds["permanent_ice"].values, lat, lon,
                               "permanent ice")
    else:
        print("         permanent_ice not in reliability file")

    ds.close()
    return water, ice


# ======================================================================
# GROW
# ======================================================================

META_HINTS = ("attribute", "meta", "station", "site", "well", "location",
              "coord", "point", "header", "info")
META_EXTS = (".csv", ".parquet", ".gpkg", ".shp", ".xlsx", ".txt", ".tsv")


def _peek_columns(fp):
    try:
        ext = os.path.splitext(fp)[1].lower()
        if ext == ".parquet":
            import pyarrow.parquet as pq
            return [f.name for f in pq.ParquetFile(fp).schema_arrow]
        if ext in (".csv", ".txt", ".tsv"):
            sep = "\t" if ext in (".txt", ".tsv") else ","
            return list(pd.read_csv(fp, nrows=5, sep=sep, engine="python").columns)
        if ext == ".xlsx":
            return list(pd.read_excel(fp, nrows=5).columns)
        if ext in (".gpkg", ".shp"):
            import geopandas as gpd
            return list(gpd.read_file(fp, rows=5).columns)
    except Exception:
        pass
    return []


def _read_table(fp):
    ext = os.path.splitext(fp)[1].lower()
    if ext == ".parquet":
        return pd.read_parquet(fp)
    if ext in (".csv", ".txt", ".tsv"):
        sep = "\t" if ext in (".txt", ".tsv") else ","
        return pd.read_csv(fp, sep=sep, engine="python")
    if ext == ".xlsx":
        return pd.read_excel(fp)
    if ext in (".gpkg", ".shp"):
        import geopandas as gpd
        g = gpd.read_file(fp)
        g = g.set_crs(4326) if g.crs is None else g.to_crs(4326)
        if _pick(g.columns, LON_KEYS) is None:
            g["longitude"], g["latitude"] = g.geometry.x, g.geometry.y
        return pd.DataFrame(g.drop(columns="geometry", errors="ignore"))
    raise ValueError(f"unsupported metadata format: {ext}")


def _find_grow_metadata(ts_fp, id_col):
    folder = os.path.dirname(ts_fp)
    cands = []
    for root in (folder, os.path.dirname(folder)):
        if not os.path.isdir(root):
            continue
        for f in glob.glob(os.path.join(root, "**", "*"), recursive=True):
            if (os.path.isfile(f) and f != ts_fp
                    and os.path.splitext(f)[1].lower() in META_EXTS):
                cands.append(f)

    def rank(f):
        n = os.path.basename(f).lower()
        return (0 if any(h in n for h in META_HINTS) else 1, os.path.getsize(f))

    print("         searching for a GROW metadata file with coordinates")
    for f in sorted(set(cands), key=rank):
        cols = _peek_columns(f)
        if not cols:
            continue
        lon_c, lat_c = _pick(cols, LON_KEYS), _pick(cols, LAT_KEYS)
        this_id = _pick(cols, (id_col.lower(),) + ID_KEYS)
        if lon_c and lat_c and this_id:
            print(f"         -> {os.path.basename(f)} "
                  f"(id={this_id}, lon={lon_c}, lat={lat_c})")
            return f, this_id, lon_c, lat_c
    print("         no companion file with coordinates found")
    return None, None, None, None


def _load_grow(fp, cache_dir):
    sites_c = os.path.join(cache_dir, "grow_sites.csv")
    years_c = os.path.join(cache_dir, "grow_site_years.csv")
    if not REBUILD_CACHE and os.path.exists(sites_c) and os.path.exists(years_c):
        sites, years = pd.read_csv(sites_c), pd.read_csv(years_c)
        print(f"  [ok]   GROW (cached): {len(sites):,} sites, "
              f"{len(years):,} site-years")
        return sites, years

    if not os.path.exists(fp):
        print(f"  [SKIP] GROW: not found ({fp})")
        return None, None

    import pyarrow.parquet as pq

    pf = pq.ParquetFile(fp)
    schema = [f.name for f in pf.schema_arrow]
    n_rows = pf.metadata.num_rows
    print(f"  [..]   GROW: {n_rows:,} rows, {len(schema)} columns, "
          f"{os.path.getsize(fp) / 1e9:.2f} GB on disk")

    id_c  = _pick(schema, ID_KEYS)
    lon_c, lat_c = _pick(schema, LON_KEYS), _pick(schema, LAT_KEYS)
    t_c   = "year" if "year" in schema else _pick(schema, TIME_KEYS)
    if id_c is None:
        raise ValueError(f"No site-id column in GROW. Schema: {schema}")

    meta = None
    if lon_c is None or lat_c is None:
        if GROW_META_FP and os.path.exists(GROW_META_FP):
            mfp = GROW_META_FP
            mc = _peek_columns(mfp)
            mid = _pick(mc, (id_c.lower(),) + ID_KEYS)
            mlon, mlat = _pick(mc, LON_KEYS), _pick(mc, LAT_KEYS)
        else:
            mfp, mid, mlon, mlat = _find_grow_metadata(fp, id_c)
        if not (mfp and mid and mlon and mlat):
            raise ValueError(
                "The GROW time series carries no coordinates and no usable "
                "companion file was found. Set GROW_META_FP in CONFIG.")
        meta = _read_table(mfp)[[mid, mlon, mlat]].rename(
            columns={mid: "site", mlon: "lon", mlat: "lat"})
        meta["lon"] = pd.to_numeric(meta["lon"], errors="coerce")
        meta["lat"] = pd.to_numeric(meta["lat"], errors="coerce")
        meta = (meta.dropna(subset=["lon", "lat"])
                    .drop_duplicates(subset=["site"]))
        meta = meta[meta.lon.between(-180, 180) & meta.lat.between(-90, 90)]
        print(f"         metadata: {len(meta):,} sites with coordinates")

    want = [c for c in dict.fromkeys([id_c, lon_c, lat_c, t_c]) if c]
    site_rows, year_rows, ids_seen = [], [], set()
    seen, nxt = 0, 5_000_000
    for batch in pf.iter_batches(batch_size=500_000, columns=want):
        df = batch.to_pandas()
        seen += len(df)
        if meta is None:
            df[lon_c] = pd.to_numeric(df[lon_c], errors="coerce")
            df[lat_c] = pd.to_numeric(df[lat_c], errors="coerce")
            df = df.dropna(subset=[lon_c, lat_c])
            if df.empty:
                continue
            site_rows.append(df[[id_c, lon_c, lat_c]].drop_duplicates(subset=[id_c]))
        else:
            ids_seen.update(df[id_c].unique().tolist())
        if t_c is not None:
            df = df.copy()
            df["_yr"] = _to_year(df[t_c])
            yr = df.dropna(subset=["_yr"])[[id_c, "_yr"]].drop_duplicates()
            if len(yr):
                year_rows.append(yr)
        if seen >= nxt:
            print(f"         ... {seen:,} / {n_rows:,} rows")
            nxt += 5_000_000

    if meta is None:
        sites = (pd.concat(site_rows, ignore_index=True)
                   .drop_duplicates(subset=[id_c])
                   .rename(columns={id_c: "site", lon_c: "lon", lat_c: "lat"}))
    else:
        sites = meta[meta["site"].isin(ids_seen)].copy()

    if year_rows:
        years = (pd.concat(year_rows, ignore_index=True).drop_duplicates()
                   .rename(columns={id_c: "site", "_yr": "year"}))
        years["year"] = years["year"].astype(int)
        years = years[years["site"].isin(set(sites["site"]))]
    else:
        years = pd.DataFrame(columns=["site", "year"])

    os.makedirs(cache_dir, exist_ok=True)
    sites.to_csv(sites_c, index=False)
    years.to_csv(years_c, index=False)
    print(f"  [ok]   GROW: {len(sites):,} georeferenced sites, "
          f"{len(years):,} site-years (cached)")
    return sites, years


# ======================================================================
# Jasechko  (handles both long and wide, one-column-per-year, layouts)
# ======================================================================

def _load_jasechko(fp, cache_dir):
    sites_c = os.path.join(cache_dir, "jas_sites.csv")
    years_c = os.path.join(cache_dir, "jas_site_years.csv")
    if not REBUILD_CACHE and os.path.exists(sites_c) and os.path.exists(years_c):
        sites, years = pd.read_csv(sites_c), pd.read_csv(years_c)
        print(f"  [ok]   Jasechko (cached): {len(sites):,} sites, "
              f"{len(years):,} site-years")
        return sites, years

    if not os.path.exists(fp):
        print(f"  [SKIP] Jasechko: not found ({fp})")
        return None, None

    cols = list(pd.read_csv(fp, nrows=200).columns)
    lon_c, lat_c = _pick(cols, LON_KEYS), _pick(cols, LAT_KEYS)
    id_c = _pick(cols, ID_KEYS)
    if lon_c is None or lat_c is None:
        raise ValueError(f"No lon/lat column in the Jasechko CSV. Columns: {cols}")

    ycols, ymap = _year_columns(cols)
    t_c = None if ycols else _pick(cols, TIME_KEYS)
    layout = "wide" if ycols else ("long" if t_c else "sites-only")
    print(f"         columns -> id={id_c} lon={lon_c} lat={lat_c} "
          f"layout={layout}"
          + (f" ({len(ycols)} year columns, {min(ymap.values())}"
             f"-{max(ymap.values())})" if ycols else ""))

    usecols = [c for c in dict.fromkeys([id_c, lon_c, lat_c, t_c]) if c] + ycols
    site_rows, year_rows = [], []
    for df in pd.read_csv(fp, usecols=usecols, chunksize=200_000):
        df[lon_c] = pd.to_numeric(df[lon_c], errors="coerce")
        df[lat_c] = pd.to_numeric(df[lat_c], errors="coerce")
        df = df.dropna(subset=[lon_c, lat_c])
        df = df[df[lon_c].between(-180, 180) & df[lat_c].between(-90, 90)]
        if df.empty:
            continue

        key = id_c
        if key is None:
            df = df.assign(_site=df[lon_c].round(6).astype(str) + "_"
                                 + df[lat_c].round(6).astype(str))
            key = "_site"

        site_rows.append(df[[key, lon_c, lat_c]].drop_duplicates(subset=[key]))

        if layout == "wide":
            m = df.melt(id_vars=[key], value_vars=ycols,
                        var_name="_col", value_name="_v").dropna(subset=["_v"])
            if len(m):
                m["year"] = m["_col"].map(ymap).astype(int)
                year_rows.append(m[[key, "year"]].drop_duplicates())
        elif layout == "long":
            d = df.copy()
            d["year"] = _to_year(d[t_c])
            d = d.dropna(subset=["year"])
            if len(d):
                year_rows.append(
                    d[[key, "year"]].astype({"year": int}).drop_duplicates())

    key = id_c if id_c is not None else "_site"
    sites = (pd.concat(site_rows, ignore_index=True)
               .drop_duplicates(subset=[key])
               .rename(columns={key: "site", lon_c: "lon", lat_c: "lat"}))
    if year_rows:
        years = (pd.concat(year_rows, ignore_index=True).drop_duplicates()
                   .rename(columns={key: "site"}))
        years["year"] = years["year"].astype(int)
    else:
        years = pd.DataFrame(columns=["site", "year"])

    os.makedirs(cache_dir, exist_ok=True)
    sites.to_csv(sites_c, index=False)
    years.to_csv(years_c, index=False)
    print(f"  [ok]   Jasechko: {len(sites):,} sites, {len(years):,} site-years")
    return sites, years


# ======================================================================
# Domain
# ======================================================================

def _domain_mask(fp):
    if not os.path.exists(fp):
        print(f"  [SKIP] domain: not found ({fp})")
        return None
    import xarray as xr

    ds = xr.open_dataset(fp)
    names = list(dict.fromkeys(list(ds.variables) + list(ds.coords)))
    print(f"         netcdf dims={dict(ds.sizes)} vars={list(ds.data_vars)[:8]}")
    lat_n = _pick(names, ("clat", "lat", "latitude", "cell_lat", "y"))
    lon_n = _pick(names, ("clon", "lon", "longitude", "cell_lon", "x"))
    if lat_n is None or lon_n is None:
        print(f"  [SKIP] domain: no lat/lon variable in {names[:15]}")
        return None

    lat = np.asarray(ds[lat_n].values).squeeze()
    lon = np.asarray(ds[lon_n].values).squeeze()
    if lat.ndim == 1 and lon.ndim == 1 and lat.size == lon.size:
        ok = np.isfinite(lat) & np.isfinite(lon)
        print(f"  [ok]   domain: {int(ok.sum()):,} cells (1-D cell list)")
        return ("points", lon[ok], lat[ok])

    var = _pick(list(ds.data_vars), ("gwsa_trend", "trend", "sigma_total"))
    if var is None and len(ds.data_vars):
        var = list(ds.data_vars)[0]
    arr = np.asarray(ds[var].values)
    while arr.ndim > 2:
        arr = arr[0]
    mask = np.isfinite(arr)
    if mask.shape == (lon.size, lat.size):
        mask = mask.T
    if mask.shape != (lat.size, lon.size):
        print("  [SKIP] domain: mask shape does not match lat/lon")
        return None
    print(f"  [ok]   domain: {int(mask.sum()):,} cells (2-D grid)")
    return ("grid", lon, lat, mask)


# ======================================================================
# Coverage
# ======================================================================

def _cellset(df, res):
    if df is None or not len(df):
        return set()
    i = np.floor((df.lon.values + 180) / res).astype(int)
    j = np.floor((df.lat.values + 90) / res).astype(int)
    return set(zip(i.tolist(), j.tolist()))


def _coverage_stats(grow, jas, res):
    g, j = _cellset(grow, res), _cellset(jas, res)
    return {"grow_sites": 0 if grow is None else len(grow),
            "jas_sites": 0 if jas is None else len(jas),
            "grow_cells": len(g), "jas_cells": len(j), "both": len(g & j),
            "grow_only": len(g - j), "jas_only": len(j - g)}


# ======================================================================
# Furniture
# ======================================================================

def _north_arrow(ax, x=0.965, y=0.86, size=0.05):
    fs = plt.rcParams["font.size"]
    ax.annotate("", xy=(x, y + size), xytext=(x, y - size * 0.5),
                xycoords="axes fraction",
                arrowprops=dict(facecolor="black", edgecolor="black",
                                width=0.9, headwidth=6.0, headlength=6.5),
                zorder=40)
    ax.text(x, y + size * 1.15, "N", transform=ax.transAxes, ha="center",
            va="bottom", fontsize=fs * 1.05, fontweight="bold", zorder=40)


def _scale_bar(ax, length_km=5000, x=0.03, y=-0.085, lw=2.6):
    """
    Drawn in the clear margin just below the map frame, at the far left
    where the ellipse narrows and no longitude labels fall. Placing it in
    axes-fraction coordinates rather than over the ocean means it cannot
    land on a coastline when the projection or extent changes.

    The bar is still scaled from true projected distance, so it remains
    exact only at the equator, as on any world projection.
    """
    fs = plt.rcParams["font.size"] * 0.88
    x0, x1 = ax.get_xlim()
    frac = (length_km * 1000.0) / (x1 - x0)      # bar length in axes fraction

    kw = dict(transform=ax.transAxes, clip_on=False, zorder=40)
    ax.plot([x, x + frac], [y, y], color="black", lw=lw,
            solid_capstyle="butt", **kw)
    ax.plot([x, x + frac / 2], [y, y], color="white", lw=lw * 0.5,
            solid_capstyle="butt", **kw)
    ax.text(x, y + 0.022, "0", ha="center", va="bottom", fontsize=fs,
            transform=ax.transAxes, clip_on=False, zorder=41)
    ax.text(x + frac, y + 0.022, f"{length_km:,} km", ha="center", va="bottom",
            fontsize=fs, transform=ax.transAxes, clip_on=False, zorder=41)


# ======================================================================
# Main
# ======================================================================

def main():
    font = apply_style(base=8, scale=PANEL_SCALE)
    print(f"font: {font}\nloading layers", flush=True)

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    cache_dir = os.path.join(OUTDIR, "_cache")
    os.makedirs(cache_dir, exist_ok=True)

    basins   = _read_vector(BASINS_FP, "basins")
    aquifers = _read_vector(_find_aquifers(), "aquifers")
    aquifers, _, _ = _split_whymap(aquifers)          # WHYMAP -> aquifers only
    water, ice = _study_masks()                        # this study's own cells
    ne_water = ne_ice = None                            # Natural Earth retired
    grow_sites, grow_years = _load_grow(GROW_FP, cache_dir)
    jas_sites, jas_years   = _load_jasechko(JASECHKO_FP, cache_dir)
    domain   = _domain_mask(DOMAIN_NC)

    stats = _coverage_stats(grow_sites, jas_sites, COVERAGE_RES)
    print(f"\ncoverage comparison ({COVERAGE_RES:g} deg cells)")
    print(f"  GROW      {stats['grow_sites']:>9,} sites   "
          f"{stats['grow_cells']:>6,} cells")
    print(f"  Jasechko  {stats['jas_sites']:>9,} sites   "
          f"{stats['jas_cells']:>6,} cells")
    print(f"  shared {stats['both']:,} | GROW-only {stats['grow_only']:,} "
          f"| Jasechko-only {stats['jas_only']:,}\n")

    proj, pc = ccrs.EqualEarth(central_longitude=0), ccrs.PlateCarree()
    fig_w = FIG_W_MM / 25.4
    fig = plt.figure(figsize=(fig_w, fig_w * 0.90))
    # rows: map | legend strip | lower panels
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[2.00, 0.16, 1.00],
                           hspace=0.62, wspace=0.30,
                           left=0.085, right=0.975, top=0.975, bottom=0.10)

    # ---------------- (a) map ----------------------------------------
    ax = fig.add_subplot(gs[0, :], projection=proj)
    ax.set_global()
    ax.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor=C_OCEAN, zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor=C_LAND, zorder=1)

    if domain is not None:
        if domain[0] == "points":
            _, dlon, dlat = domain
            ax.scatter(dlon, dlat, s=0.30, c=C_DOMAIN, marker="s", transform=pc,
                       zorder=2, linewidths=0, rasterized=True)
        else:
            _, dlon, dlat, dmask = domain
            ax.pcolormesh(dlon, dlat, np.where(dmask, 1.0, np.nan), transform=pc,
                          cmap=mcolors.ListedColormap([C_DOMAIN]),
                          shading="auto", zorder=2, rasterized=True)

    if aquifers is not None:
        ax.add_geometries(aquifers.geometry, crs=pc, facecolor=C_AQ_FACE,
                          edgecolor=C_AQ_EDGE, linewidth=0.10, alpha=0.85,
                          zorder=3, rasterized=True)
    if ice is not None:
        ax.add_geometries(ice.geometry, crs=pc, facecolor=C_ICE,
                          edgecolor=C_ICE_EDGE, linewidth=0.15, zorder=4,
                          rasterized=True)
    elif ne_ice:
        for feat in ne_ice:
            ax.add_feature(feat, facecolor=C_ICE, edgecolor=C_ICE_EDGE,
                           linewidth=0.15, zorder=4)
    if water is not None:
        ax.add_geometries(water.geometry, crs=pc, facecolor=C_WATER,
                          edgecolor=C_WATER_EDGE, linewidth=0.15, zorder=5,
                          rasterized=True)
    elif ne_water:
        for feat in ne_water:
            ax.add_feature(feat, facecolor=C_WATER, edgecolor=C_WATER_EDGE,
                           linewidth=0.15, zorder=5)
    if basins is not None:
        ax.add_geometries(basins.geometry, crs=pc, facecolor="none",
                          edgecolor=C_BASIN, linewidth=0.30, zorder=6)

    layers = []
    if grow_sites is not None and len(grow_sites):
        layers.append((grow_sites, C_GROW, GROW_MS, GROW_ALPHA, 7))
    if jas_sites is not None and len(jas_sites):
        layers.append((jas_sites, C_JAS, JAS_MS, JAS_ALPHA, 8))
    if DRAW_ORDER == "jasechko_first":
        layers = layers[::-1]
    for df, col, ms, al, z in layers:
        ax.scatter(df.lon.values, df.lat.values, s=ms, c=col, marker="o",
                   alpha=al, linewidths=0, transform=pc, zorder=z,
                   rasterized=True)

    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), edgecolor=C_COAST,
                   linewidth=0.30, zorder=10)
    gl = ax.gridlines(crs=pc, draw_labels=True, linewidth=0.3, color="#b8b8b8",
                      alpha=0.6, linestyle=":", zorder=9,
                      xlocs=np.arange(-120, 121, 60),
                      ylocs=np.arange(-60, 61, 30))
    gl.top_labels = gl.right_labels = False
    for attr in ("ylabels_right", "xlabels_top"):      # older cartopy
        try:
            setattr(gl, attr, False)
        except Exception:
            pass
    gl.xlabel_style = gl.ylabel_style = {"size": plt.rcParams["font.size"] * 0.92}
    gl.xpadding = gl.ypadding = 3
    ax.spines["geo"].set_linewidth(0.6)
    ax.spines["geo"].set_edgecolor("#4a4a4a")

    _north_arrow(ax)
    _scale_bar(ax)

    # ---------------- legend strip -----------------------------------
    lax = fig.add_subplot(gs[1, :])
    lax.axis("off")
    handles = []
    if domain is not None:
        handles.append(Patch(facecolor=C_DOMAIN, edgecolor="none",
                             label="Reconstruction domain (55,364 cells)"))
    if aquifers is not None:
        handles.append(Patch(facecolor=C_AQ_FACE, edgecolor=C_AQ_EDGE,
                             linewidth=0.4, label="Major aquifer systems"))
    if water is not None or ne_water:
        handles.append(Patch(facecolor=C_WATER, edgecolor=C_WATER_EDGE,
                             linewidth=0.4, label="Surface water bodies"))
    if ice is not None or ne_ice:
        handles.append(Patch(facecolor=C_ICE, edgecolor=C_ICE_EDGE,
                             linewidth=0.4, label="Permanent ice"))
    if basins is not None:
        handles.append(Line2D([0], [0], color=C_BASIN, lw=0.9,
                              label=f"Basins, level 3 (n = {len(basins)})"))
    if grow_sites is not None:
        handles.append(Line2D([0], [0], marker="o", color="none", markersize=3.4,
                              markerfacecolor=C_GROW, markeredgecolor="none",
                              label=f"GROW wells (n = {stats['grow_sites']:,})"))
    if jas_sites is not None:
        handles.append(Line2D([0], [0], marker="o", color="none", markersize=3.4,
                              markerfacecolor=C_JAS, markeredgecolor="none",
                              label=f"Jasechko wells (n = {stats['jas_sites']:,})"))
    if handles:
        lax.legend(handles=handles, loc="center", frameon=False, ncol=3,
                   handlelength=1.4, columnspacing=1.8, handletextpad=0.6,
                   borderaxespad=0.0)

    # ---------------- (b) temporal coverage --------------------------
    axb = fig.add_subplot(gs[2, 0])
    yr0, yr1 = ANALYSIS_YEARS
    curves = []
    for yrs, lab, col in ((grow_years, "GROW", C_GROW),
                          (jas_years, "Jasechko", C_JAS)):
        if yrs is None or not len(yrs):
            continue
        s = yrs.groupby("year")["site"].nunique()
        s = s[(s.index >= 1900) & (s.index <= 2030)]
        if len(s):
            curves.append((s, lab, col))
    if curves:
        # Plotted in thousands. A scientific-notation offset would be
        # drawn just above the axes, exactly where the panel label sits.
        top = max(float(s.max()) for s, _, _ in curves) / 1e3
        axb.axvspan(yr0, yr1, color="#ded8ca", alpha=0.55, zorder=0, lw=0)
        for s, lab, col in curves:
            axb.plot(s.index.values, s.values / 1e3, color=col, lw=1.4,
                     label=lab, zorder=3)
        axb.set_ylim(0, top * 1.30)          # headroom so nothing collides
        axb.annotate(f"{yr0}\u2013{yr1}", xy=((yr0 + yr1) / 2, top * 1.22),
                     ha="center", va="center",
                     fontsize=plt.rcParams["font.size"] * 0.9)
        axb.set_xlabel("Year")
        axb.set_ylabel("Wells reporting (thousands)")
        axb.set_xlim(1950, 2026)
        axb.legend(frameon=False, loc="upper left", borderaxespad=0.2,
                   handlelength=1.4)
    else:
        axb.text(0.5, 0.5, "no usable year information", ha="center",
                 va="center", transform=axb.transAxes)
    for s in ("top", "right"):
        axb.spines[s].set_visible(False)

    # ---------------- (c) spatial coverage ---------------------------
    axc = fig.add_subplot(gs[2, 1])
    labels = ["Shared", "GROW only", "Jasechko only"]
    vals = [stats["both"], stats["grow_only"], stats["jas_only"]]
    cols = ["#9a9a9a", C_GROW, C_JAS]
    bars = axc.barh(labels[::-1], vals[::-1], color=cols[::-1], height=0.60)
    top = max(vals) if max(vals) else 1
    for b, v in zip(bars, vals[::-1]):
        axc.text(b.get_width() + top * 0.02, b.get_y() + b.get_height() / 2,
                 f"{v:,}", va="center",
                 fontsize=plt.rcParams["font.size"] * 0.9)
    axc.set_xlabel(f"{COVERAGE_RES:g}\u00b0 cells containing \u2265 1 well")
    axc.set_xlim(0, top * 1.28)
    for s in ("top", "right"):
        axc.spines[s].set_visible(False)

    # ---------------- labels and caption -----------------------------
    for a, lab in ((ax, "(a)"), (axb, "(b)"), (axc, "(c)")):
        a.text(0.0, 1.03, lab, transform=a.transAxes, ha="left", va="bottom",
               fontsize=plt.rcParams["font.size"] * 1.25, fontweight="bold")

    src = ("Projection: Equal Earth, central meridian 0\u00b0. "
           "Basins: HydroBASINS v1c level 3. Aquifers: WHYMAP (BGR/UNESCO). ")
    if water is not None or ice is not None:
        which = " and ".join(
            [s for s, on in (("surface water", water is not None),
                             ("permanent ice", ice is not None)) if on])
        src += (f"Excluded {which} shown as this study's own 0.5\u00b0 "
                f"masks (SWE-persistence, RGI glaciers, WHYMAP icesheet). ")
    src += "Wells: GROW and Jasechko et al. (2024)."
    fig.text(0.085, 0.018, src,
             fontsize=plt.rcParams["font.size"] * 0.85, color="#3a3a3a",
             ha="left", va="bottom")

    os.makedirs(OUTDIR, exist_ok=True)
    save_fig(fig, STEM, outdir=OUTDIR)
    plt.close(fig)

    pd.DataFrame([stats]).to_csv(
        os.path.join(OUTDIR, "well_network_coverage.csv"), index=False)
    print("wrote coverage table -> "
          + os.path.join(OUTDIR, "well_network_coverage.csv"))


if __name__ == "__main__":
    main()