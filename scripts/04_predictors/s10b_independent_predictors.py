# -*- coding: utf-8 -*-
r"""
================================================================================
PHASE 2.9a  -  ASSEMBLE ON-DISK INDEPENDENT PREDICTORS  (0.5 deg, 360x720)
================================================================================
Thesis "Aquifer to Atmosphere" | Arpon, KUET

PURPOSE
  Build a per-cell predictor stack on the 0.5 degree grid from data that never
  entered the groundwater calibration. In this release the elevation, slope,
  aei and aeigw layers are read by s11_dataset_assembler.py; the remaining
  layers were computed in the same pass.

ANTI-CIRCULARITY (locked)
  INCLUDED (clean - never in calibration):
    NDVI trend+mean, EVI trend+mean      (MODIS MOD13C2)
    t2m trend                            (ERA5)
    E trend  (actual evap)               (GLEAM v4)
    Ep trend (potential evap / demand)   (GLEAM v4 Ep)
    AEI, AEIGW  (irrigation, gw-irrig)   (GMIA v5)
    elevation, slope                     (ETOPO)
  EXCLUDED (circular for the calibrated target):
    GLDAS-derived anything  (GLDAS is INSIDE GWSA = GRACE - GLDAS)
    GRACE-derived anything, Sy, recharge_class  (all entered calibration)

TARGET GRID (verified against the calibrated trend file):
    lat ascending  -89.75 .. 89.75   (360)
    lon            -179.75 .. 179.75 (720)
  Coordinates taken directly from gwd_trend_calibrated_v2.nc (an earlier
  intermediate file on the same grid) to guarantee exact alignment; only its
  lat/lon and land mask are used, none of its values enter the predictors.

TREND METHOD
  Per-cell trend = Theil-Sen slope of the ANNUAL mean series (2002-2022),
  robust to outliers, units per year. Annual means first (removes seasonal
  aliasing), then one slope per cell. A trend needs >= MIN_YEARS valid years.

OUTPUT
  OUTPUTS\09_ml_diagnostic\predictors_independent.nc  (CF metadata)
  + console sanity report (coverage %, min/median/max, NaN fraction per layer)

RUN: no arguments. Regridding the 312 MODIS files
is the slow part (~a few min on CPU); everything else is fast.
================================================================================
"""
import os, sys, glob, warnings, numpy as np, xarray as xr
warnings.filterwarnings("ignore")
from scipy.stats import theilslopes

ROOT   = r"D:\THESIS\Arpon_Thesis_Final"
DATA   = os.path.join(ROOT, "DATA")
OUTdir = os.path.join(ROOT, "OUTPUTS", "09_ml_diagnostic")
CAL    = os.path.join(ROOT, "OUTPUTS", "03_calibrated", "gwd_trend_calibrated_v2.nc")
os.makedirs(OUTdir, exist_ok=True)

Y0, Y1     = 2002, 2022
MIN_YEARS  = 10          # min valid annual points to compute a trend

def hr(t=""):
    print("\n" + "="*76)
    if t: print(t); print("="*76)

# ---- target grid from the calibrated file (exact coords) --------------------
cal = xr.open_dataset(CAL)
LAT = cal["lat"].values.astype(float)      # 360, ascending
LON = cal["lon"].values.astype(float)      # 720
nlat, nlon = len(LAT), len(LON)
print(f"target grid: {nlat} x {nlon}  lat[{LAT[0]:.2f}..{LAT[-1]:.2f}] "
      f"lon[{LON[0]:.2f}..{LON[-1]:.2f}]")
# cell edges for conservative-ish binning of finer grids
lat_edges = np.linspace(-90, 90, nlat+1)
lon_edges = np.linspace(-180, 180, nlon+1)

def regrid_mean(src, slat, slon):
    """Area-agnostic block-mean of a fine 2D field onto the 0.5deg grid using
    bin indices. src: 2D (slat,slon). Returns (nlat,nlon) with NaN where empty.
    Assumes slat/slon are cell-CENTER coords."""
    src = np.asarray(src, float)
    # bin index of each source cell center into target edges
    jb = np.clip(np.digitize(slat, lat_edges) - 1, 0, nlat-1)
    ib = np.clip(np.digitize(slon, lon_edges) - 1, 0, nlon-1)
    out_sum = np.zeros((nlat, nlon)); out_cnt = np.zeros((nlat, nlon))
    JB = jb[:, None] + np.zeros((1, len(slon)), int)
    IB = ib[None, :] + np.zeros((len(slat), 1), int)
    m = np.isfinite(src)
    np.add.at(out_sum, (JB[m], IB[m]), src[m])
    np.add.at(out_cnt, (JB[m], IB[m]), 1.0)
    with np.errstate(invalid="ignore"):
        out = out_sum / out_cnt
    out[out_cnt == 0] = np.nan
    return out

def cell_trend(stack_years):
    """stack_years: (nyear, nlat, nlon) annual means -> Theil-Sen slope/yr."""
    ny = stack_years.shape[0]
    yrs = np.arange(ny, dtype=float)
    out = np.full((nlat, nlon), np.nan, np.float32)
    valid = np.isfinite(stack_years).sum(0)
    ii, jj = np.where(valid >= MIN_YEARS)
    for a, b in zip(ii, jj):
        y = stack_years[:, a, b]
        m = np.isfinite(y)
        if m.sum() >= MIN_YEARS:
            out[a, b] = theilslopes(y[m], yrs[m])[0]
    return out

layers = {}   # name -> 2D array on target grid

# ============================================================================
# 1. MODIS NDVI + EVI  (pyhdf)  -> annual means -> trend + mean
# ============================================================================
hr("1. MODIS NDVI / EVI")
from pyhdf.SD import SD, SDC
mfiles = sorted(glob.glob(os.path.join(DATA, "modis_vegetation", "*.hdf")))
print(f"  MODIS files: {len(mfiles)}")

# MOD13C2 CMG 0.05deg: 3600 lat x 7200 lon, lat 90->-90, lon -180->180
mlat = 90 - (np.arange(3600) + 0.5) * 0.05
mlon = -180 + (np.arange(7200) + 0.5) * 0.05

def modis_year(files_of_year, sds):
    acc = None; n = 0
    for f in files_of_year:
        sd = SD(f, SDC.READ)
        arr = sd.select(sds)[:].astype(np.float32)
        rel = sd.select("CMG 0.05 Deg Monthly pixel reliability")[:].astype(np.float32)
        sd.end()
        arr[arr == -3000] = np.nan          # fill value
        arr *= 0.0001                        # scale factor
        arr[(rel < 0) | (rel > 1)] = np.nan  # keep reliability 0(good)/1(marginal)
        acc = arr if acc is None else np.nansum(np.dstack([acc, arr]), 2)
        n += 1
    return acc / n if n else None

import re
def year_of(f):
    m = re.search(r"A(\d{4})(\d{3})", os.path.basename(f))
    return int(m.group(1)) if m else None

for sds, tag in [("CMG 0.05 Deg Monthly NDVI", "ndvi"),
                 ("CMG 0.05 Deg Monthly EVI",  "evi")]:
    yr_stack = []
    for yr in range(Y0, Y1+1):
        fy = [f for f in mfiles if year_of(f) == yr]
        if not fy:
            yr_stack.append(np.full((nlat, nlon), np.nan, np.float32)); continue
        fine = modis_year(fy, sds)                 # 3600x7200
        yr_stack.append(regrid_mean(fine, mlat, mlon).astype(np.float32))
        print(f"    {tag} {yr}: {len(fy)} files")
    st = np.stack(yr_stack)
    layers[f"{tag}_mean"]  = np.nanmean(st, 0).astype(np.float32)
    layers[f"{tag}_trend"] = cell_trend(st)
    print(f"    {tag}: mean+trend done")

# ============================================================================
# 2. ERA5 t2m  -> annual means -> trend  (K/yr)
# ============================================================================
hr("2. ERA5 t2m trend")
efiles = {int(re.search(r"_(\d{4})_", os.path.basename(f)).group(1)): f
          for f in glob.glob(os.path.join(DATA,"era5_thermodynamics","extracted","*_avgua.nc"))}
e_stack = []
for yr in range(Y0, Y1+1):
    if yr not in efiles:
        e_stack.append(np.full((nlat, nlon), np.nan, np.float32)); continue
    d = xr.open_dataset(efiles[yr])
    t = d["t2m"].mean("valid_time").values           # 721x1440, lat 90->-90
    la = d["latitude"].values; lo = d["longitude"].values
    lo = np.where(lo > 180, lo - 360, lo)             # 0..360 -> -180..180 if needed
    e_stack.append(regrid_mean(t, la, lo).astype(np.float32))
    d.close()
layers["t2m_trend"] = cell_trend(np.stack(e_stack))
print("  t2m trend done")

# ============================================================================
# 3. GLEAM E and Ep  -> annual means -> trend  (mm/yr per year)
# ============================================================================
hr("3. GLEAM E / Ep trends")
def gleam_trend(folder, var):
    fs = {int(re.search(r"_(\d{4})_", os.path.basename(f)).group(1)): f
          for f in glob.glob(os.path.join(DATA, folder, "*.nc"))}
    stack = []
    for yr in range(Y0, Y1+1):
        if yr not in fs:
            stack.append(np.full((nlat, nlon), np.nan, np.float32)); continue
        d = xr.open_dataset(fs[yr])
        a = d[var].sum("time").values                # annual total (mm/yr)
        la = d["lat"].values; lo = d["lon"].values
        if la[0] > la[-1]:                            # ensure center coords fine
            pass
        stack.append(regrid_mean(a, la, lo).astype(np.float32))
        d.close()
        print(f"    {var} {yr}")
    return cell_trend(np.stack(stack))
layers["E_trend"]  = gleam_trend("GLEAM_v4",    "E")
layers["Ep_trend"] = gleam_trend("GLEAM_v4_Ep", "Ep")
print("  GLEAM done")

# ============================================================================
# 4. GMIA irrigation  (static .asc, 5 arcmin) -> AEI, AEIGW
# ============================================================================
hr("4. GMIA AEI / AEIGW")
def read_asc(path):
    hdr = {}
    with open(path) as f:
        for _ in range(6):
            k, v = f.readline().split()
            hdr[k.lower()] = float(v)
    arr = np.loadtxt(path, skiprows=6)
    nod = hdr.get("nodata_value", -9999)
    arr[arr == nod] = np.nan
    nc, nr = int(hdr["ncols"]), int(hdr["nrows"])
    xll, yll, cs = hdr["xllcorner"], hdr["yllcorner"], hdr["cellsize"]
    glon = xll + (np.arange(nc) + 0.5) * cs
    glat = yll + (np.arange(nr) + 0.5) * cs          # ascending
    arr = arr[::-1, :]                               # asc raster stored top->bottom
    return arr, glat, glon
for tag, fn in [("aei","gmia_v5_aei_pct.asc"), ("aeigw","gmia_v5_aeigw_pct_aei.asc")]:
    a, gla, glo = read_asc(os.path.join(DATA, "GMIA", fn))
    layers[tag] = regrid_mean(a, gla, glo).astype(np.float32)
    print(f"    {tag} done")

# ============================================================================
# 5. ETOPO elevation + slope
# ============================================================================
hr("5. ETOPO elevation / slope")
et = xr.open_dataset(os.path.join(DATA,"Topography","ETOPO_2022_v1_60s_surface.nc"))
zvar = [v for v in et.data_vars if et[v].ndim == 2][0]
Z = et[zvar].values.astype(np.float32)
elat = et[[c for c in et.coords if "lat" in c.lower()][0]].values
elon = et[[c for c in et.coords if "lon" in c.lower()][0]].values
if elat[0] > elat[-1]:
    Z = Z[::-1, :]; elat = elat[::-1]
layers["elevation"] = regrid_mean(Z, elat, elon).astype(np.float32)
# slope from the regridded elevation (deg): gradient over ~0.5deg cells
dy = np.gradient(layers["elevation"], axis=0)
dx = np.gradient(layers["elevation"], axis=1)
cell_m = 0.5 * 111000.0
layers["slope"] = (np.degrees(np.arctan(np.sqrt(dx**2 + dy**2) / cell_m))).astype(np.float32)
et.close()
print("  ETOPO done")

# ============================================================================
# 6. SANITY REPORT + WRITE
# ============================================================================
hr("6. SANITY REPORT (per layer)")
land = np.isfinite(cal["trend_calibrated"].values)
nland = land.sum()
print(f"  land cells (calibrated finite): {nland}")
print(f"  {'layer':16s} {'cover%':>7s} {'min':>10s} {'median':>10s} {'max':>10s} {'NaN%':>7s}")
for k, v in layers.items():
    onland = v[land]
    cov = np.isfinite(onland).mean()*100
    fin = onland[np.isfinite(onland)]
    if fin.size:
        print(f"  {k:16s} {cov:7.1f} {np.min(fin):10.4f} "
              f"{np.median(fin):10.4f} {np.max(fin):10.4f} {(1-cov/100)*100:7.1f}")
    else:
        print(f"  {k:16s} {cov:7.1f}  ALL-NAN")

ds = xr.Dataset(
    {k: (("lat","lon"), v) for k, v in layers.items()},
    coords={"lat": LAT, "lon": LON},
    attrs=dict(title="Phase 2.9a independent predictors (calibration-clean)",
               grid="0.5deg 360x720 lat-ascending",
               window=f"{Y0}-{Y1}", trend_method="Theil-Sen on annual means",
               note="GLDAS/GRACE/Sy/recharge EXCLUDED as circular"))
for k in ds.data_vars:
    ds[k].attrs["long_name"] = k
outp = os.path.join(OUTdir, "predictors_independent.nc")
ds.to_netcdf(outp)
cal.close()
print(f"\n  written: {outp}")
hr("PHASE 2.9a COMPLETE")