# -*- coding: utf-8 -*-
r"""
================================================================================
URBAN FRACTION PROCESSOR  ->  urban_fraction.nc  (0.5deg, 360x720)
================================================================================
Thesis "Aquifer to Atmosphere" | Arpon, KUET

Source: Gao & O'Neill (2020, Nat. Commun.) global 1/8-deg urban land fraction.
  urb_frac_2000.nc   = observed base year (2000)
  ssp2_2010.nc       = SSP2 projection 2010  (best-match "middle of the road")
  ssp2_2020.nc       = SSP2 projection 2020

We use SSP2 for 2010/2020 because it is the "continuation of observed trends"
scenario and all SSPs are near-identical this early; post-2000 values are
PROJECTIONS (stated limitation).

Produces per 0.5-deg cell:
  urban_frac_mean : mean of 2000/2010/2020  (structural urban presence)
  urban_frac_rate : (2020 - 2000)/20  per year (urban growth covariate)
Both provided; VIF/modelling decides which to keep.

Var name inside the files is 'urb_frac_<...>.tif'; grid is 1117x2880,
lat 83.687 -> -55.813 (descending), values are fraction 0-1 (units meta says
"Degree" which is a metadata quirk - they are fractions).

OUTPUT: OUTPUTS\09_ml_diagnostic\urban_fraction.nc  + sanity report
================================================================================
"""
import os, glob, numpy as np, xarray as xr

ROOT = r"D:\THESIS\Arpon_Thesis_Final"
UF   = os.path.join(ROOT, "DATA", "Urban Fraction")
CAL  = os.path.join(ROOT, "OUTPUTS", "03_calibrated", "gwd_trend_calibrated_v2.nc")
OUT  = os.path.join(ROOT, "OUTPUTS", "09_ml_diagnostic")
os.makedirs(OUT, exist_ok=True)

def hr(t=""):
    print("\n"+"="*72)
    if t: print(t); print("="*72)

# ---- target grid ----
cal = xr.open_dataset(CAL)
LAT = cal["lat"].values; LON = cal["lon"].values
nlat, nlon = len(LAT), len(LON)
land = np.isfinite(cal["trend_calibrated"].values)
cal.close()
lat_edges = np.linspace(-90,90,nlat+1); lon_edges = np.linspace(-180,180,nlon+1)

def regrid_mean(src, slat, slon):
    """block-mean a fine 2D field (cell centers slat/slon) onto 0.5deg grid."""
    src = np.asarray(src, float)
    jb = np.clip(np.digitize(slat, lat_edges)-1, 0, nlat-1)
    ib = np.clip(np.digitize(slon, lon_edges)-1, 0, nlon-1)
    out_sum = np.zeros((nlat,nlon)); out_cnt = np.zeros((nlat,nlon))
    JB = jb[:,None]+np.zeros((1,len(slon)),int)
    IB = ib[None,:]+np.zeros((len(slat),1),int)
    m = np.isfinite(src)
    np.add.at(out_sum,(JB[m],IB[m]),src[m])
    np.add.at(out_cnt,(JB[m],IB[m]),1.0)
    with np.errstate(invalid="ignore"):
        out = out_sum/out_cnt
    out[out_cnt==0] = np.nan
    return out

def load_uf(path):
    d = xr.open_dataset(path)
    v = list(d.data_vars)[0]           # e.g. 'urb_frac_2000.tif'
    arr = d[v].values.astype(float)
    latc = [c for c in d.coords if "lat" in c.lower()][0]
    lonc = [c for c in d.coords if "lon" in c.lower()][0]
    la = d[latc].values; lo = d[lonc].values
    d.close()
    # clamp any fill/negative to NaN; fractions should be 0..1
    arr = np.where((arr < 0) | (arr > 1), np.nan, arr)
    return regrid_mean(arr, la, lo)

hr("1. LOAD + REGRID urban fraction (2000, ssp2_2010, ssp2_2020)")
f2000 = os.path.join(UF, "UrbanFraction_1_8_dgr_NETCDF_BaseYear_2000_v1", "urb_frac_2000.nc")
proj  = os.path.join(UF, "UrbanFraction_1_8_dgr_NETCDF_Projections_SSPs1-5_2010-2100_v1")
f2010 = os.path.join(proj, "ssp2_2010.nc")
f2020 = os.path.join(proj, "ssp2_2020.nc")
for f in (f2000, f2010, f2020):
    print("  found:" if os.path.exists(f) else "  MISSING:", os.path.basename(f))

u2000 = load_uf(f2000)
u2010 = load_uf(f2010)
u2020 = load_uf(f2020)
print("  regridded all three to 0.5deg")

hr("2. DERIVE covariates")
stack = np.stack([u2000, u2010, u2020])
urban_frac_mean = np.nanmean(stack, 0)
urban_frac_rate = (u2020 - u2000) / 20.0   # per year
# fill NaN over land with 0 (no urban) - non-urban cells are legit 0, not missing
for name, arr in [("mean",urban_frac_mean),("rate",urban_frac_rate)]:
    onland_nan = np.isnan(arr) & land
    arr[onland_nan] = 0.0
print("  urban_frac_mean + urban_frac_rate computed; land NaN->0")

hr("3. SANITY REPORT")
for name, arr in [("urban_frac_mean",urban_frac_mean),("urban_frac_rate",urban_frac_rate)]:
    onland = arr[land]; fin = onland[np.isfinite(onland)]
    print(f"  {name:16s}: min={np.min(fin):.5f} med={np.median(fin):.5f} "
          f"max={np.max(fin):.5f} mean={np.mean(fin):.5f}")
# show a few known-urban cells sanity: high values should exist
print(f"  cells with urban_frac_mean>0.05 (urbanized): {int((urban_frac_mean[land]>0.05).sum())}")

hr("4. WRITE")
ds = xr.Dataset(
    {"urban_frac_mean": (("lat","lon"), urban_frac_mean.astype("float32")),
     "urban_frac_rate": (("lat","lon"), urban_frac_rate.astype("float32"))},
    coords={"lat":LAT,"lon":LON},
    attrs=dict(title="Urban land fraction (Gao & O'Neill 2020) regridded to 0.5deg",
               source="urb_frac_2000 + ssp2_2010 + ssp2_2020",
               note="post-2000 are SSP2 PROJECTIONS (stated limitation)"))
outp = os.path.join(OUT, "urban_fraction.nc")
ds.to_netcdf(outp)
print("  written:", outp)
hr("URBAN FRACTION PROCESSOR COMPLETE")