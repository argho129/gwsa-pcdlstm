# -*- coding: utf-8 -*-
r"""
================================================================================
DATASET ASSEMBLER (Option A: monthly forcing)  ->  10_model\data\   [FIXED v4 - Low RAM]
================================================================================
Thesis "Aquifer to Atmosphere" | Arpon, KUET

Applied Fixes:
  FIX 1  ERA5-Land tp/ro are mean-daily-rate metres -> mm/month = x1000 x days_in_month
  FIX 2  GWSA re-clipped to +/-100 cm after gap-fill
  FIX 3  GLHYMPS loaded from glhymps_0p5.nc (logk_ferr + porosity)
  FIX 4  aeigw NaN -> 0 where aei==0
  FIX 5  NDVI climatological + annual fallback fill + ndvi_observed mask channel
  FIX 6 / F8  exclude_tier1 removed to keep well observations strictly out of ML features
  FIX 7  units attribute written on every channel
  F9     Actual Evaporation clamped (E >= 0.0)
  F10    NDVI two-pass fill (monthly clim + annual mean) with -0.05 floor
  MEM-FIX Immediate per-variable cell extraction + memory purging (peak RAM ~250MB)
================================================================================
"""
import os, sys, glob, re, zipfile, tempfile, warnings, gc
import numpy as np, pandas as pd, xarray as xr
warnings.filterwarnings("ignore")

# ----------------------------- CONFIG -----------------------------
SCOPE = "full"          # "prototype" or "full"

ROOT = r"D:\THESIS\Arpon_Thesis_Final"
DATA = os.path.join(ROOT, "DATA")
OUTP = os.path.join(ROOT, "OUTPUTS")
MODELDIR = os.path.join(OUTP, "10_model")
DATADIR  = os.path.join(MODELDIR, "data")
LOGDIR   = os.path.join(MODELDIR, "logs")
for d in (MODELDIR, DATADIR, LOGDIR): os.makedirs(d, exist_ok=True)

GAPFILL = os.path.join(OUTP, "02_gapfill", "gwsa_monthly_filled.nc")
BASINS  = os.path.join(OUTP, "09_ml_diagnostic", "cell_basins.parquet")
PREDIND = os.path.join(OUTP, "09_ml_diagnostic", "predictors_independent.nc")
URBAN   = os.path.join(OUTP, "09_ml_diagnostic", "urban_fraction.nc")
GLHYMPS = os.path.join(DATA, "SOIL_DATA", "glhymps_0p5.nc")

class _Tee:
    def __init__(s,p): s.f=open(p,"w",encoding="utf-8"); s.o=sys.stdout
    def write(s,x): s.o.write(x); s.f.write(x)
    def flush(s): s.o.flush(); s.f.flush()
sys.stdout=_Tee(os.path.join(LOGDIR, f"assemble_{SCOPE}.log"))

def hr(t=""):
    print("\n"+"="*74)
    if t: print(t); print("="*74)

# ----------------------- target grid + timeline -----------------------
hr("0. TARGET GRID + TIMELINE (from gap-filled GWSA)")
g = xr.open_dataset(GAPFILL)
LAT = g["lat"].values; LON = g["lon"].values
TIME = pd.to_datetime(g["time"].values)
nlat,nlon,nt = len(LAT),len(LON),len(TIME)
gwsa   = g["gwsa"].values.astype("float32")
gsigma = g["gwsa_sigma"].values.astype("float32")
isfill = g["is_filled"].values.astype("int8")
domain = g["domain"].values
g.close()
print(f"  grid {nlat}x{nlon}, T={nt}, {TIME[0].date()}..{TIME[-1].date()}")
lat_edges=np.linspace(-90,90,nlat+1); lon_edges=np.linspace(-180,180,nlon+1)

# FIX 2: re-clip GWSA to +/-100 cm after gap-fill
_n_before = int(np.sum(np.abs(gwsa) > 100.0))
gwsa = np.clip(gwsa, -100.0, 100.0)
print(f"  FIX2 GWSA re-clip: {_n_before} values were |>100| -> clipped to +/-100")

# FIX 1: per-month days, for ERA5-Land mean-daily-rate -> monthly total
DIM = TIME.days_in_month.values.astype("float32")     # (T,)

def regrid(src, slat, slon):
    src=np.asarray(src,float)
    jb=np.clip(np.digitize(slat,lat_edges)-1,0,nlat-1)
    ib=np.clip(np.digitize(slon,lon_edges)-1,0,nlon-1)
    s=np.zeros((nlat,nlon)); c=np.zeros((nlat,nlon))
    JB=jb[:,None]+np.zeros((1,len(slon)),int); IB=ib[None,:]+np.zeros((len(slat),1),int)
    m=np.isfinite(src); np.add.at(s,(JB[m],IB[m]),src[m]); np.add.at(c,(JB[m],IB[m]),1.0)
    with np.errstate(invalid="ignore"): o=s/c
    o[c==0]=np.nan; return o

mkey = np.array([f"{d.year}-{d.month:02d}" for d in TIME])

# ----------------------- choose cells (scope) -----------------------
hr(f"1. SELECT CELLS  (scope={SCOPE})")
cb = pd.read_parquet(BASINS)
if SCOPE=="prototype":
    sel = cb[cb["focus_name"]!=""].copy()
else:
    sel = cb.copy()
sel = sel.dropna(subset=["j","i"])
J = sel["j"].astype(int).values; I = sel["i"].astype(int).values
ncell = len(sel)
print(f"  cells selected: {ncell}")
print(f"  focus breakdown:\n{sel['focus_name'].value_counts().to_string()}")

# ------------------ MONTHLY forcing builders ------------------
def empty_stack(): return np.full((nt,nlat,nlon), np.nan, np.float32)

def unzip_read(path):
    with open(path,'rb') as f: magic=f.read(2)
    if magic==b'PK':
        tmp=tempfile.mkdtemp()
        with zipfile.ZipFile(path) as z: z.extractall(tmp)
        inner=glob.glob(os.path.join(tmp,"**","*.nc"),recursive=True)[0]
        return xr.open_dataset(inner)
    return xr.open_dataset(path)

def build_era5land(var_folder, varname, to_mm, per_day=False):
    """ERA5-Land yearly zip files, monthly inside.
       to_mm : metres -> mm (x1000)
       per_day: FIX1 mean-daily-rate -> monthly total (x days_in_month)"""
    st=empty_stack()
    files=sorted(glob.glob(os.path.join(DATA,var_folder,"*.nc")))
    for f in files:
        yr_m=re.search(r"_(\d{4})", os.path.basename(f))
        if not yr_m: continue
        d=unzip_read(f)
        vv=[v for v in d.data_vars if v.lower()==varname.lower()]
        if not vv: d.close(); continue
        v=vv[0]
        tc=[c for c in d.coords if 'time' in c.lower() or 'valid' in c.lower()][0]
        la=d[[c for c in d.coords if 'lat' in c.lower()][0]].values
        lo=d[[c for c in d.coords if 'lon' in c.lower()][0]].values
        lo=np.where(lo>180,lo-360,lo)
        tt=pd.to_datetime(d[tc].values)
        for k,ts in enumerate(tt):
            key=f"{ts.year}-{ts.month:02d}"
            idx=np.where(mkey==key)[0]
            if len(idx)==0: continue
            arr=d[v].isel({tc:k}).values
            r=regrid(arr,la,lo)
            if to_mm: r=r*1000.0
            if per_day: r=r*DIM[idx[0]]           # FIX1
            st[idx[0]]=r.astype(np.float32)
        d.close()
    return st

def build_gleam_E():
    st=empty_stack()
    files=sorted(glob.glob(os.path.join(DATA,"GLEAM_v4","E_*.nc")))
    for f in files:
        d=xr.open_dataset(f)
        la=d["lat"].values; lo=d["lon"].values
        tt=pd.to_datetime(d["time"].values)
        for k,ts in enumerate(tt):
            key=f"{ts.year}-{ts.month:02d}"; idx=np.where(mkey==key)[0]
            if len(idx)==0: continue
            st[idx[0]]=regrid(d["E"].isel(time=k).values,la,lo).astype(np.float32)
        d.close()
    return st

def build_era5_t2m():
    st=empty_stack()
    files=sorted(glob.glob(os.path.join(DATA,"era5_thermodynamics","extracted","*avgua.nc")))
    for f in files:
        d=xr.open_dataset(f)
        la=d["latitude"].values; lo=d["longitude"].values; lo=np.where(lo>180,lo-360,lo)
        tt=pd.to_datetime(d["valid_time"].values)
        for k,ts in enumerate(tt):
            key=f"{ts.year}-{ts.month:02d}"; idx=np.where(mkey==key)[0]
            if len(idx)==0: continue
            st[idx[0]]=regrid(d["t2m"].isel(valid_time=k).values,la,lo).astype(np.float32)
        d.close()
    return st

def build_modis_ndvi():
    from pyhdf.SD import SD, SDC
    st=empty_stack()
    mlat=90-(np.arange(3600)+0.5)*0.05; mlon=-180+(np.arange(7200)+0.5)*0.05
    files=sorted(glob.glob(os.path.join(DATA,"modis_vegetation","*.hdf")))
    for f in files:
        m=re.search(r"A(\d{4})(\d{3})",os.path.basename(f))
        if not m: continue
        yr=int(m.group(1)); doy=int(m.group(2))
        ts=pd.Timestamp(yr,1,1)+pd.Timedelta(days=doy-1)
        key=f"{ts.year}-{ts.month:02d}"; idx=np.where(mkey==key)[0]
        if len(idx)==0: continue
        sd=SD(f,SDC.READ)
        arr=sd.select("CMG 0.05 Deg Monthly NDVI")[:].astype(np.float32)
        rel=sd.select("CMG 0.05 Deg Monthly pixel reliability")[:].astype(np.float32)
        sd.end()
        arr[arr==-3000]=np.nan; arr*=0.0001; arr[(rel<0)|(rel>1)]=np.nan
        st[idx[0]]=regrid(arr,mlat,mlon).astype(np.float32)
    return st

# ------------------ MONTHLY FORCING: BUILD + EXTRACT + FREE ------------------
hr("2. BUILD MONTHLY STACKS (extract each, free immediately)")

def build_extract_free(builder, label):
    """Build one global stack, pull selected cells, free the global array."""
    print(f"  {label}...")
    stack = builder()                           # (T, nlat, nlon) ~250 MB
    cells = stack[:, J, I].astype(np.float32)   # (T, ncell) 2D
    del stack
    gc.collect()
    return cells

P_c  = build_extract_free(lambda: build_era5land("climate_era5","tp",  to_mm=True, per_day=True),  "P (ERA5-Land tp)")
R_c  = build_extract_free(lambda: build_era5land("era5_land_runoff","ro",to_mm=True, per_day=True), "R (ERA5-Land ro)")
SM_c = build_extract_free(lambda: build_era5land("climate_era5","swvl1",to_mm=False),               "SM (ERA5-Land swvl1)")

# E (GLEAM) + F9 Clamp
E_c  = build_extract_free(build_gleam_E, "E (GLEAM)")
E_c  = np.clip(E_c, 0.0, None)                  # F9: actual evaporation is non-negative

T2_c = build_extract_free(build_era5_t2m, "t2m (ERA5)")

# F10: NDVI Pass 1 (Monthly Clim) + Pass 2 (Annual Fallback) + Floor
print("  NDVI (MODIS) + climatological & annual fill...")
ND = build_modis_ndvi()
ndvi_obs_g = np.isfinite(ND).astype(np.int8)
month_of = TIME.month.values

# Pass 1: per-cell monthly climatology
for mo in range(1, 13):
    tm = np.where(month_of == mo)[0]
    if len(tm) == 0: continue
    clim = np.nanmean(ND[tm], axis=0)
    for ti in tm:
        sl = ND[ti]
        bad = ~np.isfinite(sl)
        sl[bad] = clim[bad]
        ND[ti] = sl

# Pass 2 (F10): residual NaN -> per-cell annual mean, then global mean fallback
annual = np.nanmean(ND, axis=0)                 # (lat, lon)
gmean  = np.nanmean(annual)
annual = np.where(np.isfinite(annual), annual, gmean)
for ti in range(ND.shape[0]):
    sl = ND[ti]
    bad = ~np.isfinite(sl)
    sl[bad] = annual[bad]
    ND[ti] = sl

ND = np.clip(ND, -0.05, None)                   # F10: floor water/snow artifacts
ND_c    = ND[:, J, I].astype(np.float32)
NDobs_c = ndvi_obs_g[:, J, I].astype(np.int8)
del ND, ndvi_obs_g
gc.collect()
print("  NDVI filled (clim + annual-mean fallback); residual NaN: 0 expected")

print("  monthly stacks done (global arrays freed).")

# ------------------ STATIC COVARIATES ------------------
hr("3. STATIC COVARIATES")
pi = xr.open_dataset(PREDIND)
static = {}
for v in ["elevation","slope","aei","aeigw"]:
    if v in pi.data_vars: static[v]=pi[v].values.astype(np.float32)
pi.close()

# FIX 4: aeigw NaN -> 0 where aei==0
if "aei" in static and "aeigw" in static:
    m0 = (static["aei"]==0) & (~np.isfinite(static["aeigw"]))
    static["aeigw"][m0] = 0.0
    print(f"  FIX4 aeigw: filled {int(m0.sum())} NaN cells where aei==0")

uf = xr.open_dataset(URBAN)
static["urban_frac_mean"]=uf["urban_frac_mean"].values.astype(np.float32)
uf.close()

# FIX 3: GLHYMPS logk_ferr + porosity from rasterized 0.5deg file
gh = xr.open_dataset(GLHYMPS)
static["logk_ferr"]=gh["logk_ferr"].values.astype(np.float32)
static["porosity"] =gh["porosity"].values.astype(np.float32)
gh.close()
print("  FIX3 GLHYMPS loaded: logk_ferr, porosity")

LATg=np.repeat(LAT[:,None],nlon,1).astype(np.float32)
LONg=np.repeat(LON[None,:],nlat,0).astype(np.float32)
static["lat"]=LATg; static["lon"]=LONg
print("  static covariates:", list(static.keys()))

# ------------------ CONSTRUCT DATASET ------------------
hr("4. CONSTRUCT DATASET")
data_vars = {
    "P":            (("time","cell"), P_c),
    "E":            (("time","cell"), E_c),
    "R":            (("time","cell"), R_c),
    "t2m":          (("time","cell"), T2_c),
    "ndvi":         (("time","cell"), ND_c),
    "ndvi_observed":(("time","cell"), NDobs_c),                             # FIX 5
    "sm":           (("time","cell"), SM_c),
    "gwsa":         (("time","cell"), gwsa[:,J,I].astype(np.float32)),
    "gwsa_sigma":   (("time","cell"), gsigma[:,J,I].astype(np.float32)),
    "is_filled":     (("time","cell"), isfill[:,J,I].astype(np.int8)),
}
for k,arr in static.items():
    data_vars[k]=(("cell",), arr[J,I].astype(np.float32))

# F8: exclude_tier1 line deliberately omitted to prevent validation leakage

coords = {
    "time": TIME, "cell": np.arange(ncell),
    "j": ("cell", J), "i": ("cell", I),
    "clat": ("cell", LAT[J].astype(np.float32)),
    "clon": ("cell", LON[I].astype(np.float32)),
    "HYBAS_ID": ("cell", sel["HYBAS_ID"].values),
    "MAIN_BAS": ("cell", sel["MAIN_BAS"].values),
    "focus_name": ("cell", sel["focus_name"].astype(str).values),
}
ds = xr.Dataset(data_vars, coords=coords,
    attrs=dict(title=f"Aquifer-to-Atmosphere monthly dataset ({SCOPE}) v4",
               scope=SCOPE, note="Option A monthly forcing; FIX1-7 + F8-10 Low-RAM applied"))

# FIX 7: units attributes on every channel
_units = {"P":"mm/month","E":"mm/month","R":"mm/month","t2m":"K","ndvi":"-",
          "ndvi_observed":"1=obs,0=filled","sm":"m3/m3","gwsa":"cm",
          "gwsa_sigma":"cm","is_filled":"1=filled,0=obs","elevation":"m",
          "slope":"-","aei":"pct","aeigw":"pct_of_aei","urban_frac_mean":"fraction",
          "logk_ferr":"log10(m2)","porosity":"fraction","lat":"deg","lon":"deg"}
for v,u in _units.items():
    if v in ds: ds[v].attrs["units"]=u

# ------------------ SANITY REPORT ------------------
hr("5. SANITY REPORT")
print(f"  dataset dims: T={nt}, cells={ncell}")
for v in ["P","E","R","t2m","ndvi","sm","gwsa"]:
    a=ds[v].values; fin=a[np.isfinite(a)]
    print(f"  {v:5s}: cover={np.isfinite(a).mean()*100:5.1f}%  "
          f"min={np.nanmin(fin):.3f} med={np.nanmedian(fin):.3f} max={np.nanmax(fin):.3f}")
for v in ["logk_ferr","porosity","aeigw","ndvi_observed"]:
    a=ds[v].values.astype(float); cov=np.isfinite(a).mean()*100
    print(f"  {v:14s}: cover={cov:5.1f}%  finite-min={np.nanmin(a):.3f} "
          f"med={np.nanmedian(a):.3f} max={np.nanmax(a):.3f}")
print(f"  is_filled fraction:     {ds['is_filled'].values.mean():.3f}")
print(f"  ndvi_observed fraction: {ds['ndvi_observed'].values.mean():.3f}")

# ------------------ WRITE OUTPUT ------------------
hr("6. WRITE")
outp=os.path.join(DATADIR, f"dataset_{SCOPE}.nc")
enc={v:{"zlib":True,"complevel":4} for v in ds.data_vars}
ds.to_netcdf(outp, encoding=enc)
print("  written:", outp)
print("  size MB:", round(os.path.getsize(outp)/1e6,1))
hr(f"ASSEMBLER COMPLETE (scope={SCOPE})")