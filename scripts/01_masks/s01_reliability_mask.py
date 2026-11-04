#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
0.5_Build_Reliability_Mask_v5.py
v5: rasterize RGI polygons to the 0.5deg grid (seconds) instead of
    make_valid+union_all on 213k polygons (30+ min / may OOM). Identical mask.
    Loads the cached RGI from v4 -> no WFS refetch.
"""
import os, sys, glob, re, time, traceback
import numpy as np

DATA=r"D:\THESIS\Arpon_Thesis_Final\DATA"
OUT_ROOT=r"D:\THESIS\Arpon_Thesis_Final\OUTPUTS"
OUT_DIR=os.path.join(OUT_ROOT,"00_reliability_mask")
GLDAS_DIR=os.path.join(DATA,"gldas_noah_025")
ICE_SHP=os.path.join(DATA,"WHYMAP","WHYMAP_GWR","shp","whymap_icesheets__v1_poly.shp")
DEM_NC=os.path.join(DATA,"Topography","ETOPO_2022_v1_60s_surface.nc")
OUT_NC=os.path.join(OUT_DIR,"reliability_mask.nc")
OUT_FIG=os.path.join(OUT_DIR,"reliability_mask_diagnostic.png")
RGI_CACHE=os.path.join(OUT_DIR,"rgi_glaciers_cache.gpkg")

YEAR_START,YEAR_END=2002,2022
GRID=0.5
LON=np.arange(-180+GRID/2,180,GRID)
LAT=np.arange(-90+GRID/2,90,GRID)
SWE_HI=300.0; SWE_FILL_CAP=10000.0; GLDAS_SWE_VAR="SWE_inst"
# grid edges for rasterization
LON_EDGES=np.arange(-180,180+GRID,GRID)
LAT_EDGES=np.arange(-90,90+GRID,GRID)

LINE="="*78
def hdr(t):print("\n"+LINE+"\n"+t+"\n"+LINE)
def ok(t):print("  [ OK ]   "+t)
def warn(t):print("  [WARN]   "+t)
def fail(t):print("  [FAIL]   "+t)
def _imp(m,pip=None):
    try:return __import__(m)
    except Exception:fail(f"missing '{m}'. pip install {pip or m}");return None
def year_of(p):
    m=re.search(r"A(\d{4})\d{2}",os.path.basename(p));return int(m.group(1)) if m else None
def month_of(p):
    m=re.search(r"A\d{4}(\d{2})",os.path.basename(p));return int(m.group(1)) if m else None

def swe_mask_on_grid():
    xr=_imp("xarray")
    if xr is None:return None
    files=sorted(glob.glob(os.path.join(GLDAS_DIR,"**","*.nc4"),recursive=True)+
                 glob.glob(os.path.join(GLDAS_DIR,"**","*.nc"),recursive=True))
    files=[f for f in files if year_of(f) and YEAR_START<=year_of(f)<=YEAR_END]
    print(f"  GLDAS files: {len(files)}")
    if not files:fail("no GLDAS");return None
    ds0=xr.open_dataset(files[0])
    if GLDAS_SWE_VAR not in ds0.data_vars:fail("SWE absent");ds0.close();return None
    latn=[d for d in ds0[GLDAS_SWE_VAR].dims if "lat" in d.lower()][0]
    lonn=[d for d in ds0[GLDAS_SWE_VAR].dims if "lon" in d.lower()][0]
    ds0.close()
    years={}
    for f in files:years.setdefault(year_of(f),[]).append(f)
    amin=[]
    for y in sorted(years):
        arrs=[]
        for f in sorted(years[y],key=month_of):
            d=xr.open_dataset(f);a=d[GLDAS_SWE_VAR].squeeze()
            a=a.where(a<SWE_FILL_CAP).where(a>=0);arrs.append(a)
        amin.append(xr.concat(arrs,dim="t").min(dim="t"))
    allmin=xr.concat(amin,dim="yr").min(dim="yr").rename({latn:"lat",lonn:"lon"})
    if allmin.lat.values[0]>allmin.lat.values[-1]:allmin=allmin.sortby("lat")
    allmin=allmin.sortby("lon")
    swe=np.asarray(allmin.interp(lat=LAT,lon=LON,method="nearest").values,"float64")
    swe=np.where(np.isfinite(swe),swe,0.0)
    ok(f"SWE max={swe.max():.1f} cells>= {SWE_HI}: {int((swe>=SWE_HI).sum())}")
    return swe>=SWE_HI

def rasterize_polys(gdf,label,land=None,clip_land=False):
    """FAST: rasterize polygons to 0.5deg via rasterio; robust to invalid geoms."""
    if gdf is None or len(gdf)==0:
        warn(f"{label}: empty");return np.zeros((len(LAT),len(LON)),bool)
    try:
        from rasterio.features import rasterize
        from rasterio.transform import from_origin
    except Exception:
        fail("rasterio missing. pip install rasterio");return np.zeros((len(LAT),len(LON)),bool)
    # transform: top-left origin, 0.5deg pixels; raster rows go N->S
    transform=from_origin(-180,90,GRID,GRID)
    shapes=((geom,1) for geom in gdf.geometry.values if geom is not None and not geom.is_empty)
    arr=rasterize(shapes,out_shape=(len(LAT),len(LON)),transform=transform,
                  fill=0,all_touched=True,dtype="uint8")
    # rasterize gives row0=North; our LAT is ascending (row0=South) -> flip
    m=np.flipud(arr).astype(bool)
    if clip_land and land is not None:m=m&land
    ok(f"{label}: {int(m.sum())} cells")
    return m

def land_mask():
    xr=_imp("xarray")
    if not os.path.exists(DEM_NC):warn("no DEM");return None
    d=xr.open_dataset(DEM_NC)
    z=d["z" if "z" in d.data_vars else list(d.data_vars)[0]]
    latn=[c for c in z.dims if "lat" in c.lower() or c.lower()=="y"][0]
    lonn=[c for c in z.dims if "lon" in c.lower() or c.lower()=="x"][0]
    z=z.rename({latn:"lat",lonn:"lon"})
    if z.lat.values[0]>z.lat.values[-1]:z=z.sortby("lat")
    z=z.sortby("lon")
    elev=np.asarray(z.interp(lat=LAT,lon=LON,method="linear").values,"float64")
    return elev>-100

def main():
    os.makedirs(OUT_DIR,exist_ok=True)
    print(LINE);print("RELIABILITY MASK v5 (rasterized RGI, fast)");print(LINE)
    xr=_imp("xarray");gpd=_imp("geopandas")
    if xr is None or gpd is None:return
    hdr("STEP 1 -- SWE-persistence");swe=swe_mask_on_grid()
    if swe is None:fail("abort");return
    hdr("STEP 2 -- land (ETOPO)");land=land_mask()
    if land is None:land=swe.copy()
    hdr("STEP 3 -- RGI glaciers (from cache, rasterized)")
    if not os.path.exists(RGI_CACHE):
        fail(f"cache missing: {RGI_CACHE}. Re-run v4 to fetch first.");return
    rgi=gpd.read_file(RGI_CACHE)
    ok(f"RGI loaded from cache: {len(rgi)} outlines")
    if rgi.crs is None:rgi=rgi.set_crs("EPSG:4326")
    elif rgi.crs.to_epsg()!=4326:rgi=rgi.to_crs("EPSG:4326")
    glac=rasterize_polys(rgi,"RGI glaciers")
    hdr("STEP 4 -- WHYMAP icesheet (land-clipped)")
    ice=np.zeros((len(LAT),len(LON)),bool)
    if os.path.exists(ICE_SHP):
        ig=gpd.read_file(ICE_SHP)
        if ig.crs is None:ig=ig.set_crs("EPSG:4326")
        elif ig.crs.to_epsg()!=4326:ig=ig.to_crs("EPSG:4326")
        ice=rasterize_polys(ig,"WHYMAP icesheet",land=land,clip_land=True)
    hdr("STEP 5 -- combine + sanity")
    perm=(swe|glac|ice)&land
    print(f"  SWE:{int(swe.sum())}  RGI:{int(glac.sum())}  icesheet:{int(ice.sum())}")
    tot=int(land.sum())
    def rc(n,a,b,c,d,e):
        jj=(LAT>=a)&(LAT<=b);ii=(LON>=c)&(LON<=d)
        print(f"     {n:<22}: {int(perm[np.ix_(jj,ii)].sum()):>4}  (expect {e})")
    rc("High Mountain Asia",27,40,70,100,"HIGH now")
    rc("Alaska/Yukon",58,68,-155,-130,"HIGH")
    rc("Andes (S)",-55,-30,-75,-68,"some")
    rc("Indo-Gangetic plain",24,30,75,88,"~0")
    rc("North China Plain",34,40,113,119,"~0")
    rc("Canadian prairies",49,55,-114,-97,"~0")
    print(f"\n  permanent-ice total: {int(perm.sum())} ({100*perm.sum()/max(tot,1):.2f}% land)")
    hdr("STEP 6 -- write")
    ds=xr.Dataset(
        {"permanent_ice":(("lat","lon"),perm.astype("int8")),
         "swe_mask":(("lat","lon"),swe.astype("int8")),
         "rgi_glacier":(("lat","lon"),glac.astype("int8")),
         "icesheet":(("lat","lon"),ice.astype("int8")),
         "land":(("lat","lon"),land.astype("int8")),
         "exclude_tier1":(("lat","lon"),perm.astype("int8"))},
        coords={"lat":LAT,"lon":LON},
        attrs={"title":"Reliability mask v5 (SWE + RGI rasterized + WHYMAP icesheet)",
               "grid_deg":GRID,
               "rgi_source":"RGI via GLIMS OGC WFS, rasterized all_touched",
               "method":"permanent_ice = SWE OR RGI glacier OR WHYMAP icesheet; masking not subtraction"})
    ds.to_netcdf(OUT_NC);ok(f"written: {OUT_NC}")
    try:
        import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as pl
        fig,ax=pl.subplots(figsize=(13,6),dpi=200)
        ax.imshow(perm,origin="lower",extent=[-180,180,-90,90],cmap="Blues",vmin=0,vmax=1)
        ax.set_title("Permanent-ice mask v5")
        fig.savefig(OUT_FIG,bbox_inches="tight");ok(f"figure: {OUT_FIG}")
    except Exception as e:warn(f"figure skipped: {e}")
    print("\n"+LINE);print("Run complete.");print(LINE)
# ---- DIAGNOSE: where are the Indo-Gangetic flagged cells? ----
    hdr("DIAGNOSTIC -- locate IGP-flagged cells + true GRACE-domain fraction")
    jj=(LAT>=24)&(LAT<=30); ii=(LON>=75)&(LON<=88)
    sub=perm[np.ix_(jj,ii)]; la_sub=LAT[jj]; lo_sub=LON[ii]
    print("  IGP-box flagged cells (lat, lon):")
    for a in range(sub.shape[0]):
        for b in range(sub.shape[1]):
            if sub[a,b]:
                print(f"     lat={la_sub[a]:.2f}  lon={lo_sub[b]:.2f}")
    # true GRACE analysis domain: exclude poles beyond GRACE + ice sheets
    grace_dom = land.copy()
    latgrid = np.repeat(LAT[:,None], len(LON), axis=1)
    grace_dom &= (latgrid >= -60) & (latgrid <= 84)   # GRACE latitude coverage
    grace_dom &= (ice==0)                              # drop ice-sheet cells from domain
    n_dom=int(grace_dom.sum())
    n_masked_in_dom=int((perm & grace_dom).sum())
    print(f"\n  land cells (elev>-100)          : {int(land.sum())}")
    print(f"  TRUE GRACE-domain cells         : {n_dom}")
    print(f"  permanent-ice within GRACE domain: {n_masked_in_dom} "
          f"({100*n_masked_in_dom/max(n_dom,1):.2f}% of GRACE domain)")
    print("  ^ THIS percentage is the real cost of the ice mask, not the 31%.")
if __name__=="__main__":
    try:main()
    except Exception:traceback.print_exc()