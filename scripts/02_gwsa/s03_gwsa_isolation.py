#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE 1 -- MONTHLY GWSA ISOLATION  (v2: memory-safe)
Fixes vs v1: float32 throughout, lazy GRACE reads, incremental climatology,
NaN-warning suppression. Same physics, same outputs.
GWSA = TWSA_anom(GRACE, scale-factor) - (soil4+SWE+canopy)_anom(GLDAS), 2004-2009 baseline.
Outputs -> OUTPUTS\01_gwsa_monthly\
"""
import os, sys, glob, re, traceback, warnings, datetime as dt
import numpy as np
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning)

DATA      = r"D:\THESIS\Arpon_Thesis_Final\DATA"
OUT_ROOT  = r"D:\THESIS\Arpon_Thesis_Final\OUTPUTS"
OUT_DIR   = os.path.join(OUT_ROOT, "01_gwsa_monthly")
GRACE_FILE= os.path.join(DATA,"grace_jpl_mascon",
                         "GRCTellus.JPL.200204_202512.GLO.RL06.3M.MSCNv04CRI.nc")
GLDAS_DIR = os.path.join(DATA,"gldas_noah_025")
MASK_NC   = os.path.join(OUT_ROOT,"00_reliability_mask","reliability_mask.nc")
COUNTRIES = os.path.join(DATA,"Global SHapefile","World_Countries_Generalized.shp")

BASE_START, BASE_END = 2004, 2009
ANA_START, ANA_END = (2002,4),(2022,12)
GRID=0.5; CAP_CM=100.0
SOIL_VARS=["SoilMoi0_10cm_inst","SoilMoi10_40cm_inst",
           "SoilMoi40_100cm_inst","SoilMoi100_200cm_inst"]
SWE_VAR,CAN_VAR="SWE_inst","CanopInt_inst"
LON=np.arange(-180+GRID/2,180,GRID); LAT=np.arange(-90+GRID/2,90,GRID)

LINE="="*78; _LOG=[]
def _p(s=""): print(s); _LOG.append(s)
def hdr(t): _p("\n"+LINE); _p(t); _p(LINE)
def ok(t): _p("  [ OK ]   "+t)
def warn(t): _p("  [WARN]   "+t)
def fail(t): _p("  [FAIL]   "+t)
def _imp(m,pip=None):
    try: return __import__(m)
    except Exception: fail(f"missing '{m}'. pip install {pip or m}"); return None

def year_of(p):
    m=re.search(r"A(\d{4})\d{2}",os.path.basename(p)); return int(m.group(1)) if m else None
def month_of(p):
    m=re.search(r"A(\d{4})(\d{2})",os.path.basename(p)); return int(m.group(2)) if m else None

def block_mean_2x2(a):
    ny,nx=a.shape
    return np.nanmean(a.reshape(ny//2,2,nx//2,2),axis=(1,3))

def gldas_storage_month(xr,f):
    """Total storage (cm, float32) on 0.5deg 360x720 for one GLDAS month."""
    d=xr.open_dataset(f)
    comp=np.zeros((600,1440),"float32")
    for v in SOIL_VARS+[SWE_VAR,CAN_VAR]:
        comp+=np.asarray(d[v].squeeze().values,"float32")
    d.close()
    half=block_mean_2x2(comp).astype("float32")     # (300,720) lat -60..90
    full=np.full((360,720),np.nan,"float32")
    full[60:360,:]=half
    return full/10.0                                 # mm -> cm

def main():
    os.makedirs(OUT_DIR,exist_ok=True)
    _p(LINE); _p("PHASE 1 -- MONTHLY GWSA ISOLATION (memory-safe)"); _p(LINE)
    xr=_imp("xarray")
    if xr is None: return

    hdr("STEP 1 -- open GRACE (lazy; scale factor, lon roll)")
    ds=xr.open_dataset(GRACE_FILE)
    _p(f"  GRACE time={ds.sizes['time']}")
    sf=np.asarray(ds["scale_factor"].values,"float32")
    lmask=np.asarray(ds["land_mask"].values).astype(bool)
    glon=ds["lon"].values
    order=None
    if glon.max()>180:
        new_lon=np.where(glon>180,glon-360,glon); order=np.argsort(new_lon)
        sf=sf[:,order]; lmask=lmask[:,order]
        ok(f"GRACE lon rolled to -180..180 ({len(order)} cols).")
    glat=ds["lat"].values
    flip_lat = glat[0]>glat[-1]
    if flip_lat:
        sf=sf[::-1,:]; lmask=lmask[::-1,:]
        ok("GRACE lat flipped to ascending.")
    times=ds["time"].values
    grace_ym=[(int(str(t)[:4]),int(str(t)[5:7])) for t in times]
    lat_lo,lat_hi=float(np.min(glat)),float(np.max(glat))
    ok(f"GRACE lat extent {lat_lo:.2f}..{lat_hi:.2f}, span {grace_ym[0]}..{grace_ym[-1]}")

    def grace_month(idx):
        """Read one month of TWSA (cm, float32), aligned, scale-factor applied."""
        a=np.asarray(ds["lwe_thickness"].isel(time=idx).values,"float32")
        u=np.asarray(ds["uncertainty"].isel(time=idx).values,"float32")
        if order is not None: a=a[:,order]; u=u[:,order]
        if flip_lat: a=a[::-1,:]; u=u[::-1,:]
        return a*sf, u

    hdr("STEP 2 -- reliability mask")
    if not os.path.exists(MASK_NC): fail(f"missing {MASK_NC}"); return
    mds=xr.open_dataset(MASK_NC)
    perm_ice=np.asarray(mds["permanent_ice"].values,bool); mds.close()
    ok(f"permanent_ice cells: {int(perm_ice.sum())}")

    hdr("STEP 3 -- analysis domain")
    latgrid=np.repeat(LAT[:,None],len(LON),axis=1)
    domain=lmask&(latgrid>=lat_lo)&(latgrid<=lat_hi)&(~perm_ice)
    ok(f"domain cells: {int(domain.sum())}")

    hdr("STEP 4 -- GLDAS 2004-2009 climatology (incremental)")
    files=[f for f in sorted(glob.glob(os.path.join(GLDAS_DIR,"**","*.nc4"),recursive=True))
           if year_of(f)]
    by_ym={(year_of(f),month_of(f)):f for f in files}
    _p(f"  GLDAS months: {len(by_ym)}")
    clim={}
    for mo in range(1,13):
        acc=np.zeros((360,720),"float32"); cnt=np.zeros((360,720),"float32")
        for y in range(BASE_START,BASE_END+1):
            f=by_ym.get((y,mo))
            if not f: continue
            s=gldas_storage_month(xr,f)
            m=np.isfinite(s)
            acc[m]+=s[m]; cnt[m]+=1
        clim[mo]=np.where(cnt>0,acc/np.maximum(cnt,1),np.nan).astype("float32")
    ok("climatology built (12 months, incremental, low-memory).")

    hdr("STEP 5 -- monthly GWSA")
    target=[]; y,mo=ANA_START
    while (y,mo)<=ANA_END:
        target.append((y,mo)); mo+=1
        if mo>12: mo=1; y+=1
    T=len(target); _p(f"  target months: {T}")
    lut={ym:i for i,ym in enumerate(grace_ym)}
    GWSA=np.full((T,360,720),np.nan,"float32")
    UNC =np.full((T,360,720),np.nan,"float32")
    gap=np.ones(T,np.int8); n_have=0
    for k,(yy,mm) in enumerate(target):
        gi=lut.get((yy,mm)); f=by_ym.get((yy,mm))
        if gi is None or f is None: continue
        tws,u=grace_month(gi)
        stor=gldas_storage_month(xr,f)
        GWSA[k]=tws-(stor-clim[mm]); UNC[k]=u; gap[k]=0; n_have+=1
    ds.close()
    ok(f"valid months: {n_have}/{T}  (NaN gaps: {T-n_have})")

    hdr("STEP 6 -- domain + Rodell cap")
    GWSA=np.where(domain[None,:,:],GWSA,np.nan).astype("float32")
    UNC =np.where(domain[None,:,:],UNC ,np.nan).astype("float32")
    n_cap=int(np.nansum(np.abs(GWSA)>CAP_CM))
    GWSA=np.clip(GWSA,-CAP_CM,CAP_CM)
    ok(f"capped |GWSA|>{CAP_CM}cm: {n_cap} cell-months")

    hdr("STEP 7 -- SANITY GATES (hotspots MUST be negative)")
    gm=np.array([np.nanmean(GWSA[k][domain]) for k in range(T)],"float32")
    _p(f"  global-mean GWSA: {np.nanmin(gm):.2f} .. {np.nanmax(gm):.2f} cm")
    def trend_at(la,lo):
        j=np.argmin(np.abs(LAT-la)); i=np.argmin(np.abs(LON-lo))
        s=GWSA[:,j,i]; m=np.isfinite(s)
        if m.sum()<60: return None
        return np.polyfit(np.arange(T)[m],s[m],1)[0]*12
    for nm,la,lo,exp in [("NW India/Punjab",30.5,75.5,"NEGATIVE"),
                         ("North China Plain",37.0,116.0,"NEGATIVE"),
                         ("Arabia",24.0,45.0,"NEGATIVE"),
                         ("Amazon (control)",-3.0,-60.0,"~0 or positive")]:
        b=trend_at(la,lo)
        _p(f"     {nm:<26}: {b:+.3f} cm/yr  (expect {exp})" if b is not None
           else f"     {nm:<26}: insufficient data")

    hdr("STEP 8 -- write")
    tc=[dt.datetime(a,b,15) for (a,b) in target]
    out=xr.Dataset(
        {"gwsa":(("time","lat","lon"),GWSA),
         "gwsa_uncertainty":(("time","lat","lon"),UNC),
         "gap_flag":(("time",),gap),
         "domain":(("lat","lon"),domain.astype("int8")),
         "permanent_ice":(("lat","lon"),perm_ice.astype("int8"))},
        coords={"time":tc,"lat":LAT,"lon":LON},
        attrs={"title":"Monthly GWSA (Phase 1)","units":"cm",
               "baseline":f"{BASE_START}-{BASE_END}",
               "gwsa_def":"TWSA_anom(GRACE,scale-factor) - (soil4+SWE+canopy)_anom(GLDAS)",
               "cap_cm":CAP_CM,
               "note":"GRACE gaps = NaN + gap_flag; no interpolation in Phase 1"})
    p=os.path.join(OUT_DIR,"gwsa_monthly.nc"); out.to_netcdf(p); ok(f"written: {p}")

    hdr("STEP 9 -- figures (TNR, 600 dpi)")
    make_figures(GWSA,gap,domain,perm_ice,target,gm)
    with open(os.path.join(OUT_DIR,"gwsa_diagnostics.txt"),"w",encoding="utf-8") as fh:
        fh.write("\n".join(_LOG))
    _p("\n"+LINE); _p("Run complete."); _p(LINE)

def make_figures(GWSA,gap,domain,perm_ice,target,gm):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import rcParams
        rcParams["font.family"]="serif"
        rcParams["font.serif"]=["Times New Roman","DejaVu Serif"]
        rcParams.update({"font.size":15,"axes.titlesize":18,"axes.labelsize":16})
    except Exception as e: warn(f"mpl failed: {e}"); return
    gpd=_imp("geopandas"); cty=None
    if gpd and os.path.exists(COUNTRIES):
        try: cty=gpd.read_file(COUNTRIES)
        except Exception as e: warn(f"countries: {e}")
    T=len(target); x=np.arange(T)
    trend=np.full((360,720),np.nan,"float32")
    vc=np.sum(np.isfinite(GWSA),axis=0)
    for j,i in zip(*np.where(vc>=60)):
        s=GWSA[:,j,i]; m=np.isfinite(s)
        trend[j,i]=np.polyfit(x[m],s[m],1)[0]*12
    ext=[-180,180,-90,90]
    def base(ax):
        if cty is not None: cty.boundary.plot(ax=ax,linewidth=0.3,color="0.35")
        ax.set_xlim(-180,180); ax.set_ylim(-60,85)
        ax.set_xlabel("Longitude ($^\\circ$)"); ax.set_ylabel("Latitude ($^\\circ$)")
    import matplotlib.pyplot as plt
    f,ax=plt.subplots(figsize=(15,7))
    v=np.nanpercentile(np.abs(trend),98)
    im=ax.imshow(trend,origin="lower",extent=ext,cmap="RdBu",vmin=-v,vmax=v)
    base(ax); ax.set_title("Groundwater Storage Anomaly Trend, 2002–2022")
    cb=f.colorbar(im,ax=ax,shrink=0.7,pad=0.02); cb.set_label("GWSA trend (cm yr$^{-1}$)")
    f.savefig(os.path.join(OUT_DIR,"fig1_gwsa_trend_map.png"),dpi=600,bbox_inches="tight"); plt.close(f); ok("fig1")
    f,ax=plt.subplots(figsize=(13,6))
    ax.plot(x,gm,color="#1f4e79",lw=1.6); ax.axhline(0,color="0.6",lw=0.8)
    ax.set_xticks(range(0,T,12)); ax.set_xticklabels([target[k][0] for k in range(0,T,12)],rotation=45)
    ax.set_title("Global-Mean Groundwater Storage Anomaly"); ax.set_xlabel("Year"); ax.set_ylabel("GWSA (cm)")
    f.savefig(os.path.join(OUT_DIR,"fig2_global_mean_series.png"),dpi=600,bbox_inches="tight"); plt.close(f); ok("fig2")
    f,ax=plt.subplots(figsize=(15,7))
    disp=np.full((360,720),np.nan); disp[domain]=0; disp[perm_ice]=1
    ax.imshow(disp,origin="lower",extent=ext,cmap="coolwarm",vmin=0,vmax=1)
    base(ax); ax.set_title("Analysis Domain (blue) and Permanent-Ice Exclusion (red)")
    f.savefig(os.path.join(OUT_DIR,"fig3_domain_mask.png"),dpi=600,bbox_inches="tight"); plt.close(f); ok("fig3")
    f,ax=plt.subplots(figsize=(13,4))
    ax.bar(x,1-gap,color="#2c7fb8",width=1.0); ax.bar(x,gap,bottom=1-gap,color="#e34a33",width=1.0)
    ax.set_xticks(range(0,T,12)); ax.set_xticklabels([target[k][0] for k in range(0,T,12)],rotation=45)
    ax.set_yticks([]); ax.set_title("GRACE Data Availability (blue present, red gap)"); ax.set_xlabel("Year")
    f.savefig(os.path.join(OUT_DIR,"fig4_data_availability.png"),dpi=600,bbox_inches="tight"); plt.close(f); ok("fig4")

if __name__=="__main__":
    try: main()
    except Exception: traceback.print_exc()