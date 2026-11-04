#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
PHASE 8 -- FINAL PRODUCT PACKAGING (citable deliverable)
  Merges the frozen product (gwsa_final.nc) + uncertainty (gwsa_uncertainty.nc)
  into CF-compliant NetCDFs (1D compact + gridded GIS-ready), plus a README data
  descriptor and a machine-readable provenance JSON.

  Author: Arpon Sarkar, Dept. of Urban & Regional Planning (URP), KUET (ID 2017055)

OUTPUT -> OUTPUTS\08_final_product\
  GWSA_KUET_v1_1D.nc         compact (cell, time) CF-1.10
  GWSA_KUET_v1_gridded.nc    (time, lat, lon) + (lat, lon), 0.5 deg, GIS-ready
  README_GWSA.md             data descriptor
  provenance_GWSA.json       machine-readable provenance
================================================================================
"""
import os, sys, json, datetime as dt
import numpy as np

OUT=r"D:\THESIS\Arpon_Thesis_Final\OUTPUTS"
MODEL=os.path.join(OUT,"05_model_tensor")
FINAL_NC=os.path.join(MODEL,"gwsa_final.nc")
UNC_NC=os.path.join(OUT,"07_uncertainty","gwsa_uncertainty.nc")
DEST=os.path.join(OUT,"08_final_product")

AUTHOR="Arpon Sarkar"
AFFIL="Department of Urban & Regional Planning (URP), Khulna University of Engineering & Technology (KUET)"
AUTHOR_ID="2017055"
VERSION="v1"
BASE_Y0,BASE_Y1=2004,2009
NLAT,NLON=360,720
DOI_PLACEHOLDER="<DOI to be minted on deposit>"

LINE="="*78
def _p(s=""): print(s)
def hdr(t): _p(""); _p(LINE); _p(t); _p(LINE)


# CF metadata per variable: (long_name, units, comment)
CF={
 "gwsa":       ("Groundwater storage anomaly","cm","Calibrated monthly GWSA in equivalent water height; baseline 2004-2009 mean removed."),
 "gwsa_trend": ("GWSA linear trend","cm yr-1","Theil-Sen trend over 2002-2022."),
 "trend_ci_lo":("GWSA trend 2.5th percentile","cm yr-1","Lower bound, block-bootstrap 95% CI."),
 "trend_ci_hi":("GWSA trend 97.5th percentile","cm yr-1","Upper bound, block-bootstrap 95% CI."),
 "trend_significant":("Trend significance (bootstrap CI excludes zero)","1","1 = significant, 0 = not."),
 "sigma_total":("Total GWSA trend uncertainty (1 sigma)","cm yr-1","Combined obs+model+Sy, spatially-correlated Monte Carlo."),
 "sigma_obs":  ("Observation-component trend uncertainty","cm yr-1","From GRACE/GLDAS observation sigma."),
 "sigma_model":("Model-component trend uncertainty","cm yr-1","From held-out reconstruction RMSE (7.05 cm)."),
 "sigma_sy":   ("Specific-yield-component trend uncertainty","cm yr-1","From specific-yield relative error."),
 "frac_obs":   ("Observation variance fraction","1","Share of total trend variance from observations."),
 "frac_model": ("Model variance fraction","1","Share of total trend variance from model error."),
 "frac_sy":    ("Specific-yield variance fraction","1","Share of total trend variance from Sy."),
 "dominant":   ("Dominant uncertainty component","1","0=obs, 1=model, 2=Sy."),
 "snr":        ("Trend signal-to-noise ratio","1","|trend| / sigma_total."),
 "significant_full":("Significance after full uncertainty","1","1 if |trend|/sigma_total >= 1.96."),
 "tier":       ("Validation tier","1","1=well-constrained (near in-situ wells), 0=physics-extrapolated."),
 "hybas_l3":   ("HydroBASINS level-3 basin ID","1","Pfafstetter-coded basin identifier."),
}


def main():
    os.makedirs(DEST,exist_ok=True)
    import xarray as xr, pandas as pd
    _p(LINE); _p("PHASE 8 -- FINAL PRODUCT PACKAGING"); _p(f"run {dt.datetime.now().isoformat(timespec='seconds')}"); _p(LINE)

    hdr("LOAD frozen product + uncertainty")
    G=xr.open_dataset(FINAL_NC); G.load()
    U=xr.open_dataset(UNC_NC); U.load()
    keepG=G["keep"].values.astype(bool); keepU=U["keep"].values.astype(bool)
    assert keepG.sum()==keepU.sum(), "product/uncertainty keep mismatch!"
    keep=keepG & keepU
    ci=np.where(keep)[0]
    _p(f"  aligned valid cells: {ci.size}")
    tvals=pd.to_datetime(G["time"].values)
    clat=G["clat"].values; clon=G["clon"].values

    # assemble variable dict on FULL cell axis (NaN off-keep), then subset to keep
    def full1d(src,name):
        v=src[name].values.astype("float32")
        out=np.where(keep,v,np.nan) if v.ndim==1 else v
        return out
    var1d={
        "gwsa_trend":full1d(G,"gwsa_trend"),
        "trend_ci_lo":full1d(G,"trend_ci_lo"),"trend_ci_hi":full1d(G,"trend_ci_hi"),
        "trend_significant":full1d(G,"trend_significant"),
        "sigma_total":full1d(U,"sigma_total"),"sigma_obs":full1d(U,"sigma_obs"),
        "sigma_model":full1d(U,"sigma_model"),"sigma_sy":full1d(U,"sigma_sy"),
        "frac_obs":full1d(U,"frac_obs"),"frac_model":full1d(U,"frac_model"),"frac_sy":full1d(U,"frac_sy"),
        "dominant":full1d(U,"dominant"),"snr":full1d(U,"snr"),
        "significant_full":full1d(U,"significant_full"),"tier":full1d(U,"tier"),
        "hybas_l3":G["hybas_l3"].values.astype("float64"),
    }
    gwsa_ts=G["gwsa_calibrated"].values.astype("float32")   # (time, cell)
    gwsa_ts=np.where(keep[None,:],gwsa_ts,np.nan)

    # ------------------------------------------------- 1D NetCDF
    hdr("WRITE 1D NetCDF (cell, time)")
    ds1=xr.Dataset(
        data_vars={"gwsa":(("time","cell"),gwsa_ts),
                   **{k:(("cell",),v) for k,v in var1d.items()},
                   "lat":(("cell",),clat.astype("float32")),
                   "lon":(("cell",),clon.astype("float32"))},
        coords={"time":tvals,"cell":np.arange(keep.size)})
    _apply_cf(ds1); ds1.attrs.update(_global_attrs(ci.size,tvals))
    p1=os.path.join(DEST,f"GWSA_KUET_{VERSION}_1D.nc")
    enc={v:{"zlib":True,"complevel":4} for v in ds1.data_vars}
    ds1.to_netcdf(p1,encoding=enc); _p(f"  [OK] {p1}")

    # ------------------------------------------------- gridded NetCDF
    hdr("WRITE gridded NetCDF (time, lat, lon)")
    lat=np.arange(-89.75,90,0.5,dtype="float32"); lon=np.arange(-179.75,180,0.5,dtype="float32")
    ai=np.round((clat+89.75)/0.5).astype(int); aj=np.round((clon+179.75)/0.5).astype(int)
    def grid2d(v1):
        g=np.full((NLAT,NLON),np.nan,"float32"); m=keep
        g[ai[m],aj[m]]=v1[m]; return g
    def grid3d(v2):
        g=np.full((tvals.size,NLAT,NLON),np.nan,"float32")
        g[:,ai[keep],aj[keep]]=v2[:,keep]; return g
    gvars={k:(("lat","lon"),grid2d(v)) for k,v in var1d.items()}
    gvars["gwsa"]=(("time","lat","lon"),grid3d(gwsa_ts))
    dsg=xr.Dataset(data_vars=gvars,coords={"time":tvals,"lat":lat,"lon":lon})
    _apply_cf(dsg); dsg["lat"].attrs.update(units="degrees_north",long_name="latitude")
    dsg["lon"].attrs.update(units="degrees_east",long_name="longitude")
    dsg.attrs.update(_global_attrs(ci.size,tvals)); dsg.attrs["geospatial_resolution"]="0.5 degree"
    pg=os.path.join(DEST,f"GWSA_KUET_{VERSION}_gridded.nc")
    encg={v:{"zlib":True,"complevel":4,"_FillValue":np.float32(np.nan)} for v in dsg.data_vars}
    dsg.to_netcdf(pg,encoding=encg); _p(f"  [OK] {pg}")

    # ------------------------------------------------- README + provenance
    hdr("WRITE README + provenance JSON")
    _write_readme(os.path.join(DEST,"README_GWSA.md"),ci.size,tvals)
    _write_provenance(os.path.join(DEST,"provenance_GWSA.json"),ci.size,tvals)
    _p("  [OK] README_GWSA.md  +  provenance_GWSA.json")
    _p(f"\n  Phase 8 complete. Deliverables in {DEST}"); _p(LINE)


def _apply_cf(ds):
    for v in ds.data_vars:
        if v in CF:
            ln,un,cm=CF[v]; ds[v].attrs.update(long_name=ln,units=un,comment=cm)


def _global_attrs(ncell,tvals):
    return {
        "title":"Global downscaled groundwater storage anomaly (GWSA), 0.5 degree, 2002-2022",
        "summary":("Monthly groundwater storage anomalies and long-term trends over global land "
                   "(surface water bodies excluded), derived by physics-informed deep-learning "
                   "downscaling and assimilation of GRACE/GRACE-FO, hydrological, and specific-yield "
                   "data, calibrated to storage amplitude and validated against in-situ wells (GROW)."),
        "institution":AFFIL,
        "creator_name":AUTHOR,"creator_id":AUTHOR_ID,
        "Conventions":"CF-1.10",
        "source":"Physics-informed LSTM downscaling + variance-matching calibration",
        "history":f"{dt.datetime.now().isoformat(timespec='seconds')} Phase 8 packaging",
        "references":"Gou & Soja (2024); Wang et al. (2025, igGWSA); Jasechko et al. (2024); Baethge et al. (2026, GROW)",
        "baseline_period":f"{BASE_Y0}-{BASE_Y1} mean removed",
        "time_coverage_start":str(tvals[0])[:10],"time_coverage_end":str(tvals[-1])[:10],
        "valid_land_cells":int(ncell),
        "surface_water_note":"2937 surface-water cells excluded from the product and all analyses.",
        "validation_summary":("Beats WaterGAP & GRACE-SeDA-implied GWSA against independent GROW wells "
                              "at aquifer/basin scale; per-basin temporal PCC consistent with Wang et al. (2025)."),
        "uncertainty_note":"sigma_total = spatially-correlated Monte Carlo (obs+model+Sy); see sigma_* and frac_* fields.",
        "tier_note":"tier=1 well-constrained (near in-situ wells); tier=0 physics-extrapolated.",
        "license":"CC-BY-4.0",
        "doi":DOI_PLACEHOLDER,
    }


def _write_readme(path,ncell,tvals):
    rows="\n".join(f"| `{k}` | {CF[k][1]} | {CF[k][0]} |" for k in CF if k!="gwsa")
    txt=f"""# Global Groundwater Storage Anomaly (GWSA) Dataset {VERSION}

**Author:** {AUTHOR} ({AFFIL}), ID {AUTHOR_ID}
**Coverage:** {str(tvals[0])[:7]} to {str(tvals[-1])[:7]}, global land (surface water excluded), 0.5 degree
**Valid land cells:** {ncell}
**DOI:** {DOI_PLACEHOLDER}
**License:** CC-BY-4.0

## Abstract
Monthly groundwater storage anomalies (GWSA) and long-term trends over global land at 0.5 degree
resolution. The product is derived by physics-informed deep-learning downscaling and assimilation of
GRACE/GRACE-FO terrestrial water storage, hydrological drivers, and specific-yield information,
calibrated to storage amplitude and validated against independent in-situ groundwater observations
(GROW). Baseline: {BASE_Y0}-{BASE_Y1} mean removed. Surface-water bodies are excluded.

## Files
- `GWSA_KUET_{VERSION}_1D.nc` - compact (cell, time), CF-1.10.
- `GWSA_KUET_{VERSION}_gridded.nc` - (time, lat, lon), 0.5 degree, GIS-ready (QGIS/Panoply).
- `provenance_GWSA.json` - machine-readable provenance.

## Variables
| variable | units | description |
|---|---|---|
| `gwsa` (time,...) | cm | Calibrated monthly GWSA (baseline {BASE_Y0}-{BASE_Y1}) |
{rows}

## Methods (summary)
1. GWSA isolation: GRACE TWSA minus non-groundwater storage (soil, snow, canopy).
2. Physics-informed LSTM downscaling: trend branch bounded by flux-derived supply; seasonal + residual branches.
3. Self-supervised data assimilation with basin mass-conservation term.
4. Variance-matching bias correction (calibration to storage amplitude).
5. Trend estimation (Theil-Sen) with block-bootstrap confidence intervals.
6. Uncertainty: spatially-correlated Monte Carlo combining observation, model, and specific-yield components.

## Validation (summary)
- Independent in-situ wells (GROW, Baethge et al. 2026): significant, direction-correct agreement,
  strongest at hydrogeologically-coherent aquifer/basin scales; representativeness-bounded at point scale
  (sigma_MAD/|trend| = 1.88, quantified a priori).
- Benchmarked through an identical pipeline against WaterGAP (raw) and GRACE-SeDA-implied GWSA: the product
  agrees with wells at least as well as, and at aquifer/basin scale better than, both benchmarks.
- Per-basin temporal correlation (PCC) consistent with Wang et al. (2025, igGWSA) at depletion hotspots.

## How to cite
{AUTHOR} ({dt.datetime.now().year}). Global Groundwater Storage Anomaly (GWSA) Dataset {VERSION}. {DOI_PLACEHOLDER}
"""
    open(path,"w",encoding="utf-8").write(txt)


def _write_provenance(path,ncell,tvals):
    prov={
        "dataset":f"GWSA_KUET_{VERSION}",
        "author":{"name":AUTHOR,"affiliation":AFFIL,"id":AUTHOR_ID},
        "created":dt.datetime.now().isoformat(timespec="seconds"),
        "spatial_resolution":"0.5 degree","temporal_resolution":"monthly",
        "time_coverage":[str(tvals[0])[:10],str(tvals[-1])[:10]],
        "valid_land_cells":int(ncell),
        "baseline_period":f"{BASE_Y0}-{BASE_Y1}",
        "surface_water_excluded_cells":2937,
        "license":"CC-BY-4.0","doi":DOI_PLACEHOLDER,
        "input_datasets":[
            {"name":"GRACE/GRACE-FO JPL mascon","role":"TWSA source"},
            {"name":"GLDAS Noah","role":"soil moisture / non-groundwater components"},
            {"name":"ERA5-Land","role":"meteorological drivers / evaluation"},
            {"name":"WaterGAP (WGHM)","role":"hydrological structure / benchmark"},
            {"name":"GROW (Baethge et al. 2026)","role":"in-situ well validation"},
            {"name":"Jasechko et al. (2024) aquifers","role":"aquifer polygons"},
            {"name":"HydroBASINS level-3","role":"basin polygons"},
            {"name":"GRACE-SeDA (Gou & Soja 2024)","role":"cross-product consistency benchmark"},
        ],
        "processing_steps":[
            "GWSA isolation (TWSA - non-groundwater storage)",
            "Physics-informed LSTM downscaling (supply-bounded trend + seasonal + residual)",
            "Self-supervised assimilation with basin mass-conservation loss",
            "Variance-matching bias correction (amplitude calibration)",
            "Theil-Sen trend + block-bootstrap CI",
            "Spatially-correlated Monte Carlo uncertainty (obs+model+Sy)",
            "Surface-water exclusion (2937 cells)",
            "CF-1.10 packaging (1D + gridded)",
        ],
        "software":{"python":sys.version.split()[0]},
    }
    open(path,"w",encoding="utf-8").write(json.dumps(prov,indent=2))


if __name__=="__main__":
    try: main()
    except Exception:
        import traceback; traceback.print_exc()