"""
glhymps_rasterio_stream.py — ONE-TIME, low-memory, valid areal mean.
Streams GLHYMPS in feature batches -> single 0.05deg raster -> block-mean 0.5deg.
No osgeo. rasterio only. Read-only on shapefile. Writes glhymps_0p5.nc + sanity.
"""
import numpy as np, xarray as xr, pyogrio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from pathlib import Path

GLH  = Path(r"D:\THESIS\Arpon_Thesis_Final\DATA\SOIL_DATA\GLHYMPS.shp")
GRID = Path(r"D:\THESIS\Arpon_Thesis_Final\OUTPUTS\02_gapfill\gwsa_monthly_filled.nc")
OUT  = Path(r"D:\THESIS\Arpon_Thesis_Final\DATA\SOIL_DATA\glhymps_0p5.nc")

FIELDS = ["logK_Ferr_", "logK_Ice_x", "Porosity_x"]
NAMES  = {"logK_Ferr_":"logk_ferr","logK_Ice_x":"logk_ice","Porosity_x":"porosity"}
SUB   = 10                    # 0.05 -> 0.5 deg
BATCH = 50_000                # features per chunk (tune down if RAM tight)

# target grid
g = xr.open_dataset(GRID); tlat=g["lat"].values.astype("float64")
tlon=g["lon"].values.astype("float64"); g.close()
lat_asc = tlat[0] < tlat[-1]

xmin, ymax = tlon.min()-0.25, tlat.max()+0.25
res = 0.5/SUB
nx = int(round(((tlon.max()+0.25)-xmin)/res))
ny = int(round((ymax-(tlat.min()-0.25))/res))
transform = from_origin(xmin, ymax, res, res)     # 0.05deg, EPSG:4326, north-up
print(f"fine raster {ny}x{nx} @ {res}deg -> target {tlat.size}x{tlon.size}")

info = pyogrio.read_info(GLH); N = info["features"]
print(f"{N} polygons, batches of {BATCH}")

# one accumulator per field; last-writer-wins burn (like a mosaic)
grids = {f: np.full((ny, nx), np.nan, dtype="float32") for f in FIELDS}

for start in range(0, N, BATCH):
    n = min(BATCH, N - start)
    gdf = pyogrio.read_dataframe(
        GLH, columns=FIELDS, read_geometry=True,
        skip_features=start, max_features=n)
    # source CRS is equal-area; reproject THIS BATCH only (cheap, 50k geoms)
    gdf = gdf.to_crs("EPSG:4326")
    geoms = gdf.geometry.values
    for f in FIELDS:
        vals = gdf[f].to_numpy("float64") / 100.0    # DECODE x100
        burn = rasterize(
            ((geom, v) for geom, v in zip(geoms, vals)),
            out=grids[f],                # burn INTO existing accumulator
            transform=transform,
            fill=np.nan, all_touched=False, default_value=np.nan)
    print(f"  {start+n:>8d}/{N}")

out={}
for f in FIELDS:
    fine = grids[f].astype("float64")
    fb = fine.reshape(ny//SUB, SUB, nx//SUB, SUB)
    with np.errstate(invalid="ignore"):
        s=np.nansum(fb,axis=(1,3)); c=np.sum(np.isfinite(fb),axis=(1,3)); m=s/c
    m[c==0]=np.nan
    if lat_asc: m=m[::-1,:]           # fine is lat-desc; match target
    out[NAMES[f]]=m

print("\n"+"="*66+"\nSANITY\n"+"="*66)
for nm,a in out.items():
    fin=np.isfinite(a); vv=a[fin]
    print(f"  {nm:11s} cov={100*fin.mean():5.1f}%  min={vv.min():8.3f} "
          f"med={np.median(vv):8.3f} max={vv.max():8.3f}")
    if nm=="porosity":
        print(f"              porosity==0: {int((vv==0).sum())} ({100*(vv==0).mean():.1f}%)")
ci=int(np.argmin(abs(tlat-30.75))); cj=int(np.argmin(abs(tlon-74.25)))
print(f"\n  hand-check lat={tlat[ci]:.2f} lon={tlon[cj]:.2f}:")
for nm,a in out.items(): print(f"    {nm:11s} = {a[ci,cj]:.3f}")

dso=xr.Dataset({nm:(("lat","lon"),a.astype("float32")) for nm,a in out.items()},
    coords={"lat":tlat,"lon":tlon},
    attrs={"title":"GLHYMPS 2.0 areal-mean 0.5deg (rasterio streaming + block-mean)",
           "method":"batched rasterize 0.05deg, 10x10 block-mean; equal-area=>areal mean",
           "source":"GLHYMPS 2.0 (Huscroft et al.; Gleeson et al. 2011)"})
dso["logk_ferr"].attrs={"units":"log10(m2)","long_name":"permeability, no permafrost corr"}
dso["logk_ice"].attrs ={"units":"log10(m2)","long_name":"permeability, with permafrost"}
dso["porosity"].attrs ={"units":"fraction","long_name":"near-surface porosity"}
enc={nm:{"zlib":True,"complevel":4,"_FillValue":np.float32(np.nan)} for nm in out}
dso.to_netcdf(OUT, encoding=enc)
print(f"\nwritten: {OUT} ({OUT.stat().st_size/1e6:.1f} MB)\nDONE")