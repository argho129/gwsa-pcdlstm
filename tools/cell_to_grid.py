#!/usr/bin/env python3
"""
cell_to_grid.py  --  convert the 1-D (cell) product to a CF-style 2-D grid
(time, lat, lon) on the 0.5-degree global grid, so users can open it directly in
Panoply, QGIS, xarray or CDO.

The placement follows the product's own convention:
    row = round((lat + 89.75) / 0.5),  col = round((lon + 179.75) / 0.5)

Usage:
    python tools/cell_to_grid.py GWSA_KUET_v1_1D.nc GWSA_KUET_v1_2D.nc

Every variable with a 'cell' dimension is regridded; cells that are NaN in the
1-D file (surface water, ice, outside the domain) stay NaN (integer flags get -1).
Collisions and out-of-range cells are reported and make the script stop.
"""
import sys
import numpy as np
import xarray as xr

RES, NLAT, NLON = 0.5, 360, 720


def pick(ds, names):
    for n in names:
        if n in ds.variables:
            return np.asarray(ds[n].values, float)
    raise KeyError(f"none of {names} in file")


def main(src, dst):
    ds = xr.open_dataset(src)
    lat = pick(ds, ("clat", "lat"))
    lon = pick(ds, ("clon", "lon"))
    row = np.round((lat + 89.75) / RES).astype(int)
    col = np.round((lon + 179.75) / RES).astype(int)
    ok = (row >= 0) & (row < NLAT) & (col >= 0) & (col < NLON)
    if not ok.all():
        raise SystemExit(f"{int((~ok).sum())} cells fall outside the grid")
    flat = row * NLON + col
    if np.unique(flat).size != flat.size:
        raise SystemExit("two cells map to the same grid point; check lat/lon")

    glat = -89.75 + RES * np.arange(NLAT)
    glon = -179.75 + RES * np.arange(NLON)
    out = {}
    skip = {"clat", "clon", "lat", "lon"}
    for name, da in ds.data_vars.items():
        if name in skip or "cell" not in da.dims:
            continue
        other = [d for d in da.dims if d != "cell"]
        v = da.transpose(*other, "cell").values
        is_int = np.issubdtype(v.dtype, np.integer)
        fill = -1 if is_int else np.nan
        dtype = v.dtype if is_int else np.float32
        g = np.full(v.shape[:-1] + (NLAT, NLON), fill, dtype=dtype)
        g[..., row, col] = v
        out[name] = (tuple(other) + ("lat", "lon"), g, dict(da.attrs))

    coords = {"lat": ("lat", glat, {"units": "degrees_north", "standard_name": "latitude"}),
              "lon": ("lon", glon, {"units": "degrees_east", "standard_name": "longitude"})}
    for d in {d for _, (dims, _, _) in out.items() for d in dims} - {"lat", "lon"}:
        if d in ds.coords:
            coords[d] = ds[d]
    res = xr.Dataset({k: (dims, arr, attrs) for k, (dims, arr, attrs) in out.items()}, coords=coords)
    res.attrs = dict(ds.attrs)
    res.attrs.update({"grid": "regular 0.5 degree, cell centres", "Conventions": "CF-1.8",
                      "converted_from": str(src)})
    enc = {k: {"zlib": True, "complevel": 4} for k in res.data_vars}
    res.to_netcdf(dst, encoding=enc)
    print(f"wrote {dst}: {len(out)} variables, {int(ok.sum())} cells placed")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
