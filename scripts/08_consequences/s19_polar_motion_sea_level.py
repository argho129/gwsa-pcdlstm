"""
================================================================================
PHASE 13 - GROUNDWATER DEPLETION -> EARTH'S ROTATIONAL POLE DRIFT
            + GRACE-SeDA ANNUAL CROSS-CHECK + FINAL FULL-PERIOD ESTIMATE
Arpon Sarkar, URP, KUET
================================================================================

STRUCTURE
  STEPS 1-4  Full-period (2002-2022) analysis: volumes, sea-level equivalent,
             secular pole drift under three masks, and a month-by-month pole
             path simulation that doubles as an internal consistency check.
  STEP 4b    GRACE-SeDA cross-check, restricted to the overlapping window
             (2002-2019). Compares annual total groundwater storage and the
             annual depletion rate. DIAGNOSTIC ONLY.
  STEP 4c    FINAL ESTIMATE over the full 2002-2022 period. This is what is
             reported. The Step 4b window is shorter only so the two products
             can be compared like for like.

READING GRACE-SeDA (Gou & Soja 2024, doi:10.1038/s44221-024-00194-w)
  - Time is stored as Modified Julian Date (epoch 1858-11-17) at the OBSERVATION
    MID-DATE, not at month starts. Two entries can therefore fall in the same
    calendar month; duplicates are averaged rather than dropped, and month
    matching is done through an explicit month->rows map (a pandas index built
    from these months is non-unique and get_indexer would raise).
  - Values are equivalent water height in mm -> factor 0.1 to cm.
  - SeDA is TOTAL water storage, so this study's OSA is subtracted to obtain a
    comparable groundwater anomaly. CAVEAT FOR THE TEXT: the same OSA is removed
    from both, so this isolates the TWSA component and is not a fully
    independent test of the groundwater partitioning.
  - SeDA removes the 2004.0-2009.999 mean; this study uses its own baseline.
    Both series are therefore centred on the overlap before comparison, and only
    TRENDS and year-to-year CHANGES are compared, never absolute values.

EXCITATION MATH - three corrections, retained
  FIX 1 GEOMETRY. The published integral form carries sin^2(theta) because one
        sin(theta) is the dS Jacobian. Per-cell MASSES are used here, so the
        discrete kernel is sin(theta)cos(theta)cos(phi).
  FIX 2 SIGN. I_ij = INT rho (r^2 delta_ij - x_i x_j) dV, so off-diagonal terms
        carry a leading minus. Omitting it rotates the azimuth by 180 deg.
  FIX 3 ELASTIC. (1 + k2') = 0.699 for the surface load AND 1/(1 - k2/ks) ~ 1.43
        for the rotation-axis response. Their product is ~1.00.

HONEST LIMITATIONS
  - Groundwater layer only: no ice sheets, glaciers, GIA, or the compensating
    ocean load. Drift MAGNITUDE is not comparable to Seo et al. (2023);
    DIRECTION is the meaningful comparison.
  - Secular approximation; no Chandler wobble dynamics.
  - Formal uncertainty propagates per-cell trend error only, and so understates
    the true figure.

DEPENDENCIES: numpy, xarray, pandas, matplotlib. cartopy only via fig_style.

Output: OUTPUTS\\13_polar_motion\\polar_motion_summary.json
        OUTPUTS\\13_polar_motion\\pole_path_monthly.csv
        OUTPUTS\\13_polar_motion\\seda_annual_comparison.csv
        OUTPUTS\\13_polar_motion\\seda_cellwise_trends.csv
        OUTPUTS\\13_polar_motion\\phase13_diagnostics.txt
        OUTPUTS\\13_polar_motion\\figures\\fig13_*.png / .pdf
================================================================================
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ------------------------------------------------------------------- paths --
IN_PATH = Path(r"D:\THESIS\Arpon_Thesis_Final\OUTPUTS\08_final_product\GWSA_KUET_v1_1D.nc")
OUT_DIR = Path(r"D:\THESIS\Arpon_Thesis_Final\OUTPUTS\13_polar_motion")
FIG_DIR = OUT_DIR / "figures"
FIG_STYLE_DIR = r"D:\THESIS\Arpon_Thesis_Final\Codes\Into the Models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

SEDA = {
    "ENABLED": True,
    "LABEL": "GRACE-SeDA",
    "PATH": Path(r"D:\THESIS\Arpon_Thesis_Final\DATA"
                 r"\Water_Paper Validation data (TWSA)"
                 r"\GRACE-SeDA_v0_2002_2019.nc"),
    "VAR": "twsa",
    "MASK_VAR": "mask",
    "UNITS_TO_CM": 0.1,          # EWH mm -> cm
    "OSA_PATH": Path(r"D:\THESIS\Arpon_Thesis_Final\OUTPUTS\05_model_tensor"
                     r"\gwsa_reconstructed.nc"),
    "OSA_VAR": "osa",
    "MIN_MONTHS_PER_YEAR": 8,    # a year needs this many months to be plotted
}

# --------------------------------------------------------------- constants --
A_EARTH = 6.371e6
C_MINUS_A = 2.6398e35
RHO_W = 1000.0
K2_PRIME = -0.301
ROT_DEFORM = 1.43
EXC_FACTOR = (1.0 + K2_PRIME) * ROT_DEFORM

CELL_DEG = 0.5
OCEAN_AREA_KM2 = 361.8e6
GT_PER_MM_SLE = OCEAN_AREA_KM2 * 1e6 * 1e-3 / 1e9      # 361.8

SEO_RATE_CM_YR = 4.36
SEO_DIR_DEG_E = 64.16
SEO_TOTAL_GT = 2150.0
SEO_PERIOD_YR = 17.0
SEO_KM3_YR = SEO_TOTAL_GT / SEO_PERIOD_YR
SEO_SLE_MM_YR = SEO_KM3_YR / GT_PER_MM_SLE

MJD_EPOCH = pd.Timestamp("1858-11-17")

LINE = "=" * 78
_LOG = []


def _p(s=""):
    print(s)
    _LOG.append(str(s))


def hdr(t):
    _p("")
    _p(LINE)
    _p(t)
    _p(LINE)


def pick(ds, *names):
    for n in names:
        if n in ds.data_vars or n in ds.coords:
            return ds[n]
    raise KeyError(f"None of {names} found. Available: {list(ds.data_vars) + list(ds.coords)}")


def find_name(ds, candidates):
    low = {str(k).lower(): k for k in list(ds.variables)}
    for c in candidates:
        if c in low:
            return low[c]
    return None


def spherical_cell_area_m2(lat_deg, cell_deg):
    half = cell_deg / 2.0
    lat_n = np.deg2rad(np.minimum(lat_deg + half, 90.0))
    lat_s = np.deg2rad(np.maximum(lat_deg - half, -90.0))
    return (A_EARTH ** 2) * np.deg2rad(cell_deg) * (np.sin(lat_n) - np.sin(lat_s))


def angular_sep(a, b):
    d = abs(a - b) % 360.0
    return d if d <= 180.0 else 360.0 - d


def nearest_idx(grid, q):
    order = np.argsort(grid)
    g = grid[order]
    pos = np.clip(np.searchsorted(g, q), 1, len(g) - 1)
    left, right = g[pos - 1], g[pos]
    return order[np.where(q - left <= right - q, pos - 1, pos)]


def parse_time_to_periods(da, label=""):
    """Return a PeriodIndex (monthly). Handles datetime64, Modified Julian Date,
    '<unit> since <date>' including the non-CF 'months since', decimal year and
    YYYYMM. Duplicated months are NOT collapsed here - the caller does that."""
    raw = np.asarray(da.values)
    if np.issubdtype(raw.dtype, np.datetime64):
        return pd.DatetimeIndex(raw).to_period("M"), "datetime64"

    units = str((getattr(da, "attrs", {}) or {}).get("units", ""))
    ul = units.lower()
    v = np.asarray(raw, dtype=float).ravel()

    if "since" in ul:
        unit_word = ul.split("since")[0].strip()
        base = pd.Timestamp(units.split("since")[1].strip().split()[0])
        if unit_word.startswith("month"):
            # udunits has no fixed-length month, so xarray refuses this. Handle
            # it directly: integer month offsets from the base period.
            return (pd.PeriodIndex([base.to_period("M") + int(round(x)) for x in v],
                                   freq="M"), f"'{units}' (integer month offsets)")
        code = {"day": "D", "hour": "h", "minute": "m", "second": "s"}.get(
            unit_word.rstrip("s"), None)
        if code:
            return (pd.DatetimeIndex(base + pd.to_timedelta(v, unit=code)).to_period("M"),
                    f"'{units}'")

    if np.isfinite(v).all():
        if "julian" in ul or "mjd" in ul or (30000.0 < v.min() and v.max() < 90000.0):
            return (pd.DatetimeIndex(MJD_EPOCH + pd.to_timedelta(v, unit="D")).to_period("M"),
                    "Modified Julian Date")
        if 1900 < v.min() and v.max() < 2100:
            yr = np.floor(v).astype(int)
            mo = np.clip(np.floor((v - yr) * 12.0).astype(int) + 1, 1, 12)
            return (pd.PeriodIndex([f"{y:04d}-{m:02d}" for y, m in zip(yr, mo)],
                                   freq="M"), "decimal year")
        if 190001 < v.min() and v.max() < 210013 and np.allclose(v, np.round(v)):
            iv = v.astype(int)
            return (pd.PeriodIndex([f"{x // 100:04d}-{x % 100:02d}" for x in iv],
                                   freq="M"), "YYYYMM integer")

    raise ValueError(f"could not parse the {label} time axis. dtype={raw.dtype}, "
                     f"units='{units}', first={raw.ravel()[:4]}, last={raw.ravel()[-2:]}")


def ols_slope_matrix(Y, tyr, min_obs=36):
    """Per-column OLS slope of Y (time, cell) against tyr, NaN-aware."""
    good = np.isfinite(Y)
    n = good.sum(axis=0)
    Yf = np.where(good, Y, 0.0)
    t = np.where(good, tyr[:, None], 0.0)
    st_, sy = t.sum(axis=0), Yf.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        denom = (t * t).sum(axis=0) - st_ * st_ / np.maximum(n, 1)
        slope = ((t * Yf).sum(axis=0) - st_ * sy / np.maximum(n, 1)) / denom
    slope[(n < min_obs) | ~np.isfinite(denom) | (denom <= 0)] = np.nan
    return slope


# =============================================================== STEP 1 ======
hdr("STEP 1 -- load Phase 8 product")
ds = xr.open_dataset(IN_PATH)
_p(f"  variables: {list(ds.data_vars)}")

trend = pick(ds, "gwsa_trend", "trend")
sigma_total = pick(ds, "sigma_total")
significant = pick(ds, "trend_significant", "significant_full")
clat, clon = pick(ds, "clat", "lat"), pick(ds, "clon", "lon")
gwsa_ts = pick(ds, "gwsa", "gwsa_calibrated")

lat = np.asarray(clat.values, dtype=float)
lon = np.asarray(clon.values, dtype=float)
tr = np.asarray(trend.values, dtype=float)
times = pd.to_datetime(ds["time"].values)
mt = pd.PeriodIndex(times.to_period("M"))

valid = (trend.notnull() & sigma_total.notnull()).values
sig = valid & (significant.values == 1)
mask_depl = sig & (tr < 0)
mask_signet = sig
mask_allnet = valid

area_m2 = spherical_cell_area_m2(lat, CELL_DEG)
area_km2 = area_m2 / 1e6
implied_land_Mkm2 = float(np.nansum(area_km2[valid]) / 1e6)

_p(f"  valid cells        : {int(valid.sum())} / {tr.size}")
_p(f"  significant cells  : {int(sig.sum())} ({100 * sig.sum() / max(valid.sum(), 1):.1f}% of valid)")
_p(f"  depleting cells    : {int(mask_depl.sum())}")
_p(f"  implied land area  : {implied_land_Mkm2:.1f} million km^2 (ice-free land ~130)")
_p(f"  period             : {times[0]:%Y-%m} to {times[-1]:%Y-%m} ({len(times)} months)")

theta = np.deg2rad(90.0 - lat)
phi = np.deg2rad(lon)
kern13 = np.sin(theta) * np.cos(theta) * np.cos(phi)
kern23 = np.sin(theta) * np.cos(theta) * np.sin(phi)

G = np.asarray(gwsa_ts.values, dtype=float)
if G.shape[0] == lat.size and G.shape[1] != lat.size:
    G = G.T


# =============================================================== STEP 2 ======
hdr("STEP 2 -- volumes and sea-level equivalent (three masks, full period)")
vol_trend = tr * area_km2 * 1e-5
sig_vol = np.asarray(sigma_total.values, dtype=float) * area_km2 * 1e-5


def block_sigma(mask, block_deg=6.0):
    s = np.where(mask, sig_vol, np.nan)
    good = np.isfinite(s)
    if good.sum() == 0:
        return float("nan")
    df = pd.DataFrame({"sigma": s[good],
                       "lat_bin": (lat[good] // block_deg).astype(int),
                       "lon_bin": (lon[good] // block_deg).astype(int)})
    return float(np.sqrt((df.groupby(["lat_bin", "lon_bin"])["sigma"].sum() ** 2).sum()))


vol_results = {}
for name, mask in [("depletion_only", mask_depl), ("net_significant", mask_signet),
                   ("net_all_valid", mask_allnet)]:
    v = float(np.nansum(np.where(mask, vol_trend, 0.0)))
    sg = block_sigma(mask)
    vol_results[name] = dict(cells=int(mask.sum()), volume_km3_yr=round(v, 2),
                             magnitude_km3_yr=round(abs(v), 2),
                             sigma_block_km3_yr=round(sg, 2),
                             sle_mm_yr=round(abs(v) / GT_PER_MM_SLE, 4),
                             sle_sigma_mm_yr=round(sg / GT_PER_MM_SLE, 4))
    _p(f"  {name:<18} n={mask.sum():>6}  vol={v:+9.2f} km3/yr  "
       f"|SLE|={abs(v) / GT_PER_MM_SLE:.4f} mm/yr  (formal sigma {sg:.2f} km3/yr)")


# =============================================================== STEP 3 ======
hdr("STEP 3 -- secular pole drift from the trend field (full period)")


def excitation_from_trend(mask, trend_field=None):
    t_use = tr if trend_field is None else trend_field
    loss = np.nan_to_num(np.where(mask, (-t_use / 100.0) * area_m2 * RHO_W, 0.0), nan=0.0)
    dI13 = -(A_EARTH ** 2) * np.sum(-loss * kern13)
    dI23 = -(A_EARTH ** 2) * np.sum(-loss * kern23)
    d1 = EXC_FACTOR / C_MINUS_A * dI13
    d2 = EXC_FACTOR / C_MINUS_A * dI23
    return dict(rate_cm_yr=round(float(np.hypot(d1, d2)) * A_EARTH * 100.0, 4),
                direction_deg_E=round(float(np.degrees(np.arctan2(d2, d1)) % 360.0), 2),
                mass_rate_Gt_yr=round(float(np.sum(loss)) / 1e12, 2))


pole_results = {}
for name, mask in [("depletion_only", mask_depl), ("net_significant", mask_signet),
                   ("net_all_valid", mask_allnet)]:
    r = excitation_from_trend(mask)
    r["sep_from_seo_deg"] = round(angular_sep(r["direction_deg_E"], SEO_DIR_DEG_E), 2)
    pole_results[name] = r
    _p(f"  {name:<18} {r['rate_cm_yr']:>8.4f} cm/yr toward {r['direction_deg_E']:>6.1f} deg E "
       f"(Seo separation {r['sep_from_seo_deg']:.1f} deg)")

_p("")
_p(f"  Seo et al. (2023): {SEO_RATE_CM_YR} cm/yr toward {SEO_DIR_DEG_E} deg E "
   f"({SEO_KM3_YR:.1f} km3/yr = {SEO_SLE_MM_YR:.3f} mm/yr SLE)")
d_dep = pole_results["depletion_only"]["direction_deg_E"]
d_net = pole_results["net_significant"]["direction_deg_E"]
mask_spread = angular_sep(d_dep, d_net)
_p(f"  MASK ROBUSTNESS: depletion-only vs net-significant differ by {mask_spread:.1f} deg.")
if pole_results["depletion_only"]["sep_from_seo_deg"] > 150.0:
    _p("  >> WARNING: ~180 deg from Seo indicates a residual SIGN error.")


# =============================================================== STEP 4 ======
hdr("STEP 4 -- month-by-month pole path simulation (full period)")
cellw = np.where(mask_signet, area_m2 * RHO_W / 100.0, 0.0)
Gm = np.nan_to_num(G, nan=0.0) * cellw[None, :]
m1_cm = EXC_FACTOR / C_MINUS_A * (-(A_EARTH ** 2) * (Gm @ kern13)) * A_EARTH * 100.0
m2_cm = EXC_FACTOR / C_MINUS_A * (-(A_EARTH ** 2) * (Gm @ kern23)) * A_EARTH * 100.0

tyr = np.asarray((times - times[0]).days / 365.25, dtype=float)
A_des = np.vstack([tyr, np.ones_like(tyr)]).T
fit1 = np.linalg.lstsq(A_des, m1_cm, rcond=None)[0]
fit2 = np.linalg.lstsq(A_des, m2_cm, rcond=None)[0]
s1, s2 = fit1[0], fit2[0]
path_rate = float(np.hypot(s1, s2))
path_dir = float(np.degrees(np.arctan2(s2, s1)) % 360.0)
pd.DataFrame({"time": times, "m1_cm": m1_cm, "m2_cm": m2_cm}).to_csv(
    OUT_DIR / "pole_path_monthly.csv", index=False)

sec = pole_results["net_significant"]
_p(f"  simulated path drift : {path_rate:.4f} cm/yr toward {path_dir:.1f} deg E")
_p(f"  secular estimate     : {sec['rate_cm_yr']:.4f} cm/yr toward {sec['direction_deg_E']:.1f} deg E")
_p(f"  agreement            : "
   f"{100 * abs(path_rate - sec['rate_cm_yr']) / max(sec['rate_cm_yr'], 1e-9):.2f}% rate, "
   f"{angular_sep(path_dir, sec['direction_deg_E']):.2f} deg azimuth")
_p(f"  seasonal excursion   : {float(np.std(m1_cm)):.2f} / {float(np.std(m2_cm)):.2f} cm (1-sigma)")
_p("  The seasonal excursion is the annual hydrological wobble, not a secular")
_p("  signal. Report the fitted drift, not the excursion amplitude.")


# =============================================================== STEP 4b =====
hdr("STEP 4b -- GRACE-SeDA cross-check (overlap window, DIAGNOSTIC)")
seda = None


def volume_series(Field, mask):
    """Global summed storage over `mask` for each month, in km^3."""
    w = np.where(mask, area_m2 / 100.0, 0.0)          # cm -> m^3
    return np.nansum(np.where(np.isfinite(Field), Field, 0.0) * w[None, :], axis=1) / 1e9


if not SEDA["ENABLED"]:
    _p("  disabled.")
elif not SEDA["PATH"].exists():
    _p(f"  file not found: {SEDA['PATH']}")
else:
    try:
        de = xr.open_dataset(SEDA["PATH"], decode_times=False)
        _p(f"  variables: {list(de.data_vars)}   dims: {dict(de.sizes)}")
        latn = find_name(de, ["lat", "latitude", "y"])
        lonn = find_name(de, ["lon", "longitude", "x"])
        timn = find_name(de, ["time", "t"])
        vname = SEDA["VAR"] or [v for v in de.data_vars if de[v].ndim == 3][0]
        vunits = str(de[vname].attrs.get("units", "unknown"))
        _p(f"  variable '{vname}', file units '{vunits}', "
           f"factor {SEDA['UNITS_TO_CM']} -> cm")

        da = de[vname].transpose(timn, latn, lonn)
        glat = np.asarray(de[latn].values, dtype=float)
        glon = np.asarray(de[lonn].values, dtype=float)

        st, how = parse_time_to_periods(de[timn], "SeDA")
        _p(f"  time parsed as: {how}")
        _p(f"  SeDA span {st.min()} to {st.max()} ({len(st)} records)")

        # SeDA timestamps are observation MID-DATES, so a calendar month can hold
        # more than one record. Build an explicit month -> rows map and average
        # duplicates. A pandas index over these months is non-unique, which is
        # why get_indexer must not be used here.
        month_rows = {}
        for i, m in enumerate(st):
            month_rows.setdefault(m, []).append(i)
        ndup = sum(1 for v in month_rows.values() if len(v) > 1)
        if ndup:
            _p(f"  {ndup} calendar month(s) hold more than one SeDA record; "
               f"duplicates averaged")

        mine_rows = {m: i for i, m in enumerate(mt)}
        common = sorted(set(month_rows) & set(mine_rows))
        if len(common) < 60:
            raise ValueError(f"only {len(common)} overlapping months")
        _p(f"  overlap {common[0]} to {common[-1]} ({len(common)} months "
           f"of {len(mt)} available in this study)")

        lon_q = lon % 360.0 if glon.min() >= 0 else ((lon + 180.0) % 360.0) - 180.0
        ilat, ilon = nearest_idx(glat, lat), nearest_idx(glon, lon_q)
        dlat = np.abs(glat[ilat] - lat)
        dlon = np.abs(((glon[ilon] - lon_q + 180.0) % 360.0) - 180.0)
        _p(f"  grid match: max offset {dlat.max():.3f} lat, {dlon.max():.3f} lon "
           f"(expect <= {CELL_DEG / 2:.2f})")

        cell_ok = np.ones(lat.size, bool)
        mvar = SEDA.get("MASK_VAR")
        if mvar and mvar in de.variables:
            mv = np.asarray(de[mvar].values)
            if mv.ndim == 3:
                mv = mv[0]
            if mv.shape == (len(glat), len(glon)):
                cell_ok = np.asarray(mv, dtype=float)[ilat, ilon] > 0
                _p(f"  applied '{mvar}': {int(cell_ok.sum())}/{cell_ok.size} study "
                   f"cells on valid SeDA grid points")

        # read one month at a time, averaging duplicate records
        Ve = np.empty((len(common), lat.size), dtype="float32")
        for k, m in enumerate(common):
            rows = month_rows[m]
            acc = None
            for ri in rows:
                sl = np.asarray(da.isel({timn: int(ri)}).values, dtype="float32")[ilat, ilon]
                acc = sl if acc is None else acc + sl
            Ve[k] = acc / len(rows)
        Ve *= SEDA["UNITS_TO_CM"]
        Ve[(Ve < -1e19) | (Ve > 1e19)] = np.nan
        Ve[:, ~cell_ok] = np.nan

        do = xr.open_dataset(SEDA["OSA_PATH"])
        Vo = np.asarray(pick(do, SEDA["OSA_VAR"]).values, dtype=float)
        if Vo.shape[0] == lat.size and Vo.shape[1] != lat.size:
            Vo = Vo.T
        ot, ohow = parse_time_to_periods(do["time"], "OSA")
        osa_rows = {m: i for i, m in enumerate(ot)}
        miss = [str(m) for m in common if m not in osa_rows]
        if miss:
            raise ValueError(f"OSA missing {len(miss)} overlap months, e.g. {miss[:3]}")
        Ve = Ve - Vo[[osa_rows[m] for m in common]]
        _p(f"  subtracted OSA (time parsed as {ohow}) -> SeDA-derived GWSA")

        Gmine = G[[mine_rows[m] for m in common]]

        # ---- annual totals over the depleting cells ------------------------
        yrs = np.array([m.year for m in common])
        Vs = volume_series(Ve, mask_depl)
        Vm = volume_series(Gmine, mask_depl)
        Vs -= np.nanmean(Vs)                 # different baselines; centre both
        Vm -= np.nanmean(Vm)

        rows = []
        for y in np.unique(yrs):
            sel = yrs == y
            if sel.sum() < SEDA["MIN_MONTHS_PER_YEAR"]:
                _p(f"  year {y} has only {int(sel.sum())} months; excluded from the "
                   f"annual series")
                continue
            rows.append({"year": int(y), "months": int(sel.sum()),
                         "storage_mine_km3": float(np.nanmean(Vm[sel])),
                         "storage_seda_km3": float(np.nanmean(Vs[sel]))})
        ann = pd.DataFrame(rows)
        # GRACE / GRACE-FO gap years are dropped above, so consecutive ROWS can be
        # several years apart. Divide each storage difference by the actual number
        # of years spanned, otherwise the first row after a gap reports several
        # years of depletion as a single annual rate.
        ann["years_spanned"] = ann["year"].diff()
        ann["depletion_rate_mine_km3_yr"] = (-ann["storage_mine_km3"].diff()
                                             / ann["years_spanned"])
        ann["depletion_rate_seda_km3_yr"] = (-ann["storage_seda_km3"].diff()
                                             / ann["years_spanned"])
        _gapped = ann[ann["years_spanned"] > 1]
        if len(_gapped):
            _p(f"  {len(_gapped)} annual rate(s) span a gap and are divided by the "
               f"years bridged: "
               + ", ".join(f"{int(r.year)} ({int(r.years_spanned)} yr)"
                           for r in _gapped.itertuples()))
        ann.to_csv(OUT_DIR / "seda_annual_comparison.csv", index=False)

        yv = ann["year"].values.astype(float)
        sl_m = np.polyfit(yv, ann["storage_mine_km3"].values, 1)[0]
        sl_s = np.polyfit(yv, ann["storage_seda_km3"].values, 1)[0]

        _p("")
        _p(f"  ANNUAL TOTAL DEPLETION RATE over depleting cells "
           f"({int(mask_depl.sum())} cells)")
        _p(f"  {'year':<8}{'this study':>14}{'SeDA':>12}{'difference':>13}{'span':>7}")
        for _, rw in ann.iterrows():
            if np.isfinite(rw["depletion_rate_mine_km3_yr"]):
                _sp = int(rw["years_spanned"])
                _p(f"  {int(rw['year']):<8}{rw['depletion_rate_mine_km3_yr']:>14.1f}"
                   f"{rw['depletion_rate_seda_km3_yr']:>12.1f}"
                   f"{rw['depletion_rate_mine_km3_yr'] - rw['depletion_rate_seda_km3_yr']:>13.1f}"
                   f"{(str(_sp) + ' yr') if _sp > 1 else '':>7}")
        _p(f"  {'-' * 54}")
        _p(f"  {'mean':<8}{ann['depletion_rate_mine_km3_yr'].mean():>14.1f}"
           f"{ann['depletion_rate_seda_km3_yr'].mean():>12.1f}"
           f"{ann['depletion_rate_mine_km3_yr'].mean() - ann['depletion_rate_seda_km3_yr'].mean():>13.1f}")
        _p(f"  {'trend':<8}{-sl_m:>14.1f}{-sl_s:>12.1f}{-(sl_m - sl_s):>13.1f}"
           f"{'':>7}   <-- quote THIS row")
        _p("  The 'trend' row is a regression on the storage curve and is immune to")
        _p("  the gap issue; the year rows show interannual behaviour.")
        if sl_m * sl_s > 0 and abs(sl_s) > 0.05 * abs(sl_m):
            _p(f"  Ratio this study / SeDA = {sl_m / sl_s:.2f}. A value above 1 is the "
               f"expected")
            _p("  consequence of the Phase 5.2 variance matching, which restores "
               "amplitude")
            _p("  that GRACE-derived products damp; state it as such rather than as an "
               "error.")
        else:
            _p("  Slopes differ in sign or SeDA's slope is near zero, so their ratio is "
               "not")
            _p("  meaningful; compare the two slopes directly instead.")

        rr_ann = float(np.corrcoef(ann["storage_mine_km3"], ann["storage_seda_km3"])[0, 1])
        d_ann = ann.dropna(subset=["depletion_rate_mine_km3_yr"])
        rr_rate = (float(np.corrcoef(d_ann["depletion_rate_mine_km3_yr"],
                                     d_ann["depletion_rate_seda_km3_yr"])[0, 1])
                   if len(d_ann) > 3 else np.nan)
        _p(f"  annual storage correlation      r = {rr_ann:+.3f}")
        _p(f"  annual depletion-rate correlation r = {rr_rate:+.3f}")

        # ---- cell-wise trends on the same window ---------------------------
        t_ov = np.arange(len(common), dtype=float) / 12.0
        tr_seda = ols_slope_matrix(Ve, t_ov)
        tr_mine = ols_slope_matrix(Gmine, t_ov)
        both = np.isfinite(tr_seda) & np.isfinite(tr_mine)
        mk = mask_signet & both
        rr_cell = float(np.corrcoef(tr_mine[mk], tr_seda[mk])[0, 1])
        bias = float(np.mean(tr_mine[mk] - tr_seda[mk]))
        rmsd = float(np.sqrt(np.mean((tr_mine[mk] - tr_seda[mk]) ** 2)))
        agree = float(np.mean(np.sign(tr_mine[mk]) == np.sign(tr_seda[mk])) * 100.0)
        _p("")
        _p(f"  CELL-WISE TREND AGREEMENT over {int(mk.sum())} significant cells:")
        _p(f"     r = {rr_cell:+.3f}   bias (mine-SeDA) = {bias:+.4f} cm/yr")
        _p(f"     RMSD = {rmsd:.4f} cm/yr   sign agreement = {agree:.1f}%")
        pd.DataFrame({"lat": lat[mk], "lon": lon[mk], "trend_mine_cm_yr": tr_mine[mk],
                      "trend_seda_cm_yr": tr_seda[mk]}).to_csv(
            OUT_DIR / "seda_cellwise_trends.csv", index=False)

        # this study's own drift restricted to the same window
        pm_win = excitation_from_trend(mk, trend_field=np.nan_to_num(tr_mine))
        ps_win = excitation_from_trend(mk, trend_field=np.nan_to_num(tr_seda))

        seda = {
            "overlap_start": str(common[0]), "overlap_end": str(common[-1]),
            "overlap_months": int(len(common)),
            "duplicate_months_averaged": int(ndup),
            "grid_max_offset_deg": round(float(max(dlat.max(), dlon.max())), 4),
            "annual_storage_corr": round(rr_ann, 4),
            "annual_depletion_rate_corr": (round(rr_rate, 4)
                                           if np.isfinite(rr_rate) else None),
            "mean_annual_depletion_mine_km3_yr":
                round(float(ann["depletion_rate_mine_km3_yr"].mean()), 2),
            "mean_annual_depletion_seda_km3_yr":
                round(float(ann["depletion_rate_seda_km3_yr"].mean()), 2),
            "storage_slope_mine_km3_yr": round(-float(sl_m), 2),
            "storage_slope_seda_km3_yr": round(-float(sl_s), 2),
            "storage_slope_ratio_mine_over_seda": (
                round(float(sl_m / sl_s), 3)
                if sl_m * sl_s > 0 and abs(sl_s) > 0.05 * abs(sl_m) else None),
            "cellwise_trend_corr": round(rr_cell, 4),
            "cellwise_trend_bias_cm_yr": round(bias, 4),
            "cellwise_trend_rmsd_cm_yr": round(rmsd, 4),
            "cellwise_sign_agreement_pct": round(agree, 2),
            "window_drift_mine_cm_yr": pm_win["rate_cm_yr"],
            "window_azimuth_mine_deg_E": pm_win["direction_deg_E"],
            "window_drift_seda_cm_yr": ps_win["rate_cm_yr"],
            "window_azimuth_seda_deg_E": ps_win["direction_deg_E"],
            "window_azimuth_difference_deg": round(
                angular_sep(pm_win["direction_deg_E"], ps_win["direction_deg_E"]), 2),
            "_ann": ann, "_arrays": (tr_mine, tr_seda, mk),
        }
        _p("")
        _p("  CAVEAT FOR THE TEXT: this study's OSA is removed from SeDA as well,")
        _p("  so the comparison isolates the TWSA component and is not a fully")
        _p("  independent test of the groundwater partitioning.")
    except Exception as e:
        import traceback
        _p(f"  cross-check failed: {e}")
        traceback.print_exc()
        seda = None


# =============================================================== STEP 4c =====
hdr(f"STEP 4c -- FINAL ESTIMATE, FULL PERIOD {times[0]:%Y-%m} to {times[-1]:%Y-%m}")
_p("  This is the reported result. STEP 4b used a shorter window purely so the")
_p("  two products could be compared like for like; those numbers are")
_p("  diagnostic and must not be quoted as the estimate.")
_p("")
_rn, _vd, _vn = (pole_results["net_significant"], vol_results["depletion_only"],
                 vol_results["net_significant"])
_p(f"  {'quantity':<36}{'value':>16}")
_p(f"  {'-' * 52}")
for k, v in [("period", f"{times[0]:%Y-%m} to {times[-1]:%Y-%m}"),
             ("months", f"{len(times)}"),
             ("cells (valid / significant)", f"{int(valid.sum())} / {int(sig.sum())}"),
             ("gross depletion (km3/yr)", f"{_vd['magnitude_km3_yr']:.2f}"),
             ("gross depletion SLE (mm/yr)", f"{_vd['sle_mm_yr']:.4f}"),
             ("net significant (km3/yr)", f"{abs(_vn['volume_km3_yr']):.2f}"),
             ("net significant SLE (mm/yr)", f"{_vn['sle_mm_yr']:.4f}"),
             ("pole drift rate (cm/yr)", f"{_rn['rate_cm_yr']:.4f}"),
             ("pole drift azimuth (deg E)", f"{_rn['direction_deg_E']:.2f}"),
             ("separation from Seo et al. (deg)", f"{_rn['sep_from_seo_deg']:.2f}"),
             ("mask spread (deg)", f"{mask_spread:.2f}"),
             ("path-vs-secular azimuth (deg)",
              f"{angular_sep(path_dir, _rn['direction_deg_E']):.2f}"),
             ("path-vs-secular rate (%)",
              f"{100 * abs(path_rate - _rn['rate_cm_yr']) / _rn['rate_cm_yr']:.2f}")]:
    _p(f"  {k:<36}{v:>16}")

final_estimate = {
    "period": f"{times[0]:%Y-%m} to {times[-1]:%Y-%m}", "months": int(len(times)),
    "valid_cells": int(valid.sum()), "significant_cells": int(sig.sum()),
    "gross_depletion_km3_yr": _vd["magnitude_km3_yr"],
    "gross_depletion_sle_mm_yr": _vd["sle_mm_yr"],
    "net_significant_km3_yr": round(abs(_vn["volume_km3_yr"]), 2),
    "net_significant_sle_mm_yr": _vn["sle_mm_yr"],
    "pole_drift_cm_yr": _rn["rate_cm_yr"],
    "pole_drift_azimuth_deg_E": _rn["direction_deg_E"],
    "separation_from_seo_deg": _rn["sep_from_seo_deg"],
    "mask_spread_deg": round(mask_spread, 2),
}
if seda is not None:
    shift = angular_sep(seda["window_azimuth_mine_deg_E"], _rn["direction_deg_E"])
    final_estimate["window_sensitivity"] = {
        "window": f"{seda['overlap_start']}..{seda['overlap_end']}",
        "drift_cm_yr": seda["window_drift_mine_cm_yr"],
        "azimuth_deg_E": seda["window_azimuth_mine_deg_E"],
        "azimuth_shift_vs_full_deg": round(shift, 2)}
    _p("")
    _p("  WINDOW SENSITIVITY of this study's own result (net_significant):")
    _p(f"    full {times[0]:%Y-%m}..{times[-1]:%Y-%m}: "
       f"{_rn['rate_cm_yr']:.3f} cm/yr @ {_rn['direction_deg_E']:.2f} deg E")
    _p(f"    SeDA {seda['overlap_start']}..{seda['overlap_end']}: "
       f"{seda['window_drift_mine_cm_yr']:.3f} cm/yr @ "
       f"{seda['window_azimuth_mine_deg_E']:.2f} deg E  (shift {shift:.2f} deg)")
    _p("    A small shift means the reported azimuth is not an artefact of the")
    _p("    period chosen.")


# =============================================================== STEP 5 ======
hdr("STEP 5 -- figures")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HAVE_STYLE = False
try:
    sys.path.insert(0, FIG_STYLE_DIR)
    import fig_style as fs
    fs.apply()
    HAVE_STYLE = True
except Exception:
    pass

FS_TICK, FS_LABEL, FS_TITLE, FS_ANNOT, FS_EQ = 10, 11, 12, 10, 15
matplotlib.rcParams.update({
    "font.family": "serif", "font.size": FS_TICK, "axes.labelsize": FS_LABEL,
    "axes.titlesize": FS_TITLE, "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
    "legend.fontsize": FS_ANNOT, "mathtext.fontset": "dejavuserif",
    "axes.linewidth": 0.8, "savefig.dpi": 600, "figure.dpi": 120})
_p(f"  fig_style: {'loaded' if HAVE_STYLE else 'fallback'}; typography enforced")

NAVY, RED, GREEN, GREY, BLACK = "#1f4e79", "#c0392b", "#2ca02c", "#7f8c8d", "#111111"


def save(fig, nm):
    for e in (".png", ".pdf"):
        fig.savefig(FIG_DIR / (nm + e), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    _p(f"  [ OK ] {nm}")


def despine(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=FS_TICK)


# ---- FIG 1: equations + azimuth rose ---------------------------------------
fig = plt.figure(figsize=(13.0, 6.0))
axT = fig.add_axes((0.015, 0.05, 0.37, 0.90))
axT.axis("off")
axT.text(0.0, 1.00, "Excitation-function pathway", fontsize=FS_TITLE + 2,
         weight="bold", va="top")
for y, eq, cap in [
    (0.855, r"$\dot{m}_i \;=\; -\,\dot{h}_i \, A_i \, \rho_w$",
     "per-cell mass rate from the GWSA trend"),
    (0.665, r"$\dot{I}_{13} = -a^{2}\sum_i \dot{m}_i \sin\theta_i \cos\theta_i \cos\phi_i$",
     "off-diagonal inertia perturbation"),
    (0.475, r"$\dot{I}_{23} = -a^{2}\sum_i \dot{m}_i \sin\theta_i \cos\theta_i \sin\phi_i$",
     "orthogonal component"),
    (0.285, r"$\dot{\chi}_{1,2} = \dfrac{(1+k_2')\,\kappa}{C-A}\;\dot{I}_{13,23}$",
     r"$\kappa=(1-k_2/k_s)^{-1}\approx 1.43$"),
    (0.095, r"$\dot{p}=a\sqrt{\dot{\chi}_1^{2}+\dot{\chi}_2^{2}}\;,\quad "
            r"\alpha=\arctan(\dot{\chi}_2/\dot{\chi}_1)$", "drift rate and azimuth")]:
    axT.text(0.015, y, eq, fontsize=FS_EQ, va="center")
    axT.text(0.015, y - 0.072, cap, fontsize=FS_ANNOT - 1, va="center",
             color=GREY, style="italic")
axT.text(0.015, -0.03,
         f"$a$ = {A_EARTH:.3e} m      $C-A$ = {C_MINUS_A:.4e} kg m$^2$\n"
         f"$k_2'$ = {K2_PRIME}      combined factor = {EXC_FACTOR:.3f}",
         fontsize=FS_ANNOT - 1, va="center", color="0.3")

# The four azimuths agree to within a few degrees, so the full 360 deg circle
# renders them as one indistinguishable bundle. Zoom to a wedge spanning just
# the range of interest so the agreement - the actual result - is legible.
_rose = [("Depletion only", pole_results["depletion_only"]["direction_deg_E"], RED, "-", 1.00),
         ("Net (significant)", pole_results["net_significant"]["direction_deg_E"], NAVY, "-", 0.88),
         ("Net (all valid)", pole_results["net_all_valid"]["direction_deg_E"], GREEN, "--", 0.76),
         ("Seo et al. (2023)", SEO_DIR_DEG_E, BLACK, "-", 0.64)]
_angs = [a for _, a, _, _, _ in _rose]
_pad = max(8.0, 0.45 * (max(_angs) - min(_angs)))
_lo, _hi = min(_angs) - _pad, max(_angs) + _pad
if _hi - _lo < 30.0:                      # keep the wedge wide enough to read
    _mid = 0.5 * (_lo + _hi)
    _lo, _hi = _mid - 15.0, _mid + 15.0

axP = fig.add_axes((0.44, 0.20, 0.55, 0.74), projection="polar")
axP.set_theta_zero_location("E")
axP.set_theta_direction(1)
axP.set_thetamin(_lo)
axP.set_thetamax(_hi)
for lab, ang, col, ls, rad in _rose:
    axP.annotate("", xy=(np.deg2rad(ang), rad), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color=col, lw=2.6, linestyle=ls,
                                 mutation_scale=20))
    # value lives in the legend only; an inline label here collides with the
    # arrowheads once the wedge is zoomed
    axP.plot([], [], color=col, lw=2.6, ls=ls, label=f"{lab}  {ang:.1f}$\\degree$E")
axP.set_rmax(1.16)
axP.set_yticklabels([])
axP.set_thetagrids(np.arange(np.ceil(_lo / 5) * 5, _hi + 0.1, 5))
axP.tick_params(labelsize=FS_TICK, pad=2)
axP.set_title("Drift azimuth\n(radius carries no meaning; wedge is zoomed)",
              fontsize=FS_TITLE, pad=14)
axP.grid(alpha=0.35, lw=0.6)
axP.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), frameon=False,
           fontsize=FS_ANNOT, ncol=2, columnspacing=1.4)
axP.text(0.5, -0.30, f"separation from Seo et al.: {_rn['sep_from_seo_deg']:.1f}$\\degree$"
                     f"   |   mask spread: {mask_spread:.1f}$\\degree$",
         transform=axP.transAxes, ha="center", fontsize=FS_ANNOT, color=NAVY)
save(fig, "fig13_1_physics_and_azimuth")


# ---- FIG 2: pole path simulation -------------------------------------------
fig = plt.figure(figsize=(13.0, 5.6))
ax = fig.add_axes((0.055, 0.135, 0.36, 0.775))
sc = ax.scatter(m1_cm, m2_cm, c=tyr, cmap="viridis", s=11, linewidths=0, zorder=3)
ax.plot(m1_cm, m2_cm, color="0.6", lw=0.55, alpha=0.85, zorder=2)
ax.axhline(0, color="k", lw=0.6)
ax.axvline(0, color="k", lw=0.6)
span = float(np.nanmax(np.abs(np.r_[m1_cm, m2_cm]))) * 1.18
# Anchor the drift arrow on the FITTED LINE, not on the origin. The path starts
# near (fit1[1], fit2[1]), so an origin-anchored arrow would imply the pole began
# at zero displacement, which it did not.
_x0, _y0 = float(fit1[1]), float(fit2[1])
_x1, _y1 = _x0 + s1 * float(tyr.max()), _y0 + s2 * float(tyr.max())
ax.annotate("", xy=(_x1, _y1), xytext=(_x0, _y0),
            arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.6, mutation_scale=20), zorder=6)
# label sits beside the arrow tip, so no leader line crosses the panel
_perp = np.array([-(_y1 - _y0), (_x1 - _x0)], dtype=float)
_perp /= max(np.hypot(*_perp), 1e-9)
ax.annotate(f"{path_rate:.2f} cm yr$^{{-1}}$\n{path_dir:.1f}$\\degree$E",
            xy=(_x1, _y1), xytext=(_x1 + _perp[0] * span * 0.30 + span * 0.04,
                                   _y1 + _perp[1] * span * 0.30 - span * 0.06),
            color=RED, fontsize=FS_ANNOT, ha="center", va="center", zorder=7,
            bbox=dict(boxstyle="round,pad=0.30", fc="white", ec=RED, lw=0.9, alpha=0.95))
ax.plot([_x0], [_y0], marker="o", ms=5, mfc="white", mec=RED, mew=1.4, zorder=7)
ax.set_xlim(-span, span)
ax.set_ylim(-span, span)
ax.set_aspect("equal")
ax.set_xlabel("Displacement toward 0$\\degree$E (cm)", fontsize=FS_LABEL)
ax.set_ylabel("Displacement toward 90$\\degree$E (cm)", fontsize=FS_LABEL)
ax.set_title(f"Simulated pole path, {times[0]:%Y}-{times[-1]:%Y}", fontsize=FS_TITLE)
despine(ax)
cb = fig.colorbar(sc, cax=fig.add_axes((0.425, 0.20, 0.011, 0.64)))
cb.set_label("Years since start", fontsize=FS_ANNOT)
cb.ax.tick_params(labelsize=FS_TICK)
ax2 = fig.add_axes((0.565, 0.575, 0.40, 0.335))
ax2.plot(times, m1_cm, color=NAVY, lw=0.9)
ax2.plot(times, A_des @ fit1, color=RED, lw=1.5, ls="--")
ax2.set_ylabel("$\\chi_1$ (cm)", fontsize=FS_LABEL)
ax2.set_title("Excitation components with fitted secular drift", fontsize=FS_TITLE)
ax2.tick_params(labelbottom=False)
despine(ax2)
ax3 = fig.add_axes((0.565, 0.135, 0.40, 0.335))
ax3.plot(times, m2_cm, color=GREEN, lw=0.9)
ax3.plot(times, A_des @ fit2, color=RED, lw=1.5, ls="--")
ax3.set_ylabel("$\\chi_2$ (cm)", fontsize=FS_LABEL)
ax3.set_xlabel("Year", fontsize=FS_LABEL)
despine(ax3)
save(fig, "fig13_2_pole_path_simulation")


# ---- FIG 3: excitation sources ---------------------------------------------
loss_sig = np.nan_to_num(np.where(mask_signet, (-tr / 100.0) * area_m2 * RHO_W, 0.0), nan=0.0)
contrib = (A_EARTH ** 2) * np.abs(loss_sig) * np.hypot(kern13, kern23)
contrib_n = contrib / max(float(np.nanmax(contrib)), 1e-30)
show = contrib_n > 0
vmax = float(np.nanpercentile(contrib_n[show], 99)) if show.sum() else 1.0
fig = plt.figure(figsize=(13.0, 6.8))
if HAVE_STYLE:
    import cartopy.crs as ccrs
    axm = fs.global_map_axis(fig, (0.03, 0.335, 0.94, 0.635), projection="robinson", basins=True)
    o = np.argsort(contrib_n[show])
    scm = axm.scatter(lon[show][o], lat[show][o], c=contrib_n[show][o], cmap="inferno",
                      s=6, marker="s", linewidths=0, vmin=0, vmax=vmax,
                      transform=ccrs.PlateCarree(), zorder=5)
    cbm = fig.colorbar(scm, cax=fig.add_axes((0.345, 0.295, 0.31, 0.016)), orientation="horizontal")
    cbm.set_label("Relative contribution to pole excitation", fontsize=FS_LABEL)
    cbm.ax.tick_params(labelsize=FS_TICK)
    axm.set_title("Where the rotational signal originates", fontsize=FS_TITLE)
else:
    axm = fig.add_axes((0.07, 0.36, 0.80, 0.58))
    scm = axm.scatter(lon[show], lat[show], c=contrib_n[show], cmap="inferno",
                      s=2.5, linewidths=0, vmin=0, vmax=vmax)
    axm.set_xlim(-180, 180)
    axm.set_ylim(-60, 85)
    axm.set_xlabel("Longitude ($\\degree$E)", fontsize=FS_LABEL)
    axm.set_ylabel("Latitude ($\\degree$N)", fontsize=FS_LABEL)
    axm.set_title("Where the rotational signal originates", fontsize=FS_TITLE)
    fig.colorbar(scm, ax=axm, shrink=0.8, pad=0.02).set_label("Relative contribution",
                                                              fontsize=FS_ANNOT)
axb = fig.add_axes((0.10, 0.075, 0.80, 0.165))
bins = np.arange(-180, 181, 10)
bi = np.digitize(lon[show], bins) - 1
prof = np.array([contrib[show][bi == i].sum() for i in range(len(bins) - 1)])
prof = prof / max(prof.sum(), 1e-30) * 100.0
axb.bar(bins[:-1] + 5, prof, width=8.6, color=NAVY, edgecolor="none")
for xv, cl, lb, dy in [(SEO_DIR_DEG_E, BLACK, "Seo et al.", 0.98),
                       (d_net if d_net <= 180 else d_net - 360, RED, "This study", 0.72)]:
    axb.axvline(xv, color=cl, ls="--", lw=1.2)
    axb.text(xv + 4, prof.max() * dy, lb, fontsize=FS_ANNOT - 1, color=cl, va="top")
axb.set_xlim(-180, 180)
axb.set_xlabel("Longitude ($\\degree$E)", fontsize=FS_LABEL)
axb.set_ylabel("% of total", fontsize=FS_ANNOT)
despine(axb)
save(fig, "fig13_3_excitation_sources")


# ---- FIG 4: sea level, full period only ------------------------------------
fig = plt.figure(figsize=(13.0, 5.2))
axL = fig.add_axes((0.065, 0.20, 0.36, 0.66))
items = [("This study\n(gross depl.)", vol_results["depletion_only"]["sle_mm_yr"],
          vol_results["depletion_only"]["sle_sigma_mm_yr"], RED),
         ("This study\n(net, signif.)", vol_results["net_significant"]["sle_mm_yr"],
          vol_results["net_significant"]["sle_sigma_mm_yr"], NAVY),
         ("Seo et al. (2023)\n(depl.-derived)", SEO_SLE_MM_YR, 0.0, GREY)]
xs = np.arange(len(items))
axL.bar(xs, [i[1] for i in items], yerr=[i[2] for i in items], capsize=4,
        color=[i[3] for i in items], edgecolor="k", lw=0.6, width=0.62)
for i, it in enumerate(items):
    axL.text(i, it[1] + max(it[2], 0) + 0.02, f"{it[1]:.3f}", ha="center",
             fontsize=FS_ANNOT, weight="bold")
axL.set_xticks(xs)
axL.set_xticklabels([i[0] for i in items], fontsize=FS_ANNOT - 1)
axL.set_ylabel("Sea-level equivalent (mm yr$^{-1}$)", fontsize=FS_LABEL)
axL.set_ylim(0, max(i[1] for i in items) * 1.30)
axL.set_title(f"Sea-level equivalent, {times[0]:%Y}-{times[-1]:%Y}", fontsize=FS_TITLE)
despine(axL)
axR = fig.add_axes((0.545, 0.20, 0.33, 0.66))
gross = vol_results["depletion_only"]["magnitude_km3_yr"]
netsig = abs(vol_results["net_significant"]["volume_km3_yr"])
recharge = gross - netsig
axR.bar(0, gross, color=RED, edgecolor="k", lw=0.6, width=0.6)
axR.bar(1, recharge, bottom=netsig, color=GREEN, edgecolor="k", lw=0.6, width=0.6)
axR.bar(2, netsig, color=NAVY, edgecolor="k", lw=0.6, width=0.6)
axR.plot([0.3, 0.7], [gross, gross], color="0.4", lw=0.9, ls=":")
axR.plot([1.3, 1.7], [netsig, netsig], color="0.4", lw=0.9, ls=":")
for i, (v, yy) in enumerate([(gross, gross), (recharge, netsig + recharge), (netsig, netsig)]):
    axR.text(i, yy + gross * 0.035, f"{v:.0f}", ha="center", fontsize=FS_ANNOT, weight="bold")
axR.set_xticks(range(3))
axR.set_xticklabels(["Gross\ndepletion", "Offsetting\nrecharge", "Net\n(significant)"],
                    fontsize=FS_ANNOT - 1)
axR.set_ylabel("Volume (km$^3$ yr$^{-1}$)", fontsize=FS_LABEL)
axR.set_ylim(0, gross * 1.18)
axR.set_title("Gross-to-net decomposition", fontsize=FS_TITLE)
axR.spines[["top"]].set_visible(False)
axRt = axR.twinx()
axRt.set_ylim(np.array(axR.get_ylim()) / GT_PER_MM_SLE)
axRt.set_ylabel("Sea-level equivalent (mm yr$^{-1}$)", fontsize=FS_LABEL)
axRt.tick_params(labelsize=FS_TICK)
axRt.spines[["top"]].set_visible(False)
save(fig, "fig13_4_sea_level")


# ---- FIG 5: mask sensitivity + reported-result card ------------------------
fig = plt.figure(figsize=(13.0, 4.6))
keys = ["depletion_only", "net_significant", "net_all_valid"]
labs = ["Depletion\nonly", "Net\n(significant)", "Net\n(all valid)"]
cols = [RED, NAVY, GREEN]
axa = fig.add_axes((0.055, 0.20, 0.24, 0.66))
axa.bar(range(3), [pole_results[k]["rate_cm_yr"] for k in keys], color=cols,
        edgecolor="k", lw=0.6, width=0.62)
axa.axhline(SEO_RATE_CM_YR, color=BLACK, ls="--", lw=1.2)
axa.text(2.45, SEO_RATE_CM_YR, " Seo", fontsize=FS_ANNOT, va="bottom", ha="right")
axa.set_xticks(range(3))
axa.set_xticklabels(labs, fontsize=FS_ANNOT - 1)
axa.set_ylabel("Drift rate (cm yr$^{-1}$)", fontsize=FS_LABEL)
axa.set_title("Magnitude (not comparable)", fontsize=FS_TITLE)
despine(axa)
axb2 = fig.add_axes((0.375, 0.20, 0.24, 0.66))
dirs = [pole_results[k]["direction_deg_E"] for k in keys]
axb2.bar(range(3), dirs, color=cols, edgecolor="k", lw=0.6, width=0.62)
axb2.axhline(SEO_DIR_DEG_E, color=BLACK, ls="--", lw=1.2)
axb2.text(2.45, SEO_DIR_DEG_E, " Seo", fontsize=FS_ANNOT, va="bottom", ha="right")
axb2.set_ylim(0, max(max(dirs), SEO_DIR_DEG_E) * 1.28)
axb2.set_xticks(range(3))
axb2.set_xticklabels(labs, fontsize=FS_ANNOT - 1)
axb2.set_ylabel("Azimuth ($\\degree$E)", fontsize=FS_LABEL)
axb2.set_title(f"Direction (spread {mask_spread:.1f}$\\degree$)", fontsize=FS_TITLE)
despine(axb2)
axc = fig.add_axes((0.685, 0.06, 0.30, 0.88))
axc.axis("off")
axc.text(0.0, 1.00, f"Reported result  {times[0]:%Y}-{times[-1]:%Y}",
         fontsize=FS_TITLE + 1, weight="bold", va="top")
card = [("Gross depletion", f"{_vd['magnitude_km3_yr']:.0f} km$^3$ yr$^{{-1}}$"),
        ("  sea-level equiv.", f"{_vd['sle_mm_yr']:.3f} mm yr$^{{-1}}$"),
        ("Net (significant)", f"{abs(_vn['volume_km3_yr']):.0f} km$^3$ yr$^{{-1}}$"),
        ("  sea-level equiv.", f"{_vn['sle_mm_yr']:.3f} mm yr$^{{-1}}$"),
        ("", ""),
        ("Pole drift", f"{_rn['rate_cm_yr']:.2f} cm yr$^{{-1}}$"),
        ("Azimuth", f"{_rn['direction_deg_E']:.1f}$\\degree$E"),
        ("Seo separation", f"{_rn['sep_from_seo_deg']:.1f}$\\degree$"),
        ("Mask spread", f"{mask_spread:.1f}$\\degree$"),
        ("Path-vs-secular", f"{angular_sep(path_dir, _rn['direction_deg_E']):.2f}$\\degree$")]
if seda is not None:
    card += [("", ""), ("SeDA cell trend $r$*", f"{seda['cellwise_trend_corr']:+.3f}"),
             ("SeDA annual storage $r$*", f"{seda['annual_storage_corr']:+.3f}"),
             ("* matched window", "")]
for i, (k, v) in enumerate(card):
    y = 0.88 - i * 0.068
    axc.text(0.0, y, k, fontsize=FS_ANNOT, va="top", color="0.25")
    axc.text(1.0, y, v, fontsize=FS_ANNOT, va="top", ha="right", weight="bold")
save(fig, "fig13_5_mask_sensitivity")


# ---- FIG 6: GRACE-SeDA annual comparison -----------------------------------
if seda is not None:
    ann = seda["_ann"]
    tr_mine, tr_seda, mk = seda["_arrays"]
    fig = plt.figure(figsize=(13.0, 8.2))
    fig.suptitle(f"GRACE-SeDA cross-check, {seda['overlap_start']} to "
                 f"{seda['overlap_end']} (diagnostic; the reported estimate covers "
                 f"{times[0]:%Y}-{times[-1]:%Y})", fontsize=FS_TITLE + 1, y=0.985)

    # (a) annual total storage over depleting cells
    axa = fig.add_axes((0.065, 0.575, 0.40, 0.33))
    axa.plot(ann["year"], ann["storage_mine_km3"], "o-", color=NAVY, lw=1.8, ms=5,
             label="This study")
    axa.plot(ann["year"], ann["storage_seda_km3"], "s--", color=GREEN, lw=1.8, ms=5,
             label=SEDA["LABEL"])
    axa.axhline(0, color="k", lw=0.6)
    axa.set_xlabel("Year", fontsize=FS_LABEL)
    axa.set_ylabel("Storage anomaly (km$^3$)", fontsize=FS_LABEL)
    axa.set_title(f"Annual groundwater storage over depleting cells\n"
                  f"(centred; $r$ = {seda['annual_storage_corr']:+.3f})", fontsize=FS_TITLE)
    axa.legend(frameon=False, fontsize=FS_ANNOT)
    despine(axa)

    # (b) annual depletion rate
    axb3 = fig.add_axes((0.565, 0.575, 0.40, 0.33))
    dd = ann.dropna(subset=["depletion_rate_mine_km3_yr"])
    w = 0.38
    xx = np.arange(len(dd))
    axb3.bar(xx - w / 2, dd["depletion_rate_mine_km3_yr"], w, color=NAVY, label="This study")
    axb3.bar(xx + w / 2, dd["depletion_rate_seda_km3_yr"], w, color=GREEN, label=SEDA["LABEL"])
    axb3.axhline(0, color="k", lw=0.7)
    axb3.set_xticks(xx[::2])
    axb3.set_xticklabels(dd["year"].astype(int).values[::2], rotation=45, fontsize=FS_TICK)
    axb3.set_xlabel("Year", fontsize=FS_LABEL)
    axb3.set_ylabel("Depletion rate (km$^3$ yr$^{-1}$)", fontsize=FS_LABEL)
    axb3.set_title("Annual depletion rate (positive = storage lost)", fontsize=FS_TITLE)
    axb3.legend(frameon=False, fontsize=FS_ANNOT, ncol=2)
    despine(axb3)

    # (c) cell-wise trend agreement
    axc2 = fig.add_axes((0.065, 0.085, 0.36, 0.35))
    lim = float(np.nanpercentile(np.abs(np.r_[tr_mine[mk], tr_seda[mk]]), 99))
    axc2.hexbin(tr_seda[mk], tr_mine[mk], gridsize=55, cmap="magma", mincnt=1,
                extent=[-lim, lim, -lim, lim])
    axc2.plot([-lim, lim], [-lim, lim], "w--", lw=1.2)
    axc2.set_xlabel(f"{SEDA['LABEL']} trend (cm yr$^{{-1}}$)", fontsize=FS_LABEL)
    axc2.set_ylabel("This study trend (cm yr$^{-1}$)", fontsize=FS_LABEL)
    axc2.set_title(f"Cell-wise trends: $r$ = {seda['cellwise_trend_corr']:+.3f}, "
                   f"RMSD = {seda['cellwise_trend_rmsd_cm_yr']:.3f}", fontsize=FS_TITLE)

    # (d) summary numbers
    axd = fig.add_axes((0.52, 0.085, 0.45, 0.35))
    axd.axis("off")
    axd.text(0.0, 1.0, "Cross-check summary", fontsize=FS_TITLE, weight="bold", va="top")
    lines = [
        ("Overlap", f"{seda['overlap_start']} to {seda['overlap_end']} "
                    f"({seda['overlap_months']} mo)"),
        ("Mean annual depletion, this study", f"{seda['mean_annual_depletion_mine_km3_yr']:.1f} km$^3$ yr$^{{-1}}$"),
        ("Mean annual depletion, SeDA", f"{seda['mean_annual_depletion_seda_km3_yr']:.1f} km$^3$ yr$^{{-1}}$"),
        ("Storage slope, this study", f"{seda['storage_slope_mine_km3_yr']:.1f} km$^3$ yr$^{{-1}}$"),
        ("Storage slope, SeDA", f"{seda['storage_slope_seda_km3_yr']:.1f} km$^3$ yr$^{{-1}}$"),
        ("Annual storage correlation", f"{seda['annual_storage_corr']:+.3f}"),
        ("Cell-wise trend correlation", f"{seda['cellwise_trend_corr']:+.3f}"),
        ("Cell-wise sign agreement", f"{seda['cellwise_sign_agreement_pct']:.1f}%"),
        ("Azimuth, this study (window)", f"{seda['window_azimuth_mine_deg_E']:.1f}$\\degree$E"),
        ("Azimuth, SeDA (window)", f"{seda['window_azimuth_seda_deg_E']:.1f}$\\degree$E"),
    ]
    for i, (k, v) in enumerate(lines):
        yy = 0.88 - i * 0.093
        axd.text(0.0, yy, k, fontsize=FS_ANNOT, va="top", color="0.25")
        axd.text(1.0, yy, v, fontsize=FS_ANNOT, va="top", ha="right", weight="bold")
    save(fig, "fig13_6_seda_annual_comparison")

    seda.pop("_ann", None)
    seda.pop("_arrays", None)


# =============================================================== STEP 6 ======
hdr("STEP 6 -- write summary")
summary = {
    "final_estimate_full_period": final_estimate,
    "volumes_and_sea_level": vol_results,
    "secular_pole_drift": pole_results,
    "simulated_pole_path": {
        "mask": "net_significant", "drift_rate_cm_yr": round(path_rate, 4),
        "direction_deg_E": round(path_dir, 2),
        "rate_agreement_pct": round(100 * abs(path_rate - sec["rate_cm_yr"]) / sec["rate_cm_yr"], 3),
        "azimuth_agreement_deg": round(angular_sep(path_dir, sec["direction_deg_E"]), 3),
        "seasonal_sigma_chi1_cm": round(float(np.std(m1_cm)), 3),
        "seasonal_sigma_chi2_cm": round(float(np.std(m2_cm)), 3),
        "n_months": int(len(times))},
    "mask_robustness": {"direction_spread_deg": round(mask_spread, 2)},
    "seda_cross_check": seda,
    "reference_seo_et_al_2023": {
        "rate_cm_yr": SEO_RATE_CM_YR, "direction_deg_E": SEO_DIR_DEG_E,
        "implied_km3_yr": round(SEO_KM3_YR, 1), "implied_sle_mm_yr": round(SEO_SLE_MM_YR, 4),
        "citation": "Seo et al. (2023), GRL, doi:10.1029/2023GL103509"},
    "seda_reference": {
        "citation": "Gou & Soja (2024), Nature Water 2, 139-150, "
                    "doi:10.1038/s44221-024-00194-w",
        "note": "Downscaled from GRACE + WGHM. This study's OSA is subtracted to "
                "obtain a comparable groundwater anomaly, so the comparison "
                "isolates the TWSA component and is not fully independent. Both "
                "series centred on the overlap; only trends and year-to-year "
                "changes are compared."},
    "constants": {"a_earth_m": A_EARTH, "C_minus_A_kg_m2": C_MINUS_A, "k2_prime": K2_PRIME,
                  "rotational_deformation_factor": ROT_DEFORM,
                  "combined_excitation_factor": round(EXC_FACTOR, 4),
                  "cell_deg": CELL_DEG, "gt_per_mm_sle": GT_PER_MM_SLE},
    "grid_check": {"valid_cells": int(valid.sum()),
                   "implied_land_area_Mkm2": round(implied_land_Mkm2, 1)},
    "caveats": ("Groundwater layer only: no ice sheets, glaciers, GIA or the "
                "compensating ocean load, so drift MAGNITUDE is not comparable to "
                "Seo et al.; direction is the meaningful comparison. Secular "
                "approximation, no Chandler wobble dynamics. Published depletion "
                "totals are not used as benchmarks because periods, definitions "
                "and significance treatment differ; sea-level equivalent is the "
                "only cross-study quantity. Formal uncertainty propagates "
                "per-cell trend error only."),
}
with open(OUT_DIR / "polar_motion_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)
with open(OUT_DIR / "phase13_diagnostics.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(_LOG))
_p(f"  wrote {OUT_DIR / 'polar_motion_summary.json'}")
_p(f"  wrote {OUT_DIR / 'pole_path_monthly.csv'}")
if seda is not None:
    _p(f"  wrote {OUT_DIR / 'seda_annual_comparison.csv'}")
    _p(f"  wrote {OUT_DIR / 'seda_cellwise_trends.csv'}")
_p("")
_p(LINE)
_p("Final estimate: see STEP 4c and polar_motion_summary.json.")
_p("STEP 4b is a matched-window diagnostic; the reported estimate is STEP 4c.")
_p(LINE)