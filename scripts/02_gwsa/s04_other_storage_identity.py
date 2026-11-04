#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
PHASE 5.0a -- RECOVER AND PERSIST THE OTHER-STORAGE ANOMALY (OSA)
--------------------------------------------------------------------------------
WHY THIS EXISTS
  Phase 1 defined            GWSA = TWSA - (soil + SWE + canopy)_anom
  and computed the bracketed term internally as (stor - clim[mm]) -- but never
  wrote it to disk. The physics constraint in Phase 5.1 needs it, because the
  water balance governs TOTAL storage, not groundwater alone:

        d(TWSA)/dt = P - ET - R
        TWSA        = GWSA + OSA
   =>   d(GWSA_pred + OSA_obs)/dt  ~=  (NetFlux - E)/10      [cm/month]

  So the LSTM predicts GWSA, OSA is added back to reconstruct TWSA, and the
  budget is closed on TWSA. This follows the Phase 1 rule exactly and needs no
  low-frequency approximation.

CRITICAL
  OSA must be computed with BIT-IDENTICAL convention to Phase 1, or the identity
  GWSA + OSA = TWSA will not hold. This script therefore reuses Phase 1's exact
  code path: same six GLDAS variables, same block_mean_2x2 aggregation, same
  2004-2009 climatology, same mm->cm conversion, same target month list.

VERIFICATION GATE (the whole point)
  Reconstructs TWSA = GWSA + OSA and compares it against the ACTUAL GRACE
  lwe_thickness * scale_factor. If the convention is right this matches to
  float32 precision on every uncapped cell-month. If it does not match, STOP --
  do not proceed to Phase 5.1.

  Note: Phase 1 applied the Rodell +/-100 cm cap AFTER differencing, so the
  identity is broken wherever |GWSA| hit the cap. Those cell-months are excluded
  from the gate and reported separately.

OUTPUT
  OUTPUTS\05_model_tensor\other_storage_anomaly.nc
  OUTPUTS\05_model_tensor\phase5_0a_diagnostics.txt

RUN: CPU only. Memory-safe (incremental climatology, float32, lazy GRACE reads).
================================================================================
"""

import os, glob, re, warnings, traceback, datetime as dt
import numpy as np

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==============================================================================
# CONFIG -- must match Phase 1 exactly
# ==============================================================================
DATA      = r"D:\THESIS\Arpon_Thesis_Final\DATA"
OUT_ROOT  = r"D:\THESIS\Arpon_Thesis_Final\OUTPUTS"
OUT_DIR   = os.path.join(OUT_ROOT, "05_model_tensor")

GRACE_FILE = os.path.join(DATA, "grace_jpl_mascon",
                          "GRCTellus.JPL.200204_202512.GLO.RL06.3M.MSCNv04CRI.nc")
GLDAS_DIR  = os.path.join(DATA, "gldas_noah_025")
PHASE1_NC  = os.path.join(OUT_ROOT, "01_gwsa_monthly", "gwsa_monthly.nc")

BASE_START, BASE_END = 2004, 2009           # identical to Phase 1
ANA_START, ANA_END   = (2002, 4), (2022, 12)
GRID   = 0.5
CAP_CM = 100.0

SOIL_VARS = ["SoilMoi0_10cm_inst", "SoilMoi10_40cm_inst",
             "SoilMoi40_100cm_inst", "SoilMoi100_200cm_inst"]
SWE_VAR, CAN_VAR = "SWE_inst", "CanopInt_inst"

LON = np.arange(-180 + GRID / 2, 180, GRID)
LAT = np.arange(-90 + GRID / 2, 90, GRID)

LINE = "=" * 78
_LOG = []
def _p(s=""):
    print(s); _LOG.append(str(s))
def hdr(t):
    _p(""); _p(LINE); _p(t); _p(LINE)
def ok(t):   _p("  [ OK ]   " + t)
def warn(t): _p("  [WARN]   " + t)
def fail(t): _p("  [FAIL]   " + t)


# --- Phase 1 helpers, copied verbatim so the convention cannot drift ----------
def year_of(p):
    m = re.search(r"A(\d{4})\d{2}", os.path.basename(p))
    return int(m.group(1)) if m else None

def month_of(p):
    m = re.search(r"A(\d{4})(\d{2})", os.path.basename(p))
    return int(m.group(2)) if m else None

def block_mean_2x2(a):
    ny, nx = a.shape
    return np.nanmean(a.reshape(ny // 2, 2, nx // 2, 2), axis=(1, 3))

def gldas_storage_month(xr, f):
    """Total storage (cm, float32) on 0.5deg 360x720 -- IDENTICAL to Phase 1."""
    d = xr.open_dataset(f)
    comp = np.zeros((600, 1440), "float32")
    for v in SOIL_VARS + [SWE_VAR, CAN_VAR]:
        comp += np.asarray(d[v].squeeze().values, "float32")
    d.close()
    half = block_mean_2x2(comp).astype("float32")     # (300,720) lat -60..90
    full = np.full((360, 720), np.nan, "float32")
    full[60:360, :] = half
    return full / 10.0                                 # mm -> cm


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    _p(LINE)
    _p("PHASE 5.0a -- RECOVER OTHER-STORAGE ANOMALY (OSA)")
    _p(f"run: {dt.datetime.now().isoformat(timespec='seconds')}")
    _p(LINE)

    import xarray as xr

    hdr("STEP 1 -- inputs")
    for path, what in [(GRACE_FILE, "GRACE mascon"),
                       (PHASE1_NC, "Phase 1 gwsa_monthly.nc")]:
        if not os.path.exists(path):
            fail(f"{what} not found:\n           {path}")
            return
        ok(f"{what}: {os.path.basename(path)}")
    files = [f for f in sorted(glob.glob(os.path.join(GLDAS_DIR, "**", "*.nc4"),
                                         recursive=True)) if year_of(f)]
    if not files:
        fail(f"no GLDAS .nc4 files under {GLDAS_DIR}")
        return
    by_ym = {(year_of(f), month_of(f)): f for f in files}
    ok(f"GLDAS months available: {len(by_ym)}")

    # ------------------------------------------------------------------ STEP 2
    hdr("STEP 2 -- GLDAS 2004-2009 climatology (incremental, identical to Phase 1)")
    clim = {}
    for mo in range(1, 13):
        acc = np.zeros((360, 720), "float32")
        cnt = np.zeros((360, 720), "float32")
        for y in range(BASE_START, BASE_END + 1):
            f = by_ym.get((y, mo))
            if not f:
                continue
            s = gldas_storage_month(xr, f)
            m = np.isfinite(s)
            acc[m] += s[m]; cnt[m] += 1
        clim[mo] = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan).astype("float32")
        _p(f"     month {mo:02d}: {int(np.isfinite(clim[mo]).sum())} valid cells")
    ok("climatology rebuilt")

    # ------------------------------------------------------------------ STEP 3
    hdr("STEP 3 -- monthly OSA = stor - clim  (cm)")
    target = []
    y, mo = ANA_START
    while (y, mo) <= ANA_END:
        target.append((y, mo))
        mo += 1
        if mo > 12:
            mo = 1; y += 1
    T = len(target)
    _p(f"  target months: {T}")

    OSA = np.full((T, 360, 720), np.nan, "float32")
    n_have = 0
    for k, (yy, mm) in enumerate(target):
        f = by_ym.get((yy, mm))
        if f is None:
            continue
        OSA[k] = gldas_storage_month(xr, f) - clim[mm]
        n_have += 1
    ok(f"OSA months populated: {n_have}/{T}")
    fin = np.isfinite(OSA)
    _p(f"  OSA finite fraction: {100*fin.mean():.2f}%")
    _p(f"  OSA range: {np.nanmin(OSA):+.3f} .. {np.nanmax(OSA):+.3f} cm  "
       f"(mean {np.nanmean(OSA):+.4f})")

    # ------------------------------------------------------------------ STEP 4
    hdr("STEP 4 -- rebuild GRACE TWSA (same alignment as Phase 1)")
    ds = xr.open_dataset(GRACE_FILE)
    sf = np.asarray(ds["scale_factor"].values, "float32")
    glon = ds["lon"].values
    order = None
    if glon.max() > 180:
        new_lon = np.where(glon > 180, glon - 360, glon)
        order = np.argsort(new_lon)
        sf = sf[:, order]
        ok("GRACE lon rolled to -180..180")
    glat = ds["lat"].values
    flip_lat = glat[0] > glat[-1]
    if flip_lat:
        sf = sf[::-1, :]
        ok("GRACE lat flipped to ascending")
    times = ds["time"].values
    grace_ym = [(int(str(t)[:4]), int(str(t)[5:7])) for t in times]
    lut = {ym: i for i, ym in enumerate(grace_ym)}

    TWSA = np.full((T, 360, 720), np.nan, "float32")
    for k, ym in enumerate(target):
        gi = lut.get(ym)
        if gi is None:
            continue
        a = np.asarray(ds["lwe_thickness"].isel(time=gi).values, "float32")
        if order is not None:
            a = a[:, order]
        if flip_lat:
            a = a[::-1, :]
        TWSA[k] = a * sf
    ds.close()
    ok(f"TWSA months populated: {int(np.isfinite(TWSA).any(axis=(1,2)).sum())}/{T}")

    # ------------------------------------------------------------------ STEP 5
    hdr("STEP 5 -- VERIFICATION GATE:  GWSA + OSA  ==  TWSA ?")
    p1 = xr.open_dataset(PHASE1_NC); p1.load(); p1.close()
    GWSA = np.asarray(p1["gwsa"].values, "float32")
    if GWSA.shape != OSA.shape:
        fail(f"shape mismatch: Phase1 gwsa {GWSA.shape} vs OSA {OSA.shape}")
        return
    domain = np.asarray(p1["domain"].values).astype(bool)

    recon = GWSA + OSA
    capped = np.abs(GWSA) >= (CAP_CM - 1e-3)          # identity broken by the cap
    m = (np.isfinite(recon) & np.isfinite(TWSA)
         & domain[None, :, :] & ~capped)
    n_cmp = int(m.sum())
    _p(f"  comparable cell-months: {n_cmp:,}")
    _p(f"  excluded (Rodell cap)  : {int((capped & domain[None,:,:]).sum()):,}")

    if n_cmp == 0:
        fail("nothing comparable -- check that Phase 1 and this run used the "
             "same GLDAS files and baseline.")
        return

    d = recon[m] - TWSA[m]
    amax = float(np.max(np.abs(d)))
    rmse = float(np.sqrt(np.mean(d ** 2)))
    _p(f"  max |GWSA + OSA - TWSA| : {amax:.6e} cm")
    _p(f"  RMSE                    : {rmse:.6e} cm")
    _p(f"  99.9th pct |residual|   : {float(np.percentile(np.abs(d), 99.9)):.6e} cm")

    TOL = 1e-3          # cm; float32 accumulation over 6 summed GLDAS layers
    if amax < TOL:
        ok(f"IDENTITY HOLDS (max residual < {TOL} cm). Convention is bit-consistent "
           "with Phase 1. Safe to proceed.")
        gate = True
    else:
        fail(f"IDENTITY BROKEN (max residual {amax:.4e} cm >= {TOL}).")
        _p("     DO NOT PROCEED. Likely causes, in order of probability:")
        _p("       1. GLDAS file set differs from the Phase 1 run "
           "(months added/removed since).")
        _p("       2. Baseline years or variable list changed.")
        _p("       3. Phase 1 gwsa_monthly.nc was regenerated with different settings.")
        gate = False

    # ------------------------------------------------------------------ STEP 6
    hdr("STEP 6 -- closure preview: d(TWSA)/dt vs (NetFlux - E)")
    _p("  Full diagnostic runs in Phase 5.0 STEP 8 on the cell grid. This is a")
    _p("  coarse global check that the reconstructed TWSA tendency behaves.")
    tw = np.where(domain[None, :, :], TWSA, np.nan)
    gm = np.array([np.nanmean(tw[k]) for k in range(T)], "float32")
    dgm = np.full(T, np.nan, "float32")
    dgm[1:-1] = (gm[2:] - gm[:-2]) / 2.0
    fin_d = np.isfinite(dgm)
    _p(f"  global-mean TWSA tendency: {np.nanmin(dgm):+.4f} .. {np.nanmax(dgm):+.4f} cm/month")
    _p(f"  months with finite tendency: {int(fin_d.sum())}/{T}")

    # ------------------------------------------------------------------ STEP 7
    hdr("STEP 7 -- write")
    tc = [dt.datetime(a, b, 15) for (a, b) in target]
    out = xr.Dataset(
        {"osa": (("time", "lat", "lon"), OSA),
         "twsa_grace": (("time", "lat", "lon"), TWSA),
         "identity_ok": ((), np.int8(1 if gate else 0))},
        coords={"time": tc, "lat": LAT, "lon": LON},
        attrs={
            "title": "Other-storage anomaly (soil + SWE + canopy), Phase 5.0a",
            "units": "cm",
            "definition": "OSA = GLDAS(soil4 + SWE + canopy) - 2004-2009 climatology",
            "identity": "GWSA + OSA = TWSA (exact, except where Rodell cap applied)",
            "purpose": ("physics constraint for Phase 5.1: "
                        "d(GWSA_pred + OSA)/dt ~= (NetFlux - E)/10 cm/month"),
            "baseline": f"{BASE_START}-{BASE_END}",
            "convention": "identical to Phase 1 (block_mean_2x2, mm->cm /10)",
            "max_identity_residual_cm": amax,
            "identity_rmse_cm": rmse,
            "created": dt.datetime.now().isoformat(timespec="seconds")})
    enc = {v: {"zlib": True, "complevel": 4}
           for v in ("osa", "twsa_grace")}
    p_out = os.path.join(OUT_DIR, "other_storage_anomaly.nc")
    out.to_netcdf(p_out, encoding=enc)
    ok(f"written: {p_out}  ({os.path.getsize(p_out)/1e6:.0f} MB)")

    try:
        make_figures(OSA, GWSA, TWSA, domain, d, target)
    except Exception as e:
        warn(f"figures failed: {e}")

    with open(os.path.join(OUT_DIR, "phase5_0a_diagnostics.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(_LOG))

    _p("")
    _p(LINE)
    if gate:
        _p("Identity check passed: OSA is consistent with Phase 1.")
    else:
        _p("Identity check failed.")
    _p(LINE)


def make_figures(OSA, GWSA, TWSA, domain, resid, target):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    rcParams["font.family"] = "serif"
    rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    rcParams.update({"font.size": 13, "axes.titlesize": 15, "axes.labelsize": 14})

    T = OSA.shape[0]
    x = np.arange(T)
    dm = domain[None, :, :]
    g_osa = np.array([np.nanmean(np.where(domain, OSA[k], np.nan)) for k in range(T)])
    g_gw  = np.array([np.nanmean(np.where(domain, GWSA[k], np.nan)) for k in range(T)])
    g_tw  = np.array([np.nanmean(np.where(domain, TWSA[k], np.nan)) for k in range(T)])

    f, ax = plt.subplots(figsize=(13, 6))
    ax.plot(x, g_tw, lw=1.6, color="#1f4e79", label="TWSA (GRACE)")
    ax.plot(x, g_gw, lw=1.4, color="#c0392b", label="GWSA (Phase 1)")
    ax.plot(x, g_osa, lw=1.4, color="#2c7fb8", label="OSA (soil + SWE + canopy)")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xticks(range(0, T, 12))
    ax.set_xticklabels([target[k][0] for k in range(0, T, 12)], rotation=45)
    ax.set_xlabel("Year"); ax.set_ylabel("Anomaly (cm)")
    ax.set_title("Storage decomposition: TWSA = GWSA + OSA (global mean)")
    ax.legend()
    f.savefig(os.path.join(OUT_DIR, "fig5a_1_decomposition.png"),
              dpi=600, bbox_inches="tight"); plt.close(f)

    f, ax = plt.subplots(figsize=(8, 5))
    ax.hist(resid, bins=80, color="#2c7fb8", edgecolor="k", linewidth=0.3)
    ax.set_xlabel("GWSA + OSA $-$ TWSA (cm)")
    ax.set_ylabel("Cell-months")
    ax.set_title("Identity residual (uncapped cells)")
    ax.set_yscale("log")
    f.savefig(os.path.join(OUT_DIR, "fig5a_2_identity_residual.png"),
              dpi=600, bbox_inches="tight"); plt.close(f)

    ok("figures written (fig5a_1, fig5a_2)")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()