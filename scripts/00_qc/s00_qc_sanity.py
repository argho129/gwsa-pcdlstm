#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE 0 - VERIFICATION GATES  (auto-discovery version)
Author: Arpon Sarkar, KUET.
  CHECK 1  PET source + sign (GLEAM Ep primary; ERA5 pev fallback) -> gates Phase 3
  CHECK 2  WHYMAP HYGEO2 coding vs legend                          -> gates Discovery 1
  CHECK 3  GROW QC-passed well count                               -> calibration strength
"""

import os, sys, glob, traceback
import numpy as np

# ------------------------------------------------------------------ ROOTS -----
WORK        = r"D:\THESIS\Arpon_Thesis_Final\DATA"
CLIMATE_DIR = r"D:\THESIS\Arpon_Thesis_Final\DATA\climate_era5"   # ERA5-Land P/PET live here
GLEAM_DIR   = r"D:\THESIS\Arpon_Thesis_Final\DATA\GLEAM_v4_Ep"    # point at Ep — you have it now
WHYMAP_DIR  = r"D:\THESIS\Arpon_Thesis_Final\DATA\WHYMAP"

# GROW QC filter (spec section 3). Do not loosen silently.
GROW_QC = {
    "min_length_years":  10,
    "max_gap_fraction":  0.10,
    "allowed_intervals": {"YS", "MS"},
    "exclude_plateaus":  True,
    "exclude_outliers":  True,
    "exclude_neg_signs": {"All", "Some"},
}

LINE = "=" * 78
def hdr(t): print("\n" + LINE + "\n" + t + "\n" + LINE)
def ok(t):  print("  [ OK ]   " + t)
def warn(t):print("  [WARN]   " + t)
def fail(t):print("  [FAIL]   " + t)
results = {}

def find_nc(root):
    out = []
    for ext in ("*.nc", "*.nc4"):
        out += glob.glob(os.path.join(root, "**", ext), recursive=True)
    return sorted(set(out))

# ============================================================ CHECK 1 ==========
# PET sign / source. Physics ground truth:
#   Potential evaporation = atmospheric water DEMAND (surface loss).
#   ERA5/ERA5-Land accumulate pev NEGATIVE (downward-positive convention).
#   GLEAM Ep is a positive demand. The (P-PET) attribution inverts if a
#   negative pev is used without negation. This check finds the file, reports
#   sign, and states the exact ACTION for Phase 3.
def check1_pet():
    hdr("CHECK 1 -- PET source + SIGN [gates Phase 3 attribution]")
    try:
        import xarray as xr
    except Exception:
        fail("xarray missing. pip install xarray netcdf4"); return False

    # 1) Look for GLEAM Ep (potential evap) first - spec's PRIMARY PET choice.
    gleam = find_nc(GLEAM_DIR)
    ep_files = [f for f in gleam if os.path.basename(f).lower().startswith(("ep_", "ep"))
                and "ep" in os.path.basename(f).lower()]
    e_files  = [f for f in gleam if os.path.basename(f).lower().startswith("e_")]
    print(f"  GLEAM_v4 dir: {len(gleam)} nc files "
          f"(E_*: {len(e_files)}, Ep_*: {len(ep_files)})")
    if ep_files:
        ok(f"GLEAM Ep present ({len(ep_files)} files) -> spec-primary PET available.")
    else:
        warn("GLEAM Ep (potential evap) NOT found - only actual-E downloaded. "
             "Spec wants Ep as PRIMARY PET. Either download GLEAM Ep, or use "
             "ERA5 pev (below) as PET. FLAG - decide before Phase 3.")

    # 2) Scan Climate_data for any potential-evaporation variable.
    clim = find_nc(CLIMATE_DIR)
    print(f"  Climate_data dir: {len(clim)} nc files across subfolders.")
    if not clim:
        fail(f"no .nc under {CLIMATE_DIR}"); return False

    PET_KEYS = ("pev", "potential_evap", "pet", "ep")
    hits = []  # (file, varname, units, longname)
    for f in clim + ep_files:
        try:
            ds = xr.open_dataset(f, decode_times=False)
        except Exception:
            continue
        for vn in ds.data_vars:
            ln = str(ds[vn].attrs.get("long_name", "")).lower()
            sn = str(ds[vn].attrs.get("standard_name", "")).lower()
            if (vn.lower() in PET_KEYS
                    or "potential evapo" in ln or "potential_evapo" in sn):
                hits.append((f, vn, ds[vn].attrs.get("units", "?"),
                             ds[vn].attrs.get("long_name", "?")))
        ds.close()

    if not hits:
        warn("No potential-evaporation variable auto-found in Climate_data or GLEAM.")
        print("     Listing ALL variables per Climate_data file so we can locate PET:")
        for f in clim:
            try:
                ds = xr.open_dataset(f, decode_times=False)
                print(f"       {os.path.relpath(f, CLIMATE_DIR)} -> {list(ds.data_vars)}")
                ds.close()
            except Exception as e:
                print(f"       {f} -> unreadable ({e})")
        fail("PET source unresolved. See the variable listing above.")
        return False

    ok(f"potential-evap variable(s) found in {len(set(h[0] for h in hits))} file(s):")
    for f, vn, u, ln in hits:
        print(f"     {os.path.basename(f)} :: var='{vn}' units='{u}' long_name='{ln}'")

    # 3) Sign test on the first hit.
    f, vn, units, ln = hits[0]
    print(f"\n  --> sign test on: {os.path.basename(f)} :: '{vn}'")
    ds = xr.open_dataset(f, decode_times=False)
    arr = np.asarray(ds[vn].values, "float64").ravel()
    ds.close()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        fail("all-NaN."); return False
    p_neg = 100.0 * np.mean(arr < 0)
    absmax = np.nanmax(np.abs(arr))
    print(f"     min={np.nanmin(arr):.4g} max={np.nanmax(arr):.4g} "
          f"mean={np.nanmean(arr):.4g} |max|={absmax:.4g}")
    print(f"     fraction < 0 : {p_neg:.1f}%")

    verdict = True
    if p_neg > 80:
        ok("PET stored NEGATIVE (ERA5 convention).  ACTION: PET = -value")
        results["_pet_sign"] = "NEGATIVE_negate"
    elif p_neg < 20:
        ok("PET stored POSITIVE (demand).  ACTION: PET = value  (as-is)")
        results["_pet_sign"] = "POSITIVE_asis"
    else:
        fail(f"AMBIGUOUS sign ({p_neg:.1f}% neg) - mixed sign is a red flag.")
        results["_pet_sign"] = "AMBIGUOUS"; verdict = False

    if absmax < 1e-2:
        warn(f"|max|={absmax:.4g} tiny -> metres accumulation. Convert to mm "
             "(x1000, x days-in-month) before Phase 3.")
    elif absmax > 5e3:
        warn(f"|max|={absmax:.4g} large -> check units (J m-2? seconds accum?).")
    else:
        ok(f"|max|={absmax:.4g} plausible mm-scale.")
    return verdict

# ============================================================ CHECK 2 ==========
def check2_whymap():
    hdr("CHECK 2 -- WHYMAP HYGEO2 digit coding [gates Discovery 1 + pooling]")
    try:
        import geopandas as gpd
    except Exception:
        fail("geopandas missing. pip install geopandas"); return False

    shps = glob.glob(os.path.join(WHYMAP_DIR, "**", "*.shp"), recursive=True)
    print(f"  shapefiles under {WHYMAP_DIR}: {[os.path.basename(s) for s in shps]}")
    if not shps:
        fail("no .shp found. Was WHYMAP extracted here?"); return False
    poly = [s for s in shps if "poly" in s.lower()] or shps
    shp = poly[0]
    ok(f"using: {os.path.basename(shp)}")

    gdf = gpd.read_file(shp)
    print(f"  polygons: {len(gdf)}   CRS: {gdf.crs}")
    print(f"  columns: {list(gdf.columns)}")

    col = "HYGEO2" if "HYGEO2" in gdf.columns else None
    if col is None:
        cands = [c for c in gdf.columns if "hygeo" in c.lower()]
        if cands: col = cands[0]
    if col is None:
        fail("HYGEO2 column absent.")
        print("     candidate cols:", [c for c in gdf.columns if c != gdf.geometry.name])
        return False
    ok(f"class column: '{col}'  dtype={gdf[col].dtype}")

    vc = gdf[col].value_counts(dropna=False).sort_index()
    print("  distinct value counts:")
    for k, n in vc.items():
        print(f"     {repr(k):>14} : {n}")

    label_cols = [c for c in gdf.columns if c != col and gdf[c].dtype == object
                  and any(s in c.lower() for s in
                          ("name","desc","legend","class","type","aqui","hygeo"))]
    if label_cols:
        lc = label_cols[0]
        ok(f"label column '{lc}' -> digit<->label cross-tab:")
        try:
            xt = gdf.groupby([col, lc]).size().reset_index(name="n")
            for _, r in xt.iterrows():
                print(f"     {repr(r[col]):>14}  <->  {str(r[lc])[:46]:<46} ({r['n']})")
        except Exception as e:
            warn(f"cross-tab failed: {e}")
    else:
        warn("no text label column - confirm digit->class vs bundled legend PDF manually.")

    try:
        lead = gdf[col].dropna().astype(int).astype(str).str[0]
        leads = sorted(lead.unique())
        print(f"  leading digits present: {leads}")
        unexpected = set(leads) - {"1","2","3","8"}
        if unexpected:
            warn(f"leading digits {sorted(unexpected)} NOT in spec map {{1,2,3,8}}.")
        else:
            ok("all leading digits within spec map {1,2,3,8}.")
    except Exception as e:
        warn(f"non-integer codes, leading-digit extraction skipped: {e}")

    print("\n  NOTE: structural read OK. SEMANTIC digit->class confirm vs legend is")
    print("        manual step - check the value counts against the WHYMAP legend.")
    return True

# ============================================================ CHECK 3 ==========
def _truthy(s):
    import pandas as pd
    if s.dtype == bool: return s
    return s.astype(str).str.strip().str.lower().isin(["true","1","1.0","yes","t"])

def check3_grow():
    hdr("CHECK 3 -- GROW QC-passed well COUNT [calibration strength]")
    try:
        import pandas as pd
    except Exception:
        fail("pandas missing."); return False

    pqs = glob.glob(os.path.join(WORK, "**", "*.parquet"), recursive=True)
    attr = [p for p in pqs if any(k in os.path.basename(p).lower()
            for k in ("attr","grow","well","meta"))]
    print(f"  parquet files under {WORK}: {len(pqs)}")
    for p in pqs:
        print(f"     {os.path.relpath(p, WORK)}")
    if not pqs:
        fail("no .parquet found. Check the GROW path in the configuration.")
        return False
    target = (attr or pqs)[0]
    ok(f"using: {os.path.relpath(target, WORK)}")

    try:
        df = pd.read_parquet(target)
    except Exception as e:
        fail(f"read failed: {e}"); return False
    print(f"  rows: {len(df)}   cols ({len(df.columns)}): {list(df.columns)}")

    def find(cands):
        low = {c.lower(): c for c in df.columns}
        for c in cands:
            if c in df.columns: return c
            if c.lower() in low: return low[c.lower()]
        return None

    c_len  = find(["length_years"])
    c_gap  = find(["gap_fraction"])
    c_int  = find(["interval"])
    c_plat = find(["plateaus"])
    c_out  = find(["outliers_change_points"])
    c_neg  = find(["negative_signs_wtd"])
    c_slp  = find(["trend_slope_m_year-1","trend_slope_m_year_1","trend_slope"])
    c_dir  = find(["trend_direction"])
    for nm, c in [("length_years",c_len),("gap_fraction",c_gap),("interval",c_int),
                  ("plateaus",c_plat),("outliers_change_points",c_out),
                  ("negative_signs_wtd",c_neg),("trend_slope",c_slp),
                  ("trend_direction",c_dir)]:
        (ok if c else warn)(f"{nm:<24} -> {c if c else 'NOT FOUND'}")

    mask = np.ones(len(df), dtype=bool)
    def step(m, label):
        nonlocal mask
        before = mask.sum()
        mask = mask & (m.values if hasattr(m, "values") else m)
        print(f"     {label:<40} kept {mask.sum():>7} / {before:>7} (-{before-mask.sum()})")

    print("\n  applying calibration filter:")
    if c_len  is not None: step(df[c_len] >= GROW_QC["min_length_years"], f"length_years >= {GROW_QC['min_length_years']}")
    if c_gap  is not None: step(df[c_gap] <= GROW_QC["max_gap_fraction"], f"gap_fraction <= {GROW_QC['max_gap_fraction']}")
    if c_int  is not None: step(df[c_int].astype(str).isin(GROW_QC["allowed_intervals"]), f"interval in {GROW_QC['allowed_intervals']}")
    if c_plat is not None: step(~_truthy(df[c_plat]), "plateaus == False")
    if c_out  is not None: step(~_truthy(df[c_out]),  "outliers_change_points == False")
    if c_neg  is not None: step(~df[c_neg].astype(str).isin(GROW_QC["exclude_neg_signs"]), f"negative_signs_wtd not in {GROW_QC['exclude_neg_signs']}")

    n_qc = int(mask.sum())
    print(f"\n  QC-PASSED WELLS: {n_qc}  ({100.0*n_qc/max(len(df),1):.1f}% of {len(df)})")
    if c_slp is not None:
        print(f"  of QC-passed, non-null trend_slope: {int(df.loc[mask, c_slp].notna().sum())}")
    if c_dir is not None:
        print("  trend_direction (QC-passed):")
        print("     " + df.loc[mask, c_dir].astype(str).value_counts().to_string().replace("\n","\n     "))

    if   n_qc >= 20000: ok(f"calibration: STRONG branch plausible ({n_qc}).")
    elif n_qc >= 5000:  warn(f"calibration: MODERATE ({n_qc}); LORO-CV decides later.")
    else:               warn(f"calibration: THIN ({n_qc}); lean conservative, harden Tier-2.")
    results["_grow_qc_count"] = n_qc
    return n_qc > 0

# ============================================================ MAIN =============
def main():
    print(LINE); print("PHASE 0 VERIFICATION")
    print("python:", sys.version.split()[0]); print(LINE)
    checks = [("CHECK1_pet", check1_pet), ("CHECK2_whymap", check2_whymap),
              ("CHECK3_grow", check3_grow)]
    for name, fn in checks:
        try: results[name] = bool(fn())
        except Exception:
            fail(f"{name} crashed:"); traceback.print_exc(); results[name] = False
    hdr("PHASE 0 SUMMARY"); allpass = True
    for name, _ in checks:
        p = results.get(name, False); allpass &= p
        print(f"  {name:<16}: {'PASS' if p else 'FAIL'}")
    if "_pet_sign" in results:      print(f"  PET sign         : {results['_pet_sign']}")
    if "_grow_qc_count" in results: print(f"  GROW QC count    : {results['_grow_qc_count']}")
    print("\n" + LINE)
    print("All checks passed."
          if allpass else
          "One or more checks failed.")
    print(LINE)

if __name__ == "__main__":
    main()