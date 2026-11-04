#!/usr/bin/env python3
"""
make_repo_layout.py  --  copy the analysis scripts and run logs from your working
folder into this repository's layout. File CONTENTS are copied byte-for-byte;
only file names and locations change.

Usage (run from inside the repository folder):
    python tools/make_repo_layout.py --src "D:\\THESIS\\Arpon_Thesis_Final\\Codes"
    python tools/make_repo_layout.py --src "...\\Codes" --logs "...\\where\\the\\txt\\logs\\are"
    add --dry-run to only report what would happen.

It writes logs/ORIGINAL_NAMES.csv (original path -> repo path, SHA-256) and prints:
  - scripts it could not find (upload them or write them before release)
  - ambiguous matches (several files with the same name; the newest is used)
  - a scan for credentials and for hard-coded local paths
"""
import argparse, csv, hashlib, os, re, shutil, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# original file name -> path inside the repository
SCRIPTS = {
    "0.0_QC_Sanity.py": "scripts/00_qc/s00_qc_sanity.py",
    "0.5_Build_Reliability_Mask.py": "scripts/01_masks/s01_reliability_mask.py",
    "0.5b -- PERSIST SURFACE-WATER FLAG INTO THE RELIABILITY MASK.py":
        "scripts/01_masks/s02_surface_water_flag.py",
    "1.1 phase1 gwsa isolation.py": "scripts/02_gwsa/s03_gwsa_isolation.py",
    "05a_TWSA_Physics.py": "scripts/02_gwsa/s04_other_storage_identity.py",
    "2.0_Inspect_Sy_and_GROW.py": "scripts/03_wells/s05_inspect_sy_grow.py",
    "2.2 GRACE CELL TRENDS + WELL-TO-CELL PAIRING.py":
        "scripts/03_wells/s06_grace_trends_well_pairing.py",
    "2.3_WELL_SUITABILITY.py": "scripts/03_wells/s07_well_suitability.py",
    "Basin_TAGGING.py": "scripts/04_predictors/s08_basin_tagging.py",
    "glhymps_rasterizer.py": "scripts/04_predictors/s09_glhymps_rasterizer.py",
    "URBAN_Fraction_Processor.py": "scripts/04_predictors/s10_urban_fraction.py",
    "Dataset_Assembler_Full.py": "scripts/04_predictors/s11_dataset_assembler.py",
    "3a_Collinearity_Check.py": "scripts/04_predictors/s12_collinearity_check.py",
    "5.1 MODEL_v4.py": "scripts/05_model/s13_pcd_lstm_train.py",
    "5.2 Bias_Correction.py": "scripts/05_model/s14_calibration_trends.py",
    "6.0_FINAL_Validation.py": "scripts/06_validation/s15_multiscale_validation.py",
    "6.1_Validation_3.py": "scripts/06_validation/s16_temporal_validation.py",
    "7.0 Uncertainty.py": "scripts/07_uncertainty/s17_uncertainty.py",
    "Phase 9.0.py": "scripts/08_consequences/s18_depletion_sea_level.py",
    "Phase_9.0.py": "scripts/08_consequences/s18_depletion_sea_level.py",
    "9.0b_significance_sensitivity.py": "scripts/08_consequences/s18b_significance_sensitivity.py",
    "8.0_Final_Product_Dataset.py": "scripts/07_uncertainty/s18a_final_product_packaging.py",
    "13.0 Polar Motion_v2.py": "scripts/08_consequences/s19_polar_motion_sea_level.py",
    "Data_Figures.py": "figures/data_figures.py",
    "Data_Figure_Part_2.py": "figures/data_figures_part2.py",
    "Data_Figures_Part2.py": "figures/data_figures_part2.py",
    "11.0_Figs.py": "figures/manuscript_trend_map.py",
    "study_area_map.py": "figures/study_area/study_area_map.py",
    "Sy_Map.py": "figures/extra/sy_map.py",
}
# the two different 'fig_style' modules are told apart by content
FIG_STYLE_BIG = ("def global_map_axis", "figures/fig_style.py")
FIG_STYLE_SMALL = ("def apply_style", "figures/study_area/fig_style.py")

LOGS = {
    "0.0_QC_Sanity.txt": "logs/s00_qc_sanity.txt",
    "0.5_Build_Reliability_Mask.txt": "logs/s01_reliability_mask.txt",
    "0.5b -- PERSIST SURFACE-WATER FLAG INTO THE RELIABILITY MASK.txt":
        "logs/s02_surface_water_flag.txt",
    "2.2 GRACE CELL TRENDS + WELL-TO-CELL PAIRING.txt":
        "logs/s06_grace_trends_well_pairing.txt",
    "2.3_WELL_SUITABILITY.txt": "logs/s07_well_suitability.txt",
    "Dataset_Assembler_Full.txt": "logs/s11_dataset_assembler.txt",
    "3a_Collinearity_Check.txt": "logs/s12_collinearity_check.txt",
    "5.1 MODEL_v4.txt": "logs/s13_pcd_lstm_train.txt",
    "5.2 Bias_Correction.txt": "logs/s14_calibration_trends.txt",
    "6.0_Final_VALIDATION.txt": "logs/s15_multiscale_validation.txt",
    "6.1_Validation_3.txt": "logs/s16_temporal_validation.txt",
    "7.0 Uncertainty.txt": "logs/s17_uncertainty.txt",
    "Phase 9.0 Depletion_Only.txt": "logs/s18_depletion_sea_level.txt",
    "13.0 Polar Motion v2.txt": "logs/s19_polar_motion_sea_level.txt",
}

STILL_MISSING = [
    "gap-fill  -> OUTPUTS/02_gapfill/gwsa_monthly_filled.nc",
    "Phase 2.1 well trends -> OUTPUTS/02_well_trends/well_trends*.parquet",
    "GROW inventory -> OUTPUTS/06_validation/grow_inventory.parquet",
    "static predictors -> OUTPUTS/09_ml_diagnostic/predictors_independent.nc",
    "Phase 5.0 tensor -> OUTPUTS/05_model_tensor/model_ready.nc",
    "Sy sweep + block significance (Table A.1, Fig. A.12)",
    "current Fig. 14 script",
]

SECRET_RE = re.compile(r"(api[_-]?key|password|passwd|secret|token|\.cdsapirc|\.netrc)", re.I)
PATH_RE = re.compile(r"[A-Za-z]:\\\\?[^\s\"']+")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def index_tree(root):
    idx = {}
    for dp, _, fns in os.walk(root):
        for fn in fns:
            idx.setdefault(fn, []).append(Path(dp) / fn)
    return idx


def pick(cands, report, name):
    if len(cands) > 1:
        cands = sorted(cands, key=lambda p: p.stat().st_mtime, reverse=True)
        report.append(f"AMBIGUOUS {name}: {len(cands)} copies, using newest: {cands[0]}")
    return cands[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder holding your .py scripts (searched recursively)")
    ap.add_argument("--logs", default=None, help="folder holding the .txt run logs (default: --src)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    src = Path(a.src)
    if not src.is_dir():
        sys.exit(f"--src not found: {src}")
    idx = index_tree(src)
    lidx = index_tree(Path(a.logs)) if a.logs else idx

    rows, notes, missing = [], [], []
    plan = []
    for orig, dest in SCRIPTS.items():
        if orig in idx:
            plan.append((pick(idx[orig], notes, orig), dest))
        elif orig not in ("Data_Figure_Part_2.py", "Data_Figures_Part2.py"):
            missing.append(orig)
    if not any(o in idx for o in ("Data_Figure_Part_2.py", "Data_Figures_Part2.py")):
        missing.append("Data_Figure_Part_2.py")

    # fig_style modules, identified by content (also accept fig_style1.py)
    styles = idx.get("fig_style.py", []) + idx.get("fig_style1.py", [])
    for marker, dest in (FIG_STYLE_BIG, FIG_STYLE_SMALL):
        hits = [p for p in styles if marker in p.read_text(encoding="utf-8", errors="ignore")]
        if hits:
            plan.append((pick(hits, notes, f"fig_style ({marker})"), dest))
        else:
            missing.append(f"fig_style.py containing '{marker}'")

    for orig, dest in LOGS.items():
        if orig in lidx:
            plan.append((pick(lidx[orig], notes, orig), dest))
        else:
            notes.append(f"log not found (optional): {orig}")

    for s, dest in plan:
        d = REPO / dest
        if not a.dry_run:
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, d)
        rows.append({"original_path": str(s), "repo_path": dest, "sha256": sha256(s)})
        print(f"  {'would copy' if a.dry_run else 'copied'}: {s.name}  ->  {dest}")

    if not a.dry_run:
        (REPO / "logs").mkdir(exist_ok=True)
        with open(REPO / "logs" / "ORIGINAL_NAMES.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["original_path", "repo_path", "sha256"])
            w.writeheader(); w.writerows(rows)

    print("\n=== NOT FOUND in --src (add before release) ===")
    for m in missing:
        print("  -", m)
    print("=== stages whose script name is unknown (find or write) ===")
    for m in STILL_MISSING:
        print("  -", m)
    if notes:
        print("=== notes ===")
        for n in notes:
            print("  -", n)

    print("=== scan of copied .py files ===")
    for s, dest in plan:
        if not dest.endswith(".py"):
            continue
        txt = s.read_text(encoding="utf-8", errors="ignore")
        sec = sorted(set(m.group(0) for m in SECRET_RE.finditer(txt)))
        n_paths = len(PATH_RE.findall(txt))
        flag = []
        if sec:
            flag.append(f"CHECK possible credential words {sec}")
        if n_paths:
            flag.append(f"{n_paths} hard-coded Windows path(s)")
        if flag:
            print(f"  {dest}: " + "; ".join(flag))
    print("\nHard-coded paths are expected (documented in README). "
          "Credential words must be reviewed by eye before pushing.")


if __name__ == "__main__":
    main()
