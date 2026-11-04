# update_readme.py - adds the newly released scripts to README.md and
# logs/ORIGINAL_NAMES.csv, and removes the "not in this release" rows.
# A copy of the old README is kept OUTSIDE the repository first.
#
# Run:  python tools\update_readme.py        (from inside the gwsa-pcdlstm folder)
#  or:  python D:\...\gwsa-pcdlstm\tools\update_readme.py
import csv, os, re, shutil, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if len(sys.argv) > 1:
    REPO = sys.argv[1]
README = os.path.join(REPO, "README.md")
NAMES = os.path.join(REPO, "logs", "ORIGINAL_NAMES.csv")
BACKUP = r"D:\gwsa_backup_before_clean2"

# step, script, main output, manuscript item
ROWS = {
    "s03b": ("02_gwsa/s03b_gapfill.py", "OUT/02_gapfill/gwsa_monthly_filled.nc", "§2.5"),
    "s05a": ("03_wells/s05a_well_trends_v1.py",
             "OUT/02_well_trends/well_trends.parquet (read by s06)", "§2.9, §3.3"),
    "s05b": ("03_wells/s05b_well_trends.py",
             "OUT/02_well_trends/well_trends_v2.parquet (read by s07)", "§2.9"),
    "s10b": ("04_predictors/s10b_independent_predictors.py",
             "OUT/09_ml_diagnostic/predictors_independent.nc", "§2.2"),
    "s12b": ("05_model/s12b_model_tensor.py", "OUT/05_model_tensor/model_ready.nc", "§2.6–2.7"),
    "s17b": ("07_uncertainty/s17b_sy_sensitivity_sweep.py",
             "OUT/07_uncertainty/sy_sensitivity_table.csv", "supports the fixed 0.15 measured-Sy error used in s17c"),
    "s17c": ("07_uncertainty/s17c_sy_response_curve.py",
             "OUT/07_uncertainty/sy_response_curve.csv, fig7_05, fig7_06", "Table A.1, Fig. A.12"),
    "fig": ("figures/depletion_maps.py",
            "OUT/11_maps/fig11_2_globe_views, fig11_3_regional_hotspots (panels)", "Fig. 14"),
}
REPLACE = [("gap-fill", ["s03b"]), ("well trends", ["s05a", "s05b"]),
           ("static predictors", ["s10b"]), ("model tensor", ["s12b"]),
           ("sy sweep", ["s17b", "s17c"])]
ORIGINAL = {
    "scripts/02_gwsa/s03b_gapfill.py": "2.0.1 GAP-FILLED CONTINUOUS MONTHLY GWSA PRODUCT.py",
    "scripts/03_wells/s05a_well_trends_v1.py": "2.1_WELL TREND COMPUTATION.py",
    "scripts/03_wells/s05b_well_trends.py": "2.1 v2__WELL TREND COMPUTATION.py",
    "scripts/04_predictors/s10b_independent_predictors.py": "2.9a Assemble Predictors_1.py",
    "scripts/05_model/s12b_model_tensor.py": "5.0_v2_PHASE5_ASSEMBLE_MODEL_TENSOR.py",
    "scripts/07_uncertainty/s17b_sy_sensitivity_sweep.py": "7_1_Sy_Sensitivity_Sweep.py",
    "scripts/07_uncertainty/s17c_sy_response_curve.py": "7_2_Sy_ResponseCurve_and_Aggregation.py",
    "figures/depletion_maps.py": "11.0_Figs.py (Phase 11 maps, commented block at the top)",
}
NOTE = ("All pipeline stages are included. One intermediate input is not: "
        "OUT/06_validation/grow_inventory.parquet, a per-well table (GROW_ID, sampling interval, "
        "first and last observation date) derived from the GROW time series and read by s15 and "
        "s16; the script that wrote it was not retained. s10b takes its grid coordinates and land "
        "mask from OUT/03_calibrated/gwd_trend_calibrated_v2.nc, an earlier intermediate file on "
        "the same 0.5° grid; none of its values enter the predictors.")
LAYOUT = [
    (r"s03[–-]s04(\s+)GWSA isolation and the TWSA = GWSA \+ OSA identity",
     r"s03–s04\1GWSA isolation, gap-fill (s03b), TWSA = GWSA + OSA identity"),
    (r"s13[–-]s14(\s+)physics-constrained LSTM",
     r"s12b–s14\1model tensor, physics-constrained LSTM"),
    (r"(07_uncertainty/\s+)s17(\s+)uncertainty decomposition",
     r"\1s17–s18a\2uncertainty decomposition, Sy sensitivity, product packaging"),
]

if not os.path.exists(README):
    sys.exit("README.md not found in " + REPO)
raw = open(README, "rb").read()
crlf = b"\r\n" in raw
text = raw.decode("utf-8").replace("\r\n", "\n")
os.makedirs(BACKUP, exist_ok=True)
bk = os.path.join(BACKUP, "README_before_update.md")
if not os.path.exists(bk):
    shutil.copy2(README, bk)
lines = text.split("\n")
report = []

# a row that already exists, used as the formatting template
tmpl = next((l for l in lines if "s03_gwsa_isolation.py" in l and ("|" in l or "\t" in l)), None)
if tmpl is None:
    sys.exit("Could not find the s03 row of the pipeline table; README left unchanged.")
pipe = tmpl.lstrip().startswith("|")
cells_t = [c.strip() for c in (tmpl.strip().strip("|").split("|") if pipe else tmpl.split("\t"))]
bt_script = cells_t[1].startswith("`")
bt_out = len(cells_t) > 2 and cells_t[2].startswith("`")

def make_row(step):
    script, outp, ms = ROWS[step]
    if bt_out:
        base, _, tail = outp.partition(" (")
        outp = ", ".join("`%s`" % x for x in base.split(", ")) + (" (" + tail if tail else "")
    c = [step, "`%s`" % script if bt_script else script, outp, ms]
    return ("| " + " | ".join(c) + " |") if pipe else "\t".join(c)

present = "\n".join(lines)
new_lines = []
for l in lines:
    if "(not in this release)" in l or ("not in this release" in l and ("|" in l or "\t" in l)
                                         and "Stages marked" not in l):
        low = l.lower()
        hit = next((steps for key, steps in REPLACE if key in low), None)
        if hit:
            for s in hit:
                if ROWS[s][0] not in present:
                    new_lines.append(make_row(s))
            report.append("replaced row: " + l.strip()[:70] + "  ->  " + ", ".join(hit))
            continue
        report.append("NOTE: row left as is (unknown): " + l.strip()[:70])
    new_lines.append(l)

# Fig. 14 row: after the manuscript_trend_map row if there is one, else after s19
if ROWS["fig"][0] not in present:
    anchor = next((i for i, l in enumerate(new_lines) if "manuscript_trend_map.py" in l
                   and ("|" in l or "\t" in l)), None)
    if anchor is None:
        anchor = next((i for i, l in enumerate(new_lines) if "s19_polar_motion" in l), None)
    if anchor is not None:
        new_lines.insert(anchor + 1, make_row("fig"))
        report.append("added row: figures/depletion_maps.py (Fig. 14)")
    else:
        report.append("NOTE: could not place the Fig. 14 row")

text = "\n".join(new_lines)
m = re.search(r"Stages marked\W{0,3}not in this release\W{0,3} are intermediate.*?Zenodo data record\.",
              text, flags=re.S)
if m:
    text = text[:m.start()] + NOTE + text[m.end():]
    report.append("replaced the 'not in this release' note")
else:
    report.append("NOTE: 'Stages marked not in this release' sentence not found")
for pat, rep in LAYOUT:
    text, n = re.subn(pat, rep, text)
    report.append(("updated layout line: " if n else "NOTE: layout line not found: ") + pat[:40])

with open(README, "w", encoding="utf-8", newline="\r\n" if crlf else "\n") as fh:
    fh.write(text)

# ORIGINAL_NAMES.csv
if os.path.exists(NAMES):
    rows = list(csv.reader(open(NAMES, encoding="utf-8", newline="")))
    header = [h.lower() for h in rows[0]] if rows else []
    orig_first = bool(header) and "orig" in header[0]
    have = {c.replace("\\", "/") for r in rows for c in r}
    added = 0
    with open(NAMES, "a", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        for repo_path, orig in ORIGINAL.items():
            if repo_path not in have:
                w.writerow([orig, repo_path] if orig_first else [repo_path, orig]); added += 1
    report.append("ORIGINAL_NAMES.csv: %d row(s) added" % added)
else:
    report.append("NOTE: logs/ORIGINAL_NAMES.csv not found")

left = text.count("not in this release")
print("\n".join(report))
print("\n'not in this release' left in README: %d" % left)
print("Old README kept in: " + bk + "  (do NOT upload)")
