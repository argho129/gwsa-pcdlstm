"""
3a_collinearity_check.py — VIF + correlation on the prototype static/dynamic sets.
Read-only. Produces: VIF table, dynamic corr matrix, heatmap PNG (600 dpi).
"""
import numpy as np, pandas as pd, xarray as xr
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pathlib import Path

DS  = r"D:\THESIS\Arpon_Thesis_Final\OUTPUTS\10_model\data\dataset_prototype.nc"
FIG = Path(r"D:\THESIS\Arpon_Thesis_Final\OUTPUTS\10_model\figures")
FIG.mkdir(parents=True, exist_ok=True)

d = xr.open_dataset(DS)
STAT = ["elevation","slope","aei","aeigw","urban_frac_mean","logk_ferr","porosity","lat","lon"]
DYN  = ["P","E","R","t2m","ndvi","sm"]

# ---------- STATIC VIF (one row per cell) ----------
print("="*60); print("STATIC VIF (spatial multicollinearity)"); print("="*60)
X = np.column_stack([d[v].values for v in STAT]).astype(float)
X = X[np.all(np.isfinite(X), axis=1)]
Xs = (X - X.mean(0)) / X.std(0)           # standardize
vif = pd.DataFrame({"predictor":STAT,
    "VIF":[variance_inflation_factor(Xs,i) for i in range(Xs.shape[1])]})
vif = vif.sort_values("VIF", ascending=False).reset_index(drop=True)
print(vif.to_string(index=False))
print("\n  rule: <5 keep | 5-10 flag/justify | >10 drop or merge")

# ---------- STATIC correlation ----------
Cs = pd.DataFrame(X, columns=STAT).corr()

# ---------- DYNAMIC correlation (per-cell temporal, averaged) ----------
print("\n"+"="*60); print("DYNAMIC correlation (temporal, cell-averaged)"); print("="*60)
dyn = {v: d[v].values for v in DYN}       # (T, cell)
ncell = dyn["P"].shape[1]
acc = np.zeros((len(DYN), len(DYN))); cnt = 0
for c in range(ncell):
    M = np.column_stack([dyn[v][:,c] for v in DYN])
    ok = np.all(np.isfinite(M), axis=1)
    if ok.sum() < 24: continue
    acc += np.corrcoef(M[ok].T); cnt += 1
Cd = pd.DataFrame(acc/cnt, index=DYN, columns=DYN)
print(Cd.round(2).to_string())
print("\n  rule: |r|>0.9 (avg) -> drop one")

# ---------- heatmaps ----------
fig, ax = plt.subplots(1,2, figsize=(15,6))
for a,(C,ttl) in zip(ax, [(Cs,"Static predictors"),(Cd,"Dynamic predictors")]):
    im=a.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    a.set_xticks(range(len(C))); a.set_xticklabels(C.columns, rotation=45, ha="right")
    a.set_yticks(range(len(C))); a.set_yticklabels(C.columns)
    a.set_title(ttl)
    for ii in range(len(C)):
        for jj in range(len(C)):
            a.text(jj,ii,f"{C.iloc[ii,jj]:.2f}",ha="center",va="center",fontsize=7)
    fig.colorbar(im, ax=a, fraction=0.046)
plt.tight_layout(); plt.savefig(FIG/"collinearity_prototype.png", dpi=600, bbox_inches="tight")
print(f"\nsaved: {FIG/'collinearity_prototype.png'}")
d.close()