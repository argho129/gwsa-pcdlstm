#!/usr/bin/env python3
"""
export_model_metadata.py  --  package what is needed to reuse the trained model,
without redistributing the input tensor.

Reads  OUTPUTS/05_model_tensor/model_ready.nc   (feature names, normalisation stats)
       OUTPUTS/05_model_tensor/lstm_state_v4.pt (trained weights)
Writes <out>/model_checkpoint/
         lstm_state_v4.pt           (copied unchanged)
         model_metadata.json        (features, mean/sd, hyper-parameters, SHA-256)

Usage:
    python tools/export_model_metadata.py --model-dir "D:\\THESIS\\Arpon_Thesis_Final\\OUTPUTS\\05_model_tensor" --out zenodo_upload

The network class (`Decomp`) is defined inside scripts/05_model/s13_pcd_lstm_train.py;
load the weights with that class: Decomp(len(dyn_features), len(static_features), 64, 1).
"""
import argparse, hashlib, json, shutil
from pathlib import Path

import numpy as np
import xarray as xr

# values hard-coded in s13 (5.1 MODEL_v4.py) for the published run
HYPER = {
    "seed": 20260730, "hidden_units": 64, "lstm_layers": 1, "batch_cells": 512,
    "learning_rate": 1.5e-3, "weight_decay": 1e-5, "grad_clip": 1.0,
    "epochs_sweep": 45, "patience_sweep": 8, "epochs_final": 120, "patience_final": 15,
    "basin_weight_sweep": [0.0, 0.1, 0.25, 0.5, 1.0], "basin_weight_final": 0.5,
    "split_fractions_train_val_test": [0.70, 0.15, 0.15],
    "supply_bound": "b = 1.5*|mean_t(NetFlux - E)|/10*12 + 1.0  [cm/yr]",
    "standardisation": "x_z = (x - mu) / sd, sd==0 -> 1, NaN -> 0 after scaling",
    "outputs": "GWSA = trend + seasonal + residual (cm)",
}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def as_list(a):
    return [None if not np.isfinite(v) else float(v) for v in np.asarray(a, float).ravel()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", default="zenodo_upload")
    ap.add_argument("--weights", default="lstm_state_v4.pt")
    a = ap.parse_args()

    md = Path(a.model_dir)
    tensor, weights = md / "model_ready.nc", md / a.weights
    for p in (tensor, weights):
        if not p.exists():
            raise SystemExit(f"not found: {p}")

    ds = xr.open_dataset(tensor)
    meta = {
        "dyn_features": [str(v) for v in ds["dfeat"].values],
        "static_features": [str(v) for v in ds["sfeat"].values],
        "dyn_mean": as_list(ds["dyn_mu"].values), "dyn_sd": as_list(ds["dyn_sd"].values),
        "static_mean": as_list(ds["sta_mu"].values), "static_sd": as_list(ds["sta_sd"].values),
        "n_months": int(ds.sizes["time"]), "n_cells": int(ds.sizes["cell"]),
        "hyperparameters": HYPER,
    }
    ds.close()

    out = Path(a.out) / "model_checkpoint"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(weights, out / weights.name)
    meta["weights_file"] = weights.name
    meta["weights_sha256"] = sha256(out / weights.name)
    with open(out / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {out/'model_metadata.json'} and {out/weights.name}")
    print("dynamic:", meta["dyn_features"])
    print("static :", meta["static_features"])


if __name__ == "__main__":
    main()
