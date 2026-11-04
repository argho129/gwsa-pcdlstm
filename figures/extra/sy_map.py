#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone Robinson Projection Map for Specific Yield (S_y)

Spatially links well-level S_y attribute data with global aquifer and basin vector layers.
"""

import os
import warnings

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from shapely.geometry import Point

warnings.filterwarnings("ignore")

# Publication Styling Configuration
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 600,
})


def generate_sy_robinson_map(
    well_attr_path: str,
    aquifers_path: str,
    basins_path: str = None,
    out_path: str = "global_sy_robinson_map.png",
    default_sy: float = 0.10,
):
  if not os.path.exists(well_attr_path):
    raise FileNotFoundError(f"Well attribute parquet not found: {well_attr_path}")
  if not os.path.exists(aquifers_path):
    raise FileNotFoundError(f"Aquifer shapefile not found: {aquifers_path}")

  # 1. Load Well Attributes & Clean Specific Yield (S_y)
  print(f"[ INFO ] Loading well attributes: {well_attr_path}")
  wells = pd.read_parquet(well_attr_path)

  sy_col = next(
      (c for c in ["specific_y", "sy", "Sy", "SY"] if c in wells.columns), None
  )
  if sy_col is None:
    raise KeyError("Could not find a valid Specific Yield column in parquet.")

  wells[sy_col] = pd.to_numeric(wells[sy_col], errors="coerce").fillna(
      default_sy
  )
  wells[sy_col] = np.where(wells[sy_col] <= 0, default_sy, wells[sy_col])

  wells_gdf = gpd.GeoDataFrame(
      wells,
      geometry=[
          Point(lo, la)
          for lo, la in zip(wells["longitude"], wells["latitude"])
      ],
      crs=4326,
  )

  # 2. Load Aquifers & Aggregate Mean S_y Per Polygon
  print(f"[ INFO ] Loading aquifer shapefile: {aquifers_path}")
  aquifers = gpd.read_file(aquifers_path).to_crs(epsg=4326)
  aquifers["poly_id"] = "AQ_" + aquifers.index.astype(str)

  print("[ INFO ] Spatially joining well S_y to aquifer geometries...")
  joined = gpd.sjoin(
      wells_gdf[["geometry", sy_col]],
      aquifers[["poly_id", "geometry"]],
      how="inner",
      predicate="within",
  )

  # Calculate average S_y per aquifer polygon
  aq_sy_mean = joined.groupby("poly_id")[sy_col].mean().to_dict()
  aquifers["mean_sy"] = (
      aquifers["poly_id"].map(aq_sy_mean).fillna(default_sy)
  )

  # 3. Setup Robinson Projection Figure
  fig = plt.figure(figsize=(14, 7.5), constrained_layout=True)
  ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
  ax.set_global()

  # 4. Underlay Layer (Basemap, Ocean, Land, Borders)
  ax.add_feature(
      cfeature.OCEAN, facecolor="#eef2f5", zorder=0, edgecolor="none"
  )
  ax.add_feature(
      cfeature.LAND, facecolor="#f8f9fa", zorder=1, edgecolor="none"
  )
  ax.add_feature(
      cfeature.BORDERS,
      linewidth=0.4,
      edgecolor="#777777",
      linestyle="-",
      zorder=2,
      alpha=0.6,
  )
  ax.coastlines(linewidth=0.5, color="#333333", zorder=2)

  # 5. Global HydroBASINS Underlay (Semi-transparent Outlines)
  if basins_path and os.path.exists(basins_path):
    try:
      print("[ INFO ] Overlaying HydroBASINS vector layer...")
      basins = gpd.read_file(basins_path).to_crs(epsg=4326)
      basins.plot(
          ax=ax,
          facecolor="none",
          edgecolor="#1f4e79",
          linewidth=0.25,
          alpha=0.30,
          transform=ccrs.PlateCarree(),
          zorder=3,
      )
    except Exception as e:
      print(f"[ WARN ] Could not load HydroBASINS layer: {e}")

  # 6. High-Contrast Aquifer Specific Yield Plotting
  cmap = "plasma"
  vmin, vmax = aquifers["mean_sy"].min(), aquifers["mean_sy"].max()

  aquifers.plot(
      column="mean_sy",
      cmap=cmap,
      ax=ax,
      vmin=vmin,
      vmax=vmax,
      linewidth=0.3,
      edgecolor="#222222",
      alpha=0.85,
      transform=ccrs.PlateCarree(),
      zorder=4,
  )

  # Overlay well locations as high-density points
  ax.scatter(
      wells_gdf.geometry.x,
      wells_gdf.geometry.y,
      c=wells_gdf[sy_col],
      cmap=cmap,
      s=4,
      vmin=vmin,
      vmax=vmax,
      alpha=0.6,
      edgecolor="none",
      transform=ccrs.PlateCarree(),
      zorder=5,
  )

  # 7. Colorbar and Map Formatting
  norm = Normalize(vmin=vmin, vmax=vmax)
  sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
  sm._A = []
  cbar = fig.colorbar(
      sm, ax=ax, orientation="horizontal", pad=0.04, shrink=0.45, aspect=28
  )
  cbar.set_label(
      r"Specific Yield ($S_y$) $[-]$", fontsize=12, labelpad=6, fontweight="bold"
  )
  cbar.ax.tick_params(labelsize=10)

  ax.set_title(
      r"Global Distribution of Aquifer Specific Yield ($S_y$)",
      fontsize=15,
      fontweight="bold",
      pad=12,
  )
  ax.gridlines(
      draw_labels=False,
      linewidth=0.3,
      color="#cccccc",
      alpha=0.5,
      linestyle="--",
  )

  # Save Output
  os.makedirs(os.path.dirname(out_path), exist_ok=True)
  plt.savefig(out_path, dpi=600, bbox_inches="tight", facecolor="white")
  plt.close(fig)
  print(f"[  OK  ] Map saved successfully: {out_path}")


if __name__ == "__main__":
  WELL_ATTR_PARQUET = r"D:\THESIS\Arpon_Thesis_Final\OUTPUTS\02_well_trends\well_trends_v2_tiered.parquet"
  AQUIFERS_SHP = r"D:\THESIS\Arpon_Thesis_Final\DATA\Aquifers\jasechko_et_al_2024_aquifers.shp"
  HYBAS_GPKG = (
      r"D:\THESIS\Arpon_Thesis_Final\DATA\HydroBasins\hybas_lev03_global.gpkg"
  )
  OUTPUT_PNG = r"D:\THESIS\Arpon_Thesis_Final\OUTPUTS\06_validation_final\paper_style\figures\global_sy_robinson_map.png"

  generate_sy_robinson_map(
      well_attr_path=WELL_ATTR_PARQUET,
      aquifers_path=AQUIFERS_SHP,
      basins_path=HYBAS_GPKG,
      out_path=OUTPUT_PNG,
  )