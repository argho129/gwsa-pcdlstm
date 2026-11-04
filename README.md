# Global groundwater storage anomaly reconstruction (GRACE/GRACE-FO, physics-constrained decomposition LSTM)

Code for:

> Sarkar, A., Sarkar, S. K., & Haydar, M. (2026). *Reconstruction of Global Groundwater Storage Anomaly from GRACE Satellite Gravimetry Using a Physics-Constrained Decomposition LSTM.* Manuscript under review.

- **Data product (Zenodo):** https://doi.org/10.5281/zenodo.23909628 
- **This code (Zenodo archive of the v1.0.0 release):** https://doi.org/10.5281/zenodo.YYYYYYY  

The pipeline separates groundwater from GRACE/GRACE-FO total water storage (JPL RL06.3Mv04 CRI mascons minus GLDAS Noah v2.1 soil moisture, snow and canopy). It then reconstructs the monthly groundwater storage anomaly (GWSA) on a 0.5° grid for April 2002 – December 2022 with a decomposition LSTM (trend + seasonal + residual). The trend of each cell is bounded by the net atmospheric water supply, and basin-mean trends are held to observation. The pipeline then calibrates amplitude, validates against the GROW monitoring-well compilation, WaterGAP 2.2d and GRACE-SeDA, decomposes the trend uncertainty, and converts the depletion into sea-level and polar-motion terms.

---

## Repository layout

```
scripts/
  00_qc/            s00  input sanity checks
  01_masks/         s01–s02  permanent-ice and surface-water masks
  02_gwsa/          s03–s04  GWSA isolation, gap-fill (s03b), TWSA = GWSA + OSA identity
  03_wells/         s05–s07  GROW well trends, pairing, suitability classes
  04_predictors/    s08–s12  basin tags, static predictors, dataset assembly, VIF
  05_model/         s12b–s14  model tensor, physics-constrained LSTM, amplitude calibration + trends
  06_validation/    s15–s16  multi-scale trend and temporal validation
  07_uncertainty/   s17–s18a  uncertainty decomposition, Sy sensitivity, product packaging
  08_consequences/  s18–s19  depletion volumes, sea level, polar motion
figures/            manuscript figure scripts and shared style (fig_style.py)
figures/study_area/ Fig. 1 (has its own small fig_style.py)
logs/               console output of the runs that produced the published numbers
tools/              release helpers (not part of the analysis)
```

All pipeline stages are included. One intermediate input is not: OUT/06_validation/grow_inventory.parquet, a per-well table (GROW_ID, sampling interval, first and last observation date) derived from the GROW time series and read by s15 and s16; the script that wrote it was not retained. s10b takes its grid coordinates and land mask from OUT/03_calibrated/gwd_trend_calibrated_v2.nc, an earlier intermediate file on the same 0.5° grid; none of its values enter the predictors. The original file names on the authors' machine are listed in `logs/ORIGINAL_NAMES.csv`, which `tools/make_repo_layout.py` writes. The analysis code is exactly as run; only comments and console message text were tidied for release.

---

## Pipeline, outputs and manuscript items

Run the scripts in this order. `OUT` stands for `<ROOT>/OUTPUTS`.

| Step | Script | Main output(s) | Manuscript |
|---|---|---|---|
| s00 | `00_qc/s00_qc_sanity.py` | console only | – |
| s01 | `01_masks/s01_reliability_mask.py` | `OUT/00_reliability_mask/reliability_mask.nc` | §2.3, §3.1 |
| s02 | `01_masks/s02_surface_water_flag.py` | adds `surface_water` to the mask | §2.3 |
| s03 | `02_gwsa/s03_gwsa_isolation.py` | `OUT/01_gwsa_monthly/gwsa_monthly.nc` | §2.4, Eqs 1–4 |
| s03b | `02_gwsa/s03b_gapfill.py` | `OUT/02_gapfill/gwsa_monthly_filled.nc` | §2.5 |
| s04 | `02_gwsa/s04_other_storage_identity.py` | `OUT/05_model_tensor/other_storage_anomaly.nc` | Eq. 5 |
| s05 | `03_wells/s05_inspect_sy_grow.py` | console only | – |
| s05a | `03_wells/s05a_well_trends_v1.py` | `OUT/02_well_trends/well_trends.parquet` (read by s06) | §2.9, §3.3 |
| s05b | `03_wells/s05b_well_trends.py` | `OUT/02_well_trends/well_trends_v2.parquet` (read by s07) | §2.9 |
| s06 | `03_wells/s06_grace_trends_well_pairing.py` | `grace_cell_trends.nc`, `paired_cells.parquet` | §3.3 |
| s07 | `03_wells/s07_well_suitability.py` | `well_trends_by_cell.nc`, `well_trends_v2_tiered.parquet` | §3.9 |
| s08 | `04_predictors/s08_basin_tagging.py` | `OUT/09_ml_diagnostic/cell_basins.parquet` | §2.3 |
| s09 | `04_predictors/s09_glhymps_rasterizer.py` | `DATA/SOIL_DATA/glhymps_0p5.nc` | §2.2 |
| s10 | `04_predictors/s10_urban_fraction.py` | `OUT/09_ml_diagnostic/urban_fraction.nc` | §2.2 |
| s10b | `04_predictors/s10b_independent_predictors.py` | `OUT/09_ml_diagnostic/predictors_independent.nc` | §2.2 |
| s11 | `04_predictors/s11_dataset_assembler.py` | `OUT/10_model/data/dataset_full.nc` (`SCOPE="prototype"` gives the focus-basin subset) | §2.5 |
| s12 | `04_predictors/s12_collinearity_check.py` | VIF table | §2.6, §3.5 |
| s12b | `05_model/s12b_model_tensor.py` | `OUT/05_model_tensor/model_ready.nc` | §2.6–2.7 |
| s13 | `05_model/s13_pcd_lstm_train.py` | `gwsa_reconstructed.nc`, `lstm_state_v4.pt` | §2.7, §3.6, Table 3, Fig. 9 |
| s14 | `05_model/s14_calibration_trends.py` | `OUT/05_model_tensor/gwsa_final.nc` | §2.8, §3.7 |
| s15 | `06_validation/s15_multiscale_validation.py` | `OUT/06_validation_final/*_FINAL.csv` | §3.8–3.9, Table 4, Figs 10b–f, 11, 12, A.3 |
| s16 | `06_validation/s16_temporal_validation.py` | `OUT/06_validation_final/paper_style/` | §3.10, Figs A.4–A.10 |
| s17 | `07_uncertainty/s17_uncertainty.py` | `OUT/07_uncertainty/gwsa_uncertainty.nc` | §2.10, §3.11, Figs 13, A.11 |
| s17b | `07_uncertainty/s17b_sy_sensitivity_sweep.py` | `OUT/07_uncertainty/sy_sensitivity_table.csv` | supports the fixed 0.15 measured-Sy error used in s17c |
| s17c | `07_uncertainty/s17c_sy_response_curve.py` | `OUT/07_uncertainty/sy_response_curve.csv`, `fig7_05`, `fig7_06` | Table A.1, Fig. A.12 |
| s18a | `07_uncertainty/s18a_final_product_packaging.py` | `OUT/08_final_product/GWSA_KUET_v1_1D.nc`, `_gridded.nc` | released product |
| s18 | `08_consequences/s18_depletion_sea_level.py` | `OUT/09_depletion_sle/` | §3.12 |
| s18b | `08_consequences/s18b_significance_sensitivity.py` | console only | significance-filter sensitivity |
| s19 | `08_consequences/s19_polar_motion_sea_level.py` | `OUT/13_polar_motion/` | §3.12–3.13, Figs 15–16 |

**Figures**

| Script | Figures |
|---|---|
| `figures/study_area/study_area_map.py` | Fig. 1 |
| `figures/data_figures.py` | Figs 4, 5, 6, 7, 8a–b, A.1, A.2 |
| `figures/data_figures_part2.py` | Fig. 8c–f |
| `figures/manuscript_trend_map.py` | Fig. 10a |
| fig | `figures/depletion_maps.py` | `OUT/11_maps/fig11_2_globe_views`, `fig11_3_regional_hotspots` (panels) | Fig. 14 |
| s13 / s15 / s16 / s17 / s19 | Figs 9, 10b–f, 11–13, 15–16, A.3–A.11 |

Figs 2 and 3 are schematic diagrams and have no script.

---

## Requirements

- Python 3.11.1. The published runs used a CPU-only machine (Intel i5-11300H, 6 threads).
- Packages: see `requirements.txt`. The model run used `torch 2.7.1+cpu`. Exact versions of the other packages are in `requirements-lock.txt` (`pip freeze` from the environment used).
- Approximate runtimes on that machine: s13 ≈ 2.6 h, s14 ≈ 4.2 h, s15 ≈ 54 min, s17 ≈ 11 min, s16 ≈ 9 min.

## Input data

The input data are not redistributed here. Download them from the providers below and arrange them under one root folder. The scripts use `ROOT = D:\THESIS\Arpon_Thesis_Final` with `DATA` and `OUTPUTS` subfolders. **Paths are hard-coded at the top of each script**: either recreate this layout, or replace `D:\THESIS\Arpon_Thesis_Final` with your own root in every script (a single find-and-replace). The shared style module is loaded from `FIG_STYLE_DIR`; point that at this repository's `figures/` folder.

| Folder under `DATA/` | Dataset | Source |
|---|---|---|
| `grace_jpl_mascon/GRCTellus.JPL.200204_202512.GLO.RL06.3M.MSCNv04CRI.nc` | JPL GRACE/GRACE-FO mascon RL06.3 v04 CRI | PO.DAAC, doi:10.5067/TEMSC-3JC634 |
| `gldas_noah_025/` (monthly `.nc4`) | GLDAS Noah L4 0.25° v2.1 | GES DISC, doi:10.5067/SXAVCZFAQLNO |
| `climate_era5/` (tp, swvl1), `era5_land_runoff/` (ro) | ERA5-Land monthly means | CDS, doi:10.24381/cds.68d2bb30 |
| `era5_thermodynamics/extracted/*avgua.nc` | ERA5 monthly 2 m temperature | CDS, doi:10.24381/cds.f17050d7 |
| `GLEAM_v4/E_*.nc` (`GLEAM_v4_Ep/` for s00 only) | GLEAM v4.3a | https://www.gleam.eu |
| `modis_vegetation/*.hdf` | MODIS MOD13C2 C6.1 | LP DAAC, doi:10.5067/MODIS/MOD13C2.061 |
| `SOIL_DATA/GLHYMPS.shp` | GLHYMPS 2.0 | Borealis, doi:10.5683/SP2/TTJNIU |
| `Topography/ETOPO_2022_v1_60s_surface.nc` | ETOPO 2022 | NOAA NCEI, doi:10.25921/fd45-gt74 |
| `Urban Fraction/` | Gao & O'Neill (2020) urban fraction | doi:10.1038/s41467-020-15788-7 |
| FAO GMIA v5 (static predictor builder) | Global Map of Irrigation Areas | https://www.fao.org/aquastat |
| `WHYMAP/WHYMAP_GWR/shp/` | WHYMAP Groundwater Resources of the World | https://www.whymap.org |
| `HydroBasins/` (`*lev03*.shp`, `hybas_lev03_global.gpkg`) | HydroBASINS v1c level 3 | https://www.hydrosheds.org |
| RGI glacier outlines (fetched by s01 from the GLIMS WFS, cached) | Randolph Glacier Inventory | doi:10.7265/N5-RGI-60 |
| `Aquifers/jasechko_et_al_2024_aquifers.shp` | Jasechko et al. (2024) aquifers | doi:10.5281/zenodo.10003697 |
| `GROW/grow_attributes.parquet`, `grow_timeseries.parquet` | GROW (CC BY-NC-SA 4.0) | doi:10.5281/zenodo.15149480 |
| `specific yeild-上传/geishuidu-min.shp` | Global specific-yield field (T. Sun et al., 2026) | figshare, doi:10.6084/m9.figshare.31669225 |
| `WaterGAP_Data/watergap_22d_WFDEI-GPCC_histsoc_groundwstor_monthly_1901_2016.nc4` | WaterGAP 2.2d | doi:10.5194/gmd-14-1037-2021 |
| `Water_Paper Validation data (TWSA)/GRACE-SeDA_v0_2002_2019.nc` | GRACE-SeDA | doi:10.1038/s44221-024-00194-w |
| `Global SHapefile/World_Countries_Generalized.shp` | Esri World Countries (Generalized) | Esri |

---

## Key settings (as used)

| Setting | Value | Where |
|---|---|---|
| Grid / period | 0.5°, 2002-04 to 2022-12, 249 months | s03 |
| Anomaly baseline | 2004–2009; GWSA cap ±100 cm | s03 |
| Random seed | 20260730 | s13, s14, s15, s17 |
| LSTM | 1 layer, 64 hidden units; batch 512 cells | s13 |
| Optimiser | Adam, lr 1.5e-3, weight decay 1e-5, gradient clip 1.0 | s13 |
| Month split | 70 / 15 / 15 of observed months (150 / 32 / 32); 35 gap-filled months excluded | s13 |
| Supply bound | b = 1.5·\|mean(F − E)\|·12/10 + 1 cm/yr | s13 |
| Basin weight sweep | {0, 0.1, 0.25, 0.5, 1.0}, 45 epochs (patience 8); final λ = 0.5, 120 epochs (patience 15) | s13 |
| Calibration | per-cell variance matching on training months, ≥ 24 observed months | s14 |
| Trend CI | Theil–Sen, 200 year-block bootstrap replicates | s14 |
| Uncertainty | 200 Monte Carlo members; Sy relative error 0.15 (≥ 1 Class A well) / 0.40 | s17 |
| Polar motion | a = 6.371e6 m, C − A = 2.6398e35 kg m², k₂′ = −0.301, κ = 1.43 | s19 |

## Implementation notes

- **s03:** GLDAS stores are converted to anomalies about a per-calendar-month 2004–2009 climatology. GRACE `lwe_thickness × scale_factor` is used as delivered by JPL.
- **s13:** the basin mass-conservation penalty is applied once per epoch as a separate optimiser step over all cells of each basin. λ is chosen as the lowest basin-closure RMSE among arms within 3 % of the best held-out RMSE.
- **s14 and s15:** for speed, the Theil–Sen slope uses a fixed random subsample of 200 (s14) or 150 (s15) time steps.
- **s17:** the variogram range is limited to between 1° and 6°. The model-error standard deviation is set to 7.05 cm (`HELDOUT_RMSE_CM`), slightly above the final held-out RMSE of 6.973 cm, so the model uncertainty term is marginally conservative.
- The code comments contain the working project title "Aquifer to Atmosphere" and notes from earlier project phases. These do not affect the results.

## Licence and citation

The code is released under the MIT licence (see `LICENSE`). Please cite the article and the Zenodo archive; `CITATION.cff` contains the citation details.

GROW data are licensed CC BY-NC-SA 4.0. Any well-level outputs you derive from them inherit that licence.

## Contact

Arpon Sarkar, Department of Urban and Regional Planning, KUET, Khulna-9203, Bangladesh. arghosarkar79@gmail.com, ORCID 0009-0006-1722-4782.
