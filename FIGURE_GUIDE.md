# Figure and analysis guide

The files named below are in `Fig2-5_GitHub_draft_20261127.zip`. Extract it and preserve the original paths under `src`. These are working scripts rather than a self-contained, fully verified raw-data-to-results pipeline. Additional scripts in the archive are local dependencies, prior analysis or plotting variants; the entries below identify the figure drivers and selected analysis modules, not a complete dependency graph.

## Figure 2 — geomagnetic forcing and mesospheric observations

**Figure driver:** `src/元旦磁暴_跨层波数1/two_region_mechanism_profiles_20261021/plot_figure2_temperature_dailymean_20261121.py`

The driver computes nightly daily-mean MLS temperature changes and delegates the remaining panels to the earlier figure code. Related scripts include `plot_figure1_f_min30_20261109.py`, `plot_figure2_indices_mainphase_20261101.py` and `plot_figure2_Dec28_Jan11_20261022.py` in the same directory. These cover geomagnetic-index, particle-flux and ozone/temperature figure preparation.

**Inputs:** MLS temperature files, geomagnetic-index tables and previously processed MLS and MEPED products are required. The temperature driver searches several original relative directories; it selects daily files for 28 December 2024–11 January 2025. The exact minimal input set and provenance of all intermediate tables remain to be established.

## Figure 3 — atmospheric environment and planetary-wave diagnostics

**Figure driver:** `src/元旦磁暴_跨层波数1/plot_combined_environment_w1_ep_qy_legend_bottom_20261109.py`

This assembles the environmental and W1 diagnostic panels. Related modules include `plot_figure3_flux_amplitude_EP_nature_20261105.py`, `plot_W1_amplitude_flux_daily_Jan02_Jan11_2x5_20261022.py`, `plot_daily_W1_EP_directions_20261010.py`, `ep_flux_display_20261011.py` and `build_acp2018_event_figures_20261010.py`. Their roles include wave amplitude, heat-flux proxy, EP-flux display and background diagnostics. They share additional dependencies in the archive.

**Inputs:** MLS-derived fields and intermediate diagnostics are needed. The original scripts also rely on cached products. A complete list of minimal inputs and the full upstream calculation chain have not been independently verified.

## Figure 4 — wave phase and vortex geometry

**Figure driver:** `src/A研究进展报告/ERA5_2025元旦磁暴研究报告/plot_figure4_phase_envelope_180mm_aligned_20261107.py`

Related modules include `plot_figure4_wave_vortex_reconstruction_20261027.py`, `figure4_full_depth_data_20261030.py` and `analyze_w1_w2_vortex_20261025.py`. They provide wave, envelope, reconstruction and related vortex diagnostics used by the figure workflow. The driver imports existing analysis modules and reads cached data; reconstruction is a diagnostic of geometry rather than a separate forcing experiment.

**Inputs:** ERA5-related fields, processed wave/vortex products and map data are required; the driver reads an existing cache. These files are not included in the code archive.

## Figure 5 — tropospheric circulation and surface conditions

**Figure driver:** `src/A研究进展报告/ERA5_2025元旦磁暴研究报告/plot_figure5_height_wind_weather_180mm_20261108.py`

Related scripts include `plot_figure5_height_wind_weather_20261101.py`, `plot_actual_west_east_height_20261031.py`, `plot_figure5_snow_region_actual_20261031.py` and `plot_figure5_w2w3_troposphere_impact.py`. They support diagnostics and plotting of geopotential-height differences, meridional winds, horizontal temperature advection and near-surface temperature anomalies.

**Inputs:** ERA5 and NCEP–NCAR reanalysis data, regional products and map files are needed. The figure workflow reads intermediate files including `actual_height_fields.npz` and `snow_region_actual_fields.npz`. These inputs are not in the archive.

## Running and verification

The archive's `run_figures.py` identifies the four drivers. With the required data restored in the original expected locations, its command syntax is `python run_figures.py 2` (or `3`, `4`, `5`, `all`). Installing `requirements.txt` alone does not supply these inputs. See `REPRODUCIBILITY.md` for the previous rendering check and its limitations. Paths, source data and intermediate generation must be verified before describing this repository as a complete reproducibility release.
