# Figure-by-figure guide

Run commands below from the repository root after installing the required environment and placing inputs at the relative paths expected by the code. The paths are listed to identify the scripts, not to imply the repository is runnable without missing data. Outputs are created under the existing `src/` hierarchy. No complete raw-data-to-output run was repeated for this repository.

## Figure 2 — geomagnetic indices, energetic electrons and MLS observations

- Run: `python run_figures.py 2`.
- Driver: `src/元旦磁暴_跨层波数1/two_region_mechanism_profiles_20261021/plot_figure2_temperature_dailymean_20261121.py`.
- Purpose: combines the Dst, Kp, AE and PCN indices, MEPED electron flux and MLS nighttime ozone and temperature panels. The driver calculates daily nighttime MLS temperature means and delegates other panels to existing plotting/processing modules.
- Key related scripts in the same directory: `plot_figure1_f_min30_20261109.py`, `plot_figure2_indices_mainphase_20261101.py`, `plot_figure2_Dec28_Jan11_20261022.py`.
- Inputs: daily MLS v6 temperature files (28 December 2024–11 January 2025) in the paths specified by `source_files()`; preprocessed `night_ozone_temperature_native_daily.csv`, `MEPED_plotted_values.csv`, `AE_PCN_hourly.csv`, `Kp_3hourly_GFZ.csv` and `Dst_hourly_OMNI.csv`, plus related inputs. See [INTERMEDIATES.md](INTERMEDIATES.md).
- Outputs: PNG/SVG figure and daily temperature tables in `src/元旦磁暴_跨层波数1/two_region_mechanism_profiles_20261021/figure2_temperature_dailymean_20261121/`.
- Scope: all panels were rendered in the previous working-package test using a mixture of raw and preprocessed inputs. Independent regeneration of every preprocessed input has not been checked.

## Figure 3 — atmospheric environment and planetary waves

- Run: `python run_figures.py 3`.
- Driver: `src/元旦磁暴_跨层波数1/plot_combined_environment_w1_ep_qy_legend_bottom_20261109.py`.
- Purpose: assembles the atmospheric-environment, W1 amplitude and heat-flux proxy, and EP-flux diagnostic panels.
- Key modules: `plot_figure3_flux_amplitude_EP_nature_20261105.py`, `plot_daily_W1_EP_directions_20261010.py`, `ep_flux_display_20261011.py`, `build_acp2018_event_figures_20261010.py`, and `two_region_mechanism_profiles_20261021/plot_mechanism_chain_2x3_20261022.py`.
- Inputs: MLS-derived wave fields, background data, daily W1 EP fields and precomputed amplitude/proxy tables; exact files and derivations are described in [INTERMEDIATES.md](INTERMEDIATES.md).
- Outputs: PNG/SVG and plotting settings under `src/元旦磁暴_跨层波数1/combined_environment_W1_EP_qy_legend_bottom_20261109/`.
- Scope: previous rendering used precomputed input fields; the upstream wave and background inputs have not all been regenerated from original MLS files.

## Figure 4 — wave phase and polar-vortex envelope

- Run: `python run_figures.py 4`.
- Driver: `src/A研究进展报告/ERA5_2025元旦磁暴研究报告/plot_figure4_phase_envelope_180mm_aligned_20261107.py`.
- Purpose: plots wave phase, vortex envelope and same-level wave-component reconstructions.
- Key modules: `plot_figure4_wave_vortex_reconstruction_20261027.py`, `figure4_full_depth_data_20261030.py`, `plot_era5_daily_vortex_boundaries.py` and `analyze_w1_w2_vortex_20261025.py` in the same directory.
- Inputs: ERA5-derived `daily_vortex_diagnostics.npz`, MLS wave fields, supporting tables and map boundaries. See [INTERMEDIATES.md](INTERMEDIATES.md).
- Outputs: PNG/SVG and diagnostics under `src/A研究进展报告/ERA5_2025元旦磁暴研究报告/results/Figure4_phase_envelope_180mm_aligned_20261107/`.
- Scope: the previous figure rendering used existing caches; the complete derivation of all inputs from original ERA5 and MLS datasets remains to be checked.

## Figure 5 — circulation, cold-air transport and surface temperature

- Run: `python run_figures.py 5`.
- Driver: `src/A研究进展报告/ERA5_2025元旦磁暴研究报告/plot_figure5_height_wind_weather_180mm_20261108.py`.
- Purpose: plots geopotential-height differences, meridional winds, 850-hPa horizontal temperature advection and 2-m temperature anomalies.
- Key modules: `plot_figure5_height_wind_weather_20261101.py`, `plot_actual_west_east_height_20261031.py`, `plot_figure5_snow_region_actual_20261031.py` and `plot_figure5_w2w3_troposphere_impact.py`.
- Inputs: ERA5-derived `actual_height_fields.npz`, `snow_region_actual_fields.npz` and `aligned_daily_fields.npz`; surface-temperature anomaly cache; boundary table and map assets. The old snow-region module name is historical and does not mean snow markers are displayed in this final driver.
- Outputs: PNG/SVG and figure settings under `src/A研究进展报告/ERA5_2025元旦磁暴研究报告/results/Figure5_height_wind_weather_180mm_20261108/`.
- Scope: previous rendering used cached products; the complete raw reanalysis-to-cache workflow has not been independently reproduced.

For all four figures, the previous test used a separate working package with additional files. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md); the present repository currently cannot execute the four commands on its own.
