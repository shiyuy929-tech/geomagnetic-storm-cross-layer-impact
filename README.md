# Figure 2–5 code — review draft

**Status: local review draft, not a complete reproducibility release.**

This folder contains Python scripts and an environment file copied from the 2026-11-21 working package. Data, cached arrays, maps and figure outputs have deliberately not been copied. Consequently `run_figures.py` will not run successfully from this folder alone. `input_inventory.csv` lists possible non-code inputs present in the original working package; it is an inventory, not a verified minimal dependency list.

## Entry points

- Figure 2: `src/元旦磁暴_跨层波数1/two_region_mechanism_profiles_20261021/plot_figure2_temperature_dailymean_20261121.py`
- Figure 3: `src/元旦磁暴_跨层波数1/plot_combined_environment_w1_ep_qy_legend_bottom_20261109.py`
- Figure 4: `src/A研究进展报告/ERA5_2025元旦磁暴研究报告/plot_figure4_phase_envelope_180mm_aligned_20261107.py`
- Figure 5: `src/A研究进展报告/ERA5_2025元旦磁暴研究报告/plot_figure5_height_wind_weather_180mm_20261108.py`

## Before sharing

1. Check each figure against the final manuscript and establish code ownership/licences.
2. Trace each cached intermediate to its source and the corresponding processing script.
3. Determine redistributable inputs and exact download instructions for externally hosted data.
4. Check local paths, credentials, personal information and third-party assets.
5. Set up a clean environment and reproduce the outputs from documented inputs; record limitations.
6. Decide a licence with the code owners and select a repository release/version.

The existing `Fig2-5_code_package_20261121` working package records previous figure regeneration using a mixture of original and preprocessed inputs; the complete raw-data-to-results workflow has not been verified.
