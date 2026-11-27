# Data sources

This repository's code archive does not include original observations, reanalysis files, processed data or figure source data. The sources below are entry points for the data used in the working package; the precise downloads, versions, processing steps, redistribution terms and minimal inputs for each figure still need to be verified.

- MLS version 6 ozone and temperature: https://avdc.gsfc.nasa.gov/pub/data/satellite/Aura/MLS/V06/L2GP/
- Geomagnetic indices: Dst via https://omniweb.gsfc.nasa.gov/form/dx1.html ; Kp via https://kp.gfz.de/en/data ; AE via https://wdc.kugi.kyoto-u.ac.jp/hapi/ ; PCN via https://ftp.space.dtu.dk/WDC/indices/pcn/ .
- MEPED/POES energetic-particle data: https://www.ncei.noaa.gov/data/poes-metop-space-environment-monitor/access/l1b/v01r00/ . The Figure 2 working package also uses a processed flux table.
- ERA5 pressure-level data: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels . Access is subject to Copernicus account and dataset conditions.
- NCEP–NCAR Reanalysis 1: https://psl.noaa.gov/data/reanalysis/ .
- Natural Earth map data: https://www.naturalearthdata.com/ .

The archive's `input_inventory.csv` lists possible non-code inputs present in the separate working package; it is not a verified minimum list and does not include the files themselves. Some inputs are derived `.npz` and `.csv` products. Their provenance and generation from original observations need further verification.
