# Data sources and reuse

This code repository does not currently redistribute the observation, reanalysis or derived inputs. The following are provider entry points; the exact versions, retrieval dates, temporal/spatial subsets and individual intermediate-product provenance are not yet fully verified.

- **NASA Aura MLS v6**, geopotential height, temperature and ozone: https://avdc.gsfc.nasa.gov/pub/data/satellite/Aura/MLS/V06/L2GP/ . NASA's Earth science data policy supports open sharing; identify and cite the actual products: https://science.nasa.gov/earth-science/earth-science-data/data-information-policy/ .
- **MEPED/POES-MetOp energetic particles**, NOAA NCEI: https://www.ncei.noaa.gov/data/poes-metop-space-environment-monitor/access/l1b/v01r00/ . Verify the exact product and provider terms for each processed electron-flux table.
- **Geomagnetic indices:** Dst via NASA OMNI https://omniweb.gsfc.nasa.gov/form/dx1.html ; Kp via GFZ https://kp.gfz.de/en/data ; AE via Kyoto https://wdc.kugi.kyoto-u.ac.jp/hapi/ ; PCN via DTU https://ftp.space.dtu.dk/WDC/indices/pcn/ . GFZ provides Kp under CC BY 4.0 and requests acknowledgement: https://kp.gfz.de/en/about-kp . Check the individual source and conditions for the other indices.
- **ERA5 pressure levels**, Copernicus Climate Data Store: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels . The Copernicus Products licence permits distribution and adaptation with conditions including attribution: https://xds.ecmwf.int/licences/licence-to-use-copernicus-products . Verify the precise licence and citation for the actual download and derivative files.
- **NCEP–NCAR Reanalysis 1**, NOAA PSL: https://psl.noaa.gov/data/reanalysis/ . Confirm the exact surface-temperature product, climatology construction and any dataset-specific terms.
- **Natural Earth** map data: https://www.naturalearthdata.com/ . Verify the exact boundary dataset and attribution before repackaging it.

For known processed inputs and the limits of the trace, see [INTERMEDIATES.md](INTERMEDIATES.md). `input_inventory.csv` lists files in a separate working package and is not a verified minimal download manifest.
