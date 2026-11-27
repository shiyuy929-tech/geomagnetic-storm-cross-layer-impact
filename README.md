# Analysis code for the geomagnetic-storm cross-layer study

This repository provides a working code package for Figures 2–5 and an index of the scripts used for the analyses and figure production. The source scripts retain their original directory structure inside `Fig2-5_GitHub_draft_20261127.zip`.

**Current scope:** the archive contains scripts and an environment file, but not the observational data, reanalysis files, cached arrays or processed tables required to run them. The figure scripts cannot be run from this repository alone. A separate working package was used to regenerate Figures 2–5 from a mixture of original and previously processed inputs. The full workflow from source archives to every intermediate has not been verified. Figure 1 and any supplementary figures or tables are not covered by this code package.

## Repository contents

- `Fig2-5_GitHub_draft_20261127.zip`: source scripts, figure runner, environment file and an inventory of possible inputs. The archive must be extracted to inspect the code.
- `FIGURE_GUIDE.md`: figure-by-figure entry points, associated analysis and known input requirements.
- `DATA_SOURCES.md`: external data sources and the scope of the missing inputs.
- `REPRODUCIBILITY.md`: checks carried out with the separate working package and their limitations.
- `requirements.txt`: packages recorded for the working environment.

## Code structure and figure runner

After extracting the archive, keep its `src` directory structure intact. The main entries and analysis purposes are documented in `FIGURE_GUIDE.md`. The runner accepts `python run_figures.py 2`, `3`, `4`, `5` or `all`; **these commands require missing data and intermediates**, so they will not work with the archive alone. The input inventory in the archive is a list of possible inputs, not a verified minimal dependency list or a set of download instructions.

The working package was tested with Python 3.10.8 on Windows using the environment in `requirements.txt`. The scripts are being released as a working research package; no licence is specified pending confirmation by the code owners. Check the documented data access and reuse conditions before redistributing third-party materials.
