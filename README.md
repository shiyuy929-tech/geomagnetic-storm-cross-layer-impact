# Geomagnetic storm cross-layer study: analysis and figure code

Python scripts used for the working versions of Figures 2–5 are available in `src/`. The original directory hierarchy is preserved because the scripts import one another and resolve input files relative to these paths. `run_figures.py` selects a figure driver. See [FIGURE_GUIDE.md](FIGURE_GUIDE.md) for entry points, purpose, known inputs and outputs, and [INTERMEDIATES.md](INTERMEDIATES.md) for provenance and sharing status.

**Current reproducibility scope:** this repository includes scripts, but not the observation files, reanalysis data and processed caches needed to execute them. A separate working package was previously used to reproduce the four figures from a mixture of raw and processed inputs. The complete raw-to-result workflow and the exact minimal input set are not yet verified. Figure 1, supplementary figures and tables are outside this package. The script filenames describe working versions and should be checked against the final manuscript.

The four figure drivers are invoked from the repository root using `python run_figures.py 2` (or `3`, `4`, `5`, `all`). These commands require inputs that are currently **not included**. `requirements.txt` is the recorded package list; the environment and input placement must be checked before a fresh run. The earlier rendering check used Windows and Python 3.10.8; see [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

External datasets, use conditions and acknowledgements are described in [DATA_SOURCES.md](DATA_SOURCES.md). `input_inventory.csv` is an inventory of possible files in a separate working package, not a verified dependency list. No code licence has been assigned; code owners need to decide how the scripts may be reused.
