# Reproducibility checks and scope

The separate working package assembled on 21 November 2026 was tested in a copy of the original project directory using Python 3.10.8 on Windows. Figures 2–5 were regenerated from a mixture of original inputs and previously processed `.npz`/`.csv` products. Original and rerun PNG dimensions matched. After resizing to 256 × 256, the mean absolute RGB differences were 0.0, 0.0001, 0.0 and 0.0 intensity units for Figures 2–5 respectively; Figures 4 and 5 were byte-identical, whereas Figures 2 and 3 were not.

Two portability changes were made in the packaged copies: the Figure 2 driver no longer imports a panel-alignment audit from a local author-specific tools directory; and a Figure 3 dependency uses a numerical tolerance in a comparison assertion (`rtol=1e-10`, `atol=1e-20`, `equal_nan=True`) after small floating-point differences were observed. These changes did not change the intended scientific calculations.

The working package includes inputs that are **not** in this repository. These checks do not establish a verified end-to-end workflow from the external data archives or prove that all figure inputs can be regenerated from the scripts in this archive. The input provenance, minimal data requirements, permissions and exact match to the final manuscript still require review.
