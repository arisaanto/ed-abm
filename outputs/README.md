# Outputs

## `findings/`

Publication-ready aggregate findings. Figures are retained as PDF for
Overleaf/arXiv and PNG for repository previews. Tables are retained as TeX for
the paper, CSV for machine-readable reuse, and Markdown for direct review.

## `source_results/`

Local-only completed run trees and final result archives used by the figure and
table builders. This directory is excluded from version control because it is
large and may contain detailed run-level records.

The frozen Part 1 and Part 2 source trees and the final Part 3 archives belong
here. Ordinary documentation or figure edits should rebuild from these sources;
the completed simulations and LLM inference should not be rerun.
