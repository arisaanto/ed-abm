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

The final questionnaire-audit archive is also retained here locally. Only its
authorized aggregate convergence table is tracked under `findings/`; raw
hospital questionnaire records remain outside the repository.

The local source inventory is deliberately limited to:

- extracted Part 1 validation and Part 2 intervention run trees;
- the Part 2 parameter-sensitivity ZIP used directly by the Part 2 builder;
- the Part 3 closed-loop, appraisal, and architecture-ablation archives used
  directly by the Part 3 builder; and
- the final questionnaire-audit archive retained as local aggregate-analysis
  provenance.

These packages are not redundant with `findings/`: the latter contains only
reported aggregates and cannot regenerate or re-audit the analyses by itself.
