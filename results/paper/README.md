# Classical paper — result files

These files are the numerical source of the classical paper. Every table and figure with
results is generated from them (see `paper/`, added separately). They are kept in the
repository because the inputs needed to recompute them (prediction dumps, checkpoints and
offline datasets) are too large to version.

## `analysis/`

Written by `training/scripts/paper/analyse_predictions.py` from the prediction dumps of
the 40 checkpoints (4 heads × 2 input types × 5 training seeds) and the DeepVariant
internal reference. Evaluation sets: HG004 chr20 at 40× and HG005 chr20+chr21 at 40×,
20×, 10× and 5×. Model names follow `training/scripts/paper/README.md`.

| File | Content | Used for |
|---|---|---|
| `view_metrics.csv` | Accuracy, macro precision, recall and F1, and error count, per evaluation set, view and model | Tables 2 and 4, Supplementary Table 9, Figures 3 and 4 |
| `group_errors.csv` | Errors per availability group (1, 10, 11), error types, class support and confusion matrix | Table 3, Supplementary Table 7 |
| `paired_tests.csv` | Block-bootstrap confidence intervals (1 Mb blocks, 1,000 resamples) and exact McNemar tests, seed-0 models | Section 4.2 |
| `seed_summary.csv`, `group_seed_summary.csv` | Mean and standard deviation across seeds | Not used directly; the paper reports medians and ranges |
| `analysis_meta.json` | Bootstrap settings and the dumps that were read | — |

## `singleton_audit/`

Written by `scripts/multimodal/audit_singleton_loci_filter.py` on HG005 chr20+chr21 at
40× (training-mode exports). It quantifies the loci and candidate rows that the
singleton-locus rule keeps and excludes. Used for Supplementary Table 8.

The per-candidate file `singleton_loci_row_level.csv` (about 50 MB) is not included.

## Scope

All results are offline, candidate-level classifications on the singleton-locus candidate
set. They are not end-to-end variant-calling results. The coverage experiments use the
heads trained at 40×, without retraining.
