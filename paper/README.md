# Classical paper — figures and tables

This folder regenerates the figures and tables of the paper *When Does a Second
Sequencing Technology Help? Late Fusion of Technology-Specific DeepVariant Models under
Reduced Coverage* from the result files in this repository.

```bash
pip install -r paper/requirements.txt
python paper/make_all.py
```

No dataset, checkpoint or GPU is needed. The run takes a few seconds.

## Where each result comes from

| Paper | Script | Input |
|---|---|---|
| Figure 1 (workflow) | schematic: `figures/schematics/fig1_overview.tex` (TikZ) | — |
| Figure 2 (head architectures) | schematic: `figures/schematics/fig2_heads_architecture.drawio` | — |
| Figure 3 (macro-F1 at 40×) | `scripts/make_figures.py` | `results/paper/analysis/view_metrics.csv` |
| Figure 4 (coverage) | `scripts/make_figures.py` | `results/paper/analysis/view_metrics.csv` |
| Table 1 (HG003 probes) | `studies/hg003_chr20_initial_validation/` | that study's evaluation reports |
| Table 2 (errors at 40×) | `scripts/make_tables.py` | `view_metrics.csv` |
| Table 3 (errors by group) | `scripts/make_tables.py` | `group_errors.csv` |
| Table 4 (input types) | `scripts/make_tables.py` | `view_metrics.csv` |
| Table 5 (two-stage, HG003) | `studies/hg003_chr20_initial_validation/` | that study's evaluation reports |
| Table 6 (end-to-end DeepVariant) | `scripts/make_tables.py` | `studies/deepvariant_low_coverage/results/benchmark_summary.csv` |
| Table 7 (group composition) | `scripts/make_tables.py` | `group_errors.csv` (class support of the DeepVariant rows) |
| Table 8 (singleton-locus audit) | `scripts/make_tables.py` | `results/paper/singleton_audit/` |
| Table 9 (results per seed) | `scripts/make_tables.py` | `view_metrics.csv` |
| Bootstrap intervals and McNemar tests (Section 4.2) | — | `results/paper/analysis/paired_tests.csv` |

Outputs:

- `figures/generated/fig3{a-d}_*.pdf/png` and `fig4{a-d}_*.pdf/png` (not versioned);
- `tables/tab*.tex`: the rows between `\midrule` and `\bottomrule` of each table. These
  files are versioned, so `git diff paper/tables/` after a run shows any change.

## Conventions

- Values are medians over five training seeds, with the range [min–max] where shown.
- Figure 3: marker = median, bar = range. Figure 4 A–B: line = median, band = range.
  Figure 4 C–D: gain of each fusion head over the unimodal head with the same input and
  seed; marker = median, bar = interquartile range; solid = representations, dashed =
  genotype probabilities.
- Confidence intervals and McNemar tests use the seed-0 models.

## How the result files were produced

The files in `results/paper/` come from `training/scripts/paper/` (see its README), run on
the singleton-locus offline datasets. Recomputing them needs those datasets and a GPU;
regenerating the figures and tables from them does not.

## Scope

All custom-model results are offline, candidate-level classifications on the
singleton-locus candidate set, not end-to-end variant calls. The scaled experiments cover
chromosomes 20 and 21 only. The coverage experiments use the heads trained at 40×,
without retraining.
