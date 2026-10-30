# VariantCalling — reproducibility companion

Companion repository for the paper *When Does a Second Sequencing Technology Help? Late Fusion of Technology-Specific DeepVariant Models under Reduced Coverage*.

**Authors:** [to be confirmed]

**Status:** preprint draft, October 2026

**Paper artefacts:** Zenodo record `23804374`, reserved DOI [`10.5281/zenodo.23804374`](https://doi.org/10.5281/zenodo.23804374)

This repository contains the code, experiment configuration and versioned numerical results needed to reproduce the paper at four levels. DeepVariant is kept technology-specific and unchanged; lightweight downstream heads are evaluated offline on a singleton-locus candidate set built from paired Illumina and Oxford Nanopore (ONT) data. These experiments are **candidate-level analyses, not end-to-end variant-calling improvements over DeepVariant**.

## Reproducibility levels

| Level | Reproduces | Inputs / download | Hardware and runtime | Entry point and check |
|---|---|---|---|---|
| **0** | Paper tables and generated figures | Repository only | CPU; a few seconds | `pip install -r paper/requirements.txt`<br>`python paper/make_all.py`<br>`git diff --exit-code paper/tables` |
| **1** | Analysis CSVs from the published prediction dumps | `paper_predictions.tar` — 622 MB | CPU; minutes | `scripts/fetch_paper_artifacts.sh --record 23804374 --what predictions`<br>`python training/scripts/paper/analyse_predictions.py --pred_dir training/out/paper/predictions --out_dir training/out/paper/analysis`<br>`python training/scripts/paper/compare_with_reference.py --analysis_dir training/out/paper/analysis --mode exact` |
| **2** | The 40 heads, prediction dumps and analysis CSVs | Seven multimodal dataset archives — 11.4 GB | GPU recommended; hours, hardware-dependent | `scripts/fetch_paper_artifacts.sh --record 23804374 --what datasets`<br>`bash training/scripts/paper/run_paper_experiments.sh all`<br>`python training/scripts/paper/compare_with_reference.py --analysis_dir training/out/paper/analysis --mode tolerance` |
| **3** | The seven singleton-locus multimodal datasets from public sequencing inputs | Public GIAB / ONT BAMs, reference and truth data; size depends on source preparation | HPC-class; not benchmarked end to end outside the original cluster | Prepare inputs as in [`docs/reproduce_datasets.md`](docs/reproduce_datasets.md), then `bash scripts/reproduce_datasets.sh --runtime docker`<br>`python scripts/multimodal/check_group_counts.py` |

The Zenodo record also contains `paper_checkpoints.tar` (169.6 MB), with the 4 canonical and 36 paper checkpoints. Archives are extracted at the repository root; internal paths are repository-relative. While the Zenodo record remains a draft, `scripts/fetch_paper_artifacts.sh` accepts a personal token through `ZENODO_TOKEN`. No token is required after publication.

## Quick start — Level 0

```bash
pip install -r paper/requirements.txt
python paper/make_all.py
git diff --exit-code paper/tables
```

Generated figures are written under `paper/figures/generated/`. The versioned LaTeX table rows under `paper/tables/` must remain unchanged.

## Repository map

| Path | Purpose | Documentation |
|---|---|---|
| `data/` | Input layout, reference/truth provenance and local generated data | [`data/1_input_bams/README.md`](data/1_input_bams/README.md), [`data/2_reference/README.md`](data/2_reference/README.md), [`data/3_truth/README.md`](data/3_truth/README.md), [`data/4_out/README.md`](data/4_out/README.md) |
| `docs/` | Reproduction notes for large-data workflows | [`docs/reproduce_datasets.md`](docs/reproduce_datasets.md) |
| `paper/` | Regeneration of paper figures and Tables 1–9 | [`paper/README.md`](paper/README.md) |
| `requirements/` | Pinned training / analysis environment | [`requirements/training.txt`](requirements/training.txt) |
| `results/` | Versioned numerical source of the paper | [`results/paper/README.md`](results/paper/README.md) |
| `scripts/` | Data preparation, DeepVariant extraction, multimodal joins and Zenodo fetching | [`docs/reproduce_datasets.md`](docs/reproduce_datasets.md) |
| `studies/` | Supporting and historical experiments retained for provenance | [`studies/hg003_chr20_initial_validation/README.md`](studies/hg003_chr20_initial_validation/README.md) |
| `training/` | Head configs, training, prediction dumps and paper analysis | [`training/scripts/paper/README.md`](training/scripts/paper/README.md) |
| `.github/` | Level 0 continuous integration | [`.github/workflows/reproduce-level0.yml`](.github/workflows/reproduce-level0.yml) |

## Scope and limitations

The main experiments cover chromosomes 20 and 21 in HG002, HG003, HG004 and HG005; three of the four GIAB individuals are from one family. HG004 evaluation uses chr20 as the blocked test set, while HG005 uses chr20+chr21. The multimodal join uses a **singleton-locus** rule: loci with more than one candidate in either technology are excluded. The lightweight heads are trained at 40×; the 20×, 10× and 5× experiments evaluate those same heads without retraining. GPU retraining is not expected to be bit-exact, so Level 2 is checked against the published seed ranges rather than exact floating-point outputs. Level 3 is an HPC reproduction recipe and has not been re-run end to end outside the original cluster.

The repository does not provide a candidate-preserving `variant_key` join or a synchronised VCF-export path for these paper heads. Neither is required for the reported offline candidate-level results.

## Data sources

The benchmark uses GIAB HG002–HG005 Illumina data and GIAB v4.2.1 truth/confident regions on GRCh38, plus ONT Open Data GIAB release 2025.01. Exact selections, run identifiers and expected local paths are recorded under `data/` and in [`docs/reproduce_datasets.md`](docs/reproduce_datasets.md). DeepVariant stages use version 1.9.0; the technology-specific preset is selected by the existing wrapper from `--tech`.

## Citation and licences

Code in this repository is released under the [MIT licence](LICENSE). The Zenodo dataset is released under **CC BY 4.0** and has reserved DOI [`10.5281/zenodo.23804374`](https://doi.org/10.5281/zenodo.23804374). Citation metadata for the software and preferred paper citation are in [`CITATION.cff`](CITATION.cff); author and paper-DOI fields remain placeholders until the manuscript metadata is final.
