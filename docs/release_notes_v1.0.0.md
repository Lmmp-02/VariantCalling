# v1.0.0 — paper reproducibility companion

This release is the reproducibility companion for the paper *When Does a Second Sequencing Technology Help? Late Fusion of Technology-Specific DeepVariant Models under Reduced Coverage*.

The repository supports four reproduction levels:

| Level | Reproduces | Main input | Check |
|---|---|---|---|
| 0 | Paper tables and generated figures | Repository only | `python paper/make_all.py` followed by `git diff --exit-code paper/tables` |
| 1 | Analysis CSVs | `paper_predictions.tar` (622 MB) | `compare_with_reference.py --mode exact` |
| 2 | Classification heads, prediction dumps and analysis CSVs | Seven multimodal dataset archives (11.4 GB); GPU recommended | `compare_with_reference.py --mode tolerance` |
| 3 | Seven singleton-locus multimodal datasets | Public GIAB/ONT inputs, DeepVariant 1.9.0 and HPC-class compute | `scripts/multimodal/check_group_counts.py` |

The paper artefact archive contains the prediction dumps, the 40 classification-head checkpoints (4 canonical and 36 paper runs), and the seven singleton-locus multimodal datasets. The archive is licensed under CC BY 4.0 and is deposited on Zenodo under reserved DOI [`10.5281/zenodo.23804374`](https://doi.org/10.5281/zenodo.23804374). The code is released under the MIT licence.

The paper results are offline, candidate-level classifications on the singleton-locus candidate set. They are not an end-to-end variant-calling improvement over DeepVariant. HG004 is evaluated on chromosome 20 as the blocked internal test set; HG005 uses chromosomes 20 and 21. The 20×, 10× and 5× coverage experiments evaluate heads trained at 40× without retraining.

## Provenance

The reported paper results were produced from internal cluster revision `06a254f` and were subsequently ported to this public repository in the merged pull request *Add the paper's additional experiments and figure/table generation*. The measured training environment was Python 3.9.18, PyTorch 2.8.0 with a CUDA 12.8 build, NumPy 2.0.2, pandas 2.3.3 and Matplotlib 3.9.4. DeepVariant processing used version 1.9.0 with technology-specific models. Large archive metadata were sanitised before publication so that stored paths are repository-relative rather than private cluster paths.

See the root `README.md` for exact commands, hardware expectations and the four reproduction checks.
