# HG003 chr20 initial validation

This directory preserves the initial HG003 chromosome 20 proof-of-concept used to validate the feasibility of late multimodal fusion of DeepVariant representations.

It is a **historical side study**, not part of the current canonical chr20+chr21 multi-subject pipeline. The experiment was produced with an earlier implementation of dataset preparation, training, and evaluation. In particular, it used the historical singleton-only `OUTER` layout, a fixed genomic-bin split, and a sidecar metadata CSV to attach labels and variant-type information during training/evaluation.

`variant_type` was used for metadata/reporting and SNP/INDEL stratification; it was not an input feature to the classifiers.

## Preserved artifacts

- `split/split_bins.json`: exact genomic-bin split used by the experiment.
- `evaluation/`: normalized DeepVariant, unimodal, simple-hybrid and groupwise-hybrid evaluation reports.
- `results/hg003_summary_internal_test_long.csv`: compact normalized result table.
- `historical_outputs/view_specific/`: original hybrid evaluation outputs used to reconstruct the normalized reports.
- `historical_pipeline/`: archived preprocessing, training and evaluation scripts used by the earlier implementation. These are preserved for provenance and are not maintained as current entrypoints.
- `reporting/`: scripts used to normalize the historical outputs.

The normalized files do **not** represent retraining with the current pipeline. They reformat the original HG003 chr20 outputs so they can be reported consistently alongside later experiments; the underlying predictions and metrics are unchanged.

The main project results come from the newer multi-subject chr20+chr21 workflow. This directory should be interpreted only as initial feasibility / proof-of-concept evidence.

## Verification

From this directory:

```bash
bash regenerate_reports.sh
```

This reconstructs the normalized hybrid reports from the historical view-specific outputs, rebuilds the summary table, and verifies that it exactly matches the committed result.
