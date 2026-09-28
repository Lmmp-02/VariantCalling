# Variant Calling

Classical multimodal variant-calling research pipeline built around DeepVariant representations from paired Illumina and Oxford Nanopore (ONT) data.

The canonical pipeline uses DeepVariant as a feature extractor: each technology is processed independently with `make_examples` and feature export, the resulting representations are fused at the locus level into a `singleton_locus` multimodal dataset, and downstream classification heads are trained and evaluated offline.

## Happy path

1. **Prepare data** — `scripts/data/`
2. **Generate DeepVariant examples and latent features** — `scripts/deepvariant/`
3. **Build the Illumina + ONT multimodal dataset** — `scripts/multimodal/`
4. **Train and evaluate downstream heads** — `training/`
5. **Inspect curated evaluation tables** — `results/`

`scripts/baselines/` contains the full DeepVariant end-to-end baseline, while `studies/deepvariant_low_coverage/` contains the supporting low-coverage study used to motivate the benchmark.

## Repository scope

Large genomic inputs, TFRecords, offline dataset shards, checkpoints, runtime manifests and HPC-specific launch wrappers are intentionally not versioned. The repository keeps reusable code, lightweight source/coverage metadata, experiment configuration and curated result tables.

The current multimodal join is `singleton_locus`: loci with multiple candidates in either technology are excluded, so this is not a candidate-complete replacement for DeepVariant. A candidate-preserving `variant_key` join remains a planned extension.

VCF export is handled by a separate downstream step; the corresponding exporter still needs to be synchronized into this public repository.
