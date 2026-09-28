# Multimodal join policies

## `singleton_locus` — current implementation

The current multimodal dataset is written under `data/4_out/datasets/multimodal/by_subject/<dataset_id>/outer/`.

- Unit of fusion: locus.
- Retention rule: at most one Illumina candidate and at most one ONT candidate at a locus.
- Non-singleton loci are excluded.
- The physical folder name `outer/` must not be interpreted as a candidate-complete outer join.
- Intended use: the controlled downstream-head benchmark and singleton-constrained calling evaluation.

## `variant_key` — planned extension

A future candidate-preserving join should use exported `variant_key` metadata to retain all candidates and match technologies only when candidate identity agrees. This builder is not implemented in the current repository.
