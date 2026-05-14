# Experiment Registry

This file records the interpretation of experiment outputs produced during the Variant Calling workstream.

## Join-policy naming convention

Two multimodal dataset conditions are considered in the project:

- `singleton_locus`: current controlled baseline condition.
- `candidate_key`: planned candidate-preserving condition.

At the time of this registry, existing multimodal experiments were produced using the `singleton_locus` condition.

## `singleton_locus` condition

Physical dataset subdir:

```text
data/4_out/datasets/multimodal/by_subject/<dataset_id>/outer/
```

Interpretation:

- The unit of fusion is the locus.
- Only loci with at most one Illumina candidate and at most one ONT candidate are retained.
- Illumina-only, ONT-only and Illumina+ONT rows can exist.
- Non-singleton loci are excluded before training/evaluation.
- Despite the folder name `outer/`, this condition is not a full candidate-preserving outer join.

Valid use:

- controlled transfer-learning evaluation;
- head-level comparison under a restricted candidate universe;
- multimodal late-fusion benchmarking under singleton-locus constraints;
- singleton-constrained VCF/hap.py evaluation.

Invalid use:

- claiming full candidate-universe recall;
- claiming complete biological variant coverage;
- interpreting hap.py recall as head-only performance.

## Existing experiments before explicit policy naming

The following experiments were produced before explicit join-policy naming was introduced. They should be interpreted as `singleton_locus` experiments.

```
training/out/experiments/
```

Known/expected experiment families:

```
unimodal_illumina_*
unimodal_ont_*
hybrid_simple_*
hybrid_groupwise_*
```

If a specific experiment contains a resolved split or resolved eval suite pointing to:

```
data/4_out/datasets/multimodal/by_subject/<dataset_id>/outer/
```

then it belongs to the `singleton_locus` condition unless explicitly documented otherwise.

## Policy-aware experiments

Future experiments should include the join policy in either:

- the experiment config path;
- the output directory;
- the resolved split/eval JSON;
- or all of the above.

Recommended layout:

```
training/out/experiments/singleton_locus/<experiment_name>/
training/out/experiments/candidate_key/<experiment_name>/
```

## Candidate-key condition

Planned physical dataset subdir:

```
data/4_out/datasets/multimodal/by_subject/<dataset_id>/candidate_key/
```

Interpretation:

- The unit of fusion is the candidate.
- Exact `variant_key` is used to match Illumina and ONT candidates.
- Same-locus but different candidate variants are retained as separate rows.
- Non-singleton loci are retained.
- This condition preserves the exported DeepVariant candidate universe more faithfully than `singleton_locus`.

Important limitation:

This condition is still bounded by the candidates generated/exported from DeepVariant. It should not be described as complete biological truth.

## Current status

- `singleton_locus` data path annotation: completed.
- `singleton_locus` explicit split/eval configs: completed.
- `resolve_split.py` policy-aware resolution: completed.
- `resolve_eval_suite.py` policy-aware resolution: completed.
- explicit singleton-locus join runners: completed.
- `candidate_key` join builder: pending.
- singleton-constrained VCF/hap.py evaluation: pending.