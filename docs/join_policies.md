# Join policies for multimodal DeepVariant-derived datasets

## singleton_locus

- Dataset subdir: `outer/`
- Unit of fusion: locus
- Retention rule: only loci with at most one Illumina candidate and at most one ONT candidate.
- Drops non-singleton loci.
- Intended use: controlled head-level benchmark and singleton-constrained VCF/hap.py evaluation.
- Not intended as a complete representation of the biological candidate universe.

## candidate_key

- Dataset subdir: `candidate_key/`
- Unit of fusion: `variant_key`
- Retention rule: preserve all exported candidates; match modalities only when the exact candidate key agrees.
- Same locus with different candidates is retained as separate rows.
- Intended use: candidate-preserving benchmark and more realistic bridge to VCF/hap.py.