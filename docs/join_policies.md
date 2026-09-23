# Políticas de unión de datasets multimodales

Las salidas actuales bajo `data/4_out/datasets/multimodal/by_subject/<dataset_id>/outer/` corresponden a `singleton_locus`. El nombre `outer` describe la combinación de tecnologías, pero **no** incluye todos los candidatos exportados: se excluyen los loci con varios candidatos en cualquiera de las modalidades.

## singleton_locus

- Dataset subdir: `outer/`
- Unit of fusion: locus
- Retention rule: only loci with at most one Illumina candidate and at most one ONT candidate.
- Drops non-singleton loci.
- Intended use: controlled head-level benchmark and singleton-constrained VCF/hap.py evaluation.
- Not intended as a complete representation of the biological candidate universe.

## variant_key (experimental)

- Dataset subdir previsto: `variant_key/`.
- Unit of fusion: `variant_key`
- Retention rule: preserve all exported candidates; match modalities only when the exact candidate key agrees.
- Same locus with different candidates is retained as separate rows.
- Uso previsto: comparación que conserva candidatos y facilita el puente hacia VCF/hap.py.
