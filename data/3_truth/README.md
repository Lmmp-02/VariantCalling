# GIAB truth sets

GIAB small-variant benchmark VCFs and benchmark-region BED files are used as ground truth for labelled DeepVariant examples and evaluation.

The classical benchmark uses HG002, HG003, HG004 and HG005 on GRCh38 / NIST v4.2.1. Raw VCF/BED/index files are intentionally excluded from Git; lightweight source manifests are retained for provenance.

Use `scripts/data/download_giab_truthsets.sh` to download the supported truth sets. Keep BAMs, reference FASTA, truth VCFs and benchmark BEDs on the same genome build and contig naming convention.
