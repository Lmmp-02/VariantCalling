# Input BAMs

This directory is the logical location for GIAB regional BAMs used by the project. BAM/BAI files are intentionally excluded from Git; the tracked JSON files provide lightweight provenance for the selected public sources and the derived coverage conditions.

Core samples are HG002, HG003, HG004 and HG005, with paired Illumina and ONT data over chr20+chr21. ONT benchmark inputs use the merged two-run policy recorded in the metadata and benchmark configuration.

Use:

- `scripts/data/download_giab_regions.sh` to obtain the regional source BAMs.
- `scripts/data/validate_input_bams.sh` for basic validation.
- `scripts/data/downsample_multimodal_benchmark.sh` with `scripts/data/config/downsample_multimodal_benchmark.tsv` to generate the harmonized 40x and low-coverage benchmark conditions.

Derived BAMs remain local; compact mosdepth summaries are retained under `data/4_out/mosdepth_multimodal_benchmark/`.
