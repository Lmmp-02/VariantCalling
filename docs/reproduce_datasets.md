# Reproducing the paper datasets (Level 3)

Level 3 rebuilds the seven singleton-locus multimodal datasets used by the paper from public GIAB / ONT inputs. It is an HPC-class workflow: it performs BAM downsampling, DeepVariant `make_examples`, GPU-backed feature export and the multimodal join. The recipe is provided for reproducibility, but **has not been re-run end to end outside the original cluster**.

The seven dataset ids are:

| Sample | Condition | Dataset id | Paper comparison scope |
|---|---|---|---|
| HG002 | 40x | `hg002_chr20_chr21_harmonized_40x` | report only |
| HG003 | 40x | `hg003_chr20_chr21_harmonized_40x` | report only |
| HG004 | 40x | `hg004_chr20_chr21_harmonized_40x` | **chr20 only** (blocked test) |
| HG005 | 40x | `hg005_chr20_chr21_harmonized_40x` | chr20+chr21 |
| HG005 | 20x | `hg005_chr20_chr21_coverage_study_20x` | chr20+chr21 |
| HG005 | 10x | `hg005_chr20_chr21_coverage_study_10x` | chr20+chr21 |
| HG005 | 5x | `hg005_chr20_chr21_coverage_study_5x` | chr20+chr21 |

## Inputs

Run the recipe from the repository root. The large inputs are intentionally not versioned.

### Regional BAMs

`data/1_input_bams/README.md` is the source of truth for the local layout. The tracked metadata files record the public selections used by the benchmark. The source BAMs can be obtained with the existing downloader:

```bash
bash scripts/data/download_giab_regions.sh --sample HG002 --tech Illumina --regions chr20 chr21
bash scripts/data/download_giab_regions.sh --sample HG003 --tech Illumina --regions chr20 chr21
bash scripts/data/download_giab_regions.sh --sample HG004 --tech Illumina --regions chr20 chr21
bash scripts/data/download_giab_regions.sh --sample HG005 --tech Illumina --regions chr20 chr21

bash scripts/data/download_giab_regions.sh --sample HG002 --tech ONT --regions chr20 chr21 --runs PAW70337 PAW71238 --merge-ont
bash scripts/data/download_giab_regions.sh --sample HG003 --tech ONT --regions chr20 chr21 --runs PAY87794 PAY87954 --merge-ont
bash scripts/data/download_giab_regions.sh --sample HG004 --tech ONT --regions chr20 chr21 --runs PAY87778 PAY88428 --merge-ont
bash scripts/data/download_giab_regions.sh --sample HG005 --tech ONT --regions chr20 chr21 --runs PAW87816 PAW88001 --merge-ont
```

The Illumina sources are GIAB/NIST regional extractions. The ONT inputs are the two-run merged selections from ONT Open Data GIAB release 2025.01, as recorded in the tracked metadata and downsampling manifest.

Validate the resulting BAM/BAI pairs with:

```bash
bash scripts/data/validate_input_bams.sh
```

### Reference and truth sets

DeepVariant training mode expects these regional paths for each sample:

```text
data/2_reference/GRCh38_no_alt_plus_hs38d1/GRCh38_no_alt_plus_hs38d1.chr20_chr21.fa
data/2_reference/GRCh38_no_alt_plus_hs38d1/GRCh38_no_alt_plus_hs38d1.chr20_chr21.fa.fai
data/3_truth/<SAMPLE>/<SAMPLE>_chr20_chr21.truth.vcf.gz
data/3_truth/<SAMPLE>/<SAMPLE>_chr20_chr21.truth.vcf.gz.tbi
data/3_truth/<SAMPLE>/<SAMPLE>_chr20_chr21.confident.bed
```

See `data/2_reference/README.md` for the GRCh38 no-alt + hs38d1 reference and `data/3_truth/README.md` for GIAB v4.2.1 truth data. `scripts/data/download_giab_truthsets.sh` downloads the full truth sets; regional subsetting remains a separate preparation step. Keep BAM, reference, truth VCF and confident BED on the same build and contig naming convention.

### Downsampling environment

The existing `scripts/data/downsample_multimodal_benchmark.sh` activates a micromamba environment (default `hts`) containing `samtools` and `mosdepth`. Its `MICROMAMBA_BIN`, `CONDA_ENV`, `THREADS` and `FORCE` environment variables are passed through unchanged. The paper manifest fixes the downsampling seed at 42.

DeepVariant stages run in containers. `dv_make_examples.sh` defaults to `google/deepvariant:1.9.0`; `dv_export_features.sh` defaults to `google/deepvariant:1.9.0-gpu`. The DeepVariant preset is derived from `--tech`: Illumina uses `illumina_wgs`, ONT uses `ont_r104`. There is no `--preset` argument.

## Run the recipe

First inspect the complete plan. `--dry-run` is a pure planner: it does not invoke the downsampler, containers or join scripts and does not create temporary manifests or output directories.

```bash
bash scripts/reproduce_datasets.sh --dry-run
```

Run all seven datasets with Docker:

```bash
bash scripts/reproduce_datasets.sh --runtime docker
```

or use udocker on a cluster where Docker is unavailable:

```bash
bash scripts/reproduce_datasets.sh --runtime udocker
```

A single dataset can be selected without changing the canonical manifest:

```bash
bash scripts/reproduce_datasets.sh \
  --runtime udocker \
  --only hg005_chr20_chr21_coverage_study_10x
```

For each selected dataset the wrapper filters exactly the two required rows (Illumina + ONT) from `scripts/data/config/downsample_multimodal_benchmark.tsv`, then chains:

1. `scripts/data/downsample_multimodal_benchmark.sh`;
2. `scripts/deepvariant/dv_make_examples.sh --mode training` for Illumina and ONT;
3. `scripts/deepvariant/dv_export_features.sh` for Illumina and ONT;
4. `scripts/multimodal/dv_build_join_datasets_singleton_locus.sh`.

The wrapper deliberately excludes the HG004 20x/10x/5x rows that also exist in the benchmark TSV: those datasets are not part of the paper's seven archived Level 3 datasets.

## Check Table 7 group counts

After the joins exist, run:

```bash
python scripts/multimodal/check_group_counts.py
```

The check reads the `group` and `chrom` fields from the singleton-locus outer NPZ shards. Availability groups are 1 (ONT only), 10 (Illumina only) and 11 (both). The reference counts are read from the DeepVariant rows of `results/paper/analysis/group_errors.csv`, which is the source used for Table 7.

The scope is intentionally asymmetric:

- HG004 is restricted to **chr20 only**, matching the blocked test set used in the paper;
- HG005 uses chr20+chr21;
- HG002 and HG003 have no Table 7 reference and are reported without pass/fail comparison.

To inspect one dataset only:

```bash
python scripts/multimodal/check_group_counts.py \
  --only hg004_chr20_chr21_harmonized_40x
```

A non-zero exit status means a referenced HG004/HG005 group count differs from the published value or a required shard/reference field is missing.
