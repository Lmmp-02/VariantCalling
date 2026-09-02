#!/usr/bin/env bash
set -euo pipefail

cd /home/lantik-deploy/jlazaro/projects/variantcalling

REF="data/2_reference/GRCh38_no_alt_plus_hs38d1/GRCh38_no_alt_plus_hs38d1.chr20_chr21.fa"
SAMPLE="HG005"
SCOPE_ID="chr20_chr21"
THREADS="${THREADS:-2}"
FORCE="${FORCE:-0}"

ILL_STEM="HG005.GRCh38.300x.chr20_chr21"
ONT_STEM="HG005.GRCh38.ONT_R104_sup_PAW87816_PAW88001.chr20_chr21"

ITEMS=(
  "hg005_chr20_chr21_harmonized_40x|illumina|data/1_input_bams/HG005/derived/chr20_chr21/harmonized_40x/Illumina/${ILL_STEM}.cov40x.s42.bam"
  "hg005_chr20_chr21_harmonized_40x|ont|data/1_input_bams/HG005/derived/chr20_chr21/harmonized_40x/ONT/${ONT_STEM}.cov40x.s42.bam"

  "hg005_chr20_chr21_coverage_study_20x|illumina|data/1_input_bams/HG005/derived/chr20_chr21/coverage_study/Illumina/${ILL_STEM}.cov20x.s42.bam"
  "hg005_chr20_chr21_coverage_study_20x|ont|data/1_input_bams/HG005/derived/chr20_chr21/coverage_study/ONT/${ONT_STEM}.cov20x.s42.bam"

  "hg005_chr20_chr21_coverage_study_10x|illumina|data/1_input_bams/HG005/derived/chr20_chr21/coverage_study/Illumina/${ILL_STEM}.cov10x.s42.bam"
  "hg005_chr20_chr21_coverage_study_10x|ont|data/1_input_bams/HG005/derived/chr20_chr21/coverage_study/ONT/${ONT_STEM}.cov10x.s42.bam"

  "hg005_chr20_chr21_coverage_study_5x|illumina|data/1_input_bams/HG005/derived/chr20_chr21/coverage_study/Illumina/${ILL_STEM}.cov5x.s42.bam"
  "hg005_chr20_chr21_coverage_study_5x|ont|data/1_input_bams/HG005/derived/chr20_chr21/coverage_study/ONT/${ONT_STEM}.cov5x.s42.bam"
)

echo "=== PREFLIGHT ==="
test -f "$REF" || { echo "ERROR: REF not found: $REF"; exit 1; }
test -f "${REF}.fai" || { echo "ERROR: REF index not found: ${REF}.fai"; exit 1; }

for item in "${ITEMS[@]}"; do
  IFS="|" read -r ds tech bam <<< "$item"

  echo
  echo "--- $ds / $tech"
  echo "BAM=$bam"

  test -f "$bam" || { echo "ERROR: BAM not found: $bam"; exit 1; }

  if [[ -f "${bam}.bai" || -f "${bam%.bam}.bai" ]]; then
    echo "BAI: OK"
  else
    echo "ERROR: BAI not found for: $bam"
    exit 1
  fi
done

echo
echo "Preflight OK. Submitting 8 DeepVariant full-call jobs..."
echo

prev_job=""

for item in "${ITEMS[@]}"; do
  IFS="|" read -r ds tech bam <<< "$item"

  dep_arg=()
  if [[ -n "$prev_job" ]]; then
    dep_arg=(--dependency=afterok:${prev_job})
  fi

  jid=$(sbatch --parsable \
    "${dep_arg[@]}" \
    --export=ALL,DATASET_ID="${ds}",TECH="${tech}",BAM="${bam}",SAMPLE="${SAMPLE}",SCOPE_ID="${SCOPE_ID}",REF="${REF}",THREADS="${THREADS}",FORCE="${FORCE}",CLEAN_INTERMEDIATE=1 \
    cluster/run_dv_full_call.sh)

  echo "${jid}  ${ds}  ${tech}"
  prev_job="${jid}"
done

echo
echo "Submitted safe chain. Last job: ${prev_job}"
