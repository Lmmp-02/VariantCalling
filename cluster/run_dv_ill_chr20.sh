#!/bin/bash
#SBATCH --job-name=dv_ill_chr20
#SBATCH --partition=da
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=/home/lantik-deploy/jlazaro/projects/variantcalling/out/%x_%j.log
#SBATCH --error=/home/lantik-deploy/jlazaro/projects/variantcalling/out/%x_%j.err

set -euxo pipefail

REPO=/home/lantik-deploy/jlazaro/projects/variantcalling
UDOCKER=$REPO/tools/udocker/udocker-1.3.17/udocker/udocker

cd "$REPO"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

echo "=== ENV CHECK ==="
hostname
uname -m
"$UDOCKER" ps -m
echo "================="

OUTDIR=data/4_out/deepvariant/HG003/Illumina/chr20/covorig_sorig
mkdir -p "$OUTDIR" "$OUTDIR/logs" "$OUTDIR/intermediate"

time "$UDOCKER" run \
  --volume="$REPO":/work \
  dv19 \
  /opt/deepvariant/bin/run_deepvariant \
    --model_type=WGS \
    --ref=/work/data/2_reference/GRCh38_no_alt_plus_hs38d1/GRCh38_no_alt_plus_hs38d1.chr20.fa \
    --reads=/work/data/1_input_bams/HG003/Illumina/HG003.GRCh38.2x250.chr20.bam \
    --output_vcf=/work/$OUTDIR/output.vcf.gz \
    --output_gvcf=/work/$OUTDIR/output.g.vcf.gz \
    --num_shards=${SLURM_CPUS_PER_TASK} \
    --intermediate_results_dir=/work/$OUTDIR/intermediate \
    --logging_dir=/work/$OUTDIR/logs \
    --vcf_stats_report=true

echo "Finished at: $(date)"