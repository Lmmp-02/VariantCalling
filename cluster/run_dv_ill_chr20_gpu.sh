#!/bin/bash
#SBATCH --job-name=dv_ill_chr20_gpu
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

# Intenta respetar la GPU asignada por Slurm si ya viene expuesta.
# Si no, deja lo que haya. Evita fijar manualmente una GPU concreta salvo depuración.
echo "=== ENV CHECK ==="
hostname
uname -m
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi || true
"$UDOCKER" ps -m
echo "================="

OUTDIR=data/4_out/deepvariant/HG003/Illumina/chr20/covorig_sorig_gpu
mkdir -p "$OUTDIR" "$OUTDIR/logs" "$OUTDIR/intermediate"

REF=/work/data/2_reference/GRCh38_no_alt_plus_hs38d1/GRCh38_no_alt_plus_hs38d1.chr20.fa
READS=/work/data/1_input_bams/HG003/Illumina/HG003.GRCh38.2x250.chr20.bam

EXAMPLES=/work/$OUTDIR/intermediate/make_examples.tfrecord@1.gz
GVCF_TF=/work/$OUTDIR/intermediate/gvcf.tfrecord@1.gz
CVO=/work/$OUTDIR/intermediate/call_variants_output.tfrecord.gz
VCF_OUT=/work/$OUTDIR/output.vcf.gz
GVCF_OUT=/work/$OUTDIR/output.g.vcf.gz

echo "=== STEP 1/3: make_examples (CPU) ==="
time "$UDOCKER" run \
  --volume="$REPO":/work \
  dv19 \
  /opt/deepvariant/bin/make_examples \
    --mode calling \
    --ref "$REF" \
    --reads "$READS" \
    --examples "$EXAMPLES" \
    --checkpoint /opt/models/wgs \
    --call_small_model_examples \
    --gvcf "$GVCF_TF" \
    --small_model_indel_gq_threshold 28 \
    --small_model_snp_gq_threshold 20 \
    --small_model_vaf_context_window_size 51 \
    --track_ref_reads \
    --trained_small_model_path /opt/smallmodels/wgs \
    --task 0

echo "=== STEP 2/3: call_variants (GPU) ==="
time "$UDOCKER" run \
  --volume="$REPO":/work \
  dv19gpu \
  /opt/deepvariant/bin/call_variants \
    --outfile "$CVO" \
    --examples "$EXAMPLES" \
    --checkpoint /opt/models/wgs

echo "=== STEP 3/3: postprocess_variants (CPU) ==="
time "$UDOCKER" run \
  --volume="$REPO":/work \
  dv19 \
  /opt/deepvariant/bin/postprocess_variants \
    --ref "$REF" \
    --infile "$CVO" \
    --outfile "$VCF_OUT" \
    --cpus 1 \
    --small_model_cvo_records /work/$OUTDIR/intermediate/make_examples_call_variant_outputs.tfrecord@1.gz \
    --gvcf_outfile "$GVCF_OUT" \
    --nonvariant_site_tfrecord_path "$GVCF_TF"

echo "=== OPTIONAL STEP 4/4: vcf_stats_report (CPU) ==="
time "$UDOCKER" run \
  --volume="$REPO":/work \
  dv19 \
  /opt/deepvariant/bin/vcf_stats_report \
    --input_vcf "$VCF_OUT" \
    --outfile_base /work/$OUTDIR/output

echo "Finished at: $(date)"