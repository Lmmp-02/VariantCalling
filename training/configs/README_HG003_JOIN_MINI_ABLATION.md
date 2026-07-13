# HG003 chr20+chr21 join mini-ablation — Hybrid Groupwise 3C

This ablation compares two multimodal dataset construction policies while keeping the genomic split, model architecture, seed and hyperparameters fixed.

## Expected dataset layout

```text
data/4_out/datasets/multimodal/by_subject/
└── hg003_chr20_chr21_harmonized_40x/
    ├── outer/       # canonical singleton_locus multimodal dataset
    │   └── *.npz
    └── variant_key/ # Luisa's alternative multimodal join
        └── *.npz
```

Both folders must contain the canonical multimodal fields consumed by the current trainer, including at least:

- `ill_embeddings`
- `ont_embeddings`
- `label`
- `group`
- `mask_ill`
- `mask_ont`
- `chrom`
- `locus_start`

`variant_type` and `variant_type_mismatch` are strongly recommended for stratified reports, but the loader can fall back when they are absent.

## Split design

The split uses chromosome-aware 1 Mb genomic bins:

- 80 retained bins for train
- 10 retained bins for validation
- 10 retained bins for test
- 8 buffer bins dropped between blocks

The exact same bin file is reused for both datasets:

```text
training/configs/splits/bin_sources/hg003_chr20_chr21_mini_ablation_bins.json
```

Do not regenerate or edit the bin file between the two conditions.

## Resolve both splits

Run from the repository root:

```bash
python training/scripts/preprocess/resolve_split.py \
  --split_config training/configs/splits/hg003_chr20_chr21_mini_ablation__singleton_locus.json

python training/scripts/preprocess/resolve_split.py \
  --split_config training/configs/splits/hg003_chr20_chr21_mini_ablation__variant_key.json
```

This creates:

```text
training/configs/splits/resolved__hg003_chr20_chr21_mini_ablation__singleton_locus.json
training/configs/splits/resolved__hg003_chr20_chr21_mini_ablation__variant_key.json
```

These two resolved splits store dataset paths relative to the repository, so they can be generated on one machine and used from a clone located elsewhere. Launch training and evaluation from the repository root.

Check that each resolver summary reports `path_exists: yes` on the machine where the corresponding dataset currently exists.

## Local CPU training

Activate the environment first, then run:

```bash
python -u training/scripts/train/hybrid/train_hybrid.py \
  --experiment_config training/configs/experiments/hybrid_groupwise_hg003_join_mini_ablation__singleton_locus.json \
  --resolved_split training/configs/splits/resolved__hg003_chr20_chr21_mini_ablation__singleton_locus.json \
  --device cpu

python -u training/scripts/train/hybrid/train_hybrid.py \
  --experiment_config training/configs/experiments/hybrid_groupwise_hg003_join_mini_ablation__variant_key.json \
  --resolved_split training/configs/splits/resolved__hg003_chr20_chr21_mini_ablation__variant_key.json \
  --device cpu
```

Outputs are written to:

```text
training/out/experiments/hybrid_groupwise_hg003_join_mini_ablation__singleton_locus/
training/out/experiments/hybrid_groupwise_hg003_join_mini_ablation__variant_key/
```

## Local CPU evaluation

Evaluate the best validation-selected checkpoint on each condition's own test partition:

```bash
python -u training/scripts/eval/eval_checkpoint.py \
  --source checkpoint \
  --experiment_dir training/out/experiments/hybrid_groupwise_hg003_join_mini_ablation__singleton_locus \
  --checkpoint best.pt \
  --resolved_split training/configs/splits/resolved__hg003_chr20_chr21_mini_ablation__singleton_locus.json \
  --partition test \
  --batch_size 4096 \
  --device cpu \
  --out_dir training/out/reports/hg003_join_mini_ablation/singleton_locus \
  --name hybrid_groupwise_singleton_locus

python -u training/scripts/eval/eval_checkpoint.py \
  --source checkpoint \
  --experiment_dir training/out/experiments/hybrid_groupwise_hg003_join_mini_ablation__variant_key \
  --checkpoint best.pt \
  --resolved_split training/configs/splits/resolved__hg003_chr20_chr21_mini_ablation__variant_key.json \
  --partition test \
  --batch_size 4096 \
  --device cpu \
  --out_dir training/out/reports/hg003_join_mini_ablation/variant_key \
  --name hybrid_groupwise_variant_key
```

## Cluster training with Slurm

```bash
sbatch --export=ALL,\
EXPERIMENT_CONFIG=training/configs/experiments/hybrid_groupwise_hg003_join_mini_ablation__singleton_locus.json,\
RESOLVED_SPLIT=training/configs/splits/resolved__hg003_chr20_chr21_mini_ablation__singleton_locus.json,\
DEVICE=cuda \
training/runners/run_hybrid_experiment.sh

sbatch --export=ALL,\
EXPERIMENT_CONFIG=training/configs/experiments/hybrid_groupwise_hg003_join_mini_ablation__variant_key.json,\
RESOLVED_SPLIT=training/configs/splits/resolved__hg003_chr20_chr21_mini_ablation__variant_key.json,\
DEVICE=cuda \
training/runners/run_hybrid_experiment.sh
```

## Cluster evaluation with Slurm

```bash
sbatch --export=ALL,\
SOURCE=checkpoint,\
EXPERIMENT_DIR=training/out/experiments/hybrid_groupwise_hg003_join_mini_ablation__singleton_locus,\
CHECKPOINT=best.pt,\
RESOLVED_SPLIT=training/configs/splits/resolved__hg003_chr20_chr21_mini_ablation__singleton_locus.json,\
EVAL_PARTITION=test,\
DEVICE=cuda,\
OUT_DIR=training/out/reports/hg003_join_mini_ablation/singleton_locus,\
NAME=hybrid_groupwise_singleton_locus \
training/runners/run_eval_checkpoint.sh

sbatch --export=ALL,\
SOURCE=checkpoint,\
EXPERIMENT_DIR=training/out/experiments/hybrid_groupwise_hg003_join_mini_ablation__variant_key,\
CHECKPOINT=best.pt,\
RESOLVED_SPLIT=training/configs/splits/resolved__hg003_chr20_chr21_mini_ablation__variant_key.json,\
EVAL_PARTITION=test,\
DEVICE=cuda,\
OUT_DIR=training/out/reports/hg003_join_mini_ablation/variant_key,\
NAME=hybrid_groupwise_variant_key \
training/runners/run_eval_checkpoint.sh
```

## Comparison rule

Because the alternative join may retain a different candidate universe, always compare both model metrics and dataset composition:

- total rows in train/val/test
- class distribution
- group distribution (`1`, `10`, `11`)
- chromosome distribution
- macro-F1 and weighted-F1
- per-class precision/recall/F1
- results by availability group and variant type
