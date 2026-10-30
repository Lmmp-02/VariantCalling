# Classical paper — additional experiments

These scripts produce the seed and probability-fusion experiments of the classical paper
and the analysis files behind its tables and figures. They read the existing offline
datasets and checkpoints. They do not change the canonical pipeline or its outputs.

| Script | Hardware | What it does |
|---|---|---|
| `make_seed_configs.py` | any | Writes seed replicates of the four canonical configs. Only `seed`, `experiment_id`, `outputs.root_dir`, `notes` and, optionally, `hyperparams.input_features` change. |
| `dump_predictions.py` | GPU recommended | Writes one NPZ per evaluation set with, for every candidate, the DeepVariant probabilities, the probabilities of each checkpoint and the candidate metadata. |
| `analyse_predictions.py` | CPU | Computes per-view metrics, per-group errors, seed summaries, 1 Mb block-bootstrap confidence intervals and exact McNemar tests. Needs numpy and pandas only. |
| `run_paper_experiments.sh` | — | Runs the whole sequence on one machine, stage by stage. |

## Experiments

- **Seeds.** The four heads (Unimodal Illumina, Unimodal ONT, Hybrid-S, Hybrid-G) are
  trained with seeds 0–4. Seed 0 is the canonical run in `training/out/experiments/`;
  seeds 1–4 go to `training/out/experiments/paper_seeds/`.
- **Probability fusion.** The same four heads are trained on the DeepVariant genotype
  probabilities instead of the internal representations (`hyperparams.input_features:
  dv_probs`), with seeds 0–4. Everything else is unchanged.

This gives 40 checkpoints in total: 4 heads × 2 input types × 5 seeds.

## Model names

In the prediction dumps and the analysis CSVs, models are named as follows:

| Name | Meaning |
|---|---|
| `dv_illumina`, `dv_ont` | DeepVariant internal reference |
| `<head>` | representation head, seed 0 (canonical run) |
| `<head>__seed<N>` | representation head, seed N (1–4) |
| `<head>__dvprobs__seed<N>` | probability head, seed N (0–4) |

where `<head>` is `unimodal_illumina`, `unimodal_ont`, `hybrid_simple` or `hybrid_groupwise`.

## Running

Prerequisites: the singleton-locus offline datasets (built with `scripts/multimodal/`) and
the four canonical seed-0 checkpoints, trained with `training/scripts/train/` and the
configs in `training/configs/experiments/`.

```bash
bash training/scripts/paper/run_paper_experiments.sh resolve configs train dump analyse
# or
bash training/scripts/paper/run_paper_experiments.sh all
```

Set `DEVICE=cuda` (or `cpu`) to force a device. The script runs every step in sequence and
needs no job scheduler. On a cluster, the same commands can be wrapped in job scripts.

Outputs:

- `training/out/paper/predictions/<eval_set>__all.npz`: prediction dumps with all 40
  checkpoints, for `hg004_chr20_40x`, `hg005_40x`, `hg005_20x`, `hg005_10x` and `hg005_5x`;
- `training/out/paper/analysis/`: `view_metrics.csv`, `group_errors.csv`,
  `seed_summary.csv`, `group_seed_summary.csv`, `paired_tests.csv` and `analysis_meta.json`.

`analyse_predictions.py` reads every NPZ in the prediction folder. When several dumps exist
for the same evaluation set, it prefers `__all` over `__seeds0-4` over `__seed0`.

## Notes

- Confidence intervals and McNemar tests in `paired_tests.csv` use the seed-0 models.
- `--sklearn_prob_fusion` trains a preliminary probability-fusion baseline with scikit-learn.
  It is superseded by the `dv_probs` heads and was not used for the results in the paper.
- `training/scripts/call/call_checkpoint.py` does not yet read `input_features`, so the
  `dv_probs` checkpoints cannot be used in calling mode.
