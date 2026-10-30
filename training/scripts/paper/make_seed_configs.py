#!/usr/bin/env python3
"""
Create seed-replicate experiment configs from the four canonical configs.

Only `seed`, `experiment_id`, `outputs.root_dir`, `notes` and (optionally)
`hyperparams.input_features` change; every other hyperparameter is copied
unchanged. Outputs go to a separate root so canonical runs are never touched.

    # embedding heads, extra seeds (canonical seed 0 already exists)
    python3 training/scripts/paper/make_seed_configs.py --seeds 1 2 3 4

    # DV-probability heads, all seeds
    python3 training/scripts/paper/make_seed_configs.py --input_features dv_probs --seeds 0 1 2 3 4
"""

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CANONICAL = [
    "unimodal_illumina_hg002_hg003_vs_hg004",
    "unimodal_ont_hg002_hg003_vs_hg004",
    "hybrid_simple_hg002_hg003_vs_hg004",
    "hybrid_groupwise_hg002_hg003_vs_hg004",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--input_features", choices=["embeddings", "dv_probs"], default="embeddings")
    ap.add_argument("--out_dir", default="training/configs/experiments/paper_seeds")
    ap.add_argument("--root_dir", default="training/out/experiments/paper_seeds")
    args = ap.parse_args()

    tag = "" if args.input_features == "embeddings" else "__dvprobs"
    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in CANONICAL:
        src = REPO_ROOT / "training/configs/experiments" / f"{name}.json"
        base = json.loads(src.read_text(encoding="utf-8"))
        for seed in args.seeds:
            cfg = json.loads(json.dumps(base))
            cfg["experiment_id"] = f"{name}{tag}__seed{seed}"
            cfg["seed"] = seed
            cfg.setdefault("hyperparams", {})["input_features"] = args.input_features
            cfg.setdefault("outputs", {})["root_dir"] = args.root_dir
            cfg["notes"] = (f"Paper replicate of {name} (seed={seed}, "
                            f"input_features={args.input_features}).")
            dst = out_dir / f"{cfg['experiment_id']}.json"
            dst.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
            print(dst.relative_to(REPO_ROOT) if dst.is_relative_to(REPO_ROOT) else dst)


if __name__ == "__main__":
    main()
