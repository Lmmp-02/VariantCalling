#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dump per-candidate predictions for the classical paper analyses.

One compact NPZ per (resolved split, partition) with, for every candidate row
of the singleton_locus multimodal dataset (groups 1 / 10 / 11):

    chrom, locus_start, group, label, variant_type, mask_ill, mask_ont,
    dv_ill_probs, dv_ont_probs              (DeepVariant internal reference)
    probs__<model_name>                     (one array per checkpoint)

Probabilities are float32 with NaN for rows outside a model's view
(e.g. unimodal Illumina on ONT-only rows).

Use cases (all computed offline, after this dump):
  * paired block-bootstrap confidence intervals and McNemar tests;
  * per-group (1 / 10 / 11) error decomposition;
  * probability-level late-fusion baseline (train on DV probs + masks).

The script only reads existing datasets and checkpoints; it does not modify
any canonical output.

Examples
--------
# Test partition of a final-eval split, with checkpoints
python training/scripts/paper/dump_predictions.py \
  --resolved_split training/out/resolved/hg005_trainmode_5x_final_eval.json \
  --partition test \
  --model unimodal_illumina=training/out/experiments/unimodal_illumina_hg002_hg003_vs_hg004 \
  --model hybrid_simple=training/out/experiments/hybrid_simple_hg002_hg003_vs_hg004 \
  --out_npz training/out/paper/predictions/hg005_5x_test.npz

# Train / val partitions, DeepVariant outputs only (no embeddings loaded)
python training/scripts/paper/dump_predictions.py \
  --resolved_split training/out/resolved/hg002_hg003_train__hg004_valtest__singleton_locus.json \
  --partition train --teacher_only \
  --out_npz training/out/paper/predictions/train_teacher_only.npz
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.scripts.train.shared_data import (  # noqa: E402
    _as_str_array,
    build_partition_index_hybrid,
    dv_log_probs,
    load_resolved_split,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits.astype(np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


def teacher_probs(npz, rows: np.ndarray, prefix: str) -> np.ndarray:
    pk, lk = f"{prefix}_probs", f"{prefix}_logits"
    if pk in npz.files:
        return npz[pk].astype(np.float32, copy=False)[rows]
    if lk in npz.files:
        return softmax_np(npz[lk][rows])
    raise SystemExit(f"[ERROR] Missing {pk}/{lk} in shard.")


def parse_models(specs: List[str]) -> List[Tuple[str, Path]]:
    out = []
    for s in specs:
        if "=" not in s:
            raise SystemExit(f"[ERROR] --model expects name=experiment_dir, got: {s}")
        name, d = s.split("=", 1)
        p = (REPO_ROOT / d).resolve() if not Path(d).is_absolute() else Path(d)
        if not (p / "best.pt").exists():
            raise SystemExit(f"[ERROR] best.pt not found in {p}")
        out.append((name.strip(), p))
    return out


def load_models(models: List[Tuple[str, Path]], device):
    import torch
    sys.path.insert(0, str(REPO_ROOT / "training" / "scripts" / "eval"))
    from eval_checkpoint import (  # noqa: E402
        build_checkpoint_model,
        infer_checkpoint_model_info,
        load_experiment_metadata,
    )

    loaded = []
    for name, exp_dir in models:
        meta = load_experiment_metadata(exp_dir)
        ckpt = torch.load(exp_dir / "best.pt", map_location="cpu", weights_only=False)
        info = infer_checkpoint_model_info(ckpt, meta["config"])
        model = build_checkpoint_model(
            model_kind=info["model_kind"],
            model_family=info["model_family"],
            in_dim=info["input_dim"],
            n_classes=info["n_classes"],
            arch=info["arch"],
            hidden=info["hidden"],
            dropout=info["dropout"],
        )
        model.load_state_dict(ckpt["model_state"])
        model.to(device).eval()
        loaded.append({"name": name, "dir": str(exp_dir), "info": info, "model": model,
                       "best_epoch": ckpt.get("best_epoch")})
        print(f"[Model] {name}: kind={info['model_kind']} family={info['model_family']} "
              f"features={info.get('input_features', 'embeddings')} "
              f"modality={info['modality']} in_dim={info['input_dim']} best_epoch={ckpt.get('best_epoch')}")
    return loaded


def run_model(m, base, mask_ill, mask_ont, group, device, batch_size) -> np.ndarray:
    """base: dict with 'ill'/'ont' (embeddings) and 'p_ill'/'p_ont' (masked DV log-probs)."""
    import torch
    info = m["info"]
    n = group.shape[0]
    out = np.full((n, 3), np.nan, dtype=np.float32)
    probs_in = info.get("input_features", "embeddings") == "dv_probs"
    ill = base["p_ill"] if probs_in else base.get("ill")
    ont = base["p_ont"] if probs_in else base.get("ont")

    if info["model_kind"] == "unimodal":
        if info["modality"] == "illumina":
            sel, x_all = np.nonzero(mask_ill == 1)[0], ill
        else:
            sel, x_all = np.nonzero(mask_ont == 1)[0], ont
        feats = lambda idx: x_all[idx]
    else:
        sel = np.arange(n)
        add_masks = bool(info["add_masks"])

        def feats(idx):
            parts = [ill[idx], ont[idx]]
            if add_masks:
                parts.append(np.stack([mask_ill[idx], mask_ont[idx]], axis=1).astype(np.float32))
            return np.concatenate(parts, axis=1)

    with torch.no_grad():
        for i in range(0, sel.size, batch_size):
            idx = sel[i:i + batch_size]
            x = torch.from_numpy(np.ascontiguousarray(feats(idx), dtype=np.float32)).to(device)
            if info["model_family"] == "hybrid_groupwise_3c":
                g = torch.from_numpy(group[idx].astype(np.int64)).to(device)
                logits = m["model"](x, g)
            else:
                logits = m["model"](x)
            out[idx] = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
    return out


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolved_split", required=True)
    ap.add_argument("--partition", choices=["train", "val", "test"], default="test")
    ap.add_argument("--model", action="append", default=[],
                    help="name=experiment_dir (repeatable). Ignored with --teacher_only.")
    ap.add_argument("--teacher_only", action="store_true",
                    help="Dump DeepVariant outputs only; embeddings are not loaded.")
    ap.add_argument("--out_npz", required=True)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--batch_size", type=int, default=8192)
    args = ap.parse_args()

    t0 = time.time()
    rs = load_resolved_split(args.resolved_split)
    index = build_partition_index_hybrid(rs, args.partition, add_masks=True)
    print(f"[Index] partition={args.partition} n={index.n:,} shards={len(index.shard_items)}")

    loaded = []
    device = None
    if not args.teacher_only and args.model:
        import torch
        if args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)
        print(f"[Device] {device}")
        loaded = load_models(parse_models(args.model), device)

    cols: Dict[str, List[np.ndarray]] = {k: [] for k in [
        "chrom", "locus_start", "group", "label", "variant_type",
        "mask_ill", "mask_ont", "dataset_id", "dv_ill_probs", "dv_ont_probs"]}
    for m in loaded:
        cols[f"probs__{m['name']}"] = []

    for k, item in enumerate(index.shard_items):
        npz = np.load(item.shard_path, allow_pickle=False)
        rows = item.rows
        cols["chrom"].append(_as_str_array(npz["chrom"])[rows].astype("U8"))
        cols["locus_start"].append(npz["locus_start"][rows].astype(np.int64))
        cols["group"].append(item.group.astype(np.int16))
        cols["label"].append(item.y.astype(np.int8))
        cols["variant_type"].append(item.variant_type.astype(np.int8))
        cols["mask_ill"].append(item.mask_ill.astype(np.int8))
        cols["mask_ont"].append(item.mask_ont.astype(np.int8))
        cols["dataset_id"].append(np.full(rows.size, item.dataset_id, dtype="U48"))

        p_ill = teacher_probs(npz, rows, "ill")
        p_ont = teacher_probs(npz, rows, "ont")
        p_ill = p_ill.copy(); p_ont = p_ont.copy()
        p_ill[item.mask_ill != 1] = np.nan
        p_ont[item.mask_ont != 1] = np.nan
        cols["dv_ill_probs"].append(p_ill)
        cols["dv_ont_probs"].append(p_ont)

        if loaded:
            base = {"p_ill": dv_log_probs(npz, rows, "ill", item.mask_ill),
                    "p_ont": dv_log_probs(npz, rows, "ont", item.mask_ont)}
            if any(m["info"].get("input_features", "embeddings") == "embeddings" for m in loaded):
                base["ill"] = npz["ill_embeddings"].astype(np.float32, copy=False)[rows]
                base["ont"] = npz["ont_embeddings"].astype(np.float32, copy=False)[rows]
            for m in loaded:
                cols[f"probs__{m['name']}"].append(
                    run_model(m, base, item.mask_ill, item.mask_ont, item.group,
                              device, args.batch_size))
        npz.close()
        if (k + 1) % 10 == 0 or k + 1 == len(index.shard_items):
            print(f"[Progress] shards {k + 1}/{len(index.shard_items)}  elapsed={time.time() - t0:.0f}s")

    out = {k: np.concatenate(v, axis=0) for k, v in cols.items()}

    # Sanity: DV argmax accuracy per view, to compare against existing reports.
    for pref, mk in (("ill", "mask_ill"), ("ont", "mask_ont")):
        sel = out[mk] == 1
        acc = float((np.argmax(out[f"dv_{pref}_probs"][sel], axis=1) == out["label"][sel]).mean())
        print(f"[Check] dv_{pref} view n={int(sel.sum()):,} acc={acc:.6f}")
    for m in loaded:
        p = out[f"probs__{m['name']}"]
        sel = ~np.isnan(p[:, 0])
        acc = float((np.argmax(p[sel], axis=1) == out["label"][sel]).mean())
        print(f"[Check] {m['name']} n={int(sel.sum()):,} acc={acc:.6f}")

    meta = {
        "resolved_split": str(Path(args.resolved_split).resolve()),
        "split_id": rs.get("split_id"),
        "partition": args.partition,
        "n": int(out["label"].shape[0]),
        "models": [{"name": m["name"], "dir": m["dir"], "best_epoch": m["best_epoch"],
                    "info": m["info"]} for m in loaded],
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out_path = Path(args.out_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, meta_json=np.array(json.dumps(meta)), **out)
    print(f"[Saved] {out_path}  n={meta['n']:,}  elapsed={time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
