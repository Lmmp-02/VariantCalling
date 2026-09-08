#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified unimodal trainer for Illumina-only and ONT-only baselines.

This trainer consumes:
- an experiment config JSON
- a resolved split JSON

It uses the canonical multimodal OUTER datasets and applies a unimodal view:
- modality=illumina -> keep groups {10, 11}, use ill_embeddings
- modality=ont      -> keep groups {1, 11}, use ont_embeddings

The target is always the unified dataset `label`, already resolved upstream
by the multimodal dataset builder.

Example
-------
python training/scripts/train/unimodal/train_unimodal.py \
  --experiment_config training/configs/experiments/unimodal_illumina_hg002_hg003_vs_hg004.json \
  --resolved_split training/configs/splits/resolved__hg002_hg003_train__hg004_valtest.json \
  --modality illumina
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

# Make repo root importable when executing as a script
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.metrics import cm_3, metrics_from_cm_3
from training.models import LinearHead, MLP
from training.scripts.train.shared_data import (  # noqa: E402
    build_partition_index_unimodal,
    load_resolved_split,
    load_unimodal_shard_features,
    summarize_partition,
    print_partition_summary,
)


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------

def build_model(in_dim: int, n_classes: int, arch: str, hidden: int, dropout: float) -> nn.Module:
    if arch == "linear":
        return LinearHead(in_dim=in_dim, n_classes=n_classes)
    return MLP(in_dim=in_dim, n_classes=n_classes, hidden=hidden, dropout=dropout)

  
# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

@torch.no_grad()
def eval_partition_3c(index, model, modality: str, device, batch_size: int) -> Dict[str, Any]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    by_group: Dict[int, Dict[str, List[int]]] = {}

    for shard_item in index.shard_items:
        x_np, y_np = load_unimodal_shard_features(shard_item, modality)
        g_np = shard_item.group.astype(np.int64, copy=False)

        for i in range(0, len(y_np), batch_size):
            xb = x_np[i:i + batch_size]
            yb = y_np[i:i + batch_size]
            gb = g_np[i:i + batch_size]

            x = torch.from_numpy(xb).to(device)
            logits = model(x)
            pred = torch.argmax(logits, dim=1).cpu().numpy().astype(int)

            y_true += yb.tolist()
            y_pred += pred.tolist()

            for yy, pp, gg in zip(yb.tolist(), pred.tolist(), gb.tolist()):
                if int(gg) not in by_group:
                    by_group[int(gg)] = {"y": [], "p": []}
                by_group[int(gg)]["y"].append(int(yy))
                by_group[int(gg)]["p"].append(int(pp))

    cm = cm_3(y_true, y_pred, n_classes=3)
    out = {"n": len(y_true), "cm": cm.tolist(), **metrics_from_cm_3(cm), "by_group": {}}

    for gg in sorted(by_group.keys()):
        cmg = cm_3(by_group[gg]["y"], by_group[gg]["p"], n_classes=3)
        out["by_group"][str(gg)] = {
            "n": len(by_group[gg]["y"]),
            "cm": cmg.tolist(),
            **metrics_from_cm_3(cmg),
        }

    return out


# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------

def _jsonify(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    return obj


def load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"[ERROR] JSON file not found: {path}")
    except json.JSONDecodeError as e:
        raise SystemExit(f"[ERROR] Invalid JSON in {path}: {e}")


def build_class_weights(train_index, max_w: float = 10.0) -> Optional[torch.Tensor]:
    counts = {0: 0, 1: 0, 2: 0}
    for shard_item in train_index.shard_items:
        for y in shard_item.y.tolist():
            counts[int(y)] += 1

    total = float(sum(counts.values()))
    if total <= 0:
        return None

    w = np.zeros((3,), dtype=np.float32)
    for c in (0, 1, 2):
        if counts[c] > 0:
            w[c] = total / (3.0 * float(counts[c]))
        else:
            w[c] = 1.0

    w = w / max(w.mean(), 1e-12)
    w = np.clip(w, 0.0, float(max_w))
    return torch.from_numpy(w)


def infer_modality_from_experiment(exp: Dict[str, Any]) -> Optional[str]:
    candidates = [
        exp.get("modality"),
        exp.get("dataset_policy", {}).get("modality"),
        exp.get("dataset_policy", {}).get("mode"),
        exp.get("model_family"),
    ]
    for c in candidates:
        if c is None:
            continue
        s = str(c).lower()
        if "illumina" in s:
            return "illumina"
        if s == "ill":
            return "illumina"
        if "ont" in s:
            return "ont"
    return None


def modality_groups_keep(modality: str) -> List[int]:
    if modality == "illumina":
        return [10, 11]
    if modality == "ont":
        return [1, 11]
    raise SystemExit(f"[ERROR] Unsupported modality: {modality}")


def target_name_for_modality(modality: str) -> str:
    if modality == "illumina":
        return "label (unified), using ill_embeddings, groups {10,11}"
    if modality == "ont":
        return "label (unified), using ont_embeddings, groups {1,11}"
    raise SystemExit(f"[ERROR] Unsupported modality: {modality}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment_config", type=str, required=True)
    ap.add_argument("--resolved_split", type=str, required=True)

    ap.add_argument(
        "--modality",
        type=str,
        default=None,
        choices=["illumina", "ont"],
        help="Optional override. If omitted, inferred from experiment_config.",
    )
    ap.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Optional override for output directory. "
             "Default: <outputs.root_dir>/<experiment_id>",
    )
    ap.add_argument("--out_csv", type=str, default=None)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    args = ap.parse_args()

    experiment_config_path = Path(args.experiment_config).resolve()
    resolved_split_path = Path(args.resolved_split).resolve()

    exp = load_json(experiment_config_path)
    rs = load_resolved_split(resolved_split_path)

    experiment_id = str(exp.get("experiment_id", "unimodal_experiment"))
    hyper = exp.get("hyperparams", {})
    selection = exp.get("selection", {})
    outputs_cfg = exp.get("outputs", {})

    modality = args.modality or infer_modality_from_experiment(exp)
    if modality is None:
        raise SystemExit(
            "[ERROR] Could not infer modality from experiment_config. "
            "Pass --modality illumina|ont explicitly."
        )

    seed = int(exp.get("seed", 0))
    epochs = int(hyper.get("epochs", 10))
    batch_size = int(hyper.get("batch_size", 2048))
    lr = float(hyper.get("lr", 3e-4))
    weight_decay = float(hyper.get("weight_decay", 5e-4))
    hidden = int(hyper.get("hidden", 512))
    dropout = float(hyper.get("dropout", 0.1))
    arch = str(hyper.get("arch", "mlp")).lower()
    use_class_weights = bool(hyper.get("use_class_weights", False))
    max_class_weight = float(hyper.get("max_class_weight", 10.0))
    save_best_on = str(selection.get("save_best_on", "val_macro_f1"))

    if arch not in {"mlp", "linear"}:
        raise SystemExit(f"[ERROR] Unsupported arch: {arch}")

    if save_best_on not in {"val_macro_f1", "val_acc", "val_weighted_f1"}:
        raise SystemExit(f"[ERROR] Unsupported save_best_on: {save_best_on}")

    if args.save_dir:
        save_dir = Path(args.save_dir).resolve()
    else:
        root_dir = Path(outputs_cfg.get("root_dir", "training/out/experiments"))
        save_dir = (root_dir / experiment_id).resolve()

    save_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("[ERROR] --device cuda requested but CUDA is not available.")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[Experiment] id={experiment_id}")
    print(f"[Modality]   {modality}")
    print(f"[Device]     {device}")
    print(f"[Config]     {experiment_config_path}")
    print(f"[Split]      {resolved_split_path}")
    print(f"[SaveDir]    {save_dir}")

    # Build partition indices (memory-safe, shard-based)
    train_idx = build_partition_index_unimodal(rs, "train", modality)
    val_idx = build_partition_index_unimodal(rs, "val", modality)
    test_idx = build_partition_index_unimodal(rs, "test", modality)

    print_partition_summary(summarize_partition(train_idx))
    print_partition_summary(summarize_partition(val_idx))
    print_partition_summary(summarize_partition(test_idx))
    print()

    if epochs <= 0:
        print("[DryRun] epochs<=0, stopping after partition summaries.")
        return

    in_dim = int(train_idx.input_dim)
    model = build_model(in_dim=in_dim, n_classes=3, arch=arch, hidden=hidden, dropout=dropout).to(device)

    if arch == "linear":
        print(f"[Model] arch=linear in_dim={in_dim} n_classes=3")
    else:
        print(f"[Model] arch=mlp in_dim={in_dim} n_classes=3 hidden={hidden} dropout={dropout}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    class_weights_list = None
    if use_class_weights:
        cw = build_class_weights(train_idx, max_w=max_class_weight)
        class_weights_list = cw.numpy().tolist() if cw is not None else None
        print(f"[Loss] class_weights (capped, mean~1): {class_weights_list}")
        loss_fn = nn.CrossEntropyLoss(weight=cw.to(device) if cw is not None else None)
    else:
        loss_fn = nn.CrossEntropyLoss()

    # Save resolved config snapshot
    config_payload = {
        "args": {
            "experiment_config": str(experiment_config_path),
            "resolved_split": str(resolved_split_path),
            "modality": modality,
            "save_dir": str(save_dir),
            "out_csv": args.out_csv,
            "device": str(device),
        },
        "resolved": {
            "experiment_id": experiment_id,
            "input_dim": in_dim,
            "n_classes": 3,
            "device": str(device),
            "loss": "CrossEntropyLoss",
            "class_weights": class_weights_list,
            "arch": arch,
            "hidden": hidden,
            "dropout": dropout,
            "seed": seed,
        },
        "dataset_policy": {
            "mode": f"{modality}_only",
            "groups_keep": modality_groups_keep(modality),
            "target": target_name_for_modality(modality),
            "label_source": "dataset.label",
            "variant_type_source": "dataset.variant_type",
            "keep_vt_mismatch": True,
            "join_definition": "locus-level late merge",
        },
        "data_summary": {
            "train": summarize_partition(train_idx),
            "val": summarize_partition(val_idx),
            "test": summarize_partition(test_idx),
        },
        "experiment_config_path": str(experiment_config_path),
        "resolved_split_path": str(resolved_split_path),
    }

    (save_dir / "config.json").write_text(
        json.dumps(_jsonify(config_payload), indent=2),
        encoding="utf-8",
    )
    (save_dir / "experiment_config.used.json").write_text(
        json.dumps(_jsonify(exp), indent=2),
        encoding="utf-8",
    )
    (save_dir / "resolved_split.used.json").write_text(
        json.dumps(_jsonify(rs), indent=2),
        encoding="utf-8",
    )

    csv_writer = None
    csv_f = None
    if args.out_csv:
        outp = Path(args.out_csv).resolve()
        outp.parent.mkdir(parents=True, exist_ok=True)
        csv_f = open(outp, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_f)
        csv_writer.writerow([
            "epoch", "train_loss",
            "val_n", "val_acc", "val_macro_f1", "val_weighted_f1",
            "test_n", "test_acc", "test_macro_f1", "test_weighted_f1",
        ])

    def score_from_val(val_m: Dict[str, Any]) -> float:
        if save_best_on == "val_acc":
            return float(val_m["acc"])
        if save_best_on == "val_weighted_f1":
            return float(val_m["weighted_f1"])
        return float(val_m["macro_f1"])

    best_score = -1e18
    best_epoch = -1

    shard_items_master = list(train_idx.shard_items)

    for epoch in range(1, epochs + 1):
        model.train()
        rng = random.Random(seed + epoch)

        shard_items = list(shard_items_master)
        rng.shuffle(shard_items)

        total_loss = 0.0
        n_seen = 0

        for shard_item in shard_items:
            x_np, y_np = load_unimodal_shard_features(shard_item, modality)

            perm = np.arange(len(y_np))
            perm_list = perm.tolist()
            rng.shuffle(perm_list)
            perm = np.asarray(perm_list, dtype=np.int64)

            x_np = x_np[perm]
            y_np = y_np[perm]

            for i in range(0, len(y_np), batch_size):
                xb = x_np[i:i + batch_size]
                yb = y_np[i:i + batch_size]

                x = torch.from_numpy(xb).to(device)
                y = torch.from_numpy(yb.astype(np.int64)).to(device)

                opt.zero_grad(set_to_none=True)
                logits = model(x)
                loss = loss_fn(logits, y)
                loss.backward()
                opt.step()

                bs = len(yb)
                total_loss += float(loss.detach().cpu()) * bs
                n_seen += bs

        train_loss = total_loss / max(n_seen, 1)

        val_m = eval_partition_3c(val_idx, model, modality, device, batch_size)
        test_m = eval_partition_3c(test_idx, model, modality, device, batch_size)
        score = score_from_val(val_m)

        print(
            f"[Epoch {epoch:02d}] loss={train_loss:.5f} | "
            f"VAL acc={val_m['acc']:.4f} macroF1={val_m['macro_f1']:.4f} wF1={val_m['weighted_f1']:.4f} | "
            f"TEST acc={test_m['acc']:.4f} macroF1={test_m['macro_f1']:.4f} wF1={test_m['weighted_f1']:.4f} | "
            f"best_on={save_best_on} score={score:.6f}"
        )

        if csv_writer:
            csv_writer.writerow([
                epoch, train_loss,
                val_m["n"], val_m["acc"], val_m["macro_f1"], val_m["weighted_f1"],
                test_m["n"], test_m["acc"], test_m["macro_f1"], test_m["weighted_f1"],
            ])
            csv_f.flush()

        last_payload = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val": val_m,
            "test": test_m,
            "score": score,
            "best_on": save_best_on,
        }
        (save_dir / "last_metrics.json").write_text(
            json.dumps(_jsonify(last_payload), indent=2),
            encoding="utf-8",
        )

        if score > best_score:
            best_score = float(score)
            best_epoch = epoch

            best_payload = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "experiment_id": experiment_id,
                "modality": modality,
                "input_dim": in_dim,
                "n_classes": 3,
                "arch": arch,
                "hidden": hidden,
                "dropout": dropout,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "best_on": save_best_on,
                "val": val_m,
                "test": test_m,
            }
            torch.save(best_payload, save_dir / "best.pt")

            best_metrics_only = {
                "epoch": epoch,
                "best_score": best_score,
                "best_on": save_best_on,
                "val": val_m,
                "test": test_m,
            }
            (save_dir / "best_metrics.json").write_text(
                json.dumps(_jsonify(best_metrics_only), indent=2),
                encoding="utf-8",
            )

    final_payload = {
        "epoch": epochs,
        "model_state": model.state_dict(),
        "experiment_id": experiment_id,
        "modality": modality,
        "input_dim": in_dim,
        "n_classes": 3,
        "arch": arch,
        "hidden": hidden,
        "dropout": dropout,
        "best_score": best_score,
        "best_epoch": best_epoch,
        "best_on": save_best_on,
    }
    torch.save(final_payload, save_dir / "final.pt")

    if csv_f:
        csv_f.close()

    print(f"[Saved] best.pt / final.pt in {save_dir}")
    print(f"[Saved] config.json / experiment_config.used.json / resolved_split.used.json in {save_dir}")
    print(f"[Saved] best_metrics.json / last_metrics.json in {save_dir}")


if __name__ == "__main__":
    main()
