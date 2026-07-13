#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified hybrid trainer for canonical multimodal OUTER datasets.

Supported model families
------------------------
- hybrid_simple_3c
- hybrid_groupwise_3c

This trainer consumes:
- an experiment config JSON
- a resolved split JSON

It uses the canonical multimodal OUTER datasets and the unified dataset label:
- keep groups {1, 10, 11}
- target = dataset.label

Features
--------
- concat([ill_embeddings, ont_embeddings])                     if add_masks=False
- concat([ill_embeddings, ont_embeddings, mask_ill, mask_ont]) if add_masks=True

Example
-------
python training/scripts/train/hybrid/train_hybrid.py \
  --experiment_config training/configs/experiments/hybrid_groupwise_hg002_hg003_vs_hg004.json \
  --resolved_split training/configs/splits/resolved__hg002_hg003_train__hg004_valtest.json \
  --device cpu
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

from training.scripts.train.shared_data import (  # noqa: E402
    build_partition_index_hybrid,
    load_hybrid_shard_features,
    load_resolved_split,
    print_partition_summary,
    summarize_partition,
)


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------

class LinearHead(nn.Module):
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class MLP(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GroupwiseLinear3C(nn.Module):
    def __init__(self, in_dim: int, n_classes: int = 3):
        super().__init__()
        self.trunk = nn.Identity()
        self.heads = nn.ModuleDict({
            "1": nn.Linear(in_dim, n_classes),
            "10": nn.Linear(in_dim, n_classes),
            "11": nn.Linear(in_dim, n_classes),
        })

    def forward(self, x: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        out = torch.zeros((h.shape[0], 3), dtype=h.dtype, device=h.device)

        for g in (1, 10, 11):
            mask = (groups == g)
            if mask.any():
                out[mask] = self.heads[str(g)](h[mask])

        return out


class GroupwiseMLP3C(nn.Module):
    def __init__(self, in_dim: int, n_classes: int = 3, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleDict({
            "1": nn.Linear(hidden, n_classes),
            "10": nn.Linear(hidden, n_classes),
            "11": nn.Linear(hidden, n_classes),
        })

    def forward(self, x: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        out = torch.zeros((h.shape[0], 3), dtype=h.dtype, device=h.device)

        for g in (1, 10, 11):
            mask = (groups == g)
            if mask.any():
                out[mask] = self.heads[str(g)](h[mask])

        return out


def build_model(
    model_family: str,
    in_dim: int,
    n_classes: int,
    arch: str,
    hidden: int,
    dropout: float,
) -> nn.Module:
    if model_family == "hybrid_simple_3c":
        if arch == "linear":
            return LinearHead(in_dim=in_dim, n_classes=n_classes)
        return MLP(in_dim=in_dim, n_classes=n_classes, hidden=hidden, dropout=dropout)

    if model_family == "hybrid_groupwise_3c":
        if arch == "linear":
            return GroupwiseLinear3C(in_dim=in_dim, n_classes=n_classes)
        return GroupwiseMLP3C(in_dim=in_dim, n_classes=n_classes, hidden=hidden, dropout=dropout)

    raise SystemExit(f"[ERROR] Unsupported model_family: {model_family}")


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def cm_3(y_true: List[int], y_pred: List[int], n_classes: int = 3) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def metrics_from_cm_3(cm: np.ndarray) -> Dict[str, Any]:
    eps = 1e-12
    n_classes = cm.shape[0]
    total = int(cm.sum())
    acc = float(np.trace(cm) / max(total, 1))

    per_class = {}
    f1s = []
    recs = []
    precs = []
    supports = []

    for c in range(n_classes):
        tp = float(cm[c, c])
        fp = float(cm[:, c].sum() - cm[c, c])
        fn = float(cm[c, :].sum() - cm[c, c])
        support = float(cm[c, :].sum())

        prec = float(tp / (tp + fp + eps))
        rec = float(tp / (tp + fn + eps))
        f1 = float(2 * prec * rec / (prec + rec + eps))

        per_class[str(c)] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": int(support),
        }

        f1s.append(f1)
        recs.append(rec)
        precs.append(prec)
        supports.append(support)

    macro_f1 = float(np.mean(f1s))
    macro_recall = float(np.mean(recs))
    macro_precision = float(np.mean(precs))
    wsum = float(np.sum(supports)) + eps
    weighted_f1 = float(np.sum([f1s[i] * supports[i] for i in range(n_classes)]) / wsum)

    return {
        "acc": acc,
        "macro_f1": macro_f1,
        "macro_recall": macro_recall,
        "macro_precision": macro_precision,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
    }


def variant_recall_from_cm3(cm3: List[List[int]]) -> float:
    cm = np.array(cm3, dtype=np.int64)
    fn = cm[1, 0] + cm[2, 0]
    tp = cm[1, 1] + cm[1, 2] + cm[2, 1] + cm[2, 2]
    return float(tp / max(tp + fn, 1))


@torch.no_grad()
def eval_partition_3c(index, model, model_family: str, device, batch_size: int, add_masks: bool) -> Dict[str, Any]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    by_group: Dict[int, Dict[str, List[int]]] = {1: {"y": [], "p": []}, 10: {"y": [], "p": []}, 11: {"y": [], "p": []}}

    for shard_item in index.shard_items:
        x_np, y_np, g_np = load_hybrid_shard_features(shard_item, add_masks=add_masks)

        for i in range(0, len(y_np), batch_size):
            xb = x_np[i:i + batch_size]
            yb = y_np[i:i + batch_size]
            gb = g_np[i:i + batch_size]

            x = torch.from_numpy(xb).to(device)
            g = torch.from_numpy(gb.astype(np.int64)).to(device)

            if model_family == "hybrid_simple_3c":
                logits = model(x)
            elif model_family == "hybrid_groupwise_3c":
                logits = model(x, g)
            else:
                raise SystemExit(f"[ERROR] Unsupported model_family in eval: {model_family}")

            pred = torch.argmax(logits, dim=1).cpu().numpy().astype(int)

            y_true += yb.tolist()
            y_pred += pred.tolist()

            for yy, pp, gg in zip(yb.tolist(), pred.tolist(), gb.tolist()):
                by_group[int(gg)]["y"].append(int(yy))
                by_group[int(gg)]["p"].append(int(pp))

    cm = cm_3(y_true, y_pred, n_classes=3)
    out = {"n": len(y_true), "cm": cm.tolist(), **metrics_from_cm_3(cm), "by_group": {}}

    for gg in (1, 10, 11):
        if len(by_group[gg]["y"]) == 0:
            continue
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


def infer_model_family_from_experiment(exp: Dict[str, Any]) -> Optional[str]:
    candidates = [
        exp.get("model_family"),
        exp.get("dataset_policy", {}).get("mode"),
    ]
    for c in candidates:
        if c is None:
            continue
        s = str(c).lower()
        if s == "hybrid_simple_3c":
            return "hybrid_simple_3c"
        if s == "hybrid_groupwise_3c":
            return "hybrid_groupwise_3c"
    return None


def score_from_val(val_m: Dict[str, Any], save_best_on: str) -> float:
    if save_best_on == "val_acc":
        return float(val_m["acc"])
    if save_best_on == "val_weighted_f1":
        return float(val_m["weighted_f1"])
    if save_best_on == "val_variant_recall":
        return float(variant_recall_from_cm3(val_m["cm"]))
    return float(val_m["macro_f1"])


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment_config", type=str, required=True)
    ap.add_argument("--resolved_split", type=str, required=True)

    ap.add_argument(
        "--model_family",
        type=str,
        default=None,
        choices=["hybrid_simple_3c", "hybrid_groupwise_3c"],
        help="Optional override. If omitted, inferred from experiment_config.",
    )
    ap.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Optional override for output directory. Default: <outputs.root_dir>/<experiment_id>",
    )
    ap.add_argument("--out_csv", type=str, default=None)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    args = ap.parse_args()

    experiment_config_path = Path(args.experiment_config).resolve()
    resolved_split_path = Path(args.resolved_split).resolve()

    exp = load_json(experiment_config_path)
    rs = load_resolved_split(resolved_split_path)

    expected_split_id = exp.get("split_id")
    resolved_split_id = rs.get("split_id")
    if expected_split_id is not None and str(expected_split_id) != str(resolved_split_id):
        raise SystemExit(
            "[ERROR] Experiment config and resolved split do not match: "
            f"experiment split_id={expected_split_id!r}, "
            f"resolved split_id={resolved_split_id!r}."
        )

    experiment_id = str(exp.get("experiment_id", "hybrid_experiment"))
    hyper = exp.get("hyperparams", {})
    selection = exp.get("selection", {})
    outputs_cfg = exp.get("outputs", {})

    model_family = args.model_family or infer_model_family_from_experiment(exp)
    if model_family is None:
        raise SystemExit(
            "[ERROR] Could not infer model_family from experiment_config. "
            "Pass --model_family explicitly."
        )

    seed = int(exp.get("seed", 0))
    epochs = int(hyper.get("epochs", 10))
    batch_size = int(hyper.get("batch_size", 2048))
    lr = float(hyper.get("lr", 3e-4))
    weight_decay = float(hyper.get("weight_decay", 5e-4))
    hidden = int(hyper.get("hidden", 512))
    dropout = float(hyper.get("dropout", 0.1))
    arch = str(hyper.get("arch", "mlp")).lower()
    add_masks = bool(hyper.get("add_masks", True))
    use_class_weights = bool(hyper.get("use_class_weights", False))
    max_class_weight = float(hyper.get("max_class_weight", 10.0))
    save_best_on = str(selection.get("save_best_on", "val_macro_f1"))

    if arch not in {"mlp", "linear"}:
        raise SystemExit(f"[ERROR] Unsupported arch: {arch}")

    if save_best_on not in {"val_macro_f1", "val_acc", "val_weighted_f1", "val_variant_recall"}:
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
    print(f"[Family]     {model_family}")
    print(f"[Device]     {device}")
    print(f"[Config]     {experiment_config_path}")
    print(f"[Split]      {resolved_split_path}")
    print(f"[SaveDir]    {save_dir}")

    train_idx = build_partition_index_hybrid(rs, "train", add_masks=add_masks)
    val_idx = build_partition_index_hybrid(rs, "val", add_masks=add_masks)
    test_idx = build_partition_index_hybrid(rs, "test", add_masks=add_masks)

    print_partition_summary(summarize_partition(train_idx))
    print_partition_summary(summarize_partition(val_idx))
    print_partition_summary(summarize_partition(test_idx))
    print()

    if epochs <= 0:
        print("[DryRun] epochs<=0, stopping after partition summaries.")
        return

    in_dim = int(train_idx.input_dim)
    model = build_model(
        model_family=model_family,
        in_dim=in_dim,
        n_classes=3,
        arch=arch,
        hidden=hidden,
        dropout=dropout,
    ).to(device)

    if model_family == "hybrid_simple_3c":
        if arch == "linear":
            print(f"[Model] family=simple arch=linear in_dim={in_dim} n_classes=3 add_masks={add_masks}")
        else:
            print(f"[Model] family=simple arch=mlp in_dim={in_dim} n_classes=3 hidden={hidden} dropout={dropout} add_masks={add_masks}")
    else:
        if arch == "linear":
            print(f"[Model] family=groupwise arch=linear in_dim={in_dim} n_classes=3 add_masks={add_masks}")
        else:
            print(f"[Model] family=groupwise arch=mlp in_dim={in_dim} n_classes=3 hidden={hidden} dropout={dropout} add_masks={add_masks}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    class_weights_list = None
    if use_class_weights:
        cw = build_class_weights(train_idx, max_w=max_class_weight)
        class_weights_list = cw.numpy().tolist() if cw is not None else None
        print(f"[Loss] class_weights (capped, mean~1): {class_weights_list}")
        loss_fn = nn.CrossEntropyLoss(weight=cw.to(device) if cw is not None else None)
    else:
        loss_fn = nn.CrossEntropyLoss()

    config_payload = {
        "args": {
            "experiment_config": str(experiment_config_path),
            "resolved_split": str(resolved_split_path),
            "model_family": model_family,
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
            "model_family": model_family,
            "arch": arch,
            "hidden": hidden,
            "dropout": dropout,
            "add_masks": add_masks,
            "seed": seed,
        },
        "dataset_policy": {
            "mode": "hybrid",
            "groups_keep": [1, 10, 11],
            "target": "label",
            "label_source": "dataset.label",
            "variant_type_source": "dataset.variant_type",
            "keep_vt_mismatch": True,
            **exp.get("dataset_policy", {}),
        },
        "model_policy": {
            "model_family": model_family,
            "architecture": "simple_3class" if model_family == "hybrid_simple_3c" else "groupwise_3class",
            "heads": [1, 10, 11] if model_family == "hybrid_groupwise_3c" else None,
            "save_best_on": save_best_on,
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
            "val_n", "val_acc", "val_macro_f1", "val_weighted_f1", "val_variant_recall",
            "test_n", "test_acc", "test_macro_f1", "test_weighted_f1", "test_variant_recall",
        ])

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
            x_np, y_np, g_np = load_hybrid_shard_features(shard_item, add_masks=add_masks)

            perm = np.arange(len(y_np))
            perm_list = perm.tolist()
            rng.shuffle(perm_list)
            perm = np.asarray(perm_list, dtype=np.int64)

            x_np = x_np[perm]
            y_np = y_np[perm]
            g_np = g_np[perm]

            for i in range(0, len(y_np), batch_size):
                xb = x_np[i:i + batch_size]
                yb = y_np[i:i + batch_size]
                gb = g_np[i:i + batch_size]

                x = torch.from_numpy(xb).to(device)
                y = torch.from_numpy(yb.astype(np.int64)).to(device)
                g = torch.from_numpy(gb.astype(np.int64)).to(device)

                opt.zero_grad(set_to_none=True)

                if model_family == "hybrid_simple_3c":
                    logits = model(x)
                elif model_family == "hybrid_groupwise_3c":
                    logits = model(x, g)
                else:
                    raise SystemExit(f"[ERROR] Unsupported model_family in train loop: {model_family}")

                loss = loss_fn(logits, y)
                loss.backward()
                opt.step()

                bs = len(yb)
                total_loss += float(loss.detach().cpu()) * bs
                n_seen += bs

        train_loss = total_loss / max(n_seen, 1)

        val_m = eval_partition_3c(val_idx, model, model_family, device, batch_size, add_masks)
        test_m = eval_partition_3c(test_idx, model, model_family, device, batch_size, add_masks)

        val_variant_recall = variant_recall_from_cm3(val_m["cm"])
        test_variant_recall = variant_recall_from_cm3(test_m["cm"])
        score = score_from_val(val_m, save_best_on)

        print(
            f"[Epoch {epoch:02d}] loss={train_loss:.5f} | "
            f"VAL acc={val_m['acc']:.4f} macroF1={val_m['macro_f1']:.4f} "
            f"wF1={val_m['weighted_f1']:.4f} varRec={val_variant_recall:.4f} | "
            f"TEST acc={test_m['acc']:.4f} macroF1={test_m['macro_f1']:.4f} "
            f"wF1={test_m['weighted_f1']:.4f} varRec={test_variant_recall:.4f} | "
            f"best_on={save_best_on} score={score:.6f}"
        )

        if csv_writer:
            csv_writer.writerow([
                epoch, train_loss,
                val_m["n"], val_m["acc"], val_m["macro_f1"], val_m["weighted_f1"], val_variant_recall,
                test_m["n"], test_m["acc"], test_m["macro_f1"], test_m["weighted_f1"], test_variant_recall,
            ])
            csv_f.flush()

        last_payload = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val": val_m,
            "test": test_m,
            "val_variant_recall": val_variant_recall,
            "test_variant_recall": test_variant_recall,
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
                "model_family": model_family,
                "input_dim": in_dim,
                "n_classes": 3,
                "arch": arch,
                "hidden": hidden,
                "dropout": dropout,
                "add_masks": add_masks,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "best_on": save_best_on,
                "val": val_m,
                "test": test_m,
                "val_variant_recall": val_variant_recall,
                "test_variant_recall": test_variant_recall,
            }
            torch.save(best_payload, save_dir / "best.pt")

            best_metrics_only = {
                "epoch": epoch,
                "best_score": best_score,
                "best_on": save_best_on,
                "val": val_m,
                "test": test_m,
                "val_variant_recall": val_variant_recall,
                "test_variant_recall": test_variant_recall,
            }
            (save_dir / "best_metrics.json").write_text(
                json.dumps(_jsonify(best_metrics_only), indent=2),
                encoding="utf-8",
            )

    final_payload = {
        "epoch": epochs,
        "model_state": model.state_dict(),
        "experiment_id": experiment_id,
        "model_family": model_family,
        "input_dim": in_dim,
        "n_classes": 3,
        "arch": arch,
        "hidden": hidden,
        "dropout": dropout,
        "add_masks": add_masks,
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