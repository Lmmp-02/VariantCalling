#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified internal evaluator for trained checkpoints and DeepVariant teachers.

Evaluation scope
----------------
This script evaluates on labelled offline datasets, typically:

- val  = HG004 chr21
- test = HG004 chr20

It is meant for internal supervised evaluation, not for external HG005 calling.
For HG005 calling datasets, use a future call_checkpoint.py / calling script.

Supported sources
-----------------
1. source=checkpoint
   Evaluates a trained model checkpoint from training/out/experiments/<experiment_id>/

2. source=teacher
   Evaluates DeepVariant teacher logits/probs stored inside the multimodal OUTER shards:
   - teacher=illumina -> ill_probs / ill_logits over groups {10,11}
   - teacher=ont      -> ont_probs / ont_logits over groups {1,11}

Outputs
-------
- JSON report with full metrics
- CSV long/tidy summary with one row per evaluation scope

The CSV is intended for plotting/comparison. Examples of scopes:
- illumina_view: groups {10, 11}
- ont_view: groups {1, 11}
- hybrid_all: groups {1, 10, 11}
- group_1 / group_10 / group_11
- variant_SNP / variant_INDEL

Example: checkpoint
-------------------
python training/scripts/eval/eval_checkpoint.py \
  --source checkpoint \
  --experiment_dir training/out/experiments/hybrid_groupwise_hg002_hg003_vs_hg004 \
  --checkpoint best.pt \
  --partition test \
  --device cuda

Example: DV teacher
-------------------
python training/scripts/eval/eval_checkpoint.py \
  --source teacher \
  --teacher illumina \
  --resolved_split training/configs/splits/resolved__hg002_hg003_train__hg004_valtest.json \
  --partition test
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# Make repo root importable when executing as a script
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.models import (  # noqa: E402
    GroupwiseLinear3C,
    GroupwiseMLP3C,
    LinearHead,
    MLP,
)
from training.scripts.train.shared_data import (  # noqa: E402
    build_partition_index_hybrid,
    build_partition_index_unimodal,
    load_hybrid_shard_features,
    load_resolved_split,
    load_unimodal_shard_features,
    print_partition_summary,
    summarize_partition,
)


# ---------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------

def build_checkpoint_model(
    *,
    model_kind: str,
    model_family: Optional[str],
    in_dim: int,
    n_classes: int,
    arch: str,
    hidden: int,
    dropout: float,
) -> nn.Module:
    if model_kind == "unimodal":
        if arch == "linear":
            return LinearHead(in_dim=in_dim, n_classes=n_classes)
        return MLP(in_dim=in_dim, n_classes=n_classes, hidden=hidden, dropout=dropout)

    if model_kind == "hybrid":
        if model_family == "hybrid_simple_3c":
            if arch == "linear":
                return LinearHead(in_dim=in_dim, n_classes=n_classes)
            return MLP(in_dim=in_dim, n_classes=n_classes, hidden=hidden, dropout=dropout)

        if model_family == "hybrid_groupwise_3c":
            if arch == "linear":
                return GroupwiseLinear3C(in_dim=in_dim, n_classes=n_classes)
            return GroupwiseMLP3C(in_dim=in_dim, n_classes=n_classes, hidden=hidden, dropout=dropout)

    raise SystemExit(f"[ERROR] Unsupported checkpoint model kind/family: {model_kind} / {model_family}")


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"[ERROR] JSON file not found: {path}")
    except json.JSONDecodeError as e:
        raise SystemExit(f"[ERROR] Invalid JSON in {path}: {e}")


def _jsonify(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    return obj


def softmax_np(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    logits = logits.astype(np.float64)
    m = np.max(logits, axis=axis, keepdims=True)
    e = np.exp(logits - m)
    return (e / np.sum(e, axis=axis, keepdims=True)).astype(np.float32)


def teacher_modality_name(teacher: str) -> str:
    teacher = str(teacher).lower()
    if teacher in {"ill", "illumina"}:
        return "illumina"
    if teacher == "ont":
        return "ont"
    raise SystemExit(f"[ERROR] Unsupported teacher: {teacher}")


def modality_to_teacher_keys(modality: str) -> Tuple[str, str]:
    if modality == "illumina":
        return "ill_probs", "ill_logits"
    if modality == "ont":
        return "ont_probs", "ont_logits"
    raise SystemExit(f"[ERROR] Unsupported modality: {modality}")


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def cm_k(y_true: List[int], y_pred: List[int], k: int) -> np.ndarray:
    cm = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def metrics_from_cm(cm: np.ndarray) -> Dict[str, Any]:
    eps = 1e-12
    k = cm.shape[0]
    support = cm.sum(axis=1)

    per_class = {}
    f1s = []
    recs = []
    precs = []

    for c in range(k):
        tp = float(cm[c, c])
        fp = float(cm[:, c].sum() - cm[c, c])
        fn = float(cm[c, :].sum() - cm[c, c])

        prec = float(tp / (tp + fp + eps))
        rec = float(tp / (tp + fn + eps))
        f1 = float(2.0 * prec * rec / (prec + rec + eps))

        per_class[str(c)] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": int(support[c]),
        }

        f1s.append(f1)
        recs.append(rec)
        precs.append(prec)

    acc = float(np.trace(cm) / max(int(cm.sum()), 1))
    macro_f1 = float(np.mean(f1s))
    macro_recall = float(np.mean(recs))
    macro_precision = float(np.mean(precs))

    weights = support / max(float(support.sum()), 1.0)
    weighted_f1 = float(np.sum(weights * np.asarray(f1s, dtype=np.float64)))

    return {
        "cm": cm.tolist(),
        "acc": acc,
        "macro_f1": macro_f1,
        "macro_recall": macro_recall,
        "macro_precision": macro_precision,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
    }


def binary_metrics(y_true_bin: List[int], y_pred_bin: List[int]) -> Dict[str, Any]:
    cm = cm_k(y_true_bin, y_pred_bin, 2)
    tn, fp = int(cm[0, 0]), int(cm[0, 1])
    fn, tp = int(cm[1, 0]), int(cm[1, 1])
    eps = 1e-12

    precision = float(tp / (tp + fp + eps))
    recall = float(tp / (tp + fn + eps))
    specificity = float(tn / (tn + fp + eps))
    f1 = float(2.0 * precision * recall / (precision + recall + eps))
    f2 = float((5.0 * precision * recall) / (4.0 * precision + recall + eps))
    acc = float((tp + tn) / max(int(cm.sum()), 1))
    balanced_acc = float(0.5 * (recall + specificity))

    return {
        "cm": cm.tolist(),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "acc": acc,
        "balanced_acc": balanced_acc,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "f2": f2,
    }


def variant_metrics_from_cm3(cm3: np.ndarray) -> Dict[str, Any]:
    eps = 1e-12

    tp = float(cm3[1:, 1:].sum())
    fn = float(cm3[1:, 0].sum())
    fp = float(cm3[0, 1:].sum())
    tn = float(cm3[0, 0])

    precision = float(tp / (tp + fp + eps))
    recall = float(tp / (tp + fn + eps))
    specificity = float(tn / (tn + fp + eps))
    balanced_acc = float(0.5 * (recall + specificity))
    f1 = float(2.0 * precision * recall / (precision + recall + eps))
    f2 = float((5.0 * precision * recall) / (4.0 * precision + recall + eps))

    return {
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_acc": balanced_acc,
        "f1": f1,
        "f2": f2,
    }


def vt_name(v: int) -> str:
    if int(v) == 1:
        return "SNP"
    if int(v) == 2:
        return "INDEL"
    return "UNKNOWN"


def summarize_predictions(
    *,
    y_true: List[int],
    y_pred: List[int],
    groups: List[int],
    variant_types: List[int],
) -> Dict[str, Any]:
    if not y_true:
        raise SystemExit("[ERROR] No predictions collected.")

    cm3 = cm_k(y_true, y_pred, 3)

    overall_3c = {
        "n": len(y_true),
        **metrics_from_cm(cm3),
    }

    y_true_bin = [0 if y == 0 else 1 for y in y_true]
    y_pred_bin = [0 if p == 0 else 1 for p in y_pred]

    overall_binary = {
        "n": len(y_true),
        **binary_metrics(y_true_bin, y_pred_bin),
    }

    overall_variant_from_cm3 = variant_metrics_from_cm3(cm3)

    by_group = {}
    for g in sorted(set(groups)):
        idx = [i for i, gg in enumerate(groups) if gg == g]
        yt = [y_true[i] for i in idx]
        yp = [y_pred[i] for i in idx]

        cmg = cm_k(yt, yp, 3)
        ytb = [0 if y == 0 else 1 for y in yt]
        ypb = [0 if p == 0 else 1 for p in yp]

        by_group[str(g)] = {
            "n": len(idx),
            **metrics_from_cm(cmg),
            "binary_variant_vs_no": {
                "n": len(idx),
                **binary_metrics(ytb, ypb),
            },
            "variant_from_cm3": variant_metrics_from_cm3(cmg),
        }

    by_variant_type = {}
    for vt in sorted(set(variant_types)):
        idx = [i for i, vv in enumerate(variant_types) if int(vv) == int(vt)]
        yt = [y_true[i] for i in idx]
        yp = [y_pred[i] for i in idx]

        cmv = cm_k(yt, yp, 3)
        ytb = [0 if y == 0 else 1 for y in yt]
        ypb = [0 if p == 0 else 1 for p in yp]

        by_variant_type[vt_name(vt)] = {
            "variant_type": int(vt),
            "n": len(idx),
            **metrics_from_cm(cmv),
            "binary_variant_vs_no": {
                "n": len(idx),
                **binary_metrics(ytb, ypb),
            },
            "variant_from_cm3": variant_metrics_from_cm3(cmv),
        }

    return {
        "overall_3c": overall_3c,
        "overall_binary_variant_vs_no": overall_binary,
        "overall_variant_from_cm3": overall_variant_from_cm3,
        "by_group": by_group,
        "by_variant_type": by_variant_type,
    }


# ---------------------------------------------------------------------
# Prediction loops
# ---------------------------------------------------------------------

@torch.no_grad()
def predict_checkpoint_unimodal(index, model, modality: str, device, batch_size: int) -> Dict[str, List[int]]:
    model.eval()

    y_true: List[int] = []
    y_pred: List[int] = []
    groups: List[int] = []
    variant_types: List[int] = []

    for shard_item in index.shard_items:
        x_np, y_np = load_unimodal_shard_features(shard_item, modality)
        g_np = shard_item.group.astype(np.int64, copy=False)
        vt_np = shard_item.variant_type.astype(np.int64, copy=False)

        for i in range(0, len(y_np), batch_size):
            xb = x_np[i:i + batch_size]
            yb = y_np[i:i + batch_size]
            gb = g_np[i:i + batch_size]
            vtb = vt_np[i:i + batch_size]

            x = torch.from_numpy(xb).to(device)
            logits = model(x)
            pred = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(np.int64)

            y_true.extend(yb.astype(int).tolist())
            y_pred.extend(pred.astype(int).tolist())
            groups.extend(gb.astype(int).tolist())
            variant_types.extend(vtb.astype(int).tolist())

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "groups": groups,
        "variant_types": variant_types,
    }


@torch.no_grad()
def predict_checkpoint_hybrid(index, model, model_family: str, device, batch_size: int, add_masks: bool) -> Dict[str, List[int]]:
    model.eval()

    y_true: List[int] = []
    y_pred: List[int] = []
    groups: List[int] = []
    variant_types: List[int] = []

    for shard_item in index.shard_items:
        x_np, y_np, g_np = load_hybrid_shard_features(shard_item, add_masks=add_masks)
        vt_np = shard_item.variant_type.astype(np.int64, copy=False)

        for i in range(0, len(y_np), batch_size):
            xb = x_np[i:i + batch_size]
            yb = y_np[i:i + batch_size]
            gb = g_np[i:i + batch_size]
            vtb = vt_np[i:i + batch_size]

            x = torch.from_numpy(xb).to(device)
            g = torch.from_numpy(gb.astype(np.int64)).to(device)

            if model_family == "hybrid_simple_3c":
                logits = model(x)
            elif model_family == "hybrid_groupwise_3c":
                logits = model(x, g)
            else:
                raise SystemExit(f"[ERROR] Unsupported hybrid model_family: {model_family}")

            pred = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(np.int64)

            y_true.extend(yb.astype(int).tolist())
            y_pred.extend(pred.astype(int).tolist())
            groups.extend(gb.astype(int).tolist())
            variant_types.extend(vtb.astype(int).tolist())

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "groups": groups,
        "variant_types": variant_types,
    }


def predict_teacher(index, teacher: str) -> Dict[str, List[int]]:
    modality = teacher_modality_name(teacher)
    prob_key, logits_key = modality_to_teacher_keys(modality)

    y_true: List[int] = []
    y_pred: List[int] = []
    groups: List[int] = []
    variant_types: List[int] = []

    for shard_item in index.shard_items:
        npz = np.load(shard_item.shard_path, allow_pickle=False)
        files = set(npz.files)

        if prob_key in files:
            probs = npz[prob_key].astype(np.float32, copy=False)[shard_item.rows]
        elif logits_key in files:
            logits = npz[logits_key].astype(np.float32, copy=False)[shard_item.rows]
            probs = softmax_np(logits, axis=-1)
        else:
            raise SystemExit(
                f"[ERROR] Missing teacher probabilities/logits in {shard_item.shard_path}: "
                f"expected {prob_key} or {logits_key}"
            )

        pred = np.argmax(probs, axis=1).astype(np.int64)

        y_true.extend(shard_item.y.astype(int).tolist())
        y_pred.extend(pred.astype(int).tolist())
        groups.extend(shard_item.group.astype(int).tolist())
        variant_types.extend(shard_item.variant_type.astype(int).tolist())

        npz.close()

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "groups": groups,
        "variant_types": variant_types,
    }


# ---------------------------------------------------------------------
# Checkpoint metadata resolution
# ---------------------------------------------------------------------

def load_experiment_metadata(experiment_dir: Path) -> Dict[str, Any]:
    config_path = experiment_dir / "config.json"
    used_exp_path = experiment_dir / "experiment_config.used.json"
    used_split_path = experiment_dir / "resolved_split.used.json"

    config = load_json(config_path) if config_path.exists() else {}
    used_exp = load_json(used_exp_path) if used_exp_path.exists() else {}
    resolved_split = load_resolved_split(used_split_path) if used_split_path.exists() else None

    return {
        "config_path": str(config_path) if config_path.exists() else None,
        "experiment_config_used_path": str(used_exp_path) if used_exp_path.exists() else None,
        "resolved_split_used_path": str(used_split_path) if used_split_path.exists() else None,
        "config": config,
        "experiment_config": used_exp,
        "resolved_split": resolved_split,
    }


def infer_checkpoint_model_info(ckpt: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    resolved = config.get("resolved", {})
    model_policy = config.get("model_policy", {})

    modality = ckpt.get("modality", resolved.get("modality"))
    model_family = ckpt.get("model_family", resolved.get("model_family", model_policy.get("model_family")))

    if modality is not None:
        model_kind = "unimodal"
        modality = str(modality).lower()
    elif model_family is not None:
        model_kind = "hybrid"
        model_family = str(model_family)
    else:
        raise SystemExit("[ERROR] Cannot infer checkpoint type: missing modality/model_family.")

    input_dim = int(ckpt.get("input_dim", resolved.get("input_dim", 0)))
    n_classes = int(ckpt.get("n_classes", resolved.get("n_classes", 3)))
    arch = str(ckpt.get("arch", resolved.get("arch", "mlp"))).lower()
    hidden = int(ckpt.get("hidden", resolved.get("hidden", 512)))
    dropout = float(ckpt.get("dropout", resolved.get("dropout", 0.1)))
    add_masks = bool(ckpt.get("add_masks", resolved.get("add_masks", True)))

    if input_dim <= 0:
        raise SystemExit("[ERROR] Cannot infer input_dim from checkpoint/config.")

    return {
        "model_kind": model_kind,
        "modality": modality,
        "model_family": model_family,
        "input_dim": input_dim,
        "n_classes": n_classes,
        "arch": arch,
        "hidden": hidden,
        "dropout": dropout,
        "add_masks": add_masks,
    }


def default_output_paths(
    *,
    source: str,
    experiment_dir: Optional[Path],
    out_dir: Optional[Path],
    name: str,
    partition: str,
) -> Tuple[Path, Path]:
    if out_dir is not None:
        base_dir = out_dir
    elif experiment_dir is not None:
        base_dir = experiment_dir / "reports"
    else:
        base_dir = Path("training/out/reports/internal_hg004_test")

    base_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{name}_{partition}"
    return base_dir / f"{stem}.json", base_dir / f"{stem}.csv"


def _scope_indices(
    *,
    groups: List[int],
    variant_types: List[int],
    group_filter: Optional[set[int]] = None,
    variant_type_filter: Optional[set[int]] = None,
) -> List[int]:
    idx = []
    for i, (g, vt) in enumerate(zip(groups, variant_types)):
        if group_filter is not None and int(g) not in group_filter:
            continue
        if variant_type_filter is not None and int(vt) not in variant_type_filter:
            continue
        idx.append(i)
    return idx


def _subset_payload(pred_payload: Dict[str, List[int]], idx: List[int]) -> Dict[str, List[int]]:
    return {
        "y_true": [pred_payload["y_true"][i] for i in idx],
        "y_pred": [pred_payload["y_pred"][i] for i in idx],
        "groups": [pred_payload["groups"][i] for i in idx],
        "variant_types": [pred_payload["variant_types"][i] for i in idx],
    }


def _scope_metric_row(
    *,
    source: str,
    name: str,
    partition: str,
    scope: str,
    scope_type: str,
    comparable_scope: str,
    group_filter: Optional[set[int]],
    variant_type_filter: Optional[set[int]],
    pred_payload: Dict[str, List[int]],
) -> Optional[Dict[str, Any]]:
    idx = _scope_indices(
        groups=pred_payload["groups"],
        variant_types=pred_payload["variant_types"],
        group_filter=group_filter,
        variant_type_filter=variant_type_filter,
    )

    if not idx:
        return None

    sub = _subset_payload(pred_payload, idx)
    m = summarize_predictions(**sub)

    group_filter_str = (
        "ALL"
        if group_filter is None
        else ",".join(str(x) for x in sorted(group_filter))
    )
    variant_type_filter_str = (
        "ALL"
        if variant_type_filter is None
        else ",".join(vt_name(x) for x in sorted(variant_type_filter))
    )

    return {
        "source": source,
        "name": name,
        "partition": partition,
        "scope": scope,
        "scope_type": scope_type,
        "comparable_scope": comparable_scope,
        "groups": group_filter_str,
        "variant_type": variant_type_filter_str,

        "n": m["overall_3c"]["n"],

        "acc": m["overall_3c"]["acc"],
        "macro_f1": m["overall_3c"]["macro_f1"],
        "macro_recall": m["overall_3c"]["macro_recall"],
        "macro_precision": m["overall_3c"]["macro_precision"],
        "weighted_f1": m["overall_3c"]["weighted_f1"],

        "variant_precision": m["overall_variant_from_cm3"]["precision"],
        "variant_recall": m["overall_variant_from_cm3"]["recall"],
        "variant_specificity": m["overall_variant_from_cm3"]["specificity"],
        "variant_balanced_acc": m["overall_variant_from_cm3"]["balanced_acc"],
        "variant_f1": m["overall_variant_from_cm3"]["f1"],
        "variant_f2": m["overall_variant_from_cm3"]["f2"],
        "variant_tp": m["overall_variant_from_cm3"]["tp"],
        "variant_tn": m["overall_variant_from_cm3"]["tn"],
        "variant_fp": m["overall_variant_from_cm3"]["fp"],
        "variant_fn": m["overall_variant_from_cm3"]["fn"],

        "binary_acc": m["overall_binary_variant_vs_no"]["acc"],
        "binary_balanced_acc": m["overall_binary_variant_vs_no"]["balanced_acc"],
        "binary_precision": m["overall_binary_variant_vs_no"]["precision"],
        "binary_recall": m["overall_binary_variant_vs_no"]["recall"],
        "binary_specificity": m["overall_binary_variant_vs_no"]["specificity"],
        "binary_f1": m["overall_binary_variant_vs_no"]["f1"],
        "binary_f2": m["overall_binary_variant_vs_no"]["f2"],
    }


def build_long_summary_rows(
    *,
    source: str,
    name: str,
    partition: str,
    pred_payload: Dict[str, List[int]],
    eval_profile: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Build a long/tidy CSV summary.

    Comparable view scopes:
    - illumina_view: groups {10, 11}
    - ont_view: groups {1, 11}
    - hybrid_all: groups {1, 10, 11}

    Fair view+variant scopes:
    - illumina_view_SNP
    - illumina_view_INDEL
    - ont_view_SNP
    - ont_view_INDEL
    - hybrid_all_SNP
    - hybrid_all_INDEL

    Diagnostic scopes:
    - group_1 / group_10 / group_11
    - variant_SNP / variant_INDEL
    """

    rows: List[Dict[str, Any]] = []

    model_kind = eval_profile.get("model_kind")
    modality = eval_profile.get("modality")
    model_family = eval_profile.get("model_family")

    scope_defs: List[Dict[str, Any]] = []

    if model_kind == "teacher":
        if modality == "illumina":
            scope_defs.append({
                "scope": "illumina_view",
                "scope_type": "view",
                "comparable_scope": "illumina_view",
                "group_filter": {10, 11},
                "variant_type_filter": None,
            })
        elif modality == "ont":
            scope_defs.append({
                "scope": "ont_view",
                "scope_type": "view",
                "comparable_scope": "ont_view",
                "group_filter": {1, 11},
                "variant_type_filter": None,
            })

    elif model_kind == "unimodal":
        if modality == "illumina":
            scope_defs.append({
                "scope": "illumina_view",
                "scope_type": "view",
                "comparable_scope": "illumina_view",
                "group_filter": {10, 11},
                "variant_type_filter": None,
            })
        elif modality == "ont":
            scope_defs.append({
                "scope": "ont_view",
                "scope_type": "view",
                "comparable_scope": "ont_view",
                "group_filter": {1, 11},
                "variant_type_filter": None,
            })

    elif model_kind == "hybrid":
        scope_defs.extend([
            {
                "scope": "hybrid_all",
                "scope_type": "view",
                "comparable_scope": "hybrid_all",
                "group_filter": {1, 10, 11},
                "variant_type_filter": None,
            },
            {
                "scope": "illumina_view",
                "scope_type": "view",
                "comparable_scope": "illumina_view",
                "group_filter": {10, 11},
                "variant_type_filter": None,
            },
            {
                "scope": "ont_view",
                "scope_type": "view",
                "comparable_scope": "ont_view",
                "group_filter": {1, 11},
                "variant_type_filter": None,
            },
        ])

    # Fair view+variant scopes.
    # These are the important ones for comparing SNP/INDEL fairly.
    available_vts = sorted(set(int(v) for v in pred_payload["variant_types"]))
    view_scope_defs = [sd for sd in scope_defs if sd["scope_type"] == "view"]

    for base in view_scope_defs:
        for vt in available_vts:
            if vt not in {1, 2}:
                continue

            vt_label = vt_name(vt)

            scope_defs.append({
                "scope": f"{base['scope']}_{vt_label}",
                "scope_type": "view_variant_type",
                "comparable_scope": f"{base['comparable_scope']}_{vt_label}",
                "group_filter": base["group_filter"],
                "variant_type_filter": {vt},
            })

    # Diagnostic group-level scopes.
    available_groups = sorted(set(int(g) for g in pred_payload["groups"]))
    for g in available_groups:
        scope_defs.append({
            "scope": f"group_{g}",
            "scope_type": "group",
            "comparable_scope": f"group_{g}",
            "group_filter": {g},
            "variant_type_filter": None,
        })

    # Diagnostic variant-type scopes.
    # These are less fair for cross-model comparison than view+variant scopes,
    # but still useful for quick diagnostics.
    for vt in available_vts:
        if vt not in {1, 2}:
            continue

        scope_defs.append({
            "scope": f"variant_{vt_name(vt)}",
            "scope_type": "variant_type",
            "comparable_scope": f"variant_{vt_name(vt)}",
            "group_filter": None,
            "variant_type_filter": {vt},
        })

    for sd in scope_defs:
        row = _scope_metric_row(
            source=source,
            name=name,
            partition=partition,
            scope=sd["scope"],
            scope_type=sd["scope_type"],
            comparable_scope=sd["comparable_scope"],
            group_filter=sd["group_filter"],
            variant_type_filter=sd["variant_type_filter"],
            pred_payload=pred_payload,
        )
        if row is not None:
            row["model_kind"] = model_kind
            row["modality"] = modality or ""
            row["model_family"] = model_family or ""
            rows.append(row)

    return rows


def write_summary_csv_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "source",
        "model_kind",
        "modality",
        "model_family",
        "name",
        "partition",
        "scope",
        "scope_type",
        "comparable_scope",
        "groups",
        "variant_type",

        "n",

        "acc",
        "macro_f1",
        "macro_recall",
        "macro_precision",
        "weighted_f1",

        "variant_precision",
        "variant_recall",
        "variant_specificity",
        "variant_balanced_acc",
        "variant_f1",
        "variant_f2",
        "variant_tp",
        "variant_tn",
        "variant_fp",
        "variant_fn",

        "binary_acc",
        "binary_balanced_acc",
        "binary_precision",
        "binary_recall",
        "binary_specificity",
        "binary_f1",
        "binary_f2",
    ]

    write_header = not path.exists()

    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--source", choices=["checkpoint", "teacher"], required=True)

    # Checkpoint mode
    ap.add_argument("--experiment_dir", type=str, default=None)
    ap.add_argument("--checkpoint", type=str, default="best.pt")

    # Teacher mode
    ap.add_argument("--teacher", choices=["illumina", "ill", "ont"], default=None)

    # Shared
    ap.add_argument("--resolved_split", type=str, default=None)
    ap.add_argument("--partition", choices=["train", "val", "test"], default="test")
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--out_json", type=str, default=None)
    ap.add_argument("--out_csv", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--name", type=str, default=None)

    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve() if args.out_dir else None

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("[ERROR] --device cuda requested but CUDA is not available.")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.source == "teacher":
        if args.teacher is None:
            raise SystemExit("[ERROR] --teacher is required when --source teacher")
        if args.resolved_split is None:
            raise SystemExit("[ERROR] --resolved_split is required when --source teacher")

        modality = teacher_modality_name(args.teacher)
        rs = load_resolved_split(args.resolved_split)

        index = build_partition_index_unimodal(rs, args.partition, modality)
        print_partition_summary(summarize_partition(index))
        print()

        pred_payload = predict_teacher(index, teacher=modality)
        metrics = summarize_predictions(**pred_payload)

        canonical_teacher_name = f"dv_{modality}"
        name = args.name or canonical_teacher_name

        if args.name is not None and args.name != canonical_teacher_name:
            print(
                f"[WARN] Explicit --name={args.name!r} does not match canonical "
                f"teacher name {canonical_teacher_name!r}. Keeping explicit name."
            )
        
        experiment_dir = None

        source_payload = {
            "source": "teacher",
            "teacher": modality,
            "teacher_arrays": modality_to_teacher_keys(modality),
        }
        
        eval_profile = {
            "model_kind": "teacher",
            "modality": modality,
            "model_family": None,
        }

    else:
        if args.experiment_dir is None:
            raise SystemExit("[ERROR] --experiment_dir is required when --source checkpoint")

        experiment_dir = Path(args.experiment_dir).resolve()
        metadata = load_experiment_metadata(experiment_dir)

        if args.resolved_split is not None:
            rs = load_resolved_split(args.resolved_split)
            resolved_split_source = str(Path(args.resolved_split).resolve())
        else:
            if metadata["resolved_split"] is None:
                raise SystemExit(
                    "[ERROR] No --resolved_split passed and experiment_dir/resolved_split.used.json not found."
                )
            rs = metadata["resolved_split"]
            resolved_split_source = metadata["resolved_split_used_path"]

        ckpt_path = experiment_dir / args.checkpoint
        if not ckpt_path.exists():
            raise SystemExit(f"[ERROR] checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        info = infer_checkpoint_model_info(ckpt, metadata["config"])

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
        model.to(device)
        model.eval()

        if info["model_kind"] == "unimodal":
            modality = info["modality"]
            index = build_partition_index_unimodal(rs, args.partition, modality)
            print_partition_summary(summarize_partition(index))
            print()

            pred_payload = predict_checkpoint_unimodal(
                index=index,
                model=model,
                modality=modality,
                device=device,
                batch_size=args.batch_size,
            )

            name = args.name or f"{experiment_dir.name}_{Path(args.checkpoint).stem}"

            eval_profile = {
                "model_kind": "unimodal",
                "modality": modality,
                "model_family": None,
            }

        else:
            model_family = info["model_family"]
            add_masks = bool(info["add_masks"])

            index = build_partition_index_hybrid(rs, args.partition, add_masks=add_masks)
            print_partition_summary(summarize_partition(index))
            print()

            pred_payload = predict_checkpoint_hybrid(
                index=index,
                model=model,
                model_family=model_family,
                device=device,
                batch_size=args.batch_size,
                add_masks=add_masks,
            )

            name = args.name or f"{experiment_dir.name}_{Path(args.checkpoint).stem}"

            eval_profile = {
                "model_kind": "hybrid",
                "modality": "hybrid",
                "model_family": model_family,
            }

        metrics = summarize_predictions(**pred_payload)

        source_payload = {
            "source": "checkpoint",
            "experiment_dir": str(experiment_dir),
            "checkpoint": str(ckpt_path),
            "checkpoint_info": info,
            "resolved_split_source": resolved_split_source,
            "experiment_metadata": {
                "config_path": metadata["config_path"],
                "experiment_config_used_path": metadata["experiment_config_used_path"],
                "resolved_split_used_path": metadata["resolved_split_used_path"],
            },
        }

    out_json_default, out_csv_default = default_output_paths(
        source=args.source,
        experiment_dir=experiment_dir,
        out_dir=out_dir,
        name=name,
        partition=args.partition,
    )

    out_json = Path(args.out_json).resolve() if args.out_json else out_json_default
    out_csv = Path(args.out_csv).resolve() if args.out_csv else out_csv_default

    report = {
        "name": name,
        "partition": args.partition,
        "source": source_payload,
        "data_summary": summarize_partition(index),
        "metrics": metrics,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(_jsonify(report), indent=2), encoding="utf-8")

    rows = build_long_summary_rows(
        source=args.source,
        name=name,
        partition=args.partition,
        pred_payload=pred_payload,
        eval_profile=eval_profile,
    )
    write_summary_csv_rows(out_csv, rows)

    primary_row = None
    for r in rows:
        if r["scope_type"] == "view":
            primary_row = r
            break

    if primary_row is None:
        primary_row = rows[0]

    print(f"[Saved] JSON report: {out_json}")
    print(f"[Saved] CSV summary: {out_csv}")

    print("\n[Summary]")
    print(f"  name             : {name}")
    print(f"  partition        : {args.partition}")
    print(f"  primary_scope    : {primary_row['scope']}")
    print(f"  n                : {int(primary_row['n']):,}")
    print(f"  acc              : {float(primary_row['acc']):.6f}")
    print(f"  macro_f1         : {float(primary_row['macro_f1']):.6f}")
    print(f"  weighted_f1      : {float(primary_row['weighted_f1']):.6f}")
    print(f"  variant_precision: {float(primary_row['variant_precision']):.6f}")
    print(f"  variant_recall   : {float(primary_row['variant_recall']):.6f}")
    print(f"  variant_f1       : {float(primary_row['variant_f1']):.6f}")
    print(f"  csv_rows_written : {len(rows)}")


if __name__ == "__main__":
    main()
