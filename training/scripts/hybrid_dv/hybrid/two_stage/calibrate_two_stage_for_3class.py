#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fast post-hoc calibration of stage-1 thresholds for a full 2-stage hybrid pipeline,
scoring the FINAL 3-class caller instead of the binary stage-1 task.

Key idea
--------
Run stage1 and stage2 ONCE on validation, cache:
- y_true_3c
- group
- stage1_prob
- stage2_pred12

Then sweep thresholds offline without rerunning the networks.

Supports
--------
- global threshold
- per-group thresholds (coordinate ascent)
- score on final 3-class metrics:
    * macro_f1 (default)
    * weighted_f1
    * acc
    * macro_recall
    * macro_precision
"""

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn


_POS_RE = re.compile(r"(?:chr)?([0-9XYM]+)[\:\-_](\d+)", re.IGNORECASE)


def parse_locus_pos(locus: str) -> Optional[int]:
    m = _POS_RE.search(locus)
    if m:
        return int(m.group(2))
    toks = re.split(r"[^0-9]+", locus)
    toks = [t for t in toks if t]
    if toks:
        return int(toks[0])
    return None


def as_int(x, default: int = -1) -> int:
    try:
        if x is None or x == "":
            return default
        return int(x)
    except Exception:
        return default


def resolve_add_masks(ckpt_args: Dict[str, Any], default: bool = True) -> bool:
    if bool(ckpt_args.get("no_add_masks", False)):
        return False
    if bool(ckpt_args.get("add_masks", False)):
        return True
    return default


def parse_groups_csv(s: str) -> Set[int]:
    out = set()
    for tok in str(s).split(","):
        tok = tok.strip()
        if tok:
            out.add(int(tok))
    return out


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


@dataclass
class Item:
    shard: str
    row: int
    locus: str
    y3: int
    group: int
    bin_id: int
    vt_mismatch: int
    mask_ill: int
    mask_ont: int


# -----------------------------
# Model defs
# -----------------------------
class BinaryLinearHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x, groups=None):
        return self.fc(x).squeeze(1)


class BinaryMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, groups=None):
        return self.net(x).squeeze(1)


class GroupwiseBinaryMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float, trunk_mode: str = "mlp"):
        super().__init__()
        if trunk_mode == "linear":
            self.trunk = nn.Identity()
            trunk_dim = in_dim
        else:
            self.trunk = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            trunk_dim = hidden

        self.heads = nn.ModuleDict({
            "1": nn.Linear(trunk_dim, 1),
            "10": nn.Linear(trunk_dim, 1),
            "11": nn.Linear(trunk_dim, 1),
        })

    def forward(self, x, groups):
        h = self.trunk(x)
        out = torch.zeros(h.shape[0], dtype=h.dtype, device=h.device)
        for g in (1, 10, 11):
            mask = (groups == g)
            if mask.any():
                out[mask] = self.heads[str(g)](h[mask]).squeeze(1)
        return out


class TwoClassLinearHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 2)

    def forward(self, x, groups=None):
        return self.fc(x)


class TwoClassMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, x, groups=None):
        return self.net(x)


class GroupwiseTwoClassHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float, trunk_mode: str = "mlp"):
        super().__init__()
        if trunk_mode == "linear":
            self.trunk = nn.Identity()
            trunk_dim = in_dim
        else:
            self.trunk = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            trunk_dim = hidden

        self.heads = nn.ModuleDict({
            "1": nn.Linear(trunk_dim, 2),
            "10": nn.Linear(trunk_dim, 2),
            "11": nn.Linear(trunk_dim, 2),
        })

    def forward(self, x, groups):
        h = self.trunk(x)
        out = torch.zeros((h.shape[0], 2), dtype=h.dtype, device=h.device)
        for g in (1, 10, 11):
            mask = (groups == g)
            if mask.any():
                out[mask] = self.heads[str(g)](h[mask])
        return out


def build_stage1_model(ckpt_args: Dict[str, Any], input_dim: int) -> nn.Module:
    arch = ckpt_args.get("arch", "mlp")
    head_mode = ckpt_args.get("head_mode", "shared")
    hidden = int(ckpt_args.get("hidden", 512))
    dropout = float(ckpt_args.get("dropout", 0.1))

    if head_mode == "shared":
        if arch == "linear":
            return BinaryLinearHead(input_dim)
        return BinaryMLP(input_dim, hidden=hidden, dropout=dropout)

    return GroupwiseBinaryMLP(
        in_dim=input_dim,
        hidden=hidden,
        dropout=dropout,
        trunk_mode=arch,
    )


def build_stage2_model(ckpt_args: Dict[str, Any], input_dim: int) -> nn.Module:
    arch = ckpt_args.get("arch", "mlp")
    head_mode = ckpt_args.get("head_mode", "groupwise")
    hidden = int(ckpt_args.get("hidden", 512))
    dropout = float(ckpt_args.get("dropout", 0.1))

    if head_mode == "shared":
        if arch == "linear":
            return TwoClassLinearHead(input_dim)
        return TwoClassMLP(input_dim, hidden=hidden, dropout=dropout)

    return GroupwiseTwoClassHead(
        in_dim=input_dim,
        hidden=hidden,
        dropout=dropout,
        trunk_mode=arch,
    )


# -----------------------------
# Data
# -----------------------------
def load_meta(meta_csv: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with open(meta_csv, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            locus = row["locus"]
            label_ill = as_int(row.get("label_ill"), -1)
            label_ont = as_int(row.get("label_ont"), -1)
            vt_mismatch = as_int(row.get("vt_mismatch"), 0)

            out[locus] = {
                "label_ill": label_ill,
                "label_ont": label_ont,
                "vt_mismatch": vt_mismatch,
                "label_mismatch": int(label_ill >= 0 and label_ont >= 0 and label_ill != label_ont),
            }
    return out


def list_shards(outer_dir: Path) -> List[Path]:
    shards = sorted(outer_dir.glob("*.npz"))
    if not shards:
        raise SystemExit(f"No shards found in {outer_dir}")
    return shards


def split_bins_for_name(split_payload: Dict[str, Any], split_name: str) -> Set[int]:
    if split_name == "val":
        return set(split_payload["val_bins"])
    if split_name == "test":
        return set(split_payload["test_bins"])
    raise ValueError(f"Unsupported split={split_name}")


def build_items(
    outer_dir: Path,
    meta_lookup: Dict[str, Dict[str, Any]],
    split_payload: Dict[str, Any],
    split_name: str,
    bin_size: int,
    groups_keep: Set[int],
    exclude_vt_mismatch: bool,
) -> Tuple[List[Item], Dict[str, Any]]:
    target_bins = split_bins_for_name(split_payload, split_name)

    items: List[Item] = []
    stats = {
        "split": split_name,
        "groups_keep": sorted(list(groups_keep)),
        "exclude_vt_mismatch": bool(exclude_vt_mismatch),
        "scanned_rows": 0,
        "kept": 0,
        "dropped_invalid_pos": 0,
        "dropped_missing_meta": 0,
        "dropped_split": 0,
        "dropped_group": 0,
        "dropped_missing_target": 0,
        "dropped_group11_label_mismatch": 0,
        "dropped_vt_mismatch": 0,
    }

    for sp in list_shards(outer_dir):
        npz = np.load(sp, allow_pickle=True)

        loci = npz["locus"]
        if loci.dtype.kind in ("S", "O"):
            loci_list = [
                x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x)
                for x in loci.tolist()
            ]
        else:
            loci_list = [str(x) for x in loci.tolist()]

        grp = npz["group"].astype(np.int64).reshape(-1)
        mi = npz["mask_ill"].astype(np.int64).reshape(-1)
        mo = npz["mask_ont"].astype(np.int64).reshape(-1)

        for i, locus in enumerate(loci_list):
            stats["scanned_rows"] += 1

            pos = parse_locus_pos(locus)
            if pos is None:
                stats["dropped_invalid_pos"] += 1
                continue
            bin_id = int(pos // int(bin_size))
            if bin_id not in target_bins:
                stats["dropped_split"] += 1
                continue

            meta = meta_lookup.get(locus)
            if meta is None:
                stats["dropped_missing_meta"] += 1
                continue

            group = int(grp[i])
            if group not in groups_keep:
                stats["dropped_group"] += 1
                continue

            label_ill = int(meta["label_ill"])
            label_ont = int(meta["label_ont"])
            label_mismatch = int(meta["label_mismatch"])
            vt_mismatch = int(meta["vt_mismatch"])

            if exclude_vt_mismatch and vt_mismatch == 1:
                stats["dropped_vt_mismatch"] += 1
                continue

            if group == 1:
                y3 = label_ont
                if y3 < 0:
                    stats["dropped_missing_target"] += 1
                    continue
            elif group == 10:
                y3 = label_ill
                if y3 < 0:
                    stats["dropped_missing_target"] += 1
                    continue
            elif group == 11:
                if label_ill < 0 or label_ont < 0:
                    stats["dropped_missing_target"] += 1
                    continue
                if label_mismatch == 1:
                    stats["dropped_group11_label_mismatch"] += 1
                    continue
                y3 = label_ill
            else:
                stats["dropped_group"] += 1
                continue

            items.append(
                Item(
                    shard=str(sp),
                    row=i,
                    locus=locus,
                    y3=int(y3),
                    group=group,
                    bin_id=bin_id,
                    vt_mismatch=vt_mismatch,
                    mask_ill=int(mi[i]),
                    mask_ont=int(mo[i]),
                )
            )
            stats["kept"] += 1

        npz.close()

    return items, stats


def group_by_shard(items: List[Item]) -> Dict[str, List[Item]]:
    out: Dict[str, List[Item]] = {}
    for it in items:
        out.setdefault(it.shard, []).append(it)
    for k in out:
        out[k].sort(key=lambda x: x.row)
    return out


# -----------------------------
# Metrics
# -----------------------------
def cm_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> np.ndarray:
    cm = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        cm[int(t), int(p)] += 1
    return cm


def metrics_from_cm(cm: np.ndarray) -> Dict[str, Any]:
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
        f1 = float(2.0 * prec * rec / (prec + rec + eps))

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
    weighted_f1 = float(np.sum(np.asarray(f1s) * np.asarray(supports)) / (float(np.sum(supports)) + eps))

    return {
        "cm": cm.tolist(),
        "acc": acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
    }


def score_from_metrics(m: Dict[str, Any], metric: str) -> float:
    if metric == "acc":
        return float(m["acc"])
    if metric == "weighted_f1":
        return float(m["weighted_f1"])
    if metric == "macro_recall":
        return float(m["macro_recall"])
    if metric == "macro_precision":
        return float(m["macro_precision"])
    return float(m["macro_f1"])


# -----------------------------
# Cache one pass over val
# -----------------------------
@torch.no_grad()
def cache_val_outputs(
    items_by_shard: Dict[str, List[Item]],
    stage1_model: nn.Module,
    stage2_model: nn.Module,
    stage1_head_mode: str,
    stage2_head_mode: str,
    stage1_add_masks: bool,
    stage2_add_masks: bool,
    device: torch.device,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    stage1_model.eval()
    stage2_model.eval()

    y_true_3c: List[int] = []
    groups: List[int] = []
    stage1_prob: List[float] = []
    stage2_pred12: List[int] = []

    for shard, items in items_by_shard.items():
        npz = np.load(shard, allow_pickle=True)
        ill_e = npz["ill_embeddings"].astype(np.float32, copy=False)
        ont_e = npz["ont_embeddings"].astype(np.float32, copy=False)
        mi = npz["mask_ill"].astype(np.float32, copy=False).reshape(-1, 1)
        mo = npz["mask_ont"].astype(np.float32, copy=False).reshape(-1, 1)

        rows = np.array([it.row for it in items], dtype=np.int64)
        ys3 = np.array([it.y3 for it in items], dtype=np.int64)
        gs = np.array([it.group for it in items], dtype=np.int64)

        for i in range(0, len(rows), batch_size):
            rr = rows[i:i + batch_size]
            y3b = ys3[i:i + batch_size]
            gb = gs[i:i + batch_size]

            x_base = np.concatenate([ill_e[rr], ont_e[rr]], axis=1)

            if stage1_add_masks:
                x1_np = np.concatenate([x_base, mi[rr], mo[rr]], axis=1)
            else:
                x1_np = x_base

            if stage2_add_masks:
                x2_np = np.concatenate([x_base, mi[rr], mo[rr]], axis=1)
            else:
                x2_np = x_base

            x1 = torch.from_numpy(x1_np).to(device)
            x2 = torch.from_numpy(x2_np).to(device)
            g = torch.from_numpy(gb.astype(np.int64)).to(device)

            if stage1_head_mode == "groupwise":
                s1_logits = stage1_model(x1, g)
            else:
                s1_logits = stage1_model(x1)

            if stage2_head_mode == "groupwise":
                s2_logits = stage2_model(x2, g)
            else:
                s2_logits = stage2_model(x2)

            s1_prob = torch.sigmoid(s1_logits).cpu().numpy().astype(np.float64)
            s2_pred = torch.argmax(s2_logits, dim=1).cpu().numpy().astype(np.int64)

            y_true_3c.extend(y3b.tolist())
            groups.extend(gb.tolist())
            stage1_prob.extend(s1_prob.tolist())
            stage2_pred12.extend(s2_pred.tolist())

        npz.close()

    return {
        "y_true_3c": np.asarray(y_true_3c, dtype=np.int64),
        "groups": np.asarray(groups, dtype=np.int64),
        "stage1_prob": np.asarray(stage1_prob, dtype=np.float64),
        "stage2_pred12": np.asarray(stage2_pred12, dtype=np.int64),
    }


def evaluate_cached_global(cache: Dict[str, np.ndarray], thr: float) -> Dict[str, Any]:
    y_true = cache["y_true_3c"]
    prob = cache["stage1_prob"]
    pred12 = cache["stage2_pred12"]
    groups = cache["groups"]

    pred_bin = (prob >= float(thr)).astype(np.int64)
    pred3_variant = np.where(pred12 == 0, 1, 2)
    y_pred = np.where(pred_bin == 0, 0, pred3_variant)

    cm = cm_k(y_true, y_pred, 3)
    out = {"n": int(len(y_true)), **metrics_from_cm(cm), "by_group": {}}

    for gg in sorted(np.unique(groups).tolist()):
        idx = groups == int(gg)
        cmg = cm_k(y_true[idx], y_pred[idx], 3)
        out["by_group"][str(int(gg))] = {"n": int(idx.sum()), **metrics_from_cm(cmg)}

    return out


def evaluate_cached_per_group(cache: Dict[str, np.ndarray], thresholds_by_group: Dict[int, float]) -> Dict[str, Any]:
    y_true = cache["y_true_3c"]
    prob = cache["stage1_prob"]
    pred12 = cache["stage2_pred12"]
    groups = cache["groups"]

    thr_vec = np.zeros_like(prob, dtype=np.float64)
    for gg, thr in thresholds_by_group.items():
        thr_vec[groups == int(gg)] = float(thr)

    pred_bin = (prob >= thr_vec).astype(np.int64)
    pred3_variant = np.where(pred12 == 0, 1, 2)
    y_pred = np.where(pred_bin == 0, 0, pred3_variant)

    cm = cm_k(y_true, y_pred, 3)
    out = {"n": int(len(y_true)), **metrics_from_cm(cm), "by_group": {}}

    for gg in sorted(np.unique(groups).tolist()):
        idx = groups == int(gg)
        cmg = cm_k(y_true[idx], y_pred[idx], 3)
        out["by_group"][str(int(gg))] = {"n": int(idx.sum()), **metrics_from_cm(cmg)}

    return out


def candidate_thresholds(grid_min: float, grid_max: float, grid_step: float) -> List[float]:
    vals = []
    thr = float(grid_min)
    while thr <= float(grid_max) + 1e-12:
        vals.append(round(float(thr), 10))
        thr += float(grid_step)
    return vals


def best_global_threshold(cache: Dict[str, np.ndarray], score_metric: str, grid_min: float, grid_max: float, grid_step: float) -> Dict[str, Any]:
    best = None
    for thr in candidate_thresholds(grid_min, grid_max, grid_step):
        m = evaluate_cached_global(cache, thr)
        score = score_from_metrics(m, score_metric)
        cand = {
            "threshold": float(thr),
            "score_metric": score_metric,
            "score": float(score),
            "metrics": m,
        }
        if best is None or cand["score"] > best["score"]:
            best = cand
    return best


def best_threshold_for_single_group(
    target_group: int,
    cache: Dict[str, np.ndarray],
    score_metric: str,
    grid_min: float,
    grid_max: float,
    grid_step: float,
    fixed_thresholds: Dict[int, float],
) -> Dict[str, Any]:
    best = None
    for thr in candidate_thresholds(grid_min, grid_max, grid_step):
        thresholds = dict(fixed_thresholds)
        thresholds[int(target_group)] = float(thr)
        m = evaluate_cached_per_group(cache, thresholds)
        score = score_from_metrics(m, score_metric)
        cand = {
            "threshold": float(thr),
            "score_metric": score_metric,
            "score": float(score),
            "metrics": m,
        }
        if best is None or cand["score"] > best["score"]:
            best = cand
    return best


def best_per_group_thresholds(
    cache: Dict[str, np.ndarray],
    score_metric: str,
    grid_min: float,
    grid_max: float,
    grid_step: float,
    n_rounds: int = 2,
) -> Dict[str, Any]:
    global_best = best_global_threshold(cache, score_metric, grid_min, grid_max, grid_step)
    thresholds = {
        1: float(global_best["threshold"]),
        10: float(global_best["threshold"]),
        11: float(global_best["threshold"]),
    }
    selection_trace = {"init_global": global_best}

    for rnd in range(n_rounds):
        round_trace = {}
        for gg in (1, 10, 11):
            best_g = best_threshold_for_single_group(
                target_group=gg,
                cache=cache,
                score_metric=score_metric,
                grid_min=grid_min,
                grid_max=grid_max,
                grid_step=grid_step,
                fixed_thresholds=thresholds,
            )
            thresholds[gg] = float(best_g["threshold"])
            round_trace[str(gg)] = best_g
        selection_trace[f"round_{rnd+1}"] = round_trace

    final_metrics = evaluate_cached_per_group(cache, thresholds)
    final_score = score_from_metrics(final_metrics, score_metric)

    return {
        "thresholds_by_group": {str(k): float(v) for k, v in thresholds.items()},
        "score_metric": score_metric,
        "score": float(final_score),
        "metrics": final_metrics,
        "selection_trace": selection_trace,
    }


def resolve_paths(
    stage1_args: Dict[str, Any],
    stage2_args: Dict[str, Any],
    cli_args,
) -> Tuple[Path, Path, Path, int]:
    outer_dir = Path(
        cli_args.outer_dir
        if cli_args.outer_dir is not None
        else stage1_args.get("outer_dir", stage2_args.get("outer_dir"))
    )
    split_bins_json = Path(
        cli_args.split_bins_json
        if cli_args.split_bins_json is not None
        else stage1_args.get("split_json", stage2_args.get("split_json"))
    )
    meta_with_vt_csv = Path(
        cli_args.meta_with_vt_csv
        if cli_args.meta_with_vt_csv is not None
        else stage1_args.get("meta_csv", stage2_args.get("meta_csv"))
    )
    bin_size = int(
        cli_args.bin_size
        if cli_args.bin_size is not None
        else stage1_args.get("bin_size", stage2_args.get("bin_size", 1_000_000))
    )
    return outer_dir, split_bins_json, meta_with_vt_csv, bin_size


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--stage1_ckpt", type=str, required=True)
    ap.add_argument("--stage2_ckpt", type=str, required=True)

    ap.add_argument("--outer_dir", type=str, default=None)
    ap.add_argument("--split_bins_json", type=str, default=None)
    ap.add_argument("--meta_with_vt_csv", type=str, default=None)
    ap.add_argument("--bin_size", type=int, default=None)

    ap.add_argument("--split", choices=["val"], default="val")
    ap.add_argument("--groups", type=str, default="1,10,11")
    ap.add_argument("--exclude_vt_mismatch", action="store_true")

    ap.add_argument("--threshold_mode", choices=["global", "per_group"], default="per_group")
    ap.add_argument("--score_metric", choices=["macro_f1", "weighted_f1", "acc", "macro_recall", "macro_precision"], default="macro_f1")
    ap.add_argument("--grid_min", type=float, default=0.05)
    ap.add_argument("--grid_max", type=float, default=0.95)
    ap.add_argument("--grid_step", type=float, default=0.01)
    ap.add_argument("--n_rounds", type=int, default=2)

    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--out_json", type=str, required=True)

    args = ap.parse_args()

    stage1_ckpt = Path(args.stage1_ckpt)
    stage2_ckpt = Path(args.stage2_ckpt)

    if not stage1_ckpt.exists():
        raise SystemExit(f"Stage1 checkpoint not found: {stage1_ckpt}")
    if not stage2_ckpt.exists():
        raise SystemExit(f"Stage2 checkpoint not found: {stage2_ckpt}")

    stage1_payload = torch.load(stage1_ckpt, map_location="cpu")
    stage2_payload = torch.load(stage2_ckpt, map_location="cpu")

    stage1_args = stage1_payload.get("args", {})
    stage2_args = stage2_payload.get("args", {})

    outer_dir, split_bins_json, meta_with_vt_csv, bin_size = resolve_paths(stage1_args, stage2_args, args)
    groups_keep = parse_groups_csv(args.groups)

    batch_size = int(
        args.batch_size
        if args.batch_size is not None
        else max(int(stage1_args.get("batch_size", 2048)), int(stage2_args.get("batch_size", 2048)))
    )

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    split_payload = json.loads(split_bins_json.read_text(encoding="utf-8"))
    meta_lookup = load_meta(meta_with_vt_csv)

    items, filter_stats = build_items(
        outer_dir=outer_dir,
        meta_lookup=meta_lookup,
        split_payload=split_payload,
        split_name=args.split,
        bin_size=bin_size,
        groups_keep=groups_keep,
        exclude_vt_mismatch=bool(args.exclude_vt_mismatch),
    )

    if not items:
        raise SystemExit("No items left after filtering.")

    items_by_shard = group_by_shard(items)

    first_shard = next(iter(items_by_shard.keys()))
    npz0 = np.load(first_shard, allow_pickle=True)
    base_dim = int(npz0["ill_embeddings"].shape[1] + npz0["ont_embeddings"].shape[1])
    npz0.close()

    stage1_add_masks = resolve_add_masks(stage1_args, default=True)
    stage2_add_masks = resolve_add_masks(stage2_args, default=True)
    stage1_input_dim = base_dim + (2 if stage1_add_masks else 0)
    stage2_input_dim = base_dim + (2 if stage2_add_masks else 0)

    stage1_model = build_stage1_model(stage1_args, stage1_input_dim).to(device)
    stage2_model = build_stage2_model(stage2_args, stage2_input_dim).to(device)

    stage1_model.load_state_dict(stage1_payload["model_state"])
    stage2_model.load_state_dict(stage2_payload["model_state"])

    stage1_head_mode = stage1_args.get("head_mode", "shared")
    stage2_head_mode = stage2_args.get("head_mode", "groupwise")

    print("[Cache] Running one pass over validation to cache stage1 probabilities and stage2 predictions...")
    cache = cache_val_outputs(
        items_by_shard=items_by_shard,
        stage1_model=stage1_model,
        stage2_model=stage2_model,
        stage1_head_mode=stage1_head_mode,
        stage2_head_mode=stage2_head_mode,
        stage1_add_masks=stage1_add_masks,
        stage2_add_masks=stage2_add_masks,
        device=device,
        batch_size=batch_size,
    )
    print(f"[Cache] Cached n={len(cache['y_true_3c'])}")

    if args.threshold_mode == "global":
        best = best_global_threshold(
            cache=cache,
            score_metric=args.score_metric,
            grid_min=args.grid_min,
            grid_max=args.grid_max,
            grid_step=args.grid_step,
        )
        val_block = {
            "threshold": float(best["threshold"]),
            "score_metric": args.score_metric,
            "best_score": float(best["score"]),
            **best["metrics"],
        }
        out = {
            "split": args.split,
            "threshold_mode": "global",
            "checkpoint_stage1": str(stage1_ckpt),
            "checkpoint_stage2": str(stage2_ckpt),
            "resolved_paths": {
                "outer_dir": str(outer_dir),
                "split_bins_json": str(split_bins_json),
                "meta_with_vt_csv": str(meta_with_vt_csv),
                "bin_size": int(bin_size),
            },
            "groups_keep": sorted(list(groups_keep)),
            "exclude_vt_mismatch": bool(args.exclude_vt_mismatch),
            "filter_stats": filter_stats,
            "val": val_block,
        }
    else:
        best = best_per_group_thresholds(
            cache=cache,
            score_metric=args.score_metric,
            grid_min=args.grid_min,
            grid_max=args.grid_max,
            grid_step=args.grid_step,
            n_rounds=args.n_rounds,
        )
        val_block = {
            "threshold_mode": "per_group",
            "thresholds_by_group": best["thresholds_by_group"],
            "default_threshold": float(best["selection_trace"]["init_global"]["threshold"]),
            "score_metric": args.score_metric,
            "best_score": float(best["score"]),
            "selection_info": best["selection_trace"],
            **best["metrics"],
        }
        out = {
            "split": args.split,
            "threshold_mode": "per_group",
            "checkpoint_stage1": str(stage1_ckpt),
            "checkpoint_stage2": str(stage2_ckpt),
            "resolved_paths": {
                "outer_dir": str(outer_dir),
                "split_bins_json": str(split_bins_json),
                "meta_with_vt_csv": str(meta_with_vt_csv),
                "bin_size": int(bin_size),
            },
            "groups_keep": sorted(list(groups_keep)),
            "exclude_vt_mismatch": bool(args.exclude_vt_mismatch),
            "filter_stats": filter_stats,
            "val": val_block,
        }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(_jsonify(out), indent=2), encoding="utf-8")

    if args.threshold_mode == "global":
        print(
            f"[Selected] global threshold={out['val']['threshold']:.2f} "
            f"| {args.score_metric}={out['val']['best_score']:.6f}"
        )
    else:
        print(
            f"[Selected] per-group thresholds={out['val']['thresholds_by_group']} "
            f"| {args.score_metric}={out['val']['best_score']:.6f}"
        )

    print(
        f"[VAL 3C] acc={out['val']['acc']:.4f} "
        f"macroF1={out['val']['macro_f1']:.4f} "
        f"wF1={out['val']['weighted_f1']:.4f}"
    )
    print(f"[Saved] {out_json}")


if __name__ == "__main__":
    main()