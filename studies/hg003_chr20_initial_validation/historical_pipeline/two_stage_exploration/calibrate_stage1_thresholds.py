#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Post-hoc threshold calibration for stage-1 binary hybrid models.

What it does
------------
- Loads an already-trained stage-1 checkpoint (shared or groupwise).
- Rebuilds the validation/test splits with the same OUTER + split_json + meta_csv policy.
- Computes raw probabilities on val/test.
- Selects either:
    * one global threshold
    * one threshold per group {1,10,11}
- Applies the selected threshold(s) to val/test and saves a JSON report.

Supported rules
---------------
- recall_at_min_precision
- recall_at_max_fpr
- precision_at_min_recall
- best_f2
- best_balanced_acc

Recommended first experiment
----------------------------
- shared + global threshold
- shared + per-group thresholds
- groupwise + global threshold
- groupwise + per-group thresholds

Example
-------
python training/scripts/hybrid_dv/calibrate_stage1_thresholds.py \
  --ckpt training/out/models/stage1/hybrid_stage1_groupwise_mlp/best.pt \
  --threshold_mode per_group \
  --g1_rule recall_at_max_fpr --g1_max_fpr 0.01 \
  --g10_rule best_f2 \
  --g11_rule precision_at_min_recall --g11_min_recall 0.999 \
  --out_json training/out/models/stage1/hybrid_stage1_groupwise_mlp/calib_per_group.json
"""

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def as_int(x, default=-1):
    try:
        if x is None or x == "":
            return default
        return int(x)
    except Exception:
        return default


def parse_variant_type(x) -> str:
    if x is None:
        return "unknown"

    try:
        xi = int(x)
        if xi == 1:
            return "SNP"
        if xi == 2:
            return "INDEL"
    except Exception:
        pass

    s = str(x).strip()
    if not s:
        return "unknown"

    su = s.upper()
    if su in {"SNP", "SNV"}:
        return "SNP"
    if su == "INDEL":
        return "INDEL"

    return s


@dataclass
class Ref:
    shard: str
    row: int
    locus: str
    y3: int
    y_bin: int
    bin_id: int
    group: int
    mask_ill: int
    mask_ont: int
    label_ill: int
    label_ont: int
    label_mismatch: int
    vt_mismatch: int
    variant_type: str


def list_shards(outer_dir: Path) -> List[Path]:
    shards = sorted(outer_dir.glob("*.npz"))
    if not shards:
        raise SystemExit(f"No shards found in {outer_dir}")
    return shards


def load_meta(meta_csv: Path) -> Dict[str, Dict[str, Any]]:
    out = {}
    with open(meta_csv, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            locus = row["locus"]
            label_ill = as_int(row.get("label_ill"), -1)
            label_ont = as_int(row.get("label_ont"), -1)
            vt_mismatch = as_int(row.get("vt_mismatch"), 0)
            label_mismatch = int(label_ill >= 0 and label_ont >= 0 and label_ill != label_ont)

            variant_type = "unknown"
            if "variant_type" in row:
                variant_type = parse_variant_type(row.get("variant_type"))

            out[locus] = {
                "label_ill": label_ill,
                "label_ont": label_ont,
                "vt_mismatch": vt_mismatch,
                "label_mismatch": label_mismatch,
                "variant_type": variant_type,
            }
    return out


def build_refs_all(outer_dir: Path, meta_lookup: Dict[str, Dict[str, Any]], bin_size: int) -> List[Ref]:
    refs: List[Ref] = []
    for sp in list_shards(outer_dir):
        npz = np.load(sp, allow_pickle=True)

        loci = npz["locus"]
        if loci.dtype.kind in ("S", "O"):
            loci_list = [x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x) for x in loci.tolist()]
        else:
            loci_list = [str(x) for x in loci.tolist()]

        grp = npz["group"].astype(np.int64).reshape(-1)
        mi = npz["mask_ill"].astype(np.int64).reshape(-1)
        mo = npz["mask_ont"].astype(np.int64).reshape(-1)

        for i, locus in enumerate(loci_list):
            pos = parse_locus_pos(locus)
            if pos is None:
                continue
            meta = meta_lookup.get(locus)
            if meta is None:
                continue

            group = int(grp[i])
            label_ill = int(meta["label_ill"])
            label_ont = int(meta["label_ont"])
            label_mismatch = int(meta["label_mismatch"])
            vt_mismatch = int(meta["vt_mismatch"])
            variant_type = str(meta.get("variant_type", "unknown"))

            if group == 1:
                y3 = label_ont
                if y3 < 0:
                    continue
            elif group == 10:
                y3 = label_ill
                if y3 < 0:
                    continue
            elif group == 11:
                if label_ill < 0 or label_ont < 0:
                    continue
                if label_mismatch == 1:
                    continue
                y3 = label_ill
            else:
                continue

            y_bin = 0 if int(y3) == 0 else 1

            refs.append(
                Ref(
                    shard=str(sp),
                    row=i,
                    locus=locus,
                    y3=int(y3),
                    y_bin=int(y_bin),
                    bin_id=int(pos // int(bin_size)),
                    group=group,
                    mask_ill=int(mi[i]),
                    mask_ont=int(mo[i]),
                    label_ill=label_ill,
                    label_ont=label_ont,
                    label_mismatch=label_mismatch,
                    vt_mismatch=vt_mismatch,
                    variant_type=variant_type,
                )
            )

        npz.close()
    return refs


def group_by_shard(refs: List[Ref]) -> Dict[str, List[Ref]]:
    out: Dict[str, List[Ref]] = {}
    for r in refs:
        out.setdefault(r.shard, []).append(r)
    for k in out:
        out[k].sort(key=lambda rr: rr.row)
    return out


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


def build_model(in_dim: int, arch: str, hidden: int, dropout: float, head_mode: str) -> nn.Module:
    if head_mode == "shared":
        if arch == "linear":
            return BinaryLinearHead(in_dim=in_dim)
        return BinaryMLP(in_dim=in_dim, hidden=hidden, dropout=dropout)

    return GroupwiseBinaryMLP(
        in_dim=in_dim,
        hidden=hidden,
        dropout=dropout,
        trunk_mode=arch,
    )


def binary_confusion(y_true: List[int], y_pred: List[int]) -> Tuple[int, int, int, int]:
    tp = tn = fp = fn = 0
    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1:
            tp += 1
        elif t == 0 and p == 0:
            tn += 1
        elif t == 0 and p == 1:
            fp += 1
        elif t == 1 and p == 0:
            fn += 1
    return tp, tn, fp, fn


def metrics_from_ytrue_ypred(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
    eps = 1e-12
    tp, tn, fp, fn = binary_confusion(y_true, y_pred)

    precision = float(tp / max(tp + fp, eps))
    recall = float(tp / max(tp + fn, eps))
    specificity = float(tn / max(tn + fp, eps))
    fpr = float(fp / max(fp + tn, eps))
    acc = float((tp + tn) / max(tp + tn + fp + fn, 1))
    balanced_acc = float((recall + specificity) / 2.0)
    f1 = float(2 * precision * recall / max(precision + recall, eps))
    beta = 2.0
    f2 = float((1 + beta ** 2) * precision * recall / max(beta ** 2 * precision + recall, eps))
    positive_rate = float((tp + fp) / max(tp + tn + fp + fn, 1))

    return {
        "acc": acc,
        "balanced_acc": balanced_acc,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "fpr": fpr,
        "f1": f1,
        "f2": f2,
        "positive_rate": positive_rate,
        "cm": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "support": {
            "neg": int(tn + fp),
            "pos": int(tp + fn),
            "total": int(tp + tn + fp + fn),
        },
    }


def binary_metrics_from_probs(y_true: List[int], probs: List[float], thr: float) -> Dict[str, Any]:
    y_pred = [1 if p >= thr else 0 for p in probs]
    out = {"threshold": float(thr)}
    out.update(metrics_from_ytrue_ypred(y_true, y_pred))
    return out


def pr_auc_from_probs(y_true: List[int], probs: List[float]) -> float:
    if len(y_true) == 0:
        return 0.0
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(probs, dtype=np.float64)

    n_pos = int((y == 1).sum())
    if n_pos == 0:
        return 0.0

    order = np.argsort(-p)
    y = y[order]

    tp = 0
    fp = 0
    prev_recall = 0.0
    auc = 0.0

    for i in range(len(y)):
        if y[i] == 1:
            tp += 1
        else:
            fp += 1

        recall = tp / max(n_pos, 1)
        precision = tp / max(tp + fp, 1)
        auc += (recall - prev_recall) * precision
        prev_recall = recall

    return float(auc)


def build_thresholded_preds(
    probs: List[float],
    groups: List[int],
    threshold_mode: str,
    global_threshold: Optional[float] = None,
    thresholds_by_group: Optional[Dict[str, float]] = None,
    default_threshold: Optional[float] = None,
) -> List[int]:
    preds = []
    if threshold_mode == "global":
        if global_threshold is None:
            raise ValueError("global_threshold is required for threshold_mode='global'")
        for p in probs:
            preds.append(1 if p >= global_threshold else 0)
        return preds

    if thresholds_by_group is None:
        raise ValueError("thresholds_by_group is required for threshold_mode='per_group'")

    for p, g in zip(probs, groups):
        key = str(int(g))
        thr = thresholds_by_group.get(key, default_threshold)
        if thr is None:
            raise ValueError(f"Missing threshold for group={key} and no default_threshold was provided")
        preds.append(1 if p >= thr else 0)

    return preds


def summarize_slices_from_preds(
    y_true: List[int],
    y_pred: List[int],
    probs: List[float],
    groups: Optional[List[int]] = None,
    variant_types: Optional[List[str]] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    if groups is not None:
        by_group: Dict[str, Any] = {}
        for gg in (1, 10, 11):
            idx = [i for i, g in enumerate(groups) if int(g) == gg]
            if not idx:
                continue
            yg = [y_true[i] for i in idx]
            pg = [y_pred[i] for i in idx]
            probg = [probs[i] for i in idx]
            by_group[str(gg)] = {
                "n": len(idx),
                **metrics_from_ytrue_ypred(yg, pg),
                "pr_auc": pr_auc_from_probs(yg, probg),
            }
        out["by_group"] = by_group

    if variant_types is not None:
        by_variant_type: Dict[str, Any] = {}
        uniq_vt = sorted(set(str(v) for v in variant_types))
        for vt in uniq_vt:
            idx = [i for i, v in enumerate(variant_types) if str(v) == vt]
            if not idx:
                continue
            yv = [y_true[i] for i in idx]
            pv = [y_pred[i] for i in idx]
            probv = [probs[i] for i in idx]
            by_variant_type[str(vt)] = {
                "n": len(idx),
                **metrics_from_ytrue_ypred(yv, pv),
                "pr_auc": pr_auc_from_probs(yv, probv),
            }
        out["by_variant_type"] = by_variant_type

        if groups is not None:
            by_group_variant_type: Dict[str, Any] = {}
            for gg in (1, 10, 11):
                row = {}
                for vt in uniq_vt:
                    idx = [i for i, (g, v) in enumerate(zip(groups, variant_types)) if int(g) == gg and str(v) == vt]
                    if not idx:
                        continue
                    ygv = [y_true[i] for i in idx]
                    pgv = [y_pred[i] for i in idx]
                    probgv = [probs[i] for i in idx]
                    row[str(vt)] = {
                        "n": len(idx),
                        **metrics_from_ytrue_ypred(ygv, pgv),
                        "pr_auc": pr_auc_from_probs(ygv, probgv),
                    }
                if row:
                    by_group_variant_type[str(gg)] = row
            out["by_group_variant_type"] = by_group_variant_type

    return out


def _add_selection_meta(m: Dict[str, Any], mode: str, constraint_name: Optional[str], constraint_value: Optional[float]) -> Dict[str, Any]:
    out = dict(m)
    out["selection_mode"] = mode
    if constraint_name is not None:
        out[constraint_name] = float(constraint_value)
    return out


def select_threshold_with_rule(
    y_true: List[int],
    probs: List[float],
    rule: str,
    grid_min: float,
    grid_max: float,
    grid_step: float,
    min_precision: float,
    max_fpr: float,
    min_recall: float,
) -> Dict[str, Any]:
    candidates = []
    thr = grid_min
    while thr <= grid_max + 1e-12:
        candidates.append(binary_metrics_from_probs(y_true, probs, float(thr)))
        thr += grid_step

    if rule == "recall_at_min_precision":
        valid = [m for m in candidates if m["precision"] >= float(min_precision)]
        if valid:
            valid.sort(key=lambda m: (m["recall"], m["f2"], m["balanced_acc"], -m["fpr"]), reverse=True)
            return _add_selection_meta(valid[0], "recall_at_min_precision", "min_precision", min_precision)
        candidates.sort(key=lambda m: (m["f2"], m["recall"], m["balanced_acc"], -m["fpr"]), reverse=True)
        return _add_selection_meta(candidates[0], "fallback_best_f2", "min_precision", min_precision)

    if rule == "recall_at_max_fpr":
        valid = [m for m in candidates if m["fpr"] <= float(max_fpr)]
        if valid:
            valid.sort(key=lambda m: (m["recall"], m["f2"], m["precision"], m["specificity"]), reverse=True)
            return _add_selection_meta(valid[0], "recall_at_max_fpr", "max_fpr", max_fpr)
        candidates.sort(key=lambda m: (m["f2"], m["recall"], m["precision"], -m["fpr"]), reverse=True)
        return _add_selection_meta(candidates[0], "fallback_best_f2", "max_fpr", max_fpr)

    if rule == "precision_at_min_recall":
        valid = [m for m in candidates if m["recall"] >= float(min_recall)]
        if valid:
            valid.sort(key=lambda m: (m["precision"], m["f2"], m["balanced_acc"], -m["fpr"]), reverse=True)
            return _add_selection_meta(valid[0], "precision_at_min_recall", "min_recall", min_recall)
        candidates.sort(key=lambda m: (m["f2"], m["precision"], m["balanced_acc"], -m["fpr"]), reverse=True)
        return _add_selection_meta(candidates[0], "fallback_best_f2", "min_recall", min_recall)

    if rule == "best_balanced_acc":
        candidates.sort(key=lambda m: (m["balanced_acc"], m["f2"], m["recall"], m["precision"]), reverse=True)
        return _add_selection_meta(candidates[0], "best_balanced_acc", None, None)

    # default: best_f2
    candidates.sort(key=lambda m: (m["f2"], m["recall"], m["precision"], m["balanced_acc"]), reverse=True)
    return _add_selection_meta(candidates[0], "best_f2", None, None)


@torch.no_grad()
def eval_sharded_binary_raw(
    model,
    by_shard: Dict[str, List[Ref]],
    device,
    batch_size: int,
    add_masks: bool,
    head_mode: str,
) -> Dict[str, Any]:
    model.eval()

    y_true: List[int] = []
    probs: List[float] = []
    groups: List[int] = []
    variant_types: List[str] = []

    for shard, items in by_shard.items():
        npz = np.load(shard, allow_pickle=True)
        ill_e = npz["ill_embeddings"].astype(np.float32, copy=False)
        ont_e = npz["ont_embeddings"].astype(np.float32, copy=False)

        rows = np.array([r.row for r in items], dtype=np.int64)
        ys = np.array([r.y_bin for r in items], dtype=np.int64)
        gs = np.array([r.group for r in items], dtype=np.int64)
        mi = np.array([r.mask_ill for r in items], dtype=np.float32)
        mo = np.array([r.mask_ont for r in items], dtype=np.float32)
        vts = [r.variant_type for r in items]

        for i in range(0, len(rows), batch_size):
            rr = rows[i:i + batch_size]
            yb = ys[i:i + batch_size]
            gb = gs[i:i + batch_size]
            mib = mi[i:i + batch_size]
            mob = mo[i:i + batch_size]
            vtb = vts[i:i + batch_size]

            x = np.concatenate([ill_e[rr], ont_e[rr]], axis=1)
            if add_masks:
                masks = np.stack([mib, mob], axis=1).astype(np.float32)
                x = np.concatenate([x, masks], axis=1)

            x = torch.from_numpy(x).to(device)

            if head_mode == "groupwise":
                g = torch.from_numpy(gb.astype(np.int64)).to(device)
                logits = model(x, g)
            else:
                logits = model(x)

            pb = torch.sigmoid(logits).detach().cpu().numpy().astype(float)

            y_true += yb.tolist()
            probs += pb.tolist()
            groups += gb.tolist()
            variant_types += list(vtb)

        npz.close()

    return {
        "n": len(y_true),
        "y_true": y_true,
        "probs": probs,
        "groups": groups,
        "variant_type": variant_types,
        "pr_auc": pr_auc_from_probs(y_true, probs),
    }


def finalize_eval(
    raw_eval: Dict[str, Any],
    threshold_mode: str,
    global_threshold: Optional[float] = None,
    thresholds_by_group: Optional[Dict[str, float]] = None,
    default_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    y_pred = build_thresholded_preds(
        probs=raw_eval["probs"],
        groups=raw_eval["groups"],
        threshold_mode=threshold_mode,
        global_threshold=global_threshold,
        thresholds_by_group=thresholds_by_group,
        default_threshold=default_threshold,
    )

    out = {
        "n": int(raw_eval["n"]),
        "pr_auc": float(raw_eval["pr_auc"]),
        "threshold_mode": threshold_mode,
    }

    if threshold_mode == "global":
        out["threshold"] = float(global_threshold)
    else:
        out["thresholds_by_group"] = {str(k): float(v) for k, v in sorted((thresholds_by_group or {}).items())}
        if default_threshold is not None:
            out["default_threshold"] = float(default_threshold)

    out.update(metrics_from_ytrue_ypred(raw_eval["y_true"], y_pred))
    out.update(
        summarize_slices_from_preds(
            y_true=raw_eval["y_true"],
            y_pred=y_pred,
            probs=raw_eval["probs"],
            groups=raw_eval.get("groups"),
            variant_types=raw_eval.get("variant_type"),
        )
    )
    return out


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


def resolve_paths_from_checkpoint(ckpt_args: Dict[str, Any], cli_args) -> Tuple[Path, Path, Path, int]:
    outer_dir = Path(cli_args.outer_dir if cli_args.outer_dir is not None else ckpt_args["outer_dir"])
    split_json = Path(cli_args.split_json if cli_args.split_json is not None else ckpt_args["split_json"])
    meta_csv = Path(cli_args.meta_csv if cli_args.meta_csv is not None else ckpt_args["meta_csv"])
    bin_size = int(cli_args.bin_size if cli_args.bin_size is not None else ckpt_args.get("bin_size", 1_000_000))
    return outer_dir, split_json, meta_csv, bin_size


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--ckpt", type=str, required=True)

    ap.add_argument("--outer_dir", type=str, default=None)
    ap.add_argument("--split_json", type=str, default=None)
    ap.add_argument("--meta_csv", type=str, default=None)
    ap.add_argument("--bin_size", type=int, default=None)

    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--out_json", type=str, default=None)

    ap.add_argument("--threshold_mode", choices=["global", "per_group"], default="per_group")

    ap.add_argument("--grid_min", type=float, default=0.05)
    ap.add_argument("--grid_max", type=float, default=0.95)
    ap.add_argument("--grid_step", type=float, default=0.01)

    ap.add_argument("--global_rule", choices=[
        "recall_at_min_precision", "recall_at_max_fpr", "precision_at_min_recall",
        "best_f2", "best_balanced_acc"
    ], default="recall_at_min_precision")
    ap.add_argument("--global_min_precision", type=float, default=0.60)
    ap.add_argument("--global_max_fpr", type=float, default=0.01)
    ap.add_argument("--global_min_recall", type=float, default=0.999)

    ap.add_argument("--g1_rule", choices=[
        "recall_at_min_precision", "recall_at_max_fpr", "precision_at_min_recall",
        "best_f2", "best_balanced_acc"
    ], default="recall_at_max_fpr")
    ap.add_argument("--g1_min_precision", type=float, default=0.60)
    ap.add_argument("--g1_max_fpr", type=float, default=0.01)
    ap.add_argument("--g1_min_recall", type=float, default=0.999)

    ap.add_argument("--g10_rule", choices=[
        "recall_at_min_precision", "recall_at_max_fpr", "precision_at_min_recall",
        "best_f2", "best_balanced_acc"
    ], default="best_f2")
    ap.add_argument("--g10_min_precision", type=float, default=0.75)
    ap.add_argument("--g10_max_fpr", type=float, default=0.02)
    ap.add_argument("--g10_min_recall", type=float, default=0.995)

    ap.add_argument("--g11_rule", choices=[
        "recall_at_min_precision", "recall_at_max_fpr", "precision_at_min_recall",
        "best_f2", "best_balanced_acc"
    ], default="precision_at_min_recall")
    ap.add_argument("--g11_min_precision", type=float, default=0.98)
    ap.add_argument("--g11_max_fpr", type=float, default=0.25)
    ap.add_argument("--g11_min_recall", type=float, default=0.999)

    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    payload = torch.load(ckpt_path, map_location="cpu")
    ckpt_args = payload.get("args", {})

    outer_dir, split_json, meta_csv, bin_size = resolve_paths_from_checkpoint(ckpt_args, args)

    batch_size = int(args.batch_size if args.batch_size is not None else ckpt_args.get("batch_size", 2048))
    out_json = Path(args.out_json) if args.out_json else ckpt_path.parent / f"threshold_calibration_{args.threshold_mode}.json"

    arch = str(ckpt_args.get("arch", "mlp"))
    head_mode = str(ckpt_args.get("head_mode", "shared"))
    hidden = int(ckpt_args.get("hidden", 512))
    dropout = float(ckpt_args.get("dropout", 0.1))

    add_masks = True
    if "no_add_masks" in ckpt_args and ckpt_args["no_add_masks"]:
        add_masks = False
    elif "add_masks" in ckpt_args and ckpt_args["add_masks"]:
        add_masks = True

    split = json.loads(split_json.read_text(encoding="utf-8"))
    split_bin_size = int(split.get("bin_size", -1))
    if split_bin_size != int(bin_size):
        raise SystemExit(f"bin_size mismatch: split_json has {split_bin_size}, but resolved bin_size is {bin_size}")

    meta_lookup = load_meta(meta_csv)

    print(f"[Index] Scanning shards in {outer_dir} ...")
    refs_all = build_refs_all(outer_dir, meta_lookup=meta_lookup, bin_size=bin_size)
    if not refs_all:
        raise SystemExit("No refs indexed after applying hybrid consensus policy.")

    train_bins = set(split["train_bins"])
    val_bins = set(split["val_bins"])
    test_bins = set(split["test_bins"])

    val_refs = [r for r in refs_all if r.bin_id in val_bins]
    test_refs = [r for r in refs_all if r.bin_id in test_bins]

    val_by_shard = group_by_shard(val_refs)
    test_by_shard = group_by_shard(test_refs)

    in_dim = int(payload.get("input_dim", -1))
    if in_dim <= 0:
        one_shard = next(iter(val_by_shard.keys()))
        npz0 = np.load(one_shard, allow_pickle=True)
        in_dim = int(npz0["ill_embeddings"].shape[1] + npz0["ont_embeddings"].shape[1])
        npz0.close()
        if add_masks:
            in_dim += 2

    device = torch.device("cpu")
    model = build_model(
        in_dim=in_dim,
        arch=arch,
        hidden=hidden,
        dropout=dropout,
        head_mode=head_mode,
    ).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()

    print(f"[Model] arch={arch} head_mode={head_mode} in_dim={in_dim} add_masks={add_masks}")
    print(f"[Eval] batch_size={batch_size} threshold_mode={args.threshold_mode}")

    val_raw = eval_sharded_binary_raw(
        model=model,
        by_shard=val_by_shard,
        device=device,
        batch_size=batch_size,
        add_masks=add_masks,
        head_mode=head_mode,
    )

    test_raw = eval_sharded_binary_raw(
        model=model,
        by_shard=test_by_shard,
        device=device,
        batch_size=batch_size,
        add_masks=add_masks,
        head_mode=head_mode,
    )

    selection_info: Dict[str, Any] = {}

    if args.threshold_mode == "global":
        selected = select_threshold_with_rule(
            y_true=val_raw["y_true"],
            probs=val_raw["probs"],
            rule=args.global_rule,
            grid_min=args.grid_min,
            grid_max=args.grid_max,
            grid_step=args.grid_step,
            min_precision=args.global_min_precision,
            max_fpr=args.global_max_fpr,
            min_recall=args.global_min_recall,
        )
        global_thr = float(selected["threshold"])
        thresholds_by_group = None
        default_threshold = None
        selection_info["global"] = selected

        val_metrics = finalize_eval(
            raw_eval=val_raw,
            threshold_mode="global",
            global_threshold=global_thr,
        )
        test_metrics = finalize_eval(
            raw_eval=test_raw,
            threshold_mode="global",
            global_threshold=global_thr,
        )

    else:
        thresholds_by_group: Dict[str, float] = {}
        group_rules = {
            "1": (args.g1_rule, args.g1_min_precision, args.g1_max_fpr, args.g1_min_recall),
            "10": (args.g10_rule, args.g10_min_precision, args.g10_max_fpr, args.g10_min_recall),
            "11": (args.g11_rule, args.g11_min_precision, args.g11_max_fpr, args.g11_min_recall),
        }

        global_fallback = select_threshold_with_rule(
            y_true=val_raw["y_true"],
            probs=val_raw["probs"],
            rule=args.global_rule,
            grid_min=args.grid_min,
            grid_max=args.grid_max,
            grid_step=args.grid_step,
            min_precision=args.global_min_precision,
            max_fpr=args.global_max_fpr,
            min_recall=args.global_min_recall,
        )
        default_threshold = float(global_fallback["threshold"])
        selection_info["global_fallback"] = global_fallback

        for gg in ("1", "10", "11"):
            idx = [i for i, g in enumerate(val_raw["groups"]) if int(g) == int(gg)]
            if not idx:
                thresholds_by_group[gg] = default_threshold
                selection_info[gg] = {
                    "selection_mode": "missing_group_on_val_use_global_fallback",
                    "threshold": default_threshold,
                }
                continue

            yg = [val_raw["y_true"][i] for i in idx]
            pg = [val_raw["probs"][i] for i in idx]
            rule, min_prec, max_fpr, min_rec = group_rules[gg]

            selected = select_threshold_with_rule(
                y_true=yg,
                probs=pg,
                rule=rule,
                grid_min=args.grid_min,
                grid_max=args.grid_max,
                grid_step=args.grid_step,
                min_precision=min_prec,
                max_fpr=max_fpr,
                min_recall=min_rec,
            )
            thresholds_by_group[gg] = float(selected["threshold"])
            selection_info[gg] = selected

        global_thr = None

        val_metrics = finalize_eval(
            raw_eval=val_raw,
            threshold_mode="per_group",
            thresholds_by_group=thresholds_by_group,
            default_threshold=default_threshold,
        )
        test_metrics = finalize_eval(
            raw_eval=test_raw,
            threshold_mode="per_group",
            thresholds_by_group=thresholds_by_group,
            default_threshold=default_threshold,
        )

    report = {
        "checkpoint": str(ckpt_path),
        "resolved_paths": {
            "outer_dir": str(outer_dir),
            "split_json": str(split_json),
            "meta_csv": str(meta_csv),
            "bin_size": int(bin_size),
        },
        "model": {
            "arch": arch,
            "head_mode": head_mode,
            "hidden": hidden,
            "dropout": dropout,
            "add_masks": add_masks,
            "input_dim": in_dim,
        },
        "threshold_config": {
            "threshold_mode": args.threshold_mode,
            "grid_min": args.grid_min,
            "grid_max": args.grid_max,
            "grid_step": args.grid_step,
            "global_rule": args.global_rule,
            "global_min_precision": args.global_min_precision,
            "global_max_fpr": args.global_max_fpr,
            "global_min_recall": args.global_min_recall,
            "g1_rule": args.g1_rule,
            "g1_min_precision": args.g1_min_precision,
            "g1_max_fpr": args.g1_max_fpr,
            "g1_min_recall": args.g1_min_recall,
            "g10_rule": args.g10_rule,
            "g10_min_precision": args.g10_min_precision,
            "g10_max_fpr": args.g10_max_fpr,
            "g10_min_recall": args.g10_min_recall,
            "g11_rule": args.g11_rule,
            "g11_min_precision": args.g11_min_precision,
            "g11_max_fpr": args.g11_max_fpr,
            "g11_min_recall": args.g11_min_recall,
        },
        "selection_info": selection_info,
        "val": val_metrics,
        "test": test_metrics,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(_jsonify(report), indent=2), encoding="utf-8")

    if args.threshold_mode == "global":
        print(f"[Selected] global threshold = {global_thr:.4f}")
    else:
        print(f"[Selected] thresholds_by_group = {thresholds_by_group} | default={default_threshold:.4f}")

    print(
        f"[VAL ] prAUC={val_metrics['pr_auc']:.4f} acc={val_metrics['acc']:.4f} "
        f"balAcc={val_metrics['balanced_acc']:.4f} prec={val_metrics['precision']:.4f} "
        f"rec={val_metrics['recall']:.4f} f2={val_metrics['f2']:.4f}"
    )
    print(
        f"[TEST] prAUC={test_metrics['pr_auc']:.4f} acc={test_metrics['acc']:.4f} "
        f"balAcc={test_metrics['balanced_acc']:.4f} prec={test_metrics['precision']:.4f} "
        f"rec={test_metrics['recall']:.4f} f2={test_metrics['f2']:.4f}"
    )
    print(f"[Saved] {out_json}")


if __name__ == "__main__":
    main()