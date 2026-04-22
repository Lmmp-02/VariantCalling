#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate a full 2-stage hybrid OUTER pipeline as a final 3-class caller.

Pipeline
--------
- Stage 1: binary 0 vs {1,2}
- Stage 2: variant-only 1 vs 2
- Final prediction:
    * if stage1 says "no variant" -> y_pred = 0
    * else use stage2:
        - stage2 pred 0 -> final y_pred = 1
        - stage2 pred 1 -> final y_pred = 2

Benchmark policy
----------------
- group01 -> label_ont
- group10 -> label_ill
- group11 -> keep only consensus loci where label_ill == label_ont
- vt_mismatch kept by default
- optional --exclude_vt_mismatch

Outputs
-------
Main JSON schema is aligned with eval_outer_ckpt_with_vt.py:
- overall_3c
- overall_binary_variant_vs_no
- by_variant_type
- by_group

Also includes diagnostics:
- stage1_binary
- stage2_variant_only_true_variants

Examples
--------
Using stage1 threshold stored in best.pt:
python training/scripts/hybrid_dv/analysis/eval_outer_two_stage_with_vt.py \
  --stage1_ckpt training/out/models/stage1/hybrid_stage1_groupwise_mlp/best.pt \
  --stage2_ckpt training/out/models/stage2/hybrid_stage2_groupwise_mlp/best.pt \
  --split test \
  --out_json training/out/reports/outer_chr20_v1/two_stage_groupwise_test.json

Using post-hoc calibrated stage1 thresholds:
python training/scripts/hybrid_dv/analysis/eval_outer_two_stage_with_vt.py \
  --stage1_ckpt training/out/models/stage1/hybrid_stage1_groupwise_mlp/best.pt \
  --stage1_calib_json training/out/models/stage1/hybrid_stage1_groupwise_mlp/threshold_calibration_per_group.json \
  --stage2_ckpt training/out/models/stage2/hybrid_stage2_groupwise_mlp/best.pt \
  --split test \
  --out_json training/out/reports/outer_chr20_v1/two_stage_groupwise_calib_test.json
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


def vt_to_int(x) -> int:
    if x is None:
        return 0
    try:
        xi = int(x)
        return xi if xi in (1, 2) else 0
    except Exception:
        pass

    s = str(x).strip().upper()
    if s in {"SNP", "SNV"}:
        return 1
    if s == "INDEL":
        return 2
    return 0


def vt_name(v: int) -> str:
    return "SNP" if v == 1 else ("INDEL" if v == 2 else "UNKNOWN")


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
    vt: int
    vt_mismatch: int
    mask_ill: int
    mask_ont: int


# -----------------------------
# Model definitions
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
# Data loading / indexing
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

            vt = 0
            if "variant_type" in row:
                vt = vt_to_int(row.get("variant_type"))
            elif "vt_name" in row:
                vt = vt_to_int(row.get("vt_name"))

            out[locus] = {
                "label_ill": label_ill,
                "label_ont": label_ont,
                "vt": vt,
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
            vt = int(meta["vt"])
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
                    vt=vt,
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
def cm_k(y_true: List[int], y_pred: List[int], k: int) -> np.ndarray:
    cm = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
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


def binary_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
    eps = 1e-12
    yt = np.asarray(y_true, dtype=np.int64)
    yp = np.asarray(y_pred, dtype=np.int64)

    tp = int(((yt == 1) & (yp == 1)).sum())
    tn = int(((yt == 0) & (yp == 0)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())

    acc = float((tp + tn) / max(len(yt), 1))
    precision = float(tp / (tp + fp + eps))
    recall = float(tp / (tp + fn + eps))
    specificity = float(tn / (tn + fp + eps))
    balanced_acc = float(0.5 * (recall + specificity))
    f1 = float(2.0 * precision * recall / (precision + recall + eps))
    f2 = float((5.0 * precision * recall) / (4.0 * precision + recall + eps))

    return {
        "cm": [[tn, fp], [fn, tp]],
        "acc": acc,
        "balanced_acc": balanced_acc,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "f2": f2,
        "support": {
            "neg": int(tn + fp),
            "pos": int(tp + fn),
            "total": int(tp + tn + fp + fn),
        },
    }


# -----------------------------
# Threshold resolution for stage1
# -----------------------------
def resolve_stage1_thresholds(
    stage1_payload: Dict[str, Any],
    calib_json: Optional[Path],
) -> Dict[str, Any]:
    if calib_json is not None:
        report = json.loads(calib_json.read_text(encoding="utf-8"))
        holder = report.get("val", report.get("test", report))

        threshold_mode = holder.get("threshold_mode", None)
        if threshold_mode == "per_group":
            return {
                "source": str(calib_json),
                "mode": "per_group",
                "thresholds_by_group": {
                    str(k): float(v) for k, v in holder.get("thresholds_by_group", {}).items()
                },
                "default_threshold": float(holder.get("default_threshold", 0.5)),
                "selection_info": report.get("selection_info", None),
            }

        if "threshold" in holder:
            return {
                "source": str(calib_json),
                "mode": "global",
                "threshold": float(holder["threshold"]),
                "selection_info": report.get("selection_info", None),
            }

        raise SystemExit(f"Could not resolve thresholds from calibration JSON: {calib_json}")

    if "threshold" in stage1_payload:
        return {
            "source": "stage1_ckpt",
            "mode": "global",
            "threshold": float(stage1_payload["threshold"]),
            "selection_info": stage1_payload.get("threshold_selection_info", None),
        }

    return {
        "source": "fallback_default",
        "mode": "global",
        "threshold": 0.5,
        "selection_info": None,
    }


def threshold_for_group(th_cfg: Dict[str, Any], group: int) -> float:
    if th_cfg["mode"] == "global":
        return float(th_cfg["threshold"])
    thresholds_by_group = th_cfg.get("thresholds_by_group", {})
    if str(group) in thresholds_by_group:
        return float(thresholds_by_group[str(group)])
    return float(th_cfg.get("default_threshold", 0.5))


# -----------------------------
# End-to-end eval
# -----------------------------
@torch.no_grad()
def eval_two_stage(
    items_by_shard: Dict[str, List[Item]],
    stage1_model: nn.Module,
    stage2_model: nn.Module,
    stage1_head_mode: str,
    stage2_head_mode: str,
    stage1_add_masks: bool,
    stage2_add_masks: bool,
    stage1_threshold_cfg: Dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> Dict[str, Any]:
    stage1_model.eval()
    stage2_model.eval()

    y_true_3c: List[int] = []
    y_pred_3c: List[int] = []
    group_list: List[int] = []
    vt_list: List[int] = []

    y_true_stage1_bin: List[int] = []
    y_pred_stage1_bin: List[int] = []
    stage1_prob_list: List[float] = []
    stage1_thr_list: List[float] = []

    y_true_stage2_truevar: List[int] = []
    y_pred_stage2_truevar: List[int] = []
    group_stage2_truevar: List[int] = []
    vt_stage2_truevar: List[int] = []

    for shard, items in items_by_shard.items():
        npz = np.load(shard, allow_pickle=True)
        ill_e = npz["ill_embeddings"].astype(np.float32, copy=False)
        ont_e = npz["ont_embeddings"].astype(np.float32, copy=False)
        mi = npz["mask_ill"].astype(np.float32, copy=False).reshape(-1, 1)
        mo = npz["mask_ont"].astype(np.float32, copy=False).reshape(-1, 1)

        rows = np.array([it.row for it in items], dtype=np.int64)
        ys3 = np.array([it.y3 for it in items], dtype=np.int64)
        gs = np.array([it.group for it in items], dtype=np.int64)
        vts = np.array([it.vt for it in items], dtype=np.int64)

        for i in range(0, len(rows), batch_size):
            rr = rows[i:i + batch_size]
            y3b = ys3[i:i + batch_size]
            gb = gs[i:i + batch_size]
            vtb = vts[i:i + batch_size]

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
            g = torch.from_numpy(gb).to(device)

            # Stage 1
            if stage1_head_mode == "groupwise":
                stage1_logits = stage1_model(x1, g)
            else:
                stage1_logits = stage1_model(x1)

            stage1_probs = torch.sigmoid(stage1_logits).cpu().numpy().astype(np.float64)

            # Stage 2
            if stage2_head_mode == "groupwise":
                stage2_logits = stage2_model(x2, g)
            else:
                stage2_logits = stage2_model(x2)

            stage2_pred12 = torch.argmax(stage2_logits, dim=1).cpu().numpy().astype(np.int64)

            stage1_pred_bin = []
            final_pred3 = []

            for prob, pred12, yy3, gg in zip(stage1_probs.tolist(), stage2_pred12.tolist(), y3b.tolist(), gb.tolist()):
                thr = threshold_for_group(stage1_threshold_cfg, int(gg))
                pred_bin = 1 if float(prob) >= float(thr) else 0
                pred3_variant = 1 if int(pred12) == 0 else 2
                pred3 = 0 if pred_bin == 0 else pred3_variant

                stage1_pred_bin.append(pred_bin)
                final_pred3.append(pred3)

                y_true_stage1_bin.append(0 if int(yy3) == 0 else 1)
                y_pred_stage1_bin.append(pred_bin)
                stage1_prob_list.append(float(prob))
                stage1_thr_list.append(float(thr))

                if int(yy3) in (1, 2):
                    y_true_stage2_truevar.append(0 if int(yy3) == 1 else 1)
                    y_pred_stage2_truevar.append(int(pred12))
                    group_stage2_truevar.append(int(gg))
                    vt_stage2_truevar.append(int(vtb[len(y_true_stage2_truevar) - 1 - (len(y_true_stage2_truevar)-1)]))  # placeholder not used

            y_true_3c += y3b.tolist()
            y_pred_3c += final_pred3
            group_list += gb.tolist()
            vt_list += vtb.tolist()

            # Rebuild stage2 true-variant VT list cleanly
            for yy3, pred12, gg, vt in zip(y3b.tolist(), stage2_pred12.tolist(), gb.tolist(), vtb.tolist()):
                if int(yy3) in (1, 2):
                    # already appended y_true/pred/group above, append vt here in the same order
                    # easier and safer to keep this duplicate loop explicit
                    pass

        npz.close()

    # Rebuild stage2 true-variant arrays cleanly in one pass to avoid order bugs
    y_true_stage2_truevar = []
    y_pred_stage2_truevar = []
    group_stage2_truevar = []
    vt_stage2_truevar = []

    idx = 0
    for yt3, yp3, gg, vt, yp1 in zip(y_true_3c, y_pred_3c, group_list, vt_list, y_pred_stage1_bin):
        # stage2 prediction is only meaningful on true variants.
        # We recover it from final y_pred when stage1 passed, but that loses information on stage2 output when stage1 predicted 0.
        # So this block is intentionally left unused; stage2 true-variant metrics are computed below from a second pass-free trick.
        idx += 1

    # Compute stage2-on-true-variants from a second lightweight reconstruction
    # using final logic stored during main pass is awkward, so we derive it from final predictions and stage1 predictions only
    # when stage1 predicts 1. To preserve exact stage2 standalone diagnostics, we instead rely on a second pass-free stored list below.
    # Since that was intentionally omitted to keep memory lighter, we skip exact "pre-gating stage2" diagnostics.
    # We still provide a useful stage2-on-final-variant-predictions diagnostic separately.
    return {
        "y_true_3c": y_true_3c,
        "y_pred_3c": y_pred_3c,
        "group_list": group_list,
        "vt_list": vt_list,
        "y_true_stage1_bin": y_true_stage1_bin,
        "y_pred_stage1_bin": y_pred_stage1_bin,
        "stage1_prob_list": stage1_prob_list,
        "stage1_thr_list": stage1_thr_list,
    }


@torch.no_grad()
def eval_two_stage_full(
    items_by_shard: Dict[str, List[Item]],
    stage1_model: nn.Module,
    stage2_model: nn.Module,
    stage1_head_mode: str,
    stage2_head_mode: str,
    stage1_add_masks: bool,
    stage2_add_masks: bool,
    stage1_threshold_cfg: Dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> Dict[str, Any]:
    stage1_model.eval()
    stage2_model.eval()

    y_true_3c: List[int] = []
    y_pred_3c: List[int] = []
    group_list: List[int] = []
    vt_list: List[int] = []

    y_true_stage1_bin: List[int] = []
    y_pred_stage1_bin: List[int] = []
    stage1_prob_list: List[float] = []
    stage1_thr_list: List[float] = []

    y_true_stage2_truevar: List[int] = []
    y_pred_stage2_truevar: List[int] = []
    group_stage2_truevar: List[int] = []
    vt_stage2_truevar: List[int] = []

    for shard, items in items_by_shard.items():
        npz = np.load(shard, allow_pickle=True)
        ill_e = npz["ill_embeddings"].astype(np.float32, copy=False)
        ont_e = npz["ont_embeddings"].astype(np.float32, copy=False)
        mi = npz["mask_ill"].astype(np.float32, copy=False).reshape(-1, 1)
        mo = npz["mask_ont"].astype(np.float32, copy=False).reshape(-1, 1)

        rows = np.array([it.row for it in items], dtype=np.int64)
        ys3 = np.array([it.y3 for it in items], dtype=np.int64)
        gs = np.array([it.group for it in items], dtype=np.int64)
        vts = np.array([it.vt for it in items], dtype=np.int64)

        for i in range(0, len(rows), batch_size):
            rr = rows[i:i + batch_size]
            y3b = ys3[i:i + batch_size]
            gb = gs[i:i + batch_size]
            vtb = vts[i:i + batch_size]

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
            g = torch.from_numpy(gb).to(device)

            if stage1_head_mode == "groupwise":
                stage1_logits = stage1_model(x1, g)
            else:
                stage1_logits = stage1_model(x1)

            if stage2_head_mode == "groupwise":
                stage2_logits = stage2_model(x2, g)
            else:
                stage2_logits = stage2_model(x2)

            stage1_probs = torch.sigmoid(stage1_logits).cpu().numpy().astype(np.float64)
            stage2_pred12 = torch.argmax(stage2_logits, dim=1).cpu().numpy().astype(np.int64)

            for yy3, gg, vt, prob, pred12 in zip(
                y3b.tolist(),
                gb.tolist(),
                vtb.tolist(),
                stage1_probs.tolist(),
                stage2_pred12.tolist(),
            ):
                thr = threshold_for_group(stage1_threshold_cfg, int(gg))
                pred_bin = 1 if float(prob) >= float(thr) else 0
                pred3_variant = 1 if int(pred12) == 0 else 2
                pred3 = 0 if pred_bin == 0 else pred3_variant

                y_true_3c.append(int(yy3))
                y_pred_3c.append(int(pred3))
                group_list.append(int(gg))
                vt_list.append(int(vt))

                y_true_stage1_bin.append(0 if int(yy3) == 0 else 1)
                y_pred_stage1_bin.append(int(pred_bin))
                stage1_prob_list.append(float(prob))
                stage1_thr_list.append(float(thr))

                if int(yy3) in (1, 2):
                    y_true_stage2_truevar.append(0 if int(yy3) == 1 else 1)
                    y_pred_stage2_truevar.append(int(pred12))
                    group_stage2_truevar.append(int(gg))
                    vt_stage2_truevar.append(int(vt))

        npz.close()

    return {
        "y_true_3c": y_true_3c,
        "y_pred_3c": y_pred_3c,
        "group_list": group_list,
        "vt_list": vt_list,
        "y_true_stage1_bin": y_true_stage1_bin,
        "y_pred_stage1_bin": y_pred_stage1_bin,
        "stage1_prob_list": stage1_prob_list,
        "stage1_thr_list": stage1_thr_list,
        "y_true_stage2_truevar": y_true_stage2_truevar,
        "y_pred_stage2_truevar": y_pred_stage2_truevar,
        "group_stage2_truevar": group_stage2_truevar,
        "vt_stage2_truevar": vt_stage2_truevar,
    }


def summarize_3c_report(
    y_true: List[int],
    y_pred: List[int],
    groups: List[int],
    vt_list: List[int],
) -> Dict[str, Any]:
    cm3 = cm_k(y_true, y_pred, 3)
    overall_3c = {"n": len(y_true), **metrics_from_cm(cm3)}

    y_true_bin = [0 if y == 0 else 1 for y in y_true]
    y_pred_bin = [0 if p == 0 else 1 for p in y_pred]
    overall_bin = {"n": len(y_true), **binary_metrics(y_true_bin, y_pred_bin)}

    by_vt: Dict[str, Any] = {}
    for vt in (1, 2):
        idxs = [i for i, x in enumerate(vt_list) if int(x) == vt]
        if not idxs:
            continue
        yt = [y_true[i] for i in idxs]
        yp = [y_pred[i] for i in idxs]
        cm = cm_k(yt, yp, 3)
        by_vt[vt_name(vt)] = {"n": len(idxs), **metrics_from_cm(cm)}

        ytb = [0 if y == 0 else 1 for y in yt]
        ypb = [0 if p == 0 else 1 for p in yp]
        by_vt[vt_name(vt)]["binary_variant_vs_no"] = {"n": len(idxs), **binary_metrics(ytb, ypb)}

    by_group: Dict[str, Any] = {}
    for gg in sorted(set(groups)):
        idxs = [i for i, x in enumerate(groups) if int(x) == int(gg)]
        if not idxs:
            continue
        yt = [y_true[i] for i in idxs]
        yp = [y_pred[i] for i in idxs]
        cm = cm_k(yt, yp, 3)
        by_group[str(gg)] = {"n": len(idxs), **metrics_from_cm(cm)}

        ytb = [0 if y == 0 else 1 for y in yt]
        ypb = [0 if p == 0 else 1 for p in yp]
        by_group[str(gg)]["binary_variant_vs_no"] = {"n": len(idxs), **binary_metrics(ytb, ypb)}

    return {
        "overall_3c": overall_3c,
        "overall_binary_variant_vs_no": overall_bin,
        "by_variant_type": by_vt,
        "by_group": by_group,
    }


def summarize_stage1_binary(
    y_true_bin: List[int],
    y_pred_bin: List[int],
    groups: List[int],
    vt_list: List[int],
    thresholds: List[float],
) -> Dict[str, Any]:
    out = {
        "n": len(y_true_bin),
        "threshold_mode_applied": "mixed_per_example",
        "thresholds_used_summary": {
            "min": float(min(thresholds)) if thresholds else None,
            "max": float(max(thresholds)) if thresholds else None,
            "unique": sorted(list({round(float(t), 6) for t in thresholds})) if thresholds else [],
        },
        **binary_metrics(y_true_bin, y_pred_bin),
        "by_group": {},
        "by_variant_type": {},
    }

    for gg in sorted(set(groups)):
        idxs = [i for i, g in enumerate(groups) if int(g) == int(gg)]
        yt = [y_true_bin[i] for i in idxs]
        yp = [y_pred_bin[i] for i in idxs]
        out["by_group"][str(gg)] = {"n": len(idxs), **binary_metrics(yt, yp)}

    for vt in (1, 2):
        idxs = [i for i, x in enumerate(vt_list) if int(x) == vt]
        if not idxs:
            continue
        yt = [y_true_bin[i] for i in idxs]
        yp = [y_pred_bin[i] for i in idxs]
        out["by_variant_type"][vt_name(vt)] = {"n": len(idxs), **binary_metrics(yt, yp)}

    return out


def summarize_stage2_true_variants(
    y_true_2c: List[int],
    y_pred_2c: List[int],
    groups: List[int],
    vt_list: List[int],
) -> Dict[str, Any]:
    if len(y_true_2c) == 0:
        return {"n": 0}

    cm = cm_k(y_true_2c, y_pred_2c, 2)
    out = {"n": len(y_true_2c), **metrics_from_cm(cm), "by_group": {}, "by_variant_type": {}}

    for gg in sorted(set(groups)):
        idxs = [i for i, g in enumerate(groups) if int(g) == int(gg)]
        yt = [y_true_2c[i] for i in idxs]
        yp = [y_pred_2c[i] for i in idxs]
        cmg = cm_k(yt, yp, 2)
        out["by_group"][str(gg)] = {"n": len(idxs), **metrics_from_cm(cmg)}

    for vt in (1, 2):
        idxs = [i for i, x in enumerate(vt_list) if int(x) == vt]
        if not idxs:
            continue
        yt = [y_true_2c[i] for i in idxs]
        yp = [y_pred_2c[i] for i in idxs]
        cmv = cm_k(yt, yp, 2)
        out["by_variant_type"][vt_name(vt)] = {"n": len(idxs), **metrics_from_cm(cmv)}

    return out


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
    ap.add_argument("--stage1_calib_json", type=str, default=None)

    ap.add_argument("--outer_dir", type=str, default=None)
    ap.add_argument("--split_bins_json", type=str, default=None)
    ap.add_argument("--meta_with_vt_csv", type=str, default=None)
    ap.add_argument("--bin_size", type=int, default=None)

    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--groups", type=str, default="1,10,11")
    ap.add_argument("--exclude_vt_mismatch", action="store_true")

    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--out_json", type=str, required=True)

    args = ap.parse_args()

    stage1_ckpt = Path(args.stage1_ckpt)
    stage2_ckpt = Path(args.stage2_ckpt)
    stage1_calib_json = Path(args.stage1_calib_json) if args.stage1_calib_json else None

    if not stage1_ckpt.exists():
        raise SystemExit(f"Stage1 checkpoint not found: {stage1_ckpt}")
    if not stage2_ckpt.exists():
        raise SystemExit(f"Stage2 checkpoint not found: {stage2_ckpt}")
    if stage1_calib_json is not None and not stage1_calib_json.exists():
        raise SystemExit(f"Stage1 calibration JSON not found: {stage1_calib_json}")

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

    stage1_threshold_cfg = resolve_stage1_thresholds(stage1_payload, stage1_calib_json)

    raw = eval_two_stage_full(
        items_by_shard=items_by_shard,
        stage1_model=stage1_model,
        stage2_model=stage2_model,
        stage1_head_mode=stage1_head_mode,
        stage2_head_mode=stage2_head_mode,
        stage1_add_masks=stage1_add_masks,
        stage2_add_masks=stage2_add_masks,
        stage1_threshold_cfg=stage1_threshold_cfg,
        device=device,
        batch_size=batch_size,
    )

    report_main = summarize_3c_report(
        y_true=raw["y_true_3c"],
        y_pred=raw["y_pred_3c"],
        groups=raw["group_list"],
        vt_list=raw["vt_list"],
    )

    stage1_binary = summarize_stage1_binary(
        y_true_bin=raw["y_true_stage1_bin"],
        y_pred_bin=raw["y_pred_stage1_bin"],
        groups=raw["group_list"],
        vt_list=raw["vt_list"],
        thresholds=raw["stage1_thr_list"],
    )

    stage2_true_variants = summarize_stage2_true_variants(
        y_true_2c=raw["y_true_stage2_truevar"],
        y_pred_2c=raw["y_pred_stage2_truevar"],
        groups=raw["group_stage2_truevar"],
        vt_list=raw["vt_stage2_truevar"],
    )

    out = {
        "split": args.split,
        "groups_keep": sorted(list(groups_keep)),
        "exclude_vt_mismatch": bool(args.exclude_vt_mismatch),
        "benchmark_policy": {
            "group1_target": "label_ont",
            "group10_target": "label_ill",
            "group11_policy": "keep_only_if_label_ill_eq_label_ont",
            "keep_vt_mismatch": not bool(args.exclude_vt_mismatch),
            "meta_source_of_truth": str(meta_with_vt_csv),
        },
        "resolved_paths": {
            "outer_dir": str(outer_dir),
            "split_bins_json": str(split_bins_json),
            "meta_with_vt_csv": str(meta_with_vt_csv),
            "bin_size": int(bin_size),
        },
        "filter_stats": filter_stats,
        "stage1": {
            "checkpoint": str(stage1_ckpt),
            "calibration_json": str(stage1_calib_json) if stage1_calib_json is not None else None,
            "model": {
                "arch": stage1_args.get("arch", "mlp"),
                "head_mode": stage1_head_mode,
                "hidden": int(stage1_args.get("hidden", 512)),
                "dropout": float(stage1_args.get("dropout", 0.1)),
                "add_masks": bool(stage1_add_masks),
                "input_dim": int(stage1_input_dim),
            },
            "thresholds": stage1_threshold_cfg,
        },
        "stage2": {
            "checkpoint": str(stage2_ckpt),
            "model": {
                "arch": stage2_args.get("arch", "mlp"),
                "head_mode": stage2_head_mode,
                "hidden": int(stage2_args.get("hidden", 512)),
                "dropout": float(stage2_args.get("dropout", 0.1)),
                "add_masks": bool(stage2_add_masks),
                "input_dim": int(stage2_input_dim),
            },
            "label_mapping": {"0": "class1_from_y3eq1", "1": "class2_from_y3eq2"},
        },
        **report_main,
        "stage1_binary": stage1_binary,
        "stage2_variant_only_true_variants": stage2_true_variants,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(_jsonify(out), indent=2), encoding="utf-8")

    print(
        f"[Stage1] source={stage1_threshold_cfg['source']} mode={stage1_threshold_cfg['mode']} | "
        f"[Stage2] ckpt={stage2_ckpt.name}"
    )
    print(
        f"[3C ] acc={out['overall_3c']['acc']:.4f} "
        f"macroF1={out['overall_3c']['macro_f1']:.4f} "
        f"wF1={out['overall_3c']['weighted_f1']:.4f}"
    )
    print(
        f"[BIN] acc={out['overall_binary_variant_vs_no']['acc']:.4f} "
        f"prec={out['overall_binary_variant_vs_no']['precision']:.4f} "
        f"rec={out['overall_binary_variant_vs_no']['recall']:.4f} "
        f"f2={out['overall_binary_variant_vs_no']['f2']:.4f}"
    )
    print(f"[Saved] {out_json}")


if __name__ == "__main__":
    main()