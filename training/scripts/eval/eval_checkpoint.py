#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate trained checkpoints on OUTER using the consensus benchmark policy.

Benchmark policy
----------------
- Illumina mode:
    * group10 -> label_ill
    * group11 -> keep only if label_ill == label_ont
- ONT mode:
    * group1  -> label_ont
    * group11 -> keep only if label_ill == label_ont
- Hybrid mode:
    * group1  -> label_ont
    * group10 -> label_ill
    * group11 -> keep only if label_ill == label_ont

Notes
-----
- meta_with_vt_csv is the source of truth for labels / vt / vt_mismatch
- vt_mismatch is kept by default
- pass --exclude_vt_mismatch if you want an ambiguity-reduced evaluation
- model architecture is rebuilt from ckpt["args"]["arch"]

Examples
--------
Illumina-only:
python training/scripts/hybrid_dv/analysis/eval_outer_ckpt_with_vt.py \
  --outer_dir data/4_out/datasets/HG003/join_outer_singletons \
  --split_bins_json training/out/splits/outer_chr20_v1/split_bins.json \
  --meta_with_vt_csv training/out/meta/hg003_chr20_outer.meta_with_vt.csv \
  --ckpt training/out/models/baselines/illumina_only_linear/best.pt \
  --mode ill --groups 10,11 --split test \
  --out_json training/out/reports/outer_chr20_v1/illumina_only_linear_test.json

ONT-only:
python training/scripts/hybrid_dv/analysis/eval_outer_ckpt_with_vt.py \
  --outer_dir data/4_out/datasets/HG003/join_outer_singletons \
  --split_bins_json training/out/splits/outer_chr20_v1/split_bins.json \
  --meta_with_vt_csv training/out/meta/hg003_chr20_outer.meta_with_vt.csv \
  --ckpt training/out/models/baselines/ont_only_linear/best.pt \
  --mode ont --groups 1,11 --split test \
  --out_json training/out/reports/outer_chr20_v1/ont_only_linear_test.json

Hybrid:
python training/scripts/hybrid_dv/analysis/eval_outer_ckpt_with_vt.py \
  --outer_dir data/4_out/datasets/HG003/join_outer_singletons \
  --split_bins_json training/out/splits/outer_chr20_v1/split_bins.json \
  --meta_with_vt_csv training/out/meta/hg003_chr20_outer.meta_with_vt.csv \
  --ckpt training/out/models/baselines/hybrid_linear/best.pt \
  --mode hybrid --groups 1,10,11 --split test \
  --out_json training/out/reports/outer_chr20_v1/hybrid_linear_test.json
"""

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set

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


def vt_name(v: int) -> str:
    return "SNP" if v == 1 else ("INDEL" if v == 2 else "UNKNOWN")


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

    def forward(self, x):
        return self.net(x)


class LinearHead(nn.Module):
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.fc(x)

class GroupwiseLinear3C(nn.Module):
    def __init__(self, in_dim: int, n_classes: int = 3):
        super().__init__()
        self.trunk = nn.Identity()
        self.heads = nn.ModuleDict({
            "1": nn.Linear(in_dim, n_classes),
            "10": nn.Linear(in_dim, n_classes),
            "11": nn.Linear(in_dim, n_classes),
        })

    def forward(self, x, groups):
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

    def forward(self, x, groups):
        h = self.trunk(x)
        out = torch.zeros((h.shape[0], 3), dtype=h.dtype, device=h.device)
        for g in (1, 10, 11):
            mask = (groups == g)
            if mask.any():
                out[mask] = self.heads[str(g)](h[mask])
        return out


def build_model(
    in_dim: int,
    n_classes: int,
    arch: str,
    hidden: int,
    dropout: float,
    architecture: str = "shared_3class",
) -> nn.Module:
    if architecture == "groupwise_3class":
        if arch == "linear":
            return GroupwiseLinear3C(in_dim=in_dim, n_classes=n_classes)
        return GroupwiseMLP3C(in_dim=in_dim, n_classes=n_classes, hidden=hidden, dropout=dropout)

    if arch == "linear":
        return LinearHead(in_dim=in_dim, n_classes=n_classes)
    return MLP(in_dim=in_dim, n_classes=n_classes, hidden=hidden, dropout=dropout)


def resolve_add_masks(args_dict: Dict[str, Any]) -> bool:
    add_masks = True
    if args_dict.get("no_add_masks", False):
        add_masks = False
    if args_dict.get("add_masks", False):
        add_masks = True
    return add_masks


def load_split_bins(p: Path) -> Dict[str, Any]:
    payload = json.loads(p.read_text(encoding="utf-8"))
    return {
        "train": set(payload.get("train_bins", [])),
        "val": set(payload.get("val_bins", [])),
        "test": set(payload.get("test_bins", [])),
        "dropped": set(payload.get("dropped_bins", [])),
        "bin_size": int(payload.get("bin_size", 1_000_000)),
    }


def infer_split(bin_id: int, splits: Dict[str, Any]) -> str:
    for k in ("train", "val", "test", "dropped"):
        if bin_id in splits[k]:
            return k
    return "unknown"


def load_meta(meta_csv: Path) -> Dict[str, Dict[str, Any]]:
    out = {}
    with open(meta_csv, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            locus = row["locus"]
            label_ill = as_int(row.get("label_ill"), -1)
            label_ont = as_int(row.get("label_ont"), -1)
            vt = as_int(row.get("variant_type"), 0)
            vt_mismatch = as_int(row.get("vt_mismatch"), 0)
            label_mismatch = int(label_ill >= 0 and label_ont >= 0 and label_ill != label_ont)

            out[locus] = {
                "label_ill": label_ill,
                "label_ont": label_ont,
                "variant_type": vt,
                "vt_mismatch": vt_mismatch,
                "label_mismatch": label_mismatch,
            }
    return out


def choose_target(mode: str, group: int, label_ill: int, label_ont: int) -> Tuple[Optional[int], str]:
    """
    Returns:
      target_label or None,
      reason: ok | missing_label | label_mismatch | group_not_supported
    """
    if mode == "ill":
        if group == 10:
            return (label_ill if label_ill >= 0 else None, "ok" if label_ill >= 0 else "missing_label")
        if group == 11:
            if label_ill < 0 or label_ont < 0:
                return None, "missing_label"
            if label_ill != label_ont:
                return None, "label_mismatch"
            return label_ill, "ok"
        return None, "group_not_supported"

    if mode == "ont":
        if group == 1:
            return (label_ont if label_ont >= 0 else None, "ok" if label_ont >= 0 else "missing_label")
        if group == 11:
            if label_ill < 0 or label_ont < 0:
                return None, "missing_label"
            if label_ill != label_ont:
                return None, "label_mismatch"
            return label_ont, "ok"
        return None, "group_not_supported"

    # hybrid
    if group == 1:
        return (label_ont if label_ont >= 0 else None, "ok" if label_ont >= 0 else "missing_label")
    if group == 10:
        return (label_ill if label_ill >= 0 else None, "ok" if label_ill >= 0 else "missing_label")
    if group == 11:
        if label_ill < 0 or label_ont < 0:
            return None, "missing_label"
        if label_ill != label_ont:
            return None, "label_mismatch"
        return label_ill, "ok"
    return None, "group_not_supported"


def list_shards(outer_dir: Path) -> List[Path]:
    shards = sorted(outer_dir.glob("*.npz"))
    if not shards:
        raise SystemExit(f"No shards found in {outer_dir}")
    return shards


def build_items(
    outer_dir: Path,
    splits: Dict[str, Any],
    meta: Dict[str, Dict[str, Any]],
    which_split: str,
    groups_keep: Set[int],
    exclude_vt_mismatch: bool,
    mode: str,
    bin_size: int,
) -> Tuple[List[Item], Dict[str, int]]:
    items: List[Item] = []
    stats = {
        "seen_after_split_group": 0,
        "kept": 0,
        "drop_missing_meta": 0,
        "drop_missing_label": 0,
        "drop_label_mismatch": 0,
        "drop_vt_mismatch": 0,
        "drop_group_not_supported": 0,
    }

    for sp in list_shards(outer_dir):
        npz = np.load(sp, allow_pickle=True)

        loci = npz["locus"]
        if loci.dtype.kind in ("S", "O"):
            loci_list = [x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x) for x in loci.tolist()]
        else:
            loci_list = [str(x) for x in loci.tolist()]

        grp = npz["group"].astype(np.int64).reshape(-1)

        for i, locus in enumerate(loci_list):
            pos = parse_locus_pos(locus)
            if pos is None:
                continue

            b = pos // bin_size
            if infer_split(b, splits) != which_split:
                continue

            g = int(grp[i])
            if groups_keep and g not in groups_keep:
                continue

            stats["seen_after_split_group"] += 1

            meta_row = meta.get(locus)
            if meta_row is None:
                stats["drop_missing_meta"] += 1
                continue

            vt = int(meta_row["variant_type"])
            vt_mismatch = int(meta_row["vt_mismatch"])
            if exclude_vt_mismatch and vt_mismatch == 1:
                stats["drop_vt_mismatch"] += 1
                continue

            y3, reason = choose_target(
                mode=mode,
                group=g,
                label_ill=int(meta_row["label_ill"]),
                label_ont=int(meta_row["label_ont"]),
            )

            if y3 is None:
                if reason == "missing_label":
                    stats["drop_missing_label"] += 1
                elif reason == "label_mismatch":
                    stats["drop_label_mismatch"] += 1
                else:
                    stats["drop_group_not_supported"] += 1
                continue

            items.append(
                Item(
                    shard=str(sp),
                    row=i,
                    locus=locus,
                    y3=int(y3),
                    group=g,
                    bin_id=int(b),
                    vt=vt,
                    vt_mismatch=vt_mismatch,
                )
            )
            stats["kept"] += 1

        npz.close()

    return items, stats


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
    f1s, recs, precs = [], [], []

    for c in range(k):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = float(tp / (tp + fp + eps))
        rec = float(tp / (tp + fn + eps))
        f1 = float(2 * prec * rec / (prec + rec + eps))
        per_class[str(c)] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": int(support[c]),
        }
        f1s.append(f1)
        recs.append(rec)
        precs.append(prec)

    acc = float(np.trace(cm) / max(cm.sum(), 1))
    macro_f1 = float(np.mean(f1s))
    macro_recall = float(np.mean(recs))
    macro_precision = float(np.mean(precs))
    weights = support / max(support.sum(), 1)
    weighted_f1 = float(np.sum(weights * np.array(f1s)))

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
    tn, fp = cm[0, 0], cm[0, 1]
    fn, tp = cm[1, 0], cm[1, 1]
    eps = 1e-12
    prec = float(tp / (tp + fp + eps))
    rec = float(tp / (tp + fn + eps))
    f1 = float(2 * prec * rec / (prec + rec + eps))
    spec = float(tn / (tn + fp + eps))
    acc = float((tp + tn) / max(cm.sum(), 1))
    return {"cm": cm.tolist(), "acc": acc, "precision": prec, "recall": rec, "f1": f1, "specificity": spec}


@torch.no_grad()
def predict_all(
    items: List[Item],
    mode: str,
    model: nn.Module,
    batch_size: int,
    add_masks_hybrid: bool,
    architecture: str = "shared_3class",
) -> List[int]:
    model.eval()
    preds = [0] * len(items)

    by_shard: Dict[str, List[Tuple[int, Item]]] = {}
    for idx, it in enumerate(items):
        by_shard.setdefault(it.shard, []).append((idx, it))

    device = next(model.parameters()).device

    for shard, pairs in by_shard.items():
        npz = np.load(shard, allow_pickle=True)
        ill_e = npz["ill_embeddings"].astype(np.float32, copy=False)
        ont_e = npz["ont_embeddings"].astype(np.float32, copy=False)
        mi = npz["mask_ill"].astype(np.float32, copy=False).reshape(-1)
        mo = npz["mask_ont"].astype(np.float32, copy=False).reshape(-1)

        pairs.sort(key=lambda x: x[1].row)
        rows = np.array([it.row for _, it in pairs], dtype=np.int64)
        groups = np.array([it.group for _, it in pairs], dtype=np.int64)

        if mode == "ill":
            feat = ill_e[rows]
        elif mode == "ont":
            feat = ont_e[rows]
        else:
            feat = np.concatenate([ill_e[rows], ont_e[rows]], axis=1)
            if add_masks_hybrid:
                masks = np.stack([mi[rows], mo[rows]], axis=1).astype(np.float32)
                feat = np.concatenate([feat, masks], axis=1)

        for i in range(0, len(rows), batch_size):
            sl = slice(i, i + batch_size)
            x = torch.from_numpy(feat[sl]).to(device)

            if architecture == "groupwise_3class":
                g = torch.from_numpy(groups[sl].astype(np.int64)).to(device)
                logits = model(x, g)
            else:
                logits = model(x)

            p = torch.argmax(logits, dim=1).cpu().numpy().astype(int).tolist()
            for (global_idx, _it), pp in zip(pairs[i:i + batch_size], p):
                preds[global_idx] = int(pp)

        npz.close()

    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outer_dir", type=str, required=True)
    ap.add_argument("--split_bins_json", type=str, required=True)
    ap.add_argument("--meta_with_vt_csv", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--mode", type=str, choices=["ill", "ont", "hybrid"], required=True)
    ap.add_argument("--split", type=str, choices=["val", "test"], default="test")
    ap.add_argument("--groups", type=str, default="", help="Comma-separated groups, e.g. 10,11 or 1,11. Empty = keep all.")
    ap.add_argument("--exclude_vt_mismatch", action="store_true")
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--out_json", type=str, default=None)
    args = ap.parse_args()

    outer_dir = Path(args.outer_dir)
    splits = load_split_bins(Path(args.split_bins_json))
    meta = load_meta(Path(args.meta_with_vt_csv))
    bin_size = int(splits["bin_size"])

    groups_keep = set()
    if args.groups.strip():
        groups_keep = set(int(x) for x in args.groups.split(",") if x.strip())

    items, filter_stats = build_items(
        outer_dir=outer_dir,
        splits=splits,
        meta=meta,
        which_split=args.split,
        groups_keep=groups_keep,
        exclude_vt_mismatch=bool(args.exclude_vt_mismatch),
        mode=args.mode,
        bin_size=bin_size,
    )
    if not items:
        raise SystemExit("No items found after filtering. Check split/groups/meta/policy.")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    in_dim = int(ckpt.get("input_dim", 0))
    n_classes = int(ckpt.get("n_classes", 3))
    hidden = int(ckpt_args.get("hidden", 512))
    dropout = float(ckpt_args.get("dropout", 0.1))
    arch = str(ckpt_args.get("arch", "mlp"))
    architecture = str(ckpt.get("architecture", "shared_3class"))
    add_masks_hybrid = resolve_add_masks(ckpt_args)

    if in_dim <= 0:
        one = list_shards(outer_dir)[0]
        npz0 = np.load(one, allow_pickle=True)
        if args.mode == "ill":
            in_dim = int(npz0["ill_embeddings"].shape[1])
        elif args.mode == "ont":
            in_dim = int(npz0["ont_embeddings"].shape[1])
        else:
            in_dim = int(npz0["ill_embeddings"].shape[1] + npz0["ont_embeddings"].shape[1])
            if add_masks_hybrid:
                in_dim += 2
        npz0.close()

    model = build_model(
        in_dim=in_dim,
        n_classes=n_classes,
        arch=arch,
        hidden=hidden,
        dropout=dropout,
        architecture=architecture,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    if arch == "linear":
        print(f"[Model] loaded arch=linear in_dim={in_dim} n_classes={n_classes}")
    else:
        print(f"[Model] loaded arch=mlp in_dim={in_dim} n_classes={n_classes} hidden={hidden} dropout={dropout}")
    if args.mode == "hybrid":
        print(f"[Hybrid] add_masks={add_masks_hybrid}")

    preds = predict_all(items,
        args.mode,
        model,
        args.batch_size,
        add_masks_hybrid,
        architecture=architecture,
    )

    y_true = [it.y3 for it in items]
    y_pred = preds

    cm3 = cm_k(y_true, y_pred, 3)
    overall_3c = {"n": len(items), **metrics_from_cm(cm3)}

    y_true_bin = [0 if y == 0 else 1 for y in y_true]
    y_pred_bin = [0 if p == 0 else 1 for p in y_pred]
    overall_bin = {"n": len(items), **binary_metrics(y_true_bin, y_pred_bin)}

    tv_idx = [i for i, y in enumerate(y_true) if y in (1, 2)]
    tv_pred_variant = [i for i in tv_idx if y_pred[i] in (1, 2)]
    coverage = len(tv_pred_variant) / max(len(tv_idx), 1)

    if tv_pred_variant:
        gt_true = [0 if y_true[i] == 1 else 1 for i in tv_pred_variant]
        gt_pred = [0 if y_pred[i] == 1 else 1 for i in tv_pred_variant]
        gt_bin = binary_metrics(gt_true, gt_pred)
    else:
        gt_bin = {"cm": [[0, 0], [0, 0]], "acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "specificity": 0.0}

    genotype = {
        "true_variants_n": len(tv_idx),
        "predicted_variant_n": len(tv_pred_variant),
        "coverage_true_variants_pred_as_variant": coverage,
        "genotype_metrics_on_predicted_variants": gt_bin,
    }

    by_vt = {}
    for vt in (1, 2):
        idxs = [i for i, it in enumerate(items) if it.vt == vt]
        if not idxs:
            continue
        yt = [y_true[i] for i in idxs]
        yp = [y_pred[i] for i in idxs]
        cm = cm_k(yt, yp, 3)
        by_vt[vt_name(vt)] = {"n": len(idxs), **metrics_from_cm(cm)}

        ytb = [0 if y == 0 else 1 for y in yt]
        ypb = [0 if p == 0 else 1 for p in yp]
        by_vt[vt_name(vt)]["binary_variant_vs_no"] = {"n": len(idxs), **binary_metrics(ytb, ypb)}

    by_group = {}
    for gg in sorted({it.group for it in items}):
        idxs = [i for i, it in enumerate(items) if it.group == gg]
        yt = [y_true[i] for i in idxs]
        yp = [y_pred[i] for i in idxs]
        cm = cm_k(yt, yp, 3)
        by_group[str(gg)] = {"n": len(idxs), **metrics_from_cm(cm)}

        ytb = [0 if y == 0 else 1 for y in yt]
        ypb = [0 if p == 0 else 1 for p in yp]
        by_group[str(gg)]["binary_variant_vs_no"] = {"n": len(idxs), **binary_metrics(ytb, ypb)}

    out = {
        "split": args.split,
        "mode": args.mode,
        "groups_keep": sorted(list(groups_keep)) if groups_keep else "ALL",
        "exclude_vt_mismatch": bool(args.exclude_vt_mismatch),
        "benchmark_policy": {
            "group1_target": "label_ont",
            "group10_target": "label_ill",
            "group11_policy": "keep_only_if_label_ill_eq_label_ont",
            "keep_vt_mismatch": not bool(args.exclude_vt_mismatch),
            "meta_source_of_truth": str(args.meta_with_vt_csv),
        },
        "filter_stats": filter_stats,
        "overall_3c": overall_3c,
        "overall_binary_variant_vs_no": overall_bin,
        "variant_only_genotype": genotype,
        "by_variant_type": by_vt,
        "by_group": by_group,
    }

    if args.out_json:
        p = Path(args.out_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"[Saved] {p}")
    else:
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()