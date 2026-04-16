#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hybrid stage-1 baseline on OUTER (binary variant detection: 0 vs {1,2})

Policy
------
- Visible groups: group01 + group10 + group11
- Target source:
    * group01 -> label_ont
    * group10 -> label_ill
    * group11 -> keep only consensus loci where label_ill == label_ont
- vt_mismatch is NOT excluded
- label_mismatch in group11 is excluded from the main benchmark
- Binary target:
    * y_bin = 0 if genotype == 0
    * y_bin = 1 if genotype in {1,2}

Inputs
------
- OUTER shards
- fixed split_bins.json
- meta CSV with label_ill / label_ont
- optional variant_type column in meta CSV

Features
--------
- concat([ill_embeddings, ont_embeddings, mask_ill, mask_ont]) if add_masks=True
- concat([ill_embeddings, ont_embeddings]) if add_masks=False

Heads
-----
- shared:
    one shared binary head
- groupwise:
    shared trunk + one binary head per group {1,10,11}

Training / model selection
--------------------------
- BCEWithLogitsLoss
- optional pos_weight
- threshold selected on validation by:
    1) maximize recall subject to precision >= min_val_precision
    2) fallback: maximize F2

Outputs in save_dir
-------------------
- config.json
- split_bins_used.json
- best.pt / final.pt
- best_metrics.json / last_metrics.json
- optional out_csv

Example
-------
python training/scripts/hybrid_dv/hybrid/two_stage/train_outer_hybrid_stage1.py \
  --outer_dir data/4_out/datasets/HG003/join_outer_singletons \
  --split_json training/out/splits/outer_chr20_v1/split_bins.json \
  --meta_csv training/out/meta/hg003_chr20_outer.meta_with_vt.csv \
  --save_dir training/out/models/stage1/hybrid_stage1_shared_mlp \
  --arch mlp \
  --head_mode shared \
  --use_pos_weight \
  --epochs 10
"""

import argparse
import csv
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

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

            # Hybrid consensus policy
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


def summarize_split(name: str, refs: List[Ref]) -> Dict[str, Any]:
    n = len(refs)
    by_group = {1: 0, 10: 0, 11: 0}
    by_y3 = {0: 0, 1: 0, 2: 0}
    by_y_bin = {0: 0, 1: 0}
    by_mask = {"01": 0, "10": 0, "11": 0}
    by_vt_mismatch = {0: 0, 1: 0}
    by_variant_type: Dict[str, int] = {}

    for r in refs:
        if r.group in by_group:
            by_group[r.group] += 1
        by_y3[r.y3] += 1
        by_y_bin[r.y_bin] += 1
        mk = f"{r.mask_ill}{r.mask_ont}"
        by_mask[mk] = by_mask.get(mk, 0) + 1
        by_vt_mismatch[r.vt_mismatch] += 1
        by_variant_type[r.variant_type] = by_variant_type.get(r.variant_type, 0) + 1

    return {
        "name": name,
        "n": n,
        "by_group": by_group,
        "by_y3": by_y3,
        "by_y_bin": by_y_bin,
        "by_mask": by_mask,
        "by_vt_mismatch": by_vt_mismatch,
        "by_variant_type": by_variant_type,
    }


def print_split_summary(s: Dict[str, Any]) -> None:
    n = max(int(s["n"]), 1)

    def pct(x):
        return 100.0 * float(x) / float(n)

    print(f"\n[SplitSummary] {s['name']} n={int(s['n']):,}")
    print("  groups:       ", {k: f"{v:,} ({pct(v):.1f}%)" for k, v in sorted(s["by_group"].items())})
    print("  masks:        ", {k: f"{v:,} ({pct(v):.1f}%)" for k, v in sorted(s["by_mask"].items())})
    print("  y_3:          ", {k: f"{v:,} ({pct(v):.1f}%)" for k, v in sorted(s["by_y3"].items())})
    print("  y_bin:        ", {k: f"{v:,} ({pct(v):.1f}%)" for k, v in sorted(s["by_y_bin"].items())})
    print("  vt_mismatch:  ", {k: f"{v:,} ({pct(v):.3f}%)" for k, v in sorted(s["by_vt_mismatch"].items())})
    if s["by_variant_type"]:
        print("  variant_type: ", {k: f"{v:,} ({pct(v):.1f}%)" for k, v in sorted(s["by_variant_type"].items())})


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


def binary_metrics_from_preds(y_true: List[int], probs: List[float], thr: float) -> Dict[str, Any]:
    eps = 1e-12
    y_pred = [1 if p >= thr else 0 for p in probs]
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
        "threshold": float(thr),
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


def summarize_slices(
    y_true: List[int],
    probs: List[float],
    thr: float,
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
            pg = [probs[i] for i in idx]
            by_group[str(gg)] = {
                "n": len(idx),
                **binary_metrics_from_preds(yg, pg, thr),
                "pr_auc": pr_auc_from_probs(yg, pg),
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
            pv = [probs[i] for i in idx]
            by_variant_type[str(vt)] = {
                "n": len(idx),
                **binary_metrics_from_preds(yv, pv, thr),
                "pr_auc": pr_auc_from_probs(yv, pv),
            }
        out["by_variant_type"] = by_variant_type

        if groups is not None:
            by_group_variant_type: Dict[str, Any] = {}
            uniq_vt = sorted(set(str(v) for v in variant_types))
            for gg in (1, 10, 11):
                row = {}
                for vt in uniq_vt:
                    idx = [i for i, (g, v) in enumerate(zip(groups, variant_types)) if int(g) == gg and str(v) == vt]
                    if not idx:
                        continue
                    ygv = [y_true[i] for i in idx]
                    pgv = [probs[i] for i in idx]
                    row[str(vt)] = {
                        "n": len(idx),
                        **binary_metrics_from_preds(ygv, pgv, thr),
                        "pr_auc": pr_auc_from_probs(ygv, pgv),
                    }
                if row:
                    by_group_variant_type[str(gg)] = row
            out["by_group_variant_type"] = by_group_variant_type

    return out


def select_threshold(
    y_true: List[int],
    probs: List[float],
    min_precision: float = 0.60,
    grid_min: float = 0.05,
    grid_max: float = 0.95,
    grid_step: float = 0.01,
) -> Dict[str, Any]:
    candidates = []
    thr = grid_min
    while thr <= grid_max + 1e-12:
        m = binary_metrics_from_preds(y_true, probs, float(thr))
        candidates.append(m)
        thr += grid_step

    valid = [m for m in candidates if m["precision"] >= float(min_precision)]
    if valid:
        valid.sort(
            key=lambda m: (
                m["recall"],
                m["f2"],
                m["balanced_acc"],
                -m["fpr"],
            ),
            reverse=True,
        )
        best = valid[0]
        best["selection_mode"] = "recall_subject_to_min_precision"
        best["min_precision"] = float(min_precision)
        return best

    candidates.sort(
        key=lambda m: (
            m["f2"],
            m["recall"],
            m["balanced_acc"],
            -m["fpr"],
        ),
        reverse=True,
    )
    best = candidates[0]
    best["selection_mode"] = "fallback_best_f2"
    best["min_precision"] = float(min_precision)
    return best


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


def finalize_eval(raw_eval: Dict[str, Any], threshold: float) -> Dict[str, Any]:
    out = {
        "n": int(raw_eval["n"]),
        "threshold": float(threshold),
        **binary_metrics_from_preds(raw_eval["y_true"], raw_eval["probs"], threshold),
        "pr_auc": float(raw_eval["pr_auc"]),
    }
    out.update(
        summarize_slices(
            y_true=raw_eval["y_true"],
            probs=raw_eval["probs"],
            thr=threshold,
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


def build_pos_weight(train_refs: List[Ref], max_w: float = 20.0) -> Optional[torch.Tensor]:
    n_pos = sum(int(r.y_bin == 1) for r in train_refs)
    n_neg = sum(int(r.y_bin == 0) for r in train_refs)
    total = n_pos + n_neg
    if total <= 0:
        return None

    if n_pos <= 0:
        return torch.tensor(1.0, dtype=torch.float32)

    w = float(n_neg) / float(n_pos)
    w = min(w, float(max_w))
    return torch.tensor(w, dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--outer_dir", type=str, default="data/4_out/datasets/HG003/join_outer_singletons")
    ap.add_argument("--split_json", type=str, required=True)
    ap.add_argument("--meta_csv", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bin_size", type=int, default=1_000_000)

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=5e-4)

    ap.add_argument("--arch", type=str, default="mlp", choices=["mlp", "linear"])
    ap.add_argument("--head_mode", type=str, default="shared", choices=["shared", "groupwise"])
    ap.add_argument("--hidden", type=int, default=512, help="Hidden size for arch=mlp. Ignored for shared linear.")
    ap.add_argument("--dropout", type=float, default=0.1, help="Dropout for MLP trunk.")

    ap.add_argument("--add_masks", action="store_true")
    ap.add_argument("--no_add_masks", action="store_true")

    ap.add_argument("--use_pos_weight", action="store_true")
    ap.add_argument("--max_pos_weight", type=float, default=20.0)

    ap.add_argument("--min_val_precision", type=float, default=0.60)
    ap.add_argument("--threshold_grid_min", type=float, default=0.05)
    ap.add_argument("--threshold_grid_max", type=float, default=0.95)
    ap.add_argument("--threshold_grid_step", type=float, default=0.01)

    ap.add_argument("--save_dir", type=str, default="training/out/models/stage1/hybrid_stage1_mlp")
    ap.add_argument("--out_csv", type=str, default=None)

    ap.add_argument(
        "--save_best_on",
        type=str,
        default="val_recall",
        choices=["val_recall", "val_f2", "val_balanced_acc", "val_pr_auc", "val_precision"],
    )

    args = ap.parse_args()

    add_masks = True
    if args.no_add_masks:
        add_masks = False
    elif args.add_masks:
        add_masks = True

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    outer_dir = Path(args.outer_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    split = json.loads(Path(args.split_json).read_text(encoding="utf-8"))
    split_bin_size = int(split.get("bin_size", -1))
    if split_bin_size != int(args.bin_size):
        raise SystemExit(f"bin_size mismatch: split_json has {split_bin_size}, but args has {args.bin_size}")

    meta_lookup = load_meta(Path(args.meta_csv))

    train_bins = set(split["train_bins"])
    val_bins = set(split["val_bins"])
    test_bins = set(split["test_bins"])

    print(f"[Index] Scanning shards in {outer_dir} ...")
    refs_all = build_refs_all(outer_dir, meta_lookup=meta_lookup, bin_size=args.bin_size)
    if not refs_all:
        raise SystemExit("No refs indexed after applying hybrid consensus policy.")

    train_refs = [r for r in refs_all if r.bin_id in train_bins]
    val_refs = [r for r in refs_all if r.bin_id in val_bins]
    test_refs = [r for r in refs_all if r.bin_id in test_bins]

    print("[Split/FILTER] examples (Hybrid consensus: "
          "group01->label_ont, group10->label_ill, group11 keep only label consensus): "
          f"train={len(train_refs)} val={len(val_refs)} test={len(test_refs)}")
    print_split_summary(summarize_split("train", train_refs))
    print_split_summary(summarize_split("val", val_refs))
    print_split_summary(summarize_split("test", test_refs))
    print()

    if args.epochs <= 0:
        print("[DryRun] epochs<=0, stopping after split summary.")
        return

    train_by_shard = group_by_shard(train_refs)
    val_by_shard = group_by_shard(val_refs)
    test_by_shard = group_by_shard(test_refs)

    one_shard = next(iter(train_by_shard.keys()))
    npz0 = np.load(one_shard, allow_pickle=True)
    in_dim = int(npz0["ill_embeddings"].shape[1] + npz0["ont_embeddings"].shape[1])
    npz0.close()
    if add_masks:
        in_dim += 2

    device = torch.device("cpu")
    model = build_model(
        in_dim=in_dim,
        arch=args.arch,
        hidden=args.hidden,
        dropout=args.dropout,
        head_mode=args.head_mode,
    ).to(device)

    print(
        f"[Model] arch={args.arch} head_mode={args.head_mode} "
        f"in_dim={in_dim} hidden={args.hidden} dropout={args.dropout} add_masks={add_masks}"
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    pos_weight_value = None
    if args.use_pos_weight:
        pw = build_pos_weight(train_refs, max_w=args.max_pos_weight)
        pos_weight_value = float(pw.item()) if pw is not None else None
        print(f"[Loss] BCEWithLogitsLoss pos_weight={pos_weight_value}")
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw.to(device) if pw is not None else None)
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    config_payload = {
        "args": vars(args),
        "resolved": {
            "input_dim": in_dim,
            "device": str(device),
            "loss": "BCEWithLogitsLoss",
            "pos_weight": pos_weight_value,
            "add_masks": add_masks,
        },
        "dataset_policy": {
            "mode": "hybrid_stage1_binary",
            "groups_keep": [1, 10, 11],
            "group1_target": "label_ont",
            "group10_target": "label_ill",
            "group11_policy": "keep_only_if_label_ill_eq_label_ont",
            "drop_label_mismatch_in_group11": True,
            "keep_vt_mismatch": True,
            "target_mapping": {"0": "no_variant", "1_or_2": "variant"},
            "join_definition": "locus-level late merge",
        },
        "model_policy": {
            "arch": args.arch,
            "head_mode": args.head_mode,
            "threshold_selection": {
                "rule": "maximize_recall_subject_to_min_precision_else_best_f2",
                "min_val_precision": args.min_val_precision,
                "grid_min": args.threshold_grid_min,
                "grid_max": args.threshold_grid_max,
                "grid_step": args.threshold_grid_step,
            },
        },
        "data_summary": {
            "train": summarize_split("train", train_refs),
            "val": summarize_split("val", val_refs),
            "test": summarize_split("test", test_refs),
        },
        "split_source": {
            "split_json": str(Path(args.split_json)),
            "meta_csv": str(Path(args.meta_csv)),
            "bin_size": args.bin_size,
        },
    }

    (save_dir / "config.json").write_text(json.dumps(_jsonify(config_payload), indent=2), encoding="utf-8")
    (save_dir / "split_bins_used.json").write_text(json.dumps(_jsonify(split), indent=2), encoding="utf-8")

    csv_writer = None
    csv_f = None
    if args.out_csv:
        outp = Path(args.out_csv)
        outp.parent.mkdir(parents=True, exist_ok=True)
        csv_f = open(outp, "w", newline="")
        csv_writer = csv.writer(csv_f)
        csv_writer.writerow([
            "epoch", "train_loss", "selected_thr",
            "val_n", "val_pr_auc", "val_acc", "val_balanced_acc", "val_precision", "val_recall", "val_f1", "val_f2",
            "test_n", "test_pr_auc", "test_acc", "test_balanced_acc", "test_precision", "test_recall", "test_f1", "test_f2",
        ])

    def score_from_val(val_m: Dict[str, Any]) -> float:
        if args.save_best_on == "val_f2":
            return float(val_m["f2"])
        if args.save_best_on == "val_balanced_acc":
            return float(val_m["balanced_acc"])
        if args.save_best_on == "val_pr_auc":
            return float(val_m["pr_auc"])
        if args.save_best_on == "val_precision":
            return float(val_m["precision"])
        return float(val_m["recall"])

    best_score = -1e18
    best_epoch = -1
    shard_keys_master = sorted(list(train_by_shard.keys()))

    for epoch in range(1, args.epochs + 1):
        model.train()
        rng = random.Random(args.seed + epoch)

        shard_keys = list(shard_keys_master)
        rng.shuffle(shard_keys)

        total_loss = 0.0
        n_seen = 0

        for shard in shard_keys:
            items = train_by_shard[shard]
            rows = np.array([r.row for r in items], dtype=np.int64)
            ys = np.array([r.y_bin for r in items], dtype=np.float32)
            gs = np.array([r.group for r in items], dtype=np.int64)
            mi = np.array([r.mask_ill for r in items], dtype=np.float32)
            mo = np.array([r.mask_ont for r in items], dtype=np.float32)

            perm = np.arange(len(rows))
            perm_list = perm.tolist()
            rng.shuffle(perm_list)
            perm = np.array(perm_list, dtype=np.int64)
            rows = rows[perm]
            ys = ys[perm]
            gs = gs[perm]
            mi = mi[perm]
            mo = mo[perm]

            npz = np.load(shard, allow_pickle=True)
            ill_e = npz["ill_embeddings"].astype(np.float32, copy=False)
            ont_e = npz["ont_embeddings"].astype(np.float32, copy=False)

            for i in range(0, len(rows), args.batch_size):
                rr = rows[i:i + args.batch_size]
                yb = ys[i:i + args.batch_size]
                gb = gs[i:i + args.batch_size]
                mib = mi[i:i + args.batch_size]
                mob = mo[i:i + args.batch_size]

                x = np.concatenate([ill_e[rr], ont_e[rr]], axis=1)
                if add_masks:
                    masks = np.stack([mib, mob], axis=1).astype(np.float32)
                    x = np.concatenate([x, masks], axis=1)

                x = torch.from_numpy(x).to(device)
                y = torch.from_numpy(yb.astype(np.float32)).to(device)

                opt.zero_grad(set_to_none=True)

                if args.head_mode == "groupwise":
                    g = torch.from_numpy(gb.astype(np.int64)).to(device)
                    logits = model(x, g)
                else:
                    logits = model(x)

                loss = loss_fn(logits, y)
                loss.backward()
                opt.step()

                bs = len(rr)
                total_loss += float(loss.detach().cpu()) * bs
                n_seen += bs

            npz.close()

        train_loss = total_loss / max(n_seen, 1)

        val_raw = eval_sharded_binary_raw(
            model=model,
            by_shard=val_by_shard,
            device=device,
            batch_size=args.batch_size,
            add_masks=add_masks,
            head_mode=args.head_mode,
        )
        thr_info = select_threshold(
            y_true=val_raw["y_true"],
            probs=val_raw["probs"],
            min_precision=args.min_val_precision,
            grid_min=args.threshold_grid_min,
            grid_max=args.threshold_grid_max,
            grid_step=args.threshold_grid_step,
        )
        selected_thr = float(thr_info["threshold"])

        val_m = finalize_eval(val_raw, threshold=selected_thr)

        test_raw = eval_sharded_binary_raw(
            model=model,
            by_shard=test_by_shard,
            device=device,
            batch_size=args.batch_size,
            add_masks=add_masks,
            head_mode=args.head_mode,
        )
        test_m = finalize_eval(test_raw, threshold=selected_thr)

        score = score_from_val(val_m)

        print(
            f"[Epoch {epoch:02d}] loss={train_loss:.5f} | "
            f"VAL thr={selected_thr:.2f} prAUC={val_m['pr_auc']:.4f} "
            f"acc={val_m['acc']:.4f} balAcc={val_m['balanced_acc']:.4f} "
            f"prec={val_m['precision']:.4f} rec={val_m['recall']:.4f} f2={val_m['f2']:.4f} | "
            f"TEST prAUC={test_m['pr_auc']:.4f} acc={test_m['acc']:.4f} "
            f"balAcc={test_m['balanced_acc']:.4f} prec={test_m['precision']:.4f} "
            f"rec={test_m['recall']:.4f} f2={test_m['f2']:.4f} | "
            f"best_on={args.save_best_on} score={score:.6f}"
        )

        if csv_writer:
            csv_writer.writerow([
                epoch, train_loss, selected_thr,
                val_m["n"], val_m["pr_auc"], val_m["acc"], val_m["balanced_acc"], val_m["precision"], val_m["recall"], val_m["f1"], val_m["f2"],
                test_m["n"], test_m["pr_auc"], test_m["acc"], test_m["balanced_acc"], test_m["precision"], test_m["recall"], test_m["f1"], test_m["f2"],
            ])
            csv_f.flush()

        last_payload = {
            "epoch": epoch,
            "train_loss": train_loss,
            "threshold_selection_info": thr_info,
            "val": val_m,
            "test": test_m,
            "score": score,
            "best_on": args.save_best_on,
        }
        (save_dir / "last_metrics.json").write_text(
            json.dumps(_jsonify(last_payload), indent=2),
            encoding="utf-8"
        )

        if score > best_score:
            best_score = float(score)
            best_epoch = epoch

            best_payload = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "args": vars(args),
                "input_dim": in_dim,
                "n_classes": 1,
                "task": "binary_variant_detection",
                "best_score": best_score,
                "best_epoch": best_epoch,
                "best_on": args.save_best_on,
                "threshold": selected_thr,
                "threshold_selection_info": thr_info,
                "val": val_m,
                "test": test_m,
            }
            torch.save(best_payload, save_dir / "best.pt")

            best_metrics_only = {
                "epoch": epoch,
                "best_score": best_score,
                "best_on": args.save_best_on,
                "threshold": selected_thr,
                "threshold_selection_info": thr_info,
                "val": val_m,
                "test": test_m,
            }
            (save_dir / "best_metrics.json").write_text(
                json.dumps(_jsonify(best_metrics_only), indent=2),
                encoding="utf-8"
            )

    final_payload = {
        "epoch": args.epochs,
        "model_state": model.state_dict(),
        "args": vars(args),
        "input_dim": in_dim,
        "n_classes": 1,
        "task": "binary_variant_detection",
        "best_score": best_score,
        "best_epoch": best_epoch,
        "best_on": args.save_best_on,
    }
    torch.save(final_payload, save_dir / "final.pt")

    if csv_f:
        csv_f.close()

    print(f"[Saved] best.pt / final.pt in {save_dir}")
    print(f"[Saved] config.json / split_bins_used.json / best_metrics.json / last_metrics.json in {save_dir}")


if __name__ == "__main__":
    main()