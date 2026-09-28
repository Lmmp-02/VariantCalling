#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate DeepVariant teacher predictions stored inside OUTER shards,
using the same consensus benchmark policy as the trained baselines.

Teacher policy
--------------
- teacher=ill:
    * group10 -> label_ill
    * group11 -> keep only if label_ill == label_ont
- teacher=ont:
    * group1  -> label_ont
    * group11 -> keep only if label_ill == label_ont

Notes
-----
- meta_csv is the source of truth for labels / vt / vt_mismatch
- vt_mismatch is kept by default
- pass --exclude_vt_mismatch to remove those loci

Examples
--------
DeepVariant Illumina:
python training/scripts/hybrid_dv/analysis/eval_outer_teacher.py \
  --outer_dir data/4_out/datasets/HG003/join_outer_singletons \
  --split_bins_json training/out/splits/outer_chr20_v1/split_bins.json \
  --meta_csv training/out/meta/hg003_chr20_outer.meta_with_vt.csv \
  --teacher ill --groups 10,11 --split test \
  --out_json training/out/reports/outer_chr20_v1/dv_illumina_test.json

DeepVariant ONT:
python training/scripts/hybrid_dv/analysis/eval_outer_teacher.py \
  --outer_dir data/4_out/datasets/HG003/join_outer_singletons \
  --split_bins_json training/out/splits/outer_chr20_v1/split_bins.json \
  --meta_csv training/out/meta/hg003_chr20_outer.meta_with_vt.csv \
  --teacher ont --groups 1,11 --split test \
  --out_json training/out/reports/outer_chr20_v1/dv_ont_test.json
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple

import numpy as np


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


def softmax_np(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    logits = logits.astype(np.float64)
    m = np.max(logits, axis=axis, keepdims=True)
    e = np.exp(logits - m)
    return (e / np.sum(e, axis=axis, keepdims=True)).astype(np.float32)


def _as_str_list(arr: np.ndarray) -> List[str]:
    if arr.dtype.kind in ("S", "O"):
        out = []
        for v in arr.tolist():
            if isinstance(v, (bytes, np.bytes_)):
                out.append(v.decode("utf-8"))
            else:
                out.append(str(v))
        return out
    return [str(v) for v in arr.tolist()]


def list_shards(outer_dir: Path) -> List[Path]:
    shards = sorted(outer_dir.glob("*.npz"))
    if not shards:
        raise SystemExit(f"No shards found in {outer_dir}")
    return shards


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


def choose_target(teacher: str, group: int, label_ill: int, label_ont: int) -> Tuple[Optional[int], str]:
    if teacher == "ill":
        if group == 10:
            return (label_ill if label_ill >= 0 else None, "ok" if label_ill >= 0 else "missing_label")
        if group == 11:
            if label_ill < 0 or label_ont < 0:
                return None, "missing_label"
            if label_ill != label_ont:
                return None, "label_mismatch"
            return label_ill, "ok"
        return None, "group_not_supported"

    if teacher == "ont":
        if group == 1:
            return (label_ont if label_ont >= 0 else None, "ok" if label_ont >= 0 else "missing_label")
        if group == 11:
            if label_ill < 0 or label_ont < 0:
                return None, "missing_label"
            if label_ill != label_ont:
                return None, "label_mismatch"
            return label_ont, "ok"
        return None, "group_not_supported"

    return None, "group_not_supported"


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outer_dir", type=str, required=True)
    ap.add_argument("--split_bins_json", type=str, required=True)
    ap.add_argument("--meta_csv", type=str, required=True)
    ap.add_argument("--teacher", type=str, choices=["ill", "ont"], required=True)
    ap.add_argument("--groups", type=str, required=True, help="Comma-separated groups, e.g. 10,11 or 1,11")
    ap.add_argument("--split", type=str, choices=["val", "test"], default="test")
    ap.add_argument("--exclude_vt_mismatch", action="store_true")
    ap.add_argument("--out_json", type=str, required=True)
    args = ap.parse_args()

    outer_dir = Path(args.outer_dir)
    splits = load_split_bins(Path(args.split_bins_json))
    meta = load_meta(Path(args.meta_csv))
    bin_size = int(splits["bin_size"])
    groups_keep = set(int(x) for x in args.groups.split(",") if x.strip())

    y_true: List[int] = []
    y_pred: List[int] = []
    vt_list: List[int] = []
    group_list: List[int] = []

    filter_stats = {
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
        files = set(npz.files)

        loci = _as_str_list(npz["locus"])
        grp = npz["group"].astype(np.int64).reshape(-1)

        if args.teacher == "ill":
            if "ill_probs" in files:
                probs = npz["ill_probs"].astype(np.float32)
            elif "ill_logits" in files:
                probs = softmax_np(npz["ill_logits"], axis=-1)
            else:
                raise SystemExit(f"{sp}: missing ill_probs/ill_logits")
        else:
            if "ont_probs" in files:
                probs = npz["ont_probs"].astype(np.float32)
            elif "ont_logits" in files:
                probs = softmax_np(npz["ont_logits"], axis=-1)
            else:
                raise SystemExit(f"{sp}: missing ont_probs/ont_logits")

        preds = np.argmax(probs, axis=1).astype(np.int64)

        for i, locus in enumerate(loci):
            pos = parse_locus_pos(locus)
            if pos is None:
                continue

            b = pos // bin_size
            if infer_split(b, splits) != args.split:
                continue

            g = int(grp[i])
            if g not in groups_keep:
                continue

            filter_stats["seen_after_split_group"] += 1

            meta_row = meta.get(locus)
            if meta_row is None:
                filter_stats["drop_missing_meta"] += 1
                continue

            vt = int(meta_row["variant_type"])
            vt_mismatch = int(meta_row["vt_mismatch"])
            if args.exclude_vt_mismatch and vt_mismatch == 1:
                filter_stats["drop_vt_mismatch"] += 1
                continue

            true_lab, reason = choose_target(
                teacher=args.teacher,
                group=g,
                label_ill=int(meta_row["label_ill"]),
                label_ont=int(meta_row["label_ont"]),
            )
            if true_lab is None:
                if reason == "missing_label":
                    filter_stats["drop_missing_label"] += 1
                elif reason == "label_mismatch":
                    filter_stats["drop_label_mismatch"] += 1
                else:
                    filter_stats["drop_group_not_supported"] += 1
                continue

            y_true.append(int(true_lab))
            y_pred.append(int(preds[i]))
            vt_list.append(vt)
            group_list.append(g)
            filter_stats["kept"] += 1

        npz.close()

    if not y_true:
        raise SystemExit("No examples collected. Check split/groups/meta/policy.")

    cm3 = cm_k(y_true, y_pred, 3)
    overall_3c = {"n": len(y_true), **metrics_from_cm(cm3)}

    y_true_bin = [0 if y == 0 else 1 for y in y_true]
    y_pred_bin = [0 if p == 0 else 1 for p in y_pred]
    overall_bin = {"n": len(y_true), **binary_metrics(y_true_bin, y_pred_bin)}

    by_vt = {}
    for vt in (1, 2):
        idxs = [i for i, x in enumerate(vt_list) if x == vt]
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
    for gg in sorted(set(group_list)):
        idxs = [i for i, x in enumerate(group_list) if x == gg]
        yt = [y_true[i] for i in idxs]
        yp = [y_pred[i] for i in idxs]
        cm = cm_k(yt, yp, 3)
        by_group[str(gg)] = {"n": len(idxs), **metrics_from_cm(cm)}

        ytb = [0 if y == 0 else 1 for y in yt]
        ypb = [0 if p == 0 else 1 for p in yp]
        by_group[str(gg)]["binary_variant_vs_no"] = {"n": len(idxs), **binary_metrics(ytb, ypb)}

    out = {
        "split": args.split,
        "teacher": args.teacher,
        "groups_keep": sorted(list(groups_keep)),
        "exclude_vt_mismatch": bool(args.exclude_vt_mismatch),
        "benchmark_policy": {
            "group1_target": "label_ont",
            "group10_target": "label_ill",
            "group11_policy": "keep_only_if_label_ill_eq_label_ont",
            "keep_vt_mismatch": not bool(args.exclude_vt_mismatch),
            "meta_source_of_truth": str(args.meta_csv),
        },
        "filter_stats": filter_stats,
        "overall_3c": overall_3c,
        "overall_binary_variant_vs_no": overall_bin,
        "by_variant_type": by_vt,
        "by_group": by_group,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[Saved] {out_json}")


if __name__ == "__main__":
    main()