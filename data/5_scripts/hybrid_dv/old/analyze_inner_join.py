#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze INNER join dataset shards (.npz) for:
A) exploration + sanity checks
B) teacher metrics (Illumina / ONT) + agreement + stratified by variant_type
C) imbalance diagnosis + concrete training strategies recommendations

Expected INNER NPZ keys (robust: will introspect what exists):
- locus (string or bytes array)
- label (int 0/1/2)
- ill_embeddings, ont_embeddings (float arrays)
- (optional) ill_logits/ill_probs, ont_logits/ont_probs
- (optional) mask_ill, mask_ont
- (optional) group (e.g. 11)
- (optional) variant_type ("SNP"/"INDEL" or 0/1)

If variant_type missing in INNER:
- uses inner_meta.csv (expects at least: ill_npz, ill_idx, ont_npz, ont_idx)
- loads variant_type from raw NPZ referenced by ill_npz/ill_idx (fallback to ont if needed)

Outputs:
- stdout report
- optional: --out_json, --out_csv (per-slice summaries + confusion matrices flattened)
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Iterable, List, Optional, Tuple

import numpy as np

try:
    import pandas as pd
except Exception:
    pd = None


# ----------------------------
# Utilities
# ----------------------------

LABELS = [0, 1, 2]
LABEL_NAMES = {0: "hom-ref", 1: "het", 2: "hom-alt"}

VT_CANON = {"SNP": "SNP", "SNV": "SNP", "INDEL": "INDEL", "INS": "INDEL", "DEL": "INDEL"}


def _as_str_array(x: np.ndarray) -> List[str]:
    # NPZ may store bytes
    if x.dtype.kind in ("S", "O"):
        out = []
        for v in x.tolist():
            if isinstance(v, bytes):
                out.append(v.decode("utf-8"))
            else:
                out.append(str(v))
        return out
    return [str(v) for v in x.tolist()]


def softmax_np(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    logits = logits.astype(np.float64)
    m = np.max(logits, axis=axis, keepdims=True)
    e = np.exp(logits - m)
    return (e / np.sum(e, axis=axis, keepdims=True)).astype(np.float32)


def confusion_matrix_3(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = np.zeros((3, 3), dtype=np.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        if t in (0, 1, 2) and p in (0, 1, 2):
            cm[t, p] += 1
    return cm


def precision_recall_f1_from_cm(cm: np.ndarray) -> Dict[str, Any]:
    # per-class
    eps = 1e-12
    per_class = {}
    precisions, recalls, f1s = [], [], []
    supports = []
    for c in range(3):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        f1 = 2 * prec * rec / (prec + rec + eps)
        sup = cm[c, :].sum()
        per_class[c] = {
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "support": int(sup),
            "name": LABEL_NAMES[c],
        }
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        supports.append(sup)

    # macro
    macro = {
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "f1": float(np.mean(f1s)),
    }

    # weighted (by support)
    supports_arr = np.array(supports, dtype=np.float64)
    w = supports_arr / (supports_arr.sum() + eps)
    weighted = {
        "precision": float(np.sum(w * np.array(precisions))),
        "recall": float(np.sum(w * np.array(recalls))),
        "f1": float(np.sum(w * np.array(f1s))),
    }

    acc = float(np.trace(cm) / (cm.sum() + eps))

    return {
        "accuracy": acc,
        "macro": macro,
        "weighted": weighted,
        "per_class": per_class,
        "cm": cm.tolist(),
    }


def normalize_variant_type(v) -> Optional[str]:
    """
    Normalize DeepVariant variant_type to 'SNP'/'INDEL' when possible.

    In our DeepVariant 1.9.0 exports we observe:
      - variant_type=1 ~ SNP
      - variant_type=2 ~ INDEL

    We also keep support for textual encodings.
    """
    if v is None:
        return None

    # unwrap numpy scalar
    if isinstance(v, np.generic):
        v = v.item()

    # bytes -> str
    if isinstance(v, (bytes, np.bytes_)):
        v = v.decode("utf-8", errors="ignore")

    # numeric encodings (most common in your NPZ)
    if isinstance(v, (int, np.integer)):
        if int(v) == 1:
            return "SNP"
        if int(v) == 2:
            return "INDEL"
        return None

    s = str(v).strip().upper()

    # sometimes numeric strings
    if s.isdigit():
        iv = int(s)
        if iv == 1:
            return "SNP"
        if iv == 2:
            return "INDEL"
        # legacy guess (if ever appears)
        if iv == 0:
            return "SNP"
        return None

    # textual encodings
    if s in VT_CANON:
        return VT_CANON[s]
    if "SNP" in s or "SNV" in s:
        return "SNP"
    if "INDEL" in s or "INS" in s or "DEL" in s:
        return "INDEL"

    return None


def list_npz_shards(folder: Path) -> List[Path]:
    return sorted(folder.glob("*.npz"))


@dataclass
class FeatureDims:
    ill_embeddings: Optional[Tuple[int, ...]] = None
    ont_embeddings: Optional[Tuple[int, ...]] = None
    ill_logits: Optional[Tuple[int, ...]] = None
    ont_logits: Optional[Tuple[int, ...]] = None
    ill_probs: Optional[Tuple[int, ...]] = None
    ont_probs: Optional[Tuple[int, ...]] = None


# ----------------------------
# Variant type recovery (meta -> raw)
# ----------------------------

def load_inner_meta(meta_csv: Path):
    if pd is None:
        raise RuntimeError("pandas is required to use --meta_csv fallback for variant_type.")
    df = pd.read_csv(meta_csv)
    # Heuristic column names
    # expected: ill_npz, ill_idx, ont_npz, ont_idx (or similar)
    cols = {c.lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    c_ill_npz = pick("ill_npz", "illumina_npz", "ill_shard", "illumina_shard")
    c_ill_idx = pick("ill_idx", "illumina_idx", "ill_index", "illumina_index")
    c_ont_npz = pick("ont_npz", "ont_shard")
    c_ont_idx = pick("ont_idx", "ont_index")

    if c_ill_npz is None or c_ill_idx is None:
        raise RuntimeError(f"meta_csv missing illumina shard/index columns. Columns={list(df.columns)}")

    # ont columns optional for fallback
    return df, c_ill_npz, c_ill_idx, c_ont_npz, c_ont_idx


def recover_variant_types_from_raw(
    df_meta,
    raw_root: Path,
    c_npz: str,
    c_idx: str,
    prefer: str = "ill",
) -> List[Optional[str]]:
    """
    Recover variant_type per meta row by loading raw NPZ and indexing.
    We group by raw shard path to avoid re-loading repeatedly.
    """
    # Build absolute paths
    shard_series = df_meta[c_npz].astype(str)
    idx_series = df_meta[c_idx].astype(int)

    # Some metas store relative paths; resolve against raw_root
    def resolve_path(p: str) -> Path:
        """
        Resolve shard path stored in meta CSV.
        - If absolute: use it
        - If relative and exists as-is (relative to CWD/repo root): use it
        - Else fallback to raw_root/p
        """
        pth = Path(p)
        if pth.is_absolute():
            return pth
        # meta often stores repo-relative paths like data/4_out/...
        if pth.exists():
            return pth.resolve()
        # fallback: meta may store basenames; then raw_root helps
        return (raw_root / pth).resolve()

    shard_paths = shard_series.map(resolve_path)

    vt_out: List[Optional[str]] = [None] * len(df_meta)

    # group by shard
    groups = defaultdict(list)
    for i, (sp, ix) in enumerate(zip(shard_paths.tolist(), idx_series.tolist())):
        groups[str(sp)].append((i, ix))

    for sp_str, items in groups.items():
        sp = Path(sp_str)
        if not sp.exists():
            # leave as None; caller may fallback to other modality
            continue
        data = np.load(sp, allow_pickle=True)
        # heuristic keys in raw
        vt_key = None
        for k in data.files:
            if k.lower() in ("variant_type", "vartype", "varianttype", "type"):
                vt_key = k
                break
        if vt_key is None:
            # cannot recover from this shard
            continue
        vt_arr = data[vt_key]
        # supports scalar per-example or per-example strings
        for (row_i, ix) in items:
            if ix < 0 or ix >= len(vt_arr):
                vt_out[row_i] = None
                continue
            vt_out[row_i] = normalize_variant_type(vt_arr[ix])
    return vt_out


# ----------------------------
# Main analysis
# ----------------------------

def analyze_inner(
    inner_dir: Path,
    meta_csv: Optional[Path],
    raw_ill_dir: Optional[Path],
    raw_ont_dir: Optional[Path],
    max_shards: Optional[int],
    strict_masks: bool,
) -> Dict[str, Any]:
    shards = list_npz_shards(inner_dir)
    if max_shards is not None:
        shards = shards[:max_shards]

    if len(shards) == 0:
        raise RuntimeError(f"No .npz shards found in {inner_dir}")

    # Accumulators
    n_total = 0
    label_counts = Counter()
    vt_counts = Counter()
    vt_label_counts = defaultdict(Counter)

    # masks/group checks
    mask_ill_ok = True
    mask_ont_ok = True
    group_ok = True
    mask_ill_vals = Counter()
    mask_ont_vals = Counter()
    group_vals = Counter()

    # Feature dims (first non-empty shard)
    dims = FeatureDims()

    # Teacher metrics accumulators
    # global + per-variant_type
    def make_metric_bucket():
        return {
            "y_true": [],
            "ill_pred": [],
            "ont_pred": [],
        }

    buckets = {"ALL": make_metric_bucket(), "SNP": make_metric_bucket(), "INDEL": make_metric_bucket()}

    # Variant type source:
    # 1) from INNER shard if present
    # 2) from meta+raw NPZ
    use_meta_fallback = False
    meta_vt: Optional[List[Optional[str]]] = None

    # If we need meta fallback, build it once
    if meta_csv is not None:
        use_meta_fallback = True
        if pd is None:
            raise RuntimeError("pandas required to use meta_csv fallback.")
        df_meta, c_ill_npz, c_ill_idx, c_ont_npz, c_ont_idx = load_inner_meta(meta_csv)

        # Recover vt from Ill raw first (preferred)
        if raw_ill_dir is None:
            raise RuntimeError("--raw_ill_dir is required when using --meta_csv fallback.")
        meta_vt = recover_variant_types_from_raw(df_meta, raw_ill_dir, c_ill_npz, c_ill_idx, prefer="ill")

        # Fallback to ONT raw if still None and available
        if any(v is None for v in meta_vt) and (c_ont_npz is not None and c_ont_idx is not None and raw_ont_dir is not None):
            meta_vt_ont = recover_variant_types_from_raw(df_meta, raw_ont_dir, c_ont_npz, c_ont_idx, prefer="ont")
            meta_vt = [v if v is not None else meta_vt_ont[i] for i, v in enumerate(meta_vt)]

        # Still can have None; keep as UNKNOWN
        # We assume INNER shards preserve ordering with meta rows.
        # If that assumption fails, user will notice via sanity mismatches (counts, etc.)

    # iterate shards
    offset = 0
    missing_teacher = {"ill": 0, "ont": 0}

    for sp in shards:
        data = np.load(sp, allow_pickle=True)
        files = set(data.files)

        # required basics
        if "label" not in files:
            raise RuntimeError(f"Shard {sp} missing key 'label'. Keys={data.files}")
        y = data["label"].astype(np.int64)
        n = len(y)

        # locus for optional checks
        if "locus" in files:
            locus = data["locus"]
            _ = locus  # unused, but proves existence
        n_total += n
        n_total_prev = n_total - n

        # label dist
        for c in LABELS:
            label_counts[c] += int(np.sum(y == c))

        # feature dims capture (first shard that has each)
        def set_dim_if_none(attr: str, key: str):
            nonlocal dims
            if getattr(dims, attr) is None and key in files:
                setattr(dims, attr, tuple(data[key].shape[1:]))

        set_dim_if_none("ill_embeddings", "ill_embeddings")
        set_dim_if_none("ont_embeddings", "ont_embeddings")
        set_dim_if_none("ill_logits", "ill_logits")
        set_dim_if_none("ont_logits", "ont_logits")
        set_dim_if_none("ill_probs", "ill_probs")
        set_dim_if_none("ont_probs", "ont_probs")

        # masks / group checks
        if "mask_ill" in files:
            mi = data["mask_ill"].astype(np.int64).reshape(-1)
            mask_ill_vals.update(mi.tolist())
            if strict_masks and not np.all(mi == 1):
                mask_ill_ok = False
        if "mask_ont" in files:
            mo = data["mask_ont"].astype(np.int64).reshape(-1)
            mask_ont_vals.update(mo.tolist())
            if strict_masks and not np.all(mo == 1):
                mask_ont_ok = False
        if "group" in files:
            g = data["group"].astype(np.int64).reshape(-1)
            group_vals.update(g.tolist())
            if strict_masks and not np.all(g == 11):
                group_ok = False

        # variant_type
        vt_list: List[Optional[str]] = [None] * n
        inner_has_vt = None
        for k in data.files:
            if k.lower() in ("variant_type", "vartype", "varianttype", "type"):
                inner_has_vt = k
                break

        if inner_has_vt is not None:
            vt_raw = data[inner_has_vt]
            # could be per-example strings or numeric
            if np.ndim(vt_raw) == 0:
                vt_list = [normalize_variant_type(vt_raw.item())] * n
            else:
                vt_list = [normalize_variant_type(v) for v in vt_raw.tolist()]
        elif use_meta_fallback and meta_vt is not None:
            # map via meta ordering
            slice_vt = meta_vt[offset: offset + n]
            if len(slice_vt) != n:
                raise RuntimeError(
                    f"Meta rows ({len(meta_vt)}) do not match INNER examples slice [{offset}:{offset+n}]"
                )
            vt_list = [v if v is not None else "UNKNOWN" for v in slice_vt]
        else:
            vt_list = ["UNKNOWN"] * n

        # update vt counts
        for vt in vt_list:
            vt_counts[vt] += 1
        for vt, yy in zip(vt_list, y.tolist()):
            vt_label_counts[vt][yy] += 1

        # teacher predictions (argmax probs)
        def get_pred(prefix: str) -> Optional[np.ndarray]:
            # prefix in {"ill", "ont"}
            probs_key = f"{prefix}_probs"
            logits_key = f"{prefix}_logits"
            if probs_key in files:
                probs = data[probs_key].astype(np.float32)
            elif logits_key in files:
                probs = softmax_np(data[logits_key], axis=-1)
            else:
                return None
            if probs.ndim != 2 or probs.shape[1] != 3:
                raise RuntimeError(f"{sp}: {prefix} probs/logits expected shape (N,3), got {probs.shape}")
            return np.argmax(probs, axis=1).astype(np.int64)

        ill_pred = get_pred("ill")
        ont_pred = get_pred("ont")
        if ill_pred is None:
            missing_teacher["ill"] += n
        if ont_pred is None:
            missing_teacher["ont"] += n

        # Fill buckets
        # Only if we have at least one teacher (agreement needs both)
        for vt_name in ("ALL", "SNP", "INDEL"):
            pass

        # add per example
        for i in range(n):
            vt = vt_list[i]
            vt_bucket = vt if vt in ("SNP", "INDEL") else "ALL"
            y_i = int(y[i])
            buckets["ALL"]["y_true"].append(y_i)
            if ill_pred is not None:
                buckets["ALL"]["ill_pred"].append(int(ill_pred[i]))
            if ont_pred is not None:
                buckets["ALL"]["ont_pred"].append(int(ont_pred[i]))

            if vt in ("SNP", "INDEL"):
                buckets[vt]["y_true"].append(y_i)
                if ill_pred is not None:
                    buckets[vt]["ill_pred"].append(int(ill_pred[i]))
                if ont_pred is not None:
                    buckets[vt]["ont_pred"].append(int(ont_pred[i]))

        offset += n

    # finalize metrics
    def compute_teacher_metrics(y_true_list, pred_list) -> Optional[Dict[str, Any]]:
        if len(pred_list) == 0:
            return None
        y_true = np.array(y_true_list, dtype=np.int64)
        y_pred = np.array(pred_list, dtype=np.int64)
        cm = confusion_matrix_3(y_true, y_pred)
        return precision_recall_f1_from_cm(cm)

    def compute_agreement(p1_list, p2_list) -> Optional[Dict[str, Any]]:
        if len(p1_list) == 0 or len(p2_list) == 0:
            return None
        if len(p1_list) != len(p2_list):
            # if one teacher missing, agreement undefined in that slice
            return None
        p1 = np.array(p1_list, dtype=np.int64)
        p2 = np.array(p2_list, dtype=np.int64)
        acc = float(np.mean(p1 == p2))
        cm = confusion_matrix_3(p1, p2)  # treat Ill as "true" and ONT as "pred" just for structure
        return {"agreement": acc, "cm_ill_as_true_ont_as_pred": cm.tolist()}

    metrics = {}
    for slice_name, b in buckets.items():
        m = {
            "teacher_ill": compute_teacher_metrics(b["y_true"], b["ill_pred"]),
            "teacher_ont": compute_teacher_metrics(b["y_true"], b["ont_pred"]),
            "agreement_ill_ont": compute_agreement(b["ill_pred"], b["ont_pred"]),
            "n": len(b["y_true"]),
            "n_with_ill": len(b["ill_pred"]),
            "n_with_ont": len(b["ont_pred"]),
        }
        metrics[slice_name] = m

    # imbalance diagnosis
    total = sum(label_counts.values())
    label_freq = {k: (label_counts[k] / max(total, 1)) for k in LABELS}

    # simple suggested class weights: inverse frequency (normalized)
    inv = {k: (1.0 / max(label_freq[k], 1e-12)) for k in LABELS}
    inv_mean = sum(inv.values()) / 3.0
    class_weights = {k: float(inv[k] / inv_mean) for k in LABELS}

    out = {
        "paths": {
            "inner_dir": str(inner_dir),
            "meta_csv": str(meta_csv) if meta_csv else None,
            "raw_ill_dir": str(raw_ill_dir) if raw_ill_dir else None,
            "raw_ont_dir": str(raw_ont_dir) if raw_ont_dir else None,
        },
        "n_total": int(n_total),
        "label_counts": {str(k): int(label_counts[k]) for k in LABELS},
        "label_freq": {str(k): float(label_freq[k]) for k in LABELS},
        "variant_type_counts": {str(k): int(v) for k, v in vt_counts.items()},
        "variant_type_label_counts": {
            str(vt): {str(k): int(c[k]) for k in LABELS} for vt, c in vt_label_counts.items()
        },
        "feature_dims": asdict(dims),
        "mask_checks": {
            "mask_ill_values": dict(mask_ill_vals),
            "mask_ont_values": dict(mask_ont_vals),
            "group_values": dict(group_vals),
            "mask_ill_all_ones": bool(mask_ill_ok),
            "mask_ont_all_ones": bool(mask_ont_ok),
            "group_all_11": bool(group_ok),
            "strict_masks": bool(strict_masks),
        },
        "teacher_missing_examples": missing_teacher,
        "metrics": metrics,
        "imbalance_recommendations": {
            "class_weights_inverse_freq_normalized": class_weights,
            "notes": [
                "If label 0 dominates heavily, optimize macro-F1 and/or per-class recall (het/hom-alt).",
                "Consider WeightedRandomSampler or stratified batch sampling to avoid class-0 collapse.",
                "Also consider balancing by variant_type (SNP vs INDEL) if SNP >> INDEL.",
                "For INNER 'easy' set, keep an eye on recall for class 2 (hom-alt): it tends to be the rarest.",
            ],
        },
    }
    return out

def pretty_cm(cm: List[List[int]], title: str) -> str:
    # rows=true, cols=pred
    rows = ["0(href)", "1(het )", "2(halt)"]
    cols = ["0", "1", "2"]
    s = []
    s.append(f"{title}")
    s.append("     pred ->            0        1        2")
    for rname, row in zip(rows, cm):
        s.append(f"true {rname}     {row[0]:8d} {row[1]:8d} {row[2]:8d}")
    return "\n".join(s) + "\n"

def print_report(r: Dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print("INNER JOIN DATASET REPORT")
    print("=" * 80)
    print(f"INNER dir: {r['paths']['inner_dir']}")
    if r["paths"]["meta_csv"]:
        print(f"meta_csv:  {r['paths']['meta_csv']}")
    print(f"Total examples: {r['n_total']:,}")

    print("\nLabel distribution:")
    for k in LABELS:
        c = int(r["label_counts"][str(k)])
        f = float(r["label_freq"][str(k)])
        print(f"  {k} ({LABEL_NAMES[k]:7s}): {c:10d}  ({f*100:6.2f}%)")

    print("\nVariant type distribution:")
    for vt, c in sorted(r["variant_type_counts"].items(), key=lambda x: (-x[1], x[0])):
        print(f"  {vt:8s}: {c:10d}")
    print("\nVariant type × label:")
    for vt, d in sorted(r["variant_type_label_counts"].items(), key=lambda x: x[0]):
        parts = []
        for k in LABELS:
            parts.append(f"{k}:{d.get(str(k), 0)}")
        print(f"  {vt:8s} -> " + "  ".join(parts))

    print("\nFeature dimensions (per-example):")
    fd = r["feature_dims"]
    for k, v in fd.items():
        if v is not None:
            print(f"  {k:14s}: {tuple(v)}")
        else:
            print(f"  {k:14s}: (missing)")

    print("\nMask / group sanity:")
    mc = r["mask_checks"]
    print(f"  mask_ill values: {mc['mask_ill_values']}")
    print(f"  mask_ont values: {mc['mask_ont_values']}")
    print(f"  group values:    {mc['group_values']}")
    print(f"  strict_masks={mc['strict_masks']} -> mask_ill_all_ones={mc['mask_ill_all_ones']}, mask_ont_all_ones={mc['mask_ont_all_ones']}, group_all_11={mc['group_all_11']}")

    print("\nTeacher availability:")
    tm = r["teacher_missing_examples"]
    print(f"  missing ill teacher for: {tm['ill']} examples")
    print(f"  missing ont teacher for: {tm['ont']} examples")

    print("\nTeacher metrics + agreement:")
    for slice_name in ("ALL", "SNP", "INDEL"):
        m = r["metrics"].get(slice_name, {})
        if not m:
            continue
        print(f"\n  --- Slice: {slice_name} (n={m.get('n', 0)}) ---")
        ill = m.get("teacher_ill")
        ont = m.get("teacher_ont")
        agr = m.get("agreement_ill_ont")
        if ill:
            print(f"  Illumina teacher: acc={ill['accuracy']:.4f}  macroF1={ill['macro']['f1']:.4f}  macroR={ill['macro']['recall']:.4f}")
            if "cm" in ill:
                print(pretty_cm(ill["cm"], "  Illumina confusion matrix (true=label, pred=argmax ill_probs):"))
        else:
            print("  Illumina teacher: (missing)")
        if ont:
            print(f"  ONT teacher:      acc={ont['accuracy']:.4f}  macroF1={ont['macro']['f1']:.4f}  macroR={ont['macro']['recall']:.4f}")
            if "cm" in ont:
                print(pretty_cm(ont["cm"], "  ONT confusion matrix (true=label, pred=argmax ont_probs):"))
        else:
            print("  ONT teacher:      (missing)")
        if agr:
            print(f"  Ill vs ONT agreement (argmax): {agr['agreement']:.4f}")
        else:
            print("  Ill vs ONT agreement: (not available / missing one teacher)")

    print("\nImbalance: suggested starting class weights (inverse freq, normalized):")
    cw = r["imbalance_recommendations"]["class_weights_inverse_freq_normalized"]
    for k in LABELS:
        print(f"  class {k} ({LABEL_NAMES[k]}): {cw[str(k)] if isinstance(cw, dict) and str(k) in cw else cw[k]:.3f}")

    print("\nConcrete training strategy suggestions (for INNER PoC):")
    for line in r["imbalance_recommendations"]["notes"]:
        print(f"  - {line}")

    print("\n" + "=" * 80 + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inner_dir", type=str, default="data/4_out/datasets/HG003/join_inner_1to1",
                    help="Path to INNER join shards folder.")
    ap.add_argument("--meta_csv", type=str, default="data/4_out/datasets/HG003/join_inner_1to1/hg003_chr20_inner_meta.csv",
                    help="Meta CSV for INNER join (used only if variant_type missing in INNER shards).")
    ap.add_argument("--raw_ill_dir", type=str, default="data/4_out/datasets/HG003/illumina_chr20_train",
                    help="Root folder for Illumina raw shards (used with meta_csv fallback).")
    ap.add_argument("--raw_ont_dir", type=str, default="data/4_out/datasets/HG003/ont_chr20_train",
                    help="Root folder for ONT raw shards (optional fallback if Ill missing).")
    ap.add_argument("--no_meta_fallback", action="store_true",
                    help="Disable meta_csv fallback; variant_type will be UNKNOWN if not present in INNER shards.")
    ap.add_argument("--max_shards", type=int, default=None,
                    help="Analyze only first N shards (debug).")
    ap.add_argument("--strict_masks", action="store_true",
                    help="If set, require mask_ill/mask_ont all ones and group all 11 to pass sanity checks.")
    ap.add_argument("--out_json", type=str, default=None, help="Write full report JSON to file.")
    ap.add_argument("--out_csv", type=str, default=None,
                    help="Write a compact CSV row with key metrics (ALL/SNP/INDEL) for tracking.")
    args = ap.parse_args()

    inner_dir = Path(args.inner_dir)
    meta_csv = None if args.no_meta_fallback else Path(args.meta_csv)
    raw_ill_dir = Path(args.raw_ill_dir) if not args.no_meta_fallback else None
    raw_ont_dir = Path(args.raw_ont_dir) if not args.no_meta_fallback else None

    report = analyze_inner(
        inner_dir=inner_dir,
        meta_csv=meta_csv if (meta_csv and meta_csv.exists()) else None,
        raw_ill_dir=raw_ill_dir if raw_ill_dir else None,
        raw_ont_dir=raw_ont_dir if raw_ont_dir else None,
        max_shards=args.max_shards,
        strict_masks=args.strict_masks,
    )

    print_report(report)

    if args.out_json:
        outp = Path(args.out_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote JSON report to: {outp}")

    if args.out_csv:
        if pd is None:
            raise RuntimeError("pandas required for --out_csv")
        rows = []
        base = {
            "inner_dir": str(inner_dir),
            "n_total": report["n_total"],
            **{f"label_count_{k}": report["label_counts"][str(k)] for k in LABELS},
        }
        for slice_name in ("ALL", "SNP", "INDEL"):
            m = report["metrics"].get(slice_name, {})
            ill = m.get("teacher_ill") or {}
            ont = m.get("teacher_ont") or {}
            agr = m.get("agreement_ill_ont") or {}
            row = dict(base)
            row["slice"] = slice_name
            row["n_slice"] = m.get("n", 0)
            row["ill_acc"] = ill.get("accuracy", np.nan)
            row["ill_macro_f1"] = (ill.get("macro") or {}).get("f1", np.nan)
            row["ill_macro_recall"] = (ill.get("macro") or {}).get("recall", np.nan)
            row["ont_acc"] = ont.get("accuracy", np.nan)
            row["ont_macro_f1"] = (ont.get("macro") or {}).get("f1", np.nan)
            row["ont_macro_recall"] = (ont.get("macro") or {}).get("recall", np.nan)
            row["agreement"] = agr.get("agreement", np.nan)
            rows.append(row)

            # Flatten confusion matrices (9 cols) for tracking
            if ill and "cm" in ill:
                cm = ill["cm"]
                row.update({
                    "ill_cm_00": cm[0][0], "ill_cm_01": cm[0][1], "ill_cm_02": cm[0][2],
                    "ill_cm_10": cm[1][0], "ill_cm_11": cm[1][1], "ill_cm_12": cm[1][2],
                    "ill_cm_20": cm[2][0], "ill_cm_21": cm[2][1], "ill_cm_22": cm[2][2],
                })
            if ont and "cm" in ont:
                cm = ont["cm"]
                row.update({
                    "ont_cm_00": cm[0][0], "ont_cm_01": cm[0][1], "ont_cm_02": cm[0][2],
                    "ont_cm_10": cm[1][0], "ont_cm_11": cm[1][1], "ont_cm_12": cm[1][2],
                    "ont_cm_20": cm[2][0], "ont_cm_21": cm[2][1], "ont_cm_22": cm[2][2],
                })



        df = pd.DataFrame(rows)
        outp = Path(args.out_csv)
        outp.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(outp, index=False)
        print(f"Wrote CSV summary to: {outp}")


if __name__ == "__main__":
    main()