#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze OUTER join dataset (singleton-only, no ambiguous loci) for:
A) exploration + sanity checks
B) teacher metrics (Illumina/ONT) + agreement + stratified by variant_type and mask-group
C) imbalance + concrete training strategy recommendations

OUTER shards expected keys (from your builder):
- locus (str/bytes)
- label (int 0/1/2)   # chosen as Ill if present else ONT
- mask_ill, mask_ont (0/1)
- group (11=both, 10=ill-only, 1=ont-only)
- ill_embeddings, ont_embeddings
- (optional include_teacher=1): ill_logits/ill_probs/ont_logits/ont_probs

Meta CSV (recommended) expected columns (from your builder):
- locus, label, label_ill, label_ont, mask_ill, mask_ont, ill_npz, ill_idx, ont_npz, ont_idx

Variant type:
- If OUTER shards don't contain variant_type, we recover it from raw NPZ referenced by meta
  Using encoding: 1=SNP, 2=INDEL (as you confirmed).
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

LABELS = [0, 1, 2]
LABEL_NAMES = {0: "hom-ref", 1: "het", 2: "hom-alt"}

VT_CANON = {"SNP": "SNP", "SNV": "SNP", "INDEL": "INDEL", "INS": "INDEL", "DEL": "INDEL"}


def softmax_np(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    logits = logits.astype(np.float64)
    m = np.max(logits, axis=axis, keepdims=True)
    e = np.exp(logits - m)
    return (e / np.sum(e, axis=axis, keepdims=True)).astype(np.float32)


def confusion_matrix_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> np.ndarray:
    cm = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        if 0 <= t < k and 0 <= p < k:
            cm[t, p] += 1
    return cm


def prf_from_cm(cm: np.ndarray) -> Dict[str, Any]:
    eps = 1e-12
    k = cm.shape[0]
    per_class = {}
    ps, rs, f1s, sups = [], [], [], []
    for c in range(k):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        p = tp / (tp + fp + eps)
        r = tp / (tp + fn + eps)
        f1 = 2 * p * r / (p + r + eps)
        sup = int(cm[c, :].sum())
        per_class[c] = {"precision": float(p), "recall": float(r), "f1": float(f1), "support": sup}
        ps.append(p); rs.append(r); f1s.append(f1); sups.append(sup)

    acc = float(np.trace(cm) / (cm.sum() + eps))
    macro = {"precision": float(np.mean(ps)), "recall": float(np.mean(rs)), "f1": float(np.mean(f1s))}
    return {"accuracy": acc, "macro": macro, "per_class": per_class, "cm": cm.tolist()}


def normalize_variant_type(v) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, (bytes, np.bytes_)):
        v = v.decode("utf-8", errors="ignore")

    if isinstance(v, (int, np.integer)):
        iv = int(v)
        if iv == 1:
            return "SNP"
        if iv == 2:
            return "INDEL"
        return None

    s = str(v).strip().upper()
    if s.isdigit():
        iv = int(s)
        if iv == 1:
            return "SNP"
        if iv == 2:
            return "INDEL"
        return None

    if s in VT_CANON:
        return VT_CANON[s]
    if "SNP" in s or "SNV" in s:
        return "SNP"
    if "INDEL" in s or "INS" in s or "DEL" in s:
        return "INDEL"
    return None


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


def list_npz(folder: Path) -> List[Path]:
    return sorted(folder.glob("*.npz"))


def _resolve_path(p: str, raw_root: Optional[Path]) -> Path:
    pth = Path(p)
    if pth.is_absolute():
        return pth
    if pth.exists():
        return pth.resolve()
    if raw_root is None:
        return pth.resolve()
    return (raw_root / pth).resolve()


@dataclass
class FeatureDims:
    ill_embeddings: Optional[Tuple[int, ...]] = None
    ont_embeddings: Optional[Tuple[int, ...]] = None
    ill_logits: Optional[Tuple[int, ...]] = None
    ont_logits: Optional[Tuple[int, ...]] = None
    ill_probs: Optional[Tuple[int, ...]] = None
    ont_probs: Optional[Tuple[int, ...]] = None


# --------------------------
# Meta loading (locus lookup)
# --------------------------

def load_outer_meta(meta_csv: Path) -> Dict[str, Dict[str, Any]]:
    """
    Returns dict: locus -> meta row fields (label_ill/label_ont/masks/npz paths/idx)
    """
    lookup: Dict[str, Dict[str, Any]] = {}
    with open(meta_csv, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            locus = row["locus"]
            lookup[locus] = row
    return lookup


def build_vt_map_from_meta(meta_lookup: Dict[str, Dict[str, Any]], raw_ill_dir: Optional[Path], raw_ont_dir: Optional[Path]) -> Dict[str, str]:
    """
    locus -> SNP/INDEL, prefer Ill raw, fallback ONT raw
    """
    ill_groups = defaultdict(list)
    ont_groups = defaultdict(list)

    for locus, row in meta_lookup.items():
        ill_npz = row.get("ill_npz", "")
        ill_idx = row.get("ill_idx", "")
        ont_npz = row.get("ont_npz", "")
        ont_idx = row.get("ont_idx", "")
        if ill_npz and ill_idx not in ("", None):
            ill_groups[ill_npz].append((locus, int(ill_idx)))
        if ont_npz and ont_idx not in ("", None):
            ont_groups[ont_npz].append((locus, int(ont_idx)))

    vt_map: Dict[str, str] = {}

    # Illumina first
    for shard_str, items in ill_groups.items():
        sp = _resolve_path(shard_str, raw_ill_dir)
        if not sp.exists():
            continue
        npz = np.load(sp, allow_pickle=True)
        vt_key = None
        for k in npz.files:
            if k.lower() in ("variant_type", "vartype", "varianttype", "type"):
                vt_key = k
                break
        if vt_key is None:
            npz.close()
            continue
        vt_arr = npz[vt_key]
        for locus, idx in items:
            if 0 <= idx < len(vt_arr):
                vt = normalize_variant_type(vt_arr[idx])
                if vt in ("SNP", "INDEL"):
                    vt_map[locus] = vt
        npz.close()

    # ONT fallback
    for shard_str, items in ont_groups.items():
        sp = _resolve_path(shard_str, raw_ont_dir)
        if not sp.exists():
            continue
        npz = np.load(sp, allow_pickle=True)
        vt_key = None
        for k in npz.files:
            if k.lower() in ("variant_type", "vartype", "varianttype", "type"):
                vt_key = k
                break
        if vt_key is None:
            npz.close()
            continue
        vt_arr = npz[vt_key]
        for locus, idx in items:
            if locus in vt_map:
                continue
            if 0 <= idx < len(vt_arr):
                vt = normalize_variant_type(vt_arr[idx])
                if vt in ("SNP", "INDEL"):
                    vt_map[locus] = vt
        npz.close()

    return vt_map


# --------------------------
# Main analysis
# --------------------------

def analyze_outer(
    outer_dir: Path,
    meta_csv: Optional[Path],
    raw_ill_dir: Optional[Path],
    raw_ont_dir: Optional[Path],
    max_shards: Optional[int],
    zero_check_per_shard: int,
    zero_tol: float,
) -> Dict[str, Any]:

    shards = list_npz(outer_dir)
    if max_shards is not None:
        shards = shards[:max_shards]
    if not shards:
        raise RuntimeError(f"No .npz shards found in {outer_dir}")

    meta_lookup = None
    vt_map = None
    if meta_csv is not None and meta_csv.exists():
        meta_lookup = load_outer_meta(meta_csv)
        # try vt recovery if raw dirs provided
        if raw_ill_dir is not None or raw_ont_dir is not None:
            vt_map = build_vt_map_from_meta(meta_lookup, raw_ill_dir, raw_ont_dir)

    # counts
    n_total = 0
    label_counts = Counter()
    label_counts_by_group = defaultdict(Counter)
    label_counts_by_mask = defaultdict(Counter)
    group_counts = Counter()
    mask_counts = Counter()  # (mask_ill, mask_ont)

    vt_counts = Counter()
    vt_counts_by_group = defaultdict(Counter)
    vt_label_by_group = defaultdict(lambda: defaultdict(Counter))  # group -> vt -> label Counter

    # disagreement (group 11 only, from meta labels if available)
    disagree_label_ill_ont = 0
    n_both_with_labels = 0
    disagree_table = Counter()  # (ill,ont)

    # feature dims
    dims = FeatureDims()
    # determine dims from first shard (may decompress once)
    d0 = np.load(shards[0], allow_pickle=True)
    if "ill_embeddings" in d0.files:
        dims.ill_embeddings = tuple(d0["ill_embeddings"].shape[1:])
    if "ont_embeddings" in d0.files:
        dims.ont_embeddings = tuple(d0["ont_embeddings"].shape[1:])
    if "ill_logits" in d0.files:
        dims.ill_logits = tuple(d0["ill_logits"].shape[1:])
    if "ont_logits" in d0.files:
        dims.ont_logits = tuple(d0["ont_logits"].shape[1:])
    if "ill_probs" in d0.files:
        dims.ill_probs = tuple(d0["ill_probs"].shape[1:])
    if "ont_probs" in d0.files:
        dims.ont_probs = tuple(d0["ont_probs"].shape[1:])
    d0.close()

    # zero checks
    zero_violations = {
        "ill_embeddings": 0,
        "ont_embeddings": 0,
        "ill_logits": 0,
        "ont_logits": 0,
        "ill_probs": 0,
        "ont_probs": 0,
    }
    zero_checked = 0

    # teacher metrics buckets
    # keys: (slice_name, group) where slice_name in ALL/SNP/INDEL
    def new_bucket():
        return {"y": [], "ill_pred": [], "ont_pred": [], "ill_mask": [], "ont_mask": [],
                "y_ill": [], "y_ont": [], "has_y_ill": [], "has_y_ont": []}

    buckets = defaultdict(new_bucket)

    def add_to_bucket(vt: str, group: int, y: int, ill_pred, ont_pred, m_ill: int, m_ont: int,
                      y_ill: Optional[int], y_ont: Optional[int]):
        for slice_name in ("ALL", vt if vt in ("SNP", "INDEL") else "ALL"):
            b = buckets[(slice_name, group)]
            b["y"].append(y)
            b["ill_mask"].append(m_ill)
            b["ont_mask"].append(m_ont)
            if ill_pred is not None:
                b["ill_pred"].append(int(ill_pred))
            else:
                b["ill_pred"].append(None)
            if ont_pred is not None:
                b["ont_pred"].append(int(ont_pred))
            else:
                b["ont_pred"].append(None)

            if y_ill is not None:
                b["y_ill"].append(int(y_ill)); b["has_y_ill"].append(1)
            else:
                b["y_ill"].append(-1); b["has_y_ill"].append(0)

            if y_ont is not None:
                b["y_ont"].append(int(y_ont)); b["has_y_ont"].append(1)
            else:
                b["y_ont"].append(-1); b["has_y_ont"].append(0)

    # iterate shards
    for sp in shards:
        npz = np.load(sp, allow_pickle=True)
        files = set(npz.files)

        loci = _as_str_list(npz["locus"])
        y = npz["label"].astype(np.int64)
        mi = npz["mask_ill"].astype(np.int64).reshape(-1)
        mo = npz["mask_ont"].astype(np.int64).reshape(-1)
        grp = npz["group"].astype(np.int64).reshape(-1)

        # variant_type source: shard field if exists else vt_map else UNKNOWN
        vt_key = None
        for k in files:
            if k.lower() in ("variant_type", "vartype", "varianttype", "type"):
                vt_key = k
                break
        vt_arr = npz[vt_key] if vt_key else None

        # teacher preds
        def pred_from(prefix: str) -> Optional[np.ndarray]:
            pk = f"{prefix}_probs"
            lk = f"{prefix}_logits"
            if pk in files:
                p = npz[pk].astype(np.float32)
            elif lk in files:
                p = softmax_np(npz[lk], axis=-1)
            else:
                return None
            if p.ndim != 2 or p.shape[1] != 3:
                raise RuntimeError(f"{sp}: {prefix} probs/logits expected (N,3), got {p.shape}")
            return np.argmax(p, axis=1).astype(np.int64)

        ill_pred_arr = pred_from("ill")
        ont_pred_arr = pred_from("ont")

        n = len(y)
        n_total += n

        # update global/group counts
        for i in range(n):
            yi = int(y[i])
            label_counts[yi] += 1

            g = int(grp[i])
            group_counts[g] += 1
            label_counts_by_group[g][yi] += 1

            mask = (int(mi[i]), int(mo[i]))
            mask_counts[mask] += 1
            label_counts_by_mask[mask][yi] += 1

        # vt + metrics + disagreement (per example loop)
        for i in range(n):
            L = loci[i]
            yi = int(y[i])
            g = int(grp[i])
            m_ill = int(mi[i]); m_ont = int(mo[i])

            if vt_arr is not None:
                vt = normalize_variant_type(vt_arr[i]) or "UNKNOWN"
            elif vt_map is not None:
                vt = vt_map.get(L, "UNKNOWN")
            else:
                vt = "UNKNOWN"

            vt_counts[vt] += 1
            vt_counts_by_group[g][vt] += 1
            vt_label_by_group[g][vt][yi] += 1

            # meta labels (for disagreement and teacher-vs-own-modality)
            y_ill = None
            y_ont = None
            if meta_lookup is not None:
                row = meta_lookup.get(L)
                if row is not None:
                    # label_ill/label_ont may be -1 in meta for missing modality
                    try:
                        li = int(row.get("label_ill", -1))
                        lo = int(row.get("label_ont", -1))
                    except Exception:
                        li, lo = -1, -1
                    if li >= 0:
                        y_ill = li
                    if lo >= 0:
                        y_ont = lo

                    # disagreement only meaningful if both exist
                    if m_ill == 1 and m_ont == 1 and y_ill is not None and y_ont is not None:
                        n_both_with_labels += 1
                        if y_ill != y_ont:
                            disagree_label_ill_ont += 1
                        disagree_table[(y_ill, y_ont)] += 1

            ill_p = int(ill_pred_arr[i]) if ill_pred_arr is not None else None
            ont_p = int(ont_pred_arr[i]) if ont_pred_arr is not None else None

            add_to_bucket(vt, g, yi, ill_p, ont_p, m_ill, m_ont, y_ill, y_ont)

        # zero checks (sample)
        if zero_check_per_shard > 0:
            rng_idx = np.random.choice(n, size=min(zero_check_per_shard, n), replace=False)
            # We only check vectors where mask==0
            if "ill_embeddings" in files:
                ill_e = npz["ill_embeddings"].astype(np.float32)
                for j in rng_idx:
                    if int(mi[j]) == 0:
                        zero_checked += 1
                        if float(np.max(np.abs(ill_e[j]))) > zero_tol:
                            zero_violations["ill_embeddings"] += 1
            if "ont_embeddings" in files:
                ont_e = npz["ont_embeddings"].astype(np.float32)
                for j in rng_idx:
                    if int(mo[j]) == 0:
                        zero_checked += 1
                        if float(np.max(np.abs(ont_e[j]))) > zero_tol:
                            zero_violations["ont_embeddings"] += 1
            if "ill_logits" in files:
                ill_l = npz["ill_logits"].astype(np.float32)
                for j in rng_idx:
                    if int(mi[j]) == 0:
                        zero_checked += 1
                        if float(np.max(np.abs(ill_l[j]))) > zero_tol:
                            zero_violations["ill_logits"] += 1
            if "ont_logits" in files:
                ont_l = npz["ont_logits"].astype(np.float32)
                for j in rng_idx:
                    if int(mo[j]) == 0:
                        zero_checked += 1
                        if float(np.max(np.abs(ont_l[j]))) > zero_tol:
                            zero_violations["ont_logits"] += 1
            if "ill_probs" in files:
                ill_p = npz["ill_probs"].astype(np.float32)
                for j in rng_idx:
                    if int(mi[j]) == 0:
                        zero_checked += 1
                        if float(np.max(np.abs(ill_p[j]))) > zero_tol:
                            zero_violations["ill_probs"] += 1
            if "ont_probs" in files:
                ont_p = npz["ont_probs"].astype(np.float32)
                for j in rng_idx:
                    if int(mo[j]) == 0:
                        zero_checked += 1
                        if float(np.max(np.abs(ont_p[j]))) > zero_tol:
                            zero_violations["ont_probs"] += 1

        npz.close()

    # --------------------------
    # Compute metrics
    # --------------------------
    def compute_teacher_metrics(y_true: List[int], y_pred: List[Optional[int]], mask: List[int]) -> Optional[Dict[str, Any]]:
        # filter mask==1 and y_pred not None
        yt = []
        yp = []
        for t, p, m in zip(y_true, y_pred, mask):
            if m == 1 and p is not None and t in (0, 1, 2):
                yt.append(int(t)); yp.append(int(p))
        if not yt:
            return None
        cm = confusion_matrix_k(np.array(yt), np.array(yp), 3)
        return prf_from_cm(cm)

    def compute_agreement(p1: List[Optional[int]], p2: List[Optional[int]], m1: List[int], m2: List[int]) -> Optional[Dict[str, Any]]:
        a = []
        for x, y, mx, my in zip(p1, p2, m1, m2):
            if mx == 1 and my == 1 and x is not None and y is not None:
                a.append(int(x) == int(y))
        if not a:
            return None
        return {"agreement": float(np.mean(a)), "n": int(len(a))}

    def teacher_vs_own_label(y_mod: List[int], has_y: List[int], y_pred: List[Optional[int]], mask: List[int]) -> Optional[Dict[str, Any]]:
        yt = []
        yp = []
        for t, h, p, m in zip(y_mod, has_y, y_pred, mask):
            if h == 1 and m == 1 and p is not None and t in (0, 1, 2):
                yt.append(int(t)); yp.append(int(p))
        if not yt:
            return None
        cm = confusion_matrix_k(np.array(yt), np.array(yp), 3)
        return prf_from_cm(cm)

    metrics = {}
    # compute for each (slice, group)
    for (slice_name, group), b in buckets.items():
        y_true = b["y"]
        ill_pred = b["ill_pred"]
        ont_pred = b["ont_pred"]
        ill_mask = b["ill_mask"]
        ont_mask = b["ont_mask"]

        m = {
            "n": len(y_true),
            "teacher_ill_vs_label": compute_teacher_metrics(y_true, ill_pred, ill_mask),
            "teacher_ont_vs_label": compute_teacher_metrics(y_true, ont_pred, ont_mask),
            "agreement_ill_ont": compute_agreement(ill_pred, ont_pred, ill_mask, ont_mask),

            # if meta provided: teacher vs modality-specific label
            "teacher_ill_vs_label_ill": teacher_vs_own_label(b["y_ill"], b["has_y_ill"], ill_pred, ill_mask),
            "teacher_ont_vs_label_ont": teacher_vs_own_label(b["y_ont"], b["has_y_ont"], ont_pred, ont_mask),
        }
        metrics[f"{slice_name}_group{group}"] = m

    # imbalance recommendations (basic)
    total = sum(label_counts.values())
    freq = {k: label_counts[k] / max(total, 1) for k in LABELS}

    out = {
        "paths": {
            "outer_dir": str(outer_dir),
            "meta_csv": str(meta_csv) if meta_csv else None,
            "raw_ill_dir": str(raw_ill_dir) if raw_ill_dir else None,
            "raw_ont_dir": str(raw_ont_dir) if raw_ont_dir else None,
        },
        "n_total": int(n_total),
        "feature_dims": asdict(dims),

        "group_counts": {str(k): int(v) for k, v in group_counts.items()},
        "mask_counts": {f"{k[0]}{k[1]}": int(v) for k, v in mask_counts.items()},

        "label_counts": {str(k): int(label_counts[k]) for k in LABELS},
        "label_counts_by_group": {str(g): {str(k): int(c[k]) for k in LABELS} for g, c in label_counts_by_group.items()},
        "label_counts_by_mask": {f"{m[0]}{m[1]}": {str(k): int(c[k]) for k in LABELS} for m, c in label_counts_by_mask.items()},

        "variant_type_counts": {str(k): int(v) for k, v in vt_counts.items()},
        "variant_type_counts_by_group": {str(g): dict(vt_counts_by_group[g]) for g in vt_counts_by_group},

        "variant_type_label_by_group": {
            str(g): {vt: {str(k): int(vt_label_by_group[g][vt][k]) for k in LABELS}
                     for vt in vt_label_by_group[g]}
            for g in vt_label_by_group
        },

        "disagreement_group11": {
            "available": bool(meta_lookup is not None),
            "n_both_with_labels": int(n_both_with_labels),
            "disagree_count": int(disagree_label_ill_ont),
            "disagree_rate": float(disagree_label_ill_ont / max(n_both_with_labels, 1)),
            "table_label_ill_vs_label_ont": {f"{a}->{b}": int(v) for (a, b), v in disagree_table.items()},
        },

        "zero_checks": {
            "checked": int(zero_checked),
            "tol": float(zero_tol),
            "violations": zero_violations,
            "note": "Violations > 0 means some masked modality vectors are not (near) zero."
        },

        "metrics": metrics,

        "training_recommendations": {
            "label_freq": {str(k): float(freq[k]) for k in LABELS},
            "notes": [
                "OUTER usually has strong class-0 dominance. Consider 2-stage training: (0 vs {1,2}) then (1 vs 2 on positives).",
                "Always condition teacher/eval metrics on masks (ignore teacher outputs where mask==0).",
                "For group=11, quantify label_ill vs label_ont disagreement: it indicates cross-modality label noise; decide whether to drop or model it.",
                "Optimize metrics that reflect your goal under imbalance: AUPRC for variant detection, macro-F1 / recall(hom-alt) for genotyping.",
                "Sampling: balance by group (11/10/01) and by label, at least within minibatches, to avoid the model ignoring rare groups/types.",
                "Track performance per variant_type (SNP/INDEL); ONT INDEL tends to be harder."
            ],
        },
    }
    return out


def print_report(r: Dict[str, Any]) -> None:
    print("\n" + "=" * 90)
    print("OUTER JOIN DATASET REPORT")
    print("=" * 90)
    print(f"OUTER dir: {r['paths']['outer_dir']}")
    if r["paths"]["meta_csv"]:
        print(f"meta_csv:  {r['paths']['meta_csv']}")
    print(f"Total examples: {r['n_total']:,}")

    print("\nMask counts (mask_illmask_ont):")
    for k, v in sorted(r["mask_counts"].items(), key=lambda x: -x[1]):
        print(f"  {k}: {v:,}")

    print("\nGroup counts:")
    for k, v in sorted(r["group_counts"].items(), key=lambda x: int(x[0])):
        print(f"  group {k}: {v:,}")

    print("\nLabel distribution (global):")
    for k in LABELS:
        c = r["label_counts"][str(k)]
        print(f"  {k} ({LABEL_NAMES[k]}): {c:,}")

    print("\nVariant type distribution (global):")
    for vt, c in sorted(r["variant_type_counts"].items(), key=lambda x: -x[1]):
        print(f"  {vt:8s}: {c:,}")

    print("\nFeature dims (per-example):")
    for k, v in r["feature_dims"].items():
        print(f"  {k:14s}: {tuple(v) if v is not None else '(missing)'}")

    dg = r["disagreement_group11"]
    print("\nDisagreement on group=11 (label_ill vs label_ont) from meta:")
    if dg["available"]:
        print(f"  n_both_with_labels: {dg['n_both_with_labels']:,}")
        print(f"  disagree_count:     {dg['disagree_count']:,}")
        print(f"  disagree_rate:      {dg['disagree_rate']:.4f}")
    else:
        print("  (meta not provided -> cannot compute)")

    zc = r["zero_checks"]
    print("\nZero-vector sanity (masked modality should be ~0):")
    print(f"  checked samples: {zc['checked']:,} tol={zc['tol']}")
    print(f"  violations: {zc['violations']}")

    print("\nTeacher metrics (key slices):")
    # show a few important ones
    for key in [k for k in r["metrics"].keys() if k.startswith("ALL_group")]:
        m = r["metrics"][key]
        print(f"  {key}: n={m['n']}")
        if m["teacher_ill_vs_label"]:
            print(f"    ill_vs_label: acc={m['teacher_ill_vs_label']['accuracy']:.4f} macroF1={m['teacher_ill_vs_label']['macro']['f1']:.4f}")
        if m["teacher_ont_vs_label"]:
            print(f"    ont_vs_label: acc={m['teacher_ont_vs_label']['accuracy']:.4f} macroF1={m['teacher_ont_vs_label']['macro']['f1']:.4f}")
        if m["agreement_ill_ont"]:
            print(f"    ill-ont agreement: {m['agreement_ill_ont']['agreement']:.4f} (n={m['agreement_ill_ont']['n']})")

    print("\nRecommendations:")
    for line in r["training_recommendations"]["notes"]:
        print(f"  - {line}")

    print("\n" + "=" * 90 + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outer_dir", type=str, default="data/4_out/datasets/HG003/join_outer_singletons")
    ap.add_argument("--meta_csv", type=str, default="data/4_out/datasets/HG003/join_outer_singletons/hg003_chr20_outer_meta.csv")
    ap.add_argument("--raw_ill_dir", type=str, default="data/4_out/datasets/HG003/illumina_chr20_train")
    ap.add_argument("--raw_ont_dir", type=str, default="data/4_out/datasets/HG003/ont_chr20_train")
    ap.add_argument("--no_meta", action="store_true", help="Disable meta usage; variant_type/disagreement may be UNKNOWN.")
    ap.add_argument("--max_shards", type=int, default=None)
    ap.add_argument("--zero_check_per_shard", type=int, default=64, help="Sample count per shard for masked-zero sanity checks.")
    ap.add_argument("--zero_tol", type=float, default=1e-8, help="Tolerance for masked feature max-abs.")
    ap.add_argument("--out_json", type=str, default="data/4_out/reports/hybrid_outer_report.json")
    ap.add_argument("--out_csv", type=str, default="data/4_out/reports/hybrid_outer_metrics.csv",
                    help="Compact CSV summary (global + per group/slice).")
    args = ap.parse_args()

    outer_dir = Path(args.outer_dir)
    meta_csv = None if args.no_meta else Path(args.meta_csv)
    raw_ill_dir = None if args.no_meta else Path(args.raw_ill_dir)
    raw_ont_dir = None if args.no_meta else Path(args.raw_ont_dir)

    report = analyze_outer(
        outer_dir=outer_dir,
        meta_csv=meta_csv if (meta_csv and meta_csv.exists()) else None,
        raw_ill_dir=raw_ill_dir if (raw_ill_dir and raw_ill_dir.exists()) else None,
        raw_ont_dir=raw_ont_dir if (raw_ont_dir and raw_ont_dir.exists()) else None,
        max_shards=args.max_shards,
        zero_check_per_shard=args.zero_check_per_shard,
        zero_tol=args.zero_tol,
    )

    print_report(report)

    # JSON
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote JSON report to: {out_json}")

    # Compact CSV
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # create a compact table: one row per metric key
    rows = []
    for key, m in report["metrics"].items():
        row = {
            "slice_group": key,
            "n": m.get("n", 0),
            "ill_acc_vs_label": (m.get("teacher_ill_vs_label") or {}).get("accuracy", np.nan),
            "ill_macro_f1_vs_label": ((m.get("teacher_ill_vs_label") or {}).get("macro") or {}).get("f1", np.nan),
            "ont_acc_vs_label": (m.get("teacher_ont_vs_label") or {}).get("accuracy", np.nan),
            "ont_macro_f1_vs_label": ((m.get("teacher_ont_vs_label") or {}).get("macro") or {}).get("f1", np.nan),
            "agreement": (m.get("agreement_ill_ont") or {}).get("agreement", np.nan),
            "agreement_n": (m.get("agreement_ill_ont") or {}).get("n", 0),

            "ill_acc_vs_label_ill": (m.get("teacher_ill_vs_label_ill") or {}).get("accuracy", np.nan),
            "ont_acc_vs_label_ont": (m.get("teacher_ont_vs_label_ont") or {}).get("accuracy", np.nan),
        }
        rows.append(row)

    # write CSV
    cols = list(rows[0].keys()) if rows else ["slice_group"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote CSV summary to: {out_csv}")


if __name__ == "__main__":
    main()