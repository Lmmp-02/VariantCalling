#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hybrid OUTER 3-class with groupwise heads + auxiliary binary loss.

Main task
---------
- 3-class genotype classification: y3 in {0,1,2}

Auxiliary task
--------------
- Binary variant detection: y_bin = 0 if y3 == 0 else 1

Policy
------
- Visible groups: group01 + group10 + group11
- Target:
    * group01 -> label_ont
    * group10 -> label_ill
    * group11 -> keep only consensus loci where label_ill == label_ont
- vt_mismatch is NOT excluded
- label_mismatch in group11 is excluded from the main benchmark

Architecture
------------
- Shared trunk
- One 3-class head per group: {1,10,11}
- One binary auxiliary head per group: {1,10,11}

Inference
---------
- Official prediction uses ONLY argmax(logits_3c)
- Auxiliary binary head is used only as training regularizer and for diagnostics

Loss
----
L_total = CE_3class + aux_bin_weight * BCEWithLogits(binary)

Outputs
-------
- config.json
- split_bins_used.json
- best.pt / final.pt
- best_metrics.json / last_metrics.json              (standard 3-class contract)
- best_aux_metrics.json / last_aux_metrics.json      (auxiliary diagnostics)
- optional out_csv
"""

import argparse
import csv
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

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

    return s.lower()


def strip_standard_3c_metrics(m: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "n": m["n"],
        "cm": m["cm"],
        "acc": m["acc"],
        "macro_f1": m["macro_f1"],
        "macro_recall": m["macro_recall"],
        "macro_precision": m["macro_precision"],
        "weighted_f1": m["weighted_f1"],
        "per_class": m["per_class"],
    }

    if "by_group" in m:
        out["by_group"] = {}
        for g, gm in m["by_group"].items():
            out["by_group"][g] = {
                "n": gm["n"],
                "cm": gm["cm"],
                "acc": gm["acc"],
                "macro_f1": gm["macro_f1"],
                "macro_recall": gm["macro_recall"],
                "macro_precision": gm["macro_precision"],
                "weighted_f1": gm["weighted_f1"],
                "per_class": gm["per_class"],
            }

    return out


def build_aux_report(full_m: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "n": full_m["n"],
        "official_variant": full_m["official_variant"],
        "official_variant_from_cm3": full_m["official_variant_from_cm3"],
        "aux_bin": full_m["aux_bin"],
        "by_group": {},
    }

    for g, gm in full_m.get("by_group", {}).items():
        out["by_group"][g] = {
            "n": gm["n"],
            "official_variant": gm["official_variant"],
            "official_variant_from_cm3": gm["official_variant_from_cm3"],
            "aux_bin": gm["aux_bin"],
        }

    if "by_variant_type" in full_m:
        out["by_variant_type"] = {}
        for vt, vm in full_m["by_variant_type"].items():
            out["by_variant_type"][vt] = {
                "n": vm["n"],
                "official_variant": vm["official_variant"],
                "official_variant_from_cm3": vm["official_variant_from_cm3"],
                "aux_bin": vm["aux_bin"],
            }

    return out


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

            vt_raw = row.get("variant_type", None)
            if vt_raw is None or vt_raw == "":
                vt_raw = row.get("vt_name", None)

            out[locus] = {
                "label_ill": label_ill,
                "label_ont": label_ont,
                "vt_mismatch": vt_mismatch,
                "label_mismatch": label_mismatch,
                "variant_type": parse_variant_type(vt_raw),
            }

    return out


def build_refs_all(outer_dir: Path, meta_lookup: Dict[str, Dict[str, Any]], bin_size: int) -> List[Ref]:
    refs: List[Ref] = []

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
            variant_type = str(meta["variant_type"])

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

            refs.append(
                Ref(
                    shard=str(sp),
                    row=i,
                    locus=locus,
                    y3=int(y3),
                    y_bin=int(y3 != 0),
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
    by_ybin = {0: 0, 1: 0}
    by_vt = {"SNP": 0, "INDEL": 0, "unknown": 0}
    by_vt_mismatch = {0: 0, 1: 0}

    for r in refs:
        by_group[r.group] += 1
        by_y3[r.y3] += 1
        by_ybin[r.y_bin] += 1
        by_vt[r.variant_type if r.variant_type in by_vt else "unknown"] += 1
        by_vt_mismatch[r.vt_mismatch] += 1

    return {
        "name": name,
        "n": n,
        "by_group": by_group,
        "by_y3": by_y3,
        "by_ybin": by_ybin,
        "by_variant_type": by_vt,
        "by_vt_mismatch": by_vt_mismatch,
    }


def print_split_summary(s: Dict[str, Any]) -> None:
    n = max(int(s["n"]), 1)

    def pct(x):
        return 100.0 * float(x) / float(n)

    print(f"\n[SplitSummary] {s['name']} n={int(s['n']):,}")
    print("  groups:       ", {k: f"{v:,} ({pct(v):.1f}%)" for k, v in sorted(s["by_group"].items())})
    print("  y_3:          ", {k: f"{v:,} ({pct(v):.1f}%)" for k, v in sorted(s["by_y3"].items())})
    print("  y_bin:        ", {k: f"{v:,} ({pct(v):.1f}%)" for k, v in sorted(s["by_ybin"].items())})
    print("  variant_type: ", {k: f"{v:,} ({pct(v):.1f}%)" for k, v in s["by_variant_type"].items()})
    print("  vt_mismatch:  ", {k: f"{v:,} ({pct(v):.3f}%)" for k, v in sorted(s["by_vt_mismatch"].items())})


class GroupwiseLinear3CAux(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.heads_3c = nn.ModuleDict({
            "1": nn.Linear(in_dim, 3),
            "10": nn.Linear(in_dim, 3),
            "11": nn.Linear(in_dim, 3),
        })
        self.heads_bin = nn.ModuleDict({
            "1": nn.Linear(in_dim, 1),
            "10": nn.Linear(in_dim, 1),
            "11": nn.Linear(in_dim, 1),
        })

    def forward(self, x: torch.Tensor, groups: torch.Tensor):
        logits_3c = torch.zeros((x.shape[0], 3), dtype=x.dtype, device=x.device)
        logits_bin = torch.zeros((x.shape[0],), dtype=x.dtype, device=x.device)

        for g in (1, 10, 11):
            mask = (groups == g)
            if mask.any():
                logits_3c[mask] = self.heads_3c[str(g)](x[mask])
                logits_bin[mask] = self.heads_bin[str(g)](x[mask]).squeeze(1)

        return logits_3c, logits_bin


class GroupwiseMLP3CAux(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.heads_3c = nn.ModuleDict({
            "1": nn.Linear(hidden, 3),
            "10": nn.Linear(hidden, 3),
            "11": nn.Linear(hidden, 3),
        })
        self.heads_bin = nn.ModuleDict({
            "1": nn.Linear(hidden, 1),
            "10": nn.Linear(hidden, 1),
            "11": nn.Linear(hidden, 1),
        })

    def forward(self, x: torch.Tensor, groups: torch.Tensor):
        h = self.trunk(x)
        logits_3c = torch.zeros((h.shape[0], 3), dtype=h.dtype, device=h.device)
        logits_bin = torch.zeros((h.shape[0],), dtype=h.dtype, device=h.device)

        for g in (1, 10, 11):
            mask = (groups == g)
            if mask.any():
                logits_3c[mask] = self.heads_3c[str(g)](h[mask])
                logits_bin[mask] = self.heads_bin[str(g)](h[mask]).squeeze(1)

        return logits_3c, logits_bin


def build_model(in_dim: int, arch: str, hidden: int, dropout: float) -> nn.Module:
    if arch == "linear":
        return GroupwiseLinear3CAux(in_dim=in_dim)
    return GroupwiseMLP3CAux(in_dim=in_dim, hidden=hidden, dropout=dropout)


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
        "n": int(len(yt)),
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


def variant_metrics_from_cm_3(cm: np.ndarray) -> Dict[str, Any]:
    eps = 1e-12
    tp = float(cm[1:, 1:].sum())
    fn = float(cm[1:, 0].sum())
    fp = float(cm[0, 1:].sum())
    tn = float(cm[0, 0])

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


@torch.no_grad()
def eval_sharded_3c_aux(
    model,
    by_shard: Dict[str, List[Ref]],
    device,
    batch_size: int,
    add_masks: bool,
) -> Dict[str, Any]:
    model.eval()

    y_true_3c: List[int] = []
    y_pred_3c: List[int] = []

    y_true_bin: List[int] = []
    y_pred_bin_official: List[int] = []
    y_pred_bin_aux: List[int] = []

    by_group = {
        1: {"y3": [], "p3": [], "ybin": [], "pbin_off": [], "pbin_aux": []},
        10: {"y3": [], "p3": [], "ybin": [], "pbin_off": [], "pbin_aux": []},
        11: {"y3": [], "p3": [], "ybin": [], "pbin_off": [], "pbin_aux": []},
    }
    by_vt: Dict[str, Dict[str, List[int]]] = {}

    for shard, items in by_shard.items():
        npz = np.load(shard, allow_pickle=True)

        ill_e = npz["ill_embeddings"].astype(np.float32, copy=False)
        ont_e = npz["ont_embeddings"].astype(np.float32, copy=False)
        mi = npz["mask_ill"].astype(np.float32, copy=False).reshape(-1, 1)
        mo = npz["mask_ont"].astype(np.float32, copy=False).reshape(-1, 1)

        rows = np.array([r.row for r in items], dtype=np.int64)
        ys3 = np.array([r.y3 for r in items], dtype=np.int64)
        yb = np.array([r.y_bin for r in items], dtype=np.int64)
        gs = np.array([r.group for r in items], dtype=np.int64)
        vts = [r.variant_type for r in items]

        for i in range(0, len(rows), batch_size):
            rr = rows[i:i + batch_size]
            y3b = ys3[i:i + batch_size]
            ybin_b = yb[i:i + batch_size]
            gb = gs[i:i + batch_size]
            vtb = vts[i:i + batch_size]

            feats = [ill_e[rr], ont_e[rr]]
            if add_masks:
                feats.extend([mi[rr], mo[rr]])
            x_np = np.concatenate(feats, axis=1)

            x = torch.from_numpy(x_np).to(device)
            g = torch.from_numpy(gb).to(device)

            logits_3c, logits_bin = model(x, g)
            pred_3c = torch.argmax(logits_3c, dim=1).cpu().numpy().astype(np.int64)
            prob_bin = torch.sigmoid(logits_bin).cpu().numpy()
            pred_bin_aux = (prob_bin >= 0.5).astype(np.int64)
            pred_bin_official = (pred_3c != 0).astype(np.int64)

            y_true_3c += y3b.tolist()
            y_pred_3c += pred_3c.tolist()
            y_true_bin += ybin_b.tolist()
            y_pred_bin_official += pred_bin_official.tolist()
            y_pred_bin_aux += pred_bin_aux.tolist()

            for yy3, pp3, yyb, pob, pab, gg, vt in zip(
                y3b.tolist(),
                pred_3c.tolist(),
                ybin_b.tolist(),
                pred_bin_official.tolist(),
                pred_bin_aux.tolist(),
                gb.tolist(),
                vtb,
            ):
                by_group[int(gg)]["y3"].append(int(yy3))
                by_group[int(gg)]["p3"].append(int(pp3))
                by_group[int(gg)]["ybin"].append(int(yyb))
                by_group[int(gg)]["pbin_off"].append(int(pob))
                by_group[int(gg)]["pbin_aux"].append(int(pab))

                if vt not in by_vt:
                    by_vt[vt] = {"y3": [], "p3": [], "ybin": [], "pbin_off": [], "pbin_aux": []}
                by_vt[vt]["y3"].append(int(yy3))
                by_vt[vt]["p3"].append(int(pp3))
                by_vt[vt]["ybin"].append(int(yyb))
                by_vt[vt]["pbin_off"].append(int(pob))
                by_vt[vt]["pbin_aux"].append(int(pab))

        npz.close()

    cm = cm_3(y_true_3c, y_pred_3c, n_classes=3)
    out = {
        "n": len(y_true_3c),
        "cm": cm.tolist(),
        **metrics_from_cm_3(cm),
        "official_variant": binary_metrics(y_true_bin, y_pred_bin_official),
        "official_variant_from_cm3": variant_metrics_from_cm_3(cm),
        "aux_bin": binary_metrics(y_true_bin, y_pred_bin_aux),
        "by_group": {},
        "by_variant_type": {},
    }

    for gg in (1, 10, 11):
        if len(by_group[gg]["y3"]) == 0:
            continue
        cmg = cm_3(by_group[gg]["y3"], by_group[gg]["p3"], n_classes=3)
        out["by_group"][str(gg)] = {
            "n": len(by_group[gg]["y3"]),
            "cm": cmg.tolist(),
            **metrics_from_cm_3(cmg),
            "official_variant": binary_metrics(by_group[gg]["ybin"], by_group[gg]["pbin_off"]),
            "official_variant_from_cm3": variant_metrics_from_cm_3(cmg),
            "aux_bin": binary_metrics(by_group[gg]["ybin"], by_group[gg]["pbin_aux"]),
        }

    for vt, d in by_vt.items():
        if len(d["y3"]) == 0:
            continue
        cmv = cm_3(d["y3"], d["p3"], n_classes=3)
        out["by_variant_type"][str(vt)] = {
            "n": len(d["y3"]),
            "cm": cmv.tolist(),
            **metrics_from_cm_3(cmv),
            "official_variant": binary_metrics(d["ybin"], d["pbin_off"]),
            "official_variant_from_cm3": variant_metrics_from_cm_3(cmv),
            "aux_bin": binary_metrics(d["ybin"], d["pbin_aux"]),
        }

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


def build_class_weights_3c(train_refs: List[Ref], max_w: float = 10.0) -> Optional[torch.Tensor]:
    counts = {0: 0, 1: 0, 2: 0}
    for r in train_refs:
        counts[int(r.y3)] += 1

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


def build_pos_weight_bin(train_refs: List[Ref], max_w: float = 5.0) -> Optional[torch.Tensor]:
    n_pos = sum(int(r.y_bin == 1) for r in train_refs)
    n_neg = sum(int(r.y_bin == 0) for r in train_refs)
    if n_pos <= 0 or n_neg <= 0:
        return None

    pw = float(n_neg) / float(n_pos)
    pw = min(pw, float(max_w))
    return torch.tensor([pw], dtype=torch.float32)


def score_from_val(
    val_std: Dict[str, Any],
    val_variant_recall: float,
    val_aux_bin_recall: float,
    save_best_on: str,
) -> float:
    if save_best_on == "val_macro_f1":
        return float(val_std["macro_f1"])
    if save_best_on == "val_acc":
        return float(val_std["acc"])
    if save_best_on == "val_weighted_f1":
        return float(val_std["weighted_f1"])
    if save_best_on == "val_variant_recall":
        return float(val_variant_recall)
    if save_best_on == "val_aux_bin_recall":
        return float(val_aux_bin_recall)
    raise ValueError(f"Unsupported save_best_on={save_best_on}")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--outer_dir", type=str, default="data/4_out/datasets/HG003/join_outer_singletons")
    ap.add_argument("--split_json", type=str, required=True)
    ap.add_argument("--meta_csv", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bin_size", type=int, default=1_000_000)
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--arch", type=str, default="mlp", choices=["mlp", "linear"])
    ap.add_argument("--add_masks", action="store_true")

    ap.add_argument("--use_class_weights", action="store_true")
    ap.add_argument("--max_class_weight", type=float, default=10.0)

    ap.add_argument("--aux_bin_weight", type=float, default=0.25)
    ap.add_argument("--use_aux_pos_weight", action="store_true")
    ap.add_argument("--max_aux_pos_weight", type=float, default=5.0)

    ap.add_argument(
        "--save_best_on",
        type=str,
        default="val_macro_f1",
        choices=[
            "val_macro_f1",
            "val_acc",
            "val_weighted_f1",
            "val_variant_recall",
            "val_aux_bin_recall",
        ],
    )

    ap.add_argument("--save_dir", type=str, required=True)
    ap.add_argument("--out_csv", type=str, default="")

    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    outer_dir = Path(args.outer_dir)
    meta_csv = Path(args.meta_csv)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    split = json.loads(Path(args.split_json).read_text(encoding="utf-8"))
    train_bins = set(split["train_bins"])
    val_bins = set(split["val_bins"])
    test_bins = set(split["test_bins"])

    meta_lookup = load_meta(meta_csv)
    refs_all = build_refs_all(outer_dir=outer_dir, meta_lookup=meta_lookup, bin_size=args.bin_size)

    train_refs = [r for r in refs_all if r.bin_id in train_bins]
    val_refs = [r for r in refs_all if r.bin_id in val_bins]
    test_refs = [r for r in refs_all if r.bin_id in test_bins]

    print(
        "[Split/FILTER] examples "
        "(group01->label_ont, group10->label_ill, group11 consensus only): "
        f"train={len(train_refs)} val={len(val_refs)} test={len(test_refs)}"
    )
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

    if args.add_masks:
        in_dim += 2

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    model = build_model(
        in_dim=in_dim,
        arch=args.arch,
        hidden=args.hidden,
        dropout=args.dropout,
    ).to(device)

    print(
        f"[Model] arch={args.arch} in_dim={in_dim} hidden={args.hidden} "
        f"dropout={args.dropout} add_masks={args.add_masks} device={device}"
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    class_w = None
    if args.use_class_weights:
        class_w = build_class_weights_3c(train_refs, max_w=args.max_class_weight)
        print(f"[Loss-3C] class_weights={class_w.tolist() if class_w is not None else None}")

    ce_3c = nn.CrossEntropyLoss(weight=class_w.to(device) if class_w is not None else None)

    aux_pw = None
    if args.use_aux_pos_weight:
        aux_pw = build_pos_weight_bin(train_refs, max_w=args.max_aux_pos_weight)
        print(f"[Loss-BIN] pos_weight={float(aux_pw.item()) if aux_pw is not None else None}")

    bce_bin = nn.BCEWithLogitsLoss(pos_weight=aux_pw.to(device) if aux_pw is not None else None)

    config_payload = {
        "args": vars(args),
        "resolved": {
            "input_dim": in_dim,
            "device": str(device),
            "loss_main": "CrossEntropyLoss",
            "loss_aux": "BCEWithLogitsLoss",
            "aux_bin_weight": float(args.aux_bin_weight),
            "class_weights_3c": class_w.tolist() if class_w is not None else None,
            "aux_pos_weight": float(aux_pw.item()) if aux_pw is not None else None,
            "add_masks": bool(args.add_masks),
        },
        "dataset_policy": {
            "mode": "hybrid_groupwise_3c_with_aux_binary",
            "groups_keep": [1, 10, 11],
            "group1_target": "label_ont",
            "group10_target": "label_ill",
            "group11_policy": "keep_only_if_label_ill_eq_label_ont",
            "drop_label_mismatch_in_group11": True,
            "keep_vt_mismatch": True,
            "target_mapping_3c": {"0": "hom_ref", "1": "het", "2": "hom_alt"},
            "target_mapping_bin": {"0": "no_variant", "1": "variant"},
            "join_definition": "locus-level late merge",
        },
        "model_policy": {
            "arch": args.arch,
            "head_mode": "groupwise_3c_plus_groupwise_aux_bin",
            "official_inference": "argmax(logits_3c)",
            "auxiliary_head_used_for_inference": False,
            "save_best_on": args.save_best_on,
        },
        "data_summary": {
            "train": summarize_split("train", train_refs),
            "val": summarize_split("val", val_refs),
            "test": summarize_split("test", test_refs),
        },
        "split_source": {
            "split_json": str(Path(args.split_json)),
            "meta_csv": str(meta_csv),
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
            "epoch",
            "train_loss",
            "train_loss_3c",
            "train_loss_aux",
            "val_acc",
            "val_macro_f1",
            "val_weighted_f1",
            "val_variant_recall",
            "val_aux_bin_recall",
            "test_acc",
            "test_macro_f1",
            "test_weighted_f1",
            "test_variant_recall",
            "test_aux_bin_recall",
        ])

    best_score = -1e18
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_loss_3c = 0.0
        total_loss_aux = 0.0
        n_seen = 0

        shard_names = list(train_by_shard.keys())
        rng = random.Random(args.seed + epoch)
        rng.shuffle(shard_names)

        for shard in shard_names:
            items = train_by_shard[shard]
            npz = np.load(shard, allow_pickle=True)

            ill_e = npz["ill_embeddings"].astype(np.float32, copy=False)
            ont_e = npz["ont_embeddings"].astype(np.float32, copy=False)
            mi = npz["mask_ill"].astype(np.float32, copy=False).reshape(-1, 1)
            mo = npz["mask_ont"].astype(np.float32, copy=False).reshape(-1, 1)

            rows = np.array([r.row for r in items], dtype=np.int64)
            ys3 = np.array([r.y3 for r in items], dtype=np.int64)
            yb = np.array([r.y_bin for r in items], dtype=np.float32)
            gs = np.array([r.group for r in items], dtype=np.int64)

            order = list(range(len(rows)))
            rng.shuffle(order)
            rows = rows[order]
            ys3 = ys3[order]
            yb = yb[order]
            gs = gs[order]

            for i in range(0, len(rows), args.batch_size):
                rr = rows[i:i + args.batch_size]
                y3b = ys3[i:i + args.batch_size]
                ybin_b = yb[i:i + args.batch_size]
                gb = gs[i:i + args.batch_size]

                feats = [ill_e[rr], ont_e[rr]]
                if args.add_masks:
                    feats.extend([mi[rr], mo[rr]])
                x_np = np.concatenate(feats, axis=1)

                x = torch.from_numpy(x_np).to(device)
                y3t = torch.from_numpy(y3b).to(device)
                ybt = torch.from_numpy(ybin_b).to(device)
                gt = torch.from_numpy(gb).to(device)

                opt.zero_grad(set_to_none=True)

                logits_3c, logits_bin = model(x, gt)
                loss_main = ce_3c(logits_3c, y3t)
                loss_aux = bce_bin(logits_bin, ybt)
                loss = loss_main + float(args.aux_bin_weight) * loss_aux

                loss.backward()
                opt.step()

                bs = len(rr)
                total_loss += float(loss.detach().cpu()) * bs
                total_loss_3c += float(loss_main.detach().cpu()) * bs
                total_loss_aux += float(loss_aux.detach().cpu()) * bs
                n_seen += bs

            npz.close()

        train_loss = total_loss / max(n_seen, 1)
        train_loss_3c = total_loss_3c / max(n_seen, 1)
        train_loss_aux = total_loss_aux / max(n_seen, 1)

        val_full = eval_sharded_3c_aux(model, val_by_shard, device, args.batch_size, args.add_masks)
        test_full = eval_sharded_3c_aux(model, test_by_shard, device, args.batch_size, args.add_masks)

        val_m = strip_standard_3c_metrics(val_full)
        test_m = strip_standard_3c_metrics(test_full)

        val_aux = build_aux_report(val_full)
        test_aux = build_aux_report(test_full)

        val_variant_recall = float(val_full["official_variant_from_cm3"]["recall"])
        test_variant_recall = float(test_full["official_variant_from_cm3"]["recall"])
        val_aux_bin_recall = float(val_full["aux_bin"]["recall"])
        test_aux_bin_recall = float(test_full["aux_bin"]["recall"])

        score = score_from_val(
            val_std=val_m,
            val_variant_recall=val_variant_recall,
            val_aux_bin_recall=val_aux_bin_recall,
            save_best_on=args.save_best_on,
        )

        print(
            f"[Epoch {epoch:02d}] "
            f"loss={train_loss:.5f} (3c={train_loss_3c:.5f} aux={train_loss_aux:.5f}) | "
            f"VAL acc={val_m['acc']:.4f} macroF1={val_m['macro_f1']:.4f} "
            f"wF1={val_m['weighted_f1']:.4f} varRec={val_variant_recall:.4f} auxRec={val_aux_bin_recall:.4f} | "
            f"TEST acc={test_m['acc']:.4f} macroF1={test_m['macro_f1']:.4f} "
            f"wF1={test_m['weighted_f1']:.4f} varRec={test_variant_recall:.4f} auxRec={test_aux_bin_recall:.4f} | "
            f"best_on={args.save_best_on} score={score:.6f}"
        )

        if csv_writer:
            csv_writer.writerow([
                epoch,
                train_loss,
                train_loss_3c,
                train_loss_aux,
                val_m["acc"],
                val_m["macro_f1"],
                val_m["weighted_f1"],
                val_variant_recall,
                val_aux_bin_recall,
                test_m["acc"],
                test_m["macro_f1"],
                test_m["weighted_f1"],
                test_variant_recall,
                test_aux_bin_recall,
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
            "best_on": args.save_best_on,
        }
        (save_dir / "last_metrics.json").write_text(
            json.dumps(_jsonify(last_payload), indent=2),
            encoding="utf-8",
        )

        last_aux_payload = {
            "epoch": epoch,
            "score": score,
            "best_on": args.save_best_on,
            "train_loss": train_loss,
            "train_loss_3c": train_loss_3c,
            "train_loss_aux": train_loss_aux,
            "val_aux_bin_recall": val_aux_bin_recall,
            "test_aux_bin_recall": test_aux_bin_recall,
            "val": val_aux,
            "test": test_aux,
        }
        (save_dir / "last_aux_metrics.json").write_text(
            json.dumps(_jsonify(last_aux_payload), indent=2),
            encoding="utf-8",
        )

        if score > best_score:
            best_score = float(score)
            best_epoch = epoch

            best_ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "args": vars(args),
                "input_dim": in_dim,
                "n_classes": 3,
                "architecture": "groupwise_3class_aux_binary",
                "best_score": best_score,
                "best_epoch": best_epoch,
                "best_on": args.save_best_on,
                "val": val_m,
                "test": test_m,
                "val_variant_recall": val_variant_recall,
                "test_variant_recall": test_variant_recall,
                "val_aux": val_aux,
                "test_aux": test_aux,
                "val_aux_bin_recall": val_aux_bin_recall,
                "test_aux_bin_recall": test_aux_bin_recall,
            }
            torch.save(best_ckpt, save_dir / "best.pt")

            best_metrics_only = {
                "epoch": epoch,
                "best_score": best_score,
                "best_on": args.save_best_on,
                "val": val_m,
                "test": test_m,
                "val_variant_recall": val_variant_recall,
                "test_variant_recall": test_variant_recall,
            }
            (save_dir / "best_metrics.json").write_text(
                json.dumps(_jsonify(best_metrics_only), indent=2),
                encoding="utf-8",
            )

            best_aux_payload = {
                "epoch": epoch,
                "best_score": best_score,
                "best_on": args.save_best_on,
                "val_aux_bin_recall": val_aux_bin_recall,
                "test_aux_bin_recall": test_aux_bin_recall,
                "val": val_aux,
                "test": test_aux,
            }
            (save_dir / "best_aux_metrics.json").write_text(
                json.dumps(_jsonify(best_aux_payload), indent=2),
                encoding="utf-8",
            )

    final_payload = {
        "epoch": args.epochs,
        "model_state": model.state_dict(),
        "args": vars(args),
        "input_dim": in_dim,
        "n_classes": 3,
        "architecture": "groupwise_3class_aux_binary",
        "best_score": best_score,
        "best_epoch": best_epoch,
        "best_on": args.save_best_on,
    }
    torch.save(final_payload, save_dir / "final.pt")

    if csv_f:
        csv_f.close()

    print(f"[Saved] best.pt / final.pt in {save_dir}")
    print(
        "[Saved] config.json / split_bins_used.json / "
        "best_metrics.json / last_metrics.json / "
        "best_aux_metrics.json / last_aux_metrics.json"
    )


if __name__ == "__main__":
    main()