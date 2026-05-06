#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export OUTER probabilities to CSV with schema:
    locus,cls0,cls1,cls2

Supported sources
-----------------
1) Trained checkpoints (same reconstruction logic as eval_outer_ckpt_with_vt.py)
2) DeepVariant teacher probabilities/logits stored in OUTER shards

Why this script exists
----------------------
It is intended as a bridge between the existing OUTER evaluation setup and a
later postprocess stage that expects a simple CSV with per-locus class
probabilities.

The filtering policy intentionally mirrors the existing evaluation scripts so
that the exported CSV corresponds to the exact locus universe used in prior
OUTER experiments.

Direct examples
---------------
Best hybrid one-stage model:
python training/scripts/hybrid_dv/analysis/export_outer_probs_csv.py \
  --source ckpt \
  --outer_dir data/4_out/datasets/HG003/join_outer_singletons \
  --split_bins_json training/out/splits/outer_chr20_v1/split_bins.json \
  --meta_csv training/out/meta/hg003_chr20_outer.meta_with_vt.csv \
  --ckpt training/out/models/baselines/hybrid_groupwise_mlp/best.pt \
  --mode hybrid --groups 1,10,11 --split test \
  --out_csv training/out/csv_exports/outer_chr20_v1/hybrid_groupwise_mlp_best_test.csv

DeepVariant Illumina teacher:
python training/scripts/hybrid_dv/analysis/export_outer_probs_csv.py \
  --source teacher \
  --outer_dir data/4_out/datasets/HG003/join_outer_singletons \
  --split_bins_json training/out/splits/outer_chr20_v1/split_bins.json \
  --meta_csv training/out/meta/hg003_chr20_outer.meta_with_vt.csv \
  --teacher ill --groups 10,11 --split test \
  --out_csv training/out/csv_exports/outer_chr20_v1/dv_illumina_test.csv

Manifest mode
-------------
python training/scripts/hybrid_dv/analysis/export_outer_probs_csv.py \
  --manifest_json training/out/manifests/outer_chr20_v1/export_test_csvs.example.json
"""

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn


_POS_RE = re.compile(r"(?:chr)?([0-9XYM]+)[\:\-_](\d+)", re.IGNORECASE)


def as_int(x: Any, default: int = -1) -> int:
    try:
        if x is None or x == "":
            return default
        return int(x)
    except Exception:
        return default


def parse_locus_pos(locus: str) -> Optional[int]:
    m = _POS_RE.search(locus)
    if m:
        return int(m.group(2))
    toks = re.split(r"[^0-9]+", locus)
    toks = [t for t in toks if t]
    if toks:
        return int(toks[0])
    return None


def locus_sort_key(locus: str) -> Tuple[int, int, str]:
    m = _POS_RE.search(locus)
    chrom_rank = 10**9
    pos = parse_locus_pos(locus) or 10**18
    if m:
        chrom = m.group(1).upper()
        if chrom.isdigit():
            chrom_rank = int(chrom)
        elif chrom == "X":
            chrom_rank = 23
        elif chrom == "Y":
            chrom_rank = 24
        elif chrom in {"M", "MT"}:
            chrom_rank = 25
    return chrom_rank, pos, locus


def softmax_np(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    logits = logits.astype(np.float64)
    mx = np.max(logits, axis=axis, keepdims=True)
    ex = np.exp(logits - mx)
    return (ex / np.sum(ex, axis=axis, keepdims=True)).astype(np.float32)


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


@dataclass
class Item:
    shard: str
    row: int
    locus: str
    group: int
    bin_id: int
    vt: int
    vt_mismatch: int
    y3: Optional[int] = None


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
            mask = groups == g
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
            mask = groups == g
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


def choose_target_for_mode(mode: str, group: int, label_ill: int, label_ont: int) -> Tuple[Optional[int], str]:
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

    if mode == "hybrid":
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

    return None, "group_not_supported"


def build_items(
    outer_dir: Path,
    splits: Dict[str, Any],
    meta: Dict[str, Dict[str, Any]],
    which_split: str,
    groups_keep: Set[int],
    exclude_vt_mismatch: bool,
    label_policy: str,
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
        loci_list = _as_str_list(npz["locus"])
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

            y3, reason = choose_target_for_mode(
                mode=label_policy,
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

            items.append(Item(
                shard=str(sp),
                row=i,
                locus=locus,
                group=g,
                bin_id=int(b),
                vt=vt,
                vt_mismatch=vt_mismatch,
                y3=int(y3),
            ))
            stats["kept"] += 1

        npz.close()

    return items, stats


@torch.no_grad()
def predict_ckpt_probs(
    items: Sequence[Item],
    mode: str,
    ckpt_path: Path,
    batch_size: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    in_dim = int(ckpt.get("input_dim", 0))
    n_classes = int(ckpt.get("n_classes", 3))
    hidden = int(ckpt_args.get("hidden", 512))
    dropout = float(ckpt_args.get("dropout", 0.1))
    arch = str(ckpt_args.get("arch", "mlp"))
    architecture = str(ckpt.get("architecture", "shared_3class"))
    add_masks_hybrid = resolve_add_masks(ckpt_args)

    if in_dim <= 0:
        first = Path(items[0].shard)
        npz0 = np.load(first, allow_pickle=True)
        if mode == "ill":
            in_dim = int(npz0["ill_embeddings"].shape[1])
        elif mode == "ont":
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
    device = next(model.parameters()).device

    probs = np.zeros((len(items), 3), dtype=np.float32)
    by_shard: Dict[str, List[Tuple[int, Item]]] = {}
    for idx, it in enumerate(items):
        by_shard.setdefault(it.shard, []).append((idx, it))

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
            batch_probs = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
            for (global_idx, _), prob in zip(pairs[i:i + batch_size], batch_probs):
                probs[global_idx] = prob

        npz.close()

    resolved = {
        "checkpoint": str(ckpt_path),
        "architecture": architecture,
        "arch": arch,
        "input_dim": in_dim,
        "n_classes": n_classes,
        "hidden": hidden,
        "dropout": dropout,
        "add_masks_hybrid": add_masks_hybrid,
        "best_epoch": ckpt.get("best_epoch"),
        "best_score": ckpt.get("best_score"),
        "best_on": ckpt.get("best_on"),
        "saved_epoch": ckpt.get("epoch"),
    }
    return probs, resolved


def predict_teacher_probs(items: Sequence[Item], teacher: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    probs = np.zeros((len(items), 3), dtype=np.float32)

    by_shard: Dict[str, List[Tuple[int, Item]]] = {}
    for idx, it in enumerate(items):
        by_shard.setdefault(it.shard, []).append((idx, it))

    for shard, pairs in by_shard.items():
        npz = np.load(shard, allow_pickle=True)
        files = set(npz.files)
        if teacher == "ill":
            if "ill_probs" in files:
                teacher_probs = npz["ill_probs"].astype(np.float32)
                source_field = "ill_probs"
            elif "ill_logits" in files:
                teacher_probs = softmax_np(npz["ill_logits"], axis=-1)
                source_field = "ill_logits"
            else:
                raise SystemExit(f"{shard}: missing ill_probs/ill_logits")
        else:
            if "ont_probs" in files:
                teacher_probs = npz["ont_probs"].astype(np.float32)
                source_field = "ont_probs"
            elif "ont_logits" in files:
                teacher_probs = softmax_np(npz["ont_logits"], axis=-1)
                source_field = "ont_logits"
            else:
                raise SystemExit(f"{shard}: missing ont_probs/ont_logits")

        for global_idx, it in pairs:
            probs[global_idx] = teacher_probs[it.row]
        npz.close()

    resolved = {
        "teacher": teacher,
        "source_field": source_field,
    }
    return probs, resolved


def write_csv(rows: Sequence[Tuple[str, np.ndarray]], out_csv: Path, float_fmt: str = ".8f") -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["locus", "cls0", "cls1", "cls2"])
        for locus, prob in rows:
            w.writerow([
                locus,
                format(float(prob[0]), float_fmt),
                format(float(prob[1]), float_fmt),
                format(float(prob[2]), float_fmt),
            ])


def parse_groups(groups_raw: str) -> Set[int]:
    return set(int(x) for x in str(groups_raw).split(",") if str(x).strip()) if str(groups_raw).strip() else set()


def normalize_job(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(cfg)
    source = str(out.get("source", "")).strip()
    if source not in {"ckpt", "teacher"}:
        raise SystemExit(f"Invalid source='{source}'. Expected 'ckpt' or 'teacher'.")
    out["source"] = source

    split = str(out.get("split", "test")).strip()
    if split not in {"train", "val", "test"}:
        raise SystemExit(f"Invalid split='{split}'. Expected train/val/test.")
    out["split"] = split

    if source == "ckpt":
        mode = str(out.get("mode", "")).strip()
        if mode not in {"ill", "ont", "hybrid"}:
            raise SystemExit(f"Invalid mode='{mode}' for source=ckpt.")
        if not out.get("ckpt"):
            raise SystemExit("source=ckpt requires 'ckpt'.")
        out.setdefault("label_policy", mode)
    else:
        teacher = str(out.get("teacher", "")).strip()
        if teacher not in {"ill", "ont"}:
            raise SystemExit(f"Invalid teacher='{teacher}' for source=teacher.")
        out.setdefault("label_policy", teacher)

    required = ["outer_dir", "split_bins_json", "meta_csv", "out_csv"]
    for k in required:
        if not out.get(k):
            raise SystemExit(f"Missing required key '{k}'.")

    out.setdefault("groups", "")
    out.setdefault("exclude_vt_mismatch", False)
    out.setdefault("batch_size", 4096)
    out.setdefault("sort_by_locus", True)
    out.setdefault("float_fmt", ".8f")
    return out


def run_job(job_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = normalize_job(job_cfg)

    outer_dir = Path(cfg["outer_dir"])
    split_bins_json = Path(cfg["split_bins_json"])
    meta_csv = Path(cfg["meta_csv"])
    out_csv = Path(cfg["out_csv"])
    out_manifest_json = Path(cfg["out_manifest_json"]) if cfg.get("out_manifest_json") else out_csv.with_suffix(".manifest.json")

    splits = load_split_bins(split_bins_json)
    meta = load_meta(meta_csv)
    groups_keep = parse_groups(str(cfg.get("groups", "")))
    bin_size = int(splits["bin_size"])

    items, filter_stats = build_items(
        outer_dir=outer_dir,
        splits=splits,
        meta=meta,
        which_split=str(cfg["split"]),
        groups_keep=groups_keep,
        exclude_vt_mismatch=bool(cfg.get("exclude_vt_mismatch", False)),
        label_policy=str(cfg["label_policy"]),
        bin_size=bin_size,
    )
    if not items:
        raise SystemExit("No items found after filtering. Check split/groups/meta/policy.")

    if cfg["source"] == "ckpt":
        probs, source_info = predict_ckpt_probs(
            items=items,
            mode=str(cfg["mode"]),
            ckpt_path=Path(cfg["ckpt"]),
            batch_size=int(cfg["batch_size"]),
        )
    else:
        probs, source_info = predict_teacher_probs(
            items=items,
            teacher=str(cfg["teacher"]),
        )

    indexed_rows = list(zip(items, probs))
    if bool(cfg.get("sort_by_locus", True)):
        indexed_rows.sort(key=lambda x: locus_sort_key(x[0].locus))

    rows_for_csv = [(it.locus, prob) for it, prob in indexed_rows]
    write_csv(rows_for_csv, out_csv=out_csv, float_fmt=str(cfg.get("float_fmt", ".8f")))

    sidecar = {
        "job_name": cfg.get("name"),
        "source": cfg["source"],
        "split": cfg["split"],
        "groups": sorted(list(groups_keep)) if groups_keep else "ALL",
        "label_policy": cfg["label_policy"],
        "exclude_vt_mismatch": bool(cfg.get("exclude_vt_mismatch", False)),
        "outer_dir": str(outer_dir),
        "split_bins_json": str(split_bins_json),
        "meta_csv": str(meta_csv),
        "bin_size": bin_size,
        "n_rows": len(rows_for_csv),
        "filter_stats": filter_stats,
        "csv_schema": ["locus", "cls0", "cls1", "cls2"],
        "output_csv": str(out_csv),
        "source_info": source_info,
    }
    out_manifest_json.parent.mkdir(parents=True, exist_ok=True)
    out_manifest_json.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

    print(f"[Saved CSV] {out_csv}")
    print(f"[Saved Manifest] {out_manifest_json}")
    print(f"[Rows] {len(rows_for_csv)}")
    return sidecar


def run_manifest(manifest_path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    common = dict(payload.get("common", {}))
    jobs = list(payload.get("jobs", []))
    if not jobs:
        raise SystemExit(f"Manifest has no jobs: {manifest_path}")

    results = []
    for idx, job in enumerate(jobs, start=1):
        merged = dict(common)
        merged.update(job)
        print(f"\n=== Running job {idx}/{len(jobs)}: {merged.get('name', '<unnamed>')} ===")
        results.append(run_job(merged))
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_json", type=str, default=None)

    ap.add_argument("--source", type=str, choices=["ckpt", "teacher"], default=None)
    ap.add_argument("--outer_dir", type=str, default=None)
    ap.add_argument("--split_bins_json", type=str, default=None)
    ap.add_argument("--meta_csv", type=str, default=None)
    ap.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    ap.add_argument("--groups", type=str, default="")
    ap.add_argument("--exclude_vt_mismatch", action="store_true")
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--sort_by_locus", action="store_true")
    ap.add_argument("--no_sort_by_locus", action="store_true")
    ap.add_argument("--float_fmt", type=str, default=".8f")

    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--mode", type=str, choices=["ill", "ont", "hybrid"], default=None)

    ap.add_argument("--teacher", type=str, choices=["ill", "ont"], default=None)

    ap.add_argument("--name", type=str, default=None)
    ap.add_argument("--out_csv", type=str, default=None)
    ap.add_argument("--out_manifest_json", type=str, default=None)
    return ap


def main() -> None:
    ap = build_arg_parser()
    args = ap.parse_args()

    if args.manifest_json:
        run_manifest(Path(args.manifest_json))
        return

    sort_by_locus = True
    if args.no_sort_by_locus:
        sort_by_locus = False
    elif args.sort_by_locus:
        sort_by_locus = True

    cfg = {
        "name": args.name,
        "source": args.source,
        "outer_dir": args.outer_dir,
        "split_bins_json": args.split_bins_json,
        "meta_csv": args.meta_csv,
        "split": args.split,
        "groups": args.groups,
        "exclude_vt_mismatch": bool(args.exclude_vt_mismatch),
        "batch_size": int(args.batch_size),
        "sort_by_locus": sort_by_locus,
        "float_fmt": args.float_fmt,
        "ckpt": args.ckpt,
        "mode": args.mode,
        "teacher": args.teacher,
        "out_csv": args.out_csv,
        "out_manifest_json": args.out_manifest_json,
    }
    run_job(cfg)


if __name__ == "__main__":
    main()
