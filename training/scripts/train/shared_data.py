#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shared dataset loading helpers for the new training pipeline.

Memory-safe version:
- does NOT materialize whole partitions into RAM
- builds shard-level indices
- loads embeddings shard-by-shard only when requested by trainers

Designed for canonical multimodal NPZ datasets produced by:
    data/5_scripts/hybrid_dv/build_join_datasets.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------

@dataclass
class ShardSelection:
    dataset_id: str
    shard_path: str
    rows: np.ndarray
    y: np.ndarray
    group: np.ndarray
    mask_ill: np.ndarray
    mask_ont: np.ndarray
    variant_type: np.ndarray
    variant_type_mismatch: np.ndarray

    @property
    def n(self) -> int:
        return int(self.y.shape[0])


@dataclass
class PartitionIndex:
    name: str
    feature_mode: str
    input_dim: int
    shard_items: List[ShardSelection]
    summary: Dict[str, Any]

    @property
    def n(self) -> int:
        return int(sum(item.n for item in self.shard_items))


# ---------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------

def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"[ERROR] JSON file not found: {path}")
    except json.JSONDecodeError as e:
        raise SystemExit(f"[ERROR] Invalid JSON in {path}: {e}")


def load_resolved_split(path_or_obj: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(path_or_obj, dict):
        payload = path_or_obj
    else:
        payload = load_json(path_or_obj)

    required = ["split_id", "split_strategy", "source_type", "resolved_partitions"]
    missing = [k for k in required if k not in payload]
    if missing:
        raise SystemExit(f"[ERROR] resolved_split missing required keys: {missing}")

    for part in ("train", "val", "test"):
        if part not in payload["resolved_partitions"]:
            raise SystemExit(f"[ERROR] resolved_split missing partition: {part}")

    return payload


# ---------------------------------------------------------------------
# Filesystem / NPZ helpers
# ---------------------------------------------------------------------

def resolve_dataset_dir(dataset_dir: Union[str, Path]) -> Path:
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = REPO_ROOT / dataset_dir
    return dataset_dir.resolve()


def list_npz_shards(dataset_dir: Union[str, Path]) -> List[Path]:
    dataset_dir = resolve_dataset_dir(dataset_dir)
    shards = sorted(dataset_dir.glob("*.npz"))
    if not shards:
        raise SystemExit(f"[ERROR] No NPZ shards found in dataset_dir: {dataset_dir}")
    return shards


def _require_npz_keys(npz: np.lib.npyio.NpzFile, keys: Sequence[str], *, ctx: str) -> None:
    missing = [k for k in keys if k not in npz.files]
    if missing:
        raise SystemExit(f"[ERROR] Missing NPZ keys in {ctx}: {missing}")


def _as_str_array(x: np.ndarray) -> np.ndarray:
    if x.dtype.kind in ("S", "O"):
        return np.asarray(
            [v.decode("utf-8") if isinstance(v, (bytes, np.bytes_)) else str(v) for v in x.tolist()],
            dtype=np.str_,
        )
    return x.astype(np.str_)


# ---------------------------------------------------------------------
# Row selection from resolved split
# ---------------------------------------------------------------------

def _row_mask_scope_partition(npz: np.lib.npyio.NpzFile, chroms: Sequence[str]) -> np.ndarray:
    _require_npz_keys(npz, ["chrom"], ctx="scope_partition")
    chrom_arr = _as_str_array(npz["chrom"])
    keep = np.isin(chrom_arr, np.asarray(list(chroms), dtype=np.str_))
    return keep.astype(bool)


def _row_mask_locus_bins(npz: np.lib.npyio.NpzFile, bins: Sequence[int], bin_size: int) -> np.ndarray:
    _require_npz_keys(npz, ["locus_start"], ctx="locus_bins")
    if bin_size is None or int(bin_size) <= 0:
        raise SystemExit("[ERROR] locus_bins selection requires a positive bin_size.")
    starts = npz["locus_start"].astype(np.int64)
    row_bins = starts // int(bin_size)
    keep = np.isin(row_bins, np.asarray(list(bins), dtype=np.int64))
    return keep.astype(bool)


def _row_mask_chrom_locus_bins(
    npz: np.lib.npyio.NpzFile,
    bins_by_chrom: Dict[str, Sequence[int]],
    bin_size: int,
) -> np.ndarray:
    """Select 1 Mb-style bins while keeping chromosome identity in the key."""
    _require_npz_keys(npz, ["chrom", "locus_start"], ctx="chrom_locus_bins")
    if bin_size is None or int(bin_size) <= 0:
        raise SystemExit("[ERROR] chrom_locus_bins selection requires a positive bin_size.")
    if not isinstance(bins_by_chrom, dict) or not bins_by_chrom:
        raise SystemExit("[ERROR] chrom_locus_bins selection requires bins_by_chrom.")

    chrom_arr = _as_str_array(npz["chrom"])
    starts = npz["locus_start"].astype(np.int64)
    row_bins = starts // int(bin_size)
    keep = np.zeros(chrom_arr.shape[0], dtype=bool)

    for chrom, bins in bins_by_chrom.items():
        if not bins:
            continue
        chrom_keep = chrom_arr == str(chrom)
        bin_keep = np.isin(row_bins, np.asarray(list(bins), dtype=np.int64))
        keep |= chrom_keep & bin_keep

    return keep


def _row_mask_from_entry(npz: np.lib.npyio.NpzFile, entry: Dict[str, Any]) -> np.ndarray:
    if "selection" not in entry:
        raise SystemExit("[ERROR] resolved split entry missing 'selection'.")

    sel = entry["selection"]
    sel_type = sel.get("type", None)

    if sel_type == "scope_partition":
        chroms = sel.get("chroms", entry.get("chroms", None))
        if not chroms:
            raise SystemExit("[ERROR] scope_partition selection requires chroms.")
        return _row_mask_scope_partition(npz, chroms)

    if sel_type == "locus_bins":
        bins = sel.get("bins", None)
        bin_size = sel.get("bin_size", None)
        if bins is None:
            raise SystemExit("[ERROR] locus_bins selection requires bins.")
        return _row_mask_locus_bins(npz, bins=bins, bin_size=int(bin_size))

    if sel_type == "chrom_locus_bins":
        bins_by_chrom = sel.get("bins_by_chrom", None)
        bin_size = sel.get("bin_size", None)
        if bins_by_chrom is None:
            raise SystemExit("[ERROR] chrom_locus_bins selection requires bins_by_chrom.")
        return _row_mask_chrom_locus_bins(
            npz,
            bins_by_chrom=bins_by_chrom,
            bin_size=int(bin_size),
        )

    raise SystemExit(f"[ERROR] Unsupported selection type in resolved split entry: {sel_type}")


# ---------------------------------------------------------------------
# Feature policies
# ---------------------------------------------------------------------

UNIMODAL_POLICIES: Dict[str, Dict[str, Any]] = {
    "illumina": {
        "feature_key": "ill_embeddings",
        "mask_key": "mask_ill",
        "allowed_groups": {10, 11},
    },
    "ont": {
        "feature_key": "ont_embeddings",
        "mask_key": "mask_ont",
        "allowed_groups": {1, 11},
    },
}


# ---------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------

def _init_summary(name: str, input_dim: int) -> Dict[str, Any]:
    return {
        "name": name,
        "n": 0,
        "input_dim": int(input_dim),
        "by_group": defaultdict(int),
        "by_y3": defaultdict(int),
        "by_mask": defaultdict(int),
        "by_vt_mismatch": defaultdict(int),
        "by_variant_type": defaultdict(int),
        "by_chrom": defaultdict(int),
        "by_dataset_id": defaultdict(int),
        "by_dataset_id_and_group": defaultdict(lambda: defaultdict(int)),
        "by_dataset_id_and_mask": defaultdict(lambda: defaultdict(int)),
        "by_dataset_id_and_chrom": defaultdict(lambda: defaultdict(int)),
    }


def _finalize_nested_int_map(m: Dict[Any, Dict[Any, int]]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for outer_k, inner in sorted(m.items(), key=lambda kv: str(kv[0])):
        out[str(outer_k)] = {
            str(inner_k): int(inner_v)
            for inner_k, inner_v in sorted(inner.items(), key=lambda kv: str(kv[0]))
        }
    return out


def _finalize_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": summary["name"],
        "n": int(summary["n"]),
        "input_dim": int(summary["input_dim"]),
        "by_group": dict(sorted((int(k), int(v)) for k, v in summary["by_group"].items())),
        "by_y3": dict(sorted((int(k), int(v)) for k, v in summary["by_y3"].items())),
        "by_mask": dict(sorted((str(k), int(v)) for k, v in summary["by_mask"].items())),
        "by_vt_mismatch": dict(sorted((int(k), int(v)) for k, v in summary["by_vt_mismatch"].items())),
        "by_variant_type": dict(sorted((int(k), int(v)) for k, v in summary["by_variant_type"].items())),
        "by_chrom": dict(sorted((str(k), int(v)) for k, v in summary["by_chrom"].items())),
        "by_dataset_id": dict(sorted((str(k), int(v)) for k, v in summary["by_dataset_id"].items())),
        "by_dataset_id_and_group": _finalize_nested_int_map(summary["by_dataset_id_and_group"]),
        "by_dataset_id_and_mask": _finalize_nested_int_map(summary["by_dataset_id_and_mask"]),
        "by_dataset_id_and_chrom": _finalize_nested_int_map(summary["by_dataset_id_and_chrom"]),
    }


# ---------------------------------------------------------------------
# Internal index builder
# ---------------------------------------------------------------------

def _build_partition_index(
    resolved_split: Dict[str, Any],
    partition_name: str,
    feature_mode: str,
    add_masks: bool = True,
) -> PartitionIndex:
    if partition_name not in ("train", "val", "test"):
        raise SystemExit(f"[ERROR] Invalid partition_name: {partition_name}")

    entries = resolved_split["resolved_partitions"][partition_name]
    if not isinstance(entries, list) or not entries:
        raise SystemExit(f"[ERROR] Partition '{partition_name}' is empty in resolved split.")

    shard_items: List[ShardSelection] = []
    input_dim: int | None = None
    summary: Dict[str, Any] | None = None

    for entry_idx, entry in enumerate(entries):
        dataset_id = str(entry.get("dataset_id", f"entry_{entry_idx}"))
        dataset_path = resolve_dataset_dir(entry["dataset_path"])
        if not dataset_path.exists():
            raise SystemExit(
                f"[ERROR] dataset_path does not exist for partition '{partition_name}' "
                f"entry #{entry_idx}: {dataset_path}"
            )

        shards = list_npz_shards(dataset_path)

        for shard in shards:
            npz = np.load(shard, allow_pickle=False)

            _require_npz_keys(
                npz,
                [
                    "chrom",
                    "locus_start",
                    "label",
                    "group",
                    "mask_ill",
                    "mask_ont",
                    "ill_embeddings",
                    "ont_embeddings",
                ],
                ctx=f"{partition_name} / {shard}",
            )

            base_keep = _row_mask_from_entry(npz, entry)

            label = npz["label"].astype(np.int64)
            group = npz["group"].astype(np.int64)
            mask_ill = npz["mask_ill"].astype(np.int64)
            mask_ont = npz["mask_ont"].astype(np.int64)
            chrom = _as_str_array(npz["chrom"])

            if "variant_type" in npz.files:
                variant_type = npz["variant_type"].astype(np.int64)
            else:
                variant_type = np.full((label.shape[0],), -1, dtype=np.int64)

            if "variant_type_mismatch" in npz.files:
                variant_type_mismatch = npz["variant_type_mismatch"].astype(np.int64)
            else:
                variant_type_mismatch = np.zeros((label.shape[0],), dtype=np.int64)

            if feature_mode.startswith("unimodal:"):
                modality = feature_mode.split(":", 1)[1]
                if modality not in UNIMODAL_POLICIES:
                    raise SystemExit(f"[ERROR] Unsupported modality in feature_mode: {feature_mode}")

                policy = UNIMODAL_POLICIES[modality]
                modality_keep = (npz[policy["mask_key"]].astype(np.int64) == 1)
                group_keep = np.isin(group, np.asarray(sorted(policy["allowed_groups"]), dtype=np.int64))
                keep = base_keep & modality_keep & group_keep

                feature_key = policy["feature_key"]
                shard_input_dim = int(npz[feature_key].shape[1])

            elif feature_mode == "hybrid":
                keep = base_keep & np.isin(group, np.asarray([1, 10, 11], dtype=np.int64))
                shard_input_dim = int(npz["ill_embeddings"].shape[1] + npz["ont_embeddings"].shape[1])
                if add_masks:
                    shard_input_dim += 2

            else:
                raise SystemExit(f"[ERROR] Unsupported feature_mode: {feature_mode}")

            if input_dim is None:
                input_dim = shard_input_dim
                summary = _init_summary(partition_name, input_dim)
            elif input_dim != shard_input_dim:
                raise SystemExit(
                    f"[ERROR] Inconsistent input_dim across shards in partition '{partition_name}': "
                    f"{input_dim} vs {shard_input_dim}"
                )

            rows = np.nonzero(keep)[0].astype(np.int64)
            if rows.size == 0:
                npz.close()
                continue

            y_sel = label[rows].astype(np.int64, copy=True)
            group_sel = group[rows].astype(np.int64, copy=True)
            mask_ill_sel = mask_ill[rows].astype(np.int64, copy=True)
            mask_ont_sel = mask_ont[rows].astype(np.int64, copy=True)
            vt_sel = variant_type[rows].astype(np.int64, copy=True)
            vtm_sel = variant_type_mismatch[rows].astype(np.int64, copy=True)
            chrom_sel = chrom[rows]

            shard_items.append(
                ShardSelection(
                    dataset_id=dataset_id,
                    shard_path=str(shard),
                    rows=rows,
                    y=y_sel,
                    group=group_sel,
                    mask_ill=mask_ill_sel,
                    mask_ont=mask_ont_sel,
                    variant_type=vt_sel,
                    variant_type_mismatch=vtm_sel,
                )
            )

            assert summary is not None
            summary["n"] += int(rows.size)
            summary["by_dataset_id"][dataset_id] += int(rows.size)

            for g in group_sel.tolist():
                summary["by_group"][int(g)] += 1
                summary["by_dataset_id_and_group"][dataset_id][int(g)] += 1

            for y in y_sel.tolist():
                summary["by_y3"][int(y)] += 1

            for mi, mo in zip(mask_ill_sel.tolist(), mask_ont_sel.tolist()):
                mk = f"{int(mi)}{int(mo)}"
                summary["by_mask"][mk] += 1
                summary["by_dataset_id_and_mask"][dataset_id][mk] += 1

            for vtm in vtm_sel.tolist():
                summary["by_vt_mismatch"][int(vtm)] += 1

            for vt in vt_sel.tolist():
                summary["by_variant_type"][int(vt)] += 1

            for ch in chrom_sel.tolist():
                chs = str(ch)
                summary["by_chrom"][chs] += 1
                summary["by_dataset_id_and_chrom"][dataset_id][chs] += 1

            npz.close()

    if input_dim is None or summary is None or not shard_items:
        raise SystemExit(f"[ERROR] Partition '{partition_name}' resolved to zero rows.")

    return PartitionIndex(
        name=partition_name,
        feature_mode=feature_mode,
        input_dim=int(input_dim),
        shard_items=shard_items,
        summary=_finalize_summary(summary),
    )


# ---------------------------------------------------------------------
# Public index builders
# ---------------------------------------------------------------------

def build_partition_index_unimodal(
    resolved_split_path_or_obj: Union[str, Path, Dict[str, Any]],
    partition_name: str,
    modality: str,
) -> PartitionIndex:
    resolved = load_resolved_split(resolved_split_path_or_obj)
    modality = str(modality).lower()
    if modality not in UNIMODAL_POLICIES:
        raise SystemExit(
            f"[ERROR] Unsupported unimodal modality: {modality}. "
            f"Allowed: {sorted(UNIMODAL_POLICIES.keys())}"
        )
    return _build_partition_index(
        resolved_split=resolved,
        partition_name=partition_name,
        feature_mode=f"unimodal:{modality}",
        add_masks=False,
    )


def build_partition_index_hybrid(
    resolved_split_path_or_obj: Union[str, Path, Dict[str, Any]],
    partition_name: str,
    add_masks: bool = True,
) -> PartitionIndex:
    resolved = load_resolved_split(resolved_split_path_or_obj)
    return _build_partition_index(
        resolved_split=resolved,
        partition_name=partition_name,
        feature_mode="hybrid",
        add_masks=add_masks,
    )


# ---------------------------------------------------------------------
# Shard feature loaders (used by trainers)
# ---------------------------------------------------------------------

def load_unimodal_shard_features(
    shard_item: ShardSelection,
    modality: str,
) -> tuple[np.ndarray, np.ndarray]:
    modality = str(modality).lower()
    if modality not in UNIMODAL_POLICIES:
        raise SystemExit(
            f"[ERROR] Unsupported unimodal modality: {modality}. "
            f"Allowed: {sorted(UNIMODAL_POLICIES.keys())}"
        )

    feature_key = UNIMODAL_POLICIES[modality]["feature_key"]

    npz = np.load(shard_item.shard_path, allow_pickle=False)
    x = npz[feature_key].astype(np.float32, copy=False)[shard_item.rows]
    y = shard_item.y.astype(np.int64, copy=False)
    npz.close()

    return x, y


def load_hybrid_shard_features(
    shard_item: ShardSelection,
    add_masks: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    npz = np.load(shard_item.shard_path, allow_pickle=False)
    ill = npz["ill_embeddings"].astype(np.float32, copy=False)[shard_item.rows]
    ont = npz["ont_embeddings"].astype(np.float32, copy=False)[shard_item.rows]
    npz.close()

    x = np.concatenate([ill, ont], axis=1)

    if add_masks:
        masks = np.stack(
            [
                shard_item.mask_ill.astype(np.float32, copy=False),
                shard_item.mask_ont.astype(np.float32, copy=False),
            ],
            axis=1,
        )
        x = np.concatenate([x, masks], axis=1)

    y = shard_item.y.astype(np.int64, copy=False)
    g = shard_item.group.astype(np.int64, copy=False)

    return x, y, g


# ---------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------

def summarize_partition(index: PartitionIndex) -> Dict[str, Any]:
    return index.summary


def print_partition_summary(summary: Dict[str, Any]) -> None:
    n = max(int(summary["n"]), 1)

    def pct(x: int) -> str:
        return f"{x:,} ({100.0 * float(x) / float(n):.1f}%)"

    print(f"\n[PartitionSummary] {summary['name']} n={summary['n']:,} input_dim={summary['input_dim']}")
    print("  by_group:               ", {k: pct(v) for k, v in summary["by_group"].items()})
    print("  by_y3:                  ", {k: pct(v) for k, v in summary["by_y3"].items()})
    print("  by_mask:                ", {k: pct(v) for k, v in summary["by_mask"].items()})
    print("  by_vt_mismatch:         ", {k: pct(v) for k, v in summary["by_vt_mismatch"].items()})
    print("  by_variant_type:        ", {k: pct(v) for k, v in summary["by_variant_type"].items()})
    print("  by_chrom:               ", {k: pct(v) for k, v in summary["by_chrom"].items()})
    print("  by_dataset_id:          ", {k: pct(v) for k, v in summary["by_dataset_id"].items()})

    print("\n  by_dataset_id_and_group:")
    for ds, inner in summary["by_dataset_id_and_group"].items():
        print(f"    - {ds}: {inner}")

    print("\n  by_dataset_id_and_mask:")
    for ds, inner in summary["by_dataset_id_and_mask"].items():
        print(f"    - {ds}: {inner}")

    print("\n  by_dataset_id_and_chrom:")
    for ds, inner in summary["by_dataset_id_and_chrom"].items():
        print(f"    - {ds}: {inner}")