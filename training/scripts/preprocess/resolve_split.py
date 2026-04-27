#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Resolve a training split config into a normalized, ready-to-consume JSON manifest.

Supported split strategies:
- scope_partition
- locus_bins

Typical usage:
    python training/scripts/preprocess/resolve_split.py \
      --split_config training/configs/splits/hg002_hg003_train__hg004_valtest.json

    python training/scripts/preprocess/resolve_split.py \
      --split_config training/configs/splits/legacy_hg003_chr20_bins.json \
      --output_json training/out/experiments/debug_legacy/resolved_split.json

Notes:
- For scope_partition, the resolver preserves chroms as declared in the config.
- For locus_bins, the resolver loads the historical split_bins.json and exposes:
    - train_bins / val_bins / test_bins at top level
    - a normalized resolved_partitions structure
- Dataset path derivation is intentionally lightweight and convention-based.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"[ERROR] JSON file not found: {path}")
    except json.JSONDecodeError as e:
        raise SystemExit(f"[ERROR] Invalid JSON in {path}: {e}")


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def find_repo_root(start: Path) -> Path:
    """
    Heuristic repo root detection:
    walk upwards until a folder containing both 'training' and 'data' is found.
    """
    candidates = [start] + list(start.parents)
    for c in candidates:
        if (c / "training").exists() and (c / "data").exists():
            return c

    script_here = Path(__file__).resolve()
    candidates = [script_here.parent] + list(script_here.parent.parents)
    for c in candidates:
        if (c / "training").exists() and (c / "data").exists():
            return c

    raise SystemExit(
        "[ERROR] Could not infer repo root. Run this from inside the repo, "
        "or pass --repo_root explicitly."
    )


def resolve_path(path_str: str, *, split_config_path: Path, repo_root: Path) -> Path:
    """
    Resolve a path with the following precedence:
    1) absolute path
    2) relative to split config directory
    3) relative to repo root
    """
    raw = Path(path_str)
    if raw.is_absolute():
        return raw

    cand1 = (split_config_path.parent / raw).resolve()
    if cand1.exists():
        return cand1

    cand2 = (repo_root / raw).resolve()
    if cand2.exists():
        return cand2

    # Return repo-root-relative fallback even if it does not exist,
    # so the caller can still inspect the intended location.
    return cand2


def maybe_dataset_path_multimodal(repo_root: Path, dataset_id: str) -> Tuple[Path, bool]:
    """
    Convention-based path resolution for multimodal datasets.

    Preferred:
      data/4_out/datasets/multimodal/by_subject/<dataset_id>/outer

    Fallback:
      data/4_out/datasets/multimodal/<dataset_id>/outer
    """
    p1 = repo_root / "data" / "4_out" / "datasets" / "multimodal" / "by_subject" / dataset_id / "outer"
    if p1.exists():
        return p1.resolve(), True

    p2 = repo_root / "data" / "4_out" / "datasets" / "multimodal" / dataset_id / "outer"
    if p2.exists():
        return p2.resolve(), True

    # Return preferred path even if missing
    return p1.resolve(), False


def ensure_keys(d: Dict[str, Any], keys: List[str], *, ctx: str) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise SystemExit(f"[ERROR] Missing required keys in {ctx}: {missing}")


def ensure_partition_names(parts: Dict[str, Any], *, ctx: str) -> None:
    allowed = {"train", "val", "test"}
    got = set(parts.keys())
    missing = allowed - got
    extra = got - allowed
    if missing:
        raise SystemExit(f"[ERROR] Missing partitions in {ctx}: {sorted(missing)}")
    if extra:
        raise SystemExit(f"[ERROR] Unexpected partitions in {ctx}: {sorted(extra)}")


# ---------------------------------------------------------------------
# Validation + resolution
# ---------------------------------------------------------------------

@dataclass
class ResolveContext:
    repo_root: Path
    split_config_path: Path
    split_payload: Dict[str, Any]


def resolve_scope_partition(ctx: ResolveContext) -> Dict[str, Any]:
    payload = ctx.split_payload
    ensure_keys(
        payload,
        ["split_id", "split_strategy", "source_type", "partitions"],
        ctx="scope_partition split config",
    )
    ensure_partition_names(payload["partitions"], ctx="scope_partition.partitions")

    source_type = str(payload["source_type"])
    if source_type != "multimodal":
        raise SystemExit(
            f"[ERROR] scope_partition currently only supports source_type='multimodal'. "
            f"Got: {source_type}"
        )

    resolved_parts: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}

    seen_records = set()

    for partition_name in ("train", "val", "test"):
        items = payload["partitions"][partition_name]
        if not isinstance(items, list):
            raise SystemExit(f"[ERROR] Partition '{partition_name}' must be a list.")

        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                raise SystemExit(
                    f"[ERROR] Partition '{partition_name}' entry #{idx} must be an object."
                )

            ensure_keys(
                item,
                ["dataset_id", "subject", "chroms", "coverage"],
                ctx=f"{partition_name}[{idx}]",
            )

            dataset_id = str(item["dataset_id"])
            subject = str(item["subject"])
            coverage = str(item["coverage"])
            chroms = item["chroms"]

            if not isinstance(chroms, list) or not chroms or not all(isinstance(c, str) for c in chroms):
                raise SystemExit(
                    f"[ERROR] {partition_name}[{idx}].chroms must be a non-empty list[str]."
                )

            dataset_path, exists = maybe_dataset_path_multimodal(ctx.repo_root, dataset_id)

            # Duplicate guard at the declared scope level
            dup_key = (partition_name, dataset_id, tuple(chroms), coverage)
            if dup_key in seen_records:
                raise SystemExit(
                    f"[ERROR] Duplicate split entry detected in partition '{partition_name}': "
                    f"{dataset_id} / {chroms} / {coverage}"
                )
            seen_records.add(dup_key)

            resolved_parts[partition_name].append(
                {
                    "partition": partition_name,
                    "dataset_id": dataset_id,
                    "dataset_path": str(dataset_path),
                    "dataset_path_exists": exists,
                    "subject": subject,
                    "chroms": chroms,
                    "coverage": coverage,
                    "selection": {
                        "type": "scope_partition",
                        "chroms": chroms,
                    },
                }
            )

    return {
        "split_id": payload["split_id"],
        "split_strategy": payload["split_strategy"],
        "source_type": payload["source_type"],
        "description": payload.get("description", ""),
        "resolved_at": utc_now_iso(),
        "repo_root": str(ctx.repo_root),
        "split_config_path": str(ctx.split_config_path.resolve()),
        "resolved_partitions": resolved_parts,
        "summary": {
            "n_train_entries": len(resolved_parts["train"]),
            "n_val_entries": len(resolved_parts["val"]),
            "n_test_entries": len(resolved_parts["test"]),
        },
    }


def resolve_locus_bins(ctx: ResolveContext) -> Dict[str, Any]:
    payload = ctx.split_payload
    ensure_keys(
        payload,
        ["split_id", "split_strategy", "source_type", "dataset", "bin_source", "partitions"],
        ctx="locus_bins split config",
    )
    ensure_partition_names(payload["partitions"], ctx="locus_bins.partitions")

    source_type = str(payload["source_type"])
    if source_type != "multimodal":
        raise SystemExit(
            f"[ERROR] locus_bins currently only supports source_type='multimodal'. "
            f"Got: {source_type}"
        )

    dataset = payload["dataset"]
    ensure_keys(dataset, ["dataset_id", "subject", "chroms"], ctx="locus_bins.dataset")

    dataset_id = str(dataset["dataset_id"])
    subject = str(dataset["subject"])
    chroms = dataset["chroms"]
    if not isinstance(chroms, list) or not chroms or not all(isinstance(c, str) for c in chroms):
        raise SystemExit("[ERROR] locus_bins.dataset.chroms must be a non-empty list[str].")

    dataset_path, dataset_exists = maybe_dataset_path_multimodal(ctx.repo_root, dataset_id)

    bin_source = payload["bin_source"]
    ensure_keys(bin_source, ["format", "path"], ctx="locus_bins.bin_source")

    fmt = str(bin_source["format"]).lower()
    if fmt != "json":
        raise SystemExit(f"[ERROR] Unsupported locus_bins bin_source format: {fmt}")

    bin_path = resolve_path(
        str(bin_source["path"]),
        split_config_path=ctx.split_config_path,
        repo_root=ctx.repo_root,
    )
    if not bin_path.exists():
        raise SystemExit(f"[ERROR] locus_bins bin_source not found: {bin_path}")

    legacy = load_json(bin_path)
    ensure_keys(
        legacy,
        ["train_bins", "val_bins", "test_bins"],
        ctx=f"legacy bin_source payload ({bin_path})",
    )

    train_bins = sorted(list(set(int(x) for x in legacy["train_bins"])))
    val_bins = sorted(list(set(int(x) for x in legacy["val_bins"])))
    test_bins = sorted(list(set(int(x) for x in legacy["test_bins"])))

    # Validate requested aliases
    alias_map = {
        "train": train_bins,
        "val": val_bins,
        "test": test_bins,
    }
    resolved_parts: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}

    for partition_name in ("train", "val", "test"):
        aliases = payload["partitions"][partition_name]
        if not isinstance(aliases, list) or not aliases:
            raise SystemExit(
                f"[ERROR] locus_bins partition '{partition_name}' must be a non-empty list[str]."
            )

        selected_bins: List[int] = []
        for alias in aliases:
            if alias not in alias_map:
                raise SystemExit(
                    f"[ERROR] Unknown bin alias '{alias}' in partition '{partition_name}'. "
                    f"Allowed: train, val, test"
                )
            selected_bins.extend(alias_map[alias])

        selected_bins = sorted(list(set(int(x) for x in selected_bins)))

        resolved_parts[partition_name].append(
            {
                "partition": partition_name,
                "dataset_id": dataset_id,
                "dataset_path": str(dataset_path),
                "dataset_path_exists": dataset_exists,
                "subject": subject,
                "chroms": chroms,
                "selection": {
                    "type": "locus_bins",
                    "bin_source_path": str(bin_path.resolve()),
                    "bin_size": legacy.get("bin_size"),
                    "bins": selected_bins,
                },
            }
        )

    return {
        "split_id": payload["split_id"],
        "split_strategy": payload["split_strategy"],
        "source_type": payload["source_type"],
        "description": payload.get("description", ""),
        "resolved_at": utc_now_iso(),
        "repo_root": str(ctx.repo_root),
        "split_config_path": str(ctx.split_config_path.resolve()),
        "dataset": {
            "dataset_id": dataset_id,
            "dataset_path": str(dataset_path),
            "dataset_path_exists": dataset_exists,
            "subject": subject,
            "chroms": chroms,
        },
        "bin_source": {
            "format": fmt,
            "path": str(bin_path.resolve()),
        },
        "bin_size": legacy.get("bin_size"),
        "buffer_bins": legacy.get("buffer_bins"),
        "train_bins": train_bins,
        "val_bins": val_bins,
        "test_bins": test_bins,
        "dropped_bins": legacy.get("dropped_bins", []),
        "resolved_partitions": resolved_parts,
        "summary": {
            "n_train_bins": len(train_bins),
            "n_val_bins": len(val_bins),
            "n_test_bins": len(test_bins),
            "n_dropped_bins": len(legacy.get("dropped_bins", [])),
        },
    }


# ---------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------

def print_scope_summary(resolved: Dict[str, Any]) -> None:
    print("\n[ResolvedSplit]")
    print(f"  split_id       : {resolved['split_id']}")
    print(f"  strategy       : {resolved['split_strategy']}")
    print(f"  source_type    : {resolved['source_type']}")
    print(f"  config         : {resolved['split_config_path']}")
    print(f"  resolved_at    : {resolved['resolved_at']}")

    print("\n[Partitions]")
    for part in ("train", "val", "test"):
        rows = resolved["resolved_partitions"][part]
        print(f"  - {part}: {len(rows)} entr{'y' if len(rows) == 1 else 'ies'}")
        for row in rows:
            chroms = ",".join(row["chroms"])
            exists = "yes" if row["dataset_path_exists"] else "no"
            print(
                f"      dataset_id={row['dataset_id']} "
                f"subject={row['subject']} "
                f"chroms=[{chroms}] "
                f"coverage={row.get('coverage', 'NA')} "
                f"path_exists={exists}"
            )


def print_locus_bins_summary(resolved: Dict[str, Any]) -> None:
    print("\n[ResolvedSplit]")
    print(f"  split_id       : {resolved['split_id']}")
    print(f"  strategy       : {resolved['split_strategy']}")
    print(f"  source_type    : {resolved['source_type']}")
    print(f"  config         : {resolved['split_config_path']}")
    print(f"  resolved_at    : {resolved['resolved_at']}")

    ds = resolved["dataset"]
    exists = "yes" if ds["dataset_path_exists"] else "no"

    print("\n[Dataset]")
    print(f"  dataset_id     : {ds['dataset_id']}")
    print(f"  subject        : {ds['subject']}")
    print(f"  chroms         : {ds['chroms']}")
    print(f"  path_exists    : {exists}")

    print("\n[BinSource]")
    print(f"  path           : {resolved['bin_source']['path']}")
    print(f"  bin_size       : {resolved.get('bin_size')}")
    print(f"  buffer_bins    : {resolved.get('buffer_bins')}")

    print("\n[Partitions]")
    print(f"  train bins     : {len(resolved['train_bins'])}")
    print(f"  val bins       : {len(resolved['val_bins'])}")
    print(f"  test bins      : {len(resolved['test_bins'])}")
    print(f"  dropped bins   : {len(resolved.get('dropped_bins', []))}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--split_config",
        type=str,
        required=True,
        help="Path to split config JSON under training/configs/splits/.",
    )
    ap.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional output path for resolved split JSON. "
             "Default: sibling file next to split config named resolved__<split_id>.json",
    )
    ap.add_argument(
        "--repo_root",
        type=str,
        default=None,
        help="Optional explicit repo root. If omitted, it is auto-detected.",
    )
    args = ap.parse_args()

    split_config_path = Path(args.split_config).resolve()
    if not split_config_path.exists():
        raise SystemExit(f"[ERROR] split_config not found: {split_config_path}")

    repo_root = Path(args.repo_root).resolve() if args.repo_root else find_repo_root(Path.cwd())
    payload = load_json(split_config_path)

    ensure_keys(payload, ["split_id", "split_strategy"], ctx="top-level split config")
    strategy = str(payload["split_strategy"])

    ctx = ResolveContext(
        repo_root=repo_root,
        split_config_path=split_config_path,
        split_payload=payload,
    )

    if strategy == "scope_partition":
        resolved = resolve_scope_partition(ctx)
        printer = print_scope_summary
    elif strategy == "locus_bins":
        resolved = resolve_locus_bins(ctx)
        printer = print_locus_bins_summary
    else:
        raise SystemExit(
            f"[ERROR] Unsupported split_strategy: {strategy}. "
            f"Allowed: scope_partition, locus_bins"
        )

    if args.output_json:
        output_json = Path(args.output_json).resolve()
    else:
        output_json = split_config_path.parent / f"resolved__{payload['split_id']}.json"

    dump_json(output_json, resolved)

    printer(resolved)

    print("\n[Saved]")
    print(f"  {output_json.resolve()}")


if __name__ == "__main__":
    main()