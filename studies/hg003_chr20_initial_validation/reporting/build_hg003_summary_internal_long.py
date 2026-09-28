#!/usr/bin/env python3
"""
build_hg003_summary_internal_long.py

Build an HG003 summary CSV with the same column structure as the newer
HG004/HG005 `summary_internal_test_long.csv` files.

Supports:
  1) Legacy single-view JSONs with top-level overall_3c/by_variant_type/by_group.
  2) Merged multi-view JSONs with metrics.by_scope[illumina_view|ont_view|hybrid_all].

For merged hybrid files, view-specific SNP/INDEL metrics are read from
metrics.by_scope, not reconstructed from the global block.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_FILES = [
    "dv_illumina_test.json",
    "dv_ont_test.json",
    "unimodal_illumina_linear_test.json",
    "unimodal_illumina_mlp_test.json",
    "unimodal_ont_linear_test.json",
    "unimodal_ont_mlp_test.json",
    "hybrid_simple_linear_test.json",
    "hybrid_simple_mlp_test.json",
    "hybrid_groupwise_linear_test.json",
    "hybrid_groupwise_mlp_test.json",
]

# Keep this registry explicit so the output is stable and does not depend on filenames only.
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "dv_illumina_test.json": {
        "source": "teacher",
        "model_kind": "teacher",
        "modality": "illumina",
        "model_family": "",
        "name": "dv_illumina",
        "order": 10,
    },
    "dv_ont_test.json": {
        "source": "teacher",
        "model_kind": "teacher",
        "modality": "ont",
        "model_family": "",
        "name": "dv_ont",
        "order": 20,
    },
    "unimodal_illumina_linear_test.json": {
        "source": "checkpoint",
        "model_kind": "unimodal",
        "modality": "illumina",
        "model_family": "unimodal_linear_3c",
        "name": "unimodal_illumina_linear",
        "order": 30,
    },
    "unimodal_illumina_mlp_test.json": {
        "source": "checkpoint",
        "model_kind": "unimodal",
        "modality": "illumina",
        "model_family": "unimodal_mlp_3c",
        "name": "unimodal_illumina_mlp",
        "order": 40,
    },
    "unimodal_ont_linear_test.json": {
        "source": "checkpoint",
        "model_kind": "unimodal",
        "modality": "ont",
        "model_family": "unimodal_linear_3c",
        "name": "unimodal_ont_linear",
        "order": 50,
    },
    "unimodal_ont_mlp_test.json": {
        "source": "checkpoint",
        "model_kind": "unimodal",
        "modality": "ont",
        "model_family": "unimodal_mlp_3c",
        "name": "unimodal_ont_mlp",
        "order": 60,
    },
    "hybrid_simple_linear_test.json": {
        "source": "checkpoint",
        "model_kind": "hybrid",
        "modality": "hybrid",
        "model_family": "hybrid_simple_linear_3c",
        "name": "hybrid_simple_linear",
        "order": 70,
    },
    "hybrid_simple_mlp_test.json": {
        "source": "checkpoint",
        "model_kind": "hybrid",
        "modality": "hybrid",
        "model_family": "hybrid_simple_mlp_3c",
        "name": "hybrid_simple_mlp",
        "order": 80,
    },
    "hybrid_groupwise_linear_test.json": {
        "source": "checkpoint",
        "model_kind": "hybrid",
        "modality": "hybrid",
        "model_family": "hybrid_groupwise_linear_3c",
        "name": "hybrid_groupwise_linear",
        "order": 90,
    },
    "hybrid_groupwise_mlp_test.json": {
        "source": "checkpoint",
        "model_kind": "hybrid",
        "modality": "hybrid",
        "model_family": "hybrid_groupwise_mlp_3c",
        "name": "hybrid_groupwise_mlp",
        "order": 100,
    },
}

FIELDNAMES = [
    "source",
    "model_kind",
    "modality",
    "model_family",
    "name",
    "partition",
    "scope",
    "scope_type",
    "comparable_scope",
    "groups",
    "variant_type",
    "n",
    "acc",
    "macro_f1",
    "macro_recall",
    "macro_precision",
    "weighted_f1",
    "variant_precision",
    "variant_recall",
    "variant_specificity",
    "variant_balanced_acc",
    "variant_f1",
    "variant_f2",
    "variant_tp",
    "variant_tn",
    "variant_fp",
    "variant_fn",
    "binary_acc",
    "binary_balanced_acc",
    "binary_precision",
    "binary_recall",
    "binary_specificity",
    "binary_f1",
    "binary_f2",
]

SCOPE_FROM_GROUPS = {
    (10, 11): "illumina_view",
    (1, 11): "ont_view",
    (1, 10, 11): "hybrid_all",
}

SCOPE_ORDER = {
    "hybrid_all": 1,
    "illumina_view": 2,
    "ont_view": 3,
}

SCOPE_TYPE_ORDER = {
    "view": 1,
    "view_variant_type": 2,
    "group": 3,
    "variant_type": 4,
}

VARIANT_ORDER = {"ALL": 0, "SNP": 1, "INDEL": 2}


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def as_groups(groups_keep: Optional[List[int]]) -> str:
    if not groups_keep:
        return ""
    return ",".join(str(g) for g in groups_keep)


def scope_from_groups(groups_keep: Optional[List[int]]) -> str:
    return SCOPE_FROM_GROUPS.get(tuple(groups_keep or []), "unknown")


def binary_counts(binary_block: Optional[Dict[str, Any]]) -> Dict[str, Optional[int]]:
    if not binary_block:
        return {"tp": None, "tn": None, "fp": None, "fn": None}

    out = {
        "tp": binary_block.get("tp"),
        "tn": binary_block.get("tn"),
        "fp": binary_block.get("fp"),
        "fn": binary_block.get("fn"),
    }
    if all(v is not None for v in out.values()):
        return out

    cm = binary_block.get("cm")
    if (
        isinstance(cm, list)
        and len(cm) == 2
        and all(isinstance(row, list) and len(row) == 2 for row in cm)
    ):
        tn, fp = cm[0]
        fn, tp = cm[1]
        return {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)}

    return out


def balanced_acc(recall: Any, specificity: Any) -> Optional[float]:
    if recall is None or specificity is None:
        return None
    return (float(recall) + float(specificity)) / 2.0


def f_beta_2(precision: Any, recall: Any) -> Optional[float]:
    if precision is None or recall is None:
        return None
    p = float(precision)
    r = float(recall)
    denom = 4.0 * p + r
    return 0.0 if denom == 0 else 5.0 * p * r / denom


def make_row(
    *,
    registry: Dict[str, Any],
    partition: str,
    scope: str,
    scope_type: str,
    groups: str,
    variant_type: str,
    metrics_3c: Optional[Dict[str, Any]],
    binary: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    metrics_3c = metrics_3c or {}
    binary = binary or {}
    counts = binary_counts(binary)

    b_precision = binary.get("precision")
    b_recall = binary.get("recall")
    b_specificity = binary.get("specificity")
    b_balanced_acc = binary.get("balanced_acc")
    if b_balanced_acc is None:
        b_balanced_acc = balanced_acc(b_recall, b_specificity)

    b_f2 = binary.get("f2")
    if b_f2 is None:
        b_f2 = f_beta_2(b_precision, b_recall)

    return {
        "source": registry["source"],
        "model_kind": registry["model_kind"],
        "modality": registry["modality"],
        "model_family": registry["model_family"],
        "name": registry["name"],
        "partition": partition,
        "scope": scope,
        "scope_type": scope_type,
        "comparable_scope": scope,
        "groups": groups,
        "variant_type": variant_type,
        "n": metrics_3c.get("n", binary.get("n")),
        "acc": metrics_3c.get("acc"),
        "macro_f1": metrics_3c.get("macro_f1"),
        "macro_recall": metrics_3c.get("macro_recall"),
        "macro_precision": metrics_3c.get("macro_precision"),
        "weighted_f1": metrics_3c.get("weighted_f1"),
        "variant_precision": b_precision,
        "variant_recall": b_recall,
        "variant_specificity": b_specificity,
        "variant_balanced_acc": b_balanced_acc,
        "variant_f1": binary.get("f1"),
        "variant_f2": b_f2,
        "variant_tp": counts.get("tp"),
        "variant_tn": counts.get("tn"),
        "variant_fp": counts.get("fp"),
        "variant_fn": counts.get("fn"),
        "binary_acc": binary.get("acc"),
        "binary_balanced_acc": b_balanced_acc,
        "binary_precision": b_precision,
        "binary_recall": b_recall,
        "binary_specificity": b_specificity,
        "binary_f1": binary.get("f1"),
        "binary_f2": b_f2,
    }


def rows_from_eval_block(
    *,
    block: Dict[str, Any],
    registry: Dict[str, Any],
    partition: str,
    scope: str,
    groups: str,
    include_view_rows: bool,
    include_group_rows: bool,
    include_variant_type_rows: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    if include_view_rows:
        rows.append(
            make_row(
                registry=registry,
                partition=partition,
                scope=scope,
                scope_type="view",
                groups=groups,
                variant_type="ALL",
                metrics_3c=block.get("overall_3c"),
                binary=block.get("overall_binary_variant_vs_no")
                or block.get("overall_variant_from_cm3"),
            )
        )

        for vt in ["SNP", "INDEL"]:
            vt_block = (block.get("by_variant_type") or {}).get(vt)
            if vt_block is None:
                continue
            rows.append(
                make_row(
                    registry=registry,
                    partition=partition,
                    scope=f"{scope}_{vt}",
                    scope_type="view_variant_type",
                    groups=groups,
                    variant_type=vt,
                    metrics_3c=vt_block,
                    binary=vt_block.get("binary_variant_vs_no")
                    or vt_block.get("variant_from_cm3"),
                )
            )

    if include_group_rows:
        for group_key in sorted((block.get("by_group") or {}).keys(), key=lambda x: int(x)):
            group_block = block["by_group"][group_key]
            rows.append(
                make_row(
                    registry=registry,
                    partition=partition,
                    scope=f"group_{group_key}",
                    scope_type="group",
                    groups=str(group_key),
                    variant_type="ALL",
                    metrics_3c=group_block,
                    binary=group_block.get("binary_variant_vs_no")
                    or group_block.get("variant_from_cm3"),
                )
            )

    if include_variant_type_rows:
        for vt in ["SNP", "INDEL"]:
            vt_block = (block.get("by_variant_type") or {}).get(vt)
            if vt_block is None:
                continue
            rows.append(
                make_row(
                    registry=registry,
                    partition=partition,
                    scope=f"variant_{vt}",
                    scope_type="variant_type",
                    groups="ALL",
                    variant_type=vt,
                    metrics_3c=vt_block,
                    binary=vt_block.get("binary_variant_vs_no")
                    or vt_block.get("variant_from_cm3"),
                )
            )

    return rows


def rows_from_legacy(path: Path, data: Dict[str, Any], registry: Dict[str, Any]) -> List[Dict[str, Any]]:
    groups_keep = data.get("groups_keep")
    scope = scope_from_groups(groups_keep)
    groups = as_groups(groups_keep)
    partition = data.get("split", "test")
    return rows_from_eval_block(
        block=data,
        registry=registry,
        partition=partition,
        scope=scope,
        groups=groups,
        include_view_rows=True,
        include_group_rows=True,
        include_variant_type_rows=True,
    )


def rows_from_merged(path: Path, data: Dict[str, Any], registry: Dict[str, Any]) -> List[Dict[str, Any]]:
    metrics = data.get("metrics") or {}
    by_scope = metrics.get("by_scope") or {}
    partition = data.get("partition", "test")

    rows: List[Dict[str, Any]] = []

    # View rows and view_variant_type rows from each explicit scope.
    for scope in ["hybrid_all", "illumina_view", "ont_view"]:
        if scope not in by_scope:
            continue
        scope_block = by_scope[scope]
        rows.extend(
            rows_from_eval_block(
                block=scope_block,
                registry=registry,
                partition=partition,
                scope=scope,
                groups=as_groups(scope_block.get("groups_keep")),
                include_view_rows=True,
                include_group_rows=False,
                include_variant_type_rows=False,
            )
        )

    # Group and generic variant_type rows from the top-level metrics block,
    # matching the newer HG004/HG005 summary layout.
    rows.extend(
        rows_from_eval_block(
            block=metrics,
            registry=registry,
            partition=partition,
            scope="hybrid_all",
            groups=as_groups(metrics.get("groups_keep")),
            include_view_rows=False,
            include_group_rows=True,
            include_variant_type_rows=True,
        )
    )

    return rows


def extract_rows(path: Path) -> List[Dict[str, Any]]:
    if path.name not in MODEL_REGISTRY:
        raise KeyError(f"No MODEL_REGISTRY entry for {path.name}")

    registry = MODEL_REGISTRY[path.name]
    data = load_json(path)

    if "metrics" in data and "by_scope" in (data.get("metrics") or {}):
        return rows_from_merged(path, data, registry)
    return rows_from_legacy(path, data, registry)


def sort_key(row: Dict[str, Any]) -> Tuple[int, int, int, int]:
    registry_order = next(
        (v["order"] for v in MODEL_REGISTRY.values() if v["name"] == row["name"]),
        999,
    )
    return (
        registry_order,
        SCOPE_TYPE_ORDER.get(row["scope_type"], 999),
        SCOPE_ORDER.get(row["scope"].replace("_SNP", "").replace("_INDEL", ""), 999),
        VARIANT_ORDER.get(row["variant_type"], 999),
    )


def write_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=sort_key)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in FIELDNAMES})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("hg003_summary_internal_test_long.csv"))
    parser.add_argument("--files", nargs="*", default=None)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when a default file is missing. Without this flag, missing files are skipped.",
    )
    args = parser.parse_args()

    filenames = args.files if args.files is not None else DEFAULT_FILES
    rows: List[Dict[str, Any]] = []
    missing: List[str] = []

    for filename in filenames:
        path = args.input_dir / filename
        if not path.exists():
            missing.append(filename)
            continue
        rows.extend(extract_rows(path))

    if args.strict and missing:
        raise FileNotFoundError("Missing files:\n" + "\n".join(missing))

    write_csv(rows, args.out)

    print(f"Wrote {len(rows)} rows to {args.out}")
    if missing:
        print("Skipped missing files:")
        for item in missing:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
