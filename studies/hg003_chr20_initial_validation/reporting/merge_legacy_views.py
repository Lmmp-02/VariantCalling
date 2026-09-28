#!/usr/bin/env python3
"""
merge_legacy_views.py

Merge legacy HG003 view-specific evaluation JSONs into a single JSON
with a structure closer to the newer HG004/HG005 evaluation outputs.

Expected legacy inputs:
  - Illumina view: groups_keep = [10, 11]
  - ONT view:      groups_keep = [1, 11]
  - Hybrid all:    groups_keep = [1, 10, 11]

The script does not recompute SNP/INDEL metrics from the global file.
It preserves the by_variant_type blocks from each view-specific JSON.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, Optional


SCOPE_CONFIG = {
    "illumina_view": {
        "label": "Illumina view",
        "expected_groups": [10, 11],
        "arg": "ill",
    },
    "ont_view": {
        "label": "ONT view",
        "expected_groups": [1, 11],
        "arg": "ont",
    },
    "hybrid_all": {
        "label": "Hybrid all",
        "expected_groups": [1, 10, 11],
        "arg": "all",
    },
}


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def cm2_binary_counts(cm: Any) -> Optional[Dict[str, int]]:
    """
    Expects a binary confusion matrix in the form:
      [[tn, fp],
       [fn, tp]]
    """
    if (
        isinstance(cm, list)
        and len(cm) == 2
        and all(isinstance(row, list) and len(row) == 2 for row in cm)
    ):
        tn, fp = cm[0]
        fn, tp = cm[1]
        return {
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
        }
    return None


def add_binary_counts(block: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adds tp/tn/fp/fn if the block contains a binary cm and those fields
    are not already present. Also adds balanced_acc and f2 when possible.
    """
    out = copy.deepcopy(block)
    counts = cm2_binary_counts(out.get("cm"))

    if counts is not None:
        for key, value in counts.items():
            out.setdefault(key, value)

    sensitivity = out.get("recall")
    specificity = out.get("specificity")
    precision = out.get("precision")

    if "balanced_acc" not in out and sensitivity is not None and specificity is not None:
        out["balanced_acc"] = (float(sensitivity) + float(specificity)) / 2.0

    if "f2" not in out and precision is not None and sensitivity is not None:
        p = float(precision)
        r = float(sensitivity)
        denom = (4.0 * p) + r
        out["f2"] = (5.0 * p * r / denom) if denom else 0.0

    return out


def normalize_metric_tree(obj: Any) -> Any:
    """
    Recursively normalizes metric blocks. In practice, this mainly adds
    tp/tn/fp/fn to binary_variant_vs_no blocks in legacy JSONs.
    """
    if isinstance(obj, dict):
        normalized = {k: normalize_metric_tree(v) for k, v in obj.items()}
        if "cm" in normalized and {"precision", "recall", "f1"}.issubset(normalized.keys()):
            normalized = add_binary_counts(normalized)
        return normalized
    if isinstance(obj, list):
        return [normalize_metric_tree(v) for v in obj]
    return obj


def build_scope_block(scope_key: str, path: Path, data: Dict[str, Any]) -> Dict[str, Any]:
    cfg = SCOPE_CONFIG[scope_key]
    groups_keep = data.get("groups_keep")

    if groups_keep != cfg["expected_groups"]:
        raise ValueError(
            f"{path} does not match expected groups for {scope_key}: "
            f"expected {cfg['expected_groups']}, got {groups_keep}"
        )

    block: Dict[str, Any] = {
        "scope": cfg["label"],
        "scope_key": scope_key,
        "source_file": str(path),
        "split": data.get("split"),
        "mode": data.get("mode"),
        "teacher": data.get("teacher"),
        "groups_keep": groups_keep,
        "exclude_vt_mismatch": data.get("exclude_vt_mismatch"),
        "benchmark_policy": data.get("benchmark_policy"),
        "filter_stats": data.get("filter_stats"),
        "overall_3c": data.get("overall_3c"),
        "overall_binary_variant_vs_no": data.get("overall_binary_variant_vs_no"),
        "by_variant_type": data.get("by_variant_type"),
        "by_group": data.get("by_group"),
    }

    if "variant_only_genotype" in data:
        block["variant_only_genotype"] = data["variant_only_genotype"]

    block = normalize_metric_tree(block)

    # Alias used by newer JSONs.
    if "overall_binary_variant_vs_no" in block:
        block["overall_variant_from_cm3"] = copy.deepcopy(
            block["overall_binary_variant_vs_no"]
        )

    return block


def build_data_summary(by_scope: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Creates a lightweight data_summary inspired by the newer evaluation format.
    """
    summary: Dict[str, Any] = {"by_scope": {}}

    for scope_key, block in by_scope.items():
        by_group = block.get("by_group") or {}
        by_variant_type = block.get("by_variant_type") or {}

        summary["by_scope"][scope_key] = {
            "scope": block.get("scope"),
            "n": block.get("overall_3c", {}).get("n"),
            "groups_keep": block.get("groups_keep"),
            "filter_stats": block.get("filter_stats"),
            "by_group": {
                group: metrics.get("n")
                for group, metrics in by_group.items()
                if isinstance(metrics, dict)
            },
            "by_variant_type": {
                vt: metrics.get("n")
                for vt, metrics in by_variant_type.items()
                if isinstance(metrics, dict)
            },
        }

    # For compatibility, expose Hybrid all as the default summary when present.
    if "hybrid_all" in summary["by_scope"]:
        default = summary["by_scope"]["hybrid_all"]
        summary.update(
            {
                "name": "test",
                "n": default.get("n"),
                "by_group": default.get("by_group"),
                "by_variant_type": default.get("by_variant_type"),
            }
        )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Model name, e.g. hybrid_mlp")
    parser.add_argument("--ill", required=True, type=Path, help="Illumina-view legacy JSON")
    parser.add_argument("--ont", required=True, type=Path, help="ONT-view legacy JSON")
    parser.add_argument("--all", required=True, type=Path, help="Hybrid-all legacy JSON")
    parser.add_argument("--out", required=True, type=Path, help="Output merged JSON")
    args = parser.parse_args()

    input_paths = {
        "illumina_view": args.ill,
        "ont_view": args.ont,
        "hybrid_all": args.all,
    }

    by_scope: Dict[str, Dict[str, Any]] = {}

    for scope_key, path in input_paths.items():
        data = load_json(path)
        by_scope[scope_key] = build_scope_block(scope_key, path, data)

    hybrid_all = by_scope["hybrid_all"]

    merged: Dict[str, Any] = {
        "name": args.name,
        "partition": hybrid_all.get("split", "test"),
        "source": {
            "source": "legacy_merged_views",
            "inputs": {
                scope_key: str(path)
                for scope_key, path in input_paths.items()
            },
            "note": (
                "Merged from legacy HG003 per-view evaluation JSONs. "
                "View-specific SNP/INDEL metrics are preserved from each input file."
            ),
        },
        "data_summary": build_data_summary(by_scope),
        "metrics": {
            # Default/top-level metrics use Hybrid all, following the newer format.
            "overall_3c": hybrid_all.get("overall_3c"),
            "overall_binary_variant_vs_no": hybrid_all.get("overall_binary_variant_vs_no"),
            "overall_variant_from_cm3": hybrid_all.get("overall_variant_from_cm3"),
            "variant_only_genotype": hybrid_all.get("variant_only_genotype"),
            "by_group": hybrid_all.get("by_group"),
            "by_variant_type": hybrid_all.get("by_variant_type"),
            # New explicit scope block.
            "by_scope": by_scope,
        },
    }

    save_json(merged, args.out)
    print(f"Wrote merged JSON to: {args.out}")


if __name__ == "__main__":
    main()