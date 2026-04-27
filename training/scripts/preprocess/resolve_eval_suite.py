#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Resolve an evaluation suite config into a normalized, ready-to-consume JSON manifest.

Supported:
- source_type = multimodal

Typical usage:
    python training/scripts/preprocess/resolve_eval_suite.py \
      --eval_suite_config training/configs/eval_suites/hg005_final_test_40x.json

    python training/scripts/preprocess/resolve_eval_suite.py \
      --eval_suite_config training/configs/eval_suites/hg005_coverage_sweep.json \
      --output_json training/out/experiments/debug_eval/resolved_eval_suite.json
"""

from __future__ import annotations

import argparse
import json
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


def ensure_keys(d: Dict[str, Any], keys: List[str], *, ctx: str) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise SystemExit(f"[ERROR] Missing required keys in {ctx}: {missing}")


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

    return p1.resolve(), False


# ---------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------

def resolve_eval_suite(payload: Dict[str, Any], repo_root: Path, config_path: Path) -> Dict[str, Any]:
    ensure_keys(
        payload,
        ["eval_suite_id", "source_type", "targets"],
        ctx="eval_suite config",
    )

    source_type = str(payload["source_type"])
    if source_type != "multimodal":
        raise SystemExit(
            f"[ERROR] resolve_eval_suite.py currently only supports source_type='multimodal'. "
            f"Got: {source_type}"
        )

    targets = payload["targets"]
    if not isinstance(targets, list) or not targets:
        raise SystemExit("[ERROR] 'targets' must be a non-empty list.")

    resolved_targets: List[Dict[str, Any]] = []
    seen_target_ids = set()

    for idx, target in enumerate(targets):
        if not isinstance(target, dict):
            raise SystemExit(f"[ERROR] targets[{idx}] must be an object.")

        ensure_keys(
            target,
            ["target_id", "dataset_id", "subject", "chroms", "coverage", "role"],
            ctx=f"targets[{idx}]",
        )

        target_id = str(target["target_id"])
        dataset_id = str(target["dataset_id"])
        subject = str(target["subject"])
        coverage = str(target["coverage"])
        role = str(target["role"])
        chroms = target["chroms"]

        if target_id in seen_target_ids:
            raise SystemExit(f"[ERROR] Duplicate target_id in eval suite: {target_id}")
        seen_target_ids.add(target_id)

        if not isinstance(chroms, list) or not chroms or not all(isinstance(c, str) for c in chroms):
            raise SystemExit(
                f"[ERROR] targets[{idx}].chroms must be a non-empty list[str]."
            )

        dataset_path, exists = maybe_dataset_path_multimodal(repo_root, dataset_id)

        resolved_targets.append(
            {
                "target_id": target_id,
                "dataset_id": dataset_id,
                "dataset_path": str(dataset_path),
                "dataset_path_exists": exists,
                "subject": subject,
                "chroms": chroms,
                "coverage": coverage,
                "role": role,
                "selection": {
                    "type": "eval_suite_target",
                    "chroms": chroms
                }
            }
        )

    return {
        "eval_suite_id": payload["eval_suite_id"],
        "source_type": payload["source_type"],
        "description": payload.get("description", ""),
        "resolved_at": utc_now_iso(),
        "repo_root": str(repo_root),
        "eval_suite_config_path": str(config_path.resolve()),
        "resolved_targets": resolved_targets,
        "summary": {
            "n_targets": len(resolved_targets),
            "target_ids": [t["target_id"] for t in resolved_targets]
        }
    }


# ---------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------

def print_eval_suite_summary(resolved: Dict[str, Any]) -> None:
    print("\n[ResolvedEvalSuite]")
    print(f"  eval_suite_id  : {resolved['eval_suite_id']}")
    print(f"  source_type    : {resolved['source_type']}")
    print(f"  config         : {resolved['eval_suite_config_path']}")
    print(f"  resolved_at    : {resolved['resolved_at']}")

    print("\n[Targets]")
    for t in resolved["resolved_targets"]:
        chroms = ",".join(t["chroms"])
        exists = "yes" if t["dataset_path_exists"] else "no"
        print(
            f"  - target_id={t['target_id']} "
            f"dataset_id={t['dataset_id']} "
            f"subject={t['subject']} "
            f"chroms=[{chroms}] "
            f"coverage={t['coverage']} "
            f"role={t['role']} "
            f"path_exists={exists}"
        )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--eval_suite_config",
        type=str,
        required=True,
        help="Path to eval suite config JSON under training/configs/eval_suites/.",
    )
    ap.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional output path for resolved eval suite JSON. "
             "Default: sibling file next to eval suite config named resolved__<eval_suite_id>.json",
    )
    ap.add_argument(
        "--repo_root",
        type=str,
        default=None,
        help="Optional explicit repo root. If omitted, it is auto-detected.",
    )
    args = ap.parse_args()

    config_path = Path(args.eval_suite_config).resolve()
    if not config_path.exists():
        raise SystemExit(f"[ERROR] eval_suite_config not found: {config_path}")

    repo_root = Path(args.repo_root).resolve() if args.repo_root else find_repo_root(Path.cwd())
    payload = load_json(config_path)
    resolved = resolve_eval_suite(payload, repo_root, config_path)

    if args.output_json:
        output_json = Path(args.output_json).resolve()
    else:
        output_json = config_path.parent / f"resolved__{payload['eval_suite_id']}.json"

    dump_json(output_json, resolved)
    print_eval_suite_summary(resolved)

    print("\n[Saved]")
    print(f"  {output_json.resolve()}")


if __name__ == "__main__":
    main()