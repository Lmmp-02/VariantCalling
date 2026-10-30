#!/usr/bin/env python3
"""Check singleton-locus availability-group counts against the paper reference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PAPER_DATASETS = (
    "hg002_chr20_chr21_harmonized_40x",
    "hg003_chr20_chr21_harmonized_40x",
    "hg004_chr20_chr21_harmonized_40x",
    "hg005_chr20_chr21_harmonized_40x",
    "hg005_chr20_chr21_coverage_study_20x",
    "hg005_chr20_chr21_coverage_study_10x",
    "hg005_chr20_chr21_coverage_study_5x",
)

REFERENCE_EVAL_SET = {
    "hg004_chr20_chr21_harmonized_40x": "hg004_chr20_40x",
    "hg005_chr20_chr21_harmonized_40x": "hg005_40x",
    "hg005_chr20_chr21_coverage_study_20x": "hg005_20x",
    "hg005_chr20_chr21_coverage_study_10x": "hg005_10x",
    "hg005_chr20_chr21_coverage_study_5x": "hg005_5x",
}

GROUPS = (1, 10, 11)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count groups 1/10/11 in paper singleton-locus outer shards and compare "
            "HG004/HG005 with Table 7's DeepVariant support counts."
        )
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=Path("data/4_out/datasets/multimodal/by_subject"),
        help="Root containing <dataset_id>/outer/ directories.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("results/paper/analysis/group_errors.csv"),
        help="Paper group_errors.csv used as the Table 7 reference.",
    )
    parser.add_argument(
        "--only",
        choices=PAPER_DATASETS,
        help="Check one paper dataset instead of all seven.",
    )
    return parser.parse_args()


def normalise_chrom(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype.kind in ("S", "O"):
        return np.asarray(
            [
                value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
                for value in arr.tolist()
            ],
            dtype=np.str_,
        )
    return arr.astype(np.str_)


def chromosome_mask(dataset_id: str, chrom: np.ndarray) -> tuple[np.ndarray, str]:
    chrom = normalise_chrom(chrom)
    if dataset_id == "hg004_chr20_chr21_harmonized_40x":
        return np.isin(chrom, np.asarray(["chr20", "20"], dtype=np.str_)), "chr20"
    if dataset_id.startswith("hg005_"):
        return np.isin(chrom, np.asarray(["chr20", "20", "chr21", "21"], dtype=np.str_)), "chr20+chr21"
    return np.ones(chrom.shape[0], dtype=bool), "all shard rows"


def count_dataset(dataset_root: Path, dataset_id: str) -> tuple[dict[int, int], str, int]:
    outer_dir = dataset_root / dataset_id / "outer"
    shards = sorted(outer_dir.glob("*_outer_*.npz"))
    if not shards:
        raise RuntimeError(f"No singleton-locus outer NPZ shards found under: {outer_dir}")

    counts = {group: 0 for group in GROUPS}
    selected_rows = 0
    scope_label: str | None = None

    for shard in shards:
        with np.load(shard, allow_pickle=False) as npz:
            missing = [key for key in ("group", "chrom") if key not in npz.files]
            if missing:
                raise RuntimeError(f"Missing keys {missing} in shard: {shard}")

            groups = np.asarray(npz["group"], dtype=np.int64)
            chrom = np.asarray(npz["chrom"])
            if groups.shape[0] != chrom.shape[0]:
                raise RuntimeError(f"group/chrom row mismatch in shard: {shard}")

            keep, this_scope = chromosome_mask(dataset_id, chrom)
            if scope_label is None:
                scope_label = this_scope
            selected_rows += int(np.count_nonzero(keep))

            selected_groups = groups[keep]
            unexpected = sorted(set(np.unique(selected_groups).tolist()) - set(GROUPS))
            if unexpected:
                raise RuntimeError(f"Unexpected availability groups {unexpected} in shard: {shard}")
            for group in GROUPS:
                counts[group] += int(np.count_nonzero(selected_groups == group))

    return counts, scope_label or "unknown", selected_rows


def load_reference(path: Path, eval_sets: set[str]) -> dict[str, dict[int, int]]:
    if not path.is_file():
        raise RuntimeError(f"Reference CSV not found: {path}")

    df = pd.read_csv(path)
    required = {"eval_set", "model", "group", "n"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Reference CSV is missing columns: {missing}")

    dv = df[df["model"].isin(["dv_illumina", "dv_ont"])].copy()
    references: dict[str, dict[int, int]] = {}

    for eval_set in sorted(eval_sets):
        rows = dv[dv["eval_set"] == eval_set]
        if rows.empty:
            raise RuntimeError(f"No DeepVariant reference rows for eval_set={eval_set}")

        expected: dict[int, int] = {}
        for group in GROUPS:
            values = rows.loc[rows["group"] == group, "n"].astype(int).unique().tolist()
            if len(values) != 1:
                raise RuntimeError(
                    f"Expected one unique DeepVariant n for eval_set={eval_set}, group={group}; got {values}"
                )
            expected[group] = int(values[0])
        references[eval_set] = expected

    return references


def main() -> int:
    args = parse_args()
    datasets = (args.only,) if args.only else PAPER_DATASETS

    try:
        needed_eval_sets = {REFERENCE_EVAL_SET[d] for d in datasets if d in REFERENCE_EVAL_SET}
        references = load_reference(args.reference, needed_eval_sets)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    rows: list[dict[str, object]] = []
    failed = False

    for dataset_id in datasets:
        try:
            counts, scope, selected_rows = count_dataset(args.dataset_root, dataset_id)
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

        eval_set = REFERENCE_EVAL_SET.get(dataset_id)
        expected_counts = references.get(eval_set) if eval_set else None

        for group in GROUPS:
            observed = counts[group]
            if expected_counts is None:
                expected: int | None = None
                status = "REPORT"
                delta: int | None = None
            else:
                expected = expected_counts[group]
                delta = observed - expected
                status = "PASS" if delta == 0 else "FAIL"
                failed |= status == "FAIL"

            rows.append(
                {
                    "dataset": dataset_id,
                    "scope": scope,
                    "group": group,
                    "observed": observed,
                    "expected": expected,
                    "delta": delta,
                    "status": status,
                }
            )

        if sum(counts.values()) != selected_rows:
            print(
                f"ERROR: counted groups do not cover selected rows for {dataset_id}: "
                f"groups={sum(counts.values())}, selected={selected_rows}",
                file=sys.stderr,
            )
            return 1

    table = pd.DataFrame(rows)
    print(table.to_string(index=False, na_rep="-"))
    print()
    if failed:
        print("Group-count comparison: FAIL")
        return 1
    print("Group-count comparison: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
