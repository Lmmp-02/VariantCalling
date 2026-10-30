#!/usr/bin/env python3
"""Compare regenerated paper analysis CSVs with the published reference results.

Two comparison modes are intentionally provided:

``exact``
    Intended for Level 1, where the published prediction dumps are re-analysed.
    The three primary analysis CSVs must contain the same keyed rows. Integer and
    text values must match exactly; floating-point values may differ by at most
    1e-12 (absolute tolerance, no relative tolerance).

``tolerance``
    Intended for Level 2, where the lightweight heads are retrained. For every
    evaluation set, view and model family, the median error count of the rerun
    must fall inside the published minimum-to-maximum range across training
    seeds. Per-seed error differences are reported for diagnostics.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_float_dtype, is_integer_dtype, is_numeric_dtype

EXACT_FILES: Dict[str, Sequence[str]] = {
    "view_metrics.csv": ("eval_set", "view", "model"),
    "group_errors.csv": ("eval_set", "view", "model", "group"),
    "paired_tests.csv": ("eval_set", "view", "model_a", "model_b"),
}
TOLERANCE_FILE = "view_metrics.csv"
TOLERANCE_REQUIRED = ("eval_set", "view", "model", "errors")
SEED_RE = re.compile(r"__seed(\d+)$")
FLOAT_ATOL = 1e-12


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _format_table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    rows_s = [[str(cell) for cell in row] for row in rows]
    widths = [len(str(h)) for h in headers]
    for row in rows_s:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def render(row: Sequence[str]) -> str:
        return "  ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row))

    header = render([str(h) for h in headers])
    rule = render(["-" * width for width in widths])
    body = [render(row) for row in rows_s]
    return "\n".join([header, rule, *body])


def _key_index(df: pd.DataFrame, keys: Sequence[str], label: str) -> pd.DataFrame:
    missing = [key for key in keys if key not in df.columns]
    if missing:
        raise ValueError(f"{label}: missing key columns: {', '.join(missing)}")
    duplicated = df.duplicated(list(keys), keep=False)
    if duplicated.any():
        examples = df.loc[duplicated, list(keys)].head(3).to_dict("records")
        raise ValueError(f"{label}: duplicate row keys, examples: {examples}")
    return df.set_index(list(keys), drop=False).sort_index()


def _exact_column_difference(reference: pd.Series, candidate: pd.Series) -> Tuple[int, str]:
    """Return mismatch count and a compact diagnostic for one aligned column."""
    if is_integer_dtype(reference.dtype):
        equal = reference.eq(candidate) | (reference.isna() & candidate.isna())
        count = int((~equal).sum())
        return count, "integer mismatch" if count else ""

    if is_float_dtype(reference.dtype):
        if not is_numeric_dtype(candidate.dtype):
            return len(reference), f"candidate dtype {candidate.dtype} is not numeric"
        ref_values = reference.to_numpy(dtype=float)
        cand_values = candidate.to_numpy(dtype=float)
        equal = np.isclose(ref_values, cand_values, rtol=0.0, atol=FLOAT_ATOL, equal_nan=True)
        count = int((~equal).sum())
        if count:
            finite = np.isfinite(ref_values) & np.isfinite(cand_values)
            max_diff = float(np.max(np.abs(ref_values[finite] - cand_values[finite]))) if finite.any() else float("nan")
            return count, f"float mismatch (max |Δ|={max_diff:.3g})"
        return 0, ""

    equal = reference.eq(candidate) | (reference.isna() & candidate.isna())
    count = int((~equal).sum())
    return count, "value mismatch" if count else ""


def compare_exact(analysis_dir: Path, reference_dir: Path) -> bool:
    rows: List[Sequence[object]] = []
    passed = True

    for filename, keys in EXACT_FILES.items():
        try:
            reference = _read_csv(reference_dir / filename)
            candidate = _read_csv(analysis_dir / filename)

            if list(reference.columns) != list(candidate.columns):
                missing = [c for c in reference.columns if c not in candidate.columns]
                extra = [c for c in candidate.columns if c not in reference.columns]
                detail = f"columns differ; missing={missing or '-'} extra={extra or '-'}"
                rows.append((filename, len(candidate), "FAIL", detail))
                passed = False
                continue

            reference = _key_index(reference, keys, f"reference/{filename}")
            candidate = _key_index(candidate, keys, f"analysis/{filename}")

            missing_keys = reference.index.difference(candidate.index)
            extra_keys = candidate.index.difference(reference.index)
            if len(missing_keys) or len(extra_keys):
                detail = f"row keys differ; missing={len(missing_keys)} extra={len(extra_keys)}"
                rows.append((filename, len(candidate), "FAIL", detail))
                passed = False
                continue

            candidate = candidate.loc[reference.index]
            mismatches: List[str] = []
            for column in reference.columns:
                count, diagnostic = _exact_column_difference(reference[column], candidate[column])
                if count:
                    mismatches.append(f"{column}: {count} ({diagnostic})")

            if mismatches:
                rows.append((filename, len(candidate), "FAIL", "; ".join(mismatches[:3])))
                passed = False
            else:
                rows.append((filename, len(candidate), "PASS", "values match"))
        except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
            rows.append((filename, "-", "FAIL", str(exc)))
            passed = False

    print(_format_table(("file", "rows", "status", "detail"), rows))
    print(f"\nExact comparison: {'PASS' if passed else 'FAIL'}")
    return passed


def _add_family_and_seed(df: pd.DataFrame, label: str) -> pd.DataFrame:
    missing = [column for column in TOLERANCE_REQUIRED if column not in df.columns]
    if missing:
        raise ValueError(f"{label}: missing columns: {', '.join(missing)}")

    out = df.copy()
    out["model"] = out["model"].astype(str)
    out["family"] = out["model"].str.replace(r"__seed\d+$", "", regex=True)
    extracted = out["model"].str.extract(r"__seed(\d+)$", expand=False)
    out["seed"] = pd.to_numeric(extracted, errors="coerce").fillna(0).astype(int)
    out["errors"] = pd.to_numeric(out["errors"], errors="raise")

    keys = ["eval_set", "view", "family", "seed"]
    duplicated = out.duplicated(keys, keep=False)
    if duplicated.any():
        examples = out.loc[duplicated, keys].head(3).to_dict("records")
        raise ValueError(f"{label}: duplicate family/seed rows, examples: {examples}")
    return out


def compare_tolerance(analysis_dir: Path, reference_dir: Path) -> bool:
    try:
        reference = _add_family_and_seed(
            _read_csv(reference_dir / TOLERANCE_FILE), f"reference/{TOLERANCE_FILE}"
        )
        candidate = _add_family_and_seed(
            _read_csv(analysis_dir / TOLERANCE_FILE), f"analysis/{TOLERANCE_FILE}"
        )
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        print(f"Tolerance comparison: FAIL\n{exc}")
        return False

    family_keys = ["eval_set", "view", "family"]
    seed_keys = [*family_keys, "seed"]

    ref_families = set(map(tuple, reference[family_keys].drop_duplicates().itertuples(index=False, name=None)))
    new_families = set(map(tuple, candidate[family_keys].drop_duplicates().itertuples(index=False, name=None)))
    missing_families = ref_families - new_families
    extra_families = new_families - ref_families

    ref_seed = reference.set_index(seed_keys)["errors"]
    new_seed = candidate.set_index(seed_keys)["errors"]
    missing_seed_rows = ref_seed.index.difference(new_seed.index)
    extra_seed_rows = new_seed.index.difference(ref_seed.index)

    passed = not (missing_families or extra_families or len(missing_seed_rows) or len(extra_seed_rows))
    rows: List[Sequence[object]] = []

    for key in sorted(ref_families | new_families):
        eval_set, view, family = key
        ref_group = reference[
            (reference["eval_set"] == eval_set)
            & (reference["view"] == view)
            & (reference["family"] == family)
        ]
        new_group = candidate[
            (candidate["eval_set"] == eval_set)
            & (candidate["view"] == view)
            & (candidate["family"] == family)
        ]

        if ref_group.empty:
            rows.append((eval_set, view, family, "-", "-", "-", "FAIL extra family"))
            passed = False
            continue
        if new_group.empty:
            rows.append((eval_set, view, family, "-", "-", "-", "FAIL missing family"))
            passed = False
            continue

        ref_min = int(ref_group["errors"].min())
        ref_max = int(ref_group["errors"].max())
        new_median = float(new_group["errors"].median())
        in_range = ref_min <= new_median <= ref_max
        if not in_range:
            passed = False

        deltas = []
        ref_by_seed = ref_group.set_index("seed")["errors"]
        new_by_seed = new_group.set_index("seed")["errors"]
        for seed in sorted(set(ref_by_seed.index) | set(new_by_seed.index)):
            if seed not in ref_by_seed.index:
                deltas.append(f"s{seed}:extra")
            elif seed not in new_by_seed.index:
                deltas.append(f"s{seed}:missing")
            else:
                delta = int(new_by_seed.loc[seed] - ref_by_seed.loc[seed])
                deltas.append(f"s{seed}:{delta:+d}")

        status = "PASS" if in_range else "FAIL"
        rows.append((
            eval_set,
            view,
            family,
            f"{new_median:g}",
            f"[{ref_min},{ref_max}]",
            ",".join(deltas),
            status,
        ))

    print(_format_table(
        ("eval_set", "view", "family", "new_med", "ref_range", "seed Δerrors", "status"),
        rows,
    ))

    if missing_seed_rows.size:
        print(f"\nMissing seed rows: {len(missing_seed_rows)}")
    if extra_seed_rows.size:
        print(f"Extra seed rows: {len(extra_seed_rows)}")
    print(f"\nTolerance comparison: {'PASS' if passed else 'FAIL'}")
    return passed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis_dir",
        required=True,
        type=Path,
        help="Analysis directory produced by analyse_predictions.py.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("results/paper/analysis"),
        help="Published reference analysis directory (default: results/paper/analysis).",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=("exact", "tolerance"),
        help="Exact re-analysis check or seed-range retraining check.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "exact":
        passed = compare_exact(args.analysis_dir, args.reference)
    else:
        passed = compare_tolerance(args.analysis_dir, args.reference)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
