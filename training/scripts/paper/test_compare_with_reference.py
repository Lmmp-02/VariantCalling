#!/usr/bin/env python3
"""Small acceptance test for compare_with_reference.py."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


def run(command, expected: int) -> None:
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode != expected:
        print(result.stdout)
        raise SystemExit(
            f"Expected exit {expected}, got {result.returncode}: {' '.join(map(str, command))}"
        )


def main() -> int:
    repo = Path(__file__).resolve().parents[3]
    comparator = Path(__file__).with_name("compare_with_reference.py")
    reference = repo / "results" / "paper" / "analysis"

    base = [
        sys.executable,
        str(comparator),
        "--reference",
        str(reference),
    ]

    run([*base, "--analysis_dir", str(reference), "--mode", "exact"], expected=0)
    print("[PASS] exact mode accepts the published analysis against itself")

    with tempfile.TemporaryDirectory(prefix="paper_compare_") as tmp:
        tmp_path = Path(tmp)

        exact_bad = tmp_path / "exact_bad"
        exact_bad.mkdir()
        for filename in ("view_metrics.csv", "group_errors.csv", "paired_tests.csv"):
            shutil.copy2(reference / filename, exact_bad / filename)
        vm_bad = pd.read_csv(exact_bad / "view_metrics.csv")
        vm_bad.loc[0, "errors"] = int(vm_bad.loc[0, "errors"]) + 1
        vm_bad.to_csv(exact_bad / "view_metrics.csv", index=False)
        run([*base, "--analysis_dir", str(exact_bad), "--mode", "exact"], expected=1)
        print("[PASS] exact mode rejects a changed error count")

        tolerance_ok = tmp_path / "tolerance_ok"
        tolerance_ok.mkdir()
        vm = pd.read_csv(reference / "view_metrics.csv")
        vm["family"] = vm["model"].str.replace(r"__seed\d+$", "", regex=True)
        shifted = 0
        for _, idx in vm.groupby(["eval_set", "view", "family"]).groups.items():
            values = vm.loc[idx, "errors"].astype(int)
            lo, hi = int(values.min()), int(values.max())
            if lo == hi:
                continue
            width = hi - lo + 1
            replacement = lo + ((values - lo + 1) % width)
            shifted += int((replacement != values).sum())
            vm.loc[idx, "errors"] = replacement.to_numpy()
        if shifted == 0:
            raise SystemExit("Tolerance fixture did not shift any error counts")
        vm.drop(columns="family").to_csv(tolerance_ok / "view_metrics.csv", index=False)
        run([*base, "--analysis_dir", str(tolerance_ok), "--mode", "tolerance"], expected=0)
        print(f"[PASS] tolerance mode accepts {shifted} shifted counts inside reference ranges")

    return 0


if __name__ == "__main__":
    sys.exit(main())
