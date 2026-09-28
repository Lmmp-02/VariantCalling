#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

mkdir -p "$TMP/evaluation"

# Teacher and unimodal reports were already single-view outputs.
cp "$ROOT"/evaluation/dv_*_test.json "$TMP/evaluation/"
cp "$ROOT"/evaluation/unimodal_*_test.json "$TMP/evaluation/"

# Rebuild normalized multi-view hybrid reports from the historical outputs.
cd "$ROOT/historical_outputs/view_specific"

python "$ROOT/reporting/merge_legacy_views.py" \
  --name hybrid_linear \
  --ill hybrid_linear_illvis_test.json \
  --ont hybrid_linear_ontvis_test.json \
  --all hybrid_linear_test.json \
  --out "$TMP/evaluation/hybrid_simple_linear_test.json"

python "$ROOT/reporting/merge_legacy_views.py" \
  --name hybrid_mlp \
  --ill hybrid_mlp_illvis_test.json \
  --ont hybrid_mlp_ontvis_test.json \
  --all hybrid_mlp_test.json \
  --out "$TMP/evaluation/hybrid_simple_mlp_test.json"

python "$ROOT/reporting/merge_legacy_views.py" \
  --name hybrid_groupwise_linear \
  --ill hybrid_groupwise_linear_illvis_test.json \
  --ont hybrid_groupwise_linear_ontvis_test.json \
  --all hybrid_groupwise_linear_test.json \
  --out "$TMP/evaluation/hybrid_groupwise_linear_test.json"

python "$ROOT/reporting/merge_legacy_views.py" \
  --name hybrid_groupwise_mlp \
  --ill hybrid_groupwise_mlp_illvis_test.json \
  --ont hybrid_groupwise_mlp_ontvis_test.json \
  --all hybrid_groupwise_mlp_test.json \
  --out "$TMP/evaluation/hybrid_groupwise_mlp_test.json"

# Rebuild the compact normalized summary.
python "$ROOT/reporting/build_hg003_summary_internal_long.py" \
  --input-dir "$TMP/evaluation" \
  --out "$TMP/hg003_summary_internal_test_long.csv" \
  --strict

cmp \
  "$TMP/hg003_summary_internal_test_long.csv" \
  "$ROOT/results/hg003_summary_internal_test_long.csv"

echo "OK: normalized HG003 chr20 summary reproduces exactly."
