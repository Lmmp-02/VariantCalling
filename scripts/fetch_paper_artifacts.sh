#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RECORD=""
WHAT=""
FORCE=0
DOWNLOAD_DIR=""

PREDICTIONS=(
  paper_predictions.tar
)
CHECKPOINTS=(
  paper_checkpoints.tar
)
DATASETS=(
  dataset_hg002_chr20_chr21_harmonized_40x.tar
  dataset_hg003_chr20_chr21_harmonized_40x.tar
  dataset_hg004_chr20_chr21_harmonized_40x.tar
  dataset_hg005_chr20_chr21_harmonized_40x.tar
  dataset_hg005_chr20_chr21_coverage_study_20x.tar
  dataset_hg005_chr20_chr21_coverage_study_10x.tar
  dataset_hg005_chr20_chr21_coverage_study_5x.tar
)

usage() {
  cat <<'USAGE'
Usage:
  scripts/fetch_paper_artifacts.sh --record <zenodo_record_id> \
    --what predictions|checkpoints|datasets|all [--force] [--download-dir <dir>]

Downloads the paper artefacts from Zenodo, verifies each selected tarball against the
record's SHA256SUMS, and extracts it at the repository root.

Options:
  --record <id>         Zenodo record/deposition id (for this paper: 23804374).
  --what <selection>    predictions, checkpoints, datasets, or all.
  --force               Allow extraction to overwrite existing files.
  --download-dir <dir>  Archive cache directory. Default:
                        ${XDG_CACHE_HOME:-$HOME/.cache}/variantcalling/zenodo-<record>
  -h, --help            Show this help.

Draft records:
  Set ZENODO_TOKEN to a personal Zenodo token. The script then resolves file URLs through
  the deposition API. Never put the token in this repository or on the command line.

Published records:
  No token is needed; files are fetched from the public record URL.
USAGE
}

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

need_arg() {
  [[ $# -ge 2 && -n "${2:-}" ]] || fail "Missing value for $1"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --record)
      need_arg "$@"
      RECORD="$2"
      shift 2
      ;;
    --what)
      need_arg "$@"
      WHAT="$2"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --download-dir)
      need_arg "$@"
      DOWNLOAD_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "Unknown argument: $1"
      ;;
  esac
done

[[ -n "$RECORD" ]] || fail "--record is required"
[[ "$RECORD" =~ ^[0-9]+$ ]] || fail "--record must be a numeric Zenodo record id"
case "$WHAT" in
  predictions|checkpoints|datasets|all) ;;
  "") fail "--what is required" ;;
  *) fail "--what must be predictions, checkpoints, datasets, or all" ;;
esac

command -v curl >/dev/null 2>&1 || fail "curl is required"
command -v sha256sum >/dev/null 2>&1 || fail "sha256sum is required"
command -v tar >/dev/null 2>&1 || fail "tar is required"
command -v python >/dev/null 2>&1 || fail "python is required"

if [[ -z "$DOWNLOAD_DIR" ]]; then
  DOWNLOAD_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/variantcalling/zenodo-${RECORD}"
fi
mkdir -p "$DOWNLOAD_DIR"
DOWNLOAD_DIR="$(cd "$DOWNLOAD_DIR" && pwd)"

SELECTED=()
case "$WHAT" in
  predictions)
    SELECTED+=("${PREDICTIONS[@]}")
    ;;
  checkpoints)
    SELECTED+=("${CHECKPOINTS[@]}")
    ;;
  datasets)
    SELECTED+=("${DATASETS[@]}")
    ;;
  all)
    SELECTED+=("${PREDICTIONS[@]}" "${CHECKPOINTS[@]}" "${DATASETS[@]}")
    ;;
esac

CURL_AUTH=()
DEPOSITION_FILES_JSON=""
if [[ -n "${ZENODO_TOKEN:-}" ]]; then
  CURL_AUTH=(-H "Authorization: Bearer ${ZENODO_TOKEN}")
  DEPOSITION_FILES_JSON="$DOWNLOAD_DIR/deposition-files.json"
  echo "Resolving draft file URLs for Zenodo deposition ${RECORD} ..."
  curl --fail --silent --show-error --location --retry 3 \
    "${CURL_AUTH[@]}" \
    "https://zenodo.org/api/deposit/depositions/${RECORD}/files" \
    -o "$DEPOSITION_FILES_JSON"
fi

file_url() {
  local filename="$1"
  if [[ -n "$DEPOSITION_FILES_JSON" ]]; then
    python - "$DEPOSITION_FILES_JSON" "$filename" <<'PY'
import json
import sys

path, wanted = sys.argv[1:]
with open(path, "r", encoding="utf-8") as handle:
    entries = json.load(handle)
for entry in entries:
    name = entry.get("filename") or entry.get("key")
    if name != wanted:
        continue
    links = entry.get("links") or {}
    url = links.get("download") or links.get("content") or links.get("self")
    if url:
        print(url)
        raise SystemExit(0)
raise SystemExit(f"File {wanted!r} not found in Zenodo draft deposition")
PY
  else
    printf 'https://zenodo.org/records/%s/files/%s?download=1\n' "$RECORD" "$filename"
  fi
}

download_file() {
  local filename="$1"
  local destination="$DOWNLOAD_DIR/$filename"
  local url
  url="$(file_url "$filename")"

  echo "Downloading $filename"
  if [[ -f "$destination" ]]; then
    echo "  resuming cached file: $destination"
  fi

  curl --fail --show-error --location --retry 3 --continue-at - \
    "${CURL_AUTH[@]}" \
    "$url" \
    -o "$destination"
}

# SHA256SUMS is small and is always refreshed. It is the record's integrity manifest for
# the nine tarballs; README.txt is descriptive and is not needed by the fetcher.
SHA_FILE="$DOWNLOAD_DIR/SHA256SUMS"
rm -f "$SHA_FILE"
download_file SHA256SUMS

verify_archive() {
  local filename="$1"
  local archive="$DOWNLOAD_DIR/$filename"
  local expected
  expected="$(awk -v f="$filename" '$2 == f {print $1}' "$SHA_FILE")"
  [[ -n "$expected" ]] || fail "No SHA-256 entry for $filename in downloaded SHA256SUMS"
  [[ "$(printf '%s' "$expected" | wc -c)" -eq 64 ]] || fail "Invalid SHA-256 entry for $filename"

  local actual
  actual="$(sha256sum "$archive" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || fail "Checksum mismatch for $filename"
  echo "Verified   $filename"
}

validate_tar_paths() {
  local archive="$1"
  local member
  while IFS= read -r member; do
    [[ -n "$member" ]] || continue
    [[ "$member" != /* ]] || fail "Archive contains an absolute path: $member"
    case "/$member/" in
      */../*) fail "Archive contains a parent-directory path: $member" ;;
    esac
  done < <(tar -tf "$archive")
}

check_overwrite() {
  local archive="$1"
  [[ "$FORCE" -eq 1 ]] && return 0

  local member
  while IFS= read -r member; do
    [[ -n "$member" ]] || continue
    [[ "$member" == */ ]] && continue
    if [[ -e "$ROOT/$member" || -L "$ROOT/$member" ]]; then
      fail "Refusing to overwrite $member (re-run with --force to allow overwrites)"
    fi
  done < <(tar -tf "$archive")
}

for filename in "${SELECTED[@]}"; do
  archive="$DOWNLOAD_DIR/$filename"

  if [[ -f "$archive" ]]; then
    if verify_archive "$filename" >/dev/null 2>&1; then
      echo "Using cached $filename (checksum OK)"
    else
      download_file "$filename"
      verify_archive "$filename"
    fi
  else
    download_file "$filename"
    verify_archive "$filename"
  fi

  validate_tar_paths "$archive"
  check_overwrite "$archive"
  echo "Extracting  $filename at $ROOT"
  tar -xf "$archive" -C "$ROOT"
done

echo
echo "Fetched and extracted: $WHAT"
echo "Archive cache: $DOWNLOAD_DIR"
if [[ "$WHAT" == "predictions" || "$WHAT" == "all" ]]; then
  cat <<'NEXT'

Level 1 check:
  python training/scripts/paper/analyse_predictions.py \
    --pred_dir training/out/paper/predictions --out_dir /tmp/l1
  python training/scripts/paper/compare_with_reference.py \
    --analysis_dir /tmp/l1 --mode exact
NEXT
fi
