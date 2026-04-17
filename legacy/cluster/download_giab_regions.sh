#!/usr/bin/env bash
set -euo pipefail

# Download/reconstruct regional BAMs for GIAB samples used in VariantCalling.
# - Illumina:
#     * HG002-HG004 -> remote regional extraction from GIAB/NIST BAM via samtools
#     * HG005       -> download full BAM + BAI to local staging, then extract regions locally
# - ONT:
#     * remote regional extraction from ONT Open Data BAMs via samtools
#       using a canonical run per sample by default
#
# Important implementation detail:
#   samtools/htslib may cache remote .bai files into the current working directory.
#   For ONT runs the remote BAM basename is always `calls.sorted.bam`, which can lead
#   to collisions if multiple runs/samples are extracted from the same cwd.
#   To avoid that, every remote extraction is executed inside an isolated temporary
#   directory and, for ONT, the index is referenced explicitly with ##idx##.
#
# Robustness:
#   - Remote extraction is retried automatically on transient failures.
#   - Output is first written to a temporary BAM in the destination directory.
#   - The final OUT_BAM is only promoted if the temporary BAM passes samtools quickcheck.
#   - ONT region names are resolved dynamically from the remote BAM header instead of
#     assuming chr/no_chr naming.
#
# Metadata JSON is always written next to each downloaded/generated BAM.

THREADS=8
BC="sup"
RETRIES=3
RETRY_SLEEP=5
OUT_ROOT="data/1_input_bams"
STAGING_ROOT="data/0_staging"
TECH=""
SAMPLE=""
REGIONS=()
RUNS=()
MERGE_ONT=false
SELECTION_POLICY=""

log_info()  { echo "[INFO] $*" >&2; }
log_step()  { echo "[STEP] $*" >&2; }
log_warn()  { echo "[WARN] $*" >&2; }
log_error() { echo "[ERROR] $*" >&2; }

usage() {
  cat <<USAGE
Usage:
  $0 --sample HG002 --tech Illumina --regions chr20 chr21 [options]
  $0 --sample HG002 --tech ONT      --regions chr20 chr21 [options]

Required:
  --sample   HG002 | HG003 | HG004 | HG005
  --tech     Illumina | ONT
  --regions  One or more regions, e.g. chr20 chr21

Optional:
  --threads      N              [default: 8]
  --out-root     PATH           [default: data/1_input_bams]
  --staging-root PATH           [default: data/0_staging]
  --retries      N              [default: 3]
  --retry-sleep  SEC            [default: 5]

ONT options:
  --bc         sup|hac        [default: sup]
  --run        RUN_ID         Explicit single ONT run
  --runs       RUN1 RUN2 ...  Explicit ONT runs
  --merge-ont                 Merge multiple explicit ONT runs into one BAM

Defaults / policy:
  - Illumina uses a single canonical GIAB/NIST BAM per sample.
  - HG005 Illumina is handled via full local download + local regional extraction.
  - ONT uses a single canonical run per sample by default, always with bc=sup unless overridden.
  - A metadata JSON is written next to every BAM.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sample) SAMPLE="$2"; shift 2 ;;
    --tech) TECH="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --staging-root) STAGING_ROOT="$2"; shift 2 ;;
    --retries) RETRIES="$2"; shift 2 ;;
    --retry-sleep) RETRY_SLEEP="$2"; shift 2 ;;
    --bc) BC="$2"; shift 2 ;;
    --regions)
      shift
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do REGIONS+=("$1"); shift; done
      ;;
    --run)
      RUNS+=("$2")
      shift 2
      ;;
    --runs)
      shift
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do RUNS+=("$1"); shift; done
      ;;
    --merge-ont) MERGE_ONT=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) log_error "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

[[ -n "$SAMPLE" ]] || { log_error "--sample is required"; exit 1; }
[[ -n "$TECH"   ]] || { log_error "--tech is required"; exit 1; }
[[ ${#REGIONS[@]} -gt 0 ]] || { log_error "--regions is required"; exit 1; }
command -v samtools >/dev/null || { log_error "samtools not found"; exit 1; }
command -v python3 >/dev/null || { log_error "python3 not found"; exit 1; }
command -v mktemp >/dev/null || { log_error "mktemp not found"; exit 1; }

join_by() { local IFS="$1"; shift; echo "$*"; }
REGION_TAG=$(join_by "_" "${REGIONS[@]}")
REGION_SPACE=$(join_by " " "${REGIONS[@]}")
TIMESTAMP_UTC=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
SCRIPT_NAME=$(basename "$0")

cleanup_tmp() {
  local d="$1"
  [[ -n "${d:-}" && -d "$d" ]] && rm -rf "$d"
}

rm_if_exists() {
  local f="$1"
  [[ -n "${f:-}" && -e "$f" ]] && rm -f -- "$f"
}

download_file_resume() {
  local url="$1"
  local out="$2"

  log_step "Downloading: ${url}"
  log_step "-> ${out}"

  if command -v curl >/dev/null 2>&1; then
    curl -L --fail \
      --retry "$RETRIES" \
      --retry-delay "$RETRY_SLEEP" \
      -C - \
      -o "$out" \
      "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -c \
      --tries="$RETRIES" \
      --waitretry="$RETRY_SLEEP" \
      -O "$out" \
      "$url"
  else
    log_error "neither curl nor wget found; one is required for HG005 Illumina full download"
    return 1
  fi
}

fetch_remote_header() {
  local bam_spec="$1"
  local out_header="$2"
  local attempt=1
  local tmpdir=""
  local sleep_s=0

  while (( attempt <= RETRIES )); do
    tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/samtools_hdrcache_XXXX")
    log_step "Probing remote BAM header... attempt ${attempt}/${RETRIES}"

    if (
      cd "$tmpdir"
      samtools view -H "$bam_spec" > header.sam 2>/dev/null
    ); then
      mv -f "$tmpdir/header.sam" "$out_header"
      cleanup_tmp "$tmpdir"
      return 0
    fi

    cleanup_tmp "$tmpdir"
    log_warn "Could not read remote BAM header on attempt ${attempt}/${RETRIES}"
    if (( attempt < RETRIES )); then
      sleep_s=$(( RETRY_SLEEP * attempt ))
      log_warn "Retrying header probe in ${sleep_s}s..."
      sleep "$sleep_s"
    fi
    attempt=$(( attempt + 1 ))
  done

  log_error "Unable to read remote BAM header after ${RETRIES} attempts"
  return 1
}

resolve_regions_from_header_file() {
  local header_file="$1"
  shift

  python3 - "$header_file" "$@" <<'PY'
import sys

header_file = sys.argv[1]
wanted = sys.argv[2:]

refs = []
alias_to_sn = {}

with open(header_file, "r", encoding="utf-8", errors="replace") as fh:
    for line in fh:
        if not line.startswith("@SQ\t"):
            continue
        fields = line.rstrip("\n").split("\t")
        sn = None
        aliases = []
        for f in fields[1:]:
            if f.startswith("SN:"):
                sn = f[3:]
            elif f.startswith("AN:"):
                aliases.extend([a for a in f[3:].split(",") if a])
        if sn is None:
            continue
        refs.append(sn)
        for a in [sn] + aliases:
            alias_to_sn.setdefault(a, sn)

def candidate_sns(q):
    out = []
    seen = set()

    def add(alias):
        if alias in alias_to_sn:
            sn = alias_to_sn[alias]
            if sn not in seen:
                out.append(sn)
                seen.add(sn)

    add(q)
    if q.startswith("chr"):
        add(q[3:])
    else:
        add("chr" + q)

    q_norm = q[3:] if q.startswith("chr") else q
    for alias, sn in alias_to_sn.items():
        alias_norm = alias[3:] if alias.startswith("chr") else alias
        if alias_norm == q_norm and sn not in seen:
            out.append(sn)
            seen.add(sn)

    return out

resolved = []
errors = []

for q in wanted:
    cands = candidate_sns(q)
    if len(cands) == 1:
        resolved.append(cands[0])
    elif len(cands) == 0:
        errors.append(f"{q}: no match")
    else:
        errors.append(f"{q}: ambiguous -> {', '.join(cands)}")

if errors:
    sys.stderr.write("ERROR resolving regions from BAM header:\n")
    for e in errors:
        sys.stderr.write(f"  - {e}\n")
    sys.stderr.write("Available references (first 60):\n")
    for sn in refs[:60]:
        sys.stderr.write(f"  - {sn}\n")
    sys.exit(1)

for r in resolved:
    print(r)
PY
}

resolve_regions_from_remote_bam() {
  local bam_spec="$1"
  shift
  local -a wanted_regions=("$@")

  local tmpdir=""
  local header_file=""
  local resolved=""

  tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/samtools_regionmap_XXXX")
  header_file="${tmpdir}/header.sam"

  trap 'cleanup_tmp "$tmpdir"' RETURN

  fetch_remote_header "$bam_spec" "$header_file"
  resolved=$(resolve_regions_from_header_file "$header_file" "${wanted_regions[@]}")
  printf '%s\n' "$resolved"
}

run_remote_extract_sort() {
  local bam_spec="$1"
  shift
  local out_bam="$1"
  shift
  local -a regions=("$@")

  local abs_out
  abs_out=$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$out_bam")
  local out_dir
  out_dir=$(dirname "$abs_out")
  local out_base
  out_base=$(basename "$abs_out")

  local attempt=1
  local tmpdir=""
  local tmp_out=""
  local rc=0
  local sleep_s=0

  while (( attempt <= RETRIES )); do
    tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/samtools_idxcache_XXXX")
    tmp_out=$(mktemp "${out_dir}/.${out_base}.attempt${attempt}.XXXXXX.bam")

    log_step "Extracting regional BAM... attempt ${attempt}/${RETRIES}"

    if (
      cd "$tmpdir"
      samtools view -b -@ "$THREADS" "$bam_spec" "${regions[@]}" \
        | samtools sort -@ "$THREADS" -o "$tmp_out"
    ); then
      rc=0
    else
      rc=$?
    fi

    cleanup_tmp "$tmpdir"

    if (( rc != 0 )); then
      log_warn "Remote extraction failed on attempt ${attempt}/${RETRIES}"
      rm_if_exists "$tmp_out"
      if (( attempt < RETRIES )); then
        sleep_s=$(( RETRY_SLEEP * attempt ))
        log_warn "Retrying extraction in ${sleep_s}s..."
        sleep "$sleep_s"
        attempt=$(( attempt + 1 ))
        continue
      fi
      log_error "Remote extraction failed after ${RETRIES} attempts"
      return 1
    fi

    if ! samtools quickcheck "$tmp_out"; then
      log_warn "Extracted BAM failed quickcheck on attempt ${attempt}/${RETRIES}"
      rm_if_exists "$tmp_out"
      if (( attempt < RETRIES )); then
        sleep_s=$(( RETRY_SLEEP * attempt ))
        log_warn "Retrying extraction in ${sleep_s}s..."
        sleep "$sleep_s"
        attempt=$(( attempt + 1 ))
        continue
      fi
      log_error "Extracted BAM failed quickcheck after ${RETRIES} attempts"
      return 1
    fi

    mv -f "$tmp_out" "$abs_out"
    return 0
  done

  log_error "Exhausted extraction loop unexpectedly"
  return 1
}

run_local_extract_sort() {
  local src_bam="$1"
  shift
  local out_bam="$1"
  shift
  local -a regions=("$@")

  local abs_out
  abs_out=$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$out_bam")
  local out_dir
  out_dir=$(dirname "$abs_out")
  local out_base
  out_base=$(basename "$abs_out")
  local tmp_out

  samtools quickcheck "$src_bam"

  tmp_out=$(mktemp "${out_dir}/.${out_base}.local.XXXXXX.bam")

  log_step "Extracting regional BAM from local source..."
  if samtools view -b -@ "$THREADS" "$src_bam" "${regions[@]}" \
      | samtools sort -@ "$THREADS" -o "$tmp_out"; then
    :
  else
    rm_if_exists "$tmp_out"
    log_error "Local regional extraction failed"
    return 1
  fi

  samtools quickcheck "$tmp_out"
  mv -f "$tmp_out" "$abs_out"
}

illumina_meta() {
  case "$1" in
    HG002)
      FAMILY=AshkenazimTrio
      SUBDIR=HG002_NA24385_son/NIST_Illumina_2x250bps/novoalign_bams
      BAM=HG002.GRCh38.2x250.bam
      ILLUMINA_LABEL=2x250
      ;;
    HG003)
      FAMILY=AshkenazimTrio
      SUBDIR=HG003_NA24149_father/NIST_Illumina_2x250bps/novoalign_bams
      BAM=HG003.GRCh38.2x250.bam
      ILLUMINA_LABEL=2x250
      ;;
    HG004)
      FAMILY=AshkenazimTrio
      SUBDIR=HG004_NA24143_mother/NIST_Illumina_2x250bps/novoalign_bams
      BAM=HG004.GRCh38.2x250.bam
      ILLUMINA_LABEL=2x250
      ;;
    HG005)
      FAMILY=ChineseTrio
      SUBDIR=HG005_NA24631_son/HG005_NA24631_son_HiSeq_300x/NHGRI_Illumina300X_Chinesetrio_novoalign_bams
      BAM=HG005.GRCh38_full_plus_hs38d1_analysis_set_minus_alts.300x.bam
      ILLUMINA_LABEL=300x
      ;;
    *)
      log_error "unsupported sample for Illumina: $1"
      exit 1
      ;;
  esac
}

ont_canonical_run() {
  case "$1" in
    HG002) echo "PAW70337" ;;
    HG003) echo "PAY87794" ;;
    HG004) echo "PAY87778" ;;
    HG005) echo "PAW87816" ;;
    *)
      log_error "unsupported sample for ONT canonical run: $1"
      exit 1
      ;;
  esac
}

write_metadata_json() {
  local json_path="$1"
  local output_bam="$2"
  local output_bai="$3"
  local technology="$4"
  local source_type="$5"
  local source_url="$6"
  local selection_policy="$7"
  local basecalling="$8"
  local run_ids_csv="$9"
  local extra_note="${10}"

  python3 - "$json_path" "$output_bam" "$output_bai" "$SAMPLE" "$technology" "$source_type" "$source_url" \
    "$selection_policy" "$basecalling" "$run_ids_csv" "$REGION_TAG" "$TIMESTAMP_UTC" "$SCRIPT_NAME" "$extra_note" <<'PY'
import json, sys
(
    json_path, output_bam, output_bai, sample, technology, source_type, source_url,
    selection_policy, basecalling, run_ids_csv, region_tag, timestamp_utc,
    script_name, extra_note
) = sys.argv[1:]
regions = [r for r in region_tag.split("_") if r]
runs = [r for r in run_ids_csv.split(",") if r]
payload = {
    "sample": sample,
    "technology": technology,
    "regions": regions,
    "source_type": source_type,
    "source": source_url,
    "selection_policy": selection_policy,
    "basecalling": None if basecalling == "" else basecalling,
    "run_ids": runs,
    "output_bam": output_bam,
    "output_bai": output_bai,
    "created_utc": timestamp_utc,
    "script": script_name,
    "notes": extra_note,
}
with open(json_path, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2, ensure_ascii=False)
    fh.write("\n")
PY
}

if [[ "$TECH" == "Illumina" ]]; then
  illumina_meta "$SAMPLE"
  BASE="https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/data/${FAMILY}/${SUBDIR}"
  BAM_URL="${BASE}/${BAM}"
  IDX_URL="${BAM_URL}.bai"
  BAM_SPEC="${BAM_URL}##idx##${IDX_URL}"
  OUT_DIR="${OUT_ROOT}/${SAMPLE}/Illumina"
  mkdir -p "$OUT_DIR"
  OUT_BAM="${OUT_DIR}/${SAMPLE}.GRCh38.${ILLUMINA_LABEL}.${REGION_TAG}.bam"
  OUT_BAI="${OUT_BAM}.bai"
  OUT_JSON="${OUT_BAM%.bam}.metadata.json"
  SELECTION_POLICY="canonical_giab_bam"

  log_info "Sample:   $SAMPLE"
  log_info "Tech:     Illumina"
  log_info "Regions:  ${REGION_SPACE}"
  log_info "BAM URL:  ${BAM_URL}"
  log_info "OUT BAM:  ${OUT_BAM}"

  if [[ "$SAMPLE" == "HG005" ]]; then
    STAGE_DIR="${STAGING_ROOT}/${SAMPLE}/Illumina"
    mkdir -p "$STAGE_DIR"
    FULL_BAM="${STAGE_DIR}/${BAM}"
    FULL_BAI="${FULL_BAM}.bai"

    log_info "HG005 Illumina special path: full BAM download + local regional extraction"
    download_file_resume "$BAM_URL" "$FULL_BAM"
    download_file_resume "$IDX_URL" "$FULL_BAI"

    log_step "Verifying staged full BAM..."
    samtools quickcheck "$FULL_BAM"

    run_local_extract_sort "$FULL_BAM" "$OUT_BAM" "${REGIONS[@]}"

    SOURCE_TYPE="full_bam_download_then_local_region_extract"
    EXTRA_NOTE="HG005 Illumina handled via full BAM+BAI download to staging followed by local regional extraction, because remote indexed extraction proved unreliable in this environment. Output naming follows source label semantics: 300x."
  else
    run_remote_extract_sort "$BAM_SPEC" "$OUT_BAM" "${REGIONS[@]}"
    SOURCE_TYPE="remote_bam_region_extract"
    EXTRA_NOTE="Canonical GIAB/NIST Illumina BAM used as source. Output naming follows source label semantics (e.g. 2x250 for HG002-HG004). Remote index handled in isolated temp dir; extraction uses retries and promotes only BAMs that pass quickcheck."
  fi

  log_step "Indexing BAM..."
  samtools index -@ "$THREADS" "$OUT_BAM"

  log_step "Checking contents..."
  samtools idxstats "$OUT_BAM" | awk '
    $1=="*" {next}
    $3>0 {print}
  ' >&2

  write_metadata_json \
    "$OUT_JSON" \
    "$OUT_BAM" \
    "$OUT_BAI" \
    "Illumina" \
    "$SOURCE_TYPE" \
    "$BAM_URL" \
    "$SELECTION_POLICY" \
    "" \
    "" \
    "$EXTRA_NOTE"

  log_info "Metadata: ${OUT_JSON}"
  exit 0
fi

if [[ "$TECH" == "ONT" ]]; then
  OUT_DIR="${OUT_ROOT}/${SAMPLE}/ONT"
  mkdir -p "$OUT_DIR"

  if [[ ${#RUNS[@]} -eq 0 ]]; then
    RUNS+=("$(ont_canonical_run "$SAMPLE")")
    SELECTION_POLICY="canonical_ont_run"
  else
    if [[ ${#RUNS[@]} -eq 1 ]]; then
      SELECTION_POLICY="explicit_single_run"
    else
      SELECTION_POLICY="explicit_multi_run"
    fi
  fi

  [[ ${#RUNS[@]} -gt 0 ]] || { log_error "no ONT runs selected for ${SAMPLE} (${BC})"; exit 1; }

  log_info "Sample:   $SAMPLE"
  log_info "Tech:     ONT"
  log_info "BC:       $BC"
  log_info "Regions:  ${REGION_SPACE}"
  log_info "Runs:     ${RUNS[*]}"

  PER_RUN_BAMS=()
  for RUN in "${RUNS[@]}"; do
    BAM_URL="https://ont-open-data.s3.amazonaws.com/giab_2025.01/basecalling/${BC}/${SAMPLE}/${RUN}/calls.sorted.bam"
    IDX_URL="${BAM_URL}.bai"
    BAM_SPEC="${BAM_URL}##idx##${IDX_URL}"

    log_step "Resolving ONT region names from remote BAM header..."
    REMOTE_REGIONS_STR=$(resolve_regions_from_remote_bam "$BAM_SPEC" "${REGIONS[@]}")
    [[ -n "$REMOTE_REGIONS_STR" ]] || { log_error "resolved ONT region list is empty"; exit 1; }
    mapfile -t REMOTE_REGIONS <<< "$REMOTE_REGIONS_STR"

    OUT_BAM="${OUT_DIR}/${SAMPLE}.GRCh38.ONT_R104_${BC}_${RUN}.${REGION_TAG}.bam"
    OUT_BAI="${OUT_BAM}.bai"
    OUT_JSON="${OUT_BAM%.bam}.metadata.json"

    log_info "Run ${RUN} -> ${OUT_BAM}"
    log_info "Resolved remote regions: ${REMOTE_REGIONS[*]}"

    run_remote_extract_sort "$BAM_SPEC" "$OUT_BAM" "${REMOTE_REGIONS[@]}"

    log_step "Indexing BAM..."
    samtools index -@ "$THREADS" "$OUT_BAM"

    log_step "Checking contents..."
    samtools idxstats "$OUT_BAM" | awk '
      $1=="*" {next}
      $3>0 {print}
    ' >&2

    write_metadata_json \
      "$OUT_JSON" \
      "$OUT_BAM" \
      "$OUT_BAI" \
      "ONT" \
      "remote_bam_region_extract" \
      "$BAM_URL" \
      "$SELECTION_POLICY" \
      "$BC" \
      "$RUN" \
      "ONT regional BAM extracted from selected run. Remote index handled in isolated temp dir and referenced explicitly via ##idx## to avoid cwd cache collisions on calls.sorted.bam.bai. Requested regions were resolved dynamically from the BAM header/aliases before extraction. Extraction uses retries and promotes only BAMs that pass quickcheck."

    log_info "Metadata: ${OUT_JSON}"
    PER_RUN_BAMS+=("$OUT_BAM")
  done

  if [[ ${#PER_RUN_BAMS[@]} -gt 1 && "$MERGE_ONT" == true ]]; then
    RUN_TAG=$(join_by "_" "${RUNS[@]}")
    MERGED="${OUT_DIR}/${SAMPLE}.GRCh38.ONT_R104_${BC}_${RUN_TAG}.${REGION_TAG}.bam"
    MERGED_BAI="${MERGED}.bai"
    MERGED_JSON="${MERGED%.bam}.metadata.json"
    MERGED_SOURCE="https://ont-open-data.s3.amazonaws.com/giab_2025.01/basecalling/${BC}/${SAMPLE}/"

    log_info "Merging ONT runs -> ${MERGED}"
    samtools merge -@ "$THREADS" -O BAM "$MERGED" "${PER_RUN_BAMS[@]}"
    samtools quickcheck "$MERGED"
    samtools index -@ "$THREADS" "$MERGED"
    samtools idxstats "$MERGED" | awk '
      $1=="*" {next}
      $3>0 {print}
    ' >&2

    write_metadata_json \
      "$MERGED_JSON" \
      "$MERGED" \
      "$MERGED_BAI" \
      "ONT" \
      "merged_regional_bams" \
      "$MERGED_SOURCE" \
      "merged_multiple_runs" \
      "$BC" \
      "$(IFS=,; echo "${RUNS[*]}")" \
      "Merged ONT regional BAM generated from multiple selected runs."

    log_info "Metadata: ${MERGED_JSON}"
  fi
  exit 0
fi

log_error "unsupported --tech '${TECH}'"
exit 1