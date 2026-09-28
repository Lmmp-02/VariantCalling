#!/usr/bin/env bash
set -euo pipefail

MANIFEST_PATH="${1:-data/5_scripts/config/downsample_multimodal_benchmark.tsv}"
THREADS="${THREADS:-8}"
CONDA_ENV="${CONDA_ENV:-hts}"
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"
MICROMAMBA_BIN="${MICROMAMBA_BIN:-/home/lantik-deploy/jlazaro/bin/micromamba}"
MOSDEPTH_ROOT="${MOSDEPTH_ROOT:-data/4_out/mosdepth_multimodal_benchmark}"

echo "[INFO] Manifest   : $MANIFEST_PATH"
echo "[INFO] Threads    : $THREADS"
echo "[INFO] Env        : $CONDA_ENV"
echo "[INFO] Dry run    : $DRY_RUN"
echo "[INFO] Force      : $FORCE"
echo "[INFO] Micromamba : $MICROMAMBA_BIN"
echo "[INFO] Mosdepth   : $MOSDEPTH_ROOT"

if [[ ! -f "$MANIFEST_PATH" ]]; then
  echo "[ERROR] Manifest not found: $MANIFEST_PATH" >&2
  exit 1
fi

if [[ ! -x "$MICROMAMBA_BIN" ]]; then
  echo "[ERROR] micromamba not found or not executable at: $MICROMAMBA_BIN" >&2
  exit 1
fi

eval "$("$MICROMAMBA_BIN" shell hook --shell bash)" || {
  echo "[ERROR] Failed to initialize micromamba shell hook." >&2
  exit 1
}

micromamba activate "$CONDA_ENV" || {
  echo "[ERROR] Failed to activate micromamba environment: $CONDA_ENV" >&2
  exit 1
}

command -v samtools >/dev/null 2>&1 || { echo "[ERROR] samtools not found" >&2; exit 1; }
command -v mosdepth >/dev/null 2>&1 || { echo "[ERROR] mosdepth not found" >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "[ERROR] python3 not found" >&2; exit 1; }

echo "[OK] Environment looks good."

strip_cr() {
  printf '%s' "${1//$'\r'/}"
}

get_mean_depth() {
  local summary_file="$1"
  awk '$1=="total"{print $4}' "$summary_file"
}

calc_fraction() {
  local target_cov="$1"
  local mean_cov="$2"
  python3 - "$target_cov" "$mean_cov" <<'PY'
import sys

target = float(sys.argv[1])
mean = float(sys.argv[2])

if mean <= 0:
    raise SystemExit("mean coverage must be > 0")

frac = target / mean
print(f"{frac:.6f}")
PY
}

get_source_mosdepth_prefix() {
  local sample="$1"
  local regions="$2"
  local tech="$3"
  local input_bam="$4"

  local input_base input_name
  input_base="$(basename "$input_bam")"
  input_name="${input_base%.bam}"

  printf '%s' "${MOSDEPTH_ROOT}/${sample}/${regions}/sources/${tech}/${input_name}.source"
}

ensure_source_mosdepth() {
  local sample="$1"
  local regions="$2"
  local tech="$3"
  local input_bam="$4"

  local prefix summary_file
  prefix="$(get_source_mosdepth_prefix "$sample" "$regions" "$tech" "$input_bam")"
  summary_file="${prefix}.mosdepth.summary.txt"

  mkdir -p "$(dirname "$prefix")"

  if [[ -f "$summary_file" && "$FORCE" != "1" ]]; then
    echo "[SOURCE] Reusing existing mosdepth summary: $summary_file" >&2
    printf '%s\n' "$summary_file"
    return 0
  fi

  echo "[SOURCE] Running mosdepth on source BAM:" >&2
  echo "[SOURCE]   bam    : $input_bam" >&2
  echo "[SOURCE]   prefix : $prefix" >&2

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[SOURCE] DRY_RUN=1 -> mosdepth not executed" >&2
    printf '%s\n' "$summary_file"
    return 0
  fi

  mosdepth -t "$THREADS" -n "$prefix" "$input_bam"

  if [[ ! -f "$summary_file" ]]; then
    echo "[ERROR] Source mosdepth summary not created: $summary_file" >&2
    exit 1
  fi

  printf '%s\n' "$summary_file"
}

build_seed_fraction() {
  local seed="$1"
  local frac="$2"

  python3 - "$seed" "$frac" <<'PY'
import sys

seed = sys.argv[1]
frac = float(sys.argv[2])

if not (0 < frac <= 1.0):
    raise SystemExit(f"downsampling fraction must be in (0,1], got {frac}")

frac_str = f"{frac:.6f}".split(".")[1]
print(f"{seed}.{frac_str}")
PY
}

ensure_downsampled_bam() {
  local input_bam="$1"
  local outbam="$2"
  local outmeta="$3"
  local mosprefix="$4"
  local target_cov="$5"
  local mean_cov="$6"
  local seed="$7"

  local outbai
  outbai="${outbam}.bai"

  local frac seed_frac
  frac="$(calc_fraction "$target_cov" "$mean_cov")"

  if python3 - "$frac" <<'PY'
import sys
frac = float(sys.argv[1])
sys.exit(0 if 0 < frac <= 1.0 else 1)
PY
  then
    :
  else
    echo "[ERROR] Computed fraction out of range for $outbam (target=${target_cov}, mean=${mean_cov}, frac=${frac})" >&2
    exit 1
  fi

  seed_frac="$(build_seed_fraction "$seed" "$frac")"

  if [[ -f "$outbam" && -f "$outbai" && -f "${mosprefix}.mosdepth.summary.txt" && -f "$outmeta" && "$FORCE" != "1" ]]; then
    echo "[RUN ] Reusing existing derived BAM: $outbam" >&2
    echo "[RUN ] Reusing existing coverage summary: ${mosprefix}.mosdepth.summary.txt" >&2
    echo "[RUN ] Reusing existing metadata: $outmeta" >&2
    printf '%s\n' "$frac"
    return 0
  fi

  if [[ -f "$outbam" && -f "$outbai" && -f "${mosprefix}.mosdepth.summary.txt" && "$FORCE" != "1" ]]; then
    echo "[RUN ] Reusing existing derived BAM and summary, metadata will be regenerated: $outbam" >&2
    printf '%s\n' "$frac"
    return 0
  fi

  echo "[RUN ] Downsampling BAM:" >&2
  echo "[RUN ]   input_bam  : $input_bam" >&2
  echo "[RUN ]   outbam     : $outbam" >&2
  echo "[RUN ]   target_cov : ${target_cov}x" >&2
  echo "[RUN ]   mean_cov   : ${mean_cov}x" >&2
  echo "[RUN ]   fraction   : $frac" >&2
  echo "[RUN ]   seed_frac  : $seed_frac" >&2

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[RUN ] DRY_RUN=1 -> downsampling not executed" >&2
    printf '%s\n' "$frac"
    return 0
  fi

  mkdir -p "$(dirname "$outbam")" "$(dirname "$mosprefix")"

  echo "[RUN ] samtools view | samtools sort ..." >&2
  samtools view -@ "$THREADS" -s "$seed_frac" -b "$input_bam" \
    | samtools sort -@ "$THREADS" -o "$outbam" -

  echo "[RUN ] samtools index ..." >&2
  samtools index -@ "$THREADS" "$outbam"

  echo "[RUN ] samtools quickcheck ..." >&2
  samtools quickcheck -v "$outbam"

  echo "[RUN ] mosdepth derived BAM ..." >&2
  mosdepth -t "$THREADS" -n "$mosprefix" "$outbam"

  if [[ ! -f "$outbai" ]]; then
    echo "[ERROR] BAM index not created: $outbai" >&2
    exit 1
  fi

  if [[ ! -f "${mosprefix}.mosdepth.summary.txt" ]]; then
    echo "[ERROR] Derived mosdepth summary not created: ${mosprefix}.mosdepth.summary.txt" >&2
    exit 1
  fi

  printf '%s\n' "$frac"
}

write_metadata_json() {
  local metadata_path="$1"
  local sample="$2"
  local tech="$3"
  local source_policy="$4"
  local regions="$5"
  local input_bam="$6"
  local output_group="$7"
  local target_cov="$8"
  local seed="$9"
  local notes="${10}"
  local source_summary="${11}"
  local derived_summary="${12}"
  local outbam="${13}"
  local outbai="${14}"
  local frac="${15}"
  local seed_frac="${16}"

  mkdir -p "$(dirname "$metadata_path")"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[META] DRY_RUN=1 -> metadata not written: $metadata_path" >&2
    return 0
  fi

  python3 - "$metadata_path" "$sample" "$tech" "$source_policy" "$regions" "$input_bam" \
    "$output_group" "$target_cov" "$seed" "$notes" "$source_summary" "$derived_summary" \
    "$outbam" "$outbai" "$frac" "$seed_frac" <<'PY'
import csv
import json
import sys
from pathlib import Path

(
    metadata_path,
    sample,
    tech,
    source_policy,
    regions,
    input_bam,
    output_group,
    target_cov,
    seed,
    notes,
    source_summary,
    derived_summary,
    outbam,
    outbai,
    frac,
    seed_frac,
) = sys.argv[1:17]

def parse_summary(path_str: str) -> dict:
    path = Path(path_str)
    if not path.exists():
        raise SystemExit(f"Summary file not found: {path}")

    by_chrom = {}
    total_mean = None

    with path.open() as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            chrom = row["chrom"]
            mean = float(row["mean"])
            if chrom == "total":
                total_mean = mean
            else:
                by_chrom[chrom] = mean

    if total_mean is None:
        raise SystemExit(f"No total row found in summary: {path}")

    return {
        "path": str(path),
        "total_mean": total_mean,
        "by_chrom_mean": by_chrom,
    }

derived = parse_summary(derived_summary)

payload = {
    "sample": sample,
    "technology": tech,
    "regions": regions.split("_"),
    "output_group": output_group,
    "source_policy": source_policy,
    "parent_bam": input_bam,
    "output_bam": outbam,
    "output_bai": outbai,
    "target_coverage": int(target_cov),
    "observed_coverage": derived["total_mean"],
    "observed_coverage_by_chrom": derived["by_chrom_mean"],
    "seed": int(seed),
    "fraction": float(frac),
    "seed_fraction": seed_frac,
    "source_mosdepth_summary": source_summary,
    "derived_mosdepth_summary": derived["path"],
    "notes": notes,
}

Path(metadata_path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

  echo "[META] Wrote metadata: $metadata_path" >&2
}

process_row() {
  local sample="$1"
  local tech="$2"
  local source_policy="$3"
  local regions="$4"
  local input_bam="$5"
  local output_group="$6"
  local target_cov="$7"
  local seed="$8"
  local notes="$9"

  local input_base input_name input_stem
  input_base="$(basename "$input_bam")"
  input_name="${input_base%.bam}"
  input_stem="${input_name}"

  local outdir mosdir
  outdir="data/1_input_bams/${sample}/derived/${regions}/${output_group}/${tech}"
  mosdir="${MOSDEPTH_ROOT}/${sample}/${regions}/${output_group}/${tech}"

  local outbam outbai outmeta mosprefix
  outbam="${outdir}/${input_stem}.cov${target_cov}x.s${seed}.bam"
  outbai="${outbam}.bai"
  outmeta="${outdir}/${input_stem}.cov${target_cov}x.s${seed}.metadata.json"
  mosprefix="${mosdir}/${input_stem}.cov${target_cov}x.s${seed}"

  mkdir -p "$outdir" "$mosdir"

  echo ""
  echo "------------------------------------------------------------"
  echo "[PLAN] sample        : $sample"
  echo "[PLAN] tech          : $tech"
  echo "[PLAN] source_policy : $source_policy"
  echo "[PLAN] regions       : $regions"
  echo "[PLAN] input_bam     : $input_bam"
  echo "[PLAN] output_group  : $output_group"
  echo "[PLAN] target_cov    : ${target_cov}x"
  echo "[PLAN] seed          : $seed"
  echo "[PLAN] notes         : $notes"
  echo "[PLAN] outdir        : $outdir"
  echo "[PLAN] outbam        : $outbam"
  echo "[PLAN] outbai        : $outbai"
  echo "[PLAN] outmeta       : $outmeta"
  echo "[PLAN] mosdepth_dir  : $mosdir"
  echo "[PLAN] mosprefix     : $mosprefix"

  local source_summary mean_cov frac observed_cov seed_frac derived_summary
  source_summary="$(ensure_source_mosdepth "$sample" "$regions" "$tech" "$input_bam")"

  if [[ ! -f "$source_summary" && "$DRY_RUN" != "1" ]]; then
    echo "[ERROR] Source mosdepth summary not found: $source_summary" >&2
    exit 1
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[PLAN] source_sum    : $source_summary"
    echo "[PLAN] source_mean   : DRY_RUN_PENDING"
    echo "[PLAN] fraction      : DRY_RUN_PENDING"
    echo "[PLAN] seed_frac     : DRY_RUN_PENDING"
    echo "[PLAN] obs_mean      : DRY_RUN_PENDING"
    echo "[PLAN] status        : READY"
    return 0
  fi

  mean_cov="$(get_mean_depth "$source_summary")"
  if [[ -z "${mean_cov:-}" ]]; then
    echo "[ERROR] Could not read mean coverage from: $source_summary" >&2
    exit 1
  fi

  frac="$(ensure_downsampled_bam "$input_bam" "$outbam" "$outmeta" "$mosprefix" "$target_cov" "$mean_cov" "$seed")"
  seed_frac="$(build_seed_fraction "$seed" "$frac")"
  derived_summary="${mosprefix}.mosdepth.summary.txt"

  observed_cov="$(get_mean_depth "$derived_summary")"
  if [[ -z "${observed_cov:-}" ]]; then
    echo "[ERROR] Could not read observed coverage from: $derived_summary" >&2
    exit 1
  fi

  write_metadata_json \
    "$outmeta" "$sample" "$tech" "$source_policy" "$regions" "$input_bam" \
    "$output_group" "$target_cov" "$seed" "$notes" "$source_summary" "$derived_summary" \
    "$outbam" "$outbai" "$frac" "$seed_frac"

  echo "[PLAN] source_sum    : $source_summary"
  echo "[PLAN] source_mean   : ${mean_cov}x"
  echo "[PLAN] fraction      : $frac"
  echo "[PLAN] seed_frac     : $seed_frac"
  echo "[PLAN] obs_mean      : ${observed_cov}x"
  echo "[PLAN] status        : DONE"
}

row_count=0
enabled_count=0
planned_count=0

while IFS=$'\t' read -r enabled sample tech source_policy regions input_bam output_group target_cov seed notes || \
      [[ -n "${enabled:-}${sample:-}${tech:-}${source_policy:-}${regions:-}${input_bam:-}${output_group:-}${target_cov:-}${seed:-}${notes:-}" ]]; do

  enabled="$(strip_cr "${enabled:-}")"
  sample="$(strip_cr "${sample:-}")"
  tech="$(strip_cr "${tech:-}")"
  source_policy="$(strip_cr "${source_policy:-}")"
  regions="$(strip_cr "${regions:-}")"
  input_bam="$(strip_cr "${input_bam:-}")"
  output_group="$(strip_cr "${output_group:-}")"
  target_cov="$(strip_cr "${target_cov:-}")"
  seed="$(strip_cr "${seed:-}")"
  notes="$(strip_cr "${notes:-}")"

  if [[ "$enabled" == "enabled" ]]; then
    continue
  fi

  if [[ -z "$enabled" ]]; then
    continue
  fi

  row_count=$((row_count + 1))

  if [[ "$enabled" != "1" ]]; then
    continue
  fi

  enabled_count=$((enabled_count + 1))

  if [[ ! -f "$input_bam" ]]; then
    echo "[ERROR] Input BAM not found for row $row_count: $input_bam" >&2
    exit 1
  fi

  case "$tech" in
    Illumina|ONT) ;;
    *)
      echo "[ERROR] Invalid tech at row $row_count: $tech" >&2
      exit 1
      ;;
  esac

  case "$output_group" in
    harmonized_40x|coverage_study) ;;
    *)
      echo "[ERROR] Invalid output_group at row $row_count: $output_group" >&2
      exit 1
      ;;
  esac

  if [[ ! "$target_cov" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] Invalid target_cov at row $row_count: $target_cov" >&2
    exit 1
  fi

  if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] Invalid seed at row $row_count: $seed" >&2
    exit 1
  fi

  process_row \
    "$sample" "$tech" "$source_policy" "$regions" "$input_bam" \
    "$output_group" "$target_cov" "$seed" "$notes"

  planned_count=$((planned_count + 1))
done < "$MANIFEST_PATH"

echo ""
echo "============================================================"
echo "[SUMMARY] rows_seen     : $row_count"
echo "[SUMMARY] rows_enabled  : $enabled_count"
echo "[SUMMARY] rows_planned  : $planned_count"
echo "[SUMMARY] mode          : $([[ "$DRY_RUN" == "1" ]] && echo DRY_RUN || echo DOWNSAMPLING_EXECUTION)"
echo "============================================================"