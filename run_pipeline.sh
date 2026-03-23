#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# run_pipeline.sh
#
# Runs DeepVariant → hap.py sequentially for every (tech, coverage) combination:
#   Techs  : Illumina, ONT
#   Covs   : orig, 30, 15, 10, 5
#   Seed   : 42 (for downsampled; "orig" for originals)
#
# Usage:
#   bash run_pipeline.sh
#
# Requirements: run_deepvariant.sh and run_happy.sh must be in the same directory
# =============================================================================

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
SEED=42
TECHS=(Illumina ONT)
COVS=(orig 30 15 10 5)

# Absolute paths derived from the project root (where this script lives)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DV_SCRIPT="${PROJECT_ROOT}/data/5_scripts/run_deepvariant.sh"
HAPPY_SCRIPT="${PROJECT_ROOT}/data/5_scripts/run_happy.sh"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run() {
  if $DRY_RUN; then
    echo "[DRY-RUN] $*"
  else
    "$@"
  fi
}

# ----------------------------------------------------------------------------
# Sanity checks
# ----------------------------------------------------------------------------
[[ -f "$DV_SCRIPT" ]]    || { echo "ERROR: Not found: $DV_SCRIPT"; exit 1; }
[[ -f "$HAPPY_SCRIPT" ]] || { echo "ERROR: Not found: $HAPPY_SCRIPT"; exit 1; }

# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------
TOTAL=$(( ${#TECHS[@]} * ${#COVS[@]} ))
JOB=0

for TECH in "${TECHS[@]}"; do
  for COV in "${COVS[@]}"; do
    JOB=$(( JOB + 1 ))
    log "──────────────────────────────────────────────────────"
    log "Job ${JOB}/${TOTAL} │ TECH=${TECH}  COV=${COV}"
    log "──────────────────────────────────────────────────────"

    # Build seed argument: "orig" for the original BAM, 42 for downsamples
    if [[ "$COV" == "orig" ]]; then
      SEED_ARG="orig"
    else
      SEED_ARG="$SEED"
    fi

    # --- DeepVariant ---
    log "  → DeepVariant (tech=${TECH}, cov=${COV}, seed=${SEED_ARG})"
    run bash "$DV_SCRIPT" \
      --tech   "$TECH" \
      --cov    "$COV" \
      --seed   "$SEED_ARG"

    # --- hap.py ---
    log "  → hap.py (tech=${TECH}, cov=${COV}, seed=${SEED_ARG})"
    run bash "$HAPPY_SCRIPT" \
      --tech   "$TECH" \
      --cov    "$COV" \
      --seed   "$SEED_ARG"

    log "  ✓ Done: TECH=${TECH}  COV=${COV}"
  done
done

log "══════════════════════════════════════════════════════"
log "All ${TOTAL} jobs completed."
log "══════════════════════════════════════════════════════"
