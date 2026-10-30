#!/usr/bin/env bash
# LTD-C old obfs4 capture purge: EXACT directory allowlist, A/B never touched.
# --inventory is read-only. --purge additionally requires explicit environment confirmation.
set -euo pipefail
umask 077
MODE=--inventory
if (( $# > 0 )); then MODE="$1"; fi
REPORT_ROOT="$HOME/ltd-ops/dataset-c-bridge-retirement"
candidate_paths() {
  printf '%s\n' /data/ltd-c-qa-bsierra \
    "$HOME/ltd-storage/dataset-c-smoke-obfs4" \
    "$HOME/ltd-storage/dataset-c-smoke"
}
inventory() {
  echo "LTD_C_BRIDGE_PURGE_INVENTORY v1"
  echo "HOST=$(hostname) UTC_UNVERIFIED=$(date -u +%FT%TZ)"
  echo "=== Exact candidate roots ==="
  while IFS= read -r p; do
    if [[ -L "$p" ]]; then
      echo "SYMLINK_REVIEW_NOT_PURGED $p -> $(readlink -- "$p")"
    elif [[ -d "$p" ]]; then
      echo "CANDIDATE path=$p owner=$(stat -c '%u' "$p") mount=$(mountpoint -q "$p" && echo yes || echo no)"
      du -sh -- "$p"
      find "$p" -mindepth 1 -maxdepth 1 -printf '  %f\n' | head -30 || true
    elif [[ -e "$p" ]]; then
      echo "NON_DIRECTORY_REVIEW_NOT_PURGED $p"
    else
      echo "ABSENT $p"
    fi
  done < <(candidate_paths)
  echo "=== Running C containers ==="
  docker ps --format '{{.Names}} {{.Status}}' | grep -E '^ltd-c-' || true
  echo "=== Matching C cron entries ==="
  (crontab -l 2>/dev/null || true) | grep -Ei 'LTD_C_|dataset.c|run_hourly_qa|three_hourly_qa|watchdog' || true
  echo "=== Existing C Docker volumes (NOT touched) ==="
  docker volume ls --format '{{.Name}}' | grep -E '^ltd-c-' || true
  echo "=== Additional top-level C directories to inspect; NEVER auto-purged ==="
  find /data -mindepth 1 -maxdepth 2 -type d \( -iname '*dataset*c*' -o -iname 'ltd-c-*' \) -print 2>/dev/null | head -45 || true
}
guard() {
  [[ "$(printenv LTD_C_PURGE_CONFIRMED 2>/dev/null || true)" == "YES_DELETE_ONLY_OLD_OBFS4_C" ]] ||
    { echo "STOP: explicit confirmation env value required"; exit 3; }
  [[ -d /data ]] || { echo "STOP: /data missing"; exit 3; }
  local active
  active="$(docker ps --format '{{.Names}}' | grep -E '^ltd-c-' || true)"
  [[ -z "$active" ]] || { echo "STOP: active C containers: $active"; exit 3; }
  local cron
  cron="$(crontab -l 2>/dev/null | grep -Ev '^[[:space:]]*(#|$)' | grep -Ei 'LTD_C_|dataset.c|run_hourly_qa|three_hourly_qa|watchdog' || true)"
  [[ -z "$cron" ]] || { echo "STOP: active C cron; deactivate before deletion"; exit 3; }
  for n in ltd-c-tor-inst1 ltd-c-tor-inst2 ltd-c-tor-inst3 ltd-c-manual-tor; do
    if docker inspect "$n" >/dev/null 2>&1; then
      [[ "$(docker inspect --format '{{.State.Running}}' "$n")" == false ]] ||
        { echo "STOP: running $n"; exit 3; }
    fi
  done
}
purge() {
  guard
  mkdir -p -- "$REPORT_ROOT"
  chmod 700 "$REPORT_ROOT"
  local file="$REPORT_ROOT/purge_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
  {
    echo "RESEARCHER_AUTHORIZED_OLD_BRIDGE_DATA_PURGE=YES"
    echo "HOST=$(hostname) UTC_UNVERIFIED=$(date -u +%FT%TZ)"
    echo "POLICY=EXACT_ALLOWLIST_ROOTS_ONLY; no A/B, models, collector source or repo"
  } > "$file"
  chmod 600 "$file"
  local p total=0
  while IFS= read -r p; do
    if [[ -L "$p" ]]; then echo "SKIP_SYMLINK $p" | tee -a "$file"; continue; fi
    if [[ ! -e "$p" ]]; then echo "ABSENT $p" | tee -a "$file"; continue; fi
    [[ -d "$p" && "$(stat -c '%u' "$p")" == "$(id -u)" ]] ||
      { echo "STOP: not an owned directory: $p"; exit 3; }
    [[ "$(realpath -e -- "$p")" == "$p" ]] ||
      { echo "STOP: path resolves elsewhere: $p"; exit 3; }
    if mountpoint -q "$p"; then echo "STOP: mount root: $p"; exit 3; fi
    case "$p" in
      /data/ltd-c-qa-bsierra|"$HOME"/ltd-storage/dataset-c-smoke-obfs4|"$HOME"/ltd-storage/dataset-c-smoke) ;;
      *) echo "STOP: outside explicit allowlist: $p"; exit 3 ;;
    esac
    local bytes files
    bytes="$(du -sb -- "$p" | cut -f1)"
    files="$(find "$p" -xdev -type f 2>/dev/null | wc -l)"
    echo "START_DELETE path=$p bytes=$bytes regular_files=$files" | tee -a "$file"
    if [[ "$p" == /data/ltd-c-qa-bsierra ]]; then
      # Keep the user-owned /data directory because recreating it may require root.
      shopt -s dotglob nullglob
      for item in "$p"/*; do
        rm -rf --one-file-system -- "$item"
      done
      shopt -u dotglob nullglob
      [[ -z "$(ls -A -- "$p")" ]] || { echo "STOP: root not fully emptied: $p"; exit 5; }
      echo "EMPTIED_ROOT_RETAINED $p" | tee -a "$file"
    else
      rm -rf --one-file-system -- "$p"
      [[ ! -e "$p" ]] || { echo "STOP: deletion incomplete: $p"; exit 5; }
      echo "DELETED $p" | tee -a "$file"
    fi
    total=$((total+1))
  done < <(candidate_paths)
  echo "BRIDGE_DATA_PURGE_COMPLETE count=$total; REPORT=$file" | tee -a "$file"
  printf '%s\n' "$file" > "$REPORT_ROOT/last_success.txt"
}
case "$MODE" in
  --inventory) inventory ;;
  --purge) purge ;;
  *) echo "Usage: $0 --inventory | --purge"; exit 2 ;;
esac
