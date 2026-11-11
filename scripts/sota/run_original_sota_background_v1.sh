#!/usr/bin/env bash
# Detached, SSH-independent SOTA battery on Zeus. Read-only monitoring.
# start [all|NetCLR|Swallow|WFCAT|DF|VarCNN|TikTok|RoFiRe]
# status | models | follow
set -euo pipefail
cd "$(dirname "$0")/../.."
BASE=/data/datasets_brayan/ltd-sota-2026
test -f "$BASE/host_manifests/host_environment.sh" || {
  echo "BLOCKED: missing Zeus host environment"; exit 3; }
source "$BASE/host_manifests/host_environment.sh"
export LTD_SOTA_PYTORCH_PY="$HOME/modelos/venv/bin/python"
export LTD_SOTA_KERAS_PY="$BASE/environments/venv-keras-authors/bin/python"
export TMPDIR="$BASE/tmp"
export PIP_CACHE_DIR="$BASE/pip-cache"
export PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"
PY="$LTD_SOTA_PYTORCH_PY"
TRACK="$LTD_SOTA_ARTIFACT_ROOT/background_battery_v1"
mkdir -p "$TRACK"
MODE="$1"
case "$MODE" in
 start)
   MODEL="$2"
   case "$MODEL" in all|NetCLR|Swallow|WFCAT|DF|VarCNN|TikTok|RoFiRe);; *) echo "Unknown model";exit 2;;esac
   test -x "$PY" && test -x "$LTD_SOTA_KERAS_PY" || { echo "BLOCKED: missing Python interpreters";exit 3; }
   if test -f "$TRACK/latest_run";then
     OLD="$(cat "$TRACK/latest_run")"
     if test -f "$TRACK/$OLD/pid" && kill -0 "$(cat "$TRACK/$OLD/pid")" 2>/dev/null;then
        echo "BLOCKED: active prior run $OLD";exit 5
     fi
   fi
   "$PY" scripts/sota/sota_host_readiness_certificate_v1.py --stage inspect
   "$PY" scripts/sota/sota_zeus_evidence_bootstrap_v1.py --stage inspect >/dev/null
   GPU="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
   test -z "$GPU" || { echo "BLOCKED: GPU busy $GPU";exit 5; }
   echo "===== LIVE CUDA GATE BEFORE DETACH ====="
   bash scripts/sota/smoke_perseo_gpu_runtime_v1.sh
   RUN="$(date -u +%Y%m%dT%H%M%SZ)_$MODEL"
   OUT="$TRACK/$RUN"
   mkdir "$OUT"
   printf '%s\n' "$MODEL" >"$OUT/model"
   printf '%s\n' "$RUN" >"$TRACK/latest_run"
   nohup bash "$0" worker "$RUN" >"$OUT/console.log" 2>&1 </dev/null &
   printf '%s\n' "$!" >"$OUT/pid"
   echo "STARTED run=$RUN pid=$(cat "$OUT/pid")"
   echo "LOG=$OUT/console.log"
   ;;
 worker)
   RUN="$2"
   case "$RUN" in *'/'*|'') exit 2;;esac
   OUT="$TRACK/$RUN"
   MODEL="$(cat "$OUT/model")"
   exec 9>"$TRACK/single_battery.lock"
   if ! flock -n 9;then
      echo "BLOCKED: battery lock occupied";printf 'FAILED_LOCK\n' >"$OUT/status";exit 5
   fi
   printf 'RUNNING\n' >"$OUT/status"
   date -u +%FT%TZ >"$OUT/started_utc"
   git rev-parse HEAD >"$OUT/git_commit"
   finish() {
     CODE="$?"
     date -u +%FT%TZ >"$OUT/finished_utc"
     printf '%s\n' "$CODE" >"$OUT/exit_code"
     if test "$CODE" -eq 0;then
       printf 'COMPLETED\n' >"$OUT/status"
       echo "BATTERY_COMPLETED $RUN"
     else
       printf 'FAILED\n' >"$OUT/status"
       echo "BATTERY_FAILED $RUN code=$CODE"
     fi
   }
   trap finish EXIT
   echo "BATTERY_STARTED $RUN $MODEL"
   if test "$MODEL" = all;then
      "$PY" -u scripts/sota/original_sota_orchestrator_v2.py --stage all --run
   else
      "$PY" -u scripts/sota/original_sota_orchestrator_v2.py --stage all --model "$MODEL" --run
   fi
   ;;
 status|models|follow)
   if ! test -f "$TRACK/latest_run";then echo "NO_BATTERY_RUNS";exit 0;fi
   RUN="$(cat "$TRACK/latest_run")"
   OUT="$TRACK/$RUN"
   echo "RUN=$RUN MODEL=$(cat "$OUT/model") STATUS=$(cat "$OUT/status" 2>/dev/null || echo STARTING)"
   PID="$(cat "$OUT/pid" 2>/dev/null || echo unknown)"
   if kill -0 "$PID" 2>/dev/null;then echo "PID=$PID ACTIVE";else echo "PID=$PID NOT_ACTIVE";fi
   if test "$MODE" = follow;then
      tail -n 50 -f "$OUT/console.log"
   elif test "$MODE" = models;then
      grep -E 'SOTA_STAGE|_EPOCH|_PASS|_FAILED|_BLOCKED|SOTA_BATTERY_V2_EXEC' "$OUT/console.log" | tail -n 50 || true
      echo "===== MODEL LOGS ====="
      find "$LTD_SOTA_ARTIFACT_ROOT" -maxdepth 3 -type d -name logs -print
   else
      tail -n 35 "$OUT/console.log"
      nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader
   fi
   ;;
 *)
   echo "Usage: bash $0 {start MODEL|status|models|follow}";exit 2;;
esac
