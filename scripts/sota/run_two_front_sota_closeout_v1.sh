#!/usr/bin/env bash
# One-shot two-front SOTA closeout. Frozen results remain immutable.
# Explicit --full-swallow selects GPU full v2 after controlled paired pilot.
set -euo pipefail
cd "$(dirname "$0")/../.."
BASE="$(printenv LTD_SOTA_BASE || true)"
if [[ -z "$BASE" ]];then BASE="/data/datasets_brayan/ltd-sota-2026";fi
if [[ -f "$BASE/host_manifests/host_environment.sh" ]];then
 source "$BASE/host_manifests/host_environment.sh"
fi
PY="$(printenv LTD_SOTA_PYTORCH_PY || true)"
if [[ -z "$PY" ]];then PY="$HOME/modelos/venv/bin/python";fi
[[ -x "$PY" ]] || { echo "BLOCKED_PYTORCH_RUNTIME $PY";exit 5; }
AUD="$BASE/audits"
CMP="$AUD/ltd_vs_seven_sota_20261113T095045Z.json"
FUN="$AUD/author_fidelity_recovered_20261113T102600Z/certificate.json"
BC="$AUD/bulk_BC_reconciled_20261113T104446Z.json"
for f in "$CMP" "$FUN" "$BC";do
  [[ -f "$f" ]] || { echo "BLOCKED_MISSING_FROZEN_EVIDENCE $f";exit 5; }
done
MODE="--pilot-only"
if [[ "$#" -ge 1 ]];then MODE="$1";fi
[[ "$MODE" == "--full-swallow" || "$MODE" == "--pilot-only" ]] || {
 echo "Usage: bash $0 [--pilot-only|--full-swallow]";exit 2; }
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
MATRIX="$AUD/final_six_method_matrix_$STAMP"
PILOT="$AUD/swallow_target_BN_pilot_$STAMP"
echo "===== FRONT A: SIX-METHOD ARTICLE SCOPE FREEZE ====="
"$PY" scripts/sota/final_six_method_scope_matrix_v1.py \
 --comparison "$CMP" --functional-certificate "$FUN" \
 --reconciled-bc "$BC" --output-dir "$MATRIX"
echo "===== FRONT B: ORIGINAL SWALLOW TARGET-BN CAUSAL PILOT ====="
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
 "$PY" scripts/sota/swallow_byol_target_bn_controlled_pilot_v1.py \
 --repo "$PWD" --base "$BASE" --steps 120 --batch 32 --output-dir "$PILOT"
echo "===== SWALLOW PILOT INTERPRETATION ====="
"$PY" - "$PILOT/result.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
for name,a in d["arms"].items():
 f=a["probes"][-1]
 print(name,"STEP",f["step"],
       "FEATURE_STD",f["mean_inter_capture_feature_std"],
       "UNIQUE",f["unique_embeddings_rounded_6"])
print("PILOT_ONLY_NOT_FULL_CERTIFICATION",sys.argv[1])
PY
if [[ "$MODE" == "--pilot-only" ]];then
  echo "PILOT_ONLY_COMPLETE"
  echo "METHOD_MATRIX=$MATRIX/method_matrix.json"
  echo "SWALLOW_PILOT=$PILOT/result.json"
  exit 0
fi
echo "===== V2 FULL TRAIN GATE ====="
"$PY" - "$PILOT/result.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
r=d["arms"]["AUTHOR_TARGET_TRAIN"]["probes"][-1]
assert r["unique_embeddings_rounded_6"]>=3 and r["mean_inter_capture_feature_std"]>=1e-7, (
  "BLOCKED: author-target-train pilot already lost representation diversity")
print("V2_HISTORICAL_TRAIN_AUTHORIZED_BY_NONCOLLAPSED_PILOT")
PY
V2="$BASE/artifacts/sota/s11_swallow_authors_BNtrain_v2_seed42"
if [[ -e "$V2" ]];then
  echo "BLOCKED_V2_OUTPUT_EXISTS_NO_OVERWRITE $V2"
  echo "Inspect progress; this runner never restarts incomplete experiments."
  exit 5
fi
echo "===== V2 SWALLOW: HISTORICAL PRETRAIN B ====="
"$PY" scripts/sota/s11_swallow_authors_bntrain_v2.py --stage pretrain-B
echo "===== V2 SWALLOW: HISTORICAL SUPERVISED B ====="
"$PY" scripts/sota/s11_swallow_authors_bntrain_v2.py --stage supervised-B
echo "===== V2 SWALLOW: HISTORICAL REFIT C ====="
"$PY" scripts/sota/s11_swallow_authors_bntrain_v2.py --stage C
echo "===== V2 SWALLOW: FROZEN EXTERNAL INFERENCE ====="
"$PY" scripts/sota/s11_swallow_authors_bntrain_v2.py --stage evaluate --cohort 09B
"$PY" scripts/sota/s11_swallow_authors_bntrain_v2.py --stage evaluate --cohort 10B_DEV_FROZEN
REL="$AUD/swallow_BNtrain_v2_release_$STAMP.json"
echo "===== V2 SWALLOW RELEASE GATE ====="
CUDA_VISIBLE_DEVICES=-1 "$PY" \
 scripts/sota/certify_swallow_bntrain_v2_release_v1.py \
 --repo "$PWD" --base "$BASE" --output "$REL"
echo "===== V2 SWALLOW CANONICAL PAIRED STATISTICS ====="
STATS="$AUD/swallow_BNtrain_v2_vs_LTD_$STAMP.json"
CUDA_VISIBLE_DEVICES=-1 "$PY" \
 scripts/sota/compare_swallow_bntrain_v2_ltd_canonical_v1.py \
 --repo "$PWD" --base "$BASE" --release "$REL" --output "$STATS"
echo "TWO_FRONT_SOTA_COMPLETE"
echo "METHOD_MATRIX=$MATRIX/method_matrix.json"
echo "SWALLOW_PILOT=$PILOT/result.json"
echo "SWALLOW_RELEASE=$REL"
echo "SWALLOW_COMPARISON=$STATS"
