#!/usr/bin/env bash
# Read-only recovery of existing Keras runtime on Zeus; new isolated venv only
# with explicit --allow-isolated-install. Does not modify original checkpoints.
set -euo pipefail
cd "$(dirname "$0")/../.."
BASE=/data/datasets_brayan/ltd-sota-2026
if [[ -f "$BASE/host_manifests/host_environment.sh" ]]; then
  source "$BASE/host_manifests/host_environment.sh"
fi
PY="$HOME/modelos/venv/bin/python"
CMP="$BASE/audits/ltd_vs_seven_sota_20261113T095045Z.json"
[[ -x "$PY" && -f "$CMP" ]] || { echo "BLOCKED_TORCH_OR_COMPARISON_NOT_FOUND";exit 3; }
check() {
  [[ -x "$1" ]] || return 1
  CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 "$1" -c 'import keras,tensorflow,numpy,sklearn' >/dev/null 2>&1
}
chosen=""
for path in "$(printenv LTD_SOTA_KERAS_PY || true)" \
  "$BASE/runtime/venv-keras-authors/bin/python" \
  "$BASE/venv-keras-authors/bin/python" \
  "/data/ltd-sota-bsierra/venv-keras-authors/bin/python" \
  "$HOME/modelos/venv-keras-authors/bin/python" \
  "$HOME/venv-keras-authors/bin/python"
do
  if [[ -n "$path" ]] && check "$path"; then chosen="$path";break;fi
done
if [[ -z "$chosen" ]];then
  for parent in "$BASE" "$HOME/modelos";do
    [[ -d "$parent" ]] || continue
    while IFS= read -r path;do
      if check "$path";then chosen="$path";break;fi
    done < <(find "$parent" -maxdepth 6 \( -type f -o -type l \) -path '*/bin/python' 2>/dev/null | head -80)
    [[ -n "$chosen" ]] && break
  done
fi
if [[ -z "$chosen" && "$*" == *--allow-isolated-install* ]];then
  DEST="$BASE/runtime/venv-keras-authors"
  [[ ! -e "$DEST" ]] || { echo "BLOCKED_EXISTING_ENV_REFUSE_OVERWRITE $DEST";exit 5; }
  command -v python3.12 >/dev/null || { echo "BLOCKED_PYTHON312_NOT_FOUND";exit 5; }
  export LTD_SOTA_KERAS_VENV_DIR="$DEST"
  bash scripts/sota/setup_isolated_authors_keras_v1.sh install
  bash scripts/sota/setup_isolated_authors_keras_v1.sh check
  chosen="$DEST/bin/python"
fi
[[ -n "$chosen" ]] || {
  echo "BLOCKED_NO_KERAS_ENV: rerun with --allow-isolated-install for separate pinned installation";exit 5;
}
echo "FOUND_KERAS_PY $chosen"
CUDA_VISIBLE_DEVICES=-1 "$chosen" - <<'PY'
import keras,tensorflow as tf,numpy,sklearn,sys
print("RUNTIME_VERSIONS",sys.version.split()[0],keras.__version__,
      tf.__version__,numpy.__version__,sklearn.__version__)
print("VISIBLE_GPU",tf.config.list_physical_devices("GPU"))
PY
OUT="$BASE/audits/author_fidelity_recovered_$(date -u +%Y%m%dT%H%M%SZ)"
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2 \
 nice -n 10 "$PY" scripts/sota/audit_seven_author_functional_fidelity_v1.py \
  --repo "$PWD" --base "$BASE" --comparison "$CMP" \
  --keras-python "$chosen" --output-dir "$OUT"
"$PY" - "$OUT/certificate.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
for name,item in r["models"].items():
 print("AUTHOR_FIDELITY",name,item["status"])
 if item.get("error"):print("ERROR",name,item["error"][-1100:])
print("WFCAT_IAT_PARITY",r["supplementary"]["WFCAT_raw_author_IAT_numerical_parity"]["status"])
print("SUITE",r["status"],"CERTIFICATE",sys.argv[1])
PY
