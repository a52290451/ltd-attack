#!/usr/bin/env bash
# Explicit isolated Keras/TensorFlow setup for ORIGINAL DF/TikTok/VarCNN/RoFiRe.
# Never changes the existing PyTorch venv. No training and no GPU use.
set -euo pipefail
cd "$(dirname "$0")/../.."
MODE="$1"
if [[ -n "${LTD_SOTA_KERAS_VENV_DIR:-}" ]]; then
  DEST="$LTD_SOTA_KERAS_VENV_DIR"
elif [[ -n "${LTD_SOTA_KERAS_PY:-}" ]]; then
  DEST="${LTD_SOTA_KERAS_PY%/bin/python}"
else
  DEST="/data/ltd-sota-bsierra/venv-keras-authors"
fi
case "$MODE" in
  install)
    if [[ -e "$DEST" ]]; then
      echo "EXISTS_REFUSE_OVERWRITE: $DEST"
      echo "Run --check or inspect the existing env; do not silently reinstall."
      exit 3
    fi
    python3.12 -m venv "$DEST"
    CUDA_VISIBLE_DEVICES=-1 "$DEST/bin/python" -m pip install \
      --upgrade pip
    CUDA_VISIBLE_DEVICES=-1 "$DEST/bin/python" -m pip install \
      "tensorflow[and-cuda]==2.21.0" "scikit-learn>=1.5,<2.0" \
      "numpy>=1.26,<2.2"
    echo "ISOLATED_AUTHORS_KERAS_INSTALLED $DEST"
    ;;
  check)
    if [[ ! -x "$DEST/bin/python" ]];then
      echo "MISSING_AUTHORS_KERAS_ENV $DEST";exit 3
    fi
    CUDA_VISIBLE_DEVICES=-1 "$DEST/bin/python" - <<'PY'
import os
assert os.environ.get("CUDA_VISIBLE_DEVICES")=="-1"
import keras,tensorflow as tf,numpy,sklearn
print("ISOLATED_KERAS_ENV_PASS",
      "tensorflow",tf.__version__,"keras",keras.__version__,
      "numpy",numpy.__version__,"scikit-learn",sklearn.__version__)
print("GPU_PROHIBITED_IN_CHECK:",tf.config.list_physical_devices("GPU"))
assert not tf.config.list_physical_devices("GPU")
PY
    ;;
  *)
    echo "Usage: bash $0 {install|check}"
    exit 2 ;;
esac
