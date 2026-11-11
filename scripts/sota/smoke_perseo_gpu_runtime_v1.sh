#!/usr/bin/env bash
# GPU smoke only when no existing compute jobs are reported.
# Lightweight 1-MiB PyTorch allocation; TensorFlow visibility only.
set -euo pipefail
cd "$(dirname "$0")/../.."
echo "===== GPU AVAILABILITY (LIVE) ====="
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader,nounits
active="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
if [[ -n "$active" ]]; then
  echo "BLOCKED_GPU_BUSY_COMPUTE_PIDS $active"
  exit 5
fi
echo "===== PyTorch CUDA allocation smoke ====="
/data/ltd-sota-bsierra/venv-cu128/bin/python - <<'PY'
import torch
assert torch.cuda.is_available(), "PyTorch CUDA unavailable"
print("TORCH",torch.__version__,"CUDA",torch.version.cuda)
print("DEVICE",torch.cuda.get_device_name(0))
with torch.no_grad():
    x=torch.ones(256*1024,dtype=torch.float32,device="cuda:0")
    assert torch.isfinite((x+x).sum()).item()
torch.cuda.synchronize()
print("TORCH_CUDA_SMALL_ALLOCATION_PASS")
PY
echo "===== TensorFlow GPU visibility (no tensors) ====="
/data/ltd-sota-bsierra/venv-keras-authors/bin/python - <<'PY'
import tensorflow as tf
print("TENSORFLOW",tf.__version__)
gpus=tf.config.list_physical_devices("GPU")
print("TF_VISIBLE_GPUS",gpus)
if not gpus:
    raise SystemExit("BLOCKED_TF_GPU_NOT_VISIBLE")
print("TF_GPU_VISIBLE_PASS")
PY
echo "SOTA_GPU_RUNTIME_SMOKE_PASS"
