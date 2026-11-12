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
"${LTD_SOTA_PYTORCH_PY:-/data/ltd-sota-bsierra/venv-cu128/bin/python}" - <<'PY'
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
"${LTD_SOTA_KERAS_PY:-/data/ltd-sota-bsierra/venv-keras-authors/bin/python}" - <<'PY'
import tensorflow as tf
print("TENSORFLOW",tf.__version__)
gpus=tf.config.list_physical_devices("GPU")
print("TF_VISIBLE_GPUS",gpus)
if not gpus:
    raise SystemExit("BLOCKED_TF_GPU_NOT_VISIBLE")
print("TF_GPU_VISIBLE_PASS")
# GPU visibility alone does not prove that sm_120 kernels work. Explicitly
# execute float32 and Conv1D GPU kernels before allowing any training.
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)
with tf.device("/GPU:0"):
    x=tf.ones((4,128,1),dtype=tf.float32)
    y=tf.keras.layers.Conv1D(8,3)(x)
    value=tf.reduce_sum(y)
result=float(value.numpy())
assert __import__("math").isfinite(result), "TensorFlow GPU returned nonfinite value"
print("TF_CUDA_CONV_PASS",result,flush=True)
PY
echo "SOTA_GPU_RUNTIME_SMOKE_PASS"
