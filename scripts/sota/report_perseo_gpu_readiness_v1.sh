#!/usr/bin/env bash
# Non-invasive real-time GPU and two-framework visibility report for Perseo.
# No tensors allocated, no imports of torch/tensorflow, no training.
set -euo pipefail
echo "===== GPU IDENTITY / MEMORY / UTILIZATION ====="
nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used,memory.free,utilization.gpu,compute_mode \
  --format=csv,noheader,nounits
echo "===== COMPUTE PROCESSES ====="
active="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv,noheader,nounits)"
if [[ -n "$active" ]]; then
  echo "$active"
  echo "GPU_BUSY_OTHER_PROCESS_PRESENT"
else
  echo "NO_NVML_COMPUTE_PROCESS_REPORTED"
fi
echo "===== ACTIVE PROCESS DETAILS (IF ACCESSIBLE) ====="
nvidia-smi
echo "===== ORIGINAL PYTORCH / TENSORFLOW ENVS ====="
"${LTD_SOTA_PYTORCH_PY:-/data/ltd-sota-bsierra/venv-cu128/bin/python}" -m pip show torch torchvision 2>/dev/null | grep -E '^(Name|Version):' || true
"${LTD_SOTA_KERAS_PY:-/data/ltd-sota-bsierra/venv-keras-authors/bin/python}" -m pip show tensorflow keras 2>/dev/null | grep -E '^(Name|Version):' || true
echo "===== INTERPRETATION ====="
echo "This is an observation, not a reservation. Recheck immediately before GPU training."
echo "Other users may be hidden by driver visibility; check full nvidia-smi table."
