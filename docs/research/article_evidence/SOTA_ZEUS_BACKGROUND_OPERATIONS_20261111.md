# Zeus detached original-source SOTA battery: launch and monitor

Run only **after** CPU preflight, CPU integration, CPU preparation and host-check succeed on **the same commit**. All commands below run on Zeus, in ~/ltd-attack.

## One-time shell setup after each SSH login

    cd ~/ltd-attack || exit 1
    BASE=/data/datasets_brayan/ltd-sota-2026
    source "$BASE/host_manifests/host_environment.sh"
    export LTD_SOTA_PYTORCH_PY="$HOME/modelos/venv/bin/python"
    export LTD_SOTA_KERAS_PY="$BASE/environments/venv-keras-authors/bin/python"
    export TMPDIR="$BASE/tmp"
    export PIP_CACHE_DIR="$BASE/pip-cache"

## Strict readiness gate

    bash scripts/sota/run_original_sota_battery_v1.sh cpu-preflight
    bash scripts/sota/run_original_sota_battery_v1.sh cpu-integration
    bash scripts/sota/run_original_sota_battery_v1.sh cpu-prepare
    bash scripts/sota/run_original_sota_battery_v1.sh host-check
    bash scripts/sota/smoke_perseo_gpu_runtime_v1.sh
    python scripts/sota/original_sota_orchestrator_v2.py --stage all --dry-run

The detached launcher checks host/commit certificates, frozen S1/S5 hashes,
GPU idleness, a live torch CUDA tensor and TensorFlow GPU visibility again
before detaching. Do not launch if any check fails.

## First complete real model, detached

    bash scripts/sota/run_original_sota_background_v1.sh start NetCLR

This returns a run ID, process ID and absolute console.log path.
The process continues after normal SSH logout using nohup. Only one
battery is permitted at a time using flock. No training output is overwritten.

## Live monitoring from a second SSH session

    cd ~/ltd-attack
    bash scripts/sota/run_original_sota_background_v1.sh status
    bash scripts/sota/run_original_sota_background_v1.sh models
    bash scripts/sota/run_original_sota_background_v1.sh follow
    python scripts/sota/report_original_sota_progress_v1.py
    python scripts/sota/report_original_sota_progress_v1.py --json

status shows last lines and current GPU load. models shows last 50 stage
events and the locations of individual model logs. follow tails live output
until Ctrl+C; Ctrl+C in the monitor does NOT stop the detached training.
The read-only Python progress report checks manifests and SHA-256 for each
of all 30 stages. While a currently active stage is still writing its
output, RUNNING_UNFINISHED_STAGE is expected; after it finishes,
PASS_VERIFIED is required.

## After NetCLR completes, run the remaining models

    bash scripts/sota/run_original_sota_background_v1.sh start all

The full orchestrator only skips stages validated by complete manifest
and checkpoint/NPZ SHA. Missing stages run sequentially. All seven
models each produce external 09B and Future-B evaluations. Output
locations are under LTD_SOTA_ARTIFACT_ROOT, not in the Git repository.

## Final independent audit and statistical comparison

The --model all workflow automatically runs the independent UID/score
integrity check. Date-paired LTD comparisons are a separate explicit step:

    export LTD_SOTA_LTD_REFERENCE_09B="/actual/path/to/frozen_LTD_09B.npz"
    export LTD_SOTA_LTD_REFERENCE_FUTURE_B="/actual/path/to/frozen_LTD_FutureB.npz"
    bash scripts/sota/run_original_sota_battery_v1.sh statistics

Do not assume the canonical S5 NPZ automatically contains the LTD-Hybrid
probability field or the same class order; check reference schema and
sha256 before the statistics command.

## Safety

Do not launch both NetCLR and all at once. Never kill another user's
GPU processes. Do not remove partially completed directories without a
preserved log and explicit diagnosis. The script does not install packages,
send notifications or promise automatic recovery from programming errors.
A job status of FAILED must be inspected in its console.log.
