#!/usr/bin/env bash
# Dataset C: stage new direct-Tor processes, validate, and swap legacy container names.
# Docker's entrypoint is immutable: replace containers safely; never restart obfs4 entrypoint.
# NO capture, NO cron, NO NTP bypass for scientific data, NO host clock changes.
set -euo pipefail
MODE=--status
if (( $# > 0 )); then MODE="$1"; fi
IMAGE="ltd-c-tor-nyx:trixie"
RETIRE_TAG="obfs4-retired-$(date -u +%Y%m%dT%H%M%SZ)"
PURGE_MARKER="$HOME/ltd-ops/dataset-c-bridge-retirement/last_success.txt"
original_name() {
  if [[ "$1" == manual ]]; then echo ltd-c-manual-tor; else echo "ltd-c-tor-inst$1"; fi
}
stage_name() {
  if [[ "$1" == manual ]]; then echo ltd-c-direct-stage-manual; else echo "ltd-c-direct-stage-inst$1"; fi
}
network_name() {
  if [[ "$1" == manual ]]; then echo bridge; else echo "isolated_net_1site_inst$1"; fi
}
exists() { docker container inspect "$1" >/dev/null 2>&1; }
running() {
  [[ "$(docker inspect --format '{{.State.Running}}' "$1" 2>/dev/null || true)" == true ]]
}
direct_ready() {
  local n="$1" attempt response
  running "$n" || { echo "FAIL_NOT_RUNNING $n" >&2; return 1; }
  docker exec "$n" test -f /tmp/direct-torrc || return 1
  docker exec "$n" grep -qx 'UseBridges 0' /tmp/direct-torrc || return 1
  docker exec "$n" sh -ec 'test -z "$(printenv LTD_CLOCK_OFFSET_SECONDS 2>/dev/null || true)"' || return 1
  # Audit current running Tor process arguments: no obfs4 proxy or old torrc.
  docker exec "$n" sh -ec 'tr "\000" " " </proc/1/cmdline | grep -Fq "tor -f /tmp/direct-torrc"' || return 1
  docker logs "$n" 2>&1 | grep -q 'Bootstrapped 100%' || return 1
  # Per-instance bounded retries: a single exit-circuit TLS reset is not a failed Tor instance.
  for attempt in 1 2 3; do
    response="$(docker exec "$n" curl -fsS --connect-timeout 8 --max-time 25       --retry 0 --socks5-hostname 127.0.0.1:9050       https://check.torproject.org/api/ip 2>/dev/null || true)"
    if printf '%s' "$response" | grep -Eq '"IsTor"[[:space:]]*:[[:space:]]*true'; then
      return 0
    fi
    if [[ "$attempt" -lt 3 ]]; then sleep 7; fi
  done
  echo "FAIL_TOR_EXIT_PROBE $n after 3 requests" >&2
  return 1
}
check_base() {
  [[ -f "$PURGE_MARKER" ]] ||
    { echo "STOP: bridge-data purge report not found; run retire_bridge_capture_data_v1.sh first"; exit 3; }
  docker image inspect "$IMAGE" >/dev/null ||
    { echo "STOP: missing local image"; exit 3; }
  local n stage net
  for i in 1 2 3 manual; do
    n="$(original_name "$i")"; stage="$(stage_name "$i")"; net="$(network_name "$i")"
    docker network inspect "$net" >/dev/null ||
      { echo "STOP: missing network $net"; exit 3; }
    if exists "$n"; then
      [[ "$(docker inspect --format '{{.State.Running}}' "$n")" == false ]] ||
        { echo "STOP: legacy container currently running $n"; exit 3; }
      [[ "$(docker inspect --format '{{.HostConfig.RestartPolicy.Name}}' "$n")" == no ]] ||
        { echo "STOP: legacy container has restart policy $n"; exit 3; }
      [[ "$(docker inspect --format '{{.Config.Image}}' "$n")" == "$IMAGE" ]] ||
        { echo "STOP: unexpected old image $n"; exit 3; }
    fi
    exists "$stage" && { echo "STOP: stage name already exists $stage"; exit 3; }
  done
  if docker ps --format '{{.Names}}' | grep -E '^ltd-c-capture|^ltd-c-audit120'; then
    echo "STOP: active C capture container"; exit 3
  fi
}
new_stage() {
  [[ "$(printenv LTD_C_DIRECT_REBUILD_CONFIRMED 2>/dev/null || true)" == "YES_STAGE_DIRECT_C" ]] ||
    { echo "STOP: set LTD_C_DIRECT_REBUILD_CONFIRMED=YES_STAGE_DIRECT_C"; exit 3; }
  check_base
  local stage net
  for i in 1 2 3 manual; do
    stage="$(stage_name "$i")"
    net="$(network_name "$i")"
    docker create --name "$stage" --network "$net" --restart=no \
      --user 100:101 --cap-drop ALL --security-opt no-new-privileges \
      --read-only --tmpfs /tmp:rw,nosuid,nodev,size=256m \
      --entrypoint /bin/sh "$IMAGE" -ec '
        unset LD_PRELOAD LTD_CLOCK_OFFSET_SECONDS http_proxy https_proxy ALL_PROXY all_proxy
        cat >/tmp/direct-torrc <<EOF
ClientOnly 1
SocksPort 127.0.0.1:9050
ControlPort 0
UseBridges 0
DataDirectory /tmp/ltd-direct-data
AvoidDiskWrites 1
Log notice stdout
EOF
        mkdir -m 0700 /tmp/ltd-direct-data
        tor --verify-config -f /tmp/direct-torrc >/dev/null
        exec tor -f /tmp/direct-torrc
      ' >/dev/null
    docker start "$stage" >/dev/null
    echo "STAGED_DIRECT_TOR $stage network=$net"
  done
  echo "STAGE_CREATED_NO_CAPTURES; run --verify-stage before --cutover"
}
verify_stage() {
  local ok=0 n
  for i in 1 2 3 manual; do
    n="$(stage_name "$i")"
    if direct_ready "$n"; then
      echo "DIRECT_TOR_READY $n"
    else
      echo "DIRECT_TOR_NOT_READY $n"
      ok=1
    fi
  done
  [[ "$ok" -eq 0 ]] || exit 4
  echo "DIRECT_STAGE_ALL_READY"
}
cutover() {
  [[ "$(printenv LTD_C_DIRECT_CUTOVER_CONFIRMED 2>/dev/null || true)" == "YES_SWITCH_C_NAMES" ]] ||
    { echo "STOP: set LTD_C_DIRECT_CUTOVER_CONFIRMED=YES_SWITCH_C_NAMES"; exit 3; }
  verify_stage
  local n stage
  for i in 1 2 3 manual; do
    n="$(original_name "$i")"; stage="$(stage_name "$i")"
    exists "$n" || { echo "STOP: old original name missing $n"; exit 3; }
    running "$n" && { echo "STOP: old container running $n"; exit 3; }
    exists "$n-$RETIRE_TAG" && { echo "STOP: old retirement name conflict"; exit 3; }
  done
  for i in 1 2 3 manual; do
    n="$(original_name "$i")"; stage="$(stage_name "$i")"
    docker rename "$n" "$n-$RETIRE_TAG"
    docker rename "$stage" "$n"
    echo "DIRECT_TOR_REPLACED $n ; old_config_retained_as=$n-$RETIRE_TAG"
  done
  echo "DIRECT_TOR_CUTOVER_COMPLETE; old obfs4 stopped containers retained for rollback, not started"
}
verify_current() {
  local ok=0 n
  for i in 1 2 3 manual; do
    n="$(original_name "$i")"
    if direct_ready "$n"; then echo "DIRECT_TOR_READY $n"; else echo "DIRECT_TOR_NOT_READY $n";ok=1;fi
  done
  [[ "$ok" -eq 0 ]] || exit 4
  echo "DIRECT_TOR_4_INSTANCES_PASS"
  echo "CAPTURE_GATE_NTP=$(timedatectl show -p NTPSynchronized --value 2>/dev/null || echo unknown)"
  echo "NO_CAPTURE_OR_CRON_HAS_BEEN_STARTED"
}
status() {
  echo "PURGE_REPORT_PRESENT=$([[ -f "$PURGE_MARKER" ]] && echo yes || echo no)"
  for i in 1 2 3 manual; do
    n="$(original_name "$i")"
    stage="$(stage_name "$i")"
    for c in "$n" "$stage"; do
      if exists "$c"; then
        docker inspect --format '{{.Name}} state={{.State.Status}} network={{.HostConfig.NetworkMode}} image={{.Config.Image}} restart={{.HostConfig.RestartPolicy.Name}}' "$c"
      fi
    done
  done
  echo "NTP=$(timedatectl show -p NTPSynchronized --value 2>/dev/null || echo unknown)"
}
case "$MODE" in
  --status) status ;;
  --stage) new_stage ;;
  --verify-stage) verify_stage ;;
  --cutover) cutover ;;
  --verify-current) verify_current ;;
  *) echo "Usage: $0 --status|--stage|--verify-stage|--cutover|--verify-current"; exit 2 ;;
esac
