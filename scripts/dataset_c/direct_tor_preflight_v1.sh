#!/usr/bin/env bash
# Dataset C: direct-Tor inventory and one disposable bootstrap/egress probe.
# NO capture, NO bridge, NO cron, NO clock offset, NO reuse of historical containers.
set -euo pipefail
MODE="${1:---inventory}"
IMAGE="${LTD_DIRECT_TOR_IMAGE:-ltd-c-tor-nyx:trixie}"
NAME="ltd-c-direct-smoke-$(id -u)-$$"

inventory() {
  echo "=== HOST AND CLOCK ==="
  hostname
  date -u '+UTC=%Y-%m-%dT%H:%M:%SZ'
  timedatectl show -p NTP -p NTPSynchronized 2>/dev/null || true
  echo "=== DATA CAPACITY ==="
  if [[ -d /data ]]; then df -h /data; else df -h "$HOME/ltd-storage"; fi
  echo "=== LEGACY C CONTAINERS (READ ONLY) ==="
  for n in ltd-c-tor-inst1 ltd-c-tor-inst2 ltd-c-tor-inst3 ltd-c-manual-tor; do
    docker inspect --format '{{.Name}} status={{.State.Status}} restart={{.HostConfig.RestartPolicy.Name}}' "$n" 2>/dev/null || echo "NOT_FOUND $n"
  done
  echo "=== IMAGE ==="
  docker image inspect "$IMAGE" --format '{{.Id}}' 2>/dev/null || echo "IMAGE_NOT_FOUND $IMAGE"
  echo "=== MATCHING CRON ENTRIES (READ ONLY) ==="
  (crontab -l 2>/dev/null || true) | grep -Ei 'LTD_C_|dataset.c|run_hourly_qa|three_hourly_qa|watchdog' || true
}

probe() {
  echo "=== DIRECT TOR DISPOSABLE PROBE ==="
  command -v docker >/dev/null || { echo "STOP no Docker"; exit 3; }
  command -v timeout >/dev/null || { echo "STOP no timeout"; exit 3; }
  docker image inspect "$IMAGE" >/dev/null || { echo "STOP missing image: $IMAGE"; exit 3; }
  sync="$(timedatectl show -p NTPSynchronized --value 2>/dev/null || true)"
  [[ "$sync" == yes ]] || { echo "STOP NTPSynchronized=${sync:-unknown}"; exit 4; }
  timeout 240 docker run --rm --name "$NAME" --network bridge \
    --user 100:101 --cap-drop ALL --security-opt no-new-privileges \
    --read-only --tmpfs /tmp:rw,nosuid,size=64m \
    --entrypoint /bin/sh "$IMAGE" -ec '
      unset LD_PRELOAD LTD_CLOCK_OFFSET_SECONDS http_proxy https_proxy ALL_PROXY all_proxy
      command -v tor >/dev/null
      command -v curl >/dev/null
      cat >/tmp/direct-torrc <<EOF
ClientOnly 1
SocksPort 127.0.0.1:9050
ControlPort 0
UseBridges 0
DataDirectory /tmp/ltd-direct-data
AvoidDiskWrites 1
Log notice file /tmp/ltd-direct.log
EOF
      mkdir -p /tmp/ltd-direct-data
      chmod 700 /tmp/ltd-direct-data
      tor --verify-config -f /tmp/direct-torrc >/dev/null
      tor -f /tmp/direct-torrc >/tmp/ltd-direct-stdout.log 2>&1 &
      p=$!
      trap '\''kill "$p" 2>/dev/null || true; wait "$p" 2>/dev/null || true'\'' EXIT
      i=0
      while [ "$i" -lt 30 ]; do
        i=$((i+1))
        if curl -fsS --connect-timeout 4 --max-time 10 \
          --socks5-hostname 127.0.0.1:9050 \
          https://check.torproject.org/api/ip >/tmp/tor-check.json 2>/dev/null; then
          if grep -Eq '\''"IsTor"[[:space:]]*:[[:space:]]*true'\'' /tmp/tor-check.json \
            && grep -q "Bootstrapped 100%" /tmp/ltd-direct.log; then
            echo "TOR_DIRECT_BOOTSTRAP_100=YES"
            echo "TOR_DIRECT_ISTOR=TRUE"
            echo "DIRECT_TOR_SMOKE_PASS"
            exit 0
          fi
        fi
        sleep 5
      done
      echo "DIRECT_TOR_SMOKE_FAIL"
      grep "Bootstrapped" /tmp/ltd-direct.log | tail -4 || true
      exit 6
    '
}
case "$MODE" in
  --inventory) inventory ;;
  --probe) probe ;;
  *) echo 'Usage: direct_tor_preflight_v1.sh [--inventory|--probe]'; exit 2 ;;
esac
