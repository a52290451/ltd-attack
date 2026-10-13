# Dataset C — STOP bridge pilot and preflight matched direct-Tor recapture (2026-10-13)

## Decision and boundaries
- **IMMEDIATE STOP requested by researcher.** Existing C traffic was acquired using Tor bridges/obfs4 and is NOT suitable for unqualified longitudinal comparison with A/B if A/B used direct Tor. Such transport may alter visible traffic features and introduce acquisition-confounded distribution shift.
- **Current status: NOT REMOTELY STOPPED.** GitHub edits do not control Perseo processes or Docker. Operator on Perseo must perform exact service/container stop and show evidence. Do not announce operational success without Perseo verification.
- Preserve C-obfs4 pilot intact as **C_BRIDGE_PILOT_INVALID_FOR_AB_LONGITUDINAL_COMPARISON**, with provenance; do not delete, relabel as Direct or silently blend with A/B or a new C.
- No experiments, inference, model selection, predictions or aggregate result comparison using C bridge pilot.

## 0. Read-only Perseo inventory — run immediately
```bash
date -u --iso-8601=seconds
hostname -f
docker ps --format 'table {{.ID}}\t{{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Command}}\t{{.Ports}}'
ps -eo pid,ppid,lstart,cmd | grep -Ei '[t]or|[o]bfs4|[c]apture|[c]ollector|[w]atchdog|[s]elenium|[p]laywright|[p]cap|[d]ataset.c' | head -100
systemctl --user list-units --all --type=service --no-pager | grep -Ei 'tor|capture|collector|dataset|watchdog' || true
systemctl list-units --all --type=service --no-pager | grep -Ei 'tor|capture|collector|dataset|watchdog' || true
tmux ls 2>/dev/null || true
screen -ls 2>/dev/null || true
crontab -l 2>/dev/null || true
docker compose ls 2>/dev/null || true
```
**Check exact process/units/names before touching anything.** No `docker stop $(docker ps ...)`, no `pkill -f tor`, no `docker system prune` / `down -v`; never assume all Tor containers are Dataset C.

## 1. Controlled stop order (choose only actually observed managers and exact identifiers)
1. Disable/stop the specific C scheduler/watchdog/cron/systemd/tmux job FIRST, so it does not restart workers.
2. Request graceful stop of exact C capture workers, close pcap/csv writers and flush manifests.
3. Stop ONLY the exact C Tor/obfs4 containers if they are exclusively part of C (not shared Tor infra).
4. Verify no C capture jobs, timers or restart policies remain active. Keep unrelated production services untouched.

Examples **ONLY AFTER INVENTORY IDENTIFIES CORRECT NAMES**:
```bash
# If user-owned systemd capture service:
systemctl --user stop <EXACT_C_CAPTURE_UNIT>
systemctl --user disable <EXACT_C_CAPTURE_UNIT>  # if it is a boot-auto-start C-only unit
# If system-level service, use sudo systemctl stop/disable exact unit only.
# If tmux process controls a worker:
tmux send-keys -t <EXACT_C_CAPTURE_SESSION> C-c
# If Docker standalone C capture worker/container:
docker update --restart=no <EXACT_C_CONTAINER_NAME>
docker stop --time 45 <EXACT_C_CONTAINER_NAME>
# Docker Compose services: stop specifically identified C service from the correct project directory:
docker compose -p <EXACT_C_PROJECT> stop <EXACT_C_SERVICE>
```
Cron: back up exact crontab to user directory first, then manually disable only matching C line, leaving all other jobs. If systemd timer exists, stop/disable the exact C timer too. If Docker Compose orchestrator has `restart: always` / a parent scheduler, the parent must be disabled or stopping workers will be temporary. Confirm no restart on a 2–5 minute follow-up inventory.

## 2. Data quarantine and shutdown evidence
Capture (without deleting or renaming raw data):
- UTC stop time and operator; supervisor/parent PIDs; exact container names/IDs and images; original restart policies and scheduler records.
- Paths to raw pcaps, manifests and count snapshots; SHA256 of collected outputs; final capture count and partial/corrupt record counts.
- Origin of obfs4 setting (torrc UseBridges / Bridge lines, environment, compose configuration); capture viewpoint and effective traffic observation point.
- Store as `docs/operations/C_BRIDGE_PILOT_STOP_EVIDENCE_<DATE>.md` and immutable storage outside model-training datasets. Do not commit any bridge keys, authentication tokens, client secrets or other sensitive operational parameters.

## 3. A/B contract MUST BE VERIFIED, not assumed
Retrieve ORIGINAL A/B acquisition config/scripts/logs/manifests (not descriptions after the fact) and construct a side-by-side matrix:
- Tor binary version, Tor bootstrap route, bridge settings (direct vs bridge), SOCKS endpoint, browser/version, profile freshness, circuit policy, caching, session/container restart cadence.
- Traffic collection VANTAGE (client interface, eth/tun, host vs Tor-side, before/after TLS/obfs4 encapsulation); filter/BPF; pcap timestamp sources; capture protocol and process boundaries.
- Capture unit definition (one website visit), timeout/retry, pacing/concurrency, site URL list (65 canonical), geographic/network endpoint, service DNS and network stack.
- Clock UTC offset and synchronisation (pilot previously showed ~2,963 seconds offset); NTP/chrony status before and during collection.
- Direction and size feature extraction: exact definitions, truncation/padding MAX_LEN, handling of TLS/circuit control traffic, errors, duplicates, class order and quality filters.
- Privacy/ethics, Tor research resource-use constraints and network routing consistency.

**Matching direct-Tor route does not by itself guarantee scientific comparability.** A new direct capture may have other environmental shifts, which must be documented. Protocol freeze before collection; if meaningful differences remain, define separate strata rather than asserting same distribution.

## 4. Direct pilot / new cohort, only after explicit green-light
- Name future acquisition `C_DIRECT_V1` and make its root directory distinct from `C_BRIDGE_PILOT`.
- Acquire a tiny operational *QA-only* direct-Tor pilot first; DO NOT run model inference or tune network paths against model performance.
- Predeclared success gates: 65/65 sites attempted, coverage by site/day, timestamps within tolerance, no bridge in confirmed route, same capture viewpoint and feature contract as A/B, no unexpected Tor/bootstrap/circuit or pcap corruption, inventory and SHA manifests.
- Document remaining unavoidable environment shifts; do not claim equivalent if not measurable.
- After pilot QA and supervisor approval, start canonical comparable C collection and reserve it for ONE declared external evaluation.

## 5. Handoff
Provide the output of the inventory commands plus original A/B acquisition scripts and C launch mechanism. Only then can exact safe stop/start commands be produced. This protocol **does not** claim to have remotely stopped Perseo.
