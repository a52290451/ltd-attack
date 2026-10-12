#!/usr/bin/env bash
# Post-fix P16C QA: pause hourly runs, clear disposable smoke,
# install reviewed V2.4 only on an idle collector, compare a small recovery.
# DOES NOT resume Cron or declare Dataset C scientific.
set -euo pipefail
umask 077
REPO="$HOME/ltd-attack"
BASE=/data/ltd-c-qa-bsierra
APP="$HOME/ltd-collector-c-smoke-v22/P16C_RECOLECTOR_V2"
cd "$REPO"
[[ -d "$BASE" && -f "$APP/capture_dom.js" ]] || { echo "STOP: collector or QA base missing"; exit 2; }

echo "===== 1. PAUSE HOURLY QA + SAFE HOUSEKEEPING ====="
python3 scripts/dataset_c/pre_experiment_maintenance.py --apply

echo "===== 2. WAIT FOR ANY IN-FLIGHT CAPTURE TO FINISH ====="
last=0
for i in $(seq 1 100); do
  names=$(docker ps --format '{{.Names}}' |
    grep -E '^(ltd-c-capture-|ltd-c-recovery-|ltd-c-audit120_)' || true)
  if [[ -z "$names" ]]; then last=1; break; fi
  if (( i % 4 == 1 )); then echo "WAITING_FOR_COLLECTOR_IDLE $(date -u +%FT%TZ)"; fi
  sleep 15
done
(( last )) || { echo "STOP: collectors did not finish in 25 minutes"; exit 3; }
echo "COLLECTOR_IDLE_CONFIRMED"
python3 scripts/dataset_c/pre_experiment_maintenance.py --apply

echo "===== 3. AUDIT TWO FULL ROUNDS ALREADY STORED ====="
for name in three_hourly_20261012T092246Z_523162 three_hourly_20261012T100004Z_545692; do
  p="$BASE/$name"
  if [[ -f "$p/VALIDATED_THREE_FULL.txt" ]]; then
    python3 scripts/dataset_c/audit_three_hourly.py "$p" |
      grep -E 'RUN=|INTENTOS=|OK_PCAPP_VALID=|URL_UNICAS=|URL_OK_UNICAS=|PCAP_PRESENT_STRUCTURAL=|ESTADOS='
  else
    echo "SKIPPED_NOT_COMPLETE $name"
  fi
done

echo "===== 4. VERIFIED SOURCE PATCH V2.4 ====="
python3 scripts/dataset_c/patch_collector_v24.py --app "$APP"

echo "===== 5. CLASSIFIER UNIT TESTS (UNMODIFIED) ====="
docker run --rm --network none \
  --mount "type=bind,src=$APP,dst=/opt/ltd-c-v2,readonly" \
  --entrypoint /bin/bash tor_1site_noise:final -lc \
  'cd /opt/ltd-c-v2 && NODE_PATH=/app/node_modules node --test tests/quality.test.js'

echo "===== 6. PRIVATE SOURCE REVIEW, INCLUDING QUALITY CLASSIFIER ====="
python3 scripts/dataset_c/package_collector_source.py

echo "===== 7. LIMITED TARGETED QA RE-VISITS ====="
python3 scripts/dataset_c/review_and_recover_qa.py recover --max-per-category 3 |
  tee "$BASE/v24_targeted_recovery_console.log"

echo "===== 8. SCIENTIFIC CLOCK GATE ====="
timedatectl show -p NTP -p NTPSynchronized
echo "Cron remains PAUSED; V2.4 is QA-only; no scientific Dataset C claim."
echo "FINISHED_V24_QA_BATCH"
