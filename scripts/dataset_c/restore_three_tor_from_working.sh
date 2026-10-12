#!/usr/bin/env bash
# P16C: recupera obfs4proxy y la configuración Tor del contenedor que funciona.
# No modifica Dataset A/B, Noise, volúmenes Tor ni Cron hasta validar.
set -Eeuo pipefail
umask 077
BASE=/data/ltd-c-qa-bsierra
SOURCE=ltd-c-manual-tor
[[ -d "$BASE" && "$(stat -c %u "$BASE")" == "$(id -u)" ]] || { echo "STOP: almacenamiento base no pertenece al usuario"; exit 2; }
[[ -z "$(crontab -l 2>/dev/null | grep -E 'LTD_C_(THREE_QA|QA_HOURLY)' || true)" ]] || { echo "STOP: Cron previo sigue activo"; exit 2; }
[[ -z "$(docker ps --format '{{.Names}}' | grep '^ltd-c-capture-' || true)" ]] || { echo "STOP: hay navegadores capturando"; exit 2; }
[[ "$(docker inspect -f '{{.State.Running}}' "$SOURCE")" == true ]] || { echo "STOP: Tor original apagado"; exit 2; }
docker exec "$SOURCE" test -x /usr/bin/obfs4proxy || { echo "STOP: el origen no tiene ejecutable"; exit 2; }
docker exec "$SOURCE" test -s /etc/tor/torrc-obfs4 || { echo "STOP: el origen no tiene configuración"; exit 2; }

REF_IMAGE="$(docker inspect -f '{{.Image}}' "$SOURCE")"
for i in 1 2 3; do
  NAME="ltd-c-tor-inst$i"
  [[ "$(docker inspect -f '{{.Image}}' "$NAME")" == "$REF_IMAGE" ]] || { echo "STOP: imagen distinta en $NAME"; exit 3; }
  [[ "$(docker inspect -f '{{.State.Running}}' "$NAME")" == false ]] || { echo "STOP: $NAME no está detenido"; exit 3; }
done

STAGE="$(mktemp -d "$BASE/.obfs4-repair.XXXXXXXX")"
cleanup() {
  test -n "${STAGE:-}" || return
  case "$STAGE" in "$BASE"/.obfs4-repair.*)
    rm -f -- "$STAGE/obfs4proxy" "$STAGE/torrc-obfs4"
    rmdir -- "$STAGE" 2>/dev/null || true
  ;; esac
}
trap cleanup EXIT
chmod 700 "$STAGE"
# Copias privadas locales: nunca mostrar los certificados ni escribirlos en Git.
docker cp -L "$SOURCE:/usr/bin/obfs4proxy" "$STAGE/obfs4proxy"
docker cp -L "$SOURCE:/etc/tor/torrc-obfs4" "$STAGE/torrc-obfs4"
chmod 755 "$STAGE/obfs4proxy"
chmod 600 "$STAGE/torrc-obfs4"
test -s "$STAGE/obfs4proxy" && test -s "$STAGE/torrc-obfs4"

for i in 1 2 3; do
  NAME="ltd-c-tor-inst$i"
  docker cp "$STAGE/obfs4proxy" "$NAME:/usr/bin/obfs4proxy"
  docker cp "$STAGE/torrc-obfs4" "$NAME:/etc/tor/torrc-obfs4"
  echo "RESTAURADO $NAME (binario obfs4 + configuración original)"
done
# Liberar la copia privada tan pronto como se termine la restauración.
cleanup
STAGE=""

for i in 1 2 3; do
  NAME="ltd-c-tor-inst$i"
  docker start "$NAME" >/dev/null
  docker exec "$NAME" test -x /usr/bin/obfs4proxy
  docker exec "$NAME" sh -c 'test "$(stat -c %u:%g /etc/tor/torrc-obfs4)" = "100:101" && test "$(stat -c %a /etc/tor/torrc-obfs4)" = 600'
  echo "ARRANCADO $NAME"
done

ready() {
  local name="$1" j
  for j in $(seq 1 20); do
    if docker exec "$name" curl -fsS --max-time 12 --socks5-hostname 127.0.0.1:9050 https://check.torproject.org/api/ip 2>/dev/null |
      python3 -c 'import sys,json; assert json.load(sys.stdin).get("IsTor") is True' 2>/dev/null; then
      echo "TOR_OK $name"
      return 0
    fi
    sleep 10
  done
  echo "TOR_FAIL $name"
  return 1
}
declare -a PIDS=()
for i in 1 2 3; do
  ready "ltd-c-tor-inst$i" > "$BASE/three_tor_bootstrap_inst$i.log" 2>&1 &
  PIDS+=("$!")
done
FAILED=0
for idx in 0 1 2; do
  wait "${PIDS[$idx]}" || FAILED=1
  cat "$BASE/three_tor_bootstrap_inst$((idx+1)).log"
done
if (( FAILED )); then
  echo "STOP: Tor no disponible en todas las instancias; no iniciar capturas ni Cron"
  echo "Revisar exclusivamente el arranque Tor; los volúmenes permanecen intactos."
  exit 4
fi

cd "$HOME/ltd-attack"
echo "===== PRUEBA PARALELA: 12 POSICIONES ====="
python3 scripts/dataset_c/three_instance_qa.py smoke
echo "===== VALIDACIÓN PARALELA: 120 POSICIONES ====="
python3 scripts/dataset_c/three_instance_qa.py full
echo "===== ACTIVAR QA HORARIO ====="
bash scripts/dataset_c/three_hourly_qa.sh --install
echo "THREE_QA_READY"
