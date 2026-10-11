#!/usr/bin/env bash
# Mover el piloto QA P16C desde /home/bsierra a /data sin pérdida y con rollback.
# NO modifica /data/bs_1site ni /data/bs_1site_noise.
set -euo pipefail
umask 077

REPO="${REPO:-$HOME/ltd-attack}"
SRC="${SRC:-$HOME/ltd-storage/dataset-c-smoke-obfs4}"
DEST="${DEST:-/data/ltd-c-qa-bsierra}"
CRON_TAG="LTD_C_QA_HOURLY"
TORMANAGER="$REPO/scripts/dataset_c/run_hourly_qa.sh"
IMAGE="ltd-c-tor-nyx:trixie"
BACKUP_CRON="$HOME/ltd-c-crontab-before-data-migration-$(date -u +%Y%m%dT%H%M%SZ).txt"
CRON_PAUSED=0

restore() {
  rc=$?
  if [[ "$CRON_PAUSED" == "1" ]]; then
    echo "Restaurando Cron anterior por salida antes de finalizar (código $rc)..."
    crontab "$BACKUP_CRON" || true
  fi
}
trap restore EXIT

[[ -f "$TORMANAGER" ]] || { echo "STOP: falta $TORMANAGER" >&2; exit 2; }
[[ "$DEST" == /data/* && "$DEST" != /data/ ]] || { echo "STOP: DEST debe estar dentro de /data" >&2; exit 2; }
[[ "$(findmnt -n -o TARGET --target /data)" == "/data" ]] || {
  echo "STOP: /data no está montado como volumen independiente. No copiar." >&2; exit 2;
}
[[ -d "$SRC" && ! -L "$SRC" ]] || {
  if [[ -L "$SRC" && "$(readlink -f "$SRC")" == "$(readlink -m "$DEST")" ]]; then
    echo "YA MIGRADO: $(readlink -f "$SRC")"
    exit 0
  fi
  echo "STOP: SRC no es el directorio esperado; revisar manualmente." >&2; exit 2;
}
[[ ! -e "$DEST" || ! -L "$DEST" ]] || { echo "STOP: DEST es enlace simbólico." >&2; exit 2; }
if docker ps --format '{{.Names}}' | grep -Eq '^ltd-c-(audit120_|recovery-v23-)'; then
  echo "STOP: hay una captura activa. Terminarla antes de migrar." >&2
  exit 5
fi
echo "SRC=$SRC"
echo "DEST=$DEST"
df -h "$SRC" /data

# Copia privada del crontab; modificar únicamente la entrada del piloto.
crontab -l > "$BACKUP_CRON"
if grep -q "$CRON_TAG" "$BACKUP_CRON"; then
  grep -v "$CRON_TAG" "$BACKUP_CRON" | crontab -
  CRON_PAUSED=1
  echo "Cron QA pausado; crontab original respaldado en $BACKUP_CRON."
fi

# Intento normal con permisos del usuario. Si /data es root-owned, crear
# exclusivamente un NUEVO subdirectorio propio con Docker (sin --privileged).
if [[ ! -d "$DEST" ]]; then
  if mkdir -m 700 "$DEST" 2>/dev/null; then
    echo "Directorio creado directamente por $(id -un)."
  else
    if [[ -e "$DEST" ]]; then
      echo "STOP: destino existente no accesible; no cambiar propietario." >&2
      exit 6
    fi
    echo "Creando solo $DEST a través de Docker, con UID/GID del usuario..."
    docker run --rm --network none \
      --mount "type=bind,src=/data,dst=/data" \
      --entrypoint /bin/sh "$IMAGE" \
      -c 'set -eu; test ! -e "$1"; mkdir -m 700 "$1"; chown "$2:$3" "$1"' \
      sh "$DEST" "$(id -u)" "$(id -g)"
  fi
fi
[[ -d "$DEST" && ! -L "$DEST" && -O "$DEST" ]] || {
  echo "STOP: el destino debe ser un directorio real propiedad del usuario." >&2; exit 6;
}
chmod 700 "$DEST"
[[ -w "$DEST" ]] || { echo "STOP: no se puede escribir en DEST." >&2; exit 6; }
testfile="$(mktemp "$DEST/.write-test.XXXXXXXX")"
rm -f "$testfile"

# Validar capacidad antes de iniciar copia; reserva adicional en /data.
SRC_KIB="$(du -sk "$SRC" | awk '{print $1}')"
FREE_KIB="$(df -Pk "$DEST" | awk 'NR==2{print $4}')"
if (( FREE_KIB < SRC_KIB + 120 * 1024 * 1024 )); then
  echo "STOP: /data no conservaría 120 GiB tras migrar." >&2; exit 6
fi
echo "Copiando $((SRC_KIB / 1024)) MiB a $DEST ..."
if command -v rsync >/dev/null; then
  rsync -a --no-owner --no-group "$SRC/" "$DEST/"
  # Recorre todas las rutas; '--checksum' valida contenido y longitud.
  discrepancy="$(rsync -ainc --no-owner --no-group "$SRC/" "$DEST/")"
  if [[ -n "$discrepancy" ]]; then
    echo "STOP: diferencias de rsync; no cambiar ruta original." >&2
    echo "$discrepancy" | head -n 12
    exit 7
  fi
else
  cp -a "$SRC/." "$DEST/"
  python3 - "$SRC" "$DEST" <<'PY'
from pathlib import Path
import hashlib
import sys
src, dst = map(Path, sys.argv[1:])
def fingerprint(base):
    out = {}
    for p in base.rglob("*"):
        name = str(p.relative_to(base))
        if p.is_symlink():
            out[name] = ("link", p.readlink().as_posix())
        elif p.is_file():
            digest = hashlib.sha256()
            with p.open("rb") as fh:
                for b in iter(lambda: fh.read(1<<20), b""):
                    digest.update(b)
            out[name] = ("file", p.stat().st_size, digest.hexdigest())
        elif p.is_dir():
            out[name] = ("dir",)
    return out
assert fingerprint(src) == fingerprint(dst), "STOP: copia no idéntica"
print("VERIFICACIÓN: contenido y estructura idénticos.")
PY
fi
echo "Copia verificada: $(du -sh "$DEST" | cut -f1)."
OLD="$SRC.before-data-migration-$(date -u +%Y%m%dT%H%M%SZ)"
mv "$SRC" "$OLD"
if ! ln -s "$DEST" "$SRC"; then
  mv "$OLD" "$SRC"
  echo "STOP: no se pudo crear enlace; ruta original restaurada." >&2
  exit 8
fi

[[ "$(readlink -f "$SRC")" == "$(readlink -f "$DEST")" ]] || {
  echo "STOP: enlace no resuelve correctamente" >&2; exit 8;
}
echo "ENLACE_OK: $SRC -> $(readlink -f "$SRC")"
echo "COPIA_SEGURIDAD_LOCAL=$OLD"

# Reinstalar la única tarea QA original con su script actual.
bash "$TORMANAGER" --install
CRON_PAUSED=0
echo "===== COMPROBACIÓN POST-MIGRACIÓN ====="
df -h "$SRC" /data
stat -c 'PERMISOS=%a USUARIO=%U GRUPO=%G %n' "$DEST"
crontab -l | grep "$CRON_TAG"
echo "MIGRACIÓN_OK. Datos nuevos y anteriores se guardan físicamente en $DEST"
echo "NO borrar la carpeta de respaldo $OLD hasta verificar una ronda de Cron."
