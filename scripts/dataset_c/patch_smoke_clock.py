#!/usr/bin/env python3
"""Preparar piloto Tor con reloj aparente corregido SOLO para Tor.

NO sincroniza el host, NO modifica el kernel y NO corrige timestamps PCAP.
SOLO SMOKE DIAGNÓSTICO P16C. NO USAR COMO SOLUCIÓN DE PRODUCCIÓN.
"""
import base64
import hashlib
from pathlib import Path
import sys
import zlib

EXPECTED_SHA256 = "68fb2c0bf13d8b6b395ceb2810c2d4cab23108dc4ebced8751b3859a2a09b063"
START = "    tor -f /tmp/tor-c-smoke.conf --SocksPort 9050 --DataDirectory /tmp/tor-c-smoke "
INSERT = '''    # PILOTO: corrección temporal de Tor; NO modifica kernel ni marcas PCAP.
    : "${LTD_TOR_CLOCK_OFFSET_SECONDS:=-2963}"
    echo "CLOCK_HOST_UTC=$(date -u +%FT%TZ)"
    echo "CLOCK_TOR_EFFECTIVE_UTC=$(LTD_CLOCK_OFFSET_SECONDS="$LTD_TOR_CLOCK_OFFSET_SECONDS" LD_PRELOAD=/opt/ltd-c-v2/libltd_realtime.so date -u +%FT%TZ)"
    LTD_CLOCK_OFFSET_SECONDS="$LTD_TOR_CLOCK_OFFSET_SECONDS" LD_PRELOAD=/opt/ltd-c-v2/libltd_realtime.so tor -f /tmp/tor-c-smoke.conf --SocksPort 9050 --DataDirectory /tmp/tor-c-smoke '''

def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Uso: python3 patch_smoke_clock.py /ruta/smoke_inst1_belleza.sh")
    source = Path(__file__).resolve().parent
    script = Path(sys.argv[1]).expanduser().resolve()
    if script.name != "smoke_inst1_belleza.sh" or not script.is_file():
        raise SystemExit("STOP: ruta de piloto inexistente o nombre incorrecto")
    root = script.parent
    text = script.read_text(encoding="utf-8")
    if "--privileged" in text or "SYS_TIME" in text:
        raise SystemExit("STOP: prohibido modificar el reloj del kernel")
    if START in text:
        if text.count(START) != 1:
            raise SystemExit("STOP: múltiple inicio de Tor; revisar manualmente")
        patched = text.replace(START, INSERT, 1)
    elif "LD_PRELOAD=/opt/ltd-c-v2/libltd_realtime.so tor -f /tmp/tor-c-smoke.conf" in text:
        patched = text
    else:
        raise SystemExit("STOP: formato de inicio de Tor inesperado")
    enc = (source / "libltd_realtime.so.zlib.b64").read_text(encoding="ascii").strip()
    data = zlib.decompress(base64.b64decode(enc, validate=True))
    actual = hashlib.sha256(data).hexdigest()
    if actual != EXPECTED_SHA256:
        raise SystemExit("STOP: SHA256 de biblioteca no coincide")
    if not data.startswith(b"\x7fELF") or data[4] != 2 or data[18:20] != b"\x3e\x00":
        raise SystemExit("STOP: biblioteca requiere Linux x86_64")
    target = root / "libltd_realtime.so"
    if target.exists() and target.read_bytes() != data:
        raise SystemExit("STOP: biblioteca existente distinta; no se sobreescribe")
    if not target.exists():
        target.write_bytes(data)
        target.chmod(0o644)
    if patched != text:
        script.write_text(patched, encoding="utf-8")
        print("PATCH_OK")
    else:
        print("ALREADY_PATCHED")
    print("SHA256=" + actual)
    print("SOLO PILOTO: Tor -2963 s aproximados; pcap usa reloj host")
    print("SMOKE=" + str(script))

if __name__ == "__main__":
    main()
