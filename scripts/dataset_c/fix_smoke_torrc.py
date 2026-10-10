#!/usr/bin/env python3
"""P16C: parche idempotente, solo para piloto V2.2 del colector, sin sudo."""
from pathlib import Path
import sys

if len(sys.argv) != 2:
    raise SystemExit("Uso: python3 fix_smoke_torrc.py /ruta/smoke_inst1_belleza.sh")

path = Path(sys.argv[1]).expanduser().resolve()
if not path.is_file() or path.name != "smoke_inst1_belleza.sh":
    raise SystemExit("ERROR: ruta del piloto V2.2 no encontrada o nombre incorrecto")

content = path.read_text(encoding="utf-8")
old = "    tor -f /dev/null --SocksPort 9050 --DataDirectory /tmp/tor-c-smoke "
new = (
    '    # Tor requiere un torrc regular, no /dev/null (imagen histórica).\n'
    '    : > /tmp/tor-c-smoke.conf\n'
    '    chmod 644 /tmp/tor-c-smoke.conf\n'
    '    tor -f /tmp/tor-c-smoke.conf --SocksPort 9050 --DataDirectory /tmp/tor-c-smoke '
)
if content.count(old) == 1:
    path.write_text(content.replace(old, new, 1), encoding="utf-8")
    print(f"PATCH_OK: {path}")
elif content.count('tor -f /tmp/tor-c-smoke.conf --SocksPort') == 1:
    print("ALREADY_PATCHED: no se modificó ningún archivo")
else:
    raise SystemExit("STOP: contenido inesperado; no se aplicó el parche")
