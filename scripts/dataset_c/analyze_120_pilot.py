#!/usr/bin/env python3
"""Resumen de 120 capturas P16C; solo QA, nunca evaluación de modelos."""
import csv
import json
from pathlib import Path
from collections import Counter
from datetime import datetime
import sys

if len(sys.argv) != 2:
    sys.exit("Uso: python3 analyze_120_pilot.py /ruta/audit120_...")
root = Path(sys.argv[1]).expanduser().resolve()
if not root.is_dir() or not root.name.startswith("audit120_"):
    sys.exit("STOP: ruta de auditoría inválida")

categories = [(f"inst{i}", c) for i, cc in (
    (1, ["belleza", "turismo"]),
    (2, ["bancos", "salud"]),
    (3, ["deportes", "politica"])
) for c in cc]
headers = ["instancia", "categoria", "slot", "url", "status", "http",
           "duracion_s", "pcap_bytes", "paquetes", "pcap_valido", "pcap_existe",
           "nodos_dom", "texto_visible", "imagenes", "scripts", "consentimiento",
           "errores", "uid", "json_existe"]
all_rows = []
for inst, cat in categories:
    for log in sorted((root / inst / cat).glob("attempts_*.tsv")):
        with log.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                if not r.get("uid"):
                    continue
                meta = {}
                try:
                    target = (root / Path(r["json"]).relative_to("/audit120")).resolve()
                    if target.is_relative_to(root) and target.is_file():
                        meta = json.loads(target.read_text(encoding="utf-8"))
                except (OSError, ValueError, json.JSONDecodeError):
                    pass
                pcap_exists = False
                try:
                    target_pcap = (root / Path(r["pcap"]).relative_to("/audit120")).resolve()
                    pcap_exists = target_pcap.is_relative_to(root) and target_pcap.is_file()
                except (OSError, ValueError):
                    pass
                try:
                    start = datetime.fromisoformat(meta["timestamp_start_utc"].replace("Z","+00:00"))
                    end = datetime.fromisoformat(meta["timestamp_end_utc"].replace("Z","+00:00"))
                    elapsed = round((end-start).total_seconds(), 2)
                except (ValueError, KeyError, TypeError):
                    elapsed = ""
                m = meta.get("metrics") or {}
                consent = meta.get("consent") or {}
                all_rows.append({
                    "instancia": inst, "categoria": cat, "slot": r.get("slot_index",""),
                    "url": r.get("url",""), "status": r.get("status",""),
                    "http": meta.get("http_status",""), "duracion_s": elapsed,
                    "pcap_bytes": r.get("pcap_bytes",""),
                    "paquetes": r.get("packet_count",""),
                    "pcap_valido": r.get("pcap_valid",""),
                    "pcap_existe": pcap_exists,
                    "nodos_dom": m.get("domNodes",""),
                    "texto_visible": meta.get("visible_chars",""),
                    "imagenes": m.get("images",""), "scripts": m.get("scripts",""),
                    "consentimiento": consent.get("action",""),
                    "errores": "|".join(meta.get("errors",[])),
                    "uid": r["uid"], "json_existe": bool(meta)
                })
csv_path = root / "quality_120.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=headers)
    w.writeheader()
    w.writerows(all_rows)

good = lambda r: r["status"] == "OK" and r["pcap_valido"] == "true" and r["pcap_existe"]
overall = Counter(r["status"] for r in all_rows)
valid = sum(good(r) for r in all_rows)
print(f"INTENTOS={len(all_rows)}/120 OK_PCAPP_VALID={valid} URL_UNICAS={len(set(r['url'] for r in all_rows))}")
print("ESTADOS="+json.dumps(dict(overall),ensure_ascii=False))
times = [float(r["duracion_s"]) for r in all_rows if r["duracion_s"] != ""]
if times:
    print(f"TIEMPO_MEDIO_NAV={sum(times)/len(times):.1f}s MAX={max(times):.1f}s")
report = ["# P16C QA piloto 120 posiciones (NO Dataset C científico)","",
          f"- Intentos: {len(all_rows)}/120",
          f"- OK + PCAP válido: {valid}",
          f"- Estados: {json.dumps(dict(overall),ensure_ascii=False)}",
          "",
          "| Instancia | Categoría | Intentos | OK+PCAP |",
          "|---|---|---:|---:|"]
for inst,cat in categories:
    vals = [r for r in all_rows if r["instancia"]==inst and r["categoria"]==cat]
    successes = sum(good(r) for r in vals)
    print(f"{inst}/{cat}: {len(vals)}/20, OK+PCAP={successes}")
    report.append(f"| {inst} | {cat} | {len(vals)} | {successes} |")
print("===== FALLOS O CAPTURAS NO VÁLIDAS =====")
for r in all_rows:
    if not good(r):
        print(f"{r['instancia']}/{r['categoria']} slot={r['slot']} HTTP={r['http']} {r['status']} {r['url']}")
report += ["", "Limitaciones: obfs4 y reloj del host desincronizado. Solo QA operativo.",
           "El listado congelado de 65 sitios de A/B debe cruzarse después, sin modificarlo.",
           "Los detalles completos están en quality_120.csv."]
(root / "quality_120.md").write_text("\n".join(report)+"\n",encoding="utf-8")
print("CSV="+str(csv_path))
print("INFORME="+str(root/"quality_120.md"))
if len(all_rows)!=120:
    print("AVISO: auditoría incompleta, revisar el log de ejecución.")
