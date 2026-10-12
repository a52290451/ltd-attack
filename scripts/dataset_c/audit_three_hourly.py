#!/usr/bin/env python3
"""P16C: audit an already-completed three-instance QA round. Read-only captures.
The report is technical QA; it does NOT validate scientific equivalence with A/B.
"""
import csv
import json
from pathlib import Path
from collections import Counter
import sys

BASE=Path("/data/ltd-c-qa-bsierra")
SPECS={1:("belleza","turismo"),2:("bancos","salud"),3:("deportes","politica")}
def target(root, raw):
    if not raw: return None
    p=Path(raw)
    # Containers write paths relative to /audit, not the host filesystem.
    if p.is_absolute():
        if p.parts[1:2] not in [("audit",),("audit120",)]:
            return None
        p=Path(*p.parts[2:])
    try:
        candidate=(root/p).resolve()
        if candidate.is_relative_to(root.resolve()): return candidate
    except (OSError,ValueError):
        pass
    return None

def main():
    if len(sys.argv)>2: sys.exit("Uso: python3 audit_three_hourly.py [ruta_del_run]")
    if len(sys.argv)==2:
        root=Path(sys.argv[1]).expanduser().resolve()
    else:
        choices=sorted((p for p in BASE.glob("three_hourly_*") if (p/"VALIDATED_THREE_FULL.txt").is_file()),
                       key=lambda p:p.stat().st_mtime,reverse=True)
        if not choices: sys.exit("STOP: no hay full validado todavía")
        root=choices[0].resolve()
    if root.parent!=BASE or not root.name.startswith("three_hourly_") or not (root/"VALIDATED_THREE_FULL.txt").is_file():
        sys.exit("STOP: no es una ronda de 3 instancias completa y validada")
    totals=Counter()
    statuses=Counter()
    by=[]
    seen_uids=set()
    duplicates=[]
    unique_urls=set()
    usable_urls=set()
    for instance,categories in SPECS.items():
        for category in categories:
            p=root/f"inst{instance}"/category
            files=sorted(p.glob("attempts_*.tsv"))
            rows=[]
            for f in files:
                with f.open(newline="",encoding="utf-8-sig") as h:
                    rows.extend(csv.DictReader(h,delimiter="\t"))
            success=0
            with_pcap=0
            validity=0
            missing=0
            for r in rows:
                status=(r.get("status") or "").strip()
                statuses[status or "EMPTY_STATUS"]+=1
                uid=(r.get("uid") or "").strip()
                if uid:
                    if uid in seen_uids: duplicates.append(uid)
                    seen_uids.add(uid)
                if r.get("url"): unique_urls.add(r["url"])
                path=target(root,(r.get("pcap") or "").strip())
                has_pcap=path is not None and path.is_file() and path.stat().st_size>24
                flagged=(r.get("pcap_valid") or "").strip().lower() in ("true","1","yes")
                if status=="OK": success+=1
                if has_pcap: with_pcap+=1
                if flagged: validity+=1
                if status=="OK" and flagged and has_pcap:
                    totals["ok_pcap_valid"]+=1
                    if r.get("url"): usable_urls.add(r["url"])
                if flagged and not has_pcap: missing+=1
            by.append({"instancia":f"inst{instance}","categoria":category,"intentos":len(rows),"ok_status":success,"pcap_presentes":with_pcap,"pcap_flag_valid":validity,"pcap_flag_missing":missing})
            totals["intentos"]+=len(rows)
            totals["categorias_completas"]+=int(len(rows)==20)
    print("RUN="+str(root))
    print("INTENTOS="+str(totals["intentos"])+"/120")
    print("CATEGORIAS_COMPLETAS="+str(totals["categorias_completas"])+"/6")
    print("OK_PCAPP_VALID="+str(totals["ok_pcap_valid"]))
    print("URL_UNICAS="+str(len(unique_urls)))
    print("URL_OK_UNICAS="+str(len(usable_urls)))
    print("PCAP_PRESENT_STRUCTURAL="+str(sum(r["pcap_presentes"] for r in by))+"/"+str(totals["intentos"]))
    print("UID_DUPLICADOS="+str(len(duplicates)))
    print("ESTADOS="+json.dumps(dict(statuses),ensure_ascii=False,sort_keys=True))
    for r in by:
        print(f'{r["instancia"]}/{r["categoria"]}: slots={r["intentos"]}/20 OK={r["ok_status"]} PCAP_presentes={r["pcap_presentes"]} PCAP_flag_valid={r["pcap_flag_valid"]} PCAP_flag_missing={r["pcap_flag_missing"]}')
    summary={"run":str(root),"scientific_dataset":False,"transport":"obfs4",
             "clock":"unsynchronized_host_unverified","totals":dict(totals),
             "status_counts":dict(statuses),"categories":by,
             "unique_url_count":len(unique_urls),"unique_ok_url_count":len(usable_urls),"duplicate_uid_count":len(duplicates)}
    out=root/"three_quality_summary.json"
    out.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("REPORT="+str(out))
    if totals["intentos"]!=120 or totals["categorias_completas"]!=6:
        sys.exit("STOP: auditoría completa incompleta")
    if totals["ok_pcap_valid"]==0:
        sys.exit("AVISO: cero capturas OK con PCAP válido; revisar recolector")
if __name__=="__main__":
    main()
