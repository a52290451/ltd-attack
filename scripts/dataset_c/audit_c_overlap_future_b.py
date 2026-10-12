#!/usr/bin/env python3
"""Evidence-first crosswalk: newest completed 120-slot QA C vs frozen Future-B sites.
No dataset edits. If Future-B labels cannot be resolved as sites, report INCONCLUSIVE.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
from urllib.parse import urlsplit
from collections import Counter
import hashlib

HOME=Path.home()
BASE=Path("/data/ltd-c-qa-bsierra")
BASENAME="CLEAN_final_features_sites_concept_drift.csv"
ENCODER=HOME/"ltd-storage/artifacts/micro/MICRO-002-R/ds3_label_encoder_vec_3000.joblib"
EXPECTED_ENCODER_SHA="f373925b181e9700bc350a13d2f95875cec7492be66ab221fc8402de26a0013b"
SITES=("site","site_name","website","website_name","web_site","url","domain","hostname",
       "sitio","sitio_web","sitio_web_name","label","class","site_label","website_label")

def domain(raw):
    raw=str(raw or "").strip().lower()
    if not raw:return ""
    if raw.isdigit() or raw.startswith("class_") and raw[6:].isdigit():return ""
    try:
        if "://" not in raw:raw="https://"+raw
        host=(urlsplit(raw).hostname or "").strip(".")
        if host.startswith("www."):host=host[4:]
        if (not host or "." not in host or
                any(c.isspace() for c in host) or ":" in host):return ""
        return host
    except ValueError:return ""

def b_file():
    root=HOME/"ltd-storage/datasets"
    if not root.is_dir():raise RuntimeError("STOP: frozen dataset storage missing")
    choices=[p for p in root.rglob(BASENAME) if p.is_file() and not p.is_symlink()]
    if len(choices)!=1:
        raise RuntimeError(f"INCONCLUSIVE: Future-B source count={len(choices)}; provide canonical filename/path, do not guess. Matches={choices[:5]}")
    return choices[0]

def canonical_labels():
    if not ENCODER.is_file() or ENCODER.is_symlink():
        raise RuntimeError("INCONCLUSIVE: frozen label encoder missing")
    h=hashlib.sha256()
    with ENCODER.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):h.update(chunk)
    if h.hexdigest()!=EXPECTED_ENCODER_SHA:
        raise RuntimeError("STOP: frozen label encoder SHA256 mismatch")
    try:
        import joblib
    except ImportError:
        raise RuntimeError("INCONCLUSIVE: joblib unavailable in this Python. Try ~/modelos/venv/bin/python or ~/ltd-venv/bin/python without installing anything.")
    try:
        labels=joblib.load(ENCODER).classes_
    except Exception as exc:
        raise RuntimeError("INCONCLUSIVE: verified encoder cannot be loaded: "+str(exc)[:150])
    values={str(v).strip() for v in labels}
    if len(values)!=65:raise RuntimeError("STOP: canonical encoder does not have 65 distinct labels")
    parsed={domain(v) for v in values}
    if not all(parsed) or len(parsed)!=65:
        raise RuntimeError("INCONCLUSIVE: encoder class names do not map to 65 unique domain names")
    return parsed

def b_class_set(path):
    with path.open(newline="",encoding="utf-8-sig") as f:
        reader=csv.DictReader(f)
        if not reader.fieldnames:raise RuntimeError("INCONCLUSIVE: Future-B CSV has no header")
        by_lower={str(name).strip().lower():name for name in reader.fieldnames}
        candidates=[by_lower[x] for x in SITES if x in by_lower]
        if not candidates:
            raise RuntimeError("INCONCLUSIVE: Cannot identify Future-B site column; available="+",".join(reader.fieldnames[:30]))
        from itertools import chain, islice
        sample=list(islice(reader,50))
        scored=[]
        for c in candidates:
            valid=sum(bool(domain(row.get(c))) for row in sample)
            scored.append((valid,c))
        valid,chosen=max(scored,key=lambda x:x[0])
        if sample and valid < min(5,len(sample)):
            raise RuntimeError("INCONCLUSIVE: no site/domain-like column in Future-B CSV; candidates="+str(scored))
        rows=0
        uniq=set()
        invalid=Counter()
        for row in chain(sample,reader):
            rows+=1
            if rows>2_000_000:raise RuntimeError("STOP: >2 million B rows, unexpected")
            value=str(row.get(chosen) or "").strip()
            d=domain(value)
            if d:uniq.add(d)
            elif value:invalid[value[:70]]+=1
    if invalid:
        raise RuntimeError("INCONCLUSIVE: B has non-domain labels: "+str(list(invalid)[:8]))
    frozen=canonical_labels()
    absent=frozen-uniq
    if absent:raise RuntimeError("INCONCLUSIVE: frozen 65 classes absent from B source: "+str(sorted(absent)[:8]))
    retained=0
    with path.open(newline="",encoding="utf-8-sig") as source:
        for row in csv.DictReader(source):
            retained+=domain(row.get(chosen)) in frozen
    if retained!=18543:
        raise RuntimeError(f"INCONCLUSIVE: B filtered cohort {retained} rows; expected 18543; verify labels")
    return frozen,chosen,rows,retained

def c_sites():
    runs=sorted((r for r in BASE.glob("three_hourly_*") if
                 (r/"VALIDATED_THREE_FULL.txt").is_file() and
                 (r/"three_quality_summary.json").is_file()),
                key=lambda p:p.stat().st_mtime,reverse=True)
    if not runs:raise RuntimeError("STOP: no completed QA-C round with audit")
    root=runs[0]
    domains=set()
    records=[]
    for i in (1,2,3):
        for cat in ("belleza","turismo") if i==1 else ("bancos","salud") if i==2 else ("deportes","politica"):
            d=root/f"inst{i}"/cat
            for file in sorted(d.glob("attempts_*.tsv")):
                with file.open(newline="",encoding="utf-8-sig") as f:
                    for rec in csv.DictReader(f,delimiter="\t"):
                        if rec.get("status")!="OK" or str(rec.get("pcap_valid","")).lower()!="true":continue
                        p=Path(rec.get("pcap") or "")
                        if p.is_absolute() and p.parts[1:2]==("audit",):
                            p=root/Path(*p.parts[2:])
                        else:
                            p=root/p
                        if not p.resolve().is_relative_to(root.resolve()) or not p.is_file() or p.stat().st_size<=24:continue
                        site=domain(rec.get("url"))
                        if site:domains.add(site);records.append((site,rec.get("url"),cat))
    return root,domains,records

def main():
    b=b_file()
    bsites,key,brows,retained=b_class_set(b)
    root,csites,crecords=c_sites()
    common=bsites & csites
    missing=bsites-csites
    extra=csites-bsites
    result={"status":"VERIFIED_DOMAIN_EXACT","B_future_source":str(b),"B_column":key,"B_rows":brows,"B_frozen_rows":retained,
            "B_site_count":len(bsites),"C_qa_run":str(root),
            "C_eligible_attempts":len(crecords),"C_distinct_domains":len(csites),
            "matched_B_domains":len(common),"missing_B_domains":sorted(missing),
            "extra_C_domains":sorted(extra),"matching_B_domains":sorted(common)}
    out=BASE/"b_c_overlap_operational_qa.json"
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print("B_FROZEN_SITES",len(bsites),"B_COHORT_ROWS",retained,"B_SOURCE",b,"COLUMN",key)
    print("C_OK_VALID_VISITS",len(crecords),"C_OK_DISTINCT_DOMAINS",len(csites))
    print("C_B_COMMON_CLASSES",len(common),"/",len(bsites))
    print("B_CLASSES_MISSING_IN_C",len(missing))
    print("C_OK_DOMAINS_OUTSIDE_B",len(extra))
    print("MISSING_B",",".join(sorted(missing)))
    print("EXTRA_C",",".join(sorted(extra)))
    print("REPORT",out)
    print("WARNING: Domain equality does not by itself prove transport/methodological equivalence or clean longitudinal PCAP time.")
if __name__=="__main__":
    try:main()
    except Exception as e:
        print("B_C_OVERLAP_NOT_CONFIRMED",str(e))
        raise SystemExit(2)
