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
    return values

def b_class_set(path):
    """Map the canonical numeric label IDs using the immutable CSV site_label/site pairs."""
    labels=canonical_labels()
    domain_by_label={}
    labels_by_domain={}
    raw_domains=set()
    row_count=0
    rows_in_cohort=0
    with path.open(newline="",encoding="utf-8-sig") as f:
        reader=csv.DictReader(f)
        if not reader.fieldnames or not {"site","site_label"}.issubset(reader.fieldnames):
            raise RuntimeError("INCONCLUSIVE: Future-B CSV needs site and site_label")
        for r in reader:
            row_count+=1
            label=str(r.get("site_label") or "").strip()
            host=domain(r.get("site"))
            if not host or not label:
                raise RuntimeError("INCONCLUSIVE: missing site/site_label in Future-B")
            raw_domains.add(host)
            if label in domain_by_label and domain_by_label[label]!=host:
                raise RuntimeError("STOP: label points to multiple domains: "+label)
            if host in labels_by_domain and labels_by_domain[host]!=label:
                raise RuntimeError("STOP: domain points to multiple labels: "+host)
            domain_by_label[label]=host
            labels_by_domain[host]=label
            if label in labels:rows_in_cohort+=1
    unknown=labels-set(domain_by_label)
    if unknown:raise RuntimeError("INCONCLUSIVE: missing 65-class labels in B: "+str(sorted(unknown)[:10]))
    cohort={domain_by_label[label] for label in labels}
    if len(cohort)!=65:
        raise RuntimeError("STOP: labels are not bijective for the frozen 65 domains")
    if rows_in_cohort!=18543:
        raise RuntimeError(f"INCONCLUSIVE: 65-class filtered Future-B rows={rows_in_cohort}, expected 18543")
    print("FUTURE_B_RAW_ROWS",row_count,"RAW_DOMAINS",len(raw_domains))
    print("FUTURE_B_FROZEN_ROWS",rows_in_cohort,"FROZEN_DOMAINS",len(cohort))
    print("EXCLUDED_RAW_B_DOMAINS",",".join(sorted(raw_domains-cohort)))
    return cohort,"site_label/site",row_count,rows_in_cohort

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
    b=HOME/'ltd-storage/datasets/future'/BASENAME\n    if not b.is_file() or b.is_symlink():raise RuntimeError('STOP: expected frozen Future-B source unavailable')
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
