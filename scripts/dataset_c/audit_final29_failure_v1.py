#!/usr/bin/env python3
"""Read-only root-cause audit of the 2026-11-02 failed final29 QA attempt."""
import collections
import json
import pathlib
import sys

root=pathlib.Path("/data/ltd-c-qa-bsierra/C_DIRECT_V1")
stem="qa_final29_20261102T111428Z_12857"
diagnostics=root/"diagnostics"/stem
output=root/"qa_only"/stem/"outputs"
summary=diagnostics/"final29_summary.json"

if not summary.is_file():
    raise SystemExit("BLOCKED: final29 evidence not present: "+str(summary))
data=json.loads(summary.read_text())
visits=data.get("records",[])
if len(visits)!=58:
    raise SystemExit("BLOCKED: expected 58 previously recorded visits, found "+str(len(visits)))

print("FINAL29_FORENSIC_ONLY=YES")
print("ATTEMPTS",len(visits))
for field in ("phase","capture_exit","node_exit","status","qa_valid","tor_instance"):
    counts=collections.Counter(str(v.get(field)) for v in visits)
    print("EVIDENCE",field,dict(counts))
print("ARTIFACT_COUNTS",
      {"json":len(list(output.glob("*.json"))),
       "html":len(list(output.glob("*.html"))),
       "pcap":len(list(output.glob("*.pcap"))),
       "browser_logs":len(list(diagnostics.glob("*.browser.log"))),
       "capture_errors":len(list(diagnostics.glob("*.error.log"))),
       "tcpdump_logs":len(list(diagnostics.glob("*.tcpdump.log")))})
selected={}
for v in visits:
    code=(v.get("capture_exit"),v.get("node_exit"),v.get("status"))
    if str(code) not in selected:
        selected[str(code)]=v
print("DISTINCT_FAILURE_PATTERNS",len(selected))
for pattern,v in list(selected.items())[:4]:
    k=v.get("stem","")
    print("=== FAILURE",pattern,"SITE",v.get("domain"),"PHASE",v.get("phase"),"TOR",v.get("tor_instance"),"===")
    for suffix in (".error.log",".browser.log",".tcpdump.log"):
        p=diagnostics/(k+suffix)
        if p.is_file():
            txt=p.read_text(errors="replace")
            print("LOG",suffix,"BYTES",len(txt))
            print("LOG_TAIL",txt[-1600:].replace("\n"," | "))
    if k:
        artifacts=sorted(p.name for p in output.glob(k+".*"))
        print("ARTIFACTS",artifacts[:8])
print("AUDIT_ONLY_NO_SITE_VISITS=YES")
print("SCIENTIFIC_CAPTURE=NOT_AUTHORIZED")
