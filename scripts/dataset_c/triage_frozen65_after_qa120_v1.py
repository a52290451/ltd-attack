#!/usr/bin/env python3
"""Read-only scientific class-coverage triage after Dataset C full 120-slot QA.

Reconciles immutable 120-slot summary to the frozen 65-class cohort. Makes NO
attempts to evade access restrictions, navigate, collect, train or change A/B.
Writes one JSON and CSV report within the existing QA108 diagnostics folder.
"""
from __future__ import annotations
import csv
import json
import os
from collections import Counter,defaultdict
from pathlib import Path
import sys
from urllib.parse import urlsplit

BASE = Path("/data/ltd-c-qa-bsierra/C_DIRECT_V1")
REFERENCE = BASE / "diagnostics/pinned_qa108_20261101T211133Z_465267"
READINESS = BASE / "diagnostics/scientific_readiness_20261101T205205Z_312672/scientific_readiness.json"
RESTRICTED = {"BLOCKED","HTTP_ERROR","CHALLENGE","AUTH_REQUIRED"}
ACTIONS = {
 "BLOCKED":"ACCESS_RESTRICTED_NO_AUTOMATIC_RETRY",
 "HTTP_ERROR":"REVIEW_HTTP_AND_HUMAN_VERIFICATION_NO_BYPASS",
 "CHALLENGE":"HUMAN_VERIFICATION_NO_BYPASS",
 "AUTH_REQUIRED":"AUTH_REQUIRED_DO_NOT_BYPASS",
 "NAVIGATION_ERROR":"REVIEW_CHROME_NETWORK_LOG_POSSIBLE_SINGLE_CONTROLLED_RETRY",
 "TIMEOUT":"REVIEW_TIMEOUT_POSSIBLE_SINGLE_CONTROLLED_RETRY",
 "CONSENT_UNRESOLVED":"REVIEW_LEGITIMATE_COOKIE_POLICY_NO_AUTOMATED_CHALLENGE_BYPASS",
 "LOW_CONTENT":"REVIEW_HTML_AND_METADATA_NO_LOOSENING_QUALITY",
 "SERVER_ERROR":"REVIEW_UPSTREAM_HTTP_5XX_AND_SITE_AVAILABILITY",
}

def load(path):
 if not path.is_file() or path.is_symlink():
  raise RuntimeError("Required QA evidence missing or symlink: "+str(path))
 return json.loads(path.read_text(encoding="utf-8"))

def domain(value):
 try:
  host=urlsplit(str(value)).hostname
  return (host or "").lower().removeprefix("www.")
 except ValueError:
  return ""

def main():
 folder = Path(sys.argv[1]).resolve() if len(sys.argv)==2 else REFERENCE.resolve()
 if len(sys.argv)>2 or not folder.is_relative_to((BASE/"diagnostics").resolve()) or not folder.name.startswith("pinned_qa108_"):
  raise RuntimeError("Usage: triage_frozen65_after_qa120_v1.py [QA108_DIAGNOSTICS_DIRECTORY]")
 all_data=load(folder/"qa120_coverage_summary.json")
 source=load(folder/"qa108_summary.json")
 original=load(READINESS)
 records=all_data.get("records")
 frozen=set(original.get("c_inventory",{}).get("frozen_matched_domains",[]))
 if (len(frozen)!=65 or len(records or [])!=120 or
     all_data.get("inventory_slots")!=120 or all_data.get("inventory_distinct_urls")!=118 or
     all_data.get("slots_qa_valid")!=67 or all_data.get("slots_access_restricted")!=39 or
     all_data.get("slots_other_failure")!=14 or
     all_data.get("frozen_65_successful_site_count")!=41 or
     len(source.get("visits",[]))!=108):
  raise RuntimeError("QA120/Future-B frozen evidence invariants changed or incomplete")
 observed=defaultdict(list)
 statuses=Counter()
 for row in records:
  host=domain(row["url"])
  if host!=row["domain"] or (host in frozen)!=bool(row["frozen_65"]):
   raise RuntimeError("QA120 domain/frozen-label consistency mismatch")
  observed[host].append(row)
  statuses[(row["status"] if not row["qa_valid_any"] else "OK")]+=1
 missing=frozen-{r["domain"] for r in records if r["qa_valid_any"]}
 if len(missing)!=24:
  raise RuntimeError("Expected 24 frozen classes with no accepted QA sample")
 missing_rows=[]
 counts=Counter()
 for host in sorted(missing):
  rlist=observed[host]
  if not rlist:
   raise RuntimeError("Frozen class absent from C inventory: "+host)
  # The frozen classes were derived as a set, not from arbitrary QA label IDs.
  for r in rlist:
   if r.get("qa_valid_any"):raise RuntimeError("Unexpected successful site in missing cohort")
  state_counts=Counter(r.get("status") or "UNKNOWN" for r in rlist)
  state=state_counts.most_common(1)[0][0]
  category="HTTP_RESTRICTED" if state in RESTRICTED else "OTHER_FAILURE"
  counts[category]+=1
  for r in rlist:
   missing_rows.append({
     "frozen_site":host,"instance":r["instance"],"category":r["category"],
     "slot":r["slot"],"status":r.get("status") or "UNKNOWN",
     "failure_type":category,"evidence_source":r["evidence_source"],
     "review_action":ACTIONS.get(r.get("status"),"MANUAL_ROOT_CAUSE_REVIEW"),
   })
 if len(missing_rows)!=24 or counts["HTTP_RESTRICTED"]!=19 or counts["OTHER_FAILURE"]!=5:
  raise RuntimeError("Frozen 19 restricted + 5 other exact cohort invariant failed")
 # Link detailed QA108 outcomes by canonical instance/category/slot.
 indexed={(r["instance"],r["category"],int(r["slot"])):r for r in source["visits"]}
 for r in missing_rows:
  k=r["instance"],r["category"],int(r["slot"])
  rec=indexed.get(k)
  if rec:
   if rec.get("status")!=r["status"] or rec.get("qa_valid"):
    raise RuntimeError("QA108 detailed evidence does not agree with frozen summary")
   r["node_exit"]=rec.get("node_exit")
   r["pcap_bytes"]=rec.get("pcap_bytes")
   r["packet_count"]=rec.get("packet_count")
   r["pcap_structurally_valid"]=rec.get("pcap_parsed")
  else:
   r["node_exit"]="SEE_BASELINE_12_AND_CONTROLLED_RETRY"
 summary={
  "classification":"QA_ONLY_FROZEN_65_ACCESS_AND_VALIDITY_REVIEW",
  "scientific_capture_authorized":False,
  "frozen_total":65,
  "frozen_with_at_least_one_valid_QA_capture":41,
  "frozen_without_valid_QA_capture":24,
  "frozen_access_restricted":19,
  "frozen_other_failure":5,
  "all_120":{"valid":67,"access_restricted":39,"other_failure":14},
  "full_120_status_counts":dict(statuses),
  "frozen_65_missing_status_counts":dict(Counter(r["status"] for r in missing_rows)),
  "frozen_65_missing_records":missing_rows,
  "methodological_notice":"HTTP-blocked/consent/low-content samples must never be relabelled as valid positives. Frozen labels and A/B remain unchanged. No C tuning or training.",
  "remaining_external_gates":["A/B original capture-viewpoint and BPF parity","Tor background traffic isolation","trusted UTC/NTP clock","retention-capacity plan","scheduler fail-closed QA"],
  "sources":[str(folder/"qa120_coverage_summary.json"),str(folder/"qa108_summary.json"),str(READINESS)],
 }
 output=folder/"frozen65_access_triage.json"
 csv_path=folder/"frozen65_missing_24.csv"
 output.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
 with csv_path.open("w",encoding="utf-8",newline="") as f:
  writer=csv.DictWriter(f,fieldnames=["frozen_site","instance","category","slot","status","failure_type","review_action","evidence_source","node_exit","pcap_bytes","packet_count","pcap_structurally_valid"])
  writer.writeheader()
  writer.writerows(missing_rows)
 print("FROZEN65_VALID=41/65")
 print("FROZEN65_MISSING=24")
 print("FROZEN65_MISSING_ACCESS_RESTRICTED=19")
 print("FROZEN65_MISSING_OTHER_FAILURE=5")
 print("FROZEN65_MISSING_STATUS_COUNTS",json.dumps(summary["frozen_65_missing_status_counts"],ensure_ascii=False,sort_keys=True))
 for row in missing_rows:
  print("FROZEN_MISSING",row["frozen_site"],row["instance"],row["category"],"slot="+str(row["slot"]),
        "status="+row["status"],"ACTION="+row["review_action"])
 print("REVIEW_REPORT",output)
 print("REVIEW_CSV",csv_path)
 print("SCIENTIFIC_CAPTURE=NOT_AUTHORIZED")
 return 0

if __name__=="__main__":
 try:sys.exit(main())
 except Exception as exc:
  print("FROZEN65_TRIAGE_INCONCLUSIVE",type(exc).__name__,str(exc)[:450],file=sys.stderr)
  sys.exit(4)
