#!/usr/bin/env python3
"""Append-only scientific interpretation of UID collision and numeric identity evidence.

Does NOT edit the original certification result or convert incomplete method
fidelity into a pass. Requires original run plus SHA-verified numeric report.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
def need(v,msg):
    if not v:raise RuntimeError("SOTA_UID_RECONCILIATION_BLOCKED "+msg)
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--original-certificate",required=True,type=Path)
    p.add_argument("--uid-report",required=True,type=Path)
    p.add_argument("--numeric-report",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args()
    cert=json.loads(a.original_certificate.read_text())
    uid=json.loads(a.uid_report.read_text())
    num=json.loads(a.numeric_report.read_text())
    need(cert["classification"]=="LTD_SOTA_METHOD_CERTIFICATION_SUITE_V1",
         "original certificate invalid")
    need(uid["classification"]=="LTD_FROZEN_COHORT_UID_AND_RAW_TRACE_COLLISION_FORENSIC_V1",
         "UID report invalid")
    need(num["classification"]=="LTD_CANONICAL_NUMERIC_ALL_COHORT_TRACE_IDENTITY_V1",
         "numeric report invalid")
    need(num["data_source_sha_verified"] is True,
         "input data digests were not validated")
    need(num["n"]=={"DEV":57916,"09B":14687,"Future-B":18543},
         "wrong frozen populations")
    pair=num["uid_overlap_by_origin"]
    old=uid["uid_intersection_counts"]
    need(pair.get("DEV",0)==old["DEV_vs_FutureB"] and
         pair.get("09B",0)==old["09B_vs_FutureB"],
         "UID collision counts changed")
    exact=num["global_direction_and_time_exact_matches_future_to_historical"]
    same_date=num["capture_date_overlaps"]["DEV_vs_FutureB_common_dates"]
    need(not a.output.exists(),"refusing overwrite")
    tests={key:v["status"] for key,v in cert["steps"].items()}
    other_fail={k:v for k,v in tests.items() if v=="FAIL" and
                k!="historical_external_UID_disjoint"}
    no_exact=(exact==0 and same_date==0)
    r={
       "classification":"LTD_UID_COLLISION_RECONCILED_CERTIFICATION_ADDENDUM_V1",
       "original_certificate_sha256":sha(a.original_certificate),
       "uid_forensics_sha256":sha(a.uid_report),
       "numeric_forensics_sha256":sha(a.numeric_report),
       "original_certificate_status":cert["status"],
       "original_UID_disjoint_failure":"SUPERSEDED_FOR_CAPTURE_IDENTITY_INTERPRETATION_ONLY",
       "UIDs_are_not_globally_unique":bool(sum(pair.values())),
       "global_numeric_duplicate_count":exact,
       "DEV_FutureB_shared_capture_dates":same_date,
       "other_certification_failed_gates":other_fail,
       "Swallow":"FAIL_COLLAPSED_PRETRAIN",
       "own_model_comparison":tests.get("own_model_canonical_score_alignment","PENDING"),
       "method_fidelity":cert["fidelity"]["paper_exact_status"],
       "limited_identity_verdict":(
          "PASS_NO_EXACT_CROSS_COHORT_NUMERIC_DUPLICATES"
          if no_exact else "FAIL_OR_UNRESOLVED_CROSS_COHORT_CAPTURE_IDENTITY"),
       "scientific_certification":"NOT_CERTIFIED",
       "may_reuse_frozen_scores_for_descriptive_comparison":bool(no_exact and not other_fail),
       "reason":"Numeric capture distinctness and frozen metrics do not demonstrate author-method parity, no near duplicates, or own-model alignment.",
       "unresolved":["independent author functional parity across all 7 implementations",
                     "Swallow collapsed pretraining remediation on Historical DEV as new experiment",
                     "own-model frozen cohort score alignment",
                     "capture timestamp unit provenance and near-duplicate/acquisition audit"],
    }
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(r,ensure_ascii=False,indent=2)+"\n")
    print("UID_CERTIFICATE_RECONCILED",r["limited_identity_verdict"])
    print("SUITABLE_FOR_FINAL_PAPER",False)
    print("RECONCILIATION_REPORT",a.output)
if __name__=="__main__":main()
