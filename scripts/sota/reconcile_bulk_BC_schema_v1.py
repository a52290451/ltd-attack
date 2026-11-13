#!/usr/bin/env python3
"""Re-run ONLY the failed 7-model B/C population/epoch/hash gate and attach
a non-destructive addendum to prior bulk scientific certificate.

The original certificate is immutable. A fixed schema validator must NOT
turn adapted source code into a paper-exact method reproduction.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,sys
from datetime import datetime,timezone
from pathlib import Path
def check(b,msg):
    if not b:raise RuntimeError("BULK_BC_ADDENDUM_BLOCKED "+msg)
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--repo",type=Path,required=True)
    p.add_argument("--base",type=Path,required=True)
    p.add_argument("--previous",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    check(not a.output.exists(),"will not overwrite prior evidence")
    prev=json.loads(a.previous.read_text())
    check(prev["classification"]=="LTD_SEVEN_SOTA_BULK_SCIENTIFIC_EVIDENCE_V1",
          "wrong previous certificate")
    failures={k:v for k,v in prev["steps"].items() if v["status"]=="FAIL"}
    check(set(failures)=={"all_seven_BC_checkpoints_and_historical_only"},
          "unexpected previous failures; refusing limited addendum")
    module=a.repo/"scripts/sota/audit_bulk_seven_scientific_fidelity_v1.py"
    spec=importlib.util.spec_from_file_location("ltd_bulk_stage_gate",module)
    impl=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(impl)
    cases={}
    for name in impl.NAME_MAP:
        try:
            result=impl.stage_integrity(a.base/"artifacts/sota",name)
            cases[name]={"status":"PASS","evidence":result}
        except Exception as exc:
            cases[name]={"status":"FAIL","error":repr(exc)}
        print("BULK_BC_METHOD",name,cases[name]["status"],
              cases[name].get("error",""),flush=True)
    other_pass=all(s["status"]=="PASS" for k,s in prev["steps"].items()
                   if k not in failures)
    allok=all(x["status"]=="PASS" for x in cases.values())
    out={"classification":"LTD_SEVEN_SOTA_BC_MANIFEST_SCHEMA_REPAIR_ADDENDUM_V1",
         "created_utc":datetime.now(timezone.utc).isoformat(),
         "previous_certificate_sha256":sha(a.previous),
         "fixed_validator_sha256":sha(module),
         "model_checks":cases,
         "other_seven_previous_gates_unchanged":other_pass,
         "previous_bulk_technically_reconciled":bool(other_pass and allok),
         "status":("PASS_BULK_TECHNICAL_GATES_RECONCILED_WITH_LIMITATIONS"
                   if other_pass and allok else "FAIL_REQUIRES_INVESTIGATION"),
         "original_paper_method_fidelity":"NOT_CERTIFIED",
         "Swallow":"FAIL_COLLAPSED_BYOL_NOT_A_VALID_ORIGINAL_METHOD_BASELINE",
         "article_scope":"Retrospective LTD versus SIX named author-source adapted baselines only; exact article/paper method fidelity not established.",
         "limitations":[
             "This repair changes validation of manifest field names, never original model data.",
             "Previous original-method adaptation caveats remain valid.",
             "Future-B historical 65-class cohort selection was not fully blind.",
             "Any substantive defect requires a separate versioned experiment on Historical DEV."]}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,indent=2,ensure_ascii=False)+"\n")
    print("BULK_ADDENDUM_STATUS",out["status"],flush=True)
    print("BULK_ADDENDUM_REPORT",a.output,flush=True)
if __name__=="__main__":main()
