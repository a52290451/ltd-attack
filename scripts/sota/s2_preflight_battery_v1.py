#!/usr/bin/env python3
"""Read-only pre-battery audit for the LTD-Attack frozen SOTA protocol.

Runs no legacy baseline, does not import PyTorch, does not open Future-B,
does not contact the network, does not modify A/B or Dataset C. The report
is *not* permission to launch training: new adapters and pinned resources
must still be audited explicitly.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter,defaultdict
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
SOTA=Path.home()/"ltd-ops"/"sota"
ORIGINAL_METRICS=ROOT/"scripts/diagnostics/audit_frozen_pairwise_stats.py"
PROTOCOL=ROOT/"docs/sota/SOTA_COMMON_PROTOCOL_V1.md"
S1_SOURCE=ROOT/"scripts/sota/s1_build_dev_index.py"
LEGACY=ROOT/"scripts/train/run_estado_del_arte.sh"
EXPECTED_SPLITS={"DEV_EARLY":19098,"DEV_MIDDLE":20160,"DEV_LATE":18658}
EXPECTED_METRICS=("accuracy","macro_f1","top5","mrr")
PRESERVED_CHECKPOINTS=("docs/evidence/FINAL/09A_manifest_final_dev_refit.json",
                       "docs/evidence/FINAL/09B_manifest_internal_test.json",
                       "docs/evidence/FINAL/FUTURE-B/10B_manifest_future_b.json")
BANNED_TRACKED={".pyc",".pyo",".pcap",".pcapng",".pth",".pt",".ckpt",".joblib",".swp"}
BANNED_LEAFS={".DS_Store","Thumbs.db"}

def check(ok,why):
    if not ok:raise RuntimeError("SOTA_PREFLIGHT_BLOCKED "+why)

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for buf in iter(lambda:f.read(1048576),b""):
            h.update(buf)
    return h.hexdigest()

def canonical_metric_code():
    # This does not run anything; it proves the canonical frozen audit functions
    # exist and retain their four endpoint declarations.
    tree=ast.parse(ORIGINAL_METRICS.read_text())
    metrics=None
    functions=set()
    for x in tree.body:
        if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)):
            functions.add(x.name)
        if isinstance(x,ast.Assign) and any(
            isinstance(target,ast.Name) and target.id=="METRICS" for target in x.targets):
            metrics=ast.literal_eval(x.value)
    check(metrics==EXPECTED_METRICS,"frozen metric names changed")
    required={"compute_predictions","f1_metrics","metric_totals",
              "daily_stats","bootstrap_metrics","swap_p_values","holm_adjust"}
    check(required<=functions,"missing canonical metrics/statistical functions")
    return {"names":list(metrics),"metric_source":str(ORIGINAL_METRICS),
            "sha256":sha(ORIGINAL_METRICS),"functions":sorted(required)}

def metric_toy_contract():
    # 65-class predictions: ranks = 1, 2, 6. No numpy/torch, no data.
    import math
    labels=[0,1,2]
    rankings=[
        [0]+list(range(1,65)),
        [0,1]+list(range(2,65)),
        [3,4,5,6,7,2]+[i for i in range(65) if i not in (3,4,5,6,7,2)]
    ]
    pred=[x[0] for x in rankings]
    ranks=[ranking.index(y)+1 for ranking,y in zip(rankings,labels)]
    accuracy=sum(p==y for p,y in zip(pred,labels))/3
    top5=sum(r<=5 for r in ranks)/3
    mrr=sum(1/r for r in ranks)/3
    f1=[]
    for c in range(65):
        support=sum(y==c for y in labels)
        predicted=sum(p==c for p in pred)
        tp=sum(y==c and p==c for y,p in zip(labels,pred))
        f1.append(2*tp/(support+predicted) if support+predicted else 0)
    mf1=sum(f1)/65
    check(ranks==[1,2,6],"toy reciprocal rank expected [1,2,6]")
    check(math.isclose(accuracy,1/3,abs_tol=1e-12),"toy Top1 discrepancy")
    check(math.isclose(top5,2/3,abs_tol=1e-12),"toy Top5 discrepancy")
    check(math.isclose(mrr,(1+0.5+1/6)/3,abs_tol=1e-12),"toy MRR discrepancy")
    check(math.isclose(mf1,(2/3)/65,abs_tol=1e-12),"toy Macro-F1 discrepancy")
    return {"toy_accuracy":accuracy,"toy_macro_f1":mf1,"toy_top5":top5,
            "toy_mrr":mrr,"toy_ranks":ranks,"passed":True}

def inspect_hygiene():
    raw=subprocess.run(["git","ls-files","-z"],cwd=ROOT,capture_output=True,check=True).stdout
    paths=[x.decode("utf-8","replace") for x in raw.split(b"\0") if x]
    unsafe=[]
    stale_sota=[]
    for p in paths:
        q=Path(p)
        if q.suffix.lower() in BANNED_TRACKED or q.name in BANNED_LEAFS:
            unsafe.append(p)
        if p.startswith("src/models/sota/SOTA_0"):
            stale_sota.append(p)
    # Old SOTA scripts are historical source (not automatically delete-able).
    return {"tracked_files":len(paths),"tracked_generated_candidate_paths":unsafe,
            "legacy_sota_code_for_reference_only":stale_sota,
            "automated_deletion_performed":False,
            "frozen_evidence_preserved":all((ROOT/p).is_file() for p in PRESERVED_CHECKPOINTS)}

def latest_s1():
    if not SOTA.is_dir():return None
    folders=sorted(SOTA.glob("s1_dev_frozen_*/s1_manifest.json"),
                   key=lambda p:p.stat().st_mtime,reverse=True)
    return folders[0] if folders else None

def inspect_dev_manifest():
    p=latest_s1()
    if p is None:return {"status":"PENDING","reason":"run S1 full Historical DEV index first"}
    data=json.loads(p.read_text())
    check(data.get("classification")=="SOTA_S1_HISTORICAL_DEV_ONLY_FROZEN_INDEX",
          "unexpected S1 report type")
    check(data.get("uid_count")==57916 and data.get("site_count")==65,
          "S1 DEV cohort mismatch")
    check(data.get("future_b_not_opened") is True and
          data.get("training_not_started") is True,
          "S1 not marked Future-B-safe")
    origin=data.get("temporal_assignments_provenance",{})
    check(origin.get("method") in ("ORIGINAL_FROZEN_ASSIGNMENT",
                                  "RECONSTRUCTED_FROM_VERIFIED_HISTORICAL_MACRO_DATES"),
          "unknown S1 assignment lineage")
    index=Path(data["dev_index_file"])
    labels_file=Path(data["label_order_file"])
    check(index.is_file() and labels_file.is_file(),"missing S1 output files")
    check(sha(index)==data["dev_index_sha256"] and
          sha(labels_file)==data["label_order_sha256"],
          "S1 index or label list altered since created")
    labels=json.loads(labels_file.read_text())
    check(isinstance(labels,list) and len(labels)==65 and
          len(set(labels))==65,"S1 class list not 65 unique")
    counts=Counter()
    classes=defaultdict(set)
    days=defaultdict(set)
    uids=set()
    with index.open(newline="",encoding="utf-8") as f:
        r=csv.DictReader(f)
        check(set(r.fieldnames or [])==
              {"pcap_uid","site_label","capture_date","temporal_split"},
              "S1 DEV index schema changed")
        for row in r:
            uid=row["pcap_uid"]
            check(uid and uid not in uids,"duplicate S1 UID")
            uids.add(uid)
            split=row["temporal_split"]
            check(split in EXPECTED_SPLITS,"not a DEV-only temporal row")
            check(row["site_label"] in labels,"label not in saved S1 label order")
            counts[split]+=1
            classes[split].add(row["site_label"])
            days[split].add(row["capture_date"])
    check(dict(counts)==EXPECTED_SPLITS,"S1 57916 DEV split counts not frozen")
    check(all(len(classes[s])==65 for s in EXPECTED_SPLITS),
          "S1 one of the DEV splits is missing a class")
    check([len(days[s]) for s in EXPECTED_SPLITS]==[14,14,13],
          "S1 original chronology differs")
    return {"status":"PASS","source":str(p),"uid_count":len(uids),
            "split_counts":dict(counts),"days":{s:len(days[s]) for s in EXPECTED_SPLITS},
            "original_assignments_verified":origin.get("original_assignment_hash_verified"),
            "assignment_source":origin.get("method"),
            "requires_independent_split_lineage_review":
                origin.get("original_assignment_hash_verified") is not True}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--selftest",action="store_true",
                    help="small deterministic metric and source-code tests only")
    a=ap.parse_args()
    metric=canonical_metric_code()
    toy=metric_toy_contract()
    check(PROTOCOL.is_file() and LEGACY.is_file(),"canonical protocol / legacy launcher missing")
    check("FUTURE-INFORMED FEATURE SELECTION" in
          (ROOT/"docs/FEATURE_LINEAGE.md").read_text(),
          "94F leakage warning lost from feature lineage")
    if a.selftest:
        print("SOTA_METRIC_CONTRACT_SELFTEST=PASS",toy)
        print("SOTA_FROZEN_METRICS",metric["names"])
        print("NO_DATA_NO_TRAINING_NO_FUTURE_B=YES")
        return
    hygiene=inspect_hygiene()
    dev=inspect_dev_manifest()
    report={"classification":"SOTA_PRE_BATTERY_READ_ONLY_CONTRACT_V1",
            "metric_contract":metric,"metric_selftest":toy,
            "historical_dev":dev,"repo_hygiene":hygiene,
            "legacy_runner_not_executed":True,
            "future_b_data_opened":False,
            "dataset_c_touched":False,
            "training_started":False,
            "runtime_battery_ready":False,
            "pending":["Audit original chronological split reconstruction if applicable",
                       "Validate exact 65 class order vs canonical scored NPZ (evaluation gate)",
                       "Pin external DF/VarCNN repos and approve adaptations",
                       "Implement and test actual trainers and identical evaluation export",
                       "Freeze GPU/batch/seed/search budget before any training"]}
    SOTA.mkdir(mode=0o700,parents=True,exist_ok=True)
    stamp=dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target=SOTA/("sota_battery_preflight_"+stamp+".json")
    target.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    target.chmod(0o600)
    print("SOTA_FROZEN_METRICS",metric["names"])
    print("SOTA_METRICS_SELFTEST=PASS")
    print("SOTA_S1_STATUS",dev["status"])
    if dev["status"]=="PASS":
        print("SOTA_UIDS_DEV",dev["uid_count"])
        print("SOTA_SPLIT_SOURCE",dev["assignment_source"])
        print("SOTA_SPLIT_REVIEW_PENDING",dev["requires_independent_split_lineage_review"])
    print("SOTA_TRACKED_FILE_COUNT",hygiene["tracked_files"])
    print("SOTA_GENERATED_FILES_TRACKED",hygiene["tracked_generated_candidate_paths"][:25])
    print("SOTA_LEGACY_REFERENCE_ONLY",hygiene["legacy_sota_code_for_reference_only"])
    print("SOTA_PREFLIGHT_REPORT",target)
    print("SOTA_BATTERY_RELEASE=BLOCKED_UNTIL_IMPLEMENTATION_GATES")
    print("FUTURE_B_UNTOUCHED=YES")

if __name__=="__main__":
    try:main()
    except Exception as error:
        print("SOTA_PREFLIGHT_ABORTED",type(error).__name__,str(error)[:600],
              file=sys.stderr,flush=True)
        sys.exit(4)
