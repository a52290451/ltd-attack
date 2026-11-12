#!/usr/bin/env python3
"""Full-population source timestamp/packet audit for frozen 09B and Future-B.

Parses ALL cohort traces from canonical CSV, verifies source SHA/UID/labels,
summarizes timing semantics by cohort/date and 65 classes. CPU/read-only.
No model inference, training, feature changes or model selection.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np

def require(ok,msg):
    if not ok:raise RuntimeError("TEMPORAL_FULL_AUDIT_BLOCKED "+msg)
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
    return h.hexdigest()
def summary(values):
    z=np.asarray(values,dtype=np.float64)
    return {"n":int(len(z)),"median":float(np.median(z)),
            "mean":float(z.mean()),
            "p05":float(np.percentile(z,5)),
            "p95":float(np.percentile(z,95)),
            "min":float(z.min()),"max":float(z.max())}
def audit_one(repo,base,storage,cohort):
    sys.path.insert(0,str(repo))
    from src.models.sota.v1.data_adapter import parse_direction
    from src.models.sota.v1.timing_adapter import parse_times
    root=base/"artifacts/sota"
    reference=root/"s12_wfcat_authors_iat_seed42/evaluation"/cohort/"scores_sota_aligned.npz"
    with np.load(reference,allow_pickle=False) as f:
        u=f["pcap_uid"].astype(str)
        y=f["y_true"].astype(int)
        names=f["candidate_labels"].astype(str)
        dates=f["query_date"].astype("datetime64[D]").astype(str)
    require(len(u)==(14687 if cohort=="09B" else 18543),cohort+" canonical UID count")
    expected={str(v):(str(names[c]),str(dt),int(c)) for v,c,dt in zip(u,y,dates)}
    csvfile=(storage/"historical/CLEAN_final_vectors_sites.csv"
            if cohort=="09B" else
            storage/"future/CLEAN_final_vectors_sites_concept_drift.csv")
    if cohort=="09B":
        # S1 frozen source SHA
        from scripts.sota.sota_paths_v1 import S1
        m=json.loads(S1.read_text())
        checksum=m["historical_checksums"]["historical_vectors"]["sha256"]
    else:
        from scripts.sota.sota_paths_v1 import FUTURE
        m=json.loads(FUTURE.read_text())
        checksum=m["future_inputs"]["micro"]["sha256"]
    require(sha(csvfile)==checksum,cohort+" raw source SHA invalid")
    result={}
    bydate=defaultdict(list);byclass=defaultdict(list)
    csv.field_size_limit(sys.maxsize)
    with csvfile.open(newline="",encoding="utf-8") as stream:
        rd=csv.DictReader(stream)
        require({"pcap_uid","site_label","direction_vector","time_vector"}<=set(rd.fieldnames or ()),
                cohort+" missing source columns")
        for row in rd:
            uid=row["pcap_uid"]
            if uid not in expected:continue
            require(uid not in result,cohort+" duplicate raw source UID")
            label,date,cl=expected[uid]
            require(row["site_label"]==label,cohort+" UID/label mismatch")
            d=parse_direction(row["direction_vector"])
            t=np.asarray(parse_times(row["time_vector"]),dtype=np.float64)
            require(len(d)==len(t)>1,cohort+" invalid trace length")
            require(np.isfinite(t).all() and np.all(t>=0),cohort+" invalid timestamp")
            backward=int((np.diff(t)<0).sum())
            rec=(len(d),float(t[-1]),float(t.max()-t.min()),
                 backward,int(np.sum(t==t[0])))
            result[uid]=rec;bydate[date].append(rec);byclass[cl].append(rec)
    require(len(result)==len(u) and len(byclass)==65,cohort+" incomplete UID/classes")
    def profile(rows):
        a=np.asarray(rows,dtype=np.float64)
        return {
            "n":len(a),
            "packet_count":summary(a[:,0]),
            "last_timestamp":summary(a[:,1]),
            "duration_range_seconds_if_source_is_seconds":summary(a[:,2]),
            "timestamp_regression_trace_fraction":float(np.mean(a[:,3]>0)),
            "total_regression_pairs":int(a[:,3].sum()),
            "median_initial_equal_timestamps":float(np.median(a[:,4]))}
    rows=list(result.values())
    return {"status":"PASS_CANONICAL_SOURCE_SHA_UID_AND_LABEL",
            "cohort":cohort,"n":len(rows),"source_sha256":checksum,
            "all":profile(rows),
            "by_date":{k:profile(v) for k,v in sorted(bydate.items())},
            "by_class":{str(k):profile(v) for k,v in sorted(byclass.items())},
            "limitations":[
              "Durations are expressed in source timestamp units; units must be independently verified against acquisition metadata.",
              "Population shift alone does not prove causal explanation for WFCAT failure.",
              "Feature-transform equivalence to the authors' reference is a separate gate."]}
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--base",type=Path,required=True)
    p.add_argument("--repo",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    from os import environ
    storage=Path(environ.get("LTD_SOTA_STORAGE_ROOT",str(Path.home()/"ltd-storage/datasets")))
    doc={"classification":"LTD_FULL_EXTERNAL_TEMPORAL_SEMANTICS_AUDIT_V1",
         "cohorts":{}}
    for cohort in ("09B","10B_DEV_FROZEN"):
        doc["cohorts"][cohort]=audit_one(args.repo,args.base,storage,cohort)
        z=doc["cohorts"][cohort]["all"]
        print("TEMPORAL_POPULATION_AUDIT_PASS",cohort,"n",z["n"],
              "packets_median",z["packet_count"]["median"],
              "duration_median",z["duration_range_seconds_if_source_is_seconds"]["median"],
              "regression_frac",z["timestamp_regression_trace_fraction"],flush=True)
    require(not args.output.exists(),"refusing overwrite "+str(args.output))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(doc,ensure_ascii=False,indent=2)+"\n")
    print("TEMPORAL_FULL_AUDIT_REPORT",args.output,flush=True)
if __name__=="__main__":main()
