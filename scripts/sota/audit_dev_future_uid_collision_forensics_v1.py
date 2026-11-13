#!/usr/bin/env python3
"""Forensic identity audit for Historical DEV, 09B and Future-B.

Avoid asserting leakage merely from a UID collision; compare frozen raw traces,
site labels, dates, directional/time features and content digests. Read only.
No model weights or scores are touched, no personal/trace IDs printed.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,os,sys
from collections import Counter
from pathlib import Path
import numpy as np

def require(v,msg):
    if not v:raise RuntimeError("UID_FORENSICS_BLOCKED "+msg)
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda:f.read(4*1024*1024),b""):h.update(block)
    return h.hexdigest()
def get_scores(path):
    with np.load(path,allow_pickle=False) as z:
        uid=z["pcap_uid"].astype(str)
        dates=z["query_date"].astype("datetime64[D]").astype(str)
        labels=z["candidate_labels"].astype(str)
        y=z["y_true"].astype(int)
    require(len(uid)==len(set(uid)),"duplicate score IDs")
    return {str(k):(str(labels[yi]),str(date)) for k,yi,date in zip(uid,y,dates)}
def row_digest(s):
    # Literal feature representation, no decoding/normalization: identical
    # hashes prove identical frozen text payload, differences do not alone
    # establish scientifically distinct captures.
    return hashlib.sha256(s.encode("utf-8")).hexdigest()
def extract(path,wanted):
    found={}
    if not wanted:return found
    csv.field_size_limit(sys.maxsize)
    with path.open(newline="",encoding="utf-8") as f:
        reader=csv.DictReader(f)
        require({"pcap_uid","site_label","direction_vector","time_vector"}<=set(reader.fieldnames or ()),
                "missing source fields "+str(path))
        for row in reader:
            uid=row["pcap_uid"]
            if uid not in wanted:continue
            require(uid not in found,"duplicate source UID "+str(path))
            found[uid]={
                "label":row["site_label"],
                "dir_sha":row_digest(row["direction_vector"]),
                "time_sha":row_digest(row["time_vector"]),
                "both_sha":row_digest(row["direction_vector"]+"\0"+row["time_vector"]),
                "dir_raw_len":len(row["direction_vector"]),
                "time_raw_len":len(row["time_vector"]),
            }
    require(len(found)==len(wanted),str(path)+" missing overlap UID rows "+str(len(wanted)-len(found)))
    return found
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--repo",type=Path,required=True)
    p.add_argument("--base",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    from importlib.util import spec_from_file_location,module_from_spec
    module=args.repo/"scripts/sota/sota_paths_v1.py"
    spec=spec_from_file_location("ltd_sota_paths_audit",module)
    mod=module_from_spec(spec);spec.loader.exec_module(mod)
    s1=json.loads(mod.S1.read_text())
    index=Path(s1["dev_index_file"])
    require(sha(index)==s1["dev_index_sha256"],"S1 UID index SHA wrong")
    with index.open(newline="",encoding="utf-8") as f:
        dev={}
        for row in csv.DictReader(f):
            uid=row["pcap_uid"]
            require(uid not in dev,"duplicate DEV UID")
            dev[uid]={"label":row["site_label"],"date":row["capture_date"],
                      "split":row["temporal_split"]}
    require(len(dev)==57916,"DEV population changed")
    root=args.base/"artifacts/sota/s9_netclr_author_original_v1/evaluation"
    cohorts={x:get_scores(root/x/"scores_sota_aligned.npz")
             for x in ("09B","10B_DEV_FROZEN")}
    overlap={
        "DEV_vs_09B":set(dev)&set(cohorts["09B"]),
        "DEV_vs_FutureB":set(dev)&set(cohorts["10B_DEV_FROZEN"]),
        "09B_vs_FutureB":set(cohorts["09B"])&set(cohorts["10B_DEV_FROZEN"])}
    historical_overlap=overlap["DEV_vs_FutureB"]|overlap["09B_vs_FutureB"]
    # Verify original frozen CSV signatures before making any comparisons.
    hist_manifest_source=s1["historical_checksums"]["historical_vectors"]
    historical_file=Path(hist_manifest_source["path"])
    require(sha(historical_file)==hist_manifest_source["sha256"],
            "historical vector source SHA mismatch")
    future_manifest=json.loads(mod.FUTURE.read_text())
    future_file=mod.STORAGE/"future/CLEAN_final_vectors_sites_concept_drift.csv"
    require(sha(future_file)==future_manifest["future_inputs"]["micro"]["sha256"],
            "Future-B vector source SHA mismatch")
    hist=extract(historical_file,historical_overlap)
    fut=extract(future_file,historical_overlap)
    exact=sum(hist[u]["both_sha"]==fut[u]["both_sha"] for u in historical_overlap)
    direq=sum(hist[u]["dir_sha"]==fut[u]["dir_sha"] for u in historical_overlap)
    timeeq=sum(hist[u]["time_sha"]==fut[u]["time_sha"] for u in historical_overlap)
    lab_eq=sum(hist[u]["label"]==fut[u]["label"] for u in historical_overlap)
    dev_future=overlap["DEV_vs_FutureB"]
    dev_date_same=sum(dev[u]["date"]==cohorts["10B_DEV_FROZEN"][u][1]
                      for u in dev_future)
    dev_label_same=sum(dev[u]["label"]==cohorts["10B_DEV_FROZEN"][u][0]
                       for u in dev_future)
    future_date_counts=Counter(cohorts["10B_DEV_FROZEN"][u][1] for u in dev_future)
    dev_date_counts=Counter(dev[u]["date"] for u in dev_future)
    result={
        "classification":"LTD_FROZEN_COHORT_UID_AND_RAW_TRACE_COLLISION_FORENSIC_V1",
        "frozen_source_sha_valid":True,
        "population":{"DEV":len(dev),"09B":len(cohorts["09B"]),
                      "Future-B":len(cohorts["10B_DEV_FROZEN"])},
        "uid_intersection_counts":{k:len(v) for k,v in overlap.items()},
        "historical_vs_future_overlap_union":len(historical_overlap),
        "on_overlapping_UIDs":{
            "exact_text_direction_and_time":exact,
            "identical_direction_text":direq,
            "identical_time_text":timeeq,
            "identical_site_label":lab_eq,
            "DEV_vs_FutureB_same_label":dev_label_same,
            "DEV_vs_FutureB_same_capture_date":dev_date_same,
            "DEV_overlap_capture_dates":dict(sorted(dev_date_counts.items())),
            "FutureB_overlap_capture_dates":dict(sorted(future_date_counts.items())),
        },
        "scientific_verdict":(
            "CONFIRMED_EXACT_RAW_TRACE_REUSE_WITHIN_OVERLAPPING_UIDS"
            if exact>0 else
            "UID_COLLISION_WITHOUT_EXACT_RAW_TEXT_DUPLICATION_CAUSAL_IDENTITY_UNRESOLVED"
            if historical_overlap else "NO_UID_OVERLAP"
        ),
        "cautions":[
            "Non-equal raw CSV text can still represent equivalent packet sequences; use numerical fingerprints next if required.",
            "Equal UID alone does not prove same capture or leakage.",
            "Identical direction/time vector and label across DEV/Future is evidence of duplicated input.",
            "Do not relabel, filter or alter frozen Future-B to hide overlaps.",
            "Do not certify own-model comparisons until the split identity issue is resolved independently."
        ],
    }
    require(not args.output.exists(),"refusing to overwrite "+str(args.output))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print("UID_COLLISION_FORENSIC_REPORT",args.output)
    print("UID_OVERLAPS",json.dumps(result["uid_intersection_counts"]))
    print("RAW_TEXT_EXACT_DUPLICATES",exact)
    print("SAME_DIRECTION_TEXT",direq,"SAME_TIME_TEXT",timeeq)
    print("SAME_LABEL",lab_eq,"SAME_CAPTURE_DATE",dev_date_same)
    print("UID_FORENSIC_VERDICT",result["scientific_verdict"])
if __name__=="__main__":main()
