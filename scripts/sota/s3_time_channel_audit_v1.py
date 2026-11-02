#!/usr/bin/env python3
"""Read-only Historical DEV timing-channel audit for WFlib VarCNN DT2.

DO NOT synthesize timing, normalize, select samples or train models.
Inspects up to n canonical DEV records per split and records numeric
characteristics of time_vector vs direction_vector; no future data access.
"""
from __future__ import annotations
import argparse
import ast
from array import array
from collections import Counter
import csv
import datetime as dt
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.models.sota.v1.data_adapter import load_s1_dev_index,DEV_SPLITS
from src.models.sota.v1.timing_adapter import dt2_from_sequences,parse_times
OPS=Path.home()/"ltd-ops"/"sota"

def fail_if(condition,message):
    if condition:raise ValueError("SOTA_S3_TIME_AUDIT_BLOCKED "+message)

def parse_numeric(name,value):
    try:a=ast.literal_eval(value)
    except (SyntaxError,ValueError,TypeError) as e:
        raise ValueError("invalid vector "+name) from e
    fail_if(not isinstance(a,(list,tuple)) or not a,"empty/non-list "+name)
    fail_if(any(type(x) not in (int,float) or not math.isfinite(x) for x in a),
            "non-finite "+name)
    return a

def main(n):
    fail_if(n not in (1,2,3,4,5),"sample limit must be 1..5 per split")
    paths=sorted(OPS.glob("s1_dev_frozen_*/s1_manifest.json"),
                 key=lambda p:p.stat().st_mtime,reverse=True)
    fail_if(not paths,"frozen DEV S1 manifest missing")
    id_map,classes,manifest=load_s1_dev_index(paths[0])
    source=Path(manifest["historical_checksums"]["historical_vectors"]["path"])
    fail_if(source.is_symlink() or not source.is_file(),
            "original historical vector source missing")
    taken=Counter()
    audit=[]
    csv.field_size_limit(sys.maxsize)
    with source.open(newline="",encoding="utf-8") as f:
        r=csv.DictReader(f)
        fail_if(not {"pcap_uid","site_label","direction_vector","time_vector"}
                .issubset(r.fieldnames or []),"source channels missing")
        for row in r:
            metadata=id_map.get(row["pcap_uid"])
            if metadata is None:continue
            site,date,split=metadata
            if taken[split]>=n:continue
            fail_if(row["site_label"]!=site,"site label not aligned")
            d=parse_numeric("direction",row["direction_vector"])
            t=parse_times(row["time_vector"])
            fail_if(any(x not in (-1,1) for x in d),
                    "non-binary observed direction in DEV raw")
            dt2=dt2_from_sequences(array("b",(int(v) for v in d)),t,5000)
            drops=sum(t[i]<t[i-1]-1e-8 for i in range(1,len(t)))
            matched=len(d)==len(t)
            t_nonnegative=all(x>=0 for x in t)
            audit.append({
                "split":split,"date":date,"direction_length":len(d),
                "time_length":len(t),"lengths_match":matched,
                "time_min":float(min(t)),"time_max":float(max(t)),
                "time_first":float(t[0]),"time_last":float(t[-1]),
                "time_nonnegative":t_nonnegative,
                "time_non_decreasing":drops==0,
                "time_decrease_edges":drops,
                "time_all_zero":all(x==0 for x in t),
                "time_unique_values":len(set(t)),
                "dt2_input_shape":[2,dt2.input_length],
                "dt2_gaps_considered":dt2.gaps_considered,
                "dt2_first_packet_direction":float(dt2.direction[0]),
                "dt2_first_gap_seconds":float(dt2.interarrival[0]),
                "dt2_last_included_gap_seconds":
                    float(dt2.interarrival[4999]),
            })
            taken[split]+=1
            if all(taken[s]>=n for s in DEV_SPLITS):break
    fail_if(any(taken[s]!=n for s in DEV_SPLITS),
            "not enough canonical DEV timing samples in each split")
    all_aligned=all(x["lengths_match"] for x in audit)
    all_monotonic=all(x["time_non_decreasing"] and x["time_nonnegative"]
                      for x in audit)
    no_zero=all(not x["time_all_zero"] for x in audit)
    report={
        "classification":"SOTA_S3_HISTORICAL_DEV_TIME_CHANNEL_AUDIT",
        "source_s1_manifest":str(paths[0]),
        "sample_counts":dict(taken),"direction_time_same_length":all_aligned,
        "nonnegative_monotonic_time_samples":all_monotonic,
        "time_not_all_zero":no_zero,
        "time_semantics_proven":"NO",
        "wflib_dt2_requires_signed_absolute_packet_timestamps":True,
        "varcnn_dt2_samples_passed":len(audit),
        "varcnn_dt2_translation_smoke_passed":True,
        "varcnn_dt2_conversion_authorized":False,
        "training_started":False,"future_b_opened":False,
        "samples":audit,
    }
    path=OPS/("s3_time_channel_"+dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")+".json")
    path.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    path.chmod(0o600)
    print("SOTA_S3_TIME_SAMPLES",dict(taken))
    print("SOTA_S3_TIME_DIRECTION_ALIGNMENT",all_aligned)
    print("SOTA_S3_TIME_NONNEGATIVE_MONOTONIC",all_monotonic)
    print("SOTA_S3_TIME_NONZERO",no_zero)
    print("SOTA_S3_DT2_TENSOR_SHAPE","[2,5000]")
    print("SOTA_S3_DT2_SMOKE_SAMPLES",len(audit))
    print("SOTA_S3_TIME_SEMANTICS=REQUIRES_ORIGINAL_EXTRACTOR_PROVENANCE")
    print("SOTA_S3_TIMING_REPORT",path)
    print("VARCNN_DT2_TRAINING=NOT_AUTHORIZED")
    print("FUTURE_B_UNTOUCHED=YES")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--per-split",type=int,default=2)
    a=p.parse_args()
    try:main(a.per_split)
    except Exception as exc:
        print("SOTA_S3_TIME_AUDIT_ABORTED",type(exc).__name__,str(exc)[:600],
              file=sys.stderr,flush=True)
        sys.exit(4)
