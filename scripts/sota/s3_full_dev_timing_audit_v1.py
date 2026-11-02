#!/usr/bin/env python3
"""Historical DEV exhaustive time/direction quality gate, READ-ONLY.

Iterates exact 57,916 S1 UID rows, never loads INTERNAL_TEST/Future-B payload,
never tunes or trains; records negative-IAT clipping in the ORIGINAL packet
ordering (WFlib DT2 semantics) and anomalies per split/date. Does not modify
or drop captures. Uses Python stdlib only; outputs metadata JSON outside git.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter,defaultdict
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.models.sota.v1.data_adapter import load_s1_dev_index,DEV_SPLITS

OPS=Path.home()/"ltd-ops"/"sota"
EXPECTED={"DEV_EARLY":19098,"DEV_MIDDLE":20160,"DEV_LATE":18658}
FIXED_LENGTH=5000
TOLERANCE_START=1e-3
BINS=(1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,0.1,1.0)

def insist(value,message):
    if not value:raise RuntimeError("SOTA_S3_FULL_TIMING_BLOCKED "+message)

def numbers(raw,name):
    if not isinstance(raw,str) or not raw.startswith("[") or not raw.endswith("]"):
        raise ValueError(name+" not a JSON-like list")
    try:values=ast.literal_eval(raw)
    except (SyntaxError,ValueError,TypeError,RecursionError,MemoryError) as e:
        raise ValueError(name+" invalid list") from e
    if not isinstance(values,(list,tuple)) or not values:
        raise ValueError(name+" empty/not list")
    if not all(type(x) in (int,float) and math.isfinite(x) for x in values):
        raise ValueError(name+" non-finite/non-numeric value")
    return values

def summarize_trace(directions,times):
    """Pure diagnostic per trace, preserving every observed packet and time."""
    if not directions or len(directions)!=len(times):
        raise ValueError("direction/time alignment mismatch")
    if any(type(x) not in (int,float) or x not in (-1,1) for x in directions):
        raise ValueError("observed direction must be binary +/-1")
    if abs(float(times[0]))>TOLERANCE_START:
        raise ValueError("times not normalized to first packet zero")
    total=0
    used=0
    max_regression=0.0
    histogram=Counter()
    len_n=len(times)
    for i in range(len_n-1):
        delta=float(times[i+1])-float(times[i])
        if delta<0:
            total+=1
            if i<FIXED_LENGTH:used+=1
            drop=-delta
            max_regression=max(max_regression,drop)
            bucket=next((str(t) for t in BINS if drop<=t),">1s")
            histogram[bucket]+=1
    return {
        "packets":len_n,
        "negative_steps":total,
        "negative_steps_used":used,
        "negative_steps_after_window":total-used,
        "max_backward_seconds":max_regression,
        "negative_drop_bins":dict(histogram),
        "time_first_seconds":float(times[0]),
        "time_last_seconds":float(times[-1])
    }

def main(max_uids,save):
    manifest_files=sorted(OPS.glob("s1_dev_frozen_*/s1_manifest.json"),
                          key=lambda p:p.stat().st_mtime,reverse=True)
    insist(bool(manifest_files),"no S1 dev provenance manifest")
    by_uid,classes,s1=load_s1_dev_index(manifest_files[0])
    insist(len(by_uid)==57916 and len(classes)==65,"noncanonical DEV index")
    hist=Path(s1["historical_checksums"]["historical_vectors"]["path"])
    insist(hist.is_file() and not hist.is_symlink(),"historical vectors source missing")
    insist(max_uids==0 or 1<=max_uids<=57916,"max-uids outside 0..57916")
    # Historical vector file SHA was verified in S1. Rehashing bytes here
    # duplicates large IO; preserve the frozen SHA in report instead.
    counts=Counter()
    per_day=defaultdict(Counter)
    per_class=defaultdict(Counter)
    by_split=defaultdict(Counter)
    seen=set()
    errors=[]
    examples=[]
    magnitude_bins=Counter()
    aggregate=Counter()
    max_backward=0.0
    total_samples=sum(EXPECTED.values()) if max_uids==0 else max_uids
    csv.field_size_limit(sys.maxsize)
    with hist.open(newline="",encoding="utf-8") as f:
        reader=csv.DictReader(f)
        insist({"pcap_uid","site_label","direction_vector","time_vector"}
                <=set(reader.fieldnames or []),"source schema has changed")
        for row in reader:
            uid=row["pcap_uid"]
            info=by_uid.get(uid)
            if info is None:continue
            insist(uid not in seen,"duplicated canonical Historical DEV UID")
            seen.add(uid)
            label,date,split=info
            counts[split]+=1
            insist(row["site_label"]==label,"Historical Micro/Macro label mismatch")
            try:
                d=numbers(row["direction_vector"],"direction")
                t=numbers(row["time_vector"],"time")
                metrics=summarize_trace(d,t)
            except (ValueError,TypeError,OverflowError) as exc:
                aggregate["invalid"]+=1
                if len(errors)<50:
                    errors.append({"uid":uid,"split":split,"reason":str(exc)[:130]})
                by_split[split]["invalid"]+=1
                per_day[date]["invalid"]+=1
                per_class[label]["invalid"]+=1
                continue
            aggregate["valid"]+=1
            by_split[split]["valid"]+=1
            per_day[date]["valid"]+=1
            per_class[label]["valid"]+=1
            if metrics["negative_steps"]>0:
                aggregate["traces_with_negative_steps"]+=1
                by_split[split]["negative_traces"]+=1
                per_day[date]["negative_traces"]+=1
                per_class[label]["negative_traces"]+=1
                if len(examples)<20:
                    examples.append({"uid":uid,"split":split,"date":date,
                                     "negative_steps":metrics["negative_steps"],
                                     "max_backward_seconds":metrics["max_backward_seconds"]})
            for key in ("negative_steps","negative_steps_used","negative_steps_after_window"):
                aggregate[key]+=metrics[key]
                by_split[split][key]+=metrics[key]
                per_day[date][key]+=metrics[key]
            max_backward=max(max_backward,metrics["max_backward_seconds"])
            magnitude_bins.update(metrics["negative_drop_bins"])
            if aggregate["valid"]%5000==0:
                print("SOTA_S3_FULL_AUDIT_PROGRESS",len(seen),"/",total_samples,
                      "valid",aggregate["valid"],flush=True)
            if max_uids>0 and len(seen)>=max_uids:break
    current_source_sha256=None
    if max_uids==0:
        digest=hashlib.sha256()
        with hist.open("rb") as source_bytes:
            for chunk in iter(lambda:source_bytes.read(4*1024*1024),b""):
                digest.update(chunk)
        current_source_sha256=digest.hexdigest()
        insist(current_source_sha256==
               s1["historical_checksums"]["historical_vectors"]["sha256"],
               "historical vectors SHA256 changed since frozen S1")
        print("SOTA_S3_FULL_SOURCE_SHA256=PASS",flush=True)
        insist(len(seen)==57916 and dict(counts)==EXPECTED,
               "full DEV cohort not exactly 57916 and 19098/20160/18658")
        # Still write an audit report if invalid samples were encountered;
        # never release training until all 57,916 have valid aligned timing.
        insist(aggregate["valid"]==57916,
               "full Historical DEV trace count incomplete")
    result={
        "classification":"SOTA_S3_HISTORICAL_DEV_FULL_TIME_QUALITY_V1",
        "status":"PASS_FULL_POPULATION" if max_uids==0 and not aggregate["invalid"] else
                 ("BLOCKED_INVALID_TIME" if aggregate["invalid"] else "LIMITED_SMOKE_ONLY"),
        "s1_manifest":str(manifest_files[0]),
        "frozen_source_sha256":s1["historical_checksums"]["historical_vectors"]["sha256"],
        "current_source_sha256":current_source_sha256,
        "source_hash_reverified_in_full_mode":max_uids==0,
        "canonical_dev_uids":len(by_uid),
        "audited_unique_uids":len(seen),
        "max_uids_limit":max_uids,
        "split_counts_seen":dict(counts),
        "aggregate":dict(aggregate),
        "max_backward_seconds":max_backward,
        "rollback_magnitude_histogram_seconds":dict(magnitude_bins),
        "samples_by_split":{x:dict(y) for x,y in by_split.items()},
        "samples_by_day":{x:dict(y) for x,y in sorted(per_day.items())},
        "samples_by_class":{x:dict(y) for x,y in sorted(per_class.items())},
        "affected_uid_examples":examples,
        "invalid_samples":errors[:50],
        "negative_interval_policy":"WFLIB_DT2_CLIP_TO_ZERO_ORIGINAL_PACKET_ORDER",
        "source_modified":False,"future_b_opened":False,
        "trained":False,"varcnn_training_released":False,
    }
    if save:
        OPS.mkdir(mode=0o700,parents=True,exist_ok=True)
        output=OPS/("s3_full_time_quality_"+dt.datetime.now(dt.timezone.utc).
                    strftime("%Y%m%dT%H%M%SZ")+".json")
        output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
        output.chmod(0o600)
        print("SOTA_S3_FULL_AUDIT_REPORT",output)
    print("SOTA_S3_FULL_STATUS",result["status"])
    print("SOTA_S3_FULL_UIDS",len(seen),"/57916")
    print("SOTA_S3_FULL_SPLITS",dict(counts))
    print("SOTA_S3_FULL_TEMPORAL_VALID",aggregate["valid"])
    print("SOTA_S3_FULL_NEGATIVE_TRACES",aggregate["traces_with_negative_steps"])
    print("SOTA_S3_FULL_NEGATIVE_STEPS",aggregate["negative_steps"])
    print("SOTA_S3_FULL_CLIPPED_IN_5000",aggregate["negative_steps_used"])
    print("SOTA_S3_FULL_MAX_BACKWARD_SECONDS",max_backward)
    print("SOTA_S3_FULL_INVALID",aggregate["invalid"])
    print("FUTURE_B_UNTOUCHED=YES")
    if aggregate["invalid"]:
        raise RuntimeError("Invalid canonical DEV time vectors; review saved audit before model training")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--max-uids",type=int,default=0,
                   help="0 = entire DEV; >0 bounded smoke")
    p.add_argument("--save",action="store_true")
    a=p.parse_args()
    try:main(a.max_uids,a.save)
    except Exception as exc:
        print("SOTA_S3_FULL_AUDIT_ABORTED",type(exc).__name__,str(exc)[:500],
              file=sys.stderr,flush=True)
        sys.exit(4)
