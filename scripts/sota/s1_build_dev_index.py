#!/usr/bin/env python3
"""Build an auditable SOTA DEV index from the frozen 09A Historical splits.

Reads ONLY Historical A vectors, Historical A macro metadata, and historical
temporal assignments; never opens Future-B, Dataset C, trained checkpoints,
or INTERNAL_TEST predictions. Does not train or perform model selection.
Outputs metadata-only 57,916-UID index and checksums in ~/ltd-ops/sota/.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[2]
MANIFEST=ROOT/"docs/evidence/FINAL/09A_manifest_final_dev_refit.json"
HIST_VECTORS=Path.home()/"ltd-storage/datasets/historical/CLEAN_final_vectors_sites.csv"
HIST_MACRO=Path.home()/"ltd-storage/datasets/historical/CLEAN_final_features_sites.csv"
ASSIGN=Path.home()/"ltd-storage/results/feature_engineering/MACRO-V2-HIST/01_temporal_split_assignments.csv"
OUT=Path.home()/"ltd-ops/sota"
TRACK=("DEV_EARLY","DEV_MIDDLE","DEV_LATE")
EXPECTED={"DEV_EARLY":19098,"DEV_MIDDLE":20160,"DEV_LATE":18658,"INTERNAL_TEST":14687}
DAY_RANGES={
 "DEV_EARLY":("2025-11-18","2025-12-01",14),
 "DEV_MIDDLE":("2025-12-02","2025-12-15",14),
 "DEV_LATE":("2025-12-16","2025-12-28",13),
 "INTERNAL_TEST":("2025-12-29","2026-01-08",11),
}
SHA_HIST_VECTORS="a8a88041c44e5627e126e2c29c7e068b17c8598a0b01c63a5ff1cd3105c2c7c9"
SHA_HIST_MACRO="cadfc0848caca0b2556e03b9061f842e19e1d99d862bc39dc7c0bf56169375ce"
SHA_ASSIGN="d095195f7b376fb8835cb4488375de68c5692573ba4e776d0b2fb909989a030d"
DATE_COMPACT=re.compile(r"20\d{6}")

def insist(cond,reason):
    if not cond:raise RuntimeError("SOTA_S1_BLOCKED "+reason)

def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()

def load_reader(path,required):
    f=path.open("r",encoding="utf-8",newline="")
    reader=csv.DictReader(f)
    missing=set(required)-set(reader.fieldnames or [])
    if missing:
        f.close()
        raise RuntimeError(f"Missing {sorted(missing)} in {path.name}")
    return f,reader

def parse_date(date_id,pcap_name):
    for candidate in (str(date_id or "").strip(),str(pcap_name or "").strip()):
        if not candidate:continue
        match=re.search(r"20\d{2}-\d{2}-\d{2}",candidate)
        if match:
            try:return dt.date.fromisoformat(match.group(0)).isoformat()
            except ValueError:pass
        match=DATE_COMPACT.search(candidate)
        if match:
            try:return dt.datetime.strptime(match.group(0),"%Y%m%d").date().isoformat()
            except ValueError:pass
    return None

def assignments():
    index={}
    counts=Counter()
    with ASSIGN.open("r",encoding="utf-8",newline="") as f:
        r=csv.DictReader(f)
        insist({"pcap_uid","temporal_split"}<=set(r.fieldnames or []),
               "temporal assignment schema changed")
        for row in r:
            uid=row["pcap_uid"].strip()
            split=row["temporal_split"].strip()
            insist(bool(uid) and uid not in index,"duplicate/blank assigned UID")
            insist(split in EXPECTED,"unrecognized historical split")
            index[uid]=split
            counts[split]+=1
    insist(dict(counts)==EXPECTED,
           "expected 57916 DEV + 14687 INTERNAL, found "+str(dict(counts)))
    return index,counts

def metadata(dev):
    found={}
    seen=Counter()
    dates=defaultdict(set)
    labels=defaultdict(set)
    f,r=load_reader(HIST_MACRO,["pcap_uid","site_label"])
    insist("date_id" in (r.fieldnames or []) or "pcap_name" in (r.fieldnames or []),
           "no historical canonical date source")
    with f:
        for row in r:
            uid=row["pcap_uid"].strip()
            split=dev.get(uid)
            if split not in TRACK:continue
            insist(uid not in found,"duplicate DEV UID in macro metadata")
            label=row["site_label"].strip()
            date=parse_date(row.get("date_id"),row.get("pcap_name"))
            insist(bool(label) and bool(date),
                   "missing historical label/date for DEV UID")
            lo,hi,_=DAY_RANGES[split]
            insist(lo<=date<=hi,"date outside frozen temporal split: "+split)
            found[uid]=(label,date,split)
            seen[split]+=1
            dates[split].add(date)
            labels[split].add(label)
    insist(sum(seen.values())==57916 and all(seen[k]==EXPECTED[k] for k in TRACK),
           "DEV macro metadata coverage mismatch "+str(dict(seen)))
    for split in TRACK:
        insist(len(dates[split])==DAY_RANGES[split][2],
               "DEV date count mismatch "+split)
        insist(len(labels[split])==65,"missing canonical class in "+split)
    return found,seen,{k:sorted(v) for k,v in dates.items()},labels

def vector_bridge(dev,meta):
    seen=set()
    count=Counter()
    labels=set()
    vector_format_checked=0
    observed_lengths=Counter()
    f,r=load_reader(HIST_VECTORS,
                    ["pcap_uid","site_label","direction_vector","size_vector"])
    with f:
        for row in r:
            uid=row["pcap_uid"].strip()
            split=dev.get(uid)
            if split not in TRACK:
                # Historical INTERNAL_TEST resides in the same physical source,
                # but no internal payload is retained or used.
                continue
            insist(uid not in seen,"duplicate DEV UID in vector source")
            seen.add(uid)
            label=row["site_label"].strip()
            insist(uid in meta,"unmatched DEV vector UID")
            insist(label==meta[uid][0],"Micro/Macro site label mismatch at DEV UID")
            direction=row["direction_vector"].strip()
            insist(direction.startswith("[") and direction.endswith("]"),
                   "DEV direction vector is not list-shaped")
            insist(row["size_vector"].strip().startswith("["),
                   "DEV size vector shape mismatch")
            payload=direction[1:-1].strip()
            observed_lengths[0 if not payload else payload.count(",")+1]+=1
            if vector_format_checked<32:
                sample=[x.strip() for x in payload.split(",")[:16]]
                try:valid_directions=all(float(v) in (-1.0,0.0,1.0) for v in sample if v)
                except ValueError:valid_directions=False
                insist(valid_directions,"unexpected numeric direction encoding")
                vector_format_checked+=1
            count[split]+=1
            labels.add(label)
    insist(len(seen)==57916 and len(meta)==len(seen),
           "DEV vector UID coverage mismatch")
    insist(all(count[k]==EXPECTED[k] for k in TRACK),
           "DEV vector temporal counts mismatch")
    insist(len(labels)==65,"DEV vector class count mismatch")
    insist(seen==set(meta),"DEV vector and macro UID sets differ")
    stats={
        "observed_max_vector_length":max(observed_lengths),
        "observed_min_vector_length":min(observed_lengths),
        "samples_with_vector_length_gt_3000":sum(n for length,n in observed_lengths.items() if length>3000),
        "samples_with_vector_length_gt_5000":sum(n for length,n in observed_lengths.items() if length>5000),
        "length_distribution_first_25_bins":dict(sorted(observed_lengths.items())[:25]),
        "all_vectors_3000_or_shorter":max(observed_lengths)<=3000,
        "possible_source_truncation_requires_extractor_audit":max(observed_lengths)<=3000,
    }
    return count,sorted(labels),stats

def run(fast):
    insist(MANIFEST.is_file(),"09A frozen source manifest absent")
    m=json.loads(MANIFEST.read_text())
    insist(m.get("training_population",{}).get("captures")==57916,
           "09A manifest no longer agrees with 57916 DEV captures")
    insist(m.get("training_population",{}).get("sites")==65,
           "09A manifest no longer agrees with 65 classes")
    for name,p in [("vectors",HIST_VECTORS),("macro",HIST_MACRO),("splits",ASSIGN)]:
        insist(p.is_file() and not p.is_symlink(),"historical "+name+" source missing")
    # The split alone can be verified cheaply, independent of installed ML packages.
    idx,counts=assignments()
    if fast:
        print("SOTA_S1_ASSIGNMENTS_PASS",dict(counts))
        print("SOTA_S1_FAST_ONLY=YES_NO_PAYLOAD_READ")
        print("FUTURE_B_UNTOUCHED=YES")
        return
    meta,macro_counts,dates,sets=metadata(idx)
    vector_counts,classes,vector_lengths=vector_bridge(idx,meta)
    insist(set(classes)==sets["DEV_EARLY"]==sets["DEV_MIDDLE"]==sets["DEV_LATE"],
           "65 DEV class mappings are not identical across folds")
    # Full byte checks against the *already frozen* Historical manifests. This
    # operation is read-only and disk intensive; use ionice on a busy capture host.
    frozen={
        "historical_vectors":(HIST_VECTORS,SHA_HIST_VECTORS),
        "historical_macro":(HIST_MACRO,SHA_HIST_MACRO),
        "historical_temporal_splits":(ASSIGN,SHA_ASSIGN),
    }
    hashes={}
    for name,(p,expected) in frozen.items():
        actual=sha256(p)
        insist(actual==expected,name+" SHA256 changed from frozen historical source")
        hashes[name]={"path":str(p),"sha256":actual}
        print("SOTA_S1_SHA256_PASS",name,flush=True)
    OUT.mkdir(parents=True,exist_ok=True)
    OUT.chmod(0o700)
    stamp=dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder=OUT/("s1_dev_frozen_"+stamp)
    folder.mkdir(mode=0o700)
    index_file=folder/"dev_index.csv"
    with index_file.open("w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f)
        writer.writerow(["pcap_uid","site_label","capture_date","temporal_split"])
        for uid,(label,date,split) in sorted(meta.items(),key=lambda item:(item[1][1],item[0])):
            writer.writerow([uid,label,date,split])
    index_file.chmod(0o600)
    label_file=folder/"label_order.json"
    label_file.write_text(json.dumps(classes,indent=2)+"\n")
    label_file.chmod(0o600)
    info={
        "classification":"SOTA_S1_HISTORICAL_DEV_ONLY_FROZEN_INDEX",
        "uid_count":len(meta),
        "site_count":len(classes),
        "split_counts":dict(counts),
        "split_dates":dates,
        "historical_checksums":hashes,
        "dev_index_file":str(index_file),
        "dev_index_sha256":sha256(index_file),
        "label_order_file":str(label_file),
        "label_order_sha256":sha256(label_file),
        "vector_representation":"direction_vector: -1/0/+1, fixed-size input not yet chosen",
        "vector_length_audit":vector_lengths,
        "faithful_original_df_5000_input_verified":False,
        "training_not_started":True,
        "future_b_not_opened":True,
        "internal_test_not_used_in_model_computation":True,
        "dataset_c_not_read":True,
        "checkpoint_selection_not_started":True,
    }
    p=folder/"s1_manifest.json"
    p.write_text(json.dumps(info,indent=2,ensure_ascii=False)+"\n")
    p.chmod(0o600)
    print("SOTA_S1_DEV_UIDS",len(meta),flush=True)
    print("SOTA_S1_DEV_DATES",len(set(d for lst in dates.values() for d in lst)),flush=True)
    print("SOTA_S1_CLASSES",len(classes),flush=True)
    print("SOTA_S1_VECTOR_LENGTH_AUDIT",vector_lengths,flush=True)
    print("SOTA_S1_SPLITS",dict(counts),flush=True)
    print("SOTA_S1_MANIFEST",p,flush=True)
    print("SOTA_S1_FROZEN_HISTORICAL=PASS",flush=True)
    print("FUTURE_B_UNTOUCHED=YES",flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--fast",action="store_true",
                    help="Verify split counts only; no dataset payload or hashing")
    args=ap.parse_args()
    try:
        csv.field_size_limit(sys.maxsize)
        run(args.fast)
    except Exception as error:
        print("SOTA_S1_ABORTED",type(error).__name__,str(error)[:500],
              file=sys.stderr,flush=True)
        sys.exit(4)
