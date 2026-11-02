#!/usr/bin/env python3
"""S2 bounded read-only smoke test: DF 5000 / Var-CNN direction input feasibility.

Reads S1 DEV UID index and historical vector CSV only, stopping after the
configured sample count per DEV split. NEVER reads INTERNAL/Future-B/C payload.
Does not train, install dependencies, or generate tensor datasets.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

from src.models.sota.v1.data_adapter import (
    DEFAULT_LENGTH,DEV_SPLITS,digest,iter_dev_vectors,load_s1_dev_index
)

OPS=Path.home()/"ltd-ops"/"sota"
HIST=Path.home()/"ltd-storage/datasets/historical/CLEAN_final_vectors_sites.csv"

def fail_if(cond,reason):
    if cond:raise RuntimeError("SOTA_S2_BLOCKED "+reason)

def latest_manifest():
    reports=sorted(OPS.glob("s1_dev_frozen_*/s1_manifest.json"),
                   key=lambda p:p.stat().st_mtime,reverse=True)
    fail_if(not reports,"S1 verified DEV index not found")
    return reports[0]

def run(samples_per_split):
    fail_if(samples_per_split<1 or samples_per_split>10,
            "sample per DEV split must be 1..10")
    manifest_path=latest_manifest()
    uid_map,labels,manifest=load_s1_dev_index(manifest_path)
    source=Path(manifest["historical_checksums"]["historical_vectors"]["path"])
    fail_if(source.resolve()!=HIST.resolve() or source.is_symlink(),
            "S1 historical vector path does not match canonical source")
    fail_if(manifest.get("vector_length_audit",{}).get("observed_max_vector_length",0)<5000,
            "Historical DEV source does not contain long untruncated vectors")
    # No Future-B scores read. S1 order is not asserted to match the frozen
    # evaluation softmax order until dedicated G2 alignment is complete.
    fail_if(manifest.get("label_order_verified_against_frozen_score_arrays") is True,
            "unexpected S1 claim of externally verified class order")
    with source.open("r",encoding="utf-8",newline="") as f:
        header=csv.DictReader(f).fieldnames or []
    available={name:name in header for name in
               ("pcap_uid","site_label","direction_vector","size_vector","time_vector")}
    fail_if(not all(available[x] for x in
                    ("pcap_uid","site_label","direction_vector")),
            "canonical Historical vector schema missing")
    got=Counter()
    observations=[]
    for example in iter_dev_vectors(source,uid_map,labels,DEFAULT_LENGTH):
        if got[example.temporal_split]>=samples_per_split:continue
        got[example.temporal_split]+=1
        fail_if(len(example.directions)!=5000,"DF input shape not 5000")
        fail_if(example.label_id not in range(65),"out-of-range class ID")
        fail_if(set(example.directions)-{-1,0,1},"unexpected packet directions")
        observations.append({
            "pcap_uid":example.pcap_uid,
            "split":example.temporal_split,
            "original_length":example.original_length,
            "fixed_input_length":len(example.directions),
            "label_id":example.label_id,
            "padding_count":max(0,5000-example.original_length),
            "truncated_count":max(0,example.original_length-5000)
        })
        if all(got[s]==samples_per_split for s in DEV_SPLITS):break
    fail_if(any(got[s]!=samples_per_split for s in DEV_SPLITS),
            "not enough valid samples in each DEV split")
    result={
        "classification":"SOTA_S2_HISTORICAL_DEV_INPUT_SMOKE",
        "s1_manifest":str(manifest_path),
        "historical_vector_sha256_recorded_from_s1":
            manifest["historical_checksums"]["historical_vectors"]["sha256"],
        "historical_source_header_channels":available,
        "frozen_classes":len(labels),
        "dev_uid_count":len(uid_map),
        "samples_per_split":samples_per_split,
        "smoke_samples_processed":len(observations),
        "df_input_length":5000,
        "varcnn_direction_input_feasible":True,
        "varcnn_full_direction_time_metadata_fidelity_approved":False,
        "df_original_model_architecture_fidelity_approved":False,
        "canonical_65_probability_order_evaluated":False,
        "samples":observations,
        "future_b_opened":False,
        "trained":False,
    }
    OPS.mkdir(mode=0o700,parents=True,exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest=OPS/("s2_adapter_smoke_"+stamp+".json")
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    dest.chmod(0o600)
    print("SOTA_S2_DEV_SAMPLES",dict(got))
    print("SOTA_S2_SOURCE_CHANNELS",available)
    print("SOTA_S2_DF_INPUT_SHAPE",len(observations),"x",5000)
    print("SOTA_S2_INPUT_CONTRACT=PASS")
    print("SOTA_S2_REPORT",dest)
    print("SOTA_MODEL_TRAINING=NOT_STARTED")
    print("FUTURE_B_UNTOUCHED=YES")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--samples-per-split",type=int,default=2)
    args=ap.parse_args()
    try:run(args.samples_per_split)
    except Exception as exc:
        print("SOTA_S2_ABORTED",type(exc).__name__,str(exc)[:500],
              file=sys.stderr,flush=True)
        sys.exit(4)
