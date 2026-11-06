#!/usr/bin/env python3
"""S8 TikTok Historical DEV-only DT data smoke; no GPU, fit or external access."""
from __future__ import annotations
import csv
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.models.sota.v1.data_adapter import load_s1_dev_index,parse_direction,digest
from src.models.sota.v1.timing_adapter import parse_times
from src.models.sota.v1.tiktok_adapter import tiktok_dt_from_sequences

S1=Path.home()/"ltd-ops/sota/s1_dev_frozen_20261102T130232Z/s1_manifest.json"
COUNTS={"DEV_EARLY":19098,"DEV_MIDDLE":20160,"DEV_LATE":18658}
def main():
    uids,classes,manifest=load_s1_dev_index(S1)
    original=manifest["historical_checksums"]["historical_vectors"]
    source=Path(original["path"])
    if digest(source)!=original["sha256"]:
        raise RuntimeError("S8_HISTORICAL_SOURCE_SHA_MISMATCH")
    selected=Counter()
    seen=set()
    zero_dt=0
    negatives=0
    csv.field_size_limit(sys.maxsize)
    with source.open(newline="",encoding="utf-8") as f:
        reader=csv.DictReader(f)
        needed={"pcap_uid","site_label","direction_vector","time_vector"}
        if not needed.issubset(reader.fieldnames or []):
            raise RuntimeError("S8_REQUIRED_COLUMNS_MISSING")
        for row in reader:
            uid=row["pcap_uid"]
            if uid not in uids:continue
            if uid in seen:raise RuntimeError("S8_DUPLICATE_UID")
            seen.add(uid)
            label,date,split=uids[uid]
            if label!=row["site_label"]:raise RuntimeError("S8_LABEL_MISMATCH")
            directions=parse_direction(row["direction_vector"])
            times=parse_times(row["time_vector"])
            if len(directions)!=len(times):raise RuntimeError("S8_UNPAIRED_DIRECTION_TIME")
            if any(x not in (-1,1) for x in directions):
                raise RuntimeError("S8_OBSERVED_DIRECTION_NOT_SIGNED")
            negatives+=sum(times[i+1]<times[i] for i in range(len(times)-1))
            if selected[split]<100:
                dt=tiktok_dt_from_sequences(directions,times)
                x=np.asarray(dt,dtype=np.float32)
                if x.shape!=(5000,) or not np.isfinite(x).all():
                    raise RuntimeError("S8_DT_SHAPE_OR_FINITE_FAIL")
                zero_dt+=int(not np.any(x))
                print("S8_DT_SMOKE_SAMPLE",split,selected[split]+1,flush=True) if selected[split] in (0,99) else None
            selected[split]+=1
    if set(seen)!=set(uids) or dict(selected)!=COUNTS:
        raise RuntimeError(f"S8_HISTORICAL_COUNTS_FAIL {selected}")
    print("S8_HISTORICAL_COUNTS_PASS",dict(selected),flush=True)
    print("S8_DT_SMOKE_300_PASS","all_zero",zero_dt,"negative_timestamp_steps",negatives,flush=True)
    print("S8_HISTORICAL_DATA_GATE_PASS_NO_TRAINING",flush=True)
if __name__=="__main__":
    try:main()
    except Exception as exc:
        print("S8_HISTORICAL_DATA_GATE_ERROR",type(exc).__name__,str(exc)[:500],file=sys.stderr,flush=True)
        sys.exit(4)
