#!/usr/bin/env python3
"""15A historical rank-gap descriptive phase, reusing SHA-verified 13B/13C aggregate CSVs.
Does not open Future-B / Dataset-C and does not fit/evaluate new models.
"""
from pathlib import Path
import hashlib
import json
import pandas as pd

BASE=Path.home()/"ltd-storage/results"
B=BASE/"micro/MICRO-ROBUSTNESS-PHASE13-BATTERY"
C=BASE/"hybrid/LTD-ROBUSTNESS-PHASE13-BATTERY"
ORIGINS={"ORIGIN14","ORIGIN28"}
WINDOWS={"NEAR","MID","FAR"}
CATEGORIES={"TOP1_CORRECT","TOP5_RECOVERABLE","TOP5_MISSED"}

def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for buf in iter(lambda:f.read(1048576),b""):h.update(buf)
    return h.hexdigest()

def verify(stage,path):
    m=json.loads((path/f"{stage}_manifest.json").read_text())
    if m.get("stage")!=stage:raise RuntimeError("manifest stage mismatch")
    for name,digest in m["outputs_sha256"].items():
        file=path/name
        if not file.is_file() or sha256(file)!=digest:
            raise RuntimeError("SHA mismatch: "+str(file))
    print(f"INTEGRITY_{stage}_OK",len(m["outputs_sha256"]))

def validate(frame,keys,candidates):
    if frame.duplicated(keys).any():
        raise RuntimeError("duplicate keys in rank gap")
    assert set(frame.origin)==ORIGINS and set(frame.window)==WINDOWS
    assert set(frame.candidate)==candidates
    assert set(frame.category)==CATEGORIES, f"Unexpected rank categories {set(frame.category)}"
    actual=len(ORIGINS)*len(WINDOWS)*len(candidates)*len(CATEGORIES)
    assert len(frame)==actual,(len(frame),actual)
    groups=frame.groupby(keys[:-1],dropna=False)
    for _,g in groups:
        if len(g)!=3 or abs(g.fraction.sum()-1)>1e-4:
            raise RuntimeError("bad rank category fractions")
        if "count" in g and abs(g["count"].sum()-g["count"].sum().round())>1e-6:
            raise RuntimeError("noninteger support")
    return groups

def main():
    verify("13B",B)
    verify("13C",C)
    b=pd.read_csv(B/"13B_rank_gap_historical.csv")
    c=pd.read_csv(C/"13C_rank_gap_historical.csv")
    candidates_b={"U1","U3","T3_S11","T3_S42","T3_S73","T9"}
    candidates_c={"U1xM0","U1xM1","U3xM0","U3xM1",
                  "T3_S42xM0","T3_S42xM1","T9xM0","T9xM1"}
    validate(b,["origin","window","candidate","category"],candidates_b)
    validate(c,["origin","window","candidate","category"],candidates_c)
    summaries={"MICRO":pd.read_csv(B/"13B_model_summary.csv"),
               "HYBRID":pd.read_csv(C/"13C_factorial_summary.csv")}
    print("15A_HIST_AGGREGATE_DIAGNOSTIC")
    for label,frame,focus in [("MICRO",b,["U1","U3","T3_S42","T9"]),
                               ("HYBRID",c,["U1xM0","U3xM0","T3_S42xM0","T9xM0"])]:
        filtered=frame[frame.candidate.isin(focus)]
        table=filtered.pivot(index=["origin","window","candidate"],columns="category",values="fraction").reset_index()
        table["TOP5_TOTAL"]=table["TOP1_CORRECT"]+table["TOP5_RECOVERABLE"]
        print("\n"+label+"_RANK_TRAJECTORIES")
        print(table.sort_values(["origin","candidate","window"]).to_string(index=False,float_format=lambda x:f"{x:.5f}"))
        for origin in sorted(ORIGINS):
            for candidate in focus:
                subset=table[(table.origin==origin)&(table.candidate==candidate)].set_index("window")
                near=subset.loc["NEAR"]
                far=subset.loc["FAR"]
                print(f"NEAR_TO_FAR {label} {origin} {candidate} delta_top1={far.TOP1_CORRECT-near.TOP1_CORRECT:+.5f} delta_rank2_5={far.TOP5_RECOVERABLE-near.TOP5_RECOVERABLE:+.5f} delta_lost={far.TOP5_MISSED-near.TOP5_MISSED:+.5f}")
    print("\nLIMITATIONS: CSVs are aggregate-only. No per-capture alignment, confidence calibration, truncation, temporal-memory mechanism, or causal conclusions here.")
    print("NEXT_GATE: inspect compatible 13B/13C frozen per-capture caches; do NOT rerun training or open Future-B.")
if __name__=="__main__":main()
