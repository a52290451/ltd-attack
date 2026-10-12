#!/usr/bin/env python3
"""Read-only P15A-MECH closeout; validate outputs, then print compact auditable results."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd

ROOT=Path.home()/"ltd-storage/results/diagnostics/P15A-MECH"
FILES=("P15A_mechanisms_summary.csv","P15A_day_counts.csv","P15A_site_counts.csv","P15A_day_block_bootstrap.csv")
def sha256(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()
def check():
    mf=ROOT/"P15A_manifest.json"
    m=json.loads(mf.read_text())
    if m.get("stage")!="P15A_MECH" or m.get("status")!="DESCRIPTIVE_ONLY":raise RuntimeError("BAD_STAGE_STATUS")
    if m.get("training_allowed") is not False or m.get("selection_allowed") is not False or m.get("future_b_opened") is not False:
        raise RuntimeError("DATA_POLICY_VIOLATION")
    if set(m.get("outputs_sha256",{}))!=set(FILES):raise RuntimeError("OUTPUT_SET_CHANGED")
    for name,digest in m["outputs_sha256"].items():
        if sha256(ROOT/name)!=digest:raise RuntimeError("OUTPUT_HASH_MISMATCH "+name)
    print("P15A_OUTPUT_SHA_VERIFIED",len(FILES))
    summary=pd.read_csv(ROOT/FILES[0])
    days=pd.read_csv(ROOT/FILES[1])
    sites=pd.read_csv(ROOT/FILES[2])
    boots=pd.read_csv(ROOT/FILES[3])
    assert len(summary)==len(boots)==18
    assert not summary.duplicated(["origin","window","candidate"]).any()
    assert not boots.duplicated(["origin","window","candidate"]).any()
    assert set(summary.origin)=={"ORIGIN14","ORIGIN28"}
    assert set(summary.window)=={"NEAR","MID","FAR"}
    assert set(summary.candidate)=={"U1","U3","T3_S42"}
    assert len(sites.groupby(["origin","window","candidate"]))==18
    for key,g in summary.groupby(["origin","window","candidate"]):
        n=int(g.iloc[0].n)
        assert int(g.iloc[0].rescued)-int(g.iloc[0].harmed)==int(g.iloc[0].net)
        sd=days.loc[(days.origin==key[0])&(days.window==key[1])&(days.candidate==key[2])]
        ss=sites.loc[(sites.origin==key[0])&(sites.window==key[1])&(sites.candidate==key[2])]
        if sd.n.sum()!=n or ss.n.sum()!=n:raise RuntimeError("SUPPORT_INCONSISTENCY "+str(key))
        if sd.rescued.sum()!=g.iloc[0].rescued or ss.rescued.sum()!=g.iloc[0].rescued:raise RuntimeError("RESCUE_INCONSISTENCY "+str(key))
        if sd.harmed.sum()!=g.iloc[0].harmed or ss.harmed.sum()!=g.iloc[0].harmed:raise RuntimeError("HARM_INCONSISTENCY "+str(key))
    merged=summary.merge(boots,on=["origin","window","candidate"],validate="one_to_one")
    print("PAIRED_18_VALIDATED")
    print("DAY_BLOCK_BOOTSTRAP_TOP5_CI")
    print(merged[["origin","window","candidate","delta_top1","delta_top5","top1_ci_low","top1_ci_high","top5_ci_low","top5_ci_high"]].to_string(index=False,float_format=lambda z:f"{z:+.5f}"))
    print("OUTCOME_CONFIDENCE_MEANS")
    cols=["origin","window","candidate","rescued_by_hybrid_m_confidence","harmed_by_hybrid_m_confidence",
          "rescued_by_hybrid_h_confidence","harmed_by_hybrid_h_confidence",
          "m_wrong_conf_ge_08_fraction","h_wrong_conf_ge_08_fraction"]
    print(summary[cols].to_string(index=False,float_format=lambda z:f"{z:.5f}"))
    print("SITE_EFFECT_DISTRIBUTION")
    sites["net"]=sites.rescued-sites.harmed
    g=sites.groupby(["origin","window","candidate"])
    tally=g.agg(sites=("site_label","nunique"),net_positive_sites=("net",lambda x:int((x>0).sum())),
                net_negative_sites=("net",lambda x:int((x<0).sum())),
                net_zero_sites=("net",lambda x:int((x==0).sum())))
    print(tally.to_string())
    print("P15A_MECH_CLOSEOUT_VALIDATED: all files immutable, descriptive Historical only")
if __name__=="__main__":check()
