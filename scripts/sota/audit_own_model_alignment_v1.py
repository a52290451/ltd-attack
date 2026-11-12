#!/usr/bin/env python3
"""Optional comparison contract: own LTD predictions versus 7 frozen SOTA scores.

No fitting. Requires standard NPZ arrays pcap_uid, query_date, y_true,
candidate_labels, sota_probs (N x 65). Own predictions with alternative key
probabilities can be exported to canonical schema in a separately versioned
conversion; never guess class order. Output a verification JSON.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def require(ok,msg):
    if not ok:raise RuntimeError("OWN_MODEL_COMPARISON_BLOCKED "+msg)
def load(path):
    with np.load(path,allow_pickle=False) as data:
        require(set(("pcap_uid","query_date","y_true","candidate_labels","sota_probs"))<=set(data.files),
                "canonical NPZ columns missing "+str(path))
        uid=data["pcap_uid"].astype(str)
        date=data["query_date"].astype("datetime64[D]").astype(str)
        yt=data["y_true"].astype(np.int64)
        names=data["candidate_labels"].astype(str)
        probs=np.asarray(data["sota_probs"],dtype=np.float32)
    require(probs.shape==(len(uid),65) and len(names)==65,
            "prediction/class shape")
    require(len(set(uid))==len(uid) and len(set(names))==65,"UID/classes not unique")
    require(np.isfinite(probs).all() and np.all(probs>=0) and
            np.allclose(probs.sum(1),1,atol=2e-4,rtol=0),"invalid probabilities")
    return uid,date,yt,names,probs

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--base",type=Path,default=Path("/data/datasets_brayan/ltd-sota-2026"))
    p.add_argument("--own-09b",type=Path,required=True)
    p.add_argument("--own-future-b",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    out={"classification":"LTD_SOTA_OWN_MODEL_ALIGNED_COMPARISON_V1",
         "cohorts":{}}
    for cohort,own in (("09B",a.own_09b),("10B_DEV_FROZEN",a.own_future_b)):
        ref=a.base/"artifacts/sota/s9_netclr_author_original_v1/evaluation"/cohort/"scores_sota_aligned.npz"
        r=load(ref)
        x=load(own)
        names=("pcap_uid","query_date","y_true","candidate_labels")
        for name,v1,v2 in zip(names,r[:4],x[:4]):
            require(np.array_equal(v1,v2),cohort+" "+name+" is NOT identical to SOTA")
        require(len(x[0])==(14687 if cohort=="09B" else 18543),
                "wrong cohort count")
        pred=x[4].argmax(1)
        correct=pred==x[2]
        out["cohorts"][cohort]={
            "status":"OWN_MODEL_UID_DATE_LABEL_CLASS_PARITY_PASS",
            "n":len(x[0]),"own_accuracy":float(correct.mean()),
            "own_score_npz_sha256":hashlib.sha256(own.read_bytes()).hexdigest(),
            "reference_sota_npz_sha256":hashlib.sha256(ref.read_bytes()).hexdigest()}
        print("OWN_MODEL_ALIGNMENT_PASS",cohort,len(x[0]),
              out["cohorts"][cohort]["own_accuracy"],flush=True)
    require(not a.output.exists(),"refuse overwrite")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,indent=2)+"\n")
    print("OWN_MODEL_COMPARISON_ALIGNMENT_PASS",a.output,flush=True)
if __name__=="__main__":main()
