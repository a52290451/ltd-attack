#!/usr/bin/env python3
"""Paired LTD-vs-author-original SOTA cluster bootstrap + date sign-randomization.

No model training or predictions are generated here. Both supplied NPZ files
must have the SAME 09B/Future-B UID/date/label 65-way order. Computes accuracy,
macro-F1, top5, MRR, paired date-cluster percentile CI (2,500 replicates) and
date-block sign randomization Monte Carlo p-values (2,500 permutations).

A positive delta means reference (typically LTD-Hybrid) exceeds candidate.
Statistical randomization assumes date-block exchangeability under null.
This is not a re-fit, and must not be used for model selection.
"""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
FIELDS=("accuracy","macro_f1","top5","mrr")
N={"09B":14687,"10B_DEV_FROZEN":18543}
def require(ok,msg):
    if not ok:raise RuntimeError("SOTA_PAIRED_STATS_BLOCKED "+msg)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda:f.read(4*1024*1024),b""):h.update(block)
    return h.hexdigest()
def read(path):
    with np.load(path,allow_pickle=False) as f:
        require({"pcap_uid","query_date","y_true","candidate_labels"}<=set(f.files),
                "UID/date/y/labels missing in "+str(path))
        key=next((k for k in ("sota_probs","probs","probabilities",
                            "hybrid_probs","y_prob")
                  if k in f.files),None)
        require(key is not None,"65-way probabilities missing in "+str(path))
        uid=f["pcap_uid"].astype(str)
        date=f["query_date"].astype("datetime64[D]").astype(str)
        y=f["y_true"].astype(int)
        labels=f["candidate_labels"].astype(str)
        p=np.asarray(f[key],dtype=np.float32)
    require(p.shape==(len(uid),65) and len(labels)==65
            and len(set(labels))==65 and len(set(uid))==len(uid),
            "unique UID/class or 65-probability shape invalid")
    require(np.isfinite(p).all() and
            np.allclose(p.sum(axis=1),1.0,atol=2e-4,rtol=0) and
            np.all((y>=0)&(y<65)),
            "invalid probabilities, 65-class labels")
    return uid,date,y,labels,p
def profile(p,y):
    order=np.argsort(-p,axis=1)
    pos=np.argmax(order==y[:,None],axis=1)+1
    pred=order[:,0]
    return {"pred":pred,"correct":(pred==y).astype(np.float64),
            "top5":(pos<=5).astype(np.float64),"reciprocal":1.0/pos}
def stats(a,y,ids):
    pred=a["pred"][ids]
    truth=y[ids]
    tp=np.bincount(truth[truth==pred],minlength=65)
    gt=np.bincount(truth,minlength=65)
    pp=np.bincount(pred,minlength=65)
    macro=float(np.divide(2*tp,gt+pp,out=np.zeros(65),
                          where=gt+pp>0).mean())
    return np.array([a["correct"][ids].mean(),macro,
                     a["top5"][ids].mean(),a["reciprocal"][ids].mean()])
def mixed(a,b,y,date_mask):
    # On each day assign all examples either reference or SOTA predictions
    # to A or B. Data labels remain fixed and unchanged.
    pa={k:np.where(date_mask,a[k],b[k]) for k in a}
    pb={k:np.where(date_mask,b[k],a[k]) for k in a}
    ids=np.arange(len(y))
    return stats(pa,y,ids)-stats(pb,y,ids)
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cohort",choices=tuple(N),required=True)
    ap.add_argument("--reference",type=Path,required=True)
    ap.add_argument("--candidate",type=Path,required=True)
    ap.add_argument("--reference-name",default="LTD-Hybrid")
    ap.add_argument("--candidate-name",required=True)
    ap.add_argument("--replicates",type=int,default=2500)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    require(a.replicates>=1000,"insufficient bootstrap replicates")
    ref=read(a.reference);cand=read(a.candidate)
    for i,k in enumerate(("UID","date","y_true","candidate_labels")):
        require(np.array_equal(ref[i],cand[i]),"reference/candidate "+k+
                " misaligned; do NOT reorder using inferred class labels")
    uid,dates,y,labels,p=ref
    require(len(uid)==N[a.cohort],"cohort population unexpected")
    ar=profile(p,y);br=profile(cand[-1],y)
    groups=[np.flatnonzero(dates==d)
            for d in sorted(set(dates))]
    require(len(groups)>=2 and all(len(q)>0 for q in groups),
            "paired date clusters invalid")
    rng=np.random.default_rng(42)
    idx=np.arange(len(y))
    m_ref=stats(ar,y,idx);m_cand=stats(br,y,idx)
    observed=m_ref-m_cand
    n=a.replicates
    boot=np.empty((n,4))
    perm=np.empty((n,4))
    for i in range(n):
        chosen=rng.integers(0,len(groups),len(groups))
        sampled=np.concatenate([groups[j] for j in chosen])
        boot[i]=stats(ar,y,sampled)-stats(br,y,sampled)
        # Date-level paired randomization is two-sided under H0 of
        # exchangeability, preserving each site's date/UID pairing.
        flip=rng.integers(0,2,len(groups)).astype(bool)
        use_a=np.zeros(len(y),dtype=bool)
        for j,g in enumerate(groups):use_a[g]=flip[j]
        perm[i]=mixed(ar,br,y,use_a)
    lower=np.percentile(boot,2.5,axis=0)
    upper=np.percentile(boot,97.5,axis=0)
    p=(1+(np.abs(perm)>=np.abs(observed)).sum(axis=0))/(n+1)
    report={"classification":"LTD_VS_AUTHORS_SOTA_PAIRED_DATE_INFERENCE_V1",
        "cohort":a.cohort,"n":len(uid),"classes":65,
        "date_clusters":len(groups),"seed":42,"replicates":n,
        "reference":a.reference_name,"candidate":a.candidate_name,
        "reference_sha256":sha(a.reference),
        "candidate_sha256":sha(a.candidate),
        "statistical_methods":{
            "ci":"paired cluster-by-date bootstrap, percentile 95%",
            "p":"two-sided paired date-block sign randomization, Monte Carlo",
            "p_value_resolution":1/(n+1),
            "selection_or_tuning":False},
        "metrics":{k:{"reference":float(m_ref[j]),
                      "candidate":float(m_cand[j]),
                      "delta_reference_minus_candidate":float(observed[j]),
                      "ci95_lower":float(lower[j]),
                      "ci95_upper":float(upper[j]),
                      "p_two_sided":float(p[j]),
                      "ci95_excludes_zero":bool(lower[j]>0 or upper[j]<0)}
                   for j,k in enumerate(FIELDS)}}
    require(not a.output.exists(),"report exists, do not overwrite")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+"\n")
    print("SOTA_PAIRED_DATE_STATS_PASS",json.dumps(report,indent=2),flush=True)
if __name__=="__main__":main()
