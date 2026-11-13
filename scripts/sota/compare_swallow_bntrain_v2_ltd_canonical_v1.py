#!/usr/bin/env python3
"""New Swallow v2 versus frozen LTD/Micro/Hybrid: canonical S5/S6-only paired metrics.
Only use after v2 release gate passes. No model fitting, selection or overwrite.
"""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
def must(ok,msg):
    if not ok:raise RuntimeError("SWALLOW_V2_COMPARISON_BLOCKED "+msg)
def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()
def metrics(y,p):
    order=np.argsort(-p,axis=1,kind="stable")
    first=order[:,0]
    rank=np.argmax(order==y[:,None],axis=1)+1
    support=np.bincount(y,minlength=65)
    predicted=np.bincount(first,minlength=65)
    correct=np.bincount(y[first==y],minlength=65)
    macro=float(np.divide(2*correct,support+predicted,
           out=np.zeros(65,dtype=np.float64),where=(support+predicted)>0).mean())
    return np.array([np.mean(first==y),macro,np.mean(rank<=5),
                     np.mean(1.0/rank)],dtype=np.float64)
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--repo",type=Path,required=True)
    p.add_argument("--base",type=Path,required=True)
    p.add_argument("--release",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    must(not a.output.exists(),"output already exists")
    rel=json.loads(a.release.read_text())
    must(rel["status"]=="PASS_ADAPTED_SWALLOW_V2_NONCOLLAPSED_TECHNICAL_GATE_NOT_PAPER_EXACT",
         "Swallow v2 release gate has not passed")
    sys.path.insert(0,str(a.repo))
    from scripts.sota.sota_paths_v1 import S5
    s5=json.loads(S5.read_text())
    must(s5["classification"]=="SOTA_S5_FROZEN_REFITS_AND_LABEL_CROSSWALK_V1",
         "S5 canonical source mismatch")
    names=("accuracy","macro_f1","top5","mrr")
    out={"classification":"LTD_SWALLOW_V2_FROZEN_S5S6_PAIRED_COMPARISON_V1",
         "release_report_sha256":digest(a.release),
         "canonical_S5_sha256":digest(S5),
         "bootstrap_by_date":2500,
         "cohorts":{},"interpretation":"source-adapted Swallow v2, not exact original Swallow"}
    root=a.base/"artifacts/sota"
    for cohort,n in (("09B",14687),("10B_DEV_FROZEN",18543)):
        freeze=s5["label_alignment"][cohort]
        original=Path(freeze["source"])
        must(digest(original)==freeze["source_sha256"],
             "original LTD SHA mismatch")
        with np.load(original,allow_pickle=False) as data:
            day=data["query_date"].astype("datetime64[D]").astype(str)
            y=data["y_true"].astype(int)
            labels=data["candidate_labels"].astype(str)
            own={field:data[key].astype(np.float64) for field,key in
                (("LTD_MICRO","micro_probs"),("LTD_MACRO","macro_final_probs"),
                 ("LTD_HYBRID","hybrid_final_probs"))}
        orig_sota=root/"s9_netclr_author_original_v1/evaluation"/cohort/"scores_sota_aligned.npz"
        with np.load(orig_sota,allow_pickle=False) as reference:
            u=reference["pcap_uid"].astype(str)
            must(np.array_equal(reference["query_date"].astype("datetime64[D]").astype(str),day),
                 "original SOTA dates not canonical")
            must(np.array_equal(reference["y_true"].astype(int),y) and
                 np.array_equal(reference["candidate_labels"].astype(str),labels),
                 "original SOTA class labels not canonical")
        score=root/"s11_swallow_authors_BNtrain_v2_seed42/evaluation"/cohort/"scores_sota_aligned.npz"
        with np.load(score,allow_pickle=False) as z:
            must(np.array_equal(z["pcap_uid"].astype(str),u) and
                 np.array_equal(z["query_date"].astype("datetime64[D]").astype(str),day)
                 and np.array_equal(z["y_true"].astype(int),y)
                 and np.array_equal(z["candidate_labels"].astype(str),labels),
                 "V2 canonical UID/label/date mismatch")
            p_v2=z["sota_probs"].astype(np.float64)
        must(p_v2.shape==(n,65) and len(np.unique(day))==(11 if cohort=="09B" else 16),
             "cohort population/date count")
        own["SWALLOW_V2"]=p_v2
        # Reuse the already validated S5/S6 daily sufficient-statistic
        # implementation: no 2500x full-packet sorting/reprocessing.
        from scripts.sota.compare_all_seven_with_original_ltd_s5_s6_v1 import evidence,values
        dates=np.unique(day)
        day_count=np.array([int(np.sum(day==d)) for d in dates],dtype=np.float64)
        rng=np.random.default_rng(20261113)
        draws=rng.multinomial(len(dates),[1/len(dates)]*len(dates),size=2500)
        scores={};samples={}
        for k,v in own.items():
            valid_days,agg=evidence(y,day,v)
            must(np.array_equal(valid_days,dates),"daily S5/S6 mismatch")
            scores[k]=np.array(values(np.ones(len(dates)),day_count,agg))
            samples[k]=np.array([values(w,day_count,agg) for w in draws])
        diffs={}
        for ours in ("LTD_MICRO","LTD_MACRO","LTD_HYBRID"):
            dif=scores[ours]-scores["SWALLOW_V2"]
            replicates=samples[ours]-samples["SWALLOW_V2"]
            diffs[ours]={m:{"delta":float(dif[j]),
                "paired_bootstrap_ci95":np.quantile(replicates[:,j],[.025,.975]).tolist()}
                for j,m in enumerate(names)}
        out["cohorts"][cohort]={"n":n,"dates":len(dates),
            "accuracy":{k:float(v[0]) for k,v in scores.items()},
            "paired_differences":diffs,"new_sota_scores_sha256":digest(score)}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,indent=2)+"\n")
    print("SWALLOW_V2_CANONICAL_PAIRWISE_PASS")
    for cohort,x in out["cohorts"].items():
        print(cohort,"LTD_HYBRID",x["accuracy"]["LTD_HYBRID"],
              "SWALLOW_V2",x["accuracy"]["SWALLOW_V2"],
              "DELTA_CI",x["paired_differences"]["LTD_HYBRID"]["accuracy"])
    print("REPORT",a.output)
if __name__=="__main__":main()
