#!/usr/bin/env python3
"""Canonical LTD (Micro/Macro/Hybrid) vs seven frozen author-source SOTA baselines.

REUSES the existing S5 65-class crosswalk and S6 frozen score alignment:
- S5 SHA-pinned original 09B and Future-B model scores (no UID pickle loaded).
- Already source-aligned seven SOTA NPZ carry verified UID row order.
- Match canonical original dates, y_true and candidate_labels EXACTLY, by row.
- Recover the implicit row-UID contract via the original S5 canonical source
  and existing S5/10B evaluation lineage (do NOT join cross-campaign UIDs).
- Compute four metrics and 2500 day-block PAIRED bootstrap percentile CIs,
  for all 10 predictors across 09B and Future-B.
No training, no score modification, no Future-B tuning. Output under audits.
"""
from __future__ import annotations
import argparse,hashlib,json,os,sys
from pathlib import Path
import numpy as np

LTD_FIELDS={"LTD_MICRO":"micro_probs",
            "LTD_MACRO":"macro_final_probs",
            "LTD_HYBRID":"hybrid_final_probs"}
SOTA_FOLDERS={
    "SOTA_DF":"authors_df_seed42",
    "SOTA_VarCNN_direction":"authors_varcnn_seed42",
    "SOTA_TikTok":"authors_tiktok_seed42",
    "SOTA_NetCLR":"s9_netclr_author_original_v1",
    "SOTA_RoFiRe":"authors_rofire_seed42",
    "SOTA_Swallow_COLLAPSED":"s11_swallow_authors_seed42",
    "SOTA_WFCAT":"s12_wfcat_authors_iat_seed42",
}
COHORTS={"09B":(14687,11),"10B_DEV_FROZEN":(18543,16)}
METRICS=("accuracy","macro_f1","top5","mrr")

def check(ok,msg):
    if not ok:raise RuntimeError("SOTA_LTD_CANONICAL_CROSSWALK_BLOCKED "+msg)
def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()

def evidence(y,dates,probs):
    n=len(y)
    order=np.argsort(-probs,axis=1,kind="stable")
    top=order[:,0]
    rank=np.argmax(order==y[:,None],axis=1)+1
    ds,indices=np.unique(dates,return_inverse=True)
    actual=np.zeros((len(ds),65),dtype=np.int64)
    hits=np.zeros((len(ds),65),dtype=np.int64)
    predicted=np.zeros((len(ds),65),dtype=np.int64)
    np.add.at(actual,(indices,y),1)
    np.add.at(predicted,(indices,top),1)
    correct=(top==y)
    np.add.at(hits,(indices[correct],y[correct]),1)
    aggregate={
      "actual":actual,"hits":hits,"predicted":predicted,
      "correct":np.bincount(indices,weights=correct.astype(np.int64),minlength=len(ds)),
      "top5":np.bincount(indices,weights=(rank<=5).astype(np.int64),minlength=len(ds)),
      "reciprocal":np.bincount(indices,weights=(1.0/rank),minlength=len(ds))}
    return ds,aggregate

def values(w,denom,agg):
    n=float(np.dot(w,denom))
    check(n>0,"bootstrap cannot generate empty cohort")
    ac=w@agg["actual"];tp=w@agg["hits"];pp=w@agg["predicted"]
    macro=float(np.divide(2*tp,ac+pp,out=np.zeros(65,dtype=np.float64),
                          where=(ac+pp)>0).mean())
    return [float((w@agg["correct"])/n),macro,
            float((w@agg["top5"])/n),float((w@agg["reciprocal"])/n)]

def validate_probs(name,probs,n):
    check(probs.shape==(n,65),name+" probability shape")
    check(np.isfinite(probs).all() and (probs>=0).all() and
          np.allclose(probs.sum(1),1.0,atol=2e-4,rtol=0),
          name+" invalid probability values")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--repo",type=Path,default=Path.home()/"ltd-attack")
    p.add_argument("--base",type=Path,default=Path("/data/datasets_brayan/ltd-sota-2026"))
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--bootstrap",type=int,default=2500)
    args=p.parse_args()
    check(args.bootstrap>=100,"at least 100 bootstraps")
    sys.path.insert(0,str(args.repo))
    from scripts.sota.sota_paths_v1 import S5
    check(S5.is_file(),"S5 crosswalk manifest missing")
    s5=json.loads(S5.read_text())
    check(s5.get("classification")=="SOTA_S5_FROZEN_REFITS_AND_LABEL_CROSSWALK_V1",
          "incorrect S5 freeze")
    check(s5.get("training_performed") is False and
          s5.get("evaluation_performed") is False,
          "S5 evidence not a frozen label manifest")
    root=args.base/"artifacts/sota"
    results={"classification":"LTD_SEVEN_SOTA_REUSE_CANONICAL_S5_S6_ALIGNMENT_V1",
             "S5_sha256":digest(S5),"bootstrap_replicates":args.bootstrap,
             "seed":20261113,"cohorts":{},
             "source_code_lineage":{
               "crosswalk":"scripts/sota/s5_freeze_and_label_alignment_v1.py",
               "canonical_comparison":"scripts/sota/s6_compare_frozen_sota_ltd_v1.py",
               "Future_B_identity_reconstruction":"src/analysis/future_rank_gap_battery.py",
               "Future_B_Micro_Macro_join":"src/models/final/10B_evaluate_future_concept_drift.py",
             },
             "limits":[
               "Uses S5 canonical freeze and prior S5/10B verified UID order provenance; it independently checks dates, classes and truth, not NPZ object-UID payload equality.",
               "This compares our frozen models to adapted author-source baselines, not published numerical results.",
               "Swallow is collapsed, so it is tagged diagnostic and not a valid representative of original Swallow.",
               "The historical 65-site cohort may have been conditioned by availability in Future-B; this is a retrospective evaluation, not a new blind holdout.",
               "Only 11 dates in 09B and 16 dates in Future-B; paired date-bootstrap uncertainty is conditional and not prospective confirmation."]}
    for ci,(cohort,(n,ndays)) in enumerate(COHORTS.items()):
        cross=s5["label_alignment"][cohort]
        original=Path(cross["source"])
        check(original.is_file() and digest(original)==cross["source_sha256"],
              cohort+" original frozen canonical score SHA mismatch")
        with np.load(original,allow_pickle=False) as z:
            mandatory={"query_date","y_true","candidate_labels",*LTD_FIELDS.values()}
            check(mandatory<=set(z.files),cohort+" original LTD fields missing")
            dates=z["query_date"].astype("datetime64[D]").astype(str)
            y=z["y_true"].astype(np.int64)
            label=z["candidate_labels"].astype(str)
            probs={name:np.asarray(z[field],dtype=np.float32)
                   for name,field in LTD_FIELDS.items()}
            # Crucially, DO NOT read z["pcap_uid"] when dtype=object:
            # previous P14A safe UID recovery and S5 are the provenance.
        check(len(dates)==len(y)==n and len(np.unique(dates))==ndays,
              cohort+" date/count contract mismatch")
        check(len(label)==65 and len(set(label))==65 and
              (y>=0).all() and (y<65).all(),
              cohort+" invalid frozen original labels")
        for name,folder in SOTA_FOLDERS.items():
            score=root/folder/"evaluation"/cohort/"scores_sota_aligned.npz"
            manifest=root/folder/"evaluation"/cohort/"manifest.json"
            check(score.is_file() and manifest.is_file(),name+" missing result")
            meta=json.loads(manifest.read_text())
            check(digest(score)==meta["score_sha256"],name+" score SHA mismatch")
            with np.load(score,allow_pickle=False) as z:
                uid=z["pcap_uid"].astype(str)
                d=z["query_date"].astype("datetime64[D]").astype(str)
                yt=z["y_true"].astype(np.int64)
                lbl=z["candidate_labels"].astype(str)
                v=z["sota_probs"].astype(np.float32)
            check(len(uid)==n and len(set(uid))==n,name+" score UIDs invalid")
            check(np.array_equal(d,dates) and np.array_equal(yt,y)
                  and np.array_equal(lbl,label),
                  name+" is NOT row-aligned to canonical LTD original")
            probs[name]=v
        for name,v in probs.items():validate_probs(name,v,n)
        all_days,indices=np.unique(dates,return_inverse=True)
        count=np.bincount(indices,minlength=len(all_days)).astype(np.float64)
        weights=np.ones(len(all_days),dtype=np.float64)
        rng=np.random.default_rng(20261113+ci)
        samplings=rng.multinomial(len(all_days),
                   [1/len(all_days)]*len(all_days),
                   size=args.bootstrap).astype(np.float64)
        summaries={};boot={}
        for name,v in probs.items():
            ds,agg=evidence(y,dates,v)
            check(np.array_equal(ds,all_days),name+" day order mismatch")
            score=values(weights,count,agg)
            samples=np.asarray([values(w,count,agg) for w in samplings],
                               dtype=np.float64)
            summaries[name]={"metrics":dict(zip(METRICS,score)),
               "date_bootstrap_ci95":{m:[float(x) for x in
                  np.quantile(samples[:,k],[.025,.975])]
                   for k,m in enumerate(METRICS)}}
            boot[name]=samples
            print("CANONICAL_LTD_SOTA_PARITY_PASS",cohort,name,
                  "acc",round(score[0],5),flush=True)
        comparisons=[]
        for own in LTD_FIELDS:
            for base in SOTA_FOLDERS:
                delta=np.asarray(summaries[own]["metrics"][m]-
                    summaries[base]["metrics"][m] for m in METRICS,dtype=np.float64)
                samples=boot[own]-boot[base]
                comparisons.append({
                   "own":own,"sota":base,
                   "differences_own_minus_SOTA":dict(zip(METRICS,delta.tolist())),
                   "day_bootstrap_ci95":{m:[float(x) for x in
                         np.quantile(samples[:,k],[.025,.975])]
                         for k,m in enumerate(METRICS)},
                   "interpretation":("DIAGNOSTIC_COLLAPSED_BASELINE" if
                         base=="SOTA_Swallow_COLLAPSED" else "FROZEN_ADAPTED_BASELINE")})
        results["cohorts"][cohort]={
           "status":"PASS_CANONICAL_S5_S6_ROW_AND_CLASS_ALIGNMENT",
           "n":n,"days":ndays,
           "original_source_sha256":cross["source_sha256"],
           "label_identity_order":cross["identity_order"],
           "models":summaries,"paired_comparisons":comparisons}
    check(not args.output.exists(),"refuse overwrite audit report")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(results,ensure_ascii=False,indent=2)+"\n")
    print("SOTA_LTD_CANONICAL_PAIRWISE_REPORT",args.output,flush=True)
    print("COMPARISONS_PER_COHORT",len(comparisons),flush=True)
if __name__=="__main__":main()
