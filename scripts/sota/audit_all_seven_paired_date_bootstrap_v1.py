#!/usr/bin/env python3
"""Read-only full 7-model UID/class audit and paired date-cluster bootstrap.

For each frozen cohort, recompute top-1 accuracy independently from NPZ,
verify identical ordered UID, y_true, candidate_labels and dates across all
seven models, and perform 2500 paired date-cluster bootstrap replicates.
No fitting, GPU, checkpoint load or modifications to model outputs.
"""
from __future__ import annotations
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
MODELS={
 "NetCLR":"s9_netclr_author_original_v1",
 "Swallow":"s11_swallow_authors_seed42",
 "WFCAT":"s12_wfcat_authors_iat_seed42",
 "DF":"authors_df_seed42",
 "VarCNN_direction":"authors_varcnn_seed42",
 "TikTok":"authors_tiktok_seed42",
 "RoFiRe":"authors_rofire_seed42",
}
COHORTS={"09B":14687,"10B_DEV_FROZEN":18543}
def need(ok,msg):
    if not ok:raise RuntimeError("SOTA_SEVEN_AUDIT_BLOCKED "+msg)
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--base",type=Path,default=Path("/data/datasets_brayan/ltd-sota-2026"))
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--bootstrap",type=int,default=2500)
    a=p.parse_args()
    need(a.bootstrap>=100,"bootstrap replicates too small")
    root=a.base/"artifacts/sota"
    report={"classification":"LTD_SOTA_ALL_SEVEN_PAIRED_DATE_AUDIT_V1",
            "bootstrap_replicates":a.bootstrap,
            "seed":42,"cohorts":{},"statistical_protocol":
             "paired cluster bootstrap resampling distinct capture dates with replacement, within-date capture multiplicities preserved; percentile CI at 95%",
            "limitations":[
              "Not independent verification of original author code fidelity.",
              "Bootstrap CI describes date clusters in these observed cohorts only.",
              "All seven models are adapted; VarCNN is direction branch only.",
              "Do not tune any model based on the frozen Future-B results."]}
    for cohort,expected_n in COHORTS.items():
        labels0=uid0=dates0=y0=None
        corr={}; raw_metrics={}
        for model,folder in MODELS.items():
            dest=root/folder/"evaluation"/cohort
            mp=dest/"manifest.json"
            sp=dest/"scores_sota_aligned.npz"
            need(mp.is_file() and sp.is_file(),f"{model} {cohort} output missing")
            m=json.loads(mp.read_text())
            need(sha(sp)==m.get("score_sha256"),f"{model} {cohort} score SHA differs")
            need(m.get("n")==expected_n,f"{model} {cohort} count mismatch")
            need(m.get("dataset_c_opened") is False,
                 f"{model} {cohort} Dataset C exclusion")
            with np.load(sp,allow_pickle=False) as z:
                uid=z["pcap_uid"].astype(str)
                dates=z["query_date"].astype("datetime64[D]").astype(str)
                y=z["y_true"].astype(np.int64)
                names=z["candidate_labels"].astype(str)
                prob=np.asarray(z["sota_probs"],dtype=np.float32)
            need(prob.shape==(expected_n,65),f"{model} {cohort} bad probability shape")
            need(len(uid)==expected_n and len(set(uid))==expected_n,
                 f"{model} {cohort} duplicate UID")
            need(len(names)==65 and len(set(names))==65,
                 f"{model} {cohort} class list")
            need(np.isfinite(prob).all() and np.allclose(prob.sum(1),1.,atol=2e-4,rtol=0),
                 f"{model} {cohort} invalid probabilities")
            if uid0 is None:
                uid0,dates0,y0,labels0=uid,dates,y,names
            else:
                need(np.array_equal(uid,uid0),f"{model} {cohort} UID ORDER MISMATCH")
                need(np.array_equal(dates,dates0),f"{model} {cohort} DATE MISMATCH")
                need(np.array_equal(y,y0),f"{model} {cohort} LABEL MISMATCH")
                need(np.array_equal(names,labels0),f"{model} {cohort} LABEL ORDER MISMATCH")
            correct=(prob.argmax(1)==y)
            acc=float(correct.mean())
            need(abs(acc-float(m["metrics"]["accuracy"]))<1e-6,
                 f"{model} {cohort} metric mismatch")
            corr[model]=correct
            raw_metrics[model]={
              "accuracy":acc,"macro_f1_reported":m["metrics"]["macro_f1"],
              "top5_reported":m["metrics"]["top5"],
              "mrr_reported":m["metrics"]["mrr"],
              "classes_predicted":int(np.unique(prob.argmax(1)).size),
            }
            del prob
        unique,grp=np.unique(dates0,return_inverse=True)
        need(len(unique)>=2,cohort+" insufficient date clusters")
        count=np.bincount(grp,minlength=len(unique))
        correct_by_date=np.stack([
          np.bincount(grp,weights=corr[model].astype(int),minlength=len(unique))
          for model in MODELS
        ])
        rng=np.random.default_rng(42)
        take=rng.integers(0,len(unique),size=(a.bootstrap,len(unique)))
        replicate_count=count[take].sum(axis=1)
        need(np.all(replicate_count>0),cohort+" bootstrap count zero")
        estimates=np.stack([
            correct_by_date[k][take].sum(axis=1)/replicate_count
            for k in range(len(MODELS))
        ])
        model_names=list(MODELS)
        for i,model in enumerate(model_names):
            raw_metrics[model]["accuracy_date_bootstrap_95pct_ci"]=[
               float(q) for q in np.quantile(estimates[i],[.025,.975])]
        reference="VarCNN_direction"
        ref_i=model_names.index(reference)
        comparisons={}
        for k,model in enumerate(model_names):
            if model==reference:continue
            delta=estimates[ref_i]-estimates[k]
            actual=raw_metrics[reference]["accuracy"]-raw_metrics[model]["accuracy"]
            comparisons[reference+"_minus_"+model]={
               "accuracy_difference":actual,
               "paired_date_bootstrap_95pct_ci":[float(x) for x in np.quantile(delta,[.025,.975])],
               "bootstrap_nonpositive_fraction":float(np.mean(delta<=0)),
            }
        report["cohorts"][cohort]={
          "status":"SEVEN_MODEL_UID_DATE_LABEL_AND_SCORE_SHA_PASS",
          "n":expected_n,"unique_capture_dates":int(len(unique)),
          "date_cluster_size_min":int(count.min()),
          "date_cluster_size_max":int(count.max()),
          "models":raw_metrics,"paired_comparisons":comparisons}
        print("SEVEN_MODELS_COHORT_PASS",cohort,
              "n",expected_n,"days",len(unique),flush=True)
    need(not a.output.exists(),"output exists; no overwrite "+str(a.output))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print("SOTA_SEVEN_DATE_BOOTSTRAP_PASS",a.output,flush=True)
    for cohort,info in report["cohorts"].items():
        print("COHORT",cohort)
        for name,metrics in info["models"].items():
            print(name,round(metrics["accuracy"],5),metrics["accuracy_date_bootstrap_95pct_ci"])
        for name,delta in info["paired_comparisons"].items():
            print("DELTA",name,round(delta["accuracy_difference"],5),
                  delta["paired_date_bootstrap_95pct_ci"])
if __name__=="__main__":main()
