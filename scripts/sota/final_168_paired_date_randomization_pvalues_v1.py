#!/usr/bin/env python3
"""Append 168 auditable paired-date randomization p-values to frozen LTD/SOTA CIs.

Uses exactly the article's 3 LTD + 7 adapted SOTA, including noncollapsed
Swallow BNtrain v2, on 09B (11 days) and Future-B (16 days).
- Original S5 canonical hash verification and no object UID deserialization.
- Each SOTA scores NPZ and manifest verified for SHA, n, dates, y, labels, row order.
- Signed date-block label swaps at WHOLE-day level under exchangeability null.
- Macro F1 is recomputed from date-level class confusion sufficient statistics
  after swaps; accuracy/top5/MRR use daily additive counts.
- 2500 Monte Carlo swaps per paired contrast, fixed seed; p=(b+1)/(B+1).
- Holm adjusted p across 168 tests and separately 14 primary Hybrid accuracy
  tests (posthoc family defined explicitly). Existing 95% paired cluster
  bootstrap CIs kept unchanged. NO training or tuning.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.sota.compare_all_seven_with_original_ltd_s5_s6_v1 import (
    evidence,values,SOTA_FOLDERS,LTD_FIELDS,COHORTS,digest as sha)
OURS=("LTD_MICRO","LTD_MACRO","LTD_HYBRID")
SIX=("SOTA_DF","SOTA_VarCNN_direction","SOTA_TikTok","SOTA_NetCLR",
     "SOTA_RoFiRe","SOTA_WFCAT")
SWALLOW="SOTA_Swallow_v2_BNtrain"
FIELDS=("accuracy","macro_f1","top5","mrr")
def must(cond,msg):
    if not cond:raise RuntimeError("SOTA_PVALUE_CLOSEOUT_BLOCKED "+msg)
def readj(p):return json.loads(p.read_text())
def holm(values):
    """Strong FWER controlled by Holm, regardless of dependence."""
    x=np.asarray(values,dtype=float)
    must(np.isfinite(x).all() and np.all((0<=x)&(x<=1)),"invalid raw p")
    order=np.argsort(x,kind="stable")
    result=np.empty(len(x))
    running=0.
    for i,ix in enumerate(order):
        running=max(running, min(1.,float((len(x)-i)*x[ix])))
        result[ix]=running
    return result
def valid_prob(x,n,name):
    must(x.shape==(n,65) and np.isfinite(x).all() and
         np.all(x>=0) and np.allclose(x.sum(1),1.,atol=2e-4,rtol=0),
         "bad probability array "+name)
def daily_stat(y,date,probs):
    ds,agg=evidence(y,date,probs)
    return ds,agg
def randomization(a,b,den,draws):
    """Simulate paired whole-day swaps, including nonlinear macro F1."""
    D=len(den);S=draws.shape[0]
    must(draws.shape[1]==D,"wrong date draw dimensions")
    def swapped(w):
        def total(field):
            av=a[field];bv=b[field]
            selector=w.reshape((S,D)+(1,)*(av.ndim-1))
            mix_a=np.where(selector,av,bv)
            mix_b=np.where(selector,bv,av)
            return mix_a.sum(axis=1),mix_b.sum(axis=1)
        # Avoid allocating (2500, dates, 65) arrays per metric.
        # Daily mixed class counts can be calculated by a matrix product.
        flip=w.astype(np.float64)
        total0=float(den.sum())
        ac=a["actual"].sum(axis=0)
        assert np.array_equal(a["actual"],b["actual"])
        def mass(field):
            A=a[field].astype(float); B=b[field].astype(float)
            s=B.sum(axis=0)+flip@(A-B)
            r=A.sum(axis=0)+flip@(B-A)
            return s,r
        at,bt=mass("hits")
        ap,bp=mass("predicted")
        support=ac[None,:]
        ma=np.divide(2*at,support+ap,out=np.zeros_like(at),
                     where=(support+ap)>0).mean(axis=1)
        mb=np.divide(2*bt,support+bp,out=np.zeros_like(bt),
                     where=(support+bp)>0).mean(axis=1)
        ca,cb=mass("correct")
        ta,tb=mass("top5")
        ra,rb=mass("reciprocal")
        return np.stack(((ca-cb)/total0,ma-mb,(ta-tb)/total0,
                         (ra-rb)/total0),axis=1)
    return swapped(draws)
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--base",type=Path,required=True)
    p.add_argument("--final-dir",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--permutations",type=int,default=2500)
    p.add_argument("--seed",type=int,default=20261113)
    a=p.parse_args()
    must(not a.output_dir.exists(),"output directory already exists")
    must(a.permutations>=1000,"at least 1000 permutations required")
    existing=a.final_dir/"article_paired_168_ci95.csv"
    final_cert=a.final_dir/"article_certificate.json"
    must(existing.is_file() and final_cert.is_file(),"final article evidence missing")
    prior=readj(final_cert)
    must(prior["status"]=="FINAL_RETROSPECTIVE_ADAPTED_METHODS_COMPARISON_PASS" and
         prior["models_per_cohort"]==10 and prior["paired_metric_ci_count"]==168,
         "incorrect frozen article certificate")
    with existing.open(newline="") as f:
        original_rows=list(csv.DictReader(f))
    must(len(original_rows)==168,"prior 168 intervals not present")
    indexed={}
    for row in original_rows:
        k=(row["cohort"],row["own"],row["baseline"],row["metric"])
        must(k not in indexed,"duplicate prior comparison "+str(k))
        indexed[k]=row
    from scripts.sota.sota_paths_v1 import S5
    s5=readj(S5)
    must(s5["classification"]=="SOTA_S5_FROZEN_REFITS_AND_LABEL_CROSSWALK_V1",
         "S5 not canonical")
    output=[]
    source_shas={}
    for ci,(cohort,(n,days)) in enumerate(COHORTS.items()):
        canonical=Path(s5["label_alignment"][cohort]["source"])
        must(canonical.is_file() and sha(canonical)==
             s5["label_alignment"][cohort]["source_sha256"],
             "canonical LTD source hash changed "+cohort)
        with np.load(canonical,allow_pickle=False) as z:
            y=z["y_true"].astype(np.int64)
            dates=z["query_date"].astype("datetime64[D]").astype(str)
            lbl=z["candidate_labels"].astype(str)
            probs={name:z[field].astype(np.float32)
                   for name,field in LTD_FIELDS.items()}
        must(len(y)==n and len(lbl)==65 and len(np.unique(dates))==days,
             "canonical cohort size/classes/dates changed")
        source_shas[cohort]={"LTD":sha(canonical)}
        for model in SIX+(SWALLOW,):
            folder=("s11_swallow_authors_BNtrain_v2_seed42" if model==SWALLOW
                    else SOTA_FOLDERS[model])
            score=a.base/"artifacts/sota"/folder/"evaluation"/cohort/"scores_sota_aligned.npz"
            manifest=score.parent/"manifest.json"
            must(score.is_file() and manifest.is_file(),"missing frozen "+model)
            meta=readj(manifest)
            actual_hash=sha(score)
            must(actual_hash==meta["score_sha256"],"score hash mismatch "+model)
            with np.load(score,allow_pickle=False) as z:
                yp=z["y_true"].astype(np.int64)
                dp=z["query_date"].astype("datetime64[D]").astype(str)
                labels=z["candidate_labels"].astype(str)
                u=z["pcap_uid"].astype(str)
                arr=z["sota_probs"].astype(np.float32)
            must(len(u)==n and len(set(u))==n and
                 np.array_equal(yp,y) and np.array_equal(dp,dates)
                 and np.array_equal(labels,lbl),"canonical alignment "+model)
            valid_prob(arr,n,model);probs[model]=arr
            source_shas[cohort][model]=actual_hash
        all_days=np.unique(dates)
        per_day=np.bincount(np.searchsorted(all_days,dates),
                            minlength=days).astype(float)
        agg={}
        for name,pred in probs.items():
            valid_prob(pred,n,name)
            ds,dat=daily_stat(y,dates,pred)
            must(np.array_equal(ds,all_days),"date aggregate mismatch "+name)
            agg[name]=dat
        rng=np.random.default_rng(a.seed+ci)
        # Each candidate gets independent reproducible swaps; every metric
        # and comparison is evaluated with the same unbiased sign-flip null.
        for own in OURS:
            for base in SIX+(SWALLOW,):
                observed=np.asarray(values(np.ones(days),per_day,agg[own]))-np.asarray(
                    values(np.ones(days),per_day,agg[base]))
                draws=rng.integers(0,2,size=(a.permutations,days),
                                  dtype=np.int8).astype(bool)
                null=randomization(agg[own],agg[base],per_day,draws)
                extreme=(np.abs(null)>=np.abs(observed)[None,:]-1e-12).sum(axis=0)
                pv=(extreme+1)/(a.permutations+1)
                for j,m in enumerate(FIELDS):
                    key=(cohort,own,base,m)
                    must(key in indexed,"prior comparison missing "+str(key))
                    old=indexed[key]
                    must(abs(float(old["delta"])-observed[j])<1e-8,
                         "randomization observed delta differs from frozen paired CI "+str(key))
                    output.append({
                      "cohort":cohort,"own":own,"baseline":base,"metric":m,
                      "delta":float(observed[j]),
                      "ci95_lower":float(old["ci95_lower"]),
                      "ci95_upper":float(old["ci95_upper"]),
                      "p_value_two_sided":float(pv[j]),
                      "permutation_extreme_count":int(extreme[j]),
                      "permutations":a.permutations,
                      "p_method":"two-sided date-cluster paired sign swap",
                      "ci_method":"frozen paired date-cluster bootstrap 95 percent"})
                print("PVALUE_PAIR",cohort,own,base,
                      "ACC_p",float(pv[0]),"F1_p",float(pv[1]),
                      "TOP5_p",float(pv[2]),"MRR_p",float(pv[3]),flush=True)
    must(len(output)==168 and set(indexed)=={
          (x["cohort"],x["own"],x["baseline"],x["metric"]) for x in output},
          "168 comparison inventory mismatch")
    p_raw=[x["p_value_two_sided"] for x in output]
    correction=holm(p_raw)
    for row,adj in zip(output,correction):
        row["p_holm_168"]=float(adj)
        row["raw_p_below_0_05"]=bool(row["p_value_two_sided"]<.05)
        row["holm_168_below_0_05"]=bool(adj<.05)
    primary=[i for i,row in enumerate(output)
             if row["own"]=="LTD_HYBRID" and row["metric"]=="accuracy"]
    must(len(primary)==14,"14 pre-specified primary comparisons missing")
    primary_correct=holm([output[i]["p_value_two_sided"] for i in primary])
    for idx,val in zip(primary,primary_correct):
        output[idx]["p_holm_primary_14"]=float(val)
        output[idx]["holm_primary_14_below_0_05"]=bool(val<.05)
    sig={cohort:{"hybrid_accuracy_raw_positive_significant":sum(
           r["metric"]=="accuracy" and r["own"]=="LTD_HYBRID" and
           r["cohort"]==cohort and r["delta"]>0 and r["raw_p_below_0_05"]
           for r in output),
        "hybrid_accuracy_primary_holm_significant":sum(
           r["metric"]=="accuracy" and r["own"]=="LTD_HYBRID" and
           r["cohort"]==cohort and r["delta"]>0 and
           r.get("holm_primary_14_below_0_05",False) for r in output)}
         for cohort in COHORTS}
    summary={"classification":"LTD_SOTA_FINAL_168_PAIRED_DATE_PVALUES_V1",
             "status":"PVALUES_COMPUTED_AND_LINKED_TO_FROZEN_168_CIS",
             "n_metric_tests":len(output),"primary_hybrid_accuracy_tests":14,
             "permutations_per_test":a.permutations,"seed":a.seed,
             "p_resolution":1/(a.permutations+1),
             "test":"two-sided paired cluster-by-date complete predictor-output swaps; recomputes nonlinear macro-F1",
             "null_assumptions":"exchangeability of two methods' per-day predictor outcomes conditional on observed date clusters",
             "multiple_comparisons":{"family_168":"Holm FWER",
                    "primary_family_14":"LTD Hybrid versus 7 SOTA accuracy in both cohorts; Holm FWER",
                    "secondary_metrics":"exploratory unless declared a priori"},
             "confidence_intervals":"unchanged 95% paired date bootstrap percentile, 2500 draws",
             "source_sha256":{"final_cert":sha(final_cert),
                   "previous_168_ci_csv":sha(existing),
                   "canonical_S5":sha(S5),**source_shas},
             "summary":sig,
             "limitations":[
               "Date sign randomization assumes exchangeability by day; dates are not randomized model assignments.",
               "Monte Carlo p-values are limited to resolution 1/(B+1). Do not report p=0.",
               "Only 11 dates 09B and 16 dates Future-B, so effective cluster count is small.",
               "Adjusted p-values protect specified hypothesis families, not method equivalence.",
               "All results condition on retrospective 65-class cohort and source-adapted SOTA methods."]}
    a.output_dir.mkdir(parents=True)
    with (a.output_dir/"article_168_pvalues_and_ci95.csv").open("x",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(output[0].keys())+
            ["p_holm_primary_14","holm_primary_14_below_0_05"])
        # Fields already included for some primary rows: preserve unique order.
        writer.fieldnames=list(dict.fromkeys(writer.fieldnames))
        writer.writeheader()
        for row in output:writer.writerow(row)
    (a.output_dir/"statistical_certificate.json").write_text(
        json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
    with (a.output_dir/"hybrid_primary_14_tests.csv").open("x",newline="") as f:
        rows=[x for x in output if x["own"]=="LTD_HYBRID" and x["metric"]=="accuracy"]
        cols=("cohort","baseline","delta","ci95_lower","ci95_upper",
              "p_value_two_sided","p_holm_primary_14","holm_primary_14_below_0_05",
              "p_holm_168")
        writer=csv.DictWriter(f,fieldnames=cols,extrasaction="ignore")
        writer.writeheader();writer.writerows(rows)
    print("FINAL_PVALUE_GATE_PASS",summary["status"])
    for cohort,v in sig.items():print("HYBRID_PRIMARY",cohort,v)
    print("PVALUE_ROWS",len(output),"PRIMARY_ROWS",len(primary))
    print("STATISTICAL_CLOSURE",a.output_dir)
if __name__=="__main__":main()
