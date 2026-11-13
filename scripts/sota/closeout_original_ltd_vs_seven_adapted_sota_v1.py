#!/usr/bin/env python3
"""One-pass scientific closeout of already verified paired LTD/SOTA results.

Reads the canonical S5/S6 10-model comparison JSON; checks model, cohort,
metric, CI and paired-difference algebra; emits compact paper-ready CSV/JSON
without rewriting or selecting checkpoints or using future data for training.

Swallow COLLAPSED is excluded from valid benchmark ranking and marked as
failed adaptation. All comparisons retain posthoc / paper-fidelity caveats.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math
from pathlib import Path

METRICS=("accuracy","macro_f1","top5","mrr")
OURS=("LTD_MICRO","LTD_MACRO","LTD_HYBRID")
SOTA=("SOTA_DF","SOTA_VarCNN_direction","SOTA_TikTok","SOTA_NetCLR",
      "SOTA_RoFiRe","SOTA_WFCAT")
COLLAPSED="SOTA_Swallow_COLLAPSED"
COHORTS={"09B":(14687,11),"10B_DEV_FROZEN":(18543,16)}
def require(condition,message):
    if not condition:raise RuntimeError("SOTA_LTD_CLOSEOUT_BLOCKED "+message)
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
    return h.hexdigest()
def csv_write(path,rows,fields):
    with path.open("x",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader();w.writerows(rows)
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--comparison",required=True,type=Path)
    p.add_argument("--output-dir",required=True,type=Path)
    a=p.parse_args()
    require(a.comparison.is_file(),"comparison JSON missing")
    source=json.loads(a.comparison.read_text())
    require(source.get("classification")==
            "LTD_SEVEN_SOTA_REUSE_CANONICAL_S5_S6_ALIGNMENT_V1",
            "wrong canonical comparison")
    require(source.get("bootstrap_replicates")==2500,
            "paired day bootstrap protocol changed")
    require(set(source["cohorts"])==set(COHORTS),"cohorts differ")
    rankings=[];pairs=[];summaries={}
    for cohort,(n,days) in COHORTS.items():
        entry=source["cohorts"][cohort]
        require(entry["n"]==n and entry["days"]==days,"population mismatch "+cohort)
        models=entry["models"]
        require(set(models)==set(OURS)|set(SOTA)|{COLLAPSED},
                "missing/wrong benchmark methods")
        require(len(entry["paired_comparisons"])==21,
                "expected 3 LTD × 7 SOTA paired comparisons")
        pair_index={}
        for row in entry["paired_comparisons"]:
            k=(row["own"],row["sota"])
            require(k not in pair_index,"duplicate pair")
            pair_index[k]=row
        require(set(pair_index)=={(o,s) for o in OURS for s in (*SOTA,COLLAPSED)},
                "incorrect comparison grid")
        for name,info in models.items():
            for m in METRICS:
                value=float(info["metrics"][m])
                ci=[float(x) for x in info["date_bootstrap_ci95"][m]]
                require(math.isfinite(value) and 0<=value<=1,
                        name+" "+m+" invalid metric")
                require(len(ci)==2 and 0<=ci[0]<=ci[1]<=1,
                        name+" "+m+" invalid CI")
                rankings.append({"cohort":cohort,"model":name,"n":n,
                                 "n_capture_dates":days,"metric":m,
                                 "estimate":value,"ci95_low":ci[0],
                                 "ci95_high":ci[1],
                                 "status":"DIAGNOSTIC_COLLAPSED" if name==COLLAPSED
                                          else "ADAPTED_FROZEN_COMPARISON"})
        for (own,sota),rec in pair_index.items():
            for m in METRICS:
                d=float(rec["differences_own_minus_SOTA"][m])
                ci=[float(x) for x in rec["day_bootstrap_ci95"][m]]
                expect=float(models[own]["metrics"][m])-float(models[sota]["metrics"][m])
                require(abs(d-expect)<1e-12,"incorrect paired delta "+cohort+"/"+own+"/"+sota+"/"+m)
                require(-1<=ci[0]<=ci[1]<=1,"invalid paired CI")
                pairs.append({
                  "cohort":cohort,"own_model":own,"sota_model":sota,
                  "n":n,"n_capture_dates":days,"metric":m,"delta_own_minus_sota":d,
                  "ci95_low":ci[0],"ci95_high":ci[1],
                  "ci_excludes_zero":ci[0]>0 or ci[1]<0,
                  "direction":("own_higher" if ci[0]>0 else
                               "sota_higher" if ci[1]<0 else "inconclusive"),
                  "comparison_status":("DIAGNOSTIC_UNUSABLE_SWALLOW" if sota==COLLAPSED
                      else "VALID_PAIRED_POSTHOC_ADAPTED_MODEL_COMPARISON")})
        hybrid=float(models["LTD_HYBRID"]["metrics"]["accuracy"])
        eligible={k:float(models[k]["metrics"]["accuracy"]) for k in SOTA}
        best=max(eligible,key=eligible.get)
        primary=pair_index[("LTD_HYBRID",best)]
        summaries[cohort]={
            "n":n,"days":days,
            "hybrid_accuracy":hybrid,
            "highest_eligible_sota_accuracy_model":best,
            "highest_eligible_sota_accuracy":eligible[best],
            "hybrid_minus_best_sota_accuracy":hybrid-eligible[best],
            "paired_day_bootstrap_accuracy_ci95":
                primary["day_bootstrap_ci95"]["accuracy"],
            "note":"Adapted model benchmark, not original-paper exact fidelity."}
    require(not a.output_dir.exists(),"destination exists: no overwrite")
    a.output_dir.mkdir(parents=True)
    csv_write(a.output_dir/"metrics_models.csv",rankings,list(rankings[0]))
    csv_write(a.output_dir/"paired_3x7x4.csv",pairs,list(pairs[0]))
    report={"classification":"LTD_SOTA_S5_S6_PAIRED_CLOSEOUT_V1",
            "source_comparison_sha256":sha(a.comparison),
            "source":str(a.comparison),
            "bootstrap_by_date":2500,"summary":summaries,
            "excluded_as_failed_method_adaptation":[COLLAPSED],
            "limitations":[
                "Paired observational comparison on frozen 09B/Future-B, not a prospective blind benchmark.",
                "The 65 sites may historically have been selected conditional on Future-B availability.",
                "Model algorithms use author-source adaptations; original-paper fidelity is not certified.",
                "A nonzero bootstrap percentile CI is descriptive conditional on only 11/16 observed dates.",
                "LTD_HYBRID does not necessarily dominate all endpoints (notably Future-B Top-5).",
                "Swallow representations collapsed at BYOL 40; do not rank as representative original Swallow.",
                "Source document's row-level canonical alignment is assumed and checked internally, not reconstructed here."],
            "status":"PASS_POSTHOC_COMPARISON_NOT_AUTHOR_FIDELITY_CERTIFICATE"}
    (a.output_dir/"summary.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print("CLOSEOUT_PASS",report["status"])
    for cohort,v in summaries.items():
        print("COHORT",cohort,"HYBRID_ACC",v["hybrid_accuracy"],
              "BEST_SOTA",v["highest_eligible_sota_accuracy_model"],
              v["highest_eligible_sota_accuracy"],
              "DELTA",v["hybrid_minus_best_sota_accuracy"],
              "CI95",v["paired_day_bootstrap_accuracy_ci95"])
    print("PAIRWISE_CI_COUNT",len(pairs),"OUTPUT",a.output_dir)
if __name__=="__main__":main()
