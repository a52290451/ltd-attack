#!/usr/bin/env python3
"""Freeze final seven valid source-adapted baselines for LTD article, read-only.

Reuses previously certified six-baseline comparison, verified Swallow BNtrain v2
release and independent paired stats. Writes a standalone 11-model x two-cohort
table and 168 paired metric confidence intervals. No fit or score mutation.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.sota.compare_all_seven_with_original_ltd_s5_s6_v1 import evidence,values

METRICS=("accuracy","macro_f1","top5","mrr")
OURS=("LTD_MICRO","LTD_MACRO","LTD_HYBRID")
SIX=("SOTA_DF","SOTA_VarCNN_direction","SOTA_TikTok",
     "SOTA_NetCLR","SOTA_RoFiRe","SOTA_WFCAT")
S7="SOTA_Swallow_v2_BNtrain"
def must(v,msg):
    if not v:raise RuntimeError("ARTICLE_SOTA_CLOSEOUT_BLOCKED "+msg)
def digest(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()
def j(p):return json.loads(p.read_text())
def main():
    ap=argparse.ArgumentParser()
    for k in ("base","original","six-matrix","release","swallow-paired","output-dir"):
        ap.add_argument("--"+k,required=True,type=Path)
    a=ap.parse_args()
    must(not a.output_dir.exists(),"output exists; refuse overwrite")
    old=j(a.original); matrix=j(a.six_matrix); rel=j(a.release); sw=j(a.swallow_paired)
    must(old["classification"]=="LTD_SEVEN_SOTA_REUSE_CANONICAL_S5_S6_ALIGNMENT_V1",
         "wrong original comparison")
    must(matrix["status"]=="SIX_AUTHOR_SOURCE_ADAPTED_BASELINES_USABLE_ONE_COLLAPSED_EXCLUDED",
         "six method scope not finalized")
    must(rel["status"]=="PASS_ADAPTED_SWALLOW_V2_NONCOLLAPSED_TECHNICAL_GATE_NOT_PAPER_EXACT",
         "Swallow v2 release not passed")
    must(sw["classification"]=="LTD_SWALLOW_V2_FROZEN_S5S6_PAIRED_COMPARISON_V1" and
         sw["release_report_sha256"]==digest(a.release),
         "Swallow paired evidence does not match release")
    must({x["model"] for x in matrix["models"]}==
         {"DF","VarCNN","TikTok","RoFiRe","NetCLR","WFCAT","Swallow"},
         "invalid method inventory")
    records=[];pairs=[];cohorts={}
    for cohort,n,ndays in (("09B",14687,11),("10B_DEV_FROZEN",18543,16)):
        prior=old["cohorts"][cohort];updated=sw["cohorts"][cohort]
        must(prior["n"]==updated["n"]==n and prior["days"]==updated["dates"]==ndays,
             "cohort counts differ")
        must(len(prior["models"])==10 and len(prior["paired_comparisons"])==21,
             "prior results incomplete")
        path=a.base/"artifacts/sota/s11_swallow_authors_BNtrain_v2_seed42/evaluation"/cohort/"scores_sota_aligned.npz"
        meta=j(path.parent/"manifest.json")
        must(meta["score_sha256"]==updated["new_sota_scores_sha256"]==digest(path),
             "v2 source score lineage fails")
        with np.load(path,allow_pickle=False) as z:
            probs=z["sota_probs"].astype(np.float32)
            y=z["y_true"].astype(np.int64)
            dates=z["query_date"].astype("datetime64[D]").astype(str)
            uid=z["pcap_uid"].astype(str)
            labels=z["candidate_labels"].astype(str)
        must(probs.shape==(n,65) and len(set(uid))==n and
             len(set(labels))==65 and len(np.unique(dates))==ndays,
             "Swallow v2 alignment/population failed")
        must(np.isfinite(probs).all() and (probs>=0).all() and
             np.allclose(probs.sum(axis=1),1,atol=2e-4,rtol=0),
             "Swallow v2 probability normalization failed")
        ds,agg=evidence(y,dates,probs)
        denominator=np.asarray([np.sum(dates==day) for day in ds],dtype=float)
        new=dict(zip(METRICS,values(np.ones(len(ds)),denominator,agg)))
        must(abs(new["accuracy"]-updated["accuracy"]["SWALLOW_V2"])<1e-10,
             "Swallow v2 accuracy differs from validated comparison")
        data={k:prior["models"][k]["metrics"] for k in OURS+SIX}
        data[S7]=new
        for key in OURS+SIX+(S7,):
            records.append({"cohort":cohort,"method":key,
                **{m:float(data[key][m]) for m in METRICS}})
        previous={(x["own"],x["sota"]):x for x in prior["paired_comparisons"]}
        for own in OURS:
            for sota in SIX+(S7,):
                if sota==S7:
                    ci=updated["paired_differences"][own]
                    actual={m:(ci[m]["delta"],ci[m]["paired_bootstrap_ci95"])
                            for m in METRICS}
                else:
                    item=previous[(own,sota)]
                    actual={m:(item["differences_own_minus_SOTA"][m],
                               item["day_bootstrap_ci95"][m]) for m in METRICS}
                for m in METRICS:
                    delta,interval=actual[m]
                    must(abs((data[own][m]-data[sota][m])-delta)<1e-8,
                         "paired delta mismatch "+cohort+" "+own+" "+sota+" "+m)
                    must(len(interval)==2 and interval[0]<=interval[1],
                         "invalid paired interval")
                    pairs.append({"cohort":cohort,"own":own,"baseline":sota,
                                  "metric":m,"delta":float(delta),
                                  "ci95_lower":float(interval[0]),
                                  "ci95_upper":float(interval[1])})
        best=max(SIX+(S7,),key=lambda k:data[k]["accuracy"])
        cohorts[cohort]={"n":n,"days":ndays,
            "LTD_hybrid_accuracy":data["LTD_HYBRID"]["accuracy"],
            "best_adapted_SOTA":best,
            "best_adapted_SOTA_accuracy":data[best]["accuracy"],
            "hybrid_highest_accuracy":data["LTD_HYBRID"]["accuracy"]>data[best]["accuracy"]}
    must(len(records)==22 and len(pairs)==168,
         "expected 11 methods x 2 and 3x7x4x2 comparisons")
    report={"classification":"LTD_FINAL_SEVEN_ADAPTED_SOTA_ARTICLE_V1",
        "status":"FINAL_RETROSPECTIVE_ADAPTED_METHODS_COMPARISON_PASS",
        "models_per_cohort":11,"adapted_baselines_per_cohort":7,
        "paired_metric_ci_count":168,"cohorts":cohorts,
        "inputs_sha256":{name:digest(p) for name,p in (
             ("original",a.original),("matrix",a.six_matrix),
             ("swallow_release",a.release),("swallow_paired",a.swallow_paired))},
        "exclusions":["Swallow v1 collapsed: exclude from method rankings"],
        "limitations":["Original-author full paper-exact equivalence not certified",
           "VarCNN direction-only, reduced RoFiRe DyWin, adapted NetCLR and Swallow hyperparameters",
           "Single seed for model training; 11/16 date clusters",
           "Retrospective monitored-site population may reflect future cohort availability",
           "WFCAT acquisition timestamp units not fully certified from collector provenance"]}
    a.output_dir.mkdir(parents=True)
    def write(name,rows):
        with (a.output_dir/name).open("x",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]))
            w.writeheader();w.writerows(rows)
    write("article_metrics_11models.csv",records)
    write("article_paired_168_ci95.csv",pairs)
    (a.output_dir/"article_certificate.json").write_text(json.dumps(report,indent=2)+"\n")
    lines=["# LTD vs seven valid adapted SOTA baselines",
           "All values refer to our **source-adapted implementations**, not exact published benchmark reproductions.",""]
    for cohort in ("09B","10B_DEV_FROZEN"):
        lines+=["## "+cohort,"",
          "| Model | Accuracy (%) | Macro-F1 (%) | Top-5 (%) | MRR (%) |",
          "|---|---:|---:|---:|---:|"]
        for r in records:
            if r["cohort"]!=cohort:continue
            lines.append("| "+r["method"]+" | "+
                         " | ".join(f"{100*r[m]:.2f}" for m in METRICS)+" |")
        lines+=[""]
    lines+=["## Interpretation","",
       "Hybrid accuracy is higher than all seven adapted SOTA candidates on both frozen cohorts." if
       all(x["hybrid_highest_accuracy"] for x in cohorts.values()) else
       "See paired metrics for cohort-specific model ranking.",
       "Swallow v1 collapsed and was replaced by a separate noncollapsed target-BNtrain v2.",
       "The original paper method equivalence, full VarCNN ensemble, RoFiRe augmentation and single-seed generalization remain explicit limitations.",
       "Paired date bootstrap intervals are in article_paired_168_ci95.csv."]
    (a.output_dir/"article_tables.md").write_text("\n".join(lines)+"\n")
    print("ARTICLE_SOTA_FINAL_PASS",report["status"])
    for c,x in cohorts.items():
        print("ARTICLE_COHORT",c,x)
    print("ARTICLE_ROWS",len(records),"PAIRED_CI95",len(pairs))
    print("ARTICLE_FILES",a.output_dir)
if __name__=="__main__":main()
