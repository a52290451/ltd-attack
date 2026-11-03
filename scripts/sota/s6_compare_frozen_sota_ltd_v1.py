#!/usr/bin/env python3
"""S6 paired historical LTD vs frozen SOTA, two cohorts, numpy-only.

Read-only original 09B / 10B DEV_FROZEN LTD NPZ + four S5 SOTA NPZ.
Exact UID/date/label equality and source/checkpoint hashes. Retrospective
analysis, no training, hyperparameter changes, or Dataset C.
"""
from __future__ import annotations

import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.models.sota.v1.s4_protocol import sha256_file

FREEZE=Path("/data/ltd-attack/sota/s5/frozen_s5_refits_and_labels_v1.json")
S5_DIR=Path("/data/ltd-attack/sota/s5/inference")
OUTPUT=Path("/data/ltd-attack/sota/s6/comparison_v1")
LTD={"MICRO":"micro_probs","MACRO":"macro_final_probs",
     "HYBRID":"hybrid_final_probs"}
SOTA=("DF","VarCNN")
METRICS=("accuracy","macro_f1","top5","mrr")
COHORTS={"09B":(14687,11),"10B_DEV_FROZEN":(18543,16)}
BOOT=2500
RANDOM_SWAPS=9999
SEED=20261103

def check(cond,msg):
    if not cond:raise RuntimeError("S6_BLOCKED "+msg)

def file_hash(path):
    check(path.is_file() and not path.is_symlink(),"file missing or symlinked "+str(path))
    return sha256_file(path)

def load(cohort,freeze):
    n,days=COHORTS[cohort]
    canonical=freeze["label_alignment"][cohort]
    path=Path(canonical["source"])
    h=file_hash(path)
    check(h==canonical["source_sha256"],"original score SHA drift "+cohort)
    with np.load(path,allow_pickle=True) as z:
        required={"pcap_uid","query_date","y_true","candidate_labels",*LTD.values()}
        check(required<=set(z.files),"missing original LTD probability arrays")
        uid=z["pcap_uid"].astype(str)
        dates=z["query_date"].astype("datetime64[D]").astype(str)
        y=z["y_true"].astype(np.int64)
        labels=z["candidate_labels"].astype(str)
        probs={name:z[key].astype(np.float64) for name,key in LTD.items()}
    check(len(uid)==n and len(set(uid))==n and len(np.unique(dates))==days,
          "original UID/date population mismatch "+cohort)
    check(len(labels)==65 and len(set(labels))==65,
          "canonical classes invalid "+cohort)
    check(np.all((y>=0)&(y<65)),"canonical y outside 65 classes")
    for model in SOTA:
        folder=S5_DIR/cohort/model
        manifest_path=folder/"manifest.json"
        manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
        pred_path=folder/"scores_sota_aligned.npz"
        check(file_hash(pred_path)==manifest["sota_score_sha256"],
              "SOTA output checksum mismatch "+cohort+" "+model)
        check(manifest["checkpoint_sha256"]==freeze["models"][model]["sha256"]
              and manifest["protocol_sha256"]==freeze["protocol_sha256"]
              and manifest["canonical_score_sha256"]==h
              and manifest["population"]==n
              and manifest["cohort"]==cohort and manifest["model"]==model
              and manifest["training_steps"]==0,
              "S5 provenance invalid "+cohort+" "+model)
        with np.load(pred_path,allow_pickle=False) as z:
            check(set(z.files)=={"pcap_uid","query_date","y_true",
                                "candidate_labels","sota_probs"},
                  "SOTA score schema mismatch")
            check(np.array_equal(z["pcap_uid"].astype(str),uid) and
                  np.array_equal(z["query_date"].astype(str),dates) and
                  np.array_equal(z["y_true"].astype(np.int64),y) and
                  np.array_equal(z["candidate_labels"].astype(str),labels),
                  "S5 predictions not paired by exact UID/date/label "+model)
            probs[model]=z["sota_probs"].astype(np.float64)
    for model,p in probs.items():
        check(p.shape==(n,65) and np.isfinite(p).all()
              and (p>=0).all() and
              np.allclose(p.sum(axis=1),1.0,atol=2e-4,rtol=0),
              "invalid full probabilities "+model)
    return {"n":n,"days":dates,"uids":uid,"y":y,"labels":labels,
            "probs":probs,"original_sha256":h}

def prep(data):
    y,days=data["y"],data["days"]
    d,inv=np.unique(days,return_inverse=True)
    sizes=np.bincount(inv,minlength=len(d)).astype(np.float64)
    sup=np.zeros((len(d),65),np.float64)
    np.add.at(sup,(inv,y),1.0)
    pred={}
    daily={}
    for name,probs in data["probs"].items():
        order=np.argsort(-probs,axis=1,kind="stable")
        top=order[:,0]
        rank=np.argmax(order==y[:,None],axis=1)+1
        tp=np.zeros((len(d),65),np.float64)
        pc=np.zeros((len(d),65),np.float64)
        correct=top==y
        np.add.at(tp,(inv[correct],y[correct]),1.0)
        np.add.at(pc,(inv,top),1.0)
        daily[name]={
            "tp":tp,"pc":pc,
            "acc":np.bincount(inv,weights=correct.astype(float),minlength=len(d)),
            "top5":np.bincount(inv,weights=(rank<=5).astype(float),minlength=len(d)),
            "rr":np.bincount(inv,weights=1.0/rank,minlength=len(d)),
        }
        pred[name]={"top":top,"rank":rank}
    return d,sizes,sup,daily,pred

def value(w,sizes,sup,stat):
    size=float(w@sizes)
    tp=w@stat["tp"]
    pc=w@stat["pc"]
    support=w@sup
    denom=pc+support
    f=np.divide(2*tp,denom,out=np.zeros(65),where=denom>0).mean()
    return np.array([float(w@stat["acc"]/size),float(f),
                     float(w@stat["top5"]/size),float(w@stat["rr"]/size)])

def block_sign_p(sizes,sup,left,right,metric_index,rng):
    """Exact complete day-swap enumeration for additive metrics; Monte Carlo
    whole-day swaps for nonadditive 65-class Macro-F1. Exploratory conditional.
    """
    days=len(sizes)
    observed=value(np.ones(days),sizes,sup,right)-value(
        np.ones(days),sizes,sup,left)
    if metric_index in (0,2,3):
        field={0:"acc",2:"top5",3:"rr"}[metric_index]
        delta=(right[field]-left[field])
        # All 2^11 or 2^16 day-label swaps; no capture-level independence.
        masks=np.arange(1<<days,dtype=np.uint32)
        bits=((masks[:,None] >> np.arange(days,dtype=np.uint32))&1).astype(float)
        vals=(delta.sum()-2*(bits@delta))/sizes.sum()
        p=float(np.mean(np.abs(vals)>=abs(observed[metric_index])-1e-12))
        return p,"exact_day_swap"
    # Nonlinear macro-F1 requires reaggregating per-class contributions
    flips=rng.integers(0,2,size=(RANDOM_SWAPS,days),dtype=np.int8)
    tp_l=left["tp"].sum(axis=0);tp_r=right["tp"].sum(axis=0)
    pc_l=left["pc"].sum(axis=0);pc_r=right["pc"].sum(axis=0)
    dt=right["tp"]-left["tp"];dp=right["pc"]-left["pc"]
    sup_total=sup.sum(axis=0)
    extremes=0
    for start in range(0,RANDOM_SWAPS,256):
        mask=flips[start:start+256].astype(float)
        tl=tp_l+mask@dt
        tr=tp_r-mask@dt
        pl=pc_l+mask@dp
        pr=pc_r-mask@dp
        fl=np.divide(2*tl,pl+sup_total,
                     out=np.zeros_like(tl),where=(pl+sup_total)>0).mean(axis=1)
        fr=np.divide(2*tr,pr+sup_total,
                     out=np.zeros_like(tr),where=(pr+sup_total)>0).mean(axis=1)
        extremes+=int(np.count_nonzero(np.abs(fr-fl)>=abs(observed[1])-1e-12))
    return float((extremes+1)/(RANDOM_SWAPS+1)),"mc_day_swap"

def holm(rows):
    # A family of exactly 2 baselines x 4 metrics per cohort, frozen here.
    order=sorted(range(len(rows)),key=lambda i:rows[i]["p_posthoc_two_sided"])
    previous=0.0
    for rank,i in enumerate(order):
        previous=max(previous,
                     min(1.0,rows[i]["p_posthoc_two_sided"]*(len(rows)-rank)))
        rows[i]["holm8_adjusted_p_posthoc"]=previous

def write_csv(path,rows):
    check(bool(rows),"empty report "+path.name)
    with path.open("x",newline="",encoding="utf-8") as out:
        writer=csv.DictWriter(out,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)

def run():
    check(not OUTPUT.exists() and not OUTPUT.is_symlink(),
          "S6 result directory exists; never overwrite frozen reports")
    freeze=json.loads(FREEZE.read_text(encoding="utf-8"))
    check(freeze["classification"]==
          "SOTA_S5_FROZEN_REFITS_AND_LABEL_CROSSWALK_V1",
          "invalid S5 label freeze")
    summary=[];comparisons=[];daily_rows=[];site_rows=[];sources=[]
    for ci,cohort in enumerate(COHORTS):
        data=load(cohort,freeze)
        dates,sizes,sup,daily,pred=prep(data)
        n=len(data["y"])
        weights=np.ones(len(dates))
        points={model:value(weights,sizes,sup,st)
                for model,st in daily.items()}
        rng=np.random.default_rng(SEED+ci)
        # identical day bootstrap resamples for EVERY comparator, including
        # nonlinear macro-F1, and 2500 rounds as frozen scientific protocol.
        bw=rng.multinomial(len(dates),[1/len(dates)]*len(dates),
                           size=BOOT).astype(float)
        bootstrap={name:np.vstack([value(w,sizes,sup,stat) for w in bw])
                   for name,stat in daily.items()}
        for model,v in points.items():
            summary.append({"cohort":cohort,"model":model,"n":n,
                            "days":len(dates),
                            **dict(zip(METRICS,map(float,v)))})
        for name,stat in daily.items():
            for dayi,day in enumerate(dates):
                w=np.zeros(len(dates));w[dayi]=1
                v=value(w,sizes,sup,stat)
                daily_rows.append({"cohort":cohort,"model":name,"date":day,
                                   "n":int(sizes[dayi]),
                                   **dict(zip(METRICS,map(float,v)))})
            y=data["y"]
            for i,label in enumerate(data["labels"]):
                mask=y==i
                cls_pred=pred[name]["top"]
                tp=int(np.count_nonzero(mask&(cls_pred==i)))
                count=int(mask.sum())
                pred_count=int(np.count_nonzero(cls_pred==i))
                site_rows.append({"cohort":cohort,"model":name,
                                  "site_label":label,"n":count,
                                  "recall":tp/count if count else 0.0,
                                  "class_f1":2*tp/(count+pred_count)
                                  if count+pred_count else 0.0})
        # The original LTD-Hybrid is always the paired reference.
        for candidate in SOTA:
            rows_family=[]
            other="HYBRID"
            point=points[candidate]-points[other]
            paired=bootstrap[candidate]-bootstrap[other]
            low,high=np.quantile(paired,[.025,.975],axis=0)
            ca=pred[candidate]["top"]==data["y"]
            cb=pred[other]["top"]==data["y"]
            counts={
                "sota_rescues_hybrid":int(np.count_nonzero(ca&~cb)),
                "hybrid_rescues_sota":int(np.count_nonzero(cb&~ca)),
                "both_correct":int(np.count_nonzero(ca&cb)),
                "both_wrong":int(np.count_nonzero(~ca&~cb))
            }
            check(sum(counts.values())==n,"paired disagreement accounting wrong")
            for k,metric in enumerate(METRICS):
                p,method=block_sign_p(sizes,sup,daily[other],
                                      daily[candidate],k,
                                      np.random.default_rng(SEED+ci*100+k+
                                                            (0 if candidate=="DF" else 50)))
                rows_family.append({
                    "cohort":cohort,"reference":"HYBRID","candidate":candidate,
                    "metric":metric,"n":n,"days":len(dates),
                    "hybrid":float(points[other][k]),
                    "sota":float(points[candidate][k]),
                    "delta_sota_minus_hybrid":float(point[k]),
                    "ci95_low":float(low[k]),"ci95_high":float(high[k]),
                    "p_posthoc_two_sided":p,"p_method":method,
                    "holm8_adjusted_p_posthoc":None,
                    **counts})
            comparisons.extend(rows_family)
        family=[r for r in comparisons if r["cohort"]==cohort]
        check(len(family)==8,"exact Holm8 comparison family expected")
        holm(family)
        sources.append({"cohort":cohort,"original_scores_sha256":
                        data["original_sha256"],
                        "sota_scores_sha256":{
                            m:json.loads((S5_DIR/cohort/m/"manifest.json").read_text())[
                                "sota_score_sha256"] for m in SOTA},
                        "n":n,"days":len(dates)})
        print("S6_COHORT_PAIRED",cohort,n,
              "DAYS",len(dates),flush=True)
        for m in ("HYBRID","DF","VarCNN","MICRO","MACRO"):
            print("S6_METRIC",cohort,m,
                  "ACCURACY",round(points[m][0],6),
                  "MACRO_F1",round(points[m][1],6),flush=True)
    # No output until both cohorts have passed verification and analysis.
    OUTPUT.mkdir(parents=True,mode=0o700)
    write_csv(OUTPUT/"00_model_metrics.csv",summary)
    write_csv(OUTPUT/"01_sota_vs_hybrid_paired.csv",comparisons)
    write_csv(OUTPUT/"02_daily_metrics.csv",daily_rows)
    write_csv(OUTPUT/"03_site_metrics.csv",site_rows)
    note=("# S6 — retrospective paired SOTA vs LTD-Hybrid\n\n"
          "09B and Future-B DEV_FROZEN are distinct cohorts. All five models"
          " use identical UIDs, dates, and 65 class labels within each cohort.\n\n"
          "Delta = SOTA − HYBRID. Day-paired 2500 bootstrap percentile CI95"
          " (macro F1 is recomputed on each sampled block)."
          " Conditional day-label exchange p: additive metrics exact full"
          " 2^D permutations, Macro-F1 Monte Carlo 9999 draws."
          " Holm8 within each cohort: DF + VarCNN × four metrics."
          " p values are exploratory and retrospective, NOT prospective"
          " confirmatory inference; consecutive dates may be correlated."
          " No external score was used to fit or tune a model.\n"
          " Future-B 65-site cohort may historically be B-conditioned."
          " Do not interpret these as new independent blind generalization.\n")
    (OUTPUT/"README.md").write_text(note,encoding="utf-8")
    outputs={p.name:file_hash(p) for p in OUTPUT.iterdir() if p.is_file()}
    manifest={
        "classification":"S6_PAIRED_SOTA_VS_HYBRID_POSTHOC_V1",
        "created_utc":datetime.now(timezone.utc).isoformat(),
        "cohorts":sources,
        "S5_freeze_sha256":file_hash(FREEZE),
        "checkpoints_sha256":{m:freeze["models"][m]["sha256"] for m in SOTA},
        "metrics":METRICS,
        "bootstrap_by_day":BOOT,"bootstrap_ci":"percentile_95",
        "macro_f1_nonadditive_recomputed":True,
        "macro_f1_day_swap_mc":RANDOM_SWAPS,
        "additive_exact_day_swap":True,
        "holm_per_cohort":"Holm8: 2 SOTA x 4 endpoints",
        "no_model_selection":True,"no_training":True,
        "dataset_c_opened":False,
        "results_sha256":outputs
    }
    (OUTPUT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print("S6_PAIRED_COMPARISON=PASS",OUTPUT,flush=True)

if __name__=="__main__":
    try:run()
    except Exception as e:
        print("S6_COMPARISON_ERROR",type(e).__name__,str(e)[:750],
              file=sys.stderr,flush=True)
        sys.exit(4)
