#!/usr/bin/env python3
"""Audit S9 author-original-source external evidence, print metrics and create a local frozen summary.

This tool refuses to report S9 accuracy unless BOTH frozen external manifests,
NPZ files, UID populations, SHA and 65-way probability matrices validate.
Never modifies original artifacts or reports external numbers from pretraining.
"""
from __future__ import annotations
import json,hashlib
from pathlib import Path
import numpy as np

BASE=Path("/data/ltd-attack/sota/s9_netclr_author_original_v1")
COHORTS={"09B":14687,"10B_DEV_FROZEN":18543}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as fp:
        for block in iter(lambda:fp.read(4*1024*1024),b""):h.update(block)
    return h.hexdigest()

def must(ok,msg):
    if not ok:raise RuntimeError("S9_REPORT_BLOCKED "+msg)

def main():
    collected={}
    c=BASE/"refit_C_v1/manifest.json"
    must(c.is_file(),"C manifest missing")
    cinfo=json.loads(c.read_text())
    must(cinfo.get("classification")=="S9_NETCLR_AUTHORS_HISTORICAL_C_FROZEN_V1",
         "C class mismatch")
    must(sha(cinfo["checkpoint"])==cinfo["checkpoint_sha256"],
         "C checkpoint altered")
    for cohort,n in COHORTS.items():
        folder=BASE/"evaluation"/cohort
        m=folder/"manifest.json"
        scores=folder/"scores_sota_aligned.npz"
        must(m.is_file() and scores.is_file(),"incomplete "+cohort)
        x=json.loads(m.read_text())
        must(x.get("classification")=="S9_NETCLR_AUTHORS_FROZEN_EXTERNAL_V1"
             and x.get("cohort")==cohort and x.get("n")==n,
             "manifest mismatch "+cohort)
        must(x.get("C_manifest_sha256")==sha(c),
             "model provenance mismatch "+cohort)
        must(x.get("score_sha256")==sha(scores),
             "score artifact checksum mismatch "+cohort)
        with np.load(scores,allow_pickle=False) as z:
            uid=z["pcap_uid"].astype(str)
            y=z["y_true"].astype(int)
            p=z["sota_probs"]
            labels=z["candidate_labels"].astype(str)
            dates=z["query_date"].astype(str)
        must(len(uid)==n and len(set(uid))==n and
             len(labels)==65 and len(set(labels))==65 and
             p.shape==(n,65) and len(y)==n and len(dates)==n,
             "UID/class/shape mismatch "+cohort)
        must(np.isfinite(p).all() and
             np.allclose(p.sum(axis=1),1.,atol=2e-4,rtol=0),
             "probability contract mismatch "+cohort)
        must(np.all((y>=0)&(y<65)),"labels invalid "+cohort)
        top=np.argmax(p,axis=1)
        order=np.argsort(-p,axis=1)
        rank=(order==y[:,None]).argmax(axis=1)+1
        tp=np.bincount(y[top==y],minlength=65)
        support=np.bincount(y,minlength=65)
        pred=np.bincount(top,minlength=65)
        macro_f1=float(np.divide(2*tp,support+pred,
                       out=np.zeros(65,dtype=np.float64),
                       where=(support+pred)>0).mean())
        recomputed={"accuracy":float((top==y).mean()),
                    "macro_f1":macro_f1,
                    "top5":float((rank<=5).mean()),
                    "mrr":float(np.mean(1.0/rank))}
        metrics=x["metrics"]
        for k,v in recomputed.items():
            must(k in metrics and abs(float(metrics[k])-v)<1e-8,
                 "reported "+k+" differs from artifact "+cohort)
        collected[cohort]={"n":n,"metrics":metrics,
                           "score_sha256":x["score_sha256"],
                           "checkpoint_sha256":x["checkpoint_sha256"]}
    report={"classification":"S9_NETCLR_AUTHORS_FROZEN_RESULTS_V1",
            "source":"locally recomputed frozen artifact validation",
            "model_status":"SPIN_UMass_authors_source_adapted_not_paper_exact",
            "pretrain_sha256":json.loads((BASE/"pretrain_B_v1/manifest.json").read_text())["checkpoint_sha256"],
            "C_checkpoint_sha256":cinfo["checkpoint_sha256"],
            "cohorts":collected}
    target=BASE/"S9_NETCLR_AUTHORS_FROZEN_RESULTS_V1.json"
    must(not target.exists(),"summary exists; no overwrite")
    target.write_text(json.dumps(report,indent=2)+"\n")
    print("S9_AUTHOR_FINAL_RESULTS_PASS",json.dumps(report,indent=2))
    print("S9_AUTHOR_RESULTS_SAVED",str(target))

if __name__=="__main__":main()
