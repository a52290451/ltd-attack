#!/usr/bin/env python3
"""Final independent UID-aligned SOTA author-source score/metric integrity audit.

Reads (never fits or modifies) frozen score artifacts, checks exact 09B and
Future-B UID/label population, 65-class probability alignment and all four
metrics. Refuses partial or missing cohorts. No CUDA and no Dataset C.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.sota.sota_paths_v1 import ARTIFACT, S5
ROOT=ARTIFACT

N={"09B":14687,"10B_DEV_FROZEN":18543}
MODELS={
 "S3_DF_Authors":ROOT/"authors_df_seed42",
 "S4_VarCNN_direction_only_Authors":ROOT/"authors_varcnn_seed42",
 "S8_TikTok_Authors":ROOT/"authors_tiktok_seed42",
 "S9_NetCLR_Authors_Adapted":ROOT/"s9_netclr_author_original_v1",
 "S10_RoFiRe_Authors_Adapted":ROOT/"authors_rofire_seed42",
 "S11_Swallow_Authors_Adapted":ROOT/"s11_swallow_authors_seed42",
 "S12_WFCAT_Authors_Adapted":ROOT/"s12_wfcat_authors_iat_seed42",
}
def must(ok,why):
    if not ok:raise RuntimeError("SOTA_AUTHORS_RESULT_BLOCKED "+why)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b""):h.update(chunk)
    return h.hexdigest()
def metrics(p,y):
    rank=np.argsort(-p,axis=1)
    top=rank[:,0]
    pos=(rank==y[:,None]).argmax(axis=1)+1
    tp=np.bincount(y[top==y],minlength=65)
    sp=np.bincount(y,minlength=65)
    pred=np.bincount(top,minlength=65)
    macro=float(np.divide(2*tp,sp+pred,out=np.zeros(65),
                          where=sp+pred>0).mean())
    return {"accuracy":float((top==y).mean()),
            "macro_f1":macro,"top5":float((pos<=5).mean()),
            "mrr":float((1.0/pos).mean())}
def verify(name,base,ref,cohort):
    folder=base/"evaluation"/cohort
    manifest=folder/"manifest.json"
    score=folder/"scores_sota_aligned.npz"
    must(manifest.is_file() and score.is_file(),name+" missing "+cohort)
    m=json.loads(manifest.read_text())
    must(m.get("cohort")==cohort and int(m.get("n",-1))==N[cohort],
         name+" evidence contract "+cohort)
    must(m.get("dataset_c_opened") is False,name+" C source contamination")
    must(m.get("no_external_tuning") is True,name+" external adaptation flag")
    must(sha(score)==m.get("score_sha256"),name+" score hash mismatch")
    canon=Path(ref["source"])
    must(canon.is_file() and sha(canon)==ref["source_sha256"],
         "canonical cohort SHA invalid")
    with np.load(canon,allow_pickle=True) as f:
        canon_uid=f["pcap_uid"].astype(str)
        canon_y=f["y_true"].astype(np.int64)
        canon_labels=f["candidate_labels"].astype(str)
        canon_dates=f["query_date"].astype("datetime64[D]").astype(str)
    with np.load(score,allow_pickle=False) as f:
        uid=f["pcap_uid"].astype(str)
        y=f["y_true"].astype(np.int64)
        labels=f["candidate_labels"].astype(str)
        dates=f["query_date"].astype(str)
        p=f["sota_probs"]
    n=N[cohort]
    must(p.shape==(n,65) and len(uid)==n and len(set(uid))==n and
         len(labels)==65 and len(set(labels))==65,
         name+" UID/output shape")
    must(np.array_equal(uid,canon_uid) and
         np.array_equal(y,canon_y) and
         np.array_equal(labels,canon_labels) and
         np.array_equal(dates,canon_dates),
         name+" canonical alignment mismatch")
    must(np.all((y>=0)&(y<65)) and
         np.isfinite(p).all() and
         np.allclose(p.sum(axis=1),1.0,atol=2e-4,rtol=0),
         name+" invalid probability simplex")
    calc=metrics(p,y)
    reported=m.get("metrics",{})
    for k,v in calc.items():
        must(k in reported and abs(float(reported[k])-v)<1e-8,
             name+" metric mismatch "+k)
    return {"cohort":cohort,"n":n,"metrics":calc,
            "sha256":m["score_sha256"],"model_manifest_sha256":sha(manifest)}
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",choices=list(MODELS)+["all"],default="all")
    ap.add_argument("--save",action="store_true")
    opts=ap.parse_args()
    frozen=json.loads(S5.read_text())
    models=MODELS if opts.model=="all" else {opts.model:MODELS[opts.model]}
    result={"classification":"ORIGINAL_AUTHORS_SOTA_FROZEN_INTEGRITY_V1",
            "status":"all_requested_models_passed",
            "models":{},"dataset_c_used":False}
    for name,base in models.items():
        got={}
        for cohort in N:
            got[cohort]=verify(name,base,
                              frozen["label_alignment"][cohort],cohort)
        result["models"][name]=got
        print("SOTA_AUTHORS_RESULT_PASS",name,json.dumps(got),flush=True)
    if opts.save:
        folder=ROOT/"authors_source_frozen_results_v1"
        output=folder/("all_models.json" if opts.model=="all"
                       else opts.model+".json")
        must(not output.exists(),"result summary exists; refuse overwrite")
        folder.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(result,indent=2)+"\n")
        print("SOTA_AUTHORS_RESULT_SUMMARY_SAVED",str(output),flush=True)
    print("SOTA_AUTHORS_RESULT_FULL_INTEGRITY_PASS",len(models),flush=True)
if __name__=="__main__":main()
