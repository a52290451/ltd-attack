#!/usr/bin/env python3
"""NetCLR S9 Historical DEV frozen refit-C and 09B/Future-B inference.

Reuses pinned WFlib NetCLR conv backbone initialized with verified Historical
DEV_EARLY+MIDDLE contrastive weights (no external data). B selects the number
of classifier fine-tuning epochs by DEV_LATE macro F1. C freshly initializes
classification head and refits classifier on all 57,916 Historical DEV; this
is a frozen historical-only training recipe, not paper-exact NetCLR.
External cohort data is NEVER used for training, validation or selection.
"""
from __future__ import annotations
import argparse, csv, json, random, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import importlib.util
SRC=ROOT/"scripts/sota/s9_netclr_authors_supervised_b_v1.py"
SPEC=importlib.util.spec_from_file_location("s9_supervised_pinned",SRC)
B=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(B)
P=B.P
OUT=B.OUT
TARGET=OUT/"refit_C_v1"
from scripts.sota.sota_paths_v1 import S5, STORAGE, FUTURE
COHORTS={"09B":14687,"10B_DEV_FROZEN":18543}

def must(ok,why):
    if not ok:raise RuntimeError("S9_C_FROZEN_BLOCKED "+why)

def b_evidence():
    p=B.TARGET/"manifest.json"
    must(p.is_file(),"supervised B manifest not yet available")
    m=json.loads(p.read_text())
    must(m.get("classification")=="S9_NETCLR_AUTHORS_SUPERVISED_B_V1"
         and m.get("config")==B.CONFIG,"B manifest provenance invalid")
    ep=m.get("selected_epoch")
    hist=m.get("history",[])
    must(type(ep)==int and 1<=ep<=30 and len(hist)==30,
         "B selection incomplete")
    must(hist[ep-1]==m.get("validation"),"B selected record mismatch")
    best=max(float(r["val_macro_f1"]) for r in hist)
    must(ep==min(r["epoch"] for r in hist if r["val_macro_f1"]==best),
         "B selected epoch not best macro-F1")
    ck=Path(m["checkpoint"])
    must(ck.is_file() and P.sha(ck)==m["checkpoint_sha256"],
         "B checkpoint hash mismatch")
    must(m.get("external_data_opened") is False and
         m.get("dataset_c_opened") is False,"B data contamination")
    return m

def seed():
    B.seed()

def train_c():
    evidence=b_evidence()
    must(not TARGET.exists(),"C output already exists; no overwrite")
    must(torch.cuda.is_available(),"GPU required for C")
    B.gpu_idle()
    seed()
    X,y,s,cache=B.prepare_cache()
    must(X.shape==(57916,5000) and len(y)==57916 and
         set(map(int,np.unique(y)))==set(range(65)),
         "C historical population or classes mismatch")
    must(dict(zip(*np.unique(s,return_counts=True)))==
         {0:19098,1:20160,2:18658},"C chronological splits mismatch")
    loader=DataLoader(B.Subset(X,y,np.arange(57916)),batch_size=64,
                      shuffle=True,drop_last=True,num_workers=0,
                      pin_memory=True)
    model=B.classifier().to("cuda:0")
    optim=torch.optim.Adam(model.parameters(),lr=0.0003)
    criterion=nn.CrossEntropyLoss()
    epochs=evidence["selected_epoch"]
    TARGET.mkdir(parents=True,mode=0o700)
    started=time.monotonic()
    history=[]
    print("S9_NETCLR_C_BEGIN","train",57916,"epochs",epochs,
          "pretrain_sha",P.sha(B.CHECKPOINT),flush=True)
    for epoch in range(1,epochs+1):
        start=time.monotonic()
        model.train()
        running=0.
        steps=0
        for x,yb in loader:
            x=x.to("cuda:0",non_blocking=True)
            yb=yb.to("cuda:0",non_blocking=True)
            optim.zero_grad(set_to_none=True)
            z=model(x)
            loss=criterion(z,yb)
            must(bool(torch.isfinite(loss).item()),"nonfinite C loss")
            loss.backward();optim.step()
            running+=float(loss.item())
            steps+=1
            if steps%100==0:
                print("S9_NETCLR_C_PROGRESS",json.dumps({
                    "epoch":epoch,"batch":steps,"batches":len(loader),
                    "elapsed_s":round(time.monotonic()-started,1)}),flush=True)
        rec={"epoch":epoch,"train_loss":running/steps,
             "epoch_elapsed_s":round(time.monotonic()-start,1),
             "total_elapsed_s":round(time.monotonic()-started,1)}
        history.append(rec)
        print("S9_NETCLR_C_EPOCH",json.dumps(rec),flush=True)
    ck=TARGET/"model.pt"
    torch.save({"model":model.cpu().state_dict(),"epoch":epochs,
                "config":B.CONFIG,
                "pretrain_sha256":P.sha(B.CHECKPOINT),
                "B_manifest_sha256":P.sha(B.TARGET/"manifest.json")},ck)
    m={"classification":"S9_NETCLR_AUTHORS_HISTORICAL_C_FROZEN_V1",
       "config":B.CONFIG,"refit_epochs":epochs,"training_n":57916,
       "checkpoint":str(ck),"checkpoint_sha256":P.sha(ck),
       "B_manifest_sha256":P.sha(B.TARGET/"manifest.json"),
       "pretrain_sha256":P.sha(B.CHECKPOINT),
       "cache_manifest_sha256":P.sha(B.CACHE/"manifest.json"),
       "historical_source_sha256":cache["source_sha256"],
       "S1_label_sha256":cache["label_sha256"],
       "history":history,"elapsed_seconds":time.monotonic()-started,
       "external_data_opened":False,"dataset_c_opened":False,
       "created_utc":datetime.now(timezone.utc).isoformat()}
    (TARGET/"manifest.json").write_text(json.dumps(m,indent=2)+"\n")
    print("S9_NETCLR_C_PASS",P.sha(ck),flush=True)

def evaluate(cohort):
    must(cohort in COHORTS,"unknown cohort")
    target=OUT/"evaluation"/cohort
    must(not target.exists(),"evaluation already exists; no overwrite")
    evidence=b_evidence()
    mp=TARGET/"manifest.json"
    must(mp.is_file(),"C must be completed before evaluation")
    c=json.loads(mp.read_text())
    must(c.get("classification")=="S9_NETCLR_AUTHORS_HISTORICAL_C_FROZEN_V1"
         and c.get("config")==B.CONFIG and
         c.get("refit_epochs")==evidence["selected_epoch"],
         "C manifest does not match frozen B")
    must(c.get("B_manifest_sha256")==P.sha(B.TARGET/"manifest.json"),
         "B was altered after C")
    ck=Path(c["checkpoint"])
    must(ck.is_file() and P.sha(ck)==c["checkpoint_sha256"],
         "C checkpoint SHA mismatch")
    frozen=json.loads(S5.read_text())
    ref=frozen["label_alignment"][cohort]
    must(ref.get("identity_order") is True,"noncanonical class permutation")
    source=Path(ref["source"])
    must(P.sha(source)==ref["source_sha256"],"frozen score/UID index changed")
    with np.load(source,allow_pickle=True) as arr:
        uid=arr["pcap_uid"].astype(str)
        dates=arr["query_date"].astype("datetime64[D]").astype(str)
        y=arr["y_true"].astype(np.int64)
        names=arr["candidate_labels"].astype(str)
    n=COHORTS[cohort]
    must(len(uid)==n and len(set(uid))==n and len(names)==65 and
         np.all((y>=0)&(y<65)),"external population or class mismatch")
    true_by_uid={str(k):str(names[int(yy)]) for k,yy in zip(uid,y)}
    vectors=(STORAGE/"historical/CLEAN_final_vectors_sites.csv"
             if cohort=="09B" else
             STORAGE/"future/CLEAN_final_vectors_sites_concept_drift.csv")
    if cohort=="09B":
        expected=c["historical_source_sha256"]
    else:
        fm=json.loads((FUTURE).read_text())
        expected=fm["future_inputs"]["micro"]["sha256"]
    must(P.sha(vectors)==expected,"external vectors hash mismatch")
    from src.models.sota.v1.data_adapter import parse_direction, fixed_length
    observed={}
    csv.field_size_limit(sys.maxsize)
    with vectors.open(newline="",encoding="utf-8") as fh:
        reader=csv.DictReader(fh)
        must({"pcap_uid","site_label","direction_vector"}<=set(reader.fieldnames or []),
             "missing external packet direction columns")
        for row in reader:
            key=row["pcap_uid"]
            if key not in true_by_uid:continue
            must(key not in observed,"external duplicate UID")
            must(row["site_label"]==true_by_uid[key],"UID/site class mismatch")
            d=parse_direction(row["direction_vector"])
            observed[key]=np.asarray(fixed_length(d),dtype=np.float32)
    must(len(observed)==n and set(observed)==set(uid),
         "external UID coverage mismatch")
    model=B.classifier()
    checkpoint=torch.load(ck,map_location="cpu",weights_only=True)
    must(checkpoint["config"]==B.CONFIG and checkpoint["epoch"]==
         evidence["selected_epoch"],"C model content mismatch")
    model.load_state_dict(checkpoint["model"],strict=True)
    # Use CPU if GPU is occupied, so external inference never interferes.
    device="cpu"
    model=model.to(device).eval()
    probs=np.empty((n,65),dtype=np.float32)
    with torch.inference_mode():
        for start in range(0,n,128):
            stop=min(start+128,n)
            x=np.stack([observed[k] for k in uid[start:stop]])
            logits=model(torch.from_numpy(x[:,None,:]).to(device))
            p=torch.softmax(logits,dim=1).cpu().numpy()
            must(np.isfinite(p).all(),"nonfinite probabilities")
            probs[start:stop]=p
            if (start//128)%20==0:
                print("S9_NETCLR_EVAL_PROGRESS",cohort,stop,n,flush=True)
    must(np.allclose(probs.sum(axis=1),1.,rtol=0,atol=2e-4),
         "probabilities do not sum to 1")
    order=np.argsort(-probs,axis=1)
    rank=(order==y[:,None]).argmax(axis=1)+1
    top=order[:,0]
    tp=np.bincount(y[top==y],minlength=65)
    pred=np.bincount(top,minlength=65)
    support=np.bincount(y,minlength=65)
    f1=np.divide(2*tp,support+pred,out=np.zeros(65,dtype=np.float64),
                 where=(support+pred)>0).mean()
    metrics={"accuracy":float(np.mean(top==y)),"macro_f1":float(f1),
             "top5":float(np.mean(rank<=5)),"mrr":float(np.mean(1./rank))}
    target.mkdir(parents=True,mode=0o700)
    artifact=target/"scores_sota_aligned.npz"
    np.savez_compressed(artifact,pcap_uid=uid,query_date=dates,y_true=y,
                        candidate_labels=names,sota_probs=probs)
    m={"classification":"S9_NETCLR_AUTHORS_FROZEN_EXTERNAL_V1",
       "cohort":cohort,"model":"NETCLR_SPIN_UMASS_AUTHORS_ADAPTED",
       "n":n,"metrics":metrics,
       "checkpoint_sha256":c["checkpoint_sha256"],
       "C_manifest_sha256":P.sha(mp),
       "canonical_score_sha256":ref["source_sha256"],
       "vector_sha256":expected,
       "score_sha256":P.sha(artifact),
       "no_external_tuning":True,"dataset_c_opened":False,
       "created_utc":datetime.now(timezone.utc).isoformat()}
    (target/"manifest.json").write_text(json.dumps(m,indent=2)+"\n")
    print("S9_NETCLR_EXTERNAL_PASS",cohort,json.dumps(metrics),flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--stage",choices=["C","evaluate"],required=True)
    ap.add_argument("--cohort",choices=list(COHORTS))
    args=ap.parse_args()
    if args.stage=="C":train_c()
    else:
        must(args.cohort is not None,"--cohort required")
        evaluate(args.cohort)
if __name__=="__main__":
    try:main()
    except Exception as e:
        print("S9_NETCLR_FROZEN_ERROR",type(e).__name__,str(e)[:1000],
              file=sys.stderr,flush=True)
        raise
