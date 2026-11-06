#!/usr/bin/env python3
"""S8 TikTok frozen Historical DEV refit C and external inference (no evaluation tuning).

Use the WFlib code pinned by the S8 B script, best epoch selected in DEV_LATE.
Mode C: refit all 57,916 Historical DEV at fixed 23 epochs as selected by B;
mode evaluate: 09B and Future-B DEV_FROZEN, no optimizer or modifications.
Does not access Dataset C. Will not overwrite an existing scientific result.
"""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, random, sys, time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.models.sota.v1.data_adapter import parse_direction, digest
from src.models.sota.v1.timing_adapter import parse_times
from src.models.sota.v1.tiktok_adapter import tiktok_dt_from_sequences

BFILE=ROOT/"scripts/sota/s8_tiktok_train_b_v1.py"
SPEC=importlib.util.spec_from_file_location("ltd_sota_s8_b_frozen",BFILE)
B=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(B)
OUT=B.OUT
S5=Path("/data/ltd-attack/sota/s5/frozen_s5_refits_and_labels_v1.json")
COHORT_N={"09B":14687,"10B_DEV_FROZEN":18543}

def must(ok,reason):
    if not ok:raise RuntimeError("S8_FROZEN_BLOCKED "+reason)

def sha(path):return B.sha(path)

def b_evidence():
    p=OUT/"B"/"manifest.json"
    must(p.is_file(),"B validation manifest missing")
    m=json.loads(p.read_text())
    must(m.get("classification")=="S8_TIKTOK_WFLIB_HISTORICAL_B_V1","invalid B provenance")
    must(m.get("config")==B.CONFIG,"B config changed")
    epoch=m.get("selected_epoch")
    must(type(epoch)==int and 1<=epoch<=30,"invalid selected B epoch")
    hist=m.get("history",[])
    must(len(hist)==30 and hist[epoch-1]==m.get("validation"),"B selection evidence mismatch")
    must(epoch==min((r["epoch"] for r in hist if r["val_macro_f1"]==max(x["val_macro_f1"] for x in hist))),
         "B selection must maximize macro-F1, earliest tie")
    ck=Path(m["checkpoint"])
    must(ck.is_file() and sha(ck)==m["checkpoint_sha256"],"B checkpoint SHA mismatch")
    must(m.get("used_external_samples_for_fit") is False,"B external data prohibition")
    return m

def seed():
    torch.manual_seed(42);np.random.seed(42);random.seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True

def train_c():
    evidence=b_evidence()
    target=OUT/"C"
    must(not target.exists(),"refuse to overwrite C")
    must(torch.cuda.is_available(),"CUDA GPU required for S8 C")
    seed()
    X,y,s,s1,cache=B.load_data()
    must(X.shape==(57916,5000) and len(y)==57916 and len(s)==57916,"Historical shape mismatch")
    must(set(y)==set(range(65)),"class coverage mismatch")
    must(dict(zip(*np.unique(s,return_counts=True)))=={0:19098,1:20160,2:18658},
         "chronological partition mismatch")
    idx=np.arange(len(y))
    loader=DataLoader(B.NpyDataset(X,y,idx),batch_size=128,shuffle=True,
                      drop_last=True,num_workers=0,pin_memory=True)
    model=B.original_model().to("cuda:0")
    opt=torch.optim.Adamax(model.parameters(),lr=0.002)
    crit=nn.CrossEntropyLoss()
    epochs=evidence["selected_epoch"]
    target.mkdir(parents=True,mode=0o700)
    print("S8_TIKTOK_C_TRAIN_BEGIN",len(y),"epochs",epochs,flush=True)
    start=time.monotonic()
    losses=[]
    for ep in range(1,epochs+1):
        model.train()
        batch_losses=[]
        for x_batch,y_batch in loader:
            x_batch=x_batch.to("cuda:0",non_blocking=True)
            y_batch=y_batch.to("cuda:0",non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits=model(x_batch)
            loss=crit(logits,y_batch)
            must(bool(torch.isfinite(loss).item()),"nonfinite C loss")
            loss.backward();opt.step()
            batch_losses.append(float(loss.item()))
        record={"epoch":ep,"train_loss":float(np.mean(batch_losses))}
        losses.append(record)
        print("S8_TIKTOK_C_EPOCH",json.dumps(record),flush=True)
    ck=target/"model.pt"
    torch.save({"model":model.cpu().state_dict(),"epoch":epochs,
                "config":B.CONFIG,"b_checkpoint_sha256":evidence["checkpoint_sha256"]},ck)
    m={"classification":"S8_TIKTOK_HISTORICAL_C_FROZEN_V1","stage":"C",
       "config":B.CONFIG,"refit_epochs":epochs,"training_n":57916,
       "checkpoint":str(ck),"checkpoint_sha256":sha(ck),
       "B_manifest_sha256":sha(OUT/"B"/"manifest.json"),
       "B_checkpoint_sha256":evidence["checkpoint_sha256"],
       "cache_manifest_sha256":sha(OUT/"historical_cache_v1"/"manifest.json"),
       "historical_source_sha256":s1["historical_checksums"]["historical_vectors"]["sha256"],
       "S1_label_sha256":s1["label_order_sha256"],
       "loss_history":losses,"elapsed_seconds":time.monotonic()-start,
       "used_external_samples_for_fit":False,"dataset_c_opened":False,
       "created_utc":datetime.now(timezone.utc).isoformat()}
    (target/"manifest.json").write_text(json.dumps(m,indent=2)+"\n")
    print("S8_TIKTOK_C_PASS",ck,m["checkpoint_sha256"],flush=True)

def inference(cohort):
    must(cohort in COHORT_N,"unrecognized cohort")
    target=OUT/"evaluation"/cohort
    must(not target.exists(),"refuse to overwrite external evaluation")
    b=b_evidence()
    path=OUT/"C"/"manifest.json"
    must(path.is_file(),"C manifest missing")
    c=json.loads(path.read_text())
    must(c.get("stage")=="C" and c["config"]==B.CONFIG,
         "C config invalid")
    must(c["refit_epochs"]==b["selected_epoch"],"C epoch differs from B selection")
    must(c["B_manifest_sha256"]==sha(OUT/"B"/"manifest.json"),"B manifest changed")
    ck=Path(c["checkpoint"])
    must(ck.is_file() and sha(ck)==c["checkpoint_sha256"],"C checkpoint SHA changed")
    with S5.open() as f:frozen=json.load(f)
    ref=frozen["label_alignment"][cohort]
    must(ref.get("identity_order") is True,"canonical class permutation not identity")
    original=Path(ref["source"])
    must(digest(original)==ref["source_sha256"],"canonical NPZ SHA changed")
    with np.load(original,allow_pickle=True) as z:
        uid=z["pcap_uid"].astype(str)
        dates=z["query_date"].astype("datetime64[D]").astype(str)
        y=z["y_true"].astype(np.int64)
        names=z["candidate_labels"].astype(str)
    n=COHORT_N[cohort]
    must(len(uid)==n and len(set(uid))==n,"canonical UID population mismatch")
    must(len(names)==65 and np.all((y>=0)&(y<65)),"canonical class population mismatch")
    true_by_uid={str(key):str(names[int(yy)]) for key,yy in zip(uid,y)}
    source=(Path.home()/"ltd-storage/datasets/historical/CLEAN_final_vectors_sites.csv"
            if cohort=="09B" else
            Path.home()/"ltd-storage/datasets/future/CLEAN_final_vectors_sites_concept_drift.csv")
    if cohort=="09B":
        expected_sha=c["historical_source_sha256"]
    else:
        m=json.loads((ROOT/"docs/evidence/FINAL/FUTURE-B/10B_manifest_future_b.json").read_text())
        expected_sha=m["future_inputs"]["micro"]["sha256"]
    must(digest(source)==expected_sha,"packet source SHA mismatch")
    # Infer in bounded UID batches: avoid materializing full Future-B tensor in memory.
    observed={}
    csv.field_size_limit(sys.maxsize)
    with source.open(newline="",encoding="utf-8") as f:
        reader=csv.DictReader(f)
        must({"pcap_uid","site_label","direction_vector","time_vector"}<=set(reader.fieldnames or []),
             "external timing columns absent")
        for row in reader:
            key=row["pcap_uid"]
            if key not in true_by_uid:continue
            must(key not in observed,"duplicate external UID")
            must(row["site_label"]==true_by_uid[key],"UID/site label disagreement")
            ds=parse_direction(row["direction_vector"])
            ts=parse_times(row["time_vector"])
            observed[key]=np.asarray(tiktok_dt_from_sequences(ds,ts),dtype=np.float32)
    must(len(observed)==n and set(observed)==set(uid),"external UID coverage mismatch")
    checkpoint=torch.load(ck,map_location="cpu",weights_only=True)
    must(checkpoint["epoch"]==b["selected_epoch"] and checkpoint["config"]==B.CONFIG,
         "checkpoint config or epoch mismatch")
    model=B.original_model()
    model.load_state_dict(checkpoint["model"],strict=True)
    model=model.to("cuda:0" if torch.cuda.is_available() else "cpu").eval()
    device=next(model.parameters()).device
    probs=np.empty((n,65),dtype=np.float32)
    with torch.inference_mode():
        for start in range(0,n,256):
            end=min(start+256,n)
            x=np.stack([observed[k] for k in uid[start:end]])
            z=model(torch.from_numpy(x[:,None,:]).to(device))
            p=torch.softmax(z,dim=1).cpu().numpy()
            must(np.isfinite(p).all(),"nonfinite external probabilities")
            probs[start:end]=p
    must(np.allclose(probs.sum(axis=1),1.,atol=2e-4,rtol=0),"probability simplex failure")
    order=np.argsort(-probs,axis=1)
    rank=(order==y[:,None]).argmax(axis=1)+1
    tp=np.bincount(y[order[:,0]==y],minlength=65)
    pred_count=np.bincount(order[:,0],minlength=65)
    support=np.bincount(y,minlength=65)
    macro_f1=float(np.divide(2*tp,support+pred_count,out=np.zeros(65),
                             where=support+pred_count>0).mean())
    metrics={"accuracy":float((order[:,0]==y).mean()),"macro_f1":macro_f1,
             "top5":float((rank<=5).mean()),"mrr":float(np.mean(1./rank))}
    target.mkdir(parents=True,mode=0o700)
    artifact=target/"scores_sota_aligned.npz"
    np.savez_compressed(artifact,pcap_uid=uid,query_date=dates,
                        y_true=y,candidate_labels=names,sota_probs=probs)
    report={"classification":"S8_TIKTOK_WFLIB_FROZEN_EXTERNAL_V1",
            "cohort":cohort,"model":"TIKTOK_WFLIB_DT","n":n,
            "metrics":metrics,"checkpoint_sha256":c["checkpoint_sha256"],
            "C_manifest_sha256":sha(path),
            "source_sha256":expected_sha,"canonical_score_sha256":ref["source_sha256"],
            "score_sha256":sha(artifact),"no_external_tuning":True,
            "dataset_c_opened":False}
    (target/"manifest.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S8_TIKTOK_EXTERNAL_PASS",cohort,metrics,flush=True)

if __name__=="__main__":
    try:
        parser=argparse.ArgumentParser()
        parser.add_argument("--stage",choices=["C","evaluate"],required=True)
        parser.add_argument("--cohort",choices=list(COHORT_N))
        a=parser.parse_args()
        if a.stage=="C":train_c()
        else:
            must(a.cohort is not None,"--cohort required for evaluate")
            inference(a.cohort)
    except Exception as exc:
        print("S8_FROZEN_ERROR",type(exc).__name__,str(exc)[:700],file=sys.stderr,flush=True)
        sys.exit(4)
