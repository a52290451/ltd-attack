#!/usr/bin/env python3
"""Read-only Swallow BYOL epoch trajectory and augmentation forensic diagnostics.

Reads frozen 20/40/60/80/100 checkpoints, original CIF cache and original
augmentation code. CPU only, never fine-tunes or mutates source artifacts.
"""
from __future__ import annotations
import argparse,hashlib,json,random,sys
from pathlib import Path
import numpy as np

def need(condition,message):
    if not condition:raise RuntimeError("SWALLOW_TRAJECTORY_BLOCKED "+message)
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--repo",type=Path,required=True)
    p.add_argument("--base",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    import torch
    torch.set_num_threads(2)
    sys.path.insert(0,str(a.repo))
    from scripts.sota import s11_swallow_authors_v1 as S
    loc=a.base/"artifacts/sota/s11_swallow_authors_seed42"
    m=json.loads((loc/"pretrain_B/manifest.json").read_text())
    need(m["config"]==S.CONFIG,"pretrain provenance changed")
    X=np.load(loc/"historical_cif_cache/cif_uint16.npy",mmap_mode="r")
    s=np.load(loc/"historical_cif_cache/split.npy",mmap_mode="r")
    times=np.load(loc/"historical_cif_cache/load_time.npy",mmap_mode="r")
    duration=np.load(loc/"historical_cif_cache/slot_duration.npy",mmap_mode="r")
    indices=[]
    for sid in (0,1,2):
        positions=np.flatnonzero(s==sid)
        indices.extend(np.random.default_rng(42+sid).choice(
            positions,size=8,replace=False).tolist())
    values=np.asarray(X[indices],dtype=np.float32).copy()
    x=torch.from_numpy(values[:,None,:,:])
    need(x.shape==(24,1,2,1000),"CIF sample shape mismatch")
    model,_,augment_class=S.network()
    report={"classification":"SWALLOW_BYOL_CHECKPOINT_TRAJECTORY_V1",
            "checkpoint_epochs":[],"augmentation_diagnostics":{},
            "notes":["20-epoch checkpoint intervals do not identify first exact failing batch.",
                     "Author augmentation source is loaded with documented in-memory loop safety fix.",
                     "This script does not fit, rescore or modify any historical experiment."]}
    def probe(t):
        t.eval()
        with torch.inference_mode():
            z=t(x).detach().numpy().reshape(len(indices),-1)
        return {"between_sample_std_mean":float(z.std(axis=0).mean()),
                "between_sample_std_max":float(z.std(axis=0).max()),
                "unique_rounded_6":int(len(np.unique(z.round(6),axis=0))),
                "zero_fraction":float(np.mean(z==0)),
                "overall_rms":float(np.sqrt(np.mean(z.astype(np.float64)**2)))}
    report["untrained_baseline"]=probe(model)
    for ep in (20,40,60,80,100):
        ck=loc/f"pretrain_B/epoch_{ep:03d}.pt"
        need(ck.is_file(),"missing "+str(ck))
        with ck.open("rb") as f:
            dig=hashlib.file_digest(f,"sha256").hexdigest()
        state=torch.load(ck,map_location="cpu",weights_only=True)
        need(state["epoch"]==ep and state["config"]==S.CONFIG,
             f"epoch {ep} checkpoint mismatch")
        model.load_state_dict(state["model"],strict=True)
        result=probe(model)
        result.update({"epoch":ep,"sha256":dig})
        report["checkpoint_epochs"].append(result)
        print("SWALLOW_BYOL_EPOCH_AUDIT",ep,
              result["between_sample_std_mean"],
              result["unique_rounded_6"],flush=True)
    losses=m.get("history",[])
    report["pretraining_loss"]={
      "history_n":len(losses),
      "first":losses[0] if losses else None,
      "last":losses[-1] if losses else None,
      "min":min((v.get("BYOL_loss",float("inf")) for v in losses),default=None),
      "max":max((v.get("BYOL_loss",float("-inf")) for v in losses),default=None)}
    # Author-source augmentations of real cached CIF inputs, without gradients.
    random.seed(42);np.random.seed(42)
    aug=augment_class()
    measures=[]
    for row,index in zip(values[:8],indices[:8]):
        a1=np.asarray(aug.augment(row.copy(),float(times[index]),float(duration[index])),dtype=np.float32)
        a2=np.asarray(aug.augment(row.copy(),float(times[index]),float(duration[index])),dtype=np.float32)
        need(a1.shape==(2,1000) and a2.shape==(2,1000),"augmentation altered tensor shape")
        need(np.isfinite(a1).all() and np.isfinite(a2).all(),"augmentation nonfinite")
        measures.append({
          "original_sum":float(row.sum()),
          "view1_sum":float(a1.sum()),"view2_sum":float(a2.sum()),
          "views_equal":bool(np.array_equal(a1,a2)),
          "mean_view_absolute_distance":float(np.abs(a1-a2).mean()),
          "mean_original_view_absolute_distance":float(np.abs(row-a1).mean())})
    report["augmentation_diagnostics"]={
        "samples":measures,
        "identical_pair_count":sum(int(r["views_equal"]) for r in measures),
        "mean_two_view_difference":float(np.mean([r["mean_view_absolute_distance"] for r in measures])),
        "mean_original_view_difference":float(np.mean([r["mean_original_view_absolute_distance"] for r in measures]))}
    vals=report["checkpoint_epochs"]
    collapsed=[z["epoch"] for z in vals
               if z["between_sample_std_mean"]<1e-7 and z["unique_rounded_6"]==1]
    report["earliest_observed_collapsed_checkpoint"]=collapsed[0] if collapsed else None
    need(not a.output.exists(),"refuse overwrite "+str(a.output))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print("SWALLOW_TRAJECTORY_AUDIT_PASS",a.output)
    print("EARLIEST_COLLAPSED_CHECKPOINT",report["earliest_observed_collapsed_checkpoint"])
    print("AUGMENT_IDENTICAL_PAIRS",report["augmentation_diagnostics"]["identical_pair_count"])
if __name__=="__main__":main()
