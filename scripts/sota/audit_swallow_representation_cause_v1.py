#!/usr/bin/env python3
"""LTD SOTA second-stage posthoc diagnosis: Swallow representation failure and WFCAT shift.

Read only. CPU only. No relabeling, training, hyperparameter search, evaluation
updates, or edits of frozen artifacts. One JSON report in audit directory.
"""
from __future__ import annotations
import argparse,hashlib,json,os,sys
from pathlib import Path
import numpy as np

def check(v,msg):
    if not v:raise RuntimeError("SOTA_CAUSE_AUDIT_FAILED: "+msg)
def metric(a):
    a=np.asarray(a,dtype=np.float64)
    return {"shape":list(a.shape),"mean":float(np.mean(a)),
            "std":float(np.std(a)),"min":float(np.min(a)),
            "max":float(np.max(a)),"is_finite":bool(np.isfinite(a).all())}
def sample_probe(net,x):
    import torch
    net.eval()
    with torch.inference_mode():
        v=net(torch.as_tensor(x,dtype=torch.float32))
    if isinstance(v,(tuple,list)):v=v[0]
    if isinstance(v,dict):raise RuntimeError("unexpected Swallow output dictionary")
    a=v.detach().cpu().numpy().reshape(len(x),-1)
    return {"output":metric(a),"variation_mean_feature_std":
            float(a.std(axis=0).mean()),
            "variation_max_feature_std":float(a.std(axis=0).max()),
            "all_zero_fraction":float((a==0).mean()),
            "distinct_outputs_rounded_6":int(np.unique(np.round(a,6),axis=0).shape[0])}
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--base",type=Path,default=Path("/data/datasets_brayan/ltd-sota-2026"))
    p.add_argument("--repo",type=Path,default=Path.home()/"ltd-attack")
    p.add_argument("--chain-json",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    args=p.parse_args()
    check(args.chain_json.is_file(),"chained forensic report missing")
    report=json.loads(args.chain_json.read_text())
    check(report.get("status")=="CHAIN_AUDIT_DONE_INSPECT_FINDINGS",
          "chained forensic report incomplete")
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(42)
    sys.path.insert(0,str(args.repo))
    from scripts.sota import s11_swallow_authors_v1 as S
    root=args.base/"artifacts/sota/s11_swallow_authors_seed42"
    manifest=json.loads((root/"pretrain_B/manifest.json").read_text())
    ck=Path(manifest["checkpoint"])
    check(ck.is_file(),"pretrained checkpoint absent")
    sha=hashlib.sha256(ck.read_bytes()).hexdigest()
    check(sha==manifest["checkpoint_sha256"],"pretrain checkpoint SHA mismatch")
    c=np.load(root/"historical_cif_cache/cif_uint16.npy",mmap_mode="r")
    split=np.load(root/"historical_cif_cache/split.npy",mmap_mode="r")
    indices=[]
    for sid in (0,1,2):
        candidates=np.flatnonzero(split==sid)
        indices.extend(np.random.default_rng(42+sid).choice(candidates,
                                    size=8,replace=False).tolist())
    X=np.asarray(c[indices],dtype=np.float32)[:,None,:,:].copy()
    model,_,_=S.network()
    children=list(model.children())
    result={
      "classification":"LTD_S11_SWALLOW_CAUSE_DIAGNOSTIC_V1",
      "pretrain_checkpoint_sha256":sha,
      "input_shape":list(X.shape),
      "input_overall":metric(X),
      "input_sample_sum_std":float(X.reshape(len(X),-1).sum(axis=1).std()),
      "model_top_level_children":[type(module).__name__ for module in children],
      "baseline_random_full":sample_probe(model,X),
      "baseline_random_trunk_excluding_last_child":
        sample_probe(torch.nn.Sequential(*children[:-1]),X),
    }
    check(manifest["config"]==S.CONFIG,"runtime config differs from frozen pretrain")
    pretrained=torch.load(ck,map_location="cpu",weights_only=True)
    check(pretrained["config"]==S.CONFIG,"embedded pretrain config differs")
    model.load_state_dict(pretrained["model"],strict=True)
    result["trained_pretrain_full"]=sample_probe(model,X)
    result["trained_pretrain_trunk_excluding_last_child"]=sample_probe(
       torch.nn.Sequential(*list(model.children())[:-1]),X)
    s2={}
    for name,stage,fn in (("B","supervised_B","best.pt"),
                         ("C","supervised_C","model.pt")):
        m=json.loads((root/stage/"manifest.json").read_text())
        state=torch.load(m["checkpoint"],map_location="cpu",weights_only=True)
        full,_,_=S.network()
        enc=torch.nn.Sequential(*list(full.children())[:-1])
        head=torch.nn.Linear(512,65)
        enc.load_state_dict(state["encoder"],strict=True)
        head.load_state_dict(state["classifier"],strict=True)
        z=sample_probe(enc,X)
        full_logits=torch.nn.Sequential(enc,torch.nn.Flatten(),head)
        logits=sample_probe(full_logits,X)
        s2[name]={"epoch":m.get("selected_epoch"),
                  "encoder":z,"logits":logits,
                  "head_weight":metric(head.weight.detach().numpy()),
                  "head_bias":metric(head.bias.detach().numpy()),
                  "history_first":m.get("history",[None])[0],
                  "history_last":m.get("history",[None])[-1]}
    result["supervised_stages"]=s2
    result["wfcat_shift_from_chained_audit"]=report.get("external_original_feature_samples",{})
    result["historical_profiles"]=report.get("historical_cache_feature_profiles",{})
    result["interpretive_limits"]=[
       "Read-only descriptive posthoc diagnosis; no causality from correlation.",
       "Full and trunk probes reveal architectural/representation collapse but do not prove its optimizer cause.",
       "External feature samples represent only sampled traces; no adaptation or selection on Future-B allowed.",
       "Artifacts, author methods, and datasets remain frozen."]
    check(not args.output.exists(),"refusing to overwrite report "+str(args.output))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print("SWALLOW_CAUSE_AUDIT_PASS",str(args.output))
    for key in ("baseline_random_full","baseline_random_trunk_excluding_last_child",
                "trained_pretrain_full","trained_pretrain_trunk_excluding_last_child"):
        x=result[key]
        print(key,"std",x["variation_mean_feature_std"],
              "distinct",x["distinct_outputs_rounded_6"],
              "zero_fraction",x["all_zero_fraction"])
    for key,stage in s2.items():
        print("SUPERVISED",key,"encoder_std",
              stage["encoder"]["variation_mean_feature_std"],
              "logit_std",stage["logits"]["variation_mean_feature_std"],
              "epochs",stage["epoch"])
if __name__=="__main__":main()
