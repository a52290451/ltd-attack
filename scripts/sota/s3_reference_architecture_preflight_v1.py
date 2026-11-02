#!/usr/bin/env python3
"""S3 original-code architecture gate for pinned WFlib, no training, no datasets."""
from __future__ import annotations
import argparse
import datetime as dt
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.models.sota.v1.upstream_models import (
    DEFAULT_CHECKOUT,audit_signatures,verify_checkout,get_model
)

def check(ok,why):
    if not ok:raise RuntimeError("SOTA_S3_BLOCKED "+why)

def smoke(checkout):
    import torch
    torch.set_num_threads(min(2,torch.get_num_threads()))
    result={}
    for model_name,channels in (("DF",1),("VarCNN",2)):
        model=get_model(model_name,65,checkout=checkout)
        model.eval()
        params=sum(x.numel() for x in model.parameters())
        with torch.inference_mode():
            sample=torch.zeros((2,channels,5000),dtype=torch.float32)
            sample[0,:,0]=1.0
            predictions=model(sample)
        check(tuple(predictions.shape)==(2,65),
              model_name+" output shape not [2,65]")
        check(bool(torch.isfinite(predictions).all()),
              model_name+" returned NaN/Inf logits")
        result[model_name]={"parameters":params,
                            "input_shape":[2,channels,5000],
                            "output_shape":list(predictions.shape),
                            "finite_logits":True,
                            "original_upstream_source":True,
                            "training_steps":0}
        print("SOTA_S3_FORWARD_PASS",model_name,result[model_name],flush=True)
        del model,sample,predictions
    return result

def run(args):
    plan=audit_signatures()
    report={"classification":"SOTA_S3_UPSTREAM_ARCHITECTURE_PREFLIGHT",
            "model_source_pinned":plan,
            "synthetic_only":True,
            "future_b_opened":False,"dataset_c_opened":False,
            "training_started":False,
            "release_for_training":False}
    if args.selftest:
        check(plan["output"]["shape"]==["N",65],"invalid classes")
        check(plan["model_inputs"]["DF"]["channels"]==1,"DF channel mismatch")
        check(plan["model_inputs"]["VarCNN"]["channels"]==2,"DT2 channel mismatch")
        print("SOTA_S3_CONTRACT_SELFTEST=PASS")
        print("SOTA_TRAINING=NOT_AUTHORIZED")
        return
    report["checkout"]=verify_checkout(args.checkout)
    print("SOTA_S3_WFLIB_PIN_VERIFIED",report["checkout"]["commit"],flush=True)
    if args.forward_smoke:
        if importlib.util.find_spec("torch") is None:
            raise RuntimeError("PyTorch not installed in selected python; activate isolated SOTA venv")
        report["architecture_forward"]=smoke(args.checkout)
    else:
        print("SOTA_S3_FORWARD_SMOKE=SKIPPED_RUN_WITH_--forward-smoke")
    path=Path.home()/"ltd-ops"/"sota"
    path.mkdir(mode=0o700,parents=True,exist_ok=True)
    dest=path/("s3_reference_"+dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")+".json")
    dest.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    dest.chmod(0o600)
    print("SOTA_S3_REPORT",dest)
    print("SOTA_S3_TRAINING_GATE=PENDING")
    print("NO_FUTURE_B_NO_DATASET_C_NO_TRAINING=YES")
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--checkout",type=Path,default=DEFAULT_CHECKOUT)
    modes=p.add_mutually_exclusive_group()
    modes.add_argument("--selftest",action="store_true")
    modes.add_argument("--forward-smoke",action="store_true",
                       help="CPU-only synthetic [2,1/2,5000] model test")
    a=p.parse_args()
    try:run(a)
    except Exception as ex:
        print("SOTA_S3_PREFLIGHT_ABORTED",type(ex).__name__,str(ex)[:550],
              file=sys.stderr,flush=True)
        sys.exit(4)
