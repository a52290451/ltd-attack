#!/usr/bin/env python3
"""Synthetic DF/VarCNN 2x5000 CUDA smoke, no training, no datasets.

Requires a pinned local WFlib checkout and isolated PyTorch environment.
"""
from __future__ import annotations
import datetime as dt
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.models.sota.v1.upstream_models import get_model,verify_checkout

def main():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device available to isolated PyTorch")
    device=torch.device("cuda:0")
    capability=torch.cuda.get_device_capability(0)
    if capability[0]<12:
        raise RuntimeError("Unexpected CUDA compute capability: "+str(capability))
    provenance=verify_checkout()
    torch.set_num_threads(min(2,torch.get_num_threads()))
    print("SOTA_CUDA_TORCH",torch.__version__)
    print("SOTA_CUDA_RUNTIME",torch.version.cuda)
    print("SOTA_CUDA_GPU",torch.cuda.get_device_name(0))
    print("SOTA_CUDA_CAPABILITY",capability)
    records={}
    for name,channels in (("DF",1),("VarCNN",2)):
        model=get_model(name,65).eval().to(device)
        x=torch.zeros((2,channels,5000),device=device,dtype=torch.float32)
        x[0,:,0]=1.0
        with torch.inference_mode():
            result=model(x)
        torch.cuda.synchronize()
        if tuple(result.shape)!=(2,65) or not bool(torch.isfinite(result).all()):
            raise RuntimeError(name+" returned wrong shape or non-finite logits")
        records[name]={
            "input_shape":[2,channels,5000],
            "logit_shape":[2,65],
            "parameters":sum(p.numel() for p in model.parameters()),
            "finite_logits":True,
            "device":"cuda:0",
            "max_memory_allocated":torch.cuda.max_memory_allocated(0),
            "training_steps":0,
        }
        print("SOTA_CUDA_SYNTHETIC_FORWARD_PASS",name,records[name])
        del x,result,model
        torch.cuda.empty_cache()
    report={"classification":"SOTA_S3_CUDA_SYNTHETIC_NO_TRAINING",
            "torch":torch.__version__,"torch_cuda":torch.version.cuda,
            "capability":list(capability),"upstream":provenance,
            "results":records,"future_b_opened":False,
            "dataset_c_opened":False,"training_steps":0,
            "release_for_training":False}
    output_dir=Path.home()/"ltd-ops"/"sota"
    output_dir.mkdir(parents=True,exist_ok=True)
    output=output_dir/("s3_cuda_smoke_"+dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")+".json")
    output.write_text(json.dumps(report,indent=2)+"\n")
    output.chmod(0o600)
    print("SOTA_CUDA_REPORT",output)
    print("SOTA_CUDA_SMOKE=PASS_NO_TRAINING")
if __name__=="__main__":
    try:main()
    except Exception as error:
        print("SOTA_CUDA_SMOKE_ABORTED",type(error).__name__,str(error)[:400],
              file=sys.stderr,flush=True)
        sys.exit(4)
