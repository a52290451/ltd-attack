#!/usr/bin/env python3
"""S4 synthetic optimizer smoke on existing CUDA venv (no Historical rows).

Uses fixed WFlib architecture, optimization, torch runtime and one fake
training step with two examples; NO 09A/B/C, no model checkpoint, no tuning.
Only outputs tiny metadata JSON under /data/ltd-attack/sota/protocol.
"""
from __future__ import annotations
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.models.sota.v1.s4_protocol import validate_config,sha256_file
from src.models.sota.v1.upstream_models import get_model,verify_checkout

CONFIG=ROOT/"configs/sota/sota_s4_frozen_protocol_v1.json"
OUTPUT=Path("/data/ltd-attack/sota/protocol")

def ensure(value,message):
    if not value:raise RuntimeError("SOTA_S4_OPTIMIZER_BLOCKED "+message)

def main():
    config=json.loads(CONFIG.read_text())
    validate_config(config)
    import torch
    ensure(torch.__version__.startswith("2.11.0+cu128"),
           "unexpected torch wheel or CUDA runtime")
    ensure(torch.cuda.is_available(),"CUDA unavailable")
    ensure(torch.cuda.get_device_capability(0)[0]>=12,
           "RTX 5080 compute capability mismatch")
    available,total=torch.cuda.mem_get_info(0)
    ensure(available>=5*1024**3,
           "less than 5 GiB VRAM free; protect other workloads")
    torch.set_num_threads(2)
    upstream=verify_checkout()
    evidence={}
    for name in ("DF","VarCNN"):
        settings=config["model_configs"][name]
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        model=get_model(name,65).to("cuda:0")
        ensure(sum(p.numel() for p in model.parameters())==
               settings["parameter_count_verified"],
               name+" parameter count mismatch")
        model.train()
        optimizer_class=getattr(torch.optim,settings["optimizer"])
        optimizer=optimizer_class(model.parameters(),lr=settings["learning_rate"],
                                 weight_decay=settings["weight_decay"])
        inputs=torch.zeros((2,settings["channels"],5000),
                           device="cuda:0",dtype=torch.float32)
        inputs[0,:,0]=1.0
        inputs[1,:,0]=-1.0
        labels=torch.tensor([0,1],device="cuda:0",dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        logits=model(inputs)
        ensure(tuple(logits.shape)==(2,65),"invalid logit shape")
        loss=torch.nn.functional.cross_entropy(logits,labels)
        ensure(bool(torch.isfinite(loss)),"loss contains NaN/Inf")
        before=float(loss.item())
        loss.backward()
        gradients=[p.grad for p in model.parameters() if p.grad is not None]
        ensure(bool(gradients),"no parameter received a gradient")
        ensure(all(bool(torch.isfinite(g).all()) for g in gradients),
               "non-finite parameter gradients")
        optimizer.step()
        torch.cuda.synchronize()
        evidence[name]={
            "device":"cuda:0","synthetic_examples":2,
            "channels":settings["channels"],
            "output_shape":[2,65],
            "optimizer":settings["optimizer"],
            "learning_rate":settings["learning_rate"],
            "initial_synthetic_loss":before,
            "gradient_tensors":len(gradients),
            "all_gradients_finite":True,
            "synthetic_optimizer_steps":1,
            "historical_training_steps":0,
            "cuda_max_memory_allocated_bytes":torch.cuda.max_memory_allocated(0)
        }
        print("SOTA_S4_SYNTHETIC_OPTIMIZER_PASS",name,
              "LOSS",round(before,5),"GRADS",len(gradients),flush=True)
        del gradients,optimizer,loss,inputs,labels,logits,model
        torch.cuda.empty_cache()
    output={
        "classification":"SOTA_S4_SYNTHETIC_GRADIENT_TEST_NO_DATA_V1",
        "created_utc":datetime.now(timezone.utc).isoformat(),
        "protocol_sha256":sha256_file(CONFIG),
        "torch_version":torch.__version__,
        "torch_cuda":torch.version.cuda,
        "upstream_commit":upstream["commit"],
        "hardware":torch.cuda.get_device_name(0),
        "models":evidence,
        "historical_09A_payload_opened":False,
        "internal_09B_payload_opened":False,
        "future_10B_payload_opened":False,
        "dataset_c_opened":False,
        "historical_training_steps":0,
        "training_battery_released":False,
    }
    OUTPUT.mkdir(parents=True,mode=0o700,exist_ok=True)
    dest=OUTPUT/("s4_optimizer_smoke_"+datetime.now(timezone.utc).
                 strftime("%Y%m%dT%H%M%SZ")+".json")
    ensure(not dest.exists(),"report collision")
    dest.write_text(json.dumps(output,indent=2,ensure_ascii=False)+"\n")
    dest.chmod(0o600)
    print("SOTA_S4_SYNTHETIC_OPTIMIZER=PASS")
    print("SOTA_S4_REPORT",dest)
    print("SOTA_HISTORICAL_TRAINING=NOT_STARTED")
    print("FUTURE_B_UNTOUCHED=YES")

if __name__=="__main__":
    try:main()
    except Exception as e:
        print("SOTA_S4_SYNTHETIC_OPTIMIZER_ABORTED",
              type(e).__name__,str(e)[:500],file=sys.stderr,flush=True)
        sys.exit(4)
