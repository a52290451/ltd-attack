#!/usr/bin/env python3
"""Resumable authors'-source SOTA orchestration for Perseo or Zeus.

No training unless --run is explicitly supplied. GPU work is serialized,
blocked when any other NVML compute PID is observed. No stage is skipped
based on directory presence alone: completed manifest SHA, classification,
data-exclusion and key evidence must pass. Partial dirs block (no deletion).
Unverified source archives, missing S1/S5, or missing CUDA block training.

Usage:
  python scripts/sota/original_sota_orchestrator_v2.py --stage status
  python scripts/sota/original_sota_orchestrator_v2.py --stage all --dry-run
  python scripts/sota/original_sota_orchestrator_v2.py --stage all --run
  python scripts/sota/original_sota_orchestrator_v2.py --stage train --model NetCLR --run
"""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.sota.sota_paths_v1 import ARTIFACT,S1,S5,FUTURE
SOTA=ROOT/"scripts/sota"
PY=os.environ.get("LTD_SOTA_PYTORCH_PY","/data/ltd-sota-bsierra/venv-cu128/bin/python")
KPY=os.environ.get("LTD_SOTA_KERAS_PY","/data/ltd-sota-bsierra/venv-keras-authors/bin/python")
COHORTS=("09B","10B_DEV_FROZEN")
MODELS=("NetCLR","Swallow","WFCAT","DF","VarCNN","TikTok","RoFiRe")
ROOTS={
 "NetCLR":ARTIFACT/"s9_netclr_author_original_v1",
 "Swallow":ARTIFACT/"s11_swallow_authors_seed42",
 "WFCAT":ARTIFACT/"s12_wfcat_authors_iat_seed42",
 **{n:ARTIFACT/("authors_"+n.lower()+"_seed42") for n in ("DF","VarCNN","TikTok","RoFiRe")}}
DIRS={
 "NetCLR":(("pretrain-B","pretrain_B_v1","S9_NETCLR_AUTHORS_MODEL_AUGMENT_HISTORICAL_B"),
           ("supervised-B","supervised_B_v1","S9_NETCLR_AUTHORS_SUPERVISED_B_V1"),
           ("C","refit_C_v1","S9_NETCLR_AUTHORS_HISTORICAL_C_FROZEN_V1")),
 "Swallow":(("pretrain-B","pretrain_B","S11_SWALLOW_AUTHORS_B_PRETRAIN_V1"),
            ("supervised-B","supervised_B","S11_SWALLOW_SUPERVISED_B_V1"),
            ("C","supervised_C","S11_SWALLOW_SUPERVISED_C_V1")),
 "WFCAT":(("B","B","S12_WFCAT_CHRONOLOGICAL_B"),
          ("C","C","S12_WFCAT_CHRONOLOGICAL_C")),
 **{n:(("B","B","AUTHOR_SOTA_HISTORICAL_B_V1"),
        ("C","C","AUTHOR_SOTA_HISTORICAL_C_V1"))
    for n in ("DF","VarCNN","TikTok","RoFiRe")}}
EVAL_TYPES={
 "NetCLR":"S9_NETCLR_AUTHORS_FROZEN_EXTERNAL_V1",
 "Swallow":"S11_SWALLOW_EXTERNAL_FROZEN_V1",
 "WFCAT":"S12_WFCAT_FROZEN_EXTERNAL_V1",
 **{n:"AUTHOR_SOTA_EXTERNAL_FROZEN_V1"
    for n in ("DF","VarCNN","TikTok","RoFiRe")}}
RUNNERS={
 "NetCLR":"run_s9_netclr_authors_v1.sh",
 "Swallow":"run_s11_swallow_authors_v1.sh",
 "WFCAT":"run_s12_wfcat_authors_v1.sh",
 **{n:"run_authors_keras_s3_s4_s8_v1.sh"
    for n in ("DF","VarCNN","TikTok","RoFiRe")}}
def require(ok,msg):
    if not ok:raise RuntimeError("SOTA_BATTERY_V2_BLOCKED "+msg)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as fp:
        for b in iter(lambda:fp.read(4*1024*1024),b""):h.update(b)
    return h.hexdigest()
def check_manifest(path,classification,cohort=None):
    if not path.is_file():return False
    m=json.loads(path.read_text())
    require(m.get("classification")==classification,
            "unexpected stage classification: "+str(path))
    require(m.get("dataset_c_opened") is False,
            "Dataset C contamination: "+str(path))
    require(m.get("external_data_opened") is False or
            m.get("no_external_tuning") is True or
            m.get("external_data_used_for_fit") is False or
            cohort is not None,
            "cannot establish historical-only fit: "+str(path))
    if cohort:
        require(m.get("cohort")==cohort and m.get("n")==(
            14687 if cohort=="09B" else 18543),"cohort count mismatch")
        evidence=path.parent/"scores_sota_aligned.npz"
        require(evidence.is_file() and sha(evidence)==m["score_sha256"],
                "external NPZ missing/changed: "+str(evidence))
    else:
        ck=m.get("checkpoint")
        require(ck and Path(ck).is_file() and
                sha(ck)==m.get("checkpoint_sha256"),
                "checkpoint missing or mismatched: "+str(path))
    return True
def state(model,stage,folder,klass,cohort=None):
    manifest=ROOTS[model]/folder/"manifest.json"
    if manifest.is_file():
        check_manifest(manifest,klass,cohort)
        return "PASS_VERIFIED"
    if manifest.parent.exists():
        return "BLOCKED_PARTIAL_OUTPUT"
    return "PENDING"
def cmd(model,stage):
    if model in ("DF","VarCNN","TikTok","RoFiRe"):
        return ["bash",str(SOTA/RUNNERS[model]),model,stage]
    return ["bash",str(SOTA/RUNNERS[model]),stage]
def gpu_idle():
    p=subprocess.run(["nvidia-smi","--query-compute-apps=pid",
                      "--format=csv,noheader,nounits"],
                     capture_output=True,text=True,timeout=20)
    require(p.returncode==0,"GPU busy status inaccessible")
    require(not p.stdout.strip(),
            "GPU has existing compute processes: "+p.stdout.strip())
def execute(model,stage):
    gpu=stage in ("pretrain-B","supervised-B","B","C")
    if gpu:gpu_idle()
    arg=cmd(model,stage)
    print("SOTA_BATTERY_V2_EXEC",model,stage," ".join(arg),flush=True)
    subprocess.run(arg,cwd=ROOT,check=True)
def stages(model,kind):
    fit=list(DIRS[model])
    external=[("09B","evaluation/09B",EVAL_TYPES[model],"09B"),
              ("Future-B","evaluation/10B_DEV_FROZEN",EVAL_TYPES[model],
               "10B_DEV_FROZEN")]
    result=[(s,d,k,None) for s,d,k in fit]+external
    if kind=="train":return result[:len(fit)]
    if kind=="evaluate":return result[len(fit):]
    return result
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--stage",choices=("status","train","evaluate","all"),
                    required=True)
    ap.add_argument("--model",choices=("all",)+MODELS,default="all")
    g=ap.add_mutually_exclusive_group()
    g.add_argument("--run",action="store_true")
    g.add_argument("--dry-run",action="store_true")
    opts=ap.parse_args()
    selection=MODELS if opts.model=="all" else (opts.model,)
    require(opts.stage=="status" or opts.run or opts.dry_run,
            "must explicitly specify --dry-run or --run")
    if opts.run:
        require(S1.is_file() and S5.is_file() and FUTURE.is_file(),
                "missing chronological S1/S5/Future references on this host")
        # SHA inspection is intentionally skipped in status/dry-run, but is
        # mandatory before fit or final evaluation on either server.
        subprocess.run([PY,str(SOTA/"sota_zeus_evidence_bootstrap_v1.py"),
                        "--stage","inspect"],cwd=ROOT,check=True)
    for model in selection:
        for stage,folder,klass,cohort in stages(model,
                        "all" if opts.stage=="status" else opts.stage):
            stat=state(model,stage,folder,klass,cohort)
            print("SOTA_STAGE",model,stage,stat,flush=True)
            if stat=="BLOCKED_PARTIAL_OUTPUT":
                if opts.run:
                    raise RuntimeError("Partial output requires explicit recovery: "+
                                       str(ROOTS[model]/folder))
                continue
            if stat=="PASS_VERIFIED":
                print("SOTA_STAGE_SKIP_VERIFIED",model,stage,flush=True)
                continue
            if opts.run:
                execute(model,stage)
                require(state(model,stage,folder,klass,cohort)=="PASS_VERIFIED",
                        "stage returned success but evidence is not complete "+
                        model+" "+stage)
            elif opts.dry_run:print("SOTA_PLAN"," ".join(cmd(model,stage)),flush=True)
    if opts.run and opts.stage in ("evaluate","all") and opts.model=="all":
        subprocess.run([PY,str(SOTA/"original_authors_results_integrity_v1.py"),
                        "--model","all","--save"],cwd=ROOT,check=True)
    print("SOTA_BATTERY_V2_"+
          ("EXECUTION_PASS" if opts.run else "STATUS_PASS"),flush=True)
if __name__=="__main__":main()
