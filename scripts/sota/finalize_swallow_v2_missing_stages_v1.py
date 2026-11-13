#!/usr/bin/env python3
"""Swallow v2 read-only status + safe completion of missing downstream stages.

Never restarts a pretrain, overwrites checkpoints, or modifies the frozen V1.
If --advance is passed, completes ONLY absent full stages (B/C/evals), then
release certificate and paired LTD metrics. Blocks partial/corrupt stages.
"""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
def fail(s):raise RuntimeError("SWALLOW_V2_SAFE_FINALIZATION_BLOCKED "+s)
def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b""):h.update(chunk)
    return h.hexdigest()
def stage(path,expected,model):
    if not path.exists():return None
    mp=path/"manifest.json"
    if not mp.is_file():fail("Partial stage exists, no automatic restart: "+str(path))
    x=json.loads(mp.read_text())
    if x.get("classification")!=expected:fail("Wrong classification "+str(mp))
    if model:
        if x.get("config")!=model:fail("Training config mismatch "+str(mp))
        ck=Path(x["checkpoint"])
        if not ck.is_file() or digest(ck)!=x["checkpoint_sha256"]:fail("Checkpoint SHA mismatch "+str(ck))
        if x.get("external_data_opened") is not False or x.get("dataset_c_opened") is not False:
            fail("Historical only/source constraints absent "+str(mp))
    else:
        score=path/"scores_sota_aligned.npz"
        if not score.is_file() or digest(score)!=x.get("score_sha256"):
            fail("Evaluation score SHA mismatch "+str(score))
    return x
def runner(py,cmd,repo,env):
    print("RUNNING_ONLY_MISSING_STAGE", " ".join(map(str,cmd)),flush=True)
    result=subprocess.run([str(py),*map(str,cmd)],cwd=repo,env=env,check=False)
    if result.returncode:fail("Subprocess failed; no automatic restart. rc="+str(result.returncode))
def locate_report(aud,pattern,classification,release_dir):
    choices=sorted(aud.glob(pattern))
    for p in reversed(choices):
        if p.is_file():
            try:
                x=json.loads(p.read_text())
            except Exception:continue
            if x.get("classification")==classification:
                if release_dir is None or x.get("v2_distinct_output")==str(release_dir):
                    return p,x
    return None,None
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--repo",required=True,type=Path)
    p.add_argument("--base",required=True,type=Path)
    p.add_argument("--advance",action="store_true",help="Complete missing stages only; guarded, no overwrite")
    a=p.parse_args()
    sys.path.insert(0,str(a.repo))
    from scripts.sota import s11_swallow_authors_bntrain_v2 as S
    from scripts.sota import s11_swallow_authors_v1 as V1
    from scripts.sota.sota_paths_v1 import ARTIFACT
    root=S.OUT
    aud=a.base/"audits"
    if root!=ARTIFACT/"s11_swallow_authors_BNtrain_v2_seed42":
        fail("Unexpected v2 artifact root")
    if root==V1.OUT or S.CACHE!=V1.CACHE:
        fail("V1/v2 output isolation violation")
    names=[
      ("PRETRAIN",root/"pretrain_B","S11_SWALLOW_AUTHORS_B_PRETRAIN_V2",True,"pretrain-B"),
      ("SUPERVISED_B",root/"supervised_B","S11_SWALLOW_SUPERVISED_B_V2",True,"supervised-B"),
      ("REFIT_C",root/"supervised_C","S11_SWALLOW_SUPERVISED_C_V2",True,"C"),
      ("09B",root/"evaluation"/"09B","S11_SWALLOW_EXTERNAL_FROZEN_V1",False,"evaluate"),
      ("FUTURE_B",root/"evaluation"/"10B_DEV_FROZEN","S11_SWALLOW_EXTERNAL_FROZEN_V1",False,"evaluate")]
    existing={}
    for name,folder,classification,model_flag,arg in names:
        evidence=stage(folder,classification,S.CONFIG if model_flag else None)
        if evidence and not model_flag:
            expected=14687 if name=="09B" else 18543
            if evidence.get("n")!=expected:fail("External cohort N mismatch "+name)
        existing[name]=bool(evidence)
        print("SWALLOW_V2_STAGE",name,"VERIFIED" if evidence else "MISSING",flush=True)
    if not existing["PRETRAIN"]:
        if a.advance:fail("Pretraining does not have complete manifest; no automatic retraining")
        print("SWALLOW_V2_STATUS PRETRAIN_MISSING_OR_NOT_FINALIZED")
        return
    pre=json.loads((root/"pretrain_B/manifest.json").read_text())
    monitor=pre.get("probe_monitor",[])
    if len(monitor)!=100 or any(x["feature_std"]<1e-7 or x["unique"]<3 for x in monitor[2:]):
        fail("Pretraining monitor has insufficient diversity or missing epochs")
    print("SWALLOW_V2_PRETRAIN_VERIFIED",100,"FINAL_STD",monitor[-1]["feature_std"],
          "FINAL_DISTINCT",monitor[-1]["unique"],flush=True)
    if a.advance:
        for name,folder,classification,model_flag,arg in names[1:]:
            if existing[name]:continue
            # Explicit checkpoint chain; training requires GPU only for B and C,
            # evaluation CPU. Scripts themselves prevent output overwrite.
            if name=="REFIT_C" and not existing["SUPERVISED_B"]:
                fail("Supervised B must complete before C")
            if name in ("09B","FUTURE_B") and not existing["REFIT_C"]:
                fail("C must complete before external inference")
            command=["scripts/sota/s11_swallow_authors_bntrain_v2.py","--stage",arg]
            if arg=="evaluate":
                cohort="09B" if name=="09B" else "10B_DEV_FROZEN"
                command+=["--cohort",cohort]
            if arg in ("supervised-B","C"):
                r=subprocess.run(["nvidia-smi","--query-compute-apps=pid",
                    "--format=csv,noheader,nounits"],capture_output=True,text=True)
                if r.returncode or r.stdout.strip():fail("GPU occupied; do not start training")
            env=dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"]="-1" if arg=="evaluate" else env.get("CUDA_VISIBLE_DEVICES","0")
            runner(sys.executable,command,a.repo,env)
            check=stage(folder,classification,S.CONFIG if model_flag else None)
            if check is None:fail("Completed subprocess without manifest "+name)
            existing[name]=True
            print("SWALLOW_V2_STAGE_VERIFIED_AFTER_RUN",name,flush=True)
    if not all(existing.values()):
        print("SWALLOW_V2_STATUS DOWNSTREAM_STAGES_MISSING")
        return
    rel,relmeta=locate_report(aud,"swallow_BNtrain_v2_release_*.json",
                "LTD_SWALLOW_TARGET_BNTRAIN_V2_HISTORICAL_RELEASE_GATE",root)
    if rel is None:
        rel=aud/("swallow_BNtrain_v2_release_final_"+
                  datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+".json")
        if a.advance:
            runner(sys.executable,["scripts/sota/certify_swallow_bntrain_v2_release_v1.py",
                       "--repo",a.repo,"--base",a.base,"--output",rel],a.repo,
                       dict(os.environ,CUDA_VISIBLE_DEVICES="-1"))
            relmeta=json.loads(rel.read_text())
        else:
            print("SWALLOW_V2_RELEASE_NOT_FOUND; use --advance")
            return
    if relmeta.get("status")!="PASS_ADAPTED_SWALLOW_V2_NONCOLLAPSED_TECHNICAL_GATE_NOT_PAPER_EXACT":
        fail("Release certificate did not pass "+str(rel))
    print("SWALLOW_V2_RELEASE_VERIFIED",rel,flush=True)
    files=sorted(aud.glob("swallow_BNtrain_v2_vs_LTD_*.json"))
    previous=None
    for p in reversed(files):
        try:
            x=json.loads(p.read_text())
            if (x.get("classification")=="LTD_SWALLOW_V2_FROZEN_S5S6_PAIRED_COMPARISON_V1" and
                  x.get("release_report_sha256")==digest(rel)):
                previous=p;break
        except Exception:pass
    if previous is not None:
        print("SWALLOW_V2_PAIRED_RESULT_ALREADY_PRESENT",previous)
        res=json.loads(previous.read_text())
    elif a.advance:
        dest=aud/("swallow_BNtrain_v2_vs_LTD_final_"+
                  datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+".json")
        runner(sys.executable,["scripts/sota/compare_swallow_bntrain_v2_ltd_canonical_v1.py",
                  "--repo",a.repo,"--base",a.base,"--release",rel,"--output",dest],
                  a.repo,dict(os.environ,CUDA_VISIBLE_DEVICES="-1"))
        res=json.loads(dest.read_text())
    else:
        print("SWALLOW_V2_PAIRED_MISSING; use --advance")
        return
    for cohort,v in res["cohorts"].items():
        print("SWALLOW_V2_FINAL",cohort,"LTD_HYBRID_ACC",v["accuracy"]["LTD_HYBRID"],
              "SWALLOW_V2_ACC",v["accuracy"]["SWALLOW_V2"],
              "HYBRID_MINUS_V2_CI95",
              v["paired_differences"]["LTD_HYBRID"]["accuracy"]["paired_bootstrap_ci95"],flush=True)
    print("SWALLOW_V2_ALL_FINISHED_PAIRED_COMPARISON_COMPLETE")
if __name__=="__main__":main()
