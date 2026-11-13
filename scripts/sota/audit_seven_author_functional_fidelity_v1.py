#!/usr/bin/env python3
"""Executable 7-model original-code methodological gate, CPU, one pass.

- Each method launches in a separate Python process (no authors' models import
  namespace collisions), reuses existing *original source* integration smoke.
- Verifies trained B/C checkpoint/manifests and original source commit.
- WFCAT independently compares raw author feature_transform to the adapter
  on deterministic trace inputs, including timestamp regressions.
- Explicitly checks noncollapsed model predictions and Swallow failure.
- Reports method fidelity honestly: original-source adapted, NOT paper-exact.
- All original experiment/checkpoint/score data remain read-only.
"""
from __future__ import annotations
import argparse,ast,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np

SOURCE_STATUS={
 "DF":{"scope":"original DFNet author class, 65-class adapter; Keras compatibility and hyperparameter protocol adapted"},
 "VarCNN":{"scope":"only direction branch of original VarCNN, NOT paper's complete ensemble"},
 "TikTok":{"scope":"original ConvNet class, DT adapter; Keras compatibility and optimizer protocol adapted"},
 "RoFiRe":{"scope":"original DFVarNet + DyWin 10x rather than author's 200x; 65-class head"},
 "NetCLR":{"scope":"author notebook DFsimCLR/DFNet/NetAugment, modified burst CDF, 100 vs 401 epochs"},
 "WFCAT":{"scope":"author WFCAT + original IAT, 30 vs 50 epochs, batch 16 vs 64; temporal shift"},
 "Swallow":{"scope":"author CIF/ResNet/BYOL, modified augmentation loop, 100 vs 200 epochs, batch32 vs512, target eval; collapsed"}}
ARTIFACT={
 "DF":("authors_df_seed42","B","C"),
 "VarCNN":("authors_varcnn_seed42","B","C"),
 "TikTok":("authors_tiktok_seed42","B","C"),
 "RoFiRe":("authors_rofire_seed42","B","C"),
 "NetCLR":("s9_netclr_author_original_v1","supervised_B_v1","refit_C_v1"),
 "WFCAT":("s12_wfcat_authors_iat_seed42","B","C"),
 "Swallow":("s11_swallow_authors_seed42","supervised_B","supervised_C")}

def require(ok,msg):
    if not ok:raise RuntimeError(msg)
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as s:
        for b in iter(lambda:s.read(1024*1024),b""):h.update(b)
    return h.hexdigest()
def canonical_results(root,name):
    folder,b,c=ARTIFACT[name]
    out={}
    for stage,sub in (("B",b),("C",c)):
        manifest=root/folder/sub/"manifest.json"
        info=json.loads(manifest.read_text())
        checkpoint=Path(info["checkpoint"])
        require(checkpoint.is_file(),"missing "+str(checkpoint))
        require(sha(checkpoint)==info["checkpoint_sha256"],
                name+" "+stage+" checkpoint checksum changed")
        require(info.get("dataset_c_opened") is False and
                info.get("external_data_opened") is False,
                name+" "+stage+" external access assertion missing")
        out[stage]={"sha256":info["checkpoint_sha256"],
                    "training_n":info.get("training_n"),
                    "selected_epoch":info.get("selected_epoch",info.get("refit_epochs"))}
    require(out["B"]["selected_epoch"]==out["C"]["selected_epoch"],
            name+" B/C selected epoch mismatch")
    for cohort,expected in (("09B",14687),("10B_DEV_FROZEN",18543)):
        path=root/folder/"evaluation"/cohort/"manifest.json"
        data=json.loads(path.read_text())
        p=root/folder/"evaluation"/cohort/"scores_sota_aligned.npz"
        require(sha(p)==data["score_sha256"],name+" "+cohort+" score hash changed")
        with np.load(p,allow_pickle=False) as z:
            prob=np.asarray(z["sota_probs"],dtype=np.float32)
        require(prob.shape==(expected,65),"scores invalid "+name+cohort)
        require(np.isfinite(prob).all(),"nonfinite probability")
        out[cohort]={"n":expected,
                     "predicted_classes":int(np.unique(prob.argmax(1)).size),
                     "majority_fraction":float(np.bincount(prob.argmax(1),minlength=65).max()/len(prob)),
                     "score_sha256":data["score_sha256"]}
    return out

def integration(name,repo,base,timeout,keras_python):
    script=repo/"scripts/sota"/("s3_s4_s8_authors_keras_v1.py" if name in
                         ("DF","VarCNN","TikTok","RoFiRe") else
                         "s11_swallow_authors_v1.py" if name=="Swallow" else
                         "s12_wfcat_authors_v1.py" if name=="WFCAT" else
                         "s9_netclr_authors_pretrain_b_v1.py")
    python=(str(keras_python) if name in ("DF","VarCNN","TikTok","RoFiRe") else sys.executable)
    if name in ("DF","VarCNN","TikTok","RoFiRe"):
        require(keras_python.is_file() and os.access(keras_python,os.X_OK),
                "AUTHORS_KERAS_RUNTIME_MISSING: "+str(keras_python))
    args=([python,str(script),"--model",name,"--stage","integration-smoke"]
          if name in ("DF","VarCNN","TikTok","RoFiRe") else
          [sys.executable,str(script),"--stage",
           "smoke" if name=="NetCLR" else "integration-smoke"])
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="-1",TF_CPP_MIN_LOG_LEVEL="3",
             OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2",
             TF_NUM_INTEROP_THREADS="2",TF_NUM_INTRAOP_THREADS="2",
             LTD_SOTA_ARTIFACT_ROOT=str(base/"artifacts/sota"),
             LTD_SOTA_OPS_ROOT=str(base/"ops/sota"))
    p=subprocess.run(args,cwd=repo,env=env,text=True,
                     stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                     timeout=timeout,check=False)
    require(p.returncode==0,
            name+" integration exit="+str(p.returncode)+" stdout="+p.stdout[-2500:]+" stderr="+p.stderr[-1200:])
    markers={
      "DF":"AUTHOR_KERAS_TRAIN_SAVE_RELOAD_CPU_PASS",
      "VarCNN":"AUTHOR_KERAS_TRAIN_SAVE_RELOAD_CPU_PASS",
      "TikTok":"AUTHOR_KERAS_TRAIN_SAVE_RELOAD_CPU_PASS",
      "RoFiRe":"AUTHOR_KERAS_TRAIN_SAVE_RELOAD_CPU_PASS",
      "NetCLR":"S9_AUTHOR_B_CPU_SMOKE_PASS",
      "WFCAT":"S12_WFCAT_AUTHORS_INTEGRATION_CPU_PASS",
      "Swallow":"S11_SWALLOW_AUTHORS_INTEGRATION_CPU_PASS"}
    require(markers[name] in p.stdout,name+" integration marker absent")
    return {"command":args[1:],"marker":markers[name],
            "stdout_tail":p.stdout[-1800:]}

def wfcat_feature_pair(repo,base):
    # Run in a subprocess independently from Swallow's package "models".
    code=r'''
import importlib.util,json,sys,numpy as np
from pathlib import Path
repo=Path(sys.argv[1]);base=Path(sys.argv[2]);sys.path.insert(0,str(repo))
from scripts.sota.s12_wfcat_authors_v1 import create_feature,SOURCE,COMMIT
import subprocess
assert subprocess.check_output(["git","-C",str(SOURCE),"rev-parse","HEAD"],text=True).strip()==COMMIT
src=SOURCE/"src/features.py"
spec=importlib.util.spec_from_file_location("wfcat_features_pinned_ref",src)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
cases={}
for name,t,d in [
 ("monotonic",np.arange(160,dtype=np.float64)*0.013,
    np.where(np.arange(160)%3==0,1,-1)),
 ("regressions",np.array([0.,.01,.015,.013,.04,.050,.049,.07,.09]),
    np.array([1,-1,1,1,-1,-1,1,1,-1])),
 ("ties",np.array([0.,0.,.001,.001,.005,.008,.008]),
    np.array([1,-1,-1,1,1,-1,1]))]:
   got=create_feature(d,t,m.feature_transform)
   ids=np.argsort(t,kind="stable")
   direct=m.feature_transform(np.stack((t[ids],d[ids]),axis=1),
                               "iat",1800,granularity=9,time_window=.044)
   assert got.shape==(18,1800)
   assert np.array_equal(got.astype(np.float64),direct)
   cases[name]={"exact_equal":True,"shape":list(got.shape),
                "sum":float(got.sum()),"n_packets":len(t)}
   assert int(got.sum())==len(t)
print("WFCAT_FUNCTIONAL_PARITY_PASS",json.dumps(cases))
'''
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="-1",
             LTD_SOTA_OPS_ROOT=str(base/"ops/sota"),
             LTD_SOTA_ARTIFACT_ROOT=str(base/"artifacts/sota"))
    r=subprocess.run([sys.executable,"-c",code,str(repo),str(base)],
                     cwd=repo,env=env,capture_output=True,text=True,timeout=180)
    require(r.returncode==0,"WFCAT numerical parity failed "+r.stderr[-2000:])
    require("WFCAT_FUNCTIONAL_PARITY_PASS" in r.stdout,"WFCAT parity marker missing")
    return {"status":"PASS_AUTHOR_FEATURE_NUMERIC_EQUALITY_SYNTHETIC_CASES",
            "stdout":r.stdout.strip()}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--repo",required=True,type=Path)
    p.add_argument("--base",required=True,type=Path)
    p.add_argument("--comparison",required=True,type=Path)
    p.add_argument("--output-dir",required=True,type=Path)
    p.add_argument("--timeout",type=int,default=1500)
    p.add_argument("--keras-python",type=Path,
        default=Path(os.environ.get("LTD_SOTA_KERAS_PY",
                                  "/data/ltd-sota-bsierra/venv-keras-authors/bin/python")),
        help="Existing pinned Keras/TensorFlow interpreter used for DF/VarCNN/TikTok/RoFiRe. Never install or alter it.")
    a=p.parse_args()
    require(not a.output_dir.exists(),"output directory already exists")
    a.output_dir.mkdir(parents=True)
    root=a.base/"artifacts/sota"
    comparison=json.loads(a.comparison.read_text())
    require(comparison.get("classification")==
            "LTD_SEVEN_SOTA_REUSE_CANONICAL_S5_S6_ALIGNMENT_V1",
            "existing comparison must pass first")
    report={"classification":"LTD_SEVEN_ORIGINAL_SOURCE_FUNCTIONAL_FIDELITY_V1",
            "source_comparison_sha256":sha(a.comparison),
            "timestamp_utc":__import__("datetime").datetime.now(
              __import__("datetime").timezone.utc).isoformat(),
            "models":{},"supplementary":{}}
    for model in SOURCE_STATUS:
        print("SOURCE_FIDELITY_BEGIN",model,flush=True)
        record={"declared_scope":SOURCE_STATUS[model]["scope"],
                "paper_exact":False}
        try:
            record["artifact_integrity"]=canonical_results(root,model)
            record["integration_smoke"]=integration(model,a.repo,a.base,a.timeout,a.keras_python)
            learned=record["artifact_integrity"]["09B"]["predicted_classes"]
            if model=="Swallow":
                require(learned==1,"Swallow collapse changed; verify new evidence")
                record["status"]="FAIL_COLLAPSED_BYOL_REQUIRES_NEW_RUN"
            else:
                require(learned>=50,
                        model+" insufficient predicted class diversity on 09B")
                record["status"]="PASS_AUTHOR_SOURCE_ADAPTED_FUNCTIONAL_SMOKE"
        except Exception as e:
            record["status"]="FAIL_OR_PENDING_FUNCTIONAL_INTEGRITY"
            record["error"]=str(e)
        report["models"][model]=record
        print("SOURCE_FIDELITY_STATUS",model,record["status"],flush=True)
        (a.output_dir/"progress.json").write_text(json.dumps(report,indent=2)+"\n")
    try:
        report["supplementary"]["WFCAT_raw_author_IAT_numerical_parity"]=wfcat_feature_pair(a.repo,a.base)
    except Exception as exc:
        report["supplementary"]["WFCAT_raw_author_IAT_numerical_parity"]={
            "status":"FAIL","error":str(exc)}
    report["scientific_scope"]={
        "status":"NOT_PAPER_EXACT_CERTIFIED",
        "comparable_as":"frozen historical-protocol author-source adapted baseline implementations; Swallow excluded as failed run",
        "requires_for_full_author_method_equivalence":[
          "Independent end-to-end gradient and preprocessor parity on real captures for each author method",
          "Keras legacy vs modern optimizer/hyperparameter parity where changes were introduced",
          "NetCLR notebook original augmentation and contrastive objective external reference outputs",
          "RoFiRe original DyWin multiplier and data sampling adaptation effect",
          "Original complete VarCNN ensemble evaluation (current baseline direction-only)",
          "Swallow new noncollapsed BYOL training with justified original target-mode/augmentation correction",
          "WFCAT acquisition timestamp units and temporal drift interpretation",
          "Multi-seed sensitivity or clear single-seed limitation statement",
        ],
        "existing_09B_futureB_paired_statistics":"Reusable as frozen retrospective evidence; not an author's benchmark result reproduction."}
    report["status"]=(
      "FUNCTIONAL_SOURCE_GATES_COMPLETE_WITH_SWALLOW_FAILURE"
      if all(v["status"] in ("PASS_AUTHOR_SOURCE_ADAPTED_FUNCTIONAL_SMOKE",
                            "FAIL_COLLAPSED_BYOL_REQUIRES_NEW_RUN")
             for v in report["models"].values())
         and report["supplementary"]["WFCAT_raw_author_IAT_numerical_parity"]["status"].startswith("PASS")
      else "FUNCTIONAL_SOURCE_GATES_INCOMPLETE")
    (a.output_dir/"certificate.json").write_text(json.dumps(report,indent=2)+"\n")
    print("SOURCE_FIDELITY_SUITE_STATUS",report["status"])
    print("SOURCE_FIDELITY_CERTIFICATE",a.output_dir/"certificate.json")
if __name__=="__main__":main()
