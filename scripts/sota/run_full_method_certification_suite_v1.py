#!/usr/bin/env python3
"""One-shot read-only LTD 7-SOTA methodological certification suite.

Runs all prior independent probes, verifies B/C hyperparameter/selection
contracts, upstream pinning, Historical-vs-external UID disjointness, and issues
a conservative evidence-based certificate. It NEVER retrains or overwrites
original experiments. New evidence is written ONLY under --audit-root.
A fully successful run can STILL end NOT_CERTIFIED if original paper fidelity
or the existing Swallow failure remains unresolved.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,os,subprocess,sys,traceback
from datetime import datetime,timezone
from pathlib import Path

MODELS={
 "NetCLR":("s9_netclr_author_original_v1","supervised_B_v1","refit_C_v1"),
 "Swallow":("s11_swallow_authors_seed42","supervised_B","supervised_C"),
 "WFCAT":("s12_wfcat_authors_iat_seed42","B","C"),
 "DF":("authors_df_seed42","B","C"),
 "VarCNN_direction":("authors_varcnn_seed42","B","C"),
 "TikTok":("authors_tiktok_seed42","B","C"),
 "RoFiRe":("authors_rofire_seed42","B","C")}
SOURCE_REPOS={
 "Swallow":("upstream_2026/S11_Swallow","c6dc304bca6e61b10b36b00178ee84e38c17961b",
             ["CIF.py","models/resnet_base_network.py","loader/dataAugmentation.py"]),
 "WFCAT":("upstream_2026/S12_WFCAT","cd382d30f7d33714fa4d025cddb3daaf2ffb0f09",
           ["src/features.py","src/models/wfcat.py"]),
 "DF":("upstream_original_2026/S3_DF_Authors","38df0c15a089f13228e4df06bd5269c8a37340fa",["src/Model_NoDef.py"]),
 "VarCNN_direction":("upstream_original_2026/S4_VarCNN_Authors","e5db76a86fcbf8839f764b9aca241749d1c1f700",["var_cnn.py"]),
 "TikTok":("upstream_original_2026/S8_TikTok_Authors","0aa56660b435d88f3542939b3f08c0ba1409dbf5",["DL_Experiments/cw_attack.py"]),
 "RoFiRe":("upstream_2026/S10_RoFiRe","7ef45ee7c9af9ac04048456992d3d96398791bbc",["model/DF.py","feature/features.py"])}
DECLARED_DEVIATIONS={
 "NetCLR":["pretraining batch/epoch protocol adapted; burst-CDF correction; classifier 65 classes; original notebook not paper exact"],
 "Swallow":["pretrain 100 vs original 200 epochs; batch 32 vs 512; original aggregation loop patched in memory; BYOL collapsed in frozen run"],
 "WFCAT":["IAT 9 bins, 1800 steps and 0.044s window need author configuration parity; temporal unit equivalence unproven"],
 "DF":["65-way head, Keras modernized, historical B/C selection; not original study's evaluation protocol"],
 "VarCNN_direction":["ONLY direction branch, not complete published multibranch/metadata ensemble"],
 "TikTok":["DT representation adapter, Keras modernized, 65-way head and historical B/C"],
 "RoFiRe":["DyWin augmentation x10 vs source default x200; first 5 chronological samples per class; historical B/C"]}
EXPECTED_COUNTS={0:19098,1:20160,2:18658}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
    return h.hexdigest()
def jread(path):
    return json.loads(Path(path).read_text())
def must(b,msg):
    if not b:raise RuntimeError(msg)
def cmd(command,repo,extra_env=None):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="-1",OMP_NUM_THREADS="2",
             OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2")
    if extra_env:env.update(extra_env)
    p=subprocess.run(command,cwd=repo,env=env,text=True,
                     capture_output=True,check=False)
    if p.returncode:
        raise RuntimeError("returncode="+str(p.returncode)+" stderr="+p.stderr[-2000:]+
                           " stdout="+p.stdout[-1000:])
    return p.stdout.strip()
def repo_pins(ops):
    out={}
    for name,(rel,revision,files) in SOURCE_REPOS.items():
        p=ops/rel
        try:
            actual=subprocess.run(["git","-C",str(p),"rev-parse","HEAD"],
                capture_output=True,text=True,check=True).stdout.strip()
            dirty=subprocess.run(["git","-C",str(p),"status","--porcelain",
                "--untracked-files=no"],capture_output=True,text=True,check=True).stdout.strip()
            evidence={file:sha(p/file) for file in files}
            must(actual==revision and not dirty,
                 "source commit changed or tracked files dirty")
            out[name]={"status":"PASS_PINNED_SOURCE_NOT_PAPER_EQUIVALENCE",
                        "commit":actual,"source_files_sha256":evidence}
        except Exception as e:
            out[name]={"status":"FAIL","error":str(e)}
    out["NetCLR"]={"status":"PENDING_MANUAL_NOTEBOOK_MODEL_AND_AUGMENT_PARITY",
                   "reason":"Notebook AST-based import; source commit/architecture must be separately independently checked"}
    return out
def stage_checks(root):
    out={}
    for name,(rel,b,c) in MODELS.items():
        folder=root/rel
        try:
            bm=jread(folder/b/"manifest.json")
            cm=jread(folder/c/"manifest.json")
            for stage,m in (("B",bm),("C",cm)):
                fp=Path(m["checkpoint"])
                must(fp.is_file() and sha(fp)==m["checkpoint_sha256"],
                     name+" "+stage+" checkpoint mismatch")
                must(m.get("dataset_c_opened") is False,
                     name+" "+stage+" data C exclusion assertion absent")
                must(m.get("external_data_opened") is False,
                     name+" "+stage+" external exclusion assertion absent")
            be=bm["selected_epoch"]
            ce=cm.get("selected_epoch",cm.get("refit_epochs"))
            must(isinstance(be,int) and 1<=be<=100 and ce==be,
                 name+" B/C selected epochs inconsistent")
            must(int(bm.get("training_n",bm.get("train_n",0)))==39258,
                 name+" B population inconsistent")
            must(int(cm.get("training_n",cm.get("train_n",0)))==57916,
                 name+" C population inconsistent")
            history=bm.get("history",[])
            scored=[r for r in history if "val_macro_f1" in r or "validation_macro_f1" in r]
            if scored:
                def f1(r):return r.get("val_macro_f1",r.get("validation_macro_f1"))
                best=max(f1(r) for r in scored)
                first=min(int(r["epoch"]) for r in scored if f1(r)==best)
                must(be==first,name+" best epoch not selected")
            out[name]={"status":"PASS_B_C_STAGE_CONTRACT_ONLY",
                       "B_epoch":be,"C_epoch":ce,
                       "B_checkpoint_sha256":bm["checkpoint_sha256"],
                       "C_checkpoint_sha256":cm["checkpoint_sha256"],
                       "dev_late_best_macro_f1":best if scored else None,
                       "deviations":DECLARED_DEVIATIONS[name],
                       "paper_exact":False}
        except Exception as e:
            out[name]={"status":"FAIL","error":str(e),
                       "deviations":DECLARED_DEVIATIONS[name]}
    return out
def uid_disjoint(ops,root):
    import numpy as np
    manifest_path=Path(os.environ.get("LTD_SOTA_S1_MANIFEST",
                          ops/"s1_dev_frozen_20261102T130232Z/s1_manifest.json"))
    if not manifest_path.is_file():
        candidates=list(ops.glob("s1_dev_frozen_*/s1_manifest.json"))
        must(len(candidates)==1,"S1 manifest not uniquely located")
        manifest_path=candidates[0]
    s1=jread(manifest_path)
    dev_idx=Path(s1["dev_index_file"])
    must(sha(dev_idx)==s1["dev_index_sha256"],"S1 index checksum mismatch")
    with dev_idx.open(newline="",encoding="utf-8") as stream:
        dev={r["pcap_uid"] for r in csv.DictReader(stream)}
    must(len(dev)==57916,"DEV index UID duplicates")
    external={}
    for cohort,size in (("09B",14687),("10B_DEV_FROZEN",18543)):
        path=root/"s9_netclr_author_original_v1/evaluation"/cohort/"scores_sota_aligned.npz"
        with np.load(path,allow_pickle=False) as z:
            ids=z["pcap_uid"].astype(str)
        must(len(ids)==size and len(set(ids))==size,"external UIDs invalid "+cohort)
        external[cohort]=set(ids)
        must(not dev.intersection(external[cohort]),
             "Historical DEV vs "+cohort+" UID overlap")
    # 09B is from historical vectors but distinct from DEV; Future must be disjoint.
    must(not external["09B"].intersection(external["10B_DEV_FROZEN"]),
         "external cohort UID overlap")
    return {"status":"PASS_UID_DISJOINT_ALL_COHORTS",
            "DEV":len(dev),"09B":len(external["09B"]),
            "Future_B":len(external["10B_DEV_FROZEN"])}
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--repo",type=Path,default=Path.home()/"ltd-attack")
    p.add_argument("--base",type=Path,default=Path("/data/datasets_brayan/ltd-sota-2026"))
    p.add_argument("--audit-root",type=Path,required=True)
    p.add_argument("--run-id",default=None)
    a=p.parse_args()
    stamp=a.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest=a.audit_root/("certification_"+stamp)
    must(not dest.exists(),"audit output exists: "+str(dest))
    dest.mkdir(parents=True)
    root=a.base/"artifacts/sota"
    ops=Path(os.environ.get("LTD_SOTA_OPS_ROOT",str(a.base/"ops/sota")))
    env={"LTD_SOTA_ARTIFACT_ROOT":str(root),"LTD_SOTA_OPS_ROOT":str(ops)}
    out={"classification":"LTD_SOTA_METHOD_CERTIFICATION_SUITE_V1",
         "created_utc":datetime.now(timezone.utc).isoformat(),
         "repo_ref":cmd(["git","rev-parse","HEAD"],a.repo),
         "read_only_original_artifacts":True,"run_dir":str(dest),
         "steps":{},"fidelity":{},"required_corrections":[],
         "statistical_notes":[]}
    def step(label,fn):
        try:
            value=fn()
            out["steps"][label]={"status":"PASS","evidence":value}
            print("SUITE_STEP_PASS",label,flush=True)
            return True
        except Exception as exc:
            out["steps"][label]={"status":"FAIL","error":str(exc),
                                 "traceback":traceback.format_exc()[-3000:]}
            print("SUITE_STEP_FAIL",label,str(exc)[:500],flush=True)
            return False
    py=sys.executable
    def run_script(name,args):
        filename=a.repo/"scripts/sota"/name
        must(filename.is_file(),"missing script "+str(filename))
        return cmd([py,str(filename),*args],a.repo,env)
    forensic=dest/"forensic.json"
    chain=dest/"chain.json"
    cause=dest/"swallow_cause.json"
    trajectory=dest/"swallow_trajectory.json"
    seven=dest/"seven_date_bootstrap.json"
    def strict_sources():
        r=repo_pins(ops)
        must(all(v.get("status","").startswith("PASS") for k,v in r.items()
                 if k!="NetCLR"),"some pinned upstream sources failed: "+
             json.dumps({k:v for k,v in r.items() if v.get("status")=="FAIL"}))
        return r
    def strict_stages():
        r=stage_checks(root)
        must(all(v["status"]=="PASS_B_C_STAGE_CONTRACT_ONLY" for v in r.values()),
             "B/C stage contract invalid: "+
             json.dumps({k:v for k,v in r.items() if v["status"]=="FAIL"}))
        return r
    step("source_repositories_pinned",strict_sources)
    step("frozen_B_C_training_contracts",strict_stages)
    step("historical_external_UID_disjoint",lambda:uid_disjoint(ops,root))
    ok0=step("independent_3model_prediction_integrity",
       lambda:run_script("audit_existing_original_sota_readonly_v1.py",
             ["--artifact-root",str(root),"--output",str(forensic)]))
    if ok0:
        ok1=step("source_CIF_IAT_64_samples_and_Swallow_B_C_probe",
           lambda:run_script("audit_sota_full_chain_readonly_v1.py",
              ["--repo",str(a.repo),"--base",str(a.base),
               "--forensic-json",str(forensic),"--deep-external",
               "--output",str(chain)]))
        if ok1:
            step("Swallow_random_BYOL_and_B_C_representation_probe",
               lambda:run_script("audit_swallow_representation_cause_v1.py",
                ["--repo",str(a.repo),"--base",str(a.base),
                 "--chain-json",str(chain),"--output",str(cause)]))
            step("Swallow_BYOL_epoch_20_40_60_80_100_augmentation_trajectory",
               lambda:run_script("audit_swallow_pretrain_trajectory_v1.py",
                ["--repo",str(a.repo),"--base",str(a.base),
                 "--output",str(trajectory)]))
    step("all_7_prediction_alignment_2500_paired_date_bootstrap",
         lambda:run_script("audit_all_seven_paired_date_bootstrap_v1.py",
              ["--base",str(a.base),"--bootstrap","2500","--output",str(seven)]))
    out["fidelity"]={
        "paper_exact_status":"NOT_DEMONSTRATED",
        "scope":"authors-source-adapted seven-model benchmark; VarCNN direction branch only",
        "controls_not_yet_certified":[
           "functional numerical parity with author reference outputs at shared inputs",
           "independent complete hyperparameter, augmentation and training-logic comparison against upstream",
           "data capture/packet/time units and cross-campaign semantic equivalence",
           "normalization/preprocessing equivalence for every input modality",
           "independent anti-leakage proof beyond manifest assertions",
           "proper collapsed-Swallow remediation in new historical-only protocol"]}
    if cause.is_file():
        try:
            c=jread(cause)
            rnd=c["baseline_random_trunk_excluding_last_child"]["variation_mean_feature_std"]
            pr=c["trained_pretrain_trunk_excluding_last_child"]["variation_mean_feature_std"]
            out["fidelity"]["Swallow"]={
              "random_trunk_mean_std":rnd,"BYOL_checkpoint_mean_std":pr,
              "collapsed":bool(rnd>1e-6 and pr<1e-7),
              "status":"FAIL_COLLAPSED_PRETRAIN" if rnd>1e-6 and pr<1e-7 else "REVIEW"}
        except Exception as e:out["fidelity"]["Swallow"]={"status":"REVIEW","error":str(e)}
    out["required_corrections"]=[
        "Swallow: trace representation collapse through epoch_020/040/060/080/100, independently compare upstream training and augmentation; train NEW version only after root cause established with Historical DEV.",
        "WFCAT: validate IAT timestamp units and feature-shift diagnostics using original reference traces; no adjustment chosen using Future-B.",
        "All 7: independent source-to-adapter architecture/feature/loss/optimizer/augment parity certification and versioned deviation matrix.",
        "No leakage: independently audit cohort source dates and UID, selection paths and checkpoint dependency graph.",
        "Own-model comparison: enforce canonical UID/date/candidate_labels, same scores schema, identical frozen cohorts and preregistered metrics/CI. Do not compare directly to original paper numbers."]
    allok=all(x["status"]=="PASS" for x in out["steps"].values())
    out["status"]="AUDIT_STEPS_PASS_METHODS_NOT_YET_CERTIFIED" if allok else "AUDIT_INCOMPLETE_INVESTIGATE_FAILURES"
    out["suitable_for_paper_now"]=False
    out["statistical_notes"]=[
      "Bootstrap has 11 distinct 09B capture dates and 16 Future-B dates; generalization beyond these observed date clusters is unproven.",
      "Accuracy date-cluster paired CI is descriptive, not universal SOTA superiority.",
      "Source pin and checkpoint SHA prove traceability, not correct reproduction of published method."]
    report=dest/"certificate.json"
    report.write_text(json.dumps(out,indent=2,ensure_ascii=False)+"\n")
    print("SOTA_CERTIFICATION_SUITE_STATUS",out["status"])
    print("SOTA_CERTIFICATION_SUITE_REPORT",report)
    for k,x in out["steps"].items():print("SUITE_GATE",k,x["status"])
    print("SWALLOW_STATUS",out["fidelity"].get("Swallow",{}).get("status","NO_CAUSE_REPORT"))
if __name__=="__main__":
    main()
