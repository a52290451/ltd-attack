#!/usr/bin/env python3
"""Bulk, CPU-only end-to-end evidence sweep over ALL SEVEN frozen SOTA methods.

Consumes the PASS author functional certificate, S5/S6 paired comparison,
existing Swallow trajectory/temporal reports, original sources and all B/C
manifests. Never retrains or modifies original artifacts. Hard distinguishes
replicated authors source, adapted methodology and paper-exact equivalence.
"""
from __future__ import annotations
import argparse,ast,csv,hashlib,json,os,subprocess,sys,traceback
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

NAME_MAP={"DF":"SOTA_DF","VarCNN":"SOTA_VarCNN_direction","TikTok":"SOTA_TikTok",
          "RoFiRe":"SOTA_RoFiRe","NetCLR":"SOTA_NetCLR","WFCAT":"SOTA_WFCAT",
          "Swallow":"SOTA_Swallow_COLLAPSED"}
PATHS={
 "DF":("authors_df_seed42","B","C"),
 "VarCNN":("authors_varcnn_seed42","B","C"),
 "TikTok":("authors_tiktok_seed42","B","C"),
 "RoFiRe":("authors_rofire_seed42","B","C"),
 "NetCLR":("s9_netclr_author_original_v1","supervised_B_v1","refit_C_v1"),
 "WFCAT":("s12_wfcat_authors_iat_seed42","B","C"),
 "Swallow":("s11_swallow_authors_seed42","supervised_B","supervised_C")}
DECLARED_MATERIAL={
 "DF":["Keras compatibility and 65-way replacement head; historical chronological selection"],
 "VarCNN":["Only direction branch of ResNet18; published multimodal/metadata ensemble NOT reproduced"],
 "TikTok":["65-site classifier, adapted DT representation and training configuration"],
 "RoFiRe":["DyWin multiplier 10 rather than 200; 5 original samples/class for augmentation"],
 "NetCLR":["100 vs upstream notebook parameter 401 epochs; batch64 vs256; outgoing burst CDF corrected"],
 "WFCAT":["30 vs author 50 epochs; batch16 vs64; timestamp semantics under temporal dataset shift"],
 "Swallow":["100 vs author 200 epochs; batch32 vs512; original augmentation multiplier6 not applied; target network set eval(); observed collapsed representation"]}

def require(ok,why):
    if not ok:raise RuntimeError(why)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()
def j(path):return json.loads(Path(path).read_text())
def subprocess_audit(py,repo,args):
    p=subprocess.run([str(py),*map(str,args)],cwd=repo,
       env=dict(os.environ,CUDA_VISIBLE_DEVICES="-1",OMP_NUM_THREADS="2",
                MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2"),
       capture_output=True,text=True,timeout=1800,check=False)
    require(p.returncode==0,
         "audit subprocess rc="+str(p.returncode)+" stderr="+p.stderr[-1400:]+" stdout="+p.stdout[-700:])
    return p.stdout[-1700:]
def stage_integrity(root,model):
    folder,b,c=PATHS[model]
    outputs={}
    for tag,stage in (("B",b),("C",c)):
        mpath=root/folder/stage/"manifest.json"
        m=j(mpath)
        ck=Path(m["checkpoint"])
        require(ck.is_file() and sha(ck)==m["checkpoint_sha256"],
                model+" "+tag+" checkpoint integrity")
        require(m.get("external_data_opened") is False and
                m.get("dataset_c_opened") is False,
                model+" "+tag+" external access declaration")
        expected=39258 if tag=="B" else 57916
        # Original author adapters use either training_n (Keras, NetCLR,
        # Swallow) or train_n (WFCAT). Never silently default missing counts.
        declared={k:int(m[k]) for k in ("training_n","train_n") if k in m}
        require(declared and len(set(declared.values()))==1,
                model+" "+tag+" missing/conflicting training population keys")
        count=next(iter(declared.values()))
        require(count==expected,model+" "+tag+" population: "+repr(declared))
        epoch=m.get("selected_epoch",m.get("refit_epochs"))
        require(type(epoch) is int and 1<=epoch<=100,
                model+" "+tag+" selected epoch missing/invalid")
        if tag=="B":
            history=m.get("history",[])
            require(isinstance(history,list) and history,
                    model+" B validation history missing")
            vals=[(int(item["epoch"]),float(item.get("val_macro_f1",
                           item.get("validation_macro_f1"))))
                  for item in history
                  if isinstance(item,dict) and
                     ("val_macro_f1" in item or "validation_macro_f1" in item)]
            require(vals,model+" B validation macro-F1 history unavailable")
            highest=max(v for _,v in vals)
            earliest=min(e for e,v in vals if v==highest)
            require(epoch==earliest,model+" B selected epoch not first best macro-F1")
        outputs[tag]={"manifest_sha256":sha(mpath),"checkpoint_sha256":sha(ck),
                      "epoch":epoch,"training_n":count,
                      "population_manifest_field":next(iter(declared))}
    require(outputs["B"]["epoch"]==outputs["C"]["epoch"],
            model+" B/C epoch mismatch")
    return outputs
def original_code_extraction(repo,base):
    sys.path.insert(0,str(repo))
    from scripts.sota.s3_s4_s8_authors_keras_v1 import REGISTRY
    from scripts.sota.sota_paths_v1 import ORIGINAL,MODERN
    result={}
    for model,x in REGISTRY.items():
        path=(MODERN if model=="RoFiRe" else ORIGINAL)/x["repo"]/x["path"]
        require(path.is_file(),"author source missing "+model)
        parent=path.parent
        commit=x["commit"]
        actual=subprocess.check_output(["git","-C",str(parent),"rev-parse","HEAD"],text=True).strip()
        require(actual==commit,"author source not pinned "+model)
        tree=ast.parse(path.read_text(encoding="utf-8-sig"))
        author_nodes={n.name:n for n in tree.body
                      if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
        require(set(x["names"])<=set(author_nodes),"missing author functions "+model)
        result[model]={"source_file":str(path),"commit":actual,
                       "source_sha256":sha(path),"names":x["names"],
                       "author_ast_present":True,
                       "exact_full_paper":False}
    return result
def real_wfcat_pairs(repo,base,report,n_samples):
    code=r'''
import csv,hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
repo=Path(sys.argv[1]);base=Path(sys.argv[2]);dest=Path(sys.argv[3])
max_per_cohort=int(sys.argv[4])
sys.path.insert(0,str(repo))
from scripts.sota.sota_paths_v1 import S1,FUTURE,STORAGE
from scripts.sota.s12_wfcat_authors_v1 import create_feature,SOURCE
from src.models.sota.v1.data_adapter import parse_direction
from src.models.sota.v1.timing_adapter import parse_times
spec=importlib.util.spec_from_file_location("wfcat_orig_feat",SOURCE/"src/features.py")
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
root=base/"artifacts/sota/s12_wfcat_authors_iat_seed42/evaluation"
cohorts={"09B":STORAGE/"historical/CLEAN_final_vectors_sites.csv",
         "10B_DEV_FROZEN":STORAGE/"future/CLEAN_final_vectors_sites_concept_drift.csv"}
results={}
for cohort,source in cohorts.items():
    with np.load(root/cohort/"scores_sota_aligned.npz",allow_pickle=False) as z:
        ids=z["pcap_uid"].astype(str)
        labels=z["candidate_labels"].astype(str)
        true=z["y_true"].astype(int)
    want={uid:labels[k] for uid,k in zip(ids[:max_per_cohort],true[:max_per_cohort])}
    visited=set();mismatch=0;cases=[]
    csv.field_size_limit(sys.maxsize)
    with source.open(newline="",encoding="utf-8") as stream:
        rd=csv.DictReader(stream)
        for row in rd:
            uid=row["pcap_uid"]
            if uid not in want:continue
            assert uid not in visited and row["site_label"]==want[uid]
            visited.add(uid)
            d=np.asarray(parse_direction(row["direction_vector"]),dtype=np.float64)
            t=np.asarray(parse_times(row["time_vector"]),dtype=np.float64)
            assert len(d)==len(t)>1
            got=create_feature(d,t,mod.feature_transform)
            sort=np.argsort(t,kind="stable")
            raw=mod.feature_transform(np.stack((t[sort],d[sort]),axis=1),
                  "iat",1800,granularity=9,time_window=.044)
            eq=np.array_equal(got.astype(np.float64),raw)
            mismatch+=int(not eq)
            assert eq and int(got.sum())==len(d)
            cases.append({"packets":len(d),"max_timestamp":float(t.max()),
                          "regression_count":int(np.count_nonzero(np.diff(t)<0)),
                          "feature_sum":int(got.sum())})
            if len(visited)==len(want):break
    assert len(visited)==len(want),cohort+" missing sampled UID"
    results[cohort]={"n":len(visited),"author_adapter_numeric_mismatches":mismatch,
                     "with_timestamp_regression":sum(c["regression_count"]>0 for c in cases),
                     "median_packets":float(np.median([c["packets"] for c in cases]))}
assert not dest.exists()
dest.write_text(json.dumps({"classification":"WFCAT_REAL_FROZEN_SOURCE_FEATURE_PARITY",
 "results":results,"limits":["Paired inputs are real but sampled, not full population",
 "Source timestamp units still require acquisition metadata corroboration"]},indent=2)+"\n")
print("WFCAT_REAL_SOURCE_PARITY_PASS",json.dumps(results))
'''
    return subprocess_audit(sys.executable,repo,["-c",code,repo,base,report,n_samples])
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--repo",required=True,type=Path)
    p.add_argument("--base",required=True,type=Path)
    p.add_argument("--functional-certificate",required=True,type=Path)
    p.add_argument("--comparison",required=True,type=Path)
    p.add_argument("--output-dir",required=True,type=Path)
    p.add_argument("--wfcat-samples",type=int,default=48)
    a=p.parse_args()
    require(not a.output_dir.exists(),"output directory already exists")
    a.output_dir.mkdir(parents=True)
    report={"classification":"LTD_SEVEN_SOTA_BULK_SCIENTIFIC_EVIDENCE_V1",
            "created_utc":datetime.now(timezone.utc).isoformat(),
            "comparison_sha256":sha(a.comparison),
            "functional_certificate_sha256":sha(a.functional_certificate),
            "steps":{},"models":{}}
    def step(name,func):
        try:
            data=func()
            report["steps"][name]={"status":"PASS","evidence":data}
            print("BULK_SOTA_GATE_PASS",name,flush=True)
        except Exception as exc:
            report["steps"][name]={"status":"FAIL","error":str(exc),
                "traceback":traceback.format_exc()[-2000:]}
            print("BULK_SOTA_GATE_FAIL",name,str(exc)[:450],flush=True)
        (a.output_dir/"progress.json").write_text(json.dumps(report,indent=2)+"\n")
    root=a.base/"artifacts/sota"
    functional=j(a.functional_certificate)
    comparison=j(a.comparison)
    require(functional.get("classification")==
            "LTD_SEVEN_ORIGINAL_SOURCE_FUNCTIONAL_FIDELITY_V1",
            "incorrect original author functional certificate")
    require(comparison.get("classification")==
            "LTD_SEVEN_SOTA_REUSE_CANONICAL_S5_S6_ALIGNMENT_V1",
            "incorrect aligned S5/S6 comparison")
    require(set(functional["models"])==set(NAME_MAP),
            "seven-model functional certificate incomplete")
    for model in NAME_MAP:
        x=functional["models"][model]
        passed=(x["status"]=="PASS_AUTHOR_SOURCE_ADAPTED_FUNCTIONAL_SMOKE")
        failed=(model=="Swallow" and x["status"]==
                "FAIL_COLLAPSED_BYOL_REQUIRES_NEW_RUN")
        report["models"][model]={"frozen_functional_status":x["status"],
           "status":"PASS_PREVIOUS_AUTHOR_SOURCE_ADAPTED_SMOKE" if passed else
                    "FAIL_KNOWN_COLLAPSE" if failed else "INCOMPLETE",
           "methodological_deviations":DECLARED_MATERIAL[model],
           "paper_exact_equivalence":False}
    step("source_original_Keras_four_AST_and_git_pins",
         lambda:original_code_extraction(a.repo,a.base))
    step("all_seven_BC_checkpoints_and_historical_only",
         lambda:{m:stage_integrity(root,m) for m in NAME_MAP})
    def check_previous():
        require(all(m["status"] in ("PASS_PREVIOUS_AUTHOR_SOURCE_ADAPTED_SMOKE",
                                    "FAIL_KNOWN_COLLAPSE")
                 for m in report["models"].values()),
                "an author functional smoke did not pass")
        return {m:x["status"] for m,x in report["models"].items()}
    step("previous_seven_isolated_author_functional_smokes",check_previous)
    def validate_comparison():
        for cohort,n in (("09B",14687),("10B_DEV_FROZEN",18543)):
            item=comparison["cohorts"][cohort]
            require(item["n"]==n and len(item["models"])==10 and
                    len(item["paired_comparisons"])==21,
                    "ten model comparison missing "+cohort)
            for model in NAME_MAP:
                k=NAME_MAP[model]
                require(k in item["models"],"missing aligned SOTA "+k)
                value=item["models"][k]["metrics"]["accuracy"]
                require(0<=value<=1,"invalid accuracy "+k)
        return {"09B_n":14687,"FutureB_n":18543,
                "models_aligned":10,"paired_comparisons_total":42}
    step("canonical_S5S6_original_LTD_vs_seven_prediction_alignment",
         validate_comparison)
    step("WFCAT_real_Historical_Future_original_feature_numeric_parity",
         lambda:real_wfcat_pairs(a.repo,a.base,
                   a.output_dir/"wfcat_real_feature_parity.json",a.wfcat_samples))
    def temporal():
        # Reuse the complete previous population scan rather than read 33k CSVs again.
        files=sorted((a.base/"audits").glob("certification_*/full_temporal_population.json"))
        if not files:
            raise RuntimeError("no previous full-population temporal report; run full certification suite")
        prev=j(files[-1])
        require(prev.get("classification")==
                "LTD_FULL_EXTERNAL_TEMPORAL_SEMANTICS_AUDIT_V1",
                "wrong full-temporal report")
        require(prev["cohorts"]["09B"]["n"]==14687 and
                prev["cohorts"]["10B_DEV_FROZEN"]["n"]==18543,
                "population counts mismatch")
        return {"source_report":str(files[-1]),"sha256":sha(files[-1]),
                "units_verified_from_acquisition":"PENDING"}
    step("reused_complete_temporal_population_with_acquisition_caveat",temporal)
    def swallow():
        files=sorted((a.base/"audits").glob("certification_*/swallow_trajectory.json"))
        require(bool(files),"no original BYOL trajectory output")
        prev=j(files[-1])
        epochs={v["epoch"]:v for v in prev["checkpoint_epochs"]}
        require(20 in epochs and 40 in epochs,"missing BYOL 20/40")
        require(epochs[20]["unique_rounded_6"]>1 and
                epochs[40]["unique_rounded_6"]==1,
                "BYOL collapse 20-40 not reproduced")
        return {"sha256":sha(files[-1]),"source":str(files[-1]),
                "collapse_window_epochs":"20-40",
                "repair":"NEW experiment only; do not overwrite baseline"}
    step("Swallow_original_frozen_BYOL_collapse_root_narrowing",swallow)
    try:
        m=sorted((a.base/"audits").glob("certification_*/upstream_training_parity.json"))
        require(bool(m),"missing upstream-parity source report")
        doc=j(m[-1])
        require(doc["classification"]=="LTD_PINNED_UPSTREAM_TRAINING_PARITY_V1",
                "incorrect author-source parity")
        report["steps"]["original_method_training_parameter_deviations"]={
            "status":"PASS","evidence":{"source":str(m[-1]),"sha256":sha(m[-1]),
                     "Swallow":doc["Swallow"]["checked"],
                     "WFCAT":doc["WFCAT"]["checked"]}}
    except Exception as exc:
        report["steps"]["original_method_training_parameter_deviations"]={
           "status":"FAIL","error":str(exc)}
    print("BULK_SOTA_GATE", "original_method_training_parameter_deviations",
          report["steps"]["original_method_training_parameter_deviations"]["status"],flush=True)
    report["summary"]={
      "usable_frozen_adapted_baselines":[m for m in NAME_MAP if m!="Swallow" and
          report["models"][m]["status"]=="PASS_PREVIOUS_AUTHOR_SOURCE_ADAPTED_SMOKE"],
      "excluded_baseline":"Swallow: collapsed BYOL",
      "method_fidelity":"AUTHOR_SOURCE_ADAPTED_NOT_ORIGINAL_PAPER_EQUIVALENT",
      "article_claim_allowed":"Our frozen LTD-Hybrid scored higher accuracy than six specified author-source adapted implementations on the retrospective 09B and Future-B cohorts, subject to paired date CIs.",
      "article_claim_disallowed":"Our method definitively outperforms six faithfully reproduced original published SOTA methods.",
      "remaining_critical":[
       "Swallow: independently design/test corrected BYOL protocol on Historical DEV and version a new baseline.",
       "VarCNN: direction branch is not full multimodal published ensemble.",
       "RoFiRe: DyWin 10x vs author 200x needs disclosed sensitivity or new version for author equivalence.",
       "NetCLR: changed CDF, pretrain epochs/batch; source notebook numerical equivalence not demonstrated.",
       "All methods: exact optimizer/loss/hyperparameter differences documented rather than treated as paper-exact.",
       "Multi-seed sensitivity / single-seed limitation; acquisition timestamp unit provenance.",
       "Future-B 65-class cohort may historically be conditioned on external availability; not fully blind."],
    }
    report["status"]=("BULK_EVIDENCE_GATES_PASS_WITH_EXPLICIT_METHOD_LIMITATIONS"
      if all(v["status"]=="PASS" for v in report["steps"].values())
      else "BULK_EVIDENCE_HAS_FAILED_OR_MISSING_GATES")
    (a.output_dir/"certificate.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print("BULK_SOTA_SCIENTIFIC_STATUS",report["status"])
    for k,v in report["steps"].items():
        print("GATE",k,v["status"])
    print("SIX_ADAPTED_BASELINES",report["summary"]["usable_frozen_adapted_baselines"])
    print("FINAL_CERTIFICATE",a.output_dir/"certificate.json")
if __name__=="__main__":main()
