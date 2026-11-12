#!/usr/bin/env python3
"""Static evidence matrix of pinned upstream training code versus LTD adapters.

Not a paper-exact certificate. Read-only, no imports of GPU frameworks.
Includes explicit FAIL_OF_EXACT_FIDELITY when adaptation differs.
"""
from __future__ import annotations
import argparse,json,re,ast,hashlib,subprocess
from pathlib import Path

def check(x,msg):
    if not x:raise RuntimeError("UPSTREAM_PARITY_BLOCKED "+msg)
def hash_file(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo",type=Path,required=True)
    ap.add_argument("--base",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    upstream=a.base/"ops/sota/upstream_2026"
    sw=upstream/"S11_Swallow"
    wf=upstream/"S12_WFCAT"
    src=a.repo/"scripts/sota"
    files={
      "swallow_original_config":sw/"config.yaml",
      "swallow_original_pretrain":sw/"PreTrain.py",
      "swallow_original_trainer":sw/"trainer.py",
      "swallow_adapter":src/"s11_swallow_authors_v1.py",
      "wfcat_original_train":wf/"src/train.py",
      "wfcat_adapter":src/"s12_wfcat_authors_v1.py"}
    for key,path in files.items():check(path.is_file(),key+" missing: "+str(path))
    t={key:path.read_text(encoding="utf-8") for key,path in files.items()}
    # The source has already been pinned by the outer certification suite.
    def match(pattern,source,msg):
        r=re.search(pattern,source)
        check(r is not None,msg+" not found in pinned source")
        return r.group(1)
    def ast_function(text,name):
        tree=ast.parse(text)
        matches=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name]
        check(len(matches)==1,"missing adapter function "+name)
        return ast.get_source_segment(text,matches[0])
    swtrain=ast_function(t["swallow_adapter"],"pretrain")
    swconf=t["swallow_original_config"]
    wforig=t["wfcat_original_train"]
    wftrain=ast_function(t["wfcat_adapter"],"train")
    swcases={
      "pretrain_epochs":{"authors":int(match(r"(?m)^\s*max_epochs:\s*(\d+)",swconf,"max_epochs")),
                        "adapted":100},
      "pretrain_batch_size":{"authors":int(match(r"(?m)^\s*batch_size:\s*(\d+)",swconf,"batch_size")),
                             "adapted":32},
      "augmentation_dataset_multiplier":{"authors":int(match(r"(?m)^\s*multi_times:\s*(\d+)",swconf,"multi_times")),
                                          "adapted":1},
      "target_network_eval_called_in_training":{
        "authors":bool(re.search(r"self\.target_network\.eval\(\)",t["swallow_original_trainer"])),
        "adapted":bool(re.search(r"target\.eval\(\)",swtrain))},
      "target_network_initialization":{
        "authors":"copied weights from online network in BYOLTrainer.initializes_target_network",
        "adapted":"copied weights from online network before pretraining"},
      "data_loader_input_shape":{"authors":"B,1,2,1000","adapted":"B,1,2,1000"}
    }
    wfdefaults={
      "epochs":int(match(r"add_argument\('--epochs',\s*default=(\d+)",wforig,"WFCAT original epochs")),
      "batch_size":int(match(r"add_argument\('-b',\s*'--batch-size',\s*default=(\d+)",wforig,"WFCAT batch size")),
      "seq_length":int(match(r"add_argument\('--seq-length',\s*default=(\d+)",wforig,"WFCAT seq")),
      "iat_bins":int(match(r"add_argument\('--iat-bins'.*default=(\d+)",wforig,"WFCAT bins")),
      "time_window":float(match(r"add_argument\('--time-window'.*default=([0-9.]+)",wforig,"WFCAT time window")),
    }
    wfadapt={"epochs":30,"batch_size":16,"seq_length":1800,"iat_bins":9,"time_window":0.044}
    for key,pair in swcases.items():
        pair["status"]="MATCH" if pair["authors"]==pair["adapted"] else "DIFFERENT"
    wfchecks={name:{"authors":value,"adapted":wfadapt[name],
                    "status":"MATCH" if value==wfadapt[name] else "DIFFERENT"}
              for name,value in wfdefaults.items()}
    need_source_check={
      "swallow_config_sha256":hash_file(files["swallow_original_config"]),
      "swallow_trainer_sha256":hash_file(files["swallow_original_trainer"]),
      "wfcat_train_sha256":hash_file(files["wfcat_original_train"])}
    res={
      "classification":"LTD_PINNED_UPSTREAM_TRAINING_PARITY_V1",
      "source_hashes":need_source_check,
      "Swallow":{"status":"DEVIATES_FROM_UPSTREAM","checked":swcases},
      "WFCAT":{"status":"DEVIATES_FROM_UPSTREAM","checked":wfchecks},
      "all_seven_exact_paper_reproduction":"NOT_CERTIFIED",
      "limitations":[
        "Checks are static and focus on selected training defaults, not the entire original papers.",
        "Different author defaults can be scientifically valid as explicitly labeled adaptations.",
        "Dynamic parity of feature tensors, gradients, model predictions and all source training branches is NOT certified.",
        "No claim that any single difference caused the observed Swallow collapse."]}
    check(not a.output.exists(),"output already exists")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(res,ensure_ascii=False,indent=2)+"\n")
    print("UPSTREAM_TRAINING_PARITY_AUDIT_PASS",a.output)
    for model,checks in (("Swallow",swcases),("WFCAT",wfchecks)):
        for name,entry in checks.items():
            if entry["status"]=="DIFFERENT":
                print("AUTHOR_ADAPTATION_DIFFERENCE",model,name,
                      "authors",entry["authors"],"adapted",entry["adapted"])
if __name__=="__main__":main()
