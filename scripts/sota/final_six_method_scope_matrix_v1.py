#!/usr/bin/env python3
"""Final author-method scope and publication decision matrix, seven SOTA.

This is a static, source-grounded DECISION document, not a simulation of
paper-exact reproduction. It reuses completed frozen comparisons/certificates,
never reloads train arrays, never retrains, and never changes existing reports.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,os
from pathlib import Path

def must(ok,msg):
    if not ok:raise RuntimeError("FINAL_METHOD_MATRIX_BLOCKED "+msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
MODEL_SPEC={
 "DF":{"benchmark_name":"SOTA_DF","source":"S3_DF_Authors/src/Model_NoDef.py",
       "architecture":"DFNet from pinned authors AST","representation":"direction 5000",
       "adaptations":["65-class head","historical 30-epoch macro-F1 B/C protocol","Keras API and optimizer configuration"],
       "required_claim":"DFNet author-source-adapted"},
 "VarCNN":{"benchmark_name":"SOTA_VarCNN_direction","source":"S4_VarCNN_Authors/var_cnn.py",
       "architecture":"author ResNet18 direction branch only","representation":"direction 5000",
       "adaptations":["published multi-input architecture/ensemble NOT implemented","65-class output","historical B/C"],
       "required_claim":"VarCNN direction-only branch (not complete VarCNN)"},
 "TikTok":{"benchmark_name":"SOTA_TikTok","source":"S8_TikTok_Authors/DL_Experiments/cw_attack.py",
       "architecture":"author ConvNet AST","representation":"directional timing 5000",
       "adaptations":["DT input reconstruction on LTD traces","65-way output","Keras training/chronology adapter"],
       "required_claim":"Tik-Tok author-source-adapted"},
 "RoFiRe":{"benchmark_name":"SOTA_RoFiRe","source":"S10_RoFiRe/model/DF.py + feature/features.py",
       "architecture":"author DFVarNet + DyWin","representation":"directional timing 5000",
       "adaptations":["DyWin 10x rather than author default 200x","5 chronological original samples per class for augmentation","65-way output"],
       "required_claim":"RoFiRe author-source-adapted (DyWin ×10)"},
 "NetCLR":{"benchmark_name":"SOTA_NetCLR","source":"SPIN-UMass source notebook pinned AST",
       "architecture":"author DFsimCLR/DFNet/NetAugment extracted from notebook","representation":"direction 5000",
       "adaptations":["100 vs original notebook 401 epoch parameter","batch64 vs256","corrected outgoing burst CDF pairing","65-way fine-tuning"],
       "required_claim":"NetCLR author-source-adapted"},
 "WFCAT":{"benchmark_name":"SOTA_WFCAT","source":"S12_WFCAT/src/features.py + src/models/wfcat.py",
       "architecture":"original WFCAT class and IAT transform","representation":"18×1800 IAT counts",
       "adaptations":["30 vs author 50 epochs","batch16 vs64","author IAT with LTD timestamps","65-way output"],
       "required_claim":"WFCAT author-source-adapted (IAT)"},
 "Swallow":{"benchmark_name":"SOTA_Swallow_COLLAPSED","source":"S11_Swallow/CIF.py, ResNet, BYOL trainer",
       "architecture":"original CIF and ResNet plus adapted BYOL","representation":"CIF 2×1000",
       "adaptations":["target.eval() differs from author training mode","100 vs200 epochs","batch32 vs512","author 6x augmentation samples not replicated"],
       "required_claim":"Swallow failed adaptation, diagnostic only"}}
def main():
    a=argparse.ArgumentParser()
    a.add_argument("--comparison",type=Path,required=True)
    a.add_argument("--functional-certificate",type=Path,required=True)
    a.add_argument("--reconciled-bc",type=Path,required=True)
    a.add_argument("--output-dir",type=Path,required=True)
    q=a.parse_args()
    must(not q.output_dir.exists(),"output exists; refuse overwrite")
    comp=json.loads(q.comparison.read_text())
    func=json.loads(q.functional_certificate.read_text())
    bc=json.loads(q.reconciled_bc.read_text())
    must(comp["classification"]=="LTD_SEVEN_SOTA_REUSE_CANONICAL_S5_S6_ALIGNMENT_V1",
         "unexpected comparator")
    must(func["status"]=="FUNCTIONAL_SOURCE_GATES_COMPLETE_WITH_SWALLOW_FAILURE",
         "functional gates not complete")
    must(bc["status"]=="PASS_BULK_TECHNICAL_GATES_RECONCILED_WITH_LIMITATIONS",
         "B/C schema repair not approved")
    must(set(bc["model_checks"])==set(MODEL_SPEC),"B/C model inventory mismatch")
    entries=[]
    for model,spec in MODEL_SPEC.items():
        bm=bc["model_checks"][model]
        fm=func["models"][model]
        must(bm["status"]=="PASS","B/C invalid "+model)
        is_valid=model!="Swallow"
        expected=("PASS_AUTHOR_SOURCE_ADAPTED_FUNCTIONAL_SMOKE" if is_valid
                 else "FAIL_COLLAPSED_BYOL_REQUIRES_NEW_RUN")
        must(fm["status"]==expected,"functional status mismatch "+model)
        result={
          "model":model,"label_for_paper":spec["required_claim"],
          "author_reference":spec["source"],"author_architecture":spec["architecture"],
          "input_modality":spec["representation"],
          "declared_method_deviations":spec["adaptations"],
          "historical_selection_B_epoch":bm["evidence"]["B"]["epoch"],
          "historical_refit_C_epochs":bm["evidence"]["C"]["epoch"],
          "decision":("KEEP_AS_FROZEN_ADAPTED_BASELINE" if is_valid
                     else "EXCLUDE_AS_METHOD_COMPARISON_USE_DIAGNOSTIC_ONLY"),
          "paper_exact":False,"functional_original_source_gate":fm["status"],
          "publication_scope":("VALID_FOR_RETROSPECTIVE_ADAPTED_BENCHMARK_ONLY" if is_valid
                               else "INVALID_AS_REPRESENTATIVE_SWALLOW"),
          "relative_accuracy_by_cohort":{}}
        for cohort in ("09B","10B_DEV_FROZEN"):
            c=comp["cohorts"][cohort]
            acc=c["models"][spec["benchmark_name"]]["metrics"]["accuracy"]
            hybrid=c["models"]["LTD_HYBRID"]["metrics"]["accuracy"]
            delta=hybrid-acc
            pair=next(z for z in c["paired_comparisons"]
                      if z["own"]=="LTD_HYBRID" and z["sota"]==spec["benchmark_name"])
            ci=pair["day_bootstrap_ci95"]["accuracy"]
            must(abs(delta-pair["differences_own_minus_SOTA"]["accuracy"])<1e-12,
                 "paired accuracy discrepancy")
            result["relative_accuracy_by_cohort"][cohort]={
                "baseline_accuracy":acc,"hybrid_accuracy":hybrid,
                "hybrid_minus_baseline":delta,"paired_day_bootstrap_ci95":ci,
                "sample_size":c["n"],"capture_days":c["days"]}
        entries.append(result)
    report={"classification":"LTD_SEVEN_SOTA_FINAL_METHOD_SCOPE_MATRIX_V1",
            "evidence_hashes":{"comparison":sha(q.comparison),
                "functional_gate":sha(q.functional_certificate),
                "BC_reconciliation":sha(q.reconciled_bc)},
            "models":entries,
            "status":"SIX_AUTHOR_SOURCE_ADAPTED_BASELINES_USABLE_ONE_COLLAPSED_EXCLUDED",
            "allowed_conclusion":"Under a shared historical-training protocol and frozen retrospective cohorts, LTD Hybrid outperformed the six specified source-adapted benchmark variants in accuracy (day-paired bootstrap intervals).",
            "disallowed_conclusion":"LTD Hybrid is proved superior to the complete original published SOTA methods.",
            "limitations":[
              "Method-specific original-paper numerical equivalence is not proven.",
              "VarCNN is direction branch only and RoFiRe uses reduced DyWin augmentation.",
              "Single-seed original baselines; no seed-variance generality claim.",
              "The 65 monitored sites may have been chosen conditional on future availability.",
              "Swallow BYOL collapsed; original method not evaluated validly.",
              "WFCAT acquisition-time units require original collector provenance evidence.",
              "Date-block bootstrap CIs are conditional on observed 11/16 date clusters."]}
    q.output_dir.mkdir(parents=True)
    (q.output_dir/"method_matrix.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    with (q.output_dir/"method_matrix.csv").open("x",newline="") as f:
        fields=["model","decision","paper_exact","input_modality","historical_selection_B_epoch",
                "historical_refit_C_epochs","author_architecture","label_for_paper",
                "declared_method_deviations"]
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for record in entries:
            w.writerow({k:(" | ".join(record[k]) if isinstance(record.get(k),list)
                    else record.get(k)) for k in fields})
    print("METHOD_MATRIX_STATUS",report["status"])
    for item in entries:print("MODEL_FINAL_DECISION",item["model"],item["decision"])
    print("METHOD_MATRIX_REPORT",q.output_dir/"method_matrix.json")
if __name__=="__main__":main()
