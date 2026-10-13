#!/usr/bin/env python3
"""READ-ONLY extraction of existing frozen 10B/10C + P15D statistical evidence.
No training, fitting, re-ranking, hyperparameter search or raw Future-B access.
Never overwrite an already-generated paper evidence audit.
"""
from pathlib import Path
import hashlib, json
import pandas as pd
from src.utils.paths import result_path

DATA=Path(result_path("final","FUTURE-B"))
P15D=Path(result_path("diagnostics","P15D-HIST-PAIRED-V1"))
OUT=Path(result_path("reports","PAPER-EVIDENCE-AUDIT-V1"))
NAMES={"summary":"10C_mechanistic_summary.csv","bootstrap":"10C_mechanistic_bootstrap.csv",
       "rescues":"10C_rescue_analysis.csv","per_day":"10C_per_day_ablation.csv",
       "per_class":"10C_per_class_ablation.csv"}
EXPECTED={"accuracy":.063366,"macro_f1":.051259,"top5_accuracy":.081702,"mrr":.069147}

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1048576),b""):h.update(block)
    return h.hexdigest()

def main():
    if OUT.exists():raise RuntimeError("Existing evidence audit output: NO OVERWRITE")
    p10b=DATA/"10B_manifest_future_b.json"
    p10c=DATA/"10C_manifest.json"
    b=json.loads(p10b.read_text())
    c=json.loads(p10c.read_text())
    if b.get("status")!="FUTURE_B_OPENED_AND_EVALUATED":raise RuntimeError("10B NOT FROZEN")
    bp=b.get("protocol",{})
    if bp.get("model_selection_after_future_open") is not False or bp.get("hyperparameter_changes_after_future_open") is not False:
        raise RuntimeError("10B SELECTION_POLICY_INVALID")
    if c.get("status")!="POSTHOC_MECHANISTIC_ANALYSIS" or c.get("scientific_status",{}).get("may_promote_new_model") is not False:
        raise RuntimeError("10C POSTHOC_POLICY_INVALID")
    cp=c.get("data_policy",{})
    if cp.get("new_training") is not False or cp.get("future_csv_read") is not False:
        raise RuntimeError("10C DATA_POLICY_INVALID")
    files={}
    for key,name in NAMES.items():
        digest=sha(DATA/name)
        if c.get("outputs",{}).get(key)!=digest:raise RuntimeError("SHA_MISMATCH "+name)
        files[name]=digest
    df=pd.read_csv(DATA/NAMES["bootstrap"])
    required={"track","comparison","metric","observed_delta","ci95_low","ci95_high","fraction_delta_gt_0"}
    if not required.issubset(df.columns):raise RuntimeError("BOOTSTRAP_SCHEMA_MISMATCH")
    primary=df[(df.track=="DEV_FROZEN")&(df.comparison=="ACTUAL_HYBRID_vs_MICRO")].copy()
    if len(primary)!=4 or set(primary.metric)!=set(EXPECTED) or primary.metric.duplicated().any():
        raise RuntimeError("PRIMARY_COMPARISON_INCOMPLETE")
    for r in primary.itertuples():
        if abs(float(r.observed_delta)-EXPECTED[r.metric])>2e-5:raise RuntimeError("10B_10C_DELTA_MISMATCH "+r.metric)
        if float(r.ci95_low)<=0 or float(r.ci95_high)<=0:raise RuntimeError("POSITIVE_CI_NOT_CONFIRMED "+r.metric)
    m15p=P15D/"P15D_manifest.json"
    d=json.loads(m15p.read_text())
    if d.get("stage")!="P15D_HIST_PAIRED_M0_M1":raise RuntimeError("P15D_INVALID")
    for name,digest in d.get("outputs_sha256",{}).items():
        if sha(P15D/name)!=digest:raise RuntimeError("P15D_SHA_MISMATCH "+name)
    OUT.mkdir(parents=True)
    primary.to_csv(OUT/"paper_primary_metrics.csv",index=False)
    lines=["# LTD-Attack — frozen evidence, no new experiment","",
      "The 10B DEV_FROZEN Future-B Micro-vs-Hybrid result is the original frozen comparison.",
      "The following paired 95% percentile intervals come from EXISTING 10C day-block bootstrap (2000 draws).",
      "10C is a post-hoc diagnostic, NOT a new confirmatory test or model-selection stage.","",
      "| Metric | Observed paired difference | 95% day-block CI | Positive bootstrap fraction |",
      "|---|---:|---:|---:|"]
    for r in primary.sort_values("metric").itertuples():
        unit="" if r.metric=="mrr" else " pp"
        scale=1 if r.metric=="mrr" else 100
        lines.append(f"| {r.metric} | {r.observed_delta*scale:+.5f}{unit} | [{r.ci95_low*scale:+.5f}, {r.ci95_high*scale:+.5f}]{unit} | {r.fraction_delta_gt_0:.3f} |")
    lines += ["","**Caveats:** The bootstrap fraction is not a p-value; the bootstrap block is DATE; Future-B was already opened, and no new model may be selected from these findings.",
      "All component ablations in 10C are post-hoc; XGB/LTD branch-substitution contrasts differ from renormalized deletion contrasts.",
      "The C_BRIDGE_PILOT remains quarantined from the A/B comparison.",""]
    (OUT/"paper_primary_metrics.md").write_text("\n".join(lines))
    manifest={"stage":"PAPER_FROZEN_STATISTICAL_AUDIT","training":False,"future_raw_read":False,"model_selection":False,
      "sources":{str(p10b):sha(p10b),str(p10c):sha(p10c),str(m15p):sha(m15p),**files},
      "outputs_sha256":{name:sha(OUT/name) for name in ["paper_primary_metrics.csv","paper_primary_metrics.md"]}}
    (OUT/"PAPER_EVIDENCE_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True))
    print("PAPER_EVIDENCE_VERIFIED 10B 10C_SHA_5/5 P15D_SHA_"+str(len(d["outputs_sha256"])))
    print(primary[["metric","observed_delta","ci95_low","ci95_high","fraction_delta_gt_0"]].to_string(index=False))
    print("PAPER_EVIDENCE_AUDIT_COMPLETE",OUT)

if __name__=="__main__":
    main()
