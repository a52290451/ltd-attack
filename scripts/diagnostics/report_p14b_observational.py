#!/usr/bin/env python3
"""P14B-OBS: manifest-verified, strictly descriptive 14A reports. No training or model selection."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd

ROOT=Path.home()/"ltd-storage/results/final/FUTURE-RANK-GAP-PHASE14"
EXPECTED_N=18543
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()
def main():
    a=argparse.ArgumentParser()
    a.add_argument("--root",type=Path,default=ROOT)
    a.add_argument("--top",type=int,default=15)
    opt=a.parse_args()
    root=opt.root
    manifest=json.loads((root/"14A_manifest.json").read_text())
    assert manifest["stage"]=="14A"
    assert manifest["scientific_status"]=="POSTHOC_DIAGNOSTIC_ONLY"
    assert manifest["training_allowed"] is False and manifest["selection_allowed"] is False
    expected=manifest["outputs_sha256"]
    assert len(expected)==10, f"Expected ten frozen diagnostic outputs, got {len(expected)}"
    for name,digest in expected.items():
        p=root/name
        if not p.is_file() or sha(p)!=digest:
            raise RuntimeError("14A_MANIFEST_HASH_MISMATCH: "+name)
    print("14A_INTEGRITY_VERIFIED 10/10")
    ranks=pd.read_csv(root/"14A_rank_distribution.csv")
    h=ranks[ranks.component=="HYBRID_FINAL"].set_index("bucket")["count"]
    print("RANK_DISTRIBUTION_SUM",repr(h.sum()),"EXPECTED",EXPECTED_N)
    print("RANK_DISTRIBUTION_HYBRID",h.to_dict())
    if int(h.sum())!=EXPECTED_N:
        raise RuntimeError("RANK_DISTRIBUTION_TOTAL_MISMATCH: inspect bucket counts and component labels; do not alter frozen evidence")
    top1=int(h["rank_1"])
    rescued=int(h[["rank_2","rank_3","rank_4","rank_5"]].sum())
    lost=EXPECTED_N-top1-rescued
    assert (top1,rescued,lost)==(9170,5768,3605)
    print(f"HYBRID_TOP1 {top1}/{EXPECTED_N} = {top1/EXPECTED_N:.4%}")
    print(f"HYBRID_RANK2_5 {rescued}/{EXPECTED_N} = {rescued/EXPECTED_N:.4%}")
    print(f"HYBRID_OUTSIDE_TOP5 {lost}/{EXPECTED_N} = {lost/EXPECTED_N:.4%}")
    print(f"FRACTION_OF_ERRORS_RANK2_5 {rescued/(EXPECTED_N-top1):.4%}")
    sites=pd.read_csv(root/"14A_per_site_rank_gap.csv")
    assert len(sites)==65 and int(sites.support.sum())==EXPECTED_N, "CLASS_SUPPORT_MISMATCH"
    assert abs((sites.support*sites.accuracy).sum()/EXPECTED_N-top1/EXPECTED_N)<1e-5
    assert abs((sites.support*sites.top5_accuracy).sum()/EXPECTED_N-(top1+rescued)/EXPECTED_N)<1e-5
    assert sites.category.isin(["high_top1","recoverable_gap","lost_beyond_top5"]).all()
    # Fixed preregistered categories; no changing thresholds or optimizing variants.
    print("SITE_BUCKET_COUNTS",sites.category.value_counts().to_dict())
    cols=["site_label","support","accuracy","top5_accuracy","top5_minus_top1","mrr","median_true_rank","category"]
    print("SITE_METRICS_ALL_65")
    print(sites[cols].sort_values("site_label").to_string(index=False))
    print("TOP_DESCRIPTIVE_CONFUSIONS")
    edges=pd.read_csv(root/"14A_confusion_edges.csv")
    print(edges.head(max(opt.top,0)).to_string(index=False))
    captures=pd.read_csv(root/"14A_capture_groups.csv")
    assert len(captures)==EXPECTED_N and captures.pcap_uid.astype(str).is_unique
    assert captures.top5_recoverable.astype(bool).sum()==rescued
    assert captures.top5_missed.astype(bool).sum()==lost
    print("CAPTURE_CATEGORY_COUNTS",captures.category.value_counts().to_dict())
    for name in ("14A_component_rescue.csv","14A_per_site_size_metrics.csv","14A_per_site_drift_metrics.csv"):
        d=pd.read_csv(root/name)
        print("DESCRIPTIVE_SCHEMA",name,"rows",len(d),"columns",list(d.columns))
        print(d.head(5).to_string(index=False))
    print("LIMITATION: Future-B is OPEN/FROZEN. DO NOT choose models, thresholds, or sites for experiments from these observations.")
if __name__=="__main__":
    main()
