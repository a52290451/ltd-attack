#!/usr/bin/env python3
"""Read-only per-site annotated 14B diagnostics. Future-B is post-hoc only.
No training, parameter choice, model selection, or mutation of frozen artifacts.
"""
from pathlib import Path
import argparse
import hashlib
import json
import pandas as pd

HOME=Path.home()
ROOT=HOME/"ltd-storage/results/final/FUTURE-RANK-GAP-PHASE14"
SOURCE=HOME/"ltd-storage/datasets/future/CLEAN_final_features_sites_concept_drift.csv"
EXPECTED_N=18543

def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--source",type=Path,default=SOURCE)
    args=parser.parse_args()
    m=json.loads((args.root/"14A_manifest.json").read_text())
    if (m.get("stage")!="14A" or
        m.get("scientific_status")!="POSTHOC_DIAGNOSTIC_ONLY" or
        m.get("training_allowed") is not False or
        m.get("selection_allowed") is not False):
        raise RuntimeError("STOP: 14A scientific status not frozen diagnostic")
    files=m.get("outputs_sha256",{})
    if len(files)!=10:raise RuntimeError("STOP: 14A expected ten diagnostic outputs")
    for name,sha in files.items():
        p=args.root/name
        if not p.is_file() or digest(p)!=sha:
            raise RuntimeError("STOP: 14A output differs from manifest: "+name)
    print("14A_HASHES_VERIFIED 10/10")

    b=pd.read_csv(args.source,usecols=["site_label","site"])
    b["site_label"]=b.site_label.astype(str)
    b["site"]=b.site.astype(str).str.strip().str.lower().str.removeprefix("www.")
    if b["site_label"].isna().any() or b["site"].isna().any():
        raise RuntimeError("STOP: malformed labels/domains")
    pairs=b.drop_duplicates()
    if (pairs.groupby("site_label").site.nunique()>1).any() or (pairs.groupby("site").site_label.nunique()>1).any():
        raise RuntimeError("STOP: site_label/site mapping not one-to-one")
    sites=pd.read_csv(args.root/"14A_per_site_rank_gap.csv")
    sites["site_label"]=sites.site_label.astype(str)
    if len(sites)!=65 or int(sites.support.sum())!=EXPECTED_N:
        raise RuntimeError("STOP: not canonical 65 sites/18543 examples")
    annotated=sites.merge(pairs,on="site_label",how="left",validate="one_to_one",indicator=True)
    if not annotated["_merge"].eq("both").all():
        raise RuntimeError("STOP: unknown canonical site labels")
    annotated=annotated.drop(columns="_merge")
    annotated["recoverable_count"]=(annotated.support*annotated.top5_minus_top1).round().astype(int)
    annotated["outside_top5_count"]=(annotated.support*(1-annotated.top5_accuracy)).round().astype(int)
    if annotated.recoverable_count.sum()!=5768 or annotated.outside_top5_count.sum()!=3605:
        raise RuntimeError("STOP: canonical count mismatch")

    print("SITE_CATEGORIES",annotated.category.value_counts().to_dict())
    print("CANONICAL_DOMAINS_65_WITH_RANK_METRICS")
    cols=["site_label","site","support","accuracy","top5_accuracy","top5_minus_top1",
          "recoverable_count","outside_top5_count","category"]
    print(annotated[cols].sort_values(["site"],ascending=[True]).to_string(index=False,
          formatters={k:"{:.2%}".format for k in ["accuracy","top5_accuracy","top5_minus_top1"]}))
    print("CONFUSION_EDGES_WITH_DOMAINS")
    edges=pd.read_csv(args.root/"14A_confusion_edges.csv").head(20).copy()
    edge_map=dict(zip(pairs.site_label,pairs.site))
    edges["true_site"]=edges.true_site.astype(str).map(edge_map)
    edges["predicted_site"]=edges.predicted_site.astype(str).map(edge_map)
    if edges[["true_site","predicted_site"]].isna().any().any():
        raise RuntimeError("STOP: unmapped confusion label")
    print(edges.to_string(index=False))
    drift=pd.read_csv(args.root/"14A_per_site_drift_metrics.csv")
    print("DRIFT_METRIC_ROWS",len(drift),"CANONICAL_SITES",len(sites))
    if len(drift)!=65:
        print("DRIFT_CAVEAT: drift table includes raw/unfiltered sites; do not aggregate without canonical filtering")
    print("NOTE: only descriptive analysis of OPEN/FROZEN Future-B; cannot determine model parameters or promote candidates.")
if __name__=="__main__":main()
