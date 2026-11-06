#!/usr/bin/env python3
"""Print existing S6 inferential findings and inspect CUMUL input viability.

Read-only: no new inference, training, array parsing, Dataset C, or write.
CUMUL's signed packet-size feature extraction needs packet sizes, and the
direction-only vector must NOT be mislabeled as a faithful CUMUL reproduction.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
OUTPUT=Path("/data/ltd-attack/sota/s6/comparison_v1")
S1=Path.home()/"ltd-ops/sota/s1_dev_frozen_20261102T130232Z/s1_manifest.json"
FUTURE=Path.home()/"ltd-storage/datasets/future/CLEAN_final_vectors_sites_concept_drift.csv"

def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()

def fail(condition,message):
    if not condition:raise RuntimeError("S6_S7_REVIEW_BLOCKED "+message)

def rows(path):
    with path.open(newline="",encoding="utf-8") as f:
        return list(csv.DictReader(f))

def columns(path):
    with path.open(newline="",encoding="utf-8") as f:
        return next(csv.reader(f))

def main():
    m=OUTPUT/"manifest.json"
    fail(m.is_file(),"S6 output manifest missing; no duplicate run")
    manifest=json.loads(m.read_text())
    fail(manifest.get("classification")=="S6_PAIRED_SOTA_VS_HYBRID_POSTHOC_V1",
         "S6 manifest classification invalid")
    for name,digest in manifest["results_sha256"].items():
        candidate=OUTPUT/name
        fail(candidate.is_file() and sha(candidate)==digest,
             "S6 report checksum mismatch "+name)
    metrics=rows(OUTPUT/"00_model_metrics.csv")
    paired=rows(OUTPUT/"01_sota_vs_hybrid_paired.csv")
    fail(len(metrics)==10 and len(paired)==16,
         "expected 5 models x 2 cohorts and 2 x 4 comparisons per cohort")
    for cohort in ("09B","10B_DEV_FROZEN"):
        print("\n=== S6",cohort,"===")
        for r in metrics:
            if r["cohort"]==cohort:
                print("MODEL",r["model"],"N",r["n"],
                      "ACCURACY",round(float(r["accuracy"])*100,3),
                      "MACRO_F1",round(float(r["macro_f1"])*100,3),
                      "TOP5",round(float(r["top5"])*100,3),
                      "MRR",round(float(r["mrr"]),4))
        for r in paired:
            if r["cohort"]!=cohort:continue
            print("PAIRED",r["candidate"],r["metric"],
                  "SOTA_MINUS_HYBRID_PP",round(float(r["delta_sota_minus_hybrid"])*100,3),
                  "CI95_PP",[round(float(r[k])*100,3) for k in ("ci95_low","ci95_high")],
                  "P",r["p_posthoc_two_sided"],
                  "HOLM8",r["holm8_adjusted_p_posthoc"],
                  "METHOD",r["p_method"])
    # Input feasibility: inspect header, without loading any future feature
    # values; future source has already been used for S5, but is not needed
    # for a new model's selection, and must not inform architecture tuning.
    frozen=json.loads(S1.read_text())
    historical=Path(frozen["historical_checksums"]["historical_vectors"]["path"])
    fail(historical.is_file(),"historical input not found")
    cols=columns(historical)
    names=[name for name in cols if any(tag in name.lower() for tag in
           ("size","length","bytes","packet","direction","time"))]
    print("\n=== S7 CUMUL SOURCE FEASIBILITY ===")
    print("HISTORICAL_PACKET_RELATED_COLUMNS",names)
    print("HISTORICAL_VECTOR_SOURCE",historical)
    print("NOTE: column availability alone does not establish original CUMUL"
          " signed-size fidelity. Verify definition and original PCAP or a"
          " provenance-audited packet-size vector before calling CUMUL faithful.")
    print("S6_REPORT_AND_S7_INPUT_GATE=PASS_READ_ONLY")

if __name__=="__main__":
    try:main()
    except Exception as e:
        print("S6_S7_REPORT_ERROR",type(e).__name__,str(e)[:700],
              file=sys.stderr,flush=True)
        sys.exit(4)
