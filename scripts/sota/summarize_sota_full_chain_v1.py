#!/usr/bin/env python3
"""Print decision-relevant conclusions from chained SOTA read-only JSON."""
import argparse,json
from pathlib import Path
def fmt(v):
    if v is None:return "N/D"
    return f"{v:.5f}" if isinstance(v,(int,float)) else str(v)
def main():
    p=argparse.ArgumentParser()
    p.add_argument("report",type=Path)
    a=p.parse_args()
    r=json.loads(a.report.read_text())
    print("CHAIN_STATUS",r.get("status"),"SECONDS",r.get("elapsed_seconds"))
    print("BATTERY",r.get("battery"))
    print("\n======= FROZEN MODEL RESULTS =======")
    for name,entry in r["reconstructed_score_audit"].items():
        print(name,"B DEV_LATE accuracy",fmt(entry["B_best_dev_late_accuracy"]),
              "macro-F1",fmt(entry["B_best_dev_late_macro_f1"]))
        for cohort in ("09B","Future-B"):
            x=entry[cohort]
            print(" ",cohort,"accuracy",fmt(x["accuracy"]),
                  "macro-F1",fmt(x["macro_f1"]),
                  "classes",x["predicted_classes_present"],
                  "dominant",fmt(x["majority_predicted_fraction"]),
                  "entropy",fmt(x["normalized_prediction_entropy"]),
                  "confidence",fmt(x["mean_max_confidence"]))
    print("\n======= SWALLOW CHECKPOINT PROBE =======")
    for stage,info in r["swallow_cpu_checkpoint_probe"].items():
        print("SWALLOW",stage,"selected_epoch",info["selected_epoch"],
              "head_weight_norm",fmt(info["classifier_weight_norm"]),
              "head_weight_std",fmt(info["classifier_weight_std"]),
              "head_bias_std",fmt(info["classifier_bias_std"]))
        for split,v in info["groups"].items():
            print(" ",split,"sample_acc",fmt(v["sampled_accuracy"]),
                  "predicted_classes",v["predicted_unique_classes"],
                  "dominant",fmt(v["dominant_predicted_fraction"]),
                  "confidence",fmt(v["mean_top_probability"]),
                  "embedding_std",fmt(v["embedding_across_samples_std"]),
                  "logit_std",fmt(v["logit_across_samples_std"]))
    print("\n======= HISTORICAL INPUT PROFILES =======")
    for model,item in r["historical_cache_feature_profiles"].items():
        print(model,"feature_shape",item["feature_shape"],
              "timestamp_regressions",item.get("timestamp_regressions"))
        for split,info in item["splits"].items():
            print(" ",split,"nonzero_ratio",fmt(info["feature_nonzero_ratio"]),
                  "all_zero",fmt(info["sample_fraction_all_zero"]),
                  "saturated",fmt(info["feature_saturated_uint16_ratio"]),
                  "sum",info["feature_per_capture_sum"])
    print("\n======= EXTERNAL ORIGINAL FEATURE SAMPLES =======")
    for cohort,item in r.get("external_original_feature_samples",{}).items():
        print(cohort,"samples",item["sample_n"],"UID alignment",item["sample_uid_label_alignment"])
        for key,profile in item["raw_vectors_and_original_features"].items():
            print(" ",key,"median",fmt(profile["median"]),
                  "p05",fmt(profile["p05"]),"p95",fmt(profile["p95"]))
    print("\nIMPORTANT: descriptive audit, not an exact original-method reproduction.")
if __name__=="__main__":main()
