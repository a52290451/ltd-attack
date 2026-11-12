#!/usr/bin/env python3
"""Compact, read-only report for LTD SOTA forensic JSON; no GPU or model imports."""
import argparse
import json
from pathlib import Path


def pcent(x):
    return f"{100 * x:.2f}%" if isinstance(x, (float, int)) else "N/D"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    result = json.loads(args.report.read_text(encoding="utf-8"))
    print("AUDITORÍA:", result.get("classification", "N/D"))
    print("ESTADO:", result.get("overall_status", "NO REPORTADO"))
    print()
    for model, detail in result.get("models", {}).items():
        print("======", model, "======")
        for stage, stage_rec in detail.get("training", {}).items():
            print(stage, "época seleccionada:", stage_rec.get("selected_epoch"),
                  "validación macro-F1:", pcent(stage_rec.get("best_dev_late_macro_f1")),
                  "validación accuracy:", pcent(stage_rec.get("best_dev_late_accuracy")),
                  "mejor época:", stage_rec.get("best_dev_late_epoch"),
                  "selección correcta:", stage_rec.get("selection_matches_best_epoch"))
        for cohort, ev in detail.get("evaluation", {}).items():
            print(cohort,
                  "accuracy:", pcent(ev.get("accuracy")),
                  "macro-F1:", pcent(ev.get("macro_f1")),
                  "Top-5:", pcent(ev.get("top5")),
                  "MRR:", pcent(ev.get("mrr")))
            print("  clases predichas:", ev.get("predicted_classes_present"),
                  "top1 dominante:", pcent(ev.get("majority_predicted_fraction")),
                  "confianza media:", pcent(ev.get("mean_max_confidence")),
                  "entropía relativa:", ev.get("normalized_prediction_entropy"),
                  "clase real dominante:", pcent(ev.get("majority_true_fraction")))
            print("  top clases predichas:", ev.get("top_predicted_class_counts"))
        print()
    for cohort, align in result.get("cross_model_alignment", {}).items():
        mismatches = [(model, key) for model, vals in align.items()
                      for key, ok in vals.items() if not ok]
        print("ALINEACIÓN", cohort, "PASS" if not mismatches else "FAIL",
              mismatches if mismatches else "")
    print("ADVERTENCIA: Esta auditoría no prueba equivalencia con el artículo original.")


if __name__ == "__main__":
    main()
