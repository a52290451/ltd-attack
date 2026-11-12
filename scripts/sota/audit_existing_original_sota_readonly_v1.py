#!/usr/bin/env python3
"""Read-only post-hoc audit of S9 NetCLR, S11 Swallow, S12 WFCAT.

This is a forensic integrity/prediction audit, NOT a methodological equivalence
certificate. No CUDA, TensorFlow, model loading, checkpoint mutation or training.
Outputs one JSON object on stdout. Defaults to Zeus's separated artifact root.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np

MODELS = {
    "NetCLR": ("s9_netclr_author_original_v1", "supervised_B_v1", "refit_C_v1"),
    "Swallow": ("s11_swallow_authors_seed42", "supervised_B", "supervised_C"),
    "WFCAT": ("s12_wfcat_authors_iat_seed42", "B", "C"),
}
COHORTS = {"09B": 14687, "10B_DEV_FROZEN": 18543}
METRICS = ("accuracy", "macro_f1", "top5", "mrr")
DEFAULT_ROOT = "/data/datasets_brayan/ltd-sota-2026/artifacts/sota"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError("AUDIT_FAIL " + message)


def metrics_and_distribution(y: np.ndarray, probs: np.ndarray) -> dict:
    n = len(y)
    top = probs.argmax(axis=1)
    order = np.argsort(-probs, axis=1, kind="stable")
    ranks = np.argmax(order == y[:, None], axis=1) + 1
    predicted = np.bincount(top, minlength=65)
    actual = np.bincount(y, minlength=65)
    correct = np.bincount(y[top == y], minlength=65)
    macro = np.divide(
        2 * correct, actual + predicted, out=np.zeros(65, dtype=np.float64),
        where=(actual + predicted) > 0
    ).mean()
    ent = -np.sum(probs * np.log(np.clip(probs, 1e-30, 1.0)), axis=1)
    return {
        "n": n,
        "accuracy": float((top == y).mean()),
        "macro_f1": float(macro),
        "top5": float((ranks <= 5).mean()),
        "mrr": float((1.0 / ranks).mean()),
        "true_classes_present": int((actual > 0).sum()),
        "predicted_classes_present": int((predicted > 0).sum()),
        "majority_predicted_fraction": float(predicted.max() / n),
        "majority_true_fraction": float(actual.max() / n),
        "normalized_prediction_entropy": float(ent.mean() / np.log(65)),
        "mean_max_confidence": float(probs.max(axis=1).mean()),
        "median_max_confidence": float(np.median(probs.max(axis=1))),
        "top_predicted_class_counts": [
            {"class_id": int(i), "count": int(predicted[i])}
            for i in np.argsort(-predicted, kind="stable")[:8]
        ],
        "top_true_class_counts": [
            {"class_id": int(i), "count": int(actual[i])}
            for i in np.argsort(-actual, kind="stable")[:8]
        ],
        "chance_uniform_accuracy": 1.0 / 65.0,
        "chance_uniform_top5": 5.0 / 65.0,
    }


def validate_training(path: Path, b_name: str, c_name: str) -> dict:
    result = {}
    for stage, folder in (("B", b_name), ("C", c_name)):
        mp = path / folder / "manifest.json"
        require(mp.is_file(), str(mp) + " missing")
        meta = json.loads(mp.read_text())
        ck = meta.get("checkpoint")
        require(ck, str(mp) + " missing checkpoint")
        checkpoint = Path(ck)
        require(checkpoint.is_file(), str(checkpoint) + " missing")
        require(digest(checkpoint) == meta.get("checkpoint_sha256"),
                str(checkpoint) + " sha256 mismatch")
        require(meta.get("dataset_c_opened") is False,
                str(mp) + " dataset C exclusion missing")
        require(meta.get("external_data_opened") is False,
                str(mp) + " external exclusion missing")
        hist = meta.get("history", [])
        rec = {"classification": meta.get("classification"),
               "checkpoint_sha256": meta["checkpoint_sha256"],
               "selected_epoch": meta.get("selected_epoch", meta.get("refit_epochs")),
               "training_n": meta.get("training_n", meta.get("train_n")),
               "history_epochs": len(hist),
               "paper_exact": meta.get("config", {}).get("paper_exact", "UNDECLARED")}
        if stage == "B":
            scored = [entry for entry in hist if "val_macro_f1" in entry]
            if scored:
                best = max(scored, key=lambda x: float(x["val_macro_f1"]))
                rec["best_dev_late_macro_f1"] = float(best["val_macro_f1"])
                rec["best_dev_late_accuracy"] = float(best.get("val_accuracy", best.get("validation_accuracy", float("nan"))))
                rec["best_dev_late_epoch"] = best.get("epoch")
                rec["selection_matches_best_epoch"] = (
                    rec["selected_epoch"] == best.get("epoch")
                )
        result[stage] = rec
    return result


def evaluate(path: Path, cohort: str) -> tuple[dict, tuple]:
    directory = path / "evaluation" / cohort
    mp = directory / "manifest.json"
    scores = directory / "scores_sota_aligned.npz"
    require(mp.is_file() and scores.is_file(),
            str(directory) + " incomplete external result")
    manifest = json.loads(mp.read_text())
    require(digest(scores) == manifest.get("score_sha256"),
            str(scores) + " sha256 mismatch")
    require(manifest.get("cohort") == cohort, str(mp) + " wrong cohort")
    require(manifest.get("dataset_c_opened") is False,
            str(mp) + " data C claim incorrect")
    require(manifest.get("no_external_tuning") is True,
            str(mp) + " frozen evaluation claim missing")
    with np.load(scores, allow_pickle=False) as z:
        for key in ("pcap_uid", "query_date", "y_true",
                    "candidate_labels", "sota_probs"):
            require(key in z.files, str(scores) + " missing " + key)
        uid = z["pcap_uid"].astype(str)
        dates = z["query_date"].astype(str)
        true = np.asarray(z["y_true"], dtype=np.int64)
        labels = z["candidate_labels"].astype(str)
        probs = np.asarray(z["sota_probs"], dtype=np.float64)
    n = COHORTS[cohort]
    require(len(uid) == n and len(dates) == n and len(true) == n,
            str(scores) + " wrong cohort length")
    require(len(set(uid)) == n, str(scores) + " duplicate UIDs")
    require(len(labels) == 65 and len(set(labels)) == 65,
            str(scores) + " label order/coverage invalid")
    require(probs.shape == (n, 65), str(scores) + " incorrect score shape")
    require(np.isfinite(probs).all() and (probs >= 0).all()
            and (probs <= 1 + 1e-6).all(), str(scores) + " invalid probabilities")
    require(np.allclose(probs.sum(axis=1), 1, atol=2e-4, rtol=0),
            str(scores) + " rows not normalized")
    require(np.all((true >= 0) & (true < 65)),
            str(scores) + " true label out of range")
    observed = metrics_and_distribution(true, probs)
    require(manifest.get("n") == n, str(mp) + " count mismatch")
    for key in METRICS:
        require(abs(observed[key] - float(manifest["metrics"][key])) < 1e-6,
                str(mp) + " independently recomputed " + key + " mismatch")
    observed.update({
        "status": "PASS_RECOMPUTED_WITH_SHA",
        "cohort": cohort,
        "score_sha256": manifest["score_sha256"],
        "distinct_dates": len(set(dates)),
    })
    return observed, (uid, dates, true, labels)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artifact-root", type=Path,
                   default=Path(os.environ.get("LTD_SOTA_ARTIFACT_ROOT", DEFAULT_ROOT)))
    p.add_argument("--output", type=Path, default=None,
                   help="Optional JSON output path OUTSIDE checkpoints")
    args = p.parse_args()
    root = args.artifact_root
    result = {
        "classification": "LTD_READONLY_SOTA_FORENSIC_V1",
        "limitations": [
            "This verifies score integrity and model prediction distributions, not author-paper equivalence.",
            "No source-dataset packet/time semantic equivalence test is performed.",
            "Model evaluation manifests prove claims, not independently observed exclusion of Future-B from all prior processes.",
            "No model retraining or tuning is authorized by this audit.",
        ],
        "artifact_root": str(root), "models": {},
        "cross_model_alignment": {},
    }
    by_cohort = {name: {} for name in COHORTS}
    for model, (folder, b, c) in MODELS.items():
        loc = root / folder
        record = {"training": validate_training(loc, b, c),
                  "evaluation": {}}
        for cohort in COHORTS:
            eval_record, alignment = evaluate(loc, cohort)
            record["evaluation"][cohort] = eval_record
            by_cohort[cohort][model] = alignment
        result["models"][model] = record
    for cohort, by_model in by_cohort.items():
        first = next(iter(by_model.values()))
        aligned = {}
        for model, pack in by_model.items():
            aligned[model] = {
                "same_uid_order": bool(np.array_equal(pack[0], first[0])),
                "same_dates": bool(np.array_equal(pack[1], first[1])),
                "same_true_labels": bool(np.array_equal(pack[2], first[2])),
                "same_candidate_labels": bool(np.array_equal(pack[3], first[3])),
            }
        require(all(all(v.values()) for v in aligned.values()),
                cohort + " cross-model UID/date/y/class order mismatch")
        result["cross_model_alignment"][cohort] = aligned
    result["overall_status"] = "PASS_PREDICTION_INTEGRITY_ONLY"
    blob = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        dest = args.output
        require(not dest.exists(), "refusing overwrite of " + str(dest))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(blob)
        print("LTD_SOTA_FORENSIC_WRITTEN", str(dest), file=sys.stderr)
    else:
        sys.stdout.write(blob)


if __name__ == "__main__":
    main()
