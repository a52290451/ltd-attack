#!/usr/bin/env python3
"""Read-only model-comparison audit of FROZEN 09B/10B probability exports.

Never reads raw capture CSV, trains a model, selects hyperparameters, or updates
stored 09B/10B/10C artifacts. Outputs a NEW versioned report directory only.

Run from repository root: LTD_ENV=perseo python scripts/diagnostics/audit_frozen_pairwise_stats.py

Two-sided day-block label-swap p-values are POST-HOC conditional randomization
checks (whole day swaps; not independent confirmatory p-values). Day block
bootstrap CIs are percentile intervals; overlapping days may violate IID blocks.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import math
import numpy as np
import pandas as pd

from src.utils.paths import result_path

VERSION = "FROZEN-PAIRWISE-EVIDENCE-V1"
MODELS = {
    "MICRO": "micro_probs",
    "XGB": "macro_xgb_probs",
    "LTD": "macro_ltd_probs",
    "MACRO": "macro_final_probs",
    "HYBRID": "hybrid_final_probs",
}
COMPARISONS = [
    ("HYBRID_vs_MICRO", "MICRO", "HYBRID", "primary"),
    ("HYBRID_vs_MACRO", "MACRO", "HYBRID", "primary"),
    ("HYBRID_vs_XGB", "XGB", "HYBRID", "primary"),
    ("HYBRID_vs_LTD", "LTD", "HYBRID", "secondary"),
    ("MACRO_vs_XGB", "XGB", "MACRO", "component"),
    ("MACRO_vs_LTD", "LTD", "MACRO", "component"),
    ("XGB_vs_LTD", "LTD", "XGB", "component"),
]
METRICS = ("accuracy", "macro_f1", "top5", "mrr")
EXPECTED = {"INTERNAL_09B": (14687, 11), "FUTURE_10B_DEV_FROZEN": (18543, 16)}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf8"))


def read_verified(npz_path, manifest_path, origin, future_10c=None):
    if not (npz_path.is_file() and manifest_path.is_file()):
        raise FileNotFoundError(f"Frozen files missing: {npz_path} or {manifest_path}")
    manifest = read_json(manifest_path)
    digest = sha256(npz_path)
    if origin == "INTERNAL_09B":
        assert manifest.get("status") == "INTERNAL_TEST_OPENED_AND_EVALUATED"
        assert manifest.get("outputs", {}).get("scores") == digest, "09B score SHA mismatch"
    else:
        assert manifest.get("status") == "FUTURE_B_OPENED_AND_EVALUATED"
        prot = manifest.get("protocol", {})
        assert prot.get("model_selection_after_future_open") is False
        assert prot.get("hyperparameter_changes_after_future_open") is False
        if not future_10c or not future_10c.is_file():
            raise FileNotFoundError("10C manifest required to verify 10B score SHA")
        c = read_json(future_10c)
        assert c.get("status") == "POSTHOC_MECHANISTIC_ANALYSIS"
        assert c.get("score_inputs", {}).get("DEV_FROZEN", {}).get("sha256") == digest, "10C→10B score SHA mismatch"
    with np.load(npz_path, allow_pickle=True) as f:
        required = ["pcap_uid", "query_date", "y_true", "candidate_labels", *MODELS.values()]
        missing = [x for x in required if x not in f]
        if missing:
            raise ValueError(f"missing {missing}: {npz_path}")
        uid = f["pcap_uid"].astype(str)
        days = f["query_date"].astype("datetime64[D]").astype(str)
        y = f["y_true"].astype(np.int64)
        labels = f["candidate_labels"].astype(str)
        probs = {k: f[v].astype(np.float64) for k, v in MODELS.items()}
    n, n_days = EXPECTED[origin]
    assert len(y) == n and len(uid) == n and len(np.unique(uid)) == n
    assert len(np.unique(days)) == n_days and len(labels) == 65 and len(set(labels)) == 65
    assert y.min() >= 0 and y.max() <= 64
    for k, p in probs.items():
        assert p.shape == (n, 65) and np.isfinite(p).all() and (p >= 0).all(), k
        assert np.allclose(p.sum(axis=1), 1.0, atol=2e-4, rtol=0), f"probability sum {k}"
    return {"y": y, "days": days, "labels": labels, "uid": uid, "probs": probs,
            "source_sha": digest, "manifest_sha": sha256(manifest_path),
            "source": str(npz_path), "manifest": str(manifest_path)}


def compute_predictions(y, probs):
    ranks = {}
    for name, p in probs.items():
        order = np.argsort(-p, axis=1)
        pred = order[:, 0]
        pos = (order == y[:, None]).argmax(axis=1) + 1
        ranks[name] = {"pred": pred, "rank": pos}
    return ranks


def f1_metrics(tp, pcount, support):
    denom = (pcount + support).astype(float)
    vals = np.divide(2.0 * tp, denom, out=np.zeros_like(denom, dtype=float), where=denom > 0)
    return float(vals.mean())


def daily_stats(y, days, predictions):
    unique_days, date_index = np.unique(days, return_inverse=True)
    d, c = len(unique_days), 65
    sup = np.zeros((d, c), dtype=np.float64)
    sizes = np.bincount(date_index, minlength=d).astype(float)
    np.add.at(sup, (date_index, y), 1)
    out = {}
    for name, p in predictions.items():
        pred, rank = p["pred"], p["rank"]
        tp = np.zeros((d, c), dtype=np.float64)
        pc = np.zeros((d, c), dtype=np.float64)
        mask = pred == y
        np.add.at(tp, (date_index[mask], y[mask]), 1)
        np.add.at(pc, (date_index, pred), 1)
        out[name] = {"tp": tp, "pc": pc,
                     "acc": np.bincount(date_index, weights=mask.astype(float), minlength=d),
                     "top5": np.bincount(date_index, weights=(rank <= 5).astype(float), minlength=d),
                     "rr": np.bincount(date_index, weights=1.0/rank, minlength=d)}
    return unique_days, sizes, sup, out


def metric_totals(w, size, sup, st):
    N = float(w @ size)
    t = w @ st["tp"]
    p = w @ st["pc"]
    s = w @ sup
    return np.array([w @ st["acc"] / N, f1_metrics(t, p, s),
                     w @ st["top5"] / N, w @ st["rr"] / N])


def bootstrap_metrics(size, sup, daily, iterations, seed):
    rng = np.random.default_rng(seed)
    d = len(size)
    weights = rng.multinomial(d, np.full(d, 1 / d), size=iterations).astype(float)
    # All models use IDENTICAL block samples for each bootstrap replicate.
    bootstrap = {}
    for name, st in daily.items():
        arr = np.zeros((iterations, len(METRICS)), dtype=float)
        for i, w in enumerate(weights):
            arr[i, :] = metric_totals(w, size, sup, st)
        bootstrap[name] = arr
    return bootstrap


def swap_p_values(size, sup, a, b, seed, n_perm):
    """Whole-day model-label swap, TWO-SIDED and conditional on exchangeability.

    The test is an exploratory paired block randomization test, not a causal
    claim or a fully calibrated test under arbitrary serial dependence.
    """
    rng = np.random.default_rng(seed)
    d = len(size)
    signs = rng.integers(0, 2, size=(n_perm, d), dtype=np.int8).astype(float)
    n = size.sum()
    obs = metric_totals(np.ones(d), size, sup, b) - metric_totals(np.ones(d), size, sup, a)
    dist = np.empty((n_perm, len(METRICS)))
    for j, k in enumerate(("acc", "top5", "rr")):
        daily_delta = b[k] - a[k]
        null_delta = (daily_delta.sum() - 2 * (signs @ daily_delta)) / n
        dist[:, {"acc": 0, "top5": 2, "rr": 3}[k]] = null_delta
    diff_tp = b["tp"] - a["tp"]
    diff_pc = b["pc"] - a["pc"]
    sum_a_tp, sum_b_tp = a["tp"].sum(axis=0), b["tp"].sum(axis=0)
    sum_a_pc, sum_b_pc = a["pc"].sum(axis=0), b["pc"].sum(axis=0)
    total_sup = sup.sum(axis=0)
    chunk = 512
    for start in range(0, n_perm, chunk):
        q = signs[start:start+chunk]
        dt = q @ diff_tp
        dp = q @ diff_pc
        at, bt = sum_a_tp[None, :] + dt, sum_b_tp[None, :] - dt
        ap, bp = sum_a_pc[None, :] + dp, sum_b_pc[None, :] - dp
        f1a = np.divide(2*at, ap + total_sup, out=np.zeros_like(at), where=(ap+total_sup)>0).mean(axis=1)
        f1b = np.divide(2*bt, bp + total_sup, out=np.zeros_like(bt), where=(bp+total_sup)>0).mean(axis=1)
        dist[start:start+chunk, 1] = f1b-f1a
    return [(np.sum(np.abs(dist[:, j]) >= abs(obs[j]) - 1e-12) + 1) / (n_perm+1)
            for j in range(len(METRICS))]


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    order = np.argsort(p_values)
    adj = np.zeros(n)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, p_values[idx] * (n-rank)))
        adj[idx] = running
    return adj


def verify_archived_metrics(csv_path, period, observed):
    table = pd.read_csv(csv_path)
    # Existing 09B/10B summaries must match the recalculated frozen predictions.
    expected_names = {"MICRO": "MICRO_FINAL", "XGB": "MACRO_XGB", "LTD": "MACRO_LTD",
                      "MACRO": "MACRO_LTD_FINAL", "HYBRID": "LTD_HYBRID_FINAL"}
    if period == "FUTURE_10B_DEV_FROZEN":
        table = table.loc[table["track"].astype(str) == "DEV_FROZEN"]
    for name, model in expected_names.items():
        matched = table.loc[table["model"].astype(str) == model]
        if len(matched) != 1:
            raise ValueError(f"Missing or multiple archived {model} in {csv_path}")
        row = matched.iloc[0]
        for i, k in enumerate(("accuracy", "macro_f1", "top5_accuracy", "mrr")):
            if k not in table.columns:
                raise ValueError(f"Missing {k} in {csv_path}")
            if abs(float(row[k]) - observed[name][i]) > 4e-4:
                raise ValueError(f"ARCHIVE_METRIC_MISMATCH {name} {k}: archive={row[k]} recalc={observed[name][i]}")
    return sha256(csv_path)


def audit_one(origin, score, manifest, table, future_10c, boot, permutations):
    bundle = read_verified(score, manifest, origin, future_10c)
    y, dates = bundle["y"], bundle["days"]
    preds = compute_predictions(y, bundle["probs"])
    day_names, sizes, sup, daily = daily_stats(y, dates, preds)
    observed = {m: metric_totals(np.ones(len(day_names)), sizes, sup, v) for m,v in daily.items()}
    table_sha = verify_archived_metrics(table, origin, observed)
    boots = bootstrap_metrics(sizes, sup, daily, boot, 20261026)
    summary = []
    for m in MODELS:
        row = {"period":origin, "model":m, "n":len(y), "days":len(day_names)}
        row.update({metric: float(observed[m][i]) for i,metric in enumerate(METRICS)})
        summary.append(row)
    comp = []
    for idx,(nm,a,b,role) in enumerate(COMPARISONS):
        ac = preds[a]["pred"] == y
        bc = preds[b]["pred"] == y
        fixed_counts = {"both_correct":int((ac&bc).sum()), "rescued":int((~ac&bc).sum()),
                        "harmed":int((ac&~bc).sum()), "both_wrong":int((~ac&~bc).sum())}
        assert sum(fixed_counts.values()) == len(y)
        diff = observed[b] - observed[a]
        bs = boots[b] - boots[a]
        lower, upper = np.quantile(bs, [0.025,0.975], axis=0)
        ps = swap_p_values(sizes, sup, daily[a], daily[b], seed=20261026+idx,
                           n_perm=permutations)
        for j,metric in enumerate(METRICS):
            comp.append({"period":origin,"comparison":nm,"family":role,"metric":metric,
                         "reference":a,"candidate":b,"n":len(y), "days":len(day_names),
                         "reference_value":float(observed[a][j]),"candidate_value":float(observed[b][j]),
                         "delta":float(diff[j]),"ci95_low":float(lower[j]),"ci95_high":float(upper[j]),
                         "day_swap_p_two_sided_posthoc":float(ps[j]),
                         "bootstrap_fraction_gt_zero_NOT_P":float(np.mean(bs[:,j] > 0)),
                         **fixed_counts})
    comparisons = pd.DataFrame(comp)
    # Holm family of 3 prespecified auditing comparisons x 4 metrics, within each period;
    # this is a POST-HOC audit-family, NOT a prospective claim about original 10B.
    mask = comparisons.family == "primary"
    comparisons["holm_p_12_tests_primary_posthoc"] = np.nan
    comparisons.loc[mask, "holm_p_12_tests_primary_posthoc"] = holm_adjust(
        comparisons.loc[mask, "day_swap_p_two_sided_posthoc"].to_numpy())
    days_out=[]; site_out=[]
    for m, p in preds.items():
        for i, dt in enumerate(day_names):
            ids = np.flatnonzero(dates == dt)
            stats = metric_totals(np.eye(len(day_names))[i], sizes, sup, daily[m])
            days_out.append({"period":origin,"date":str(dt),"model":m,"n":len(ids),
                             **{k:float(stats[j]) for j,k in enumerate(METRICS)}})
        for cls, label in enumerate(bundle["labels"]):
            ix = y == cls
            cp = p["pred"][ix]
            tpr = np.mean(cp == cls)
            predicted_total = np.sum(p["pred"] == cls)
            tp = np.sum(cp == cls)
            f1 = 2*tp/(int(ix.sum()) + int(predicted_total)) if ix.sum()+predicted_total else 0.
            precision = float(tp/predicted_total) if predicted_total else 0.0
            site_out.append({"period":origin,"model":m,"site_label":str(label),"n":int(ix.sum()),
                             "precision":precision,"recall":float(tpr),"class_f1":float(f1),
                             "top5":float(np.mean(p["rank"][ix] <= 5))})
    meta={"period":origin,"npz":bundle["source"], "score_sha256":bundle["source_sha"],
          "manifest":bundle["manifest"],"manifest_sha256":bundle["manifest_sha"],
          "summary_sha256":table_sha,"n":len(y), "days":len(day_names),"sites":65}
    return summary, comparisons, days_out, site_out, meta


def audit_10c(c_dir):
    """Extract existing immutable marginal ablation estimates; no new model selection."""
    path = c_dir/"10C_manifest.json"
    c = read_json(path)
    assert c.get("status") == "POSTHOC_MECHANISTIC_ANALYSIS"
    out = {}
    for key, file in {"summary":"10C_mechanistic_summary.csv", "bootstrap":"10C_mechanistic_bootstrap.csv",
                      "rescues":"10C_rescue_analysis.csv"}.items():
        src = c_dir/file
        digest = sha256(src)
        assert c.get("outputs",{}).get(key) == digest, f"10C SHA mismatch: {file}"
        out[key] = pd.read_csv(src)
    out["bootstrap"] = out["bootstrap"].loc[out["bootstrap"]["track"] == "DEV_FROZEN"].copy()
    out["summary"] = out["summary"].loc[out["summary"]["track"] == "DEV_FROZEN"].copy()
    return out, {"10C_manifest_sha256":sha256(path),"10C_bootstrap_sha256":sha256(c_dir/"10C_mechanistic_bootstrap.csv")}


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--out", type=Path, default=Path(result_path("reports", VERSION)))
    pa.add_argument("--bootstrap", type=int, default=2500)
    pa.add_argument("--permutations", type=int, default=9999)
    args = pa.parse_args()
    if args.bootstrap < 200 or args.permutations < 500:
        raise ValueError("Use at least 200 bootstrap draws and 500 block swaps")
    if args.out.exists():
        raise FileExistsError(f"OUTPUT EXISTS; refusing to overwrite: {args.out}")
    internal = Path(result_path("final", "INTERNAL-TEST"))
    future = Path(result_path("final", "FUTURE-B"))
    specs=[("INTERNAL_09B",internal/"09B_internal_scores.npz", internal/"09B_manifest_internal_test.json",
            internal/"09B_internal_summary.csv",None),
           ("FUTURE_10B_DEV_FROZEN",future/"10B_scores_DEV_FROZEN.npz",future/"10B_manifest_future_b.json",
            future/"10B_future_summary.csv",future/"10C_manifest.json")]
    sumrows, comparison_frames, dates, classes, source_meta=[],[],[],[],[]
    for spec in specs:
        print("VERIFY FROZEN SOURCE",spec[0],flush=True)
        a,b,c,d,m=audit_one(*spec, args.bootstrap,args.permutations)
        sumrows.extend(a);comparison_frames.append(b); dates.extend(c);classes.extend(d); source_meta.append(m)
        print("  PASS n=",m["n"]," days=",m["days"]," verified score SHA",m["score_sha256"][:14],flush=True)
    ablations, meta10c = audit_10c(future)
    # Only now create output. Original inputs untouched.
    args.out.mkdir(parents=True,exist_ok=False)
    pd.DataFrame(sumrows).to_csv(args.out/"00_model_metrics.csv",index=False)
    pd.concat(comparison_frames,ignore_index=True).to_csv(args.out/"01_paired_comparisons.csv",index=False)
    pd.DataFrame(dates).to_csv(args.out/"02_per_day_metrics.csv",index=False)
    pd.DataFrame(classes).to_csv(args.out/"03_per_site_metrics.csv",index=False)
    ablations["summary"].to_csv(args.out/"04_existing_10C_ablation_summary.csv",index=False)
    ablations["bootstrap"].to_csv(args.out/"05_existing_10C_ablation_bootstrap.csv",index=False)
    ablations["rescues"].to_csv(args.out/"06_existing_10C_rescues.csv",index=False)
    note=("# Frozen LTD-Attack statistical evidence — post-hoc extraction\n\n"
          "Read-only audit of the ORIGINAL 09B and 10B DEV_FROZEN scores. No fitting,\n"
          "retraining, architecture selection, source score overwrites, or raw PCAP/Future CSV reads.\n\n"
          "01_paired_comparisons.csv: differences = candidate minus reference. Same UID/date/labels.\n"
          "Day-block bootstrap percentile 95% CI, 2500 draws (or configured). Bootstrap positive fraction is NOT p-value.\n"
          "Two-sided whole-day model-label swap Monte Carlo p-values CONDITIONAL on day-block exchangeability,\n"
          "with randomization; potentially invalid under strong serial dependencies.\n"
          "Holm 12-test adjustments concern 3 Hybrid-vs-Micro/Macro/XGB x 4 metrics per cohort only;\n"
          "a NEW POST-HOC audit family, not a retroactive prospective multiplicity claim.\n"
          "McNemar rescue/harm counts are paired descriptive evidence; no iid capture-level p-values are invented.\n\n"
          "04–06 are EXISTING 10C posthoc ablations, not separate independently validated classifiers.\n"
          "Use no 10C result to select a Future-B tuned model. The 65-site cohort was conditioned on B coverage.\n"
          "The two cohorts are DISTINCT: never pair 09B and 10B sample indices.\n"
          "No claim of statistical superiority is justified until outputs are inspected, including per-site\n"
          "heterogeneity, day/block robustness, and effects of only 11/16 dates.\n")
    (args.out/"README.md").write_text(note)
    paths=sorted(p for p in args.out.iterdir() if p.is_file())
    mani={"stage":VERSION,"read_only_source":True,"training":False,"model_selection":False,
          "calibration_or_threshold_search":False,"source_checks":source_meta,"source_10c":meta10c,
          "parameters":{"bootstrap":args.bootstrap,"permutations":args.permutations,"seed":20261026,
                        "significance":"posthoc_conditional_block_swap","bootstrap":"paired_day_block_percentile"},
          "outputs_sha256":{p.name:sha256(p) for p in paths}}
    (args.out/"manifest.json").write_text(json.dumps(mani,indent=2,ensure_ascii=False)+"\n")
    print("EVIDENCE_AUDIT_COMPLETE",args.out,flush=True)
    for f in ["00_model_metrics.csv","01_paired_comparisons.csv","05_existing_10C_ablation_bootstrap.csv"]:
        print("  ",args.out/f)

if __name__ == "__main__":
    main()