from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import torch

from _common import (
    git_commit,
    load_historical_65,
    output_dir,
    sha256,
    write_json,
)

FOLDS = [
    "A_EARLY_TO_MIDDLE",
    "B_EARLY_MIDDLE_TO_LATE",
]

V1_CONTEXT = 7
V2_CONTEXT = 5
ALPHA = 0.375

N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20260920
EPS = 1e-12


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(path)

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fuse(xgb, ltd):
    z = (
        (1.0 - ALPHA)
        * np.log(np.clip(xgb, EPS, 1.0))
        +
        ALPHA
        * np.log(np.clip(ltd, EPS, 1.0))
    )

    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)

    return p / p.sum(axis=1, keepdims=True)


def predictions(p, y):
    order = np.argsort(-p, axis=1)
    pred = order[:, 0]

    ranks = (
        np.argmax(
            order == y[:, None],
            axis=1,
        )
        + 1
    )

    return pred, ranks


def metrics(y, pred, ranks, idx=None):
    if idx is None:
        idx = np.arange(len(y))

    yy = y[idx]
    pp = pred[idx]
    rr = ranks[idx]

    return {
        "accuracy":
            float(np.mean(yy == pp)),

        "macro_f1":
            float(
                f1_score(
                    yy,
                    pp,
                    labels=np.arange(65),
                    average="macro",
                    zero_division=0,
                )
            ),

        "top5_accuracy":
            float(np.mean(rr <= 5)),

        "mrr":
            float(np.mean(1.0 / rr)),
    }


def deltas(
    y,
    v1_pred,
    v1_rank,
    v2_pred,
    v2_rank,
    idx=None,
):
    a = metrics(
        y,
        v1_pred,
        v1_rank,
        idx,
    )

    b = metrics(
        y,
        v2_pred,
        v2_rank,
        idx,
    )

    return {
        k: b[k] - a[k]
        for k in a
    }


def block_bootstrap(
    y,
    dates,
    v1_pred,
    v1_rank,
    v2_pred,
    v2_rank,
):
    unique_dates = np.array(
        sorted(pd.unique(dates))
    )

    by_date = {
        d: np.flatnonzero(dates == d)
        for d in unique_dates
    }

    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    rows = []

    for _ in range(N_BOOTSTRAP):

        sampled = rng.choice(
            unique_dates,
            size=len(unique_dates),
            replace=True,
        )

        idx = np.concatenate(
            [by_date[d] for d in sampled]
        )

        rows.append(
            deltas(
                y,
                v1_pred,
                v1_rank,
                v2_pred,
                v2_rank,
                idx,
            )
        )

    return pd.DataFrame(rows)


def main():
    print("=" * 78)
    print("07D-1R — CONTEXT ROBUSTNESS AUDIT")
    print("=" * 78)

    print("V1 context:", V1_CONTEXT)
    print("V2 candidate context:", V2_CONTEXT)
    print("Alpha frozen:", ALPHA)
    print("No model/context selection.")
    print("INTERNAL_TEST: CLOSED")
    print("Future-B: CLOSED")

    here = Path(__file__).resolve().parent

    mod07d = load_module(
        here / "07D_01_context_length_ablation.py",
        "macro_v2_07d_context",
    )

    mod07a, source07a = mod07d.load_07a()

    features, frozen_path = (
        mod07a.load_frozen_features()
    )

    out = output_dir()

    daily_path = (
        out
        / "06C_daily_base_profiles.csv"
    )

    assignment_path = (
        out
        / "01_temporal_split_assignments.csv"
    )

    daily = pd.read_csv(daily_path)
    daily["site_label"] = (
        daily["site_label"].astype(str)
    )
    daily["date"] = pd.to_datetime(
        daily["date"]
    )

    captures, historical_path = (
        load_historical_65()
    )

    assignments = pd.read_csv(
        assignment_path
    )

    captures = captures.merge(
        assignments[
            [
                "pcap_uid",
                "temporal_split",
            ]
        ],
        on="pcap_uid",
        how="inner",
        validate="one_to_one",
    )

    captures["site_label"] = (
        captures["site_label"].astype(str)
    )

    dates, _ = mod07a.derive_dates(
        captures
    )

    captures["_query_date"] = (
        pd.to_datetime(dates)
    )

    fold_specs = {
        f["fold"]: f
        for f in mod07a.FOLDS
    }

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    bootstrap_rows = []
    fold_rows = []
    day_rows = []
    class_rows = []

    for fold in FOLDS:

        print()
        print("=" * 78)
        print(fold)
        print("=" * 78)

        fold_data = mod07d.prepare_fold(
            mod07a,
            captures,
            daily,
            features,
            fold_specs[fold],
        )

        score_path = (
            out
            / f"07B_01_scores_{fold}.npz"
        )

        old = np.load(
            score_path,
            allow_pickle=True,
        )

        xgb_probs = old[
            "xgb_probs"
        ].astype(np.float64)

        v1_ltd = old[
            "ltd_ensemble_probs"
        ].astype(np.float64)

        y = old[
            "y_true"
        ].astype(int)

        labels = old[
            "candidate_labels"
        ].astype(str)

        pcap_uid = old[
            "pcap_uid"
        ].astype(str)

        expected_uid = (
            fold_data[
                "test"
            ][
                "pcap_uid"
            ]
            .astype(str)
            .to_numpy()
        )

        if not np.array_equal(
            pcap_uid,
            expected_uid,
        ):
            raise RuntimeError(
                f"{fold}: pcap_uid alignment mismatch "
                "between stored V1 scores and reconstructed fold."
            )

        if not np.array_equal(
            labels,
            fold_data[
                "candidate_labels"
            ].astype(str),
        ):
            raise RuntimeError(
                f"{fold}: candidate-label order mismatch."
            )

        # Re-run only the preselected W=5 candidate.
        result = mod07d.run_window(
            mod07a,
            fold_data,
            features,
            V2_CONTEXT,
            xgb_probs,
            device,
        )

        v2_ltd = result[
            "ensemble_probs"
        ].astype(np.float64)

        v1_probs = fuse(
            xgb_probs,
            v1_ltd,
        )

        v2_probs = fuse(
            xgb_probs,
            v2_ltd,
        )

        v1_pred, v1_rank = predictions(
            v1_probs,
            y,
        )

        v2_pred, v2_rank = predictions(
            v2_probs,
            y,
        )

        observed = deltas(
            y,
            v1_pred,
            v1_rank,
            v2_pred,
            v2_rank,
        )

        query_dates = (
            pd.to_datetime(
                fold_data["test_dates"]
            )
            .strftime("%Y-%m-%d")
            .to_numpy()
        )

        if len(query_dates) != len(y):
            raise RuntimeError(
                f"{fold}: date length mismatch"
            )

        boot = block_bootstrap(
            y,
            query_dates,
            v1_pred,
            v1_rank,
            v2_pred,
            v2_rank,
        )

        for metric in [
            "accuracy",
            "macro_f1",
            "top5_accuracy",
            "mrr",
        ]:
            values = boot[
                metric
            ].to_numpy()

            bootstrap_rows.append(
                {
                    "fold":
                        fold,

                    "metric":
                        metric,

                    "observed_delta":
                        observed[metric],

                    "bootstrap_mean":
                        float(values.mean()),

                    "ci95_low":
                        float(
                            np.quantile(
                                values,
                                0.025,
                            )
                        ),

                    "ci95_high":
                        float(
                            np.quantile(
                                values,
                                0.975,
                            )
                        ),

                    "p_delta_gt_0":
                        float(
                            np.mean(
                                values > 0
                            )
                        ),
                }
            )

        improved = 0
        worse = 0
        tied = 0

        for date in sorted(
            pd.unique(query_dates)
        ):
            idx = np.flatnonzero(
                query_dates == date
            )

            d = deltas(
                y,
                v1_pred,
                v1_rank,
                v2_pred,
                v2_rank,
                idx,
            )

            if d["accuracy"] > 0:
                improved += 1
            elif d["accuracy"] < 0:
                worse += 1
            else:
                tied += 1

            day_rows.append(
                {
                    "fold":
                        fold,
                    "date":
                        date,
                    "n":
                        len(idx),
                    **{
                        f"delta_{k}": v
                        for k, v in d.items()
                    },
                }
            )

        v1_correct = (
            v1_pred == y
        )
        v2_correct = (
            v2_pred == y
        )

        v2_only = (
            ~v1_correct
            & v2_correct
        )

        v1_only = (
            v1_correct
            & ~v2_correct
        )

        fold_rows.append(
            {
                "fold":
                    fold,

                "v1_context":
                    V1_CONTEXT,

                "v2_context":
                    V2_CONTEXT,

                **{
                    f"delta_{k}": v
                    for k, v in
                    observed.items()
                },

                "v2_only_correct_n":
                    int(v2_only.sum()),

                "v1_only_correct_n":
                    int(v1_only.sum()),

                "net_correct_n":
                    int(
                        v2_only.sum()
                        -
                        v1_only.sum()
                    ),

                "improved_days":
                    improved,

                "worse_days":
                    worse,

                "tied_days":
                    tied,
            }
        )

        for cls in range(65):
            idx = np.flatnonzero(
                y == cls
            )

            a = v1_correct[idx]
            b = v2_correct[idx]

            class_rows.append(
                {
                    "fold":
                        fold,

                    "site_label":
                        labels[cls],

                    "n":
                        len(idx),

                    "v1_accuracy":
                        float(a.mean()),

                    "v2_accuracy":
                        float(b.mean()),

                    "delta_accuracy":
                        float(
                            b.mean()
                            -
                            a.mean()
                        ),

                    "v2_only_correct_n":
                        int(
                            (
                                ~a
                                & b
                            ).sum()
                        ),

                    "v1_only_correct_n":
                        int(
                            (
                                a
                                & ~b
                            ).sum()
                        ),
                }
            )

        score_out = (
            out
            / f"07D_01R_scores_{fold}.npz"
        )

        np.savez_compressed(
            score_out,
            pcap_uid=
                pcap_uid.astype(str),

            y_true=
                y,

            candidate_labels=
                labels.astype(str),

            xgb_probs=
                xgb_probs.astype(np.float32),

            v1_ltd_probs=
                v1_ltd.astype(np.float32),

            v2_ltd_probs=
                v2_ltd.astype(np.float32),

            v1_fusion_probs=
                v1_probs.astype(np.float32),

            v2_fusion_probs=
                v2_probs.astype(np.float32),
        )

    boot_df = pd.DataFrame(
        bootstrap_rows
    )

    folds_df = pd.DataFrame(
        fold_rows
    )

    days_df = pd.DataFrame(
        day_rows
    )

    classes_df = pd.DataFrame(
        class_rows
    )

    b = (
        boot_df[
            boot_df["fold"]
            == "B_EARLY_MIDDLE_TO_LATE"
        ]
        .set_index("metric")
    )

    robustness_pass = bool(
        b.loc[
            "accuracy",
            "observed_delta",
        ] > 0
        and
        b.loc[
            "macro_f1",
            "observed_delta",
        ] > 0
        and
        b.loc[
            "accuracy",
            "p_delta_gt_0",
        ] >= 0.90
        and
        b.loc[
            "macro_f1",
            "p_delta_gt_0",
        ] >= 0.90
        and
        b.loc[
            "top5_accuracy",
            "observed_delta",
        ] >= 0
        and
        b.loc[
            "mrr",
            "observed_delta",
        ] >= 0
    )

    folds_path = (
        out
        / "07D_01R_robustness_summary.csv"
    )

    boot_path = (
        out
        / "07D_01R_day_block_bootstrap.csv"
    )

    days_path = (
        out
        / "07D_01R_per_day_deltas.csv"
    )

    classes_path = (
        out
        / "07D_01R_per_class_deltas.csv"
    )

    folds_df.to_csv(
        folds_path,
        index=False,
    )

    boot_df.to_csv(
        boot_path,
        index=False,
    )

    days_df.to_csv(
        days_path,
        index=False,
    )

    classes_df.to_csv(
        classes_path,
        index=False,
    )

    manifest = {
        "stage":
            "07D_01R_context_robustness_audit",

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "git_commit":
            git_commit(),

        "data_policy": {
            "development_only":
                True,
            "context_selection":
                False,
            "alpha_tuning":
                False,
            "internal_test_values_used":
                False,
            "external_future_used":
                False,
        },

        "candidate": {
            "v1_context_days":
                V1_CONTEXT,
            "v2_context_days":
                V2_CONTEXT,
            "alpha":
                ALPHA,
            "seeds":
                mod07a.SEEDS,
        },

        "bootstrap": {
            "unit":
                "query_date",
            "iterations":
                N_BOOTSTRAP,
            "seed":
                BOOTSTRAP_SEED,
        },

        "promotion_gate": {
            "passed":
                robustness_pass,

            "requirements": [
                "Fold-B Accuracy delta > 0",
                "Fold-B Macro-F1 delta > 0",
                "Fold-B bootstrap P(delta Accuracy > 0) >= 0.90",
                "Fold-B bootstrap P(delta Macro-F1 > 0) >= 0.90",
                "Fold-B Top-5 delta >= 0",
                "Fold-B MRR delta >= 0",
            ],
        },

        "next_if_pass":
            "promote context=5 candidate to MACRO-LTD-V2",

        "next_if_fail":
            "retain MACRO-LTD-V1 and continue targeted Macro research",

        "inputs": {
            "historical_sha256":
                sha256(historical_path),

            "daily_profiles_sha256":
                sha256(daily_path),

            "frozen_features_sha256":
                sha256(frozen_path),
        },
    }

    manifest_path = (
        out
        / "07D_01R_manifest_context_robustness.json"
    )

    write_json(
        manifest_path,
        manifest,
    )

    print()
    print("=" * 78)
    print("ROBUSTNESS SUMMARY")
    print("=" * 78)

    print(
        folds_df.to_string(
            index=False
        )
    )

    print()
    print(
        boot_df.to_string(
            index=False
        )
    )

    print()
    print(
        "MACRO-LTD-V2 PROMOTION PASS:",
        robustness_pass,
    )

    print()
    print(
        "INTERNAL_TEST remains CLOSED."
    )

    print(
        "Future-B remains CLOSED."
    )


if __name__ == "__main__":
    main()
