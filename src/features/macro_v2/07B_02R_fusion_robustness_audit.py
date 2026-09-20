from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from _common import (
    git_commit,
    output_dir,
    sha256,
    write_json,
)

FOLDS = [
    "A_EARLY_TO_MIDDLE",
    "B_EARLY_MIDDLE_TO_LATE",
]

N_CLASSES = 65
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20260920
ALPHA_LOCAL_TOL_F1 = 0.001
EPS = 1e-12


def fuse(xgb_probs, ltd_probs, alpha):
    score = (
        (1.0 - alpha)
        * np.log(
            np.clip(
                xgb_probs,
                EPS,
                1.0,
            )
        )
        +
        alpha
        * np.log(
            np.clip(
                ltd_probs,
                EPS,
                1.0,
            )
        )
    )

    score -= score.max(
        axis=1,
        keepdims=True,
    )

    p = np.exp(score)

    p /= p.sum(
        axis=1,
        keepdims=True,
    )

    return p


def predictions(probs, y):
    order = np.argsort(
        -probs,
        axis=1,
    )

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
            float(
                np.mean(
                    yy == pp
                )
            ),

        "macro_f1":
            float(
                f1_score(
                    yy,
                    pp,
                    labels=np.arange(
                        N_CLASSES
                    ),
                    average="macro",
                    zero_division=0,
                )
            ),

        "top5_accuracy":
            float(
                np.mean(
                    rr <= 5
                )
            ),

        "mrr":
            float(
                np.mean(
                    1.0 / rr
                )
            ),
    }


def delta_metrics(
    y,
    x_pred,
    x_rank,
    f_pred,
    f_rank,
    idx=None,
):
    a = metrics(
        y,
        x_pred,
        x_rank,
        idx,
    )

    b = metrics(
        y,
        f_pred,
        f_rank,
        idx,
    )

    return {
        k: b[k] - a[k]
        for k in a
    }


def block_bootstrap(
    y,
    dates,
    x_pred,
    x_rank,
    f_pred,
    f_rank,
):
    unique_dates = np.array(
        sorted(
            pd.unique(dates)
        )
    )

    date_indices = {
        d: np.flatnonzero(
            dates == d
        )
        for d in unique_dates
    }

    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    rows = []

    for _ in range(
        N_BOOTSTRAP
    ):
        sampled_dates = rng.choice(
            unique_dates,
            size=len(unique_dates),
            replace=True,
        )

        idx = np.concatenate(
            [
                date_indices[d]
                for d in sampled_dates
            ]
        )

        rows.append(
            delta_metrics(
                y,
                x_pred,
                x_rank,
                f_pred,
                f_rank,
                idx,
            )
        )

    return pd.DataFrame(
        rows
    )


def main():
    print("=" * 78)
    print(
        "07B-2R — FIXED FUSION ROBUSTNESS AUDIT"
    )
    print("=" * 78)

    print(
        "No architecture tuning."
    )

    print(
        "No alpha tuning."
    )

    print(
        "INTERNAL_TEST: CLOSED"
    )

    print(
        "Future-B: CLOSED"
    )

    out = output_dir()

    manifest_path = (
        out
        / "07B_02_manifest_geometric_fusion.json"
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            manifest_path
        )

    with open(
        manifest_path,
        "r",
        encoding="utf-8",
    ) as f:
        fusion_manifest = json.load(f)

    alpha = float(
        fusion_manifest[
            "fusion"
        ][
            "selected_alpha"
        ]
    )

    if alpha != 0.375:
        raise RuntimeError(
            f"Expected frozen alpha=0.375, got {alpha}"
        )

    curve_path = (
        out
        / "07B_02_alpha_curve_fold_A.csv"
    )

    curve = pd.read_csv(
        curve_path
    )

    best_f1 = float(
        curve[
            "macro_f1"
        ].max()
    )

    local = curve[
        curve[
            "macro_f1"
        ]
        >=
        (
            best_f1
            -
            ALPHA_LOCAL_TOL_F1
        )
    ]

    alpha_stability = {
        "best_fold_a_macro_f1":
            best_f1,

        "tolerance":
            ALPHA_LOCAL_TOL_F1,

        "stable_alpha_min":
            float(
                local[
                    "alpha"
                ].min()
            ),

        "stable_alpha_max":
            float(
                local[
                    "alpha"
                ].max()
            ),

        "stable_alpha_count":
            int(
                len(local)
            ),
    }

    paired_path = (
        out
        / "07B_01_paired_predictions.csv"
    )

    paired = pd.read_csv(
        paired_path
    )

    daily_rows = []
    class_rows = []
    bootstrap_rows = []
    fold_rows = []

    for fold in FOLDS:
        score_path = (
            out
            / f"07B_01_scores_{fold}.npz"
        )

        # 07B-1 stores pcap_uid metadata through pandas -> NumPy.
        # That array is serialized as dtype=object inside the trusted,
        # pipeline-generated NPZ artifact. Numerical score tensors remain
        # unchanged. allow_pickle=True is therefore required only to read
        # this internally generated metadata.
        data = np.load(
            score_path,
            allow_pickle=True,
        )

        xgb_probs = data[
            "xgb_probs"
        ].astype(
            np.float64
        )

        ltd_probs = data[
            "ltd_ensemble_probs"
        ].astype(
            np.float64
        )

        y = data[
            "y_true"
        ].astype(int)

        pcap_uid = (
            data[
                "pcap_uid"
            ]
            .astype(str)
        )

        labels = (
            data[
                "candidate_labels"
            ]
            .astype(str)
        )

        fold_paired = (
            paired[
                paired[
                    "fold"
                ]
                == fold
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        if len(
            fold_paired
        ) != len(y):
            raise RuntimeError(
                f"{fold}: paired length mismatch"
            )

        if not np.array_equal(
            fold_paired[
                "pcap_uid"
            ].astype(str).to_numpy(),
            pcap_uid,
        ):
            raise RuntimeError(
                f"{fold}: pcap_uid order mismatch"
            )

        dates = (
            pd.to_datetime(
                fold_paired[
                    "query_date"
                ]
            )
            .dt.strftime(
                "%Y-%m-%d"
            )
            .to_numpy()
        )

        fusion_probs = fuse(
            xgb_probs,
            ltd_probs,
            alpha,
        )

        x_pred, x_rank = predictions(
            xgb_probs,
            y,
        )

        f_pred, f_rank = predictions(
            fusion_probs,
            y,
        )

        observed = delta_metrics(
            y,
            x_pred,
            x_rank,
            f_pred,
            f_rank,
        )

        boot = block_bootstrap(
            y,
            dates,
            x_pred,
            x_rank,
            f_pred,
            f_rank,
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
                        observed[
                            metric
                        ],

                    "bootstrap_mean":
                        float(
                            values.mean()
                        ),

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

        improved_days = 0
        worse_days = 0
        tied_days = 0

        for date in sorted(
            pd.unique(dates)
        ):
            idx = np.flatnonzero(
                dates == date
            )

            d = delta_metrics(
                y,
                x_pred,
                x_rank,
                f_pred,
                f_rank,
                idx,
            )

            if d[
                "accuracy"
            ] > 0:
                improved_days += 1
            elif d[
                "accuracy"
            ] < 0:
                worse_days += 1
            else:
                tied_days += 1

            daily_rows.append(
                {
                    "fold":
                        fold,

                    "date":
                        date,

                    "n_queries":
                        len(idx),

                    **{
                        f"delta_{k}": v
                        for k, v in d.items()
                    },
                }
            )

        x_correct = (
            x_pred == y
        )

        f_correct = (
            f_pred == y
        )

        for cls in range(
            N_CLASSES
        ):
            idx = np.flatnonzero(
                y == cls
            )

            xc = x_correct[
                idx
            ]

            fc = f_correct[
                idx
            ]

            class_rows.append(
                {
                    "fold":
                        fold,

                    "site_label":
                        labels[
                            cls
                        ],

                    "n_queries":
                        len(idx),

                    "xgb_accuracy":
                        float(
                            xc.mean()
                        ),

                    "fusion_accuracy":
                        float(
                            fc.mean()
                        ),

                    "delta_accuracy":
                        float(
                            fc.mean()
                            -
                            xc.mean()
                        ),

                    "fusion_only_correct_n":
                        int(
                            (
                                ~xc
                                & fc
                            ).sum()
                        ),

                    "xgb_only_correct_n":
                        int(
                            (
                                xc
                                & ~fc
                            ).sum()
                        ),

                    "net_correct_n":
                        int(
                            (
                                ~xc
                                & fc
                            ).sum()
                            -
                            (
                                xc
                                & ~fc
                            ).sum()
                        ),
                }
            )

        fold_rows.append(
            {
                "fold":
                    fold,

                "alpha":
                    alpha,

                "n_dates":
                    len(
                        pd.unique(
                            dates
                        )
                    ),

                "delta_accuracy":
                    observed[
                        "accuracy"
                    ],

                "delta_macro_f1":
                    observed[
                        "macro_f1"
                    ],

                "delta_top5":
                    observed[
                        "top5_accuracy"
                    ],

                "delta_mrr":
                    observed[
                        "mrr"
                    ],

                "accuracy_improved_days":
                    improved_days,

                "accuracy_worse_days":
                    worse_days,

                "accuracy_tied_days":
                    tied_days,
            }
        )

    bootstrap_df = pd.DataFrame(
        bootstrap_rows
    )

    daily_df = pd.DataFrame(
        daily_rows
    )

    classes_df = pd.DataFrame(
        class_rows
    )

    folds_df = pd.DataFrame(
        fold_rows
    )

    fold_b_boot = (
        bootstrap_df[
            bootstrap_df[
                "fold"
            ]
            == "B_EARLY_MIDDLE_TO_LATE"
        ]
        .set_index(
            "metric"
        )
    )

    robustness_pass = bool(
        (
            fold_b_boot.loc[
                "accuracy",
                "observed_delta",
            ]
            > 0
        )
        and
        (
            fold_b_boot.loc[
                "macro_f1",
                "observed_delta",
            ]
            > 0
        )
        and
        (
            fold_b_boot.loc[
                "accuracy",
                "p_delta_gt_0",
            ]
            >= 0.90
        )
        and
        (
            fold_b_boot.loc[
                "macro_f1",
                "p_delta_gt_0",
            ]
            >= 0.90
        )
    )

    bootstrap_path = (
        out
        / "07B_02R_day_block_bootstrap.csv"
    )

    daily_path = (
        out
        / "07B_02R_per_day_deltas.csv"
    )

    classes_path = (
        out
        / "07B_02R_per_class_deltas.csv"
    )

    summary_path = (
        out
        / "07B_02R_robustness_summary.csv"
    )

    bootstrap_df.to_csv(
        bootstrap_path,
        index=False,
    )

    daily_df.to_csv(
        daily_path,
        index=False,
    )

    classes_df.to_csv(
        classes_path,
        index=False,
    )

    folds_df.to_csv(
        summary_path,
        index=False,
    )

    manifest = {
        "stage":
            "07B_02R_fusion_robustness_audit",

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "git_commit":
            git_commit(),

        "data_policy": {
            "development_only":
                True,

            "internal_test_values_used":
                False,

            "external_future_used":
                False,

            "architecture_tuning":
                False,

            "alpha_tuning":
                False,
        },

        "frozen_candidate": {
            "alpha":
                alpha,

            "fusion":
                "weighted geometric probability",

            "context_days":
                7,

            "base_features":
                128,

            "ltd_seeds":
                [
                    11,
                    42,
                    73,
                ],
        },

        "robustness": {
            "bootstrap_unit":
                "query_date",

            "bootstrap_iterations":
                N_BOOTSTRAP,

            "bootstrap_seed":
                BOOTSTRAP_SEED,

            "alpha_local_stability":
                alpha_stability,
        },

        "freeze_gate": {
            "fold":
                "B_EARLY_MIDDLE_TO_LATE",

            "requirements": [
                "observed accuracy delta > 0",
                "observed Macro-F1 delta > 0",
                "P_day_bootstrap(delta accuracy > 0) >= 0.90",
                "P_day_bootstrap(delta Macro-F1 > 0) >= 0.90",
            ],

            "passed":
                robustness_pass,
        },

        "next_if_pass":
            "FINAL_ARCHITECTURE_FREEZE",

        "next_if_fail":
            "review fusion robustness before any new architecture experiment",
    }

    manifest_out = (
        out
        / "07B_02R_manifest_robustness_audit.json"
    )

    write_json(
        manifest_out,
        manifest,
    )

    print()
    print("=" * 78)
    print(
        "ROBUSTNESS SUMMARY"
    )
    print("=" * 78)

    print(
        folds_df.to_string(
            index=False
        )
    )

    print()
    print(
        bootstrap_df.to_string(
            index=False
        )
    )

    print()
    print(
        "Alpha local stability:",
        alpha_stability,
    )

    print()
    print(
        "ROBUSTNESS PASS:",
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
