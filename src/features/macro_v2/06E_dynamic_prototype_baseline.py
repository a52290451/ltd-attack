from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re

import numpy as np
import pandas as pd

import sklearn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)
from sklearn.preprocessing import StandardScaler

from _common import (
    git_commit,
    load_historical_65,
    output_dir,
    sha256,
    write_json,
)


DEV_SPLITS = [
    "DEV_EARLY",
    "DEV_MIDDLE",
    "DEV_LATE",
]

FOLDS = [
    {
        "fold": "A_EARLY_TO_MIDDLE",
        "train_splits": [
            "DEV_EARLY",
        ],
        "test_split": "DEV_MIDDLE",
    },
    {
        "fold": "B_EARLY_MIDDLE_TO_LATE",
        "train_splits": [
            "DEV_EARLY",
            "DEV_MIDDLE",
        ],
        "test_split": "DEV_LATE",
    },
]

PROTOTYPE_METHODS = [
    "static_capture_median",
    "all_days_median",
    "recent_7_days_median",
    "recent_3_days_median",
    "last_day",
    "linear_trend",
]

DISTANCE_METRIC = "euclidean"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_frozen_features():
    path = (
        repo_root()
        / "configs/features/MACRO-V2-BASE-128.txt"
    )

    features = [
        x.strip()
        for x in path.read_text().splitlines()
        if x.strip()
    ]

    if len(features) != 128:
        raise RuntimeError(
            f"Expected 128 BASE features, "
            f"found {len(features)}."
        )

    if len(features) != len(set(features)):
        raise RuntimeError(
            "Duplicate BASE features."
        )

    return features, path


def normalize_date(value):
    if pd.isna(value):
        return None

    text = str(value).strip()

    parsed = pd.to_datetime(
        text,
        errors="coerce",
    )

    if not pd.isna(parsed):
        return parsed.normalize()

    match = re.search(
        r"(\d{8})",
        text,
    )

    if match:
        parsed = pd.to_datetime(
            match.group(1),
            format="%Y%m%d",
            errors="coerce",
        )

        if not pd.isna(parsed):
            return parsed.normalize()

    return None


def derive_dates(df):
    for col in [
        "_audit_date",
        "date",
        "date_id",
        "capture_date",
        "pcap_name",
    ]:
        if col not in df.columns:
            continue

        values = df[col].apply(
            normalize_date
        )

        if values.notna().mean() > 0.95:
            return values, col

    raise RuntimeError(
        "Could not derive capture dates."
    )


def numeric_frame(
    df,
    features,
):
    return (
        df[features]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )


def fit_preprocessing(
    train_df,
    features,
):
    x = numeric_frame(
        train_df,
        features,
    )

    medians = (
        x
        .median(
            axis=0,
            skipna=True,
        )
        .fillna(0.0)
    )

    x_imp = (
        x
        .fillna(medians)
        .to_numpy(
            dtype=np.float64
        )
    )

    scaler = StandardScaler()

    scaler.fit(
        x_imp
    )

    return medians, scaler


def transform(
    df,
    features,
    medians,
    scaler,
):
    x = (
        numeric_frame(
            df,
            features,
        )
        .fillna(medians)
        .to_numpy(
            dtype=np.float64
        )
    )

    z = scaler.transform(
        x
    )

    if not np.isfinite(z).all():
        raise RuntimeError(
            "Non-finite standardized values."
        )

    return z


def euclidean_scores(
    queries,
    prototypes,
):
    q2 = np.sum(
        queries ** 2,
        axis=1,
        keepdims=True,
    )

    p2 = np.sum(
        prototypes ** 2,
        axis=1,
        keepdims=True,
    ).T

    d2 = (
        q2
        + p2
        - 2.0
        * (
            queries
            @ prototypes.T
        )
    )

    return -np.maximum(
        d2,
        0.0,
    )


def evaluate(
    scores,
    true_labels,
    prototype_labels,
):
    true_labels = np.asarray(
        true_labels,
        dtype=str,
    )

    prototype_labels = np.asarray(
        prototype_labels,
        dtype=str,
    )

    order = np.argsort(
        -scores,
        axis=1,
    )

    pred = prototype_labels[
        order[:, 0]
    ]

    label_to_idx = {
        label: i
        for i, label
        in enumerate(
            prototype_labels
        )
    }

    true_idx = np.array(
        [
            label_to_idx[x]
            for x in true_labels
        ],
        dtype=int,
    )

    ranks = (
        np.argmax(
            order
            == true_idx[:, None],
            axis=1,
        )
        + 1
    )

    return {
        "accuracy":
            float(
                accuracy_score(
                    true_labels,
                    pred,
                )
            ),

        "macro_f1":
            float(
                f1_score(
                    true_labels,
                    pred,
                    average="macro",
                    zero_division=0,
                )
            ),

        "top5_accuracy":
            float(
                np.mean(
                    ranks <= 5
                )
            ),

        "mrr":
            float(
                np.mean(
                    1.0 / ranks
                )
            ),

        "mean_true_rank":
            float(
                np.mean(
                    ranks
                )
            ),

        "median_true_rank":
            float(
                np.median(
                    ranks
                )
            ),
    }


def build_static_capture_prototypes(
    z_train,
    labels,
    features,
):
    frame = pd.DataFrame(
        z_train,
        columns=features,
    )

    frame.insert(
        0,
        "site_label",
        np.asarray(
            labels,
            dtype=str,
        ),
    )

    result = (
        frame
        .groupby(
            "site_label",
            sort=True,
        )[features]
        .median()
    )

    return result


def prepare_daily_history(
    train_daily,
    features,
    medians,
    scaler,
):
    z = transform(
        train_daily,
        features,
        medians,
        scaler,
    )

    hist = pd.DataFrame(
        z,
        columns=features,
    )

    hist.insert(
        0,
        "date",
        pd.to_datetime(
            train_daily[
                "date"
            ]
        ).to_numpy(),
    )

    hist.insert(
        0,
        "site_label",
        train_daily[
            "site_label"
        ]
        .astype(str)
        .to_numpy(),
    )

    return (
        hist
        .sort_values(
            [
                "site_label",
                "date",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def make_time_conditioned_prototypes(
    method,
    test_date,
    static_capture,
    daily_history,
    features,
):
    site_labels = (
        static_capture
        .index
        .astype(str)
        .tolist()
    )

    if method == "static_capture_median":
        return (
            np.array(site_labels),
            static_capture[
                features
            ]
            .to_numpy(
                dtype=np.float64
            ),
        )

    rows = []

    test_date = pd.Timestamp(
        test_date
    )

    for site in site_labels:
        h = (
            daily_history[
                daily_history[
                    "site_label"
                ]
                == site
            ]
            .sort_values(
                "date"
            )
        )

        if h.empty:
            raise RuntimeError(
                f"No history for site {site}"
            )

        if (
            h["date"]
            >= test_date
        ).any():
            raise RuntimeError(
                "Temporal leakage: history "
                "contains date >= test date."
            )

        values = h[
            features
        ].to_numpy(
            dtype=np.float64
        )

        if method == "all_days_median":
            vector = np.median(
                values,
                axis=0,
            )

        elif method == "recent_7_days_median":
            vector = np.median(
                values[-7:],
                axis=0,
            )

        elif method == "recent_3_days_median":
            vector = np.median(
                values[-3:],
                axis=0,
            )

        elif method == "last_day":
            vector = values[-1]

        elif method == "linear_trend":
            dates = pd.to_datetime(
                h["date"]
            )

            base_date = dates.min()

            x = (
                (
                    dates
                    - base_date
                )
                .dt.days
                .to_numpy(
                    dtype=np.float64
                )
            )

            target_x = float(
                (
                    test_date
                    - base_date
                ).days
            )

            if len(x) < 2:
                vector = values[-1]
            else:
                x_mean = x.mean()

                centered = (
                    x
                    - x_mean
                )

                denom = np.sum(
                    centered ** 2
                )

                if denom <= 0:
                    vector = np.median(
                        values,
                        axis=0,
                    )
                else:
                    y_mean = values.mean(
                        axis=0
                    )

                    slopes = (
                        centered[:, None]
                        * (
                            values
                            - y_mean
                        )
                    ).sum(
                        axis=0
                    ) / denom

                    vector = (
                        y_mean
                        + slopes
                        * (
                            target_x
                            - x_mean
                        )
                    )

        else:
            raise ValueError(
                f"Unknown method {method}"
            )

        rows.append(
            vector
        )

    matrix = np.vstack(
        rows
    )

    if not np.isfinite(
        matrix
    ).all():
        raise RuntimeError(
            f"Non-finite prototype for {method}"
        )

    return (
        np.array(
            site_labels
        ),
        matrix,
    )


def main():
    print("=" * 78)
    print(
        "MACRO-V2-HIST — "
        "06E DYNAMIC PROTOTYPE BASELINE"
    )
    print("=" * 78)

    print(
        "Single-capture queries only."
    )

    print(
        "Distance metric frozen from 06D: "
        "Euclidean."
    )

    print(
        "INTERNAL_TEST: PROHIBITED"
    )

    print(
        "External Future: PROHIBITED"
    )

    features, frozen_path = (
        load_frozen_features()
    )

    out = output_dir()

    daily_path = (
        out
        / "06C_daily_base_profiles.csv"
    )

    assignments_path = (
        out
        / "01_temporal_split_assignments.csv"
    )

    daily = pd.read_csv(
        daily_path
    )

    daily[
        "site_label"
    ] = (
        daily[
            "site_label"
        ]
        .astype(str)
    )

    daily[
        "date"
    ] = pd.to_datetime(
        daily[
            "date"
        ]
    )

    (
        captures,
        dataset_path,
    ) = load_historical_65()

    assignments = pd.read_csv(
        assignments_path
    )

    dev_assignments = (
        assignments[
            assignments[
                "temporal_split"
            ].isin(
                DEV_SPLITS
            )
        ][
            [
                "pcap_uid",
                "temporal_split",
            ]
        ]
        .copy()
    )

    captures = (
        captures
        .merge(
            dev_assignments,
            on="pcap_uid",
            how="inner",
            validate="one_to_one",
        )
    )

    captures[
        "site_label"
    ] = (
        captures[
            "site_label"
        ]
        .astype(str)
    )

    capture_dates, date_source = (
        derive_dates(
            captures
        )
    )

    captures = captures.copy()

    captures[
        "_query_date"
    ] = pd.to_datetime(
        capture_dates
    )

    if captures[
        "_query_date"
    ].isna().any():
        raise RuntimeError(
            "Missing capture dates."
        )

    results = []

    for fold_spec in FOLDS:
        fold = fold_spec[
            "fold"
        ]

        train_splits = fold_spec[
            "train_splits"
        ]

        test_split = fold_spec[
            "test_split"
        ]

        print()
        print("=" * 78)
        print(fold)
        print("=" * 78)

        train_caps = (
            captures[
                captures[
                    "temporal_split"
                ].isin(
                    train_splits
                )
            ]
            .copy()
        )

        test_caps = (
            captures[
                captures[
                    "temporal_split"
                ]
                == test_split
            ]
            .copy()
        )

        train_daily = (
            daily[
                daily[
                    "temporal_split"
                ].isin(
                    train_splits
                )
            ]
            .copy()
        )

        if (
            train_caps[
                "site_label"
            ].nunique()
            != 65
        ):
            raise RuntimeError(
                "Training captures do not "
                "contain 65 sites."
            )

        if (
            train_daily[
                "site_label"
            ].nunique()
            != 65
        ):
            raise RuntimeError(
                "Training daily history does "
                "not contain 65 sites."
            )

        (
            medians,
            scaler,
        ) = fit_preprocessing(
            train_caps,
            features,
        )

        z_train = transform(
            train_caps,
            features,
            medians,
            scaler,
        )

        static_capture = (
            build_static_capture_prototypes(
                z_train,
                train_caps[
                    "site_label"
                ],
                features,
            )
        )

        daily_history = (
            prepare_daily_history(
                train_daily,
                features,
                medians,
                scaler,
            )
        )

        fold_predictions = {
            method: {
                "true": [],
                "scores": [],
                "labels": None,
            }
            for method in PROTOTYPE_METHODS
        }

        for test_date, day_queries in (
            test_caps
            .groupby(
                "_query_date",
                sort=True,
            )
        ):
            # Hard assertion:
            # all context must precede query day.
            if (
                train_daily[
                    "date"
                ].max()
                >= test_date
            ):
                raise RuntimeError(
                    "Train/test temporal boundary "
                    "violation."
                )

            z_query = transform(
                day_queries,
                features,
                medians,
                scaler,
            )

            true_labels = (
                day_queries[
                    "site_label"
                ]
                .astype(str)
                .to_numpy()
            )

            for method in PROTOTYPE_METHODS:
                (
                    proto_labels,
                    proto_matrix,
                ) = make_time_conditioned_prototypes(
                    method,
                    test_date,
                    static_capture,
                    daily_history,
                    features,
                )

                scores = euclidean_scores(
                    z_query,
                    proto_matrix,
                )

                store = (
                    fold_predictions[
                        method
                    ]
                )

                if store[
                    "labels"
                ] is None:
                    store[
                        "labels"
                    ] = proto_labels
                else:
                    if not np.array_equal(
                        store[
                            "labels"
                        ],
                        proto_labels,
                    ):
                        raise RuntimeError(
                            "Prototype label order "
                            "changed across dates."
                        )

                store[
                    "true"
                ].append(
                    true_labels
                )

                store[
                    "scores"
                ].append(
                    scores
                )

        for method in PROTOTYPE_METHODS:
            store = (
                fold_predictions[
                    method
                ]
            )

            true_labels = np.concatenate(
                store[
                    "true"
                ]
            )

            scores = np.vstack(
                store[
                    "scores"
                ]
            )

            metrics = evaluate(
                scores,
                true_labels,
                store[
                    "labels"
                ],
            )

            results.append(
                {
                    "fold":
                        fold,

                    "train_splits":
                        "|".join(
                            train_splits
                        ),

                    "test_split":
                        test_split,

                    "prototype_method":
                        method,

                    "distance_metric":
                        DISTANCE_METRIC,

                    "n_train_captures":
                        len(
                            train_caps
                        ),

                    "n_train_daily_profiles":
                        len(
                            train_daily
                        ),

                    "n_queries":
                        len(
                            true_labels
                        ),

                    **metrics,
                }
            )

            print(
                f"{method:24s} "
                f"acc={metrics['accuracy']:.4f} "
                f"F1={metrics['macro_f1']:.4f} "
                f"Top5={metrics['top5_accuracy']:.4f} "
                f"MRR={metrics['mrr']:.4f} "
                f"rank={metrics['mean_true_rank']:.2f}"
            )

    results_df = pd.DataFrame(
        results
    )

    metrics_path = (
        out
        / "06E_dynamic_prototype_metrics.csv"
    )

    results_df.to_csv(
        metrics_path,
        index=False,
    )

    static = (
        results_df[
            results_df[
                "prototype_method"
            ]
            == "static_capture_median"
        ][
            [
                "fold",
                "accuracy",
                "macro_f1",
                "top5_accuracy",
                "mrr",
            ]
        ]
        .rename(
            columns={
                "accuracy":
                    "static_accuracy",
                "macro_f1":
                    "static_macro_f1",
                "top5_accuracy":
                    "static_top5",
                "mrr":
                    "static_mrr",
            }
        )
    )

    deltas = (
        results_df
        .merge(
            static,
            on="fold",
            validate="many_to_one",
        )
    )

    deltas[
        "delta_accuracy_vs_static"
    ] = (
        deltas[
            "accuracy"
        ]
        -
        deltas[
            "static_accuracy"
        ]
    )

    deltas[
        "delta_macro_f1_vs_static"
    ] = (
        deltas[
            "macro_f1"
        ]
        -
        deltas[
            "static_macro_f1"
        ]
    )

    deltas[
        "delta_top5_vs_static"
    ] = (
        deltas[
            "top5_accuracy"
        ]
        -
        deltas[
            "static_top5"
        ]
    )

    deltas[
        "delta_mrr_vs_static"
    ] = (
        deltas[
            "mrr"
        ]
        -
        deltas[
            "static_mrr"
        ]
    )

    delta_path = (
        out
        / "06E_dynamic_prototype_delta.csv"
    )

    deltas.to_csv(
        delta_path,
        index=False,
    )

    summary = (
        results_df
        .groupby(
            "prototype_method",
            as_index=False,
        )
        .agg(
            worst_fold_macro_f1=(
                "macro_f1",
                "min",
            ),
            mean_macro_f1=(
                "macro_f1",
                "mean",
            ),
            worst_fold_top5=(
                "top5_accuracy",
                "min",
            ),
            mean_top5=(
                "top5_accuracy",
                "mean",
            ),
            worst_fold_mrr=(
                "mrr",
                "min",
            ),
            mean_mrr=(
                "mrr",
                "mean",
            ),
        )
        .sort_values(
            [
                "worst_fold_macro_f1",
                "mean_macro_f1",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    summary.insert(
        0,
        "rank",
        np.arange(
            1,
            len(summary) + 1,
        ),
    )

    summary_path = (
        out
        / "06E_dynamic_prototype_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    manifest = {
        "feature_set_id":
            "MACRO-V2-BASE-128",

        "stage":
            "06E_dynamic_prototype_baseline",

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "git_commit":
            git_commit(),

        "environment": {
            "sklearn":
                sklearn.__version__,
        },

        "inputs": {
            "historical_dataset": {
                "path":
                    str(
                        dataset_path
                    ),
                "sha256":
                    sha256(
                        dataset_path
                    ),
            },

            "daily_profiles": {
                "path":
                    str(
                        daily_path
                    ),
                "sha256":
                    sha256(
                        daily_path
                    ),
            },

            "frozen_base_features": {
                "path":
                    str(
                        frozen_path
                    ),
                "sha256":
                    sha256(
                        frozen_path
                    ),
                "count":
                    128,
            },

            "temporal_assignments": {
                "path":
                    str(
                        assignments_path
                    ),
                "sha256":
                    sha256(
                        assignments_path
                    ),
            },
        },

        "data_policy": {
            "development_only":
                True,

            "internal_test_values_used":
                False,

            "external_future_used":
                False,

            "test_observations_used_to_update_history":
                False,

            "query_true_label_used_to_choose_prototype":
                False,

            "query_scored_against_all_65_candidates":
                True,
        },

        "query_unit":
            "single_capture",

        "date_source":
            date_source,

        "distance_metric":
            DISTANCE_METRIC,

        "distance_metric_selection":
            (
                "Euclidean frozen after 06D, "
                "where it consistently exceeded "
                "cosine similarity."
            ),

        "prototype_methods":
            PROTOTYPE_METHODS,

        "temporal_folds":
            FOLDS,

        "interpretation": {
            "purpose":
                (
                    "test whether recency or "
                    "explicit temporal trend "
                    "improves candidate matching "
                    "over static historical "
                    "prototypes"
                ),

            "no_learned_temporal_encoder":
                True,

            "next_stage":
                (
                    "if temporal conditioning adds "
                    "signal, proceed to learned "
                    "candidate-conditioned temporal "
                    "encoder and fusion with "
                    "BASE-128"
                ),
        },
    }

    write_json(
        out
        / "06E_manifest_dynamic_prototype_baseline.json",
        manifest,
    )

    print()
    print("=" * 78)
    print(
        "SUMMARY"
    )
    print("=" * 78)

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        "INTERNAL_TEST remains CLOSED."
    )

    print(
        "External Future remains CLOSED."
    )

    print()
    print(
        f"Metrics: {metrics_path}"
    )

    print(
        f"Summary: {summary_path}"
    )

    print(
        f"Deltas: {delta_path}"
    )

    print(
        "06E DYNAMIC PROTOTYPE BASELINE COMPLETE"
    )


if __name__ == "__main__":
    main()
