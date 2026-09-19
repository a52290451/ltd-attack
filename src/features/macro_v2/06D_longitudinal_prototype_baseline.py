from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

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
        "test_split":
            "DEV_MIDDLE",
    },
    {
        "fold": "B_EARLY_MIDDLE_TO_LATE",
        "train_splits": [
            "DEV_EARLY",
            "DEV_MIDDLE",
        ],
        "test_split":
            "DEV_LATE",
    },
]

PROTOTYPE_TYPES = [
    "capture_median",
    "daily_balanced_median",
]

DISTANCE_METRICS = [
    "cosine",
    "euclidean",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_frozen_features():
    path = (
        repo_root()
        / "configs/features/MACRO-V2-BASE-128.txt"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    features = [
        x.strip()
        for x in path.read_text().splitlines()
        if x.strip()
    ]

    if len(features) != 128:
        raise RuntimeError(
            f"Expected 128 frozen BASE features, "
            f"found {len(features)}."
        )

    if len(features) != len(set(features)):
        raise RuntimeError(
            "Duplicate frozen BASE features."
        )

    return features, path


def numeric_frame(
    df: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:

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
    train_df: pd.DataFrame,
    features: list[str],
):
    x = numeric_frame(
        train_df,
        features,
    )

    medians = x.median(
        axis=0,
        skipna=True,
    )

    all_missing = (
        medians[
            medians.isna()
        ]
        .index
        .tolist()
    )

    medians = (
        medians
        .fillna(0.0)
    )

    x_imp = (
        x
        .fillna(medians)
        .to_numpy(
            dtype=np.float64
        )
    )

    if not np.isfinite(
        x_imp
    ).all():
        raise RuntimeError(
            "Non-finite training values remain."
        )

    scaler = StandardScaler()

    scaler.fit(
        x_imp
    )

    return (
        medians,
        scaler,
        all_missing,
    )


def transform(
    df: pd.DataFrame,
    features: list[str],
    medians: pd.Series,
    scaler: StandardScaler,
) -> np.ndarray:

    x = numeric_frame(
        df,
        features,
    )

    x = (
        x
        .fillna(medians)
        .to_numpy(
            dtype=np.float64
        )
    )

    if not np.isfinite(
        x
    ).all():
        raise RuntimeError(
            "Non-finite values remain "
            "before scaling."
        )

    z = scaler.transform(
        x
    )

    if not np.isfinite(
        z
    ).all():
        raise RuntimeError(
            "Non-finite values after scaling."
        )

    return z


def build_prototype(
    values: np.ndarray,
    labels: np.ndarray,
    features: list[str],
) -> pd.DataFrame:

    frame = pd.DataFrame(
        values,
        columns=features,
    )

    frame.insert(
        0,
        "site_label",
        labels.astype(str),
    )

    prototype = (
        frame
        .groupby(
            "site_label",
            sort=True,
        )[
            features
        ]
        .median()
        .reset_index()
    )

    if len(prototype) != 65:
        raise RuntimeError(
            "Expected 65 prototypes, "
            f"found {len(prototype)}."
        )

    return prototype


def score_matrix(
    queries: np.ndarray,
    prototypes: np.ndarray,
    metric: str,
) -> np.ndarray:

    if metric == "cosine":
        q_norm = np.linalg.norm(
            queries,
            axis=1,
            keepdims=True,
        )

        p_norm = np.linalg.norm(
            prototypes,
            axis=1,
            keepdims=True,
        )

        q_norm = np.where(
            q_norm > 0,
            q_norm,
            1.0,
        )

        p_norm = np.where(
            p_norm > 0,
            p_norm,
            1.0,
        )

        q = queries / q_norm
        p = prototypes / p_norm

        return q @ p.T

    if metric == "euclidean":
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

        dist2 = (
            q2
            + p2
            - 2.0
            * (
                queries
                @ prototypes.T
            )
        )

        dist2 = np.maximum(
            dist2,
            0.0,
        )

        # Higher score = better candidate.
        return -dist2

    raise ValueError(
        f"Unknown metric: {metric}"
    )


def evaluate_scores(
    scores: np.ndarray,
    true_labels: np.ndarray,
    prototype_labels: np.ndarray,
):
    prototype_labels = (
        prototype_labels
        .astype(str)
    )

    true_labels = (
        true_labels
        .astype(str)
    )

    if len(
        set(
            prototype_labels
        )
    ) != 65:
        raise RuntimeError(
            "Prototype label set is not 65."
        )

    label_to_idx = {
        label: idx
        for idx, label
        in enumerate(
            prototype_labels
        )
    }

    missing_true = sorted(
        set(true_labels)
        - set(prototype_labels)
    )

    if missing_true:
        raise RuntimeError(
            "Query contains classes without "
            f"prototype: {missing_true}"
        )

    order = np.argsort(
        -scores,
        axis=1,
    )

    pred_idx = order[
        :,
        0
    ]

    pred_labels = (
        prototype_labels[
            pred_idx
        ]
    )

    true_idx = np.array(
        [
            label_to_idx[label]
            for label in true_labels
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

    accuracy = accuracy_score(
        true_labels,
        pred_labels,
    )

    macro_f1 = f1_score(
        true_labels,
        pred_labels,
        average="macro",
        zero_division=0,
    )

    top5 = float(
        np.mean(
            ranks <= 5
        )
    )

    mrr = float(
        np.mean(
            1.0 / ranks
        )
    )

    mean_rank = float(
        np.mean(
            ranks
        )
    )

    median_rank = float(
        np.median(
            ranks
        )
    )

    return {
        "accuracy":
            float(accuracy),

        "macro_f1":
            float(macro_f1),

        "top5_accuracy":
            top5,

        "mrr":
            mrr,

        "mean_true_rank":
            mean_rank,

        "median_true_rank":
            median_rank,
    }, pred_labels, ranks


def main():
    print("=" * 78)
    print(
        "MACRO-V2-HIST — "
        "06D LONGITUDINAL PROTOTYPE BASELINE"
    )
    print("=" * 78)

    print(
        "Purpose: test whether historical "
        "site context helps identify later "
        "observations."
    )

    print(
        "INTERNAL_TEST: PROHIBITED"
    )

    print(
        "External Future: PROHIBITED"
    )

    print(
        "No temporal encoder training."
    )

    print(
        "Every query is scored against "
        "ALL 65 candidate site prototypes."
    )

    (
        features,
        frozen_path,
    ) = load_frozen_features()

    out = output_dir()

    daily_path = (
        out
        / "06C_daily_base_profiles.csv"
    )

    assignments_path = (
        out
        / "01_temporal_split_assignments.csv"
    )

    if not daily_path.exists():
        raise FileNotFoundError(
            daily_path
        )

    if not assignments_path.exists():
        raise FileNotFoundError(
            assignments_path
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

    (
        captures,
        dataset_path,
    ) = load_historical_65()

    assignments = pd.read_csv(
        assignments_path
    )

    if assignments[
        "pcap_uid"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate pcap_uid in assignments."
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

    if len(captures) != 57916:
        raise RuntimeError(
            "Expected 57,916 DEV captures, "
            f"found {len(captures)}."
        )

    if captures[
        "site_label"
    ].nunique() != 65:
        raise RuntimeError(
            "Expected 65 DEV sites."
        )

    missing_features = [
        f
        for f in features
        if (
            f not in captures.columns
            or
            f not in daily.columns
        )
    ]

    if missing_features:
        raise RuntimeError(
            f"Missing frozen features: "
            f"{missing_features}"
        )

    metrics_rows = []
    per_site_rows = []
    prototype_rows = []

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

        test_daily = (
            daily[
                daily[
                    "temporal_split"
                ]
                == test_split
            ]
            .copy()
        )

        for name, frame in [
            ("train_caps", train_caps),
            ("test_caps", test_caps),
            ("train_daily", train_daily),
            ("test_daily", test_daily),
        ]:
            if frame.empty:
                raise RuntimeError(
                    f"{fold}: {name} is empty."
                )

            if frame[
                "site_label"
            ].nunique() != 65:
                raise RuntimeError(
                    f"{fold}: {name} does not "
                    "contain all 65 sites."
                )

        (
            medians,
            scaler,
            all_missing,
        ) = fit_preprocessing(
            train_caps,
            features,
        )

        z_train_caps = transform(
            train_caps,
            features,
            medians,
            scaler,
        )

        z_test_caps = transform(
            test_caps,
            features,
            medians,
            scaler,
        )

        z_train_daily = transform(
            train_daily,
            features,
            medians,
            scaler,
        )

        z_test_daily = transform(
            test_daily,
            features,
            medians,
            scaler,
        )

        prototypes = {
            "capture_median":
                build_prototype(
                    z_train_caps,
                    train_caps[
                        "site_label"
                    ].to_numpy(),
                    features,
                ),

            "daily_balanced_median":
                build_prototype(
                    z_train_daily,
                    train_daily[
                        "site_label"
                    ].to_numpy(),
                    features,
                ),
        }

        print(
            "Train captures:",
            len(train_caps),
        )

        print(
            "Train daily profiles:",
            len(train_daily),
        )

        print(
            "Test captures:",
            len(test_caps),
        )

        print(
            "Test daily profiles:",
            len(test_daily),
        )

        print(
            "All-missing training features:",
            len(all_missing),
        )

        for (
            prototype_type,
            prototype_df,
        ) in prototypes.items():

            prototype_labels = (
                prototype_df[
                    "site_label"
                ]
                .astype(str)
                .to_numpy()
            )

            prototype_matrix = (
                prototype_df[
                    features
                ]
                .to_numpy(
                    dtype=np.float64
                )
            )

            for _, row in (
                prototype_df
                .iterrows()
            ):
                proto_record = {
                    "fold":
                        fold,

                    "prototype_type":
                        prototype_type,

                    "site_label":
                        str(
                            row[
                                "site_label"
                            ]
                        ),
                }

                for feature in features:
                    proto_record[
                        feature
                    ] = float(
                        row[
                            feature
                        ]
                    )

                prototype_rows.append(
                    proto_record
                )

            queries = {
                "capture": (
                    z_test_caps,
                    test_caps[
                        "site_label"
                    ]
                    .astype(str)
                    .to_numpy(),
                ),

                "daily_profile": (
                    z_test_daily,
                    test_daily[
                        "site_label"
                    ]
                    .astype(str)
                    .to_numpy(),
                ),
            }

            for (
                query_type,
                (
                    query_matrix,
                    query_labels,
                ),
            ) in queries.items():

                for metric in DISTANCE_METRICS:

                    scores = score_matrix(
                        query_matrix,
                        prototype_matrix,
                        metric,
                    )

                    (
                        metrics,
                        predictions,
                        ranks,
                    ) = evaluate_scores(
                        scores,
                        query_labels,
                        prototype_labels,
                    )

                    metrics_rows.append(
                        {
                            "fold":
                                fold,

                            "train_splits":
                                "|".join(
                                    train_splits
                                ),

                            "test_split":
                                test_split,

                            "prototype_type":
                                prototype_type,

                            "query_type":
                                query_type,

                            "distance_metric":
                                metric,

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
                                    query_labels
                                ),

                            "all_missing_train_features":
                                len(
                                    all_missing
                                ),

                            **metrics,
                        }
                    )

                    for site in sorted(
                        set(
                            query_labels
                        )
                    ):
                        mask = (
                            query_labels
                            == site
                        )

                        site_true = (
                            query_labels[
                                mask
                            ]
                        )

                        site_pred = (
                            predictions[
                                mask
                            ]
                        )

                        site_ranks = (
                            ranks[
                                mask
                            ]
                        )

                        per_site_rows.append(
                            {
                                "fold":
                                    fold,

                                "prototype_type":
                                    prototype_type,

                                "query_type":
                                    query_type,

                                "distance_metric":
                                    metric,

                                "site_label":
                                    site,

                                "n_queries":
                                    int(
                                        mask.sum()
                                    ),

                                "top1_accuracy":
                                    float(
                                        np.mean(
                                            site_pred
                                            == site_true
                                        )
                                    ),

                                "top5_accuracy":
                                    float(
                                        np.mean(
                                            site_ranks
                                            <= 5
                                        )
                                    ),

                                "mrr":
                                    float(
                                        np.mean(
                                            1.0
                                            / site_ranks
                                        )
                                    ),

                                "mean_true_rank":
                                    float(
                                        np.mean(
                                            site_ranks
                                        )
                                    ),
                            }
                        )

                    print(
                        f"{prototype_type:22s} "
                        f"{query_type:13s} "
                        f"{metric:9s} "
                        f"acc={metrics['accuracy']:.4f} "
                        f"F1={metrics['macro_f1']:.4f} "
                        f"Top5={metrics['top5_accuracy']:.4f} "
                        f"MRR={metrics['mrr']:.4f} "
                        f"rank={metrics['mean_true_rank']:.2f}"
                    )

    metrics_df = pd.DataFrame(
        metrics_rows
    )

    per_site_df = pd.DataFrame(
        per_site_rows
    )

    prototypes_df = pd.DataFrame(
        prototype_rows
    )

    metrics_path = (
        out
        / "06D_prototype_metrics.csv"
    )

    per_site_path = (
        out
        / "06D_prototype_per_site_metrics.csv"
    )

    prototypes_path = (
        out
        / "06D_site_prototypes.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    per_site_df.to_csv(
        per_site_path,
        index=False,
    )

    prototypes_df.to_csv(
        prototypes_path,
        index=False,
    )

    comparison_rows = []

    keys = [
        "fold",
        "query_type",
        "distance_metric",
    ]

    for key_values, group in (
        metrics_df
        .groupby(
            keys,
            sort=True,
        )
    ):
        if len(group) != 2:
            continue

        by_type = (
            group
            .set_index(
                "prototype_type"
            )
        )

        if not all(
            x in by_type.index
            for x in PROTOTYPE_TYPES
        ):
            continue

        static = by_type.loc[
            "capture_median"
        ]

        longitudinal = by_type.loc[
            "daily_balanced_median"
        ]

        comparison_rows.append(
            {
                "fold":
                    key_values[0],

                "query_type":
                    key_values[1],

                "distance_metric":
                    key_values[2],

                "delta_accuracy_daily_vs_capture":
                    float(
                        longitudinal[
                            "accuracy"
                        ]
                        -
                        static[
                            "accuracy"
                        ]
                    ),

                "delta_macro_f1_daily_vs_capture":
                    float(
                        longitudinal[
                            "macro_f1"
                        ]
                        -
                        static[
                            "macro_f1"
                        ]
                    ),

                "delta_top5_daily_vs_capture":
                    float(
                        longitudinal[
                            "top5_accuracy"
                        ]
                        -
                        static[
                            "top5_accuracy"
                        ]
                    ),

                "delta_mrr_daily_vs_capture":
                    float(
                        longitudinal[
                            "mrr"
                        ]
                        -
                        static[
                            "mrr"
                        ]
                    ),
            }
        )

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    comparison_path = (
        out
        / "06D_daily_vs_capture_prototype_delta.csv"
    )

    comparison_df.to_csv(
        comparison_path,
        index=False,
    )

    manifest = {
        "feature_set_id":
            "MACRO-V2-BASE-128",

        "stage":
            "06D_longitudinal_prototype_baseline",

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
                    len(features),
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
            "development_splits_used":
                DEV_SPLITS,

            "internal_test_values_used":
                False,

            "external_future_used":
                False,

            "query_true_label_used_to_choose_prototype":
                False,

            "query_scored_against_all_65_candidates":
                True,
        },

        "preprocessing": {
            "imputation":
                (
                    "feature median fitted only "
                    "on fold training captures"
                ),

            "scaling":
                (
                    "StandardScaler fitted only "
                    "on fold training captures"
                ),
        },

        "prototype_types": {
            "capture_median":
                (
                    "coordinate-wise median of "
                    "standardized training captures "
                    "for each site"
                ),

            "daily_balanced_median":
                (
                    "coordinate-wise median of "
                    "standardized site-day BASE "
                    "profiles for each site; "
                    "each observed day contributes "
                    "one profile"
                ),
        },

        "distance_metrics":
            DISTANCE_METRICS,

        "query_types": [
            "capture",
            "daily_profile",
        ],

        "temporal_folds":
            FOLDS,

        "metrics": [
            "accuracy",
            "macro_f1",
            "top5_accuracy",
            "mrr",
            "mean_true_rank",
            "median_true_rank",
        ],

        "interpretation": {
            "purpose":
                (
                    "diagnostic baseline for "
                    "candidate-conditioned "
                    "longitudinal matching"
                ),

            "not_final_architecture":
                True,

            "no_model_training":
                True,

            "no_new_feature_selection":
                True,

            "next_stage":
                (
                    "use 06D evidence to decide "
                    "whether to build a learned "
                    "Inter-Day Encoder / "
                    "Cross-Attention model"
                ),
        },
    }

    write_json(
        out
        / "06D_manifest_longitudinal_prototype_baseline.json",
        manifest,
    )

    print()
    print("=" * 78)
    print(
        "06D METRICS"
    )
    print("=" * 78)

    print(
        metrics_df[
            [
                "fold",
                "prototype_type",
                "query_type",
                "distance_metric",
                "accuracy",
                "macro_f1",
                "top5_accuracy",
                "mrr",
                "mean_true_rank",
            ]
        ]
        .sort_values(
            [
                "fold",
                "query_type",
                "distance_metric",
                "prototype_type",
            ]
        )
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 78)
    print(
        "DAILY-BALANCED PROTOTYPE DELTA"
    )
    print("=" * 78)

    print(
        comparison_df.to_string(
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
        f"Comparison: {comparison_path}"
    )

    print(
        "06D LONGITUDINAL PROTOTYPE "
        "BASELINE COMPLETE"
    )


if __name__ == "__main__":
    main()
