from __future__ import annotations

from datetime import datetime, timezone
import platform
import time

import numpy as np
import pandas as pd

import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
)

try:
    import xgboost
    from xgboost import XGBClassifier
except ImportError as exc:
    raise RuntimeError(
        "Stage 06B requires xgboost. "
        "Do not silently change the model family."
    ) from exc

from _common import (
    git_commit,
    load_historical_65,
    output_dir,
    sha256,
    write_json,
)


SEED = 42

DEV_SPLITS = [
    "DEV_EARLY",
    "DEV_MIDDLE",
    "DEV_LATE",
]

SUBSET_SIZES = [
    8,
    16,
    24,
    32,
    48,
    64,
    96,
    128,
    148,
]

PRIMARY_TOLERANCE = 0.005
SECONDARY_TOLERANCE = 0.005

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
        "fold":
            "B_EARLY_MIDDLE_TO_LATE",
        "train_splits": [
            "DEV_EARLY",
            "DEV_MIDDLE",
        ],
        "test_split":
            "DEV_LATE",
    },
]


def load_representatives():
    out = output_dir()

    path = (
        out
        / "06A_cluster_representatives.csv"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    reps = pd.read_csv(path)

    required = [
        "feature",
        "cluster_id",
        "dtp_min",
        "dtp_geometric_mean",
        "dtp_arithmetic_mean",
        "dtp_balance_gap",
        "is_cluster_representative",
    ]

    missing = [
        x
        for x in required
        if x not in reps.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing 06A columns: {missing}"
        )

    if len(reps) != 148:
        raise RuntimeError(
            "Expected 148 representatives, "
            f"found {len(reps)}."
        )

    if reps["feature"].duplicated().any():
        raise RuntimeError(
            "Duplicate representative features."
        )

    representative_flag = (
        reps[
            "is_cluster_representative"
        ]
        .astype(str)
        .str.lower()
        .isin(
            ["true", "1"]
        )
    )

    if not representative_flag.all():
        raise RuntimeError(
            "06A file contains "
            "non-representatives."
        )

    for col in [
        "dtp_min",
        "dtp_geometric_mean",
        "dtp_arithmetic_mean",
        "dtp_balance_gap",
    ]:
        values = pd.to_numeric(
            reps[col],
            errors="coerce",
        )

        if (
            values.isna().any()
            or
            not np.isfinite(
                values.to_numpy(
                    dtype=float
                )
            ).all()
        ):
            raise RuntimeError(
                f"Invalid values in {col}."
            )

        reps[col] = values

    # IMPORTANT:
    # ignore Pareto ordering from 06A.
    # Reconstruct the predeclared robust order.
    reps = (
        reps
        .sort_values(
            [
                "dtp_min",
                "dtp_geometric_mean",
                "dtp_arithmetic_mean",
                "dtp_balance_gap",
                "feature",
            ],
            ascending=[
                False,
                False,
                False,
                True,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    reps.insert(
        0,
        "robust_order",
        np.arange(
            1,
            len(reps) + 1,
        ),
    )

    return reps, path


def load_dev():
    out = output_dir()

    assignments_path = (
        out
        / "01_temporal_split_assignments.csv"
    )

    eligibility_path = (
        out
        / "01A_candidate_features_dev_eligible.txt"
    )

    if not assignments_path.exists():
        raise FileNotFoundError(
            assignments_path
        )

    if not eligibility_path.exists():
        raise FileNotFoundError(
            eligibility_path
        )

    assignments = pd.read_csv(
        assignments_path,
        dtype={
            "site_label": str,
        },
    )

    if (
        "pcap_uid"
        not in assignments.columns
        or
        "temporal_split"
        not in assignments.columns
    ):
        raise RuntimeError(
            "Invalid temporal assignments."
        )

    if assignments[
        "pcap_uid"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate assignment pcap_uid."
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

    df, dataset_path = (
        load_historical_65()
    )

    if "pcap_uid" not in df.columns:
        raise RuntimeError(
            "Historical pcap_uid required."
        )

    if df[
        "pcap_uid"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate Historical pcap_uid."
        )

    # INTERNAL_TEST never enters
    # the analytical dataframe.
    dev = df.merge(
        dev_assignments,
        on="pcap_uid",
        how="inner",
        validate="one_to_one",
    )

    observed = set(
        dev[
            "temporal_split"
        ].unique()
    )

    if observed != set(DEV_SPLITS):
        raise RuntimeError(
            "Unexpected DEV composition: "
            f"{observed}"
        )

    if dev[
        "site_label"
    ].nunique() != 65:
        raise RuntimeError(
            "Expected 65 sites in DEV."
        )

    return (
        dev,
        dataset_path,
        assignments_path,
        eligibility_path,
    )


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


def train_only_impute(
    train: pd.DataFrame,
    test: pd.DataFrame,
):
    medians = (
        train
        .median(
            axis=0,
            skipna=True,
        )
    )

    all_missing = (
        medians[
            medians.isna()
        ]
        .index
        .tolist()
    )

    # If a variable has no finite training
    # values, it cannot convey training signal.
    # Fill with a fixed constant without
    # inspecting the test distribution.
    medians = (
        medians
        .fillna(0.0)
    )

    train_imp = (
        train
        .fillna(medians)
        .to_numpy(
            dtype=np.float64
        )
    )

    test_imp = (
        test
        .fillna(medians)
        .to_numpy(
            dtype=np.float64
        )
    )

    if (
        not np.isfinite(
            train_imp
        ).all()
        or
        not np.isfinite(
            test_imp
        ).all()
    ):
        raise RuntimeError(
            "Non-finite values remain "
            "after train-only imputation."
        )

    return (
        train_imp,
        test_imp,
        all_missing,
    )


def build_models(
    n_classes: int,
):
    return {
        "logistic_regression":
            LogisticRegression(
                C=1.0,
                solver="lbfgs",
                max_iter=600,
                tol=1e-4,
            ),

        "random_forest":
            RandomForestClassifier(
                n_estimators=250,
                max_features="sqrt",
                min_samples_leaf=1,
                random_state=SEED,
                n_jobs=-1,
            ),

        "xgboost":
            XGBClassifier(
                n_estimators=120,
                max_depth=5,
                learning_rate=0.10,
                min_child_weight=1.0,
                subsample=0.80,
                colsample_bytree=0.80,
                reg_lambda=1.0,
                objective="multi:softprob",
                num_class=n_classes,
                eval_metric="mlogloss",
                tree_method="hist",
                random_state=SEED,
                n_jobs=-1,
                verbosity=0,
            ),
    }


def run_one_model(
    model_name: str,
    model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
):
    if model_name == "logistic_regression":
        scaler = StandardScaler()

        x_train_model = (
            scaler.fit_transform(
                x_train
            )
        )

        x_test_model = (
            scaler.transform(
                x_test
            )
        )
    else:
        x_train_model = x_train
        x_test_model = x_test

    start_fit = time.perf_counter()

    model.fit(
        x_train_model,
        y_train,
    )

    fit_seconds = (
        time.perf_counter()
        - start_fit
    )

    start_pred = time.perf_counter()

    prediction = model.predict(
        x_test_model
    )

    predict_seconds = (
        time.perf_counter()
        - start_pred
    )

    accuracy = accuracy_score(
        y_test,
        prediction,
    )

    macro_f1 = f1_score(
        y_test,
        prediction,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy":
            float(accuracy),

        "macro_f1":
            float(macro_f1),

        "fit_seconds":
            float(fit_seconds),

        "predict_seconds":
            float(predict_seconds),
    }


def build_summary(
    runs: pd.DataFrame,
) -> pd.DataFrame:

    model_summary = (
        runs
        .groupby(
            [
                "subset_size",
                "model",
            ],
            as_index=False,
        )
        .agg(
            model_worst_macro_f1=(
                "macro_f1",
                "min",
            ),

            model_mean_macro_f1=(
                "macro_f1",
                "mean",
            ),

            model_worst_accuracy=(
                "accuracy",
                "min",
            ),

            model_mean_accuracy=(
                "accuracy",
                "mean",
            ),
        )
    )

    subset_summary = (
        model_summary
        .groupby(
            "subset_size",
            as_index=False,
        )
        .agg(
            primary_mean_model_worst_macro_f1=(
                "model_worst_macro_f1",
                "mean",
            ),

            min_model_worst_macro_f1=(
                "model_worst_macro_f1",
                "min",
            ),

            mean_model_worst_macro_f1=(
                "model_worst_macro_f1",
                "mean",
            ),

            mean_model_worst_accuracy=(
                "model_worst_accuracy",
                "mean",
            ),
        )
    )

    overall = (
        runs
        .groupby(
            "subset_size",
            as_index=False,
        )
        .agg(
            secondary_mean_all_macro_f1=(
                "macro_f1",
                "mean",
            ),

            mean_all_accuracy=(
                "accuracy",
                "mean",
            ),

            macro_f1_std_all_runs=(
                "macro_f1",
                "std",
            ),

            total_fit_seconds=(
                "fit_seconds",
                "sum",
            ),
        )
    )

    summary = (
        subset_summary
        .merge(
            overall,
            on="subset_size",
            validate="one_to_one",
        )
    )

    pivot = (
        model_summary
        .pivot(
            index="subset_size",
            columns="model",
            values="model_worst_macro_f1",
        )
        .reset_index()
    )

    pivot.columns = [
        (
            col
            if col == "subset_size"
            else f"{col}_worst_macro_f1"
        )
        for col in pivot.columns
    ]

    summary = (
        summary
        .merge(
            pivot,
            on="subset_size",
            how="left",
            validate="one_to_one",
        )
    )

    best_primary = float(
        summary[
            "primary_mean_model_worst_macro_f1"
        ].max()
    )

    summary[
        "within_primary_tolerance"
    ] = (
        summary[
            "primary_mean_model_worst_macro_f1"
        ]
        >= (
            best_primary
            - PRIMARY_TOLERANCE
            - 1e-12
        )
    )

    primary_plateau = (
        summary[
            summary[
                "within_primary_tolerance"
            ]
        ]
    )

    best_secondary_plateau = float(
        primary_plateau[
            "secondary_mean_all_macro_f1"
        ].max()
    )

    summary[
        "within_secondary_tolerance"
    ] = (
        summary[
            "within_primary_tolerance"
        ]
        &
        (
            summary[
                "secondary_mean_all_macro_f1"
            ]
            >= (
                best_secondary_plateau
                - SECONDARY_TOLERANCE
                - 1e-12
            )
        )
    )

    eligible = (
        summary[
            summary[
                "within_secondary_tolerance"
            ]
        ]
        .sort_values(
            [
                "subset_size",
                "primary_mean_model_worst_macro_f1",
                "secondary_mean_all_macro_f1",
            ],
            ascending=[
                True,
                False,
                False,
            ],
        )
    )

    if eligible.empty:
        raise RuntimeError(
            "No subset survived "
            "selection tolerances."
        )

    selected_size = int(
        eligible.iloc[0][
            "subset_size"
        ]
    )

    summary[
        "selected_canonical"
    ] = (
        summary[
            "subset_size"
        ]
        == selected_size
    )

    raw_best = (
        summary
        .sort_values(
            [
                "primary_mean_model_worst_macro_f1",
                "secondary_mean_all_macro_f1",
                "subset_size",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )
        .iloc[0]
    )

    return (
        summary.sort_values(
            "subset_size"
        ),
        model_summary,
        selected_size,
        int(
            raw_best[
                "subset_size"
            ]
        ),
        best_primary,
        best_secondary_plateau,
    )


def main():
    print("=" * 78)
    print(
        "MACRO-V2-HIST — "
        "06B DEV-ONLY MODEL SELECTION"
    )
    print("=" * 78)

    print(
        "Purpose: select BASE feature count "
        "using temporal DEV folds."
    )

    print(
        "INTERNAL_TEST: PROHIBITED"
    )

    print(
        "External Future: PROHIBITED"
    )

    print(
        "This is model selection/tuning, "
        "NOT independent holdout evaluation."
    )

    reps, reps_path = (
        load_representatives()
    )

    (
        dev,
        dataset_path,
        assignments_path,
        eligibility_path,
    ) = load_dev()

    ordered_features = (
        reps[
            "feature"
        ]
        .astype(str)
        .tolist()
    )

    missing_features = [
        f
        for f in ordered_features
        if f not in dev.columns
    ]

    if missing_features:
        raise RuntimeError(
            f"Missing features: "
            f"{missing_features}"
        )

    print()
    print(
        f"DEV rows: {len(dev):,}"
    )

    print(
        f"Sites: "
        f"{dev['site_label'].nunique()}"
    )

    print(
        f"Representatives: "
        f"{len(ordered_features)}"
    )

    print(
        "Subset sizes: "
        f"{SUBSET_SIZES}"
    )

    out = output_dir()

    order_path = (
        out
        / "06B_robust_feature_order.csv"
    )

    reps.to_csv(
        order_path,
        index=False,
    )

    subset_rows = []

    for size in SUBSET_SIZES:
        for rank, feature in enumerate(
            ordered_features[:size],
            start=1,
        ):
            subset_rows.append(
                {
                    "subset_size":
                        size,

                    "feature_rank":
                        rank,

                    "feature":
                        feature,
                }
            )

    candidate_subsets = pd.DataFrame(
        subset_rows
    )

    subsets_path = (
        out
        / "06B_candidate_subsets.csv"
    )

    candidate_subsets.to_csv(
        subsets_path,
        index=False,
    )

    run_records = []

    for subset_size in SUBSET_SIZES:

        features = (
            ordered_features[
                :subset_size
            ]
        )

        print()
        print("=" * 78)
        print(
            f"SUBSET TOP-{subset_size}"
        )
        print("=" * 78)

        for fold_spec in FOLDS:

            train_mask = (
                dev[
                    "temporal_split"
                ]
                .isin(
                    fold_spec[
                        "train_splits"
                    ]
                )
            )

            test_mask = (
                dev[
                    "temporal_split"
                ]
                == fold_spec[
                    "test_split"
                ]
            )

            train_df = (
                dev.loc[
                    train_mask
                ]
            )

            test_df = (
                dev.loc[
                    test_mask
                ]
            )

            if (
                train_df.empty
                or test_df.empty
            ):
                raise RuntimeError(
                    "Empty temporal fold."
                )

            train_classes = set(
                train_df[
                    "site_label"
                ].astype(str)
            )

            test_classes = set(
                test_df[
                    "site_label"
                ].astype(str)
            )

            if (
                train_classes
                != test_classes
            ):
                raise RuntimeError(
                    "Train/test class sets differ "
                    f"in {fold_spec['fold']}."
                )

            x_train_df = (
                numeric_frame(
                    train_df,
                    features,
                )
            )

            x_test_df = (
                numeric_frame(
                    test_df,
                    features,
                )
            )

            (
                x_train,
                x_test,
                all_missing,
            ) = train_only_impute(
                x_train_df,
                x_test_df,
            )

            encoder = LabelEncoder()

            y_train = (
                encoder
                .fit_transform(
                    train_df[
                        "site_label"
                    ].astype(str)
                )
            )

            y_test = (
                encoder
                .transform(
                    test_df[
                        "site_label"
                    ].astype(str)
                )
            )

            n_classes = len(
                encoder.classes_
            )

            if n_classes != 65:
                raise RuntimeError(
                    f"Expected 65 classes, "
                    f"found {n_classes}."
                )

            models = build_models(
                n_classes
            )

            print(
                f"{fold_spec['fold']}: "
                f"train={len(train_df):,}, "
                f"test={len(test_df):,}, "
                f"all-missing-train="
                f"{len(all_missing)}"
            )

            for (
                model_name,
                model,
            ) in models.items():

                print(
                    f"  {model_name}...",
                    flush=True,
                )

                metrics = (
                    run_one_model(
                        model_name,
                        model,
                        x_train,
                        y_train,
                        x_test,
                        y_test,
                    )
                )

                record = {
                    "subset_size":
                        subset_size,

                    "model":
                        model_name,

                    "fold":
                        fold_spec[
                            "fold"
                        ],

                    "train_splits":
                        "|".join(
                            fold_spec[
                                "train_splits"
                            ]
                        ),

                    "test_split":
                        fold_spec[
                            "test_split"
                        ],

                    "n_train":
                        len(train_df),

                    "n_test":
                        len(test_df),

                    "n_classes":
                        n_classes,

                    "all_missing_train_features":
                        len(all_missing),

                    "all_missing_train_feature_names":
                        "|".join(
                            all_missing
                        ),

                    **metrics,
                }

                run_records.append(
                    record
                )

                print(
                    "    "
                    f"acc={metrics['accuracy']:.4f} "
                    f"macroF1={metrics['macro_f1']:.4f} "
                    f"fit={metrics['fit_seconds']:.1f}s"
                )

    runs = pd.DataFrame(
        run_records
    )

    runs_path = (
        out
        / "06B_model_runs.csv"
    )

    runs.to_csv(
        runs_path,
        index=False,
    )

    (
        summary,
        model_summary,
        selected_size,
        raw_best_size,
        best_primary,
        best_secondary_plateau,
    ) = build_summary(
        runs
    )

    model_summary_path = (
        out
        / "06B_model_summary.csv"
    )

    model_summary.to_csv(
        model_summary_path,
        index=False,
    )

    summary_path = (
        out
        / "06B_subset_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    selected_features = (
        ordered_features[
            :selected_size
        ]
    )

    selected_path = (
        out
        / "06B_selected_base_features.txt"
    )

    selected_path.write_text(
        "\n".join(
            selected_features
        )
        + "\n"
    )

    manifest = {
        "feature_set_id":
            "MACRO-V2-HIST",

        "stage":
            "06B_dev_model_selection",

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "git_commit":
            git_commit(),

        "environment": {
            "python":
                platform.python_version(),

            "sklearn":
                sklearn.__version__,

            "xgboost":
                xgboost.__version__,

            "seed":
                SEED,
        },

        "inputs": {
            "historical_dataset": {
                "path":
                    str(dataset_path),

                "sha256":
                    sha256(
                        dataset_path
                    ),
            },

            "split_assignments": {
                "path":
                    str(
                        assignments_path
                    ),

                "sha256":
                    sha256(
                        assignments_path
                    ),
            },

            "dev_eligibility": {
                "path":
                    str(
                        eligibility_path
                    ),

                "sha256":
                    sha256(
                        eligibility_path
                    ),
            },

            "06A_representatives": {
                "path":
                    str(
                        reps_path
                    ),

                "sha256":
                    sha256(
                        reps_path
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

            "interpretation":
                (
                    "DEV-only feature-count "
                    "model selection/tuning; "
                    "not independent holdout "
                    "performance estimation."
                ),
        },

        "feature_order": {
            "primary":
                "dtp_min descending",

            "secondary":
                "dtp_geometric_mean descending",

            "tertiary":
                "dtp_arithmetic_mean descending",

            "quaternary":
                "dtp_balance_gap ascending",

            "final_tie":
                "feature ascending",

            "pareto_front_used_for_order":
                False,
        },

        "candidate_subset_sizes":
            SUBSET_SIZES,

        "temporal_folds":
            FOLDS,

        "preprocessing": {
            "imputation":
                (
                    "per-feature median learned "
                    "from fold training data only"
                ),

            "all_missing_train_feature":
                (
                    "fixed zero constant, without "
                    "consulting test values"
                ),

            "logistic_scaling":
                (
                    "StandardScaler fitted on "
                    "fold training data only"
                ),

            "tree_scaling":
                False,
        },

        "models": {
            "logistic_regression": {
                "C": 1.0,
                "solver": "lbfgs",
                "max_iter": 600,
            },

            "random_forest": {
                "n_estimators": 250,
                "max_features": "sqrt",
                "min_samples_leaf": 1,
            },

            "xgboost": {
                "n_estimators": 120,
                "max_depth": 5,
                "learning_rate": 0.10,
                "subsample": 0.80,
                "colsample_bytree": 0.80,
                "tree_method": "hist",
            },
        },

        "selection_rule": {
            "primary":
                (
                    "mean across models of each "
                    "model's worst temporal-fold "
                    "Macro-F1"
                ),

            "primary_tolerance":
                PRIMARY_TOLERANCE,

            "secondary":
                (
                    "mean Macro-F1 across all "
                    "3 models x 2 temporal folds"
                ),

            "secondary_tolerance":
                SECONDARY_TOLERANCE,

            "parsimony":
                (
                    "choose the smallest subset "
                    "within both practical-"
                    "equivalence tolerances"
                ),
        },

        "results": {
            "raw_best_subset_size":
                raw_best_size,

            "canonical_selected_subset_size":
                selected_size,

            "best_primary_score":
                best_primary,

            "best_secondary_within_primary_plateau":
                best_secondary_plateau,

            "selected_feature_count":
                len(
                    selected_features
                ),
        },

        "selection_status": {
            "base_feature_count_selected":
                True,

            "base_feature_values_frozen":
                False,

            "internal_test_opened":
                False,

            "external_future_opened":
                False,

            "next_stage":
                (
                    "document 06B and construct "
                    "META using DEV only before "
                    "opening INTERNAL_TEST"
                ),
        },
    }

    write_json(
        out
        / "06B_manifest_dev_model_selection.json",
        manifest,
    )

    print()
    print("=" * 78)
    print(
        "SUBSET SUMMARY"
    )
    print("=" * 78)

    display_cols = [
        "subset_size",
        "primary_mean_model_worst_macro_f1",
        "secondary_mean_all_macro_f1",
        "min_model_worst_macro_f1",
        "logistic_regression_worst_macro_f1",
        "random_forest_worst_macro_f1",
        "xgboost_worst_macro_f1",
        "within_primary_tolerance",
        "within_secondary_tolerance",
        "selected_canonical",
    ]

    print(
        summary[
            display_cols
        ]
        .to_string(
            index=False
        )
    )

    print()
    print(
        f"Raw best subset: "
        f"TOP-{raw_best_size}"
    )

    print(
        f"Canonical parsimonious subset: "
        f"TOP-{selected_size}"
    )

    print()
    print(
        "Selected BASE features:"
    )

    for idx, feature in enumerate(
        selected_features,
        start=1,
    ):
        print(
            f"{idx:03d}. {feature}"
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
        f"Runs: {runs_path}"
    )

    print(
        f"Summary: {summary_path}"
    )

    print(
        f"Selected list: {selected_path}"
    )

    print()
    print(
        "06B DEV MODEL SELECTION COMPLETE"
    )


if __name__ == "__main__":
    main()
