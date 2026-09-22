from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import f1_score
import xgboost as xgb

from src.utils.paths import result_path

from src.features.macro_v2._common import (
    extract_capture_dates,
    git_commit,
    load_historical_65,
    sha256,
    write_json,
)


EXPECTED_SITES = 65

SUBSET_SIZES = [
    32,
    64,
    96,
    128,
]

SELECTION_TOLERANCE = 0.003

MIN_FAR_IMPROVEMENT = 0.005
MAX_NEAR_DEGRADATION = 0.010


XGB_CONFIG = {
    "n_estimators": 120,
    "max_depth": 5,
    "learning_rate": 0.10,
    "min_child_weight": 1.0,
    "subsample": 0.80,
    "colsample_bytree": 0.80,
    "reg_lambda": 1.0,
    "objective": "multi:softprob",
    "num_class": EXPECTED_SITES,
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}


OLD_OUT = Path(
    result_path(
        "feature_engineering",
        "MACRO-V2-HIST",
    )
)

OUT = Path(
    result_path(
        "feature_engineering",
        "LTD-ROBUSTNESS-PHASE11",
    )
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


CANDIDATE_PATH = (
    OLD_OUT
    / "00_candidate_features_nonconstant.txt"
)

BASE128_PATH = (
    OLD_OUT
    / "06B_R1_selected_base_features.txt"
)


def read_feature_list(
    path,
):

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    features = [
        x.strip()
        for x
        in path.read_text().splitlines()
        if x.strip()
    ]

    if len(
        features
    ) != len(
        set(
            features
        )
    ):
        raise RuntimeError(
            f"Duplicate features in {path}"
        )

    return features


def iqr(
    values,
):

    x = np.asarray(
        values,
        dtype=float,
    )

    x = x[
        np.isfinite(
            x
        )
    ]

    if len(x) == 0:
        return np.nan

    return float(
        np.quantile(
            x,
            0.75,
        )
        -
        np.quantile(
            x,
            0.25,
        )
    )


def safe_spearman(
    a,
    b,
):

    a = np.asarray(
        a,
        dtype=float,
    )

    b = np.asarray(
        b,
        dtype=float,
    )

    mask = (
        np.isfinite(a)
        &
        np.isfinite(b)
    )

    if mask.sum() < 10:
        return np.nan

    aa = a[
        mask
    ]

    bb = b[
        mask
    ]

    if (
        np.nanstd(aa) < 1e-12
        or
        np.nanstd(bb) < 1e-12
    ):
        return np.nan

    result = spearmanr(
        aa,
        bb,
    )

    if hasattr(
        result,
        "statistic",
    ):
        return float(
            result.statistic
        )

    return float(
        result[0]
    )


def build_daily_profiles(
    df,
    features,
):

    print(
        "Building site-day medians..."
    )

    numeric = (
        df[
            features
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
    )

    work = pd.concat(
        [
            df[
                [
                    "site_label",
                    "_date",
                ]
            ].reset_index(
                drop=True
            ),
            numeric.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    daily = (
        work
        .groupby(
            [
                "site_label",
                "_date",
            ],
            observed=True,
        )[
            features
        ]
        .median()
        .reset_index()
    )

    return daily


def feature_stability(
    daily,
    feature,
    quarters,
):

    tmp = daily[
        [
            "site_label",
            "_date",
            feature,
        ]
    ].copy()

    tmp[
        feature
    ] = pd.to_numeric(
        tmp[
            feature
        ],
        errors="coerce",
    )

    valid = tmp[
        np.isfinite(
            tmp[
                feature
            ]
        )
    ].copy()

    coverage = (
        len(valid)
        /
        len(tmp)
        if len(tmp)
        else 0.0
    )

    if valid.empty:
        return {
            "feature":
                feature,

            "coverage":
                0.0,

            "between_site_iqr":
                0.0,

            "median_within_site_iqr":
                np.inf,

            "separation_to_drift":
                0.0,

            "worst_adjacent_shift_median_norm":
                np.inf,

            "extreme_shift_p90_norm":
                np.inf,

            "min_quarter_rank_spearman":
                -1.0,
        }

    site_medians = (
        valid
        .groupby(
            "site_label"
        )[
            feature
        ]
        .median()
    )

    between = iqr(
        site_medians.to_numpy()
    )

    if (
        not np.isfinite(
            between
        )
        or between
        <= 1e-12
    ):
        between = 1e-12

    q25 = (
        valid
        .groupby(
            "site_label"
        )[
            feature
        ]
        .quantile(
            0.25
        )
    )

    q75 = (
        valid
        .groupby(
            "site_label"
        )[
            feature
        ]
        .quantile(
            0.75
        )
    )

    within_iqrs = (
        q75
        -
        q25
    )

    median_within = float(
        np.nanmedian(
            within_iqrs.to_numpy(
                dtype=float
            )
        )
    )

    separation_to_drift = float(
        between
        /
        (
            median_within
            +
            1e-12
        )
    )

    date_to_quarter = {}

    for name, dates in (
        quarters.items()
    ):
        for date in dates:
            date_to_quarter[
                pd.Timestamp(
                    date
                )
            ] = name

    valid[
        "_quarter"
    ] = (
        valid[
            "_date"
        ]
        .map(
            date_to_quarter
        )
    )

    valid = valid.dropna(
        subset=[
            "_quarter"
        ]
    )

    qmed = (
        valid
        .groupby(
            [
                "site_label",
                "_quarter",
            ],
            observed=True,
        )[
            feature
        ]
        .median()
        .unstack()
    )

    required = [
        "Q1",
        "Q2",
        "Q3",
        "Q4",
    ]

    for col in required:
        if col not in qmed.columns:
            qmed[
                col
            ] = np.nan

    correlations = []

    pairs = [
        (
            "Q1",
            "Q2",
        ),
        (
            "Q2",
            "Q3",
        ),
        (
            "Q3",
            "Q4",
        ),
        (
            "Q1",
            "Q4",
        ),
    ]

    adjacent_shift_medians = []

    for qa, qb in pairs:

        a = qmed[
            qa
        ].to_numpy(
            dtype=float
        )

        b = qmed[
            qb
        ].to_numpy(
            dtype=float
        )

        rho = safe_spearman(
            a,
            b,
        )

        if np.isfinite(
            rho
        ):
            correlations.append(
                rho
            )

        if (
            qa,
            qb,
        ) in [
            (
                "Q1",
                "Q2",
            ),
            (
                "Q2",
                "Q3",
            ),
            (
                "Q3",
                "Q4",
            ),
        ]:

            mask = (
                np.isfinite(a)
                &
                np.isfinite(b)
            )

            if mask.any():
                adjacent_shift_medians.append(
                    float(
                        np.median(
                            np.abs(
                                a[
                                    mask
                                ]
                                -
                                b[
                                    mask
                                ]
                            )
                        )
                        /
                        between
                    )
                )

    min_spearman = (
        float(
            min(
                correlations
            )
        )
        if correlations
        else -1.0
    )

    worst_adjacent = (
        float(
            max(
                adjacent_shift_medians
            )
        )
        if adjacent_shift_medians
        else np.inf
    )

    a = qmed[
        "Q1"
    ].to_numpy(
        dtype=float
    )

    b = qmed[
        "Q4"
    ].to_numpy(
        dtype=float
    )

    mask = (
        np.isfinite(a)
        &
        np.isfinite(b)
    )

    if mask.any():
        extreme_p90 = float(
            np.quantile(
                np.abs(
                    a[
                        mask
                    ]
                    -
                    b[
                        mask
                    ]
                )
                /
                between,
                0.90,
            )
        )

    else:
        extreme_p90 = np.inf

    return {
        "feature":
            feature,

        "coverage":
            float(
                coverage
            ),

        "between_site_iqr":
            float(
                between
            ),

        "median_within_site_iqr":
            float(
                median_within
            ),

        "separation_to_drift":
            separation_to_drift,

        "worst_adjacent_shift_median_norm":
            worst_adjacent,

        "extreme_shift_p90_norm":
            extreme_p90,

        "min_quarter_rank_spearman":
            min_spearman,
    }


def percentile_scores(
    table,
):

    out = table.copy()

    # Higher is better.
    out[
        "p_separation_to_drift"
    ] = (
        out[
            "separation_to_drift"
        ]
        .rank(
            pct=True,
            ascending=True,
        )
    )

    out[
        "p_rank_persistence"
    ] = (
        out[
            "min_quarter_rank_spearman"
        ]
        .rank(
            pct=True,
            ascending=True,
        )
    )

    out[
        "p_coverage"
    ] = (
        out[
            "coverage"
        ]
        .rank(
            pct=True,
            ascending=True,
        )
    )

    # Lower is better.
    out[
        "p_adjacent_stability"
    ] = (
        out[
            "worst_adjacent_shift_median_norm"
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .rank(
            pct=True,
            ascending=False,
            na_option="bottom",
        )
    )

    out[
        "p_long_horizon_stability"
    ] = (
        out[
            "extreme_shift_p90_norm"
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .rank(
            pct=True,
            ascending=False,
            na_option="bottom",
        )
    )

    score_columns = [
        "p_separation_to_drift",
        "p_rank_persistence",
        "p_adjacent_stability",
        "p_long_horizon_stability",
        "p_coverage",
    ]

    out[
        "temporal_identity_score"
    ] = (
        out[
            score_columns
        ]
        .mean(
            axis=1
        )
    )

    out = (
        out
        .sort_values(
            [
                "temporal_identity_score",
                "separation_to_drift",
                "min_quarter_rank_spearman",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    out.insert(
        0,
        "rank",
        np.arange(
            1,
            len(out) + 1,
        ),
    )

    return out


def prepare_matrix(
    train,
    test,
    features,
):

    train_x = (
        train[
            features
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
    )

    test_x = (
        test[
            features
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
    )

    medians = train_x.median(
        axis=0,
        skipna=True,
    ).fillna(
        0.0
    )

    return (
        train_x
        .fillna(
            medians
        )
        .to_numpy(
            dtype=np.float32
        ),

        test_x
        .fillna(
            medians
        )
        .to_numpy(
            dtype=np.float32
        ),
    )


def evaluate(
    probs,
    y,
):

    order = np.argsort(
        -probs,
        axis=1,
    )

    pred = order[
        :,
        0
    ]

    ranks = (
        np.argmax(
            order
            ==
            y[
                :,
                None
            ],
            axis=1,
        )
        +
        1
    )

    return {
        "accuracy":
            float(
                np.mean(
                    pred == y
                )
            ),

        "macro_f1":
            float(
                f1_score(
                    y,
                    pred,
                    labels=np.arange(
                        EXPECTED_SITES
                    ),
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
                    1.0
                    /
                    ranks
                )
            ),
    }


def main():

    print("=" * 78)
    print(
        "11A — HISTORICAL TEMPORAL-STABLE MACRO BASIS"
    )
    print("=" * 78)

    print(
        "POST-CONFIRMATORY ROBUSTNESS DEVELOPMENT."
    )

    print(
        "Historical data only."
    )

    print(
        "Future-B is PROHIBITED."
    )

    candidates = read_feature_list(
        CANDIDATE_PATH
    )

    base128 = read_feature_list(
        BASE128_PATH
    )

    if len(
        base128
    ) != 128:
        raise RuntimeError(
            "Expected frozen BASE128."
        )

    df, historical_path = (
        load_historical_65()
    )

    df[
        "site_label"
    ] = (
        df[
            "site_label"
        ]
        .astype(str)
    )

    df[
        "_date"
    ] = (
        extract_capture_dates(
            df
        )
    )

    if df[
        "_date"
    ].isna().any():
        raise RuntimeError(
            "Missing Historical dates."
        )

    missing = [
        f
        for f in candidates
        if f not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing candidates: {missing[:20]}"
        )

    unique_dates = np.array(
        sorted(
            df[
                "_date"
            ].unique()
        )
    )

    if len(
        unique_dates
    ) < 49:
        raise RuntimeError(
            "Not enough Historical dates."
        )

    print(
        "Historical captures:",
        len(
            df
        ),
    )

    print(
        "Sites:",
        df[
            "site_label"
        ].nunique(),
    )

    print(
        "Dates:",
        len(
            unique_dates
        ),
    )

    print(
        "Range:",
        pd.Timestamp(
            unique_dates[
                0
            ]
        ).date(),
        "->",
        pd.Timestamp(
            unique_dates[
                -1
            ]
        ).date(),
    )

    # ==================================================
    # QUARTERS FOR FEATURE STABILITY
    # ==================================================

    quarter_arrays = np.array_split(
        unique_dates,
        4,
    )

    quarters = {
        f"Q{i + 1}":
            values
        for i, values in enumerate(
            quarter_arrays
        )
    }

    print()
    print(
        "Stability quarters:"
    )

    for name, dates in (
        quarters.items()
    ):
        print(
            name,
            pd.Timestamp(
                dates[
                    0
                ]
            ).date(),
            "->",
            pd.Timestamp(
                dates[
                    -1
                ]
            ).date(),
            f"({len(dates)} dates)",
        )

    daily = build_daily_profiles(
        df,
        candidates,
    )

    records = []

    for i, feature in enumerate(
        candidates,
        start=1,
    ):

        if (
            i == 1
            or i % 25 == 0
            or i == len(
                candidates
            )
        ):
            print(
                f"Feature {i}/{len(candidates)}: "
                f"{feature}"
            )

        records.append(
            feature_stability(
                daily,
                feature,
                quarters,
            )
        )

    stability = percentile_scores(
        pd.DataFrame(
            records
        )
    )

    stability_path = (
        OUT
        / "11A_feature_stability.csv"
    )

    stability.to_csv(
        stability_path,
        index=False,
    )

    ordered = (
        stability[
            "feature"
        ]
        .astype(str)
        .tolist()
    )

    # ==================================================
    # FIXED-ORIGIN DRIFT BACKTEST
    # ==================================================
    #
    # Same model / same training window.
    # Only the temporal distance of evaluation changes.
    #
    # Train:
    # first 14 dates
    #
    # NEAR:
    # dates 15-21
    #
    # MID:
    # dates 29-35
    #
    # FAR:
    # dates 43-49
    #

    train_dates = unique_dates[
        :14
    ]

    windows = {
        "NEAR":
            unique_dates[
                14:21
            ],

        "MID":
            unique_dates[
                28:35
            ],

        "FAR":
            unique_dates[
                42:49
            ],
    }

    print()
    print(
        "Fixed-origin train:",
        pd.Timestamp(
            train_dates[
                0
            ]
        ).date(),
        "->",
        pd.Timestamp(
            train_dates[
                -1
            ]
        ).date(),
    )

    for name, dates in (
        windows.items()
    ):
        print(
            name,
            pd.Timestamp(
                dates[
                    0
                ]
            ).date(),
            "->",
            pd.Timestamp(
                dates[
                    -1
                ]
            ).date(),
        )

    labels = np.array(
        sorted(
            df[
                "site_label"
            ].unique()
        ),
        dtype=str,
    )

    if len(
        labels
    ) != EXPECTED_SITES:
        raise RuntimeError(
            "Expected 65 sites."
        )

    label_to_idx = {
        label: i
        for i, label in
        enumerate(
            labels
        )
    }

    df[
        "_y"
    ] = (
        df[
            "site_label"
        ]
        .map(
            label_to_idx
        )
        .astype(
            np.int64
        )
    )

    train = (
        df[
            df[
                "_date"
            ].isin(
                train_dates
            )
        ]
        .copy()
    )

    if (
        train[
            "site_label"
        ].nunique()
        != EXPECTED_SITES
    ):
        raise RuntimeError(
            "Training window lacks site coverage."
        )

    subsets = {
        "BASE128":
            base128,
    }

    for size in SUBSET_SIZES:

        subsets[
            f"STABLE_TOP{size}"
        ] = ordered[
            :size
        ]

    run_rows = []

    for subset_name, features in (
        subsets.items()
    ):

        print()
        print(
            subset_name,
            f"({len(features)} features)"
        )

        for window_name, test_dates in (
            windows.items()
        ):

            test = (
                df[
                    df[
                        "_date"
                    ].isin(
                        test_dates
                    )
                ]
                .copy()
            )

            if (
                test[
                    "site_label"
                ].nunique()
                != EXPECTED_SITES
            ):
                raise RuntimeError(
                    f"{window_name} lacks site coverage."
                )

            (
                x_train,
                x_test,
            ) = prepare_matrix(
                train,
                test,
                features,
            )

            y_train = train[
                "_y"
            ].to_numpy(
                dtype=np.int64
            )

            y_test = test[
                "_y"
            ].to_numpy(
                dtype=np.int64
            )

            model = xgb.XGBClassifier(
                **XGB_CONFIG
            )

            model.fit(
                x_train,
                y_train,
            )

            if not np.array_equal(
                model.classes_,
                np.arange(
                    EXPECTED_SITES
                ),
            ):
                raise RuntimeError(
                    "Unexpected XGB class order."
                )

            probs = (
                model
                .predict_proba(
                    x_test
                )
                .astype(
                    np.float64
                )
            )

            metric = evaluate(
                probs,
                y_test,
            )

            row = {
                "subset":
                    subset_name,

                "feature_count":
                    len(
                        features
                    ),

                "window":
                    window_name,

                "train_date_min":
                    str(
                        pd.Timestamp(
                            train_dates[
                                0
                            ]
                        ).date()
                    ),

                "train_date_max":
                    str(
                        pd.Timestamp(
                            train_dates[
                                -1
                            ]
                        ).date()
                    ),

                "test_date_min":
                    str(
                        pd.Timestamp(
                            test_dates[
                                0
                            ]
                        ).date()
                    ),

                "test_date_max":
                    str(
                        pd.Timestamp(
                            test_dates[
                                -1
                            ]
                        ).date()
                    ),

                "n_train":
                    len(
                        train
                    ),

                "n_test":
                    len(
                        test
                    ),

                **metric,
            }

            run_rows.append(
                row
            )

            print(
                window_name,
                f"acc={metric['accuracy']:.4f}",
                f"f1={metric['macro_f1']:.4f}",
                f"top5={metric['top5_accuracy']:.4f}",
            )

    runs = pd.DataFrame(
        run_rows
    )

    runs_path = (
        OUT
        / "11A_backtest_runs.csv"
    )

    runs.to_csv(
        runs_path,
        index=False,
    )

    summary_rows = []

    for subset, group in (
        runs.groupby(
            "subset"
        )
    ):

        by_window = (
            group
            .set_index(
                "window"
            )
        )

        summary_rows.append(
            {
                "subset":
                    subset,

                "feature_count":
                    int(
                        group[
                            "feature_count"
                        ].iloc[
                            0
                        ]
                    ),

                "worst_macro_f1":
                    float(
                        group[
                            "macro_f1"
                        ].min()
                    ),

                "mean_macro_f1":
                    float(
                        group[
                            "macro_f1"
                        ].mean()
                    ),

                "near_macro_f1":
                    float(
                        by_window.loc[
                            "NEAR",
                            "macro_f1",
                        ]
                    ),

                "mid_macro_f1":
                    float(
                        by_window.loc[
                            "MID",
                            "macro_f1",
                        ]
                    ),

                "far_macro_f1":
                    float(
                        by_window.loc[
                            "FAR",
                            "macro_f1",
                        ]
                    ),

                "far_accuracy":
                    float(
                        by_window.loc[
                            "FAR",
                            "accuracy",
                        ]
                    ),

                "far_top5":
                    float(
                        by_window.loc[
                            "FAR",
                            "top5_accuracy",
                        ]
                    ),

                "far_mrr":
                    float(
                        by_window.loc[
                            "FAR",
                            "mrr",
                        ]
                    ),
            }
        )

    subset_summary = pd.DataFrame(
        summary_rows
    )

    stable_summary = (
        subset_summary[
            subset_summary[
                "subset"
            ].str.startswith(
                "STABLE_"
            )
        ]
        .copy()
    )

    best_worst = float(
        stable_summary[
            "worst_macro_f1"
        ].max()
    )

    eligible = (
        stable_summary[
            stable_summary[
                "worst_macro_f1"
            ]
            >=
            best_worst
            -
            SELECTION_TOLERANCE
        ]
        .copy()
    )

    best_far = float(
        eligible[
            "far_macro_f1"
        ].max()
    )

    eligible = (
        eligible[
            eligible[
                "far_macro_f1"
            ]
            >=
            best_far
            -
            SELECTION_TOLERANCE
        ]
        .copy()
    )

    selected_row = (
        eligible
        .sort_values(
            [
                "feature_count",
                "subset",
            ]
        )
        .iloc[
            0
        ]
    )

    selected_name = str(
        selected_row[
            "subset"
        ]
    )

    selected_size = int(
        selected_row[
            "feature_count"
        ]
    )

    selected_features = (
        subsets[
            selected_name
        ]
    )

    baseline = (
        subset_summary[
            subset_summary[
                "subset"
            ]
            ==
            "BASE128"
        ]
        .iloc[
            0
        ]
    )

    promotion_pass = bool(
        selected_row[
            "far_macro_f1"
        ]
        >=
        baseline[
            "far_macro_f1"
        ]
        +
        MIN_FAR_IMPROVEMENT

        and

        selected_row[
            "worst_macro_f1"
        ]
        >=
        baseline[
            "worst_macro_f1"
        ]

        and

        selected_row[
            "near_macro_f1"
        ]
        >=
        baseline[
            "near_macro_f1"
        ]
        -
        MAX_NEAR_DEGRADATION
    )

    subset_summary[
        "selected_stable_candidate"
    ] = (
        subset_summary[
            "subset"
        ]
        ==
        selected_name
    )

    subset_summary[
        "baseline"
    ] = (
        subset_summary[
            "subset"
        ]
        ==
        "BASE128"
    )

    summary_path = (
        OUT
        / "11A_subset_summary.csv"
    )

    subset_summary.to_csv(
        summary_path,
        index=False,
    )

    selected_path = (
        OUT
        / "11A_selected_stable_features.txt"
    )

    selected_path.write_text(
        "\n".join(
            selected_features
        )
        +
        "\n"
    )

    manifest = {
        "stage":
            "11A_historical_robust_feature_basis",

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "git_commit":
            git_commit(),

        "scientific_status":
            "POST_CONFIRMATORY_ROBUSTNESS_DEVELOPMENT",

        "hypothesis":
            (
                "Features preserving site identity "
                "while minimizing within-site temporal "
                "movement improve long-horizon Macro "
                "classification."
            ),

        "data_policy": {
            "historical_only":
                True,

            "future_b_used":
                False,

            "future_b_scores_used":
                False,

            "internal_period_is_now_development_data":
                True,

            "future_b_may_not_select_this_model":
                True,
        },

        "population": {
            "captures":
                len(
                    df
                ),

            "sites":
                int(
                    df[
                        "site_label"
                    ].nunique()
                ),

            "dates":
                len(
                    unique_dates
                ),

            "date_min":
                str(
                    pd.Timestamp(
                        unique_dates[
                            0
                        ]
                    ).date()
                ),

            "date_max":
                str(
                    pd.Timestamp(
                        unique_dates[
                            -1
                        ]
                    ).date()
                ),
        },

        "feature_scoring": {
            "candidate_count":
                len(
                    candidates
                ),

            "metrics": [
                "between-site separation / within-site temporal IQR",
                "minimum quarter-to-quarter site rank Spearman",
                "worst adjacent-quarter normalized site shift",
                "Q1-to-Q4 p90 normalized site shift",
                "site-day coverage",
            ],

            "aggregation":
                "equal-weight mean percentile",
        },

        "fixed_origin_backtest": {
            "train_dates":
                [
                    str(
                        pd.Timestamp(
                            x
                        ).date()
                    )
                    for x in
                    train_dates
                ],

            "test_windows": {
                name: [
                    str(
                        pd.Timestamp(
                            x
                        ).date()
                    )
                    for x in
                    dates
                ]
                for name, dates in
                windows.items()
            },

            "candidate_subset_sizes":
                SUBSET_SIZES,

            "model":
                XGB_CONFIG,
        },

        "selection": {
            "primary":
                "worst Macro-F1 across NEAR/MID/FAR",

            "primary_tolerance":
                SELECTION_TOLERANCE,

            "secondary":
                "FAR Macro-F1",

            "secondary_tolerance":
                SELECTION_TOLERANCE,

            "parsimony":
                "smallest eligible feature count",

            "selected_candidate":
                selected_name,

            "selected_size":
                selected_size,
        },

        "promotion_gate": {
            "far_macro_f1_improvement_required":
                MIN_FAR_IMPROVEMENT,

            "worst_macro_f1_not_below_BASE128":
                True,

            "max_near_macro_f1_degradation":
                MAX_NEAR_DEGRADATION,

            "passed":
                promotion_pass,
        },

        "inputs": {
            "historical": {
                "path":
                    str(
                        historical_path
                    ),

                "sha256":
                    sha256(
                        historical_path
                    ),
            },

            "candidates": {
                "path":
                    str(
                        CANDIDATE_PATH
                    ),

                "sha256":
                    sha256(
                        CANDIDATE_PATH
                    ),
            },

            "base128": {
                "path":
                    str(
                        BASE128_PATH
                    ),

                "sha256":
                    sha256(
                        BASE128_PATH
                    ),
            },
        },

        "outputs": {
            "feature_stability":
                sha256(
                    stability_path
                ),

            "backtest_runs":
                sha256(
                    runs_path
                ),

            "subset_summary":
                sha256(
                    summary_path
                ),

            "selected_features":
                sha256(
                    selected_path
                ),
        },

        "next":
            (
                "11B multiscale longitudinal memory; "
                "use selected stable basis as candidate "
                "only if promotion gate passes."
            ),
    }

    write_json(
        OUT
        / "11A_manifest.json",
        manifest,
    )

    print()
    print("=" * 78)
    print(
        "TOP 20 TEMPORAL-IDENTITY FEATURES"
    )
    print("=" * 78)

    print(
        stability[
            [
                "rank",
                "feature",
                "temporal_identity_score",
                "separation_to_drift",
                "min_quarter_rank_spearman",
                "extreme_shift_p90_norm",
            ]
        ]
        .head(
            20
        )
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 78)
    print(
        "SUBSET BACKTEST"
    )
    print("=" * 78)

    print(
        subset_summary
        .sort_values(
            "feature_count"
        )
        .to_string(
            index=False
        )
    )

    print()
    print(
        "Selected stable candidate:",
        selected_name,
    )

    print(
        "PROMOTION PASS:",
        promotion_pass,
    )

    print()
    print(
        "Future-B was not accessed."
    )

    print(
        "Next: 11B multiscale longitudinal memory."
    )


if __name__ == "__main__":
    main()
