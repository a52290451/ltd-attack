from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import gc
import importlib.util
import sys
import time

import numpy as np
import pandas as pd
import torch

from src.utils.paths import result_path

from src.features.macro_v2._common import (
    extract_capture_dates,
    git_commit,
    load_historical_65,
    sha256,
    write_json,
)


EXPECTED_SITES = 65
CONTEXT_DAYS = 5

MIN_MEAN_FAR_FUSION_F1_GAIN = 0.005
MIN_BOOTSTRAP_POSITIVE = 0.90


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


def repo_root():
    return (
        Path(__file__)
        .resolve()
        .parents[3]
    )


def load_module(
    path,
    name,
):
    path = Path(path)

    parent = str(path.parent)

    if parent not in sys.path:
        sys.path.insert(
            0,
            parent,
        )

    spec = (
        importlib.util
        .spec_from_file_location(
            name,
            path,
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            f"Could not load {path}"
        )

    mod = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(
        mod
    )

    return mod


def train_models(
    mod,
    x_train,
    y_train,
    train_dates,
    daily_history,
    candidate_labels,
    features,
    device,
    origin,
    seed_rows,
):
    mod.CONTEXT_DAYS = CONTEXT_DAYS

    (
        train_cache,
        skipped_train,
    ) = mod.build_context_cache(
        daily_history,
        candidate_labels,
        train_dates,
        features,
    )

    if not train_cache:
        raise RuntimeError(
            f"{origin}: no causal train contexts."
        )

    models = []

    for seed in mod.SEEDS:
        print(
            origin,
            "seed=",
            seed,
        )

        mod.set_seed(seed)

        model = (
            mod.LTDPairScorer()
            .to(device)
        )

        t0 = time.perf_counter()

        (
            train_loss,
            n_train_queries,
            n_train_dates,
        ) = mod.train_ltd_model(
            model,
            x_train,
            y_train,
            train_dates,
            train_cache,
            device,
            seed,
        )

        seed_rows.append(
            {
                "origin":
                    origin,

                "seed":
                    seed,

                "train_loss":
                    train_loss,

                "n_train_queries":
                    n_train_queries,

                "n_train_context_dates":
                    n_train_dates,

                "skipped_train_dates":
                    len(
                        skipped_train
                    ),

                "fit_seconds":
                    time.perf_counter()
                    -
                    t0,
            }
        )

        models.append(model)

    return models


def ensemble_predict(
    mod,
    helpers,
    models,
    x,
    dates,
    history,
    candidate_labels,
    features,
    device,
):
    (
        cache,
        skipped,
    ) = mod.build_context_cache(
        history,
        candidate_labels,
        dates,
        features,
    )

    if skipped:
        raise RuntimeError(
            f"Missing contexts: {skipped}"
        )

    probs = []

    for model in models:
        p = helpers.predict_ltd(
            mod,
            model,
            x,
            dates,
            cache,
            device,
        )

        probs.append(p)

    ensemble = (
        np.stack(
            probs,
            axis=0,
        )
        .mean(
            axis=0
        )
    )

    ensemble /= ensemble.sum(
        axis=1,
        keepdims=True,
    )

    return ensemble


def oracle_refresh_window(
    mod,
    helpers,
    models,
    test,
    x_test,
    y_test,
    dates,
    initial_history,
    candidate_labels,
    features,
    medians,
    scaler,
    device,
):
    history = (
        initial_history
        .copy()
        .reset_index(
            drop=True
        )
    )

    output = np.full(
        (
            len(test),
            EXPECTED_SITES,
        ),
        np.nan,
        dtype=np.float64,
    )

    day_rows = []

    unique_dates = np.array(
        sorted(
            pd.unique(
                pd.to_datetime(
                    dates
                )
            )
        )
    )

    for day_index, date in enumerate(
        unique_dates,
        start=1,
    ):
        date = pd.Timestamp(date)

        idx = np.flatnonzero(
            pd.to_datetime(dates)
            ==
            date
        )

        if len(idx) == 0:
            raise RuntimeError(
                f"No observations for {date}"
            )

        p = ensemble_predict(
            mod,
            helpers,
            models,
            x_test[
                idx
            ],
            pd.to_datetime(
                dates[
                    idx
                ]
            ).to_numpy(),
            history,
            candidate_labels,
            features,
            device,
        )

        output[
            idx
        ] = p

        (
            metrics,
            _,
            _,
        ) = helpers.evaluate(
            p,
            y_test[
                idx
            ],
        )

        day_rows.append(
            {
                "date":
                    str(
                        date.date()
                    ),

                "day_in_window":
                    day_index,

                "n":
                    len(idx),

                "history_rows_before_update":
                    len(history),

                **metrics,
            }
        )

        # ==================================================
        # ORACLE UPDATE
        #
        # IMPORTANT:
        # The current date is updated only AFTER its
        # predictions have already been produced.
        #
        # Therefore no current-day true label can affect
        # its own prediction.
        #
        # This is non-deployable and diagnostic only.
        # ==================================================

        day_frame = (
            test.iloc[
                idx
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        day_daily_raw = (
            helpers.build_daily(
                day_frame,
                features,
            )
        )

        day_daily_std = (
            mod.standardized_daily_frame(
                day_daily_raw,
                features,
                medians,
                scaler,
            )
        )

        if (
            day_daily_std[
                "date"
            ]
            >=
            date
        ).all() is False:
            raise RuntimeError(
                "Unexpected oracle update date."
            )

        history = pd.concat(
            [
                history,
                day_daily_std,
            ],
            ignore_index=True,
        )

        history = (
            history
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

    if np.isnan(output).any():
        raise RuntimeError(
            "Incomplete oracle predictions."
        )

    return (
        output,
        pd.DataFrame(
            day_rows
        ),
    )


def main():
    print("=" * 78)
    print(
        "11F — CAUSAL ORACLE MEMORY REFRESH"
    )
    print("=" * 78)

    print(
        "Historical-only mechanism diagnostic."
    )

    print(
        "No Future-B values/scores are used."
    )

    print(
        "Oracle labels are used ONLY after each day "
        "has already been predicted."
    )

    print(
        "This is NOT a deployable model."
    )

    root = repo_root()

    source07a = (
        root
        / "src/features/macro_v2/"
        "07A_candidate_conditioned_temporal_encoder.py"
    )

    source11b = (
        root
        / "src/features/macro_v2/"
        "11B_multiscale_longitudinal_memory.py"
    )

    mod = load_module(
        source07a,
        "ltd_07a_for_11f",
    )

    helpers = load_module(
        source11b,
        "ltd_11b_helpers_for_11f",
    )

    mod.CONTEXT_DAYS = CONTEXT_DAYS

    (
        features,
        frozen_path,
    ) = mod.load_frozen_features()

    if len(features) != 128:
        raise RuntimeError(
            "Expected BASE128."
        )

    (
        captures,
        historical_path,
    ) = load_historical_65()

    captures[
        "site_label"
    ] = (
        captures[
            "site_label"
        ]
        .astype(str)
    )

    captures[
        "_query_date"
    ] = (
        extract_capture_dates(
            captures
        )
    )

    if captures[
        "_query_date"
    ].isna().any():
        raise RuntimeError(
            "Missing Historical dates."
        )

    unique_dates = np.array(
        sorted(
            captures[
                "_query_date"
            ].unique()
        )
    )

    if len(unique_dates) != 52:
        raise RuntimeError(
            "Expected 52 Historical dates."
        )

    candidate_labels = np.array(
        sorted(
            captures[
                "site_label"
            ].unique()
        ),
        dtype=str,
    )

    if len(candidate_labels) != EXPECTED_SITES:
        raise RuntimeError(
            "Expected 65 sites."
        )

    label_to_idx = (
        mod.make_label_mapping(
            candidate_labels
        )
    )

    daily_all = helpers.build_daily(
        captures,
        features,
    )

    origins = {
        "ORIGIN14": {
            "train":
                unique_dates[
                    :14
                ],

            "windows": {
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
            },
        },

        "ORIGIN28": {
            "train":
                unique_dates[
                    :28
                ],

            "windows": {
                "NEAR":
                    unique_dates[
                        28:35
                    ],

                "MID":
                    unique_dates[
                        35:42
                    ],

                "FAR":
                    unique_dates[
                        45:52
                    ],
            },
        },
    }

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

    summary_rows = []
    comparison_rows = []
    bootstrap_rows = []
    per_day_rows = []
    seed_rows = []

    for origin, spec in origins.items():
        print()
        print("#" * 78)
        print(origin)
        print("#" * 78)

        train = (
            captures[
                captures[
                    "_query_date"
                ].isin(
                    spec[
                        "train"
                    ]
                )
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        tests = {
            name:
                captures[
                    captures[
                        "_query_date"
                    ].isin(
                        dates
                    )
                ]
                .copy()
                .reset_index(
                    drop=True
                )
            for name, dates
            in spec[
                "windows"
            ].items()
        }

        y_train = (
            mod.labels_to_indices(
                train[
                    "site_label"
                ],
                label_to_idx,
            )
        )

        y_tests = {
            name:
                mod.labels_to_indices(
                    frame[
                        "site_label"
                    ],
                    label_to_idx,
                )
            for name, frame
            in tests.items()
        }

        (
            medians,
            scaler,
            x_train,
            x_tests,
        ) = helpers.prepare_matrix(
            mod,
            train,
            tests,
            features,
        )

        train_dates = (
            pd.to_datetime(
                train[
                    "_query_date"
                ]
            )
            .to_numpy()
        )

        test_dates = {
            name:
                pd.to_datetime(
                    frame[
                        "_query_date"
                    ]
                )
                .to_numpy()
            for name, frame
            in tests.items()
        }

        train_daily = (
            daily_all[
                daily_all[
                    "date"
                ].isin(
                    spec[
                        "train"
                    ]
                )
            ]
            .copy()
        )

        daily_history = (
            mod.standardized_daily_frame(
                train_daily,
                features,
                medians,
                scaler,
            )
        )

        xgb_probs = (
            helpers.train_xgb(
                train,
                tests,
                features,
                y_train,
            )
        )

        models = train_models(
            mod,
            x_train,
            y_train,
            train_dates,
            daily_history,
            candidate_labels,
            features,
            device,
            origin,
            seed_rows,
        )

        for window in [
            "NEAR",
            "MID",
            "FAR",
        ]:
            print(
                origin,
                window,
            )

            # ------------------------------------------
            # Frozen control
            # ------------------------------------------

            frozen_ltd = ensemble_predict(
                mod,
                helpers,
                models,
                x_tests[
                    window
                ],
                test_dates[
                    window
                ],
                daily_history,
                candidate_labels,
                features,
                device,
            )

            frozen_fusion = helpers.fuse(
                xgb_probs[
                    window
                ],
                frozen_ltd,
            )

            # ------------------------------------------
            # Oracle causal refresh
            # Window starts again from training history.
            # ------------------------------------------

            (
                oracle_ltd,
                day_table,
            ) = oracle_refresh_window(
                mod,
                helpers,
                models,
                tests[
                    window
                ],
                x_tests[
                    window
                ],
                y_tests[
                    window
                ],
                test_dates[
                    window
                ],
                daily_history,
                candidate_labels,
                features,
                medians,
                scaler,
                device,
            )

            oracle_fusion = helpers.fuse(
                xgb_probs[
                    window
                ],
                oracle_ltd,
            )

            # Day 1 must be identical because no oracle
            # update has happened yet.
            first_date = pd.Timestamp(
                min(
                    pd.to_datetime(
                        test_dates[
                            window
                        ]
                    )
                )
            )

            first_idx = np.flatnonzero(
                pd.to_datetime(
                    test_dates[
                        window
                    ]
                )
                ==
                first_date
            )

            max_first_day_error = float(
                np.max(
                    np.abs(
                        frozen_ltd[
                            first_idx
                        ]
                        -
                        oracle_ltd[
                            first_idx
                        ]
                    )
                )
            )

            if max_first_day_error > 1e-6:
                raise RuntimeError(
                    f"{origin}/{window}: "
                    "oracle affected first-day prediction."
                )

            variants = {
                "FROZEN_RECENT5": {
                    "ltd":
                        frozen_ltd,

                    "fusion":
                        frozen_fusion,
                },

                "ORACLE_REFRESH_RECENT5": {
                    "ltd":
                        oracle_ltd,

                    "fusion":
                        oracle_fusion,
                },
            }

            variant_metrics = {}

            for variant, scores in variants.items():
                ltd_metrics, _, _ = (
                    helpers.evaluate(
                        scores[
                            "ltd"
                        ],
                        y_tests[
                            window
                        ],
                    )
                )

                fusion_metrics, _, _ = (
                    helpers.evaluate(
                        scores[
                            "fusion"
                        ],
                        y_tests[
                            window
                        ],
                    )
                )

                variant_metrics[
                    variant
                ] = {
                    "ltd":
                        ltd_metrics,

                    "fusion":
                        fusion_metrics,
                }

                summary_rows.append(
                    {
                        "origin":
                            origin,

                        "window":
                            window,

                        "variant":
                            variant,

                        "n":
                            len(
                                y_tests[
                                    window
                                ]
                            ),

                        **{
                            f"ltd_{k}":
                                v
                            for k, v
                            in ltd_metrics.items()
                        },

                        **{
                            f"fusion_{k}":
                                v
                            for k, v
                            in fusion_metrics.items()
                        },
                    }
                )

            row = {
                "origin":
                    origin,

                "window":
                    window,

                "first_day_max_probability_difference":
                    max_first_day_error,
            }

            for model in [
                "ltd",
                "fusion",
            ]:
                for metric in [
                    "accuracy",
                    "macro_f1",
                    "top5_accuracy",
                    "mrr",
                ]:
                    row[
                        f"delta_{model}_{metric}"
                    ] = (
                        variant_metrics[
                            "ORACLE_REFRESH_RECENT5"
                        ][
                            model
                        ][
                            metric
                        ]
                        -
                        variant_metrics[
                            "FROZEN_RECENT5"
                        ][
                            model
                        ][
                            metric
                        ]
                    )

            comparison_rows.append(
                row
            )

            for model in [
                "ltd",
                "fusion",
            ]:
                boot = helpers.bootstrap_delta(
                    y_tests[
                        window
                    ],
                    pd.to_datetime(
                        test_dates[
                            window
                        ]
                    )
                    .to_numpy(
                        dtype="datetime64[D]"
                    ),
                    variants[
                        "FROZEN_RECENT5"
                    ][
                        model
                    ],
                    variants[
                        "ORACLE_REFRESH_RECENT5"
                    ][
                        model
                    ],
                )

                for metric, values in (
                    boot.items()
                ):
                    bootstrap_rows.append(
                        {
                            "origin":
                                origin,

                            "window":
                                window,

                            "model":
                                model,

                            "metric":
                                metric,

                            "observed_delta":
                                row[
                                    (
                                        f"delta_"
                                        f"{model}_"
                                        f"{metric}"
                                    )
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

                            "fraction_delta_gt_0":
                                float(
                                    np.mean(
                                        values > 0
                                    )
                                ),
                        }
                    )

            # ------------------------------------------
            # Per-day frozen vs oracle
            # ------------------------------------------

            unique_window_dates = np.array(
                sorted(
                    pd.unique(
                        pd.to_datetime(
                            test_dates[
                                window
                            ]
                        )
                    )
                )
            )

            for day_index, date in enumerate(
                unique_window_dates,
                start=1,
            ):
                idx = np.flatnonzero(
                    pd.to_datetime(
                        test_dates[
                            window
                        ]
                    )
                    ==
                    pd.Timestamp(
                        date
                    )
                )

                for variant, scores in (
                    variants.items()
                ):
                    ltd_m, _, _ = helpers.evaluate(
                        scores[
                            "ltd"
                        ][
                            idx
                        ],
                        y_tests[
                            window
                        ][
                            idx
                        ],
                    )

                    fusion_m, _, _ = helpers.evaluate(
                        scores[
                            "fusion"
                        ][
                            idx
                        ],
                        y_tests[
                            window
                        ][
                            idx
                        ],
                    )

                    per_day_rows.append(
                        {
                            "origin":
                                origin,

                            "window":
                                window,

                            "date":
                                str(
                                    pd.Timestamp(
                                        date
                                    ).date()
                                ),

                            "day_in_window":
                                day_index,

                            "variant":
                                variant,

                            "n":
                                len(idx),

                            "ltd_accuracy":
                                ltd_m[
                                    "accuracy"
                                ],

                            "ltd_macro_f1":
                                ltd_m[
                                    "macro_f1"
                                ],

                            "fusion_accuracy":
                                fusion_m[
                                    "accuracy"
                                ],

                            "fusion_macro_f1":
                                fusion_m[
                                    "macro_f1"
                                ],
                        }
                    )

        for model in models:
            del model

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = pd.DataFrame(
        summary_rows
    )

    comparison = pd.DataFrame(
        comparison_rows
    )

    bootstrap = pd.DataFrame(
        bootstrap_rows
    )

    per_day = pd.DataFrame(
        per_day_rows
    )

    far = comparison[
        comparison[
            "window"
        ]
        ==
        "FAR"
    ]

    far_boot = bootstrap[
        (
            bootstrap[
                "window"
            ]
            ==
            "FAR"
        )
        &
        (
            bootstrap[
                "model"
            ]
            ==
            "fusion"
        )
        &
        (
            bootstrap[
                "metric"
            ]
            ==
            "macro_f1"
        )
    ]

    feasibility_pass = bool(
        far[
            "delta_fusion_macro_f1"
        ].mean()
        >=
        MIN_MEAN_FAR_FUSION_F1_GAIN

        and

        (
            far[
                "delta_fusion_macro_f1"
            ]
            >
            0
        ).all()

        and

        far[
            "delta_ltd_macro_f1"
        ].mean()
        >
        0

        and

        (
            far_boot[
                "fraction_delta_gt_0"
            ]
            >=
            MIN_BOOTSTRAP_POSITIVE
        ).all()
    )

    summary_path = (
        OUT
        / "11F_oracle_refresh_summary.csv"
    )

    comparison_path = (
        OUT
        / "11F_oracle_refresh_delta.csv"
    )

    bootstrap_path = (
        OUT
        / "11F_day_block_bootstrap.csv"
    )

    per_day_path = (
        OUT
        / "11F_per_day_recovery.csv"
    )

    seed_path = (
        OUT
        / "11F_seed_training.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    bootstrap.to_csv(
        bootstrap_path,
        index=False,
    )

    per_day.to_csv(
        per_day_path,
        index=False,
    )

    pd.DataFrame(
        seed_rows
    ).to_csv(
        seed_path,
        index=False,
    )

    manifest = {
        "stage":
            "11F_oracle_memory_refresh",

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "git_commit":
            git_commit(),

        "scientific_status":
            "NON_DEPLOYABLE_MECHANISM_DIAGNOSTIC",

        "question":
            (
                "Would perfectly refreshed recent "
                "candidate history materially improve "
                "stale-context robustness?"
            ),

        "data_policy": {
            "historical_only":
                True,

            "future_b_values_used":
                False,

            "future_b_scores_used":
                False,

            "current_day_label_affects_current_prediction":
                False,

            "oracle_update_lag":
                "one day",

            "window_reset":
                True,

            "deployable":
                False,
        },

        "control":
            "FROZEN_RECENT5",

        "oracle":
            (
                "After scoring each day, construct true-label "
                "site-day profiles and make them available "
                "only to later days in the same window."
            ),

        "feasibility_gate": {
            "mean_far_fusion_macro_f1_gain_min":
                MIN_MEAN_FAR_FUSION_F1_GAIN,

            "all_far_fusion_macro_f1_positive":
                True,

            "mean_far_ltd_macro_f1_positive":
                True,

            "all_far_bootstrap_positive_fraction_min":
                MIN_BOOTSTRAP_POSITIVE,

            "passed":
                feasibility_pass,
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

            "base128": {
                "path":
                    str(
                        frozen_path
                    ),

                "sha256":
                    sha256(
                        frozen_path
                    ),
            },

            "07A_source": {
                "path":
                    str(
                        source07a
                    ),

                "sha256":
                    sha256(
                        source07a
                    ),
            },

            "11B_source": {
                "path":
                    str(
                        source11b
                    ),

                "sha256":
                    sha256(
                        source11b
                    ),
            },
        },

        "outputs": {
            "summary":
                sha256(
                    summary_path
                ),

            "delta":
                sha256(
                    comparison_path
                ),

            "bootstrap":
                sha256(
                    bootstrap_path
                ),

            "per_day":
                sha256(
                    per_day_path
                ),

            "seed_training":
                sha256(
                    seed_path
                ),
        },

        "next_if_pass":
            (
                "11G deployable causal pseudo-label "
                "self-updating memory"
            ),

        "next_if_fail":
            (
                "stop recent-memory refresh line; "
                "staleness alone does not explain "
                "the LTD degradation"
            ),
    }

    write_json(
        OUT
        / "11F_manifest.json",
        manifest,
    )

    print()
    print("=" * 78)
    print(
        "11F DELTAS"
    )
    print("=" * 78)

    print(
        comparison.to_string(
            index=False
        )
    )

    print()
    print("=" * 78)
    print(
        "FAR BOOTSTRAP"
    )
    print("=" * 78)

    print(
        far_boot.to_string(
            index=False
        )
    )

    print()
    print(
        "ORACLE MEMORY REFRESH FEASIBILITY:",
        feasibility_pass,
    )

    print()
    print(
        "Future-B was not accessed."
    )

    print(
        "Oracle results are NON-DEPLOYABLE."
    )


if __name__ == "__main__":
    main()
