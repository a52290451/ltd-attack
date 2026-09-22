from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import gc
import importlib.util
import sys

import numpy as np
import pandas as pd
import torch
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

EDGE_WEIGHT_RATIO = 4.0

EPS = 1e-12

MIN_MEAN_FAR_FUSION_F1_GAIN = 0.005
MAX_NEAR_FUSION_F1_DROP = 0.010
MIN_FAR_BOOTSTRAP_POSITIVE = 0.90


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

    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not load {path}"
        )

    mod = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        mod
    )

    return mod


def geometric_mean(
    probabilities,
):
    logp = np.mean(
        [
            np.log(
                np.clip(
                    p,
                    EPS,
                    1.0,
                )
            )
            for p in probabilities
        ],
        axis=0,
    )

    logp -= logp.max(
        axis=1,
        keepdims=True,
    )

    p = np.exp(
        logp
    )

    return (
        p
        /
        p.sum(
            axis=1,
            keepdims=True,
        )
    )


def temporal_weights(
    dates,
    mode,
):
    dates = pd.to_datetime(
        dates
    )

    unique = np.array(
        sorted(
            pd.unique(
                dates
            )
        )
    )

    rank = {
        pd.Timestamp(d):
            i
        for i, d
        in enumerate(unique)
    }

    if len(unique) <= 1:
        t = np.zeros(
            len(dates),
            dtype=float,
        )
    else:
        t = np.asarray(
            [
                rank[
                    pd.Timestamp(d)
                ]
                /
                (
                    len(unique)
                    -
                    1
                )
                for d in dates
            ],
            dtype=float,
        )

    gamma = np.log(
        EDGE_WEIGHT_RATIO
    )

    if mode == "UNIFORM":
        w = np.ones_like(
            t
        )

    elif mode == "EARLY":
        w = np.exp(
            -gamma
            *
            t
        )

    elif mode == "LATE":
        w = np.exp(
            -gamma
            *
            (
                1.0
                -
                t
            )
        )

    else:
        raise ValueError(
            mode
        )

    # Same mean sample weight across experts.
    w /= w.mean()

    return w.astype(
        np.float64
    )


def prepare_xgb(
    train,
    tests,
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

    medians = (
        train_x
        .median(
            axis=0,
            skipna=True,
        )
        .fillna(
            0.0
        )
    )

    x_train = (
        train_x
        .fillna(
            medians
        )
        .to_numpy(
            dtype=np.float32
        )
    )

    x_tests = {}

    for name, frame in tests.items():
        x_tests[
            name
        ] = (
            frame[
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
            .fillna(
                medians
            )
            .to_numpy(
                dtype=np.float32
            )
        )

    return (
        x_train,
        x_tests,
    )


def train_temporal_experts(
    helpers,
    train,
    tests,
    features,
    y_train,
):
    (
        x_train,
        x_tests,
    ) = prepare_xgb(
        train,
        tests,
        features,
    )

    modes = [
        "UNIFORM",
        "EARLY",
        "LATE",
    ]

    outputs = {}

    for mode in modes:
        print(
            "  XGB expert:",
            mode,
        )

        weights = temporal_weights(
            train[
                "_query_date"
            ],
            mode,
        )

        model = xgb.XGBClassifier(
            **helpers.XGB_CONFIG
        )

        model.fit(
            x_train,
            y_train,
            sample_weight=weights,
        )

        if not np.array_equal(
            model.classes_,
            np.arange(
                EXPECTED_SITES
            ),
        ):
            raise RuntimeError(
                "Unexpected XGB classes."
            )

        outputs[
            mode
        ] = {
            name:
                model.predict_proba(
                    x
                )
                .astype(
                    np.float64
                )
            for name, x
            in x_tests.items()
        }

    candidates = {
        "UNIFORM":
            {
                window:
                    outputs[
                        "UNIFORM"
                    ][
                        window
                    ]
                for window in tests
            },

        "TEMPORAL_SYMMETRIC3":
            {
                window:
                    geometric_mean(
                        [
                            outputs[
                                "EARLY"
                            ][
                                window
                            ],
                            outputs[
                                "UNIFORM"
                            ][
                                window
                            ],
                            outputs[
                                "LATE"
                            ][
                                window
                            ],
                        ]
                    )
                for window in tests
            },

        "TEMPORAL_RECENCY2":
            {
                window:
                    geometric_mean(
                        [
                            outputs[
                                "UNIFORM"
                            ][
                                window
                            ],
                            outputs[
                                "LATE"
                            ][
                                window
                            ],
                        ]
                    )
                for window in tests
            },
    }

    return (
        outputs,
        candidates,
    )


def prepare_origin(
    mod,
    helpers,
    m11f,
    captures,
    daily_all,
    candidate_labels,
    label_to_idx,
    features,
    spec,
    origin,
    device,
    seed_rows,
):
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

    y_train = mod.labels_to_indices(
        train[
            "site_label"
        ],
        label_to_idx,
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

    daily_history = mod.standardized_daily_frame(
        train_daily,
        features,
        medians,
        scaler,
    )

    models = m11f.train_models(
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

    ltd_probs = {}

    for window in tests:
        ltd_probs[
            window
        ] = m11f.ensemble_predict(
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

    (
        expert_probs,
        xgb_candidates,
    ) = train_temporal_experts(
        helpers,
        train,
        tests,
        features,
        y_train,
    )

    return {
        "train":
            train,

        "tests":
            tests,

        "y_tests":
            y_tests,

        "test_dates":
            test_dates,

        "ltd_probs":
            ltd_probs,

        "expert_probs":
            expert_probs,

        "xgb_candidates":
            xgb_candidates,

        "models":
            models,
    }


def evaluate_candidate(
    helpers,
    data,
    candidate_name,
):
    rows = []
    scores = {}

    for window in [
        "NEAR",
        "MID",
        "FAR",
    ]:
        y = data[
            "y_tests"
        ][
            window
        ]

        xgb_probs = data[
            "xgb_candidates"
        ][
            candidate_name
        ][
            window
        ]

        ltd = data[
            "ltd_probs"
        ][
            window
        ]

        macro = helpers.fuse(
            xgb_probs,
            ltd,
        )

        (
            xgb_metrics,
            _,
            _,
        ) = helpers.evaluate(
            xgb_probs,
            y,
        )

        (
            macro_metrics,
            _,
            _,
        ) = helpers.evaluate(
            macro,
            y,
        )

        rows.append(
            {
                "candidate":
                    candidate_name,

                "window":
                    window,

                **{
                    f"xgb_{k}":
                        v
                    for k, v
                    in xgb_metrics.items()
                },

                **{
                    f"macro_{k}":
                        v
                    for k, v
                    in macro_metrics.items()
                },
            }
        )

        scores[
            window
        ] = {
            "y":
                y,

            "dates":
                pd.to_datetime(
                    data[
                        "test_dates"
                    ][
                        window
                    ]
                )
                .to_numpy(
                    dtype="datetime64[D]"
                ),

            "xgb":
                xgb_probs,

            "macro":
                macro,
        }

    return (
        pd.DataFrame(
            rows
        ),
        scores,
    )


def main():
    print("=" * 78)
    print(
        "11H — FROZEN TEMPORAL ENVIRONMENT ENSEMBLE"
    )
    print("=" * 78)

    print(
        "No online adaptation."
    )

    print(
        "No memory updates at inference."
    )

    print(
        "No Future-B values/scores used."
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

    source11f = (
        root
        / "src/features/macro_v2/"
        "11F_oracle_memory_refresh.py"
    )

    mod = load_module(
        source07a,
        "ltd_07a_for_11h",
    )

    helpers = load_module(
        source11b,
        "ltd_11b_helpers_for_11h",
    )

    m11f = load_module(
        source11f,
        "ltd_11f_for_11h",
    )

    (
        features,
        frozen_path,
    ) = mod.load_frozen_features()

    (
        captures,
        historical_path,
    ) = load_historical_65()

    captures[
        "site_label"
    ] = captures[
        "site_label"
    ].astype(str)

    captures[
        "_query_date"
    ] = extract_capture_dates(
        captures
    )

    unique_dates = np.array(
        sorted(
            captures[
                "_query_date"
            ].unique()
        )
    )

    candidate_labels = np.array(
        sorted(
            captures[
                "site_label"
            ].unique()
        ),
        dtype=str,
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

    seed_rows = []

    origin_data = {}

    for origin, spec in origins.items():
        print()
        print(
            origin
        )

        origin_data[
            origin
        ] = prepare_origin(
            mod,
            helpers,
            m11f,
            captures,
            daily_all,
            candidate_labels,
            label_to_idx,
            features,
            spec,
            origin,
            device,
            seed_rows,
        )

    # ==================================================
    # ORIGIN14 selection
    # ==================================================

    grid_parts = []
    score14 = {}

    for candidate in [
        "UNIFORM",
        "TEMPORAL_SYMMETRIC3",
        "TEMPORAL_RECENCY2",
    ]:
        (
            result,
            scores,
        ) = evaluate_candidate(
            helpers,
            origin_data[
                "ORIGIN14"
            ],
            candidate,
        )

        grid_parts.append(
            result
        )

        score14[
            candidate
        ] = scores

    grid14 = pd.concat(
        grid_parts,
        ignore_index=True,
    )

    candidate_rows = []

    for candidate, group in (
        grid14.groupby(
            "candidate"
        )
    ):
        by = group.set_index(
            "window"
        )

        candidate_rows.append(
            {
                "candidate":
                    candidate,

                "far_macro_f1":
                    float(
                        by.loc[
                            "FAR",
                            "macro_macro_f1",
                        ]
                    ),

                "mean_macro_f1":
                    float(
                        group[
                            "macro_macro_f1"
                        ].mean()
                    ),

                "far_xgb_macro_f1":
                    float(
                        by.loc[
                            "FAR",
                            "xgb_macro_f1",
                        ]
                    ),

                "near_macro_f1":
                    float(
                        by.loc[
                            "NEAR",
                            "macro_macro_f1",
                        ]
                    ),
            }
        )

    candidate_summary = pd.DataFrame(
        candidate_rows
    )

    nonbaseline = (
        candidate_summary[
            candidate_summary[
                "candidate"
            ]
            !=
            "UNIFORM"
        ]
        .copy()
    )

    selected_row = (
        nonbaseline
        .sort_values(
            [
                "far_macro_f1",
                "mean_macro_f1",
                "candidate",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )
        .iloc[
            0
        ]
    )

    selected = str(
        selected_row[
            "candidate"
        ]
    )

    print(
        "Selected on ORIGIN14:",
        selected,
    )

    # ==================================================
    # Exact transfer ORIGIN28
    # ==================================================

    (
        baseline14,
        baseline14_scores,
    ) = evaluate_candidate(
        helpers,
        origin_data[
            "ORIGIN14"
        ],
        "UNIFORM",
    )

    (
        selected14,
        selected14_scores,
    ) = evaluate_candidate(
        helpers,
        origin_data[
            "ORIGIN14"
        ],
        selected,
    )

    (
        baseline28,
        baseline28_scores,
    ) = evaluate_candidate(
        helpers,
        origin_data[
            "ORIGIN28"
        ],
        "UNIFORM",
    )

    (
        selected28,
        selected28_scores,
    ) = evaluate_candidate(
        helpers,
        origin_data[
            "ORIGIN28"
        ],
        selected,
    )

    comparison_rows = []
    bootstrap_rows = []

    for (
        origin,
        base,
        cand,
        base_scores,
        cand_scores,
    ) in [
        (
            "ORIGIN14",
            baseline14,
            selected14,
            baseline14_scores,
            selected14_scores,
        ),
        (
            "ORIGIN28",
            baseline28,
            selected28,
            baseline28_scores,
            selected28_scores,
        ),
    ]:
        base = base.set_index(
            "window"
        )

        cand = cand.set_index(
            "window"
        )

        for window in [
            "NEAR",
            "MID",
            "FAR",
        ]:
            row = {
                "origin":
                    origin,

                "candidate":
                    selected,

                "window":
                    window,
            }

            for model in [
                "xgb",
                "macro",
            ]:
                for metric in [
                    "accuracy",
                    "macro_f1",
                    "top5_accuracy",
                    "mrr",
                ]:
                    key = (
                        f"{model}_{metric}"
                    )

                    row[
                        f"delta_{key}"
                    ] = (
                        float(
                            cand.loc[
                                window,
                                key,
                            ]
                        )
                        -
                        float(
                            base.loc[
                                window,
                                key,
                            ]
                        )
                    )

            comparison_rows.append(
                row
            )

            for model in [
                "xgb",
                "macro",
            ]:
                boot = helpers.bootstrap_delta(
                    base_scores[
                        window
                    ][
                        "y"
                    ],
                    base_scores[
                        window
                    ][
                        "dates"
                    ],
                    base_scores[
                        window
                    ][
                        model
                    ],
                    cand_scores[
                        window
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
                                        values
                                        >
                                        0
                                    )
                                ),
                        }
                    )

    comparison = pd.DataFrame(
        comparison_rows
    )

    bootstrap = pd.DataFrame(
        bootstrap_rows
    )

    far = comparison[
        comparison[
            "window"
        ]
        ==
        "FAR"
    ]

    near = comparison[
        comparison[
            "window"
        ]
        ==
        "NEAR"
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
            "macro"
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

    promotion_pass = bool(
        far[
            "delta_macro_macro_f1"
        ].mean()
        >=
        MIN_MEAN_FAR_FUSION_F1_GAIN

        and

        (
            far[
                "delta_macro_macro_f1"
            ]
            >
            0
        ).all()

        and

        far[
            "delta_xgb_macro_f1"
        ].mean()
        >
        0

        and

        near[
            "delta_macro_macro_f1"
        ].min()
        >=
        -MAX_NEAR_FUSION_F1_DROP

        and

        (
            far_boot[
                "fraction_delta_gt_0"
            ]
            >=
            MIN_FAR_BOOTSTRAP_POSITIVE
        ).all()
    )

    grid_path = (
        OUT
        / "11H_origin14_candidate_grid.csv"
    )

    summary_path = (
        OUT
        / "11H_origin14_candidate_summary.csv"
    )

    transfer_path = (
        OUT
        / "11H_selected_transfer.csv"
    )

    bootstrap_path = (
        OUT
        / "11H_day_block_bootstrap.csv"
    )

    seed_path = (
        OUT
        / "11H_seed_training.csv"
    )

    grid14.to_csv(
        grid_path,
        index=False,
    )

    candidate_summary.to_csv(
        summary_path,
        index=False,
    )

    comparison.to_csv(
        transfer_path,
        index=False,
    )

    bootstrap.to_csv(
        bootstrap_path,
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
            "11H_frozen_temporal_environment_ensemble",

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "git_commit":
            git_commit(),

        "scientific_status":
            "FROZEN_MODEL_TEMPORAL_GENERALIZATION",

        "research_objective":
            (
                "Reduce longitudinal degradation without "
                "retraining, fine-tuning, test-time adaptation "
                "or memory updates."
            ),

        "data_policy": {
            "historical_only":
                True,

            "future_b_values_used":
                False,

            "future_b_scores_used":
                False,

            "test_time_parameter_updates":
                False,

            "test_time_memory_updates":
                False,

            "origin14_selection":
                True,

            "origin28_selection":
                False,
        },

        "experts": {
            "UNIFORM":
                "all training samples weight 1",

            "EARLY":
                (
                    "all training samples retained; "
                    "earlier dates receive up to "
                    f"{EDGE_WEIGHT_RATIO}x relative edge weight"
                ),

            "LATE":
                (
                    "all training samples retained; "
                    "later dates receive up to "
                    f"{EDGE_WEIGHT_RATIO}x relative edge weight"
                ),

            "edge_weight_ratio":
                EDGE_WEIGHT_RATIO,
        },

        "candidate_ensembles": [
            "TEMPORAL_SYMMETRIC3",
            "TEMPORAL_RECENCY2",
        ],

        "selected":
            selected,

        "promotion_gate": {
            "mean_far_macro_f1_gain_min":
                MIN_MEAN_FAR_FUSION_F1_GAIN,

            "all_far_macro_f1_positive":
                True,

            "mean_far_xgb_f1_positive":
                True,

            "max_near_macro_f1_drop":
                MAX_NEAR_FUSION_F1_DROP,

            "all_far_bootstrap_positive_fraction_min":
                MIN_FAR_BOOTSTRAP_POSITIVE,

            "passed":
                promotion_pass,
        },

        "inputs": {
            "historical_sha256":
                sha256(
                    historical_path
                ),

            "base128_sha256":
                sha256(
                    frozen_path
                ),

            "11F_source_sha256":
                sha256(
                    source11f
                ),
        },

        "outputs": {
            "grid":
                sha256(
                    grid_path
                ),

            "summary":
                sha256(
                    summary_path
                ),

            "transfer":
                sha256(
                    transfer_path
                ),

            "bootstrap":
                sha256(
                    bootstrap_path
                ),

            "seed_training":
                sha256(
                    seed_path
                ),
        },

        "next_if_pass":
            (
                "11I frozen robust temporal training "
                "around selected temporal ensemble"
            ),

        "next_if_fail":
            (
                "11I temporal GroupDRO / worst-environment "
                "optimization with frozen inference"
            ),
    }

    write_json(
        OUT
        / "11H_manifest.json",
        manifest,
    )

    print()
    print("=" * 78)
    print(
        "ORIGIN14 CANDIDATES"
    )
    print("=" * 78)

    print(
        candidate_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "SELECTED:",
        selected,
    )

    print()
    print("=" * 78)
    print(
        "TRANSFER"
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
        "PROMOTION PASS:",
        promotion_pass,
    )

    print(
        "Frozen inference: YES"
    )

    print(
        "Future-B accessed: NO"
    )


if __name__ == "__main__":
    main()
