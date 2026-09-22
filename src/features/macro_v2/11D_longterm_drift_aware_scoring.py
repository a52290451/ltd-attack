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

PROTO_BETAS = [
    0.25,
    0.50,
    0.75,
]

PROTO_METHODS = [
    "LONGTERM",
    "DUALANCHOR",
]

EPS = 1e-12

MIN_MEAN_FAR_FUSION_F1_GAIN = 0.005
MAX_NEAR_FUSION_F1_DROP = 0.010

BOOTSTRAP_POSITIVE_FRACTION = 0.90


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


def softmax(
    logits,
):
    z = (
        logits
        -
        logits.max(
            axis=1,
            keepdims=True,
        )
    )

    p = np.exp(z)

    return (
        p
        /
        p.sum(
            axis=1,
            keepdims=True,
        )
    )


def geometric_fusion(
    first,
    second,
    second_weight,
):
    logits = (
        (
            1.0
            -
            second_weight
        )
        *
        np.log(
            np.clip(
                first,
                EPS,
                1.0,
            )
        )
        +
        second_weight
        *
        np.log(
            np.clip(
                second,
                EPS,
                1.0,
            )
        )
    )

    return softmax(
        logits
    )


def cosine_zscores(
    queries,
    prototypes,
):
    q = np.asarray(
        queries,
        dtype=np.float64,
    )

    p = np.asarray(
        prototypes,
        dtype=np.float64,
    )

    q_norm = np.linalg.norm(
        q,
        axis=1,
        keepdims=True,
    )

    p_norm = np.linalg.norm(
        p,
        axis=1,
        keepdims=True,
    )

    q_norm = np.where(
        q_norm > 1e-12,
        q_norm,
        1.0,
    )

    p_norm = np.where(
        p_norm > 1e-12,
        p_norm,
        1.0,
    )

    similarity = (
        q
        /
        q_norm
    ) @ (
        p
        /
        p_norm
    ).T

    mean = similarity.mean(
        axis=1,
        keepdims=True,
    )

    std = similarity.std(
        axis=1,
        keepdims=True,
    )

    std = np.where(
        std > 1e-6,
        std,
        1.0,
    )

    return (
        similarity
        -
        mean
    ) / std


def build_prototypes(
    daily_history,
    candidate_labels,
    features,
):
    longterm = []
    recent5 = []

    for site in candidate_labels:
        h = (
            daily_history[
                daily_history[
                    "site_label"
                ]
                ==
                site
            ]
            .sort_values(
                "date"
            )
        )

        if h.empty:
            raise RuntimeError(
                f"No daily history for {site}"
            )

        values = (
            h[
                features
            ]
            .to_numpy(
                dtype=np.float32
            )
        )

        longterm.append(
            np.median(
                values,
                axis=0,
            )
        )

        recent5.append(
            np.median(
                values[
                    -5:
                ],
                axis=0,
            )
        )

    return (
        np.stack(
            longterm,
            axis=0,
        ).astype(
            np.float32
        ),
        np.stack(
            recent5,
            axis=0,
        ).astype(
            np.float32
        ),
    )


def prototype_probabilities(
    x,
    longterm,
    recent5,
):
    z_long = cosine_zscores(
        x,
        longterm,
    )

    z_recent = cosine_zscores(
        x,
        recent5,
    )

    long_probs = softmax(
        z_long
    )

    # Drift-tolerant dual anchor:
    # each candidate may match either its long-term
    # identity or its most recent training state.
    dual_scores = np.maximum(
        z_long,
        z_recent,
    )

    dual_probs = softmax(
        dual_scores
    )

    return {
        "LONGTERM":
            long_probs,

        "DUALANCHOR":
            dual_probs,
    }


def train_recent_ltd(
    mod,
    helpers,
    x_train,
    y_train,
    train_dates,
    daily_history,
    candidate_labels,
    features,
    x_tests,
    test_dates,
    device,
    origin,
    seed_rows,
):
    mod.CONTEXT_DAYS = (
        CONTEXT_DAYS
    )

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
            f"{origin}: no causal LTD training contexts."
        )

    test_caches = {}

    for window in x_tests:
        (
            cache,
            skipped,
        ) = mod.build_context_cache(
            daily_history,
            candidate_labels,
            test_dates[
                window
            ],
            features,
        )

        if skipped:
            raise RuntimeError(
                f"{origin}/{window}: "
                f"missing test contexts {skipped}"
            )

        test_caches[
            window
        ] = cache

    seed_probs = {
        window: []
        for window in x_tests
    }

    for seed in mod.SEEDS:
        print(
            origin,
            "RECENT5 seed=",
            seed,
        )

        mod.set_seed(
            seed
        )

        mod.CONTEXT_DAYS = (
            CONTEXT_DAYS
        )

        model = (
            mod.LTDPairScorer()
            .to(
                device
            )
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

        fit_seconds = (
            time.perf_counter()
            -
            t0
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
                    fit_seconds,
            }
        )

        for window in x_tests:
            probs = helpers.predict_ltd(
                mod,
                model,
                x_tests[
                    window
                ],
                test_dates[
                    window
                ],
                test_caches[
                    window
                ],
                device,
            )

            seed_probs[
                window
            ].append(
                probs
            )

        del model

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result = {}

    for window in x_tests:
        ensemble = (
            np.stack(
                seed_probs[
                    window
                ],
                axis=0,
            )
            .mean(
                axis=0
            )
        )

        ensemble /= (
            ensemble.sum(
                axis=1,
                keepdims=True,
            )
        )

        result[
            window
        ] = ensemble

    return result


def prepare_origin(
    mod,
    helpers,
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

    if (
        train[
            "site_label"
        ].nunique()
        != EXPECTED_SITES
    ):
        raise RuntimeError(
            f"{origin}: incomplete train coverage."
        )

    for name, frame in tests.items():
        if (
            frame[
                "site_label"
            ].nunique()
            != EXPECTED_SITES
        ):
            raise RuntimeError(
                f"{origin}/{name}: incomplete coverage."
            )

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

    daily_history = (
        mod.standardized_daily_frame(
            train_daily,
            features,
            medians,
            scaler,
        )
    )

    xgb_probs = helpers.train_xgb(
        train,
        tests,
        features,
        y_train,
    )

    ltd_probs = train_recent_ltd(
        mod,
        helpers,
        x_train,
        y_train,
        train_dates,
        daily_history,
        candidate_labels,
        features,
        x_tests,
        test_dates,
        device,
        origin,
        seed_rows,
    )

    (
        longterm_proto,
        recent5_proto,
    ) = build_prototypes(
        daily_history,
        candidate_labels,
        features,
    )

    proto_probs = {
        window:
            prototype_probabilities(
                x_tests[
                    window
                ],
                longterm_proto,
                recent5_proto,
            )
        for window in tests
    }

    return {
        "train":
            train,

        "tests":
            tests,

        "y_tests":
            y_tests,

        "test_dates":
            test_dates,

        "xgb_probs":
            xgb_probs,

        "ltd_probs":
            ltd_probs,

        "proto_probs":
            proto_probs,
    }


def evaluate_candidate(
    helpers,
    origin_data,
    method,
    beta,
):
    rows = []
    scores = {}

    for window in [
        "NEAR",
        "MID",
        "FAR",
    ]:
        y = origin_data[
            "y_tests"
        ][
            window
        ]

        xgb = origin_data[
            "xgb_probs"
        ][
            window
        ]

        ltd = origin_data[
            "ltd_probs"
        ][
            window
        ]

        baseline_fusion = (
            helpers.fuse(
                xgb,
                ltd,
            )
        )

        if method == "BASELINE":
            branch = ltd
            fusion = baseline_fusion
            proto = None

        else:
            proto = (
                origin_data[
                    "proto_probs"
                ][
                    window
                ][
                    method
                ]
            )

            branch = geometric_fusion(
                ltd,
                proto,
                beta,
            )

            # Preserve the canonical outer Macro weight:
            # XGB=.625 / longitudinal branch=.375.
            fusion = helpers.fuse(
                xgb,
                branch,
            )

        (
            ltd_metrics,
            _,
            _,
        ) = helpers.evaluate(
            branch,
            y,
        )

        (
            fusion_metrics,
            _,
            _,
        ) = helpers.evaluate(
            fusion,
            y,
        )

        if proto is not None:
            (
                proto_metrics,
                _,
                _,
            ) = helpers.evaluate(
                proto,
                y,
            )
        else:
            proto_metrics = {
                "accuracy":
                    np.nan,
                "macro_f1":
                    np.nan,
                "top5_accuracy":
                    np.nan,
                "mrr":
                    np.nan,
                "mean_true_rank":
                    np.nan,
            }

        candidate = (
            "BASELINE_RECENT5"
            if method == "BASELINE"
            else
            (
                f"{method}_"
                f"BETA{beta:.2f}"
            )
        )

        rows.append(
            {
                "candidate":
                    candidate,

                "method":
                    method,

                "beta_proto_within_longitudinal":
                    beta,

                "window":
                    window,

                "n_test":
                    len(y),

                **{
                    f"prototype_{k}":
                        v
                    for k, v
                    in proto_metrics.items()
                },

                **{
                    f"branch_{k}":
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

        scores[
            window
        ] = {
            "y":
                y,

            "dates":
                pd.to_datetime(
                    origin_data[
                        "test_dates"
                    ][
                        window
                    ]
                )
                .to_numpy(
                    dtype="datetime64[D]"
                ),

            "branch":
                branch,

            "fusion":
                fusion,

            "baseline_branch":
                ltd,

            "baseline_fusion":
                baseline_fusion,
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
        "11D — LONG-TERM / DRIFT-AWARE CANDIDATE SCORING"
    )
    print("=" * 78)

    print(
        "Historical-only robustness development."
    )

    print(
        "BASE128 + RECENT5 remains control."
    )

    print(
        "Future-B values/scores are not used."
    )

    source07a = (
        repo_root()
        / "src/features/macro_v2/"
        "07A_candidate_conditioned_temporal_encoder.py"
    )

    source11b = (
        repo_root()
        / "src/features/macro_v2/"
        "11B_multiscale_longitudinal_memory.py"
    )

    mod = load_module(
        source07a,
        "ltd_07a_for_11d",
    )

    helpers = load_module(
        source11b,
        "ltd_11b_helpers_for_11d",
    )

    mod.CONTEXT_DAYS = (
        CONTEXT_DAYS
    )

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

    seed_rows = []

    # ==================================================
    # ORIGIN14 — candidate selection
    # ==================================================

    print()
    print("#" * 78)
    print(
        "ORIGIN14 — CANDIDATE SELECTION"
    )
    print("#" * 78)

    origin14 = prepare_origin(
        mod,
        helpers,
        captures,
        daily_all,
        candidate_labels,
        label_to_idx,
        features,
        origins[
            "ORIGIN14"
        ],
        "ORIGIN14",
        device,
        seed_rows,
    )

    (
        baseline14,
        baseline14_scores,
    ) = evaluate_candidate(
        helpers,
        origin14,
        "BASELINE",
        0.0,
    )

    grid_parts = []

    score14 = {}

    for method in PROTO_METHODS:
        for beta in PROTO_BETAS:
            (
                result,
                scores,
            ) = evaluate_candidate(
                helpers,
                origin14,
                method,
                beta,
            )

            grid_parts.append(
                result
            )

            candidate = (
                f"{method}_"
                f"BETA{beta:.2f}"
            )

            score14[
                candidate
            ] = scores

    grid14 = pd.concat(
        grid_parts,
        ignore_index=True,
    )

    candidate_summary_rows = []

    for candidate, group in (
        grid14.groupby(
            "candidate"
        )
    ):
        by_window = (
            group
            .set_index(
                "window"
            )
        )

        candidate_summary_rows.append(
            {
                "candidate":
                    candidate,

                "method":
                    group[
                        "method"
                    ].iloc[
                        0
                    ],

                "beta":
                    float(
                        group[
                            "beta_proto_within_longitudinal"
                        ].iloc[
                            0
                        ]
                    ),

                "far_fusion_macro_f1":
                    float(
                        by_window.loc[
                            "FAR",
                            "fusion_macro_f1",
                        ]
                    ),

                "far_fusion_accuracy":
                    float(
                        by_window.loc[
                            "FAR",
                            "fusion_accuracy",
                        ]
                    ),

                "mean_fusion_macro_f1":
                    float(
                        group[
                            "fusion_macro_f1"
                        ].mean()
                    ),

                "near_fusion_macro_f1":
                    float(
                        by_window.loc[
                            "NEAR",
                            "fusion_macro_f1",
                        ]
                    ),

                "far_branch_macro_f1":
                    float(
                        by_window.loc[
                            "FAR",
                            "branch_macro_f1",
                        ]
                    ),
            }
        )

    candidate_summary = pd.DataFrame(
        candidate_summary_rows
    )

    # Predeclared selection:
    # 1. highest FAR Fusion Macro-F1
    # 2. highest mean Fusion Macro-F1
    # 3. highest FAR Accuracy
    # 4. smaller prototype beta
    # 5. lexical method tie
    selected = (
        candidate_summary
        .sort_values(
            [
                "far_fusion_macro_f1",
                "mean_fusion_macro_f1",
                "far_fusion_accuracy",
                "beta",
                "method",
            ],
            ascending=[
                False,
                False,
                False,
                True,
                True,
            ],
        )
        .iloc[
            0
        ]
    )

    selected_candidate = str(
        selected[
            "candidate"
        ]
    )

    selected_method = str(
        selected[
            "method"
        ]
    )

    selected_beta = float(
        selected[
            "beta"
        ]
    )

    print()
    print(
        "Selected on ORIGIN14:",
        selected_candidate,
    )

    # ==================================================
    # ORIGIN28 — exactly one transfer candidate
    # ==================================================

    print()
    print("#" * 78)
    print(
        "ORIGIN28 — TRANSFER ONLY"
    )
    print("#" * 78)

    origin28 = prepare_origin(
        mod,
        helpers,
        captures,
        daily_all,
        candidate_labels,
        label_to_idx,
        features,
        origins[
            "ORIGIN28"
        ],
        "ORIGIN28",
        device,
        seed_rows,
    )

    (
        baseline28,
        baseline28_scores,
    ) = evaluate_candidate(
        helpers,
        origin28,
        "BASELINE",
        0.0,
    )

    (
        selected28,
        selected28_scores,
    ) = evaluate_candidate(
        helpers,
        origin28,
        selected_method,
        selected_beta,
    )

    selected14 = (
        grid14[
            grid14[
                "candidate"
            ]
            ==
            selected_candidate
        ]
        .copy()
    )

    # ==================================================
    # DELTAS
    # ==================================================

    comparison_rows = []

    bootstrap_rows = []

    score_sets = {
        "ORIGIN14": {
            "baseline":
                baseline14_scores,

            "candidate":
                score14[
                    selected_candidate
                ],
        },

        "ORIGIN28": {
            "baseline":
                baseline28_scores,

            "candidate":
                selected28_scores,
        },
    }

    result_sets = {
        "ORIGIN14": {
            "baseline":
                baseline14,

            "candidate":
                selected14,
        },

        "ORIGIN28": {
            "baseline":
                baseline28,

            "candidate":
                selected28,
        },
    }

    for origin in [
        "ORIGIN14",
        "ORIGIN28",
    ]:
        base = (
            result_sets[
                origin
            ][
                "baseline"
            ]
            .set_index(
                "window"
            )
        )

        cand = (
            result_sets[
                origin
            ][
                "candidate"
            ]
            .set_index(
                "window"
            )
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
                    selected_candidate,

                "window":
                    window,
            }

            for model in [
                "branch",
                "fusion",
            ]:
                for metric in [
                    "accuracy",
                    "macro_f1",
                    "top5_accuracy",
                    "mrr",
                ]:
                    key = (
                        f"{model}_"
                        f"{metric}"
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
                "branch",
                "fusion",
            ]:
                boot = helpers.bootstrap_delta(
                    score_sets[
                        origin
                    ][
                        "baseline"
                    ][
                        window
                    ][
                        "y"
                    ],
                    score_sets[
                        origin
                    ][
                        "baseline"
                    ][
                        window
                    ][
                        "dates"
                    ],
                    score_sets[
                        origin
                    ][
                        "baseline"
                    ][
                        window
                    ][
                        model
                    ],
                    score_sets[
                        origin
                    ][
                        "candidate"
                    ][
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
                                        values > 0
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

    far = (
        comparison[
            comparison[
                "window"
            ]
            ==
            "FAR"
        ]
    )

    near = (
        comparison[
            comparison[
                "window"
            ]
            ==
            "NEAR"
        ]
    )

    far_boot = (
        bootstrap[
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
    )

    promotion_pass = bool(
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
            "delta_branch_macro_f1"
        ].mean()
        >
        0

        and

        near[
            "delta_fusion_macro_f1"
        ].min()
        >=
        -MAX_NEAR_FUSION_F1_DROP

        and

        (
            far_boot[
                "fraction_delta_gt_0"
            ]
            >=
            BOOTSTRAP_POSITIVE_FRACTION
        ).all()
    )

    # ==================================================
    # SAVE
    # ==================================================

    grid_path = (
        OUT
        / "11D_origin14_candidate_grid.csv"
    )

    candidate_summary_path = (
        OUT
        / "11D_origin14_candidate_summary.csv"
    )

    comparison_path = (
        OUT
        / "11D_selected_transfer.csv"
    )

    bootstrap_path = (
        OUT
        / "11D_day_block_bootstrap.csv"
    )

    seed_path = (
        OUT
        / "11D_seed_training.csv"
    )

    grid14.to_csv(
        grid_path,
        index=False,
    )

    candidate_summary.to_csv(
        candidate_summary_path,
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

    pd.DataFrame(
        seed_rows
    ).to_csv(
        seed_path,
        index=False,
    )

    manifest = {
        "stage":
            "11D_longterm_drift_aware_scoring",

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
                "Separating recent candidate state from "
                "long-term candidate identity improves "
                "stale-history robustness without forcing "
                "temporal invariance."
            ),

        "data_policy": {
            "historical_only":
                True,

            "future_b_values_used":
                False,

            "future_b_scores_used":
                False,

            "test_observations_update_history":
                False,

            "candidate_selection_origin":
                "ORIGIN14",

            "origin28_used_for_selection":
                False,
        },

        "control": {
            "features":
                "BASE128",

            "memory":
                "RECENT5",

            "macro_fusion":
                "XGB=.625 / LTD=.375",
        },

        "prototype_scoring": {
            "space":
                "training-standardized BASE128",

            "longterm":
                (
                    "per-site median across all training "
                    "site-day profiles"
                ),

            "recent_anchor":
                (
                    "per-site median of last five training "
                    "site-day profiles"
                ),

            "similarity":
                "cosine",

            "query_normalization":
                (
                    "candidate cosine similarities converted "
                    "to per-query z-scores before softmax"
                ),

            "DUALANCHOR":
                (
                    "per candidate maximum of long-term "
                    "and recent-anchor z-score"
                ),

            "test_history_updates":
                False,
        },

        "candidate_grid": {
            "methods":
                PROTO_METHODS,

            "prototype_beta_within_longitudinal_branch":
                PROTO_BETAS,

            "outer_xgb_ltd_alpha":
                helpers.ALPHA_LTD,

            "selection_primary":
                "ORIGIN14 FAR Fusion Macro-F1",

            "selection_secondary":
                "ORIGIN14 mean NEAR/MID/FAR Fusion Macro-F1",

            "selected_candidate":
                selected_candidate,

            "selected_method":
                selected_method,

            "selected_beta":
                selected_beta,

            "transfer":
                "exactly one selected candidate to ORIGIN28",
        },

        "promotion_gate": {
            "mean_far_fusion_macro_f1_gain_min":
                MIN_MEAN_FAR_FUSION_F1_GAIN,

            "all_far_fusion_macro_f1_positive":
                True,

            "mean_far_branch_macro_f1_positive":
                True,

            "max_near_fusion_macro_f1_drop":
                MAX_NEAR_FUSION_F1_DROP,

            "all_far_bootstrap_positive_fraction_min":
                BOOTSTRAP_POSITIVE_FRACTION,

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
            "origin14_grid":
                sha256(
                    grid_path
                ),

            "candidate_summary":
                sha256(
                    candidate_summary_path
                ),

            "selected_transfer":
                sha256(
                    comparison_path
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
                "11E combine supported long-term scoring "
                "with the independently positive multiscale "
                "memory mechanism"
            ),

        "next_if_fail":
            (
                "11E test explicit longitudinal trajectory "
                "and velocity scoring without changing "
                "BASE128 or Future-B"
            ),
    }

    write_json(
        OUT
        / "11D_manifest.json",
        manifest,
    )

    print()
    print("=" * 78)
    print(
        "ORIGIN14 CANDIDATE SUMMARY"
    )
    print("=" * 78)

    print(
        candidate_summary
        .sort_values(
            "far_fusion_macro_f1",
            ascending=False,
        )
        .to_string(
            index=False
        )
    )

    print()
    print(
        "SELECTED:",
        selected_candidate,
    )

    print()
    print("=" * 78)
    print(
        "SELECTED CANDIDATE TRANSFER"
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
        bootstrap[
            bootstrap[
                "window"
            ]
            ==
            "FAR"
        ]
        .to_string(
            index=False
        )
    )

    print()
    print(
        "PROMOTION PASS:",
        promotion_pass,
    )

    print(
        "Future-B was not accessed."
    )


if __name__ == "__main__":
    main()
