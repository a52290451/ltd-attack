from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import gc
import importlib.util
import sys

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

MIN_MEAN_FAR_GAIN_VS_UNIFORM = 0.005
MAX_NEAR_DROP_VS_UNIFORM = 0.010
MIN_BOOTSTRAP_POSITIVE = 0.90

MAX_NEAR_DROP_VS_11H = 0.005


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

    if (
        spec is None
        or spec.loader is None
    ):
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


def arithmetic_mixture(
    probabilities,
):
    p = np.mean(
        np.stack(
            probabilities,
            axis=0,
        ),
        axis=0,
    )

    p = np.clip(
        p,
        1e-12,
        None,
    )

    return (
        p
        /
        p.sum(
            axis=1,
            keepdims=True,
        )
    )


def expert_diversity(
    helpers,
    expert_probs,
    y,
):
    names = [
        "EARLY",
        "UNIFORM",
        "LATE",
    ]

    pred = {
        name:
            np.argmax(
                expert_probs[
                    name
                ],
                axis=1,
            )
        for name in names
    }

    rows = []

    all_matrix = np.stack(
        [
            pred[
                name
            ]
            for name in names
        ],
        axis=1,
    )

    all_agree = np.all(
        all_matrix
        ==
        all_matrix[
            :,
            [0]
        ],
        axis=1,
    )

    any_correct = np.any(
        all_matrix
        ==
        y[
            :,
            None
        ],
        axis=1,
    )

    uniform_correct = (
        pred[
            "UNIFORM"
        ]
        ==
        y
    )

    early_rescue = (
        (~uniform_correct)
        &
        (
            pred[
                "EARLY"
            ]
            ==
            y
        )
    )

    late_rescue = (
        (~uniform_correct)
        &
        (
            pred[
                "LATE"
            ]
            ==
            y
        )
    )

    for a, b in [
        (
            "EARLY",
            "UNIFORM",
        ),
        (
            "UNIFORM",
            "LATE",
        ),
        (
            "EARLY",
            "LATE",
        ),
    ]:
        rows.append(
            {
                "measure":
                    f"top1_disagreement_{a}_{b}",

                "value":
                    float(
                        np.mean(
                            pred[
                                a
                            ]
                            !=
                            pred[
                                b
                            ]
                        )
                    ),
            }
        )

    rows.extend(
        [
            {
                "measure":
                    "all_three_top1_agree",

                "value":
                    float(
                        np.mean(
                            all_agree
                        )
                    ),
            },

            {
                "measure":
                    "top1_union_oracle_DIAGNOSTIC",

                "value":
                    float(
                        np.mean(
                            any_correct
                        )
                    ),
            },

            {
                "measure":
                    "early_rescue_uniform_errors",

                "value":
                    float(
                        np.mean(
                            early_rescue
                        )
                    ),
            },

            {
                "measure":
                    "late_rescue_uniform_errors",

                "value":
                    float(
                        np.mean(
                            late_rescue
                        )
                    ),
            },
        ]
    )

    for name in names:
        m, _, _ = helpers.evaluate(
            expert_probs[
                name
            ],
            y,
        )

        rows.append(
            {
                "measure":
                    f"{name}_macro_f1",

                "value":
                    float(
                        m[
                            "macro_f1"
                        ]
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def evaluate_origin(
    helpers,
    data,
    origin,
):
    comparison_rows = []
    diversity_rows = []

    score_store = {}

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

        ltd = data[
            "ltd_probs"
        ][
            window
        ]

        expert_probs = {
            name:
                data[
                    "expert_probs"
                ][
                    name
                ][
                    window
                ]
            for name in [
                "EARLY",
                "UNIFORM",
                "LATE",
            ]
        }

        uniform_xgb = (
            expert_probs[
                "UNIFORM"
            ]
        )

        geometric_xgb = (
            data[
                "xgb_candidates"
            ][
                "TEMPORAL_SYMMETRIC3"
            ][
                window
            ]
        )

        arithmetic_xgb = (
            arithmetic_mixture(
                [
                    expert_probs[
                        "EARLY"
                    ],
                    expert_probs[
                        "UNIFORM"
                    ],
                    expert_probs[
                        "LATE"
                    ],
                ]
            )
        )

        uniform_macro = helpers.fuse(
            uniform_xgb,
            ltd,
        )

        geometric_macro = helpers.fuse(
            geometric_xgb,
            ltd,
        )

        arithmetic_macro = helpers.fuse(
            arithmetic_xgb,
            ltd,
        )

        probabilities = {
            "uniform_xgb":
                uniform_xgb,

            "11H_xgb":
                geometric_xgb,

            "mixture_xgb":
                arithmetic_xgb,

            "uniform_macro":
                uniform_macro,

            "11H_macro":
                geometric_macro,

            "mixture_macro":
                arithmetic_macro,
        }

        metrics = {}

        for name, probs in (
            probabilities.items()
        ):
            m, _, _ = helpers.evaluate(
                probs,
                y,
            )

            metrics[
                name
            ] = m

        row = {
            "origin":
                origin,

            "window":
                window,

            "n":
                len(y),

            "mixture_xgb_accuracy":
                metrics[
                    "mixture_xgb"
                ][
                    "accuracy"
                ],

            "mixture_xgb_macro_f1":
                metrics[
                    "mixture_xgb"
                ][
                    "macro_f1"
                ],

            "mixture_xgb_top5":
                metrics[
                    "mixture_xgb"
                ][
                    "top5_accuracy"
                ],

            "mixture_xgb_mrr":
                metrics[
                    "mixture_xgb"
                ][
                    "mrr"
                ],

            "mixture_macro_accuracy":
                metrics[
                    "mixture_macro"
                ][
                    "accuracy"
                ],

            "mixture_macro_macro_f1":
                metrics[
                    "mixture_macro"
                ][
                    "macro_f1"
                ],

            "mixture_macro_top5":
                metrics[
                    "mixture_macro"
                ][
                    "top5_accuracy"
                ],

            "mixture_macro_mrr":
                metrics[
                    "mixture_macro"
                ][
                    "mrr"
                ],
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
                candidate_value = (
                    metrics[
                        f"mixture_{model}"
                    ][
                        metric
                    ]
                )

                row[
                    (
                        f"delta_{model}_{metric}"
                        "_vs_uniform"
                    )
                ] = (
                    candidate_value
                    -
                    metrics[
                        f"uniform_{model}"
                    ][
                        metric
                    ]
                )

                row[
                    (
                        f"delta_{model}_{metric}"
                        "_vs_11H"
                    )
                ] = (
                    candidate_value
                    -
                    metrics[
                        f"11H_{model}"
                    ][
                        metric
                    ]
                )

        comparison_rows.append(
            row
        )

        diversity = expert_diversity(
            helpers,
            expert_probs,
            y,
        )

        diversity.insert(
            0,
            "window",
            window,
        )

        diversity.insert(
            0,
            "origin",
            origin,
        )

        diversity_rows.append(
            diversity
        )

        score_store[
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

            **probabilities,
        }

    return (
        pd.DataFrame(
            comparison_rows
        ),

        pd.concat(
            diversity_rows,
            ignore_index=True,
        ),

        score_store,
    )


def main():
    print("=" * 78)
    print(
        "11K — FROZEN TEMPORAL MIXTURE OF EXPERTS"
    )
    print("=" * 78)

    print(
        "Same EARLY / UNIFORM / LATE experts as 11H."
    )

    print(
        "11H = geometric product-style ensemble."
    )

    print(
        "11K = arithmetic mixture-style ensemble."
    )

    print(
        "No online adaptation."
    )

    print(
        "No memory updates."
    )

    print(
        "No Future-B values or scores."
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

    source11h = (
        root
        / "src/features/macro_v2/"
        "11H_frozen_temporal_environment_ensemble.py"
    )

    m11h = load_module(
        source11h,
        "m11h_for_11k",
    )

    mod = load_module(
        source07a,
        "ltd_07a_for_11k",
    )

    helpers = load_module(
        source11b,
        "ltd_11b_for_11k",
    )

    m11f = load_module(
        source11f,
        "ltd_11f_for_11k",
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
    ] = (
        captures[
            "site_label"
        ]
        .astype(str)
    )

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

    if len(
        unique_dates
    ) != 52:
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

    if len(
        candidate_labels
    ) != EXPECTED_SITES:
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

    comparison_parts = []
    diversity_parts = []

    all_scores = {}

    origin_data = {}

    for origin, spec in origins.items():
        print()
        print("#" * 78)
        print(origin)
        print("#" * 78)

        data = m11h.prepare_origin(
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

        (
            comparison,
            diversity,
            scores,
        ) = evaluate_origin(
            helpers,
            data,
            origin,
        )

        comparison_parts.append(
            comparison
        )

        diversity_parts.append(
            diversity
        )

        all_scores[
            origin
        ] = scores

        origin_data[
            origin
        ] = data

    comparison = pd.concat(
        comparison_parts,
        ignore_index=True,
    )

    diversity = pd.concat(
        diversity_parts,
        ignore_index=True,
    )

    # ==================================================
    # Paired day-block bootstrap
    # ==================================================

    bootstrap_rows = []

    for origin in origins:
        for window in [
            "NEAR",
            "MID",
            "FAR",
        ]:
            score = all_scores[
                origin
            ][
                window
            ]

            observed = (
                comparison[
                    (
                        comparison[
                            "origin"
                        ]
                        ==
                        origin
                    )
                    &
                    (
                        comparison[
                            "window"
                        ]
                        ==
                        window
                    )
                ]
                .iloc[
                    0
                ]
            )

            for model in [
                "xgb",
                "macro",
            ]:
                for reference in [
                    "uniform",
                    "11H",
                ]:
                    reference_key = (
                        f"{reference}_{model}"
                    )

                    candidate_key = (
                        f"mixture_{model}"
                    )

                    boot = helpers.bootstrap_delta(
                        score[
                            "y"
                        ],
                        score[
                            "dates"
                        ],
                        score[
                            reference_key
                        ],
                        score[
                            candidate_key
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

                                "reference":
                                    reference,

                                "metric":
                                    metric,

                                "observed_delta":
                                    float(
                                        observed[
                                            (
                                                f"delta_"
                                                f"{model}_"
                                                f"{metric}_"
                                                f"vs_{reference}"
                                            )
                                        ]
                                    ),

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

    far_boot_uniform = (
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
                "macro"
            )
            &
            (
                bootstrap[
                    "reference"
                ]
                ==
                "uniform"
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

    far_boot_11h = (
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
                "macro"
            )
            &
            (
                bootstrap[
                    "reference"
                ]
                ==
                "11H"
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
            "delta_macro_macro_f1_vs_uniform"
        ].mean()
        >=
        MIN_MEAN_FAR_GAIN_VS_UNIFORM

        and

        (
            far[
                "delta_macro_macro_f1_vs_uniform"
            ]
            >
            0
        ).all()

        and

        far[
            "delta_xgb_macro_f1_vs_uniform"
        ].mean()
        >
        0

        and

        near[
            "delta_macro_macro_f1_vs_uniform"
        ].min()
        >=
        -MAX_NEAR_DROP_VS_UNIFORM

        and

        (
            far_boot_uniform[
                "fraction_delta_gt_0"
            ]
            >=
            MIN_BOOTSTRAP_POSITIVE
        ).all()
    )

    beats_11h = bool(
        far[
            "delta_macro_macro_f1_vs_11H"
        ].mean()
        >
        0

        and

        (
            far[
                "delta_macro_macro_f1_vs_11H"
            ]
            >
            0
        ).all()

        and

        near[
            "delta_macro_macro_f1_vs_11H"
        ].min()
        >=
        -MAX_NEAR_DROP_VS_11H

        and

        (
            far_boot_11h[
                "fraction_delta_gt_0"
            ]
            >=
            MIN_BOOTSTRAP_POSITIVE
        ).all()
    )

    mean_far_vs_uniform = float(
        far[
            "delta_macro_macro_f1_vs_uniform"
        ].mean()
    )

    mean_far_vs_11h = float(
        far[
            "delta_macro_macro_f1_vs_11H"
        ].mean()
    )

    # ==================================================
    # Save
    # ==================================================

    comparison_path = (
        OUT
        / "11K_transfer.csv"
    )

    bootstrap_path = (
        OUT
        / "11K_day_block_bootstrap.csv"
    )

    diversity_path = (
        OUT
        / "11K_expert_diversity.csv"
    )

    seed_path = (
        OUT
        / "11K_seed_training.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    bootstrap.to_csv(
        bootstrap_path,
        index=False,
    )

    diversity.to_csv(
        diversity_path,
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
            "11K_frozen_temporal_mixture_of_experts",

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "git_commit":
            git_commit(),

        "scientific_status":
            "FROZEN_MODEL_TEMPORAL_GENERALIZATION",

        "hypothesis":
            (
                "Temporal expert diversity is useful because "
                "different historical temporal states provide "
                "alternative valid hypotheses. Arithmetic "
                "mixture should therefore retain robustness "
                "better than geometric consensus."
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
                False,

            "origin28_selection":
                False,
        },

        "experts": {
            "same_as_11H":
                True,

            "members": [
                "EARLY",
                "UNIFORM",
                "LATE",
            ],

            "features":
                "BASE128",

            "edge_weight_ratio":
                m11h.EDGE_WEIGHT_RATIO,
        },

        "control_11H": {
            "aggregation":
                "geometric mean",

            "interpretation":
                "product / consensus style",
        },

        "candidate_11K": {
            "aggregation":
                "arithmetic probability mean",

            "interpretation":
                "mixture / alternative-hypothesis style",

            "new_hyperparameters":
                False,
        },

        "promotion_gate_vs_uniform": {
            "mean_far_macro_f1_gain_min":
                MIN_MEAN_FAR_GAIN_VS_UNIFORM,

            "all_far_positive":
                True,

            "mean_far_xgb_positive":
                True,

            "max_near_drop":
                MAX_NEAR_DROP_VS_UNIFORM,

            "all_far_bootstrap_fraction_positive_min":
                MIN_BOOTSTRAP_POSITIVE,

            "passed":
                promotion_pass,
        },

        "leading_gate_vs_11H": {
            "mean_far_macro_f1_positive":
                True,

            "all_far_macro_f1_positive":
                True,

            "max_near_drop":
                MAX_NEAR_DROP_VS_11H,

            "all_far_bootstrap_fraction_positive_min":
                MIN_BOOTSTRAP_POSITIVE,

            "passed":
                beats_11h,
        },

        "results_summary": {
            "mean_far_macro_f1_delta_vs_uniform":
                mean_far_vs_uniform,

            "mean_far_macro_f1_delta_vs_11H":
                mean_far_vs_11h,
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

            "11H_source_sha256":
                sha256(
                    source11h
                ),
        },

        "outputs": {
            "transfer":
                sha256(
                    comparison_path
                ),

            "bootstrap":
                sha256(
                    bootstrap_path
                ),

            "expert_diversity":
                sha256(
                    diversity_path
                ),

            "seed_training":
                sha256(
                    seed_path
                ),
        },

        "next_if_beats_11H":
            (
                "retain arithmetic temporal mixture and "
                "test controlled combination with the "
                "independently positive MULTISCALE5 "
                "longitudinal mechanism"
            ),

        "next_if_not":
            (
                "retain 11H geometric temporal ensemble "
                "and combine it directly with MULTISCALE5; "
                "stop expert aggregation search"
            ),
    }

    write_json(
        OUT
        / "11K_manifest.json",
        manifest,
    )

    print()
    print("=" * 78)
    print(
        "11K TRANSFER"
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
        "FAR VS UNIFORM"
    )
    print("=" * 78)

    print(
        far_boot_uniform.to_string(
            index=False
        )
    )

    print()
    print("=" * 78)
    print(
        "FAR VS 11H"
    )
    print("=" * 78)

    print(
        far_boot_11h.to_string(
            index=False
        )
    )

    print()
    print(
        "Mean FAR Macro-F1 vs Uniform:",
        f"{mean_far_vs_uniform:.6f}",
    )

    print(
        "Mean FAR Macro-F1 vs 11H:",
        f"{mean_far_vs_11h:.6f}",
    )

    print(
        "PROMOTION PASS:",
        promotion_pass,
    )

    print(
        "BEATS 11H:",
        beats_11h,
    )

    print(
        "Frozen inference: YES"
    )

    print(
        "Future-B accessed: NO"
    )

    for data in origin_data.values():
        for model in data[
            "models"
        ]:
            del model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
