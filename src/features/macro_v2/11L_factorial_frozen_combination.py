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

MEMORY = "MULTISCALE5"

MIN_MEAN_FAR_D_VS_A_F1 = 0.005
MAX_NEAR_D_VS_A_DROP = 0.010
MIN_FAR_BOOTSTRAP_POSITIVE = 0.90

EPS = 1e-12


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

    parent = str(
        path.parent
    )

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


def train_multiscale_ltd(
    mod,
    helpers,
    train,
    tests,
    daily_all,
    candidate_labels,
    label_to_idx,
    features,
    spec,
    origin,
    device,
    seed_rows,
):
    y_train = mod.labels_to_indices(
        train[
            "site_label"
        ],
        label_to_idx,
    )

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

    (
        train_cache,
        skipped_train,
    ) = helpers.context_cache(
        mod,
        MEMORY,
        daily_history,
        candidate_labels,
        train_dates,
        features,
    )

    if not train_cache:
        raise RuntimeError(
            f"{origin}: no MULTISCALE training contexts."
        )

    window_caches = {}

    for window in tests:
        (
            cache,
            skipped,
        ) = helpers.context_cache(
            mod,
            MEMORY,
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
                f"missing MULTISCALE contexts {skipped}"
            )

        window_caches[
            window
        ] = cache

    seed_probs = {
        window: []
        for window in tests
    }

    for seed in mod.SEEDS:
        print(
            origin,
            "MULTISCALE5 seed=",
            seed,
        )

        mod.set_seed(
            seed
        )

        mod.CONTEXT_DAYS = (
            helpers.CONTEXT_TOKENS
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

        seed_rows.append(
            {
                "origin":
                    origin,

                "memory":
                    MEMORY,

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

        for window in tests:
            probs = helpers.predict_ltd(
                mod,
                model,
                x_tests[
                    window
                ],
                test_dates[
                    window
                ],
                window_caches[
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

    for window in tests:
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


def evaluate_variant(
    helpers,
    xgb_probs,
    ltd_probs,
    y,
):
    fusion = helpers.fuse(
        xgb_probs,
        ltd_probs,
    )

    xgb_m, _, _ = helpers.evaluate(
        xgb_probs,
        y,
    )

    ltd_m, _, _ = helpers.evaluate(
        ltd_probs,
        y,
    )

    fusion_m, _, _ = helpers.evaluate(
        fusion,
        y,
    )

    return {
        "xgb":
            xgb_m,

        "ltd":
            ltd_m,

        "fusion":
            fusion_m,

        "fusion_probs":
            fusion,
    }


def main():
    print("=" * 78)
    print(
        "11L — FACTORIAL FROZEN COMBINATION"
    )
    print("=" * 78)

    print(
        "A = UNIFORM + RECENT5"
    )

    print(
        "B = TEMPORAL_SYMMETRIC3 + RECENT5"
    )

    print(
        "C = UNIFORM + MULTISCALE5"
    )

    print(
        "D = TEMPORAL_SYMMETRIC3 + MULTISCALE5"
    )

    print(
        "No new hyperparameters."
    )

    print(
        "No online adaptation."
    )

    print(
        "No test-time memory updates."
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

    mod = load_module(
        source07a,
        "ltd_07a_for_11l",
    )

    helpers = load_module(
        source11b,
        "ltd_11b_for_11l",
    )

    m11f = load_module(
        source11f,
        "ltd_11f_for_11l",
    )

    m11h = load_module(
        source11h,
        "ltd_11h_for_11l",
    )

    (
        features,
        frozen_path,
    ) = mod.load_frozen_features()

    if len(
        features
    ) != 128:
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

    summary_rows = []

    score_store = {}

    for origin, spec in origins.items():
        print()
        print("#" * 78)
        print(origin)
        print("#" * 78)

        # --------------------------------------------------
        # 11H infrastructure:
        #
        # - RECENT5 LTD
        # - UNIFORM XGB
        # - TEMPORAL_SYMMETRIC3 XGB
        #
        # No Future-B and no adaptive updates.
        # --------------------------------------------------

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

        train = data[
            "train"
        ]

        tests = data[
            "tests"
        ]

        y_tests = data[
            "y_tests"
        ]

        recent_probs = data[
            "ltd_probs"
        ]

        uniform_probs = (
            data[
                "xgb_candidates"
            ][
                "UNIFORM"
            ]
        )

        temporal_probs = (
            data[
                "xgb_candidates"
            ][
                "TEMPORAL_SYMMETRIC3"
            ]
        )

        # --------------------------------------------------
        # Independently positive 11B mechanism.
        # --------------------------------------------------

        multiscale_probs = (
            train_multiscale_ltd(
                mod,
                helpers,
                train,
                tests,
                daily_all,
                candidate_labels,
                label_to_idx,
                features,
                spec,
                origin,
                device,
                seed_rows,
            )
        )

        for window in [
            "NEAR",
            "MID",
            "FAR",
        ]:
            y = y_tests[
                window
            ]

            variants = {
                "A_UNIFORM_RECENT5":
                    (
                        uniform_probs[
                            window
                        ],
                        recent_probs[
                            window
                        ],
                    ),

                "B_TEMPORAL_RECENT5":
                    (
                        temporal_probs[
                            window
                        ],
                        recent_probs[
                            window
                        ],
                    ),

                "C_UNIFORM_MULTISCALE5":
                    (
                        uniform_probs[
                            window
                        ],
                        multiscale_probs[
                            window
                        ],
                    ),

                "D_TEMPORAL_MULTISCALE5":
                    (
                        temporal_probs[
                            window
                        ],
                        multiscale_probs[
                            window
                        ],
                    ),
            }

            for variant, (
                xgb_probs,
                ltd_probs,
            ) in variants.items():
                result = evaluate_variant(
                    helpers,
                    xgb_probs,
                    ltd_probs,
                    y,
                )

                summary_rows.append(
                    {
                        "origin":
                            origin,

                        "window":
                            window,

                        "variant":
                            variant,

                        "n":
                            len(y),

                        **{
                            f"xgb_{k}":
                                v
                            for k, v
                            in result[
                                "xgb"
                            ].items()
                        },

                        **{
                            f"ltd_{k}":
                                v
                            for k, v
                            in result[
                                "ltd"
                            ].items()
                        },

                        **{
                            f"fusion_{k}":
                                v
                            for k, v
                            in result[
                                "fusion"
                            ].items()
                        },
                    }
                )

                score_store[
                    (
                        origin,
                        window,
                        variant,
                    )
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

                    "fusion":
                        result[
                            "fusion_probs"
                        ],

                    "xgb":
                        xgb_probs,

                    "ltd":
                        ltd_probs,
                }

        for model in data[
            "models"
        ]:
            del model

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = pd.DataFrame(
        summary_rows
    )

    # ==================================================
    # Deltas and factorial interaction
    # ==================================================

    delta_rows = []
    interaction_rows = []
    bootstrap_rows = []

    variants = {
        "A":
            "A_UNIFORM_RECENT5",

        "B":
            "B_TEMPORAL_RECENT5",

        "C":
            "C_UNIFORM_MULTISCALE5",

        "D":
            "D_TEMPORAL_MULTISCALE5",
    }

    for origin in origins:
        for window in [
            "NEAR",
            "MID",
            "FAR",
        ]:
            rows = {}

            for short, full in (
                variants.items()
            ):
                rows[
                    short
                ] = (
                    summary[
                        (
                            summary[
                                "origin"
                            ]
                            ==
                            origin
                        )
                        &
                        (
                            summary[
                                "window"
                            ]
                            ==
                            window
                        )
                        &
                        (
                            summary[
                                "variant"
                            ]
                            ==
                            full
                        )
                    ]
                    .iloc[
                        0
                    ]
                )

            delta_row = {
                "origin":
                    origin,

                "window":
                    window,
            }

            for reference in [
                "A",
                "B",
                "C",
            ]:
                for metric in [
                    "accuracy",
                    "macro_f1",
                    "top5_accuracy",
                    "mrr",
                ]:
                    key = (
                        f"fusion_{metric}"
                    )

                    delta_row[
                        (
                            f"delta_D_vs_"
                            f"{reference}_"
                            f"{metric}"
                        )
                    ] = (
                        float(
                            rows[
                                "D"
                            ][
                                key
                            ]
                        )
                        -
                        float(
                            rows[
                                reference
                            ][
                                key
                            ]
                        )
                    )

            delta_rows.append(
                delta_row
            )

            interaction_row = {
                "origin":
                    origin,

                "window":
                    window,
            }

            for metric in [
                "accuracy",
                "macro_f1",
                "top5_accuracy",
                "mrr",
            ]:
                key = (
                    f"fusion_{metric}"
                )

                interaction_row[
                    f"interaction_{metric}"
                ] = (
                    float(
                        rows[
                            "D"
                        ][
                            key
                        ]
                    )
                    -
                    float(
                        rows[
                            "B"
                        ][
                            key
                        ]
                    )
                    -
                    float(
                        rows[
                            "C"
                        ][
                            key
                        ]
                    )
                    +
                    float(
                        rows[
                            "A"
                        ][
                            key
                        ]
                    )
                )

            interaction_rows.append(
                interaction_row
            )

            for reference in [
                "A",
                "B",
                "C",
            ]:
                ref_scores = score_store[
                    (
                        origin,
                        window,
                        variants[
                            reference
                        ],
                    )
                ]

                d_scores = score_store[
                    (
                        origin,
                        window,
                        variants[
                            "D"
                        ],
                    )
                ]

                boot = helpers.bootstrap_delta(
                    d_scores[
                        "y"
                    ],
                    d_scores[
                        "dates"
                    ],
                    ref_scores[
                        "fusion"
                    ],
                    d_scores[
                        "fusion"
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

                            "reference":
                                reference,

                            "candidate":
                                "D",

                            "metric":
                                metric,

                            "observed_delta":
                                delta_row[
                                    (
                                        f"delta_D_vs_"
                                        f"{reference}_"
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

    deltas = pd.DataFrame(
        delta_rows
    )

    interactions = pd.DataFrame(
        interaction_rows
    )

    bootstrap = pd.DataFrame(
        bootstrap_rows
    )

    far = deltas[
        deltas[
            "window"
        ]
        ==
        "FAR"
    ]

    near = deltas[
        deltas[
            "window"
        ]
        ==
        "NEAR"
    ]

    far_boot_a = bootstrap[
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
                "reference"
            ]
            ==
            "A"
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

    mean_far_d_vs_a = float(
        far[
            "delta_D_vs_A_macro_f1"
        ].mean()
    )

    mean_far_d_vs_b = float(
        far[
            "delta_D_vs_B_macro_f1"
        ].mean()
    )

    mean_far_d_vs_c = float(
        far[
            "delta_D_vs_C_macro_f1"
        ].mean()
    )

    mean_far_interaction = float(
        interactions[
            interactions[
                "window"
            ]
            ==
            "FAR"
        ][
            "interaction_macro_f1"
        ].mean()
    )

    promotion_pass = bool(
        mean_far_d_vs_a
        >=
        MIN_MEAN_FAR_D_VS_A_F1

        and

        (
            far[
                "delta_D_vs_A_macro_f1"
            ]
            >
            0
        ).all()

        and

        near[
            "delta_D_vs_A_macro_f1"
        ].min()
        >=
        -MAX_NEAR_D_VS_A_DROP

        and

        (
            far_boot_a[
                "fraction_delta_gt_0"
            ]
            >=
            MIN_FAR_BOOTSTRAP_POSITIVE
        ).all()

        and

        mean_far_d_vs_b
        >=
        0.0

        and

        mean_far_d_vs_c
        >=
        0.0
    )

    # ==================================================
    # Save
    # ==================================================

    summary_path = (
        OUT
        / "11L_factorial_summary.csv"
    )

    delta_path = (
        OUT
        / "11L_combination_deltas.csv"
    )

    interaction_path = (
        OUT
        / "11L_factorial_interaction.csv"
    )

    bootstrap_path = (
        OUT
        / "11L_day_block_bootstrap.csv"
    )

    seed_path = (
        OUT
        / "11L_seed_training.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    deltas.to_csv(
        delta_path,
        index=False,
    )

    interactions.to_csv(
        interaction_path,
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
            "11L_factorial_frozen_combination",

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "git_commit":
            git_commit(),

        "scientific_status":
            "FROZEN_MODEL_TEMPORAL_GENERALIZATION",

        "research_question":
            (
                "Do the two independently positive frozen "
                "mechanisms from 11H and 11B combine "
                "additively or synergistically?"
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

            "candidate_selection":
                False,

            "alpha_search":
                False,
        },

        "factorial_design": {
            "A":
                "UNIFORM XGB + RECENT5 LTD",

            "B":
                (
                    "11H TEMPORAL_SYMMETRIC3 XGB "
                    "+ RECENT5 LTD"
                ),

            "C":
                (
                    "UNIFORM XGB "
                    "+ 11B MULTISCALE5 LTD"
                ),

            "D":
                (
                    "11H TEMPORAL_SYMMETRIC3 XGB "
                    "+ 11B MULTISCALE5 LTD"
                ),

            "features":
                "BASE128",

            "alpha_ltd":
                helpers.ALPHA_LTD,

            "new_hyperparameters":
                False,
        },

        "promotion_gate": {
            "mean_far_D_vs_A_macro_f1_min":
                MIN_MEAN_FAR_D_VS_A_F1,

            "all_far_D_vs_A_positive":
                True,

            "max_near_D_vs_A_drop":
                MAX_NEAR_D_VS_A_DROP,

            "all_far_D_vs_A_bootstrap_fraction_positive_min":
                MIN_FAR_BOOTSTRAP_POSITIVE,

            "mean_far_D_vs_B_nonnegative":
                True,

            "mean_far_D_vs_C_nonnegative":
                True,

            "passed":
                promotion_pass,
        },

        "results_summary": {
            "mean_far_D_vs_A_macro_f1":
                mean_far_d_vs_a,

            "mean_far_D_vs_B_macro_f1":
                mean_far_d_vs_b,

            "mean_far_D_vs_C_macro_f1":
                mean_far_d_vs_c,

            "mean_far_factorial_interaction_macro_f1":
                mean_far_interaction,
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

            "11B_source_sha256":
                sha256(
                    source11b
                ),

            "11H_source_sha256":
                sha256(
                    source11h
                ),
        },

        "outputs": {
            "factorial_summary":
                sha256(
                    summary_path
                ),

            "combination_deltas":
                sha256(
                    delta_path
                ),

            "factorial_interaction":
                sha256(
                    interaction_path
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
                "freeze D as LTD-HYBRID-ROBUST-CANDIDATE; "
                "close Phase 11; only then perform one "
                "explicitly post-hoc exploratory Future-B "
                "evaluation"
            ),

        "next_if_fail":
            (
                "close Phase 11 without additional tuning; "
                "retain 11H as leading frozen XGB mechanism "
                "and 11B as positive mechanistic LTD result"
            ),
    }

    write_json(
        OUT
        / "11L_manifest.json",
        manifest,
    )

    print()
    print("=" * 78)
    print(
        "FACTORIAL SUMMARY"
    )
    print("=" * 78)

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 78)
    print(
        "COMBINATION DELTAS"
    )
    print("=" * 78)

    print(
        deltas.to_string(
            index=False
        )
    )

    print()
    print("=" * 78)
    print(
        "FACTORIAL INTERACTION"
    )
    print("=" * 78)

    print(
        interactions.to_string(
            index=False
        )
    )

    print()
    print("=" * 78)
    print(
        "FAR D vs A BOOTSTRAP"
    )
    print("=" * 78)

    print(
        far_boot_a.to_string(
            index=False
        )
    )

    print()
    print(
        "Mean FAR D vs A Macro-F1:",
        f"{mean_far_d_vs_a:.6f}",
    )

    print(
        "Mean FAR D vs B Macro-F1:",
        f"{mean_far_d_vs_b:.6f}",
    )

    print(
        "Mean FAR D vs C Macro-F1:",
        f"{mean_far_d_vs_c:.6f}",
    )

    print(
        "Mean FAR interaction Macro-F1:",
        f"{mean_far_interaction:.6f}",
    )

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
