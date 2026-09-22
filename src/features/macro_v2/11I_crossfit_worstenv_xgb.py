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

from sklearn.metrics import log_loss

from src.utils.paths import result_path

from src.features.macro_v2._common import (
    extract_capture_dates,
    git_commit,
    load_historical_65,
    sha256,
    write_json,
)


EXPECTED_SITES = 65
N_ENVIRONMENTS = 3

EPS = 1e-12

MIN_MEAN_FAR_MACRO_F1_GAIN = 0.005
MAX_NEAR_MACRO_F1_DROP = 0.010
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


def fit_matrices(
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


def environment_assignment(
    dates,
):
    dates = pd.to_datetime(
        dates
    )

    unique_dates = np.array(
        sorted(
            pd.unique(
                dates
            )
        )
    )

    blocks = np.array_split(
        unique_dates,
        N_ENVIRONMENTS,
    )

    mapping = {}

    rows = []

    for env_id, block in enumerate(
        blocks
    ):
        if len(block) == 0:
            raise RuntimeError(
                "Empty temporal environment."
            )

        for date in block:
            mapping[
                pd.Timestamp(
                    date
                )
            ] = env_id

        rows.append(
            {
                "environment":
                    env_id,

                "date_min":
                    str(
                        pd.Timestamp(
                            block[
                                0
                            ]
                        ).date()
                    ),

                "date_max":
                    str(
                        pd.Timestamp(
                            block[
                                -1
                            ]
                        ).date()
                    ),

                "n_dates":
                    len(
                        block
                    ),
            }
        )

    env = np.asarray(
        [
            mapping[
                pd.Timestamp(
                    date
                )
            ]
            for date in dates
        ],
        dtype=np.int64,
    )

    return (
        env,
        pd.DataFrame(
            rows
        ),
    )


def provisional_environment_losses(
    helpers,
    train,
    features,
    y_train,
    env,
):
    losses = []

    audit_rows = []

    for held_env in range(
        N_ENVIRONMENTS
    ):
        train_mask = (
            env != held_env
        )

        held_mask = (
            env == held_env
        )

        provisional_train = (
            train.loc[
                train_mask
            ]
            .reset_index(
                drop=True
            )
        )

        provisional_test = (
            train.loc[
                held_mask
            ]
            .reset_index(
                drop=True
            )
        )

        y_sub = y_train[
            train_mask
        ]

        y_held = y_train[
            held_mask
        ]

        if (
            provisional_train[
                "site_label"
            ].nunique()
            != EXPECTED_SITES
        ):
            raise RuntimeError(
                f"Cross-fit environment {held_env}: "
                "provisional training lacks 65 classes."
            )

        (
            x_sub,
            x_held_map,
        ) = fit_matrices(
            provisional_train,
            {
                "HELD":
                    provisional_test,
            },
            features,
        )

        model = xgb.XGBClassifier(
            **helpers.XGB_CONFIG
        )

        model.fit(
            x_sub,
            y_sub,
        )

        if not np.array_equal(
            model.classes_,
            np.arange(
                EXPECTED_SITES
            ),
        ):
            raise RuntimeError(
                "Unexpected provisional XGB classes."
            )

        probs = (
            model.predict_proba(
                x_held_map[
                    "HELD"
                ]
            )
            .astype(
                np.float64
            )
        )

        loss = float(
            log_loss(
                y_held,
                np.clip(
                    probs,
                    EPS,
                    1.0,
                ),
                labels=np.arange(
                    EXPECTED_SITES
                ),
            )
        )

        losses.append(
            loss
        )

        audit_rows.append(
            {
                "environment":
                    held_env,

                "n_provisional_train":
                    int(
                        train_mask.sum()
                    ),

                "n_heldout":
                    int(
                        held_mask.sum()
                    ),

                "heldout_log_loss":
                    loss,
            }
        )

    losses = np.asarray(
        losses,
        dtype=np.float64,
    )

    # One-step exponential worst-environment weighting.
    #
    # Higher cross-fitted loss => larger final group weight.
    #
    # No validation/test metric is used here.
    centered = (
        losses
        -
        losses.mean()
    )

    raw = np.exp(
        centered
    )

    group_weights = (
        raw
        /
        raw.sum()
    )

    audit = pd.DataFrame(
        audit_rows
    )

    audit[
        "group_weight"
    ] = group_weights

    return (
        group_weights,
        audit,
    )


def sample_weights_from_groups(
    env,
    group_weights,
):
    weights = np.zeros(
        len(env),
        dtype=np.float64,
    )

    for env_id in range(
        N_ENVIRONMENTS
    ):
        mask = (
            env == env_id
        )

        n = int(
            mask.sum()
        )

        if n == 0:
            raise RuntimeError(
                f"Environment {env_id} has no samples."
            )

        # GroupDRO-style group balancing:
        # total contribution of group g is proportional
        # to q_g rather than to its number of captures.
        weights[
            mask
        ] = (
            group_weights[
                env_id
            ]
            /
            n
        )

    # XGBoost sample-weight scale normalization.
    weights /= weights.mean()

    return weights


def train_worstenv_xgb(
    helpers,
    train,
    tests,
    features,
    y_train,
):
    (
        env,
        environment_table,
    ) = environment_assignment(
        train[
            "_query_date"
        ]
    )

    (
        group_weights,
        loss_audit,
    ) = provisional_environment_losses(
        helpers,
        train,
        features,
        y_train,
        env,
    )

    environment_table = (
        environment_table
        .merge(
            loss_audit,
            on="environment",
            how="left",
            validate="one_to_one",
        )
    )

    sample_weight = (
        sample_weights_from_groups(
            env,
            group_weights,
        )
    )

    environment_table[
        "mean_final_sample_weight"
    ] = [
        float(
            sample_weight[
                env == env_id
            ].mean()
        )
        for env_id in range(
            N_ENVIRONMENTS
        )
    ]

    (
        x_train,
        x_tests,
    ) = fit_matrices(
        train,
        tests,
        features,
    )

    model = xgb.XGBClassifier(
        **helpers.XGB_CONFIG
    )

    model.fit(
        x_train,
        y_train,
        sample_weight=sample_weight,
    )

    if not np.array_equal(
        model.classes_,
        np.arange(
            EXPECTED_SITES
        ),
    ):
        raise RuntimeError(
            "Unexpected final XGB classes."
        )

    probabilities = {
        window:
            model.predict_proba(
                x_test
            )
            .astype(
                np.float64
            )
        for window, x_test
        in x_tests.items()
    }

    return (
        probabilities,
        environment_table,
    )


def evaluate(
    helpers,
    data,
    robust_probs,
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

        ltd = data[
            "ltd_probs"
        ][
            window
        ]

        uniform_xgb = (
            data[
                "xgb_candidates"
            ][
                "UNIFORM"
            ][
                window
            ]
        )

        temporal_xgb = (
            data[
                "xgb_candidates"
            ][
                "TEMPORAL_SYMMETRIC3"
            ][
                window
            ]
        )

        robust_xgb = (
            robust_probs[
                window
            ]
        )

        uniform_macro = helpers.fuse(
            uniform_xgb,
            ltd,
        )

        temporal_macro = helpers.fuse(
            temporal_xgb,
            ltd,
        )

        robust_macro = helpers.fuse(
            robust_xgb,
            ltd,
        )

        metrics = {}

        for name, probs in [
            (
                "uniform_xgb",
                uniform_xgb,
            ),
            (
                "temporal_xgb",
                temporal_xgb,
            ),
            (
                "robust_xgb",
                robust_xgb,
            ),
            (
                "uniform_macro",
                uniform_macro,
            ),
            (
                "temporal_macro",
                temporal_macro,
            ),
            (
                "robust_macro",
                robust_macro,
            ),
        ]:
            m, _, _ = helpers.evaluate(
                probs,
                y,
            )

            metrics[
                name
            ] = m

        row = {
            "window":
                window,

            "n":
                len(y),

            "robust_xgb_accuracy":
                metrics[
                    "robust_xgb"
                ][
                    "accuracy"
                ],

            "robust_xgb_macro_f1":
                metrics[
                    "robust_xgb"
                ][
                    "macro_f1"
                ],

            "robust_xgb_top5":
                metrics[
                    "robust_xgb"
                ][
                    "top5_accuracy"
                ],

            "robust_xgb_mrr":
                metrics[
                    "robust_xgb"
                ][
                    "mrr"
                ],

            "robust_macro_accuracy":
                metrics[
                    "robust_macro"
                ][
                    "accuracy"
                ],

            "robust_macro_macro_f1":
                metrics[
                    "robust_macro"
                ][
                    "macro_f1"
                ],

            "robust_macro_top5":
                metrics[
                    "robust_macro"
                ][
                    "top5_accuracy"
                ],

            "robust_macro_mrr":
                metrics[
                    "robust_macro"
                ][
                    "mrr"
                ],
        }

        for model in [
            "xgb",
            "macro",
        ]:
            robust_key = (
                f"robust_{model}"
            )

            uniform_key = (
                f"uniform_{model}"
            )

            temporal_key = (
                f"temporal_{model}"
            )

            for metric in [
                "accuracy",
                "macro_f1",
                "top5_accuracy",
                "mrr",
            ]:
                row[
                    (
                        f"delta_{model}_{metric}"
                        "_vs_uniform"
                    )
                ] = (
                    metrics[
                        robust_key
                    ][
                        metric
                    ]
                    -
                    metrics[
                        uniform_key
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
                    metrics[
                        robust_key
                    ][
                        metric
                    ]
                    -
                    metrics[
                        temporal_key
                    ][
                        metric
                    ]
                )

        rows.append(
            row
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

            "uniform_xgb":
                uniform_xgb,

            "temporal_xgb":
                temporal_xgb,

            "robust_xgb":
                robust_xgb,

            "uniform_macro":
                uniform_macro,

            "temporal_macro":
                temporal_macro,

            "robust_macro":
                robust_macro,
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
        "11I — CROSS-FITTED TEMPORAL WORST-ENVIRONMENT XGB"
    )
    print("=" * 78)

    print(
        "Frozen inference only."
    )

    print(
        "No test-time adaptation."
    )

    print(
        "No memory update."
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
        "m11h_for_11i",
    )

    mod = load_module(
        source07a,
        "ltd_07a_for_11i",
    )

    helpers = load_module(
        source11b,
        "ltd_11b_helpers_for_11i",
    )

    m11f = load_module(
        source11f,
        "ltd_11f_for_11i",
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
    environment_rows = []
    comparison_parts = []

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

        y_train = mod.labels_to_indices(
            data[
                "train"
            ][
                "site_label"
            ],
            label_to_idx,
        )

        (
            robust_probs,
            environment_table,
        ) = train_worstenv_xgb(
            helpers,
            data[
                "train"
            ],
            data[
                "tests"
            ],
            features,
            y_train,
        )

        environment_table.insert(
            0,
            "origin",
            origin,
        )

        environment_rows.append(
            environment_table
        )

        (
            comparison,
            scores,
        ) = evaluate(
            helpers,
            data,
            robust_probs,
        )

        comparison.insert(
            0,
            "origin",
            origin,
        )

        comparison_parts.append(
            comparison
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

    environments = pd.concat(
        environment_rows,
        ignore_index=True,
    )

    # ==================================================
    # Bootstrap
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

            for model in [
                "xgb",
                "macro",
            ]:
                for reference_name in [
                    "uniform",
                    "temporal",
                ]:
                    reference = score[
                        (
                            f"{reference_name}_"
                            f"{model}"
                        )
                    ]

                    candidate = score[
                        (
                            f"robust_"
                            f"{model}"
                        )
                    ]

                    boot = helpers.bootstrap_delta(
                        score[
                            "y"
                        ],
                        score[
                            "dates"
                        ],
                        reference,
                        candidate,
                    )

                    observed_row = (
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

                    suffix = (
                        "uniform"
                        if reference_name
                        ==
                        "uniform"
                        else
                        "11H"
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
                                    suffix,

                                "metric":
                                    metric,

                                "observed_delta":
                                    float(
                                        observed_row[
                                            (
                                                f"delta_"
                                                f"{model}_"
                                                f"{metric}_"
                                                f"vs_{suffix}"
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

    promotion_pass = bool(
        far[
            "delta_macro_macro_f1_vs_uniform"
        ].mean()
        >=
        MIN_MEAN_FAR_MACRO_F1_GAIN

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
        -MAX_NEAR_MACRO_F1_DROP

        and

        (
            far_boot_uniform[
                "fraction_delta_gt_0"
            ]
            >=
            MIN_FAR_BOOTSTRAP_POSITIVE
        ).all()
    )

    mean_far_vs_11h = float(
        far[
            "delta_macro_macro_f1_vs_11H"
        ].mean()
    )

    # ==================================================
    # Save
    # ==================================================

    env_path = (
        OUT
        / "11I_environment_weights.csv"
    )

    comparison_path = (
        OUT
        / "11I_worstenv_transfer.csv"
    )

    bootstrap_path = (
        OUT
        / "11I_day_block_bootstrap.csv"
    )

    seed_path = (
        OUT
        / "11I_seed_training.csv"
    )

    environments.to_csv(
        env_path,
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
            "11I_crossfit_worstenv_xgb",

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
                "Reduce longitudinal degradation using "
                "training-time worst-environment weighting "
                "while keeping inference fully frozen."
            ),

        "method": {
            "name":
                "cross-fitted temporal worst-environment weighting",

            "groupdro_claim":
                (
                    "GroupDRO-inspired one-step reweighting; "
                    "not claimed as full iterative GroupDRO"
                ),

            "temporal_environments":
                N_ENVIRONMENTS,

            "environment_definition":
                "contiguous chronological blocks",

            "difficulty_estimation":
                (
                    "leave-one-environment-out cross-fitted "
                    "multiclass log loss"
                ),

            "group_weight":
                (
                    "softmax of centered cross-fitted "
                    "environment losses"
                ),

            "final_model":
                (
                    "single XGB trained on all Historical "
                    "training captures with group-balanced "
                    "sample weights"
                ),
        },

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

        "comparators": [
            "UNIFORM canonical XGB",
            "11H TEMPORAL_SYMMETRIC3",
        ],

        "promotion_gate": {
            "mean_far_macro_f1_gain_vs_uniform_min":
                MIN_MEAN_FAR_MACRO_F1_GAIN,

            "all_far_macro_f1_vs_uniform_positive":
                True,

            "mean_far_xgb_f1_vs_uniform_positive":
                True,

            "max_near_macro_f1_drop_vs_uniform":
                MAX_NEAR_MACRO_F1_DROP,

            "all_far_bootstrap_positive_fraction_min":
                MIN_FAR_BOOTSTRAP_POSITIVE,

            "passed":
                promotion_pass,
        },

        "comparison_to_11H": {
            "mean_far_macro_f1_delta":
                mean_far_vs_11h,

            "positive_means_11I_better":
                True,
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
            "environment_weights":
                sha256(
                    env_path
                ),

            "comparison":
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
                "11J freeze robust XGB candidate and "
                "evaluate controlled combination with "
                "the strongest independently supported "
                "longitudinal mechanism"
            ),

        "next_if_fail":
            (
                "retain 11H as leading frozen candidate "
                "and test full training-time temporal "
                "distributionally robust optimization"
            ),
    }

    write_json(
        OUT
        / "11I_manifest.json",
        manifest,
    )

    print()
    print("=" * 78)
    print(
        "TEMPORAL ENVIRONMENT DIFFICULTY"
    )
    print("=" * 78)

    print(
        environments.to_string(
            index=False
        )
    )

    print()
    print("=" * 78)
    print(
        "11I TRANSFER"
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
        "FAR MACRO-F1 BOOTSTRAP VS UNIFORM"
    )
    print("=" * 78)

    print(
        far_boot_uniform.to_string(
            index=False
        )
    )

    print()
    print(
        "Mean FAR Macro-F1 delta vs 11H:",
        f"{mean_far_vs_11h:.6f}",
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
