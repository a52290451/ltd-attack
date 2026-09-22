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

DRO_ITERATIONS = 5
DRO_ETA = 1.0

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


def load_module(path, name):
    path = Path(path)

    parent = str(path.parent)

    if parent not in sys.path:
        sys.path.insert(0, parent)

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

    spec.loader.exec_module(mod)

    return mod


def normalize_probabilities(p):
    p = np.asarray(
        p,
        dtype=np.float64,
    )

    p = np.clip(
        p,
        EPS,
        1.0,
    )

    return (
        p
        /
        p.sum(
            axis=1,
            keepdims=True,
        )
    )


def group_balanced_weights(
    env,
    q,
):
    weights = np.zeros(
        len(env),
        dtype=np.float64,
    )

    present = np.unique(env)

    q_present = q[
        present
    ].astype(
        np.float64
    )

    q_present /= q_present.sum()

    for local_i, env_id in enumerate(
        present
    ):
        mask = env == env_id

        n = int(mask.sum())

        weights[
            mask
        ] = (
            q_present[
                local_i
            ]
            /
            n
        )

    weights /= weights.mean()

    return weights


def crossfit_losses(
    helpers,
    m11i,
    train,
    features,
    y_train,
    env,
    q,
):
    losses = np.zeros(
        N_ENVIRONMENTS,
        dtype=np.float64,
    )

    for held_env in range(
        N_ENVIRONMENTS
    ):
        train_mask = (
            env != held_env
        )

        held_mask = (
            env == held_env
        )

        train_sub = (
            train.loc[
                train_mask
            ]
            .reset_index(
                drop=True
            )
        )

        held = (
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

        sub_env = env[
            train_mask
        ]

        if (
            train_sub[
                "site_label"
            ].nunique()
            != EXPECTED_SITES
        ):
            raise RuntimeError(
                f"held env {held_env}: "
                "training lacks 65 classes"
            )

        (
            x_train,
            x_test,
        ) = m11i.fit_matrices(
            train_sub,
            {
                "HELD":
                    held,
            },
            features,
        )

        sample_weight = (
            group_balanced_weights(
                sub_env,
                q,
            )
        )

        model = xgb.XGBClassifier(
            **helpers.XGB_CONFIG
        )

        model.fit(
            x_train,
            y_sub,
            sample_weight=sample_weight,
        )

        probs = normalize_probabilities(
            model.predict_proba(
                x_test[
                    "HELD"
                ]
            )
        )

        losses[
            held_env
        ] = float(
            log_loss(
                y_held,
                probs,
                labels=np.arange(
                    EXPECTED_SITES
                ),
            )
        )

    return losses


def train_iterative_dro(
    helpers,
    m11i,
    train,
    tests,
    features,
    y_train,
):
    (
        env,
        environment_table,
    ) = m11i.environment_assignment(
        train[
            "_query_date"
        ]
    )

    q = np.full(
        N_ENVIRONMENTS,
        1.0
        /
        N_ENVIRONMENTS,
        dtype=np.float64,
    )

    iteration_rows = []

    for iteration in range(
        1,
        DRO_ITERATIONS
        +
        1,
    ):
        losses = crossfit_losses(
            helpers,
            m11i,
            train,
            features,
            y_train,
            env,
            q,
        )

        centered = (
            losses
            -
            losses.mean()
        )

        q = (
            q
            *
            np.exp(
                DRO_ETA
                *
                centered
            )
        )

        q /= q.sum()

        for env_id in range(
            N_ENVIRONMENTS
        ):
            iteration_rows.append(
                {
                    "iteration":
                        iteration,

                    "environment":
                        env_id,

                    "crossfit_log_loss":
                        float(
                            losses[
                                env_id
                            ]
                        ),

                    "group_weight":
                        float(
                            q[
                                env_id
                            ]
                        ),
                }
            )

        print(
            "iteration",
            iteration,
            "losses=",
            np.round(
                losses,
                6,
            ),
            "q=",
            np.round(
                q,
                6,
            ),
        )

    sample_weight = (
        group_balanced_weights(
            env,
            q,
        )
    )

    (
        x_train,
        x_tests,
    ) = m11i.fit_matrices(
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

    probabilities = {
        window:
            normalize_probabilities(
                model.predict_proba(
                    x
                )
            )
        for window, x
        in x_tests.items()
    }

    final_table = (
        environment_table.copy()
    )

    final_table[
        "final_group_weight"
    ] = q

    final_table[
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

    return (
        probabilities,
        pd.DataFrame(
            iteration_rows
        ),
        final_table,
    )


def main():
    print("=" * 78)
    print(
        "11J — ITERATIVE TEMPORAL GROUPDRO-STYLE XGB"
    )
    print("=" * 78)

    print(
        "Frozen inference only."
    )

    print(
        "No adaptation or memory update."
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

    source11i = (
        root
        / "src/features/macro_v2/"
        "11I_crossfit_worstenv_xgb.py"
    )

    m11h = load_module(
        source11h,
        "m11h_for_11j",
    )

    m11i = load_module(
        source11i,
        "m11i_for_11j",
    )

    mod = load_module(
        source07a,
        "ltd_07a_for_11j",
    )

    helpers = load_module(
        source11b,
        "ltd_11b_for_11j",
    )

    m11f = load_module(
        source11f,
        "ltd_11f_for_11j",
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
                unique_dates[:14],

            "windows": {
                "NEAR":
                    unique_dates[14:21],

                "MID":
                    unique_dates[28:35],

                "FAR":
                    unique_dates[42:49],
            },
        },

        "ORIGIN28": {
            "train":
                unique_dates[:28],

            "windows": {
                "NEAR":
                    unique_dates[28:35],

                "MID":
                    unique_dates[35:42],

                "FAR":
                    unique_dates[45:52],
            },
        },
    }

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    seed_rows = []
    iteration_parts = []
    final_weight_parts = []
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
            iterations,
            final_weights,
        ) = train_iterative_dro(
            helpers,
            m11i,
            data[
                "train"
            ],
            data[
                "tests"
            ],
            features,
            y_train,
        )

        iterations.insert(
            0,
            "origin",
            origin,
        )

        final_weights.insert(
            0,
            "origin",
            origin,
        )

        iteration_parts.append(
            iterations
        )

        final_weight_parts.append(
            final_weights
        )

        (
            comparison,
            scores,
        ) = m11i.evaluate(
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

    iterations = pd.concat(
        iteration_parts,
        ignore_index=True,
    )

    final_weights = pd.concat(
        final_weight_parts,
        ignore_index=True,
    )

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
                    "temporal",
                ]:
                    suffix = (
                        "uniform"
                        if reference
                        ==
                        "uniform"
                        else
                        "11H"
                    )

                    boot = helpers.bootstrap_delta(
                        score[
                            "y"
                        ],
                        score[
                            "dates"
                        ],
                        score[
                            f"{reference}_{model}"
                        ],
                        score[
                            f"robust_{model}"
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
                                    suffix,

                                "metric":
                                    metric,

                                "observed_delta":
                                    float(
                                        observed[
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
                                            values > 0
                                        )
                                    ),
                            }
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
            far_boot[
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

    leading_candidate = bool(
        promotion_pass
        and
        mean_far_vs_11h
        >=
        0.0
    )

    iteration_path = (
        OUT
        / "11J_groupdro_iterations.csv"
    )

    weights_path = (
        OUT
        / "11J_final_environment_weights.csv"
    )

    comparison_path = (
        OUT
        / "11J_transfer.csv"
    )

    bootstrap_path = (
        OUT
        / "11J_day_block_bootstrap.csv"
    )

    seed_path = (
        OUT
        / "11J_seed_training.csv"
    )

    iterations.to_csv(
        iteration_path,
        index=False,
    )

    final_weights.to_csv(
        weights_path,
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
            "11J_iterative_temporal_groupdro_xgb",

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "git_commit":
            git_commit(),

        "scientific_status":
            "FROZEN_MODEL_TEMPORAL_GENERALIZATION",

        "method": {
            "name":
                (
                    "iterative cross-fitted temporal "
                    "GroupDRO-style XGB"
                ),

            "iterations":
                DRO_ITERATIONS,

            "eta":
                DRO_ETA,

            "environments":
                N_ENVIRONMENTS,

            "probabilities_explicitly_renormalized":
                True,

            "group_update":
                (
                    "multiplicative q_g <- "
                    "q_g * exp(eta * centered_loss_g)"
                ),

            "final_model":
                "single frozen XGB",
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

        "promotion_gate": {
            "mean_far_macro_f1_gain_vs_uniform_min":
                MIN_MEAN_FAR_MACRO_F1_GAIN,

            "all_far_positive":
                True,

            "mean_far_xgb_positive":
                True,

            "max_near_drop":
                MAX_NEAR_MACRO_F1_DROP,

            "all_far_bootstrap_fraction_positive_min":
                MIN_FAR_BOOTSTRAP_POSITIVE,

            "passed":
                promotion_pass,
        },

        "leading_candidate_gate": {
            "promotion_pass_required":
                True,

            "mean_far_macro_f1_vs_11H_nonnegative":
                True,

            "mean_far_macro_f1_delta_vs_11H":
                mean_far_vs_11h,

            "passed":
                leading_candidate,
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

            "11I_source_sha256":
                sha256(
                    source11i
                ),
        },

        "outputs": {
            "iterations":
                sha256(
                    iteration_path
                ),

            "final_environment_weights":
                sha256(
                    weights_path
                ),

            "transfer":
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
                "freeze 11J if it also matches or "
                "exceeds 11H"
            ),

        "next_if_fail":
            (
                "close temporal reweighting family; "
                "retain 11H and investigate frozen "
                "temporal expert diversity directly"
            ),
    }

    write_json(
        OUT
        / "11J_manifest.json",
        manifest,
    )

    print()
    print("=" * 78)
    print(
        "GROUPDRO ITERATIONS"
    )
    print("=" * 78)

    print(
        iterations.to_string(
            index=False
        )
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
    print(
        "Mean FAR Macro-F1 delta vs 11H:",
        f"{mean_far_vs_11h:.6f}",
    )

    print(
        "PROMOTION PASS:",
        promotion_pass,
    )

    print(
        "LEADING CANDIDATE:",
        leading_candidate,
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
