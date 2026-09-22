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

MIN_PSEUDO_PER_SITE_DAY = 3

POLICIES = [
    {
        "name": "CONSENSUS_ONLY",
        "min_confidence": None,
        "min_margin": None,
    },
    {
        "name": "CONSENSUS_CONF60",
        "min_confidence": 0.60,
        "min_margin": None,
    },
    {
        "name": "CONSENSUS_MARGIN20",
        "min_confidence": None,
        "min_margin": 0.20,
    },
]

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


def policy_mask(
    policy,
    xgb_probs,
    ltd_probs,
    fusion_probs,
):
    xgb_pred = np.argmax(
        xgb_probs,
        axis=1,
    )

    ltd_pred = np.argmax(
        ltd_probs,
        axis=1,
    )

    agreement = (
        xgb_pred
        ==
        ltd_pred
    )

    order = np.argsort(
        -fusion_probs,
        axis=1,
    )

    top1 = fusion_probs[
        np.arange(
            len(fusion_probs)
        ),
        order[
            :,
            0
        ],
    ]

    top2 = fusion_probs[
        np.arange(
            len(fusion_probs)
        ),
        order[
            :,
            1
        ],
    ]

    margin = (
        top1
        -
        top2
    )

    mask = agreement.copy()

    if (
        policy[
            "min_confidence"
        ]
        is not None
    ):
        mask &= (
            top1
            >=
            policy[
                "min_confidence"
            ]
        )

    if (
        policy[
            "min_margin"
        ]
        is not None
    ):
        mask &= (
            margin
            >=
            policy[
                "min_margin"
            ]
        )

    # If XGB and LTD agree, that consensus class
    # is used as the pseudo-label.
    pseudo_idx = xgb_pred

    return (
        mask,
        pseudo_idx,
        top1,
        margin,
    )


def self_update_window(
    mod,
    helpers,
    m11f,
    models,
    test,
    x_test,
    y_test,
    dates,
    xgb_probs,
    initial_history,
    candidate_labels,
    features,
    medians,
    scaler,
    device,
    policy,
):
    history = (
        initial_history
        .copy()
        .reset_index(
            drop=True
        )
    )

    ltd_output = np.full(
        (
            len(test),
            EXPECTED_SITES,
        ),
        np.nan,
        dtype=np.float64,
    )

    fusion_output = np.full_like(
        ltd_output,
        np.nan,
    )

    audit_rows = []

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
        date = pd.Timestamp(
            date
        )

        idx = np.flatnonzero(
            pd.to_datetime(
                dates
            )
            ==
            date
        )

        if len(idx) == 0:
            raise RuntimeError(
                f"No observations for {date}"
            )

        ltd = m11f.ensemble_predict(
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

        fusion = helpers.fuse(
            xgb_probs[
                idx
            ],
            ltd,
        )

        ltd_output[
            idx
        ] = ltd

        fusion_output[
            idx
        ] = fusion

        (
            accepted,
            pseudo_idx,
            confidence,
            margin,
        ) = policy_mask(
            policy,
            xgb_probs[
                idx
            ],
            ltd,
            fusion,
        )

        pseudo_labels = (
            candidate_labels[
                pseudo_idx
            ]
        )

        accepted_initial_n = int(
            accepted.sum()
        )

        # --------------------------------------------------
        # Minimum support per predicted site/day.
        # --------------------------------------------------

        accepted_labels = (
            pseudo_labels[
                accepted
            ]
        )

        if len(
            accepted_labels
        ):
            counts = pd.Series(
                accepted_labels
            ).value_counts()

            supported_sites = set(
                counts[
                    counts
                    >=
                    MIN_PSEUDO_PER_SITE_DAY
                ]
                .index
                .astype(str)
                .tolist()
            )

            accepted &= np.array(
                [
                    str(label)
                    in
                    supported_sites
                    for label in
                    pseudo_labels
                ],
                dtype=bool,
            )

        else:
            supported_sites = set()

        accepted_n = int(
            accepted.sum()
        )

        # Ground truth is used ONLY for diagnostics.
        # It does not affect mask, pseudo-labels, or update.
        pseudo_precision = (
            float(
                np.mean(
                    pseudo_idx[
                        accepted
                    ]
                    ==
                    y_test[
                        idx
                    ][
                        accepted
                    ]
                )
            )
            if accepted_n
            else np.nan
        )

        true_accepted_n = (
            int(
                np.sum(
                    pseudo_idx[
                        accepted
                    ]
                    ==
                    y_test[
                        idx
                    ][
                        accepted
                    ]
                )
            )
            if accepted_n
            else 0
        )

        audit_rows.append(
            {
                "date":
                    str(
                        date.date()
                    ),

                "day_in_window":
                    day_index,

                "policy":
                    policy[
                        "name"
                    ],

                "n_queries":
                    len(idx),

                "accepted_before_site_support":
                    accepted_initial_n,

                "accepted_final":
                    accepted_n,

                "accepted_fraction":
                    (
                        accepted_n
                        /
                        len(idx)
                    ),

                "sites_updated":
                    len(
                        supported_sites
                    ),

                "mean_confidence_accepted":
                    (
                        float(
                            confidence[
                                accepted
                            ].mean()
                        )
                        if accepted_n
                        else np.nan
                    ),

                "mean_margin_accepted":
                    (
                        float(
                            margin[
                                accepted
                            ].mean()
                        )
                        if accepted_n
                        else np.nan
                    ),

                "pseudo_label_precision_DIAGNOSTIC_ONLY":
                    pseudo_precision,

                "correct_pseudo_labels_DIAGNOSTIC_ONLY":
                    true_accepted_n,

                "history_rows_before_update":
                    len(history),
            }
        )

        # ==================================================
        # DEPLOYABLE PSEUDO UPDATE
        #
        # Only predictions and model confidence are used.
        # y_test is NOT used below.
        # ==================================================

        if accepted_n:
            accepted_global_idx = (
                idx[
                    accepted
                ]
            )

            pseudo_frame = (
                test.iloc[
                    accepted_global_idx
                ]
                .copy()
                .reset_index(
                    drop=True
                )
            )

            pseudo_frame[
                "site_label"
            ] = (
                pseudo_labels[
                    accepted
                ]
                .astype(str)
            )

            pseudo_daily_raw = (
                helpers.build_daily(
                    pseudo_frame,
                    features,
                )
            )

            pseudo_daily_std = (
                mod.standardized_daily_frame(
                    pseudo_daily_raw,
                    features,
                    medians,
                    scaler,
                )
            )

            if not (
                pseudo_daily_std[
                    "date"
                ]
                ==
                date
            ).all():
                raise RuntimeError(
                    "Unexpected pseudo-update date."
                )

            history = pd.concat(
                [
                    history,
                    pseudo_daily_std,
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

    if (
        np.isnan(
            ltd_output
        ).any()
        or
        np.isnan(
            fusion_output
        ).any()
    ):
        raise RuntimeError(
            "Incomplete pseudo-update predictions."
        )

    return (
        ltd_output,
        fusion_output,
        pd.DataFrame(
            audit_rows
        ),
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

    xgb_probs = helpers.train_xgb(
        train,
        tests,
        features,
        y_train,
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

    baseline = {}

    for window in [
        "NEAR",
        "MID",
        "FAR",
    ]:
        ltd = m11f.ensemble_predict(
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

        fusion = helpers.fuse(
            xgb_probs[
                window
            ],
            ltd,
        )

        baseline[
            window
        ] = {
            "ltd":
                ltd,

            "fusion":
                fusion,
        }

    return {
        "train":
            train,

        "tests":
            tests,

        "y_tests":
            y_tests,

        "medians":
            medians,

        "scaler":
            scaler,

        "x_tests":
            x_tests,

        "test_dates":
            test_dates,

        "daily_history":
            daily_history,

        "xgb_probs":
            xgb_probs,

        "models":
            models,

        "baseline":
            baseline,
    }


def evaluate_policy(
    mod,
    helpers,
    m11f,
    origin_data,
    candidate_labels,
    features,
    device,
    policy,
):
    summary_rows = []
    audit_parts = []
    scores = {}

    for window in [
        "NEAR",
        "MID",
        "FAR",
    ]:
        (
            ltd,
            fusion,
            audit,
        ) = self_update_window(
            mod,
            helpers,
            m11f,
            origin_data[
                "models"
            ],
            origin_data[
                "tests"
            ][
                window
            ],
            origin_data[
                "x_tests"
            ][
                window
            ],
            origin_data[
                "y_tests"
            ][
                window
            ],
            origin_data[
                "test_dates"
            ][
                window
            ],
            origin_data[
                "xgb_probs"
            ][
                window
            ],
            origin_data[
                "daily_history"
            ],
            candidate_labels,
            features,
            origin_data[
                "medians"
            ],
            origin_data[
                "scaler"
            ],
            device,
            policy,
        )

        y = origin_data[
            "y_tests"
        ][
            window
        ]

        (
            ltd_metrics,
            _,
            _,
        ) = helpers.evaluate(
            ltd,
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

        base_ltd = (
            origin_data[
                "baseline"
            ][
                window
            ][
                "ltd"
            ]
        )

        base_fusion = (
            origin_data[
                "baseline"
            ][
                window
            ][
                "fusion"
            ]
        )

        (
            base_ltd_metrics,
            _,
            _,
        ) = helpers.evaluate(
            base_ltd,
            y,
        )

        (
            base_fusion_metrics,
            _,
            _,
        ) = helpers.evaluate(
            base_fusion,
            y,
        )

        row = {
            "policy":
                policy[
                    "name"
                ],

            "window":
                window,

            "n":
                len(y),

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

        for model_name, candidate_m, base_m in [
            (
                "ltd",
                ltd_metrics,
                base_ltd_metrics,
            ),
            (
                "fusion",
                fusion_metrics,
                base_fusion_metrics,
            ),
        ]:
            for metric in [
                "accuracy",
                "macro_f1",
                "top5_accuracy",
                "mrr",
            ]:
                row[
                    (
                        f"delta_"
                        f"{model_name}_"
                        f"{metric}"
                    )
                ] = (
                    candidate_m[
                        metric
                    ]
                    -
                    base_m[
                        metric
                    ]
                )

        summary_rows.append(
            row
        )

        audit.insert(
            0,
            "window",
            window,
        )

        audit_parts.append(
            audit
        )

        scores[
            window
        ] = {
            "ltd":
                ltd,

            "fusion":
                fusion,

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
        }

    return (
        pd.DataFrame(
            summary_rows
        ),
        pd.concat(
            audit_parts,
            ignore_index=True,
        ),
        scores,
    )


def main():
    print("=" * 78)
    print(
        "11G — PSEUDO-LABEL SELF-UPDATING MEMORY"
    )
    print("=" * 78)

    print(
        "Historical-only deployable adaptation development."
    )

    print(
        "No Future-B values/scores are used."
    )

    print(
        "Ground truth is diagnostic only and never "
        "controls a memory update."
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
        "ltd_07a_for_11g",
    )

    helpers = load_module(
        source11b,
        "ltd_11b_helpers_for_11g",
    )

    m11f = load_module(
        source11f,
        "m11f_for_11g",
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
    # ORIGIN14 — policy selection
    # ==================================================

    print()
    print("#" * 78)
    print(
        "ORIGIN14 — POLICY SELECTION"
    )
    print("#" * 78)

    origin14 = prepare_origin(
        mod,
        helpers,
        m11f,
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

    grid_parts = []
    audit14_parts = []
    score14 = {}

    for policy in POLICIES:
        print(
            "Testing",
            policy[
                "name"
            ],
        )

        (
            summary,
            audit,
            scores,
        ) = evaluate_policy(
            mod,
            helpers,
            m11f,
            origin14,
            candidate_labels,
            features,
            device,
            policy,
        )

        grid_parts.append(
            summary
        )

        audit14_parts.append(
            audit
        )

        score14[
            policy[
                "name"
            ]
        ] = scores

    grid14 = pd.concat(
        grid_parts,
        ignore_index=True,
    )

    audit14 = pd.concat(
        audit14_parts,
        ignore_index=True,
    )

    policy_summary_rows = []

    for policy, group in (
        grid14.groupby(
            "policy"
        )
    ):
        by = group.set_index(
            "window"
        )

        policy_summary_rows.append(
            {
                "policy":
                    policy,

                "far_fusion_macro_f1":
                    float(
                        by.loc[
                            "FAR",
                            "fusion_macro_f1",
                        ]
                    ),

                "far_delta_fusion_macro_f1":
                    float(
                        by.loc[
                            "FAR",
                            "delta_fusion_macro_f1",
                        ]
                    ),

                "mean_fusion_macro_f1":
                    float(
                        group[
                            "fusion_macro_f1"
                        ].mean()
                    ),

                "near_delta_fusion_macro_f1":
                    float(
                        by.loc[
                            "NEAR",
                            "delta_fusion_macro_f1",
                        ]
                    ),

                "far_delta_ltd_macro_f1":
                    float(
                        by.loc[
                            "FAR",
                            "delta_ltd_macro_f1",
                        ]
                    ),
            }
        )

    policy_summary = pd.DataFrame(
        policy_summary_rows
    )

    selected_row = (
        policy_summary
        .sort_values(
            [
                "far_fusion_macro_f1",
                "mean_fusion_macro_f1",
                "policy",
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

    selected_name = str(
        selected_row[
            "policy"
        ]
    )

    selected_policy = next(
        p
        for p in POLICIES
        if p[
            "name"
        ]
        ==
        selected_name
    )

    print(
        "Selected on ORIGIN14:",
        selected_name,
    )

    # ==================================================
    # ORIGIN28 — exact transfer
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
        m11f,
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
        transfer28,
        audit28,
        score28,
    ) = evaluate_policy(
        mod,
        helpers,
        m11f,
        origin28,
        candidate_labels,
        features,
        device,
        selected_policy,
    )

    selected14 = (
        grid14[
            grid14[
                "policy"
            ]
            ==
            selected_name
        ]
        .copy()
    )

    comparison = pd.concat(
        [
            selected14.assign(
                origin="ORIGIN14"
            ),
            transfer28.assign(
                origin="ORIGIN28"
            ),
        ],
        ignore_index=True,
    )

    audit14_selected = (
        audit14[
            audit14[
                "policy"
            ]
            ==
            selected_name
        ]
        .copy()
    )

    audit14_selected.insert(
        0,
        "origin",
        "ORIGIN14",
    )

    audit28.insert(
        0,
        "origin",
        "ORIGIN28",
    )

    update_audit = pd.concat(
        [
            audit14_selected,
            audit28,
        ],
        ignore_index=True,
    )

    # ==================================================
    # Bootstrap selected policy against frozen baseline
    # ==================================================

    bootstrap_rows = []

    for origin, data, scores in [
        (
            "ORIGIN14",
            origin14,
            score14[
                selected_name
            ],
        ),
        (
            "ORIGIN28",
            origin28,
            score28,
        ),
    ]:
        for window in [
            "NEAR",
            "MID",
            "FAR",
        ]:
            for model in [
                "ltd",
                "fusion",
            ]:
                boot = helpers.bootstrap_delta(
                    scores[
                        window
                    ][
                        "y"
                    ],
                    scores[
                        window
                    ][
                        "dates"
                    ],
                    data[
                        "baseline"
                    ][
                        window
                    ][
                        model
                    ],
                    scores[
                        window
                    ][
                        model
                    ],
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
                                float(
                                    observed_row[
                                        (
                                            f"delta_"
                                            f"{model}_"
                                            f"{metric}"
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
            "delta_ltd_macro_f1"
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
            MIN_FAR_BOOTSTRAP_POSITIVE
        ).all()
    )

    # ==================================================
    # SAVE
    # ==================================================

    grid_path = (
        OUT
        / "11G_origin14_policy_grid.csv"
    )

    policy_summary_path = (
        OUT
        / "11G_origin14_policy_summary.csv"
    )

    transfer_path = (
        OUT
        / "11G_selected_transfer.csv"
    )

    bootstrap_path = (
        OUT
        / "11G_day_block_bootstrap.csv"
    )

    audit_path = (
        OUT
        / "11G_update_audit.csv"
    )

    seed_path = (
        OUT
        / "11G_seed_training.csv"
    )

    grid14.to_csv(
        grid_path,
        index=False,
    )

    policy_summary.to_csv(
        policy_summary_path,
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

    update_audit.to_csv(
        audit_path,
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
            "11G_pseudolabel_self_updating_memory",

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
                "High-confidence agreement between static "
                "and longitudinal branches can provide "
                "sufficiently precise pseudo-labels to "
                "refresh RECENT5 memory causally."
            ),

        "data_policy": {
            "historical_only":
                True,

            "future_b_values_used":
                False,

            "future_b_scores_used":
                False,

            "true_labels_used_for_memory_update":
                False,

            "true_labels_used_for_policy_gate":
                (
                    "ORIGIN14 evaluation metrics only; "
                    "never individual update decisions"
                ),

            "pseudo_precision":
                "diagnostic only",

            "window_reset":
                True,

            "origin14_policy_selection":
                True,

            "origin28_policy_selection":
                False,
        },

        "update": {
            "timing":
                (
                    "score full current day first; "
                    "pseudo-update only afterward"
                ),

            "memory":
                "RECENT5",

            "pseudo_label":
                "XGB/LTD consensus class",

            "minimum_pseudo_captures_per_site_day":
                MIN_PSEUDO_PER_SITE_DAY,

            "policies":
                POLICIES,

            "selected_policy":
                selected_name,
        },

        "promotion_gate": {
            "mean_far_fusion_macro_f1_gain_min":
                MIN_MEAN_FAR_FUSION_F1_GAIN,

            "all_far_fusion_macro_f1_positive":
                True,

            "mean_far_ltd_macro_f1_positive":
                True,

            "max_near_fusion_macro_f1_drop":
                MAX_NEAR_FUSION_F1_DROP,

            "all_far_bootstrap_positive_fraction_min":
                MIN_FAR_BOOTSTRAP_POSITIVE,

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

            "11F_source": {
                "path":
                    str(
                        source11f
                    ),

                "sha256":
                    sha256(
                        source11f
                    ),
            },
        },

        "outputs": {
            "grid":
                sha256(
                    grid_path
                ),

            "policy_summary":
                sha256(
                    policy_summary_path
                ),

            "transfer":
                sha256(
                    transfer_path
                ),

            "bootstrap":
                sha256(
                    bootstrap_path
                ),

            "update_audit":
                sha256(
                    audit_path
                ),

            "seed_training":
                sha256(
                    seed_path
                ),
        },

        "next_if_pass":
            (
                "11H continuous deployable adaptation "
                "and controlled combination with "
                "MULTISCALE memory"
            ),

        "next_if_fail":
            (
                "analyze pseudo-label contamination and "
                "test safer soft/consensus memory updates "
                "without Future-B"
            ),
    }

    write_json(
        OUT
        / "11G_manifest.json",
        manifest,
    )

    print()
    print("=" * 78)
    print(
        "ORIGIN14 POLICY SUMMARY"
    )
    print("=" * 78)

    print(
        policy_summary
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
        selected_name,
    )

    print()
    print("=" * 78)
    print(
        "SELECTED POLICY TRANSFER"
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
        "UPDATE DIAGNOSTIC"
    )
    print("=" * 78)

    print(
        update_audit.to_string(
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
        "Future-B was not accessed."
    )


if __name__ == "__main__":
    main()
