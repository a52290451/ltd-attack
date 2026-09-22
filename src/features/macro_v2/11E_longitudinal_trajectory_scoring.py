from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
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

TRAJECTORY_BETAS = [
    0.10,
    0.25,
    0.40,
]

TRAJECTORY_METHODS = [
    "RAY_DISTANCE",
    "RAY_DIRECTIONAL",
]

EPS = 1e-12

MIN_MEAN_FAR_FUSION_F1_GAIN = 0.005
MAX_NEAR_FUSION_F1_DROP = 0.010
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


def row_zscore(
    values,
):
    mean = values.mean(
        axis=1,
        keepdims=True,
    )

    std = values.std(
        axis=1,
        keepdims=True,
    )

    std = np.where(
        std > 1e-8,
        std,
        1.0,
    )

    return (
        values
        -
        mean
    ) / std


def cosine_vector(
    a,
    b,
):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)

    if (
        na <= 1e-12
        or
        nb <= 1e-12
    ):
        return 0.0

    return float(
        np.dot(
            a,
            b,
        )
        /
        (
            na
            *
            nb
        )
    )


def build_site_trajectories(
    daily_history,
    candidate_labels,
    features,
):
    records = []

    early_rows = []
    middle_rows = []
    late_rows = []

    directions = []
    consistencies = []

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

        values = (
            h[
                features
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        if len(values) < 3:
            raise RuntimeError(
                f"{site}: insufficient trajectory history."
            )

        blocks = np.array_split(
            np.arange(
                len(values)
            ),
            3,
        )

        if any(
            len(x) == 0
            for x in blocks
        ):
            raise RuntimeError(
                f"{site}: empty trajectory block."
            )

        early = np.median(
            values[
                blocks[
                    0
                ]
            ],
            axis=0,
        )

        middle = np.median(
            values[
                blocks[
                    1
                ]
            ],
            axis=0,
        )

        late = np.median(
            values[
                blocks[
                    2
                ]
            ],
            axis=0,
        )

        v1 = (
            middle
            -
            early
        )

        v2 = (
            late
            -
            middle
        )

        direction = (
            late
            -
            early
        )

        consistency_raw = (
            cosine_vector(
                v1,
                v2,
            )
        )

        # Only coherent historical motion is trusted.
        consistency = max(
            0.0,
            consistency_raw,
        )

        early_rows.append(
            early
        )

        middle_rows.append(
            middle
        )

        late_rows.append(
            late
        )

        directions.append(
            direction
        )

        consistencies.append(
            consistency
        )

        records.append(
            {
                "site_label":
                    site,

                "history_days":
                    len(
                        values
                    ),

                "trajectory_norm":
                    float(
                        np.linalg.norm(
                            direction
                        )
                    ),

                "velocity_consistency_raw":
                    consistency_raw,

                "velocity_consistency_clipped":
                    consistency,
            }
        )

    return {
        "early":
            np.stack(
                early_rows
            ),

        "middle":
            np.stack(
                middle_rows
            ),

        "late":
            np.stack(
                late_rows
            ),

        "direction":
            np.stack(
                directions
            ),

        "consistency":
            np.asarray(
                consistencies,
                dtype=np.float64,
            ),

        "audit":
            pd.DataFrame(
                records
            ),
    }


def trajectory_probabilities(
    queries,
    trajectory,
):
    queries = np.asarray(
        queries,
        dtype=np.float64,
    )

    late = trajectory[
        "late"
    ]

    direction = trajectory[
        "direction"
    ]

    consistency = trajectory[
        "consistency"
    ]

    n = len(
        queries
    )

    ray_distance = np.zeros(
        (
            n,
            EXPECTED_SITES,
        ),
        dtype=np.float64,
    )

    alignment = np.zeros_like(
        ray_distance
    )

    for c in range(
        EXPECTED_SITES
    ):
        anchor = late[
            c
        ]

        velocity = direction[
            c
        ]

        velocity_norm2 = float(
            np.dot(
                velocity,
                velocity,
            )
        )

        displacement = (
            queries
            -
            anchor[
                None,
                :
            ]
        )

        if velocity_norm2 <= 1e-12:
            projected = np.repeat(
                anchor[
                    None,
                    :
                ],
                n,
                axis=0,
            )

            align = np.zeros(
                n,
                dtype=np.float64,
            )

        else:
            t = (
                displacement
                @
                velocity
            ) / velocity_norm2

            # Forward ray:
            # never extrapolate backwards.
            t = np.maximum(
                t,
                0.0,
            )

            projected = (
                anchor[
                    None,
                    :
                ]
                +
                t[
                    :,
                    None
                ]
                *
                velocity[
                    None,
                    :
                ]
            )

            displacement_norm = np.linalg.norm(
                displacement,
                axis=1,
            )

            velocity_norm = np.sqrt(
                velocity_norm2
            )

            denom = (
                displacement_norm
                *
                velocity_norm
            )

            align = np.divide(
                displacement
                @
                velocity,
                denom,
                out=np.zeros_like(
                    displacement_norm
                ),
                where=denom > 1e-12,
            )

        residual = (
            queries
            -
            projected
        )

        # Scale-independent RMS trajectory distance.
        distance = np.sqrt(
            np.mean(
                residual ** 2,
                axis=1,
            )
        )

        ray_distance[
            :,
            c
        ] = -distance

        alignment[
            :,
            c
        ] = (
            align
            *
            consistency[
                c
            ]
        )

    z_distance = row_zscore(
        ray_distance
    )

    z_alignment = row_zscore(
        alignment
    )

    directional_score = (
        0.75
        *
        z_distance
        +
        0.25
        *
        z_alignment
    )

    return {
        "RAY_DISTANCE":
            softmax(
                z_distance
            ),

        "RAY_DIRECTIONAL":
            softmax(
                directional_score
            ),
    }


def reconstruct_standardized_origin(
    mod,
    helpers,
    origin_data,
    daily_all,
    spec,
    features,
):
    (
        medians,
        scaler,
        _,
        x_tests,
    ) = helpers.prepare_matrix(
        mod,
        origin_data[
            "train"
        ],
        origin_data[
            "tests"
        ],
        features,
    )

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

    return (
        x_tests,
        daily_history,
    )


def evaluate_variant(
    helpers,
    origin_data,
    trajectory_probs,
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

        recent_ltd = origin_data[
            "ltd_probs"
        ][
            window
        ]

        trajectory = (
            trajectory_probs[
                window
            ][
                method
            ]
        )

        branch = geometric_fusion(
            recent_ltd,
            trajectory,
            beta,
        )

        fusion = helpers.fuse(
            xgb,
            branch,
        )

        (
            trajectory_metrics,
            _,
            _,
        ) = helpers.evaluate(
            trajectory,
            y,
        )

        (
            branch_metrics,
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

        candidate = (
            f"{method}_"
            f"BETA{beta:.2f}"
        )

        rows.append(
            {
                "candidate":
                    candidate,

                "method":
                    method,

                "beta":
                    beta,

                "window":
                    window,

                **{
                    f"trajectory_{k}":
                        v
                    for k, v
                    in trajectory_metrics.items()
                },

                **{
                    f"branch_{k}":
                        v
                    for k, v
                    in branch_metrics.items()
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
        }

    return (
        pd.DataFrame(
            rows
        ),
        scores,
    )


def baseline_scores(
    helpers,
    origin_data,
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

        branch = origin_data[
            "ltd_probs"
        ][
            window
        ]

        fusion = helpers.fuse(
            origin_data[
                "xgb_probs"
            ][
                window
            ],
            branch,
        )

        bm, _, _ = helpers.evaluate(
            branch,
            y,
        )

        fm, _, _ = helpers.evaluate(
            fusion,
            y,
        )

        rows.append(
            {
                "window":
                    window,

                **{
                    f"branch_{k}":
                        v
                    for k, v
                    in bm.items()
                },

                **{
                    f"fusion_{k}":
                        v
                    for k, v
                    in fm.items()
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
        "11E — EXPLICIT LONGITUDINAL TRAJECTORY SCORING"
    )
    print("=" * 78)

    print(
        "Historical-only robustness development."
    )

    print(
        "BASE128 + RECENT5 remains control."
    )

    print(
        "No Future-B values/scores are used."
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

    source11d = (
        root
        / "src/features/macro_v2/"
        "11D_longterm_drift_aware_scoring.py"
    )

    m11d = load_module(
        source11d,
        "m11d_for_11e",
    )

    mod = m11d.load_module(
        source07a,
        "ltd_07a_for_11e",
    )

    helpers = m11d.load_module(
        source11b,
        "ltd_11b_helpers_for_11e",
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
    trajectory_audit_rows = []

    origin_data = {}
    trajectory_scores = {}

    for origin, spec in (
        origins.items()
    ):
        print()
        print("#" * 78)
        print(origin)
        print("#" * 78)

        data = m11d.prepare_origin(
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
        )

        (
            x_tests,
            daily_history,
        ) = reconstruct_standardized_origin(
            mod,
            helpers,
            data,
            daily_all,
            spec,
            features,
        )

        trajectory = build_site_trajectories(
            daily_history,
            candidate_labels,
            features,
        )

        audit = trajectory[
            "audit"
        ].copy()

        audit.insert(
            0,
            "origin",
            origin,
        )

        trajectory_audit_rows.append(
            audit
        )

        trajectory_probs = {
            window:
                trajectory_probabilities(
                    x_tests[
                        window
                    ],
                    trajectory,
                )
            for window in [
                "NEAR",
                "MID",
                "FAR",
            ]
        }

        origin_data[
            origin
        ] = data

        trajectory_scores[
            origin
        ] = trajectory_probs

    # --------------------------------------------------
    # ORIGIN14 selection only
    # --------------------------------------------------

    grid = []
    score14 = {}

    for method in TRAJECTORY_METHODS:
        for beta in TRAJECTORY_BETAS:
            (
                rows,
                scores,
            ) = evaluate_variant(
                helpers,
                origin_data[
                    "ORIGIN14"
                ],
                trajectory_scores[
                    "ORIGIN14"
                ],
                method,
                beta,
            )

            grid.append(
                rows
            )

            name = (
                f"{method}_"
                f"BETA{beta:.2f}"
            )

            score14[
                name
            ] = scores

    grid14 = pd.concat(
        grid,
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

                "method":
                    group[
                        "method"
                    ].iloc[
                        0
                    ],

                "beta":
                    float(
                        group[
                            "beta"
                        ].iloc[
                            0
                        ]
                    ),

                "far_fusion_macro_f1":
                    float(
                        by.loc[
                            "FAR",
                            "fusion_macro_f1",
                        ]
                    ),

                "mean_fusion_macro_f1":
                    float(
                        group[
                            "fusion_macro_f1"
                        ].mean()
                    ),

                "far_fusion_accuracy":
                    float(
                        by.loc[
                            "FAR",
                            "fusion_accuracy",
                        ]
                    ),

                "far_branch_macro_f1":
                    float(
                        by.loc[
                            "FAR",
                            "branch_macro_f1",
                        ]
                    ),
            }
        )

    candidate_summary = pd.DataFrame(
        candidate_rows
    )

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

    selected_name = str(
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

    print(
        "Selected on ORIGIN14:",
        selected_name,
    )

    # --------------------------------------------------
    # Baseline + exact transfer
    # --------------------------------------------------

    baseline = {}
    baseline_score = {}

    for origin in [
        "ORIGIN14",
        "ORIGIN28",
    ]:
        (
            baseline[
                origin
            ],
            baseline_score[
                origin
            ],
        ) = baseline_scores(
            helpers,
            origin_data[
                origin
            ],
        )

    selected14 = (
        grid14[
            grid14[
                "candidate"
            ]
            ==
            selected_name
        ]
        .copy()
    )

    (
        selected28,
        selected28_score,
    ) = evaluate_variant(
        helpers,
        origin_data[
            "ORIGIN28"
        ],
        trajectory_scores[
            "ORIGIN28"
        ],
        selected_method,
        selected_beta,
    )

    result = {
        "ORIGIN14":
            selected14,

        "ORIGIN28":
            selected28,
    }

    selected_scores = {
        "ORIGIN14":
            score14[
                selected_name
            ],

        "ORIGIN28":
            selected28_score,
    }

    comparison_rows = []
    bootstrap_rows = []

    for origin in [
        "ORIGIN14",
        "ORIGIN28",
    ]:
        base = (
            baseline[
                origin
            ]
            .set_index(
                "window"
            )
        )

        cand = (
            result[
                origin
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
                    selected_name,

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
                    baseline_score[
                        origin
                    ][
                        window
                    ][
                        "y"
                    ],
                    baseline_score[
                        origin
                    ][
                        window
                    ][
                        "dates"
                    ],
                    baseline_score[
                        origin
                    ][
                        window
                    ][
                        model
                    ],
                    selected_scores[
                        origin
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
            MIN_BOOTSTRAP_POSITIVE
        ).all()
    )

    grid_path = (
        OUT
        / "11E_origin14_candidate_grid.csv"
    )

    summary_path = (
        OUT
        / "11E_origin14_candidate_summary.csv"
    )

    transfer_path = (
        OUT
        / "11E_selected_transfer.csv"
    )

    bootstrap_path = (
        OUT
        / "11E_day_block_bootstrap.csv"
    )

    audit_path = (
        OUT
        / "11E_trajectory_audit.csv"
    )

    seed_path = (
        OUT
        / "11E_seed_training.csv"
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

    pd.concat(
        trajectory_audit_rows,
        ignore_index=True,
    ).to_csv(
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
            "11E_longitudinal_trajectory_scoring",

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
                "Candidate-specific direction of temporal "
                "change contains longitudinal identity "
                "information that is lost by static "
                "prototypes."
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

            "selection_origin":
                "ORIGIN14",

            "origin28_selection_use":
                False,
        },

        "control":
            "BASE128 + RECENT5 LTD + canonical XGB",

        "trajectory": {
            "blocks":
                "early/middle/late thirds per site",

            "direction":
                "late median - early median",

            "consistency":
                (
                    "max(0, cosine(mid-early, "
                    "late-mid))"
                ),

            "query_geometry":
                "distance to forward candidate ray",

            "directional_variant":
                (
                    "0.75 standardized ray proximity + "
                    "0.25 consistency-weighted alignment"
                ),
        },

        "candidate_grid": {
            "methods":
                TRAJECTORY_METHODS,

            "beta_within_longitudinal_branch":
                TRAJECTORY_BETAS,

            "selection":
                "ORIGIN14 FAR Fusion Macro-F1",

            "selected":
                selected_name,

            "selected_method":
                selected_method,

            "selected_beta":
                selected_beta,

            "origin28":
                "exact one-candidate transfer",
        },

        "promotion_gate": {
            "mean_far_fusion_f1_gain_min":
                MIN_MEAN_FAR_FUSION_F1_GAIN,

            "all_far_fusion_f1_positive":
                True,

            "mean_far_branch_f1_positive":
                True,

            "max_near_fusion_f1_drop":
                MAX_NEAR_FUSION_F1_DROP,

            "all_far_bootstrap_positive_fraction_min":
                MIN_BOOTSTRAP_POSITIVE,

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

            "11D_source_sha256":
                sha256(
                    source11d
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

            "trajectory_audit":
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
                "11F controlled combination with "
                "MULTISCALE5"
            ),

        "next_if_fail":
            (
                "stop hand-designed Macro trajectory "
                "search and move to adaptive/self-updating "
                "longitudinal memory"
            ),
    }

    write_json(
        OUT
        / "11E_manifest.json",
        manifest,
    )

    print()
    print("=" * 78)
    print(
        "ORIGIN14 CANDIDATES"
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
        selected_name,
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
