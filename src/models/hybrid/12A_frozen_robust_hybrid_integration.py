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
    sha256,
    write_json,
)


EXPECTED_SITES = 65

MICRO_EPOCHS = 30

HYBRID_MICRO_WEIGHT = 0.45
HYBRID_MACRO_WEIGHT = 0.55

EPS = 1e-12

MIN_MEAN_FAR_HD_VS_HA_F1 = 0.0025
MAX_NEAR_HD_VS_HA_DROP = 0.010
MIN_FAR_BOOTSTRAP_POSITIVE = 0.90


OUT = Path(
    result_path(
        "hybrid",
        "LTD-ROBUSTNESS-PHASE12",
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


def align_micro_to_macro(
    micro_df,
    micro_probs,
    micro_y,
    test_idx,
    macro_test,
    macro_y,
):
    test_uids = (
        micro_df
        .iloc[
            test_idx
        ][
            "pcap_uid"
        ]
        .astype(str)
        .to_numpy()
    )

    if len(
        np.unique(
            test_uids
        )
    ) != len(
        test_uids
    ):
        raise RuntimeError(
            "Duplicate Micro test pcap_uid."
        )

    lookup = {
        uid: i
        for i, uid in
        enumerate(
            test_uids
        )
    }

    macro_uids = (
        macro_test[
            "pcap_uid"
        ]
        .astype(str)
        .to_numpy()
    )

    missing = [
        uid
        for uid in macro_uids
        if uid not in lookup
    ]

    if missing:
        raise RuntimeError(
            f"Missing Micro predictions for "
            f"{len(missing)} Macro captures."
        )

    order = np.asarray(
        [
            lookup[
                uid
            ]
            for uid in macro_uids
        ],
        dtype=np.int64,
    )

    aligned_probs = (
        micro_probs[
            order
        ]
    )

    aligned_y = (
        micro_y[
            order
        ]
    )

    if not np.array_equal(
        aligned_y,
        macro_y,
    ):
        raise RuntimeError(
            "Micro/Macro label alignment mismatch."
        )

    return aligned_probs


def train_micro_origin(
    mod_micro,
    micro_df,
    x_dir,
    x_size,
    y_micro,
    spec,
    macro_tests,
    macro_y_tests,
    device,
    origin,
):
    train_mask = (
        micro_df[
            "_query_date"
        ].isin(
            spec[
                "train"
            ]
        )
        .to_numpy()
    )

    train_idx = np.flatnonzero(
        train_mask
    )

    if len(train_idx) == 0:
        raise RuntimeError(
            f"{origin}: empty Micro train."
        )

    if (
        micro_df
        .iloc[
            train_idx
        ][
            "site_label"
        ]
        .nunique()
        !=
        EXPECTED_SITES
    ):
        raise RuntimeError(
            f"{origin}: Micro train lacks 65 sites."
        )

    print()
    print(
        origin,
        "MICRO-FINAL training captures:",
        len(train_idx),
    )

    t0 = time.perf_counter()

    (
        model,
        min_size,
        max_size,
        best_epoch,
        curve,
    ) = mod_micro.train(
        x_dir,
        x_size,
        y_micro,
        train_idx,
        device,
        MICRO_EPOCHS,
        val_idx=None,
    )

    seconds = (
        time.perf_counter()
        -
        t0
    )

    if best_epoch != MICRO_EPOCHS:
        raise RuntimeError(
            "Unexpected Micro epoch selection."
        )

    curve = curve.copy()

    curve.insert(
        0,
        "origin",
        origin,
    )

    window_probs = {}

    for window, dates in (
        spec[
            "windows"
        ].items()
    ):
        test_idx = np.flatnonzero(
            micro_df[
                "_query_date"
            ]
            .isin(
                dates
            )
            .to_numpy()
        )

        if len(test_idx) == 0:
            raise RuntimeError(
                f"{origin}/{window}: "
                "empty Micro test."
            )

        test_loader = mod_micro.loader(
            x_dir,
            x_size,
            y_micro,
            test_idx,
            False,
        )

        (
            _,
            probs,
            y_true,
        ) = mod_micro.evaluate(
            model,
            test_loader,
            min_size,
            max_size,
            device,
        )

        aligned = align_micro_to_macro(
            micro_df,
            probs,
            y_true,
            test_idx,
            macro_tests[
                window
            ],
            macro_y_tests[
                window
            ],
        )

        window_probs[
            window
        ] = aligned

    del model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (
        window_probs,
        curve,
        {
            "origin":
                origin,

            "epochs":
                MICRO_EPOCHS,

            "seed":
                mod_micro.SEED,

            "n_train":
                len(
                    train_idx
                ),

            "size_min":
                min_size,

            "size_max":
                max_size,

            "fit_seconds":
                seconds,
        },
    )


def evaluate_hybrid(
    helpers,
    micro_probs,
    macro_probs,
    y,
):
    hybrid = geometric_fusion(
        micro_probs,
        macro_probs,
        HYBRID_MACRO_WEIGHT,
    )

    micro_m, _, _ = helpers.evaluate(
        micro_probs,
        y,
    )

    macro_m, _, _ = helpers.evaluate(
        macro_probs,
        y,
    )

    hybrid_m, _, _ = helpers.evaluate(
        hybrid,
        y,
    )

    return {
        "micro_metrics":
            micro_m,

        "macro_metrics":
            macro_m,

        "hybrid_metrics":
            hybrid_m,

        "hybrid_probs":
            hybrid,
    }


def main():
    print("=" * 78)
    print(
        "12A — FROZEN ROBUST HYBRID INTEGRATION"
    )
    print("=" * 78)

    print(
        "MICRO-FINAL architecture/hyperparameters frozen."
    )

    print(
        "MICRO epochs = 30, seed = 42."
    )

    print(
        "Hybrid weights fixed:"
    )

    print(
        "Micro =",
        HYBRID_MICRO_WEIGHT,
        "Macro =",
        HYBRID_MACRO_WEIGHT,
    )

    print(
        "No alpha search."
    )

    print(
        "No online adaptation."
    )

    print(
        "No test-time memory updates."
    )

    print(
        "No Future-B numerical values or scores."
    )

    root = repo_root()

    source_micro = (
        root
        / "src/models/micro/"
        "08B_clean_temporal_micro_baseline.py"
    )

    source10a = (
        root
        / "src/models/final/"
        "10A_refit_full_historical.py"
    )

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

    source11l = (
        root
        / "src/features/macro_v2/"
        "11L_factorial_frozen_combination.py"
    )

    mod_micro = load_module(
        source_micro,
        "micro_12a",
    )

    m10a = load_module(
        source10a,
        "full_hist_10a_for_12a",
    )

    mod = load_module(
        source07a,
        "ltd_07a_for_12a",
    )

    helpers = load_module(
        source11b,
        "ltd_11b_for_12a",
    )

    m11f = load_module(
        source11f,
        "ltd_11f_for_12a",
    )

    m11h = load_module(
        source11h,
        "ltd_11h_for_12a",
    )

    m11l = load_module(
        source11l,
        "ltd_11l_for_12a",
    )

    if (
        helpers.ALPHA_LTD
        !=
        0.375
    ):
        raise RuntimeError(
            "Unexpected Macro LTD alpha."
        )

    if mod_micro.SEED != 42:
        raise RuntimeError(
            "Unexpected MICRO seed."
        )

    # ==================================================
    # MACRO Historical
    # ==================================================

    (
        features,
        frozen_path,
    ) = mod.load_frozen_features()

    (
        captures,
        historical_path,
    ) = mod.load_historical_65()

    captures[
        "site_label"
    ] = (
        captures[
            "site_label"
        ]
        .astype(str)
    )

    captures[
        "pcap_uid"
    ] = (
        captures[
            "pcap_uid"
        ]
        .astype(str)
    )

    captures[
        "_query_date"
    ] = extract_capture_dates(
        captures
    )

    if captures[
        "_query_date"
    ].isna().any():
        raise RuntimeError(
            "Missing Historical dates."
        )

    if captures[
        "pcap_uid"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate Macro pcap_uid."
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

    # ==================================================
    # MICRO Historical
    # ==================================================

    (
        micro_df,
        x_dir,
        x_size,
        y_micro,
        micro_labels,
        micro_source,
    ) = m10a.load_full_micro(
        mod_micro
    )

    micro_df[
        "pcap_uid"
    ] = (
        micro_df[
            "pcap_uid"
        ]
        .astype(str)
    )

    micro_df[
        "site_label"
    ] = (
        micro_df[
            "site_label"
        ]
        .astype(str)
    )

    if not np.array_equal(
        micro_labels,
        candidate_labels,
    ):
        raise RuntimeError(
            "Micro/Macro candidate-label mismatch."
        )

    metadata = (
        captures[
            [
                "pcap_uid",
                "site_label",
                "_query_date",
            ]
        ]
        .rename(
            columns={
                "site_label":
                    "_macro_site_label",
            }
        )
    )

    micro_df = micro_df.merge(
        metadata,
        on="pcap_uid",
        how="left",
        validate="one_to_one",
    )

    if micro_df[
        "_query_date"
    ].isna().any():
        raise RuntimeError(
            "Missing Micro date after Macro bridge."
        )

    if not (
        micro_df[
            "site_label"
        ]
        ==
        micro_df[
            "_macro_site_label"
        ]
    ).all():
        raise RuntimeError(
            "Micro/Macro site-label mismatch."
        )

    micro_df[
        "_query_date"
    ] = pd.to_datetime(
        micro_df[
            "_query_date"
        ]
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
    score_store = {}

    ltd_seed_rows = []
    micro_curves = []
    micro_training_rows = []

    for origin, spec in origins.items():
        print()
        print("#" * 78)
        print(origin)
        print("#" * 78)

        # ==============================================
        # Macro infrastructure from 11H:
        # RECENT5 + UNIFORM + TEMPORAL_SYMMETRIC3
        # ==============================================

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
            ltd_seed_rows,
        )

        (
            micro_probs,
            micro_curve,
            micro_training,
        ) = train_micro_origin(
            mod_micro,
            micro_df,
            x_dir,
            x_size,
            y_micro,
            spec,
            data[
                "tests"
            ],
            data[
                "y_tests"
            ],
            device,
            origin,
        )

        micro_curves.append(
            micro_curve
        )

        micro_training_rows.append(
            micro_training
        )

        # ==============================================
        # MULTISCALE5 from 11B/11L
        # ==============================================

        multiscale_probs = (
            m11l.train_multiscale_ltd(
                mod,
                helpers,
                data[
                    "train"
                ],
                data[
                    "tests"
                ],
                daily_all,
                candidate_labels,
                label_to_idx,
                features,
                spec,
                origin,
                device,
                ltd_seed_rows,
            )
        )

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

        recent_probs = (
            data[
                "ltd_probs"
            ]
        )

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

            macro_variants = {
                "A":
                    helpers.fuse(
                        uniform_probs[
                            window
                        ],
                        recent_probs[
                            window
                        ],
                    ),

                "B":
                    helpers.fuse(
                        temporal_probs[
                            window
                        ],
                        recent_probs[
                            window
                        ],
                    ),

                "C":
                    helpers.fuse(
                        uniform_probs[
                            window
                        ],
                        multiscale_probs[
                            window
                        ],
                    ),

                "D":
                    helpers.fuse(
                        temporal_probs[
                            window
                        ],
                        multiscale_probs[
                            window
                        ],
                    ),
            }

            full_names = {
                "A":
                    "HA_MICRO_UNIFORM_RECENT5",

                "B":
                    "HB_MICRO_TEMPORAL_RECENT5",

                "C":
                    "HC_MICRO_UNIFORM_MULTISCALE5",

                "D":
                    "HD_MICRO_TEMPORAL_MULTISCALE5",
            }

            for short, macro_probs in (
                macro_variants.items()
            ):
                result = evaluate_hybrid(
                    helpers,
                    micro_probs[
                        window
                    ],
                    macro_probs,
                    y,
                )

                summary_rows.append(
                    {
                        "origin":
                            origin,

                        "window":
                            window,

                        "variant":
                            full_names[
                                short
                            ],

                        "n":
                            len(y),

                        **{
                            f"micro_{k}":
                                v
                            for k, v
                            in result[
                                "micro_metrics"
                            ].items()
                        },

                        **{
                            f"macro_{k}":
                                v
                            for k, v
                            in result[
                                "macro_metrics"
                            ].items()
                        },

                        **{
                            f"hybrid_{k}":
                                v
                            for k, v
                            in result[
                                "hybrid_metrics"
                            ].items()
                        },
                    }
                )

                score_store[
                    (
                        origin,
                        window,
                        short,
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

                    "micro":
                        micro_probs[
                            window
                        ],

                    "macro":
                        macro_probs,

                    "hybrid":
                        result[
                            "hybrid_probs"
                        ],
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
    # HD vs HA/HB/HC
    # ==================================================

    delta_rows = []
    interaction_rows = []
    bootstrap_rows = []

    names = {
        "A":
            "HA_MICRO_UNIFORM_RECENT5",

        "B":
            "HB_MICRO_TEMPORAL_RECENT5",

        "C":
            "HC_MICRO_UNIFORM_MULTISCALE5",

        "D":
            "HD_MICRO_TEMPORAL_MULTISCALE5",
    }

    for origin in origins:
        for window in [
            "NEAR",
            "MID",
            "FAR",
        ]:
            rows = {}

            for short, full in names.items():
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
                        f"hybrid_{metric}"
                    )

                    delta_row[
                        (
                            f"delta_HD_vs_H"
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

            # Robust Hybrid vs Micro alone.
            for metric in [
                "accuracy",
                "macro_f1",
                "top5_accuracy",
                "mrr",
            ]:
                delta_row[
                    (
                        f"delta_HD_vs_MICRO_"
                        f"{metric}"
                    )
                ] = (
                    float(
                        rows[
                            "D"
                        ][
                            f"hybrid_{metric}"
                        ]
                    )
                    -
                    float(
                        rows[
                            "D"
                        ][
                            f"micro_{metric}"
                        ]
                    )
                )

            delta_rows.append(
                delta_row
            )

            interaction_rows.append(
                {
                    "origin":
                        origin,

                    "window":
                        window,

                    "interaction_macro_f1":
                        (
                            float(
                                rows[
                                    "D"
                                ][
                                    "hybrid_macro_f1"
                                ]
                            )
                            -
                            float(
                                rows[
                                    "B"
                                ][
                                    "hybrid_macro_f1"
                                ]
                            )
                            -
                            float(
                                rows[
                                    "C"
                                ][
                                    "hybrid_macro_f1"
                                ]
                            )
                            +
                            float(
                                rows[
                                    "A"
                                ][
                                    "hybrid_macro_f1"
                                ]
                            )
                        ),
                }
            )

            candidate_scores = (
                score_store[
                    (
                        origin,
                        window,
                        "D",
                    )
                ]
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
                        reference,
                    )
                ]

                boot = helpers.bootstrap_delta(
                    candidate_scores[
                        "y"
                    ],
                    candidate_scores[
                        "dates"
                    ],
                    ref_scores[
                        "hybrid"
                    ],
                    candidate_scores[
                        "hybrid"
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
                                f"H{reference}",

                            "candidate":
                                "HD",

                            "metric":
                                metric,

                            "observed_delta":
                                delta_row[
                                    (
                                        f"delta_HD_vs_H"
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

    far_boot_ha = bootstrap[
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
            "HA"
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

    mean_far_hd_vs_ha = float(
        far[
            "delta_HD_vs_HA_macro_f1"
        ].mean()
    )

    mean_far_hd_vs_hb = float(
        far[
            "delta_HD_vs_HB_macro_f1"
        ].mean()
    )

    mean_far_hd_vs_hc = float(
        far[
            "delta_HD_vs_HC_macro_f1"
        ].mean()
    )

    mean_far_hd_vs_micro = float(
        far[
            "delta_HD_vs_MICRO_macro_f1"
        ].mean()
    )

    promotion_pass = bool(
        mean_far_hd_vs_ha
        >=
        MIN_MEAN_FAR_HD_VS_HA_F1

        and

        (
            far[
                "delta_HD_vs_HA_macro_f1"
            ]
            >
            0
        ).all()

        and

        near[
            "delta_HD_vs_HA_macro_f1"
        ].min()
        >=
        -MAX_NEAR_HD_VS_HA_DROP

        and

        (
            far_boot_ha[
                "fraction_delta_gt_0"
            ]
            >=
            MIN_FAR_BOOTSTRAP_POSITIVE
        ).all()

        and

        mean_far_hd_vs_hb
        >=
        0.0

        and

        mean_far_hd_vs_hc
        >=
        0.0
    )

    # ==================================================
    # SAVE
    # ==================================================

    summary_path = (
        OUT
        / "12A_hybrid_summary.csv"
    )

    delta_path = (
        OUT
        / "12A_hybrid_deltas.csv"
    )

    interaction_path = (
        OUT
        / "12A_factorial_interaction.csv"
    )

    bootstrap_path = (
        OUT
        / "12A_day_block_bootstrap.csv"
    )

    micro_curve_path = (
        OUT
        / "12A_micro_train_curves.csv"
    )

    micro_training_path = (
        OUT
        / "12A_micro_training_summary.csv"
    )

    ltd_seed_path = (
        OUT
        / "12A_ltd_seed_training.csv"
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

    pd.concat(
        micro_curves,
        ignore_index=True,
    ).to_csv(
        micro_curve_path,
        index=False,
    )

    pd.DataFrame(
        micro_training_rows
    ).to_csv(
        micro_training_path,
        index=False,
    )

    pd.DataFrame(
        ltd_seed_rows
    ).to_csv(
        ltd_seed_path,
        index=False,
    )

    manifest = {
        "stage":
            "12A_frozen_robust_hybrid_integration",

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "git_commit":
            git_commit(),

        "scientific_status":
            "FROZEN_HYBRID_ROBUSTNESS_DEVELOPMENT",

        "research_question":
            (
                "Does the Phase-11 robust Macro gain survive "
                "the frozen MICRO + MACRO final fusion?"
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

        "micro": {
            "architecture":
                "MICRO-FINAL",

            "epochs":
                MICRO_EPOCHS,

            "seed":
                mod_micro.SEED,

            "max_len":
                mod_micro.MAX_LEN,

            "d_model":
                mod_micro.D_MODEL,

            "heads":
                mod_micro.NHEAD,

            "layers":
                mod_micro.NUM_LAYERS,

            "learning_rate":
                mod_micro.LEARNING_RATE,

            "weight_decay":
                mod_micro.WEIGHT_DECAY,

            "label_smoothing":
                mod_micro.LABEL_SMOOTHING,
        },

        "macro": {
            "A":
                "UNIFORM + RECENT5",

            "B":
                "TEMPORAL_SYMMETRIC3 + RECENT5",

            "C":
                "UNIFORM + MULTISCALE5",

            "D":
                "TEMPORAL_SYMMETRIC3 + MULTISCALE5",

            "ltd_weight":
                helpers.ALPHA_LTD,
        },

        "hybrid": {
            "micro_weight":
                HYBRID_MICRO_WEIGHT,

            "macro_weight":
                HYBRID_MACRO_WEIGHT,

            "fusion":
                "weighted geometric probability",
        },

        "promotion_gate": {
            "mean_far_HD_vs_HA_macro_f1_min":
                MIN_MEAN_FAR_HD_VS_HA_F1,

            "all_far_HD_vs_HA_positive":
                True,

            "max_near_HD_vs_HA_drop":
                MAX_NEAR_HD_VS_HA_DROP,

            "all_far_HD_vs_HA_bootstrap_fraction_positive_min":
                MIN_FAR_BOOTSTRAP_POSITIVE,

            "mean_far_HD_vs_HB_nonnegative":
                True,

            "mean_far_HD_vs_HC_nonnegative":
                True,

            "passed":
                promotion_pass,
        },

        "results_summary": {
            "mean_far_HD_vs_HA_macro_f1":
                mean_far_hd_vs_ha,

            "mean_far_HD_vs_HB_macro_f1":
                mean_far_hd_vs_hb,

            "mean_far_HD_vs_HC_macro_f1":
                mean_far_hd_vs_hc,

            "mean_far_HD_vs_MICRO_macro_f1":
                mean_far_hd_vs_micro,
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

            "micro_source_sha256":
                sha256(
                    micro_source
                ),

            "micro_source_code_sha256":
                sha256(
                    source_micro
                ),

            "11L_source_sha256":
                sha256(
                    source11l
                ),
        },

        "outputs": {
            "summary":
                sha256(
                    summary_path
                ),

            "deltas":
                sha256(
                    delta_path
                ),

            "interaction":
                sha256(
                    interaction_path
                ),

            "bootstrap":
                sha256(
                    bootstrap_path
                ),

            "micro_curves":
                sha256(
                    micro_curve_path
                ),

            "micro_training":
                sha256(
                    micro_training_path
                ),

            "ltd_seed_training":
                sha256(
                    ltd_seed_path
                ),
        },

        "next_if_pass":
            (
                "freeze HD as LTD-HYBRID-ROBUST-CANDIDATE; "
                "refit robust architecture on full Historical; "
                "then perform one post-hoc Future-B evaluation"
            ),

        "next_if_fail":
            (
                "close Phase 12 development; "
                "retain Phase-11 Macro result as mechanistic "
                "evidence and keep original LTD-HYBRID-FINAL"
            ),
    }

    write_json(
        OUT
        / "12A_manifest.json",
        manifest,
    )

    print()
    print("=" * 78)
    print(
        "12A HYBRID SUMMARY"
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
        "12A DELTAS"
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
        "12A FAR HD vs HA BOOTSTRAP"
    )
    print("=" * 78)

    print(
        far_boot_ha.to_string(
            index=False
        )
    )

    print()
    print(
        "Mean FAR HD vs HA Macro-F1:",
        f"{mean_far_hd_vs_ha:.6f}",
    )

    print(
        "Mean FAR HD vs HB Macro-F1:",
        f"{mean_far_hd_vs_hb:.6f}",
    )

    print(
        "Mean FAR HD vs HC Macro-F1:",
        f"{mean_far_hd_vs_hc:.6f}",
    )

    print(
        "Mean FAR HD vs MICRO Macro-F1:",
        f"{mean_far_hd_vs_micro:.6f}",
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
