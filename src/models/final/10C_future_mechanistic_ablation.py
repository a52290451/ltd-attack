from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.utils.paths import result_path

from src.features.macro_v2._common import (
    git_commit,
    sha256,
    write_json,
)


OUT = Path(
    result_path(
        "final",
        "FUTURE-B",
    )
)

MANIFEST_10B = (
    OUT
    / "10B_manifest_future_b.json"
)

SUMMARY_10B = (
    OUT
    / "10B_future_summary.csv"
)


TRACKS = [
    "DEV_FROZEN",
    "HISTORICAL_FINAL",
]

EXPECTED_SITES = 65

EPS = 1e-12

MACRO_ALPHA = 0.375
OUTER_MACRO_WEIGHT = 0.55
OUTER_MICRO_WEIGHT = 0.45

N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20260922


# Effective coefficients of the actual final hybrid:
#
# 0.45 * log(Micro)
# +
# 0.55 * (
#     0.625 * log(XGB)
#     +
#     0.375 * log(LTD)
# )
#
# =
#
# 0.45    Micro
# 0.34375 XGB
# 0.20625 LTD

W_MICRO = OUTER_MICRO_WEIGHT

W_XGB = (
    OUTER_MACRO_WEIGHT
    *
    (
        1.0
        -
        MACRO_ALPHA
    )
)

W_LTD = (
    OUTER_MACRO_WEIGHT
    *
    MACRO_ALPHA
)


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

    p = np.exp(
        z
    )

    return (
        p
        /
        p.sum(
            axis=1,
            keepdims=True,
        )
    )


def geometric_two(
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


def weighted_log_fusion(
    probs_and_weights,
):

    logits = None

    total_weight = 0.0

    for probs, weight in probs_and_weights:

        current = (
            weight
            *
            np.log(
                np.clip(
                    probs,
                    EPS,
                    1.0,
                )
            )
        )

        if logits is None:
            logits = current.copy()

        else:
            logits += current

        total_weight += weight

    if not np.isclose(
        total_weight,
        1.0,
        atol=1e-10,
    ):
        raise RuntimeError(
            f"Fusion weights do not sum to 1: "
            f"{total_weight}"
        )

    return softmax(
        logits
    )


def metrics(
    probs,
    y,
):

    order = np.argsort(
        -probs,
        axis=1,
    )

    pred = order[
        :,
        0
    ]

    ranks = (
        np.argmax(
            order
            ==
            y[
                :,
                None
            ],
            axis=1,
        )
        +
        1
    )

    return (
        {
            "accuracy":
                float(
                    np.mean(
                        pred
                        ==
                        y
                    )
                ),

            "macro_f1":
                float(
                    f1_score(
                        y,
                        pred,
                        labels=np.arange(
                            EXPECTED_SITES
                        ),
                        average="macro",
                        zero_division=0,
                    )
                ),

            "top5_accuracy":
                float(
                    np.mean(
                        ranks <= 5
                    )
                ),

            "mrr":
                float(
                    np.mean(
                        1.0
                        /
                        ranks
                    )
                ),

            "mean_true_rank":
                float(
                    np.mean(
                        ranks
                    )
                ),
        },
        pred,
        ranks,
    )


def verify_10b():

    if not MANIFEST_10B.exists():
        raise FileNotFoundError(
            MANIFEST_10B
        )

    if not SUMMARY_10B.exists():
        raise FileNotFoundError(
            SUMMARY_10B
        )

    manifest = json.loads(
        MANIFEST_10B.read_text()
    )

    if (
        manifest[
            "status"
        ]
        !=
        "FUTURE_B_OPENED_AND_EVALUATED"
    ):
        raise RuntimeError(
            "10B Future-B evaluation is not frozen."
        )

    if (
        manifest[
            "protocol"
        ][
            "model_selection_after_future_open"
        ]
        is not False
    ):
        raise RuntimeError(
            "Unexpected 10B model-selection policy."
        )

    if (
        manifest[
            "protocol"
        ][
            "hyperparameter_changes_after_future_open"
        ]
        is not False
    ):
        raise RuntimeError(
            "Unexpected 10B tuning policy."
        )

    print(
        "PASS: frozen 10B manifest."
    )

    print(
        "PASS: 10C is POST-HOC DIAGNOSTIC ONLY."
    )

    return (
        manifest,
        pd.read_csv(
            SUMMARY_10B
        ),
    )


def load_track(
    track,
):

    path = (
        OUT
        / f"10B_scores_{track}.npz"
    )

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    d = np.load(
        path,
        allow_pickle=True,
    )

    required = [
        "pcap_uid",
        "query_date",
        "y_true",
        "candidate_labels",
        "micro_probs",
        "macro_xgb_probs",
        "macro_ltd_probs",
        "macro_final_probs",
        "hybrid_final_probs",
    ]

    missing = [
        key
        for key in required
        if key not in d.files
    ]

    if missing:
        raise RuntimeError(
            f"{track} score file missing: "
            f"{missing}"
        )

    labels = d[
        "candidate_labels"
    ].astype(str)

    if len(labels) != EXPECTED_SITES:
        raise RuntimeError(
            f"{track}: expected 65 labels."
        )

    return {
        "path":
            path,

        "pcap_uid":
            d[
                "pcap_uid"
            ].astype(str),

        "dates":
            d[
                "query_date"
            ].astype(
                "datetime64[D]"
            ),

        "y":
            d[
                "y_true"
            ].astype(
                np.int64
            ),

        "labels":
            labels,

        "micro":
            d[
                "micro_probs"
            ].astype(
                np.float64
            ),

        "xgb":
            d[
                "macro_xgb_probs"
            ].astype(
                np.float64
            ),

        "ltd":
            d[
                "macro_ltd_probs"
            ].astype(
                np.float64
            ),

        "macro":
            d[
                "macro_final_probs"
            ].astype(
                np.float64
            ),

        "hybrid":
            d[
                "hybrid_final_probs"
            ].astype(
                np.float64
            ),
    }


def verify_scores_against_10b(
    track,
    data,
    summary_10b,
):

    references = {
        "MICRO_FINAL":
            data[
                "micro"
            ],

        "MACRO_XGB":
            data[
                "xgb"
            ],

        "MACRO_LTD":
            data[
                "ltd"
            ],

        "MACRO_LTD_FINAL":
            data[
                "macro"
            ],

        "LTD_HYBRID_FINAL":
            data[
                "hybrid"
            ],
    }

    expected = (
        summary_10b[
            summary_10b[
                "track"
            ]
            ==
            track
        ]
        .set_index(
            "model"
        )
    )

    for model, probs in (
        references.items()
    ):

        m, _, _ = metrics(
            probs,
            data[
                "y"
            ],
        )

        for metric in [
            "accuracy",
            "macro_f1",
            "top5_accuracy",
            "mrr",
        ]:

            observed = m[
                metric
            ]

            target = float(
                expected.loc[
                    model,
                    metric,
                ]
            )

            if not np.isclose(
                observed,
                target,
                atol=2e-6,
            ):
                raise RuntimeError(
                    f"{track} {model} {metric} "
                    "does not reproduce 10B.\n"
                    f"observed={observed}\n"
                    f"expected={target}"
                )

    print(
        f"PASS {track}: stored scores reproduce 10B."
    )


def reconstruct_actual(
    data,
):

    macro_reconstructed = (
        geometric_two(
            data[
                "xgb"
            ],
            data[
                "ltd"
            ],
            MACRO_ALPHA,
        )
    )

    hybrid_reconstructed = (
        geometric_two(
            data[
                "micro"
            ],
            macro_reconstructed,
            OUTER_MACRO_WEIGHT,
        )
    )

    macro_error = float(
        np.max(
            np.abs(
                macro_reconstructed
                -
                data[
                    "macro"
                ]
            )
        )
    )

    hybrid_error = float(
        np.max(
            np.abs(
                hybrid_reconstructed
                -
                data[
                    "hybrid"
                ]
            )
        )
    )

    if macro_error > 5e-6:
        raise RuntimeError(
            "Stored Macro cannot be reconstructed."
        )

    if hybrid_error > 5e-6:
        raise RuntimeError(
            "Stored Hybrid cannot be reconstructed."
        )

    return (
        macro_error,
        hybrid_error,
    )


def build_variants(
    data,
):

    # --------------------------------------------------
    # A. Branch substitutions
    # --------------------------------------------------
    #
    # Preserve the frozen outer Micro/Macro weighting:
    #
    # Micro 0.45
    # candidate Macro branch 0.55
    #

    micro_xgb_outer55 = (
        geometric_two(
            data[
                "micro"
            ],
            data[
                "xgb"
            ],
            OUTER_MACRO_WEIGHT,
        )
    )

    micro_ltd_outer55 = (
        geometric_two(
            data[
                "micro"
            ],
            data[
                "ltd"
            ],
            OUTER_MACRO_WEIGHT,
        )
    )

    # --------------------------------------------------
    # B. Component deletion from the exact three-way
    #    effective hybrid.
    # --------------------------------------------------
    #
    # Actual effective weights:
    #
    # Micro = .45
    # XGB   = .34375
    # LTD   = .20625
    #
    # Delete one component and renormalize remaining
    # coefficients without any optimization.
    #

    no_ltd_sum = (
        W_MICRO
        +
        W_XGB
    )

    no_ltd_micro = (
        W_MICRO
        /
        no_ltd_sum
    )

    no_ltd_xgb = (
        W_XGB
        /
        no_ltd_sum
    )

    no_ltd_renorm = (
        weighted_log_fusion(
            [
                (
                    data[
                        "micro"
                    ],
                    no_ltd_micro,
                ),
                (
                    data[
                        "xgb"
                    ],
                    no_ltd_xgb,
                ),
            ]
        )
    )

    no_xgb_sum = (
        W_MICRO
        +
        W_LTD
    )

    no_xgb_micro = (
        W_MICRO
        /
        no_xgb_sum
    )

    no_xgb_ltd = (
        W_LTD
        /
        no_xgb_sum
    )

    no_xgb_renorm = (
        weighted_log_fusion(
            [
                (
                    data[
                        "micro"
                    ],
                    no_xgb_micro,
                ),
                (
                    data[
                        "ltd"
                    ],
                    no_xgb_ltd,
                ),
            ]
        )
    )

    return {
        "MICRO":
            data[
                "micro"
            ],

        "MICRO_XGB_OUTER55":
            micro_xgb_outer55,

        "MICRO_LTD_OUTER55":
            micro_ltd_outer55,

        "HYBRID_NO_LTD_RENORM":
            no_ltd_renorm,

        "HYBRID_NO_XGB_RENORM":
            no_xgb_renorm,

        "ACTUAL_HYBRID":
            data[
                "hybrid"
            ],
    }


def bootstrap_delta(
    y,
    dates,
    reference_probs,
    candidate_probs,
):

    (
        _,
        ref_pred,
        ref_rank,
    ) = metrics(
        reference_probs,
        y,
    )

    (
        _,
        cand_pred,
        cand_rank,
    ) = metrics(
        candidate_probs,
        y,
    )

    unique_dates = np.array(
        sorted(
            np.unique(
                dates
            )
        )
    )

    by_date = {
        date:
            np.flatnonzero(
                dates == date
            )
        for date in
        unique_dates
    }

    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    values = {
        "accuracy": [],
        "macro_f1": [],
        "top5_accuracy": [],
        "mrr": [],
    }

    for _ in range(
        N_BOOTSTRAP
    ):

        sampled_dates = rng.choice(
            unique_dates,
            size=len(
                unique_dates
            ),
            replace=True,
        )

        idx = np.concatenate(
            [
                by_date[
                    date
                ]
                for date in
                sampled_dates
            ]
        )

        yy = y[
            idx
        ]

        ref_values = {
            "accuracy":
                float(
                    np.mean(
                        ref_pred[
                            idx
                        ]
                        ==
                        yy
                    )
                ),

            "macro_f1":
                float(
                    f1_score(
                        yy,
                        ref_pred[
                            idx
                        ],
                        labels=np.arange(
                            EXPECTED_SITES
                        ),
                        average="macro",
                        zero_division=0,
                    )
                ),

            "top5_accuracy":
                float(
                    np.mean(
                        ref_rank[
                            idx
                        ]
                        <= 5
                    )
                ),

            "mrr":
                float(
                    np.mean(
                        1.0
                        /
                        ref_rank[
                            idx
                        ]
                    )
                ),
        }

        cand_values = {
            "accuracy":
                float(
                    np.mean(
                        cand_pred[
                            idx
                        ]
                        ==
                        yy
                    )
                ),

            "macro_f1":
                float(
                    f1_score(
                        yy,
                        cand_pred[
                            idx
                        ],
                        labels=np.arange(
                            EXPECTED_SITES
                        ),
                        average="macro",
                        zero_division=0,
                    )
                ),

            "top5_accuracy":
                float(
                    np.mean(
                        cand_rank[
                            idx
                        ]
                        <= 5
                    )
                ),

            "mrr":
                float(
                    np.mean(
                        1.0
                        /
                        cand_rank[
                            idx
                        ]
                    )
                ),
        }

        for metric in values:

            values[
                metric
            ].append(
                cand_values[
                    metric
                ]
                -
                ref_values[
                    metric
                ]
            )

    return {
        metric:
            np.asarray(
                vals,
                dtype=float,
            )
        for metric, vals in
        values.items()
    }


def rescue_row(
    track,
    name,
    probs,
    micro_probs,
    actual_probs,
    y,
):

    _, pred, _ = metrics(
        probs,
        y,
    )

    _, micro_pred, _ = metrics(
        micro_probs,
        y,
    )

    _, actual_pred, _ = metrics(
        actual_probs,
        y,
    )

    candidate_correct = (
        pred == y
    )

    micro_correct = (
        micro_pred == y
    )

    actual_correct = (
        actual_pred == y
    )

    micro_wrong = (
        ~micro_correct
    )

    return {
        "track":
            track,

        "variant":
            name,

        "n":
            len(y),

        "correct_n":
            int(
                candidate_correct.sum()
            ),

        "micro_error_n":
            int(
                micro_wrong.sum()
            ),

        "rescues_among_micro_errors_n":
            int(
                (
                    candidate_correct
                    &
                    micro_wrong
                ).sum()
            ),

        "rescue_rate_given_micro_wrong":
            float(
                (
                    candidate_correct
                    &
                    micro_wrong
                ).sum()
                /
                micro_wrong.sum()
            ),

        "candidate_only_vs_actual_n":
            int(
                (
                    candidate_correct
                    &
                    ~actual_correct
                ).sum()
            ),

        "actual_only_vs_candidate_n":
            int(
                (
                    actual_correct
                    &
                    ~candidate_correct
                ).sum()
            ),

        "net_correct_vs_actual":
            int(
                candidate_correct.sum()
                -
                actual_correct.sum()
            ),
    }


def main():

    print("=" * 78)
    print(
        "10C — FUTURE-B MECHANISTIC ABLATION"
    )
    print("=" * 78)

    print(
        "POST-HOC DIAGNOSTIC ONLY."
    )

    print(
        "No training."
    )

    print(
        "No hyperparameter search."
    )

    print(
        "No Future-B CSV is read."
    )

    print(
        "Only frozen 10B probability outputs are used."
    )

    print()

    (
        manifest_10b,
        summary_10b,
    ) = verify_10b()

    summary_rows = []
    bootstrap_rows = []
    rescue_rows = []
    day_rows = []
    class_rows = []

    score_hashes = {}

    reconstruction = {}

    effective_weights = {
        "micro":
            W_MICRO,

        "xgb":
            W_XGB,

        "ltd":
            W_LTD,
    }

    no_ltd_sum = (
        W_MICRO
        +
        W_XGB
    )

    no_xgb_sum = (
        W_MICRO
        +
        W_LTD
    )

    ablation_weights = {
        "HYBRID_NO_LTD_RENORM": {
            "micro":
                W_MICRO
                /
                no_ltd_sum,

            "xgb":
                W_XGB
                /
                no_ltd_sum,
        },

        "HYBRID_NO_XGB_RENORM": {
            "micro":
                W_MICRO
                /
                no_xgb_sum,

            "ltd":
                W_LTD
                /
                no_xgb_sum,
        },
    }

    for track in TRACKS:

        print()
        print("#" * 78)
        print(
            f"TRACK {track}"
        )
        print("#" * 78)

        data = load_track(
            track
        )

        score_hashes[
            track
        ] = {
            "path":
                str(
                    data[
                        "path"
                    ]
                ),

            "sha256":
                sha256(
                    data[
                        "path"
                    ]
                ),
        }

        verify_scores_against_10b(
            track,
            data,
            summary_10b,
        )

        (
            macro_error,
            hybrid_error,
        ) = reconstruct_actual(
            data
        )

        reconstruction[
            track
        ] = {
            "max_abs_macro_probability_error":
                macro_error,

            "max_abs_hybrid_probability_error":
                hybrid_error,
        }

        print(
            "PASS reconstruction:",
            reconstruction[
                track
            ],
        )

        variants = build_variants(
            data
        )

        metrics_by_variant = {}

        for name, probs in variants.items():

            (
                m,
                pred,
                rank,
            ) = metrics(
                probs,
                data[
                    "y"
                ],
            )

            metrics_by_variant[
                name
            ] = m

            summary_rows.append(
                {
                    "track":
                        track,

                    "variant":
                        name,

                    "n":
                        len(
                            data[
                                "y"
                            ]
                        ),

                    **m,
                }
            )

            rescue_rows.append(
                rescue_row(
                    track,
                    name,
                    probs,
                    data[
                        "micro"
                    ],
                    data[
                        "hybrid"
                    ],
                    data[
                        "y"
                    ],
                )
            )

            for date in sorted(
                np.unique(
                    data[
                        "dates"
                    ]
                )
            ):

                idx = np.flatnonzero(
                    data[
                        "dates"
                    ]
                    ==
                    date
                )

                dm, _, _ = metrics(
                    probs[
                        idx
                    ],
                    data[
                        "y"
                    ][
                        idx
                    ],
                )

                day_rows.append(
                    {
                        "track":
                            track,

                        "variant":
                            name,

                        "date":
                            str(
                                date
                            ),

                        "n":
                            len(
                                idx
                            ),

                        **dm,
                    }
                )

            for cls in range(
                EXPECTED_SITES
            ):

                idx = (
                    data[
                        "y"
                    ]
                    ==
                    cls
                )

                if not idx.any():
                    continue

                cm, _, _ = metrics(
                    probs[
                        idx
                    ],
                    data[
                        "y"
                    ][
                        idx
                    ],
                )

                class_rows.append(
                    {
                        "track":
                            track,

                        "variant":
                            name,

                        "site_label":
                            data[
                                "labels"
                            ][
                                cls
                            ],

                        "n":
                            int(
                                idx.sum()
                            ),

                        "accuracy":
                            cm[
                                "accuracy"
                            ],

                        "top5_accuracy":
                            cm[
                                "top5_accuracy"
                            ],

                        "mrr":
                            cm[
                                "mrr"
                            ],

                        "mean_true_rank":
                            cm[
                                "mean_true_rank"
                            ],
                    }
                )

        comparisons = [
            (
                "ACTUAL_HYBRID_vs_MICRO",
                "MICRO",
                "ACTUAL_HYBRID",
            ),

            (
                "MICRO_XGB_OUTER55_vs_MICRO",
                "MICRO",
                "MICRO_XGB_OUTER55",
            ),

            (
                "MICRO_LTD_OUTER55_vs_MICRO",
                "MICRO",
                "MICRO_LTD_OUTER55",
            ),

            (
                "ACTUAL_HYBRID_vs_NO_LTD_RENORM",
                "HYBRID_NO_LTD_RENORM",
                "ACTUAL_HYBRID",
            ),

            (
                "ACTUAL_HYBRID_vs_NO_XGB_RENORM",
                "HYBRID_NO_XGB_RENORM",
                "ACTUAL_HYBRID",
            ),

            (
                "ACTUAL_HYBRID_vs_MICRO_XGB_OUTER55",
                "MICRO_XGB_OUTER55",
                "ACTUAL_HYBRID",
            ),

            (
                "ACTUAL_HYBRID_vs_MICRO_LTD_OUTER55",
                "MICRO_LTD_OUTER55",
                "ACTUAL_HYBRID",
            ),
        ]

        for (
            comparison,
            reference_name,
            candidate_name,
        ) in comparisons:

            reference = variants[
                reference_name
            ]

            candidate = variants[
                candidate_name
            ]

            bootstrap = bootstrap_delta(
                data[
                    "y"
                ],
                data[
                    "dates"
                ],
                reference,
                candidate,
            )

            for metric, values in (
                bootstrap.items()
            ):

                observed = (
                    metrics_by_variant[
                        candidate_name
                    ][
                        metric
                    ]
                    -
                    metrics_by_variant[
                        reference_name
                    ][
                        metric
                    ]
                )

                bootstrap_rows.append(
                    {
                        "track":
                            track,

                        "comparison":
                            comparison,

                        "reference":
                            reference_name,

                        "candidate":
                            candidate_name,

                        "metric":
                            metric,

                        "observed_delta":
                            observed,

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

    summary = pd.DataFrame(
        summary_rows
    )

    bootstrap = pd.DataFrame(
        bootstrap_rows
    )

    rescues = pd.DataFrame(
        rescue_rows
    )

    per_day = pd.DataFrame(
        day_rows
    )

    per_class = pd.DataFrame(
        class_rows
    )

    summary_path = (
        OUT
        / "10C_mechanistic_summary.csv"
    )

    bootstrap_path = (
        OUT
        / "10C_mechanistic_bootstrap.csv"
    )

    rescue_path = (
        OUT
        / "10C_rescue_analysis.csv"
    )

    day_path = (
        OUT
        / "10C_per_day_ablation.csv"
    )

    class_path = (
        OUT
        / "10C_per_class_ablation.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    bootstrap.to_csv(
        bootstrap_path,
        index=False,
    )

    rescues.to_csv(
        rescue_path,
        index=False,
    )

    per_day.to_csv(
        day_path,
        index=False,
    )

    per_class.to_csv(
        class_path,
        index=False,
    )

    manifest = {
        "stage":
            "10C_future_mechanistic_ablation",

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "git_commit":
            git_commit(),

        "status":
            "POSTHOC_MECHANISTIC_ANALYSIS",

        "purpose":
            (
                "Decompose the frozen Future-B Hybrid gain "
                "into static-XGB and longitudinal-LTD "
                "contributions without model training."
            ),

        "scientific_status": {
            "confirmatory_model_selection":
                False,

            "posthoc_analysis":
                True,

            "future_b_already_open":
                True,

            "may_promote_new_model":
                False,

            "may_change_hyperparameters":
                False,
        },

        "data_policy": {
            "future_csv_read":
                False,

            "stored_10b_probabilities_only":
                True,

            "new_training":
                False,

            "new_fitting":
                False,

            "new_alpha_search":
                False,
        },

        "frozen_weights": {
            "macro_alpha":
                MACRO_ALPHA,

            "outer_micro":
                OUTER_MICRO_WEIGHT,

            "outer_macro":
                OUTER_MACRO_WEIGHT,

            "effective_three_way":
                effective_weights,

            "component_deletion_weights":
                ablation_weights,
        },

        "variants": {
            "MICRO":
                "Frozen Micro probabilities.",

            "MICRO_XGB_OUTER55":
                (
                    "Branch substitution diagnostic: "
                    "Micro=.45, XGB=.55."
                ),

            "MICRO_LTD_OUTER55":
                (
                    "Branch substitution diagnostic: "
                    "Micro=.45, LTD=.55."
                ),

            "HYBRID_NO_LTD_RENORM":
                (
                    "Remove LTD from the exact effective "
                    "Hybrid coefficients and renormalize "
                    "Micro/XGB."
                ),

            "HYBRID_NO_XGB_RENORM":
                (
                    "Remove XGB from the exact effective "
                    "Hybrid coefficients and renormalize "
                    "Micro/LTD."
                ),

            "ACTUAL_HYBRID":
                "Frozen LTD-HYBRID-FINAL from 10B.",
        },

        "score_inputs":
            score_hashes,

        "reconstruction_checks":
            reconstruction,

        "bootstrap": {
            "unit":
                "Future-B query date",

            "iterations":
                N_BOOTSTRAP,

            "seed":
                BOOTSTRAP_SEED,

            "fraction_delta_gt_0_is_p_value":
                False,
        },

        "outputs": {
            "summary":
                sha256(
                    summary_path
                ),

            "bootstrap":
                sha256(
                    bootstrap_path
                ),

            "rescues":
                sha256(
                    rescue_path
                ),

            "per_day":
                sha256(
                    day_path
                ),

            "per_class":
                sha256(
                    class_path
                ),
        },

        "interpretation_rule":
            (
                "Results are mechanistic evidence only. "
                "They cannot be used to select or promote "
                "a new Future-B model."
            ),

        "next":
            (
                "Use 10C to define Historical-only "
                "robustness hypotheses for Phase 11."
            ),
    }

    manifest_path = (
        OUT
        / "10C_manifest.json"
    )

    write_json(
        manifest_path,
        manifest,
    )

    print()
    print("=" * 78)
    print(
        "10C SUMMARY"
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
        "RESCUE ANALYSIS"
    )
    print("=" * 78)

    print(
        rescues.to_string(
            index=False
        )
    )

    print()
    print("=" * 78)
    print(
        "KEY BOOTSTRAP COMPARISONS"
    )
    print("=" * 78)

    key = bootstrap[
        bootstrap[
            "metric"
        ].isin(
            [
                "accuracy",
                "macro_f1",
                "top5_accuracy",
                "mrr",
            ]
        )
    ]

    print(
        key.to_string(
            index=False
        )
    )

    print()
    print(
        "10C complete."
    )

    print(
        "No model was trained or selected."
    )

    print(
        "Future-B confirmatory result remains unchanged."
    )


if __name__ == "__main__":
    main()
