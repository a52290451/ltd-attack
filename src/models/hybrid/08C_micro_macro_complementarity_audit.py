from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
)

from src.utils.paths import result_path

from src.features.macro_v2._common import (
    git_commit,
    sha256,
    write_json,
)


FOLDS = [
    "A_EARLY_TO_MIDDLE",
    "B_EARLY_MIDDLE_TO_LATE",
]


MICRO_DIR = Path(
    result_path(
        "micro",
        "MICRO-TEMPORAL-V1",
    )
)

MACRO_DIR = Path(
    result_path(
        "feature_engineering",
        "MACRO-V2-HIST",
    )
)

OUT = Path(
    result_path(
        "hybrid",
        "LTD-HYBRID-DEV",
    )
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


def evaluate(
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
        + 1
    )

    return {
        "accuracy":
            float(
                accuracy_score(
                    y,
                    pred,
                )
            ),

        "macro_f1":
            float(
                f1_score(
                    y,
                    pred,
                    labels=np.arange(
                        len(
                            probs[0]
                        )
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
                    1.0 / ranks
                )
            ),

        "mean_true_rank":
            float(
                np.mean(
                    ranks
                )
            ),
    }, pred, ranks


def align_fold(
    fold,
):

    micro_path = (
        MICRO_DIR
        / f"08B_scores_{fold}.npz"
    )

    macro_path = (
        MACRO_DIR
        / f"07D_01R_scores_{fold}.npz"
    )

    if not micro_path.exists():
        raise FileNotFoundError(
            micro_path
        )

    if not macro_path.exists():
        raise FileNotFoundError(
            macro_path
        )

    micro = np.load(
        micro_path,
        allow_pickle=True,
    )

    macro = np.load(
        macro_path,
        allow_pickle=True,
    )

    micro_uid = (
        micro[
            "pcap_uid"
        ].astype(str)
    )

    macro_uid = (
        macro[
            "pcap_uid"
        ].astype(str)
    )

    if (
        len(
            np.unique(
                micro_uid
            )
        )
        !=
        len(
            micro_uid
        )
    ):
        raise RuntimeError(
            f"{fold}: duplicate Micro pcap_uid"
        )

    if (
        len(
            np.unique(
                macro_uid
            )
        )
        !=
        len(
            macro_uid
        )
    ):
        raise RuntimeError(
            f"{fold}: duplicate Macro pcap_uid"
        )

    if set(
        micro_uid
    ) != set(
        macro_uid
    ):
        raise RuntimeError(
            f"{fold}: Micro/Macro capture sets differ."
        )

    macro_row = {
        uid: i
        for i, uid
        in enumerate(
            macro_uid
        )
    }

    macro_idx = np.array(
        [
            macro_row[
                uid
            ]
            for uid in
            micro_uid
        ],
        dtype=np.int64,
    )

    micro_labels = (
        micro[
            "candidate_labels"
        ].astype(str)
    )

    macro_labels = (
        macro[
            "candidate_labels"
        ].astype(str)
    )

    if set(
        micro_labels
    ) != set(
        macro_labels
    ):
        raise RuntimeError(
            f"{fold}: candidate class sets differ."
        )

    macro_col = {
        label: i
        for i, label
        in enumerate(
            macro_labels
        )
    }

    macro_column_order = np.array(
        [
            macro_col[
                label
            ]
            for label in
            micro_labels
        ],
        dtype=np.int64,
    )

    micro_y = (
        micro[
            "y_true"
        ].astype(
            np.int64
        )
    )

    macro_y_original = (
        macro[
            "y_true"
        ][
            macro_idx
        ]
        .astype(
            np.int64
        )
    )

    micro_true_label = (
        micro_labels[
            micro_y
        ]
    )

    macro_true_label = (
        macro_labels[
            macro_y_original
        ]
    )

    if not np.array_equal(
        micro_true_label,
        macro_true_label,
    ):
        raise RuntimeError(
            f"{fold}: true-label mismatch after pcap alignment."
        )

    micro_probs = (
        micro[
            "micro_probs"
        ].astype(
            np.float64
        )
    )

    macro_probs = (
        macro[
            "v2_fusion_probs"
        ][
            macro_idx
        ][
            :,
            macro_column_order
        ]
        .astype(
            np.float64
        )
    )

    if (
        micro_probs.shape
        !=
        macro_probs.shape
    ):
        raise RuntimeError(
            f"{fold}: probability shape mismatch."
        )

    if (
        micro_probs.shape[
            1
        ]
        != 65
    ):
        raise RuntimeError(
            f"{fold}: expected 65 classes."
        )

    return {
        "micro_path":
            micro_path,

        "macro_path":
            macro_path,

        "pcap_uid":
            micro_uid,

        "labels":
            micro_labels,

        "y":
            micro_y,

        "micro_probs":
            micro_probs,

        "macro_probs":
            macro_probs,
    }


def main():

    print("=" * 78)
    print(
        "08C — MICRO / MACRO COMPLEMENTARITY AUDIT"
    )
    print("=" * 78)

    print(
        "No model training."
    )

    print(
        "No fusion tuning."
    )

    print(
        "Exact pcap_uid pairing."
    )

    print(
        "Micro: MICRO-TEMPORAL-V1"
    )

    print(
        "Macro: MACRO-LTD-FINAL"
    )

    print(
        "INTERNAL_TEST: CLOSED"
    )

    print(
        "Future-B: CLOSED"
    )

    summary_rows = []
    class_rows = []
    input_hashes = {}

    for fold in FOLDS:

        print()
        print("=" * 78)
        print(
            fold
        )
        print("=" * 78)

        data = align_fold(
            fold
        )

        y = data[
            "y"
        ]

        labels = data[
            "labels"
        ]

        micro_probs = data[
            "micro_probs"
        ]

        macro_probs = data[
            "macro_probs"
        ]

        (
            micro_metrics,
            micro_pred,
            micro_rank,
        ) = evaluate(
            micro_probs,
            y,
        )

        (
            macro_metrics,
            macro_pred,
            macro_rank,
        ) = evaluate(
            macro_probs,
            y,
        )

        micro_correct = (
            micro_pred
            ==
            y
        )

        macro_correct = (
            macro_pred
            ==
            y
        )

        both_correct = (
            micro_correct
            &
            macro_correct
        )

        micro_only = (
            micro_correct
            &
            ~macro_correct
        )

        macro_only = (
            ~micro_correct
            &
            macro_correct
        )

        both_wrong = (
            ~micro_correct
            &
            ~macro_correct
        )

        n = len(
            y
        )

        micro_errors = int(
            (
                ~micro_correct
            ).sum()
        )

        macro_errors = int(
            (
                ~macro_correct
            ).sum()
        )

        oracle_correct = (
            micro_correct
            |
            macro_correct
        )

        oracle_accuracy = float(
            oracle_correct.mean()
        )

        micro_top5 = (
            micro_rank
            <= 5
        )

        macro_top5 = (
            macro_rank
            <= 5
        )

        oracle_top5 = float(
            (
                micro_top5
                |
                macro_top5
            ).mean()
        )

        true_idx = np.arange(
            n
        )

        micro_true_prob = (
            micro_probs[
                true_idx,
                y
            ]
        )

        macro_true_prob = (
            macro_probs[
                true_idx,
                y
            ]
        )

        true_prob_corr = float(
            np.corrcoef(
                micro_true_prob,
                macro_true_prob,
            )[
                0,
                1
            ]
        )

        if micro_errors:

            macro_rescue_rate = float(
                macro_only.sum()
                /
                micro_errors
            )

            macro_rank_better_when_micro_wrong = float(
                np.mean(
                    macro_rank[
                        ~micro_correct
                    ]
                    <
                    micro_rank[
                        ~micro_correct
                    ]
                )
            )

        else:

            macro_rescue_rate = np.nan
            macro_rank_better_when_micro_wrong = np.nan

        if macro_errors:

            micro_rescue_rate = float(
                micro_only.sum()
                /
                macro_errors
            )

            micro_rank_better_when_macro_wrong = float(
                np.mean(
                    micro_rank[
                        ~macro_correct
                    ]
                    <
                    macro_rank[
                        ~macro_correct
                    ]
                )
            )

        else:

            micro_rescue_rate = np.nan
            micro_rank_better_when_macro_wrong = np.nan

        summary_rows.append(
            {
                "fold":
                    fold,

                "n":
                    n,

                **{
                    f"micro_{k}":
                        v
                    for k, v
                    in micro_metrics.items()
                },

                **{
                    f"macro_{k}":
                        v
                    for k, v
                    in macro_metrics.items()
                },

                "both_correct_n":
                    int(
                        both_correct.sum()
                    ),

                "micro_only_correct_n":
                    int(
                        micro_only.sum()
                    ),

                "macro_only_correct_n":
                    int(
                        macro_only.sum()
                    ),

                "both_wrong_n":
                    int(
                        both_wrong.sum()
                    ),

                "oracle_accuracy":
                    oracle_accuracy,

                "oracle_gain_over_micro":
                    float(
                        oracle_accuracy
                        -
                        micro_metrics[
                            "accuracy"
                        ]
                    ),

                "oracle_gain_over_best":
                    float(
                        oracle_accuracy
                        -
                        max(
                            micro_metrics[
                                "accuracy"
                            ],
                            macro_metrics[
                                "accuracy"
                            ],
                        )
                    ),

                "macro_rescue_rate_given_micro_wrong":
                    macro_rescue_rate,

                "micro_rescue_rate_given_macro_wrong":
                    micro_rescue_rate,

                "prediction_agreement_rate":
                    float(
                        np.mean(
                            micro_pred
                            ==
                            macro_pred
                        )
                    ),

                "macro_rank_better_given_micro_wrong":
                    macro_rank_better_when_micro_wrong,

                "micro_rank_better_given_macro_wrong":
                    micro_rank_better_when_macro_wrong,

                "oracle_top5_accuracy":
                    oracle_top5,

                "true_class_probability_correlation":
                    true_prob_corr,
            }
        )

        for cls in range(
            65
        ):

            idx = (
                y
                ==
                cls
            )

            class_n = int(
                idx.sum()
            )

            if class_n == 0:
                continue

            mc = micro_correct[
                idx
            ]

            xc = macro_correct[
                idx
            ]

            class_rows.append(
                {
                    "fold":
                        fold,

                    "site_label":
                        labels[
                            cls
                        ],

                    "n":
                        class_n,

                    "micro_accuracy":
                        float(
                            mc.mean()
                        ),

                    "macro_accuracy":
                        float(
                            xc.mean()
                        ),

                    "oracle_accuracy":
                        float(
                            (
                                mc
                                |
                                xc
                            ).mean()
                        ),

                    "micro_only_correct_n":
                        int(
                            (
                                mc
                                &
                                ~xc
                            ).sum()
                        ),

                    "macro_only_correct_n":
                        int(
                            (
                                ~mc
                                &
                                xc
                            ).sum()
                        ),

                    "both_wrong_n":
                        int(
                            (
                                ~mc
                                &
                                ~xc
                            ).sum()
                        ),
                }
            )

        paired = pd.DataFrame(
            {
                "pcap_uid":
                    data[
                        "pcap_uid"
                    ],

                "true_site":
                    labels[
                        y
                    ],

                "micro_pred":
                    labels[
                        micro_pred
                    ],

                "macro_pred":
                    labels[
                        macro_pred
                    ],

                "micro_correct":
                    micro_correct,

                "macro_correct":
                    macro_correct,

                "micro_rank":
                    micro_rank,

                "macro_rank":
                    macro_rank,

                "micro_true_prob":
                    micro_true_prob,

                "macro_true_prob":
                    macro_true_prob,

                "micro_top1_prob":
                    micro_probs.max(
                        axis=1
                    ),

                "macro_top1_prob":
                    macro_probs.max(
                        axis=1
                    ),
            }
        )

        paired.to_csv(
            OUT
            / f"08C_paired_predictions_{fold}.csv",
            index=False,
        )

        input_hashes[
            fold
        ] = {
            "micro_scores":
                {
                    "path":
                        str(
                            data[
                                "micro_path"
                            ]
                        ),

                    "sha256":
                        sha256(
                            data[
                                "micro_path"
                            ]
                        ),
                },

            "macro_scores":
                {
                    "path":
                        str(
                            data[
                                "macro_path"
                            ]
                        ),

                    "sha256":
                        sha256(
                            data[
                                "macro_path"
                            ]
                        ),
                },
        }

    summary = pd.DataFrame(
        summary_rows
    )

    classes = pd.DataFrame(
        class_rows
    )

    summary_path = (
        OUT
        / "08C_complementarity_summary.csv"
    )

    class_path = (
        OUT
        / "08C_per_class_complementarity.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    classes.to_csv(
        class_path,
        index=False,
    )

    manifest = {
        "stage":
            "08C_micro_macro_complementarity_audit",

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "git_commit":
            git_commit(),

        "models": {
            "micro":
                "MICRO-TEMPORAL-V1",

            "macro":
                "MACRO-LTD-FINAL",
        },

        "method": {
            "pairing":
                "exact pcap_uid",

            "candidate_alignment":
                "explicit label-based probability-column alignment",

            "oracle_definition":
                (
                    "correct if either Micro or Macro "
                    "predicts the true class"
                ),

            "oracle_is_deployable_model":
                False,

            "fusion_tuning":
                False,

            "model_training":
                False,
        },

        "data_policy": {
            "development_only":
                True,

            "internal_test_values_used":
                False,

            "external_future_used":
                False,
        },

        "primary_questions": [
            (
                "How often does MACRO-LTD-FINAL rescue "
                "errors made by MICRO-TEMPORAL-V1?"
            ),
            (
                "What is the oracle union headroom above "
                "the stronger standalone model?"
            ),
            (
                "Are the true-class ranks and errors "
                "sufficiently different to justify fusion?"
            ),
        ],

        "inputs":
            input_hashes,

        "next":
            (
                "08D simple probability-level Micro/Macro "
                "fusion using DEV-only protocol"
            ),
    }

    write_json(
        OUT
        / "08C_manifest.json",
        manifest,
    )

    print()
    print("=" * 78)
    print(
        "08C COMPLEMENTARITY SUMMARY"
    )
    print("=" * 78)

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        "No model training."
    )

    print(
        "No fusion tuning."
    )

    print(
        "INTERNAL_TEST remains CLOSED."
    )

    print(
        "Future-B remains CLOSED."
    )


if __name__ == "__main__":
    main()
