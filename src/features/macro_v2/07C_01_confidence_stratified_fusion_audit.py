from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json

import numpy as np
import pandas as pd

from _common import (
    git_commit,
    output_dir,
    sha256,
    write_json,
)


FOLD_A = "A_EARLY_TO_MIDDLE"
FOLD_B = "B_EARLY_MIDDLE_TO_LATE"

FOLDS = [
    FOLD_A,
    FOLD_B,
]

ALPHA = 0.375
EPS = 1e-12


def entropy(p):
    p = np.clip(
        p,
        EPS,
        1.0,
    )

    h = -np.sum(
        p * np.log(p),
        axis=1,
    )

    return (
        h
        / np.log(
            p.shape[1]
        )
    )


def margins(p):
    order = np.sort(
        p,
        axis=1,
    )

    return (
        order[:, -1]
        -
        order[:, -2]
    )


def fuse(
    xgb_probs,
    ltd_probs,
):
    score = (
        (1.0 - ALPHA)
        * np.log(
            np.clip(
                xgb_probs,
                EPS,
                1.0,
            )
        )
        +
        ALPHA
        * np.log(
            np.clip(
                ltd_probs,
                EPS,
                1.0,
            )
        )
    )

    score -= score.max(
        axis=1,
        keepdims=True,
    )

    p = np.exp(score)

    return (
        p
        / p.sum(
            axis=1,
            keepdims=True,
        )
    )


def prediction(p):
    return np.argmax(
        p,
        axis=1,
    )


def summarize(
    frame,
    group_name,
    group_value,
):
    n = len(frame)

    if n == 0:
        return None

    xgb_correct = (
        frame[
            "xgb_correct"
        ].to_numpy()
    )

    ltd_correct = (
        frame[
            "ltd_correct"
        ].to_numpy()
    )

    fusion_correct = (
        frame[
            "fusion_correct"
        ].to_numpy()
    )

    xgb_wrong = ~xgb_correct

    fusion_only = (
        ~xgb_correct
        & fusion_correct
    )

    xgb_only = (
        xgb_correct
        & ~fusion_correct
    )

    ltd_only = (
        ~xgb_correct
        & ltd_correct
    )

    return {
        "group":
            group_name,

        "value":
            str(group_value),

        "n":
            n,

        "xgb_accuracy":
            float(
                xgb_correct.mean()
            ),

        "ltd_accuracy":
            float(
                ltd_correct.mean()
            ),

        "fusion_accuracy":
            float(
                fusion_correct.mean()
            ),

        "delta_fusion_vs_xgb":
            float(
                fusion_correct.mean()
                -
                xgb_correct.mean()
            ),

        "fusion_only_correct_n":
            int(
                fusion_only.sum()
            ),

        "xgb_only_correct_n":
            int(
                xgb_only.sum()
            ),

        "fusion_net_correct_n":
            int(
                fusion_only.sum()
                -
                xgb_only.sum()
            ),

        "ltd_only_correct_n":
            int(
                ltd_only.sum()
            ),

        "rescue_rate_given_xgb_wrong":
            (
                float(
                    fusion_only.sum()
                    / xgb_wrong.sum()
                )
                if xgb_wrong.sum()
                else 0.0
            ),

        "damage_rate_given_xgb_correct":
            (
                float(
                    xgb_only.sum()
                    / xgb_correct.sum()
                )
                if xgb_correct.sum()
                else 0.0
            ),

        "xgb_mean_margin":
            float(
                frame[
                    "xgb_margin"
                ].mean()
            ),

        "xgb_mean_entropy":
            float(
                frame[
                    "xgb_entropy"
                ].mean()
            ),

        "ltd_mean_margin":
            float(
                frame[
                    "ltd_margin"
                ].mean()
            ),

        "disagreement_rate":
            float(
                frame[
                    "disagree"
                ].mean()
            ),
    }


def main():
    print("=" * 78)
    print(
        "07C-1 — CONFIDENCE-STRATIFIED "
        "FUSION AUDIT"
    )
    print("=" * 78)

    print(
        "No model training."
    )

    print(
        "No alpha tuning."
    )

    print(
        "Frozen alpha:",
        ALPHA,
    )

    print(
        "INTERNAL_TEST: CLOSED"
    )

    print(
        "Future-B: CLOSED"
    )

    out = output_dir()

    paired_path = (
        out
        / "07B_01_paired_predictions.csv"
    )

    if not paired_path.exists():
        raise FileNotFoundError(
            paired_path
        )

    paired = pd.read_csv(
        paired_path
    )

    # ----------------------------------------------------
    # Fold-A defines confidence thresholds.
    # Fold-B never defines or adjusts them.
    # ----------------------------------------------------

    frames = {}

    fold_a_margins = None

    for fold in FOLDS:

        score_path = (
            out
            / f"07B_01_scores_{fold}.npz"
        )

        data = np.load(
            score_path,
            allow_pickle=True,
        )

        xgb_probs = (
            data[
                "xgb_probs"
            ]
            .astype(
                np.float64
            )
        )

        ltd_probs = (
            data[
                "ltd_ensemble_probs"
            ]
            .astype(
                np.float64
            )
        )

        y = (
            data[
                "y_true"
            ]
            .astype(int)
        )

        pcap_uid = (
            data[
                "pcap_uid"
            ]
            .astype(str)
        )

        fusion_probs = fuse(
            xgb_probs,
            ltd_probs,
        )

        xgb_pred = prediction(
            xgb_probs
        )

        ltd_pred = prediction(
            ltd_probs
        )

        fusion_pred = prediction(
            fusion_probs
        )

        frame = pd.DataFrame(
            {
                "pcap_uid":
                    pcap_uid,

                "y":
                    y,

                "xgb_pred":
                    xgb_pred,

                "ltd_pred":
                    ltd_pred,

                "fusion_pred":
                    fusion_pred,

                "xgb_correct":
                    xgb_pred == y,

                "ltd_correct":
                    ltd_pred == y,

                "fusion_correct":
                    fusion_pred == y,

                "disagree":
                    xgb_pred
                    != ltd_pred,

                "xgb_margin":
                    margins(
                        xgb_probs
                    ),

                "ltd_margin":
                    margins(
                        ltd_probs
                    ),

                "xgb_entropy":
                    entropy(
                        xgb_probs
                    ),

                "ltd_entropy":
                    entropy(
                        ltd_probs
                    ),
            }
        )

        if fold == FOLD_A:
            fold_a_margins = (
                frame[
                    "xgb_margin"
                ].to_numpy()
            )

        frames[
            fold
        ] = frame

    thresholds = np.quantile(
        fold_a_margins,
        [
            0.20,
            0.40,
            0.60,
            0.80,
        ],
    )

    threshold_labels = [
        "Q1_lowest",
        "Q2",
        "Q3",
        "Q4",
        "Q5_highest",
    ]

    print()
    print(
        "Fold-A XGB margin thresholds:",
        thresholds.tolist(),
    )

    rows = []

    for fold, frame in frames.items():

        frame = frame.copy()

        frame[
            "xgb_margin_bin_id"
        ] = np.digitize(
            frame[
                "xgb_margin"
            ].to_numpy(),
            thresholds,
            right=True,
        )

        frame[
            "xgb_margin_bin"
        ] = [
            threshold_labels[i]
            for i in frame[
                "xgb_margin_bin_id"
            ]
        ]

        # Overall
        row = summarize(
            frame,
            "overall",
            "all",
        )

        row[
            "fold"
        ] = fold

        rows.append(row)

        # XGB confidence quintiles
        for value in threshold_labels:

            subset = frame[
                frame[
                    "xgb_margin_bin"
                ]
                == value
            ]

            row = summarize(
                subset,
                "xgb_margin_bin",
                value,
            )

            row[
                "fold"
            ] = fold

            rows.append(row)

        # Agreement / disagreement
        for disagree in [
            False,
            True,
        ]:

            subset = frame[
                frame[
                    "disagree"
                ]
                == disagree
            ]

            row = summarize(
                subset,
                "model_disagreement",
                "disagree"
                if disagree
                else "agree",
            )

            row[
                "fold"
            ] = fold

            rows.append(row)

        # Joint condition:
        # confidence quintile + disagreement.
        for value in threshold_labels:

            subset = frame[
                (
                    frame[
                        "xgb_margin_bin"
                    ]
                    == value
                )
                &
                (
                    frame[
                        "disagree"
                    ]
                )
            ]

            if len(subset) == 0:
                continue

            row = summarize(
                subset,
                "xgb_margin_bin_disagreement",
                value,
            )

            row[
                "fold"
            ] = fold

            rows.append(row)

    summary = pd.DataFrame(
        rows
    )

    summary = summary[
        [
            "fold",
            "group",
            "value",
            "n",
            "xgb_accuracy",
            "ltd_accuracy",
            "fusion_accuracy",
            "delta_fusion_vs_xgb",
            "fusion_only_correct_n",
            "xgb_only_correct_n",
            "fusion_net_correct_n",
            "ltd_only_correct_n",
            "rescue_rate_given_xgb_wrong",
            "damage_rate_given_xgb_correct",
            "xgb_mean_margin",
            "xgb_mean_entropy",
            "ltd_mean_margin",
            "disagreement_rate",
        ]
    ]

    summary_path = (
        out
        / "07C_01_confidence_strata.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    threshold_path = (
        out
        / "07C_01_foldA_margin_thresholds.csv"
    )

    pd.DataFrame(
        {
            "boundary": [
                "q20",
                "q40",
                "q60",
                "q80",
            ],
            "xgb_margin":
                thresholds,
        }
    ).to_csv(
        threshold_path,
        index=False,
    )

    # ----------------------------------------------------
    # Diagnostic for adaptive fusion.
    # No new model is selected here.
    # ----------------------------------------------------

    b = summary[
        summary[
            "fold"
        ]
        == FOLD_B
    ]

    margin_b = (
        b[
            b[
                "group"
            ]
            == "xgb_margin_bin"
        ]
        .set_index(
            "value"
        )
    )

    low_delta = float(
        margin_b.loc[
            [
                "Q1_lowest",
                "Q2",
            ],
            "delta_fusion_vs_xgb",
        ].mean()
    )

    high_delta = float(
        margin_b.loc[
            [
                "Q4",
                "Q5_highest",
            ],
            "delta_fusion_vs_xgb",
        ].mean()
    )

    disagree_b = (
        b[
            (
                b[
                    "group"
                ]
                == "model_disagreement"
            )
            &
            (
                b[
                    "value"
                ]
                == "disagree"
            )
        ]
        .iloc[0]
    )

    adaptive_signal = bool(
        (
            low_delta
            >
            high_delta
        )
        or
        (
            disagree_b[
                "fusion_net_correct_n"
            ]
            > 0
        )
    )

    manifest = {
        "stage":
            "07C_01_confidence_stratified_fusion_audit",

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "git_commit":
            git_commit(),

        "data_policy": {
            "development_only":
                True,

            "model_training":
                False,

            "alpha_tuning":
                False,

            "internal_test_values_used":
                False,

            "external_future_used":
                False,

            "confidence_thresholds_defined_on":
                FOLD_A,

            "fold_b_used_to_define_thresholds":
                False,
        },

        "frozen_macro_v1": {
            "alpha":
                ALPHA,

            "base_features":
                128,

            "context_days":
                7,

            "ltd_seeds":
                [
                    11,
                    42,
                    73,
                ],
        },

        "confidence_strata": {
            "metric":
                "XGB top1-top2 probability margin",

            "fold_a_thresholds":
                thresholds.tolist(),

            "labels":
                threshold_labels,
        },

        "diagnostic": {
            "fold_b_low_confidence_mean_delta":
                low_delta,

            "fold_b_high_confidence_mean_delta":
                high_delta,

            "fold_b_disagreement_net_correct":
                int(
                    disagree_b[
                        "fusion_net_correct_n"
                    ]
                ),

            "adaptive_fusion_signal":
                adaptive_signal,
        },

        "decision": {
            "if_signal":
                (
                    "07C-2 adaptive fusion may be "
                    "evaluated while keeping "
                    "MACRO-LTD-V1 frozen"
                ),

            "if_no_signal":
                (
                    "retain fixed alpha fusion and "
                    "investigate a different Macro "
                    "hypothesis"
                ),
        },

        "inputs": {
            "paired_predictions": {
                "path":
                    str(
                        paired_path
                    ),

                "sha256":
                    sha256(
                        paired_path
                    ),
            },
        },
    }

    manifest_path = (
        out
        / "07C_01_manifest_confidence_audit.json"
    )

    write_json(
        manifest_path,
        manifest,
    )

    print()
    print("=" * 78)
    print("CONFIDENCE STRATA")
    print("=" * 78)

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        "Fold-B low-confidence mean delta:",
        low_delta,
    )

    print(
        "Fold-B high-confidence mean delta:",
        high_delta,
    )

    print(
        "Fold-B disagreement net correct:",
        int(
            disagree_b[
                "fusion_net_correct_n"
            ]
        ),
    )

    print(
        "Adaptive fusion signal:",
        adaptive_signal,
    )

    print()
    print(
        "INTERNAL_TEST remains CLOSED."
    )

    print(
        "Future-B remains CLOSED."
    )


if __name__ == "__main__":
    main()
