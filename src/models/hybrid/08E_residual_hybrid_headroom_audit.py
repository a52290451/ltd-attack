from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score

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

OUT = Path(
    result_path(
        "hybrid",
        "LTD-HYBRID-DEV",
    )
)

EPS = 1e-12


def ranks_and_pred(
    probs,
    y,
):
    order = np.argsort(
        -probs,
        axis=1,
    )

    pred = order[:, 0]

    rank = (
        np.argmax(
            order == y[:, None],
            axis=1,
        )
        + 1
    )

    return (
        pred,
        rank,
    )


def margin(
    probs,
):
    top2 = np.partition(
        probs,
        -2,
        axis=1,
    )[:, -2:]

    top2.sort(
        axis=1
    )

    return (
        top2[:, 1]
        -
        top2[:, 0]
    )


def entropy(
    probs,
):
    p = np.clip(
        probs,
        EPS,
        1.0,
    )

    return (
        -np.sum(
            p * np.log(p),
            axis=1,
        )
        /
        np.log(
            p.shape[1]
        )
    )


def geometric(
    micro,
    macro,
    alpha,
):
    z = (
        (1.0 - alpha)
        * np.log(
            np.clip(
                micro,
                EPS,
                1.0,
            )
        )
        +
        alpha
        * np.log(
            np.clip(
                macro,
                EPS,
                1.0,
            )
        )
    )

    return np.argmax(
        z,
        axis=1,
    )


def main():

    print("=" * 78)
    print(
        "08E — RESIDUAL HYBRID HEADROOM AUDIT"
    )
    print("=" * 78)

    print(
        "No model training."
    )

    print(
        "No fusion selection."
    )

    print(
        "INTERNAL_TEST: CLOSED"
    )

    print(
        "Future-B: CLOSED"
    )

    manifest08d = json.loads(
        (
            OUT
            / "08D_manifest.json"
        ).read_text()
    )

    selected_alpha = float(
        manifest08d[
            "fusion"
        ][
            "selected_alpha"
        ]
    )

    alpha_grid = [
        float(x)
        for x in
        manifest08d[
            "fusion"
        ][
            "candidate_grid"
        ]
    ]

    summary_rows = []
    strata_rows = []
    class_rows = []
    input_hashes = {}

    for fold in FOLDS:

        path = (
            OUT
            / f"08D_scores_{fold}.npz"
        )

        if not path.exists():
            raise FileNotFoundError(
                path
            )

        data = np.load(
            path,
            allow_pickle=True,
        )

        uid = (
            data[
                "pcap_uid"
            ].astype(str)
        )

        y = (
            data[
                "y_true"
            ].astype(int)
        )

        labels = (
            data[
                "candidate_labels"
            ].astype(str)
        )

        micro = (
            data[
                "micro_probs"
            ].astype(
                np.float64
            )
        )

        macro = (
            data[
                "macro_probs"
            ].astype(
                np.float64
            )
        )

        hybrid = (
            data[
                "fusion_probs"
            ].astype(
                np.float64
            )
        )

        micro_pred, micro_rank = (
            ranks_and_pred(
                micro,
                y,
            )
        )

        macro_pred, macro_rank = (
            ranks_and_pred(
                macro,
                y,
            )
        )

        hybrid_pred, hybrid_rank = (
            ranks_and_pred(
                hybrid,
                y,
            )
        )

        micro_correct = (
            micro_pred == y
        )

        macro_correct = (
            macro_pred == y
        )

        hybrid_correct = (
            hybrid_pred == y
        )

        top1_union = (
            micro_correct
            |
            macro_correct
        )

        any_alpha_correct = np.zeros(
            len(y),
            dtype=bool,
        )

        for alpha in alpha_grid:

            pred = geometric(
                micro,
                macro,
                alpha,
            )

            any_alpha_correct |= (
                pred == y
            )

        micro_margin = margin(
            micro
        )

        macro_margin = margin(
            macro
        )

        hybrid_margin = margin(
            hybrid
        )

        micro_entropy = entropy(
            micro
        )

        macro_entropy = entropy(
            macro
        )

        hybrid_entropy = entropy(
            hybrid
        )

        branch_agree = (
            micro_pred
            ==
            macro_pred
        )

        hybrid_wrong = (
            ~hybrid_correct
        )

        hybrid_wrong_either_top1 = (
            hybrid_wrong
            &
            top1_union
        )

        hybrid_wrong_both_top1 = (
            hybrid_wrong
            &
            ~top1_union
        )

        hybrid_correct_both_top1_wrong = (
            hybrid_correct
            &
            ~top1_union
        )

        recoverable_alpha = (
            hybrid_wrong
            &
            any_alpha_correct
        )

        union_top2 = (
            (micro_rank <= 2)
            |
            (macro_rank <= 2)
        )

        union_top3 = (
            (micro_rank <= 3)
            |
            (macro_rank <= 3)
        )

        union_top5 = (
            (micro_rank <= 5)
            |
            (macro_rank <= 5)
        )

        n_errors = int(
            hybrid_wrong.sum()
        )

        summary_rows.append(
            {
                "fold":
                    fold,

                "n":
                    len(y),

                "selected_alpha":
                    selected_alpha,

                "micro_accuracy":
                    float(
                        micro_correct.mean()
                    ),

                "macro_accuracy":
                    float(
                        macro_correct.mean()
                    ),

                "hybrid_accuracy":
                    float(
                        hybrid_correct.mean()
                    ),

                "hybrid_error_n":
                    n_errors,

                "top1_union_oracle_accuracy":
                    float(
                        top1_union.mean()
                    ),

                "union_top2_accuracy":
                    float(
                        union_top2.mean()
                    ),

                "union_top3_accuracy":
                    float(
                        union_top3.mean()
                    ),

                "union_top5_accuracy":
                    float(
                        union_top5.mean()
                    ),

                "any_alpha_grid_oracle_accuracy":
                    float(
                        any_alpha_correct.mean()
                    ),

                "any_alpha_gain_over_fixed":
                    float(
                        any_alpha_correct.mean()
                        -
                        hybrid_correct.mean()
                    ),

                "hybrid_wrong_either_branch_top1_correct_n":
                    int(
                        hybrid_wrong_either_top1.sum()
                    ),

                "hybrid_wrong_both_branch_top1_wrong_n":
                    int(
                        hybrid_wrong_both_top1.sum()
                    ),

                "hybrid_correct_when_both_branch_top1_wrong_n":
                    int(
                        hybrid_correct_both_top1_wrong.sum()
                    ),

                "hybrid_errors_recoverable_by_some_alpha_n":
                    int(
                        recoverable_alpha.sum()
                    ),

                "hybrid_errors_recoverable_by_some_alpha_rate":
                    (
                        float(
                            recoverable_alpha.sum()
                            /
                            n_errors
                        )
                        if n_errors
                        else np.nan
                    ),

                "hybrid_error_true_rank_le2_rate":
                    float(
                        np.mean(
                            hybrid_rank[
                                hybrid_wrong
                            ]
                            <= 2
                        )
                    ),

                "hybrid_error_true_rank_le3_rate":
                    float(
                        np.mean(
                            hybrid_rank[
                                hybrid_wrong
                            ]
                            <= 3
                        )
                    ),

                "hybrid_error_true_rank_le5_rate":
                    float(
                        np.mean(
                            hybrid_rank[
                                hybrid_wrong
                            ]
                            <= 5
                        )
                    ),

                "branch_prediction_agreement_rate":
                    float(
                        branch_agree.mean()
                    ),

                "hybrid_error_branch_disagreement_rate":
                    float(
                        np.mean(
                            ~branch_agree[
                                hybrid_wrong
                            ]
                        )
                    ),
            }
        )

        for correct_value, correct_name in [
            (
                True,
                "HYBRID_CORRECT",
            ),
            (
                False,
                "HYBRID_WRONG",
            ),
        ]:

            for agree_value, agree_name in [
                (
                    True,
                    "BRANCH_AGREE",
                ),
                (
                    False,
                    "BRANCH_DISAGREE",
                ),
            ]:

                idx = (
                    (hybrid_correct == correct_value)
                    &
                    (branch_agree == agree_value)
                )

                if not idx.any():
                    continue

                strata_rows.append(
                    {
                        "fold":
                            fold,

                        "hybrid_status":
                            correct_name,

                        "branch_status":
                            agree_name,

                        "n":
                            int(
                                idx.sum()
                            ),

                        "fraction":
                            float(
                                idx.mean()
                            ),

                        "micro_margin_mean":
                            float(
                                micro_margin[
                                    idx
                                ].mean()
                            ),

                        "macro_margin_mean":
                            float(
                                macro_margin[
                                    idx
                                ].mean()
                            ),

                        "hybrid_margin_mean":
                            float(
                                hybrid_margin[
                                    idx
                                ].mean()
                            ),

                        "micro_entropy_mean":
                            float(
                                micro_entropy[
                                    idx
                                ].mean()
                            ),

                        "macro_entropy_mean":
                            float(
                                macro_entropy[
                                    idx
                                ].mean()
                            ),

                        "hybrid_entropy_mean":
                            float(
                                hybrid_entropy[
                                    idx
                                ].mean()
                            ),

                        "any_alpha_correct_rate":
                            float(
                                any_alpha_correct[
                                    idx
                                ].mean()
                            ),
                    }
                )

        for cls in range(
            len(
                labels
            )
        ):

            idx = (
                y == cls
            )

            if not idx.any():
                continue

            hc = hybrid_correct[
                idx
            ]

            aa = any_alpha_correct[
                idx
            ]

            u5 = union_top5[
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
                        int(
                            idx.sum()
                        ),

                    "hybrid_accuracy":
                        float(
                            hc.mean()
                        ),

                    "any_alpha_oracle_accuracy":
                        float(
                            aa.mean()
                        ),

                    "union_top5_accuracy":
                        float(
                            u5.mean()
                        ),

                    "hybrid_error_n":
                        int(
                            (~hc).sum()
                        ),

                    "recoverable_by_alpha_n":
                        int(
                            (
                                ~hc
                                &
                                aa
                            ).sum()
                        ),
                }
            )

        input_hashes[
            fold
        ] = {
            "path":
                str(
                    path
                ),

            "sha256":
                sha256(
                    path
                ),
        }

    summary = pd.DataFrame(
        summary_rows
    )

    strata = pd.DataFrame(
        strata_rows
    )

    classes = pd.DataFrame(
        class_rows
    )

    summary.to_csv(
        OUT
        / "08E_residual_summary.csv",
        index=False,
    )

    strata.to_csv(
        OUT
        / "08E_confidence_strata.csv",
        index=False,
    )

    classes.to_csv(
        OUT
        / "08E_per_class_residual.csv",
        index=False,
    )

    manifest = {
        "stage":
            "08E_residual_hybrid_headroom_audit",

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "git_commit":
            git_commit(),

        "reference":
            "LTD-HYBRID-V1",

        "selected_alpha":
            selected_alpha,

        "analysis": {
            "model_training":
                False,

            "fusion_selection":
                False,

            "per_query_alpha_oracle":
                (
                    "diagnostic only; uses true labels "
                    "to determine whether any predeclared "
                    "alpha would have been correct"
                ),

            "topk_union":
                "diagnostic only",
        },

        "data_policy": {
            "development_only":
                True,

            "internal_test_values_used":
                False,

            "external_future_used":
                False,
        },

        "inputs":
            input_hashes,

        "next":
            (
                "Decide prospectively whether residual structure "
                "justifies one learned/calibrated fusion stage."
            ),
    }

    write_json(
        OUT
        / "08E_manifest.json",
        manifest,
    )

    print()
    print("=" * 78)
    print(
        "08E RESIDUAL SUMMARY"
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
        "CONFIDENCE STRATA"
    )
    print("=" * 78)

    print(
        strata.to_string(
            index=False
        )
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
