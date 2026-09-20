from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
)

from _common import (
    git_commit,
    output_dir,
    sha256,
    write_json,
)


FOLD_A = "A_EARLY_TO_MIDDLE"
FOLD_B = "B_EARLY_MIDDLE_TO_LATE"

ALPHAS = np.round(
    np.linspace(
        0.0,
        1.0,
        41,
    ),
    3,
)

EPS = 1e-12


def evaluate(
    probs,
    y_true,
):
    order = np.argsort(
        -probs,
        axis=1,
    )

    pred = order[:, 0]

    ranks = (
        np.argmax(
            order
            == y_true[:, None],
            axis=1,
        )
        + 1
    )

    return {
        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    pred,
                )
            ),

        "macro_f1":
            float(
                f1_score(
                    y_true,
                    pred,
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


def geometric_fusion(
    xgb_probs,
    ltd_probs,
    alpha,
):
    log_score = (
        (1.0 - alpha)
        * np.log(
            np.clip(
                xgb_probs,
                EPS,
                1.0,
            )
        )
        +
        alpha
        * np.log(
            np.clip(
                ltd_probs,
                EPS,
                1.0,
            )
        )
    )

    log_score -= log_score.max(
        axis=1,
        keepdims=True,
    )

    score = np.exp(
        log_score
    )

    score /= score.sum(
        axis=1,
        keepdims=True,
    )

    return score


def load_scores(
    path,
):
    x = np.load(
        path,
        allow_pickle=False,
    )

    required = {
        "xgb_probs",
        "ltd_ensemble_probs",
        "y_true",
        "candidate_labels",
    }

    missing = (
        required
        - set(x.files)
    )

    if missing:
        raise RuntimeError(
            f"{path}: missing {missing}"
        )

    xgb_probs = x[
        "xgb_probs"
    ].astype(
        np.float64
    )

    ltd_probs = x[
        "ltd_ensemble_probs"
    ].astype(
        np.float64
    )

    y_true = x[
        "y_true"
    ].astype(
        int
    )

    labels = x[
        "candidate_labels"
    ]

    if (
        xgb_probs.shape
        != ltd_probs.shape
    ):
        raise RuntimeError(
            "XGB/LTD score shape mismatch."
        )

    if (
        xgb_probs.shape[1]
        != 65
    ):
        raise RuntimeError(
            "Expected 65 candidates."
        )

    return (
        xgb_probs,
        ltd_probs,
        y_true,
        labels,
    )


def main():

    print("=" * 78)

    print(
        "MACRO-V2-HIST — "
        "07B-2A GEOMETRIC PROBABILITY FUSION"
    )

    print("=" * 78)

    print(
        "Alpha selected ONLY on Fold A."
    )

    print(
        "Fold B does NOT participate "
        "in alpha selection."
    )

    print(
        "INTERNAL_TEST: CLOSED"
    )

    print(
        "Future-B: CLOSED"
    )

    out = output_dir()

    path_a = (
        out
        / f"07B_01_scores_{FOLD_A}.npz"
    )

    path_b = (
        out
        / f"07B_01_scores_{FOLD_B}.npz"
    )

    if not path_a.exists():
        raise FileNotFoundError(
            path_a
        )

    if not path_b.exists():
        raise FileNotFoundError(
            path_b
        )

    (
        xgb_a,
        ltd_a,
        y_a,
        labels_a,
    ) = load_scores(
        path_a
    )

    (
        xgb_b,
        ltd_b,
        y_b,
        labels_b,
    ) = load_scores(
        path_b
    )

    if not np.array_equal(
        labels_a,
        labels_b,
    ):
        raise RuntimeError(
            "Candidate label order differs."
        )

    curve = []

    for alpha in ALPHAS:

        fused = geometric_fusion(
            xgb_a,
            ltd_a,
            float(alpha),
        )

        metrics, _, _ = evaluate(
            fused,
            y_a,
        )

        curve.append(
            {
                "alpha":
                    float(alpha),

                **metrics,
            }
        )

    curve_df = pd.DataFrame(
        curve
    )

    # Primary:
    # maximize Fold-A Macro-F1.
    #
    # Secondary:
    # maximize accuracy.
    #
    # Final tie:
    # prefer smaller alpha, i.e.
    # remain closer to the strong BASE model.
    selected = (
        curve_df
        .sort_values(
            [
                "macro_f1",
                "accuracy",
                "alpha",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )
        .iloc[0]
    )

    alpha = float(
        selected[
            "alpha"
        ]
    )

    print()
    print(
        f"Selected alpha on Fold A: "
        f"{alpha:.3f}"
    )

    summary_rows = []

    fold_data = [
        (
            FOLD_A,
            xgb_a,
            ltd_a,
            y_a,
            "selection",
        ),

        (
            FOLD_B,
            xgb_b,
            ltd_b,
            y_b,
            "temporal_transfer",
        ),
    ]

    comparison_rows = []

    for (
        fold,
        xgb_probs,
        ltd_probs,
        y_true,
        role,
    ) in fold_data:

        fusion_probs = (
            geometric_fusion(
                xgb_probs,
                ltd_probs,
                alpha,
            )
        )

        (
            xgb_metrics,
            xgb_pred,
            xgb_rank,
        ) = evaluate(
            xgb_probs,
            y_true,
        )

        (
            ltd_metrics,
            ltd_pred,
            ltd_rank,
        ) = evaluate(
            ltd_probs,
            y_true,
        )

        (
            fusion_metrics,
            fusion_pred,
            fusion_rank,
        ) = evaluate(
            fusion_probs,
            y_true,
        )

        for model, metrics in [
            (
                "xgb",
                xgb_metrics,
            ),
            (
                "ltd",
                ltd_metrics,
            ),
            (
                "fusion",
                fusion_metrics,
            ),
        ]:

            summary_rows.append(
                {
                    "fold":
                        fold,

                    "role":
                        role,

                    "model":
                        model,

                    "selected_alpha":
                        alpha,

                    **metrics,
                }
            )

        xgb_correct = (
            xgb_pred
            == y_true
        )

        fusion_correct = (
            fusion_pred
            == y_true
        )

        fusion_only = (
            ~xgb_correct
            & fusion_correct
        )

        xgb_only = (
            xgb_correct
            & ~fusion_correct
        )

        comparison_rows.append(
            {
                "fold":
                    fold,

                "role":
                    role,

                "selected_alpha":
                    alpha,

                "delta_accuracy_fusion_vs_xgb":
                    (
                        fusion_metrics[
                            "accuracy"
                        ]
                        -
                        xgb_metrics[
                            "accuracy"
                        ]
                    ),

                "delta_macro_f1_fusion_vs_xgb":
                    (
                        fusion_metrics[
                            "macro_f1"
                        ]
                        -
                        xgb_metrics[
                            "macro_f1"
                        ]
                    ),

                "delta_top5_fusion_vs_xgb":
                    (
                        fusion_metrics[
                            "top5_accuracy"
                        ]
                        -
                        xgb_metrics[
                            "top5_accuracy"
                        ]
                    ),

                "delta_mrr_fusion_vs_xgb":
                    (
                        fusion_metrics[
                            "mrr"
                        ]
                        -
                        xgb_metrics[
                            "mrr"
                        ]
                    ),

                "fusion_only_correct_n":
                    int(
                        fusion_only.sum()
                    ),

                "xgb_only_correct_n":
                    int(
                        xgb_only.sum()
                    ),

                "net_top1_gain_n":
                    int(
                        fusion_only.sum()
                        -
                        xgb_only.sum()
                    ),
            }
        )

        print()
        print(fold)

        print(
            "  XGB    "
            f"acc={xgb_metrics['accuracy']:.4f} "
            f"F1={xgb_metrics['macro_f1']:.4f}"
        )

        print(
            "  LTD    "
            f"acc={ltd_metrics['accuracy']:.4f} "
            f"F1={ltd_metrics['macro_f1']:.4f}"
        )

        print(
            "  FUSION "
            f"acc={fusion_metrics['accuracy']:.4f} "
            f"F1={fusion_metrics['macro_f1']:.4f}"
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    fold_b_cmp = (
        comparison_df[
            comparison_df[
                "fold"
            ]
            == FOLD_B
        ]
        .iloc[0]
    )

    positive_transfer = bool(
        (
            fold_b_cmp[
                "delta_accuracy_fusion_vs_xgb"
            ]
            > 0
        )
        and
        (
            fold_b_cmp[
                "delta_macro_f1_fusion_vs_xgb"
            ]
            > 0
        )
    )

    curve_path = (
        out
        / "07B_02_alpha_curve_fold_A.csv"
    )

    summary_path = (
        out
        / "07B_02_fusion_summary.csv"
    )

    comparison_path = (
        out
        / "07B_02_fusion_vs_xgb.csv"
    )

    curve_df.to_csv(
        curve_path,
        index=False,
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    comparison_df.to_csv(
        comparison_path,
        index=False,
    )

    manifest = {
        "stage":
            "07B_02_geometric_probability_fusion",

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "git_commit":
            git_commit(),

        "data_policy": {
            "development_only":
                True,

            "internal_test_values_used":
                False,

            "external_future_used":
                False,

            "alpha_selected_on":
                FOLD_A,

            "fold_b_used_for_alpha_selection":
                False,

            "fold_b_previously_seen_in_dev_diagnostics":
                True,
        },

        "fusion": {
            "type":
                "weighted geometric probability",

            "formula":
                (
                    "(1-alpha)*log(P_XGB) "
                    "+ alpha*log(P_LTD)"
                ),

            "alpha_grid":
                ALPHAS.tolist(),

            "selected_alpha":
                alpha,

            "selection_primary":
                "Fold-A Macro-F1",

            "selection_secondary":
                "Fold-A Accuracy",

            "tie_break":
                "smallest alpha",
        },

        "inputs": {
            "fold_a_scores": {
                "path":
                    str(path_a),

                "sha256":
                    sha256(path_a),
            },

            "fold_b_scores": {
                "path":
                    str(path_b),

                "sha256":
                    sha256(path_b),
            },
        },

        "decision": {
            "positive_temporal_transfer":
                positive_transfer,

            "rule":
                (
                    "positive only if Fold-B "
                    "Accuracy and Macro-F1 both "
                    "improve over BASE-XGB"
                ),

            "next_if_positive":
                "candidate final fusion / freeze",

            "next_if_negative":
                (
                    "07B-2B conditional reranker "
                    "because 07B-1 established "
                    "substantial oracle complementarity"
                ),
        },
    }

    manifest_path = (
        out
        / "07B_02_manifest_geometric_fusion.json"
    )

    write_json(
        manifest_path,
        manifest,
    )

    print()
    print("=" * 78)
    print(
        "FUSION VS XGB"
    )
    print("=" * 78)

    print(
        comparison_df.to_string(
            index=False
        )
    )

    print()
    print(
        "Positive temporal transfer:",
        positive_transfer,
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
