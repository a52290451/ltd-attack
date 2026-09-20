from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json

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

ALPHA = 0.375
EPS = 1e-12

GATES = [
    "NEVER",
    "Q20",
    "Q40",
    "Q60",
    "Q80",
    "ALWAYS",
]


def fuse(
    xgb_probs,
    ltd_probs,
):
    z = (
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

    z -= z.max(
        axis=1,
        keepdims=True,
    )

    p = np.exp(z)

    return (
        p
        / p.sum(
            axis=1,
            keepdims=True,
        )
    )


def margin(
    probs,
):
    s = np.sort(
        probs,
        axis=1,
    )

    return (
        s[:, -1]
        -
        s[:, -2]
    )


def evaluate(
    probs,
    y,
):
    order = np.argsort(
        -probs,
        axis=1,
    )

    pred = order[:, 0]

    ranks = (
        np.argmax(
            order
            == y[:, None],
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
    }, pred


def load_scores(
    out,
    fold,
):
    path = (
        out
        / f"07B_01_scores_{fold}.npz"
    )

    data = np.load(
        path,
        allow_pickle=True,
    )

    return (
        data[
            "xgb_probs"
        ].astype(
            np.float64
        ),

        data[
            "ltd_ensemble_probs"
        ].astype(
            np.float64
        ),

        data[
            "y_true"
        ].astype(int),

        path,
    )


def gated_probs(
    xgb,
    fused,
    xgb_margin,
    gate,
    thresholds,
):
    if gate == "NEVER":
        mask = np.zeros(
            len(xgb),
            dtype=bool,
        )

    elif gate == "ALWAYS":
        mask = np.ones(
            len(xgb),
            dtype=bool,
        )

    else:
        threshold = thresholds[
            gate
        ]

        mask = (
            xgb_margin
            <= threshold
        )

    out = xgb.copy()

    out[
        mask
    ] = fused[
        mask
    ]

    return out, mask


def main():

    print("=" * 78)
    print(
        "07C-2 — CONFIDENCE-GATED FUSION"
    )
    print("=" * 78)

    print(
        "Gate selected ONLY on Fold A."
    )

    print(
        "Fold B does not participate "
        "in gate selection."
    )

    print(
        "Alpha remains frozen at",
        ALPHA,
    )

    print(
        "INTERNAL_TEST: CLOSED"
    )

    print(
        "Future-B: CLOSED"
    )

    out = output_dir()

    threshold_path = (
        out
        / "07C_01_foldA_margin_thresholds.csv"
    )

    t = pd.read_csv(
        threshold_path
    )

    thresholds = {
        "Q20":
            float(
                t.loc[
                    t["boundary"] == "q20",
                    "xgb_margin",
                ].iloc[0]
            ),

        "Q40":
            float(
                t.loc[
                    t["boundary"] == "q40",
                    "xgb_margin",
                ].iloc[0]
            ),

        "Q60":
            float(
                t.loc[
                    t["boundary"] == "q60",
                    "xgb_margin",
                ].iloc[0]
            ),

        "Q80":
            float(
                t.loc[
                    t["boundary"] == "q80",
                    "xgb_margin",
                ].iloc[0]
            ),
    }

    fold_data = {}

    input_paths = {}

    for fold in [
        FOLD_A,
        FOLD_B,
    ]:

        (
            xgb,
            ltd,
            y,
            score_path,
        ) = load_scores(
            out,
            fold,
        )

        fixed = fuse(
            xgb,
            ltd,
        )

        fold_data[
            fold
        ] = {
            "xgb":
                xgb,

            "ltd":
                ltd,

            "fixed":
                fixed,

            "y":
                y,

            "margin":
                margin(
                    xgb
                ),
        }

        input_paths[
            fold
        ] = score_path

    # --------------------------------------------------
    # SELECT GATE ONLY ON FOLD A
    # --------------------------------------------------

    a = fold_data[
        FOLD_A
    ]

    selection_rows = []

    for gate in GATES:

        probs, mask = gated_probs(
            a["xgb"],
            a["fixed"],
            a["margin"],
            gate,
            thresholds,
        )

        metrics, _ = evaluate(
            probs,
            a["y"],
        )

        selection_rows.append(
            {
                "gate":
                    gate,

                "fusion_coverage":
                    float(
                        mask.mean()
                    ),

                **metrics,
            }
        )

    selection_df = pd.DataFrame(
        selection_rows
    )

    complexity = {
        "NEVER": 0,
        "Q20": 1,
        "Q40": 2,
        "Q60": 3,
        "Q80": 4,
        "ALWAYS": 5,
    }

    selection_df[
        "_complexity"
    ] = (
        selection_df[
            "gate"
        ].map(
            complexity
        )
    )

    selected = (
        selection_df
        .sort_values(
            [
                "macro_f1",
                "accuracy",
                "_complexity",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )
        .iloc[0]
    )

    selected_gate = str(
        selected[
            "gate"
        ]
    )

    print()
    print(
        "Selected Fold-A gate:",
        selected_gate,
    )

    print(
        "Fold-A fusion coverage:",
        float(
            selected[
                "fusion_coverage"
            ]
        ),
    )

    # --------------------------------------------------
    # FROZEN TRANSFER TO BOTH FOLDS
    # --------------------------------------------------

    comparison_rows = []
    summary_rows = []

    for fold in [
        FOLD_A,
        FOLD_B,
    ]:

        d = fold_data[
            fold
        ]

        adaptive, mask = gated_probs(
            d["xgb"],
            d["fixed"],
            d["margin"],
            selected_gate,
            thresholds,
        )

        models = {
            "XGB":
                d["xgb"],

            "MACRO_LTD_V1_FIXED":
                d["fixed"],

            "MACRO_LTD_V2_GATE":
                adaptive,
        }

        metrics = {}

        preds = {}

        for name, probs in models.items():

            m, pred = evaluate(
                probs,
                d["y"],
            )

            metrics[
                name
            ] = m

            preds[
                name
            ] = pred

            summary_rows.append(
                {
                    "fold":
                        fold,

                    "model":
                        name,

                    "selected_gate":
                        selected_gate,

                    "fusion_coverage":
                        (
                            float(
                                mask.mean()
                            )
                            if name
                            ==
                            "MACRO_LTD_V2_GATE"
                            else np.nan
                        ),

                    **m,
                }
            )

        v1 = metrics[
            "MACRO_LTD_V1_FIXED"
        ]

        v2 = metrics[
            "MACRO_LTD_V2_GATE"
        ]

        v1_pred = preds[
            "MACRO_LTD_V1_FIXED"
        ]

        v2_pred = preds[
            "MACRO_LTD_V2_GATE"
        ]

        y = d[
            "y"
        ]

        v2_only = (
            (v2_pred == y)
            &
            (v1_pred != y)
        )

        v1_only = (
            (v1_pred == y)
            &
            (v2_pred != y)
        )

        comparison_rows.append(
            {
                "fold":
                    fold,

                "selected_gate":
                    selected_gate,

                "fusion_coverage":
                    float(
                        mask.mean()
                    ),

                "delta_accuracy_v2_vs_v1":
                    (
                        v2[
                            "accuracy"
                        ]
                        -
                        v1[
                            "accuracy"
                        ]
                    ),

                "delta_macro_f1_v2_vs_v1":
                    (
                        v2[
                            "macro_f1"
                        ]
                        -
                        v1[
                            "macro_f1"
                        ]
                    ),

                "delta_top5_v2_vs_v1":
                    (
                        v2[
                            "top5_accuracy"
                        ]
                        -
                        v1[
                            "top5_accuracy"
                        ]
                    ),

                "delta_mrr_v2_vs_v1":
                    (
                        v2[
                            "mrr"
                        ]
                        -
                        v1[
                            "mrr"
                        ]
                    ),

                "v2_only_correct_n":
                    int(
                        v2_only.sum()
                    ),

                "v1_only_correct_n":
                    int(
                        v1_only.sum()
                    ),

                "net_correct_v2_vs_v1":
                    int(
                        v2_only.sum()
                        -
                        v1_only.sum()
                    ),
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    cmp_a = (
        comparison_df[
            comparison_df[
                "fold"
            ]
            == FOLD_A
        ]
        .iloc[0]
    )

    cmp_b = (
        comparison_df[
            comparison_df[
                "fold"
            ]
            == FOLD_B
        ]
        .iloc[0]
    )

    v2_pass = bool(
        selected_gate
        not in {
            "NEVER",
            "ALWAYS",
        }
        and
        cmp_a[
            "delta_accuracy_v2_vs_v1"
        ]
        >= 0
        and
        cmp_a[
            "delta_macro_f1_v2_vs_v1"
        ]
        > 0
        and
        cmp_b[
            "delta_accuracy_v2_vs_v1"
        ]
        >= 0
        and
        cmp_b[
            "delta_macro_f1_v2_vs_v1"
        ]
        > 0
    )

    selection_path = (
        out
        / "07C_02_gate_selection_fold_A.csv"
    )

    summary_path = (
        out
        / "07C_02_gated_fusion_summary.csv"
    )

    comparison_path = (
        out
        / "07C_02_v2_vs_v1.csv"
    )

    selection_df.drop(
        columns=[
            "_complexity"
        ]
    ).to_csv(
        selection_path,
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
            "07C_02_confidence_gated_fusion",

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

            "gate_selected_on":
                FOLD_A,

            "fold_b_used_for_gate_selection":
                False,

            "alpha_tuning":
                False,
        },

        "macro_ltd_v1": {
            "alpha":
                ALPHA,

            "status":
                "FROZEN",
        },

        "adaptive_policy": {
            "type":
                "hard XGB confidence gate",

            "metric":
                "XGB top1-top2 probability margin",

            "candidate_gates":
                GATES,

            "thresholds":
                thresholds,

            "selected_gate":
                selected_gate,
        },

        "v2_gate": {
            "requirements": [
                "selected gate is adaptive",
                "Fold-A Accuracy delta vs V1 >= 0",
                "Fold-A Macro-F1 delta vs V1 > 0",
                "Fold-B Accuracy delta vs V1 >= 0",
                "Fold-B Macro-F1 delta vs V1 > 0",
            ],

            "passed":
                v2_pass,
        },

        "next_if_pass":
            "07C-2R robustness audit for MACRO-LTD-V2",

        "next_if_fail":
            "retain MACRO-LTD-V1 and evaluate another targeted Macro hypothesis",

        "inputs": {
            fold: {
                "path":
                    str(path),

                "sha256":
                    sha256(
                        path
                    ),
            }
            for fold, path
            in input_paths.items()
        },
    }

    manifest_path = (
        out
        / "07C_02_manifest_gated_fusion.json"
    )

    write_json(
        manifest_path,
        manifest,
    )

    print()
    print("=" * 78)
    print(
        "FOLD-A GATE SELECTION"
    )
    print("=" * 78)

    print(
        selection_df.drop(
            columns=[
                "_complexity"
            ]
        ).to_string(
            index=False
        )
    )

    print()
    print("=" * 78)
    print(
        "V2 VS V1"
    )
    print("=" * 78)

    print(
        comparison_df.to_string(
            index=False
        )
    )

    print()
    print(
        "MACRO-LTD-V2 GATE PASS:",
        v2_pass,
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
