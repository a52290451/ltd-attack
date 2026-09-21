from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
)

from src.utils.paths import (
    data_path,
    result_path,
)

from src.features.macro_v2._common import (
    extract_capture_dates,
    git_commit,
    sha256,
    write_json,
)


FOLD_A = "A_EARLY_TO_MIDDLE"
FOLD_B = "B_EARLY_MIDDLE_TO_LATE"

ALPHAS = np.round(
    np.arange(
        0.0,
        1.0001,
        0.025,
    ),
    3,
)

EPS = 1e-12

N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20260921


OUT = Path(
    result_path(
        "hybrid",
        "LTD-HYBRID-DEV",
    )
)

MACRO_METADATA = Path(
    data_path(
        "historical",
        "CLEAN_final_features_sites.csv",
    )
)


def load_08c():

    path = (
        Path(__file__)
        .resolve()
        .parent
        / "08C_micro_macro_complementarity_audit.py"
    )

    spec = importlib.util.spec_from_file_location(
        "ltd_08c",
        path,
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            "Could not import 08C."
        )

    mod = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        mod
    )

    return mod, path


def geometric_fusion(
    micro,
    macro,
    alpha,
):

    z = (
        (1.0 - alpha)
        *
        np.log(
            np.clip(
                micro,
                EPS,
                1.0,
            )
        )
        +
        alpha
        *
        np.log(
            np.clip(
                macro,
                EPS,
                1.0,
            )
        )
    )

    z -= z.max(
        axis=1,
        keepdims=True,
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
                        probs.shape[1]
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
                ranks.mean()
            ),
    }, pred, ranks


def load_dates(
    pcap_uid,
):

    target = set(
        pcap_uid
    )

    header = pd.read_csv(
        MACRO_METADATA,
        nrows=0,
    ).columns.tolist()

    cols = [
        "pcap_uid",
    ]

    for col in [
        "date_id",
        "pcap_name",
    ]:
        if col in header:
            cols.append(
                col
            )

    if len(
        cols
    ) == 1:
        raise RuntimeError(
            "No canonical date metadata."
        )

    chunks = []

    for chunk in pd.read_csv(
        MACRO_METADATA,
        usecols=cols,
        chunksize=4000,
    ):

        chunk[
            "pcap_uid"
        ] = (
            chunk[
                "pcap_uid"
            ].astype(str)
        )

        keep = (
            chunk[
                "pcap_uid"
            ].isin(
                target
            )
        )

        if keep.any():
            chunks.append(
                chunk.loc[
                    keep
                ].copy()
            )

    meta = pd.concat(
        chunks,
        ignore_index=True,
    )

    if meta[
        "pcap_uid"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate date metadata."
        )

    meta[
        "_date"
    ] = (
        extract_capture_dates(
            meta
        )
    )

    date_map = dict(
        zip(
            meta[
                "pcap_uid"
            ],
            meta[
                "_date"
            ],
        )
    )

    dates = np.array(
        [
            date_map[
                uid
            ]
            for uid in
            pcap_uid
        ],
        dtype="datetime64[D]",
    )

    return dates


def delta_metrics(
    y,
    base_pred,
    base_rank,
    fusion_pred,
    fusion_rank,
    idx=None,
):

    if idx is None:
        idx = np.arange(
            len(y)
        )

    yy = y[
        idx
    ]

    bp = base_pred[
        idx
    ]

    fp = fusion_pred[
        idx
    ]

    br = base_rank[
        idx
    ]

    fr = fusion_rank[
        idx
    ]

    base = {
        "accuracy":
            float(
                np.mean(
                    bp == yy
                )
            ),

        "macro_f1":
            float(
                f1_score(
                    yy,
                    bp,
                    labels=np.arange(65),
                    average="macro",
                    zero_division=0,
                )
            ),

        "top5_accuracy":
            float(
                np.mean(
                    br <= 5
                )
            ),

        "mrr":
            float(
                np.mean(
                    1.0 / br
                )
            ),
    }

    fusion = {
        "accuracy":
            float(
                np.mean(
                    fp == yy
                )
            ),

        "macro_f1":
            float(
                f1_score(
                    yy,
                    fp,
                    labels=np.arange(65),
                    average="macro",
                    zero_division=0,
                )
            ),

        "top5_accuracy":
            float(
                np.mean(
                    fr <= 5
                )
            ),

        "mrr":
            float(
                np.mean(
                    1.0 / fr
                )
            ),
    }

    return {
        k:
            fusion[k]
            -
            base[k]
        for k in base
    }


def bootstrap_dates(
    y,
    dates,
    micro_pred,
    micro_rank,
    fusion_pred,
    fusion_rank,
):

    unique_dates = np.array(
        sorted(
            np.unique(
                dates
            )
        )
    )

    by_date = {
        d:
            np.flatnonzero(
                dates
                ==
                d
            )
        for d in unique_dates
    }

    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    rows = []

    for _ in range(
        N_BOOTSTRAP
    ):

        sampled = rng.choice(
            unique_dates,
            size=len(
                unique_dates
            ),
            replace=True,
        )

        idx = np.concatenate(
            [
                by_date[
                    d
                ]
                for d in sampled
            ]
        )

        rows.append(
            delta_metrics(
                y,
                micro_pred,
                micro_rank,
                fusion_pred,
                fusion_rank,
                idx,
            )
        )

    return pd.DataFrame(
        rows
    )


def run_fold(
    mod08c,
    fold,
    alpha,
):

    data = mod08c.align_fold(
        fold
    )

    y = data[
        "y"
    ]

    micro = data[
        "micro_probs"
    ]

    macro = data[
        "macro_probs"
    ]

    fusion = geometric_fusion(
        micro,
        macro,
        alpha,
    )

    micro_m, micro_pred, micro_rank = (
        evaluate(
            micro,
            y,
        )
    )

    macro_m, macro_pred, macro_rank = (
        evaluate(
            macro,
            y,
        )
    )

    fusion_m, fusion_pred, fusion_rank = (
        evaluate(
            fusion,
            y,
        )
    )

    return {
        "data":
            data,

        "y":
            y,

        "micro":
            (
                micro_m,
                micro_pred,
                micro_rank,
            ),

        "macro":
            (
                macro_m,
                macro_pred,
                macro_rank,
            ),

        "fusion":
            (
                fusion_m,
                fusion_pred,
                fusion_rank,
            ),

        "fusion_probs":
            fusion,
    }


def main():

    print("=" * 78)
    print(
        "08D — SIMPLE MICRO/MACRO PROBABILITY FUSION"
    )
    print("=" * 78)

    print(
        "Fusion: weighted geometric probability."
    )

    print(
        "Alpha = Macro weight."
    )

    print(
        "Alpha selected ONLY on Fold A."
    )

    print(
        "Fold B does not participate in alpha selection."
    )

    print(
        "No model training."
    )

    print(
        "INTERNAL_TEST: CLOSED"
    )

    print(
        "Future-B: CLOSED"
    )

    mod08c, source08c = (
        load_08c()
    )

    a_data = mod08c.align_fold(
        FOLD_A
    )

    curve_rows = []

    for alpha in ALPHAS:

        probs = geometric_fusion(
            a_data[
                "micro_probs"
            ],
            a_data[
                "macro_probs"
            ],
            float(
                alpha
            ),
        )

        m, _, _ = evaluate(
            probs,
            a_data[
                "y"
            ],
        )

        curve_rows.append(
            {
                "alpha_macro":
                    float(
                        alpha
                    ),

                **m,
            }
        )

    curve = pd.DataFrame(
        curve_rows
    )

    selected = (
        curve
        .sort_values(
            [
                "macro_f1",
                "accuracy",
                "alpha_macro",
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

    alpha = float(
        selected[
            "alpha_macro"
        ]
    )

    print()
    print(
        "SELECTED ALPHA:",
        alpha,
    )

    print(
        "Micro weight:",
        1.0 - alpha,
    )

    fold_rows = []
    bootstrap_rows = []
    per_class_rows = []
    per_day_rows = []

    input_hashes = {}

    for fold in [
        FOLD_A,
        FOLD_B,
    ]:

        result = run_fold(
            mod08c,
            fold,
            alpha,
        )

        data = result[
            "data"
        ]

        y = result[
            "y"
        ]

        micro_m, micro_pred, micro_rank = (
            result[
                "micro"
            ]
        )

        macro_m, _, _ = (
            result[
                "macro"
            ]
        )

        fusion_m, fusion_pred, fusion_rank = (
            result[
                "fusion"
            ]
        )

        micro_correct = (
            micro_pred == y
        )

        fusion_correct = (
            fusion_pred == y
        )

        fusion_only = (
            ~micro_correct
            &
            fusion_correct
        )

        micro_only = (
            micro_correct
            &
            ~fusion_correct
        )

        fold_rows.append(
            {
                "fold":
                    fold,

                "alpha_macro":
                    alpha,

                "micro_weight":
                    1.0 - alpha,

                **{
                    f"micro_{k}":
                        v
                    for k, v
                    in micro_m.items()
                },

                **{
                    f"macro_{k}":
                        v
                    for k, v
                    in macro_m.items()
                },

                **{
                    f"fusion_{k}":
                        v
                    for k, v
                    in fusion_m.items()
                },

                "delta_accuracy_vs_micro":
                    fusion_m[
                        "accuracy"
                    ]
                    -
                    micro_m[
                        "accuracy"
                    ],

                "delta_macro_f1_vs_micro":
                    fusion_m[
                        "macro_f1"
                    ]
                    -
                    micro_m[
                        "macro_f1"
                    ],

                "delta_top5_vs_micro":
                    fusion_m[
                        "top5_accuracy"
                    ]
                    -
                    micro_m[
                        "top5_accuracy"
                    ],

                "delta_mrr_vs_micro":
                    fusion_m[
                        "mrr"
                    ]
                    -
                    micro_m[
                        "mrr"
                    ],

                "fusion_only_correct_n":
                    int(
                        fusion_only.sum()
                    ),

                "micro_only_correct_n":
                    int(
                        micro_only.sum()
                    ),

                "net_correct_vs_micro":
                    int(
                        fusion_only.sum()
                        -
                        micro_only.sum()
                    ),
            }
        )

        dates = load_dates(
            data[
                "pcap_uid"
            ]
        )

        boot = bootstrap_dates(
            y,
            dates,
            micro_pred,
            micro_rank,
            fusion_pred,
            fusion_rank,
        )

        for metric in [
            "accuracy",
            "macro_f1",
            "top5_accuracy",
            "mrr",
        ]:

            values = boot[
                metric
            ].to_numpy()

            observed = (
                fusion_m[
                    metric
                ]
                -
                micro_m[
                    metric
                ]
            )

            bootstrap_rows.append(
                {
                    "fold":
                        fold,

                    "metric":
                        metric,

                    "observed_fusion_minus_micro":
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

        for date in sorted(
            np.unique(
                dates
            )
        ):

            idx = np.flatnonzero(
                dates
                ==
                date
            )

            d = delta_metrics(
                y,
                micro_pred,
                micro_rank,
                fusion_pred,
                fusion_rank,
                idx,
            )

            per_day_rows.append(
                {
                    "fold":
                        fold,

                    "date":
                        str(
                            date
                        ),

                    "n":
                        len(
                            idx
                        ),

                    **{
                        f"delta_{k}":
                            v
                        for k, v
                        in d.items()
                    },
                }
            )

        labels = data[
            "labels"
        ]

        for cls in range(
            65
        ):

            idx = (
                y
                ==
                cls
            )

            if not idx.any():
                continue

            mc = micro_correct[
                idx
            ]

            fc = fusion_correct[
                idx
            ]

            per_class_rows.append(
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

                    "micro_accuracy":
                        float(
                            mc.mean()
                        ),

                    "fusion_accuracy":
                        float(
                            fc.mean()
                        ),

                    "delta_accuracy":
                        float(
                            fc.mean()
                            -
                            mc.mean()
                        ),

                    "fusion_only_correct_n":
                        int(
                            (
                                ~mc
                                &
                                fc
                            ).sum()
                        ),

                    "micro_only_correct_n":
                        int(
                            (
                                mc
                                &
                                ~fc
                            ).sum()
                        ),
                }
            )

        np.savez_compressed(
            OUT
            / f"08D_scores_{fold}.npz",

            pcap_uid=
                data[
                    "pcap_uid"
                ].astype(str),

            y_true=
                y,

            candidate_labels=
                data[
                    "labels"
                ].astype(str),

            micro_probs=
                data[
                    "micro_probs"
                ].astype(
                    np.float32
                ),

            macro_probs=
                data[
                    "macro_probs"
                ].astype(
                    np.float32
                ),

            fusion_probs=
                result[
                    "fusion_probs"
                ].astype(
                    np.float32
                ),
        )

        input_hashes[
            fold
        ] = {
            "micro":
                sha256(
                    data[
                        "micro_path"
                    ]
                ),

            "macro":
                sha256(
                    data[
                        "macro_path"
                    ]
                ),
        }

    summary = pd.DataFrame(
        fold_rows
    )

    bootstrap = pd.DataFrame(
        bootstrap_rows
    )

    per_class = pd.DataFrame(
        per_class_rows
    )

    per_day = pd.DataFrame(
        per_day_rows
    )

    b = (
        bootstrap[
            bootstrap[
                "fold"
            ]
            ==
            FOLD_B
        ]
        .set_index(
            "metric"
        )
    )

    a = (
        summary[
            summary[
                "fold"
            ]
            ==
            FOLD_A
        ]
        .iloc[
            0
        ]
    )

    b_summary = (
        summary[
            summary[
                "fold"
            ]
            ==
            FOLD_B
        ]
        .iloc[
            0
        ]
    )

    fusion_pass = bool(
        0.0
        <
        alpha
        <
        1.0
        and
        a[
            "delta_accuracy_vs_micro"
        ]
        >=
        0
        and
        a[
            "delta_macro_f1_vs_micro"
        ]
        >
        0
        and
        b_summary[
            "delta_accuracy_vs_micro"
        ]
        >
        0
        and
        b_summary[
            "delta_macro_f1_vs_micro"
        ]
        >
        0
        and
        b.loc[
            "accuracy",
            "fraction_delta_gt_0",
        ]
        >=
        0.90
        and
        b.loc[
            "macro_f1",
            "fraction_delta_gt_0",
        ]
        >=
        0.90
    )

    curve.to_csv(
        OUT
        / "08D_alpha_curve_fold_A.csv",
        index=False,
    )

    summary.to_csv(
        OUT
        / "08D_fusion_summary.csv",
        index=False,
    )

    bootstrap.to_csv(
        OUT
        / "08D_day_block_bootstrap.csv",
        index=False,
    )

    per_class.to_csv(
        OUT
        / "08D_per_class_deltas.csv",
        index=False,
    )

    per_day.to_csv(
        OUT
        / "08D_per_day_deltas.csv",
        index=False,
    )

    manifest = {
        "stage":
            "08D_simple_probability_fusion",

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "git_commit":
            git_commit(),

        "fusion": {
            "type":
                "weighted geometric probability",

            "formula":
                (
                    "(1-alpha)*log(P_micro) "
                    "+ alpha*log(P_macro)"
                ),

            "alpha_definition":
                "Macro weight",

            "candidate_grid":
                [
                    float(x)
                    for x in
                    ALPHAS
                ],

            "selected_alpha":
                alpha,

            "selected_on":
                FOLD_A,

            "selection_primary":
                "Macro-F1",

            "selection_secondary":
                "Accuracy",

            "tie_break":
                "smaller Macro alpha",
        },

        "data_policy": {
            "development_only":
                True,

            "model_training":
                False,

            "fold_b_used_for_alpha_selection":
                False,

            "internal_test_values_used":
                False,

            "external_future_used":
                False,
        },

        "bootstrap": {
            "unit":
                "query_date",

            "iterations":
                N_BOOTSTRAP,

            "seed":
                BOOTSTRAP_SEED,

            "fraction_delta_gt_0_is_p_value":
                False,
        },

        "promotion_gate": {
            "passed":
                fusion_pass,

            "requirements": [
                "0 < alpha < 1",
                "Fold-A fusion Accuracy >= Micro",
                "Fold-A fusion Macro-F1 > Micro",
                "Fold-B fusion Accuracy > Micro",
                "Fold-B fusion Macro-F1 > Micro",
                "Fold-B day-bootstrap positive fraction Accuracy >= 0.90",
                "Fold-B day-bootstrap positive fraction Macro-F1 >= 0.90",
            ],
        },

        "inputs":
            input_hashes,

        "next_if_pass":
            (
                "freeze simple hybrid DEV milestone and "
                "assess remaining oracle headroom before "
                "considering learned fusion"
            ),

        "next_if_fail":
            (
                "evaluate probability calibration before "
                "any learned hybrid architecture"
            ),
    }

    write_json(
        OUT
        / "08D_manifest.json",
        manifest,
    )

    print()
    print("=" * 78)
    print(
        "FOLD-A ALPHA CURVE"
    )
    print("=" * 78)

    print(
        curve.to_string(
            index=False
        )
    )

    print()
    print("=" * 78)
    print(
        "FUSION SUMMARY"
    )
    print("=" * 78)

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        bootstrap.to_string(
            index=False
        )
    )

    print()
    print(
        "HYBRID PROMOTION PASS:",
        fusion_pass,
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
