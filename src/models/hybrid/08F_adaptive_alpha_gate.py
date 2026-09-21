from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import importlib.util
import math

import numpy as np
import pandas as pd

from sklearn.metrics import f1_score

import torch
import torch.nn.functional as F

from src.utils.paths import result_path

from src.features.macro_v2._common import (
    git_commit,
    sha256,
    write_json,
)


FOLD_A = "A_EARLY_TO_MIDDLE"
FOLD_B = "B_EARLY_MIDDLE_TO_LATE"

REFERENCE_ALPHA = 0.55

EPS = 1e-12

L2 = 1e-3
MAX_LBFGS_ITER = 250

N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20260921


OUT = Path(
    result_path(
        "hybrid",
        "LTD-HYBRID-DEV",
    )
)


def load_08d():

    path = (
        Path(__file__)
        .resolve()
        .parent
        / "08D_simple_probability_fusion.py"
    )

    spec = importlib.util.spec_from_file_location(
        "ltd_08d",
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Could not load 08D."
        )

    mod = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        mod
    )

    return mod, path


def load_fold(
    fold,
):

    path = (
        OUT
        / f"08D_scores_{fold}.npz"
    )

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    d = np.load(
        path,
        allow_pickle=True,
    )

    return {
        "path":
            path,

        "pcap_uid":
            d[
                "pcap_uid"
            ].astype(str),

        "y":
            d[
                "y_true"
            ].astype(
                np.int64
            ),

        "labels":
            d[
                "candidate_labels"
            ].astype(str),

        "micro":
            d[
                "micro_probs"
            ].astype(
                np.float64
            ),

        "macro":
            d[
                "macro_probs"
            ].astype(
                np.float64
            ),

        "fixed":
            d[
                "fusion_probs"
            ].astype(
                np.float64
            ),
    }


def normalized_entropy(
    p,
):

    q = np.clip(
        p,
        EPS,
        1.0,
    )

    return (
        -np.sum(
            q
            *
            np.log(
                q
            ),
            axis=1,
        )
        /
        np.log(
            q.shape[
                1
            ]
        )
    )


def margin(
    p,
):

    top2 = np.partition(
        p,
        -2,
        axis=1,
    )[
        :,
        -2:
    ]

    top2.sort(
        axis=1
    )

    return (
        top2[
            :,
            1
        ]
        -
        top2[
            :,
            0
        ]
    )


def js_divergence(
    p,
    q,
):

    p = np.clip(
        p,
        EPS,
        1.0,
    )

    q = np.clip(
        q,
        EPS,
        1.0,
    )

    m = (
        0.5
        *
        (
            p
            +
            q
        )
    )

    return (
        0.5
        *
        np.sum(
            p
            *
            np.log(
                p
                /
                m
            ),
            axis=1,
        )
        +
        0.5
        *
        np.sum(
            q
            *
            np.log(
                q
                /
                m
            ),
            axis=1,
        )
    )


def gate_features(
    micro,
    macro,
):

    n = len(
        micro
    )

    rows = np.arange(
        n
    )

    micro_top = np.argmax(
        micro,
        axis=1,
    )

    macro_top = np.argmax(
        macro,
        axis=1,
    )

    micro_conf = micro[
        rows,
        micro_top
    ]

    macro_conf = macro[
        rows,
        macro_top
    ]

    micro_margin = margin(
        micro
    )

    macro_margin = margin(
        macro
    )

    micro_entropy = (
        normalized_entropy(
            micro
        )
    )

    macro_entropy = (
        normalized_entropy(
            macro
        )
    )

    agree = (
        micro_top
        ==
        macro_top
    ).astype(
        np.float64
    )

    macro_on_micro_top = macro[
        rows,
        micro_top
    ]

    micro_on_macro_top = micro[
        rows,
        macro_top
    ]

    js = js_divergence(
        micro,
        macro,
    )

    names = [
        "micro_conf",
        "macro_conf",
        "micro_margin",
        "macro_margin",
        "micro_entropy",
        "macro_entropy",
        "macro_minus_micro_conf",
        "macro_minus_micro_margin",
        "macro_minus_micro_entropy",
        "branch_agree",
        "macro_prob_on_micro_top1",
        "micro_prob_on_macro_top1",
        "js_divergence",
    ]

    x = np.column_stack(
        [
            micro_conf,
            macro_conf,
            micro_margin,
            macro_margin,
            micro_entropy,
            macro_entropy,
            macro_conf
            -
            micro_conf,
            macro_margin
            -
            micro_margin,
            macro_entropy
            -
            micro_entropy,
            agree,
            macro_on_micro_top,
            micro_on_macro_top,
            js,
        ]
    ).astype(
        np.float64
    )

    return (
        x,
        names,
        agree,
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


def fit_gate(
    x,
    micro,
    macro,
    y,
):

    mu = x.mean(
        axis=0
    )

    sigma = x.std(
        axis=0
    )

    sigma[
        sigma
        <
        1e-8
    ] = 1.0

    z = (
        x
        -
        mu
    ) / sigma

    z_t = torch.tensor(
        z,
        dtype=torch.float64,
    )

    micro_log_t = torch.tensor(
        np.log(
            np.clip(
                micro,
                EPS,
                1.0,
            )
        ),
        dtype=torch.float64,
    )

    macro_log_t = torch.tensor(
        np.log(
            np.clip(
                macro,
                EPS,
                1.0,
            )
        ),
        dtype=torch.float64,
    )

    y_t = torch.tensor(
        y,
        dtype=torch.long,
    )

    w = torch.nn.Parameter(
        torch.zeros(
            z.shape[
                1
            ],
            dtype=torch.float64,
        )
    )

    initial_logit = math.log(
        REFERENCE_ALPHA
        /
        (
            1.0
            -
            REFERENCE_ALPHA
        )
    )

    b = torch.nn.Parameter(
        torch.tensor(
            initial_logit,
            dtype=torch.float64,
        )
    )

    optimizer = torch.optim.LBFGS(
        [
            w,
            b,
        ],
        lr=1.0,
        max_iter=
            MAX_LBFGS_ITER,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        line_search_fn=
            "strong_wolfe",
    )

    def closure():

        optimizer.zero_grad()

        alpha = torch.sigmoid(
            z_t
            @
            w
            +
            b
        ).unsqueeze(
            1
        )

        logits = (
            (
                1.0
                -
                alpha
            )
            *
            micro_log_t
            +
            alpha
            *
            macro_log_t
        )

        ce = F.cross_entropy(
            logits,
            y_t,
        )

        reg = (
            L2
            *
            torch.sum(
                w
                *
                w
            )
        )

        loss = (
            ce
            +
            reg
        )

        loss.backward()

        return loss

    final_loss = optimizer.step(
        closure
    )

    return {
        "mu":
            mu,

        "sigma":
            sigma,

        "w":
            w.detach()
            .cpu()
            .numpy(),

        "b":
            float(
                b.detach()
                .cpu()
                .item()
            ),

        "objective":
            float(
                final_loss
                .detach()
                .cpu()
                .item()
            ),
    }


def apply_gate(
    params,
    x,
    micro,
    macro,
):

    z = (
        x
        -
        params[
            "mu"
        ]
    ) / params[
        "sigma"
    ]

    linear = (
        z
        @
        params[
            "w"
        ]
        +
        params[
            "b"
        ]
    )

    alpha = (
        1.0
        /
        (
            1.0
            +
            np.exp(
                -np.clip(
                    linear,
                    -30.0,
                    30.0,
                )
            )
        )
    )

    logits = (
        (
            1.0
            -
            alpha[
                :,
                None
            ]
        )
        *
        np.log(
            np.clip(
                micro,
                EPS,
                1.0,
            )
        )
        +
        alpha[
            :,
            None
        ]
        *
        np.log(
            np.clip(
                macro,
                EPS,
                1.0,
            )
        )
    )

    return (
        softmax(
            logits
        ),
        alpha,
    )


def delta(
    candidate,
    reference,
):

    return {
        k:
            candidate[
                k
            ]
            -
            reference[
                k
            ]
        for k in [
            "accuracy",
            "macro_f1",
            "top5_accuracy",
            "mrr",
        ]
    }


def alpha_summary(
    fold,
    alpha,
    agree,
):

    rows = []

    for name, idx in [
        (
            "ALL",
            np.ones(
                len(
                    alpha
                ),
                dtype=bool,
            ),
        ),
        (
            "BRANCH_AGREE",
            agree.astype(
                bool
            ),
        ),
        (
            "BRANCH_DISAGREE",
            ~agree.astype(
                bool
            ),
        ),
    ]:

        if not idx.any():
            continue

        a = alpha[
            idx
        ]

        rows.append(
            {
                "fold":
                    fold,

                "stratum":
                    name,

                "n":
                    len(
                        a
                    ),

                "alpha_mean":
                    float(
                        a.mean()
                    ),

                "alpha_median":
                    float(
                        np.median(
                            a
                        )
                    ),

                "alpha_q10":
                    float(
                        np.quantile(
                            a,
                            0.10,
                        )
                    ),

                "alpha_q90":
                    float(
                        np.quantile(
                            a,
                            0.90,
                        )
                    ),

                "alpha_min":
                    float(
                        a.min()
                    ),

                "alpha_max":
                    float(
                        a.max()
                    ),
            }
        )

    return rows


def main():

    print("=" * 78)
    print(
        "08F — ADAPTIVE ALPHA GATE"
    )
    print("=" * 78)

    print(
        "Low-DOF label-free confidence gate."
    )

    print(
        "No class IDs."
    )

    print(
        "Gate training uses Fold A only."
    )

    print(
        "Fold B is temporal transfer only."
    )

    print(
        "INTERNAL_TEST: CLOSED"
    )

    print(
        "Future-B: CLOSED"
    )

    mod08d, source08d = (
        load_08d()
    )

    a = load_fold(
        FOLD_A
    )

    a_dates = mod08d.load_dates(
        a[
            "pcap_uid"
        ]
    )

    unique_a_dates = np.array(
        sorted(
            np.unique(
                a_dates
            )
        )
    )

    if len(
        unique_a_dates
    ) < 4:
        raise RuntimeError(
            "Not enough Fold-A dates."
        )

    midpoint = (
        len(
            unique_a_dates
        )
        //
        2
    )

    gate_train_dates = (
        unique_a_dates[
            :midpoint
        ]
    )

    gate_val_dates = (
        unique_a_dates[
            midpoint:
        ]
    )

    train_idx = np.isin(
        a_dates,
        gate_train_dates,
    )

    val_idx = np.isin(
        a_dates,
        gate_val_dates,
    )

    x_a, feature_names, agree_a = (
        gate_features(
            a[
                "micro"
            ],
            a[
                "macro"
            ],
        )
    )

    inner_params = fit_gate(
        x_a[
            train_idx
        ],
        a[
            "micro"
        ][
            train_idx
        ],
        a[
            "macro"
        ][
            train_idx
        ],
        a[
            "y"
        ][
            train_idx
        ],
    )

    inner_probs, inner_alpha = (
        apply_gate(
            inner_params,
            x_a[
                val_idx
            ],
            a[
                "micro"
            ][
                val_idx
            ],
            a[
                "macro"
            ][
                val_idx
            ],
        )
    )

    fixed_val = a[
        "fixed"
    ][
        val_idx
    ]

    y_val = a[
        "y"
    ][
        val_idx
    ]

    fixed_val_m, _, _ = (
        mod08d.evaluate(
            fixed_val,
            y_val,
        )
    )

    adaptive_val_m, _, _ = (
        mod08d.evaluate(
            inner_probs,
            y_val,
        )
    )

    inner_delta = delta(
        adaptive_val_m,
        fixed_val_m,
    )

    print()
    print(
        "INNER FOLD-A TEMPORAL VALIDATION"
    )

    print(
        "train dates:",
        [
            str(x)
            for x in
            gate_train_dates
        ],
    )

    print(
        "validation dates:",
        [
            str(x)
            for x in
            gate_val_dates
        ],
    )

    print(
        "fixed:",
        fixed_val_m
    )

    print(
        "adaptive:",
        adaptive_val_m
    )

    print(
        "delta:",
        inner_delta
    )

    final_params = fit_gate(
        x_a,
        a[
            "micro"
        ],
        a[
            "macro"
        ],
        a[
            "y"
        ],
    )

    summary_rows = []
    alpha_rows = []
    bootstrap_rows = []

    b_reference = None
    b_candidate = None

    input_hashes = {}

    for fold in [
        FOLD_A,
        FOLD_B,
    ]:

        d = (
            a
            if fold
            ==
            FOLD_A
            else load_fold(
                FOLD_B
            )
        )

        x, _, agree = gate_features(
            d[
                "micro"
            ],
            d[
                "macro"
            ],
        )

        probs, alpha = apply_gate(
            final_params,
            x,
            d[
                "micro"
            ],
            d[
                "macro"
            ],
        )

        fixed_m, fixed_pred, fixed_rank = (
            mod08d.evaluate(
                d[
                    "fixed"
                ],
                d[
                    "y"
                ],
            )
        )

        adaptive_m, adaptive_pred, adaptive_rank = (
            mod08d.evaluate(
                probs,
                d[
                    "y"
                ],
            )
        )

        dd = delta(
            adaptive_m,
            fixed_m,
        )

        summary_rows.append(
            {
                "fold":
                    fold,

                **{
                    f"fixed_{k}":
                        v
                    for k, v
                    in fixed_m.items()
                },

                **{
                    f"adaptive_{k}":
                        v
                    for k, v
                    in adaptive_m.items()
                },

                **{
                    f"delta_{k}":
                        v
                    for k, v
                    in dd.items()
                },

                "alpha_mean":
                    float(
                        alpha.mean()
                    ),

                "alpha_median":
                    float(
                        np.median(
                            alpha
                        )
                    ),
            }
        )

        alpha_rows.extend(
            alpha_summary(
                fold,
                alpha,
                agree,
            )
        )

        np.savez_compressed(
            OUT
            / f"08F_scores_{fold}.npz",

            pcap_uid=
                d[
                    "pcap_uid"
                ].astype(str),

            y_true=
                d[
                    "y"
                ],

            candidate_labels=
                d[
                    "labels"
                ].astype(str),

            micro_probs=
                d[
                    "micro"
                ].astype(
                    np.float32
                ),

            macro_probs=
                d[
                    "macro"
                ].astype(
                    np.float32
                ),

            fixed_fusion_probs=
                d[
                    "fixed"
                ].astype(
                    np.float32
                ),

            adaptive_fusion_probs=
                probs.astype(
                    np.float32
                ),

            adaptive_alpha=
                alpha.astype(
                    np.float32
                ),
        )

        if fold == FOLD_B:

            dates = mod08d.load_dates(
                d[
                    "pcap_uid"
                ]
            )

            boot = mod08d.bootstrap_dates(
                d[
                    "y"
                ],
                dates,
                fixed_pred,
                fixed_rank,
                adaptive_pred,
                adaptive_rank,
            )

            for metric in [
                "accuracy",
                "macro_f1",
                "top5_accuracy",
                "mrr",
            ]:

                values = (
                    boot[
                        metric
                    ]
                    .to_numpy()
                )

                bootstrap_rows.append(
                    {
                        "fold":
                            fold,

                        "metric":
                            metric,

                        "observed_adaptive_minus_fixed":
                            dd[
                                metric
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

            b_reference = (
                fixed_m
            )

            b_candidate = (
                adaptive_m
            )

        input_hashes[
            fold
        ] = {
            "path":
                str(
                    d[
                        "path"
                    ]
                ),

            "sha256":
                sha256(
                    d[
                        "path"
                    ]
                ),
        }

    summary = pd.DataFrame(
        summary_rows
    )

    alpha_df = pd.DataFrame(
        alpha_rows
    )

    bootstrap = pd.DataFrame(
        bootstrap_rows
    )

    b_boot = (
        bootstrap
        .set_index(
            "metric"
        )
    )

    promotion_pass = bool(
        inner_delta[
            "accuracy"
        ] >= 0
        and
        inner_delta[
            "macro_f1"
        ] > 0
        and
        b_candidate[
            "accuracy"
        ]
        >
        b_reference[
            "accuracy"
        ]
        and
        b_candidate[
            "macro_f1"
        ]
        >
        b_reference[
            "macro_f1"
        ]
        and
        b_candidate[
            "top5_accuracy"
        ]
        >=
        b_reference[
            "top5_accuracy"
        ]
        and
        b_candidate[
            "mrr"
        ]
        >
        b_reference[
            "mrr"
        ]
        and
        b_boot.loc[
            "accuracy",
            "fraction_delta_gt_0",
        ]
        >=
        0.90
        and
        b_boot.loc[
            "macro_f1",
            "fraction_delta_gt_0",
        ]
        >=
        0.90
    )

    summary.to_csv(
        OUT
        / "08F_adaptive_gate_summary.csv",
        index=False,
    )

    alpha_df.to_csv(
        OUT
        / "08F_alpha_distribution.csv",
        index=False,
    )

    bootstrap.to_csv(
        OUT
        / "08F_day_block_bootstrap.csv",
        index=False,
    )

    parameters = {
        "feature_names":
            feature_names,

        "mu":
            final_params[
                "mu"
            ].tolist(),

        "sigma":
            final_params[
                "sigma"
            ].tolist(),

        "weights":
            final_params[
                "w"
            ].tolist(),

        "bias":
            final_params[
                "b"
            ],

        "l2":
            L2,

        "reference_alpha":
            REFERENCE_ALPHA,
    }

    write_json(
        OUT
        / "08F_gate_parameters.json",
        parameters,
    )

    manifest = {
        "stage":
            "08F_adaptive_alpha_gate",

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "git_commit":
            git_commit(),

        "reference":
            "LTD-HYBRID-V1",

        "gate": {
            "type":
                "logistic scalar-alpha gate",

            "class_identity_features":
                False,

            "feature_count":
                len(
                    feature_names
                ),

            "features":
                feature_names,

            "objective":
                "cross-entropy of fused class distribution",

            "regularization_l2":
                L2,

            "initial_alpha":
                REFERENCE_ALPHA,
        },

        "protocol": {
            "inner_fold_a_train_dates":
                [
                    str(x)
                    for x in
                    gate_train_dates
                ],

            "inner_fold_a_validation_dates":
                [
                    str(x)
                    for x in
                    gate_val_dates
                ],

            "final_gate_fit":
                "all Fold-A predictions",

            "fold_b_used_for_gate_training":
                False,

            "fold_b_used_for_feature_selection":
                False,

            "manual_site_rules":
                False,
        },

        "inner_validation":
            {
                "fixed":
                    fixed_val_m,

                "adaptive":
                    adaptive_val_m,

                "delta":
                    inner_delta,
            },

        "data_policy": {
            "development_only":
                True,

            "internal_test_values_used":
                False,

            "external_future_used":
                False,
        },

        "promotion_gate": {
            "passed":
                promotion_pass,

            "requirements": [
                "Inner Fold-A validation Accuracy delta >= 0",
                "Inner Fold-A validation Macro-F1 delta > 0",
                "Fold-B Accuracy > LTD-HYBRID-V1",
                "Fold-B Macro-F1 > LTD-HYBRID-V1",
                "Fold-B Top-5 >= LTD-HYBRID-V1",
                "Fold-B MRR > LTD-HYBRID-V1",
                "Fold-B bootstrap positive fraction Accuracy >= 0.90",
                "Fold-B bootstrap positive fraction Macro-F1 >= 0.90",
            ],
        },

        "inputs":
            input_hashes,

        "next":
            (
                "08G final candidate-level learned reranker; "
                "predeclared before observing 08F Fold-B result"
            ),
    }

    write_json(
        OUT
        / "08F_manifest.json",
        manifest,
    )

    print()
    print("=" * 78)
    print(
        "08F SUMMARY"
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
        "PROMOTION PASS:",
        promotion_pass
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
