from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd

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

EPS = 1e-12

L2 = 1e-3
MAX_LBFGS_ITER = 300

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
        "ltd_08d_for_08g",
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


def rank_matrix(
    probs,
):

    order = np.argsort(
        -probs,
        axis=1,
    )

    ranks = np.empty_like(
        order
    )

    n, c = probs.shape

    ranks[
        np.arange(
            n
        )[
            :,
            None
        ],
        order
    ] = np.arange(
        1,
        c + 1,
    )[
        None,
        :
    ]

    return ranks.astype(
        np.float64
    )


def relative_representation(
    probs,
):

    logp = np.log(
        np.clip(
            probs,
            EPS,
            1.0,
        )
    )

    gap = (
        logp
        -
        logp.max(
            axis=1,
            keepdims=True,
        )
    )

    mean = logp.mean(
        axis=1,
        keepdims=True,
    )

    std = logp.std(
        axis=1,
        keepdims=True,
    )

    std = np.maximum(
        std,
        1e-8,
    )

    z = (
        logp
        -
        mean
    ) / std

    ranks = rank_matrix(
        probs
    )

    rr = (
        1.0
        /
        ranks
    )

    return {
        "logp":
            logp,

        "gap":
            gap,

        "z":
            z,

        "rank":
            ranks,

        "rr":
            rr,

        "top1":
            (
                ranks
                <= 1
            ).astype(
                np.float64
            ),

        "top3":
            (
                ranks
                <= 3
            ).astype(
                np.float64
            ),

        "top5":
            (
                ranks
                <= 5
            ).astype(
                np.float64
            ),
    }


def candidate_features(
    micro,
    macro,
    fixed,
):

    m = relative_representation(
        micro
    )

    a = relative_representation(
        macro
    )

    h = relative_representation(
        fixed
    )

    rr_product = (
        m[
            "rr"
        ]
        *
        a[
            "rr"
        ]
    )

    rr_absdiff = np.abs(
        m[
            "rr"
        ]
        -
        a[
            "rr"
        ]
    )

    rank_agreement = (
        1.0
        -
        np.abs(
            m[
                "rank"
            ]
            -
            a[
                "rank"
            ]
        )
        /
        (
            micro.shape[
                1
            ]
            -
            1
        )
    )

    z_sum = (
        m[
            "z"
        ]
        +
        a[
            "z"
        ]
    )

    z_absdiff = np.abs(
        m[
            "z"
        ]
        -
        a[
            "z"
        ]
    )

    both_top5 = (
        m[
            "top5"
        ]
        *
        a[
            "top5"
        ]
    )

    names = [
        "micro_log_gap",
        "macro_log_gap",
        "fixed_log_gap",
        "micro_z",
        "macro_z",
        "fixed_z",
        "micro_reciprocal_rank",
        "macro_reciprocal_rank",
        "fixed_reciprocal_rank",
        "micro_top1",
        "macro_top1",
        "fixed_top1",
        "micro_top3",
        "macro_top3",
        "fixed_top3",
        "micro_top5",
        "macro_top5",
        "fixed_top5",
        "both_branch_top5",
        "reciprocal_rank_product",
        "reciprocal_rank_absdiff",
        "rank_agreement",
        "z_sum",
        "z_absdiff",
    ]

    x = np.stack(
        [
            m[
                "gap"
            ],
            a[
                "gap"
            ],
            h[
                "gap"
            ],
            m[
                "z"
            ],
            a[
                "z"
            ],
            h[
                "z"
            ],
            m[
                "rr"
            ],
            a[
                "rr"
            ],
            h[
                "rr"
            ],
            m[
                "top1"
            ],
            a[
                "top1"
            ],
            h[
                "top1"
            ],
            m[
                "top3"
            ],
            a[
                "top3"
            ],
            h[
                "top3"
            ],
            m[
                "top5"
            ],
            a[
                "top5"
            ],
            h[
                "top5"
            ],
            both_top5,
            rr_product,
            rr_absdiff,
            rank_agreement,
            z_sum,
            z_absdiff,
        ],
        axis=-1,
    ).astype(
        np.float64
    )

    return (
        x,
        names,
    )


def fit_scaler(
    x,
):

    flat = x.reshape(
        -1,
        x.shape[
            -1
        ],
    )

    mu = flat.mean(
        axis=0
    )

    sigma = flat.std(
        axis=0
    )

    sigma[
        sigma
        <
        1e-8
    ] = 1.0

    return (
        mu,
        sigma,
    )


def standardize(
    x,
    mu,
    sigma,
):

    return (
        x
        -
        mu[
            None,
            None,
            :
        ]
    ) / sigma[
        None,
        None,
        :
    ]


def fit_reranker(
    x,
    fixed,
    y,
):

    mu, sigma = fit_scaler(
        x
    )

    z = standardize(
        x,
        mu,
        sigma,
    )

    z_t = torch.tensor(
        z,
        dtype=torch.float64,
    )

    base_t = torch.tensor(
        np.log(
            np.clip(
                fixed,
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
                -1
            ],
            dtype=torch.float64,
        )
    )

    optimizer = torch.optim.LBFGS(
        [
            w,
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

        residual = torch.einsum(
            "ncf,f->nc",
            z_t,
            w,
        )

        logits = (
            base_t
            +
            residual
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

    objective = optimizer.step(
        closure
    )

    return {
        "mu":
            mu,

        "sigma":
            sigma,

        "weights":
            w.detach()
            .cpu()
            .numpy(),

        "objective":
            float(
                objective
                .detach()
                .cpu()
                .item()
            ),
    }


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


def apply_reranker(
    params,
    x,
    fixed,
):

    z = standardize(
        x,
        params[
            "mu"
        ],
        params[
            "sigma"
        ],
    )

    residual = np.einsum(
        "ncf,f->nc",
        z,
        params[
            "weights"
        ],
    )

    logits = (
        np.log(
            np.clip(
                fixed,
                EPS,
                1.0,
            )
        )
        +
        residual
    )

    return (
        softmax(
            logits
        ),
        residual,
    )


def metric_delta(
    candidate,
    reference,
):

    return {
        metric:
            candidate[
                metric
            ]
            -
            reference[
                metric
            ]
        for metric in [
            "accuracy",
            "macro_f1",
            "top5_accuracy",
            "mrr",
        ]
    }


def main():

    print("=" * 78)
    print(
        "08G — RESIDUAL CANDIDATE-LEVEL RERANKER"
    )
    print("=" * 78)

    print(
        "Final planned DEV hybrid architecture experiment."
    )

    print(
        "Reference: LTD-HYBRID-V1."
    )

    print(
        "No candidate/site identity features."
    )

    print(
        "Relative/rank-based candidate features only."
    )

    print(
        "Fold A only for learning."
    )

    print(
        "Fold B temporal transfer only."
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

    x_a, feature_names = (
        candidate_features(
            a[
                "micro"
            ],
            a[
                "macro"
            ],
            a[
                "fixed"
            ],
        )
    )

    dates_a = mod08d.load_dates(
        a[
            "pcap_uid"
        ]
    )

    unique_dates = np.array(
        sorted(
            np.unique(
                dates_a
            )
        )
    )

    midpoint = (
        len(
            unique_dates
        )
        //
        2
    )

    inner_train_dates = (
        unique_dates[
            :midpoint
        ]
    )

    inner_val_dates = (
        unique_dates[
            midpoint:
        ]
    )

    train_mask = np.isin(
        dates_a,
        inner_train_dates,
    )

    val_mask = np.isin(
        dates_a,
        inner_val_dates,
    )

    inner_params = fit_reranker(
        x_a[
            train_mask
        ],
        a[
            "fixed"
        ][
            train_mask
        ],
        a[
            "y"
        ][
            train_mask
        ],
    )

    inner_probs, _ = apply_reranker(
        inner_params,
        x_a[
            val_mask
        ],
        a[
            "fixed"
        ][
            val_mask
        ],
    )

    fixed_inner_m, _, _ = (
        mod08d.evaluate(
            a[
                "fixed"
            ][
                val_mask
            ],
            a[
                "y"
            ][
                val_mask
            ],
        )
    )

    rerank_inner_m, _, _ = (
        mod08d.evaluate(
            inner_probs,
            a[
                "y"
            ][
                val_mask
            ],
        )
    )

    inner_delta = metric_delta(
        rerank_inner_m,
        fixed_inner_m,
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
            inner_train_dates
        ],
    )

    print(
        "validation dates:",
        [
            str(x)
            for x in
            inner_val_dates
        ],
    )

    print(
        "fixed:",
        fixed_inner_m
    )

    print(
        "reranker:",
        rerank_inner_m
    )

    print(
        "delta:",
        inner_delta
    )

    final_params = fit_reranker(
        x_a,
        a[
            "fixed"
        ],
        a[
            "y"
        ],
    )

    summary_rows = []
    bootstrap_rows = []
    day_rows = []
    class_rows = []

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

        x, _ = candidate_features(
            d[
                "micro"
            ],
            d[
                "macro"
            ],
            d[
                "fixed"
            ],
        )

        rerank_probs, residual = (
            apply_reranker(
                final_params,
                x,
                d[
                    "fixed"
                ],
            )
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

        rerank_m, rerank_pred, rerank_rank = (
            mod08d.evaluate(
                rerank_probs,
                d[
                    "y"
                ],
            )
        )

        dd = metric_delta(
            rerank_m,
            fixed_m,
        )

        y = d[
            "y"
        ]

        fixed_correct = (
            fixed_pred
            ==
            y
        )

        rerank_correct = (
            rerank_pred
            ==
            y
        )

        micro_pred = np.argmax(
            d[
                "micro"
            ],
            axis=1,
        )

        macro_pred = np.argmax(
            d[
                "macro"
            ],
            axis=1,
        )

        both_branches_wrong = (
            (micro_pred != y)
            &
            (macro_pred != y)
        )

        reranker_only = (
            ~fixed_correct
            &
            rerank_correct
        )

        fixed_only = (
            fixed_correct
            &
            ~rerank_correct
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
                    f"reranker_{k}":
                        v
                    for k, v
                    in rerank_m.items()
                },

                **{
                    f"delta_{k}":
                        v
                    for k, v
                    in dd.items()
                },

                "reranker_only_correct_n":
                    int(
                        reranker_only.sum()
                    ),

                "fixed_only_correct_n":
                    int(
                        fixed_only.sum()
                    ),

                "net_correct_n":
                    int(
                        reranker_only.sum()
                        -
                        fixed_only.sum()
                    ),

                "reranker_correct_when_both_branches_top1_wrong_n":
                    int(
                        (
                            rerank_correct
                            &
                            both_branches_wrong
                        ).sum()
                    ),

                "fixed_correct_when_both_branches_top1_wrong_n":
                    int(
                        (
                            fixed_correct
                            &
                            both_branches_wrong
                        ).sum()
                    ),

                "residual_abs_mean":
                    float(
                        np.mean(
                            np.abs(
                                residual
                            )
                        )
                    ),

                "residual_abs_q95":
                    float(
                        np.quantile(
                            np.abs(
                                residual
                            ),
                            0.95,
                        )
                    ),
            }
        )

        dates = mod08d.load_dates(
            d[
                "pcap_uid"
            ]
        )

        if fold == FOLD_B:

            boot = mod08d.bootstrap_dates(
                y,
                dates,
                fixed_pred,
                fixed_rank,
                rerank_pred,
                rerank_rank,
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

                        "observed_reranker_minus_fixed":
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

            b_reference = fixed_m
            b_candidate = rerank_m

        for date in sorted(
            np.unique(
                dates
            )
        ):

            idx = np.flatnonzero(
                dates == date
            )

            fm, _, _ = (
                mod08d.evaluate(
                    d[
                        "fixed"
                    ][
                        idx
                    ],
                    y[
                        idx
                    ],
                )
            )

            rm, _, _ = (
                mod08d.evaluate(
                    rerank_probs[
                        idx
                    ],
                    y[
                        idx
                    ],
                )
            )

            day_rows.append(
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
                            rm[
                                k
                            ]
                            -
                            fm[
                                k
                            ]
                        for k in [
                            "accuracy",
                            "macro_f1",
                            "top5_accuracy",
                            "mrr",
                        ]
                    },
                }
            )

        labels = d[
            "labels"
        ]

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

            fc = fixed_correct[
                idx
            ]

            rc = rerank_correct[
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

                    "fixed_accuracy":
                        float(
                            fc.mean()
                        ),

                    "reranker_accuracy":
                        float(
                            rc.mean()
                        ),

                    "delta_accuracy":
                        float(
                            rc.mean()
                            -
                            fc.mean()
                        ),

                    "reranker_only_correct_n":
                        int(
                            (
                                ~fc
                                &
                                rc
                            ).sum()
                        ),

                    "fixed_only_correct_n":
                        int(
                            (
                                fc
                                &
                                ~rc
                            ).sum()
                        ),
                }
            )

        np.savez_compressed(
            OUT
            / f"08G_scores_{fold}.npz",

            pcap_uid=
                d[
                    "pcap_uid"
                ].astype(str),

            y_true=
                y,

            candidate_labels=
                labels.astype(str),

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

            reranker_probs=
                rerank_probs.astype(
                    np.float32
                ),
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

    bootstrap = pd.DataFrame(
        bootstrap_rows
    )

    per_day = pd.DataFrame(
        day_rows
    )

    per_class = pd.DataFrame(
        class_rows
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
        ] > 0
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
        (
            b_candidate[
                "top5_accuracy"
            ]
            -
            b_reference[
                "top5_accuracy"
            ]
        )
        >=
        -0.001
        and
        b_candidate[
            "mrr"
        ]
        >=
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
        / "08G_reranker_summary.csv",
        index=False,
    )

    bootstrap.to_csv(
        OUT
        / "08G_day_block_bootstrap.csv",
        index=False,
    )

    per_day.to_csv(
        OUT
        / "08G_per_day_deltas.csv",
        index=False,
    )

    per_class.to_csv(
        OUT
        / "08G_per_class_deltas.csv",
        index=False,
    )

    params_json = {
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
                "weights"
            ].tolist(),

        "objective":
            final_params[
                "objective"
            ],

        "l2":
            L2,

        "base_score":
            "log LTD-HYBRID-V1 probability",
    }

    write_json(
        OUT
        / "08G_reranker_parameters.json",
        params_json,
    )

    manifest = {
        "stage":
            "08G_residual_candidate_reranker",

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "git_commit":
            git_commit(),

        "reference":
            "LTD-HYBRID-V1",

        "model": {
            "type":
                "shared linear residual candidate reranker",

            "candidate_identity_features":
                False,

            "site_specific_parameters":
                False,

            "manual_site_rules":
                False,

            "feature_count":
                len(
                    feature_names
                ),

            "features":
                feature_names,

            "base_score":
                "log probability from LTD-HYBRID-V1",

            "objective":
                "listwise 65-class cross-entropy",

            "regularization_l2":
                L2,

            "hyperparameter_search":
                False,
        },

        "protocol": {
            "inner_fold_a_train_dates":
                [
                    str(x)
                    for x in
                    inner_train_dates
                ],

            "inner_fold_a_validation_dates":
                [
                    str(x)
                    for x in
                    inner_val_dates
                ],

            "final_reranker_fit":
                "all Fold-A predictions",

            "fold_b_used_for_training":
                False,

            "fold_b_used_for_feature_selection":
                False,
        },

        "inner_validation": {
            "fixed":
                fixed_inner_m,

            "reranker":
                rerank_inner_m,

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
                "Inner Fold-A validation Accuracy delta > 0",
                "Inner Fold-A validation Macro-F1 delta > 0",
                "Fold-B Accuracy > LTD-HYBRID-V1",
                "Fold-B Macro-F1 > LTD-HYBRID-V1",
                "Fold-B Top-5 delta >= -0.001",
                "Fold-B MRR >= LTD-HYBRID-V1",
                "Fold-B bootstrap positive fraction Accuracy >= 0.90",
                "Fold-B bootstrap positive fraction Macro-F1 >= 0.90",
            ],
        },

        "stop_rule": {
            "if_pass":
                "promote reranker candidate and freeze HYBRID-FINAL",

            "if_fail":
                "LTD-HYBRID-V1 becomes HYBRID-FINAL",

            "additional_dev_hybrid_architecture_search":
                False,
        },

        "inputs":
            input_hashes,
    }

    write_json(
        OUT
        / "08G_manifest.json",
        manifest,
    )

    print()
    print("=" * 78)
    print(
        "08G SUMMARY"
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
