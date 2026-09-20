from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import importlib.util
import time

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
)

import torch

from _common import (
    git_commit,
    load_historical_65,
    output_dir,
    sha256,
    write_json,
)


FOLD_A = "A_EARLY_TO_MIDDLE"
FOLD_B = "B_EARLY_MIDDLE_TO_LATE"

CONTEXT_WINDOWS = [
    1,
    3,
    5,
    7,
]

ALPHA = 0.375
EPS = 1e-12


def load_07a():
    path = (
        Path(__file__).resolve().parent
        / "07A_candidate_conditioned_temporal_encoder.py"
    )

    spec = importlib.util.spec_from_file_location(
        "macro_v2_07a_context_ablation",
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Could not load 07A module."
        )

    mod = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        mod
    )

    return mod, path


def softmax_numpy(logits):
    z = (
        logits
        -
        logits.max(
            axis=1,
            keepdims=True,
        )
    )

    e = np.exp(z)

    return (
        e
        /
        e.sum(
            axis=1,
            keepdims=True,
        )
    )


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
        /
        p.sum(
            axis=1,
            keepdims=True,
        )
    )


def evaluate_probs(
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
                ranks.mean()
            ),
    }


def predict_ltd(
    mod,
    model,
    x,
    dates,
    context_cache,
    device,
):
    output = np.full(
        (
            len(x),
            65,
        ),
        np.nan,
        dtype=np.float32,
    )

    model.eval()

    with torch.no_grad():

        for date in sorted(
            pd.unique(
                pd.to_datetime(
                    dates
                )
            )
        ):

            date = pd.Timestamp(
                date
            )

            if date not in context_cache:
                raise RuntimeError(
                    f"No context for {date}"
                )

            idx = np.flatnonzero(
                pd.to_datetime(
                    dates
                )
                == date
            )

            (
                context_np,
                padding_np,
                _,
            ) = context_cache[
                date
            ]

            context = (
                torch.from_numpy(
                    context_np
                )
                .to(device)
            )

            padding = (
                torch.from_numpy(
                    padding_np
                )
                .to(device)
            )

            for start in range(
                0,
                len(idx),
                mod.BATCH_SIZE,
            ):

                batch_idx = idx[
                    start:
                    start
                    + mod.BATCH_SIZE
                ]

                query = (
                    torch.from_numpy(
                        x[
                            batch_idx
                        ]
                    )
                    .to(device)
                )

                logits = model(
                    query,
                    context,
                    padding,
                )

                output[
                    batch_idx
                ] = (
                    logits
                    .cpu()
                    .numpy()
                )

    if np.isnan(
        output
    ).any():
        raise RuntimeError(
            "Incomplete LTD predictions."
        )

    return output


def prepare_fold(
    mod,
    captures,
    daily,
    features,
    fold_spec,
):
    train_splits = (
        fold_spec[
            "train_splits"
        ]
    )

    test_split = (
        fold_spec[
            "test_split"
        ]
    )

    train = (
        captures[
            captures[
                "temporal_split"
            ].isin(
                train_splits
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    test = (
        captures[
            captures[
                "temporal_split"
            ]
            == test_split
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    train_daily = (
        daily[
            daily[
                "temporal_split"
            ].isin(
                train_splits
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    candidate_labels = np.array(
        sorted(
            captures[
                "site_label"
            ].unique()
        ),
        dtype=str,
    )

    label_to_idx = (
        mod.make_label_mapping(
            candidate_labels
        )
    )

    y_train = (
        mod.labels_to_indices(
            train[
                "site_label"
            ],
            label_to_idx,
        )
    )

    y_test = (
        mod.labels_to_indices(
            test[
                "site_label"
            ],
            label_to_idx,
        )
    )

    (
        medians,
        scaler,
        missing,
    ) = mod.fit_preprocessing(
        train,
        features,
    )

    if missing:
        raise RuntimeError(
            f"All-missing features: {missing}"
        )

    x_train = mod.transform(
        train,
        features,
        medians,
        scaler,
    )

    x_test = mod.transform(
        test,
        features,
        medians,
        scaler,
    )

    train_dates = (
        pd.to_datetime(
            train[
                "_query_date"
            ]
        )
        .to_numpy()
    )

    test_dates = (
        pd.to_datetime(
            test[
                "_query_date"
            ]
        )
        .to_numpy()
    )

    daily_history = (
        mod.standardized_daily_frame(
            train_daily,
            features,
            medians,
            scaler,
        )
    )

    return {
        "train":
            train,

        "test":
            test,

        "train_daily":
            train_daily,

        "candidate_labels":
            candidate_labels,

        "y_train":
            y_train,

        "y_test":
            y_test,

        "x_train":
            x_train,

        "x_test":
            x_test,

        "train_dates":
            train_dates,

        "test_dates":
            test_dates,

        "daily_history":
            daily_history,
    }


def run_window(
    mod,
    fold_data,
    features,
    context_days,
    xgb_probs,
    device,
):
    # 07A construction and positional encoding
    # both use this predeclared module-level value.
    mod.CONTEXT_DAYS = int(
        context_days
    )

    (
        train_context,
        skipped_train,
    ) = mod.build_context_cache(
        fold_data[
            "daily_history"
        ],
        fold_data[
            "candidate_labels"
        ],
        fold_data[
            "train_dates"
        ],
        features,
    )

    (
        test_context,
        skipped_test,
    ) = mod.build_context_cache(
        fold_data[
            "daily_history"
        ],
        fold_data[
            "candidate_labels"
        ],
        fold_data[
            "test_dates"
        ],
        features,
    )

    if skipped_test:
        raise RuntimeError(
            f"W={context_days}: "
            f"missing test histories {skipped_test}"
        )

    seed_probs = []

    seed_rows = []

    for seed in mod.SEEDS:

        mod.set_seed(
            seed
        )

        model = (
            mod.LTDPairScorer()
            .to(device)
        )

        t0 = time.perf_counter()

        (
            train_loss,
            train_n,
            train_dates_n,
        ) = mod.train_ltd_model(
            model,
            fold_data[
                "x_train"
            ],
            fold_data[
                "y_train"
            ],
            fold_data[
                "train_dates"
            ],
            train_context,
            device,
            seed,
        )

        fit_seconds = (
            time.perf_counter()
            - t0
        )

        logits = predict_ltd(
            mod,
            model,
            fold_data[
                "x_test"
            ],
            fold_data[
                "test_dates"
            ],
            test_context,
            device,
        )

        probs = (
            softmax_numpy(
                logits
            )
            .astype(
                np.float32
            )
        )

        m = evaluate_probs(
            probs,
            fold_data[
                "y_test"
            ],
        )

        seed_rows.append(
            {
                "context_days":
                    context_days,

                "seed":
                    seed,

                "train_loss":
                    train_loss,

                "n_train_queries":
                    train_n,

                "n_train_context_dates":
                    train_dates_n,

                "fit_seconds":
                    fit_seconds,

                **m,
            }
        )

        seed_probs.append(
            probs
        )

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ensemble = (
        np.stack(
            seed_probs,
            axis=0,
        )
        .mean(
            axis=0
        )
    )

    ensemble /= ensemble.sum(
        axis=1,
        keepdims=True,
    )

    ltd_metrics = evaluate_probs(
        ensemble,
        fold_data[
            "y_test"
        ],
    )

    fused = fuse(
        xgb_probs,
        ensemble,
    )

    fusion_metrics = evaluate_probs(
        fused,
        fold_data[
            "y_test"
        ],
    )

    return {
        "context_days":
            context_days,

        "skipped_train_dates":
            len(
                skipped_train
            ),

        "ensemble_probs":
            ensemble,

        "ltd_metrics":
            ltd_metrics,

        "fusion_metrics":
            fusion_metrics,

        "seed_rows":
            seed_rows,
    }


def main():

    print("=" * 78)
    print(
        "07D-1 — LTD CONTEXT-LENGTH ABLATION"
    )
    print("=" * 78)

    print(
        "Candidate windows:",
        CONTEXT_WINDOWS,
    )

    print(
        "All windows evaluated only on Fold A."
    )

    print(
        "Only selected window transferred to Fold B."
    )

    print(
        "Fusion alpha remains frozen:",
        ALPHA,
    )

    print(
        "INTERNAL_TEST: CLOSED"
    )

    print(
        "Future-B: CLOSED"
    )

    mod, source07a = (
        load_07a()
    )

    (
        features,
        frozen_path,
    ) = mod.load_frozen_features()

    out = output_dir()

    daily_path = (
        out
        / "06C_daily_base_profiles.csv"
    )

    assignment_path = (
        out
        / "01_temporal_split_assignments.csv"
    )

    daily = pd.read_csv(
        daily_path
    )

    daily[
        "site_label"
    ] = (
        daily[
            "site_label"
        ].astype(str)
    )

    daily[
        "date"
    ] = pd.to_datetime(
        daily[
            "date"
        ]
    )

    (
        captures,
        historical_path,
    ) = load_historical_65()

    assignments = pd.read_csv(
        assignment_path
    )

    captures = captures.merge(
        assignments[
            [
                "pcap_uid",
                "temporal_split",
            ]
        ],
        on="pcap_uid",
        how="inner",
        validate="one_to_one",
    )

    captures[
        "site_label"
    ] = (
        captures[
            "site_label"
        ].astype(str)
    )

    dates, _ = mod.derive_dates(
        captures
    )

    captures[
        "_query_date"
    ] = pd.to_datetime(
        dates
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

    fold_specs = {
        f[
            "fold"
        ]: f
        for f in mod.FOLDS
    }

    fold_a = prepare_fold(
        mod,
        captures,
        daily,
        features,
        fold_specs[
            FOLD_A
        ],
    )

    score_a_path = (
        out
        / f"07B_01_scores_{FOLD_A}.npz"
    )

    score_a = np.load(
        score_a_path,
        allow_pickle=True,
    )

    xgb_a = (
        score_a[
            "xgb_probs"
        ]
        .astype(
            np.float64
        )
    )

    curve_rows = []
    seed_rows = []
    fold_a_outputs = {}

    for window in CONTEXT_WINDOWS:

        print()
        print(
            f"Fold A — context={window}"
        )

        result = run_window(
            mod,
            fold_a,
            features,
            window,
            xgb_a,
            device,
        )

        fold_a_outputs[
            window
        ] = result

        row = {
            "context_days":
                window,

            "skipped_train_dates":
                result[
                    "skipped_train_dates"
                ],

            **{
                f"ltd_{k}": v
                for k, v in
                result[
                    "ltd_metrics"
                ].items()
            },

            **{
                f"fusion_{k}": v
                for k, v in
                result[
                    "fusion_metrics"
                ].items()
            },
        }

        curve_rows.append(
            row
        )

        for r in result[
            "seed_rows"
        ]:
            seed_rows.append(
                {
                    "fold":
                        FOLD_A,
                    **r,
                }
            )

        print(
            "LTD "
            f"F1={result['ltd_metrics']['macro_f1']:.4f} "
            "Fusion "
            f"F1={result['fusion_metrics']['macro_f1']:.4f}"
        )

    curve = pd.DataFrame(
        curve_rows
    )

    # Primary:
    # fused Fold-A Macro-F1.
    #
    # Secondary:
    # fused Accuracy.
    #
    # Tie:
    # smaller window, favoring simpler history.
    selected = (
        curve
        .sort_values(
            [
                "fusion_macro_f1",
                "fusion_accuracy",
                "context_days",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )
        .iloc[0]
    )

    selected_window = int(
        selected[
            "context_days"
        ]
    )

    print()
    print(
        "Selected context on Fold A:",
        selected_window,
    )

    # --------------------------------------------------
    # Transfer exactly one selected window to Fold B.
    # --------------------------------------------------

    fold_b = prepare_fold(
        mod,
        captures,
        daily,
        features,
        fold_specs[
            FOLD_B
        ],
    )

    score_b_path = (
        out
        / f"07B_01_scores_{FOLD_B}.npz"
    )

    score_b = np.load(
        score_b_path,
        allow_pickle=True,
    )

    xgb_b = (
        score_b[
            "xgb_probs"
        ]
        .astype(
            np.float64
        )
    )

    print()
    print(
        "Fold B transfer — context=",
        selected_window,
    )

    transfer = run_window(
        mod,
        fold_b,
        features,
        selected_window,
        xgb_b,
        device,
    )

    for r in transfer[
        "seed_rows"
    ]:
        seed_rows.append(
            {
                "fold":
                    FOLD_B,
                **r,
            }
        )

    v1_a = fold_a_outputs[
        7
    ][
        "fusion_metrics"
    ]

    selected_a = fold_a_outputs[
        selected_window
    ][
        "fusion_metrics"
    ]

    # Existing V1 Fold-B scores are used only as
    # immutable reference, not for selection.
    v1_summary_path = (
        out
        / "07B_02_fusion_summary.csv"
    )

    v1_summary = pd.read_csv(
        v1_summary_path
    )

    v1_b_row = (
        v1_summary[
            (
                v1_summary[
                    "fold"
                ]
                == FOLD_B
            )
            &
            (
                v1_summary[
                    "model"
                ]
                == "fusion"
            )
        ]
        .iloc[0]
    )

    comparison = pd.DataFrame(
        [
            {
                "fold":
                    FOLD_A,

                "selected_context_days":
                    selected_window,

                "v1_context_days":
                    7,

                "delta_accuracy_vs_v1":
                    selected_a[
                        "accuracy"
                    ]
                    -
                    v1_a[
                        "accuracy"
                    ],

                "delta_macro_f1_vs_v1":
                    selected_a[
                        "macro_f1"
                    ]
                    -
                    v1_a[
                        "macro_f1"
                    ],

                "delta_top5_vs_v1":
                    selected_a[
                        "top5_accuracy"
                    ]
                    -
                    v1_a[
                        "top5_accuracy"
                    ],

                "delta_mrr_vs_v1":
                    selected_a[
                        "mrr"
                    ]
                    -
                    v1_a[
                        "mrr"
                    ],
            },

            {
                "fold":
                    FOLD_B,

                "selected_context_days":
                    selected_window,

                "v1_context_days":
                    7,

                "delta_accuracy_vs_v1":
                    transfer[
                        "fusion_metrics"
                    ][
                        "accuracy"
                    ]
                    -
                    float(
                        v1_b_row[
                            "accuracy"
                        ]
                    ),

                "delta_macro_f1_vs_v1":
                    transfer[
                        "fusion_metrics"
                    ][
                        "macro_f1"
                    ]
                    -
                    float(
                        v1_b_row[
                            "macro_f1"
                        ]
                    ),

                "delta_top5_vs_v1":
                    transfer[
                        "fusion_metrics"
                    ][
                        "top5_accuracy"
                    ]
                    -
                    float(
                        v1_b_row[
                            "top5_accuracy"
                        ]
                    ),

                "delta_mrr_vs_v1":
                    transfer[
                        "fusion_metrics"
                    ][
                        "mrr"
                    ]
                    -
                    float(
                        v1_b_row[
                            "mrr"
                        ]
                    ),
            },
        ]
    )

    a_cmp = comparison.iloc[
        0
    ]

    b_cmp = comparison.iloc[
        1
    ]

    context_pass = bool(
        selected_window != 7
        and
        a_cmp[
            "delta_accuracy_vs_v1"
        ]
        >= 0
        and
        a_cmp[
            "delta_macro_f1_vs_v1"
        ]
        > 0
        and
        b_cmp[
            "delta_accuracy_vs_v1"
        ]
        >= 0
        and
        b_cmp[
            "delta_macro_f1_vs_v1"
        ]
        > 0
    )

    curve_path = (
        out
        / "07D_01_context_curve_fold_A.csv"
    )

    seed_path = (
        out
        / "07D_01_seed_metrics.csv"
    )

    comparison_path = (
        out
        / "07D_01_selected_context_vs_v1.csv"
    )

    transfer_path = (
        out
        / "07D_01_fold_B_transfer.csv"
    )

    curve.to_csv(
        curve_path,
        index=False,
    )

    pd.DataFrame(
        seed_rows
    ).to_csv(
        seed_path,
        index=False,
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    pd.DataFrame(
        [
            {
                "selected_context_days":
                    selected_window,

                **{
                    f"ltd_{k}": v
                    for k, v in
                    transfer[
                        "ltd_metrics"
                    ].items()
                },

                **{
                    f"fusion_{k}": v
                    for k, v in
                    transfer[
                        "fusion_metrics"
                    ].items()
                },
            }
        ]
    ).to_csv(
        transfer_path,
        index=False,
    )

    manifest = {
        "stage":
            "07D_01_context_length_ablation",

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

            "context_selected_on":
                FOLD_A,

            "fold_b_used_for_context_selection":
                False,

            "all_candidate_contexts_evaluated_on_fold_b":
                False,

            "alpha_tuning":
                False,
        },

        "candidates":
            CONTEXT_WINDOWS,

        "selected_context_days":
            selected_window,

        "frozen_fusion_alpha":
            ALPHA,

        "selection": {
            "primary":
                "Fold-A fused Macro-F1",

            "secondary":
                "Fold-A fused Accuracy",

            "tie_break":
                "smaller context",
        },

        "v2_context_gate": {
            "passed":
                context_pass,

            "requirements": [
                "selected context differs from V1 context=7",
                "Fold-A Accuracy delta >= 0",
                "Fold-A Macro-F1 delta > 0",
                "Fold-B Accuracy delta >= 0",
                "Fold-B Macro-F1 delta > 0",
            ],
        },

        "inputs": {
            "07A_source_sha256":
                sha256(
                    source07a
                ),

            "frozen_features_sha256":
                sha256(
                    frozen_path
                ),

            "historical_sha256":
                sha256(
                    historical_path
                ),

            "daily_profiles_sha256":
                sha256(
                    daily_path
                ),

            "fold_a_score_sha256":
                sha256(
                    score_a_path
                ),

            "fold_b_score_sha256":
                sha256(
                    score_b_path
                ),
        },

        "next_if_pass":
            "07D-1R robustness audit before MACRO-LTD-V2",

        "next_if_fail":
            "retain MACRO-LTD-V1 and move to temporal-order/model ablation",
    }

    manifest_path = (
        out
        / "07D_01_manifest_context_ablation.json"
    )

    write_json(
        manifest_path,
        manifest,
    )

    print()
    print("=" * 78)
    print(
        "FOLD-A CONTEXT CURVE"
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
        "SELECTED CONTEXT VS V1"
    )
    print("=" * 78)

    print(
        comparison.to_string(
            index=False
        )
    )

    print()
    print(
        "CONTEXT V2 PASS:",
        context_pass,
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
