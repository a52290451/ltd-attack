from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import importlib.util
import time

import numpy as np
import pandas as pd

import sklearn
from sklearn.metrics import accuracy_score, f1_score

import torch
import xgboost as xgb

from _common import (
    git_commit,
    load_historical_65,
    output_dir,
    sha256,
    write_json,
)


STAGE = "07B_01_complementarity_audit"

XGB_CONFIG = {
    "n_estimators": 120,
    "max_depth": 5,
    "learning_rate": 0.10,
    "min_child_weight": 1,
    "subsample": 0.80,
    "colsample_bytree": 0.80,
    "reg_lambda": 1.0,
    "objective": "multi:softprob",
    "num_class": 65,
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
}


def load_07a():
    path = (
        Path(__file__).resolve().parent
        / "07A_candidate_conditioned_temporal_encoder.py"
    )

    spec = importlib.util.spec_from_file_location(
        "macro_v2_07a",
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load 07A module.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module, path


def numeric_frame(df, features):
    return (
        df[features]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )


def fit_xgb_preprocessing(df, features):
    x = numeric_frame(df, features)

    medians = x.median(axis=0, skipna=True)

    all_missing = (
        medians[
            medians.isna()
        ]
        .index
        .tolist()
    )

    medians = medians.fillna(0.0)

    x = (
        x
        .fillna(medians)
        .to_numpy(dtype=np.float32)
    )

    return medians, x, all_missing


def transform_xgb(
    df,
    features,
    medians,
):
    x = (
        numeric_frame(df, features)
        .fillna(medians)
        .to_numpy(dtype=np.float32)
    )

    if not np.isfinite(x).all():
        raise RuntimeError(
            "Non-finite XGB values."
        )

    return x


def softmax_numpy(logits):
    z = logits - logits.max(
        axis=1,
        keepdims=True,
    )

    e = np.exp(z)

    return (
        e
        / e.sum(
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

    true_prob = probs[
        np.arange(len(probs)),
        y_true,
    ]

    top1 = probs[
        np.arange(len(probs)),
        order[:, 0],
    ]

    top2 = probs[
        np.arange(len(probs)),
        order[:, 1],
    ]

    metrics = {
        "accuracy": float(
            accuracy_score(
                y_true,
                pred,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                pred,
                average="macro",
                zero_division=0,
            )
        ),
        "top5_accuracy": float(
            np.mean(
                ranks <= 5
            )
        ),
        "mrr": float(
            np.mean(
                1.0 / ranks
            )
        ),
        "mean_true_rank": float(
            ranks.mean()
        ),
    }

    return (
        metrics,
        pred,
        ranks,
        true_prob,
        top1 - top2,
        order[:, :5],
    )


def predict_ltd_aligned(
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
                pd.to_datetime(dates)
            )
        ):
            date = pd.Timestamp(date)

            if date not in context_cache:
                raise RuntimeError(
                    f"No LTD context for {date}"
                )

            idx = np.flatnonzero(
                pd.to_datetime(dates)
                == date
            )

            (
                context_np,
                mask_np,
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

            mask = (
                torch.from_numpy(
                    mask_np
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
                    mask,
                )

                output[
                    batch_idx
                ] = (
                    logits
                    .cpu()
                    .numpy()
                )

    if np.isnan(output).any():
        raise RuntimeError(
            "Incomplete LTD predictions."
        )

    return output


def top5_jaccard(a, b):
    values = []

    for x, y in zip(a, b):
        sx = set(x.tolist())
        sy = set(y.tolist())

        values.append(
            len(
                sx & sy
            )
            /
            len(
                sx | sy
            )
        )

    return np.asarray(
        values,
        dtype=np.float32,
    )


def load_06b_reference(out):
    path = (
        out
        / "06B_R1_model_runs.csv"
    )

    if not path.exists():
        return None, path

    df = pd.read_csv(path)

    ref = df[
        (
            df["subset_size"]
            == 128
        )
        &
        (
            df["model"]
            == "xgboost"
        )
    ][
        [
            "fold",
            "accuracy",
            "macro_f1",
        ]
    ]

    if len(ref) != 2:
        return None, path

    return (
        ref.set_index("fold"),
        path,
    )


def main():

    print("=" * 78)
    print(
        "MACRO-V2-HIST — "
        "07B-1 COMPLEMENTARITY AUDIT"
    )
    print("=" * 78)

    print(
        "No fusion is fitted."
    )

    print(
        "INTERNAL_TEST: PROHIBITED"
    )

    print(
        "External Future: PROHIBITED"
    )

    mod, source07a = load_07a()

    (
        features,
        frozen_path,
    ) = mod.load_frozen_features()

    out = output_dir()

    daily_path = (
        out
        / "06C_daily_base_profiles.csv"
    )

    assignments_path = (
        out
        / "01_temporal_split_assignments.csv"
    )

    daily = pd.read_csv(
        daily_path
    )

    daily["site_label"] = (
        daily[
            "site_label"
        ].astype(str)
    )

    daily["date"] = pd.to_datetime(
        daily["date"]
    )

    (
        captures,
        dataset_path,
    ) = load_historical_65()

    assignments = pd.read_csv(
        assignments_path
    )

    dev_assignments = (
        assignments[
            assignments[
                "temporal_split"
            ].isin(
                mod.DEV_SPLITS
            )
        ][
            [
                "pcap_uid",
                "temporal_split",
            ]
        ]
        .copy()
    )

    captures = (
        captures
        .merge(
            dev_assignments,
            on="pcap_uid",
            how="inner",
            validate="one_to_one",
        )
    )

    captures[
        "site_label"
    ] = (
        captures[
            "site_label"
        ].astype(str)
    )

    dates, date_source = (
        mod.derive_dates(
            captures
        )
    )

    captures[
        "_query_date"
    ] = pd.to_datetime(
        dates
    )

    candidate_labels = np.array(
        sorted(
            captures[
                "site_label"
            ].unique()
        ),
        dtype=str,
    )

    if len(candidate_labels) != 65:
        raise RuntimeError(
            "Expected 65 classes."
        )

    label_to_idx = (
        mod.make_label_mapping(
            candidate_labels
        )
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    ref06b, ref06b_path = (
        load_06b_reference(
            out
        )
    )

    summaries = []
    paired_outputs = []
    seed_outputs = []
    per_class_outputs = []
    score_paths = []

    for fold_spec in mod.FOLDS:

        fold = fold_spec[
            "fold"
        ]

        train_splits = fold_spec[
            "train_splits"
        ]

        test_split = fold_spec[
            "test_split"
        ]

        print()
        print("=" * 78)
        print(fold)
        print("=" * 78)

        train = (
            captures[
                captures[
                    "temporal_split"
                ].isin(
                    train_splits
                )
            ]
            .copy()
            .reset_index(drop=True)
        )

        test = (
            captures[
                captures[
                    "temporal_split"
                ]
                == test_split
            ]
            .copy()
            .reset_index(drop=True)
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
            .reset_index(drop=True)
        )

        y_train = (
            mod.labels_to_indices(
                train["site_label"],
                label_to_idx,
            )
        )

        y_test = (
            mod.labels_to_indices(
                test["site_label"],
                label_to_idx,
            )
        )

        test_dates = (
            pd.to_datetime(
                test[
                    "_query_date"
                ]
            )
            .to_numpy()
        )

        # ====================================================
        # BASE-XGB
        # ====================================================

        (
            base_medians,
            x_train,
            missing,
        ) = fit_xgb_preprocessing(
            train,
            features,
        )

        if missing:
            raise RuntimeError(
                f"All-missing BASE features: {missing}"
            )

        x_test = transform_xgb(
            test,
            features,
            base_medians,
        )

        base = xgb.XGBClassifier(
            **XGB_CONFIG
        )

        t0 = time.perf_counter()

        base.fit(
            x_train,
            y_train,
        )

        base_seconds = (
            time.perf_counter()
            - t0
        )

        base_probs = (
            base.predict_proba(
                x_test
            )
            .astype(
                np.float32
            )
        )

        (
            base_metrics,
            base_pred,
            base_rank,
            base_true_prob,
            base_margin,
            base_top5,
        ) = evaluate_probs(
            base_probs,
            y_test,
        )

        print(
            "BASE-XGB "
            f"acc={base_metrics['accuracy']:.4f} "
            f"F1={base_metrics['macro_f1']:.4f} "
            f"Top5={base_metrics['top5_accuracy']:.4f} "
            f"MRR={base_metrics['mrr']:.4f}"
        )

        if (
            ref06b is not None
            and fold in ref06b.index
        ):
            ref_acc = float(
                ref06b.loc[
                    fold,
                    "accuracy",
                ]
            )

            ref_f1 = float(
                ref06b.loc[
                    fold,
                    "macro_f1",
                ]
            )

            delta_acc = (
                base_metrics[
                    "accuracy"
                ]
                - ref_acc
            )

            delta_f1 = (
                base_metrics[
                    "macro_f1"
                ]
                - ref_f1
            )

            print(
                "06B-R1 reproduction delta: "
                f"acc={delta_acc:+.10f} "
                f"F1={delta_f1:+.10f}"
            )

            if (
                abs(delta_acc) > 1e-8
                or
                abs(delta_f1) > 1e-8
            ):
                raise RuntimeError(
                    "BASE-XGB does not reproduce "
                    "06B-R1 exactly."
                )

        # ====================================================
        # LTD
        # ====================================================

        (
            ltd_medians,
            ltd_scaler,
            ltd_missing,
        ) = mod.fit_preprocessing(
            train,
            features,
        )

        if ltd_missing:
            raise RuntimeError(
                f"All-missing LTD features: {ltd_missing}"
            )

        x_train_ltd = mod.transform(
            train,
            features,
            ltd_medians,
            ltd_scaler,
        )

        x_test_ltd = mod.transform(
            test,
            features,
            ltd_medians,
            ltd_scaler,
        )

        train_dates = (
            pd.to_datetime(
                train[
                    "_query_date"
                ]
            )
            .to_numpy()
        )

        daily_history = (
            mod.standardized_daily_frame(
                train_daily,
                features,
                ltd_medians,
                ltd_scaler,
            )
        )

        (
            train_context,
            _,
        ) = mod.build_context_cache(
            daily_history,
            candidate_labels,
            train_dates,
            features,
        )

        (
            test_context,
            skipped_test,
        ) = mod.build_context_cache(
            daily_history,
            candidate_labels,
            test_dates,
            features,
        )

        if skipped_test:
            raise RuntimeError(
                f"Missing test histories: {skipped_test}"
            )

        if (
            daily_history[
                "date"
            ].max()
            >=
            pd.to_datetime(
                test_dates
            ).min()
        ):
            raise RuntimeError(
                "Temporal leakage boundary violation."
            )

        seed_probs = []

        for seed in mod.SEEDS:

            mod.set_seed(seed)

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
                x_train_ltd,
                y_train,
                train_dates,
                train_context,
                device,
                seed,
            )

            fit_seconds = (
                time.perf_counter()
                - t0
            )

            logits = predict_ltd_aligned(
                mod,
                model,
                x_test_ltd,
                test_dates,
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

            (
                metrics,
                _,
                _,
                _,
                _,
                _,
            ) = evaluate_probs(
                probs,
                y_test,
            )

            seed_outputs.append(
                {
                    "fold":
                        fold,

                    "seed":
                        seed,

                    "n_train_queries":
                        train_n,

                    "n_train_context_dates":
                        train_dates_n,

                    "n_test_queries":
                        len(y_test),

                    "train_loss":
                        train_loss,

                    "fit_seconds":
                        fit_seconds,

                    **metrics,
                }
            )

            print(
                f"LTD seed={seed} "
                f"acc={metrics['accuracy']:.4f} "
                f"F1={metrics['macro_f1']:.4f} "
                f"Top5={metrics['top5_accuracy']:.4f}"
            )

            seed_probs.append(
                probs
            )

            del model

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        seed_probs = np.stack(
            seed_probs,
            axis=0,
        )

        ltd_probs = (
            seed_probs.mean(
                axis=0
            )
        )

        ltd_probs /= (
            ltd_probs.sum(
                axis=1,
                keepdims=True,
            )
        )

        (
            ltd_metrics,
            ltd_pred,
            ltd_rank,
            ltd_true_prob,
            ltd_margin,
            ltd_top5,
        ) = evaluate_probs(
            ltd_probs,
            y_test,
        )

        print(
            "LTD-ENSEMBLE "
            f"acc={ltd_metrics['accuracy']:.4f} "
            f"F1={ltd_metrics['macro_f1']:.4f} "
            f"Top5={ltd_metrics['top5_accuracy']:.4f} "
            f"MRR={ltd_metrics['mrr']:.4f}"
        )

        # ====================================================
        # COMPLEMENTARITY
        # ====================================================

        base_correct = (
            base_pred
            == y_test
        )

        ltd_correct = (
            ltd_pred
            == y_test
        )

        both_correct = (
            base_correct
            & ltd_correct
        )

        base_only = (
            base_correct
            & ~ltd_correct
        )

        ltd_only = (
            ~base_correct
            & ltd_correct
        )

        both_wrong = (
            ~base_correct
            & ~ltd_correct
        )

        oracle = (
            base_correct
            | ltd_correct
        )

        base_wrong = (
            ~base_correct
        )

        rank_better = (
            ltd_rank
            < base_rank
        )

        rescue_rate = float(
            ltd_only.sum()
            / base_wrong.sum()
        )

        rank_better_on_error = float(
            rank_better[
                base_wrong
            ].mean()
        )

        mean_rank_gain = float(
            (
                base_rank[
                    base_wrong
                ]
                -
                ltd_rank[
                    base_wrong
                ]
            ).mean()
        )

        jaccard = top5_jaccard(
            base_top5,
            ltd_top5,
        )

        summaries.append(
            {
                "fold":
                    fold,

                "n_queries":
                    len(y_test),

                "xgb_accuracy":
                    base_metrics[
                        "accuracy"
                    ],

                "xgb_macro_f1":
                    base_metrics[
                        "macro_f1"
                    ],

                "xgb_top5":
                    base_metrics[
                        "top5_accuracy"
                    ],

                "xgb_mrr":
                    base_metrics[
                        "mrr"
                    ],

                "ltd_accuracy":
                    ltd_metrics[
                        "accuracy"
                    ],

                "ltd_macro_f1":
                    ltd_metrics[
                        "macro_f1"
                    ],

                "ltd_top5":
                    ltd_metrics[
                        "top5_accuracy"
                    ],

                "ltd_mrr":
                    ltd_metrics[
                        "mrr"
                    ],

                "both_correct_n":
                    int(
                        both_correct.sum()
                    ),

                "xgb_only_correct_n":
                    int(
                        base_only.sum()
                    ),

                "ltd_only_correct_n":
                    int(
                        ltd_only.sum()
                    ),

                "both_wrong_n":
                    int(
                        both_wrong.sum()
                    ),

                "oracle_union_accuracy":
                    float(
                        oracle.mean()
                    ),

                "oracle_gain_vs_xgb":
                    float(
                        oracle.mean()
                        -
                        base_metrics[
                            "accuracy"
                        ]
                    ),

                "rescue_rate_given_xgb_wrong":
                    rescue_rate,

                "ltd_rank_better_given_xgb_wrong":
                    rank_better_on_error,

                "mean_rank_gain_given_xgb_wrong":
                    mean_rank_gain,

                "mean_top5_jaccard":
                    float(
                        jaccard.mean()
                    ),

                "xgb_fit_seconds":
                    base_seconds,
            }
        )

        category = np.full(
            len(y_test),
            "both_wrong",
            dtype=object,
        )

        category[
            both_correct
        ] = "both_correct"

        category[
            base_only
        ] = "xgb_only_correct"

        category[
            ltd_only
        ] = "ltd_only_correct"

        paired_outputs.append(
            pd.DataFrame(
                {
                    "fold":
                        fold,

                    "pcap_uid":
                        test[
                            "pcap_uid"
                        ].astype(str),

                    "query_date":
                        pd.to_datetime(
                            test_dates
                        ).astype(str),

                    "true_label":
                        candidate_labels[
                            y_test
                        ],

                    "xgb_pred":
                        candidate_labels[
                            base_pred
                        ],

                    "xgb_true_prob":
                        base_true_prob,

                    "xgb_true_rank":
                        base_rank,

                    "xgb_margin":
                        base_margin,

                    "ltd_pred":
                        candidate_labels[
                            ltd_pred
                        ],

                    "ltd_true_prob":
                        ltd_true_prob,

                    "ltd_true_rank":
                        ltd_rank,

                    "ltd_margin":
                        ltd_margin,

                    "category":
                        category,

                    "xgb_top5_hit":
                        base_rank <= 5,

                    "ltd_top5_hit":
                        ltd_rank <= 5,

                    "ltd_rank_better":
                        rank_better,

                    "top5_jaccard":
                        jaccard,
                }
            )
        )

        for class_idx, label in enumerate(
            candidate_labels
        ):
            mask = (
                y_test
                == class_idx
            )

            b = base_correct[
                mask
            ]

            l = ltd_correct[
                mask
            ]

            per_class_outputs.append(
                {
                    "fold":
                        fold,

                    "site_label":
                        label,

                    "n_queries":
                        int(
                            mask.sum()
                        ),

                    "xgb_accuracy":
                        float(
                            b.mean()
                        ),

                    "ltd_accuracy":
                        float(
                            l.mean()
                        ),

                    "ltd_only_rescues":
                        int(
                            (
                                ~b
                                & l
                            ).sum()
                        ),

                    "xgb_only_correct":
                        int(
                            (
                                b
                                & ~l
                            ).sum()
                        ),

                    "both_wrong":
                        int(
                            (
                                ~b
                                & ~l
                            ).sum()
                        ),

                    "mean_xgb_true_rank":
                        float(
                            base_rank[
                                mask
                            ].mean()
                        ),

                    "mean_ltd_true_rank":
                        float(
                            ltd_rank[
                                mask
                            ].mean()
                        ),
                }
            )

        score_path = (
            out
            / f"07B_01_scores_{fold}.npz"
        )

        np.savez_compressed(
            score_path,

            xgb_probs=
                base_probs,

            ltd_seed_probs=
                seed_probs,

            ltd_ensemble_probs=
                ltd_probs,

            y_true=
                y_test,

            pcap_uid=
                test[
                    "pcap_uid"
                ]
                .astype(str)
                .to_numpy(),

            candidate_labels=
                candidate_labels,
        )

    summary_df = pd.DataFrame(
        summaries
    )

    paired_df = pd.concat(
        paired_outputs,
        ignore_index=True,
    )

    per_class_df = pd.DataFrame(
        per_class_outputs
    )

    seed_df = pd.DataFrame(
        seed_outputs
    )

    summary_path = (
        out
        / "07B_01_complementarity_summary.csv"
    )

    paired_path = (
        out
        / "07B_01_paired_predictions.csv"
    )

    class_path = (
        out
        / "07B_01_per_class_complementarity.csv"
    )

    seed_path = (
        out
        / "07B_01_ltd_seed_metrics.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    paired_df.to_csv(
        paired_path,
        index=False,
    )

    per_class_df.to_csv(
        class_path,
        index=False,
    )

    seed_df.to_csv(
        seed_path,
        index=False,
    )

    manifest = {
        "feature_set_id":
            "MACRO-V2-BASE-128",

        "stage":
            STAGE,

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "git_commit":
            git_commit(),

        "environment": {
            "torch":
                torch.__version__,

            "xgboost":
                xgb.__version__,

            "sklearn":
                sklearn.__version__,

            "device":
                str(device),
        },

        "data_policy": {
            "development_only":
                True,

            "internal_test_values_used":
                False,

            "external_future_used":
                False,

            "fusion_fitted":
                False,

            "query_true_label_used_to_select_context":
                False,

            "query_scored_against_all_65_candidates":
                True,
        },

        "inputs": {
            "historical_dataset": {
                "path":
                    str(dataset_path),

                "sha256":
                    sha256(
                        dataset_path
                    ),
            },

            "daily_profiles": {
                "path":
                    str(daily_path),

                "sha256":
                    sha256(
                        daily_path
                    ),
            },

            "frozen_base_features": {
                "path":
                    str(frozen_path),

                "sha256":
                    sha256(
                        frozen_path
                    ),

                "count":
                    128,
            },

            "stage07a_source": {
                "path":
                    str(source07a),

                "sha256":
                    sha256(
                        source07a
                    ),
            },

            "stage06b_reference": {
                "path":
                    str(ref06b_path),

                "available":
                    bool(
                        ref06b is not None
                    ),
            },
        },

        "xgb_configuration":
            XGB_CONFIG,

        "ltd": {
            "architecture":
                "exact 07A LTDPairScorer",

            "context_days":
                mod.CONTEXT_DAYS,

            "seeds":
                mod.SEEDS,

            "ensemble":
                "mean softmax probability",
        },

        "decision_gate": {
            "next":
                "07B-2 fusion / reranking",

            "condition":
                (
                    "non-trivial LTD rescue pool "
                    "or rank complementarity on "
                    "BASE-XGB errors"
                ),
        },
    }

    manifest_path = (
        out
        / "07B_01_manifest_complementarity_audit.json"
    )

    write_json(
        manifest_path,
        manifest,
    )

    print()
    print("=" * 78)
    print(
        "COMPLEMENTARITY SUMMARY"
    )
    print("=" * 78)

    print(
        summary_df.to_string(
            index=False
        )
    )

    print()
    print(
        "INTERNAL_TEST remains CLOSED."
    )

    print(
        "External Future remains CLOSED."
    )

    print()
    print(
        f"Summary: {summary_path}"
    )

    print(
        f"Paired: {paired_path}"
    )

    print(
        f"Per-class: {class_path}"
    )

    print(
        f"Seeds: {seed_path}"
    )

    print(
        f"Manifest: {manifest_path}"
    )

    print()
    print(
        "07B-1 COMPLEMENTARITY AUDIT COMPLETE"
    )


if __name__ == "__main__":
    main()
