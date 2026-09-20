from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import importlib.util
import random

import numpy as np
import pandas as pd

from sklearn.metrics import f1_score

import torch
import torch.nn.functional as F

from _common import (
    git_commit,
    load_historical_65,
    output_dir,
    sha256,
    write_json,
)


FOLDS = [
    "A_EARLY_TO_MIDDLE",
    "B_EARLY_MIDDLE_TO_LATE",
]

CONTEXT_DAYS = 5
MAX_STALENESS_DAYS = 14

ALPHA = 0.375
EPS = 1e-12

N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20260920


def load_module(path, name):

    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(path)

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    return mod


def fuse(xgb, ltd):

    z = (
        (1.0 - ALPHA)
        * np.log(
            np.clip(
                xgb,
                EPS,
                1.0,
            )
        )
        +
        ALPHA
        * np.log(
            np.clip(
                ltd,
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


def softmax_numpy(logits):

    z = (
        logits
        -
        logits.max(
            axis=1,
            keepdims=True,
        )
    )

    p = np.exp(z)

    return (
        p
        / p.sum(
            axis=1,
            keepdims=True,
        )
    )


def predictions(probs, y):

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

    return pred, ranks


def metrics(
    y,
    pred,
    ranks,
    idx=None,
):

    if idx is None:
        idx = np.arange(
            len(y)
        )

    yy = y[idx]
    pp = pred[idx]
    rr = ranks[idx]

    return {
        "accuracy":
            float(
                np.mean(
                    yy == pp
                )
            ),

        "macro_f1":
            float(
                f1_score(
                    yy,
                    pp,
                    labels=np.arange(65),
                    average="macro",
                    zero_division=0,
                )
            ),

        "top5_accuracy":
            float(
                np.mean(
                    rr <= 5
                )
            ),

        "mrr":
            float(
                np.mean(
                    1.0 / rr
                )
            ),

        "mean_true_rank":
            float(
                np.mean(
                    rr
                )
            ),
    }


def paired_delta(
    y,
    ref_pred,
    ref_rank,
    candidate_pred,
    candidate_rank,
    idx=None,
):

    a = metrics(
        y,
        ref_pred,
        ref_rank,
        idx,
    )

    b = metrics(
        y,
        candidate_pred,
        candidate_rank,
        idx,
    )

    return {
        k:
            b[k]
            -
            a[k]
        for k in a
    }


def build_all_cutoff_contexts(
    mod07a,
    daily_history,
    candidate_labels,
    features,
    min_date,
    max_date,
):

    cache = {}

    calendar = pd.date_range(
        pd.Timestamp(
            min_date
        ),
        pd.Timestamp(
            max_date
        ),
        freq="D",
    )

    for cutoff in calendar:

        result = (
            mod07a.build_context_for_date(
                daily_history,
                candidate_labels,
                cutoff,
                features,
            )
        )

        if result is not None:
            cache[
                pd.Timestamp(
                    cutoff
                )
            ] = result

    return cache


def eligible_cutoffs(
    query_date,
    context_cache,
):

    query_date = pd.Timestamp(
        query_date
    )

    candidates = []

    for cutoff in context_cache:

        # Context builder uses history < cutoff.
        # Therefore cutoff=query_date corresponds
        # to a nominal latest age of >=1 day.

        nominal_staleness = (
            query_date
            -
            cutoff
        ).days

        if (
            0
            <= nominal_staleness
            <= MAX_STALENESS_DAYS - 1
        ):
            candidates.append(
                cutoff
            )

    return sorted(
        candidates
    )


def train_stale_model(
    mod07a,
    model,
    x_train,
    y_train,
    train_dates,
    context_cache,
    device,
    seed,
):

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=mod07a.LEARNING_RATE,
        weight_decay=
            mod07a.WEIGHT_DECAY,
    )

    rng = np.random.default_rng(
        seed
    )

    train_dates_pd = pd.to_datetime(
        train_dates
    )

    unique_query_dates = sorted(
        pd.unique(
            train_dates_pd
        )
    )

    index_by_date = {}

    eligible_by_date = {}

    for date in unique_query_dates:

        date = pd.Timestamp(
            date
        )

        idx = np.flatnonzero(
            train_dates_pd
            == date
        )

        cutoffs = eligible_cutoffs(
            date,
            context_cache,
        )

        if (
            len(idx)
            and
            cutoffs
        ):
            index_by_date[
                date
            ] = idx

            eligible_by_date[
                date
            ] = cutoffs

    if not index_by_date:
        raise RuntimeError(
            "No stale-context training dates."
        )

    final_loss = None

    sampled_records = []

    model.train()

    for epoch in range(
        mod07a.EPOCHS
    ):

        date_order = list(
            index_by_date.keys()
        )

        rng.shuffle(
            date_order
        )

        total_loss = 0.0
        total_n = 0

        for query_date in date_order:

            query_indices = (
                index_by_date[
                    query_date
                ].copy()
            )

            rng.shuffle(
                query_indices
            )

            choices = (
                eligible_by_date[
                    query_date
                ]
            )

            # One context age is selected independently
            # for each query-date and training epoch.
            selected_cutoff = choices[
                int(
                    rng.integers(
                        0,
                        len(
                            choices
                        ),
                    )
                )
            ]

            nominal_staleness = int(
                (
                    query_date
                    -
                    selected_cutoff
                ).days
                + 1
            )

            sampled_records.append(
                {
                    "seed":
                        seed,

                    "epoch":
                        epoch + 1,

                    "query_date":
                        str(
                            query_date.date()
                        ),

                    "context_cutoff":
                        str(
                            selected_cutoff.date()
                        ),

                    "nominal_latest_age_days":
                        nominal_staleness,
                }
            )

            (
                context_np,
                padding_np,
                _,
            ) = context_cache[
                selected_cutoff
            ]

            context = (
                torch.from_numpy(
                    context_np
                )
                .to(
                    device
                )
            )

            padding = (
                torch.from_numpy(
                    padding_np
                )
                .to(
                    device
                )
            )

            for start in range(
                0,
                len(
                    query_indices
                ),
                mod07a.BATCH_SIZE,
            ):

                idx = query_indices[
                    start:
                    start
                    +
                    mod07a.BATCH_SIZE
                ]

                x = (
                    torch.from_numpy(
                        x_train[
                            idx
                        ]
                    )
                    .to(
                        device
                    )
                )

                y = (
                    torch.from_numpy(
                        y_train[
                            idx
                        ]
                    )
                    .to(
                        device
                    )
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                logits = model(
                    x,
                    context,
                    padding,
                )

                loss = (
                    F.cross_entropy(
                        logits,
                        y,
                    )
                )

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    mod07a.GRAD_CLIP,
                )

                optimizer.step()

                n = len(
                    idx
                )

                total_loss += (
                    float(
                        loss.item()
                    )
                    *
                    n
                )

                total_n += n

        final_loss = (
            total_loss
            /
            total_n
        )

    return (
        float(
            final_loss
        ),
        int(
            sum(
                len(v)
                for v in
                index_by_date.values()
            )
        ),
        len(
            index_by_date
        ),
        sampled_records,
    )


def predict_model(
    mod07d,
    mod07a,
    model,
    fold_data,
    test_context,
    device,
):

    return mod07d.predict_ltd(
        mod07a,
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


def bootstrap_dates(
    y,
    dates,
    ref_pred,
    ref_rank,
    candidate_pred,
    candidate_rank,
):

    unique_dates = np.array(
        sorted(
            pd.unique(
                dates
            )
        )
    )

    by_date = {
        d:
            np.flatnonzero(
                dates
                == d
            )
        for d in
        unique_dates
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
            paired_delta(
                y,
                ref_pred,
                ref_rank,
                candidate_pred,
                candidate_rank,
                idx,
            )
        )

    return pd.DataFrame(
        rows
    )


def main():

    print("=" * 78)
    print(
        "07G-1 — STALE-CONTEXT TRAINING"
    )
    print("=" * 78)

    print(
        "Reference: MACRO-LTD-V2"
    )

    print(
        "Architecture unchanged."
    )

    print(
        "Context days:",
        CONTEXT_DAYS,
    )

    print(
        "Maximum training staleness:",
        MAX_STALENESS_DAYS,
    )

    print(
        "Alpha frozen:",
        ALPHA,
    )

    print(
        "No architecture/context/alpha search."
    )

    print(
        "INTERNAL_TEST: CLOSED"
    )

    print(
        "Future-B: CLOSED"
    )

    here = Path(
        __file__
    ).resolve().parent

    mod07d = load_module(
        here
        / "07D_01_context_length_ablation.py",
        "macro_v2_07d_stale",
    )

    mod07a, source07a = (
        mod07d.load_07a()
    )

    mod07a.CONTEXT_DAYS = (
        CONTEXT_DAYS
    )

    features, frozen_path = (
        mod07a.load_frozen_features()
    )

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

    captures, historical_path = (
        load_historical_65()
    )

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

    dates, _ = (
        mod07a.derive_dates(
            captures
        )
    )

    captures[
        "_query_date"
    ] = pd.to_datetime(
        dates
    )

    fold_specs = {
        f["fold"]:
            f
        for f in
        mod07a.FOLDS
    }

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    summary_rows = []
    comparison_rows = []
    bootstrap_rows = []
    seed_rows = []
    sampling_rows = []
    day_rows = []

    for fold in FOLDS:

        print()
        print("=" * 78)
        print(fold)
        print("=" * 78)

        fold_data = (
            mod07d.prepare_fold(
                mod07a,
                captures,
                daily,
                features,
                fold_specs[
                    fold
                ],
            )
        )

        train_dates_pd = pd.to_datetime(
            fold_data[
                "train_dates"
            ]
        )

        all_contexts = (
            build_all_cutoff_contexts(
                mod07a,
                fold_data[
                    "daily_history"
                ],
                fold_data[
                    "candidate_labels"
                ],
                features,
                train_dates_pd.min(),
                train_dates_pd.max(),
            )
        )

        (
            test_context,
            skipped_test,
        ) = mod07a.build_context_cache(
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
                f"{fold}: test context missing."
            )

        reference_path = (
            out
            / f"07D_01R_scores_{fold}.npz"
        )

        ref = np.load(
            reference_path,
            allow_pickle=True,
        )

        xgb = ref[
            "xgb_probs"
        ].astype(
            np.float64
        )

        v2_ltd = ref[
            "v2_ltd_probs"
        ].astype(
            np.float64
        )

        v2_fusion = ref[
            "v2_fusion_probs"
        ].astype(
            np.float64
        )

        y = ref[
            "y_true"
        ].astype(int)

        saved_uid = ref[
            "pcap_uid"
        ].astype(str)

        expected_uid = (
            fold_data[
                "test"
            ][
                "pcap_uid"
            ]
            .astype(str)
            .to_numpy()
        )

        if not np.array_equal(
            saved_uid,
            expected_uid,
        ):
            raise RuntimeError(
                f"{fold}: pcap alignment mismatch."
            )

        probs_by_seed = []

        for seed in mod07a.SEEDS:

            print(
                f"{fold} — stale seed {seed}"
            )

            mod07a.set_seed(
                seed
            )

            model = (
                mod07a.LTDPairScorer()
                .to(
                    device
                )
            )

            (
                train_loss,
                train_n,
                train_dates_n,
                sampled,
            ) = train_stale_model(
                mod07a,
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
                all_contexts,
                device,
                seed,
            )

            logits = predict_model(
                mod07d,
                mod07a,
                model,
                fold_data,
                test_context,
                device,
            )

            probs = (
                softmax_numpy(
                    logits.astype(
                        np.float64
                    )
                )
            )

            pred, rank = predictions(
                probs,
                y,
            )

            m = metrics(
                y,
                pred,
                rank,
            )

            seed_rows.append(
                {
                    "fold":
                        fold,

                    "seed":
                        seed,

                    "train_loss":
                        train_loss,

                    "n_train_queries":
                        train_n,

                    "n_train_context_dates":
                        train_dates_n,

                    **m,
                }
            )

            for row in sampled:
                sampling_rows.append(
                    {
                        "fold":
                            fold,
                        **row,
                    }
                )

            probs_by_seed.append(
                probs
            )

            del model

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        stale_ltd = (
            np.stack(
                probs_by_seed,
                axis=0,
            )
            .mean(
                axis=0
            )
        )

        stale_ltd /= (
            stale_ltd.sum(
                axis=1,
                keepdims=True,
            )
        )

        stale_fusion = fuse(
            xgb,
            stale_ltd,
        )

        model_probs = {
            "V2_LTD":
                v2_ltd,

            "STALE_LTD":
                stale_ltd,

            "V2_FUSION":
                v2_fusion,

            "STALE_FUSION":
                stale_fusion,
        }

        outputs = {}

        for name, probs in (
            model_probs.items()
        ):

            pred, rank = predictions(
                probs,
                y,
            )

            m = metrics(
                y,
                pred,
                rank,
            )

            outputs[
                name
            ] = (
                pred,
                rank,
                m,
            )

            summary_rows.append(
                {
                    "fold":
                        fold,

                    "model":
                        name,

                    **m,
                }
            )

        (
            ref_pred,
            ref_rank,
            _,
        ) = outputs[
            "V2_FUSION"
        ]

        (
            candidate_pred,
            candidate_rank,
            _,
        ) = outputs[
            "STALE_FUSION"
        ]

        observed = paired_delta(
            y,
            ref_pred,
            ref_rank,
            candidate_pred,
            candidate_rank,
        )

        query_dates = (
            pd.to_datetime(
                fold_data[
                    "test_dates"
                ]
            )
            .strftime(
                "%Y-%m-%d"
            )
            .to_numpy()
        )

        boot = bootstrap_dates(
            y,
            query_dates,
            ref_pred,
            ref_rank,
            candidate_pred,
            candidate_rank,
        )

        ref_correct = (
            ref_pred == y
        )

        candidate_correct = (
            candidate_pred == y
        )

        candidate_only = (
            ~ref_correct
            &
            candidate_correct
        )

        ref_only = (
            ref_correct
            &
            ~candidate_correct
        )

        comparison_rows.append(
            {
                "fold":
                    fold,

                **{
                    f"stale_minus_v2_{k}":
                        v
                    for k, v
                    in observed.items()
                },

                "stale_only_correct_n":
                    int(
                        candidate_only.sum()
                    ),

                "v2_only_correct_n":
                    int(
                        ref_only.sum()
                    ),

                "net_stale_correct_n":
                    int(
                        candidate_only.sum()
                        -
                        ref_only.sum()
                    ),
            }
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

            bootstrap_rows.append(
                {
                    "fold":
                        fold,

                    "metric":
                        metric,

                    "observed_stale_minus_v2":
                        observed[
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

                    "p_stale_gt_v2":
                        float(
                            np.mean(
                                values > 0
                            )
                        ),
                }
            )

        for date in sorted(
            pd.unique(
                query_dates
            )
        ):

            idx = np.flatnonzero(
                query_dates == date
            )

            d = paired_delta(
                y,
                ref_pred,
                ref_rank,
                candidate_pred,
                candidate_rank,
                idx,
            )

            day_rows.append(
                {
                    "fold":
                        fold,

                    "date":
                        date,

                    "n":
                        len(idx),

                    **{
                        f"delta_{k}":
                            v
                        for k, v
                        in d.items()
                    },
                }
            )

        np.savez_compressed(
            out
            / f"07G_01_scores_{fold}.npz",

            pcap_uid=
                saved_uid.astype(str),

            y_true=
                y,

            xgb_probs=
                xgb.astype(
                    np.float32
                ),

            v2_ltd_probs=
                v2_ltd.astype(
                    np.float32
                ),

            stale_ltd_probs=
                stale_ltd.astype(
                    np.float32
                ),

            v2_fusion_probs=
                v2_fusion.astype(
                    np.float32
                ),

            stale_fusion_probs=
                stale_fusion.astype(
                    np.float32
                ),
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    bootstrap_df = pd.DataFrame(
        bootstrap_rows
    )

    seed_df = pd.DataFrame(
        seed_rows
    )

    sampling_df = pd.DataFrame(
        sampling_rows
    )

    day_df = pd.DataFrame(
        day_rows
    )

    a = (
        comparison_df[
            comparison_df[
                "fold"
            ]
            == "A_EARLY_TO_MIDDLE"
        ]
        .iloc[0]
    )

    b = (
        bootstrap_df[
            bootstrap_df[
                "fold"
            ]
            == "B_EARLY_MIDDLE_TO_LATE"
        ]
        .set_index(
            "metric"
        )
    )

    promotion_pass = bool(
        a[
            "stale_minus_v2_accuracy"
        ] >= 0
        and
        a[
            "stale_minus_v2_macro_f1"
        ] >= 0
        and
        b.loc[
            "accuracy",
            "observed_stale_minus_v2",
        ] > 0
        and
        b.loc[
            "macro_f1",
            "observed_stale_minus_v2",
        ] > 0
        and
        b.loc[
            "accuracy",
            "p_stale_gt_v2",
        ] >= 0.90
        and
        b.loc[
            "macro_f1",
            "p_stale_gt_v2",
        ] >= 0.90
        and
        b.loc[
            "top5_accuracy",
            "observed_stale_minus_v2",
        ] >= 0
        and
        b.loc[
            "mrr",
            "observed_stale_minus_v2",
        ] >= 0
    )

    summary_df.to_csv(
        out
        / "07G_01_stale_summary.csv",
        index=False,
    )

    comparison_df.to_csv(
        out
        / "07G_01_stale_vs_v2.csv",
        index=False,
    )

    bootstrap_df.to_csv(
        out
        / "07G_01_day_block_bootstrap.csv",
        index=False,
    )

    seed_df.to_csv(
        out
        / "07G_01_seed_metrics.csv",
        index=False,
    )

    sampling_df.to_csv(
        out
        / "07G_01_sampling_audit.csv",
        index=False,
    )

    day_df.to_csv(
        out
        / "07G_01_per_day_deltas.csv",
        index=False,
    )

    manifest = {
        "stage":
            "07G_01_stale_context_training",

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "git_commit":
            git_commit(),

        "data_policy": {
            "development_only":
                True,

            "feature_tuning":
                False,

            "context_tuning":
                False,

            "alpha_tuning":
                False,

            "architecture_search":
                False,

            "internal_test_values_used":
                False,

            "external_future_used":
                False,
        },

        "reference": {
            "model":
                "MACRO-LTD-V2",

            "context_days":
                CONTEXT_DAYS,

            "alpha":
                ALPHA,

            "seeds":
                mod07a.SEEDS,
        },

        "candidate": {
            "architecture":
                "identical to MACRO-LTD-V2",

            "training_change":
                (
                    "uniform sampling of an eligible "
                    "historical candidate-context cutoff "
                    "for each training query-date and epoch"
                ),

            "max_nominal_latest_age_days":
                MAX_STALENESS_DAYS,

            "inference_protocol":
                "unchanged frozen-history context",
        },

        "hypothesis":
            (
                "exposure to stale candidate histories during "
                "training reduces the train/inference context-age "
                "mismatch and improves temporal transfer"
            ),

        "promotion_gate": {
            "passed":
                promotion_pass,

            "requirements": [
                "Fold-A Accuracy delta >= 0",
                "Fold-A Macro-F1 delta >= 0",
                "Fold-B Accuracy delta > 0",
                "Fold-B Macro-F1 delta > 0",
                "Fold-B bootstrap P(delta Accuracy > 0) >= 0.90",
                "Fold-B bootstrap P(delta Macro-F1 > 0) >= 0.90",
                "Fold-B Top-5 delta >= 0",
                "Fold-B MRR delta >= 0",
            ],
        },

        "stop_rule": {
            "if_pass":
                (
                    "perform robustness confirmation and "
                    "promote candidate to MACRO-FINAL if confirmed"
                ),

            "if_fail":
                (
                    "retain MACRO-LTD-V2 as MACRO-FINAL "
                    "and proceed to clean Micro phase"
                ),

            "additional_macro_architecture_search":
                False,
        },

        "inputs": {
            "historical_sha256":
                sha256(
                    historical_path
                ),

            "daily_profiles_sha256":
                sha256(
                    daily_path
                ),

            "frozen_features_sha256":
                sha256(
                    frozen_path
                ),

            "07A_source_sha256":
                sha256(
                    source07a
                ),
        },
    }

    write_json(
        out
        / "07G_01_manifest_stale_training.json",
        manifest,
    )

    print()
    print("=" * 78)
    print("STALE-CONTEXT SUMMARY")
    print("=" * 78)

    print(
        summary_df.to_string(
            index=False
        )
    )

    print()
    print(
        comparison_df.to_string(
            index=False
        )
    )

    print()
    print(
        bootstrap_df.to_string(
            index=False
        )
    )

    print()
    print(
        "PROMOTION PASS:",
        promotion_pass,
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
