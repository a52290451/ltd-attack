from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import importlib.util
import sys
import time
import gc

import numpy as np
import pandas as pd

from sklearn.metrics import f1_score

import torch
import xgboost as xgb

from src.utils.paths import result_path

from src.features.macro_v2._common import (
    extract_capture_dates,
    git_commit,
    load_historical_65,
    sha256,
    write_json,
)


EXPECTED_SITES = 65

CONTEXT_TOKENS = 5

ALPHA_LTD = 0.375

EPS = 1e-12

N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20260922


OUT = Path(
    result_path(
        "feature_engineering",
        "LTD-ROBUSTNESS-PHASE11",
    )
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


XGB_CONFIG = {
    "n_estimators": 120,
    "max_depth": 5,
    "learning_rate": 0.10,
    "min_child_weight": 1.0,
    "subsample": 0.80,
    "colsample_bytree": 0.80,
    "reg_lambda": 1.0,
    "objective": "multi:softprob",
    "num_class": EXPECTED_SITES,
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}


def repo_root():

    return (
        Path(__file__)
        .resolve()
        .parents[3]
    )


def load_07a():

    path = (
        repo_root()
        / "src/features/macro_v2/"
        "07A_candidate_conditioned_temporal_encoder.py"
    )

    parent = str(
        path.parent
    )

    if parent not in sys.path:
        sys.path.insert(
            0,
            parent,
        )

    spec = importlib.util.spec_from_file_location(
        "ltd_07a_for_11b",
        path,
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            "Could not load 07A."
        )

    mod = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        mod
    )

    return mod, path


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


def fuse(
    xgb_probs,
    ltd_probs,
):

    logits = (
        (
            1.0
            -
            ALPHA_LTD
        )
        *
        np.log(
            np.clip(
                xgb_probs,
                EPS,
                1.0,
            )
        )
        +
        ALPHA_LTD
        *
        np.log(
            np.clip(
                ltd_probs,
                EPS,
                1.0,
            )
        )
    )

    return softmax(
        logits
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
        +
        1
    )

    return (
        {
            "accuracy":
                float(
                    np.mean(
                        pred == y
                    )
                ),

            "macro_f1":
                float(
                    f1_score(
                        y,
                        pred,
                        labels=np.arange(
                            EXPECTED_SITES
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
                        1.0
                        /
                        ranks
                    )
                ),

            "mean_true_rank":
                float(
                    np.mean(
                        ranks
                    )
                ),
        },
        pred,
        ranks,
    )


def build_daily(
    captures,
    features,
):

    numeric = (
        captures[
            features
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
    )

    work = pd.concat(
        [
            captures[
                [
                    "site_label",
                    "_query_date",
                ]
            ].reset_index(
                drop=True
            ),
            numeric.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    daily = (
        work
        .groupby(
            [
                "site_label",
                "_query_date",
            ],
            observed=True,
        )[
            features
        ]
        .median()
        .reset_index()
        .rename(
            columns={
                "_query_date":
                    "date",
            }
        )
    )

    daily[
        "date"
    ] = pd.to_datetime(
        daily[
            "date"
        ]
    )

    return (
        daily
        .sort_values(
            [
                "site_label",
                "date",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def multiscale_context_for_date(
    daily_history,
    candidate_labels,
    cutoff_date,
    features,
):

    cutoff = pd.Timestamp(
        cutoff_date
    )

    contexts = np.zeros(
        (
            len(
                candidate_labels
            ),
            CONTEXT_TOKENS,
            len(
                features
            ),
        ),
        dtype=np.float32,
    )

    padding = np.zeros(
        (
            len(
                candidate_labels
            ),
            CONTEXT_TOKENS,
        ),
        dtype=bool,
    )

    history_lengths = []

    for idx, site in enumerate(
        candidate_labels
    ):

        h = (
            daily_history[
                (
                    daily_history[
                        "site_label"
                    ]
                    ==
                    site
                )
                &
                (
                    daily_history[
                        "date"
                    ]
                    <
                    cutoff
                )
            ]
            .sort_values(
                "date"
            )
        )

        if h.empty:
            return None

        values = (
            h[
                features
            ]
            .to_numpy(
                dtype=np.float32
            )
        )

        tokens = np.stack(
            [
                # Long-term identity.
                np.median(
                    values,
                    axis=0,
                ),

                # Medium-term history.
                np.median(
                    values[
                        -10:
                    ],
                    axis=0,
                ),

                # Same horizon as the
                # current recent context,
                # but aggregated.
                np.median(
                    values[
                        -5:
                    ],
                    axis=0,
                ),

                # Short-term state.
                np.median(
                    values[
                        -3:
                    ],
                    axis=0,
                ),

                # Most recent observed day.
                values[
                    -1
                ],
            ],
            axis=0,
        )

        contexts[
            idx
        ] = tokens

        history_lengths.append(
            len(
                values
            )
        )

    return (
        contexts,
        padding,
        history_lengths,
    )


def build_multiscale_cache(
    daily_history,
    candidate_labels,
    query_dates,
    features,
):

    cache = {}
    skipped = []

    for date in sorted(
        pd.unique(
            pd.to_datetime(
                query_dates
            )
        )
    ):

        result = (
            multiscale_context_for_date(
                daily_history,
                candidate_labels,
                date,
                features,
            )
        )

        if result is None:

            skipped.append(
                pd.Timestamp(
                    date
                )
            )

        else:

            cache[
                pd.Timestamp(
                    date
                )
            ] = result

    return (
        cache,
        skipped,
    )


def context_cache(
    mod,
    memory,
    daily_history,
    candidate_labels,
    query_dates,
    features,
):

    mod.CONTEXT_DAYS = (
        CONTEXT_TOKENS
    )

    if memory == "RECENT5_SEQUENCE":

        return mod.build_context_cache(
            daily_history,
            candidate_labels,
            query_dates,
            features,
        )

    if memory == "MULTISCALE5":

        return build_multiscale_cache(
            daily_history,
            candidate_labels,
            query_dates,
            features,
        )

    raise ValueError(
        memory
    )


def predict_ltd(
    mod,
    model,
    x,
    dates,
    cache,
    device,
):

    output = np.full(
        (
            len(x),
            EXPECTED_SITES,
        ),
        np.nan,
        dtype=np.float32,
    )

    model.eval()

    with torch.inference_mode():

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

            if date not in cache:
                raise RuntimeError(
                    f"No context for {date}"
                )

            idx = np.flatnonzero(
                pd.to_datetime(
                    dates
                )
                ==
                date
            )

            (
                context_np,
                padding_np,
                _,
            ) = cache[
                date
            ]

            context = (
                torch
                .from_numpy(
                    context_np
                )
                .to(
                    device
                )
            )

            padding = (
                torch
                .from_numpy(
                    padding_np
                )
                .to(
                    device
                )
            )

            for start in range(
                0,
                len(idx),
                mod.BATCH_SIZE,
            ):

                batch_idx = idx[
                    start:
                    start
                    +
                    mod.BATCH_SIZE
                ]

                query = (
                    torch
                    .from_numpy(
                        x[
                            batch_idx
                        ]
                    )
                    .to(
                        device
                    )
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

    return softmax(
        output
    )


def prepare_matrix(
    mod,
    train,
    tests,
    features,
):

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

    transformed = {
        name:
            mod.transform(
                frame,
                features,
                medians,
                scaler,
            )
        for name, frame
        in tests.items()
    }

    return (
        medians,
        scaler,
        x_train,
        transformed,
    )


def train_xgb(
    train,
    tests,
    features,
    y_train,
):

    train_x = (
        train[
            features
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
    )

    medians = (
        train_x
        .median(
            axis=0,
            skipna=True,
        )
        .fillna(
            0.0
        )
    )

    x_train = (
        train_x
        .fillna(
            medians
        )
        .to_numpy(
            dtype=np.float32
        )
    )

    model = xgb.XGBClassifier(
        **XGB_CONFIG
    )

    model.fit(
        x_train,
        y_train,
    )

    if not np.array_equal(
        model.classes_,
        np.arange(
            EXPECTED_SITES
        ),
    ):
        raise RuntimeError(
            "Unexpected XGB class order."
        )

    result = {}

    for name, test in (
        tests.items()
    ):

        x = (
            test[
                features
            ]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
            .replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )
            .fillna(
                medians
            )
            .to_numpy(
                dtype=np.float32
            )
        )

        result[
            name
        ] = (
            model
            .predict_proba(
                x
            )
            .astype(
                np.float64
            )
        )

    return result


def bootstrap_delta(
    y,
    dates,
    reference_probs,
    candidate_probs,
):

    (
        _,
        ref_pred,
        ref_rank,
    ) = evaluate(
        reference_probs,
        y,
    )

    (
        _,
        cand_pred,
        cand_rank,
    ) = evaluate(
        candidate_probs,
        y,
    )

    unique_dates = np.array(
        sorted(
            np.unique(
                dates
            )
        )
    )

    by_date = {
        date:
            np.flatnonzero(
                dates == date
            )
        for date in
        unique_dates
    }

    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    values = {
        "accuracy": [],
        "macro_f1": [],
        "top5_accuracy": [],
        "mrr": [],
    }

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
                for d in
                sampled
            ]
        )

        yy = y[
            idx
        ]

        ref_values = {
            "accuracy":
                float(
                    np.mean(
                        ref_pred[
                            idx
                        ]
                        ==
                        yy
                    )
                ),

            "macro_f1":
                float(
                    f1_score(
                        yy,
                        ref_pred[
                            idx
                        ],
                        labels=np.arange(
                            EXPECTED_SITES
                        ),
                        average="macro",
                        zero_division=0,
                    )
                ),

            "top5_accuracy":
                float(
                    np.mean(
                        ref_rank[
                            idx
                        ]
                        <= 5
                    )
                ),

            "mrr":
                float(
                    np.mean(
                        1.0
                        /
                        ref_rank[
                            idx
                        ]
                    )
                ),
        }

        cand_values = {
            "accuracy":
                float(
                    np.mean(
                        cand_pred[
                            idx
                        ]
                        ==
                        yy
                    )
                ),

            "macro_f1":
                float(
                    f1_score(
                        yy,
                        cand_pred[
                            idx
                        ],
                        labels=np.arange(
                            EXPECTED_SITES
                        ),
                        average="macro",
                        zero_division=0,
                    )
                ),

            "top5_accuracy":
                float(
                    np.mean(
                        cand_rank[
                            idx
                        ]
                        <= 5
                    )
                ),

            "mrr":
                float(
                    np.mean(
                        1.0
                        /
                        cand_rank[
                            idx
                        ]
                    )
                ),
        }

        for metric in values:

            values[
                metric
            ].append(
                cand_values[
                    metric
                ]
                -
                ref_values[
                    metric
                ]
            )

    return {
        metric:
            np.asarray(
                vals,
                dtype=float,
            )
        for metric, vals
        in values.items()
    }


def main():

    print("=" * 78)
    print(
        "11B — MULTISCALE LONGITUDINAL MEMORY"
    )
    print("=" * 78)

    print(
        "Historical-only robustness development."
    )

    print(
        "BASE128 remains frozen."
    )

    print(
        "RECENT5 sequence vs MULTISCALE5."
    )

    print(
        "Same LTD architecture and same 5-token budget."
    )

    print(
        "Test observations NEVER update history."
    )

    print(
        "Future-B values/scores are not used."
    )

    mod, source07a = load_07a()

    # Same architecture/hyperparameters.
    mod.CONTEXT_DAYS = (
        CONTEXT_TOKENS
    )

    (
        features,
        frozen_path,
    ) = mod.load_frozen_features()

    if len(
        features
    ) != 128:

        raise RuntimeError(
            "Expected BASE128."
        )

    (
        captures,
        historical_path,
    ) = load_historical_65()

    captures[
        "site_label"
    ] = (
        captures[
            "site_label"
        ]
        .astype(str)
    )

    captures[
        "_query_date"
    ] = (
        extract_capture_dates(
            captures
        )
    )

    if captures[
        "_query_date"
    ].isna().any():

        raise RuntimeError(
            "Missing Historical dates."
        )

    unique_dates = np.array(
        sorted(
            captures[
                "_query_date"
            ].unique()
        )
    )

    if len(
        unique_dates
    ) != 52:

        raise RuntimeError(
            "Expected 52 Historical dates."
        )

    candidate_labels = np.array(
        sorted(
            captures[
                "site_label"
            ].unique()
        ),
        dtype=str,
    )

    if len(
        candidate_labels
    ) != EXPECTED_SITES:

        raise RuntimeError(
            "Expected 65 sites."
        )

    label_to_idx = (
        mod.make_label_mapping(
            candidate_labels
        )
    )

    daily_all = build_daily(
        captures,
        features,
    )

    origins = {
        "ORIGIN14": {
            "train":
                unique_dates[
                    :14
                ],

            "windows": {
                "NEAR":
                    unique_dates[
                        14:21
                    ],

                "MID":
                    unique_dates[
                        28:35
                    ],

                "FAR":
                    unique_dates[
                        42:49
                    ],
            },
        },

        "ORIGIN28": {
            "train":
                unique_dates[
                    :28
                ],

            "windows": {
                "NEAR":
                    unique_dates[
                        28:35
                    ],

                "MID":
                    unique_dates[
                        35:42
                    ],

                "FAR":
                    unique_dates[
                        45:52
                    ],
            },
        },
    }

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

    summary_rows = []
    seed_rows = []
    bootstrap_rows = []

    score_store = {}

    for origin, spec in (
        origins.items()
    ):

        print()
        print("#" * 78)
        print(origin)
        print("#" * 78)

        train = (
            captures[
                captures[
                    "_query_date"
                ].isin(
                    spec[
                        "train"
                    ]
                )
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        tests = {
            name:
                captures[
                    captures[
                        "_query_date"
                    ].isin(
                        dates
                    )
                ]
                .copy()
                .reset_index(
                    drop=True
                )
            for name, dates
            in spec[
                "windows"
            ].items()
        }

        if (
            train[
                "site_label"
            ].nunique()
            != EXPECTED_SITES
        ):

            raise RuntimeError(
                f"{origin}: incomplete train coverage."
            )

        for name, frame in (
            tests.items()
        ):

            if (
                frame[
                    "site_label"
                ].nunique()
                != EXPECTED_SITES
            ):

                raise RuntimeError(
                    f"{origin}/{name}: "
                    "incomplete site coverage."
                )

        y_train = (
            mod.labels_to_indices(
                train[
                    "site_label"
                ],
                label_to_idx,
            )
        )

        y_tests = {
            name:
                mod.labels_to_indices(
                    frame[
                        "site_label"
                    ],
                    label_to_idx,
                )
            for name, frame
            in tests.items()
        }

        (
            medians,
            scaler,
            x_train,
            x_tests,
        ) = prepare_matrix(
            mod,
            train,
            tests,
            features,
        )

        train_dates = (
            pd.to_datetime(
                train[
                    "_query_date"
                ]
            )
            .to_numpy()
        )

        test_dates = {
            name:
                pd.to_datetime(
                    frame[
                        "_query_date"
                    ]
                )
                .to_numpy()
            for name, frame
            in tests.items()
        }

        train_daily = (
            daily_all[
                daily_all[
                    "date"
                ].isin(
                    spec[
                        "train"
                    ]
                )
            ]
            .copy()
        )

        daily_history = (
            mod.standardized_daily_frame(
                train_daily,
                features,
                medians,
                scaler,
            )
        )

        xgb_probs = train_xgb(
            train,
            tests,
            features,
            y_train,
        )

        for memory in [
            "RECENT5_SEQUENCE",
            "MULTISCALE5",
        ]:

            print()
            print(
                origin,
                memory,
            )

            (
                train_cache,
                skipped_train,
            ) = context_cache(
                mod,
                memory,
                daily_history,
                candidate_labels,
                train_dates,
                features,
            )

            if not train_cache:

                raise RuntimeError(
                    f"{origin}/{memory}: "
                    "no causal training contexts."
                )

            window_caches = {}

            for window in tests:

                (
                    cache,
                    skipped,
                ) = context_cache(
                    mod,
                    memory,
                    daily_history,
                    candidate_labels,
                    test_dates[
                        window
                    ],
                    features,
                )

                if skipped:

                    raise RuntimeError(
                        f"{origin}/{memory}/{window}: "
                        f"missing contexts {skipped}"
                    )

                window_caches[
                    window
                ] = cache

            seed_probs = {
                window: []
                for window in tests
            }

            for seed in mod.SEEDS:

                print(
                    " seed=",
                    seed,
                )

                mod.set_seed(
                    seed
                )

                # Critical:
                # both experiments instantiate
                # the exact same 5-token model.
                mod.CONTEXT_DAYS = (
                    CONTEXT_TOKENS
                )

                model = (
                    mod.LTDPairScorer()
                    .to(
                        device
                    )
                )

                t0 = time.perf_counter()

                (
                    train_loss,
                    n_train_queries,
                    n_train_dates,
                ) = (
                    mod.train_ltd_model(
                        model,
                        x_train,
                        y_train,
                        train_dates,
                        train_cache,
                        device,
                        seed,
                    )
                )

                fit_seconds = (
                    time.perf_counter()
                    -
                    t0
                )

                seed_rows.append(
                    {
                        "origin":
                            origin,

                        "memory":
                            memory,

                        "seed":
                            seed,

                        "train_loss":
                            train_loss,

                        "n_train_queries":
                            n_train_queries,

                        "n_train_context_dates":
                            n_train_dates,

                        "skipped_train_dates":
                            len(
                                skipped_train
                            ),

                        "fit_seconds":
                            fit_seconds,
                    }
                )

                for window in tests:

                    probs = predict_ltd(
                        mod,
                        model,
                        x_tests[
                            window
                        ],
                        test_dates[
                            window
                        ],
                        window_caches[
                            window
                        ],
                        device,
                    )

                    seed_probs[
                        window
                    ].append(
                        probs
                    )

                del model

                gc.collect()

                if torch.cuda.is_available():

                    torch.cuda.empty_cache()

            for window in tests:

                ensemble = (
                    np.stack(
                        seed_probs[
                            window
                        ],
                        axis=0,
                    )
                    .mean(
                        axis=0
                    )
                )

                ensemble /= (
                    ensemble.sum(
                        axis=1,
                        keepdims=True,
                    )
                )

                fusion = fuse(
                    xgb_probs[
                        window
                    ],
                    ensemble,
                )

                ltd_metrics, _, _ = evaluate(
                    ensemble,
                    y_tests[
                        window
                    ],
                )

                fusion_metrics, _, _ = evaluate(
                    fusion,
                    y_tests[
                        window
                    ],
                )

                xgb_metrics, _, _ = evaluate(
                    xgb_probs[
                        window
                    ],
                    y_tests[
                        window
                    ],
                )

                summary_rows.append(
                    {
                        "origin":
                            origin,

                        "memory":
                            memory,

                        "window":
                            window,

                        "n_train":
                            len(
                                train
                            ),

                        "n_test":
                            len(
                                tests[
                                    window
                                ]
                            ),

                        "history_frozen":
                            True,

                        **{
                            f"xgb_{k}":
                                v
                            for k, v
                            in xgb_metrics.items()
                        },

                        **{
                            f"ltd_{k}":
                                v
                            for k, v
                            in ltd_metrics.items()
                        },

                        **{
                            f"fusion_{k}":
                                v
                            for k, v
                            in fusion_metrics.items()
                        },
                    }
                )

                score_store[
                    (
                        origin,
                        memory,
                        window,
                    )
                ] = {
                    "y":
                        y_tests[
                            window
                        ],

                    "dates":
                        pd.to_datetime(
                            test_dates[
                                window
                            ]
                        )
                        .to_numpy(
                            dtype="datetime64[D]"
                        ),

                    "ltd":
                        ensemble,

                    "fusion":
                        fusion,

                    "xgb":
                        xgb_probs[
                            window
                        ],
                }

    summary = pd.DataFrame(
        summary_rows
    )

    comparisons = []

    for origin in origins:

        for window in [
            "NEAR",
            "MID",
            "FAR",
        ]:

            base = (
                summary[
                    (
                        summary[
                            "origin"
                        ]
                        ==
                        origin
                    )
                    &
                    (
                        summary[
                            "window"
                        ]
                        ==
                        window
                    )
                    &
                    (
                        summary[
                            "memory"
                        ]
                        ==
                        "RECENT5_SEQUENCE"
                    )
                ]
                .iloc[
                    0
                ]
            )

            multi = (
                summary[
                    (
                        summary[
                            "origin"
                        ]
                        ==
                        origin
                    )
                    &
                    (
                        summary[
                            "window"
                        ]
                        ==
                        window
                    )
                    &
                    (
                        summary[
                            "memory"
                        ]
                        ==
                        "MULTISCALE5"
                    )
                ]
                .iloc[
                    0
                ]
            )

            row = {
                "origin":
                    origin,

                "window":
                    window,
            }

            for model in [
                "ltd",
                "fusion",
            ]:

                for metric in [
                    "accuracy",
                    "macro_f1",
                    "top5_accuracy",
                    "mrr",
                ]:

                    key = (
                        f"{model}_{metric}"
                    )

                    row[
                        f"delta_{key}"
                    ] = (
                        float(
                            multi[
                                key
                            ]
                        )
                        -
                        float(
                            base[
                                key
                            ]
                        )
                    )

            comparisons.append(
                row
            )

            recent_scores = score_store[
                (
                    origin,
                    "RECENT5_SEQUENCE",
                    window,
                )
            ]

            multi_scores = score_store[
                (
                    origin,
                    "MULTISCALE5",
                    window,
                )
            ]

            for model in [
                "ltd",
                "fusion",
            ]:

                boot = bootstrap_delta(
                    recent_scores[
                        "y"
                    ],
                    recent_scores[
                        "dates"
                    ],
                    recent_scores[
                        model
                    ],
                    multi_scores[
                        model
                    ],
                )

                for metric, values in (
                    boot.items()
                ):

                    observed = (
                        row[
                            (
                                f"delta_{model}_"
                                f"{metric}"
                            )
                        ]
                    )

                    bootstrap_rows.append(
                        {
                            "origin":
                                origin,

                            "window":
                                window,

                            "model":
                                model,

                            "metric":
                                metric,

                            "observed_delta":
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

    comparison = pd.DataFrame(
        comparisons
    )

    bootstrap = pd.DataFrame(
        bootstrap_rows
    )

    far = (
        comparison[
            comparison[
                "window"
            ]
            ==
            "FAR"
        ]
    )

    near = (
        comparison[
            comparison[
                "window"
            ]
            ==
            "NEAR"
        ]
    )

    far_boot = (
        bootstrap[
            (
                bootstrap[
                    "window"
                ]
                ==
                "FAR"
            )
            &
            (
                bootstrap[
                    "model"
                ]
                ==
                "fusion"
            )
            &
            (
                bootstrap[
                    "metric"
                ]
                ==
                "macro_f1"
            )
        ]
    )

    promotion_pass = bool(
        far[
            "delta_fusion_macro_f1"
        ].mean()
        >=
        0.005

        and

        (
            far[
                "delta_fusion_macro_f1"
            ]
            >
            0
        ).all()

        and

        far[
            "delta_ltd_macro_f1"
        ].mean()
        >
        0

        and

        near[
            "delta_fusion_macro_f1"
        ].min()
        >=
        -0.010

        and

        (
            far_boot[
                "fraction_delta_gt_0"
            ]
            >=
            0.90
        ).all()
    )

    summary_path = (
        OUT
        / "11B_memory_summary.csv"
    )

    comparison_path = (
        OUT
        / "11B_multiscale_vs_recent5.csv"
    )

    seed_path = (
        OUT
        / "11B_seed_training.csv"
    )

    bootstrap_path = (
        OUT
        / "11B_day_block_bootstrap.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    pd.DataFrame(
        seed_rows
    ).to_csv(
        seed_path,
        index=False,
    )

    bootstrap.to_csv(
        bootstrap_path,
        index=False,
    )

    manifest = {
        "stage":
            "11B_multiscale_longitudinal_memory",

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "git_commit":
            git_commit(),

        "scientific_status":
            "POST_CONFIRMATORY_ROBUSTNESS_DEVELOPMENT",

        "hypothesis":
            (
                "A multiscale candidate history "
                "retains site identity under stale "
                "context better than the last five "
                "raw site-day profiles."
            ),

        "data_policy": {
            "historical_only":
                True,

            "future_b_values_used":
                False,

            "future_b_scores_used":
                False,

            "test_observations_update_history":
                False,

            "history_bank":
                "training-window only",
        },

        "representation": {
            "features":
                "MACRO-V2-BASE-128",

            "context_tokens":
                CONTEXT_TOKENS,

            "control":
                (
                    "last five chronological "
                    "site-day profiles"
                ),

            "multiscale_tokens": [
                "median all prior site-days",
                "median last 10 prior site-days",
                "median last 5 prior site-days",
                "median last 3 prior site-days",
                "last prior site-day",
            ],

            "same_model_architecture":
                True,

            "same_context_token_budget":
                True,
        },

        "model": {
            "source":
                "07A LTDPairScorer",

            "seeds":
                [
                    int(x)
                    for x in
                    mod.SEEDS
                ],

            "epochs":
                mod.EPOCHS,

            "fusion_alpha_ltd":
                ALPHA_LTD,
        },

        "origins": {
            origin: {
                "train_dates":
                    [
                        str(
                            pd.Timestamp(
                                x
                            ).date()
                        )
                        for x in
                        spec[
                            "train"
                        ]
                    ],

                "windows": {
                    name: [
                        str(
                            pd.Timestamp(
                                x
                            ).date()
                        )
                        for x in
                        dates
                    ]
                    for name, dates
                    in spec[
                        "windows"
                    ].items()
                },
            }
            for origin, spec
            in origins.items()
        },

        "promotion_gate": {
            "mean_far_fusion_macro_f1_delta_min":
                0.005,

            "all_far_fusion_macro_f1_positive":
                True,

            "mean_far_ltd_macro_f1_positive":
                True,

            "max_near_fusion_macro_f1_degradation":
                0.010,

            "all_far_fusion_macro_f1_bootstrap_positive_fraction_min":
                0.90,

            "passed":
                promotion_pass,
        },

        "inputs": {
            "historical": {
                "path":
                    str(
                        historical_path
                    ),

                "sha256":
                    sha256(
                        historical_path
                    ),
            },

            "base128": {
                "path":
                    str(
                        frozen_path
                    ),

                "sha256":
                    sha256(
                        frozen_path
                    ),
            },

            "07A_source": {
                "path":
                    str(
                        source07a
                    ),

                "sha256":
                    sha256(
                        source07a
                    ),
            },
        },

        "outputs": {
            "summary":
                sha256(
                    summary_path
                ),

            "comparison":
                sha256(
                    comparison_path
                ),

            "seed_training":
                sha256(
                    seed_path
                ),

            "bootstrap":
                sha256(
                    bootstrap_path
                ),
        },

        "next_if_pass":
            (
                "11C temporal contrastive "
                "representation using multiscale memory"
            ),

        "next_if_fail":
            (
                "11C temporal contrastive representation "
                "using BASE128 and retain RECENT5 control"
            ),
    }

    write_json(
        OUT
        / "11B_manifest.json",
        manifest,
    )

    print()
    print("=" * 78)
    print(
        "11B SUMMARY"
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
        "MULTISCALE5 VS RECENT5"
    )
    print("=" * 78)

    print(
        comparison.to_string(
            index=False
        )
    )

    print()
    print("=" * 78)
    print(
        "FAR BOOTSTRAP"
    )
    print("=" * 78)

    print(
        bootstrap[
            bootstrap[
                "window"
            ]
            ==
            "FAR"
        ]
        .to_string(
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
        "Future-B was not accessed."
    )


if __name__ == "__main__":
    main()
