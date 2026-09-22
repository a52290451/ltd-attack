from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import gc
import importlib.util
import random
import sys
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.paths import result_path

from src.features.macro_v2._common import (
    extract_capture_dates,
    git_commit,
    load_historical_65,
    sha256,
    write_json,
)


EXPECTED_SITES = 65

INPUT_DIM = 128
CONTRASTIVE_DIM = 128

CONTEXT_DAYS = 5

TCL_SEED = 42
TCL_EPOCHS = 20
TCL_STEPS_PER_EPOCH = 50
TCL_TEMPERATURE = 0.07
TCL_LR = 1e-3
TCL_WEIGHT_DECAY = 1e-4
TCL_DROPOUT = 0.10
TCL_GRAD_CLIP = 1.0

N_BOOTSTRAP = 2000

MIN_FAR_FUSION_F1_GAIN = 0.005
MAX_NEAR_FUSION_F1_DROP = 0.010


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


def repo_root():

    return (
        Path(__file__)
        .resolve()
        .parents[3]
    )


def load_module(
    path,
    name,
):

    path = Path(
        path
    )

    parent = str(
        path.parent
    )

    if parent not in sys.path:

        sys.path.insert(
            0,
            parent,
        )

    spec = (
        importlib.util
        .spec_from_file_location(
            name,
            path,
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):

        raise RuntimeError(
            f"Could not load {path}"
        )

    mod = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(
        mod
    )

    return mod


def set_seed(
    seed,
):

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )

    if hasattr(
        torch.backends,
        "cudnn",
    ):

        torch.backends.cudnn.deterministic = True

        torch.backends.cudnn.benchmark = False


class TemporalContrastiveEncoder(
    nn.Module
):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                INPUT_DIM,
                256,
            ),

            nn.GELU(),

            nn.LayerNorm(
                256
            ),

            nn.Dropout(
                TCL_DROPOUT
            ),

            nn.Linear(
                256,
                CONTRASTIVE_DIM,
            ),

            nn.LayerNorm(
                CONTRASTIVE_DIM
            ),
        )

    def forward(
        self,
        x,
    ):

        z = self.net(
            x
        )

        return F.normalize(
            z,
            p=2,
            dim=-1,
        )


def build_temporal_pools(
    y,
    dates,
):

    unique_dates = np.array(
        sorted(
            pd.unique(
                pd.to_datetime(
                    dates
                )
            )
        )
    )

    block = max(
        3,
        len(
            unique_dates
        )
        //
        3,
    )

    early_dates = unique_dates[
        :block
    ]

    late_dates = unique_dates[
        -block:
    ]

    early_mask = np.isin(
        pd.to_datetime(
            dates
        ),
        early_dates,
    )

    late_mask = np.isin(
        pd.to_datetime(
            dates
        ),
        late_dates,
    )

    early_pools = {}
    late_pools = {}

    for cls in range(
        EXPECTED_SITES
    ):

        early = np.flatnonzero(
            (
                y
                ==
                cls
            )
            &
            early_mask
        )

        late = np.flatnonzero(
            (
                y
                ==
                cls
            )
            &
            late_mask
        )

        if (
            len(
                early
            )
            ==
            0
            or
            len(
                late
            )
            ==
            0
        ):

            raise RuntimeError(
                f"Class {cls} has no "
                "early/late contrastive pool."
            )

        early_pools[
            cls
        ] = early

        late_pools[
            cls
        ] = late

    return (
        early_pools,
        late_pools,
        early_dates,
        late_dates,
    )


def train_contrastive(
    x_train,
    y_train,
    train_dates,
    device,
):

    set_seed(
        TCL_SEED
    )

    (
        early_pools,
        late_pools,
        early_dates,
        late_dates,
    ) = build_temporal_pools(
        y_train,
        train_dates,
    )

    model = (
        TemporalContrastiveEncoder()
        .to(
            device
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TCL_LR,
        weight_decay=TCL_WEIGHT_DECAY,
    )

    target = torch.arange(
        EXPECTED_SITES,
        dtype=torch.long,
        device=device,
    )

    rng = np.random.default_rng(
        TCL_SEED
    )

    rows = []

    model.train()

    for epoch in range(
        1,
        TCL_EPOCHS
        +
        1,
    ):

        total_loss = 0.0

        for _ in range(
            TCL_STEPS_PER_EPOCH
        ):

            idx_early = np.array(
                [
                    rng.choice(
                        early_pools[
                            cls
                        ]
                    )
                    for cls in range(
                        EXPECTED_SITES
                    )
                ],
                dtype=np.int64,
            )

            idx_late = np.array(
                [
                    rng.choice(
                        late_pools[
                            cls
                        ]
                    )
                    for cls in range(
                        EXPECTED_SITES
                    )
                ],
                dtype=np.int64,
            )

            early = (
                torch
                .from_numpy(
                    x_train[
                        idx_early
                    ]
                )
                .to(
                    device
                )
            )

            late = (
                torch
                .from_numpy(
                    x_train[
                        idx_late
                    ]
                )
                .to(
                    device
                )
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            z_early = model(
                early
            )

            z_late = model(
                late
            )

            similarity = (
                z_early
                @
                z_late.T
            ) / TCL_TEMPERATURE

            loss_forward = (
                F.cross_entropy(
                    similarity,
                    target,
                )
            )

            loss_backward = (
                F.cross_entropy(
                    similarity.T,
                    target,
                )
            )

            loss = (
                0.5
                *
                (
                    loss_forward
                    +
                    loss_backward
                )
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                TCL_GRAD_CLIP,
            )

            optimizer.step()

            total_loss += float(
                loss.item()
            )

        mean_loss = (
            total_loss
            /
            TCL_STEPS_PER_EPOCH
        )

        rows.append(
            {
                "epoch":
                    epoch,

                "loss":
                    mean_loss,
            }
        )

        print(
            " TCL epoch",
            epoch,
            "loss=",
            f"{mean_loss:.6f}",
        )

    return {
        "model":
            model,

        "curve":
            pd.DataFrame(
                rows
            ),

        "early_dates":
            early_dates,

        "late_dates":
            late_dates,
    }


def encode_numpy(
    model,
    x,
    device,
    batch_size=2048,
):

    model.eval()

    output = []

    with torch.inference_mode():

        for start in range(
            0,
            len(
                x
            ),
            batch_size,
        ):

            batch = (
                torch
                .from_numpy(
                    x[
                        start:
                        start
                        +
                        batch_size
                    ]
                )
                .to(
                    device
                )
            )

            z = model(
                batch
            )

            output.append(
                z
                .cpu()
                .numpy()
                .astype(
                    np.float32
                )
            )

    return np.vstack(
        output
    )


def encoded_daily_frame(
    encoder,
    raw_daily,
    raw_features,
    device,
):

    values = (
        raw_daily[
            raw_features
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    encoded = encode_numpy(
        encoder,
        values,
        device,
    )

    encoded_features = [
        f"tcl_{i:03d}"
        for i in range(
            CONTRASTIVE_DIM
        )
    ]

    out = pd.DataFrame(
        encoded,
        columns=encoded_features,
    )

    out.insert(
        0,
        "date",
        pd.to_datetime(
            raw_daily[
                "date"
            ]
        ).to_numpy(),
    )

    out.insert(
        0,
        "site_label",
        raw_daily[
            "site_label"
        ]
        .astype(str)
        .to_numpy(),
    )

    return (
        out,
        encoded_features,
    )


def main():

    print("=" * 78)
    print(
        "11C — TEMPORAL CONTRASTIVE REPRESENTATION"
    )
    print("=" * 78)

    print(
        "Historical-only robustness development."
    )

    print(
        "BASE128 remains canonical."
    )

    print(
        "XGB remains unchanged."
    )

    print(
        "Only the LTD representation changes."
    )

    print(
        "RECENT5 remains the memory control."
    )

    print(
        "No Future-B values or scores are used."
    )

    source07a = (
        repo_root()
        / "src/features/macro_v2/"
        "07A_candidate_conditioned_temporal_encoder.py"
    )

    source11b = (
        repo_root()
        / "src/features/macro_v2/"
        "11B_multiscale_longitudinal_memory.py"
    )

    mod = load_module(
        source07a,
        "ltd_07a_for_11c",
    )

    helpers = load_module(
        source11b,
        "ltd_11b_helpers_for_11c",
    )

    mod.CONTEXT_DAYS = (
        CONTEXT_DAYS
    )

    (
        features,
        frozen_path,
    ) = (
        mod.load_frozen_features()
    )

    if len(
        features
    ) != INPUT_DIM:

        raise RuntimeError(
            "Expected BASE128."
        )

    (
        captures,
        historical_path,
    ) = (
        load_historical_65()
    )

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
            "Expected 65 classes."
        )

    label_to_idx = (
        mod.make_label_mapping(
            candidate_labels
        )
    )

    daily_all = (
        helpers.build_daily(
            captures,
            features,
        )
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
    comparison_rows = []
    bootstrap_rows = []
    seed_rows = []
    curve_rows = []

    encoder_artifacts = {}

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
            !=
            EXPECTED_SITES
        ):

            raise RuntimeError(
                f"{origin}: incomplete train coverage."
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
        ) = helpers.prepare_matrix(
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

        raw_daily_history = (
            mod.standardized_daily_frame(
                train_daily,
                features,
                medians,
                scaler,
            )
        )

        # ---------------------------------------------
        # Canonical static branch.
        # Exactly the same XGB for RAW and TCL.
        # ---------------------------------------------

        xgb_probs = (
            helpers.train_xgb(
                train,
                tests,
                features,
                y_train,
            )
        )

        # ---------------------------------------------
        # Train temporal contrastive encoder.
        # Historical training window only.
        # ---------------------------------------------

        print()
        print(
            "Training temporal contrastive encoder..."
        )

        t0 = time.perf_counter()

        tcl = train_contrastive(
            x_train,
            y_train,
            train_dates,
            device,
        )

        tcl_seconds = (
            time.perf_counter()
            -
            t0
        )

        encoder = tcl[
            "model"
        ]

        for _, row in (
            tcl[
                "curve"
            ].iterrows()
        ):

            curve_rows.append(
                {
                    "origin":
                        origin,

                    **row.to_dict(),
                }
            )

        encoded_train = (
            encode_numpy(
                encoder,
                x_train,
                device,
            )
        )

        encoded_tests = {
            name:
                encode_numpy(
                    encoder,
                    values,
                    device,
                )
            for name, values
            in x_tests.items()
        }

        (
            encoded_daily_history,
            encoded_features,
        ) = encoded_daily_frame(
            encoder,
            raw_daily_history,
            features,
            device,
        )

        encoder_path = (
            OUT
            / (
                "11C_TCL128_"
                f"{origin}.pth"
            )
        )

        torch.save(
            {
                "state_dict":
                    encoder.state_dict(),

                "origin":
                    origin,

                "seed":
                    TCL_SEED,

                "input_dim":
                    INPUT_DIM,

                "output_dim":
                    CONTRASTIVE_DIM,

                "temperature":
                    TCL_TEMPERATURE,

                "epochs":
                    TCL_EPOCHS,

                "steps_per_epoch":
                    TCL_STEPS_PER_EPOCH,

                "early_dates":
                    [
                        str(
                            pd.Timestamp(
                                x
                            ).date()
                        )
                        for x in
                        tcl[
                            "early_dates"
                        ]
                    ],

                "late_dates":
                    [
                        str(
                            pd.Timestamp(
                                x
                            ).date()
                        )
                        for x in
                        tcl[
                            "late_dates"
                        ]
                    ],
            },
            encoder_path,
        )

        encoder_artifacts[
            origin
        ] = {
            "path":
                str(
                    encoder_path
                ),

            "sha256":
                sha256(
                    encoder_path
                ),

            "fit_seconds":
                tcl_seconds,
        }

        del encoder

        gc.collect()

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

        representations = {
            "RAW128_RECENT5": {
                "x_train":
                    x_train,

                "x_tests":
                    x_tests,

                "daily_history":
                    raw_daily_history,

                "context_features":
                    features,
            },

            "TCL128_RECENT5": {
                "x_train":
                    encoded_train,

                "x_tests":
                    encoded_tests,

                "daily_history":
                    encoded_daily_history,

                "context_features":
                    encoded_features,
            },
        }

        for (
            representation,
            rep,
        ) in representations.items():

            print()
            print(
                origin,
                representation,
            )

            (
                train_cache,
                skipped_train,
            ) = (
                mod.build_context_cache(
                    rep[
                        "daily_history"
                    ],
                    candidate_labels,
                    train_dates,
                    rep[
                        "context_features"
                    ],
                )
            )

            if not train_cache:

                raise RuntimeError(
                    f"{origin}/{representation}: "
                    "no training context."
                )

            test_caches = {}

            for window in tests:

                (
                    cache,
                    skipped,
                ) = (
                    mod.build_context_cache(
                        rep[
                            "daily_history"
                        ],
                        candidate_labels,
                        test_dates[
                            window
                        ],
                        rep[
                            "context_features"
                        ],
                    )
                )

                if skipped:

                    raise RuntimeError(
                        f"{origin}/{representation}/"
                        f"{window}: missing context."
                    )

                test_caches[
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

                mod.CONTEXT_DAYS = (
                    CONTEXT_DAYS
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
                        rep[
                            "x_train"
                        ],
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

                        "representation":
                            representation,

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

                    probs = (
                        helpers.predict_ltd(
                            mod,
                            model,
                            rep[
                                "x_tests"
                            ][
                                window
                            ],
                            test_dates[
                                window
                            ],
                            test_caches[
                                window
                            ],
                            device,
                        )
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

                ltd = (
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

                ltd /= (
                    ltd.sum(
                        axis=1,
                        keepdims=True,
                    )
                )

                fusion = (
                    helpers.fuse(
                        xgb_probs[
                            window
                        ],
                        ltd,
                    )
                )

                (
                    ltd_metrics,
                    _,
                    _,
                ) = (
                    helpers.evaluate(
                        ltd,
                        y_tests[
                            window
                        ],
                    )
                )

                (
                    fusion_metrics,
                    _,
                    _,
                ) = (
                    helpers.evaluate(
                        fusion,
                        y_tests[
                            window
                        ],
                    )
                )

                (
                    xgb_metrics,
                    _,
                    _,
                ) = (
                    helpers.evaluate(
                        xgb_probs[
                            window
                        ],
                        y_tests[
                            window
                        ],
                    )
                )

                summary_rows.append(
                    {
                        "origin":
                            origin,

                        "representation":
                            representation,

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
                        representation,
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
                        ltd,

                    "fusion":
                        fusion,
                }

    summary = pd.DataFrame(
        summary_rows
    )

    for origin in origins:

        for window in [
            "NEAR",
            "MID",
            "FAR",
        ]:

            raw = (
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
                            "representation"
                        ]
                        ==
                        "RAW128_RECENT5"
                    )
                    &
                    (
                        summary[
                            "window"
                        ]
                        ==
                        window
                    )
                ]
                .iloc[
                    0
                ]
            )

            tcl = (
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
                            "representation"
                        ]
                        ==
                        "TCL128_RECENT5"
                    )
                    &
                    (
                        summary[
                            "window"
                        ]
                        ==
                        window
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
                            tcl[
                                key
                            ]
                        )
                        -
                        float(
                            raw[
                                key
                            ]
                        )
                    )

            comparison_rows.append(
                row
            )

            raw_scores = score_store[
                (
                    origin,
                    "RAW128_RECENT5",
                    window,
                )
            ]

            tcl_scores = score_store[
                (
                    origin,
                    "TCL128_RECENT5",
                    window,
                )
            ]

            for model in [
                "ltd",
                "fusion",
            ]:

                bootstrap = (
                    helpers.bootstrap_delta(
                        raw_scores[
                            "y"
                        ],
                        raw_scores[
                            "dates"
                        ],
                        raw_scores[
                            model
                        ],
                        tcl_scores[
                            model
                        ],
                    )
                )

                for metric, values in (
                    bootstrap.items()
                ):

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
                                row[
                                    (
                                        f"delta_"
                                        f"{model}_"
                                        f"{metric}"
                                    )
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

    comparison = pd.DataFrame(
        comparison_rows
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
        MIN_FAR_FUSION_F1_GAIN

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
        -MAX_NEAR_FUSION_F1_DROP

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
        / "11C_representation_summary.csv"
    )

    comparison_path = (
        OUT
        / "11C_contrastive_vs_raw.csv"
    )

    bootstrap_path = (
        OUT
        / "11C_day_block_bootstrap.csv"
    )

    seed_path = (
        OUT
        / "11C_ltd_seed_training.csv"
    )

    curve_path = (
        OUT
        / "11C_contrastive_training_curve.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    bootstrap.to_csv(
        bootstrap_path,
        index=False,
    )

    pd.DataFrame(
        seed_rows
    ).to_csv(
        seed_path,
        index=False,
    )

    pd.DataFrame(
        curve_rows
    ).to_csv(
        curve_path,
        index=False,
    )

    manifest = {
        "stage":
            "11C_temporal_contrastive_representation",

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
                "Explicitly aligning same-site captures "
                "across separated Historical periods "
                "learns a representation that preserves "
                "candidate identity under temporal drift."
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
        },

        "control": {
            "representation":
                "BASE128 standardized",

            "memory":
                "RECENT5",

            "xgb":
                "canonical BASE128 XGB",
        },

        "candidate": {
            "representation":
                "TCL128",

            "input_dim":
                INPUT_DIM,

            "output_dim":
                CONTRASTIVE_DIM,

            "objective":
                (
                    "symmetric temporal cross-period "
                    "contrastive classification"
                ),

            "positive":
                (
                    "same site sampled from separated "
                    "early and late training blocks"
                ),

            "negatives":
                "all other sites in the 65-site batch",

            "temperature":
                TCL_TEMPERATURE,

            "seed":
                TCL_SEED,

            "epochs":
                TCL_EPOCHS,

            "steps_per_epoch":
                TCL_STEPS_PER_EPOCH,

            "memory":
                "RECENT5",

            "xgb":
                (
                    "UNCHANGED canonical BASE128 XGB; "
                    "TCL modifies LTD branch only"
                ),
        },

        "downstream_ltd": {
            "source":
                "07A LTDPairScorer",

            "context_days":
                CONTEXT_DAYS,

            "seeds":
                [
                    int(x)
                    for x in
                    mod.SEEDS
                ],

            "epochs":
                mod.EPOCHS,

            "fusion_alpha_ltd":
                helpers.ALPHA_LTD,
        },

        "promotion_gate": {
            "mean_far_fusion_macro_f1_gain_min":
                MIN_FAR_FUSION_F1_GAIN,

            "all_far_fusion_macro_f1_positive":
                True,

            "mean_far_ltd_macro_f1_positive":
                True,

            "max_near_fusion_macro_f1_drop":
                MAX_NEAR_FUSION_F1_DROP,

            "all_far_bootstrap_positive_fraction_min":
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

            "11B_source": {
                "path":
                    str(
                        source11b
                    ),

                "sha256":
                    sha256(
                        source11b
                    ),
            },
        },

        "encoder_artifacts":
            encoder_artifacts,

        "outputs": {
            "summary":
                sha256(
                    summary_path
                ),

            "comparison":
                sha256(
                    comparison_path
                ),

            "bootstrap":
                sha256(
                    bootstrap_path
                ),

            "ltd_seed_training":
                sha256(
                    seed_path
                ),

            "contrastive_curve":
                sha256(
                    curve_path
                ),
        },

        "next_if_pass":
            (
                "11D combine independently supported "
                "TCL representation with a controlled "
                "longitudinal memory/adaptation experiment"
            ),

        "next_if_fail":
            (
                "11D retain BASE128+RECENT5 and test "
                "long-term prototype / drift-aware "
                "candidate scoring"
            ),
    }

    write_json(
        OUT
        / "11C_manifest.json",
        manifest,
    )

    print()
    print("=" * 78)
    print(
        "11C REPRESENTATION SUMMARY"
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
        "TCL128 VS RAW128"
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
