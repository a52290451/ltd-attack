from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import gc
import importlib.util
import sys
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import (
    Dataset,
    DataLoader,
)

from src.utils.paths import result_path

from src.features.macro_v2._common import (
    extract_capture_dates,
    git_commit,
    sha256,
    write_json,
)


EXPECTED_SITES = 65

MICRO_EPOCHS = 30

EDGE_WEIGHT_RATIO = 4.0

MIN_MEAN_FAR_F1_GAIN = 0.005
MAX_NEAR_F1_DROP = 0.010
MIN_FAR_BOOTSTRAP_POSITIVE = 0.90


OUT = Path(
    result_path(
        "micro",
        "MICRO-ROBUSTNESS-PHASE13",
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
    path = Path(path)

    parent = str(
        path.parent
    )

    if parent not in sys.path:
        sys.path.insert(
            0,
            parent,
        )

    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            f"Could not load {path}"
        )

    mod = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        mod
    )

    return mod


class WeightedIndexedDataset(
    Dataset
):

    def __init__(
        self,
        x_dir,
        x_size,
        y,
        weights,
        indices,
    ):
        self.x_dir = x_dir
        self.x_size = x_size
        self.y = y
        self.weights = weights

        self.indices = np.asarray(
            indices,
            dtype=np.int64,
        )

    def __len__(
        self,
    ):
        return len(
            self.indices
        )

    def __getitem__(
        self,
        idx,
    ):
        i = self.indices[
            idx
        ]

        return (
            self.x_dir[
                i
            ],
            self.x_size[
                i
            ],
            self.y[
                i
            ],
            self.weights[
                i
            ],
        )


def weighted_loader(
    mod_micro,
    x_dir,
    x_size,
    y,
    weights,
    idx,
):
    ds = WeightedIndexedDataset(
        x_dir,
        x_size,
        y,
        weights,
        idx,
    )

    generator = torch.Generator()

    generator.manual_seed(
        mod_micro.SEED
    )

    return DataLoader(
        ds,
        batch_size=
            mod_micro.BATCH_SIZE,

        shuffle=True,

        generator=
            generator,

        pin_memory=True,
        num_workers=0,
    )


def train_weighted_micro(
    mod_micro,
    x_dir,
    x_size,
    y,
    train_idx,
    sample_weights,
    device,
    epochs,
    expert,
):
    mod_micro.set_seed(
        mod_micro.SEED
    )

    (
        min_size,
        max_size,
    ) = mod_micro.scaler_from_indices(
        x_size,
        train_idx,
    )

    train_loader = weighted_loader(
        mod_micro,
        x_dir,
        x_size,
        y,
        sample_weights,
        train_idx,
    )

    model = (
        mod_micro
        .MultimodalTransformer(
            EXPECTED_SITES
        )
        .to(
            device
        )
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=
            mod_micro.LEARNING_RATE,
        weight_decay=
            mod_micro.WEIGHT_DECAY,
    )

    criterion = nn.CrossEntropyLoss(
        label_smoothing=
            mod_micro.LABEL_SMOOTHING,
        reduction="none",
    )

    scheduler = (
        optim.lr_scheduler
        .CosineAnnealingWarmRestarts(
            optimizer,
            T_0=15,
            T_mult=2,
        )
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            device.type
            ==
            "cuda"
        ),
    )

    amp_enabled = (
        device.type
        ==
        "cuda"
    )

    rows = []

    for epoch in range(
        1,
        epochs + 1,
    ):
        model.train()

        loss_sum = 0.0
        n_batches = 0

        t0 = time.perf_counter()

        for (
            b_dir,
            b_size,
            b_y,
            b_weight,
        ) in train_loader:
            b_dir = (
                b_dir
                .long()
                .to(
                    device,
                    non_blocking=True,
                )
            )

            b_size = (
                b_size
                .float()
                .to(
                    device,
                    non_blocking=True,
                )
            )

            b_y = (
                b_y
                .long()
                .to(
                    device,
                    non_blocking=True,
                )
            )

            b_weight = (
                b_weight
                .float()
                .to(
                    device,
                    non_blocking=True,
                )
            )

            b_size = (
                mod_micro
                .normalize_size(
                    b_size,
                    min_size,
                    max_size,
                )
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                device_type=
                    device.type,
                dtype=
                    torch.bfloat16,
                enabled=
                    amp_enabled,
            ):
                logits = model(
                    b_dir,
                    b_size,
                )

                per_sample_loss = (
                    criterion(
                        logits,
                        b_y,
                    )
                )

                loss = (
                    (
                        per_sample_loss
                        *
                        b_weight
                    ).sum()
                    /
                    torch.clamp(
                        b_weight.sum(),
                        min=1e-12,
                    )
                )

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            scaler.step(
                optimizer
            )

            scaler.update()

            scheduler.step()

            loss_sum += float(
                loss.item()
            )

            n_batches += 1

        row = {
            "expert":
                expert,

            "epoch":
                epoch,

            "train_loss":
                loss_sum
                /
                max(
                    n_batches,
                    1,
                ),

            "seconds":
                time.perf_counter()
                -
                t0,
        }

        rows.append(
            row
        )

        print(
            f"{expert} "
            f"epoch={epoch:03d} "
            f"loss="
            f"{row['train_loss']:.4f}"
        )

    return (
        model,
        min_size,
        max_size,
        pd.DataFrame(
            rows
        ),
    )


def evaluate_windows(
    mod_micro,
    model,
    min_size,
    max_size,
    micro_df,
    x_dir,
    x_size,
    y,
    windows,
    device,
):
    result = {}

    for window, dates in (
        windows.items()
    ):
        idx = np.flatnonzero(
            micro_df[
                "_query_date"
            ]
            .isin(
                dates
            )
            .to_numpy()
        )

        loader = mod_micro.loader(
            x_dir,
            x_size,
            y,
            idx,
            False,
        )

        (
            metrics,
            probs,
            y_true,
        ) = mod_micro.evaluate(
            model,
            loader,
            min_size,
            max_size,
            device,
        )

        result[
            window
        ] = {
            "metrics":
                metrics,

            "probs":
                probs,

            "y":
                y_true,

            "dates":
                pd.to_datetime(
                    micro_df
                    .iloc[
                        idx
                    ][
                        "_query_date"
                    ]
                )
                .to_numpy(
                    dtype="datetime64[D]"
                ),
        }

    return result


def expert_diversity(
    expert_probs,
    y,
):
    names = [
        "EARLY",
        "UNIFORM",
        "LATE",
    ]

    pred = {
        name:
            np.argmax(
                expert_probs[
                    name
                ],
                axis=1,
            )
        for name in names
    }

    rows = []

    for a, b in [
        (
            "EARLY",
            "UNIFORM",
        ),
        (
            "UNIFORM",
            "LATE",
        ),
        (
            "EARLY",
            "LATE",
        ),
    ]:
        rows.append(
            {
                "measure":
                    (
                        "top1_disagreement_"
                        f"{a}_{b}"
                    ),

                "value":
                    float(
                        np.mean(
                            pred[
                                a
                            ]
                            !=
                            pred[
                                b
                            ]
                        )
                    ),
            }
        )

    matrix = np.stack(
        [
            pred[
                x
            ]
            for x in names
        ],
        axis=1,
    )

    any_correct = np.any(
        matrix
        ==
        y[
            :,
            None
        ],
        axis=1,
    )

    rows.append(
        {
            "measure":
                "top1_union_oracle_DIAGNOSTIC",

            "value":
                float(
                    np.mean(
                        any_correct
                    )
                ),
        }
    )

    return pd.DataFrame(
        rows
    )


def main():
    print("=" * 78)
    print(
        "13A — FROZEN TEMPORAL MICRO ENSEMBLE"
    )
    print("=" * 78)

    print(
        "Standalone MICRO temporal robustness development."
    )

    print(
        "Same MICRO-FINAL architecture and hyperparameters."
    )

    print(
        "EARLY / UNIFORM / LATE use all training captures."
    )

    print(
        "Only training-time temporal sample importance differs."
    )

    print(
        "No test-time adaptation."
    )

    print(
        "No parameter updates during inference."
    )

    print(
        "No Future-B numerical values or scores."
    )

    root = repo_root()

    source_micro = (
        root
        / "src/models/micro/"
        "08B_clean_temporal_micro_baseline.py"
    )

    source10a = (
        root
        / "src/models/final/"
        "10A_refit_full_historical.py"
    )

    source_macro = (
        root
        / "src/features/macro_v2/"
        "07A_candidate_conditioned_temporal_encoder.py"
    )

    source11b = (
        root
        / "src/features/macro_v2/"
        "11B_multiscale_longitudinal_memory.py"
    )

    source11h = (
        root
        / "src/features/macro_v2/"
        "11H_frozen_temporal_environment_ensemble.py"
    )

    mod_micro = load_module(
        source_micro,
        "micro_13a",
    )

    m10a = load_module(
        source10a,
        "m10a_for_13a",
    )

    mod_macro = load_module(
        source_macro,
        "macro_for_13a",
    )

    helpers = load_module(
        source11b,
        "helpers_for_13a",
    )

    m11h = load_module(
        source11h,
        "m11h_for_13a",
    )

    if mod_micro.SEED != 42:
        raise RuntimeError(
            "Unexpected MICRO seed."
        )

    if m11h.EDGE_WEIGHT_RATIO != EDGE_WEIGHT_RATIO:
        raise RuntimeError(
            "Unexpected temporal edge-weight ratio."
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

    # ==================================================
    # MICRO payload
    # ==================================================

    (
        micro_df,
        x_dir,
        x_size,
        y_micro,
        micro_labels,
        micro_source,
    ) = m10a.load_full_micro(
        mod_micro
    )

    micro_df[
        "pcap_uid"
    ] = (
        micro_df[
            "pcap_uid"
        ]
        .astype(str)
    )

    micro_df[
        "site_label"
    ] = (
        micro_df[
            "site_label"
        ]
        .astype(str)
    )

    # ==================================================
    # Authoritative date bridge from Macro
    # ==================================================

    (
        captures,
        historical_path,
    ) = mod_macro.load_historical_65()

    captures[
        "pcap_uid"
    ] = (
        captures[
            "pcap_uid"
        ]
        .astype(str)
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
    ] = extract_capture_dates(
        captures
    )

    if captures[
        "_query_date"
    ].isna().any():
        raise RuntimeError(
            "Missing Historical Macro dates."
        )

    if captures[
        "pcap_uid"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate Macro pcap_uid."
        )

    macro_labels = np.array(
        sorted(
            captures[
                "site_label"
            ].unique()
        ),
        dtype=str,
    )

    if not np.array_equal(
        micro_labels,
        macro_labels,
    ):
        raise RuntimeError(
            "Micro/Macro label order mismatch."
        )

    metadata = (
        captures[
            [
                "pcap_uid",
                "site_label",
                "_query_date",
            ]
        ]
        .rename(
            columns={
                "site_label":
                    "_macro_site_label",
            }
        )
    )

    micro_df = micro_df.merge(
        metadata,
        on="pcap_uid",
        how="left",
        validate="one_to_one",
    )

    if micro_df[
        "_query_date"
    ].isna().any():
        raise RuntimeError(
            "Micro date bridge incomplete."
        )

    if not (
        micro_df[
            "site_label"
        ]
        ==
        micro_df[
            "_macro_site_label"
        ]
    ).all():
        raise RuntimeError(
            "Micro/Macro site mismatch."
        )

    micro_df[
        "_query_date"
    ] = pd.to_datetime(
        micro_df[
            "_query_date"
        ]
    )

    unique_dates = np.array(
        sorted(
            micro_df[
                "_query_date"
            ].unique()
        )
    )

    if len(unique_dates) != 52:
        raise RuntimeError(
            "Expected 52 Historical dates."
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

    summary_rows = []
    delta_rows = []
    bootstrap_rows = []
    diversity_rows = []
    curve_parts = []
    training_rows = []

    for origin, spec in (
        origins.items()
    ):
        print()
        print("#" * 78)
        print(origin)
        print("#" * 78)

        train_idx = np.flatnonzero(
            micro_df[
                "_query_date"
            ]
            .isin(
                spec[
                    "train"
                ]
            )
            .to_numpy()
        )

        train_dates = (
            micro_df
            .iloc[
                train_idx
            ][
                "_query_date"
            ]
        )

        if (
            micro_df
            .iloc[
                train_idx
            ][
                "site_label"
            ]
            .nunique()
            !=
            EXPECTED_SITES
        ):
            raise RuntimeError(
                f"{origin}: "
                "training lacks 65 sites."
            )

        # ==============================================
        # UNIFORM canonical MICRO-FINAL
        # ==============================================

        print()
        print(
            origin,
            "UNIFORM canonical MICRO-FINAL"
        )

        t0 = time.perf_counter()

        (
            uniform_model,
            uniform_min,
            uniform_max,
            uniform_best_epoch,
            uniform_curve,
        ) = mod_micro.train(
            x_dir,
            x_size,
            y_micro,
            train_idx,
            device,
            MICRO_EPOCHS,
            val_idx=None,
        )

        if (
            uniform_best_epoch
            !=
            MICRO_EPOCHS
        ):
            raise RuntimeError(
                "Unexpected uniform epoch."
            )

        uniform_seconds = (
            time.perf_counter()
            -
            t0
        )

        uniform_curve = (
            uniform_curve.copy()
        )

        uniform_curve.insert(
            0,
            "expert",
            "UNIFORM",
        )

        uniform_curve.insert(
            0,
            "origin",
            origin,
        )

        curve_parts.append(
            uniform_curve
        )

        uniform_result = evaluate_windows(
            mod_micro,
            uniform_model,
            uniform_min,
            uniform_max,
            micro_df,
            x_dir,
            x_size,
            y_micro,
            spec[
                "windows"
            ],
            device,
        )

        training_rows.append(
            {
                "origin":
                    origin,

                "expert":
                    "UNIFORM",

                "n_train":
                    len(
                        train_idx
                    ),

                "epochs":
                    MICRO_EPOCHS,

                "seed":
                    mod_micro.SEED,

                "fit_seconds":
                    uniform_seconds,
            }
        )

        del uniform_model

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        expert_results = {
            "UNIFORM":
                uniform_result
        }

        # ==============================================
        # EARLY / LATE
        # ==============================================

        for mode in [
            "EARLY",
            "LATE",
        ]:
            temporal_weights = (
                m11h.temporal_weights(
                    train_dates,
                    mode,
                )
            )

            full_weights = np.ones(
                len(
                    micro_df
                ),
                dtype=np.float32,
            )

            full_weights[
                train_idx
            ] = (
                temporal_weights
                .astype(
                    np.float32
                )
            )

            print()
            print(
                origin,
                mode,
                "weight min/max:",
                float(
                    temporal_weights.min()
                ),
                float(
                    temporal_weights.max()
                ),
            )

            t0 = time.perf_counter()

            (
                model,
                min_size,
                max_size,
                curve,
            ) = train_weighted_micro(
                mod_micro,
                x_dir,
                x_size,
                y_micro,
                train_idx,
                full_weights,
                device,
                MICRO_EPOCHS,
                mode,
            )

            seconds = (
                time.perf_counter()
                -
                t0
            )

            curve.insert(
                0,
                "origin",
                origin,
            )

            curve_parts.append(
                curve
            )

            expert_results[
                mode
            ] = evaluate_windows(
                mod_micro,
                model,
                min_size,
                max_size,
                micro_df,
                x_dir,
                x_size,
                y_micro,
                spec[
                    "windows"
                ],
                device,
            )

            training_rows.append(
                {
                    "origin":
                        origin,

                    "expert":
                        mode,

                    "n_train":
                        len(
                            train_idx
                        ),

                    "epochs":
                        MICRO_EPOCHS,

                    "seed":
                        mod_micro.SEED,

                    "fit_seconds":
                        seconds,

                    "weight_min":
                        float(
                            temporal_weights.min()
                        ),

                    "weight_max":
                        float(
                            temporal_weights.max()
                        ),
                }
            )

            del model

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ==============================================
        # Ensemble evaluation
        # ==============================================

        for window in [
            "NEAR",
            "MID",
            "FAR",
        ]:
            y = (
                expert_results[
                    "UNIFORM"
                ][
                    window
                ][
                    "y"
                ]
            )

            dates = (
                expert_results[
                    "UNIFORM"
                ][
                    window
                ][
                    "dates"
                ]
            )

            for mode in [
                "EARLY",
                "LATE",
            ]:
                if not np.array_equal(
                    expert_results[
                        mode
                    ][
                        window
                    ][
                        "y"
                    ],
                    y,
                ):
                    raise RuntimeError(
                        "Expert y mismatch."
                    )

            probs = {
                mode:
                    expert_results[
                        mode
                    ][
                        window
                    ][
                        "probs"
                    ]
                for mode in [
                    "EARLY",
                    "UNIFORM",
                    "LATE",
                ]
            }

            ensemble = (
                m11h.geometric_mean(
                    [
                        probs[
                            "EARLY"
                        ],
                        probs[
                            "UNIFORM"
                        ],
                        probs[
                            "LATE"
                        ],
                    ]
                )
            )

            ensemble_metrics, _, _ = (
                helpers.evaluate(
                    ensemble,
                    y,
                )
            )

            uniform_metrics = (
                expert_results[
                    "UNIFORM"
                ][
                    window
                ][
                    "metrics"
                ]
            )

            for mode in [
                "EARLY",
                "UNIFORM",
                "LATE",
            ]:
                metrics = (
                    expert_results[
                        mode
                    ][
                        window
                    ][
                        "metrics"
                    ]
                )

                summary_rows.append(
                    {
                        "origin":
                            origin,

                        "window":
                            window,

                        "model":
                            mode,

                        "n":
                            len(
                                y
                            ),

                        **metrics,
                    }
                )

            summary_rows.append(
                {
                    "origin":
                        origin,

                    "window":
                        window,

                    "model":
                        "TEMPORAL_SYMMETRIC3",

                    "n":
                        len(
                            y
                        ),

                    **ensemble_metrics,
                }
            )

            delta_row = {
                "origin":
                    origin,

                "window":
                    window,
            }

            for metric in [
                "accuracy",
                "macro_f1",
                "top5_accuracy",
                "mrr",
            ]:
                delta_row[
                    f"delta_{metric}"
                ] = (
                    ensemble_metrics[
                        metric
                    ]
                    -
                    uniform_metrics[
                        metric
                    ]
                )

            delta_rows.append(
                delta_row
            )

            boot = helpers.bootstrap_delta(
                y,
                dates,
                probs[
                    "UNIFORM"
                ],
                ensemble,
            )

            for metric, values in (
                boot.items()
            ):
                bootstrap_rows.append(
                    {
                        "origin":
                            origin,

                        "window":
                            window,

                        "metric":
                            metric,

                        "observed_delta":
                            delta_row[
                                f"delta_{metric}"
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

            diversity = expert_diversity(
                probs,
                y,
            )

            diversity.insert(
                0,
                "window",
                window,
            )

            diversity.insert(
                0,
                "origin",
                origin,
            )

            diversity_rows.append(
                diversity
            )

    summary = pd.DataFrame(
        summary_rows
    )

    deltas = pd.DataFrame(
        delta_rows
    )

    bootstrap = pd.DataFrame(
        bootstrap_rows
    )

    diversity = pd.concat(
        diversity_rows,
        ignore_index=True,
    )

    far = deltas[
        deltas[
            "window"
        ]
        ==
        "FAR"
    ]

    near = deltas[
        deltas[
            "window"
        ]
        ==
        "NEAR"
    ]

    far_boot = bootstrap[
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
                "metric"
            ]
            ==
            "macro_f1"
        )
    ]

    mean_far_f1_gain = float(
        far[
            "delta_macro_f1"
        ].mean()
    )

    promotion_pass = bool(
        mean_far_f1_gain
        >=
        MIN_MEAN_FAR_F1_GAIN

        and

        (
            far[
                "delta_macro_f1"
            ]
            >
            0
        ).all()

        and

        near[
            "delta_macro_f1"
        ].min()
        >=
        -MAX_NEAR_F1_DROP

        and

        (
            far_boot[
                "fraction_delta_gt_0"
            ]
            >=
            MIN_FAR_BOOTSTRAP_POSITIVE
        ).all()
    )

    # ==================================================
    # SAVE
    # ==================================================

    summary_path = (
        OUT
        / "13A_micro_summary.csv"
    )

    delta_path = (
        OUT
        / "13A_micro_deltas.csv"
    )

    bootstrap_path = (
        OUT
        / "13A_day_block_bootstrap.csv"
    )

    diversity_path = (
        OUT
        / "13A_expert_diversity.csv"
    )

    curve_path = (
        OUT
        / "13A_training_curves.csv"
    )

    training_path = (
        OUT
        / "13A_training_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    deltas.to_csv(
        delta_path,
        index=False,
    )

    bootstrap.to_csv(
        bootstrap_path,
        index=False,
    )

    diversity.to_csv(
        diversity_path,
        index=False,
    )

    pd.concat(
        curve_parts,
        ignore_index=True,
    ).to_csv(
        curve_path,
        index=False,
    )

    pd.DataFrame(
        training_rows
    ).to_csv(
        training_path,
        index=False,
    )

    manifest = {
        "stage":
            "13A_frozen_temporal_micro_ensemble",

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "git_commit":
            git_commit(),

        "scientific_status":
            "FROZEN_MICRO_TEMPORAL_ROBUSTNESS_DEVELOPMENT",

        "research_question":
            (
                "Can training-time temporal expert diversity "
                "reduce Micro degradation while keeping "
                "inference fully frozen?"
            ),

        "data_policy": {
            "historical_only":
                True,

            "future_b_values_used":
                False,

            "future_b_scores_used":
                False,

            "test_time_parameter_updates":
                False,

            "test_time_memory_updates":
                False,

            "candidate_selection":
                False,

            "ensemble_weight_search":
                False,
        },

        "architecture": {
            "source":
                "MICRO-FINAL",

            "epochs":
                MICRO_EPOCHS,

            "seed":
                mod_micro.SEED,

            "max_len":
                mod_micro.MAX_LEN,

            "d_model":
                mod_micro.D_MODEL,

            "heads":
                mod_micro.NHEAD,

            "layers":
                mod_micro.NUM_LAYERS,
        },

        "experts": {
            "UNIFORM":
                (
                    "canonical MICRO-FINAL "
                    "training"
                ),

            "EARLY":
                (
                    "all training captures retained; "
                    "earlier dates weighted more"
                ),

            "LATE":
                (
                    "all training captures retained; "
                    "later dates weighted more"
                ),

            "edge_weight_ratio":
                EDGE_WEIGHT_RATIO,

            "aggregation":
                "geometric probability mean",
        },

        "promotion_gate": {
            "mean_far_macro_f1_gain_min":
                MIN_MEAN_FAR_F1_GAIN,

            "all_far_macro_f1_positive":
                True,

            "max_near_macro_f1_drop":
                MAX_NEAR_F1_DROP,

            "all_far_bootstrap_fraction_positive_min":
                MIN_FAR_BOOTSTRAP_POSITIVE,

            "passed":
                promotion_pass,
        },

        "results_summary": {
            "mean_far_macro_f1_gain":
                mean_far_f1_gain,
        },

        "inputs": {
            "historical_sha256":
                sha256(
                    historical_path
                ),

            "micro_source_sha256":
                sha256(
                    micro_source
                ),

            "micro_code_sha256":
                sha256(
                    source_micro
                ),

            "11H_source_sha256":
                sha256(
                    source11h
                ),
        },

        "outputs": {
            "summary":
                sha256(
                    summary_path
                ),

            "deltas":
                sha256(
                    delta_path
                ),

            "bootstrap":
                sha256(
                    bootstrap_path
                ),

            "diversity":
                sha256(
                    diversity_path
                ),

            "training_curves":
                sha256(
                    curve_path
                ),

            "training_summary":
                sha256(
                    training_path
                ),
        },

        "next_if_pass":
            (
                "13B factorial integration: "
                "canonical vs robust Micro crossed with "
                "canonical vs Phase-11 robust Macro"
            ),

        "next_if_fail":
            (
                "close temporal-expert transfer to Micro; "
                "test one distinct single-model frozen "
                "temporal-regularization hypothesis"
            ),
    }

    write_json(
        OUT
        / "13A_manifest.json",
        manifest,
    )

    print()
    print("=" * 78)
    print(
        "13A MICRO SUMMARY"
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
        "13A DELTAS"
    )
    print("=" * 78)

    print(
        deltas.to_string(
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
        far_boot.to_string(
            index=False
        )
    )

    print()
    print(
        "Mean FAR Micro Macro-F1 gain:",
        f"{mean_far_f1_gain:.6f}",
    )

    print(
        "PROMOTION PASS:",
        promotion_pass,
    )

    print(
        "Frozen inference: YES"
    )

    print(
        "Future-B accessed: NO"
    )


if __name__ == "__main__":
    main()
