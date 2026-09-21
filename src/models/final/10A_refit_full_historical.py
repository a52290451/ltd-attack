from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import gc
import importlib.util
import sys
import time

import joblib
import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from src.utils.paths import (
    artifact_path,
    data_path,
    result_path,
)

from src.features.macro_v2._common import (
    git_commit,
    load_elite_sites,
    sha256,
    write_json,
)


ROOT = (
    Path(__file__)
    .resolve()
    .parents[3]
)

OUT = Path(
    result_path(
        "final",
        "HISTORICAL-FINAL",
    )
)

ART = Path(
    artifact_path(
        "final",
        "HISTORICAL-FINAL",
    )
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

ART.mkdir(
    parents=True,
    exist_ok=True,
)


EXPECTED_N = 72603
EXPECTED_SITES = 65

MICRO_EPOCHS = 30
MACRO_CONTEXT_DAYS = 5

MACRO_ALPHA = 0.375

HYBRID_MICRO_WEIGHT = 0.45
HYBRID_MACRO_WEIGHT = 0.55


XGB_CONFIG = {
    "n_estimators": 120,
    "max_depth": 5,
    "learning_rate": 0.10,
    "min_child_weight": 1.0,
    "subsample": 0.80,
    "colsample_bytree": 0.80,
    "reg_lambda": 1.0,
    "objective": "multi:softprob",
    "num_class": 65,
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}


def load_module(path, name):

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


def numeric_frame(df, features):

    return (
        df[
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


def load_full_micro(
    mod_micro,
):

    source = Path(
        mod_micro.SOURCE
    )

    elite = {
        str(x)
        for x in
        load_elite_sites()
    }

    cols = [
        "pcap_uid",
        "site_label",
        "direction_vector",
        "size_vector",
    ]

    df = pd.read_csv(
        source,
        usecols=cols,
    )

    df[
        "site_label"
    ] = (
        df[
            "site_label"
        ]
        .astype(str)
    )

    df = (
        df[
            df[
                "site_label"
            ].isin(
                elite
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    if len(df) != EXPECTED_N:
        raise RuntimeError(
            f"Expected {EXPECTED_N:,} Historical "
            f"Micro captures, found {len(df):,}."
        )

    if df[
        "pcap_uid"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate Historical Micro pcap_uid."
        )

    labels = np.array(
        sorted(
            elite
        ),
        dtype=str,
    )

    if len(labels) != EXPECTED_SITES:
        raise RuntimeError(
            "Expected 65 classes."
        )

    label_to_idx = {
        label: i
        for i, label in
        enumerate(
            labels
        )
    }

    y = np.array(
        [
            label_to_idx[
                x
            ]
            for x in
            df[
                "site_label"
            ]
        ],
        dtype=np.int64,
    )

    n = len(df)

    x_dir = np.zeros(
        (
            n,
            mod_micro.MAX_LEN,
        ),
        dtype=np.int8,
    )

    x_size = np.zeros(
        (
            n,
            mod_micro.MAX_LEN,
        ),
        dtype=np.float32,
    )

    print(
        f"Parsing {n:,} full-Historical Micro captures..."
    )

    for i, (
        direction,
        size,
    ) in enumerate(
        zip(
            df[
                "direction_vector"
            ],
            df[
                "size_vector"
            ],
        )
    ):

        x_dir[i] = (
            mod_micro.parse_vector(
                direction,
                mod_micro.MAX_LEN,
                np.int8,
            )
        )

        x_size[i] = (
            mod_micro.parse_vector(
                size,
                mod_micro.MAX_LEN,
                np.float32,
            )
        )

    return (
        df,
        x_dir,
        x_size,
        y,
        labels,
        source,
    )


def fit_xgb_data(
    df,
    features,
):

    x = numeric_frame(
        df,
        features,
    )

    medians = x.median(
        axis=0,
        skipna=True,
    )

    missing = (
        medians[
            medians.isna()
        ]
        .index
        .tolist()
    )

    if missing:
        raise RuntimeError(
            f"All-missing Macro features: {missing}"
        )

    medians = medians.fillna(
        0.0
    )

    x = (
        x
        .fillna(
            medians
        )
        .to_numpy(
            dtype=np.float32
        )
    )

    if not np.isfinite(x).all():
        raise RuntimeError(
            "Non-finite XGB matrix."
        )

    return (
        medians,
        x,
    )


def build_full_daily_profiles(
    captures,
    features,
):

    numeric = numeric_frame(
        captures,
        features,
    )

    work = captures[
        [
            "site_label",
            "_query_date",
        ]
    ].copy()

    for feature in features:
        work[
            feature
        ] = numeric[
            feature
        ].to_numpy()

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

    daily = (
        daily
        .sort_values(
            [
                "date",
                "site_label",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    if daily[
        [
            "site_label",
            "date",
        ]
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate Historical site-day profile."
        )

    return daily


def main():

    print("=" * 78)
    print(
        "10A — FULL HISTORICAL FINAL REFIT"
    )
    print("=" * 78)

    print(
        "Hyperparameters remain frozen."
    )

    print(
        "Historical includes the previously evaluated "
        "INTERNAL_TEST period."
    )

    print(
        "No holdout result is used for tuning."
    )

    print(
        "Future-B numerical values are NOT used."
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

    mod_micro = load_module(
        ROOT
        / "src/models/micro/"
        "08B_clean_temporal_micro_baseline.py",
        "micro_full_hist_10a",
    )

    mod_macro = load_module(
        ROOT
        / "src/features/macro_v2/"
        "07A_candidate_conditioned_temporal_encoder.py",
        "macro_full_hist_10a",
    )

    # =====================================================
    # MICRO-FINAL
    # =====================================================

    print()
    print("=" * 78)
    print(
        "MICRO-FINAL — FULL HISTORICAL"
    )
    print("=" * 78)

    (
        micro_df,
        x_dir,
        x_size,
        y_micro,
        micro_labels,
        micro_source,
    ) = load_full_micro(
        mod_micro
    )

    micro_idx = np.arange(
        len(
            y_micro
        ),
        dtype=np.int64,
    )

    t0 = time.perf_counter()

    (
        micro_model,
        micro_min,
        micro_max,
        _,
        micro_curve,
    ) = mod_micro.train(
        x_dir,
        x_size,
        y_micro,
        micro_idx,
        device,
        MICRO_EPOCHS,
        val_idx=None,
    )

    micro_seconds = (
        time.perf_counter()
        -
        t0
    )

    micro_checkpoint = (
        ART
        / "MICRO-FINAL_HISTORICAL.pth"
    )

    torch.save(
        {
            "state_dict":
                micro_model.state_dict(),

            "candidate_labels":
                micro_labels.tolist(),

            "max_len":
                mod_micro.MAX_LEN,

            "size_min":
                micro_min,

            "size_max":
                micro_max,

            "epochs":
                MICRO_EPOCHS,

            "seed":
                mod_micro.SEED,

            "training_population":
                "FULL_HISTORICAL",
        },
        micro_checkpoint,
    )

    micro_curve.to_csv(
        OUT
        / "10A_micro_train_curve.csv",
        index=False,
    )

    print(
        "Micro fit seconds:",
        micro_seconds,
    )

    del (
        micro_model,
        micro_df,
        x_dir,
        x_size,
        y_micro,
        micro_idx,
    )

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # =====================================================
    # MACRO-FINAL
    # =====================================================

    print()
    print("=" * 78)
    print(
        "MACRO-LTD-FINAL — FULL HISTORICAL"
    )
    print("=" * 78)

    (
        features,
        feature_path,
    ) = (
        mod_macro.load_frozen_features()
    )

    (
        captures,
        historical_path,
    ) = (
        mod_macro.load_historical_65()
    )

    if len(
        captures
    ) != EXPECTED_N:
        raise RuntimeError(
            f"Expected {EXPECTED_N:,} Macro captures, "
            f"found {len(captures):,}."
        )

    captures[
        "site_label"
    ] = (
        captures[
            "site_label"
        ]
        .astype(str)
    )

    dates, date_source = (
        mod_macro.derive_dates(
            captures
        )
    )

    captures[
        "_query_date"
    ] = pd.to_datetime(
        dates
    )

    if captures[
        "_query_date"
    ].isna().any():
        raise RuntimeError(
            "Missing Historical dates."
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
            "Micro/Macro label-order mismatch."
        )

    label_to_idx = (
        mod_macro.make_label_mapping(
            macro_labels
        )
    )

    y_macro = (
        mod_macro.labels_to_indices(
            captures[
                "site_label"
            ],
            label_to_idx,
        )
    )

    print(
        "Historical date range:",
        captures[
            "_query_date"
        ].min(),
        "->",
        captures[
            "_query_date"
        ].max(),
    )

    # --------------------------
    # XGB
    # --------------------------

    (
        xgb_medians,
        x_xgb,
    ) = fit_xgb_data(
        captures,
        features,
    )

    xgb_model = (
        xgb.XGBClassifier(
            **XGB_CONFIG
        )
    )

    t0 = time.perf_counter()

    xgb_model.fit(
        x_xgb,
        y_macro,
    )

    xgb_seconds = (
        time.perf_counter()
        -
        t0
    )

    xgb_path = (
        ART
        / "MACRO-FINAL_XGB_HISTORICAL.joblib"
    )

    joblib.dump(
        {
            "model":
                xgb_model,

            "medians":
                xgb_medians,

            "features":
                features,

            "candidate_labels":
                macro_labels,

            "config":
                XGB_CONFIG,
        },
        xgb_path,
    )

    del x_xgb

    gc.collect()

    # --------------------------
    # LTD preprocessing
    # --------------------------

    (
        ltd_medians,
        ltd_scaler,
        ltd_missing,
    ) = (
        mod_macro.fit_preprocessing(
            captures,
            features,
        )
    )

    if ltd_missing:
        raise RuntimeError(
            f"All-missing LTD features: {ltd_missing}"
        )

    x_ltd = (
        mod_macro.transform(
            captures,
            features,
            ltd_medians,
            ltd_scaler,
        )
    )

    daily = build_full_daily_profiles(
        captures,
        features,
    )

    raw_daily_path = (
        OUT
        / "10A_historical_daily_profiles.csv"
    )

    daily.to_csv(
        raw_daily_path,
        index=False,
    )

    standardized_daily = (
        mod_macro.standardized_daily_frame(
            daily,
            features,
            ltd_medians,
            ltd_scaler,
        )
    )

    standardized_daily_path = (
        OUT
        / "10A_historical_standardized_daily_history.csv"
    )

    standardized_daily.to_csv(
        standardized_daily_path,
        index=False,
    )

    mod_macro.CONTEXT_DAYS = (
        MACRO_CONTEXT_DAYS
    )

    train_dates = (
        pd.to_datetime(
            captures[
                "_query_date"
            ]
        )
        .to_numpy()
    )

    (
        train_context,
        skipped_dates,
    ) = (
        mod_macro.build_context_cache(
            standardized_daily,
            macro_labels,
            train_dates,
            features,
        )
    )

    if not train_context:
        raise RuntimeError(
            "No causal Historical context."
        )

    preprocessing_path = (
        ART
        / "MACRO-FINAL_LTD_PREPROCESSING_HISTORICAL.joblib"
    )

    joblib.dump(
        {
            "medians":
                ltd_medians,

            "scaler":
                ltd_scaler,

            "features":
                features,

            "candidate_labels":
                macro_labels,

            "context_days":
                MACRO_CONTEXT_DAYS,

            "date_source":
                date_source,
        },
        preprocessing_path,
    )

    ltd_rows = []
    ltd_paths = []

    for seed in mod_macro.SEEDS:

        print()
        print(
            f"LTD seed={seed}"
        )

        mod_macro.set_seed(
            seed
        )

        model = (
            mod_macro.LTDPairScorer()
            .to(
                device
            )
        )

        t0 = time.perf_counter()

        (
            loss,
            n_queries,
            n_context_dates,
        ) = (
            mod_macro.train_ltd_model(
                model,
                x_ltd,
                y_macro,
                train_dates,
                train_context,
                device,
                seed,
            )
        )

        seconds = (
            time.perf_counter()
            -
            t0
        )

        checkpoint = (
            ART
            / (
                "MACRO-FINAL_LTD_"
                f"seed{seed}_HISTORICAL.pth"
            )
        )

        torch.save(
            {
                "state_dict":
                    model.state_dict(),

                "seed":
                    seed,

                "candidate_labels":
                    macro_labels.tolist(),

                "context_days":
                    MACRO_CONTEXT_DAYS,

                "features":
                    features,

                "training_population":
                    "FULL_HISTORICAL",
            },
            checkpoint,
        )

        ltd_paths.append(
            checkpoint
        )

        ltd_rows.append(
            {
                "seed":
                    seed,

                "train_loss":
                    loss,

                "n_train_queries":
                    n_queries,

                "n_train_context_dates":
                    n_context_dates,

                "fit_seconds":
                    seconds,
            }
        )

        print(
            "loss=",
            loss,
            "queries=",
            n_queries,
            "context_dates=",
            n_context_dates,
        )

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ltd_table = pd.DataFrame(
        ltd_rows
    )

    ltd_table.to_csv(
        OUT
        / "10A_macro_ltd_training.csv",
        index=False,
    )

    manifest = {
        "stage":
            "10A_refit_full_historical",

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "git_commit":
            git_commit(),

        "population": {
            "captures":
                len(
                    captures
                ),

            "sites":
                len(
                    macro_labels
                ),

            "date_min":
                str(
                    captures[
                        "_query_date"
                    ].min().date()
                ),

            "date_max":
                str(
                    captures[
                        "_query_date"
                    ].max().date()
                ),
        },

        "frozen_configuration": {
            "micro_epochs":
                MICRO_EPOCHS,

            "micro_seed":
                mod_micro.SEED,

            "macro_context_days":
                MACRO_CONTEXT_DAYS,

            "macro_ltd_seeds":
                [
                    int(x)
                    for x in
                    mod_macro.SEEDS
                ],

            "macro_internal_alpha":
                MACRO_ALPHA,

            "hybrid_micro_weight":
                HYBRID_MICRO_WEIGHT,

            "hybrid_macro_weight":
                HYBRID_MACRO_WEIGHT,
        },

        "macro_causal_training": {
            "skipped_dates":
                [
                    str(
                        pd.Timestamp(x).date()
                    )
                    for x in
                    skipped_dates
                ],
        },

        "data_policy": {
            "internal_test_previously_evaluated":
                True,

            "internal_test_now_in_training_population":
                True,

            "future_values_used":
                False,

            "hyperparameter_tuning":
                False,
        },

        "inputs": {
            "historical_macro": {
                "path":
                    str(
                        historical_path
                    ),

                "sha256":
                    sha256(
                        historical_path
                    ),
            },

            "historical_micro": {
                "path":
                    str(
                        micro_source
                    ),

                "sha256":
                    sha256(
                        micro_source
                    ),
            },

            "frozen_features": {
                "path":
                    str(
                        feature_path
                    ),

                "sha256":
                    sha256(
                        feature_path
                    ),
            },
        },

        "artifacts": {
            "micro_checkpoint": {
                "path":
                    str(
                        micro_checkpoint
                    ),

                "sha256":
                    sha256(
                        micro_checkpoint
                    ),
            },

            "macro_xgb": {
                "path":
                    str(
                        xgb_path
                    ),

                "sha256":
                    sha256(
                        xgb_path
                    ),
            },

            "macro_ltd_preprocessing": {
                "path":
                    str(
                        preprocessing_path
                    ),

                "sha256":
                    sha256(
                        preprocessing_path
                    ),
            },

            "historical_daily_profiles": {
                "path":
                    str(
                        raw_daily_path
                    ),

                "sha256":
                    sha256(
                        raw_daily_path
                    ),
            },

            "historical_standardized_daily_history": {
                "path":
                    str(
                        standardized_daily_path
                    ),

                "sha256":
                    sha256(
                        standardized_daily_path
                    ),
            },

            "macro_ltd_checkpoints": [
                {
                    "path":
                        str(p),

                    "sha256":
                        sha256(p),
                }
                for p in
                ltd_paths
            ],
        },

        "next":
            (
                "10B frozen Future-B "
                "Concept Drift evaluation"
            ),
    }

    write_json(
        OUT
        / "10A_manifest_full_historical_refit.json",
        manifest,
    )

    print()
    print("=" * 78)
    print(
        "10A COMPLETE"
    )
    print("=" * 78)

    print(
        ltd_table.to_string(
            index=False
        )
    )

    print()
    print(
        "Micro checkpoint:",
        sha256(
            micro_checkpoint
        ),
    )

    print(
        "Macro XGB:",
        sha256(
            xgb_path
        ),
    )

    print()
    print(
        "Future-B remains numerically UNUSED."
    )


if __name__ == "__main__":
    main()
