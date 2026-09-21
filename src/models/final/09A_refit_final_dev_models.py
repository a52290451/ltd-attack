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
    result_path,
)

from src.features.macro_v2._common import (
    git_commit,
    sha256,
    write_json,
)


DEV_SPLITS = [
    "DEV_EARLY",
    "DEV_MIDDLE",
    "DEV_LATE",
]

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


ROOT = (
    Path(__file__)
    .resolve()
    .parents[3]
)

OUT = Path(
    result_path(
        "final",
        "DEV-FINAL",
    )
)

ART = Path(
    artifact_path(
        "final",
        "DEV-FINAL",
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


def numeric_frame(
    df,
    features,
):
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
            f"All-missing XGB features: {missing}"
        )

    medians = (
        medians
        .fillna(
            0.0
        )
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

    if not np.isfinite(
        x
    ).all():
        raise RuntimeError(
            "Non-finite XGB DEV matrix."
        )

    return (
        medians,
        x,
    )


def main():

    print("=" * 78)
    print(
        "09A — FINAL ALL-DEV REFIT"
    )
    print("=" * 78)

    print(
        "Architecture/hyperparameters are FROZEN."
    )

    print(
        "Training population:"
    )

    print(
        "DEV_EARLY + DEV_MIDDLE + DEV_LATE"
    )

    print(
        "INTERNAL_TEST numerical values are NOT used."
    )

    print(
        "Future-B: CLOSED"
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

    # =====================================================
    # LOAD FROZEN MODULES
    # =====================================================

    mod_micro = load_module(
        ROOT
        / "src/models/micro/"
        "08B_clean_temporal_micro_baseline.py",
        "micro_08b_final",
    )

    mod_macro = load_module(
        ROOT
        / "src/features/macro_v2/"
        "07A_candidate_conditioned_temporal_encoder.py",
        "macro_07a_final",
    )

    # =====================================================
    # MICRO-FINAL — REFIT ON ALL DEV
    # =====================================================

    print()
    print("=" * 78)
    print(
        "MICRO-FINAL — ALL DEV REFIT"
    )
    print("=" * 78)

    (
        micro_df,
        x_dir,
        x_size,
        y_micro,
        micro_labels,
    ) = (
        mod_micro
        .load_dev_vectors()
    )

    if len(
        micro_df
    ) != 57916:
        raise RuntimeError(
            "Expected 57,916 Micro DEV captures."
        )

    micro_idx = np.arange(
        len(
            micro_df
        ),
        dtype=np.int64,
    )

    t0 = time.perf_counter()

    (
        micro_model,
        micro_size_min,
        micro_size_max,
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
        / "MICRO-FINAL_DEV.pth"
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
                micro_size_min,

            "size_max":
                micro_size_max,

            "epochs":
                MICRO_EPOCHS,

            "seed":
                mod_micro.SEED,

            "training_population":
                DEV_SPLITS,
        },
        micro_checkpoint,
    )

    micro_curve_path = (
        OUT
        / "09A_micro_train_curve.csv"
    )

    micro_curve.to_csv(
        micro_curve_path,
        index=False,
    )

    print(
        "Micro checkpoint:",
        micro_checkpoint,
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
    # MACRO-LTD-FINAL — ALL DEV REFIT
    # =====================================================

    print()
    print("=" * 78)
    print(
        "MACRO-LTD-FINAL — ALL DEV REFIT"
    )
    print("=" * 78)

    (
        features,
        feature_path,
    ) = (
        mod_macro
        .load_frozen_features()
    )

    assignments_path = (
        result_path(
            "feature_engineering",
            "MACRO-V2-HIST",
            "01_temporal_split_assignments.csv",
        )
    )

    daily_path = (
        result_path(
            "feature_engineering",
            "MACRO-V2-HIST",
            "06C_daily_base_profiles.csv",
        )
    )

    assignments = pd.read_csv(
        assignments_path
    )

    if assignments[
        "pcap_uid"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate temporal assignment UID."
        )

    dev_assignments = (
        assignments[
            assignments[
                "temporal_split"
            ].isin(
                DEV_SPLITS
            )
        ][
            [
                "pcap_uid",
                "temporal_split",
            ]
        ]
        .copy()
    )

    (
        captures,
        historical_path,
    ) = (
        mod_macro
        .load_historical_65()
    )

    captures = captures.merge(
        dev_assignments,
        on="pcap_uid",
        how="inner",
        validate="one_to_one",
    )

    if len(
        captures
    ) != 57916:
        raise RuntimeError(
            "Expected 57,916 Macro DEV captures."
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
        mod_macro
        .derive_dates(
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
            "Missing DEV Macro query date."
        )

    macro_labels = np.array(
        sorted(
            captures[
                "site_label"
            ].unique()
        ),
        dtype=str,
    )

    if len(
        macro_labels
    ) != 65:
        raise RuntimeError(
            "Expected 65 Macro classes."
        )

    if not np.array_equal(
        micro_labels,
        macro_labels,
    ):
        raise RuntimeError(
            "Micro/Macro candidate order mismatch."
        )

    label_to_idx = (
        mod_macro
        .make_label_mapping(
            macro_labels
        )
    )

    y_macro = (
        mod_macro
        .labels_to_indices(
            captures[
                "site_label"
            ],
            label_to_idx,
        )
    )

    # -------------------------
    # XGBoost BASE
    # -------------------------

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

    xgb_artifact = (
        ART
        / "MACRO-FINAL_XGB_DEV.joblib"
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
        xgb_artifact,
    )

    print(
        "XGB fit seconds:",
        xgb_seconds,
    )

    del x_xgb

    gc.collect()

    # -------------------------
    # LTD preprocessing
    # -------------------------

    (
        ltd_medians,
        ltd_scaler,
        ltd_missing,
    ) = (
        mod_macro
        .fit_preprocessing(
            captures,
            features,
        )
    )

    if ltd_missing:
        raise RuntimeError(
            f"All-missing LTD features: {ltd_missing}"
        )

    x_ltd = (
        mod_macro
        .transform(
            captures,
            features,
            ltd_medians,
            ltd_scaler,
        )
    )

    train_dates = (
        pd.to_datetime(
            captures[
                "_query_date"
            ]
        )
        .to_numpy()
    )

    daily = pd.read_csv(
        daily_path
    )

    daily[
        "site_label"
    ] = (
        daily[
            "site_label"
        ]
        .astype(str)
    )

    daily[
        "date"
    ] = pd.to_datetime(
        daily[
            "date"
        ]
    )

    observed_daily_splits = set(
        daily[
            "temporal_split"
        ].unique()
    )

    if observed_daily_splits != set(
        DEV_SPLITS
    ):
        raise RuntimeError(
            "Unexpected daily DEV split population."
        )

    daily_history = (
        mod_macro
        .standardized_daily_frame(
            daily,
            features,
            ltd_medians,
            ltd_scaler,
        )
    )

    daily_history_path = (
        OUT
        / "09A_macro_standardized_daily_history.csv"
    )

    daily_history.to_csv(
        daily_history_path,
        index=False,
    )

    mod_macro.CONTEXT_DAYS = (
        MACRO_CONTEXT_DAYS
    )

    (
        train_context,
        skipped_train_dates,
    ) = (
        mod_macro
        .build_context_cache(
            daily_history,
            macro_labels,
            train_dates,
            features,
        )
    )

    if not train_context:
        raise RuntimeError(
            "No causal LTD training context."
        )

    ltd_preprocessing_artifact = (
        ART
        / "MACRO-FINAL_LTD_PREPROCESSING_DEV.joblib"
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
        ltd_preprocessing_artifact,
    )

    seed_rows = []
    ltd_checkpoints = []

    for seed in (
        mod_macro.SEEDS
    ):

        print()
        print(
            f"LTD final seed={seed}"
        )

        mod_macro.set_seed(
            seed
        )

        model = (
            mod_macro
            .LTDPairScorer()
            .to(
                device
            )
        )

        t0 = time.perf_counter()

        (
            loss,
            n_train_queries,
            n_train_context_dates,
        ) = (
            mod_macro
            .train_ltd_model(
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
                f"seed{seed}_DEV.pth"
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
                    DEV_SPLITS,
            },
            checkpoint,
        )

        ltd_checkpoints.append(
            checkpoint
        )

        seed_rows.append(
            {
                "seed":
                    seed,

                "train_loss":
                    loss,

                "n_train_queries":
                    n_train_queries,

                "n_train_context_dates":
                    n_train_context_dates,

                "fit_seconds":
                    seconds,
            }
        )

        print(
            "loss=",
            loss,
            "queries=",
            n_train_queries,
            "context_dates=",
            n_train_context_dates,
            "seconds=",
            seconds,
        )

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    seed_path = (
        OUT
        / "09A_macro_ltd_seed_training.csv"
    )

    pd.DataFrame(
        seed_rows
    ).to_csv(
        seed_path,
        index=False,
    )

    # =====================================================
    # MANIFEST
    # =====================================================

    artifacts = {
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
                    xgb_artifact
                ),

            "sha256":
                sha256(
                    xgb_artifact
                ),
        },

        "macro_ltd_preprocessing": {
            "path":
                str(
                    ltd_preprocessing_artifact
                ),

            "sha256":
                sha256(
                    ltd_preprocessing_artifact
                ),
        },

        "macro_daily_history": {
            "path":
                str(
                    daily_history_path
                ),

            "sha256":
                sha256(
                    daily_history_path
                ),
        },

        "macro_ltd_checkpoints": [
            {
                "path":
                    str(
                        p
                    ),

                "sha256":
                    sha256(
                        p
                    ),
            }
            for p in
            ltd_checkpoints
        ],
    }

    manifest = {
        "stage":
            "09A_refit_final_dev_models",

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "git_commit":
            git_commit(),

        "training_population": {
            "splits":
                DEV_SPLITS,

            "captures":
                57916,

            "sites":
                65,
        },

        "micro_final": {
            "epochs":
                MICRO_EPOCHS,

            "seed":
                mod_micro.SEED,

            "size_min":
                micro_size_min,

            "size_max":
                micro_size_max,

            "fit_seconds":
                micro_seconds,
        },

        "macro_final": {
            "base":
                "XGBoost",

            "context_days":
                MACRO_CONTEXT_DAYS,

            "ltd_seeds":
                [
                    int(x)
                    for x in
                    mod_macro.SEEDS
                ],

            "ltd_ensemble":
                "mean softmax probability",

            "internal_macro_fusion_alpha":
                MACRO_ALPHA,

            "skipped_causal_training_dates":
                [
                    str(
                        pd.Timestamp(x).date()
                    )
                    for x in
                    skipped_train_dates
                ],
        },

        "hybrid_final": {
            "type":
                "weighted geometric probability",

            "micro_weight":
                HYBRID_MICRO_WEIGHT,

            "macro_weight":
                HYBRID_MACRO_WEIGHT,
        },

        "data_policy": {
            "development_only":
                True,

            "internal_test_values_used":
                False,

            "external_future_used":
                False,

            "source_files_may_physically_contain_internal_rows":
                True,

            "internal_rows_used_in_model_computation":
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

            "temporal_assignments": {
                "path":
                    str(
                        assignments_path
                    ),

                "sha256":
                    sha256(
                        assignments_path
                    ),
            },

            "daily_profiles": {
                "path":
                    str(
                        daily_path
                    ),

                "sha256":
                    sha256(
                        daily_path
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

        "artifacts":
            artifacts,

        "next":
            (
                "09B one-time INTERNAL_TEST evaluation "
                "after artifact verification"
            ),
    }

    write_json(
        OUT
        / "09A_manifest_final_dev_refit.json",
        manifest,
    )

    print()
    print("=" * 78)
    print(
        "09A COMPLETE"
    )
    print("=" * 78)

    print(
        pd.DataFrame(
            seed_rows
        ).to_string(
            index=False
        )
    )

    print()
    print(
        "MICRO-FINAL artifact:",
        artifacts[
            "micro_checkpoint"
        ][
            "sha256"
        ],
    )

    print(
        "MACRO XGB artifact:",
        artifacts[
            "macro_xgb"
        ][
            "sha256"
        ],
    )

    print()
    print(
        "INTERNAL_TEST numerical values remain UNUSED."
    )

    print(
        "Future-B remains CLOSED."
    )


if __name__ == "__main__":
    main()
