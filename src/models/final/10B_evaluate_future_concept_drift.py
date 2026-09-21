from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import argparse
import gc
import importlib.util
import json
import sys

import joblib
import numpy as np
import pandas as pd
import torch

from src.utils.paths import (
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

DEV_RESULT = Path(
    result_path(
        "final",
        "DEV-FINAL",
    )
)

HIST_RESULT = Path(
    result_path(
        "final",
        "HISTORICAL-FINAL",
    )
)

INTERNAL_RESULT = Path(
    result_path(
        "final",
        "INTERNAL-TEST",
    )
)

OUT = Path(
    result_path(
        "final",
        "FUTURE-B",
    )
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


FUTURE_MACRO = Path(
    data_path(
        "future",
        "CLEAN_final_features_sites_concept_drift.csv",
    )
)

FUTURE_MICRO = Path(
    data_path(
        "future",
        "CLEAN_final_vectors_sites_concept_drift.csv",
    )
)


EXPECTED_N = 18543
EXPECTED_SITES = 65

EXPECTED_MACRO_ALPHA = 0.375
EXPECTED_HYBRID_MACRO_WEIGHT = 0.55

MARKER = (
    OUT
    / "10B_FUTURE_B_OPENED.json"
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


def verify_file(
    path,
    expected_hash,
    name,
):

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"{name}: {path}"
        )

    observed = sha256(
        path
    )

    if observed != expected_hash:
        raise RuntimeError(
            f"{name} hash mismatch.\n"
            f"expected={expected_hash}\n"
            f"observed={observed}"
        )

    print(
        f"PASS {name}: {observed}"
    )


def verify_manifest_artifacts(
    manifest,
    track,
):

    print()
    print(
        f"VERIFY TRACK: {track}"
    )

    artifacts = manifest[
        "artifacts"
    ]

    for key in [
        "micro_checkpoint",
        "macro_xgb",
        "macro_ltd_preprocessing",
    ]:

        verify_file(
            artifacts[
                key
            ][
                "path"
            ],
            artifacts[
                key
            ][
                "sha256"
            ],
            f"{track} {key}",
        )

    if track == "DEV_FROZEN":

        history_key = (
            "macro_daily_history"
        )

    else:

        history_key = (
            "historical_standardized_daily_history"
        )

    verify_file(
        artifacts[
            history_key
        ][
            "path"
        ],
        artifacts[
            history_key
        ][
            "sha256"
        ],
        f"{track} history bank",
    )

    checkpoints = artifacts[
        "macro_ltd_checkpoints"
    ]

    if len(
        checkpoints
    ) != 3:
        raise RuntimeError(
            f"{track}: expected three LTD checkpoints."
        )

    for item in checkpoints:

        verify_file(
            item[
                "path"
            ],
            item[
                "sha256"
            ],
            (
                f"{track} LTD checkpoint "
                f"{Path(item['path']).name}"
            ),
        )


def verify_only():

    print("=" * 78)
    print(
        "10B — FUTURE-B PRE-OPEN VERIFICATION"
    )
    print("=" * 78)

    dev_manifest_path = (
        DEV_RESULT
        / "09A_manifest_final_dev_refit.json"
    )

    hist_manifest_path = (
        HIST_RESULT
        / "10A_manifest_full_historical_refit.json"
    )

    internal_manifest_path = (
        INTERNAL_RESULT
        / "09B_manifest_internal_test.json"
    )

    internal_summary_path = (
        INTERNAL_RESULT
        / "09B_internal_summary.csv"
    )

    for p in [
        dev_manifest_path,
        hist_manifest_path,
        internal_manifest_path,
        internal_summary_path,
    ]:

        if not p.exists():
            raise FileNotFoundError(
                p
            )

    dev = json.loads(
        dev_manifest_path.read_text()
    )

    hist = json.loads(
        hist_manifest_path.read_text()
    )

    internal = json.loads(
        internal_manifest_path.read_text()
    )

    if (
        internal[
            "status"
        ]
        !=
        "INTERNAL_TEST_OPENED_AND_EVALUATED"
    ):
        raise RuntimeError(
            "09B INTERNAL_TEST status is unexpected."
        )

    verify_manifest_artifacts(
        dev,
        "DEV_FROZEN",
    )

    verify_manifest_artifacts(
        hist,
        "HISTORICAL_FINAL",
    )

    if not FUTURE_MACRO.exists():
        raise FileNotFoundError(
            FUTURE_MACRO
        )

    if not FUTURE_MICRO.exists():
        raise FileNotFoundError(
            FUTURE_MICRO
        )

    print()
    print(
        "PASS Future-B Macro path exists "
        "(contents NOT read)"
    )

    print(
        FUTURE_MACRO
    )

    print(
        "PASS Future-B Micro path exists "
        "(contents NOT read)"
    )

    print(
        FUTURE_MICRO
    )

    print()
    print(
        "Frozen Future protocol:"
    )

    print(
        "Track A: DEV_FROZEN"
    )

    print(
        "Track B: HISTORICAL_FINAL"
    )

    print(
        "Macro fusion alpha: 0.375"
    )

    print(
        "Hybrid: Micro=.45 / Macro=.55"
    )

    print(
        "Future observations will NOT update LTD history."
    )

    print()
    print(
        "VERIFY PASS."
    )

    print(
        "No Future-B numerical values were read."
    )

    return (
        dev,
        hist,
        internal,
    )


def read_future_macro(
    mod_macro,
    historical_macro_path,
):

    print()
    print(
        "Reading Future-B Macro payload..."
    )

    future = pd.read_csv(
        FUTURE_MACRO
    )

    if "pcap_uid" not in future.columns:
        raise RuntimeError(
            "Future Macro has no pcap_uid."
        )

    if "site_label" not in future.columns:

        if "site" not in future.columns:
            raise RuntimeError(
                "Future Macro has neither "
                "site_label nor site."
            )

        historic_header = pd.read_csv(
            historical_macro_path,
            nrows=0,
        ).columns.tolist()

        if (
            "site" not in historic_header
            or
            "site_label" not in historic_header
        ):
            raise RuntimeError(
                "Historical source cannot provide "
                "site -> site_label mapping."
            )

        mapping_df = (
            pd.read_csv(
                historical_macro_path,
                usecols=[
                    "site",
                    "site_label",
                ],
            )
            .dropna()
            .drop_duplicates()
        )

        conflicts = (
            mapping_df
            .groupby(
                "site"
            )[
                "site_label"
            ]
            .nunique()
        )

        if (
            conflicts > 1
        ).any():
            raise RuntimeError(
                "Non-unique historical site mapping."
            )

        site_map = dict(
            zip(
                mapping_df[
                    "site"
                ].astype(str),
                mapping_df[
                    "site_label"
                ].astype(str),
            )
        )

        future[
            "site_label"
        ] = (
            future[
                "site"
            ]
            .astype(str)
            .map(
                site_map
            )
        )

    future = future.dropna(
        subset=[
            "site_label",
        ]
    ).copy()

    future[
        "site_label"
    ] = (
        future[
            "site_label"
        ]
        .astype(str)
    )

    elite = {
        str(x)
        for x in
        load_elite_sites()
    }

    future = (
        future[
            future[
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

    if future[
        "pcap_uid"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate Future Macro pcap_uid."
        )

    if len(
        future
    ) != EXPECTED_N:
        raise RuntimeError(
            "Unexpected Future-B 65-site count: "
            f"{len(future):,}; "
            f"expected {EXPECTED_N:,}."
        )

    if (
        future[
            "site_label"
        ].nunique()
        != EXPECTED_SITES
    ):
        raise RuntimeError(
            "Expected 65 Future-B sites."
        )

    dates, date_source = (
        mod_macro.derive_dates(
            future
        )
    )

    future[
        "_query_date"
    ] = pd.to_datetime(
        dates
    )

    if future[
        "_query_date"
    ].isna().any():
        raise RuntimeError(
            "Missing Future-B dates."
        )

    print(
        "Future-B Macro:",
        len(
            future
        ),
        "captures",
    )

    print(
        "Date range:",
        future[
            "_query_date"
        ].min(),
        "->",
        future[
            "_query_date"
        ].max(),
    )

    return (
        future,
        date_source,
    )


def read_future_micro(
    mod_micro,
    macro_future,
    labels,
):

    print()
    print(
        "Reading and aligning Future-B Micro payload..."
    )

    header = pd.read_csv(
        FUTURE_MICRO,
        nrows=0,
    ).columns.tolist()

    required = [
        "pcap_uid",
        "direction_vector",
        "size_vector",
    ]

    missing = [
        x
        for x in required
        if x not in header
    ]

    if missing:
        raise RuntimeError(
            f"Future Micro missing: {missing}"
        )

    usecols = required.copy()

    if "site_label" in header:
        usecols.append(
            "site_label"
        )

    target_order = (
        macro_future[
            "pcap_uid"
        ]
        .astype(str)
        .tolist()
    )

    target = set(
        target_order
    )

    pieces = []

    for chunk in pd.read_csv(
        FUTURE_MICRO,
        usecols=usecols,
        chunksize=4000,
    ):

        chunk[
            "pcap_uid"
        ] = (
            chunk[
                "pcap_uid"
            ]
            .astype(str)
        )

        keep = chunk[
            "pcap_uid"
        ].isin(
            target
        )

        if keep.any():

            pieces.append(
                chunk.loc[
                    keep
                ].copy()
            )

    future = pd.concat(
        pieces,
        ignore_index=True,
    )

    if len(
        future
    ) != EXPECTED_N:
        raise RuntimeError(
            "Future Micro coverage mismatch: "
            f"{len(future):,}."
        )

    if future[
        "pcap_uid"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate Future Micro pcap_uid."
        )

    lookup = future.set_index(
        "pcap_uid"
    )

    if set(
        lookup.index
    ) != target:
        raise RuntimeError(
            "Future Micro/Macro UID sets differ."
        )

    future = (
        lookup
        .loc[
            target_order
        ]
        .reset_index()
    )

    macro_sites = (
        macro_future[
            "site_label"
        ]
        .astype(str)
        .to_numpy()
    )

    if "site_label" in future.columns:

        future[
            "site_label"
        ] = (
            future[
                "site_label"
            ]
            .astype(str)
        )

        if not np.array_equal(
            future[
                "site_label"
            ].to_numpy(),
            macro_sites,
        ):
            raise RuntimeError(
                "Future Micro/Macro labels differ."
            )

    else:

        future[
            "site_label"
        ] = macro_sites

    label_to_idx = {
        label: i
        for i, label
        in enumerate(
            labels
        )
    }

    y = np.array(
        [
            label_to_idx[
                site
            ]
            for site in
            future[
                "site_label"
            ]
        ],
        dtype=np.int64,
    )

    n = len(
        future
    )

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
        f"Parsing {n:,} Future-B Micro captures..."
    )

    for i, (
        direction,
        size,
    ) in enumerate(
        zip(
            future[
                "direction_vector"
            ],
            future[
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
        future,
        x_dir,
        x_size,
        y,
    )


def frozen_values(
    manifest,
    track,
):

    if track == "DEV_FROZEN":

        macro_alpha = float(
            manifest[
                "macro_final"
            ][
                "internal_macro_fusion_alpha"
            ]
        )

        hybrid_macro_weight = float(
            manifest[
                "hybrid_final"
            ][
                "macro_weight"
            ]
        )

    else:

        macro_alpha = float(
            manifest[
                "frozen_configuration"
            ][
                "macro_internal_alpha"
            ]
        )

        hybrid_macro_weight = float(
            manifest[
                "frozen_configuration"
            ][
                "hybrid_macro_weight"
            ]
        )

    if (
        abs(
            macro_alpha
            -
            EXPECTED_MACRO_ALPHA
        )
        >
        1e-12
    ):
        raise RuntimeError(
            f"{track}: Macro alpha changed."
        )

    if (
        abs(
            hybrid_macro_weight
            -
            EXPECTED_HYBRID_MACRO_WEIGHT
        )
        >
        1e-12
    ):
        raise RuntimeError(
            f"{track}: Hybrid weight changed."
        )

    return (
        macro_alpha,
        hybrid_macro_weight,
    )


def evaluate_track(
    track,
    manifest,
    macro_future,
    x_dir,
    x_size,
    y,
    labels,
    dates,
    mod_micro,
    mod_macro,
    mod09b,
    device,
):

    print()
    print("#" * 78)
    print(
        f"TRACK: {track}"
    )
    print("#" * 78)

    (
        macro_alpha,
        hybrid_macro_weight,
    ) = frozen_values(
        manifest,
        track,
    )

    artifacts = manifest[
        "artifacts"
    ]

    # ==================================================
    # MICRO
    # ==================================================

    micro_ckpt = torch.load(
        artifacts[
            "micro_checkpoint"
        ][
            "path"
        ],
        map_location=device,
        weights_only=False,
    )

    micro_labels = np.array(
        micro_ckpt[
            "candidate_labels"
        ],
        dtype=str,
    )

    if not np.array_equal(
        micro_labels,
        labels,
    ):
        raise RuntimeError(
            f"{track}: Micro class order mismatch."
        )

    micro_model = (
        mod_micro.MultimodalTransformer(
            EXPECTED_SITES
        )
        .to(
            device
        )
    )

    micro_model.load_state_dict(
        micro_ckpt[
            "state_dict"
        ]
    )

    micro_loader = (
        mod_micro.loader(
            x_dir,
            x_size,
            y,
            np.arange(
                len(y)
            ),
            False,
        )
    )

    (
        _,
        micro_probs,
        micro_y,
    ) = mod_micro.evaluate(
        micro_model,
        micro_loader,
        float(
            micro_ckpt[
                "size_min"
            ]
        ),
        float(
            micro_ckpt[
                "size_max"
            ]
        ),
        device,
    )

    if not np.array_equal(
        micro_y,
        y,
    ):
        raise RuntimeError(
            f"{track}: Micro output order mismatch."
        )

    del (
        micro_model,
        micro_loader,
    )

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ==================================================
    # XGB
    # ==================================================

    xgb_art = joblib.load(
        artifacts[
            "macro_xgb"
        ][
            "path"
        ]
    )

    xgb_labels = np.array(
        xgb_art[
            "candidate_labels"
        ],
        dtype=str,
    )

    if not np.array_equal(
        xgb_labels,
        labels,
    ):
        raise RuntimeError(
            f"{track}: XGB class order mismatch."
        )

    features = list(
        xgb_art[
            "features"
        ]
    )

    missing_features = [
        feature
        for feature in features
        if feature
        not in macro_future.columns
    ]

    if missing_features:
        raise RuntimeError(
            f"Future Macro missing frozen features: "
            f"{missing_features}"
        )

    x_xgb = (
        mod09b.numeric_frame(
            macro_future,
            features,
        )
        .fillna(
            xgb_art[
                "medians"
            ]
        )
        .to_numpy(
            dtype=np.float32
        )
    )

    if not np.isfinite(
        x_xgb
    ).all():
        raise RuntimeError(
            f"{track}: non-finite XGB Future values."
        )

    xgb_probs = (
        xgb_art[
            "model"
        ]
        .predict_proba(
            x_xgb
        )
        .astype(
            np.float64
        )
    )

    del x_xgb

    # ==================================================
    # LTD
    # ==================================================

    ltd_pre = joblib.load(
        artifacts[
            "macro_ltd_preprocessing"
        ][
            "path"
        ]
    )

    ltd_labels = np.array(
        ltd_pre[
            "candidate_labels"
        ],
        dtype=str,
    )

    if not np.array_equal(
        ltd_labels,
        labels,
    ):
        raise RuntimeError(
            f"{track}: LTD class order mismatch."
        )

    if list(
        ltd_pre[
            "features"
        ]
    ) != features:
        raise RuntimeError(
            f"{track}: XGB/LTD feature mismatch."
        )

    context_days = int(
        ltd_pre[
            "context_days"
        ]
    )

    if context_days != 5:
        raise RuntimeError(
            f"{track}: expected W=5."
        )

    mod_macro.CONTEXT_DAYS = (
        context_days
    )

    x_ltd = mod_macro.transform(
        macro_future,
        features,
        ltd_pre[
            "medians"
        ],
        ltd_pre[
            "scaler"
        ],
    )

    if track == "DEV_FROZEN":

        history_path = Path(
            artifacts[
                "macro_daily_history"
            ][
                "path"
            ]
        )

    else:

        history_path = Path(
            artifacts[
                "historical_standardized_daily_history"
            ][
                "path"
            ]
        )

    history = pd.read_csv(
        history_path
    )

    history[
        "site_label"
    ] = (
        history[
            "site_label"
        ]
        .astype(str)
    )

    history[
        "date"
    ] = pd.to_datetime(
        history[
            "date"
        ]
    )

    if (
        history[
            "date"
        ].max()
        >=
        pd.Timestamp(
            dates.min()
        )
    ):
        raise RuntimeError(
            f"{track}: Future leakage in history bank."
        )

    (
        context_cache,
        skipped,
    ) = mod_macro.build_context_cache(
        history,
        labels,
        dates,
        features,
    )

    if skipped:
        raise RuntimeError(
            f"{track}: missing Future contexts: "
            f"{skipped}"
        )

    ltd_seed_probs = []

    for item in artifacts[
        "macro_ltd_checkpoints"
    ]:

        checkpoint = torch.load(
            item[
                "path"
            ],
            map_location=device,
            weights_only=False,
        )

        ck_labels = np.array(
            checkpoint[
                "candidate_labels"
            ],
            dtype=str,
        )

        if not np.array_equal(
            ck_labels,
            labels,
        ):
            raise RuntimeError(
                f"{track}: LTD checkpoint order mismatch."
            )

        model = (
            mod_macro.LTDPairScorer()
            .to(
                device
            )
        )

        model.load_state_dict(
            checkpoint[
                "state_dict"
            ]
        )

        logits = mod09b.predict_ltd(
            mod_macro,
            model,
            x_ltd,
            dates,
            context_cache,
            device,
        )

        probs = mod09b.softmax_numpy(
            logits
        )

        ltd_seed_probs.append(
            probs
        )

        print(
            track,
            "LTD seed evaluated:",
            checkpoint[
                "seed"
            ],
        )

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ltd_probs = (
        np.stack(
            ltd_seed_probs,
            axis=0,
        )
        .mean(
            axis=0
        )
    )

    ltd_probs /= ltd_probs.sum(
        axis=1,
        keepdims=True,
    )

    macro_probs = (
        mod09b.geometric_fusion(
            xgb_probs,
            ltd_probs,
            macro_alpha,
        )
    )

    hybrid_probs = (
        mod09b.geometric_fusion(
            micro_probs,
            macro_probs,
            hybrid_macro_weight,
        )
    )

    return {
        "MICRO_FINAL":
            micro_probs,

        "MACRO_XGB":
            xgb_probs,

        "MACRO_LTD":
            ltd_probs,

        "MACRO_LTD_FINAL":
            macro_probs,

        "LTD_HYBRID_FINAL":
            hybrid_probs,
    }


def complementarity_row(
    track,
    probs,
    y,
    mod09b,
):

    _, micro_pred, _ = (
        mod09b.metrics(
            probs[
                "MICRO_FINAL"
            ],
            y,
        )
    )

    _, macro_pred, _ = (
        mod09b.metrics(
            probs[
                "MACRO_LTD_FINAL"
            ],
            y,
        )
    )

    _, hybrid_pred, _ = (
        mod09b.metrics(
            probs[
                "LTD_HYBRID_FINAL"
            ],
            y,
        )
    )

    micro_correct = (
        micro_pred
        ==
        y
    )

    macro_correct = (
        macro_pred
        ==
        y
    )

    hybrid_correct = (
        hybrid_pred
        ==
        y
    )

    both_correct = (
        micro_correct
        &
        macro_correct
    )

    micro_only = (
        micro_correct
        &
        ~macro_correct
    )

    macro_only = (
        ~micro_correct
        &
        macro_correct
    )

    both_wrong = (
        ~micro_correct
        &
        ~macro_correct
    )

    micro_errors = int(
        (
            ~micro_correct
        ).sum()
    )

    return {
        "track":
            track,

        "n":
            len(y),

        "both_correct_n":
            int(
                both_correct.sum()
            ),

        "micro_only_correct_n":
            int(
                micro_only.sum()
            ),

        "macro_only_correct_n":
            int(
                macro_only.sum()
            ),

        "both_wrong_n":
            int(
                both_wrong.sum()
            ),

        "top1_union_oracle_accuracy":
            float(
                np.mean(
                    micro_correct
                    |
                    macro_correct
                )
            ),

        "macro_rescue_rate_given_micro_wrong":
            (
                float(
                    macro_only.sum()
                    /
                    micro_errors
                )
                if micro_errors
                else np.nan
            ),

        "branch_prediction_agreement_rate":
            float(
                np.mean(
                    micro_pred
                    ==
                    macro_pred
                )
            ),

        "hybrid_correct_when_both_branches_wrong_n":
            int(
                (
                    hybrid_correct
                    &
                    both_wrong
                ).sum()
            ),

        "hybrid_accuracy":
            float(
                hybrid_correct.mean()
            ),

        "hybrid_gain_over_micro":
            float(
                hybrid_correct.mean()
                -
                micro_correct.mean()
            ),

        "hybrid_gain_over_macro":
            float(
                hybrid_correct.mean()
                -
                macro_correct.mean()
            ),
    }


def main(
    open_future,
    allow_repeat,
):

    (
        dev_manifest,
        hist_manifest,
        internal_manifest,
    ) = verify_only()

    if not open_future:
        return

    if (
        MARKER.exists()
        and
        not allow_repeat
    ):
        raise RuntimeError(
            "Future-B has already been opened.\n"
            f"Marker: {MARKER}\n"
            "Refusing accidental repeat."
        )

    write_json(
        MARKER,
        {
            "stage":
                "10B_future_b_opening",

            "opened_at_utc":
                datetime.now(
                    timezone.utc
                ).isoformat(),

            "git_commit":
                git_commit(),

            "future_macro":
                str(
                    FUTURE_MACRO
                ),

            "future_micro":
                str(
                    FUTURE_MICRO
                ),

            "purpose":
                (
                    "one-time frozen Concept Drift "
                    "evaluation"
                ),

            "model_changes_after_open":
                False,
        },
    )

    print()
    print("!" * 78)
    print(
        "FUTURE-B / CONCEPT DRIFT IS NOW OPEN"
    )
    print("!" * 78)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    mod_micro = load_module(
        ROOT
        / "src/models/micro/"
        "08B_clean_temporal_micro_baseline.py",
        "micro_future_10b",
    )

    mod_macro = load_module(
        ROOT
        / "src/features/macro_v2/"
        "07A_candidate_conditioned_temporal_encoder.py",
        "macro_future_10b",
    )

    mod09b = load_module(
        ROOT
        / "src/models/final/"
        "09B_evaluate_internal_test.py",
        "eval_internal_helpers_10b",
    )

    historical_macro_path = Path(
        hist_manifest[
            "inputs"
        ][
            "historical_macro"
        ][
            "path"
        ]
    )

    (
        macro_future,
        date_source,
    ) = read_future_macro(
        mod_macro,
        historical_macro_path,
    )

    labels = np.array(
        sorted(
            macro_future[
                "site_label"
            ].unique()
        ),
        dtype=str,
    )

    if len(labels) != EXPECTED_SITES:
        raise RuntimeError(
            "Expected 65 Future labels."
        )

    label_to_idx = (
        mod_macro.make_label_mapping(
            labels
        )
    )

    y_macro = (
        mod_macro.labels_to_indices(
            macro_future[
                "site_label"
            ],
            label_to_idx,
        )
    )

    (
        micro_future,
        x_dir,
        x_size,
        y_micro,
    ) = read_future_micro(
        mod_micro,
        macro_future,
        labels,
    )

    if not np.array_equal(
        y_macro,
        y_micro,
    ):
        raise RuntimeError(
            "Future Micro/Macro y mismatch."
        )

    dates = (
        pd.to_datetime(
            macro_future[
                "_query_date"
            ]
        )
        .to_numpy(
            dtype="datetime64[D]"
        )
    )

    tracks = {
        "DEV_FROZEN":
            dev_manifest,

        "HISTORICAL_FINAL":
            hist_manifest,
    }

    all_probs = {}
    summary_rows = []
    complementarity_rows = []
    bootstrap_rows = []
    day_rows = []
    class_rows = []

    for track, manifest in (
        tracks.items()
    ):

        probs = evaluate_track(
            track,
            manifest,
            macro_future,
            x_dir,
            x_size,
            y_macro,
            labels,
            dates,
            mod_micro,
            mod_macro,
            mod09b,
            device,
        )

        all_probs[
            track
        ] = probs

        predictions = {}

        for model_name, p in (
            probs.items()
        ):

            (
                metric,
                pred,
                rank,
            ) = mod09b.metrics(
                p,
                y_macro,
            )

            summary_rows.append(
                {
                    "track":
                        track,

                    "model":
                        model_name,

                    "n":
                        len(
                            y_macro
                        ),

                    **metric,
                }
            )

            predictions[
                model_name
            ] = (
                pred,
                rank,
            )

        complementarity_rows.append(
            complementarity_row(
                track,
                probs,
                y_macro,
                mod09b,
            )
        )

        for (
            comparison,
            reference,
            candidate,
        ) in [
            (
                "HYBRID_vs_MICRO",
                probs[
                    "MICRO_FINAL"
                ],
                probs[
                    "LTD_HYBRID_FINAL"
                ],
            ),
            (
                "HYBRID_vs_MACRO_FINAL",
                probs[
                    "MACRO_LTD_FINAL"
                ],
                probs[
                    "LTD_HYBRID_FINAL"
                ],
            ),
            (
                "MACRO_FINAL_vs_XGB",
                probs[
                    "MACRO_XGB"
                ],
                probs[
                    "MACRO_LTD_FINAL"
                ],
            ),
        ]:

            rows = mod09b.bootstrap_pair(
                (
                    track
                    +
                    "::"
                    +
                    comparison
                ),
                y_macro,
                dates,
                reference,
                candidate,
            )

            bootstrap_rows.extend(
                rows
            )

        for date in sorted(
            np.unique(
                dates
            )
        ):

            idx = np.flatnonzero(
                dates
                ==
                date
            )

            for model_name, p in (
                probs.items()
            ):

                metric, _, _ = (
                    mod09b.metrics(
                        p[
                            idx
                        ],
                        y_macro[
                            idx
                        ],
                    )
                )

                day_rows.append(
                    {
                        "track":
                            track,

                        "date":
                            str(
                                date
                            ),

                        "model":
                            model_name,

                        "n":
                            len(
                                idx
                            ),

                        **metric,
                    }
                )

        for cls in range(
            EXPECTED_SITES
        ):

            idx = (
                y_macro
                ==
                cls
            )

            for model_name, (
                pred,
                rank,
            ) in predictions.items():

                class_rows.append(
                    {
                        "track":
                            track,

                        "site_label":
                            labels[
                                cls
                            ],

                        "model":
                            model_name,

                        "n":
                            int(
                                idx.sum()
                            ),

                        "accuracy":
                            float(
                                np.mean(
                                    pred[
                                        idx
                                    ]
                                    ==
                                    cls
                                )
                            ),

                        "mean_true_rank":
                            float(
                                np.mean(
                                    rank[
                                        idx
                                    ]
                                )
                            ),
                    }
                )

        score_path = (
            OUT
            / f"10B_scores_{track}.npz"
        )

        np.savez_compressed(
            score_path,

            pcap_uid=
                macro_future[
                    "pcap_uid"
                ]
                .astype(str)
                .to_numpy(),

            query_date=
                dates,

            y_true=
                y_macro,

            candidate_labels=
                labels,

            micro_probs=
                probs[
                    "MICRO_FINAL"
                ].astype(
                    np.float32
                ),

            macro_xgb_probs=
                probs[
                    "MACRO_XGB"
                ].astype(
                    np.float32
                ),

            macro_ltd_probs=
                probs[
                    "MACRO_LTD"
                ].astype(
                    np.float32
                ),

            macro_final_probs=
                probs[
                    "MACRO_LTD_FINAL"
                ].astype(
                    np.float32
                ),

            hybrid_final_probs=
                probs[
                    "LTD_HYBRID_FINAL"
                ].astype(
                    np.float32
                ),
        )

    # ==================================================
    # EXACT SAME-CHECKPOINT RETENTION
    # ==================================================

    future_summary = pd.DataFrame(
        summary_rows
    )

    internal_summary = pd.read_csv(
        INTERNAL_RESULT
        / "09B_internal_summary.csv"
    )

    retention_rows = []

    future_dev = (
        future_summary[
            future_summary[
                "track"
            ]
            ==
            "DEV_FROZEN"
        ]
        .set_index(
            "model"
        )
    )

    internal_by_model = (
        internal_summary
        .set_index(
            "model"
        )
    )

    for model in [
        "MICRO_FINAL",
        "MACRO_XGB",
        "MACRO_LTD",
        "MACRO_LTD_FINAL",
        "LTD_HYBRID_FINAL",
    ]:

        for metric in [
            "accuracy",
            "macro_f1",
            "top5_accuracy",
            "mrr",
        ]:

            internal_value = float(
                internal_by_model.loc[
                    model,
                    metric,
                ]
            )

            future_value = float(
                future_dev.loc[
                    model,
                    metric,
                ]
            )

            retention_rows.append(
                {
                    "model":
                        model,

                    "metric":
                        metric,

                    "internal_value":
                        internal_value,

                    "future_value":
                        future_value,

                    "absolute_delta":
                        future_value
                        -
                        internal_value,

                    "absolute_drop":
                        internal_value
                        -
                        future_value,

                    "retention_ratio":
                        (
                            future_value
                            /
                            internal_value
                            if internal_value
                            else np.nan
                        ),

                    "relative_drop_fraction":
                        (
                            1.0
                            -
                            future_value
                            /
                            internal_value
                            if internal_value
                            else np.nan
                        ),

                    "comparison":
                        "EXACT_SAME_CHECKPOINT",
                }
            )

    retention = pd.DataFrame(
        retention_rows
    )

    # ==================================================
    # BENEFIT OF FULL-HISTORICAL REFIT ON SAME FUTURE-B
    # ==================================================

    for model in [
        "MICRO_FINAL",
        "MACRO_XGB",
        "MACRO_LTD",
        "MACRO_LTD_FINAL",
        "LTD_HYBRID_FINAL",
    ]:

        rows = mod09b.bootstrap_pair(
            (
                "HISTORICAL_REFIT_vs_DEV_FROZEN"
                f"::{model}"
            ),
            y_macro,
            dates,
            all_probs[
                "DEV_FROZEN"
            ][
                model
            ],
            all_probs[
                "HISTORICAL_FINAL"
            ][
                model
            ],
        )

        bootstrap_rows.extend(
            rows
        )

    bootstrap = pd.DataFrame(
        bootstrap_rows
    )

    complementarity = pd.DataFrame(
        complementarity_rows
    )

    per_day = pd.DataFrame(
        day_rows
    )

    per_class = pd.DataFrame(
        class_rows
    )

    # ==================================================
    # SAVE
    # ==================================================

    summary_path = (
        OUT
        / "10B_future_summary.csv"
    )

    retention_path = (
        OUT
        / "10B_same_checkpoint_retention.csv"
    )

    complementarity_path = (
        OUT
        / "10B_complementarity.csv"
    )

    bootstrap_path = (
        OUT
        / "10B_day_block_bootstrap.csv"
    )

    day_path = (
        OUT
        / "10B_per_day_metrics.csv"
    )

    class_path = (
        OUT
        / "10B_per_class_metrics.csv"
    )

    future_summary.to_csv(
        summary_path,
        index=False,
    )

    retention.to_csv(
        retention_path,
        index=False,
    )

    complementarity.to_csv(
        complementarity_path,
        index=False,
    )

    bootstrap.to_csv(
        bootstrap_path,
        index=False,
    )

    per_day.to_csv(
        day_path,
        index=False,
    )

    per_class.to_csv(
        class_path,
        index=False,
    )

    future_hashes = {
        "macro": {
            "path":
                str(
                    FUTURE_MACRO
                ),

            "sha256":
                sha256(
                    FUTURE_MACRO
                ),
        },

        "micro": {
            "path":
                str(
                    FUTURE_MICRO
                ),

            "sha256":
                sha256(
                    FUTURE_MICRO
                ),
        },
    }

    manifest = {
        "stage":
            "10B_future_concept_drift_evaluation",

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "git_commit":
            git_commit(),

        "status":
            "FUTURE_B_OPENED_AND_EVALUATED",

        "future_population": {
            "captures":
                len(
                    y_macro
                ),

            "sites":
                len(
                    labels
                ),

            "date_min":
                str(
                    pd.Timestamp(
                        dates.min()
                    ).date()
                ),

            "date_max":
                str(
                    pd.Timestamp(
                        dates.max()
                    ).date()
                ),

            "date_source":
                date_source,
        },

        "tracks": {
            "DEV_FROZEN": {
                "purpose":
                    (
                        "exact same-checkpoint retention "
                        "relative to INTERNAL_TEST"
                    ),

                "training_end":
                    "2025-12-28",
            },

            "HISTORICAL_FINAL": {
                "purpose":
                    (
                        "final operational Future-B "
                        "performance"
                    ),

                "training_end":
                    "2026-01-08",
            },
        },

        "frozen_configuration": {
            "micro_epochs":
                30,

            "macro_context_days":
                5,

            "macro_internal_alpha":
                0.375,

            "hybrid_micro_weight":
                0.45,

            "hybrid_macro_weight":
                0.55,
        },

        "protocol": {
            "future_observations_update_history":
                False,

            "model_selection_after_future_open":
                False,

            "hyperparameter_changes_after_future_open":
                False,

            "bootstrap_unit":
                "future_query_date",

            "bootstrap_iterations":
                2000,

            "bootstrap_fraction_is_p_value":
                False,

            "cohort_membership_future_blind":
                False,

            "cohort_caveat":
                (
                    "The 65-site cohort may historically "
                    "have been conditioned by Future-B "
                    "availability. Future feature values "
                    "were not used for current feature or "
                    "model selection."
                ),
        },

        "future_inputs":
            future_hashes,

        "outputs": {
            "summary":
                sha256(
                    summary_path
                ),

            "retention":
                sha256(
                    retention_path
                ),

            "complementarity":
                sha256(
                    complementarity_path
                ),

            "bootstrap":
                sha256(
                    bootstrap_path
                ),

            "per_day":
                sha256(
                    day_path
                ),

            "per_class":
                sha256(
                    class_path
                ),
        },

        "next":
            (
                "freeze Future-B evidence; "
                "then prepare virgin Dataset C "
                "external validation"
            ),
    }

    write_json(
        OUT
        / "10B_manifest_future_b.json",
        manifest,
    )

    print()
    print("=" * 78)
    print(
        "10B — FUTURE-B RESULTS"
    )
    print("=" * 78)

    print(
        future_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 78)
    print(
        "SAME-CHECKPOINT RETENTION"
    )
    print("=" * 78)

    print(
        retention.to_string(
            index=False
        )
    )

    print()
    print("=" * 78)
    print(
        "COMPLEMENTARITY"
    )
    print("=" * 78)

    print(
        complementarity.to_string(
            index=False
        )
    )

    print()
    print("=" * 78)
    print(
        "DAY-BLOCK BOOTSTRAP"
    )
    print("=" * 78)

    print(
        bootstrap.to_string(
            index=False
        )
    )

    print()
    print(
        "FUTURE-B evaluation COMPLETE."
    )

    print(
        "No model tuning is permitted after this point."
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    group = (
        parser
        .add_mutually_exclusive_group(
            required=True
        )
    )

    group.add_argument(
        "--verify-only",
        action="store_true",
    )

    group.add_argument(
        "--open-future",
        action="store_true",
    )

    parser.add_argument(
        "--allow-repeat",
        action="store_true",
        help=(
            "Technical recovery only; "
            "never for model tuning."
        ),
    )

    args = parser.parse_args()

    main(
        open_future=
            args.open_future,

        allow_repeat=
            args.allow_repeat,
    )
