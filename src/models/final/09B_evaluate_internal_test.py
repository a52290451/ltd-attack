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
from sklearn.metrics import f1_score

from src.utils.paths import (
    artifact_path,
    data_path,
    result_path,
)

from src.features.macro_v2._common import (
    git_commit,
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

DEV_ART = Path(
    artifact_path(
        "final",
        "DEV-FINAL",
    )
)

OUT = Path(
    result_path(
        "final",
        "INTERNAL-TEST",
    )
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


EXPECTED_INTERNAL_N = 14687
EXPECTED_SITES = 65

MACRO_ALPHA = 0.375

HYBRID_MACRO_ALPHA = 0.55

EPS = 1e-12

MARKER = (
    OUT
    / "09B_INTERNAL_TEST_OPENED.json"
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


def verify_file(
    path,
    expected_hash,
    label,
):

    path = Path(
        path
    )

    if not path.exists():
        raise FileNotFoundError(
            f"{label}: {path}"
        )

    observed = sha256(
        path
    )

    if observed != expected_hash:
        raise RuntimeError(
            f"{label} hash mismatch.\n"
            f"expected={expected_hash}\n"
            f"observed={observed}\n"
            f"path={path}"
        )

    print(
        f"PASS {label}: {observed}"
    )

    return observed


def verify_frozen_state():

    print("=" * 78)
    print(
        "09B — FROZEN ARTIFACT VERIFICATION"
    )
    print("=" * 78)

    manifest_path = (
        DEV_RESULT
        / "09A_manifest_final_dev_refit.json"
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            manifest_path
        )

    manifest = json.loads(
        manifest_path.read_text()
    )

    if (
        manifest[
            "training_population"
        ][
            "captures"
        ]
        != 57916
    ):
        raise RuntimeError(
            "Unexpected 09A DEV population."
        )

    if (
        manifest[
            "training_population"
        ][
            "sites"
        ]
        != 65
    ):
        raise RuntimeError(
            "Unexpected 09A site population."
        )

    artifacts = manifest[
        "artifacts"
    ]

    verify_file(
        artifacts[
            "micro_checkpoint"
        ][
            "path"
        ],
        artifacts[
            "micro_checkpoint"
        ][
            "sha256"
        ],
        "MICRO-FINAL checkpoint",
    )

    verify_file(
        artifacts[
            "macro_xgb"
        ][
            "path"
        ],
        artifacts[
            "macro_xgb"
        ][
            "sha256"
        ],
        "MACRO XGB",
    )

    verify_file(
        artifacts[
            "macro_ltd_preprocessing"
        ][
            "path"
        ],
        artifacts[
            "macro_ltd_preprocessing"
        ][
            "sha256"
        ],
        "MACRO LTD preprocessing",
    )

    verify_file(
        artifacts[
            "macro_daily_history"
        ][
            "path"
        ],
        artifacts[
            "macro_daily_history"
        ][
            "sha256"
        ],
        "MACRO frozen DEV history",
    )

    for i, item in enumerate(
        artifacts[
            "macro_ltd_checkpoints"
        ]
    ):

        verify_file(
            item[
                "path"
            ],
            item[
                "sha256"
            ],
            f"MACRO LTD checkpoint {i + 1}",
        )

    for name, item in (
        manifest[
            "inputs"
        ].items()
    ):

        verify_file(
            item[
                "path"
            ],
            item[
                "sha256"
            ],
            f"09A input {name}",
        )

    # Canonical Micro source was frozen in 08B.
    micro_manifest_path = (
        ROOT
        / "docs/evidence/"
        "MICRO-TEMPORAL-V1/"
        "08B_manifest.json"
    )

    if not micro_manifest_path.exists():
        raise FileNotFoundError(
            micro_manifest_path
        )

    micro_manifest = json.loads(
        micro_manifest_path.read_text()
    )

    micro_source = Path(
        micro_manifest[
            "source"
        ][
            "path"
        ]
    )

    verify_file(
        micro_source,
        micro_manifest[
            "source"
        ][
            "sha256"
        ],
        "canonical Micro source",
    )

    print()
    print(
        "Frozen architecture:"
    )

    print(
        "MICRO-FINAL: 30 epochs, seed 42"
    )

    print(
        "MACRO-LTD-FINAL: W=5, "
        "seeds=[11,42,73], alpha=0.375"
    )

    print(
        "LTD-HYBRID-FINAL: "
        "Micro=.45 / Macro=.55"
    )

    print()
    print(
        "VERIFY PASS."
    )

    print(
        "No INTERNAL_TEST numerical evaluation "
        "was performed."
    )

    print(
        "Future-B remains CLOSED."
    )

    return (
        manifest,
        micro_manifest,
    )


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


def softmax_numpy(
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


def geometric_fusion(
    first,
    second,
    second_weight,
):

    logits = (
        (
            1.0
            -
            second_weight
        )
        *
        np.log(
            np.clip(
                first,
                EPS,
                1.0,
            )
        )
        +
        second_weight
        *
        np.log(
            np.clip(
                second,
                EPS,
                1.0,
            )
        )
    )

    return softmax_numpy(
        logits
    )


def metrics(
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


def load_internal_macro(
    mod_macro,
):

    assignments_path = (
        result_path(
            "feature_engineering",
            "MACRO-V2-HIST",
            "01_temporal_split_assignments.csv",
        )
    )

    assignments = pd.read_csv(
        assignments_path
    )

    internal_ids = (
        assignments[
            assignments[
                "temporal_split"
            ]
            ==
            "INTERNAL_TEST"
        ][
            [
                "pcap_uid",
                "temporal_split",
            ]
        ]
        .copy()
    )

    if len(
        internal_ids
    ) != EXPECTED_INTERNAL_N:
        raise RuntimeError(
            "Unexpected INTERNAL_TEST assignment count: "
            f"{len(internal_ids)}"
        )

    (
        historical,
        historical_path,
    ) = (
        mod_macro
        .load_historical_65()
    )

    internal = historical.merge(
        internal_ids,
        on="pcap_uid",
        how="inner",
        validate="one_to_one",
    )

    if len(
        internal
    ) != EXPECTED_INTERNAL_N:
        raise RuntimeError(
            "Unexpected Macro INTERNAL_TEST count."
        )

    internal[
        "site_label"
    ] = (
        internal[
            "site_label"
        ]
        .astype(str)
    )

    dates, date_source = (
        mod_macro
        .derive_dates(
            internal
        )
    )

    internal[
        "_query_date"
    ] = pd.to_datetime(
        dates
    )

    if internal[
        "_query_date"
    ].isna().any():
        raise RuntimeError(
            "Missing INTERNAL_TEST capture dates."
        )

    labels = np.array(
        sorted(
            internal[
                "site_label"
            ].unique()
        ),
        dtype=str,
    )

    if len(
        labels
    ) != EXPECTED_SITES:
        raise RuntimeError(
            "Expected 65 INTERNAL_TEST classes."
        )

    label_to_idx = (
        mod_macro
        .make_label_mapping(
            labels
        )
    )

    y = (
        mod_macro
        .labels_to_indices(
            internal[
                "site_label"
            ],
            label_to_idx,
        )
    )

    return (
        internal,
        y,
        labels,
        date_source,
        historical_path,
    )


def load_internal_micro(
    mod_micro,
    macro_internal,
    labels,
):

    target_order = (
        macro_internal[
            "pcap_uid"
        ]
        .astype(str)
        .tolist()
    )

    target = set(
        target_order
    )

    source = Path(
        mod_micro.SOURCE
    )

    cols = [
        "pcap_uid",
        "site_label",
        "direction_vector",
        "size_vector",
    ]

    pieces = []

    for chunk in pd.read_csv(
        source,
        usecols=cols,
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

    df = pd.concat(
        pieces,
        ignore_index=True,
    )

    if len(
        df
    ) != EXPECTED_INTERNAL_N:
        raise RuntimeError(
            "Micro INTERNAL_TEST coverage mismatch."
        )

    if df[
        "pcap_uid"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate Micro INTERNAL_TEST pcap_uid."
        )

    df[
        "site_label"
    ] = (
        df[
            "site_label"
        ]
        .astype(str)
    )

    lookup = (
        df
        .set_index(
            "pcap_uid"
        )
    )

    if set(
        lookup.index
    ) != target:
        raise RuntimeError(
            "Micro/Macro INTERNAL_TEST UID sets differ."
        )

    df = (
        lookup
        .loc[
            target_order
        ]
        .reset_index()
    )

    macro_sites = (
        macro_internal[
            "site_label"
        ]
        .astype(str)
        .to_numpy()
    )

    if not np.array_equal(
        df[
            "site_label"
        ].to_numpy(),
        macro_sites,
    ):
        raise RuntimeError(
            "Micro/Macro site labels differ."
        )

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
            df[
                "site_label"
            ]
        ],
        dtype=np.int64,
    )

    n = len(
        df
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
        f"Parsing {n:,} INTERNAL_TEST "
        "Micro captures..."
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

        x_dir[
            i
        ] = (
            mod_micro
            .parse_vector(
                direction,
                mod_micro.MAX_LEN,
                np.int8,
            )
        )

        x_size[
            i
        ] = (
            mod_micro
            .parse_vector(
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
    )


def predict_ltd(
    mod_macro,
    model,
    x,
    dates,
    context_cache,
    device,
):

    output = np.full(
        (
            len(
                x
            ),
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

            if date not in context_cache:
                raise RuntimeError(
                    f"No frozen LTD context for {date}"
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
            ) = context_cache[
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
                len(
                    idx
                ),
                mod_macro.BATCH_SIZE,
            ):

                batch_idx = idx[
                    start:
                    start
                    +
                    mod_macro.BATCH_SIZE
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
            "Incomplete LTD INTERNAL_TEST predictions."
        )

    return output


def bootstrap_pair(
    name,
    y,
    dates,
    reference_probs,
    candidate_probs,
    iterations=2000,
):

    (
        ref_m,
        ref_pred,
        ref_rank,
    ) = metrics(
        reference_probs,
        y,
    )

    (
        cand_m,
        cand_pred,
        cand_rank,
    ) = metrics(
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
        d:
            np.flatnonzero(
                dates == d
            )
        for d in
        unique_dates
    }

    rng = np.random.default_rng(
        20260921
    )

    values = {
        "accuracy": [],
        "macro_f1": [],
        "top5_accuracy": [],
        "mrr": [],
    }

    for _ in range(
        iterations
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

        for model_name, pred, rank in [
            (
                "ref",
                ref_pred[
                    idx
                ],
                ref_rank[
                    idx
                ],
            ),
            (
                "cand",
                cand_pred[
                    idx
                ],
                cand_rank[
                    idx
                ],
            ),
        ]:

            current = {
                "accuracy":
                    float(
                        np.mean(
                            pred
                            ==
                            yy
                        )
                    ),

                "macro_f1":
                    float(
                        f1_score(
                            yy,
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
                            rank
                            <= 5
                        )
                    ),

                "mrr":
                    float(
                        np.mean(
                            1.0
                            /
                            rank
                        )
                    ),
            }

            if model_name == "ref":
                ref_current = current

            else:
                for metric in values:
                    values[
                        metric
                    ].append(
                        current[
                            metric
                        ]
                        -
                        ref_current[
                            metric
                        ]
                    )

    rows = []

    for metric, vals in (
        values.items()
    ):

        vals = np.asarray(
            vals,
            dtype=float,
        )

        observed = (
            cand_m[
                metric
            ]
            -
            ref_m[
                metric
            ]
        )

        rows.append(
            {
                "comparison":
                    name,

                "metric":
                    metric,

                "observed_delta":
                    observed,

                "bootstrap_mean":
                    float(
                        vals.mean()
                    ),

                "ci95_low":
                    float(
                        np.quantile(
                            vals,
                            0.025,
                        )
                    ),

                "ci95_high":
                    float(
                        np.quantile(
                            vals,
                            0.975,
                        )
                    ),

                "fraction_delta_gt_0":
                    float(
                        np.mean(
                            vals > 0
                        )
                    ),
            }
        )

    return rows


def main(
    open_internal,
    allow_repeat,
):

    manifest09a, micro_manifest = (
        verify_frozen_state()
    )

    if not open_internal:
        return

    if (
        MARKER.exists()
        and
        not allow_repeat
    ):
        raise RuntimeError(
            "INTERNAL_TEST opening marker already exists:\n"
            f"{MARKER}\n"
            "Refusing an accidental repeat."
        )

    marker = {
        "stage":
            "09B_internal_test_opening",

        "opened_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "git_commit":
            git_commit(),

        "purpose":
            (
                "one-time numerical evaluation of "
                "frozen final models"
            ),

        "architecture_changes_after_open":
            False,
    }

    write_json(
        MARKER,
        marker,
    )

    print()
    print("!" * 78)
    print(
        "INTERNAL_TEST NUMERICAL HOLDOUT IS NOW OPEN"
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
        "micro_final_09b",
    )

    mod_macro = load_module(
        ROOT
        / "src/features/macro_v2/"
        "07A_candidate_conditioned_temporal_encoder.py",
        "macro_final_09b",
    )

    # =====================================================
    # CANONICAL INTERNAL POPULATION
    # =====================================================

    (
        macro_internal,
        y_macro,
        labels,
        date_source,
        historical_path,
    ) = load_internal_macro(
        mod_macro
    )

    dates = (
        pd.to_datetime(
            macro_internal[
                "_query_date"
            ]
        )
        .to_numpy(
            dtype="datetime64[D]"
        )
    )

    print()
    print(
        "INTERNAL_TEST captures:",
        len(
            macro_internal
        ),
    )

    print(
        "Sites:",
        len(
            labels
        ),
    )

    print(
        "Date range:",
        pd.Timestamp(
            dates.min()
        ).date(),
        "->",
        pd.Timestamp(
            dates.max()
        ).date(),
    )

    # =====================================================
    # MICRO-FINAL
    # =====================================================

    print()
    print("=" * 78)
    print(
        "MICRO-FINAL"
    )
    print("=" * 78)

    (
        micro_df,
        x_dir,
        x_size,
        y_micro,
    ) = load_internal_micro(
        mod_micro,
        macro_internal,
        labels,
    )

    if not np.array_equal(
        y_micro,
        y_macro,
    ):
        raise RuntimeError(
            "Micro/Macro y_true mismatch."
        )

    micro_checkpoint_path = Path(
        manifest09a[
            "artifacts"
        ][
            "micro_checkpoint"
        ][
            "path"
        ]
    )

    micro_checkpoint = torch.load(
        micro_checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    checkpoint_labels = np.array(
        micro_checkpoint[
            "candidate_labels"
        ],
        dtype=str,
    )

    if not np.array_equal(
        checkpoint_labels,
        labels,
    ):
        raise RuntimeError(
            "MICRO-FINAL class order mismatch."
        )

    micro_model = (
        mod_micro
        .MultimodalTransformer(
            EXPECTED_SITES
        )
        .to(
            device
        )
    )

    micro_model.load_state_dict(
        micro_checkpoint[
            "state_dict"
        ]
    )

    micro_loader = (
        mod_micro.loader(
            x_dir,
            x_size,
            y_micro,
            np.arange(
                len(
                    y_micro
                )
            ),
            False,
        )
    )

    (
        _,
        micro_probs,
        micro_eval_y,
    ) = (
        mod_micro.evaluate(
            micro_model,
            micro_loader,
            float(
                micro_checkpoint[
                    "size_min"
                ]
            ),
            float(
                micro_checkpoint[
                    "size_max"
                ]
            ),
            device,
        )
    )

    if not np.array_equal(
        micro_eval_y,
        y_macro,
    ):
        raise RuntimeError(
            "Micro evaluation order mismatch."
        )

    del (
        micro_model,
        micro_loader,
        x_dir,
        x_size,
    )

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # =====================================================
    # MACRO XGB
    # =====================================================

    print()
    print("=" * 78)
    print(
        "MACRO-XGB"
    )
    print("=" * 78)

    xgb_art = joblib.load(
        manifest09a[
            "artifacts"
        ][
            "macro_xgb"
        ][
            "path"
        ]
    )

    macro_features = list(
        xgb_art[
            "features"
        ]
    )

    x_xgb = (
        numeric_frame(
            macro_internal,
            macro_features,
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
            "Non-finite INTERNAL_TEST XGB matrix."
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

    # =====================================================
    # MACRO LTD
    # =====================================================

    print()
    print("=" * 78)
    print(
        "MACRO-LTD"
    )
    print("=" * 78)

    ltd_pre = joblib.load(
        manifest09a[
            "artifacts"
        ][
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
            "MACRO LTD class order mismatch."
        )

    if list(
        ltd_pre[
            "features"
        ]
    ) != macro_features:
        raise RuntimeError(
            "MACRO XGB/LTD feature mismatch."
        )

    context_days = int(
        ltd_pre[
            "context_days"
        ]
    )

    if context_days != 5:
        raise RuntimeError(
            f"Expected W=5, got {context_days}"
        )

    mod_macro.CONTEXT_DAYS = (
        context_days
    )

    x_ltd = (
        mod_macro.transform(
            macro_internal,
            macro_features,
            ltd_pre[
                "medians"
            ],
            ltd_pre[
                "scaler"
            ],
        )
    )

    daily_history = pd.read_csv(
        manifest09a[
            "artifacts"
        ][
            "macro_daily_history"
        ][
            "path"
        ]
    )

    daily_history[
        "site_label"
    ] = (
        daily_history[
            "site_label"
        ]
        .astype(str)
    )

    daily_history[
        "date"
    ] = pd.to_datetime(
        daily_history[
            "date"
        ]
    )

    internal_dates = pd.to_datetime(
        macro_internal[
            "_query_date"
        ]
    ).to_numpy()

    if (
        daily_history[
            "date"
        ].max()
        >=
        pd.Timestamp(
            internal_dates.min()
        )
    ):
        raise RuntimeError(
            "Frozen-history temporal boundary violation."
        )

    (
        context_cache,
        skipped_internal_dates,
    ) = (
        mod_macro
        .build_context_cache(
            daily_history,
            labels,
            internal_dates,
            macro_features,
        )
    )

    if skipped_internal_dates:
        raise RuntimeError(
            "Missing INTERNAL_TEST candidate histories: "
            f"{skipped_internal_dates}"
        )

    ltd_seed_probs = []

    for item in (
        manifest09a[
            "artifacts"
        ][
            "macro_ltd_checkpoints"
        ]
    ):

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
                "LTD checkpoint class order mismatch."
            )

        model = (
            mod_macro
            .LTDPairScorer()
            .to(
                device
            )
        )

        model.load_state_dict(
            checkpoint[
                "state_dict"
            ]
        )

        logits = predict_ltd(
            mod_macro,
            model,
            x_ltd,
            internal_dates,
            context_cache,
            device,
        )

        probs = softmax_numpy(
            logits
        )

        ltd_seed_probs.append(
            probs
        )

        print(
            "Evaluated LTD seed:",
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

    # =====================================================
    # FROZEN MACRO + HYBRID FUSIONS
    # =====================================================

    macro_final_probs = (
        geometric_fusion(
            xgb_probs,
            ltd_probs,
            MACRO_ALPHA,
        )
    )

    hybrid_probs = (
        geometric_fusion(
            micro_probs,
            macro_final_probs,
            HYBRID_MACRO_ALPHA,
        )
    )

    model_probs = {
        "MICRO_FINAL":
            micro_probs,

        "MACRO_XGB":
            xgb_probs,

        "MACRO_LTD":
            ltd_probs,

        "MACRO_LTD_FINAL":
            macro_final_probs,

        "LTD_HYBRID_FINAL":
            hybrid_probs,
    }

    # =====================================================
    # OVERALL METRICS
    # =====================================================

    summary_rows = []

    predictions = {}

    for name, probs in (
        model_probs.items()
    ):

        (
            m,
            pred,
            rank,
        ) = metrics(
            probs,
            y_macro,
        )

        summary_rows.append(
            {
                "model":
                    name,

                "n":
                    len(
                        y_macro
                    ),

                **m,
            }
        )

        predictions[
            name
        ] = {
            "pred":
                pred,

            "rank":
                rank,
        }

    summary = pd.DataFrame(
        summary_rows
    )

    # =====================================================
    # COMPLEMENTARITY
    # =====================================================

    micro_correct = (
        predictions[
            "MICRO_FINAL"
        ][
            "pred"
        ]
        ==
        y_macro
    )

    macro_correct = (
        predictions[
            "MACRO_LTD_FINAL"
        ][
            "pred"
        ]
        ==
        y_macro
    )

    hybrid_correct = (
        predictions[
            "LTD_HYBRID_FINAL"
        ][
            "pred"
        ]
        ==
        y_macro
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

    macro_errors = int(
        (
            ~macro_correct
        ).sum()
    )

    complementarity = pd.DataFrame(
        [
            {
                "n":
                    len(
                        y_macro
                    ),

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

                "micro_rescue_rate_given_macro_wrong":
                    (
                        float(
                            micro_only.sum()
                            /
                            macro_errors
                        )
                        if macro_errors
                        else np.nan
                    ),

                "branch_prediction_agreement_rate":
                    float(
                        np.mean(
                            predictions[
                                "MICRO_FINAL"
                            ][
                                "pred"
                            ]
                            ==
                            predictions[
                                "MACRO_LTD_FINAL"
                            ][
                                "pred"
                            ]
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
        ]
    )

    # =====================================================
    # DAY-BLOCK BOOTSTRAP
    # =====================================================

    bootstrap_rows = []

    comparisons = [
        (
            "HYBRID_vs_MICRO",
            micro_probs,
            hybrid_probs,
        ),
        (
            "HYBRID_vs_MACRO_FINAL",
            macro_final_probs,
            hybrid_probs,
        ),
        (
            "MACRO_FINAL_vs_XGB",
            xgb_probs,
            macro_final_probs,
        ),
    ]

    for (
        name,
        reference,
        candidate,
    ) in comparisons:

        bootstrap_rows.extend(
            bootstrap_pair(
                name,
                y_macro,
                dates,
                reference,
                candidate,
            )
        )

    bootstrap = pd.DataFrame(
        bootstrap_rows
    )

    # =====================================================
    # PER-DAY
    # =====================================================

    day_rows = []

    for date in sorted(
        np.unique(
            dates
        )
    ):

        idx = np.flatnonzero(
            dates == date
        )

        for name, probs in (
            model_probs.items()
        ):

            m, _, _ = metrics(
                probs[
                    idx
                ],
                y_macro[
                    idx
                ],
            )

            day_rows.append(
                {
                    "date":
                        str(
                            date
                        ),

                    "model":
                        name,

                    "n":
                        len(
                            idx
                        ),

                    **m,
                }
            )

    per_day = pd.DataFrame(
        day_rows
    )

    # =====================================================
    # PER-CLASS
    # =====================================================

    class_rows = []

    for cls in range(
        EXPECTED_SITES
    ):

        idx = (
            y_macro
            ==
            cls
        )

        for name in (
            model_probs
        ):

            pred = predictions[
                name
            ][
                "pred"
            ][
                idx
            ]

            class_rows.append(
                {
                    "site_label":
                        labels[
                            cls
                        ],

                    "model":
                        name,

                    "n":
                        int(
                            idx.sum()
                        ),

                    "accuracy":
                        float(
                            np.mean(
                                pred == cls
                            )
                        ),
                }
            )

    per_class = pd.DataFrame(
        class_rows
    )

    # =====================================================
    # SAVE ALL EVIDENCE
    # =====================================================

    summary_path = (
        OUT
        / "09B_internal_summary.csv"
    )

    complementarity_path = (
        OUT
        / "09B_complementarity.csv"
    )

    bootstrap_path = (
        OUT
        / "09B_day_block_bootstrap.csv"
    )

    per_day_path = (
        OUT
        / "09B_per_day_metrics.csv"
    )

    per_class_path = (
        OUT
        / "09B_per_class_metrics.csv"
    )

    score_path = (
        OUT
        / "09B_internal_scores.npz"
    )

    summary.to_csv(
        summary_path,
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
        per_day_path,
        index=False,
    )

    per_class.to_csv(
        per_class_path,
        index=False,
    )

    np.savez_compressed(
        score_path,

        pcap_uid=
            macro_internal[
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
            micro_probs.astype(
                np.float32
            ),

        macro_xgb_probs=
            xgb_probs.astype(
                np.float32
            ),

        macro_ltd_probs=
            ltd_probs.astype(
                np.float32
            ),

        macro_final_probs=
            macro_final_probs.astype(
                np.float32
            ),

        hybrid_final_probs=
            hybrid_probs.astype(
                np.float32
            ),
    )

    manifest = {
        "stage":
            "09B_one_time_internal_test",

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "git_commit":
            git_commit(),

        "status":
            "INTERNAL_TEST_OPENED_AND_EVALUATED",

        "population": {
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

        "models": {
            "micro":
                "MICRO-FINAL",

            "macro":
                "MACRO-LTD-FINAL",

            "hybrid":
                "LTD-HYBRID-FINAL",

            "macro_internal_alpha":
                MACRO_ALPHA,

            "hybrid_macro_weight":
                HYBRID_MACRO_ALPHA,
        },

        "protocol": {
            "model_selection_after_open":
                False,

            "hyperparameter_changes_after_open":
                False,

            "test_observations_update_ltd_context":
                False,

            "ltd_history_source":
                (
                    "DEV-only frozen standardized "
                    "site-day history"
                ),

            "day_block_bootstrap_iterations":
                2000,

            "bootstrap_fraction_is_p_value":
                False,
        },

        "data_policy": {
            "internal_test_values_used":
                True,

            "internal_test_opened":
                True,

            "external_future_used":
                False,

            "future_b_opened":
                False,
        },

        "frozen_refit_manifest_sha256":
            sha256(
                DEV_RESULT
                / "09A_manifest_final_dev_refit.json"
            ),

        "outputs": {
            "summary":
                sha256(
                    summary_path
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
                    per_day_path
                ),

            "per_class":
                sha256(
                    per_class_path
                ),

            "scores":
                sha256(
                    score_path
                ),
        },

        "next":
            (
                "full-Historical refit followed by "
                "one-time Future-B concept-drift evaluation"
            ),
    }

    manifest_path = (
        OUT
        / "09B_manifest_internal_test.json"
    )

    write_json(
        manifest_path,
        manifest,
    )

    print()
    print("=" * 78)
    print(
        "09B INTERNAL_TEST RESULTS"
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
        "MICRO / MACRO COMPLEMENTARITY"
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
        "PAIRED DAY-BLOCK BOOTSTRAP"
    )
    print("=" * 78)

    print(
        bootstrap.to_string(
            index=False
        )
    )

    print()
    print(
        "INTERNAL_TEST evaluation COMPLETE."
    )

    print(
        "No tuning is permitted from this point."
    )

    print(
        "Future-B remains CLOSED."
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(
        required=True
    )

    group.add_argument(
        "--verify-only",
        action="store_true",
    )

    group.add_argument(
        "--open-internal",
        action="store_true",
    )

    parser.add_argument(
        "--allow-repeat",
        action="store_true",
        help=(
            "Technical recovery only. "
            "Never use for model tuning."
        ),
    )

    args = parser.parse_args()

    main(
        open_internal=
            args.open_internal,

        allow_repeat=
            args.allow_repeat,
    )
