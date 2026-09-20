from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import importlib.util
import math

import numpy as np
import pandas as pd

from sklearn.metrics import f1_score

import torch
import torch.nn as nn
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


def fuse(
    xgb,
    ltd,
):
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

    p = np.exp(z)

    return (
        p
        / p.sum(
            axis=1,
            keepdims=True,
        )
    )


def predictions(
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
        + 1
    )

    return (
        pred,
        ranks,
    )


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

    yy = y[
        idx
    ]

    pp = pred[
        idx
    ]

    rr = ranks[
        idx
    ]

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
                    labels=np.arange(
                        65
                    ),
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
    v2_pred,
    v2_rank,
    age_pred,
    age_rank,
    idx=None,
):
    a = metrics(
        y,
        v2_pred,
        v2_rank,
        idx,
    )

    b = metrics(
        y,
        age_pred,
        age_rank,
        idx,
    )

    return {
        k:
            b[
                k
            ]
            -
            a[
                k
            ]

        for k in a
    }


def age_sinusoidal_encoding(
    age_days,
    dim,
):
    """
    age_days:
        [C, W]

    Returns:
        [C, W, dim]

    Same deterministic sinusoidal basis as standard positional
    encoding, but position is the actual age in calendar days.
    """

    if dim % 2 != 0:
        raise RuntimeError(
            "Age sinusoidal encoding requires even dim."
        )

    age = age_days.to(
        dtype=torch.float32
    )

    div_term = torch.exp(
        torch.arange(
            0,
            dim,
            2,
            device=age.device,
            dtype=torch.float32,
        )
        *
        (
            -math.log(
                10000.0
            )
            / dim
        )
    )

    angles = (
        age[
            :,
            :,
            None
        ]
        *
        div_term[
            None,
            None,
            :
        ]
    )

    out = torch.zeros(
        (
            age.shape[
                0
            ],
            age.shape[
                1
            ],
            dim,
        ),
        dtype=torch.float32,
        device=age.device,
    )

    out[
        :,
        :,
        0::2
    ] = torch.sin(
        angles
    )

    out[
        :,
        :,
        1::2
    ] = torch.cos(
        angles
    )

    return out


def build_age_context_for_date(
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
            CONTEXT_DAYS,
            len(
                features
            ),
        ),
        dtype=np.float32,
    )

    padding = np.ones(
        (
            len(
                candidate_labels
            ),
            CONTEXT_DAYS,
        ),
        dtype=bool,
    )

    ages = np.zeros(
        (
            len(
                candidate_labels
            ),
            CONTEXT_DAYS,
        ),
        dtype=np.float32,
    )

    history_lengths = []

    for candidate_idx, site in enumerate(
        candidate_labels
    ):

        hist = (
            daily_history[
                (
                    daily_history[
                        "site_label"
                    ]
                    == site
                )
                &
                (
                    daily_history[
                        "date"
                    ]
                    < cutoff
                )
            ]
            .sort_values(
                "date"
            )
            .tail(
                CONTEXT_DAYS
            )
        )

        if hist.empty:
            return None

        values = (
            hist[
                features
            ]
            .to_numpy(
                dtype=np.float32
            )
        )

        hist_dates = pd.to_datetime(
            hist[
                "date"
            ]
        )

        age_values = (
            (
                cutoff
                -
                hist_dates
            )
            .dt.days
            .to_numpy(
                dtype=np.float32
            )
        )

        if np.any(
            age_values < 1
        ):
            raise RuntimeError(
                "Age must be >= 1 because context must be strictly historical."
            )

        n = len(
            values
        )

        contexts[
            candidate_idx,
            -n:,
            :
        ] = values

        padding[
            candidate_idx,
            -n:
        ] = False

        ages[
            candidate_idx,
            -n:
        ] = age_values

        history_lengths.append(
            n
        )

    return (
        contexts,
        padding,
        ages,
        history_lengths,
    )


def build_age_context_cache(
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

        date = pd.Timestamp(
            date
        )

        result = (
            build_age_context_for_date(
                daily_history,
                candidate_labels,
                date,
                features,
            )
        )

        if result is None:
            skipped.append(
                date
            )
        else:
            cache[
                date
            ] = result

    return (
        cache,
        skipped,
    )


def make_age_model(
    mod07a,
):

    class CandidateAgeTemporalEncoder(
        mod07a.CandidateTemporalEncoder
    ):
        """
        Capacity-matched replacement for the V2 encoder.

        V2:
            FeatureProjection
            + ordinal sinusoidal position encoding.

        Age-aware:
            FeatureProjection
            + sinusoidal encoding of actual age in days.

        No new trainable parameters.
        """

        def forward(
            self,
            history,
            padding_mask,
            age_days,
        ):

            x = self.input_projection(
                history
            )

            age_pe = (
                age_sinusoidal_encoding(
                    age_days,
                    mod07a.LATENT_DIM,
                )
                .to(
                    dtype=x.dtype
                )
            )

            x = (
                x
                +
                age_pe
            )

            x = self.transformer(
                x,
                src_key_padding_mask=
                    padding_mask,
            )

            x = self.final_norm(
                x
            )

            valid = (
                ~padding_mask
            ).unsqueeze(
                -1
            ).float()

            denom = (
                valid.sum(
                    dim=1
                )
                .clamp_min(
                    1.0
                )
            )

            pooled = (
                (
                    x
                    * valid
                )
                .sum(
                    dim=1
                )
                /
                denom
            )

            return pooled

    class LTDPairScorerAge(
        mod07a.LTDPairScorer
    ):

        def __init__(
            self,
        ):
            super().__init__()

            self.temporal_encoder = (
                CandidateAgeTemporalEncoder()
            )

        def forward(
            self,
            query,
            candidate_history,
            candidate_padding_mask,
            candidate_age_days,
        ):

            q = self.query_encoder(
                query
            )

            c = self.temporal_encoder(
                candidate_history,
                candidate_padding_mask,
                candidate_age_days,
            )

            B = q.shape[
                0
            ]

            C = c.shape[
                0
            ]

            q_exp = (
                q[
                    :,
                    None,
                    :
                ]
                .expand(
                    B,
                    C,
                    mod07a.LATENT_DIM,
                )
            )

            c_exp = (
                c[
                    None,
                    :,
                    :
                ]
                .expand(
                    B,
                    C,
                    mod07a.LATENT_DIM,
                )
            )

            pair = torch.cat(
                [
                    q_exp,
                    c_exp,
                    torch.abs(
                        q_exp
                        -
                        c_exp
                    ),
                    q_exp
                    *
                    c_exp,
                ],
                dim=-1,
            )

            return (
                self.pair_head(
                    pair
                )
                .squeeze(
                    -1
                )
            )

    return (
        CandidateAgeTemporalEncoder,
        LTDPairScorerAge,
    )


def train_age_model(
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

    usable_dates = sorted(
        context_cache.keys()
    )

    index_by_date = {}

    train_dates_pd = pd.to_datetime(
        train_dates
    )

    for date in usable_dates:

        idx = np.flatnonzero(
            train_dates_pd
            ==
            date
        )

        if len(
            idx
        ):
            index_by_date[
                date
            ] = idx

    if not index_by_date:
        raise RuntimeError(
            "No causal age-aware training dates."
        )

    final_loss = None

    model.train()

    for _ in range(
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

        for date in date_order:

            query_indices = (
                index_by_date[
                    date
                ].copy()
            )

            rng.shuffle(
                query_indices
            )

            (
                context_np,
                padding_np,
                age_np,
                _,
            ) = context_cache[
                date
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

            age_days = (
                torch.from_numpy(
                    age_np
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
                    age_days,
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
                len(
                    x
                )
                for x in
                index_by_date.values()
            )
        ),
        len(
            index_by_date
        ),
    )


def predict_age_model(
    mod07a,
    model,
    x_test,
    test_dates,
    context_cache,
    device,
):
    output = np.full(
        (
            len(
                x_test
            ),
            65,
        ),
        np.nan,
        dtype=np.float32,
    )

    model.eval()

    test_dates_pd = pd.to_datetime(
        test_dates
    )

    with torch.no_grad():

        for date in sorted(
            pd.unique(
                test_dates_pd
            )
        ):

            date = pd.Timestamp(
                date
            )

            if date not in context_cache:
                raise RuntimeError(
                    f"No age-aware context for test date {date}"
                )

            idx = np.flatnonzero(
                test_dates_pd
                ==
                date
            )

            (
                context_np,
                padding_np,
                age_np,
                _,
            ) = context_cache[
                date
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

            age_days = (
                torch.from_numpy(
                    age_np
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
                mod07a.BATCH_SIZE,
            ):

                batch_idx = idx[
                    start:
                    start
                    +
                    mod07a.BATCH_SIZE
                ]

                query = (
                    torch.from_numpy(
                        x_test[
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
                    age_days,
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
            "Incomplete age-aware predictions."
        )

    return output


def bootstrap_dates(
    y,
    dates,
    v2_pred,
    v2_rank,
    age_pred,
    age_rank,
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
                ==
                d
            )
        for d in unique_dates
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
                v2_pred,
                v2_rank,
                age_pred,
                age_rank,
                idx,
            )
        )

    return pd.DataFrame(
        rows
    )


def age_cache_audit(
    fold,
    split_type,
    cache,
):
    rows = []

    for date, value in sorted(
        cache.items()
    ):

        (
            _,
            padding,
            ages,
            _,
        ) = value

        valid_ages = ages[
            ~padding
        ]

        latest_per_candidate = []

        for c in range(
            ages.shape[
                0
            ]
        ):

            valid = ages[
                c
            ][
                ~padding[
                    c
                ]
            ]

            latest_per_candidate.append(
                float(
                    valid.min()
                )
            )

        rows.append(
            {
                "fold":
                    fold,

                "split_type":
                    split_type,

                "query_date":
                    str(
                        pd.Timestamp(
                            date
                        ).date()
                    ),

                "n_valid_tokens":
                    int(
                        len(
                            valid_ages
                        )
                    ),

                "token_age_min":
                    float(
                        valid_ages.min()
                    ),

                "token_age_median":
                    float(
                        np.median(
                            valid_ages
                        )
                    ),

                "token_age_max":
                    float(
                        valid_ages.max()
                    ),

                "latest_profile_age_min":
                    float(
                        np.min(
                            latest_per_candidate
                        )
                    ),

                "latest_profile_age_median":
                    float(
                        np.median(
                            latest_per_candidate
                        )
                    ),

                "latest_profile_age_max":
                    float(
                        np.max(
                            latest_per_candidate
                        )
                    ),
            }
        )

    return rows


def main():

    print("=" * 78)
    print(
        "07F-1 — QUERY-AGE-AWARE LTD"
    )
    print("=" * 78)

    print(
        "Reference: MACRO-LTD-V2."
    )

    print(
        "Context: 5 site-days."
    )

    print(
        "Ordinal position encoding is replaced "
        "with actual query-to-profile age encoding."
    )

    print(
        "No feature/context/alpha/hyperparameter tuning."
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
        "macro_v2_07d_age",
    )

    mod07a, source07a = (
        mod07d.load_07a()
    )

    mod07a.CONTEXT_DAYS = (
        CONTEXT_DAYS
    )

    (
        _,
        AgeScorer,
    ) = make_age_model(
        mod07a
    )

    reference_model = (
        mod07a.LTDPairScorer()
    )

    age_model_check = (
        AgeScorer()
    )

    reference_params = (
        mod07a.count_parameters(
            reference_model
        )
    )

    age_params = (
        mod07a.count_parameters(
            age_model_check
        )
    )

    del reference_model
    del age_model_check

    print(
        "Reference trainable parameters:",
        reference_params,
    )

    print(
        "Age-aware trainable parameters:",
        age_params,
    )

    if (
        reference_params
        !=
        age_params
    ):
        raise RuntimeError(
            "Parameter count mismatch."
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
        f[
            "fold"
        ]:
            f
        for f in
        mod07a.FOLDS
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
    per_day_rows = []
    per_class_rows = []
    age_audit_rows = []

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

        (
            train_cache,
            skipped_train,
        ) = (
            build_age_context_cache(
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
        )

        (
            test_cache,
            skipped_test,
        ) = (
            build_age_context_cache(
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
        )

        if skipped_test:
            raise RuntimeError(
                f"{fold}: skipped test dates: {skipped_test}"
            )

        age_audit_rows.extend(
            age_cache_audit(
                fold,
                "train",
                train_cache,
            )
        )

        age_audit_rows.extend(
            age_cache_audit(
                fold,
                "test",
                test_cache,
            )
        )

        reference_path = (
            out
            / f"07D_01R_scores_{fold}.npz"
        )

        if not reference_path.exists():
            raise FileNotFoundError(
                reference_path
            )

        saved = np.load(
            reference_path,
            allow_pickle=True,
        )

        xgb = (
            saved[
                "xgb_probs"
            ]
            .astype(
                np.float64
            )
        )

        v2_ltd = (
            saved[
                "v2_ltd_probs"
            ]
            .astype(
                np.float64
            )
        )

        v2_fusion = (
            saved[
                "v2_fusion_probs"
            ]
            .astype(
                np.float64
            )
        )

        y = (
            saved[
                "y_true"
            ]
            .astype(int)
        )

        labels = (
            saved[
                "candidate_labels"
            ]
            .astype(str)
        )

        saved_uid = (
            saved[
                "pcap_uid"
            ]
            .astype(str)
        )

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

        if not np.array_equal(
            labels,
            fold_data[
                "candidate_labels"
            ].astype(str),
        ):
            raise RuntimeError(
                f"{fold}: candidate order mismatch."
            )

        probs_by_seed = []

        for seed in mod07a.SEEDS:

            print(
                f"{fold} — age-aware seed {seed}"
            )

            mod07a.set_seed(
                seed
            )

            model = (
                AgeScorer()
                .to(
                    device
                )
            )

            (
                train_loss,
                train_n,
                train_dates_n,
            ) = train_age_model(
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
                train_cache,
                device,
                seed,
            )

            logits = predict_age_model(
                mod07a,
                model,
                fold_data[
                    "x_test"
                ],
                fold_data[
                    "test_dates"
                ],
                test_cache,
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

            probs_by_seed.append(
                probs
            )

            del model

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        age_ltd = (
            np.stack(
                probs_by_seed,
                axis=0,
            )
            .mean(
                axis=0
            )
        )

        age_ltd /= (
            age_ltd.sum(
                axis=1,
                keepdims=True,
            )
        )

        age_fusion = fuse(
            xgb,
            age_ltd,
        )

        model_probs = {
            "V2_LTD":
                v2_ltd,

            "AGE_LTD":
                age_ltd,

            "V2_FUSION":
                v2_fusion,

            "AGE_FUSION":
                age_fusion,
        }

        model_outputs = {}

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

            model_outputs[
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

                    "context_days":
                        CONTEXT_DAYS,

                    "trainable_parameters":
                        (
                            age_params
                            if "AGE"
                            in name
                            else reference_params
                        ),

                    **m,
                }
            )

        (
            v2_pred,
            v2_rank,
            _,
        ) = model_outputs[
            "V2_FUSION"
        ]

        (
            age_pred,
            age_rank,
            _,
        ) = model_outputs[
            "AGE_FUSION"
        ]

        observed = paired_delta(
            y,
            v2_pred,
            v2_rank,
            age_pred,
            age_rank,
        )

        test_dates = (
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
            test_dates,
            v2_pred,
            v2_rank,
            age_pred,
            age_rank,
        )

        v2_correct = (
            v2_pred == y
        )

        age_correct = (
            age_pred == y
        )

        age_only = (
            ~v2_correct
            &
            age_correct
        )

        v2_only = (
            v2_correct
            &
            ~age_correct
        )

        comparison_rows.append(
            {
                "fold":
                    fold,

                **{
                    f"age_minus_v2_{k}":
                        v
                    for k, v
                    in observed.items()
                },

                "age_only_correct_n":
                    int(
                        age_only.sum()
                    ),

                "v2_only_correct_n":
                    int(
                        v2_only.sum()
                    ),

                "net_age_correct_n":
                    int(
                        age_only.sum()
                        -
                        v2_only.sum()
                    ),
            }
        )

        for metric in [
            "accuracy",
            "macro_f1",
            "top5_accuracy",
            "mrr",
        ]:

            values = (
                boot[
                    metric
                ].to_numpy()
            )

            bootstrap_rows.append(
                {
                    "fold":
                        fold,

                    "metric":
                        metric,

                    "observed_age_minus_v2":
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

                    "p_age_gt_v2":
                        float(
                            np.mean(
                                values > 0
                            )
                        ),
                }
            )

        for date in sorted(
            pd.unique(
                test_dates
            )
        ):

            idx = np.flatnonzero(
                test_dates
                ==
                date
            )

            d = paired_delta(
                y,
                v2_pred,
                v2_rank,
                age_pred,
                age_rank,
                idx,
            )

            per_day_rows.append(
                {
                    "fold":
                        fold,

                    "date":
                        date,

                    "n":
                        len(
                            idx
                        ),

                    **{
                        f"delta_{k}":
                            v
                        for k, v in
                        d.items()
                    },
                }
            )

        for cls in range(
            65
        ):

            idx = np.flatnonzero(
                y == cls
            )

            a = v2_correct[
                idx
            ]

            b = age_correct[
                idx
            ]

            per_class_rows.append(
                {
                    "fold":
                        fold,

                    "site_label":
                        labels[
                            cls
                        ],

                    "n":
                        len(
                            idx
                        ),

                    "v2_accuracy":
                        float(
                            a.mean()
                        ),

                    "age_accuracy":
                        float(
                            b.mean()
                        ),

                    "delta_accuracy":
                        float(
                            b.mean()
                            -
                            a.mean()
                        ),

                    "age_only_correct_n":
                        int(
                            (
                                ~a
                                &
                                b
                            ).sum()
                        ),

                    "v2_only_correct_n":
                        int(
                            (
                                a
                                &
                                ~b
                            ).sum()
                        ),
                }
            )

        np.savez_compressed(
            out
            / f"07F_01_scores_{fold}.npz",

            pcap_uid=
                saved_uid.astype(str),

            y_true=
                y,

            candidate_labels=
                labels.astype(str),

            xgb_probs=
                xgb.astype(
                    np.float32
                ),

            v2_ltd_probs=
                v2_ltd.astype(
                    np.float32
                ),

            age_ltd_probs=
                age_ltd.astype(
                    np.float32
                ),

            v2_fusion_probs=
                v2_fusion.astype(
                    np.float32
                ),

            age_fusion_probs=
                age_fusion.astype(
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

    seeds_df = pd.DataFrame(
        seed_rows
    )

    per_day_df = pd.DataFrame(
        per_day_rows
    )

    per_class_df = pd.DataFrame(
        per_class_rows
    )

    age_audit_df = pd.DataFrame(
        age_audit_rows
    )

    a = (
        comparison_df[
            comparison_df[
                "fold"
            ]
            == "A_EARLY_TO_MIDDLE"
        ]
        .iloc[
            0
        ]
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

    age_pass = bool(
        a[
            "age_minus_v2_accuracy"
        ] >= 0
        and
        a[
            "age_minus_v2_macro_f1"
        ] >= 0
        and
        b.loc[
            "accuracy",
            "observed_age_minus_v2",
        ] > 0
        and
        b.loc[
            "macro_f1",
            "observed_age_minus_v2",
        ] > 0
        and
        b.loc[
            "accuracy",
            "p_age_gt_v2",
        ] >= 0.90
        and
        b.loc[
            "macro_f1",
            "p_age_gt_v2",
        ] >= 0.90
        and
        b.loc[
            "top5_accuracy",
            "observed_age_minus_v2",
        ] >= 0
        and
        b.loc[
            "mrr",
            "observed_age_minus_v2",
        ] >= 0
    )

    summary_df.to_csv(
        out
        / "07F_01_age_aware_summary.csv",
        index=False,
    )

    comparison_df.to_csv(
        out
        / "07F_01_age_vs_v2.csv",
        index=False,
    )

    bootstrap_df.to_csv(
        out
        / "07F_01_day_block_bootstrap.csv",
        index=False,
    )

    seeds_df.to_csv(
        out
        / "07F_01_seed_metrics.csv",
        index=False,
    )

    per_day_df.to_csv(
        out
        / "07F_01_per_day_deltas.csv",
        index=False,
    )

    per_class_df.to_csv(
        out
        / "07F_01_per_class_deltas.csv",
        index=False,
    )

    age_audit_df.to_csv(
        out
        / "07F_01_age_audit.csv",
        index=False,
    )

    manifest = {
        "stage":
            "07F_01_query_age_aware_ltd",

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

            "temporal_encoding":
                "ordinal sinusoidal position",
        },

        "candidate": {
            "name":
                "QUERY_AGE_AWARE",

            "context_days":
                CONTEXT_DAYS,

            "alpha":
                ALPHA,

            "seeds":
                mod07a.SEEDS,

            "temporal_encoding":
                "actual query-to-profile age in days using sinusoidal encoding",

            "ordinal_position_encoding":
                False,

            "trainable_parameters":
                age_params,

            "capacity_matched":
                bool(
                    age_params
                    ==
                    reference_params
                ),
        },

        "hypothesis":
            (
                "absolute staleness of historical profiles "
                "contains useful longitudinal information "
                "beyond ordinal recency position"
            ),

        "promotion_gate": {
            "passed":
                age_pass,

            "requirements": [
                "Fold-A Accuracy delta vs V2 >= 0",
                "Fold-A Macro-F1 delta vs V2 >= 0",
                "Fold-B Accuracy delta vs V2 > 0",
                "Fold-B Macro-F1 delta vs V2 > 0",
                "Fold-B bootstrap P(delta Accuracy > 0) >= 0.90",
                "Fold-B bootstrap P(delta Macro-F1 > 0) >= 0.90",
                "Fold-B Top-5 delta >= 0",
                "Fold-B MRR delta >= 0",
            ],
        },

        "next_if_pass":
            (
                "promote to MACRO-LTD-V3 "
                "and perform final Macro robustness/freeze review"
            ),

        "next_if_fail":
            (
                "retain MACRO-LTD-V2; "
                "allow at most one final targeted Macro hypothesis "
                "only if a concrete failure mode is identified"
            ),

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
        / "07F_01_manifest_age_aware.json",
        manifest,
    )

    print()
    print("=" * 78)
    print(
        "AGE-AWARE SUMMARY"
    )
    print("=" * 78)

    print(
        summary_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 78)
    print(
        "AGE-AWARE VS V2"
    )
    print("=" * 78)

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
        "AGE-AWARE PROMOTION PASS:",
        age_pass,
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
