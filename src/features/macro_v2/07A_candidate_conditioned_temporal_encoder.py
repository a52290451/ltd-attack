from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import math
import random
import re
import time

import numpy as np
import pandas as pd

import sklearn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)
from sklearn.preprocessing import StandardScaler

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


# ============================================================
# PREDECLARED DEVELOPMENT CONFIGURATION
# ============================================================

DEV_SPLITS = [
    "DEV_EARLY",
    "DEV_MIDDLE",
    "DEV_LATE",
]

FOLDS = [
    {
        "fold": "A_EARLY_TO_MIDDLE",
        "train_splits": [
            "DEV_EARLY",
        ],
        "test_split":
            "DEV_MIDDLE",
    },
    {
        "fold": "B_EARLY_MIDDLE_TO_LATE",
        "train_splits": [
            "DEV_EARLY",
            "DEV_MIDDLE",
        ],
        "test_split":
            "DEV_LATE",
    },
]

SEEDS = [
    11,
    42,
    73,
]

CONTEXT_DAYS = 7

INPUT_DIM = 128
LATENT_DIM = 64

NUM_HEADS = 4
NUM_LAYERS = 2
FF_DIM = 128

DROPOUT = 0.10

EPOCHS = 20
BATCH_SIZE = 256

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

GRAD_CLIP = 1.0


# ============================================================
# GENERAL UTILITIES
# ============================================================

def repo_root() -> Path:
    return (
        Path(__file__)
        .resolve()
        .parents[3]
    )


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(
        torch.backends,
        "cudnn",
    ):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_frozen_features():
    path = (
        repo_root()
        / "configs/features/MACRO-V2-BASE-128.txt"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    features = [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip()
    ]

    if len(features) != 128:
        raise RuntimeError(
            "Expected exactly 128 frozen "
            f"BASE features, got {len(features)}."
        )

    if len(features) != len(set(features)):
        raise RuntimeError(
            "Frozen BASE list contains duplicates."
        )

    return features, path


def normalize_date(value):
    if pd.isna(value):
        return None

    text = str(value).strip()

    parsed = pd.to_datetime(
        text,
        errors="coerce",
    )

    if not pd.isna(parsed):
        return parsed.normalize()

    match = re.search(
        r"(\d{8})",
        text,
    )

    if match:
        parsed = pd.to_datetime(
            match.group(1),
            format="%Y%m%d",
            errors="coerce",
        )

        if not pd.isna(parsed):
            return parsed.normalize()

    return None


def derive_dates(df):
    candidates = [
        "_audit_date",
        "date",
        "date_id",
        "capture_date",
        "pcap_name",
    ]

    for col in candidates:
        if col not in df.columns:
            continue

        values = df[
            col
        ].apply(
            normalize_date
        )

        if (
            values.notna().mean()
            > 0.95
        ):
            return values, col

    raise RuntimeError(
        "Could not derive capture dates."
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


def fit_preprocessing(
    train_df,
    features,
):
    x = numeric_frame(
        train_df,
        features,
    )

    medians = x.median(
        axis=0,
        skipna=True,
    )

    all_missing = (
        medians[
            medians.isna()
        ]
        .index
        .tolist()
    )

    medians = (
        medians
        .fillna(0.0)
    )

    x_imp = (
        x
        .fillna(medians)
        .to_numpy(
            dtype=np.float32
        )
    )

    if not np.isfinite(
        x_imp
    ).all():
        raise RuntimeError(
            "Training preprocessing "
            "contains non-finite values."
        )

    scaler = StandardScaler()

    scaler.fit(
        x_imp
    )

    return (
        medians,
        scaler,
        all_missing,
    )


def transform(
    df,
    features,
    medians,
    scaler,
):
    x = (
        numeric_frame(
            df,
            features,
        )
        .fillna(
            medians
        )
        .to_numpy(
            dtype=np.float32
        )
    )

    z = scaler.transform(
        x
    ).astype(
        np.float32
    )

    if not np.isfinite(
        z
    ).all():
        raise RuntimeError(
            "Non-finite standardized values."
        )

    return z


# ============================================================
# CONTEXT CONSTRUCTION
# ============================================================

def standardized_daily_frame(
    train_daily,
    features,
    medians,
    scaler,
):
    z = transform(
        train_daily,
        features,
        medians,
        scaler,
    )

    out = pd.DataFrame(
        z,
        columns=features,
    )

    out.insert(
        0,
        "date",
        pd.to_datetime(
            train_daily[
                "date"
            ]
        ).to_numpy(),
    )

    out.insert(
        0,
        "site_label",
        train_daily[
            "site_label"
        ]
        .astype(str)
        .to_numpy(),
    )

    return (
        out
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


def build_context_for_date(
    daily_history,
    candidate_labels,
    cutoff_date,
    features,
):
    """
    Construct one candidate history tensor:

        [65 candidates, CONTEXT_DAYS, 128]

    Only records with date < cutoff_date are allowed.

    Padding occurs on the LEFT so that the most recent
    observed day is always the final valid token.
    """

    cutoff_date = pd.Timestamp(
        cutoff_date
    )

    contexts = np.zeros(
        (
            len(candidate_labels),
            CONTEXT_DAYS,
            len(features),
        ),
        dtype=np.float32,
    )

    # True = padding for PyTorch Transformer.
    padding_mask = np.ones(
        (
            len(candidate_labels),
            CONTEXT_DAYS,
        ),
        dtype=bool,
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
                    < cutoff_date
                )
            ]
            .sort_values(
                "date"
            )
        )

        if hist.empty:
            return None

        values = (
            hist[
                features
            ]
            .tail(
                CONTEXT_DAYS
            )
            .to_numpy(
                dtype=np.float32
            )
        )

        n = len(values)

        contexts[
            candidate_idx,
            -n:,
            :,
        ] = values

        padding_mask[
            candidate_idx,
            -n:,
        ] = False

        history_lengths.append(
            n
        )

    return (
        contexts,
        padding_mask,
        history_lengths,
    )


def build_context_cache(
    daily_history,
    candidate_labels,
    query_dates,
    features,
):
    cache = {}
    skipped_dates = []

    for date in sorted(
        pd.unique(
            pd.to_datetime(
                query_dates
            )
        )
    ):
        result = build_context_for_date(
            daily_history,
            candidate_labels,
            date,
            features,
        )

        if result is None:
            skipped_dates.append(
                pd.Timestamp(
                    date
                )
            )

            continue

        cache[
            pd.Timestamp(
                date
            )
        ] = result

    return (
        cache,
        skipped_dates,
    )


# ============================================================
# MODELS
# ============================================================

def sinusoidal_encoding(
    length,
    dim,
):
    pe = torch.zeros(
        length,
        dim,
        dtype=torch.float32,
    )

    position = torch.arange(
        length,
        dtype=torch.float32,
    ).unsqueeze(1)

    div_term = torch.exp(
        torch.arange(
            0,
            dim,
            2,
            dtype=torch.float32,
        )
        * (
            -math.log(10000.0)
            / dim
        )
    )

    pe[
        :,
        0::2
    ] = torch.sin(
        position
        * div_term
    )

    pe[
        :,
        1::2
    ] = torch.cos(
        position
        * div_term
    )

    return pe


class QueryEncoder(
    nn.Module
):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                INPUT_DIM,
                LATENT_DIM,
            ),
            nn.GELU(),
            nn.LayerNorm(
                LATENT_DIM
            ),
            nn.Dropout(
                DROPOUT
            ),
        )

    def forward(
        self,
        x,
    ):
        return self.net(x)


class BaseMLPClassifier(
    nn.Module
):
    """
    Matched neural control.

    It uses exactly the same query encoder
    dimensionality but NO longitudinal context.
    """

    def __init__(
        self,
        n_classes,
    ):
        super().__init__()

        self.query_encoder = (
            QueryEncoder()
        )

        self.classifier = nn.Linear(
            LATENT_DIM,
            n_classes,
        )

    def forward(
        self,
        query,
    ):
        q = self.query_encoder(
            query
        )

        return self.classifier(
            q
        )


class CandidateTemporalEncoder(
    nn.Module
):
    def __init__(self):
        super().__init__()

        self.input_projection = nn.Sequential(
            nn.Linear(
                INPUT_DIM,
                LATENT_DIM,
            ),
            nn.GELU(),
            nn.LayerNorm(
                LATENT_DIM
            ),
        )

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=LATENT_DIM,
                nhead=NUM_HEADS,
                dim_feedforward=FF_DIM,
                dropout=DROPOUT,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
        )

        self.transformer = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=NUM_LAYERS,
            )
        )

        self.final_norm = (
            nn.LayerNorm(
                LATENT_DIM
            )
        )

        self.register_buffer(
            "position_encoding",
            sinusoidal_encoding(
                CONTEXT_DAYS,
                LATENT_DIM,
            ),
        )

    def forward(
        self,
        history,
        padding_mask,
    ):
        """
        history:
            [65, W, 128]

        padding_mask:
            [65, W]
            True = padded token
        """

        x = self.input_projection(
            history
        )

        x = (
            x
            + self.position_encoding[
                None,
                :,
                :
            ]
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

        denom = valid.sum(
            dim=1
        ).clamp_min(
            1.0
        )

        pooled = (
            x
            * valid
        ).sum(
            dim=1
        ) / denom

        return pooled


class LTDPairScorer(
    nn.Module
):
    """
    Query BASE-128 versus every candidate
    longitudinal history.

    No candidate identity embedding exists.
    """

    def __init__(self):
        super().__init__()

        self.query_encoder = (
            QueryEncoder()
        )

        self.temporal_encoder = (
            CandidateTemporalEncoder()
        )

        pair_dim = (
            LATENT_DIM
            * 4
        )

        self.pair_head = nn.Sequential(
            nn.Linear(
                pair_dim,
                128,
            ),
            nn.GELU(),
            nn.Dropout(
                DROPOUT
            ),
            nn.Linear(
                128,
                1,
            ),
        )

    def forward(
        self,
        query,
        candidate_history,
        candidate_padding_mask,
    ):
        """
        query:
            [B, 128]

        candidate_history:
            [65, W, 128]

        logits:
            [B, 65]
        """

        q = self.query_encoder(
            query
        )

        c = self.temporal_encoder(
            candidate_history,
            candidate_padding_mask,
        )

        B = q.shape[0]
        C = c.shape[0]

        q_exp = q[
            :,
            None,
            :
        ].expand(
            B,
            C,
            LATENT_DIM,
        )

        c_exp = c[
            None,
            :,
            :
        ].expand(
            B,
            C,
            LATENT_DIM,
        )

        pair = torch.cat(
            [
                q_exp,
                c_exp,
                torch.abs(
                    q_exp
                    - c_exp
                ),
                q_exp
                * c_exp,
            ],
            dim=-1,
        )

        logits = (
            self.pair_head(
                pair
            )
            .squeeze(
                -1
            )
        )

        return logits


# ============================================================
# TRAINING
# ============================================================

def count_parameters(
    model,
):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def make_label_mapping(
    candidate_labels,
):
    return {
        label: i
        for i, label
        in enumerate(
            candidate_labels
        )
    }


def labels_to_indices(
    labels,
    label_to_idx,
):
    return np.array(
        [
            label_to_idx[
                str(x)
            ]
            for x in labels
        ],
        dtype=np.int64,
    )


def train_base_model(
    model,
    x_train,
    y_train,
    device,
    seed,
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    rng = np.random.default_rng(
        seed
    )

    final_loss = None

    model.train()

    for epoch in range(
        EPOCHS
    ):
        order = rng.permutation(
            len(x_train)
        )

        total_loss = 0.0
        total_n = 0

        for start in range(
            0,
            len(order),
            BATCH_SIZE,
        ):
            idx = order[
                start:
                start
                + BATCH_SIZE
            ]

            x = torch.from_numpy(
                x_train[
                    idx
                ]
            ).to(
                device
            )

            y = torch.from_numpy(
                y_train[
                    idx
                ]
            ).to(
                device
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                x
            )

            loss = F.cross_entropy(
                logits,
                y,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP,
            )

            optimizer.step()

            n = len(idx)

            total_loss += (
                float(
                    loss.item()
                )
                * n
            )

            total_n += n

        final_loss = (
            total_loss
            / total_n
        )

    return float(
        final_loss
    )


def train_ltd_model(
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
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    rng = np.random.default_rng(
        seed
    )

    usable_dates = sorted(
        context_cache.keys()
    )

    index_by_date = {}

    for date in usable_dates:
        idx = np.flatnonzero(
            pd.to_datetime(
                train_dates
            )
            == date
        )

        if len(idx):
            index_by_date[
                date
            ] = idx

    if not index_by_date:
        raise RuntimeError(
            "No causal training dates "
            "available for LTD."
        )

    final_loss = None

    model.train()

    for epoch in range(
        EPOCHS
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

            for start in range(
                0,
                len(query_indices),
                BATCH_SIZE,
            ):
                idx = query_indices[
                    start:
                    start
                    + BATCH_SIZE
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
                    GRAD_CLIP,
                )

                optimizer.step()

                n = len(idx)

                total_loss += (
                    float(
                        loss.item()
                    )
                    * n
                )

                total_n += n

        final_loss = (
            total_loss
            / total_n
        )

    return (
        float(
            final_loss
        ),
        int(
            sum(
                len(x)
                for x in index_by_date.values()
            )
        ),
        len(
            index_by_date
        ),
    )


# ============================================================
# EVALUATION
# ============================================================

def metrics_from_logits(
    logits,
    y_true,
):
    order = np.argsort(
        -logits,
        axis=1,
    )

    pred = order[
        :,
        0
    ]

    ranks = (
        np.argmax(
            order
            == y_true[
                :,
                None
            ],
            axis=1,
        )
        + 1
    )

    return {
        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    pred,
                )
            ),

        "macro_f1":
            float(
                f1_score(
                    y_true,
                    pred,
                    average="macro",
                    zero_division=0,
                )
            ),

        "top5_accuracy":
            float(
                np.mean(
                    ranks
                    <= 5
                )
            ),

        "mrr":
            float(
                np.mean(
                    1.0
                    / ranks
                )
            ),

        "mean_true_rank":
            float(
                np.mean(
                    ranks
                )
            ),

        "median_true_rank":
            float(
                np.median(
                    ranks
                )
            ),
    }


def evaluate_base(
    model,
    x_test,
    y_test,
    device,
):
    model.eval()

    logits_all = []

    with torch.no_grad():
        for start in range(
            0,
            len(x_test),
            BATCH_SIZE,
        ):
            x = (
                torch.from_numpy(
                    x_test[
                        start:
                        start
                        + BATCH_SIZE
                    ]
                )
                .to(
                    device
                )
            )

            logits = model(
                x
            )

            logits_all.append(
                logits
                .cpu()
                .numpy()
            )

    logits_all = np.vstack(
        logits_all
    )

    return metrics_from_logits(
        logits_all,
        y_test,
    )


def evaluate_ltd(
    model,
    x_test,
    y_test,
    test_dates,
    context_cache,
    device,
):
    model.eval()

    logits_all = []
    labels_all = []

    unique_dates = sorted(
        pd.unique(
            pd.to_datetime(
                test_dates
            )
        )
    )

    with torch.no_grad():
        for date in unique_dates:
            if date not in context_cache:
                raise RuntimeError(
                    f"No candidate context "
                    f"available for test date {date}"
                )

            idx = np.flatnonzero(
                pd.to_datetime(
                    test_dates
                )
                == date
            )

            (
                context_np,
                padding_np,
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

            for start in range(
                0,
                len(idx),
                BATCH_SIZE,
            ):
                batch_idx = idx[
                    start:
                    start
                    + BATCH_SIZE
                ]

                x = (
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
                    x,
                    context,
                    padding,
                )

                logits_all.append(
                    logits
                    .cpu()
                    .numpy()
                )

                labels_all.append(
                    y_test[
                        batch_idx
                    ]
                )

    logits_all = np.vstack(
        logits_all
    )

    labels_all = np.concatenate(
        labels_all
    )

    return metrics_from_logits(
        logits_all,
        labels_all,
    )


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():
    print("=" * 78)

    print(
        "MACRO-V2-HIST — "
        "07A CANDIDATE-CONDITIONED "
        "TEMPORAL ENCODER"
    )

    print("=" * 78)

    print(
        "BASE-128: FROZEN"
    )

    print(
        f"Context window: "
        f"up to {CONTEXT_DAYS} prior site-days"
    )

    print(
        "Test observations do NOT update "
        "candidate histories."
    )

    print(
        "Every LTD query is scored against "
        "ALL 65 candidate histories."
    )

    print(
        "INTERNAL_TEST: PROHIBITED"
    )

    print(
        "External Future: PROHIBITED"
    )

    (
        features,
        frozen_path,
    ) = load_frozen_features()

    out = output_dir()

    daily_path = (
        out
        / "06C_daily_base_profiles.csv"
    )

    assignments_path = (
        out
        / "01_temporal_split_assignments.csv"
    )

    if not daily_path.exists():
        raise FileNotFoundError(
            daily_path
        )

    if not assignments_path.exists():
        raise FileNotFoundError(
            assignments_path
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

    (
        captures,
        dataset_path,
    ) = load_historical_65()

    assignments = pd.read_csv(
        assignments_path
    )

    if assignments[
        "pcap_uid"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate split assignment UID."
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

    captures = (
        captures
        .merge(
            dev_assignments,
            on="pcap_uid",
            how="inner",
            validate="one_to_one",
        )
    )

    if len(captures) != 57916:
        raise RuntimeError(
            "Expected 57,916 DEV captures, "
            f"found {len(captures)}."
        )

    captures[
        "site_label"
    ] = (
        captures[
            "site_label"
        ]
        .astype(str)
    )

    capture_dates, date_source = (
        derive_dates(
            captures
        )
    )

    captures = captures.copy()

    captures[
        "_query_date"
    ] = pd.to_datetime(
        capture_dates
    )

    if captures[
        "_query_date"
    ].isna().any():
        raise RuntimeError(
            "Some DEV captures have "
            "no usable date."
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
    ) != 65:
        raise RuntimeError(
            "Expected 65 candidate labels."
        )

    label_to_idx = (
        make_label_mapping(
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

    print(
        f"PyTorch: {torch.__version__}"
    )

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            ),
        )

    runs = []

    for fold_spec in FOLDS:
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

        train_caps = (
            captures[
                captures[
                    "temporal_split"
                ].isin(
                    train_splits
                )
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        test_caps = (
            captures[
                captures[
                    "temporal_split"
                ]
                == test_split
            ]
            .copy()
            .reset_index(
                drop=True
            )
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
            .reset_index(
                drop=True
            )
        )

        if (
            train_caps[
                "site_label"
            ].nunique()
            != 65
        ):
            raise RuntimeError(
                f"{fold}: training does not "
                "contain all 65 sites."
            )

        if (
            test_caps[
                "site_label"
            ].nunique()
            != 65
        ):
            raise RuntimeError(
                f"{fold}: test does not "
                "contain all 65 sites."
            )

        (
            medians,
            scaler,
            all_missing,
        ) = fit_preprocessing(
            train_caps,
            features,
        )

        if all_missing:
            raise RuntimeError(
                f"{fold}: all-missing "
                f"training features: "
                f"{all_missing}"
            )

        x_train = transform(
            train_caps,
            features,
            medians,
            scaler,
        )

        x_test = transform(
            test_caps,
            features,
            medians,
            scaler,
        )

        y_train = labels_to_indices(
            train_caps[
                "site_label"
            ],
            label_to_idx,
        )

        y_test = labels_to_indices(
            test_caps[
                "site_label"
            ],
            label_to_idx,
        )

        train_dates = (
            pd.to_datetime(
                train_caps[
                    "_query_date"
                ]
            )
            .to_numpy()
        )

        test_dates = (
            pd.to_datetime(
                test_caps[
                    "_query_date"
                ]
            )
            .to_numpy()
        )

        daily_history = (
            standardized_daily_frame(
                train_daily,
                features,
                medians,
                scaler,
            )
        )

        # ----------------------------------------------------
        # TRAIN context:
        # query at t only sees site-days < t.
        # ----------------------------------------------------

        (
            train_context_cache,
            skipped_train_dates,
        ) = build_context_cache(
            daily_history,
            candidate_labels,
            train_dates,
            features,
        )

        usable_train_dates = set(
            train_context_cache.keys()
        )

        valid_train_mask = np.array(
            [
                pd.Timestamp(d)
                in usable_train_dates
                for d in train_dates
            ],
            dtype=bool,
        )

        if not valid_train_mask.any():
            raise RuntimeError(
                f"{fold}: no causal "
                "training queries."
            )

        # Matched BASE control:
        # it receives exactly the same training
        # queries as LTD.
        x_train_matched = (
            x_train[
                valid_train_mask
            ]
        )

        y_train_matched = (
            y_train[
                valid_train_mask
            ]
        )

        train_dates_matched = (
            train_dates[
                valid_train_mask
            ]
        )

        # ----------------------------------------------------
        # TEST context:
        # constructed ONLY from fold training daily history.
        # Therefore no test observation can update context.
        # ----------------------------------------------------

        (
            test_context_cache,
            skipped_test_dates,
        ) = build_context_cache(
            daily_history,
            candidate_labels,
            test_dates,
            features,
        )

        if skipped_test_dates:
            raise RuntimeError(
                f"{fold}: missing historical "
                f"context for test dates: "
                f"{skipped_test_dates}"
            )

        max_train_history_date = (
            daily_history[
                "date"
            ].max()
        )

        min_test_date = (
            pd.to_datetime(
                test_dates
            ).min()
        )

        if (
            max_train_history_date
            >= min_test_date
        ):
            raise RuntimeError(
                f"{fold}: temporal boundary "
                "violation."
            )

        history_lengths = []

        for (
            _ctx,
            _mask,
            lengths,
        ) in test_context_cache.values():
            history_lengths.extend(
                lengths
            )

        print(
            "Train captures:",
            len(train_caps),
        )

        print(
            "Matched train queries:",
            len(
                x_train_matched
            ),
        )

        print(
            "Test queries:",
            len(x_test),
        )

        print(
            "Train daily profiles:",
            len(train_daily),
        )

        print(
            "Skipped earliest train dates:",
            len(
                skipped_train_dates
            ),
        )

        print(
            "Test context length "
            "(min/median/max):",
            min(history_lengths),
            float(
                np.median(
                    history_lengths
                )
            ),
            max(history_lengths),
        )

        for seed in SEEDS:
            print()
            print(
                f"{fold} — seed={seed}"
            )

            # ================================================
            # MATCHED BASE MLP
            # ================================================

            set_seed(seed)

            base_model = (
                BaseMLPClassifier(
                    n_classes=65
                )
                .to(
                    device
                )
            )

            start = time.perf_counter()

            base_train_loss = (
                train_base_model(
                    base_model,
                    x_train_matched,
                    y_train_matched,
                    device,
                    seed,
                )
            )

            base_seconds = (
                time.perf_counter()
                - start
            )

            base_metrics = (
                evaluate_base(
                    base_model,
                    x_test,
                    y_test,
                    device,
                )
            )

            print(
                "  BASE_MLP "
                f"acc={base_metrics['accuracy']:.4f} "
                f"F1={base_metrics['macro_f1']:.4f} "
                f"Top5={base_metrics['top5_accuracy']:.4f} "
                f"MRR={base_metrics['mrr']:.4f}"
            )

            runs.append(
                {
                    "fold":
                        fold,

                    "model":
                        "base_mlp",

                    "seed":
                        seed,

                    "train_splits":
                        "|".join(
                            train_splits
                        ),

                    "test_split":
                        test_split,

                    "context_days":
                        0,

                    "n_train_queries":
                        len(
                            x_train_matched
                        ),

                    "n_test_queries":
                        len(
                            x_test
                        ),

                    "train_loss":
                        base_train_loss,

                    "fit_seconds":
                        base_seconds,

                    "parameters":
                        count_parameters(
                            base_model
                        ),

                    **base_metrics,
                }
            )

            del base_model

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # ================================================
            # LTD TEMPORAL PAIR MODEL
            # ================================================

            set_seed(seed)

            ltd_model = (
                LTDPairScorer()
                .to(
                    device
                )
            )

            start = time.perf_counter()

            (
                ltd_train_loss,
                ltd_train_n,
                ltd_train_dates_n,
            ) = train_ltd_model(
                ltd_model,
                x_train,
                y_train,
                train_dates,
                train_context_cache,
                device,
                seed,
            )

            ltd_seconds = (
                time.perf_counter()
                - start
            )

            if (
                ltd_train_n
                != len(
                    x_train_matched
                )
            ):
                raise RuntimeError(
                    "Matched control and LTD "
                    "did not use the same number "
                    "of training queries."
                )

            ltd_metrics = (
                evaluate_ltd(
                    ltd_model,
                    x_test,
                    y_test,
                    test_dates,
                    test_context_cache,
                    device,
                )
            )

            print(
                "  LTD_TEMP "
                f"acc={ltd_metrics['accuracy']:.4f} "
                f"F1={ltd_metrics['macro_f1']:.4f} "
                f"Top5={ltd_metrics['top5_accuracy']:.4f} "
                f"MRR={ltd_metrics['mrr']:.4f}"
            )

            runs.append(
                {
                    "fold":
                        fold,

                    "model":
                        "ltd_temporal_pair",

                    "seed":
                        seed,

                    "train_splits":
                        "|".join(
                            train_splits
                        ),

                    "test_split":
                        test_split,

                    "context_days":
                        CONTEXT_DAYS,

                    "n_train_queries":
                        ltd_train_n,

                    "n_train_context_dates":
                        ltd_train_dates_n,

                    "n_test_queries":
                        len(
                            x_test
                        ),

                    "train_loss":
                        ltd_train_loss,

                    "fit_seconds":
                        ltd_seconds,

                    "parameters":
                        count_parameters(
                            ltd_model
                        ),

                    **ltd_metrics,
                }
            )

            del ltd_model

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ========================================================
    # OUTPUTS
    # ========================================================

    runs_df = pd.DataFrame(
        runs
    )

    runs_path = (
        out
        / "07A_model_runs.csv"
    )

    runs_df.to_csv(
        runs_path,
        index=False,
    )

    fold_summary = (
        runs_df
        .groupby(
            [
                "model",
                "fold",
            ],
            as_index=False,
        )
        .agg(
            seeds=(
                "seed",
                "nunique",
            ),

            accuracy_mean=(
                "accuracy",
                "mean",
            ),

            accuracy_std=(
                "accuracy",
                "std",
            ),

            macro_f1_mean=(
                "macro_f1",
                "mean",
            ),

            macro_f1_std=(
                "macro_f1",
                "std",
            ),

            top5_mean=(
                "top5_accuracy",
                "mean",
            ),

            top5_std=(
                "top5_accuracy",
                "std",
            ),

            mrr_mean=(
                "mrr",
                "mean",
            ),

            mrr_std=(
                "mrr",
                "std",
            ),

            mean_true_rank_mean=(
                "mean_true_rank",
                "mean",
            ),
        )
    )

    fold_summary_path = (
        out
        / "07A_fold_summary.csv"
    )

    fold_summary.to_csv(
        fold_summary_path,
        index=False,
    )

    pivot = (
        fold_summary
        .pivot(
            index="fold",
            columns="model",
            values=[
                "accuracy_mean",
                "macro_f1_mean",
                "top5_mean",
                "mrr_mean",
            ],
        )
    )

    delta_rows = []

    for fold in sorted(
        fold_summary[
            "fold"
        ].unique()
    ):
        b = (
            fold_summary[
                (
                    fold_summary[
                        "fold"
                    ]
                    == fold
                )
                &
                (
                    fold_summary[
                        "model"
                    ]
                    == "base_mlp"
                )
            ]
            .iloc[0]
        )

        l = (
            fold_summary[
                (
                    fold_summary[
                        "fold"
                    ]
                    == fold
                )
                &
                (
                    fold_summary[
                        "model"
                    ]
                    == "ltd_temporal_pair"
                )
            ]
            .iloc[0]
        )

        delta_rows.append(
            {
                "fold":
                    fold,

                "delta_accuracy_ltd_vs_base":
                    float(
                        l[
                            "accuracy_mean"
                        ]
                        -
                        b[
                            "accuracy_mean"
                        ]
                    ),

                "delta_macro_f1_ltd_vs_base":
                    float(
                        l[
                            "macro_f1_mean"
                        ]
                        -
                        b[
                            "macro_f1_mean"
                        ]
                    ),

                "delta_top5_ltd_vs_base":
                    float(
                        l[
                            "top5_mean"
                        ]
                        -
                        b[
                            "top5_mean"
                        ]
                    ),

                "delta_mrr_ltd_vs_base":
                    float(
                        l[
                            "mrr_mean"
                        ]
                        -
                        b[
                            "mrr_mean"
                        ]
                    ),
            }
        )

    delta_df = pd.DataFrame(
        delta_rows
    )

    delta_path = (
        out
        / "07A_ltd_vs_base_delta.csv"
    )

    delta_df.to_csv(
        delta_path,
        index=False,
    )

    model_summary_rows = []

    for model, group in (
        fold_summary
        .groupby(
            "model"
        )
    ):
        model_summary_rows.append(
            {
                "model":
                    model,

                "worst_fold_macro_f1":
                    float(
                        group[
                            "macro_f1_mean"
                        ].min()
                    ),

                "mean_macro_f1":
                    float(
                        group[
                            "macro_f1_mean"
                        ].mean()
                    ),

                "worst_fold_accuracy":
                    float(
                        group[
                            "accuracy_mean"
                        ].min()
                    ),

                "mean_accuracy":
                    float(
                        group[
                            "accuracy_mean"
                        ].mean()
                    ),

                "worst_fold_top5":
                    float(
                        group[
                            "top5_mean"
                        ].min()
                    ),

                "mean_top5":
                    float(
                        group[
                            "top5_mean"
                        ].mean()
                    ),

                "worst_fold_mrr":
                    float(
                        group[
                            "mrr_mean"
                        ].min()
                    ),

                "mean_mrr":
                    float(
                        group[
                            "mrr_mean"
                        ].mean()
                    ),
            }
        )

    model_summary = (
        pd.DataFrame(
            model_summary_rows
        )
        .sort_values(
            [
                "worst_fold_macro_f1",
                "mean_macro_f1",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    model_summary_path = (
        out
        / "07A_model_summary.csv"
    )

    model_summary.to_csv(
        model_summary_path,
        index=False,
    )

    manifest = {
        "feature_set_id":
            "MACRO-V2-BASE-128",

        "stage":
            "07A_candidate_conditioned_temporal_encoder",

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "git_commit":
            git_commit(),

        "environment": {
            "python_model":
                "PyTorch",

            "torch":
                torch.__version__,

            "sklearn":
                sklearn.__version__,

            "device":
                str(
                    device
                ),

            "cuda_available":
                bool(
                    torch.cuda.is_available()
                ),
        },

        "inputs": {
            "historical_dataset": {
                "path":
                    str(
                        dataset_path
                    ),

                "sha256":
                    sha256(
                        dataset_path
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

            "frozen_base_features": {
                "path":
                    str(
                        frozen_path
                    ),

                "sha256":
                    sha256(
                        frozen_path
                    ),

                "count":
                    128,
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
        },

        "data_policy": {
            "development_only":
                True,

            "internal_test_values_used":
                False,

            "external_future_used":
                False,

            "test_observations_update_history":
                False,

            "same_day_context_allowed":
                False,

            "candidate_identity_embedding":
                False,

            "query_true_label_used_to_select_context":
                False,

            "query_scored_against_all_65_candidates":
                True,
        },

        "architecture": {
            "input_dimensions":
                INPUT_DIM,

            "latent_dimensions":
                LATENT_DIM,

            "context_days":
                CONTEXT_DAYS,

            "context_window_origin":
                (
                    "DEV design choice informed by "
                    "06E recent-7 diagnostic; "
                    "not an independent result"
                ),

            "temporal_encoder": {
                "type":
                    "TransformerEncoder",

                "layers":
                    NUM_LAYERS,

                "heads":
                    NUM_HEADS,

                "feedforward_dim":
                    FF_DIM,

                "dropout":
                    DROPOUT,

                "positional_encoding":
                    "fixed sinusoidal",
            },

            "pair_features": [
                "query",
                "candidate_context",
                "absolute_difference",
                "elementwise_product",
            ],

            "pair_head":
                "256 -> 128 -> 1",

            "matched_control":
                "BASE MLP using same causal training queries",
        },

        "training": {
            "epochs":
                EPOCHS,

            "batch_size":
                BATCH_SIZE,

            "learning_rate":
                LEARNING_RATE,

            "weight_decay":
                WEIGHT_DECAY,

            "gradient_clip":
                GRAD_CLIP,

            "optimizer":
                "AdamW",

            "seeds":
                SEEDS,

            "early_stopping":
                False,

            "model_selection_on_test_fold":
                False,
        },

        "preprocessing": {
            "imputation":
                (
                    "feature medians fitted on "
                    "fold training captures only"
                ),

            "scaling":
                (
                    "StandardScaler fitted on "
                    "fold training captures only"
                ),
        },

        "temporal_folds":
            FOLDS,

        "interpretation": {
            "stage_type":
                (
                    "DEV architecture experiment; "
                    "not independent generalization "
                    "estimate"
                ),

            "primary_question":
                (
                    "does learned candidate "
                    "longitudinal context improve "
                    "single-capture classification "
                    "relative to a matched BASE-only "
                    "neural control?"
                ),

            "strong_external_dev_reference":
                (
                    "MACRO-V2-BASE-128 XGBoost "
                    "from 06B-R1 remains the stronger "
                    "non-neural reference and is not "
                    "replaced by this matched control"
                ),
        },
    }

    write_json(
        out
        / "07A_manifest_candidate_conditioned_temporal_encoder.json",
        manifest,
    )

    print()
    print("=" * 78)
    print(
        "07A FOLD SUMMARY"
    )
    print("=" * 78)

    print(
        fold_summary
        .sort_values(
            [
                "fold",
                "model",
            ]
        )
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 78)
    print(
        "LTD DELTA VS MATCHED BASE MLP"
    )
    print("=" * 78)

    print(
        delta_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 78)
    print(
        "MODEL SUMMARY"
    )
    print("=" * 78)

    print(
        model_summary.to_string(
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
        f"Runs: {runs_path}"
    )

    print(
        f"Fold summary: "
        f"{fold_summary_path}"
    )

    print(
        f"Delta: {delta_path}"
    )

    print(
        f"Model summary: "
        f"{model_summary_path}"
    )

    print()

    print(
        "07A CANDIDATE-CONDITIONED "
        "TEMPORAL ENCODER COMPLETE"
    )


if __name__ == "__main__":
    main()
