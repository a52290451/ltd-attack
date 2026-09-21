from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import math
import random
import time

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import (
    Dataset,
    DataLoader,
)

from src.utils.paths import (
    data_path,
    artifact_path,
    result_path,
)

from src.features.macro_v2._common import (
    extract_capture_dates,
    git_commit,
    load_elite_sites,
    sha256,
    write_json,
)


SOURCE = Path(
    data_path(
        "historical",
        "CLEAN_final_vectors_sites.csv",
    )
)

ASSIGNMENTS = Path(
    result_path(
        "feature_engineering",
        "MACRO-V2-HIST",
        "01_temporal_split_assignments.csv",
    )
)

OUT = Path(
    result_path(
        "micro",
        "MICRO-TEMPORAL-V1",
    )
)

ART = Path(
    artifact_path(
        "micro",
        "MICRO-TEMPORAL-V1",
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


MAX_LEN = 3000
BATCH_SIZE = 32
MAX_EPOCHS = 100

D_MODEL = 256
NHEAD = 8
NUM_LAYERS = 4

LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
LABEL_SMOOTHING = 0.1

EARLY_VAL_FRACTION = 0.20
PATIENCE = 10

SEED = 42


FOLDS = [
    {
        "fold":
            "A_EARLY_TO_MIDDLE",

        "train_splits":
            [
                "DEV_EARLY",
            ],

        "test_split":
            "DEV_MIDDLE",
    },

    {
        "fold":
            "B_EARLY_MIDDLE_TO_LATE",

        "train_splits":
            [
                "DEV_EARLY",
                "DEV_MIDDLE",
            ],

        "test_split":
            "DEV_LATE",
    },
]


def set_seed(seed):

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


def parse_vector(
    value,
    max_len,
    dtype,
):

    text = str(value).strip()

    if (
        text.startswith("[")
        and
        text.endswith("]")
    ):
        text = text[1:-1]

    arr = np.fromstring(
        text,
        sep=",",
        dtype=np.float32,
    )

    if len(arr) > max_len:
        arr = arr[:max_len]

    out = np.zeros(
        max_len,
        dtype=dtype,
    )

    if len(arr):
        out[
            :len(arr)
        ] = arr.astype(
            dtype,
            copy=False,
        )

    return out


def load_dev_vectors():

    print(
        "Loading canonical DEV Micro payload..."
    )

    assignments = pd.read_csv(
        ASSIGNMENTS,
        usecols=[
            "pcap_uid",
            "temporal_split",
        ],
    )

    assignments[
        "pcap_uid"
    ] = (
        assignments[
            "pcap_uid"
        ].astype(str)
    )

    dev_assign = (
        assignments[
            assignments[
                "temporal_split"
            ].isin(
                [
                    "DEV_EARLY",
                    "DEV_MIDDLE",
                    "DEV_LATE",
                ]
            )
        ]
        .copy()
    )

    dev_ids = set(
        dev_assign[
            "pcap_uid"
        ]
    )

    header = pd.read_csv(
        SOURCE,
        nrows=0,
    ).columns.tolist()

    required = [
        "pcap_uid",
        "site_label",
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
            f"Missing columns: {missing}"
        )

    optional = [
        x
        for x in [
            "date_id",
            "pcap_name",
        ]
        if x in header
    ]

    cols = (
        required
        +
        optional
    )

    chunks = []

    for chunk in pd.read_csv(
        SOURCE,
        usecols=cols,
        chunksize=4000,
    ):

        chunk[
            "pcap_uid"
        ] = (
            chunk[
                "pcap_uid"
            ].astype(str)
        )

        keep = (
            chunk[
                "pcap_uid"
            ].isin(
                dev_ids
            )
        )

        if keep.any():
            chunks.append(
                chunk.loc[
                    keep
                ].copy()
            )

    df = pd.concat(
        chunks,
        ignore_index=True,
    )

    df[
        "site_label"
    ] = (
        df[
            "site_label"
        ].astype(str)
    )

    df = df.merge(
        dev_assign,
        on="pcap_uid",
        how="inner",
        validate="one_to_one",
    )

    if len(df) != len(
        dev_assign
    ):
        raise RuntimeError(
            "DEV vector coverage mismatch."
        )

    elite = {
        str(x)
        for x in
        load_elite_sites()
    }

    if set(
        df[
            "site_label"
        ].unique()
    ) != elite:
        raise RuntimeError(
            "Micro site population mismatch."
        )

    df[
        "_date"
    ] = (
        extract_capture_dates(
            df
        )
    )

    if df[
        "_date"
    ].isna().any():
        raise RuntimeError(
            "Missing capture date."
        )

    labels = np.array(
        sorted(
            elite
        ),
        dtype=str,
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
            MAX_LEN,
        ),
        dtype=np.int8,
    )

    x_size = np.zeros(
        (
            n,
            MAX_LEN,
        ),
        dtype=np.float32,
    )

    print(
        f"Parsing {n:,} DEV captures..."
    )

    for i, (
        d,
        s,
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
        ] = parse_vector(
            d,
            MAX_LEN,
            np.int8,
        )

        x_size[
            i
        ] = parse_vector(
            s,
            MAX_LEN,
            np.float32,
        )

    unique_dir = set(
        np.unique(
            x_dir
        ).tolist()
    )

    if not unique_dir.issubset(
        {
            -1,
            0,
            1,
        }
    ):
        raise RuntimeError(
            f"Unexpected directions: "
            f"{sorted(unique_dir)}"
        )

    print(
        "DEV vectors loaded:",
        len(df),
    )

    print(
        "Sites:",
        len(labels),
    )

    return (
        df,
        x_dir,
        x_size,
        y,
        labels,
    )


class IndexedDataset(
    Dataset
):

    def __init__(
        self,
        x_dir,
        x_size,
        y,
        indices,
    ):

        self.x_dir = x_dir
        self.x_size = x_size
        self.y = y

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
        )


class PositionalEncoding(
    nn.Module
):

    def __init__(
        self,
        d_model,
        dropout=0.1,
        max_len=5000,
    ):
        super().__init__()

        self.dropout = (
            nn.Dropout(
                p=dropout
            )
        )

        pe = torch.zeros(
            max_len,
            d_model,
        )

        position = torch.arange(
            0,
            max_len,
            dtype=torch.float32,
        ).unsqueeze(
            1
        )

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2,
            ).float()
            *
            (
                -math.log(
                    10000.0
                )
                /
                d_model
            )
        )

        pe[
            :,
            0::2
        ] = torch.sin(
            position
            *
            div_term
        )

        pe[
            :,
            1::2
        ] = torch.cos(
            position
            *
            div_term
        )

        pe = pe.unsqueeze(
            0
        )

        self.register_buffer(
            "pe",
            pe,
        )

    def forward(
        self,
        x,
    ):

        x = (
            x
            +
            self.pe[
                :,
                :x.size(1),
                :
            ]
        )

        return self.dropout(
            x
        )


class MultimodalTransformer(
    nn.Module
):

    def __init__(
        self,
        num_classes,
    ):
        super().__init__()

        self.dir_embedding = (
            nn.Embedding(
                3,
                D_MODEL,
            )
        )

        self.weight_proj = (
            nn.Linear(
                1,
                D_MODEL,
            )
        )

        self.fusion = (
            nn.Linear(
                D_MODEL * 2,
                D_MODEL,
            )
        )

        self.conv_local = (
            nn.Conv1d(
                D_MODEL,
                D_MODEL,
                kernel_size=7,
                stride=2,
                padding=3,
            )
        )

        self.pos_encoder = (
            PositionalEncoding(
                D_MODEL,
                max_len=MAX_LEN,
            )
        )

        layer = (
            nn.TransformerEncoderLayer(
                d_model=D_MODEL,
                nhead=NHEAD,
                dim_feedforward=1024,
                dropout=0.2,
                batch_first=True,
                activation="gelu",
            )
        )

        self.transformer = (
            nn.TransformerEncoder(
                layer,
                num_layers=NUM_LAYERS,
            )
        )

        self.ln = (
            nn.LayerNorm(
                D_MODEL * 2
            )
        )

        self.fc = nn.Sequential(
            nn.Linear(
                D_MODEL * 2,
                512,
            ),
            nn.GELU(),
            nn.Dropout(
                0.4
            ),
            nn.Linear(
                512,
                num_classes,
            ),
        )

    def forward(
        self,
        x_dir,
        x_weight,
    ):

        e_dir = self.dir_embedding(
            x_dir
            + 1
        )

        e_weight = self.weight_proj(
            x_weight.unsqueeze(
                -1
            )
        )

        x = torch.cat(
            [
                e_dir,
                e_weight,
            ],
            dim=-1,
        )

        x = self.fusion(
            x
        )

        x = x.transpose(
            1,
            2,
        )

        x = self.conv_local(
            x
        )

        x = x.transpose(
            1,
            2,
        )

        x = self.pos_encoder(
            x
        )

        x = self.transformer(
            x
        )

        avg_pool = torch.mean(
            x,
            dim=1,
        )

        max_pool, _ = torch.max(
            x,
            dim=1,
        )

        x = torch.cat(
            (
                avg_pool,
                max_pool,
            ),
            dim=1,
        )

        x = self.ln(
            x
        )

        return self.fc(
            x
        )


def loader(
    x_dir,
    x_size,
    y,
    idx,
    shuffle,
):

    ds = IndexedDataset(
        x_dir,
        x_size,
        y,
        idx,
    )

    generator = (
        torch.Generator()
    )

    generator.manual_seed(
        SEED
    )

    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        generator=(
            generator
            if shuffle
            else None
        ),
        pin_memory=True,
        num_workers=0,
    )


def scaler_from_indices(
    x_size,
    idx,
):

    x = x_size[
        idx
    ]

    return (
        float(
            x.min()
        ),
        float(
            x.max()
        ),
    )


def normalize_size(
    x,
    min_value,
    max_value,
):

    return (
        x
        -
        min_value
    ) / (
        max_value
        -
        min_value
        +
        1e-8
    )


def evaluate(
    model,
    data_loader,
    min_size,
    max_size,
    device,
):

    model.eval()

    probs_all = []
    y_all = []

    amp_enabled = (
        device.type
        == "cuda"
    )

    with torch.inference_mode():

        for (
            b_dir,
            b_size,
            b_y,
        ) in data_loader:

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

            b_size = normalize_size(
                b_size,
                min_size,
                max_size,
            )

            with torch.autocast(
                device_type=
                    device.type,
                dtype=torch.bfloat16,
                enabled=amp_enabled,
            ):

                logits = model(
                    b_dir,
                    b_size,
                )

            probs = torch.softmax(
                logits.float(),
                dim=1,
            )

            probs_all.append(
                probs.cpu().numpy()
            )

            y_all.append(
                b_y.numpy()
            )

    probs = np.concatenate(
        probs_all,
        axis=0,
    )

    y_true = np.concatenate(
        y_all,
        axis=0,
    )

    order = np.argsort(
        -probs,
        axis=1,
    )

    pred = order[:, 0]

    ranks = (
        np.argmax(
            order
            ==
            y_true[:, None],
            axis=1,
        )
        + 1
    )

    metrics = {
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
                    labels=np.arange(65),
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
                ranks.mean()
            ),
    }

    return (
        metrics,
        probs,
        y_true,
    )


def train(
    x_dir,
    x_size,
    y,
    train_idx,
    device,
    epochs,
    val_idx=None,
):

    set_seed(
        SEED
    )

    min_size, max_size = (
        scaler_from_indices(
            x_size,
            train_idx,
        )
    )

    train_loader = loader(
        x_dir,
        x_size,
        y,
        train_idx,
        True,
    )

    val_loader = (
        loader(
            x_dir,
            x_size,
            y,
            val_idx,
            False,
        )
        if val_idx
        is not None
        else None
    )

    model = (
        MultimodalTransformer(
            65
        )
        .to(
            device
        )
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=
            WEIGHT_DECAY,
    )

    criterion = (
        nn.CrossEntropyLoss(
            label_smoothing=
                LABEL_SMOOTHING
        )
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
            == "cuda"
        ),
    )

    amp_enabled = (
        device.type
        == "cuda"
    )

    curve = []

    best_key = None
    best_epoch = None
    bad_epochs = 0

    for epoch in range(
        1,
        epochs + 1,
    ):

        model.train()

        loss_sum = 0.0
        n_batches = 0

        t0 = time.time()

        for (
            b_dir,
            b_size,
            b_y,
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

            b_size = normalize_size(
                b_size,
                min_size,
                max_size,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                device_type=
                    device.type,
                dtype=torch.bfloat16,
                enabled=amp_enabled,
            ):

                logits = model(
                    b_dir,
                    b_size,
                )

                loss = criterion(
                    logits,
                    b_y,
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
                time.time()
                -
                t0,
        }

        if val_loader is not None:

            m, _, _ = evaluate(
                model,
                val_loader,
                min_size,
                max_size,
                device,
            )

            row.update(
                {
                    f"val_{k}":
                        v
                    for k, v
                    in m.items()
                }
            )

            key = (
                m[
                    "macro_f1"
                ],
                m[
                    "accuracy"
                ],
                -epoch,
            )

            if (
                best_key is None
                or
                key > best_key
            ):

                best_key = key
                best_epoch = epoch
                bad_epochs = 0

            else:

                bad_epochs += 1

            print(
                f"epoch={epoch:03d} "
                f"loss={row['train_loss']:.4f} "
                f"val_acc={m['accuracy']:.4f} "
                f"val_f1={m['macro_f1']:.4f}"
            )

            if (
                bad_epochs
                >= PATIENCE
            ):
                break

        else:

            print(
                f"epoch={epoch:03d} "
                f"loss={row['train_loss']:.4f}"
            )

        curve.append(
            row
        )

    if val_loader is None:

        best_epoch = epochs

    return (
        model,
        min_size,
        max_size,
        best_epoch,
        pd.DataFrame(
            curve
        ),
    )


def main():

    print("=" * 78)
    print(
        "08B — CLEAN TEMPORAL MICRO BASELINE"
    )
    print("=" * 78)

    print(
        "Canonical source:",
        SOURCE,
    )

    print(
        "Temporal protocol:"
    )

    print(
        "A: EARLY -> MIDDLE"
    )

    print(
        "B: EARLY+MIDDLE -> LATE"
    )

    print(
        "Epoch selection uses EARLY only."
    )

    print(
        "INTERNAL_TEST values are not used."
    )

    print(
        "Future-B: CLOSED"
    )

    (
        df,
        x_dir,
        x_size,
        y,
        candidate_labels,
    ) = load_dev_vectors()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

    early_idx = np.flatnonzero(
        df[
            "temporal_split"
        ].to_numpy()
        ==
        "DEV_EARLY"
    )

    early_dates = np.array(
        sorted(
            pd.unique(
                df.loc[
                    early_idx,
                    "_date",
                ]
            )
        )
    )

    n_val_dates = max(
        1,
        int(
            math.ceil(
                len(
                    early_dates
                )
                *
                EARLY_VAL_FRACTION
            )
        ),
    )

    calibration_val_dates = (
        early_dates[
            -n_val_dates:
        ]
    )

    calibration_train_dates = (
        early_dates[
            :-n_val_dates
        ]
    )

    cal_train_idx = np.flatnonzero(
        df[
            "_date"
        ].isin(
            calibration_train_dates
        ).to_numpy()
        &
        (
            df[
                "temporal_split"
            ].to_numpy()
            ==
            "DEV_EARLY"
        )
    )

    cal_val_idx = np.flatnonzero(
        df[
            "_date"
        ].isin(
            calibration_val_dates
        ).to_numpy()
        &
        (
            df[
                "temporal_split"
            ].to_numpy()
            ==
            "DEV_EARLY"
        )
    )

    print()
    print(
        "Epoch calibration:"
    )

    print(
        "train dates:",
        [
            str(
                pd.Timestamp(x).date()
            )
            for x in
            calibration_train_dates
        ],
    )

    print(
        "validation dates:",
        [
            str(
                pd.Timestamp(x).date()
            )
            for x in
            calibration_val_dates
        ],
    )

    (
        calibration_model,
        _,
        _,
        selected_epoch,
        curve,
    ) = train(
        x_dir,
        x_size,
        y,
        cal_train_idx,
        device,
        MAX_EPOCHS,
        val_idx=
            cal_val_idx,
    )

    del calibration_model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print()
    print(
        "SELECTED EPOCH:",
        selected_epoch,
    )

    curve_path = (
        OUT
        / "08B_epoch_selection.csv"
    )

    curve.to_csv(
        curve_path,
        index=False,
    )

    fold_rows = []

    checkpoint_hashes = {}

    for spec in FOLDS:

        fold = spec[
            "fold"
        ]

        print()
        print("=" * 78)
        print(
            fold
        )
        print("=" * 78)

        train_idx = np.flatnonzero(
            df[
                "temporal_split"
            ].isin(
                spec[
                    "train_splits"
                ]
            ).to_numpy()
        )

        test_idx = np.flatnonzero(
            (
                df[
                    "temporal_split"
                ]
                ==
                spec[
                    "test_split"
                ]
            ).to_numpy()
        )

        (
            model,
            min_size,
            max_size,
            _,
            _,
        ) = train(
            x_dir,
            x_size,
            y,
            train_idx,
            device,
            selected_epoch,
            val_idx=None,
        )

        test_loader = loader(
            x_dir,
            x_size,
            y,
            test_idx,
            False,
        )

        (
            m,
            probs,
            y_true,
        ) = evaluate(
            model,
            test_loader,
            min_size,
            max_size,
            device,
        )

        checkpoint_path = (
            ART
            / f"08B_{fold}.pth"
        )

        torch.save(
            {
                "state_dict":
                    model.state_dict(),

                "candidate_labels":
                    candidate_labels.tolist(),

                "max_len":
                    MAX_LEN,

                "min_size":
                    min_size,

                "max_size":
                    max_size,

                "epochs":
                    selected_epoch,

                "seed":
                    SEED,
            },
            checkpoint_path,
        )

        checkpoint_hashes[
            fold
        ] = sha256(
            checkpoint_path
        )

        score_path = (
            OUT
            / f"08B_scores_{fold}.npz"
        )

        np.savez_compressed(
            score_path,

            pcap_uid=
                df.iloc[
                    test_idx
                ][
                    "pcap_uid"
                ]
                .astype(str)
                .to_numpy(),

            y_true=
                y_true,

            candidate_labels=
                candidate_labels,

            micro_probs=
                probs.astype(
                    np.float32
                ),
        )

        fold_rows.append(
            {
                "fold":
                    fold,

                "train_n":
                    len(
                        train_idx
                    ),

                "test_n":
                    len(
                        test_idx
                    ),

                "epochs":
                    selected_epoch,

                "seed":
                    SEED,

                "size_min_train":
                    min_size,

                "size_max_train":
                    max_size,

                **m,
            }
        )

        print(
            pd.DataFrame(
                [
                    fold_rows[
                        -1
                    ]
                ]
            ).to_string(
                index=False
            )
        )

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = pd.DataFrame(
        fold_rows
    )

    summary_path = (
        OUT
        / "08B_fold_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    manifest = {
        "stage":
            "08B_clean_temporal_micro_baseline",

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "git_commit":
            git_commit(),

        "source": {
            "path":
                str(
                    SOURCE
                ),

            "sha256":
                sha256(
                    SOURCE
                ),

            "join_key":
                "pcap_uid",

            "sites":
                65,

            "max_sequence_length":
                MAX_LEN,
        },

        "architecture": {
            "name":
                "legacy multimodal direction+size Transformer",

            "direction":
                "Embedding(3,256)",

            "size":
                "Linear(1,256)",

            "fusion":
                "Linear(512,256)",

            "local":
                "Conv1d kernel=7 stride=2",

            "transformer_layers":
                NUM_LAYERS,

            "heads":
                NHEAD,

            "d_model":
                D_MODEL,

            "pooling":
                "mean+max",

            "classes":
                65,
        },

        "training": {
            "seed":
                SEED,

            "selected_epoch":
                int(
                    selected_epoch
                ),

            "epoch_selection":
                (
                    "chronological validation using "
                    "last 20% of DEV_EARLY dates only; "
                    "primary Macro-F1, secondary Accuracy"
                ),

            "learning_rate":
                LEARNING_RATE,

            "weight_decay":
                WEIGHT_DECAY,

            "label_smoothing":
                LABEL_SMOOTHING,

            "size_scaling":
                (
                    "fold-specific min/max fitted "
                    "on training captures only"
                ),
        },

        "temporal_folds":
            FOLDS,

        "data_policy": {
            "development_only":
                True,

            "test_fold_used_for_epoch_selection":
                False,

            "internal_test_vector_values_used":
                False,

            "external_future_used":
                False,

            "random_capture_split":
                False,
        },

        "checkpoints":
            checkpoint_hashes,

        "next":
            (
                "08C Micro/Macro complementarity audit "
                "using exact pcap_uid-aligned predictions"
            ),
    }

    write_json(
        OUT
        / "08B_manifest.json",
        manifest,
    )

    print()
    print("=" * 78)
    print(
        "08B SUMMARY"
    )
    print("=" * 78)

    print(
        summary.to_string(
            index=False
        )
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
