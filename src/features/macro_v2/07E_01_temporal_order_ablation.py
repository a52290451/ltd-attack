from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import torch

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
        raise RuntimeError(path)

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    return mod


def fuse(xgb, ltd):
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


def softmax(logits):
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


def predictions(p, y):
    order = np.argsort(
        -p,
        axis=1,
    )

    pred = order[:, 0]

    ranks = (
        np.argmax(
            order
            == y[:, None],
            axis=1,
        )
        + 1
    )

    return pred, ranks


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

    yy = y[idx]
    pp = pred[idx]
    rr = ranks[idx]

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
                    labels=np.arange(65),
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
    }


def paired_delta(
    y,
    nopos_pred,
    nopos_rank,
    ordered_pred,
    ordered_rank,
    idx=None,
):
    a = metrics(
        y,
        nopos_pred,
        nopos_rank,
        idx,
    )

    b = metrics(
        y,
        ordered_pred,
        ordered_rank,
        idx,
    )

    return {
        k: b[k] - a[k]
        for k in a
    }


def bootstrap_dates(
    y,
    dates,
    nopos_pred,
    nopos_rank,
    ordered_pred,
    ordered_rank,
):
    unique_dates = np.array(
        sorted(
            pd.unique(
                dates
            )
        )
    )

    date_idx = {
        d: np.flatnonzero(
            dates == d
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
            size=len(unique_dates),
            replace=True,
        )

        idx = np.concatenate(
            [
                date_idx[d]
                for d in sampled
            ]
        )

        rows.append(
            paired_delta(
                y,
                nopos_pred,
                nopos_rank,
                ordered_pred,
                ordered_rank,
                idx,
            )
        )

    return pd.DataFrame(
        rows
    )


def make_nopos_classes(mod):

    class CandidateTemporalEncoderNoPos(
        mod.CandidateTemporalEncoder
    ):
        """
        Capacity-matched temporal encoder without
        positional information.

        Self-attention remains permutation-equivariant;
        mean pooling makes the final candidate
        representation invariant to history order.
        """

        def forward(
            self,
            history,
            padding_mask,
        ):
            x = self.input_projection(
                history
            )

            # Deliberately NO positional encoding.
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

            return (
                (
                    x
                    * valid
                ).sum(
                    dim=1
                )
                / denom
            )

    class LTDPairScorerNoPos(
        mod.LTDPairScorer
    ):
        def __init__(self):
            super().__init__()

            self.temporal_encoder = (
                CandidateTemporalEncoderNoPos()
            )

    return (
        CandidateTemporalEncoderNoPos,
        LTDPairScorerNoPos,
    )


def train_nopos(
    mod07a,
    mod07d,
    scorer_class,
    fold_data,
    features,
    device,
):
    mod07a.CONTEXT_DAYS = (
        CONTEXT_DAYS
    )

    (
        train_context,
        skipped_train,
    ) = mod07a.build_context_cache(
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

    (
        test_context,
        skipped_test,
    ) = mod07a.build_context_cache(
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

    if skipped_test:
        raise RuntimeError(
            f"Missing test context: "
            f"{skipped_test}"
        )

    probs_by_seed = []
    seed_rows = []

    for seed in mod07a.SEEDS:

        mod07a.set_seed(
            seed
        )

        model = scorer_class().to(
            device
        )

        (
            loss,
            train_n,
            train_dates_n,
        ) = mod07a.train_ltd_model(
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
            train_context,
            device,
            seed,
        )

        logits = mod07d.predict_ltd(
            mod07a,
            model,
            fold_data[
                "x_test"
            ],
            fold_data[
                "test_dates"
            ],
            test_context,
            device,
        )

        probs = softmax(
            logits.astype(
                np.float64
            )
        )

        pred, rank = predictions(
            probs,
            fold_data[
                "y_test"
            ],
        )

        m = metrics(
            fold_data[
                "y_test"
            ],
            pred,
            rank,
        )

        seed_rows.append(
            {
                "seed":
                    seed,

                "train_loss":
                    loss,

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

    ensemble = (
        np.stack(
            probs_by_seed,
            axis=0,
        )
        .mean(
            axis=0
        )
    )

    ensemble /= ensemble.sum(
        axis=1,
        keepdims=True,
    )

    return (
        ensemble,
        seed_rows,
        len(skipped_train),
    )


def main():

    print("=" * 78)
    print(
        "07E-1 — TEMPORAL ORDER ABLATION"
    )
    print("=" * 78)

    print(
        "Ordered reference: MACRO-LTD-V2"
    )

    print(
        "Control: identical W=5 encoder "
        "without positional encoding."
    )

    print(
        "No feature/context/alpha selection."
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
        "macro_v2_07d_order_ablation",
    )

    mod07a, source07a = (
        mod07d.load_07a()
    )

    mod07a.CONTEXT_DAYS = (
        CONTEXT_DAYS
    )

    (
        _,
        scorer_nopos,
    ) = make_nopos_classes(
        mod07a
    )

    ordered_model = (
        mod07a.LTDPairScorer()
    )

    nopos_model = (
        scorer_nopos()
    )

    ordered_params = (
        mod07a.count_parameters(
            ordered_model
        )
    )

    nopos_params = (
        mod07a.count_parameters(
            nopos_model
        )
    )

    del ordered_model
    del nopos_model

    if (
        ordered_params
        != nopos_params
    ):
        raise RuntimeError(
            "Parameter-count mismatch: "
            f"{ordered_params} vs "
            f"{nopos_params}"
        )

    print(
        "Trainable parameters:",
        ordered_params,
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

    dates, _ = mod07a.derive_dates(
        captures
    )

    captures[
        "_query_date"
    ] = pd.to_datetime(
        dates
    )

    fold_specs = {
        f[
            "fold"
        ]: f
        for f in mod07a.FOLDS
    }

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    summary_rows = []
    comparison_rows = []
    bootstrap_rows = []
    seed_rows = []

    for fold in FOLDS:

        print()
        print("=" * 78)
        print(fold)
        print("=" * 78)

        fold_data = mod07d.prepare_fold(
            mod07a,
            captures,
            daily,
            features,
            fold_specs[
                fold
            ],
        )

        ordered_path = (
            out
            / f"07D_01R_scores_{fold}.npz"
        )

        if not ordered_path.exists():
            raise FileNotFoundError(
                ordered_path
            )

        saved = np.load(
            ordered_path,
            allow_pickle=True,
        )

        xgb = (
            saved[
                "xgb_probs"
            ].astype(
                np.float64
            )
        )

        y = (
            saved[
                "y_true"
            ].astype(int)
        )

        ordered_ltd = (
            saved[
                "v2_ltd_probs"
            ].astype(
                np.float64
            )
        )

        ordered_fusion = (
            saved[
                "v2_fusion_probs"
            ].astype(
                np.float64
            )
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

        saved_uid = (
            saved[
                "pcap_uid"
            ].astype(str)
        )

        if not np.array_equal(
            expected_uid,
            saved_uid,
        ):
            raise RuntimeError(
                f"{fold}: pcap alignment "
                "mismatch."
            )

        (
            nopos_ltd,
            fold_seed_rows,
            skipped_train,
        ) = train_nopos(
            mod07a,
            mod07d,
            scorer_nopos,
            fold_data,
            features,
            device,
        )

        for row in fold_seed_rows:
            seed_rows.append(
                {
                    "fold":
                        fold,
                    **row,
                }
            )

        nopos_fusion = fuse(
            xgb,
            nopos_ltd,
        )

        models = {
            "ORDERED_LTD":
                ordered_ltd,

            "NOPOS_LTD":
                nopos_ltd,

            "ORDERED_FUSION":
                ordered_fusion,

            "NOPOS_FUSION":
                nopos_fusion,
        }

        model_metrics = {}

        for name, probs in models.items():

            pred, rank = predictions(
                probs,
                y,
            )

            m = metrics(
                y,
                pred,
                rank,
            )

            model_metrics[
                name
            ] = (
                m,
                pred,
                rank,
            )

            summary_rows.append(
                {
                    "fold":
                        fold,

                    "model":
                        name,

                    "context_days":
                        CONTEXT_DAYS,

                    "skipped_train_dates":
                        (
                            skipped_train
                            if "NOPOS"
                            in name
                            else np.nan
                        ),

                    **m,
                }
            )

        (
            ordered_metrics,
            ordered_pred,
            ordered_rank,
        ) = model_metrics[
            "ORDERED_FUSION"
        ]

        (
            nopos_metrics,
            nopos_pred,
            nopos_rank,
        ) = model_metrics[
            "NOPOS_FUSION"
        ]

        observed = paired_delta(
            y,
            nopos_pred,
            nopos_rank,
            ordered_pred,
            ordered_rank,
        )

        dates_test = (
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
            dates_test,
            nopos_pred,
            nopos_rank,
            ordered_pred,
            ordered_rank,
        )

        ordered_only = (
            (ordered_pred == y)
            &
            (nopos_pred != y)
        )

        nopos_only = (
            (nopos_pred == y)
            &
            (ordered_pred != y)
        )

        comparison_rows.append(
            {
                "fold":
                    fold,

                **{
                    f"ordered_minus_nopos_{k}":
                        v
                    for k, v
                    in observed.items()
                },

                "ordered_only_correct_n":
                    int(
                        ordered_only.sum()
                    ),

                "nopos_only_correct_n":
                    int(
                        nopos_only.sum()
                    ),

                "net_ordered_correct_n":
                    int(
                        ordered_only.sum()
                        -
                        nopos_only.sum()
                    ),
            }
        )

        for metric in [
            "accuracy",
            "macro_f1",
            "top5_accuracy",
            "mrr",
        ]:

            values = boot[
                metric
            ].to_numpy()

            bootstrap_rows.append(
                {
                    "fold":
                        fold,

                    "metric":
                        metric,

                    "observed_ordered_minus_nopos":
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

                    "p_ordered_gt_nopos":
                        float(
                            np.mean(
                                values > 0
                            )
                        ),
                }
            )

        np.savez_compressed(
            out
            / f"07E_01_scores_{fold}.npz",

            y_true=
                y,

            xgb_probs=
                xgb.astype(
                    np.float32
                ),

            ordered_ltd_probs=
                ordered_ltd.astype(
                    np.float32
                ),

            nopos_ltd_probs=
                nopos_ltd.astype(
                    np.float32
                ),

            ordered_fusion_probs=
                ordered_fusion.astype(
                    np.float32
                ),

            nopos_fusion_probs=
                nopos_fusion.astype(
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

    seed_df = pd.DataFrame(
        seed_rows
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

    a_cmp = (
        comparison_df[
            comparison_df[
                "fold"
            ]
            == "A_EARLY_TO_MIDDLE"
        ]
        .iloc[0]
    )

    ordered_signal = bool(
        a_cmp[
            "ordered_minus_nopos_accuracy"
        ] >= 0
        and
        a_cmp[
            "ordered_minus_nopos_macro_f1"
        ] >= 0
        and
        b.loc[
            "accuracy",
            "observed_ordered_minus_nopos",
        ] > 0
        and
        b.loc[
            "macro_f1",
            "observed_ordered_minus_nopos",
        ] > 0
        and
        b.loc[
            "accuracy",
            "p_ordered_gt_nopos",
        ] >= 0.90
        and
        b.loc[
            "macro_f1",
            "p_ordered_gt_nopos",
        ] >= 0.90
    )

    nopos_signal = bool(
        a_cmp[
            "ordered_minus_nopos_accuracy"
        ] <= 0
        and
        a_cmp[
            "ordered_minus_nopos_macro_f1"
        ] <= 0
        and
        b.loc[
            "accuracy",
            "observed_ordered_minus_nopos",
        ] < 0
        and
        b.loc[
            "macro_f1",
            "observed_ordered_minus_nopos",
        ] < 0
        and
        b.loc[
            "accuracy",
            "p_ordered_gt_nopos",
        ] <= 0.10
        and
        b.loc[
            "macro_f1",
            "p_ordered_gt_nopos",
        ] <= 0.10
    )

    if ordered_signal:
        decision = (
            "ORDERED_TEMPORAL_SIGNAL_SUPPORTED"
        )
    elif nopos_signal:
        decision = (
            "ORDERLESS_CONTEXT_SUPPORTED"
        )
    else:
        decision = (
            "ORDER_EFFECT_INCONCLUSIVE"
        )

    summary_path = (
        out
        / "07E_01_order_ablation_summary.csv"
    )

    comparison_path = (
        out
        / "07E_01_ordered_vs_nopos.csv"
    )

    bootstrap_path = (
        out
        / "07E_01_day_block_bootstrap.csv"
    )

    seeds_path = (
        out
        / "07E_01_nopos_seed_metrics.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    comparison_df.to_csv(
        comparison_path,
        index=False,
    )

    bootstrap_df.to_csv(
        bootstrap_path,
        index=False,
    )

    seed_df.to_csv(
        seeds_path,
        index=False,
    )

    manifest = {
        "stage":
            "07E_01_temporal_order_ablation",

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "git_commit":
            git_commit(),

        "data_policy": {
            "development_only":
                True,

            "model_selection":
                False,

            "context_tuning":
                False,

            "alpha_tuning":
                False,

            "internal_test_values_used":
                False,

            "external_future_used":
                False,
        },

        "frozen_reference": {
            "model":
                "MACRO-LTD-V2",

            "context_days":
                CONTEXT_DAYS,

            "alpha":
                ALPHA,

            "seeds":
                mod07a.SEEDS,
        },

        "ablation": {
            "ordered":
                (
                    "current CandidateTemporalEncoder "
                    "with sinusoidal positional encoding"
                ),

            "control":
                (
                    "same architecture and trainable "
                    "parameter count without positional "
                    "encoding"
                ),

            "trainable_parameters":
                ordered_params,

            "hypothesis":
                (
                    "ordered temporal information "
                    "contributes beyond the unordered "
                    "set of historical daily profiles"
                ),
        },

        "decision":
            decision,

        "interpretation_policy": {
            "if_ordered":
                (
                    "retain MACRO-LTD-V2 and support "
                    "a temporal-order contribution claim"
                ),

            "if_nopos":
                (
                    "do not claim temporal-order benefit; "
                    "consider simpler orderless Macro "
                    "candidate"
                ),

            "if_inconclusive":
                (
                    "retain V2 milestone but describe "
                    "evidence as longitudinal-context "
                    "rather than order-specific"
                ),
        },

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

    manifest_path = (
        out
        / "07E_01_manifest_order_ablation.json"
    )

    write_json(
        manifest_path,
        manifest,
    )

    print()
    print("=" * 78)
    print("ORDER ABLATION")
    print("=" * 78)

    print(
        summary_df.to_string(
            index=False
        )
    )

    print()
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
        "DECISION:",
        decision,
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
