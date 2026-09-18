from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from _common import (
    git_commit,
    load_historical_65,
    output_dir,
    sha256,
    write_json,
)


DEV_SPLITS = [
    "DEV_EARLY",
    "DEV_MIDDLE",
    "DEV_LATE",
]

PAIRWISE_SPLITS = [
    ("DEV_EARLY", "DEV_MIDDLE"),
    ("DEV_MIDDLE", "DEV_LATE"),
    ("DEV_EARLY", "DEV_LATE"),
]

EPS = 1e-12


def robust_scale(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return 1.0

    q25, q75 = np.percentile(
        values,
        [25, 75],
    )

    iqr = q75 - q25

    if np.isfinite(iqr) and iqr > EPS:
        return float(iqr)

    median = np.median(values)

    mad = np.median(
        np.abs(values - median)
    )

    robust_mad = 1.4826 * mad

    if (
        np.isfinite(robust_mad)
        and robust_mad > EPS
    ):
        return float(robust_mad)

    std = np.std(values)

    if np.isfinite(std) and std > EPS:
        return float(std)

    return 1.0


def identity_direction(
    source: pd.Series,
    target: pd.Series,
) -> dict:

    common = (
        source.index
        .intersection(target.index)
    )

    source = source.loc[common].astype(float)
    target = target.loc[common].astype(float)

    valid = (
        np.isfinite(source.to_numpy())
        & np.isfinite(target.to_numpy())
    )

    source = source.iloc[
        np.where(valid)[0]
    ]

    target = target.iloc[
        np.where(valid)[0]
    ]

    sites = list(source.index)

    if len(sites) < 2:
        return {
            "valid_sites": len(sites),
            "mrr": np.nan,
            "top1": np.nan,
            "top5": np.nan,
            "drift_to_separation_median": np.nan,
            "drift_to_separation_p90": np.nan,
        }

    target_values = target.to_numpy(
        dtype=float
    )

    reciprocal_ranks = []
    top1 = []
    top5 = []
    drift_ratios = []

    for idx, site in enumerate(sites):
        source_value = float(
            source.loc[site]
        )

        distances = np.abs(
            target_values - source_value
        )

        same_distance = float(
            distances[idx]
        )

        ranks = rankdata(
            distances,
            method="average",
        )

        same_rank = float(
            ranks[idx]
        )

        reciprocal_ranks.append(
            1.0 / same_rank
        )

        top1.append(
            1.0
            if same_rank <= 1.0 + EPS
            else 0.0
        )

        top5.append(
            1.0
            if same_rank <= 5.0 + EPS
            else 0.0
        )

        other_distances = np.delete(
            distances,
            idx,
        )

        between_separation = float(
            np.median(
                other_distances
            )
        )

        # Si no existe separación entre sitios,
        # esta feature no puede retener identidad,
        # incluso si el mismo sitio tampoco cambia.
        if (
            not np.isfinite(
                between_separation
            )
            or between_separation <= EPS
        ):
            drift_ratio = np.inf
        else:
            drift_ratio = (
                same_distance
                / between_separation
            )

        drift_ratios.append(
            drift_ratio
        )

    ratios = np.asarray(
        drift_ratios,
        dtype=float,
    )

    finite_ratios = ratios[
        np.isfinite(ratios)
    ]

    if len(finite_ratios) == 0:
        ratio_median = np.inf
        ratio_p90 = np.inf
    else:
        ratio_median = float(
            np.median(
                finite_ratios
            )
        )

        ratio_p90 = float(
            np.percentile(
                finite_ratios,
                90,
            )
        )

        # Si algunos sitios carecen totalmente
        # de separación, el peor comportamiento
        # debe quedar penalizado.
        if len(finite_ratios) < len(ratios):
            ratio_p90 = np.inf

    return {
        "valid_sites": len(sites),
        "mrr": float(
            np.mean(
                reciprocal_ranks
            )
        ),
        "top1": float(
            np.mean(top1)
        ),
        "top5": float(
            np.mean(top5)
        ),
        "drift_to_separation_median":
            ratio_median,
        "drift_to_separation_p90":
            ratio_p90,
    }


def pair_metrics(
    site_medians: pd.DataFrame,
    feature: str,
    split_a: str,
    split_b: str,
) -> dict:

    frame = (
        site_medians[
            [split_a, split_b]
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )

    a = frame[split_a].astype(float)
    b = frame[split_b].astype(float)

    valid_sites = len(frame)

    if valid_sites < 2:
        return {
            "feature": feature,
            "split_a": split_a,
            "split_b": split_b,
            "valid_sites": valid_sites,
            "spearman_rho": np.nan,
            "site_shift_norm_median": np.nan,
            "site_shift_norm_p90": np.nan,
            "identity_mrr_symmetric": np.nan,
            "identity_top1_symmetric": np.nan,
            "identity_top5_symmetric": np.nan,
            "drift_to_separation_median_symmetric": np.nan,
            "drift_to_separation_p90_symmetric": np.nan,
        }

    rho = spearmanr(
        a.to_numpy(),
        b.to_numpy(),
    ).statistic

    if not np.isfinite(rho):
        rho = 0.0

    pooled = np.concatenate(
        [
            a.to_numpy(dtype=float),
            b.to_numpy(dtype=float),
        ]
    )

    scale = robust_scale(
        pooled
    )

    normalized_shift = (
        np.abs(
            a.to_numpy(dtype=float)
            - b.to_numpy(dtype=float)
        )
        / scale
    )

    shift_median = float(
        np.median(
            normalized_shift
        )
    )

    shift_p90 = float(
        np.percentile(
            normalized_shift,
            90,
        )
    )

    ab = identity_direction(
        a,
        b,
    )

    ba = identity_direction(
        b,
        a,
    )

    # Conservador:
    # para métricas donde mayor es mejor,
    # usamos la peor dirección.
    mrr_symmetric = min(
        ab["mrr"],
        ba["mrr"],
    )

    top1_symmetric = min(
        ab["top1"],
        ba["top1"],
    )

    top5_symmetric = min(
        ab["top5"],
        ba["top5"],
    )

    # Para drift donde menor es mejor,
    # usamos la peor dirección.
    ratio_median_symmetric = max(
        ab[
            "drift_to_separation_median"
        ],
        ba[
            "drift_to_separation_median"
        ],
    )

    ratio_p90_symmetric = max(
        ab[
            "drift_to_separation_p90"
        ],
        ba[
            "drift_to_separation_p90"
        ],
    )

    return {
        "feature": feature,
        "split_a": split_a,
        "split_b": split_b,
        "valid_sites": valid_sites,
        "spearman_rho": float(rho),
        "site_shift_norm_median":
            shift_median,
        "site_shift_norm_p90":
            shift_p90,
        "identity_mrr_ab":
            ab["mrr"],
        "identity_mrr_ba":
            ba["mrr"],
        "identity_mrr_symmetric":
            mrr_symmetric,
        "identity_top1_symmetric":
            top1_symmetric,
        "identity_top5_symmetric":
            top5_symmetric,
        "drift_to_separation_median_symmetric":
            ratio_median_symmetric,
        "drift_to_separation_p90_symmetric":
            ratio_p90_symmetric,
    }


def load_dev_data():
    out = output_dir()

    assignments_path = (
        out
        / "01_temporal_split_assignments.csv"
    )

    candidates_path = (
        out
        / "00_candidate_features_nonconstant.txt"
    )

    if not assignments_path.exists():
        raise FileNotFoundError(
            assignments_path
        )

    if not candidates_path.exists():
        raise FileNotFoundError(
            candidates_path
        )

    assignments = pd.read_csv(
        assignments_path,
        dtype={
            "site_label": str,
        },
    )

    features = [
        x.strip()
        for x in (
            candidates_path
            .read_text()
            .splitlines()
        )
        if x.strip()
    ]

    df, dataset_path = (
        load_historical_65()
    )

    if (
        "pcap_uid" not in df.columns
        or "pcap_uid"
        not in assignments.columns
    ):
        raise RuntimeError(
            "pcap_uid required."
        )

    if df[
        "pcap_uid"
    ].duplicated().any():
        raise RuntimeError(
            "Historical pcap_uid "
            "is not unique."
        )

    if assignments[
        "pcap_uid"
    ].duplicated().any():
        raise RuntimeError(
            "Split pcap_uid "
            "is not unique."
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

    # INTERNAL_TEST nunca entra
    # en el dataframe de trabajo.
    df = df.merge(
        dev_assignments,
        on="pcap_uid",
        how="inner",
        validate="one_to_one",
    )

    if set(
        df[
            "temporal_split"
        ].unique()
    ) != set(DEV_SPLITS):
        raise RuntimeError(
            "Unexpected temporal "
            "split composition."
        )

    missing_features = [
        f
        for f in features
        if f not in df.columns
    ]

    if missing_features:
        raise RuntimeError(
            f"Missing features: "
            f"{missing_features}"
        )

    return (
        df,
        features,
        dataset_path,
        assignments_path,
        candidates_path,
    )


def main():
    print("=" * 78)
    print(
        "MACRO-V2-HIST — "
        "04 SITE PERSISTENCE"
    )
    print("=" * 78)

    print(
        "Allowed data: "
        "DEV_EARLY / DEV_MIDDLE / DEV_LATE"
    )

    print(
        "INTERNAL_TEST: PROHIBITED"
    )

    print(
        "External Future: PROHIBITED"
    )

    (
        df,
        features,
        dataset_path,
        assignments_path,
        candidates_path,
    ) = load_dev_data()

    print()
    print(
        f"DEV rows loaded: "
        f"{len(df):,}"
    )

    print(
        f"Candidate features: "
        f"{len(features)}"
    )

    print(
        f"Sites: "
        f"{df['site_label'].nunique()}"
    )

    site_counts = (
        df.groupby(
            [
                "site_label",
                "temporal_split",
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
    )

    min_samples = int(
        site_counts[
            DEV_SPLITS
        ]
        .to_numpy()
        .min()
    )

    print(
        f"Minimum captures per "
        f"site/split: {min_samples}"
    )

    # Una sola agregación:
    # 65 sites × 3 periods × 311 features.
    print()
    print(
        "Computing site-period "
        "median fingerprints..."
    )

    working = (
        df[
            [
                "site_label",
                "temporal_split",
                *features,
            ]
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    site_period_medians = (
        working
        .groupby(
            [
                "site_label",
                "temporal_split",
            ],
            observed=True,
        )[features]
        .median()
    )

    records = []

    for idx, feature in enumerate(
        features,
        start=1,
    ):
        if (
            idx == 1
            or idx % 25 == 0
            or idx == len(features)
        ):
            print(
                f"Processing "
                f"{idx}/{len(features)}: "
                f"{feature}"
            )

        feature_matrix = (
            site_period_medians[
                feature
            ]
            .unstack(
                "temporal_split"
            )
        )

        for (
            split_a,
            split_b,
        ) in PAIRWISE_SPLITS:

            records.append(
                pair_metrics(
                    feature_matrix,
                    feature,
                    split_a,
                    split_b,
                )
            )

    pairwise = pd.DataFrame(
        records
    )

    out = output_dir()

    pairwise_path = (
        out
        / "04_site_persistence_pairwise.csv"
    )

    pairwise.to_csv(
        pairwise_path,
        index=False,
    )

    rows = []

    for feature, group in (
        pairwise.groupby(
            "feature"
        )
    ):
        rows.append(
            {
                "feature": feature,

                "valid_sites_min": int(
                    group[
                        "valid_sites"
                    ].min()
                ),

                "spearman_mean": float(
                    group[
                        "spearman_rho"
                    ].mean()
                ),

                "spearman_worst": float(
                    group[
                        "spearman_rho"
                    ].min()
                ),

                "site_shift_norm_median_mean":
                    float(
                        group[
                            "site_shift_norm_median"
                        ].mean()
                    ),

                "site_shift_norm_p90_worst":
                    float(
                        group[
                            "site_shift_norm_p90"
                        ].max()
                    ),

                "identity_mrr_mean":
                    float(
                        group[
                            "identity_mrr_symmetric"
                        ].mean()
                    ),

                "identity_mrr_worst":
                    float(
                        group[
                            "identity_mrr_symmetric"
                        ].min()
                    ),

                "identity_top1_worst":
                    float(
                        group[
                            "identity_top1_symmetric"
                        ].min()
                    ),

                "identity_top5_worst":
                    float(
                        group[
                            "identity_top5_symmetric"
                        ].min()
                    ),

                "drift_to_separation_median_mean":
                    float(
                        group[
                            "drift_to_separation_median_symmetric"
                        ].mean()
                    ),

                "drift_to_separation_p90_worst":
                    float(
                        group[
                            "drift_to_separation_p90_symmetric"
                        ].max()
                    ),
            }
        )

    summary = pd.DataFrame(
        rows
    )

    # Higher is better.
    summary[
        "spearman_worst_persistence_percentile"
    ] = (
        summary[
            "spearman_worst"
        ]
        .rank(
            pct=True,
            ascending=True,
            method="average",
        )
    )

    summary[
        "identity_mrr_worst_persistence_percentile"
    ] = (
        summary[
            "identity_mrr_worst"
        ]
        .rank(
            pct=True,
            ascending=True,
            method="average",
        )
    )

    # Lower is better.
    summary[
        "site_shift_p90_persistence_percentile"
    ] = (
        summary[
            "site_shift_norm_p90_worst"
        ]
        .rank(
            pct=True,
            ascending=False,
            method="average",
        )
    )

    summary[
        "drift_separation_p90_persistence_percentile"
    ] = (
        summary[
            "drift_to_separation_p90_worst"
        ]
        .rank(
            pct=True,
            ascending=False,
            method="average",
        )
    )

    score_cols = [
        "spearman_worst_persistence_percentile",
        "identity_mrr_worst_persistence_percentile",
        "site_shift_p90_persistence_percentile",
        "drift_separation_p90_persistence_percentile",
    ]

    summary[
        "site_persistence_score"
    ] = (
        summary[
            score_cols
        ]
        .mean(axis=1)
    )

    summary = (
        summary
        .sort_values(
            [
                "site_persistence_score",
                "identity_mrr_worst",
                "spearman_worst",
                "drift_to_separation_p90_worst",
            ],
            ascending=[
                False,
                False,
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    summary.insert(
        0,
        "rank_site_persistence",
        np.arange(
            1,
            len(summary) + 1,
        ),
    )

    summary_path = (
        out
        / "04_site_persistence_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    manifest = {
        "feature_set_id":
            "MACRO-V2-HIST",

        "stage":
            "04_site_persistence",

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "git_commit":
            git_commit(),

        "historical_dataset": {
            "path": str(
                dataset_path
            ),
            "sha256": sha256(
                dataset_path
            ),
        },

        "split_assignments": {
            "path": str(
                assignments_path
            ),
            "sha256": sha256(
                assignments_path
            ),
        },

        "candidate_features": {
            "path": str(
                candidates_path
            ),
            "sha256": sha256(
                candidates_path
            ),
            "count": len(features),
        },

        "data_policy": {
            "development_splits_used":
                DEV_SPLITS,

            "internal_test_values_used":
                False,

            "external_future_used":
                False,
        },

        "site_representation": {
            "aggregation":
                "median per site and temporal split",

            "expected_sites":
                65,

            "minimum_samples_per_site_split":
                min_samples,
        },

        "comparisons": [
            list(x)
            for x in PAIRWISE_SPLITS
        ],

        "metrics": {
            "spearman_rho": (
                "rank consistency of "
                "site median fingerprints"
            ),

            "site_shift_normalized": (
                "absolute same-site median "
                "shift divided by robust "
                "pooled site-median scale"
            ),

            "identity_mrr": (
                "mean reciprocal rank of "
                "the same site when retrieving "
                "target-period site medians"
            ),

            "identity_top1_top5": (
                "same-site retrieval accuracy "
                "using single-feature median "
                "fingerprints"
            ),

            "drift_to_separation": (
                "same-site temporal distance "
                "divided by median distance "
                "to other sites"
            ),
        },

        "ranking": {
            "temporal_aggregation":
                "worst pairwise behavior",

            "score": (
                "mean percentile of "
                "worst Spearman, worst MRR, "
                "inverse P90 normalized "
                "same-site shift, and inverse "
                "P90 drift/separation"
            ),

            "fixed_feature_count":
                False,
        },
    }

    write_json(
        out
        / "04_manifest_site_persistence.json",
        manifest,
    )

    print()
    print("=" * 78)
    print(
        "TOP 20 SITE-PERSISTENT FEATURES"
    )
    print("=" * 78)

    print(
        summary[
            [
                "rank_site_persistence",
                "feature",
                "site_persistence_score",
                "spearman_worst",
                "identity_mrr_worst",
                "identity_top5_worst",
                "site_shift_norm_p90_worst",
                "drift_to_separation_p90_worst",
                "valid_sites_min",
            ]
        ]
        .head(20)
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 78)
    print(
        "BOTTOM 10 SITE-PERSISTENT FEATURES"
    )
    print("=" * 78)

    print(
        summary[
            [
                "rank_site_persistence",
                "feature",
                "site_persistence_score",
                "spearman_worst",
                "identity_mrr_worst",
                "identity_top5_worst",
                "site_shift_norm_p90_worst",
                "drift_to_separation_p90_worst",
                "valid_sites_min",
            ]
        ]
        .tail(10)
        .to_string(
            index=False
        )
    )

    print()
    print(
        f"Pairwise metrics: "
        f"{pairwise_path}"
    )

    print(
        f"Summary: "
        f"{summary_path}"
    )

    print(
        f"Manifest: "
        f"{out / '04_manifest_site_persistence.json'}"
    )

    print()
    print(
        "SITE PERSISTENCE COMPLETE"
    )


if __name__ == "__main__":
    main()
