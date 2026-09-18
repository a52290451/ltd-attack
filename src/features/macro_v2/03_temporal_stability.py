from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

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


def weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:

    mask = (
        np.isfinite(values)
        & np.isfinite(weights)
        & (weights > 0)
    )

    values = values[mask]
    weights = weights[mask]

    if len(values) == 0:
        return np.nan

    order = np.argsort(values)

    values = values[order]
    weights = weights[order]

    cumulative = np.cumsum(weights)

    total = cumulative[-1]

    if total <= 0:
        return np.nan

    target = quantile * total

    idx = np.searchsorted(
        cumulative,
        target,
        side="left",
    )

    idx = min(idx, len(values) - 1)

    return float(values[idx])


def weighted_std(
    values: np.ndarray,
    weights: np.ndarray,
) -> float:

    mask = (
        np.isfinite(values)
        & np.isfinite(weights)
        & (weights > 0)
    )

    x = values[mask]
    w = weights[mask]

    if len(x) == 0:
        return np.nan

    w_sum = np.sum(w)

    if w_sum <= 0:
        return np.nan

    mean = np.sum(w * x) / w_sum

    variance = (
        np.sum(w * (x - mean) ** 2)
        / w_sum
    )

    return float(np.sqrt(max(variance, 0.0)))


def robust_scale(
    values: np.ndarray,
    weights: np.ndarray,
) -> float:

    q25 = weighted_quantile(
        values,
        weights,
        0.25,
    )

    q75 = weighted_quantile(
        values,
        weights,
        0.75,
    )

    iqr = q75 - q25

    if np.isfinite(iqr) and iqr > 1e-12:
        return float(iqr)

    median = weighted_quantile(
        values,
        weights,
        0.50,
    )

    abs_dev = np.abs(
        values - median
    )

    mad = weighted_quantile(
        abs_dev,
        weights,
        0.50,
    )

    robust_mad = 1.4826 * mad

    if (
        np.isfinite(robust_mad)
        and robust_mad > 1e-12
    ):
        return float(robust_mad)

    std = weighted_std(
        values,
        weights,
    )

    if np.isfinite(std) and std > 1e-12:
        return float(std)

    return 1.0


def balanced_feature_values(
    df: pd.DataFrame,
    feature: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    int,
    int,
]:

    values = pd.to_numeric(
        df[feature],
        errors="coerce",
    )

    valid = (
        values.notna()
        & np.isfinite(values)
    )

    tmp = pd.DataFrame(
        {
            "site_label": (
                df.loc[
                    valid,
                    "site_label",
                ]
                .astype(str)
                .to_numpy()
            ),
            "value": (
                values.loc[
                    valid
                ]
                .astype(float)
                .to_numpy()
            ),
        }
    )

    if tmp.empty:
        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
            0,
            int(len(df)),
        )

    counts = (
        tmp.groupby(
            "site_label"
        )["site_label"]
        .transform("count")
        .astype(float)
    )

    # Cada sitio aporta peso total = 1.
    tmp["weight"] = 1.0 / counts

    return (
        tmp["value"].to_numpy(
            dtype=float
        ),
        tmp["weight"].to_numpy(
            dtype=float
        ),
        int(
            tmp["site_label"].nunique()
        ),
        int(
            len(df) - len(tmp)
        ),
    )


def weighted_ks_distance(
    x: np.ndarray,
    wx: np.ndarray,
    y: np.ndarray,
    wy: np.ndarray,
) -> float:

    if len(x) == 0 or len(y) == 0:
        return np.nan

    order_x = np.argsort(x)
    order_y = np.argsort(y)

    x_sorted = x[order_x]
    y_sorted = y[order_y]

    wx_sorted = wx[order_x]
    wy_sorted = wy[order_y]

    wx_sorted = (
        wx_sorted
        / np.sum(wx_sorted)
    )

    wy_sorted = (
        wy_sorted
        / np.sum(wy_sorted)
    )

    cwx = np.cumsum(
        wx_sorted
    )

    cwy = np.cumsum(
        wy_sorted
    )

    grid = np.unique(
        np.concatenate(
            [
                x_sorted,
                y_sorted,
            ]
        )
    )

    idx_x = np.searchsorted(
        x_sorted,
        grid,
        side="right",
    )

    idx_y = np.searchsorted(
        y_sorted,
        grid,
        side="right",
    )

    cdf_x = np.where(
        idx_x > 0,
        cwx[
            np.maximum(
                idx_x - 1,
                0,
            )
        ],
        0.0,
    )

    cdf_y = np.where(
        idx_y > 0,
        cwy[
            np.maximum(
                idx_y - 1,
                0,
            )
        ],
        0.0,
    )

    return float(
        np.max(
            np.abs(
                cdf_x - cdf_y
            )
        )
    )


def pair_metrics(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    feature: str,
    split_a: str,
    split_b: str,
) -> dict:

    (
        x,
        wx,
        sites_a,
        missing_a,
    ) = balanced_feature_values(
        df_a,
        feature,
    )

    (
        y,
        wy,
        sites_b,
        missing_b,
    ) = balanced_feature_values(
        df_b,
        feature,
    )

    if (
        len(x) == 0
        or len(y) == 0
    ):
        return {
            "feature": feature,
            "split_a": split_a,
            "split_b": split_b,
            "sites_a": sites_a,
            "sites_b": sites_b,
            "missing_a": missing_a,
            "missing_b": missing_b,
            "ks_balanced": np.nan,
            "wasserstein_raw": np.nan,
            "robust_scale": np.nan,
            "wasserstein_normalized": np.nan,
            "median_a": np.nan,
            "median_b": np.nan,
            "median_shift_normalized": np.nan,
        }

    ks = weighted_ks_distance(
        x,
        wx,
        y,
        wy,
    )

    wasserstein_raw = float(
        wasserstein_distance(
            x,
            y,
            u_weights=wx,
            v_weights=wy,
        )
    )

    pooled_values = np.concatenate(
        [x, y]
    )

    pooled_weights = np.concatenate(
        [wx, wy]
    )

    scale = robust_scale(
        pooled_values,
        pooled_weights,
    )

    wasserstein_norm = (
        wasserstein_raw / scale
    )

    median_a = weighted_quantile(
        x,
        wx,
        0.50,
    )

    median_b = weighted_quantile(
        y,
        wy,
        0.50,
    )

    median_shift_norm = (
        abs(
            median_a - median_b
        )
        / scale
    )

    return {
        "feature": feature,
        "split_a": split_a,
        "split_b": split_b,
        "sites_a": sites_a,
        "sites_b": sites_b,
        "missing_a": missing_a,
        "missing_b": missing_b,
        "ks_balanced": ks,
        "wasserstein_raw": wasserstein_raw,
        "robust_scale": scale,
        "wasserstein_normalized": wasserstein_norm,
        "median_a": median_a,
        "median_b": median_b,
        "median_shift_normalized": median_shift_norm,
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
        dtype={"site_label": str},
    )

    features = [
        x.strip()
        for x in candidates_path
        .read_text()
        .splitlines()
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

    # INTERNAL_TEST rows are never loaded
    # into the working dataframe.
    df = df.merge(
        dev_assignments,
        on="pcap_uid",
        how="inner",
        validate="one_to_one",
    )

    if (
        set(
            df[
                "temporal_split"
            ].unique()
        )
        != set(DEV_SPLITS)
    ):
        raise RuntimeError(
            "Unexpected split composition."
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
        "03 TEMPORAL STABILITY"
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

    split_frames = {
        split: (
            df[
                df[
                    "temporal_split"
                ] == split
            ]
            .copy()
        )
        for split
        in DEV_SPLITS
    }

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

        for (
            split_a,
            split_b,
        ) in PAIRWISE_SPLITS:

            records.append(
                pair_metrics(
                    split_frames[
                        split_a
                    ],
                    split_frames[
                        split_b
                    ],
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
        / "03_temporal_stability_pairwise.csv"
    )

    pairwise.to_csv(
        pairwise_path,
        index=False,
    )

    rows = []

    for feature, group in pairwise.groupby(
        "feature"
    ):
        row = {
            "feature": feature,
            "valid_sites_min": int(
                min(
                    group["sites_a"].min(),
                    group["sites_b"].min(),
                )
            ),
            "missing_rows_max": int(
                max(
                    group["missing_a"].max(),
                    group["missing_b"].max(),
                )
            ),
            "ks_mean": float(
                group[
                    "ks_balanced"
                ].mean()
            ),
            "ks_worst": float(
                group[
                    "ks_balanced"
                ].max()
            ),
            "wasserstein_norm_mean": float(
                group[
                    "wasserstein_normalized"
                ].mean()
            ),
            "wasserstein_norm_worst": float(
                group[
                    "wasserstein_normalized"
                ].max()
            ),
            "median_shift_norm_mean": float(
                group[
                    "median_shift_normalized"
                ].mean()
            ),
            "median_shift_norm_worst": float(
                group[
                    "median_shift_normalized"
                ].max()
            ),
        }

        rows.append(row)

    summary = pd.DataFrame(
        rows
    )

    # Lower drift = higher stability percentile.
    for metric in [
        "ks_worst",
        "wasserstein_norm_worst",
        "median_shift_norm_worst",
    ]:
        summary[
            metric
            + "_stability_percentile"
        ] = (
            summary[
                metric
            ]
            .rank(
                pct=True,
                ascending=False,
                method="average",
            )
        )

    stability_cols = [
        "ks_worst_stability_percentile",
        "wasserstein_norm_worst_stability_percentile",
        "median_shift_norm_worst_stability_percentile",
    ]

    summary[
        "temporal_stability_score"
    ] = (
        summary[
            stability_cols
        ]
        .mean(axis=1)
    )

    summary = (
        summary
        .sort_values(
            [
                "temporal_stability_score",
                "ks_worst",
                "wasserstein_norm_worst",
            ],
            ascending=[
                False,
                True,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    summary.insert(
        0,
        "rank_temporal_stability",
        np.arange(
            1,
            len(summary) + 1,
        ),
    )

    summary_path = (
        out
        / "03_temporal_stability_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    manifest = {
        "feature_set_id":
            "MACRO-V2-HIST",
        "stage":
            "03_temporal_stability",
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
        "site_balancing": {
            "enabled": True,
            "method": (
                "each site contributes "
                "total weight 1 inside "
                "each temporal split"
            ),
        },
        "comparisons": [
            list(x)
            for x
            in PAIRWISE_SPLITS
        ],
        "metrics": {
            "ks_balanced": (
                "weighted empirical CDF "
                "KS distance"
            ),
            "wasserstein_normalized": (
                "weighted Wasserstein "
                "distance divided by "
                "pooled robust scale"
            ),
            "median_shift_normalized": (
                "absolute weighted median "
                "difference divided by "
                "pooled robust scale"
            ),
            "robust_scale": (
                "IQR; fallback MAD*1.4826; "
                "fallback weighted STD"
            ),
        },
        "ranking": {
            "metric_aggregation":
                "worst pairwise drift",
            "score": (
                "mean percentile stability "
                "of KS, normalized Wasserstein "
                "and normalized median shift"
            ),
            "fixed_feature_count":
                False,
        },
    }

    write_json(
        out
        / "03_manifest_temporal_stability.json",
        manifest,
    )

    print()
    print("=" * 78)
    print(
        "TOP 20 TEMPORALLY "
        "STABLE FEATURES"
    )
    print("=" * 78)

    print(
        summary[
            [
                "rank_temporal_stability",
                "feature",
                "temporal_stability_score",
                "ks_worst",
                "wasserstein_norm_worst",
                "median_shift_norm_worst",
                "valid_sites_min",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    print()
    print("=" * 78)
    print(
        "BOTTOM 10 TEMPORALLY "
        "STABLE FEATURES"
    )
    print("=" * 78)

    print(
        summary[
            [
                "rank_temporal_stability",
                "feature",
                "temporal_stability_score",
                "ks_worst",
                "wasserstein_norm_worst",
                "median_shift_norm_worst",
                "valid_sites_min",
            ]
        ]
        .tail(10)
        .to_string(index=False)
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
        "TEMPORAL STABILITY COMPLETE"
    )


if __name__ == "__main__":
    main()
