from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal
from sklearn.feature_selection import mutual_info_classif

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

RANDOM_STATE = 42


def eta_squared(values: np.ndarray, labels: np.ndarray) -> float:
    mask = np.isfinite(values)

    x = values[mask]
    y = labels[mask]

    if len(x) < 2:
        return 0.0

    grand_mean = np.mean(x)

    ss_total = np.sum(
        (x - grand_mean) ** 2
    )

    if ss_total <= 0:
        return 0.0

    ss_between = 0.0

    for label in np.unique(y):
        group = x[y == label]

        if len(group) == 0:
            continue

        ss_between += (
            len(group)
            * (np.mean(group) - grand_mean) ** 2
        )

    return float(
        max(
            0.0,
            min(1.0, ss_between / ss_total),
        )
    )


def kruskal_effect(
    values: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float, float]:

    mask = np.isfinite(values)

    x = values[mask]
    y = labels[mask]

    groups = [
        x[y == label]
        for label in np.unique(y)
    ]

    groups = [
        g for g in groups
        if len(g) > 0
    ]

    k = len(groups)
    n = sum(len(g) for g in groups)

    if k < 2 or n <= k:
        return 0.0, 1.0, 0.0

    if np.nanstd(x) == 0:
        return 0.0, 1.0, 0.0

    try:
        h, p = kruskal(*groups)
    except ValueError:
        return 0.0, 1.0, 0.0

    epsilon_squared = (
        (h - k + 1) / (n - k)
    )

    epsilon_squared = float(
        max(
            0.0,
            min(1.0, epsilon_squared),
        )
    )

    return (
        float(h),
        float(p),
        epsilon_squared,
    )


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
            f"Missing temporal split: {assignments_path}"
        )

    if not candidates_path.exists():
        raise FileNotFoundError(
            f"Missing candidate list: {candidates_path}"
        )

    assignments = pd.read_csv(
        assignments_path,
        dtype={"site_label": str},
    )

    candidates = [
        x.strip()
        for x in candidates_path.read_text().splitlines()
        if x.strip()
    ]

    df, dataset_path = load_historical_65()

    # Prefer pcap_uid because it is unique within Historical.
    if (
        "pcap_uid" not in df.columns
        or "pcap_uid" not in assignments.columns
    ):
        raise RuntimeError(
            "pcap_uid is required to reproduce "
            "the frozen temporal split."
        )

    if df["pcap_uid"].duplicated().any():
        raise RuntimeError(
            "pcap_uid is not unique in Historical."
        )

    if assignments["pcap_uid"].duplicated().any():
        raise RuntimeError(
            "pcap_uid is not unique in split assignments."
        )

    dev_assignments = assignments[
        assignments["temporal_split"].isin(
            DEV_SPLITS
        )
    ][
        ["pcap_uid", "temporal_split"]
    ].copy()

    # IMPORTANT:
    # INTERNAL_TEST rows are never merged into the
    # discriminability working dataframe.
    df = df.merge(
        dev_assignments,
        on="pcap_uid",
        how="inner",
        validate="one_to_one",
    )

    if set(df["temporal_split"].unique()) != set(DEV_SPLITS):
        raise RuntimeError(
            "Unexpected temporal split composition."
        )

    missing_features = [
        f for f in candidates
        if f not in df.columns
    ]

    if missing_features:
        raise RuntimeError(
            f"Missing candidate features: {missing_features}"
        )

    return (
        df,
        candidates,
        dataset_path,
        assignments_path,
        candidates_path,
    )


def compute_split_metrics(
    df: pd.DataFrame,
    features: list[str],
    split_name: str,
) -> pd.DataFrame:

    print(
        f"\n===== {split_name} ====="
    )

    if split_name == "DEV_ALL":
        work = df.copy()
    else:
        work = df[
            df["temporal_split"] == split_name
        ].copy()

    print(
        f"Rows={len(work):,} | "
        f"Sites={work['site_label'].nunique()}"
    )

    labels_text = work[
        "site_label"
    ].astype(str)

    categories = sorted(
        labels_text.unique()
    )

    label_map = {
        label: i
        for i, label in enumerate(categories)
    }

    y = labels_text.map(
        label_map
    ).to_numpy(dtype=int)

    X_raw = (
        work[features]
        .replace([np.inf, -np.inf], np.nan)
        .apply(pd.to_numeric, errors="coerce")
    )

    # Median imputation is fitted independently inside
    # the current DEV block.
    medians = X_raw.median()

    X_imp = (
        X_raw
        .fillna(medians)
        .fillna(0.0)
    )

    print("Computing Mutual Information...")

    mi = mutual_info_classif(
        X_imp.to_numpy(dtype=np.float64),
        y,
        discrete_features=False,
        random_state=RANDOM_STATE,
        n_neighbors=3,
    )

    class_counts = np.bincount(y)

    probs = (
        class_counts[class_counts > 0]
        / len(y)
    )

    label_entropy = float(
        -np.sum(probs * np.log(probs))
    )

    if label_entropy <= 0:
        raise RuntimeError(
            "Label entropy is zero."
        )

    mi_norm = np.clip(
        mi / label_entropy,
        0.0,
        1.0,
    )

    rows = []

    print(
        "Computing effect sizes "
        "(eta² + Kruskal epsilon²)..."
    )

    for idx, feature in enumerate(features):
        values = (
            pd.to_numeric(
                work[feature],
                errors="coerce",
            )
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .to_numpy(dtype=float)
        )

        eta2 = eta_squared(
            values,
            y,
        )

        h, p, kw_eps2 = kruskal_effect(
            values,
            y,
        )

        missing_rate = float(
            np.mean(~np.isfinite(values))
        )

        rows.append(
            {
                "feature": feature,
                "split": split_name,
                "rows": len(work),
                "sites": len(categories),
                "mutual_information": float(
                    mi[idx]
                ),
                "mutual_information_normalized": float(
                    mi_norm[idx]
                ),
                "eta_squared": eta2,
                "kruskal_h": h,
                "kruskal_p": p,
                "kruskal_epsilon_squared": kw_eps2,
                "missing_rate": missing_rate,
            }
        )

    result = pd.DataFrame(rows)

    # Percentile ranks avoid imposing arbitrary scales
    # between the three discrimination metrics.
    for metric in [
        "mutual_information_normalized",
        "eta_squared",
        "kruskal_epsilon_squared",
    ]:
        result[
            metric + "_percentile"
        ] = result[metric].rank(
            pct=True,
            method="average",
        )

    result["discriminability_score"] = (
        result[
            [
                "mutual_information_normalized_percentile",
                "eta_squared_percentile",
                "kruskal_epsilon_squared_percentile",
            ]
        ]
        .mean(axis=1)
    )

    return result


def main():
    print("=" * 78)
    print(
        "MACRO-V2-HIST — 02 DISCRIMINABILITY"
    )
    print("=" * 78)
    print(
        "Allowed data: DEV_EARLY / DEV_MIDDLE / DEV_LATE"
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
        f"DEV rows loaded: {len(df):,}"
    )
    print(
        f"Candidate features: {len(features)}"
    )
    print(
        f"Sites: {df['site_label'].nunique()}"
    )

    if len(features) != 311:
        print(
            "WARNING: expected 311 candidates, "
            f"found {len(features)}."
        )

    frames = []

    for split_name in DEV_SPLITS:
        frames.append(
            compute_split_metrics(
                df,
                features,
                split_name,
            )
        )

    # DEV_ALL is descriptive only.
    # It is not used to determine temporal robustness.
    frames.append(
        compute_split_metrics(
            df,
            features,
            "DEV_ALL",
        )
    )

    long_df = pd.concat(
        frames,
        ignore_index=True,
    )

    out = output_dir()

    long_path = (
        out
        / "02_discriminability_long.csv"
    )

    long_df.to_csv(
        long_path,
        index=False,
    )

    dev_periods = long_df[
        long_df["split"].isin(DEV_SPLITS)
    ]

    score_pivot = (
        dev_periods.pivot(
            index="feature",
            columns="split",
            values="discriminability_score",
        )
        .reset_index()
    )

    score_pivot.columns.name = None

    score_pivot = score_pivot.rename(
        columns={
            "DEV_EARLY": "score_early",
            "DEV_MIDDLE": "score_middle",
            "DEV_LATE": "score_late",
        }
    )

    score_cols = [
        "score_early",
        "score_middle",
        "score_late",
    ]

    score_pivot["score_mean"] = (
        score_pivot[score_cols]
        .mean(axis=1)
    )

    score_pivot["score_min"] = (
        score_pivot[score_cols]
        .min(axis=1)
    )

    score_pivot["score_std"] = (
        score_pivot[score_cols]
        .std(axis=1)
    )

    dev_all = (
        long_df[
            long_df["split"] == "DEV_ALL"
        ][
            [
                "feature",
                "mutual_information_normalized",
                "eta_squared",
                "kruskal_epsilon_squared",
                "discriminability_score",
            ]
        ]
        .rename(
            columns={
                "mutual_information_normalized":
                    "dev_all_mi_norm",
                "eta_squared":
                    "dev_all_eta_squared",
                "kruskal_epsilon_squared":
                    "dev_all_kw_epsilon_squared",
                "discriminability_score":
                    "dev_all_score",
            }
        )
    )

    summary = score_pivot.merge(
        dev_all,
        on="feature",
        how="left",
        validate="one_to_one",
    )

    # Primary ranking:
    # a feature must remain discriminative in its
    # weakest development period.
    summary = summary.sort_values(
        [
            "score_min",
            "score_mean",
            "dev_all_score",
        ],
        ascending=False,
    ).reset_index(drop=True)

    summary.insert(
        0,
        "rank_discriminability",
        np.arange(
            1,
            len(summary) + 1,
        ),
    )

    summary_path = (
        out
        / "02_discriminability_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    manifest = {
        "feature_set_id": "MACRO-V2-HIST",
        "stage": "02_discriminability",
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "git_commit": git_commit(),
        "historical_dataset": {
            "path": str(dataset_path),
            "sha256": sha256(dataset_path),
        },
        "split_assignments": {
            "path": str(assignments_path),
            "sha256": sha256(assignments_path),
        },
        "candidate_features": {
            "path": str(candidates_path),
            "sha256": sha256(candidates_path),
            "count": len(features),
        },
        "data_policy": {
            "development_splits_used": DEV_SPLITS,
            "internal_test_values_used": False,
            "external_future_used": False,
        },
        "metrics": {
            "mutual_information": {
                "normalized_by": "label_entropy",
                "n_neighbors": 3,
            },
            "eta_squared": (
                "between-site sum-of-squares / total sum-of-squares"
            ),
            "kruskal_epsilon_squared": (
                "(H-k+1)/(n-k)"
            ),
        },
        "ranking": {
            "per_period_score": (
                "mean percentile rank of normalized MI, "
                "eta_squared and Kruskal epsilon_squared"
            ),
            "primary": "minimum period score",
            "secondary": "mean period score",
            "fixed_feature_count": False,
        },
    }

    write_json(
        out
        / "02_manifest_discriminability.json",
        manifest,
    )

    print()
    print("=" * 78)
    print("TOP 20 DISCRIMINATIVE FEATURES")
    print("=" * 78)

    print(
        summary[
            [
                "rank_discriminability",
                "feature",
                "score_min",
                "score_mean",
                "score_std",
                "dev_all_mi_norm",
                "dev_all_eta_squared",
                "dev_all_kw_epsilon_squared",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    print()
    print(
        f"Full ranking: {summary_path}"
    )
    print(
        f"Manifest: "
        f"{out / '02_manifest_discriminability.json'}"
    )
    print()
    print("DISCRIMINABILITY COMPLETE")


if __name__ == "__main__":
    main()
