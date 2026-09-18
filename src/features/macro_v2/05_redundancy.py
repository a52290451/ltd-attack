from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import (
    fcluster,
    linkage,
)
from scipy.spatial.distance import squareform

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

CANONICAL_THRESHOLD = 0.95

SENSITIVITY_THRESHOLDS = [
    0.90,
    0.95,
    0.98,
    0.99,
]

EPS = 1e-12


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
        or
        "pcap_uid"
        not in assignments.columns
    ):
        raise RuntimeError(
            "pcap_uid required."
        )

    if df["pcap_uid"].duplicated().any():
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

    # INTERNAL_TEST never enters
    # the redundancy dataframe.
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

    missing = [
        f
        for f in features
        if f not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing features: {missing}"
        )

    return (
        df,
        features,
        dataset_path,
        assignments_path,
        candidates_path,
    )


def safe_numeric_frame(
    df: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:

    x = (
        df[features]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    return x


def correlation_matrices(
    df: pd.DataFrame,
    features: list[str],
):
    print(
        "Computing sample-level "
        "correlations..."
    )

    x = safe_numeric_frame(
        df,
        features,
    )

    sample_nunique = (
        x.nunique(
            dropna=True
        )
    )

    sample_constant = (
        sample_nunique <= 1
    )

    sample_spearman = (
        x.corr(
            method="spearman",
            min_periods=100,
        )
        .reindex(
            index=features,
            columns=features,
        )
    )

    sample_pearson = (
        x.corr(
            method="pearson",
            min_periods=100,
        )
        .reindex(
            index=features,
            columns=features,
        )
    )

    print(
        "Computing 65×3 longitudinal "
        "site-period median profiles..."
    )

    work = pd.concat(
        [
            df[
                [
                    "site_label",
                    "temporal_split",
                ]
            ].reset_index(
                drop=True
            ),
            x.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    site_period = (
        work
        .groupby(
            [
                "site_label",
                "temporal_split",
            ],
            observed=True,
        )[features]
        .median()
    )

    expected_profiles = (
        df["site_label"].nunique()
        * len(DEV_SPLITS)
    )

    if len(site_period) != expected_profiles:
        raise RuntimeError(
            "Unexpected number of "
            "site-period profiles: "
            f"{len(site_period)} "
            f"!= {expected_profiles}"
        )

    longitudinal_nunique = (
        site_period.nunique(
            dropna=True
        )
    )

    longitudinal_constant = (
        longitudinal_nunique <= 1
    )

    longitudinal_spearman = (
        site_period
        .corr(
            method="spearman",
            min_periods=30,
        )
        .reindex(
            index=features,
            columns=features,
        )
    )

    stats = pd.DataFrame(
        {
            "feature": features,
            "sample_nunique": [
                int(
                    sample_nunique.get(
                        f,
                        0,
                    )
                )
                for f in features
            ],
            "sample_constant_dev": [
                bool(
                    sample_constant.get(
                        f,
                        True,
                    )
                )
                for f in features
            ],
            "longitudinal_nunique": [
                int(
                    longitudinal_nunique.get(
                        f,
                        0,
                    )
                )
                for f in features
            ],
            "longitudinal_constant": [
                bool(
                    longitudinal_constant.get(
                        f,
                        True,
                    )
                )
                for f in features
            ],
        }
    )

    return (
        sample_spearman,
        sample_pearson,
        longitudinal_spearman,
        site_period,
        stats,
    )


def build_pair_table(
    features,
    sample_spearman,
    sample_pearson,
    longitudinal_spearman,
):

    records = []

    n = len(features)

    for i in range(n):
        f1 = features[i]

        for j in range(
            i + 1,
            n,
        ):
            f2 = features[j]

            rho_sample = (
                sample_spearman.loc[
                    f1,
                    f2,
                ]
            )

            pearson_sample = (
                sample_pearson.loc[
                    f1,
                    f2,
                ]
            )

            rho_long = (
                longitudinal_spearman.loc[
                    f1,
                    f2,
                ]
            )

            valid_sample = (
                np.isfinite(
                    rho_sample
                )
            )

            valid_long = (
                np.isfinite(
                    rho_long
                )
            )

            if (
                valid_sample
                and valid_long
            ):
                abs_sample = abs(
                    float(rho_sample)
                )

                abs_long = abs(
                    float(rho_long)
                )

                same_sign = bool(
                    (
                        rho_sample
                        * rho_long
                    ) > 0
                    or (
                        abs(rho_sample)
                        <= EPS
                        and
                        abs(rho_long)
                        <= EPS
                    )
                )

                joint_similarity = (
                    min(
                        abs_sample,
                        abs_long,
                    )
                    if same_sign
                    else 0.0
                )
            else:
                abs_sample = np.nan
                abs_long = np.nan
                same_sign = False
                joint_similarity = 0.0

            strong = bool(
                joint_similarity
                >= CANONICAL_THRESHOLD
            )

            records.append(
                {
                    "feature_a": f1,
                    "feature_b": f2,

                    "sample_spearman":
                        rho_sample,

                    "sample_abs_spearman":
                        abs_sample,

                    "sample_pearson":
                        pearson_sample,

                    "sample_abs_pearson":
                        (
                            abs(
                                pearson_sample
                            )
                            if np.isfinite(
                                pearson_sample
                            )
                            else np.nan
                        ),

                    "longitudinal_spearman":
                        rho_long,

                    "longitudinal_abs_spearman":
                        abs_long,

                    "sign_consistent":
                        same_sign,

                    "joint_similarity":
                        joint_similarity,

                    "strong_redundancy":
                        strong,
                }
            )

    return pd.DataFrame(
        records
    )


def similarity_matrix_from_pairs(
    features,
    pair_table,
):

    n = len(features)

    similarity = np.eye(
        n,
        dtype=float,
    )

    index = {
        feature: idx
        for idx, feature
        in enumerate(features)
    }

    for row in (
        pair_table
        .itertuples(
            index=False
        )
    ):
        i = index[
            row.feature_a
        ]

        j = index[
            row.feature_b
        ]

        value = float(
            row.joint_similarity
        )

        similarity[i, j] = value
        similarity[j, i] = value

    return similarity


def complete_linkage_clusters(
    features,
    similarity,
    threshold,
):

    distance = (
        1.0
        - similarity
    )

    distance = np.clip(
        distance,
        0.0,
        1.0,
    )

    np.fill_diagonal(
        distance,
        0.0,
    )

    condensed = squareform(
        distance,
        checks=True,
    )

    z = linkage(
        condensed,
        method="complete",
    )

    labels = fcluster(
        z,
        t=1.0 - threshold,
        criterion="distance",
    )

    cluster_map = {}

    for feature, label in zip(
        features,
        labels,
    ):
        cluster_map.setdefault(
            int(label),
            [],
        ).append(feature)

    ordered = sorted(
        cluster_map.values(),
        key=lambda members: (
            -len(members),
            members[0],
        ),
    )

    canonical = {}

    for cluster_idx, members in enumerate(
        ordered,
        start=1,
    ):
        for feature in members:
            canonical[
                feature
            ] = cluster_idx

    return canonical


def cluster_table(
    features,
    canonical_clusters,
    pair_table,
    feature_stats,
):

    size_map = {}

    for feature in features:
        cid = canonical_clusters[
            feature
        ]

        size_map[cid] = (
            size_map.get(
                cid,
                0,
            )
            + 1
        )

    pair_lookup = {}

    for row in pair_table.itertuples(
        index=False
    ):
        key = tuple(
            sorted(
                [
                    row.feature_a,
                    row.feature_b,
                ]
            )
        )

        pair_lookup[key] = float(
            row.joint_similarity
        )

    rows = []

    stats_idx = (
        feature_stats
        .set_index(
            "feature"
        )
    )

    for feature in features:
        cid = canonical_clusters[
            feature
        ]

        members = [
            f
            for f in features
            if canonical_clusters[
                f
            ] == cid
        ]

        pairwise_sims = []

        if len(members) > 1:
            for other in members:
                if other == feature:
                    continue

                key = tuple(
                    sorted(
                        [
                            feature,
                            other,
                        ]
                    )
                )

                pairwise_sims.append(
                    pair_lookup[
                        key
                    ]
                )

        rows.append(
            {
                "feature":
                    feature,

                "cluster_id":
                    cid,

                "cluster_size":
                    size_map[cid],

                "cluster_members":
                    "|".join(
                        sorted(
                            members
                        )
                    ),

                "min_similarity_to_cluster":
                    (
                        min(
                            pairwise_sims
                        )
                        if pairwise_sims
                        else 1.0
                    ),

                "sample_nunique":
                    int(
                        stats_idx.loc[
                            feature,
                            "sample_nunique",
                        ]
                    ),

                "sample_constant_dev":
                    bool(
                        stats_idx.loc[
                            feature,
                            "sample_constant_dev",
                        ]
                    ),

                "longitudinal_nunique":
                    int(
                        stats_idx.loc[
                            feature,
                            "longitudinal_nunique",
                        ]
                    ),

                "longitudinal_constant":
                    bool(
                        stats_idx.loc[
                            feature,
                            "longitudinal_constant",
                        ]
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def sensitivity_analysis(
    features,
    similarity,
):

    rows = []

    for threshold in (
        SENSITIVITY_THRESHOLDS
    ):
        clusters = (
            complete_linkage_clusters(
                features,
                similarity,
                threshold,
            )
        )

        counts = {}

        for cid in clusters.values():
            counts[cid] = (
                counts.get(
                    cid,
                    0,
                )
                + 1
            )

        sizes = list(
            counts.values()
        )

        rows.append(
            {
                "threshold":
                    threshold,

                "n_clusters":
                    len(sizes),

                "redundant_clusters":
                    int(
                        sum(
                            s > 1
                            for s in sizes
                        )
                    ),

                "singletons":
                    int(
                        sum(
                            s == 1
                            for s in sizes
                        )
                    ),

                "features_in_redundant_clusters":
                    int(
                        sum(
                            s
                            for s in sizes
                            if s > 1
                        )
                    ),

                "largest_cluster":
                    int(
                        max(sizes)
                    ),

                "effective_feature_count_if_one_per_cluster":
                    len(sizes),
            }
        )

    return pd.DataFrame(
        rows
    )


def main():
    print("=" * 78)
    print(
        "MACRO-V2-HIST — "
        "05 REDUNDANCY"
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

    print(
        "Canonical redundancy threshold: "
        f"{CANONICAL_THRESHOLD}"
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

    (
        sample_spearman,
        sample_pearson,
        longitudinal_spearman,
        site_period,
        feature_stats,
    ) = correlation_matrices(
        df,
        features,
    )

    out = output_dir()

    sample_spearman.to_csv(
        out
        / "05_sample_spearman_matrix.csv"
    )

    sample_pearson.to_csv(
        out
        / "05_sample_pearson_matrix.csv"
    )

    longitudinal_spearman.to_csv(
        out
        / "05_longitudinal_spearman_matrix.csv"
    )

    feature_stats.to_csv(
        out
        / "05_redundancy_feature_stats.csv",
        index=False,
    )

    print()
    print(
        "Building pairwise redundancy table..."
    )

    pairs = build_pair_table(
        features,
        sample_spearman,
        sample_pearson,
        longitudinal_spearman,
    )

    pairs = pairs.sort_values(
        [
            "joint_similarity",
            "sample_abs_spearman",
            "longitudinal_abs_spearman",
        ],
        ascending=False,
    ).reset_index(
        drop=True
    )

    pairs_path = (
        out
        / "05_redundancy_pairs.csv"
    )

    pairs.to_csv(
        pairs_path,
        index=False,
    )

    strong_pairs = (
        pairs[
            pairs[
                "strong_redundancy"
            ]
        ]
        .copy()
    )

    strong_pairs_path = (
        out
        / "05_redundancy_strong_pairs.csv"
    )

    strong_pairs.to_csv(
        strong_pairs_path,
        index=False,
    )

    similarity = (
        similarity_matrix_from_pairs(
            features,
            pairs,
        )
    )

    canonical_clusters = (
        complete_linkage_clusters(
            features,
            similarity,
            CANONICAL_THRESHOLD,
        )
    )

    clusters = cluster_table(
        features,
        canonical_clusters,
        pairs,
        feature_stats,
    )

    clusters = clusters.sort_values(
        [
            "cluster_size",
            "cluster_id",
            "feature",
        ],
        ascending=[
            False,
            True,
            True,
        ],
    ).reset_index(
        drop=True
    )

    clusters_path = (
        out
        / "05_redundancy_clusters.csv"
    )

    clusters.to_csv(
        clusters_path,
        index=False,
    )

    sensitivity = (
        sensitivity_analysis(
            features,
            similarity,
        )
    )

    sensitivity_path = (
        out
        / "05_redundancy_sensitivity.csv"
    )

    sensitivity.to_csv(
        sensitivity_path,
        index=False,
    )

    canonical_row = (
        sensitivity[
            sensitivity[
                "threshold"
            ]
            == CANONICAL_THRESHOLD
        ]
        .iloc[0]
    )

    cluster_sizes = (
        clusters[
            [
                "cluster_id",
                "cluster_size",
            ]
        ]
        .drop_duplicates()
    )

    manifest = {
        "feature_set_id":
            "MACRO-V2-HIST",

        "stage":
            "05_redundancy",

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
            "count":
                len(features),
        },

        "data_policy": {
            "development_splits_used":
                DEV_SPLITS,

            "internal_test_values_used":
                False,

            "external_future_used":
                False,
        },

        "spaces": {
            "sample_level": {
                "rows":
                    len(df),

                "metrics": [
                    "Spearman",
                    "Pearson",
                ],
            },

            "longitudinal": {
                "representation":
                    "median per site and temporal split",

                "profiles":
                    len(site_period),

                "metric":
                    "Spearman",
            },
        },

        "canonical_rule": {
            "sample_abs_spearman_min":
                CANONICAL_THRESHOLD,

            "longitudinal_abs_spearman_min":
                CANONICAL_THRESHOLD,

            "sign_consistency_required":
                True,

            "joint_similarity":
                "min(abs(sample Spearman), "
                "abs(longitudinal Spearman))",

            "clustering":
                "complete linkage",

            "distance":
                "1 - joint_similarity",
        },

        "sensitivity_thresholds":
            SENSITIVITY_THRESHOLDS,

        "results": {
            "strong_pair_count":
                len(strong_pairs),

            "clusters":
                int(
                    canonical_row[
                        "n_clusters"
                    ]
                ),

            "redundant_clusters":
                int(
                    canonical_row[
                        "redundant_clusters"
                    ]
                ),

            "singletons":
                int(
                    canonical_row[
                        "singletons"
                    ]
                ),

            "features_in_redundant_clusters":
                int(
                    canonical_row[
                        "features_in_redundant_clusters"
                    ]
                ),

            "largest_cluster":
                int(
                    cluster_sizes[
                        "cluster_size"
                    ].max()
                ),

            "effective_feature_count_if_one_per_cluster":
                int(
                    canonical_row[
                        "effective_feature_count_if_one_per_cluster"
                    ]
                ),
        },

        "selection_policy": {
            "feature_removed_in_stage_05":
                False,

            "cluster_representative_selected":
                False,

            "representative_selection_deferred_to":
                "06_select_features.py",
        },
    }

    write_json(
        out
        / "05_manifest_redundancy.json",
        manifest,
    )

    print()
    print("=" * 78)
    print(
        "REDUNDANCY SUMMARY"
    )
    print("=" * 78)

    print(
        sensitivity.to_string(
            index=False
        )
    )

    print()
    print(
        f"Strong redundant pairs "
        f"@ {CANONICAL_THRESHOLD}: "
        f"{len(strong_pairs)}"
    )

    print(
        "Effective dimensions if "
        "one representative per cluster: "
        f"{int(canonical_row['n_clusters'])}"
    )

    print()
    print(
        "Largest clusters:"
    )

    largest = (
        clusters[
            [
                "cluster_id",
                "cluster_size",
                "cluster_members",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "cluster_size",
                "cluster_id",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .head(20)
    )

    print(
        largest.to_string(
            index=False
        )
    )

    print()
    print(
        "Top 30 strongest "
        "redundancy relationships:"
    )

    print(
        strong_pairs[
            [
                "feature_a",
                "feature_b",
                "sample_spearman",
                "sample_pearson",
                "longitudinal_spearman",
                "joint_similarity",
            ]
        ]
        .head(30)
        .to_string(
            index=False
        )
    )

    print()
    print(
        "NOTE: no feature has "
        "been removed."
    )

    print(
        "Representative selection "
        "is deferred to Stage 06."
    )

    print()
    print(
        f"Clusters: {clusters_path}"
    )

    print(
        f"Manifest: "
        f"{out / '05_manifest_redundancy.json'}"
    )

    print()
    print(
        "REDUNDANCY COMPLETE"
    )


if __name__ == "__main__":
    main()
