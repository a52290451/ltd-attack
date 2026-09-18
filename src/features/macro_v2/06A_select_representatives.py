from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from _common import (
    git_commit,
    output_dir,
    sha256,
    write_json,
)


EPS = 1e-12

D_COL = "score_min"
T_COL = "temporal_stability_score"
P_COL = "site_persistence_score"

OBJECTIVES = [
    D_COL,
    T_COL,
    P_COL,
]


def validate_unique_features(
    df: pd.DataFrame,
    name: str,
) -> None:

    if "feature" not in df.columns:
        raise RuntimeError(
            f"{name}: missing 'feature' column."
        )

    if df["feature"].isna().any():
        raise RuntimeError(
            f"{name}: null feature names found."
        )

    if df["feature"].duplicated().any():
        duplicated = (
            df.loc[
                df["feature"].duplicated(
                    keep=False
                ),
                "feature",
            ]
            .astype(str)
            .tolist()
        )

        raise RuntimeError(
            f"{name}: duplicate features found: "
            f"{duplicated[:20]}"
        )


def validate_score(
    df: pd.DataFrame,
    column: str,
) -> None:

    if column not in df.columns:
        raise RuntimeError(
            f"Missing score column: {column}"
        )

    values = pd.to_numeric(
        df[column],
        errors="coerce",
    )

    if values.isna().any():
        raise RuntimeError(
            f"{column}: NaN/non-numeric values."
        )

    if not np.isfinite(
        values.to_numpy(dtype=float)
    ).all():
        raise RuntimeError(
            f"{column}: infinite values."
        )

    if (
        (values < -EPS).any()
        or
        (values > 1.0 + EPS).any()
    ):
        raise RuntimeError(
            f"{column}: values outside [0,1]."
        )


def load_inputs():
    out = output_dir()

    paths = {
        "discriminability":
            out
            / "02_discriminability_summary.csv",

        "temporal_stability":
            out
            / "03_temporal_stability_summary.csv",

        "site_persistence":
            out
            / "04_site_persistence_summary.csv",

        "redundancy_clusters":
            out
            / "05_redundancy_clusters.csv",
    }

    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"{name}: {path}"
            )

    d = pd.read_csv(
        paths["discriminability"]
    )

    t = pd.read_csv(
        paths["temporal_stability"]
    )

    p = pd.read_csv(
        paths["site_persistence"]
    )

    c = pd.read_csv(
        paths["redundancy_clusters"]
    )

    for name, df in [
        ("discriminability", d),
        ("temporal_stability", t),
        ("site_persistence", p),
        ("redundancy_clusters", c),
    ]:
        validate_unique_features(
            df,
            name,
        )

    sets = {
        "D": set(
            d["feature"].astype(str)
        ),
        "T": set(
            t["feature"].astype(str)
        ),
        "P": set(
            p["feature"].astype(str)
        ),
        "R": set(
            c["feature"].astype(str)
        ),
    }

    reference = sets["D"]

    for name, values in sets.items():
        if values != reference:
            raise RuntimeError(
                "Feature-set mismatch between "
                f"D/T/P/redundancy: {name}"
            )

    if len(reference) != 311:
        raise RuntimeError(
            "Expected exactly 311 features, "
            f"found {len(reference)}."
        )

    required_cluster_cols = [
        "feature",
        "cluster_id",
        "cluster_size",
        "cluster_members",
        "min_similarity_to_cluster",
    ]

    missing_cluster_cols = [
        x
        for x in required_cluster_cols
        if x not in c.columns
    ]

    if missing_cluster_cols:
        raise RuntimeError(
            "Missing redundancy columns: "
            f"{missing_cluster_cols}"
        )

    merged = (
        d[
            [
                "feature",
                "rank_discriminability",
                D_COL,
            ]
        ]
        .merge(
            t[
                [
                    "feature",
                    "rank_temporal_stability",
                    T_COL,
                ]
            ],
            on="feature",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            p[
                [
                    "feature",
                    "rank_site_persistence",
                    P_COL,
                ]
            ],
            on="feature",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            c[
                required_cluster_cols
            ],
            on="feature",
            how="inner",
            validate="one_to_one",
        )
    )

    if len(merged) != 311:
        raise RuntimeError(
            "Merged table does not contain "
            "exactly 311 features."
        )

    for col in OBJECTIVES:
        validate_score(
            merged,
            col,
        )

    if merged[
        "cluster_id"
    ].isna().any():
        raise RuntimeError(
            "Null cluster IDs."
        )

    return (
        merged,
        paths,
    )


def compute_multiobjective_scores(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    values = result[
        OBJECTIVES
    ].to_numpy(
        dtype=float
    )

    result["dtp_min"] = (
        np.min(
            values,
            axis=1,
        )
    )

    result["dtp_geometric_mean"] = (
        np.prod(
            np.clip(
                values,
                0.0,
                1.0,
            ),
            axis=1,
        )
        ** (1.0 / 3.0)
    )

    result["dtp_arithmetic_mean"] = (
        np.mean(
            values,
            axis=1,
        )
    )

    result["dtp_balance_gap"] = (
        np.max(
            values,
            axis=1,
        )
        -
        np.min(
            values,
            axis=1,
        )
    )

    return result


def select_cluster_representatives(
    df: pd.DataFrame,
) -> pd.DataFrame:

    ranked = (
        df
        .sort_values(
            [
                "cluster_id",
                "dtp_min",
                "dtp_geometric_mean",
                "dtp_arithmetic_mean",
                "dtp_balance_gap",
                "feature",
            ],
            ascending=[
                True,
                False,
                False,
                False,
                True,
                True,
            ],
        )
        .copy()
    )

    ranked[
        "rank_within_cluster"
    ] = (
        ranked
        .groupby(
            "cluster_id"
        )
        .cumcount()
        + 1
    )

    ranked[
        "is_cluster_representative"
    ] = (
        ranked[
            "rank_within_cluster"
        ]
        == 1
    )

    representatives = (
        ranked[
            ranked[
                "is_cluster_representative"
            ]
        ]
        .copy()
    )

    cluster_count = (
        ranked[
            "cluster_id"
        ]
        .nunique()
    )

    if (
        len(representatives)
        != cluster_count
    ):
        raise RuntimeError(
            "Representative count does not "
            "equal cluster count."
        )

    if representatives[
        "cluster_id"
    ].duplicated().any():
        raise RuntimeError(
            "More than one representative "
            "selected for a cluster."
        )

    return (
        ranked,
        representatives,
    )


def dominates(
    a: np.ndarray,
    b: np.ndarray,
) -> bool:

    # Maximization problem:
    # a dominates b iff a is no worse
    # in every objective and strictly
    # better in at least one.
    no_worse = np.all(
        a >= (
            b - EPS
        )
    )

    strictly_better = np.any(
        a > (
            b + EPS
        )
    )

    return bool(
        no_worse
        and strictly_better
    )


def assign_pareto_fronts(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    matrix = result[
        OBJECTIVES
    ].to_numpy(
        dtype=float
    )

    remaining = list(
        range(
            len(result)
        )
    )

    fronts = np.zeros(
        len(result),
        dtype=int,
    )

    current_front = 1

    while remaining:
        nondominated = []

        for i in remaining:
            dominated = False

            for j in remaining:
                if i == j:
                    continue

                if dominates(
                    matrix[j],
                    matrix[i],
                ):
                    dominated = True
                    break

            if not dominated:
                nondominated.append(
                    i
                )

        if not nondominated:
            raise RuntimeError(
                "Pareto sorting failed: "
                "empty front."
            )

        for idx in nondominated:
            fronts[idx] = (
                current_front
            )

        nondominated_set = set(
            nondominated
        )

        remaining = [
            idx
            for idx in remaining
            if idx not in nondominated_set
        ]

        current_front += 1

    result[
        "pareto_front"
    ] = fronts

    result = (
        result
        .sort_values(
            [
                "pareto_front",
                "dtp_min",
                "dtp_geometric_mean",
                "dtp_arithmetic_mean",
                "dtp_balance_gap",
                "feature",
            ],
            ascending=[
                True,
                False,
                False,
                False,
                True,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    result.insert(
        0,
        "rank_representatives",
        np.arange(
            1,
            len(result) + 1,
        ),
    )

    return result


def build_front_summary(
    representatives:
        pd.DataFrame,
) -> pd.DataFrame:

    rows = []
    cumulative = 0

    for front, group in (
        representatives
        .groupby(
            "pareto_front",
            sort=True,
        )
    ):
        n = len(group)

        cumulative += n

        rows.append(
            {
                "pareto_front":
                    int(front),

                "features":
                    int(n),

                "cumulative_features":
                    int(
                        cumulative
                    ),

                "D_min":
                    float(
                        group[
                            D_COL
                        ].min()
                    ),

                "D_median":
                    float(
                        group[
                            D_COL
                        ].median()
                    ),

                "D_max":
                    float(
                        group[
                            D_COL
                        ].max()
                    ),

                "T_min":
                    float(
                        group[
                            T_COL
                        ].min()
                    ),

                "T_median":
                    float(
                        group[
                            T_COL
                        ].median()
                    ),

                "T_max":
                    float(
                        group[
                            T_COL
                        ].max()
                    ),

                "P_min":
                    float(
                        group[
                            P_COL
                        ].min()
                    ),

                "P_median":
                    float(
                        group[
                            P_COL
                        ].median()
                    ),

                "P_max":
                    float(
                        group[
                            P_COL
                        ].max()
                    ),

                "dtp_min_min":
                    float(
                        group[
                            "dtp_min"
                        ].min()
                    ),

                "dtp_min_median":
                    float(
                        group[
                            "dtp_min"
                        ].median()
                    ),

                "dtp_min_max":
                    float(
                        group[
                            "dtp_min"
                        ].max()
                    ),

                "geometric_mean_median":
                    float(
                        group[
                            "dtp_geometric_mean"
                        ].median()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def write_feature_list(
    path: Path,
    features: list[str],
) -> None:

    path.write_text(
        "\n".join(
            features
        )
        + "\n"
    )


def main():
    print("=" * 78)
    print(
        "MACRO-V2-HIST — "
        "06A CLUSTER REPRESENTATIVES "
        "+ PARETO"
    )
    print("=" * 78)

    print(
        "Inputs: DEV-derived artifacts "
        "from Stages 02-05"
    )

    print(
        "INTERNAL_TEST: PROHIBITED"
    )

    print(
        "External Future: PROHIBITED"
    )

    print(
        "No final feature-count "
        "selection in this stage."
    )

    (
        merged,
        input_paths,
    ) = load_inputs()

    print()
    print(
        f"Features loaded: "
        f"{len(merged)}"
    )

    print(
        f"Redundancy clusters: "
        f"{merged['cluster_id'].nunique()}"
    )

    scored = (
        compute_multiobjective_scores(
            merged
        )
    )

    (
        full_table,
        representatives,
    ) = (
        select_cluster_representatives(
            scored
        )
    )

    expected_clusters = (
        full_table[
            "cluster_id"
        ].nunique()
    )

    print(
        "Cluster representatives: "
        f"{len(representatives)}"
    )

    if expected_clusters != 148:
        print(
            "WARNING: expected 148 "
            "canonical clusters, found "
            f"{expected_clusters}."
        )

    representatives = (
        assign_pareto_fronts(
            representatives
        )
    )

    pareto_fronts = int(
        representatives[
            "pareto_front"
        ].max()
    )

    front1 = (
        representatives[
            representatives[
                "pareto_front"
            ]
            == 1
        ]
        .copy()
    )

    print(
        f"Pareto fronts: "
        f"{pareto_fronts}"
    )

    print(
        f"Pareto Front 1 features: "
        f"{len(front1)}"
    )

    representative_map = (
        representatives[
            [
                "feature",
                "cluster_id",
                "pareto_front",
                "rank_representatives",
            ]
        ]
        .rename(
            columns={
                "feature":
                    "representative_feature",

                "pareto_front":
                    "representative_pareto_front",

                "rank_representatives":
                    "representative_global_rank",
            }
        )
    )

    full_table = (
        full_table
        .merge(
            representative_map,
            on="cluster_id",
            how="left",
            validate="many_to_one",
        )
    )

    if full_table[
        "representative_feature"
    ].isna().any():
        raise RuntimeError(
            "Missing representative mapping."
        )

    front_summary = (
        build_front_summary(
            representatives
        )
    )

    out = output_dir()

    full_path = (
        out
        / "06A_feature_selection_table.csv"
    )

    reps_path = (
        out
        / "06A_cluster_representatives.csv"
    )

    fronts_path = (
        out
        / "06A_pareto_front_summary.csv"
    )

    reps_txt_path = (
        out
        / "06A_cluster_representatives.txt"
    )

    front1_txt_path = (
        out
        / "06A_pareto_front_01.txt"
    )

    full_table.to_csv(
        full_path,
        index=False,
    )

    representatives.to_csv(
        reps_path,
        index=False,
    )

    front_summary.to_csv(
        fronts_path,
        index=False,
    )

    write_feature_list(
        reps_txt_path,
        representatives[
            "feature"
        ].tolist(),
    )

    write_feature_list(
        front1_txt_path,
        front1[
            "feature"
        ].tolist(),
    )

    singleton_clusters = int(
        (
            representatives[
                "cluster_size"
            ]
            == 1
        ).sum()
    )

    redundant_cluster_reps = int(
        (
            representatives[
                "cluster_size"
            ]
            > 1
        ).sum()
    )

    manifest = {
        "feature_set_id":
            "MACRO-V2-HIST",

        "stage":
            "06A_cluster_representatives_pareto",

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "git_commit":
            git_commit(),

        "input_artifacts": {
            name: {
                "path": str(path),
                "sha256": sha256(path),
            }
            for name, path
            in input_paths.items()
        },

        "data_policy": {
            "source":
                "DEV-derived Stage 02-05 artifacts only",

            "raw_internal_test_loaded":
                False,

            "internal_test_metrics_used":
                False,

            "external_future_used":
                False,
        },

        "objectives": {
            "D":
                D_COL,

            "T":
                T_COL,

            "P":
                P_COL,

            "direction":
                "maximize all",
        },

        "cluster_representative_rule": {
            "primary":
                "maximize min(D,T,P)",

            "tie_break_1":
                "maximize geometric mean(D,T,P)",

            "tie_break_2":
                "maximize arithmetic mean(D,T,P)",

            "tie_break_3":
                "minimize max(D,T,P)-min(D,T,P)",

            "tie_break_4":
                "feature name ascending for deterministic reproducibility",

            "weights_used":
                False,
        },

        "pareto_rule": {
            "dominance":
                (
                    "A dominates B if A is no worse "
                    "in D,T,P and strictly better "
                    "in at least one objective"
                ),

            "objectives":
                [
                    "D",
                    "T",
                    "P",
                ],

            "weights_used":
                False,

            "fronts":
                pareto_fronts,
        },

        "results": {
            "input_features":
                len(full_table),

            "redundancy_clusters":
                expected_clusters,

            "cluster_representatives":
                len(representatives),

            "singleton_representatives":
                singleton_clusters,

            "redundant_cluster_representatives":
                redundant_cluster_reps,

            "redundant_nonrepresentatives":
                (
                    len(full_table)
                    - len(representatives)
                ),

            "pareto_fronts":
                pareto_fronts,

            "pareto_front_1_features":
                len(front1),
        },

        "selection_status": {
            "final_feature_set_frozen":
                False,

            "final_feature_count_selected":
                False,

            "internal_test_opened":
                False,

            "next_stage":
                "06B DEV-only temporal model validation",
        },
    }

    write_json(
        out
        / "06A_manifest_representatives_pareto.json",
        manifest,
    )

    print()
    print("=" * 78)
    print(
        "PARETO FRONT SUMMARY"
    )
    print("=" * 78)

    print(
        front_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 78)
    print(
        "TOP 30 REPRESENTATIVES"
    )
    print("=" * 78)

    print(
        representatives[
            [
                "rank_representatives",
                "feature",
                "cluster_id",
                "cluster_size",
                D_COL,
                T_COL,
                P_COL,
                "dtp_min",
                "dtp_geometric_mean",
                "dtp_balance_gap",
                "pareto_front",
            ]
        ]
        .head(30)
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 78)
    print(
        "PARETO FRONT 1"
    )
    print("=" * 78)

    print(
        front1[
            [
                "feature",
                "cluster_id",
                D_COL,
                T_COL,
                P_COL,
                "dtp_min",
                "dtp_geometric_mean",
            ]
        ]
        .sort_values(
            [
                "dtp_min",
                "dtp_geometric_mean",
            ],
            ascending=False,
        )
        .to_string(
            index=False
        )
    )

    print()
    print(
        "No final feature set "
        "has been selected."
    )

    print(
        "No INTERNAL_TEST or "
        "Future data was used."
    )

    print()
    print(
        f"Full table: {full_path}"
    )

    print(
        f"Representatives: {reps_path}"
    )

    print(
        f"Pareto summary: {fronts_path}"
    )

    print(
        "06A REPRESENTATIVE SELECTION "
        "COMPLETE"
    )


if __name__ == "__main__":
    main()
