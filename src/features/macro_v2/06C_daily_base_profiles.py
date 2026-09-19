from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re

import numpy as np
import pandas as pd

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


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_frozen_features() -> tuple[list[str], Path]:
    path = (
        repo_root()
        / "configs/features/MACRO-V2-BASE-128.txt"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    features = [
        x.strip()
        for x in path.read_text().splitlines()
        if x.strip()
    ]

    if len(features) != 128:
        raise RuntimeError(
            f"Expected 128 frozen BASE features, "
            f"found {len(features)}."
        )

    if len(features) != len(set(features)):
        raise RuntimeError(
            "Duplicate features in frozen BASE list."
        )

    return features, path


def normalize_date_value(value):
    if pd.isna(value):
        return None

    text = str(value).strip()

    # YYYY-MM-DD / datetime-like
    parsed = pd.to_datetime(
        text,
        errors="coerce",
    )

    if not pd.isna(parsed):
        return parsed.strftime("%Y-%m-%d")

    # YYYYMMDD
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
            return parsed.strftime("%Y-%m-%d")

    return None


def derive_dates(
    df: pd.DataFrame,
) -> tuple[pd.Series, str]:

    candidates = [
        "_audit_date",
        "date",
        "date_id",
        "capture_date",
    ]

    for col in candidates:
        if col not in df.columns:
            continue

        dates = df[col].apply(
            normalize_date_value
        )

        valid = dates.notna().mean()

        if valid > 0.95:
            return dates, col

    if "pcap_name" in df.columns:
        dates = df[
            "pcap_name"
        ].apply(
            normalize_date_value
        )

        if dates.notna().mean() > 0.95:
            return dates, "pcap_name"

    raise RuntimeError(
        "Could not derive capture date "
        "without consulting external data."
    )


def main():
    print("=" * 78)
    print(
        "MACRO-V2-HIST — "
        "06C DAILY BASE PROFILES"
    )
    print("=" * 78)

    print(
        "Purpose: construct deterministic "
        "daily longitudinal tokens from "
        "frozen MACRO-V2-BASE-128."
    )

    print(
        "INTERNAL_TEST: PROHIBITED"
    )

    print(
        "External Future: PROHIBITED"
    )

    print(
        "No model training."
    )

    print(
        "No new feature selection."
    )

    (
        features,
        frozen_path,
    ) = load_frozen_features()

    out = output_dir()

    assignments_path = (
        out
        / "01_temporal_split_assignments.csv"
    )

    if not assignments_path.exists():
        raise FileNotFoundError(
            assignments_path
        )

    assignments = pd.read_csv(
        assignments_path
    )

    required_assignment_cols = [
        "pcap_uid",
        "temporal_split",
    ]

    missing = [
        x
        for x in required_assignment_cols
        if x not in assignments.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing assignment columns: {missing}"
        )

    if assignments[
        "pcap_uid"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate pcap_uid in assignments."
        )

    # Only DEV IDs are retained before
    # touching the analytical dataframe.
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

    (
        df,
        dataset_path,
    ) = load_historical_65()

    required_dataset_cols = [
        "pcap_uid",
        "site_label",
    ]

    missing = [
        x
        for x in required_dataset_cols
        if x not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing dataset columns: {missing}"
        )

    missing_features = [
        f
        for f in features
        if f not in df.columns
    ]

    if missing_features:
        raise RuntimeError(
            "Frozen BASE features missing "
            f"from Historical: {missing_features}"
        )

    if df[
        "pcap_uid"
    ].duplicated().any():
        raise RuntimeError(
            "Historical pcap_uid is not unique."
        )

    # Critical boundary:
    # INTERNAL_TEST rows never enter DEV.
    dev = df.merge(
        dev_assignments,
        on="pcap_uid",
        how="inner",
        validate="one_to_one",
    )

    if len(dev) != len(
        dev_assignments
    ):
        raise RuntimeError(
            "DEV merge row mismatch."
        )

    observed_splits = set(
        dev[
            "temporal_split"
        ].unique()
    )

    if observed_splits != set(
        DEV_SPLITS
    ):
        raise RuntimeError(
            "Unexpected temporal splits: "
            f"{observed_splits}"
        )

    if dev[
        "site_label"
    ].nunique() != 65:
        raise RuntimeError(
            "Expected 65 sites."
        )

    dates, date_source = (
        derive_dates(
            dev
        )
    )

    dev = dev.copy()

    dev["_daily_date"] = dates

    if dev[
        "_daily_date"
    ].isna().any():
        raise RuntimeError(
            "Some DEV captures have no date."
        )

    # Ensure each calendar date belongs
    # to one temporal split only.
    date_split_counts = (
        dev[
            [
                "_daily_date",
                "temporal_split",
            ]
        ]
        .drop_duplicates()
        .groupby(
            "_daily_date"
        )[
            "temporal_split"
        ]
        .nunique()
    )

    if (
        date_split_counts
        > 1
    ).any():
        raise RuntimeError(
            "A date crosses temporal split "
            "boundaries."
        )

    print()
    print(
        f"Frozen BASE features: "
        f"{len(features)}"
    )

    print(
        f"DEV captures: "
        f"{len(dev):,}"
    )

    print(
        f"Sites: "
        f"{dev['site_label'].nunique()}"
    )

    print(
        f"Date source: {date_source}"
    )

    print(
        "DEV date range: "
        f"{dev['_daily_date'].min()} "
        f"→ {dev['_daily_date'].max()}"
    )

    # Convert BASE data to numeric.
    X = (
        dev[
            features
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    for feature in features:
        dev[feature] = X[feature]

    # Number of captures used to form
    # each site-day token.
    counts = (
        dev
        .groupby(
            [
                "site_label",
                "_daily_date",
                "temporal_split",
            ],
            observed=True,
        )
        .size()
        .rename(
            "capture_count"
        )
        .reset_index()
    )

    # Daily vector:
    # robust median of capture-level
    # frozen BASE-128 features.
    daily_values = (
        dev
        .groupby(
            [
                "site_label",
                "_daily_date",
                "temporal_split",
            ],
            observed=True,
        )[
            features
        ]
        .median()
        .reset_index()
    )

    daily = (
        daily_values
        .merge(
            counts,
            on=[
                "site_label",
                "_daily_date",
                "temporal_split",
            ],
            how="left",
            validate="one_to_one",
        )
        .rename(
            columns={
                "_daily_date":
                    "date",
            }
        )
    )

    daily = (
        daily
        .sort_values(
            [
                "date",
                "site_label",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    daily.insert(
        0,
        "daily_uid",
        (
            daily[
                "site_label"
            ].astype(str)
            + "__"
            + daily[
                "date"
            ].astype(str)
        ),
    )

    if daily[
        "daily_uid"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate site-day profiles."
        )

    # Missingness is audited but NOT
    # imputed here. Imputation must later
    # be fitted independently inside each
    # temporal training fold.
    feature_matrix = daily[
        features
    ]

    missing_cells = int(
        feature_matrix
        .isna()
        .sum()
        .sum()
    )

    total_cells = int(
        feature_matrix.shape[0]
        * feature_matrix.shape[1]
    )

    missing_fraction = (
        missing_cells
        / total_cells
        if total_cells
        else 0.0
    )

    all_missing_rows = int(
        feature_matrix
        .isna()
        .all(
            axis=1
        )
        .sum()
    )

    if all_missing_rows > 0:
        raise RuntimeError(
            "At least one daily profile has "
            "all 128 BASE features missing."
        )

    profiles_path = (
        out
        / "06C_daily_base_profiles.csv"
    )

    daily.to_csv(
        profiles_path,
        index=False,
    )

    summary_rows = []

    for split in DEV_SPLITS:
        s = daily[
            daily[
                "temporal_split"
            ]
            == split
        ]

        capture_counts = (
            s[
                "capture_count"
            ]
        )

        summary_rows.append(
            {
                "temporal_split":
                    split,

                "site_day_profiles":
                    len(s),

                "sites":
                    s[
                        "site_label"
                    ].nunique(),

                "dates":
                    s[
                        "date"
                    ].nunique(),

                "date_min":
                    s[
                        "date"
                    ].min(),

                "date_max":
                    s[
                        "date"
                    ].max(),

                "captures_min":
                    int(
                        capture_counts.min()
                    ),

                "captures_median":
                    float(
                        capture_counts.median()
                    ),

                "captures_mean":
                    float(
                        capture_counts.mean()
                    ),

                "captures_max":
                    int(
                        capture_counts.max()
                    ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary_path = (
        out
        / "06C_daily_profile_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    site_counts = (
        daily
        .groupby(
            [
                "site_label",
                "temporal_split",
            ],
            observed=True,
        )
        .size()
        .rename(
            "daily_profiles"
        )
        .reset_index()
    )

    site_counts_path = (
        out
        / "06C_site_split_daily_counts.csv"
    )

    site_counts.to_csv(
        site_counts_path,
        index=False,
    )

    manifest = {
        "feature_set_id":
            "MACRO-V2-BASE-128",

        "stage":
            "06C_daily_base_profiles",

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "git_commit":
            git_commit(),

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
                    len(features),
            },
        },

        "data_policy": {
            "development_splits_used":
                DEV_SPLITS,

            "internal_test_values_used":
                False,

            "external_future_used":
                False,
        },

        "representation": {
            "unit":
                "site-day",

            "input_unit":
                "capture",

            "input_dimensions":
                128,

            "aggregation":
                "per-feature median",

            "learned_parameters":
                False,

            "new_feature_selection":
                False,

            "imputation":
                "none",

            "date_source":
                date_source,
        },

        "results": {
            "dev_captures":
                len(dev),

            "sites":
                int(
                    daily[
                        "site_label"
                    ].nunique()
                ),

            "dates":
                int(
                    daily[
                        "date"
                    ].nunique()
                ),

            "site_day_profiles":
                len(daily),

            "missing_cells":
                missing_cells,

            "missing_fraction":
                missing_fraction,

            "all_missing_daily_profiles":
                all_missing_rows,
        },

        "selection_status": {
            "base_128_frozen":
                True,

            "meta_selected":
                False,

            "temporal_encoder_selected":
                False,

            "internal_test_opened":
                False,

            "external_future_opened":
                False,

            "next_stage":
                (
                    "06D longitudinal prototype "
                    "baseline using DEV-only "
                    "daily BASE-128 sequences"
                ),
        },
    }

    manifest_path = (
        out
        / "06C_manifest_daily_base_profiles.json"
    )

    write_json(
        manifest_path,
        manifest,
    )

    print()
    print("=" * 78)
    print(
        "DAILY PROFILE SUMMARY"
    )
    print("=" * 78)

    print(
        summary.to_string(
            index=False
        )
    )

    print()

    print(
        "Total site-day profiles:",
        len(daily),
    )

    print(
        "Missing BASE cells:",
        missing_cells,
    )

    print(
        "Missing fraction:",
        f"{missing_fraction:.8f}",
    )

    print()
    print(
        f"Profiles: {profiles_path}"
    )

    print(
        f"Summary: {summary_path}"
    )

    print(
        f"Manifest: {manifest_path}"
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
        "06C DAILY BASE PROFILES COMPLETE"
    )


if __name__ == "__main__":
    main()
