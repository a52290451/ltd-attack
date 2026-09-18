from __future__ import annotations

from datetime import datetime, timezone
import math

import pandas as pd

from _common import (
    extract_capture_dates,
    git_commit,
    load_historical_65,
    output_dir,
    sha256,
    write_json,
)


DEV_FRACTION = 0.80


def date_str(x):
    return str(pd.Timestamp(x).date())


def main():
    print("=" * 78)
    print("MACRO-V2-HIST — 01 TEMPORAL SPLIT")
    print("=" * 78)

    out = output_dir()

    df, dataset_path = load_historical_65()

    df["_split_date"] = extract_capture_dates(df)

    unique_dates = sorted(
        df["_split_date"]
        .dropna()
        .unique()
    )

    n_dates = len(unique_dates)

    if n_dates < 10:
        raise RuntimeError(
            f"Solo existen {n_dates} fechas. "
            "No es suficiente para el protocolo temporal."
        )

    # Outer split:
    # first 80% = development
    # last 20% = untouched historical internal test
    dev_n = math.floor(
        n_dates * DEV_FRACTION
    )

    dev_n = max(3, min(dev_n, n_dates - 1))

    dev_dates = unique_dates[:dev_n]
    internal_test_dates = unique_dates[dev_n:]

    # Three chronological development windows.
    base = len(dev_dates) // 3
    remainder = len(dev_dates) % 3

    sizes = [
        base + (1 if i < remainder else 0)
        for i in range(3)
    ]

    cursor = 0
    dev_early = dev_dates[
        cursor : cursor + sizes[0]
    ]
    cursor += sizes[0]

    dev_middle = dev_dates[
        cursor : cursor + sizes[1]
    ]
    cursor += sizes[1]

    dev_late = dev_dates[
        cursor : cursor + sizes[2]
    ]

    date_to_split = {}

    for d in dev_early:
        date_to_split[pd.Timestamp(d)] = "DEV_EARLY"

    for d in dev_middle:
        date_to_split[pd.Timestamp(d)] = "DEV_MIDDLE"

    for d in dev_late:
        date_to_split[pd.Timestamp(d)] = "DEV_LATE"

    for d in internal_test_dates:
        date_to_split[pd.Timestamp(d)] = "INTERNAL_TEST"

    df["temporal_split"] = (
        df["_split_date"]
        .map(date_to_split)
    )

    if df["temporal_split"].isna().any():
        raise RuntimeError(
            "Existen capturas sin asignación temporal."
        )

    metadata_cols = [
        c
        for c in [
            "pcap_uid",
            "pcap_name",
            "site_label",
        ]
        if c in df.columns
    ]

    assignments = df[
        metadata_cols
    ].copy()

    assignments["date_id"] = (
        df["_split_date"]
        .dt.strftime("%Y-%m-%d")
    )

    assignments["temporal_split"] = (
        df["temporal_split"]
    )

    assignments.to_csv(
        out / "01_temporal_split_assignments.csv",
        index=False,
    )

    summary = (
        df.groupby("temporal_split")
        .agg(
            rows=("site_label", "size"),
            sites=("site_label", "nunique"),
            dates=("_split_date", "nunique"),
            first_date=("_split_date", "min"),
            last_date=("_split_date", "max"),
        )
        .reset_index()
    )

    summary.to_csv(
        out / "01_temporal_split_summary.csv",
        index=False,
    )

    per_site = (
        df.groupby(
            ["site_label", "temporal_split"]
        )
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    per_site.to_csv(
        out / "01_per_site_split_counts.csv",
        index=False,
    )

    expected_splits = [
        "DEV_EARLY",
        "DEV_MIDDLE",
        "DEV_LATE",
        "INTERNAL_TEST",
    ]

    missing_coverage = []

    for split in expected_splits:
        if split not in per_site.columns:
            missing_coverage.append(
                {
                    "split": split,
                    "missing_sites": 65,
                }
            )
            continue

        n_missing = int(
            (per_site[split] == 0).sum()
        )

        if n_missing:
            missing_coverage.append(
                {
                    "split": split,
                    "missing_sites": n_missing,
                }
            )

    if missing_coverage:
        raise RuntimeError(
            "No todos los sitios tienen representación "
            f"en todos los splits: {missing_coverage}"
        )

    manifest = {
        "feature_set_id": "MACRO-V2-HIST",
        "stage": "01_temporal_split",
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "future_used": False,
        "split_strategy": (
            "global chronological dates; "
            "80% development / 20% untouched internal test; "
            "development divided into chronological thirds"
        ),
        "git_commit": git_commit(),
        "dataset": {
            "path": str(dataset_path),
            "sha256": sha256(dataset_path),
            "rows": len(df),
            "sites": int(df["site_label"].nunique()),
            "dates": n_dates,
        },
        "splits": {
            "DEV_EARLY": {
                "dates": len(dev_early),
                "first": date_str(dev_early[0]),
                "last": date_str(dev_early[-1]),
            },
            "DEV_MIDDLE": {
                "dates": len(dev_middle),
                "first": date_str(dev_middle[0]),
                "last": date_str(dev_middle[-1]),
            },
            "DEV_LATE": {
                "dates": len(dev_late),
                "first": date_str(dev_late[0]),
                "last": date_str(dev_late[-1]),
            },
            "INTERNAL_TEST": {
                "dates": len(internal_test_dates),
                "first": date_str(
                    internal_test_dates[0]
                ),
                "last": date_str(
                    internal_test_dates[-1]
                ),
            },
        },
    }

    write_json(
        out / "01_manifest_temporal_split.json",
        manifest,
    )

    print()
    print(summary.to_string(index=False))

    print()
    print("65-site coverage in all splits: PASS")
    print(f"Artifacts: {out}")
    print("TEMPORAL SPLIT FROZEN")


if __name__ == "__main__":
    main()
