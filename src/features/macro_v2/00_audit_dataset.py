from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from _common import (
    extract_capture_dates,
    git_commit,
    label_encoder_path,
    load_historical_65,
    numeric_feature_columns,
    output_dir,
    sha256,
    write_json,
)


def main():
    print("=" * 78)
    print("MACRO-V2-HIST — 00 DATASET AUDIT")
    print("=" * 78)

    out = output_dir()

    df, dataset_path = load_historical_65()

    print(f"Dataset: {dataset_path}")
    print(f"Rows after 65-site parity: {len(df):,}")
    print(f"Sites: {df['site_label'].nunique()}")

    capture_dates = extract_capture_dates(df)

    df["_audit_date"] = capture_dates

    feature_cols = numeric_feature_columns(df)

    print(f"Numeric candidate features: {len(feature_cols)}")

    records = []

    for feature in feature_cols:
        raw = pd.to_numeric(
            df[feature],
            errors="coerce",
        )

        inf_count = int(
            np.isinf(raw.to_numpy(dtype=float, na_value=np.nan)).sum()
        )

        clean = raw.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        non_null = int(clean.notna().sum())
        missing = int(clean.isna().sum())

        variance = (
            float(clean.var(ddof=0))
            if non_null > 0
            else np.nan
        )

        unique_count = int(clean.nunique(dropna=True))

        records.append(
            {
                "feature": feature,
                "dtype": str(df[feature].dtype),
                "rows": len(df),
                "non_null": non_null,
                "missing": missing,
                "missing_rate": missing / len(df),
                "inf_count": inf_count,
                "unique_count": unique_count,
                "variance": variance,
                "zero_variance": bool(
                    non_null == 0
                    or not np.isfinite(variance)
                    or variance == 0
                ),
            }
        )

    audit = pd.DataFrame(records)

    audit.to_csv(
        out / "00_candidate_features_audit.csv",
        index=False,
    )

    all_candidates = audit["feature"].tolist()

    informative_candidates = audit.loc[
        ~audit["zero_variance"],
        "feature",
    ].tolist()

    (out / "00_candidate_features_all.txt").write_text(
        "\n".join(all_candidates) + "\n"
    )

    (
        out
        / "00_candidate_features_nonconstant.txt"
    ).write_text(
        "\n".join(informative_candidates) + "\n"
    )

    class_counts = (
        df.groupby("site_label")
        .size()
        .rename("rows")
        .reset_index()
        .sort_values("site_label")
    )

    class_counts.to_csv(
        out / "00_class_counts.csv",
        index=False,
    )

    date_counts = (
        df.groupby("_audit_date")
        .agg(
            rows=("site_label", "size"),
            sites=("site_label", "nunique"),
        )
        .reset_index()
        .sort_values("_audit_date")
    )

    date_counts.to_csv(
        out / "00_date_counts.csv",
        index=False,
    )

    coverage = (
        df.groupby(
            ["site_label", "_audit_date"]
        )
        .size()
        .rename("rows")
        .reset_index()
    )

    coverage.to_csv(
        out / "00_site_date_coverage.csv",
        index=False,
    )

    unique_dates = sorted(
        df["_audit_date"]
        .dropna()
        .unique()
    )

    manifest = {
        "feature_set_id": "MACRO-V2-HIST",
        "stage": "00_dataset_audit",
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "future_used": False,
        "git_commit": git_commit(),
        "dataset": {
            "path": str(dataset_path),
            "sha256": sha256(dataset_path),
            "rows_after_65_site_filter": len(df),
            "sites": int(df["site_label"].nunique()),
            "first_date": str(
                pd.Timestamp(unique_dates[0]).date()
            ),
            "last_date": str(
                pd.Timestamp(unique_dates[-1]).date()
            ),
            "calendar_dates": len(unique_dates),
        },
        "label_encoder": {
            "path": str(label_encoder_path()),
            "sha256": sha256(label_encoder_path()),
            "classes": 65,
        },
        "features": {
            "numeric_candidates": len(all_candidates),
            "nonconstant_candidates": len(
                informative_candidates
            ),
            "zero_variance_candidates": int(
                audit["zero_variance"].sum()
            ),
        },
    }

    write_json(
        out / "00_manifest_dataset_audit.json",
        manifest,
    )

    print()
    print("SUMMARY")
    print("-" * 78)
    print(
        f"Date range: "
        f"{manifest['dataset']['first_date']} -> "
        f"{manifest['dataset']['last_date']}"
    )
    print(
        f"Calendar dates: "
        f"{manifest['dataset']['calendar_dates']}"
    )
    print(
        f"Candidate features: "
        f"{len(all_candidates)}"
    )
    print(
        f"Nonconstant candidates: "
        f"{len(informative_candidates)}"
    )
    print(
        f"Zero-variance: "
        f"{int(audit['zero_variance'].sum())}"
    )
    print()
    print(f"Artifacts: {out}")
    print("AUDIT COMPLETE")


if __name__ == "__main__":
    main()
