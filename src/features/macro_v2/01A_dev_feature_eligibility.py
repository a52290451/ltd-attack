from __future__ import annotations

from datetime import datetime, timezone

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


def read_feature_names(
    audit_path,
) -> list[str]:

    audit = pd.read_csv(
        audit_path,
        usecols=["feature"],
    )

    features = (
        audit["feature"]
        .astype(str)
        .str.strip()
        .tolist()
    )

    if len(features) != len(set(features)):
        raise RuntimeError(
            "Duplicate feature names in "
            "00_candidate_features_audit.csv"
        )

    return features


def main():
    print("=" * 78)
    print(
        "MACRO-V2-HIST — "
        "01A DEV-ONLY FEATURE ELIGIBILITY"
    )
    print("=" * 78)

    print(
        "Purpose: verify whether the historical-wide "
        "nonconstant audit changed the candidate universe."
    )

    print(
        "INTERNAL_TEST feature values: PROHIBITED"
    )

    print(
        "External Future: PROHIBITED"
    )

    out = output_dir()

    audit_path = (
        out
        / "00_candidate_features_audit.csv"
    )

    legacy_path = (
        out
        / "00_candidate_features_nonconstant.txt"
    )

    assignments_path = (
        out
        / "01_temporal_split_assignments.csv"
    )

    for path in [
        audit_path,
        legacy_path,
        assignments_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    # IMPORTANT:
    # Only the structural feature-name column is read
    # from Stage 00. No variance/nunique/statistics from
    # the full Historical audit are used here.
    structural_features = (
        read_feature_names(
            audit_path
        )
    )

    print()
    print(
        "Structural numeric candidates: "
        f"{len(structural_features)}"
    )

    assignments = pd.read_csv(
        assignments_path,
        dtype={
            "site_label": str,
        },
    )

    if (
        "pcap_uid"
        not in assignments.columns
        or
        "temporal_split"
        not in assignments.columns
    ):
        raise RuntimeError(
            "Split assignments require "
            "pcap_uid and temporal_split."
        )

    if assignments[
        "pcap_uid"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate pcap_uid in assignments."
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

    if (
        dev_assignments[
            "temporal_split"
        ]
        .isin(
            ["INTERNAL_TEST"]
        )
        .any()
    ):
        raise RuntimeError(
            "INTERNAL_TEST entered DEV assignments."
        )

    df, dataset_path = (
        load_historical_65()
    )

    if "pcap_uid" not in df.columns:
        raise RuntimeError(
            "Historical dataset requires pcap_uid."
        )

    if df[
        "pcap_uid"
    ].duplicated().any():
        raise RuntimeError(
            "Historical pcap_uid is not unique."
        )

    missing_columns = [
        f
        for f in structural_features
        if f not in df.columns
    ]

    if missing_columns:
        raise RuntimeError(
            "Structural candidates missing from "
            f"Historical dataset: {missing_columns}"
        )

    # Critical boundary:
    # only DEV rows enter the analytical dataframe.
    dev = df.merge(
        dev_assignments,
        on="pcap_uid",
        how="inner",
        validate="one_to_one",
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
            "Unexpected DEV split composition: "
            f"{observed_splits}"
        )

    expected_dev_rows = len(
        dev_assignments
    )

    if len(dev) != expected_dev_rows:
        raise RuntimeError(
            "DEV merge row mismatch: "
            f"{len(dev)} != {expected_dev_rows}"
        )

    print(
        f"DEV rows loaded: {len(dev):,}"
    )

    print(
        "Sites in DEV: "
        f"{dev['site_label'].nunique()}"
    )

    records = []
    eligible = []

    for feature in structural_features:

        series = pd.to_numeric(
            dev[feature],
            errors="coerce",
        )

        values = series.to_numpy(
            dtype=float,
        )

        finite_mask = np.isfinite(
            values
        )

        finite_values = values[
            finite_mask
        ]

        finite_count = int(
            len(finite_values)
        )

        finite_fraction = float(
            finite_count
            / len(values)
        )

        if finite_count == 0:
            nunique_finite = 0
            variance = np.nan

        else:
            nunique_finite = int(
                pd.Series(
                    finite_values
                ).nunique(
                    dropna=True
                )
            )

            variance = float(
                np.var(
                    finite_values,
                    ddof=0,
                )
            )

        if finite_count < 2:
            is_eligible = False
            reason = "insufficient_finite"

        elif nunique_finite <= 1:
            is_eligible = False
            reason = "constant_dev"

        else:
            is_eligible = True
            reason = "eligible"

        if is_eligible:
            eligible.append(
                feature
            )

        records.append(
            {
                "feature":
                    feature,

                "finite_count":
                    finite_count,

                "finite_fraction":
                    finite_fraction,

                "nunique_finite":
                    nunique_finite,

                "variance_dev":
                    variance,

                "eligible_dev":
                    is_eligible,

                "eligibility_reason":
                    reason,
            }
        )

    eligibility = pd.DataFrame(
        records
    )

    eligibility_path = (
        out
        / "01A_dev_feature_eligibility.csv"
    )

    eligible_path = (
        out
        / "01A_candidate_features_dev_eligible.txt"
    )

    eligibility.to_csv(
        eligibility_path,
        index=False,
    )

    eligible_path.write_text(
        "\n".join(
            eligible
        )
        + "\n"
    )

    legacy = [
        x.strip()
        for x in (
            legacy_path
            .read_text()
            .splitlines()
        )
        if x.strip()
    ]

    legacy_set = set(
        legacy
    )

    eligible_set = set(
        eligible
    )

    legacy_only = sorted(
        legacy_set
        - eligible_set
    )

    dev_only = sorted(
        eligible_set
        - legacy_set
    )

    set_equal = (
        legacy_set
        == eligible_set
    )

    ordered_equal = (
        legacy
        == eligible
    )

    comparison = pd.DataFrame(
        [
            {
                "comparison":
                    "structural_candidates",
                "count":
                    len(
                        structural_features
                    ),
            },
            {
                "comparison":
                    "legacy_full_historical_nonconstant",
                "count":
                    len(legacy),
            },
            {
                "comparison":
                    "dev_only_eligible",
                "count":
                    len(eligible),
            },
            {
                "comparison":
                    "legacy_only",
                "count":
                    len(legacy_only),
            },
            {
                "comparison":
                    "dev_only_only",
                "count":
                    len(dev_only),
            },
        ]
    )

    comparison_path = (
        out
        / "01A_dev_eligibility_comparison.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    manifest = {
        "feature_set_id":
            "MACRO-V2-HIST",

        "stage":
            "01A_dev_feature_eligibility",

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "git_commit":
            git_commit(),

        "historical_dataset": {
            "path":
                str(dataset_path),

            "sha256":
                sha256(
                    dataset_path
                ),
        },

        "structural_candidate_source": {
            "path":
                str(audit_path),

            "sha256":
                sha256(
                    audit_path
                ),

            "column_used":
                "feature",

            "full_historical_statistics_used":
                False,

            "count":
                len(
                    structural_features
                ),
        },

        "split_assignments": {
            "path":
                str(
                    assignments_path
                ),

            "sha256":
                sha256(
                    assignments_path
                ),
        },

        "eligibility_rule": {
            "finite_count_min":
                2,

            "nunique_finite_min":
                2,

            "missingness_threshold":
                None,

            "variance_used_for_decision":
                False,
        },

        "data_policy": {
            "development_splits_used":
                DEV_SPLITS,

            "dev_rows":
                len(dev),

            "internal_test_values_used":
                False,

            "external_future_used":
                False,
        },

        "results": {
            "structural_candidates":
                len(
                    structural_features
                ),

            "dev_only_eligible":
                len(eligible),

            "legacy_nonconstant_count":
                len(legacy),

            "set_equal_to_legacy":
                set_equal,

            "ordered_equal_to_legacy":
                ordered_equal,

            "legacy_only_features":
                legacy_only,

            "dev_only_features":
                dev_only,
        },

        "interpretation": {
            "if_set_equal":
                (
                    "The previous 311-feature universe "
                    "is retrospectively equivalent to "
                    "DEV-only eligibility; downstream "
                    "Stages 02-06A need not be recomputed."
                ),

            "if_set_differs":
                (
                    "The previous universe was affected "
                    "by full-Historical eligibility and "
                    "Stages 02-06A must be recomputed "
                    "from the DEV-only list."
                ),
        },
    }

    manifest_path = (
        out
        / "01A_manifest_dev_feature_eligibility.json"
    )

    write_json(
        manifest_path,
        manifest,
    )

    print()
    print("=" * 78)
    print(
        "DEV-ONLY ELIGIBILITY RESULT"
    )
    print("=" * 78)

    print(
        "Structural candidates: "
        f"{len(structural_features)}"
    )

    print(
        "Legacy full-Historical "
        "nonconstant: "
        f"{len(legacy)}"
    )

    print(
        "DEV-only eligible: "
        f"{len(eligible)}"
    )

    print(
        f"Set equal: {set_equal}"
    )

    print(
        f"Ordered equal: {ordered_equal}"
    )

    print()

    print(
        "Legacy-only features:"
    )

    if legacy_only:
        for feature in legacy_only:
            print(
                f"  - {feature}"
            )
    else:
        print(
            "  NONE"
        )

    print()

    print(
        "DEV-only-only features:"
    )

    if dev_only:
        for feature in dev_only:
            print(
                f"  - {feature}"
            )
    else:
        print(
            "  NONE"
        )

    print()

    if set_equal:
        print(
            "PASS: DEV-only eligibility "
            "produces the same feature set."
        )
    else:
        print(
            "FAIL: candidate universe differs. "
            "Do NOT proceed to Stage 06B."
        )

    print()
    print(
        f"Eligibility table: "
        f"{eligibility_path}"
    )

    print(
        f"DEV-only list: "
        f"{eligible_path}"
    )

    print(
        f"Manifest: "
        f"{manifest_path}"
    )


if __name__ == "__main__":
    main()
