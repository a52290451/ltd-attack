from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.paths import data_path
from src.features.macro_v2._common import (
    git_commit,
    historical_csv,
    load_elite_sites,
    output_dir,
    sha256,
    write_json,
)


VECTOR_CANDIDATES = [
    "CLEAN_final_vectors_sites.csv",
    "cached_vectors_with_dates.csv",
]

SPLITS = [
    "DEV_EARLY",
    "DEV_MIDDLE",
    "DEV_LATE",
    "INTERNAL_TEST",
]


def metadata_columns(path: Path):

    header = pd.read_csv(
        path,
        nrows=0,
    ).columns.tolist()

    wanted = [
        "pcap_uid",
        "pcap_name",
        "site_label",
        "date_id",
    ]

    return [
        c
        for c in wanted
        if c in header
    ], header


def read_metadata(path: Path):

    cols, header = metadata_columns(
        path
    )

    if "site_label" not in cols:
        raise RuntimeError(
            f"{path} has no site_label."
        )

    if (
        "pcap_uid" not in cols
        and
        "pcap_name" not in cols
    ):
        raise RuntimeError(
            f"{path} has neither pcap_uid nor pcap_name."
        )

    df = pd.read_csv(
        path,
        usecols=cols,
    )

    df["site_label"] = (
        df["site_label"]
        .astype(str)
    )

    return df, header


def choose_key(
    macro,
    vectors,
):

    if (
        "pcap_uid" in macro.columns
        and
        "pcap_uid" in vectors.columns
    ):
        return "pcap_uid"

    if (
        "pcap_name" in macro.columns
        and
        "pcap_name" in vectors.columns
    ):
        return "pcap_name"

    return None


def main():

    print("=" * 78)
    print(
        "08A — MICRO TEMPORAL ALIGNMENT AUDIT"
    )
    print("=" * 78)

    print(
        "Metadata only."
    )

    print(
        "No direction/size vectors are read."
    )

    print(
        "No model training."
    )

    print(
        "INTERNAL_TEST vector values: CLOSED"
    )

    print(
        "Future-B: CLOSED"
    )

    elite = {
        str(x)
        for x in load_elite_sites()
    }

    macro_path = historical_csv()

    macro, macro_header = (
        read_metadata(
            macro_path
        )
    )

    macro = (
        macro[
            macro[
                "site_label"
            ].isin(
                elite
            )
        ]
        .copy()
    )

    assignment_path = (
        output_dir()
        / "01_temporal_split_assignments.csv"
    )

    assignments = pd.read_csv(
        assignment_path,
        usecols=[
            "pcap_uid",
            "temporal_split",
        ],
    )

    if "pcap_uid" not in macro.columns:
        raise RuntimeError(
            "Canonical Macro dataset requires pcap_uid "
            "for temporal assignment."
        )

    macro = macro.merge(
        assignments,
        on="pcap_uid",
        how="inner",
        validate="one_to_one",
    )

    if set(
        macro[
            "temporal_split"
        ].unique()
    ) - set(SPLITS):
        raise RuntimeError(
            "Unexpected temporal split."
        )

    source_rows = []
    split_rows = []

    eligible_sources = []

    for filename in VECTOR_CANDIDATES:

        path = Path(
            data_path(
                "historical",
                filename,
            )
        )

        if not path.exists():

            source_rows.append(
                {
                    "source":
                        filename,

                    "exists":
                        False,

                    "status":
                        "MISSING",
                }
            )

            continue

        vectors, header = (
            read_metadata(
                path
            )
        )

        total_rows = len(
            vectors
        )

        vectors65 = (
            vectors[
                vectors[
                    "site_label"
                ].isin(
                    elite
                )
            ]
            .copy()
        )

        key = choose_key(
            macro,
            vectors65,
        )

        if key is None:

            source_rows.append(
                {
                    "source":
                        filename,

                    "exists":
                        True,

                    "rows_total":
                        total_rows,

                    "rows_elite65":
                        len(
                            vectors65
                        ),

                    "sites_elite":
                        vectors65[
                            "site_label"
                        ].nunique(),

                    "status":
                        "NO_COMMON_CAPTURE_KEY",
                }
            )

            continue

        macro_dup = int(
            macro[
                key
            ].duplicated(
                keep=False
            ).sum()
        )

        vector_dup = int(
            vectors65[
                key
            ].duplicated(
                keep=False
            ).sum()
        )

        left = macro[
            [
                key,
                "site_label",
                "temporal_split",
            ]
        ].rename(
            columns={
                "site_label":
                    "macro_site"
            }
        )

        right = vectors65[
            [
                key,
                "site_label",
            ]
        ].rename(
            columns={
                "site_label":
                    "vector_site"
            }
        )

        merged = left.merge(
            right,
            on=key,
            how="left",
            indicator=True,
        )

        merged[
            "matched"
        ] = (
            merged[
                "_merge"
            ]
            == "both"
        )

        merged[
            "label_match"
        ] = (
            merged[
                "macro_site"
            ]
            ==
            merged[
                "vector_site"
            ]
        )

        for split in SPLITS:

            g = merged[
                merged[
                    "temporal_split"
                ]
                == split
            ]

            n = len(
                g
            )

            matched = int(
                g[
                    "matched"
                ].sum()
            )

            mismatch = int(
                (
                    g[
                        "matched"
                    ]
                    &
                    ~g[
                        "label_match"
                    ]
                ).sum()
            )

            split_rows.append(
                {
                    "source":
                        filename,

                    "key":
                        key,

                    "split":
                        split,

                    "macro_rows":
                        n,

                    "matched_rows":
                        matched,

                    "coverage":
                        (
                            matched / n
                            if n
                            else np.nan
                        ),

                    "label_mismatch_n":
                        mismatch,
                }
            )

        dev = merged[
            merged[
                "temporal_split"
            ].isin(
                [
                    "DEV_EARLY",
                    "DEV_MIDDLE",
                    "DEV_LATE",
                ]
            )
        ]

        internal = merged[
            merged[
                "temporal_split"
            ]
            == "INTERNAL_TEST"
        ]

        dev_coverage = float(
            dev[
                "matched"
            ].mean()
        )

        internal_metadata_coverage = (
            float(
                internal[
                    "matched"
                ].mean()
            )
            if len(
                internal
            )
            else np.nan
        )

        mismatch_total = int(
            (
                merged[
                    "matched"
                ]
                &
                ~merged[
                    "label_match"
                ]
            ).sum()
        )

        vector_payload_columns = [
            c
            for c in [
                "direction_vector",
                "size_vector",
                "time_vector",
            ]
            if c in header
        ]

        eligible_source = bool(
            macro_dup == 0
            and
            vector_dup == 0
            and
            dev_coverage == 1.0
            and
            mismatch_total == 0
            and
            "direction_vector"
            in header
            and
            "size_vector"
            in header
        )

        if eligible_source:

            eligible_sources.append(
                {
                    "source":
                        filename,

                    "path":
                        path,

                    "key":
                        key,

                    "dev_coverage":
                        dev_coverage,

                    "internal_metadata_coverage":
                        internal_metadata_coverage,

                    "key_priority":
                        (
                            1
                            if key
                            == "pcap_uid"
                            else 0
                        ),

                    "legacy_micro_priority":
                        (
                            1
                            if filename
                            ==
                            "CLEAN_final_vectors_sites.csv"
                            else 0
                        ),
                }
            )

        source_rows.append(
            {
                "source":
                    filename,

                "exists":
                    True,

                "rows_total":
                    total_rows,

                "rows_elite65":
                    len(
                        vectors65
                    ),

                "sites_elite":
                    vectors65[
                        "site_label"
                    ].nunique(),

                "join_key":
                    key,

                "macro_duplicate_key_rows":
                    macro_dup,

                "vector_duplicate_key_rows":
                    vector_dup,

                "dev_coverage":
                    dev_coverage,

                "internal_metadata_coverage":
                    internal_metadata_coverage,

                "label_mismatch_n":
                    mismatch_total,

                "payload_columns_present":
                    ",".join(
                        vector_payload_columns
                    ),

                "payload_columns_read":
                    False,

                "eligible_for_clean_micro":
                    eligible_source,

                "status":
                    (
                        "ELIGIBLE"
                        if eligible_source
                        else "NOT_ELIGIBLE"
                    ),
            }
        )

    source_df = pd.DataFrame(
        source_rows
    )

    split_df = pd.DataFrame(
        split_rows
    )

    selected = None

    if eligible_sources:

        selected = sorted(
            eligible_sources,
            key=lambda x: (
                x[
                    "dev_coverage"
                ],
                x[
                    "internal_metadata_coverage"
                ],
                x[
                    "key_priority"
                ],
                x[
                    "legacy_micro_priority"
                ],
            ),
            reverse=True,
        )[0]

    source_path = (
        output_dir()
        / "08A_micro_alignment_sources.csv"
    )

    split_path = (
        output_dir()
        / "08A_micro_alignment_by_split.csv"
    )

    source_df.to_csv(
        source_path,
        index=False,
    )

    split_df.to_csv(
        split_path,
        index=False,
    )

    manifest = {
        "stage":
            "08A_micro_temporal_alignment_audit",

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "git_commit":
            git_commit(),

        "data_policy": {
            "metadata_only":
                True,

            "model_training":
                False,

            "vector_payload_columns_read":
                False,

            "internal_test_metadata_audited":
                True,

            "internal_test_vector_values_used":
                False,

            "external_future_used":
                False,
        },

        "macro_population": {
            "sites":
                len(
                    elite
                ),

            "rows":
                len(
                    macro
                ),

            "split_counts":
                {
                    k:
                        int(v)
                    for k, v in
                    macro[
                        "temporal_split"
                    ]
                    .value_counts()
                    .to_dict()
                    .items()
                },
        },

        "selection_policy": {
            "requirements": [
                "100% DEV metadata capture coverage",
                "zero duplicate capture keys",
                "zero site-label mismatches",
                "direction_vector and size_vector columns present",
            ],

            "tie_break": [
                "INTERNAL_TEST metadata coverage",
                "pcap_uid preferred over pcap_name",
                "legacy canonical Micro source preferred",
            ],
        },

        "selected_source":
            (
                {
                    "source":
                        selected[
                            "source"
                        ],

                    "path":
                        str(
                            selected[
                                "path"
                            ]
                        ),

                    "join_key":
                        selected[
                            "key"
                        ],

                    "dev_coverage":
                        selected[
                            "dev_coverage"
                        ],

                    "internal_metadata_coverage":
                        selected[
                            "internal_metadata_coverage"
                        ],
                }
                if selected
                else None
            ),

        "next_if_selected":
            "08B clean temporal Micro baseline",

        "next_if_none":
            (
                "resolve Micro/Macro capture provenance "
                "before model training"
            ),

        "inputs": {
            "macro_metadata_path":
                str(
                    macro_path
                ),

            "macro_sha256":
                sha256(
                    macro_path
                ),

            "split_assignment_path":
                str(
                    assignment_path
                ),

            "split_assignment_sha256":
                sha256(
                    assignment_path
                ),
        },
    }

    manifest_path = (
        output_dir()
        / "08A_micro_alignment_manifest.json"
    )

    write_json(
        manifest_path,
        manifest,
    )

    print()
    print("=" * 78)
    print(
        "SOURCE SUMMARY"
    )
    print("=" * 78)

    print(
        source_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 78)
    print(
        "SPLIT COVERAGE"
    )
    print("=" * 78)

    print(
        split_df.to_string(
            index=False
        )
    )

    print()
    print(
        "SELECTED SOURCE:"
    )

    print(
        manifest[
            "selected_source"
        ]
    )

    print()
    print(
        "No vector payload was read."
    )

    print(
        "INTERNAL_TEST numerical/vector values remain CLOSED."
    )

    print(
        "Future-B remains CLOSED."
    )


if __name__ == "__main__":
    main()
