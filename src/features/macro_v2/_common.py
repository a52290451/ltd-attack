from __future__ import annotations

from pathlib import Path
import hashlib
import json
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.paths import data_path, artifact_path, result_path


METADATA_COLUMNS = {
    "pcap_uid",
    "site_label",
    "pcap_name",
    "category",
    "site",
    "vector_len",
    "matches",
    "n_errors",
    "direction_vector",
    "time_vector",
    "size_vector",
    "hour_bin",
    "date_id",
}


def historical_csv() -> Path:
    return Path(
        data_path(
            "historical",
            "CLEAN_final_features_sites.csv"
        )
    )


def label_encoder_path() -> Path:
    candidates = [
        Path(
            artifact_path(
                "micro",
                "MICRO-002-R",
                "ds3_label_encoder_vec_3000.joblib"
            )
        ),
        Path.home()
        / "ltd-storage"
        / "artifacts"
        / "micro"
        / "MICRO-002-R"
        / "ds3_label_encoder_vec_3000.joblib",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def output_dir() -> Path:
    p = Path(
        result_path(
            "feature_engineering",
            "MACRO-V2-HIST"
        )
    )
    p.mkdir(parents=True, exist_ok=True)
    return p


def sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    except Exception:
        return "UNKNOWN"


def load_elite_sites() -> set[str]:
    path = label_encoder_path()

    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró LabelEncoder Micro: {path}"
        )

    encoder = joblib.load(path)

    classes = {
        str(x)
        for x in encoder.classes_
    }

    if len(classes) != 65:
        raise RuntimeError(
            f"Se esperaban 65 sitios y se encontraron {len(classes)}"
        )

    return classes


def load_historical_65() -> tuple[pd.DataFrame, Path]:
    path = historical_csv()

    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró dataset Historical: {path}"
        )

    df = pd.read_csv(path)

    if "site_label" not in df.columns:
        raise RuntimeError(
            "Historical no contiene columna site_label"
        )

    elite_sites = load_elite_sites()

    df["site_label"] = df["site_label"].astype(str)

    df = df[
        df["site_label"].isin(elite_sites)
    ].copy()

    found_sites = set(df["site_label"].unique())

    missing = elite_sites - found_sites

    if missing:
        raise RuntimeError(
            "Faltan sitios de la población Micro: "
            + ", ".join(sorted(missing))
        )

    return df, path


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    numeric = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    return [
        c
        for c in numeric
        if c not in METADATA_COLUMNS
    ]


def extract_capture_dates(df: pd.DataFrame) -> pd.Series:
    # Preferimos date_id si ya existe y es utilizable.
    if "date_id" in df.columns:
        parsed = pd.to_datetime(
            df["date_id"],
            errors="coerce",
        ).dt.normalize()

        if parsed.notna().mean() >= 0.95:
            return parsed

    if "pcap_name" not in df.columns:
        raise RuntimeError(
            "No existen date_id ni pcap_name para reconstruir fechas."
        )

    names = df["pcap_name"].astype(str)

    # Formato histórico observado:
    # ..._20251118-090002...
    extracted = names.str.extract(
        r"(?P<date>20\d{6})"
    )["date"]

    parsed = pd.to_datetime(
        extracted,
        format="%Y%m%d",
        errors="coerce",
    )

    success = parsed.notna().mean()

    if success < 0.95:
        raise RuntimeError(
            f"Solo se pudo reconstruir fecha para {success:.2%} "
            "de las capturas."
        )

    return parsed.dt.normalize()


def write_json(path: Path, obj: dict) -> None:
    path.write_text(
        json.dumps(
            obj,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
        + "\n"
    )
