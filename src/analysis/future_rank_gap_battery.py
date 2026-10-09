"""Diagnósticos post-hoc de rank-gap sobre el Future-B ya congelado.

El módulo no importa entrenadores, no abre datos para volver a entrenar y no
contiene ninguna ruta de selección. Su única fuente de scores es el NPZ de
10B DEV_FROZEN.
"""

from __future__ import annotations

from datetime import datetime, timezone
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any
import zipfile

import numpy as np
import pandas as pd
from src.utils.paths import data_path, result_path
from src.experiments.phase13_battery import (
    EPS,
    entropy_margin,
    metric_values,
    rank_gap_group,
    rank_values,
    sha256,
    _git_commit,
)


COMPONENTS = ("MICRO", "MACRO_XGB", "MACRO_LTD", "MACRO_FINAL", "HYBRID_FINAL")
FUTURE_DIAGNOSTIC_FLAGS = {
    "future_b_open": True,
    "future_b_use": "POSTHOC_DIAGNOSTIC_ONLY",
    "selection_allowed": False,
    "training_allowed": False,
}
OUTPUT = Path(result_path("final", "FUTURE-RANK-GAP-PHASE14"))
FUTURE_SCORE_PATH = Path(result_path("final", "FUTURE-B")) / "10B_scores_DEV_FROZEN.npz"
FUTURE_MICRO_PATH = Path(data_path("future", "CLEAN_final_vectors_sites_concept_drift.csv"))
FUTURE_MACRO_PATH = Path(data_path("future", "CLEAN_final_features_sites_concept_drift.csv"))
HISTORICAL_MACRO_PATH = Path(data_path("historical", "CLEAN_final_features_sites.csv"))
MAX_LEN = 3000
FUTURE_SCORE_SHA256 = "d0edcd2780454b5cd39ef962ec45375d8822dc26b4b3c0b02cf4c06b0b29c7bb"
_SAFE_NPZ_FIELDS = (
    "query_date",
    "y_true",
    "candidate_labels",
    "micro_probs",
    "macro_xgb_probs",
    "macro_ltd_probs",
    "macro_final_probs",
    "hybrid_final_probs",
)
_PROBABILITY_FIELDS = _SAFE_NPZ_FIELDS[3:]


def _load_future_yaml(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML es obligatorio para validar la configuración 14A") from exc
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("La configuración 14A debe ser un mapping")
    return payload


def validate_future_config(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("stage") != "14A" or payload.get("status") != "POSTHOC_DIAGNOSTIC_ONLY":
        raise ValueError("guardrail: configuración 14A debe ser POSTHOC_DIAGNOSTIC_ONLY")
    if payload.get("components") != list(COMPONENTS):
        raise ValueError("guardrail: componentes Future-B incompatibles")
    if payload.get("micro", {}).get("max_len") != MAX_LEN:
        raise ValueError("guardrail: MAX_LEN 14A debe ser 3000")
    if payload.get("guardrails") != FUTURE_DIAGNOSTIC_FLAGS:
        raise ValueError("guardrail: flags Future-B incompatibles")
    if payload.get("metrics") != ["accuracy", "top5_accuracy", "mrr", "mean_true_rank", "entropy", "top1_top2_margin"]:
        raise ValueError("guardrail: métricas 14A incompatibles")
    expected_paths = {
        "result_stage": "final", "result_name": "FUTURE-RANK-GAP-PHASE14",
        "source_stage": "final", "source_name": "FUTURE-B", "source_file": "10B_scores_DEV_FROZEN.npz",
        "micro_source": "CLEAN_final_vectors_sites_concept_drift.csv",
        "future_macro_source": "CLEAN_final_features_sites_concept_drift.csv",
        "historical_macro_source": "CLEAN_final_features_sites.csv",
    }
    if payload.get("paths") != expected_paths:
        raise ValueError("guardrail: rutas lógicas 14A incompatibles")
    return payload


def load_future_config(path: str | Path | None = None) -> tuple[Path, dict[str, Any]]:
    config_path = Path(path) if path else Path(__file__).resolve().parents[2] / "configs/experiments/FUTURE_RANK_GAP_V1.yaml"
    payload = _load_future_yaml(config_path)
    return config_path, validate_future_config(payload)


def build_future_paths(payload: dict[str, Any]) -> dict[str, Path]:
    paths = payload["paths"]
    return {
        "scores": Path(result_path(paths["source_stage"], paths["source_name"])) / paths["source_file"],
        "output": Path(result_path(paths["result_stage"], paths["result_name"])),
        "micro": Path(data_path("future", paths["micro_source"])),
        "future_macro": Path(data_path("future", paths["future_macro_source"])),
        "historical_macro": Path(data_path("historical", paths["historical_macro_source"])),
    }


def build_future_manifest(*, inputs: dict[str, Any], code_hashes: dict[str, Any], config_hash: str, outputs: dict[str, Any], git_commit: str | None = None) -> dict[str, Any]:
    return {
        "stage": "14A",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit or _git_commit(),
        "scientific_status": "POSTHOC_DIAGNOSTIC_ONLY",
        "research_question": "¿Qué parte del gap Top-1→Top-5 de Future-B es recuperable y cómo se relaciona descriptivamente con tamaño, drift y confusión?",
        "future_b_open": True,
        "future_b_use": "POSTHOC_DIAGNOSTIC_ONLY",
        "selection_allowed": False,
        "training_allowed": False,
        "data_policy": FUTURE_DIAGNOSTIC_FLAGS,
        "architecture": {"micro_max_len": MAX_LEN, "no_model_updates": True},
        "candidate_matrix": {"components": list(COMPONENTS)},
        "metrics": ["accuracy", "top5_accuracy", "mrr", "mean_true_rank", "entropy", "top1_top2_margin"],
        "predeclared_rules": {"per_site_category": "accuracy>=0.75; else top5-top1>=0.25; else lost"},
        "inputs_sha256": inputs,
        "code_sha256": code_hashes,
        "config_sha256": config_hash,
        "outputs_sha256": outputs,
    }


def rank_distribution(ranks: np.ndarray) -> dict[str, int]:
    ranks = np.asarray(ranks, dtype=np.int64)
    return {
        "rank_1": int(np.sum(ranks == 1)),
        "rank_2": int(np.sum(ranks == 2)),
        "rank_3": int(np.sum(ranks == 3)),
        "rank_4": int(np.sum(ranks == 4)),
        "rank_5": int(np.sum(ranks == 5)),
        "rank_6_10": int(np.sum((ranks >= 6) & (ranks <= 10))),
        "rank_gt_10": int(np.sum(ranks > 10)),
    }


def classify_rank_gap(rank: int) -> str:
    return rank_gap_group(int(rank))


def entropy_margin(probabilities: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(probabilities, dtype=np.float64)
    order = np.argsort(-values, axis=1, kind="stable")
    top1 = values[np.arange(len(values)), order[:, 0]]
    top2 = values[np.arange(len(values)), order[:, 1]] if values.shape[1] > 1 else np.zeros(len(values))
    entropy = -np.sum(np.clip(values, EPS, 1.0) * np.log(np.clip(values, EPS, 1.0)), axis=1)
    return {"top1_confidence": top1, "top1_top2_margin": top1 - top2, "entropy": entropy}


def build_confusion_edges(y_true: np.ndarray, probabilities: np.ndarray, labels: list[str] | np.ndarray) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(probabilities)
    labels = np.asarray(labels, dtype=str)
    order = np.argsort(-probabilities, axis=1, kind="stable")
    predicted = order[:, 0]
    ranks = rank_values(probabilities, y_true)
    errors = predicted != y_true
    rows = []
    for true_index, pred_index, rank in zip(y_true[errors], predicted[errors], ranks[errors]):
        rows.append({"true_site": labels[true_index], "predicted_site": labels[pred_index], "count": 1, "true_rank": int(rank), "true_in_top5": bool(rank <= 5)})
    if not rows:
        return pd.DataFrame(columns=["true_site", "predicted_site", "count", "normalized_fraction", "mean_true_rank", "true_in_top5_fraction"])
    frame = pd.DataFrame(rows)
    grouped = frame.groupby(["true_site", "predicted_site"], as_index=False).agg(count=("count", "sum"), mean_true_rank=("true_rank", "mean"), true_in_top5_fraction=("true_in_top5", "mean"))
    totals = grouped.groupby("true_site")["count"].transform("sum")
    grouped["normalized_fraction"] = grouped["count"] / totals
    return grouped.sort_values(["count", "true_site", "predicted_site"], ascending=[False, True, True]).reset_index(drop=True)


def _parse_vector(value: Any) -> list[float]:
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(item) for item in value]
    if pd.isna(value):
        return []
    parsed = ast.literal_eval(str(value))
    if not isinstance(parsed, (list, tuple)):
        raise ValueError("Vector no es una lista")
    return [float(item) for item in parsed]


def _read_npy_header(archive: zipfile.ZipFile, member: str) -> tuple[tuple[int, ...], np.dtype, bool]:
    try:
        with archive.open(member, "r") as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
                return shape, dtype, fortran_order
            if version in {(2, 0), (3, 0)}:
                shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
                return shape, dtype, fortran_order
    except (KeyError, ValueError, OSError) as exc:
        raise RuntimeError(f"Cabecera NPY inválida para {member}") from exc
    raise RuntimeError(f"Versión NPY no soportada para {member}: {version}")


def _validate_npz_headers(score_path: Path, *, expected_n: int, expected_classes: int) -> None:
    expected_fields = set(_SAFE_NPZ_FIELDS) | {"pcap_uid"}
    try:
        with zipfile.ZipFile(score_path) as archive:
            members = [name for name in archive.namelist() if name.endswith(".npy")]
            fields = {name[:-4] for name in members}
            missing = sorted(expected_fields - fields)
            unexpected = sorted(fields - expected_fields)
            if missing:
                raise RuntimeError(f"NPZ Future-B incompleto; faltan: {missing}")
            if unexpected:
                raise RuntimeError(f"NPZ Future-B contiene campos inesperados: {unexpected}")

            headers = {
                field: _read_npy_header(archive, f"{field}.npy")
                for field in sorted(fields)
            }
    except zipfile.BadZipFile as exc:
        raise RuntimeError(f"NPZ Future-B no es un ZIP válido: {score_path}") from exc

    uid_shape, uid_dtype, _ = headers["pcap_uid"]
    if uid_dtype.kind != "O" or uid_shape != (expected_n,):
        raise RuntimeError(
            "La cabecera de pcap_uid debe ser dtype=object y shape "
            f"({expected_n},); observado dtype={uid_dtype}, shape={uid_shape}"
        )

    query_shape, query_dtype, _ = headers["query_date"]
    if query_dtype != np.dtype("datetime64[D]") or query_shape != (expected_n,):
        raise RuntimeError("query_date debe ser datetime64[D] unidimensional con la longitud congelada")

    y_shape, y_dtype, _ = headers["y_true"]
    if y_dtype != np.dtype("int64") or y_shape != (expected_n,):
        raise RuntimeError("y_true debe ser int64 unidimensional con la longitud congelada")

    labels_shape, labels_dtype, _ = headers["candidate_labels"]
    if labels_dtype.kind != "U" or labels_shape != (expected_classes,):
        raise RuntimeError("candidate_labels debe ser Unicode unidimensional con 65 clases congeladas")

    for field in _PROBABILITY_FIELDS:
        shape, dtype, _ = headers[field]
        if dtype != np.dtype("float32") or shape != (expected_n, expected_classes):
            raise RuntimeError(
                f"{field} debe ser float32 con shape ({expected_n}, {expected_classes})"
            )


def _load_10b_date_deriver():
    root = Path(__file__).resolve().parents[2]
    module_path = root / "src/features/macro_v2/07A_candidate_conditioned_temporal_encoder.py"
    if str(module_path.parent) not in sys.path:
        sys.path.insert(0, str(module_path.parent))
    spec = importlib.util.spec_from_file_location("p14_10b_date_deriver", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("No se pudo cargar la función de fechas congelada de 10B")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.derive_dates


def _load_10b_elite_sites() -> set[str]:
    from src.features.macro_v2._common import load_elite_sites

    return {str(value) for value in load_elite_sites()}


def _reconstruct_future_identity(
    *,
    future_macro_path: Path,
    historical_macro_path: Path,
    future_micro_path: Path,
    candidate_labels: np.ndarray,
    expected_n: int,
    expected_classes: int,
    elite_sites: set[str] | None = None,
    date_deriver=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not future_macro_path.is_file():
        raise FileNotFoundError(f"No existe el CSV Macro canónico Future-B: {future_macro_path}")
    future = pd.read_csv(future_macro_path)
    if "pcap_uid" not in future.columns:
        raise RuntimeError("Future Macro no contiene pcap_uid")

    if "site_label" not in future.columns:
        if "site" not in future.columns:
            raise RuntimeError("Future Macro no contiene site_label ni site")
        if not historical_macro_path.is_file():
            raise FileNotFoundError(f"No existe la fuente histórica para site -> site_label: {historical_macro_path}")
        historic_header = pd.read_csv(historical_macro_path, nrows=0).columns.tolist()
        if "site" not in historic_header or "site_label" not in historic_header:
            raise RuntimeError("La fuente histórica no permite reconstruir site -> site_label")
        mapping_df = pd.read_csv(historical_macro_path, usecols=["site", "site_label"]).dropna().drop_duplicates()
        conflicts = mapping_df.groupby("site")["site_label"].nunique()
        if (conflicts > 1).any():
            raise RuntimeError("Mapping histórico site -> site_label no único")
        site_map = dict(zip(mapping_df["site"].astype(str), mapping_df["site_label"].astype(str)))
        future["site_label"] = future["site"].astype(str).map(site_map)

    future = future.dropna(subset=["site_label"]).copy()
    future["site_label"] = future["site_label"].astype(str)
    elite = {str(value) for value in (elite_sites if elite_sites is not None else _load_10b_elite_sites())}
    future = future[future["site_label"].isin(elite)].copy().reset_index(drop=True)

    if len(future) != expected_n:
        raise RuntimeError(f"Unexpected Future-B capture count: {len(future)}; expected {expected_n}")
    if future["site_label"].nunique() != expected_classes:
        raise RuntimeError(f"Unexpected Future-B class count: {future['site_label'].nunique()}; expected {expected_classes}")
    if future["pcap_uid"].isna().any():
        raise RuntimeError("Future Macro contiene pcap_uid nulo")
    uids = future["pcap_uid"].astype(str).to_numpy()
    if np.any(uids == "") or len(np.unique(uids)) != len(uids):
        raise RuntimeError("Future Macro contiene pcap_uid duplicado o vacío")

    derive_dates = date_deriver or _load_10b_date_deriver()
    dates, _ = derive_dates(future)
    query_date = pd.to_datetime(dates, errors="coerce").to_numpy(dtype="datetime64[D]")
    if np.isnat(query_date).any():
        raise RuntimeError("Future Macro contiene fechas no derivables")

    expected_labels = np.asarray(sorted(future["site_label"].unique()), dtype=str)
    if not np.array_equal(candidate_labels, expected_labels):
        raise RuntimeError("candidate_labels no coincide con el orden congelado de Future-B")
    label_to_index = {label: index for index, label in enumerate(candidate_labels)}
    reconstructed_y = np.asarray([label_to_index[label] for label in future["site_label"]], dtype=np.int64)

    if not future_micro_path.is_file():
        raise FileNotFoundError(f"No existe el dataset Micro Future-B: {future_micro_path}")
    # Replicar 10B: solo los UID presentes en la cohorte Macro seleccionada.
    target_uids = set(uids)
    micro = pd.read_csv(future_micro_path, usecols=["pcap_uid"])
    micro_uids = micro["pcap_uid"].astype(str)
    micro_uids = micro_uids[micro_uids.isin(target_uids)].to_numpy()
    if len(np.unique(micro_uids)) != len(micro_uids):
        raise RuntimeError("El dataset Micro Future-B contiene pcap_uid duplicado")
    if len(micro_uids) != expected_n or set(micro_uids) != target_uids:
        missing = sorted(target_uids - set(micro_uids))
        extra = sorted(set(micro_uids) - target_uids)
        raise RuntimeError(f"Cobertura Micro Future-B incompleta; missing={missing[:5]}, extra={extra[:5]}")

    return uids, query_date, reconstructed_y


def _load_npz(
    score_path: Path = FUTURE_SCORE_PATH,
    *,
    future_macro_path: Path = FUTURE_MACRO_PATH,
    historical_macro_path: Path = HISTORICAL_MACRO_PATH,
    future_micro_path: Path = FUTURE_MICRO_PATH,
    expected_sha256: str = FUTURE_SCORE_SHA256,
    elite_sites: set[str] | None = None,
    date_deriver=None,
    expected_n: int = 18543,
    expected_classes: int = 65,
) -> dict[str, np.ndarray]:
    score_path = Path(score_path)
    if not score_path.is_file():
        raise FileNotFoundError(f"No existe el NPZ congelado de Future-B; no se reejecuta 10B: {score_path}")
    observed_sha256 = sha256(score_path)
    if observed_sha256 != expected_sha256:
        raise RuntimeError(f"SHA256 NPZ Future-B incompatible; expected={expected_sha256}; observed={observed_sha256}")

    _validate_npz_headers(score_path, expected_n=expected_n, expected_classes=expected_classes)
    with np.load(score_path, allow_pickle=False) as payload:
        # Deliberadamente no se accede a payload["pcap_uid"]: ese campo es object.
        result = {field: np.array(payload[field], copy=True) for field in _SAFE_NPZ_FIELDS}

    reconstructed_uids, reconstructed_dates, reconstructed_y = _reconstruct_future_identity(
        future_macro_path=Path(future_macro_path),
        historical_macro_path=Path(historical_macro_path),
        future_micro_path=Path(future_micro_path),
        candidate_labels=result["candidate_labels"].astype(str),
        expected_n=expected_n,
        expected_classes=expected_classes,
        elite_sites=elite_sites,
        date_deriver=date_deriver,
    )
    if not np.array_equal(result["y_true"], reconstructed_y):
        raise RuntimeError("y_true del NPZ no coincide fila a fila con las etiquetas reconstruidas")
    if not np.array_equal(result["query_date"], reconstructed_dates):
        raise RuntimeError("query_date del NPZ no coincide fila a fila con las fechas reconstruidas")
    result["pcap_uid"] = reconstructed_uids
    return result


def _component_map(payload: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {"MICRO": payload["micro_probs"], "MACRO_XGB": payload["macro_xgb_probs"], "MACRO_LTD": payload["macro_ltd_probs"], "MACRO_FINAL": payload["macro_final_probs"], "HYBRID_FINAL": payload["hybrid_final_probs"]}


def _rank_rows(payload: dict[str, np.ndarray]) -> pd.DataFrame:
    y = payload["y_true"].astype(np.int64)
    rows = []
    for component, probabilities in _component_map(payload).items():
        ranks = rank_values(probabilities, y)
        distribution = rank_distribution(ranks)
        n = len(ranks)
        for bucket, count in distribution.items():
            rows.append({"component": component, "bucket": bucket, "count": count, "fraction": count / max(n, 1)})
        if component == "HYBRID_FINAL":
            rows.append({"component": component, "bucket": "top1_accuracy", "count": int(np.sum(ranks == 1)), "fraction": float(np.mean(ranks == 1))})
            rows.append({"component": component, "bucket": "top5_accuracy", "count": int(np.sum(ranks <= 5)), "fraction": float(np.mean(ranks <= 5))})
            rows.append({"component": component, "bucket": "top5_minus_top1", "count": int(np.sum(ranks <= 5) - np.sum(ranks == 1)), "fraction": float(np.mean(ranks <= 5) - np.mean(ranks == 1))})
            top1_errors = ranks > 1
            rows.append({"component": component, "bucket": "fraction_of_top1_errors_still_in_top5", "count": int(np.sum(top1_errors & (ranks <= 5))), "fraction": float(np.mean(top1_errors & (ranks <= 5)) / max(np.mean(top1_errors), EPS))})
    return pd.DataFrame(rows)


def _capture_group_frame(payload: dict[str, np.ndarray]) -> pd.DataFrame:
    y = payload["y_true"].astype(np.int64)
    labels = payload["candidate_labels"].astype(str)
    probabilities = _component_map(payload)["HYBRID_FINAL"]
    ranks = rank_values(probabilities, y)
    confidence = entropy_margin(probabilities)
    return pd.DataFrame({
        "pcap_uid": payload["pcap_uid"].astype(str),
        "query_date": payload["query_date"],
        "true_site": labels[y],
        "true_rank": ranks,
        "category": [classify_rank_gap(value) for value in ranks],
        "top1_correct": ranks == 1,
        "top5_recoverable": (ranks >= 2) & (ranks <= 5),
        "top5_missed": ranks > 5,
        "top1_confidence": confidence["top1_confidence"],
        "top1_top2_margin": confidence["top1_top2_margin"],
        "entropy": confidence["entropy"],
        "true_class_probability": probabilities[np.arange(len(y)), y],
    })


def _site_rank_metrics(payload: dict[str, np.ndarray]) -> pd.DataFrame:
    y = payload["y_true"].astype(np.int64)
    labels = payload["candidate_labels"].astype(str)
    probabilities = _component_map(payload)["HYBRID_FINAL"]
    ranks = rank_values(probabilities, y)
    rows = []
    for index, label in enumerate(labels):
        mask = y == index
        if not mask.any():
            continue
        site_ranks = ranks[mask]
        accuracy = float(np.mean(site_ranks == 1))
        top5 = float(np.mean(site_ranks <= 5))
        if accuracy >= 0.75:
            category = "high_top1"
        elif top5 - accuracy >= 0.25:
            category = "recoverable_gap"
        else:
            category = "lost_beyond_top5"
        rows.append({"site_label": label, "support": int(mask.sum()), "accuracy": accuracy, "top5_accuracy": top5, "top5_minus_top1": top5 - accuracy, "mrr": float(np.mean(1.0 / site_ranks)), "mean_true_rank": float(site_ranks.mean()), "median_true_rank": float(np.median(site_ranks)), "category_rule": "accuracy>=0.75; else top5-top1>=0.25; else lost", "category": category})
    return pd.DataFrame(rows)


def _size_metrics(payload: dict[str, np.ndarray], micro_path: Path = FUTURE_MICRO_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not micro_path.is_file():
        raise FileNotFoundError(f"No existe la fuente Micro Future-B para el diagnóstico de tamaño: {micro_path}")
    frame = pd.read_csv(micro_path, usecols=["pcap_uid", "site_label", "direction_vector", "size_vector"])
    frame["pcap_uid"] = frame["pcap_uid"].astype(str)
    order = pd.DataFrame({"pcap_uid": payload["pcap_uid"].astype(str), "y_true": payload["y_true"]})
    frame = order.merge(frame, on="pcap_uid", how="left", validate="one_to_one")
    if frame[["direction_vector", "size_vector"]].isna().any().any():
        raise RuntimeError("La fuente Micro Future-B no cubre todas las capturas del NPZ")
    frame["sequence_length"] = frame["direction_vector"].map(lambda value: len(_parse_vector(value)))
    frame["bytes_total"] = frame["size_vector"].map(lambda value: float(np.sum(np.abs(_parse_vector(value)))))
    hybrid_ranks = rank_values(_component_map(payload)["HYBRID_FINAL"], payload["y_true"])
    frame["rank"] = hybrid_ranks
    frame["group"] = [classify_rank_gap(value) for value in hybrid_ranks]
    frame["truncated_at_3000"] = frame["sequence_length"] >= MAX_LEN
    frame["exceeds_3000"] = frame["sequence_length"] > MAX_LEN
    summary = frame.groupby("group", dropna=False).agg(count=("pcap_uid", "size"), median_length=("sequence_length", "median"), p90_length=("sequence_length", lambda values: np.quantile(values, 0.90)), p95_length=("sequence_length", lambda values: np.quantile(values, 0.95)), fraction_ge_3000=("truncated_at_3000", "mean"), fraction_gt_3000=("exceeds_3000", "mean"), median_bytes=("bytes_total", "median")).reset_index()
    per_site = frame.groupby("site_label", dropna=False).agg(support=("pcap_uid", "size"), median_length=("sequence_length", "median"), p90_length=("sequence_length", lambda values: np.quantile(values, 0.90)), p95_length=("sequence_length", lambda values: np.quantile(values, 0.95)), fraction_ge_3000=("truncated_at_3000", "mean"), fraction_gt_3000=("exceeds_3000", "mean"), median_bytes=("bytes_total", "median"), top1_correct=("rank", lambda values: np.mean(values == 1)), top5_recoverable=("rank", lambda values: np.mean((values >= 2) & (values <= 5))), top5_missed=("rank", lambda values: np.mean(values > 5))).reset_index()
    return summary, per_site


def _drift_metrics(payload: dict[str, np.ndarray], future_macro_path: Path = FUTURE_MACRO_PATH, historical_macro_path: Path = HISTORICAL_MACRO_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not future_macro_path.is_file() or not historical_macro_path.is_file():
        raise FileNotFoundError("No están disponibles las fuentes Macro Historical/Future necesarias para BASE-128")
    root = Path(__file__).resolve().parents[2]
    import importlib.util
    macro_module_path = root / "src/features/macro_v2/07A_candidate_conditioned_temporal_encoder.py"
    if str(macro_module_path.parent) not in sys.path:
        sys.path.insert(0, str(macro_module_path.parent))
    spec = importlib.util.spec_from_file_location("p14_macro_features", macro_module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("No se pudo cargar el selector BASE-128 congelado")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    features, _ = module.load_frozen_features()
    hist = pd.read_csv(historical_macro_path, usecols=["site_label"] + list(features))
    future = pd.read_csv(future_macro_path, usecols=["site_label"] + list(features))
    hist["site_label"] = hist["site_label"].astype(str)
    future["site_label"] = future["site_label"].astype(str)
    hist_values = hist[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    medians = hist_values.median().fillna(0.0)
    mad = (hist_values - medians).abs().median().replace(0, np.nan).fillna(1.0)
    hist_z = (hist_values.fillna(medians) - medians) / mad
    future_z = (future[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(medians) - medians) / mad
    hist_centroids = hist_z.assign(site_label=hist["site_label"]).groupby("site_label")[features].median()
    future_centroids = future_z.assign(site_label=future["site_label"]).groupby("site_label")[features].median()
    labels = sorted(set(hist_centroids.index) & set(future_centroids.index))
    site_ranks = rank_values(_component_map(payload)["HYBRID_FINAL"], payload["y_true"])
    y = payload["y_true"].astype(np.int64)
    candidate_labels = payload["candidate_labels"].astype(str)
    rank_by_site = {candidate_labels[index]: site_ranks[y == index] for index in range(len(candidate_labels))}
    rows, vectors = [], {}
    for label in labels:
        h = hist_centroids.loc[label].to_numpy(dtype=float)
        f = future_centroids.loc[label].to_numpy(dtype=float)
        vector = f - h
        vectors[label] = vector
        h_norm, f_norm = np.linalg.norm(h), np.linalg.norm(f)
        cosine = float(np.dot(h, f) / max(h_norm * f_norm, EPS))
        ranks = rank_by_site.get(label, np.array([], dtype=float))
        rows.append({"site_label": label, "robust_standardized_l2_shift": float(np.linalg.norm(vector)), "median_absolute_standardized_shift": float(np.median(np.abs(vector))), "cosine_similarity_historical_future_centroid": cosine, "accuracy": float(np.mean(ranks == 1)) if len(ranks) else np.nan, "top5_accuracy": float(np.mean(ranks <= 5)) if len(ranks) else np.nan, "top5_minus_top1": float(np.mean(ranks <= 5) - np.mean(ranks == 1)) if len(ranks) else np.nan, "mean_rank": float(np.mean(ranks)) if len(ranks) else np.nan})
    metrics = pd.DataFrame(rows)
    pair_rows = []
    for left_index, left in enumerate(labels):
        for right in labels[left_index + 1:]:
            a, b = vectors[left], vectors[right]
            pair_rows.append({"site_a": left, "site_b": right, "drift_cosine_similarity": float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), EPS)), "drift_l2_distance": float(np.linalg.norm(a - b)), "future_centroid_distance": float(np.linalg.norm(future_centroids.loc[left].to_numpy(dtype=float) - future_centroids.loc[right].to_numpy(dtype=float)))})
    pair_frame = pd.DataFrame(pair_rows)
    if not pair_frame.empty:
        pair_frame["nearest_rank_a"] = pair_frame.groupby("site_a")["drift_cosine_similarity"].rank(method="first", ascending=False).astype(int)
        pair_frame["nearest_rank_b"] = pair_frame.groupby("site_b")["drift_cosine_similarity"].rank(method="first", ascending=False).astype(int)
    return metrics, pair_frame


def _rescue(payload: dict[str, np.ndarray]) -> pd.DataFrame:
    y = payload["y_true"].astype(np.int64)
    components = _component_map(payload)
    hybrid_ranks = rank_values(components["HYBRID_FINAL"], y)
    rows = []
    for index in np.flatnonzero(hybrid_ranks > 1):
        row = {"capture_index": int(index), "hybrid_true_rank": int(hybrid_ranks[index])}
        for name, probs in components.items():
            ranks = rank_values(probs[index:index + 1], y[index:index + 1])[0]
            row[f"{name.lower()}_top1"] = bool(ranks <= 1)
            row[f"{name.lower()}_top3"] = bool(ranks <= 3)
            row[f"{name.lower()}_top5"] = bool(ranks <= 5)
        row["rescue_top1_component"] = ",".join(name for name in components if row[f"{name.lower()}_top1"] and name != "HYBRID_FINAL") or "none"
        rows.append(row)
    return pd.DataFrame(rows)


def _correlations(site: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for left, right in (("robust_standardized_l2_shift", "accuracy"), ("robust_standardized_l2_shift", "top5_accuracy"), ("robust_standardized_l2_shift", "top5_minus_top1"), ("robust_standardized_l2_shift", "mean_rank")):
        valid = site[[left, right]].dropna()
        if len(valid) >= 3:
            coefficient = valid[left].rank(method="average").corr(valid[right].rank(method="average"))
        else:
            coefficient = np.nan
        rows.append({"x": left, "y": right, "spearman_rho": float(coefficient), "p_value_descriptive": np.nan, "interpretation": "descriptivo; no implica causalidad ni selección"})
    return pd.DataFrame(rows)


def _pair_correlations(pairs: pd.DataFrame) -> pd.DataFrame:
    if pairs.empty or pairs["confusion_count"].nunique() <= 1:
        rho = np.nan
    else:
        rho = pairs["drift_cosine_similarity"].rank(method="average").corr(pairs["confusion_count"].rank(method="average"))
    return pd.DataFrame([{"x": "drift_cosine_similarity", "y": "confusion_count", "spearman_rho": float(rho), "p_value_descriptive": np.nan, "interpretation": "descriptivo; no implica causalidad ni selección"}])


def _dry_run(paths: dict[str, Path]) -> dict[str, Any]:
    print("14A — DRY-RUN; no entrena, no selecciona y no reejecuta 10B")
    print(f"Fuente NPZ: {paths['scores']}")
    print(f"Fuente Micro para tamaños: {paths['micro']}")
    print(f"Resultados: {paths['output']}")
    print("Componentes: " + ", ".join(COMPONENTS))
    print("Flags: future_b_open=true; selection_allowed=false; training_allowed=false")
    return {"dry_run": True, "future_b_open": True, "selection_allowed": False, "training_allowed": False}


def run_future_rank_gap(config_path: str | Path | None = None, *, dry_run: bool = False) -> dict[str, Any]:
    config_file, config = load_future_config(config_path)
    paths = build_future_paths(config)
    if dry_run:
        return _dry_run(paths)
    payload = _load_npz(paths["scores"])
    paths["output"].mkdir(parents=True, exist_ok=True)
    rank_frame = _rank_rows(payload)
    capture_groups = _capture_group_frame(payload)
    site_frame = _site_rank_metrics(payload)
    size_summary, site_size = _size_metrics(payload, paths["micro"])
    drift_frame, drift_pairs = _drift_metrics(payload, paths["future_macro"], paths["historical_macro"])
    confusion = build_confusion_edges(payload["y_true"], payload["hybrid_final_probs"], payload["candidate_labels"])
    if not drift_pairs.empty:
        confusion_counts = {}
        for row in confusion.itertuples(index=False):
            key = tuple(sorted((str(row.true_site), str(row.predicted_site))))
            confusion_counts[key] = confusion_counts.get(key, 0) + int(row.count)
        drift_pairs["confusion_count"] = [confusion_counts.get(tuple(sorted((row.site_a, row.site_b))), 0) for row in drift_pairs.itertuples(index=False)]
        drift_pairs["confusion_fraction"] = drift_pairs["confusion_count"] / max(float(drift_pairs["confusion_count"].sum()), 1.0)
        size_lookup = site_size.set_index("site_label")["median_length"].to_dict()
        drift_pairs["sequence_length_median_difference"] = [abs(float(size_lookup.get(row.site_a, np.nan)) - float(size_lookup.get(row.site_b, np.nan))) for row in drift_pairs.itertuples(index=False)]
    rescue = _rescue(payload)
    correlations = pd.concat([_correlations(drift_frame), _pair_correlations(drift_pairs)], ignore_index=True)
    outputs_frames = {
        "14A_rank_distribution.csv": rank_frame,
        "14A_per_site_rank_gap.csv": site_frame,
        "14A_capture_groups.csv": capture_groups,
        "14A_sequence_size_summary.csv": size_summary,
        "14A_per_site_size_metrics.csv": site_size,
        "14A_per_site_drift_metrics.csv": drift_frame,
        "14A_drift_similarity_pairs.csv": drift_pairs,
        "14A_confusion_edges.csv": confusion,
        "14A_component_rescue.csv": rescue,
        "14A_correlations.csv": correlations,
    }
    for name, frame in outputs_frames.items():
        frame.to_csv(paths["output"] / name, index=False)
    output_hashes = {name: sha256(paths["output"] / name) for name in outputs_frames}
    root = Path(__file__).resolve().parents[2]
    input_paths = {"config": config_file, "future_scores": paths["scores"], "future_micro": paths["micro"], "future_macro": paths["future_macro"], "historical_macro": paths["historical_macro"]}
    inputs = {name: sha256(path) for name, path in input_paths.items()}
    code_paths = {"future_rank_gap": Path(__file__), "paths": root / "src/utils/paths.py", "phase13_helpers": root / "src/experiments/phase13_battery.py", "base128_selector": root / "src/features/macro_v2/07A_candidate_conditioned_temporal_encoder.py", "future_source_schema": root / "src/models/final/10B_evaluate_future_concept_drift.py"}
    code_hashes = {name: sha256(path) for name, path in code_paths.items()}
    manifest = build_future_manifest(inputs=inputs, code_hashes=code_hashes, config_hash=sha256(config_file), outputs=output_hashes)
    (paths["output"] / "14A_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"rank_distribution": rank_frame, "per_site": site_frame, "manifest": manifest}
