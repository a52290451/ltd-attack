"""Orquestador congelado de la batería P13-B1.

Este módulo mantiene separadas la planificación, la ejecución Historical y el
diagnóstico Future-B. Las funciones matemáticas no dependen de datasets y se
prueban con datos pequeños; los entrenadores históricos solo se cargan al
entrar en una ejecución que no sea dry-run.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.utils.paths import artifact_path, data_path, result_path


ORIGINS = ("ORIGIN14", "ORIGIN28")
WINDOWS = ("NEAR", "MID", "FAR")
BATTERY_MODES = ("EARLY", "UNIFORM", "LATE")
MICRO_SEEDS = (11, 42, 73)
MICRO_CANDIDATES = ("U1", "U3", "T3_S11", "T3_S42", "T3_S73", "T9")
HYBRID_MICRO_CANDIDATES = ("U1", "U3", "T3_S42", "T9")
HYBRID_MACRO_CANDIDATES = ("M0", "M1")
MAX_LEN = 3000
MICRO_EPOCHS = 30
EDGE_WEIGHT_RATIO = 4.0
HYBRID_MICRO_WEIGHT = 0.45
HYBRID_MACRO_WEIGHT = 0.55
EPS = 1e-12
BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_SEED = 20261004

EXPECTED_MICRO_ARCHITECTURE = {
    "max_len": MAX_LEN,
    "d_model": 256,
    "heads": 8,
    "layers": 4,
    "epochs": MICRO_EPOCHS,
    "edge_weight_ratio": EDGE_WEIGHT_RATIO,
}

# Contrato inmutable para reutilizar exclusivamente los caches producidos por
# la ejecución Micro histórica de P13-B1 en el commit original.
LEGACY_CACHE_SOURCE_COMMIT = "7cbf0627334ee6c1be59a56199a5f2abc9bbd62d"
LEGACY_CACHE_SOURCE_FINGERPRINT = "345f49ca1dfeddf93e8fa2ee4bf2b43a8f6990b10b9ae9f045586cd38535eeb8"
LEGACY_CACHE_SOURCE_CODE_FINGERPRINT = "7beffe2de894fb185bd334fca1909e809a23bbb3f43247da29ac87c31d8633a7"
LEGACY_CACHE_SOURCE_CONFIG_SHA256 = "f12bb55d513df8a81344a950a5d8f38e21a7085b03736a3215d05c4e4d8887c6"
LEGACY_CACHE_SOURCE_INPUT_FINGERPRINT = "89f2b1fa344a1cb0e543583e36ee4ce437fb27e2096986078989bb3d5025f9cd"
EXPECTED_EXPERT_IDENTITIES = frozenset((mode, seed) for mode in BATTERY_MODES for seed in MICRO_SEEDS)

OUTPUT_13B = Path(result_path("micro", "MICRO-ROBUSTNESS-PHASE13-BATTERY"))
OUTPUT_13C = Path(result_path("hybrid", "LTD-ROBUSTNESS-PHASE13-BATTERY"))
CACHE_ROOT = Path(artifact_path("experiments", "PHASE13_BATTERY_V1"))

PREDECLARED_GATES = {
    "mechanism_temporal_specific_pass": {
        "primary": "T3_S42_vs_U3",
        "far_macro_f1_positive_both_origins": True,
        "mean_far_macro_f1_gain_min": 0.005,
        "mean_far_accuracy_gain_strictly_positive": True,
        "far_macro_f1_fraction_delta_gt_0_min": 0.90,
        "replications": ["T3_S11_vs_U3", "T3_S73_vs_U3"],
        "replications_descriptive_only": True,
    },
    "generic_ensemble_gain": "U3_vs_U1_report_separately",
}


@dataclass(frozen=True)
class BatteryConfig:
    """Configuración mínima validada de la batería."""

    path: Path
    payload: dict[str, Any]


class CacheState(str, Enum):
    """Estados explícitos de un artefacto cacheado."""

    ABSENT = "ABSENT"
    COMPLETE_COMPATIBLE = "COMPLETE_COMPATIBLE"
    COMPLETE_COMPATIBLE_LEGACY = "COMPLETE_COMPATIBLE_LEGACY"
    COMPLETE_INCOMPATIBLE = "COMPLETE_INCOMPATIBLE"
    PARTIAL = "PARTIAL"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: str | Path) -> str:
    """Calcula SHA256 de un archivo sin asumir una ruta del sistema."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        raise RuntimeError("PyYAML es obligatorio para validar la configuración científica")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"La configuración debe ser un mapping: {path}")
    return loaded


def load_battery_config(path: str | Path | None = None) -> BatteryConfig:
    config_path = Path(path) if path else Path(__file__).resolve().parents[2] / "configs/experiments/PHASE13_BATTERY_V1.yaml"
    payload = _load_yaml(config_path)
    validate_phase13_config(payload)
    return BatteryConfig(config_path, payload)


def validate_phase13_config(payload: dict[str, Any]) -> dict[str, Any]:
    """Valida que la configuración sea la fuente de verdad congelada."""
    required_origins = ["ORIGIN14", "ORIGIN28"]
    required_windows = ["NEAR", "MID", "FAR"]
    required_modes = ["EARLY", "UNIFORM", "LATE"]
    required_seeds = [11, 42, 73]
    if payload.get("origins") != required_origins:
        raise ValueError("guardrail: origins deben ser ORIGIN14 y ORIGIN28")
    if payload.get("windows") != required_windows:
        raise ValueError("guardrail: windows deben ser NEAR, MID y FAR")
    micro = payload.get("micro", {})
    if micro.get("architecture") != "MICRO-FINAL" or micro.get("max_len") != 3000:
        raise ValueError("guardrail: arquitectura MICRO-FINAL/MAX_LEN congelada")
    if micro.get("d_model") != 256 or micro.get("heads") != 8 or micro.get("layers") != 4:
        raise ValueError("guardrail: dimensiones MICRO-FINAL congeladas")
    if micro.get("epochs") != 30 or micro.get("seeds") != required_seeds or micro.get("temporal_modes") != required_modes:
        raise ValueError("guardrail: epochs, seeds o modos temporales incompatibles")
    if float(micro.get("edge_weight_ratio", 0)) != EDGE_WEIGHT_RATIO:
        raise ValueError("guardrail: EDGE_WEIGHT_RATIO debe ser 4.0")
    if payload.get("micro_candidates") != list(MICRO_CANDIDATES):
        raise ValueError("guardrail: matriz de candidatos Micro incompatible")
    if payload.get("hybrid_micro_candidates") != list(HYBRID_MICRO_CANDIDATES):
        raise ValueError("guardrail: candidatos Micro 13C incompatibles")
    if payload.get("macro_candidates") != list(HYBRID_MACRO_CANDIDATES):
        raise ValueError("guardrail: candidatos Macro 13C incompatibles")
    expected_ensembles = {
        "U1": ["UNIFORM/42"],
        "U3": ["UNIFORM/11", "UNIFORM/42", "UNIFORM/73"],
        "T3_S11": ["EARLY/11", "UNIFORM/11", "LATE/11"],
        "T3_S42": ["EARLY/42", "UNIFORM/42", "LATE/42"],
        "T3_S73": ["EARLY/73", "UNIFORM/73", "LATE/73"],
        "T9": ["EARLY/11", "UNIFORM/11", "LATE/11", "EARLY/42", "UNIFORM/42", "LATE/42", "EARLY/73", "UNIFORM/73", "LATE/73"],
    }
    if payload.get("ensemble_definitions") != expected_ensembles:
        raise ValueError("guardrail: definiciones de ensemble incompatibles")
    weights = payload.get("hybrid_weights", {})
    if float(weights.get("micro", 0)) != HYBRID_MICRO_WEIGHT or float(weights.get("macro", 0)) != HYBRID_MACRO_WEIGHT:
        raise ValueError("guardrail: pesos Hybrid deben ser 0.45/0.55")
    bootstrap = payload.get("bootstrap", {})
    if bootstrap.get("unit") != "day_block" or int(bootstrap.get("iterations", 0)) != BOOTSTRAP_ITERATIONS:
        raise ValueError("guardrail: bootstrap day_block/iterations incompatible")
    if payload.get("metrics") != ["accuracy", "macro_f1", "top5_accuracy", "mrr", "mean_true_rank"]:
        raise ValueError("guardrail: métricas co-principales o métricas de ranking incompatibles")
    guardrails = payload.get("guardrails", {})
    if guardrails.get("historical_only") is not True or guardrails.get("future_b_scores_used") is not False or guardrails.get("adaptive_selection_during_run") is not False or guardrails.get("inference_updates") is not False or guardrails.get("dataset_c_opened") is not False:
        raise ValueError("guardrail: política científica Historical incompatible")
    if payload.get("paths") != {"result_13b": ["micro", "MICRO-ROBUSTNESS-PHASE13-BATTERY"], "result_13c": ["hybrid", "LTD-ROBUSTNESS-PHASE13-BATTERY"], "cache": ["experiments", "PHASE13_BATTERY_V1"]}:
        raise ValueError("guardrail: rutas lógicas P13 incompatibles")
    return payload


def build_macro_candidate_definitions() -> dict[str, str]:
    return {"M0": "UNIFORM_XGB+RECENT5_LTD", "M1": "TEMPORAL_SYMMETRIC3_XGB+MULTISCALE5_LTD"}


def _ensemble_from_config(lookup: dict[tuple[str, int], np.ndarray], payload: dict[str, Any]) -> dict[str, np.ndarray]:
    candidates = {}
    for name, references in payload["ensemble_definitions"].items():
        experts = []
        for reference in references:
            mode, seed = str(reference).split("/")
            experts.append(lookup[(mode, int(seed))])
        candidates[name] = geometric_probability_mean(experts)
    return candidates


def geometric_probability_mean(probabilities: Iterable[np.ndarray]) -> np.ndarray:
    """Media geométrica igual ponderada con normalización fila a fila."""
    arrays = [np.asarray(value, dtype=np.float64) for value in probabilities]
    if not arrays:
        raise ValueError("Se necesita al menos una matriz de probabilidades")
    shape = arrays[0].shape
    if len(shape) != 2 or any(value.shape != shape for value in arrays):
        raise ValueError("Todas las matrices deben compartir shape (n, clases)")
    logp = np.mean([np.log(np.clip(value, EPS, 1.0)) for value in arrays], axis=0)
    logp -= logp.max(axis=1, keepdims=True)
    result = np.exp(logp)
    return result / np.clip(result.sum(axis=1, keepdims=True), EPS, None)


def temporal_weights(dates: Iterable[Any], mode: str, ratio: float = EDGE_WEIGHT_RATIO) -> np.ndarray:
    """Reproduce los pesos EARLY/UNIFORM/LATE de 13A/11H."""
    if ratio <= 0:
        raise ValueError("El ratio temporal debe ser positivo")
    values = pd.to_datetime(pd.Series(list(dates)))
    unique = np.array(sorted(values.dropna().unique()))
    if values.isna().any() or len(unique) == 0:
        raise ValueError("Las fechas de entrenamiento no pueden estar vacías")
    rank = {pd.Timestamp(value): index for index, value in enumerate(unique)}
    if len(unique) == 1:
        t = np.zeros(len(values), dtype=float)
    else:
        t = np.asarray([rank[pd.Timestamp(value)] / (len(unique) - 1) for value in values], dtype=float)
    gamma = np.log(float(ratio))
    if mode == "UNIFORM":
        weights = np.ones_like(t)
    elif mode == "EARLY":
        weights = np.exp(-gamma * t)
    elif mode == "LATE":
        weights = np.exp(-gamma * (1.0 - t))
    else:
        raise ValueError(f"Modo temporal desconocido: {mode}")
    return (weights / weights.mean()).astype(np.float64)


def rank_values(probabilities: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities)
    y_true = np.asarray(y_true, dtype=np.int64)
    if probabilities.ndim != 2 or len(probabilities) != len(y_true):
        raise ValueError("Probabilidades y etiquetas no están alineadas")
    ordering = np.argsort(-probabilities, axis=1, kind="stable")
    positions = np.argmax(ordering == y_true[:, None], axis=1)
    return positions.astype(np.int64) + 1


def metric_values(probabilities: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    """Calcula las cinco métricas declaradas para una matriz congelada."""
    y_true = np.asarray(y_true, dtype=np.int64)
    ranks = rank_values(probabilities, y_true)
    prediction = np.argmax(probabilities, axis=1)
    n_classes = probabilities.shape[1]
    f1_values = []
    for label in range(n_classes):
        true_positive = np.sum((y_true == label) & (prediction == label))
        false_positive = np.sum((y_true != label) & (prediction == label))
        false_negative = np.sum((y_true == label) & (prediction != label))
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1_values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {
        "n": int(len(y_true)),
        "accuracy": float(np.mean(prediction == y_true)),
        "macro_f1": float(np.mean(f1_values)),
        "top5_accuracy": float(np.mean(ranks <= 5)),
        "mrr": float(np.mean(1.0 / ranks)),
        "mean_true_rank": float(np.mean(ranks)),
    }


def rank_gap_group(rank: int) -> str:
    if int(rank) == 1:
        return "TOP1_CORRECT"
    if 2 <= int(rank) <= 5:
        return "TOP5_RECOVERABLE"
    return "TOP5_MISSED"


def entropy_margin(probabilities: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(probabilities, dtype=np.float64)
    ordering = np.argsort(-values, axis=1, kind="stable")
    top1 = values[np.arange(len(values)), ordering[:, 0]]
    top2 = values[np.arange(len(values)), ordering[:, 1]] if values.shape[1] > 1 else np.zeros(len(values))
    entropy = -np.sum(np.clip(values, EPS, 1.0) * np.log(np.clip(values, EPS, 1.0)), axis=1)
    return {
        "top1_confidence": top1,
        "top1_top2_margin": top1 - top2,
        "entropy": entropy,
    }


def _as_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _as_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_json(item) for item in value]
    return value


def build_experiment_matrix(config: BatteryConfig | dict[str, Any] | None = None) -> dict[str, Any]:
    payload = config.payload if isinstance(config, BatteryConfig) else config
    origins = tuple(payload.get("origins", ORIGINS)) if payload else ORIGINS
    modes = tuple(payload.get("micro", {}).get("temporal_modes", BATTERY_MODES)) if payload else BATTERY_MODES
    seeds = tuple(payload.get("micro", {}).get("seeds", MICRO_SEEDS)) if payload else MICRO_SEEDS
    windows = tuple(payload.get("windows", WINDOWS)) if payload else WINDOWS
    micro_candidates = list(payload.get("micro_candidates", MICRO_CANDIDATES)) if payload else list(MICRO_CANDIDATES)
    hybrid_micro = list(payload.get("hybrid_micro_candidates", HYBRID_MICRO_CANDIDATES)) if payload else list(HYBRID_MICRO_CANDIDATES)
    macro_candidates = list(payload.get("macro_candidates", HYBRID_MACRO_CANDIDATES)) if payload else list(HYBRID_MACRO_CANDIDATES)
    training = [
        {"origin": origin, "temporal_mode": mode, "seed": seed, "id": f"{origin}/{mode}/{seed}"}
        for origin in origins
        for mode in modes
        for seed in seeds
    ]
    return {
        "origins": list(origins),
        "windows": list(windows),
        "micro_training": training,
        "micro_candidates": micro_candidates,
        "hybrid_micro_candidates": hybrid_micro,
        "hybrid_macro_candidates": macro_candidates,
        "hybrid_factorial": [f"{micro}x{macro}" for micro in hybrid_micro for macro in macro_candidates],
    }


def build_manifest(
    *, stage: str, research_question: str, candidate_matrix: dict[str, Any], inputs: dict[str, Any],
    code_hashes: dict[str, Any], config_hash: str, outputs: dict[str, Any], scientific_status: str = "PREDECLARED_CHECKPOINT",
) -> dict[str, Any]:
    return {
        "stage": stage,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "scientific_status": scientific_status,
        "research_question": research_question,
        "data_policy": {
            "historical_only": True,
            "future_b_scores_used": False,
            "internal_test_used_for_decision": False,
            "dataset_c_opened": False,
            "continual_learning": False,
            "test_time_adaptation": False,
            "training_inference_updates": False,
            "selection_allowed": False,
            "adaptive_decisions_during_run": False,
        },
        "architecture": {
            "micro": {"max_len": MAX_LEN, "d_model": 256, "heads": 8, "layers": 4, "epochs": MICRO_EPOCHS},
            "hybrid_weights": {"micro": HYBRID_MICRO_WEIGHT, "macro": HYBRID_MACRO_WEIGHT},
        },
        "candidate_matrix": _as_json(candidate_matrix),
        "metrics": ["accuracy", "macro_f1", "top5_accuracy", "mrr", "mean_true_rank"],
        "predeclared_gates": _as_json(PREDECLARED_GATES),
        "inputs_sha256": _as_json(inputs),
        "code_sha256": _as_json(code_hashes),
        "config_sha256": config_hash,
        "outputs_sha256": _as_json(outputs),
    }


def _git_commit() -> str:
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN"


def _config_hash(config: BatteryConfig) -> str:
    return _sha256_bytes(json.dumps(_as_json(config.payload), sort_keys=True).encode("utf-8"))


def _load_module(path: str | Path, name: str):
    module_path = Path(path)
    parent = str(module_path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"No se pudo cargar el módulo: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metrics_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["origin", "window", "candidate"])


def _bootstrap_metric(y: np.ndarray, dates: np.ndarray, first: np.ndarray, second: np.ndarray, metric: str, rng: np.random.Generator) -> np.ndarray:
    unique_dates = np.array(sorted(pd.unique(pd.to_datetime(dates))))
    if len(unique_dates) == 0:
        return np.zeros(0, dtype=float)
    values = []
    date_values = pd.to_datetime(dates)
    for _ in range(BOOTSTRAP_ITERATIONS):
        sampled = rng.choice(unique_dates, size=len(unique_dates), replace=True)
        indices = np.concatenate([np.flatnonzero(date_values == day) for day in sampled])
        left = metric_values(first[indices], y[indices])[metric]
        right = metric_values(second[indices], y[indices])[metric]
        values.append(right - left)
    return np.asarray(values, dtype=float)


def bootstrap_delta(y: np.ndarray, dates: np.ndarray, reference: np.ndarray, candidate: np.ndarray, seed_offset: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(BOOTSTRAP_SEED + int(seed_offset))
    return {metric: _bootstrap_metric(y, dates, reference, candidate, metric, rng) for metric in ("accuracy", "macro_f1", "top5_accuracy", "mrr")}


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_as_json(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _artifact_fingerprint(config_hash: str, inputs: dict[str, str], code_hashes: dict[str, str]) -> str:
    return _sha256_bytes(json.dumps({"config": config_hash, "inputs": inputs, "code": code_hashes}, sort_keys=True).encode("utf-8"))


def classify_cache_state(expected_paths: Iterable[str | Path], manifest_path: str | Path, expected_fingerprint: str) -> CacheState:
    """Clasifica un conjunto sin entrenar ni reparar artefactos."""
    paths = [Path(path) for path in expected_paths]
    present = [path.exists() for path in paths]
    if not any(present):
        return CacheState.ABSENT
    if not all(present):
        return CacheState.PARTIAL
    try:
        payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return CacheState.COMPLETE_INCOMPATIBLE
    return CacheState.COMPLETE_COMPATIBLE if payload.get("fingerprint") == expected_fingerprint else CacheState.COMPLETE_INCOMPATIBLE


def _validate_cache(manifest_path: Path, expected_fingerprint: str) -> dict[str, Any]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Falta manifest de cache: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    observed = payload.get("fingerprint")
    if observed != expected_fingerprint:
        raise RuntimeError(f"Cache incompatible: {manifest_path}; expected={expected_fingerprint}; observed={observed}")
    return payload


def _array_equal(left: Any, right: Any) -> bool:
    return np.array_equal(np.asarray(left).astype(str), np.asarray(right).astype(str))


def validate_expert_prediction_alignment(records: list[dict[str, Any]]) -> None:
    """Valida captures comunes y el producto completo de expertos Micro."""
    if not records:
        raise RuntimeError("No hay artefactos para validar alineación")
    reference = records[0]
    for field in ("origin", "window"):
        if field not in reference:
            raise RuntimeError(f"Fallo de identidad en alineación: falta {field}")
    for index, current in enumerate(records):
        for field in ("origin", "window"):
            if current.get(field) != reference.get(field):
                raise RuntimeError(f"Fallo de identidad en alineación del experto {index}: {field}")
        mode = current.get("temporal_mode")
        seed = current.get("seed")
        if mode not in BATTERY_MODES:
            raise RuntimeError(f"Fallo de identidad en alineación del experto {index}: mode inesperado")
        if seed not in MICRO_SEEDS:
            raise RuntimeError(f"Fallo de identidad en alineación del experto {index}: seed inesperado")
        for field in ("pcap_uid", "query_date", "y_true", "candidate_labels"):
            if not _array_equal(current.get(field), reference.get(field)):
                raise RuntimeError(f"Fallo de alineación del experto {index}: {field}")
        probs = np.asarray(current.get("probs"))
        labels = np.asarray(current.get("candidate_labels"))
        reference_probs = np.asarray(reference.get("probs"))
        if probs.ndim != 2 or reference_probs.ndim != 2 or probs.shape != reference_probs.shape:
            raise RuntimeError(f"Fallo de alineación del experto {index}: shape de probabilidades")
        if len(probs) != len(np.asarray(reference["pcap_uid"])) or probs.shape[1] != len(labels):
            raise RuntimeError(f"Fallo de alineación del experto {index}: número de filas/clases")

    identities = [(record.get("temporal_mode"), record.get("seed")) for record in records]
    seen = set(identities)
    if len(seen) != len(identities):
        raise RuntimeError("Fallo de identidad en alineación: experto duplicado")
    missing = sorted(EXPECTED_EXPERT_IDENTITIES - seen)
    unexpected = sorted(seen - EXPECTED_EXPERT_IDENTITIES)
    if missing:
        raise RuntimeError(f"Fallo de identidad en alineación: experto faltante {missing[0]}")
    if unexpected:
        raise RuntimeError(f"Fallo de identidad en alineación: experto inesperado {unexpected[0]}")


def validate_hybrid_alignment(micro: dict[str, Any], macro: dict[str, Any]) -> None:
    """Exige identidad exacta de captura y clases antes de fusionar Micro/Macro."""
    for field in ("pcap_uid", "query_date", "y_true", "candidate_labels"):
        if not _array_equal(micro.get(field), macro.get(field)):
            raise RuntimeError(f"Micro/Macro desalineados: {field}")
    for field in ("origin", "window"):
        if micro.get(field) != macro.get(field):
            raise RuntimeError(f"Micro/Macro desalineados: identidad {field}")


def _phase13_code_paths(root: Path) -> dict[str, Path]:
    return {
        "phase13": root / "src/experiments/phase13_battery.py",
        "micro08b": root / "src/models/micro/08B_clean_temporal_micro_baseline.py",
        "micro13a": root / "src/models/micro/13A_frozen_temporal_micro_ensemble.py",
        "refit10a": root / "src/models/final/10A_refit_full_historical.py",
        "paths": root / "src/utils/paths.py",
        "common": root / "src/features/macro_v2/_common.py",
        "macro07a": root / "src/features/macro_v2/07A_candidate_conditioned_temporal_encoder.py",
        "macro11b": root / "src/features/macro_v2/11B_multiscale_longitudinal_memory.py",
        "macro11h": root / "src/features/macro_v2/11H_frozen_temporal_environment_ensemble.py",
        "macro11f": root / "src/features/macro_v2/11F_oracle_memory_refresh.py",
        "macro11l": root / "src/features/macro_v2/11L_factorial_frozen_combination.py",
        "hybrid12a": root / "src/models/hybrid/12A_frozen_robust_hybrid_integration.py",
    }


def _hash_paths(paths: dict[str, Path]) -> dict[str, str]:
    return {name: sha256(path) for name, path in paths.items()}


def _fingerprint_digest(values: dict[str, str]) -> str:
    return _sha256_bytes(json.dumps(values, sort_keys=True).encode("utf-8"))


def _expert_expected_paths(cache_dir: Path) -> list[Path]:
    return [cache_dir / "manifest.json", cache_dir / "model.pt", cache_dir / "scaler.json"] + [cache_dir / f"{window}.npz" for window in WINDOWS]


def _macro_expected_paths(origin: str) -> list[Path]:
    cache_path = CACHE_ROOT / "macro" / f"{origin}.npz"
    return [cache_path, cache_path.with_suffix(".json")]


def _factorial_expected_paths() -> list[Path]:
    return [OUTPUT_13C / name for name in ("13C_factorial_summary.csv", "13C_factorial_effects.csv", "13C_pairwise_deltas.csv", "13C_day_block_bootstrap.csv", "13C_rank_gap_historical.csv", "13C_manifest.json")]


def _macro_provenance(config: BatteryConfig, context: dict[str, Any], origin: str) -> tuple[str, dict[str, str], dict[str, str]]:
    root = Path(__file__).resolve().parents[2]
    input_hashes = _hash_paths({"historical": Path(context["historical_path"]), "config": config.path})
    code_paths = {key: _phase13_code_paths(root)[key] for key in ("phase13", "paths", "common", "macro07a", "macro11b", "macro11f", "macro11h", "macro11l", "hybrid12a")}
    code_hashes = _hash_paths(code_paths)
    return _artifact_fingerprint(_config_hash(config), input_hashes, code_hashes), input_hashes, code_hashes


def _factorial_fingerprint(config: BatteryConfig) -> tuple[str, dict[str, str], dict[str, str]]:
    root = Path(__file__).resolve().parents[2]
    input_hashes = {"13B_manifest": sha256(OUTPUT_13B / "13B_manifest.json"), "config": sha256(config.path), "micro_source": sha256(data_path("historical", "CLEAN_final_vectors_sites.csv")), "historical_source": sha256(data_path("historical", "CLEAN_final_features_sites.csv"))}
    code_paths = _phase13_code_paths(root)
    code_hashes = _hash_paths(code_paths)
    return _artifact_fingerprint(_config_hash(config), input_hashes, code_hashes), input_hashes, code_hashes


def _raise_cache_state(cache_dir: Path, state: CacheState, expected_paths: list[Path]) -> None:
    if state == CacheState.PARTIAL:
        present = [path.name for path in expected_paths if path.exists()]
        missing = [path.name for path in expected_paths if not path.exists()]
        raise RuntimeError(f"Cache PARTIAL en {cache_dir}; presentes={present}; faltantes={missing}")
    if state == CacheState.COMPLETE_INCOMPATIBLE:
        raise RuntimeError(f"Cache COMPLETE_INCOMPATIBLE en {cache_dir}; no se sobreescribe")


def validate_file_sha256(path: str | Path, expected: str, label: str) -> str:
    observed = sha256(path)
    if observed != expected:
        raise RuntimeError(f"Provenance incompatible: SHA256 {label} en {path}")
    return observed


def _load_checkpoint_metadata(path: Path) -> dict[str, Any]:
    try:
        import torch
    except ModuleNotFoundError:
        # Permite auditar fixtures sintéticos sin instalar el runtime de
        # entrenamiento. En Perseo, donde existe Torch, siempre se usa el
        # formato real producido por torch.save.
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Checkpoint ilegible: {path}")
    return payload


def _save_prediction(path: Path, *, probs: np.ndarray, pcap_uid: np.ndarray, query_date: np.ndarray, y: np.ndarray, origin: str, window: str, temporal_mode: str, seed: int, candidate_labels: np.ndarray, fingerprint: str, checkpoint_sha256: str, scaler_sha256: str, config_sha256: str, input_fingerprint: str, code_fingerprint: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, probs=probs.astype(np.float32), pcap_uid=np.asarray(pcap_uid).astype(str), query_date=np.asarray(query_date).astype("datetime64[D]"), y_true=y.astype(np.int64), origin=np.asarray(origin), window=np.asarray(window), temporal_mode=np.asarray(temporal_mode), seed=np.asarray(seed), candidate_labels=np.asarray(candidate_labels).astype(str), fingerprint=np.asarray(fingerprint), checkpoint_sha256=np.asarray(checkpoint_sha256), scaler_sha256=np.asarray(scaler_sha256), config_sha256=np.asarray(config_sha256), input_fingerprint=np.asarray(input_fingerprint), code_fingerprint=np.asarray(code_fingerprint))


def _load_prediction(path: Path, fingerprint: str, expected: dict[str, Any] | None = None) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        observed = str(payload["fingerprint"].item())
        if observed != fingerprint:
            raise RuntimeError(f"Predicción incompatible: {path}")
        required = {"probs", "pcap_uid", "query_date", "y_true", "origin", "window", "temporal_mode", "seed", "candidate_labels", "fingerprint", "checkpoint_sha256", "scaler_sha256", "config_sha256", "input_fingerprint", "code_fingerprint"}
        missing = sorted(required - set(payload.files))
        if missing:
            raise RuntimeError(f"Predicción incompatible: faltan campos {missing} en {path}")
        result = {key: payload[key] for key in required}
    if expected:
        for key in ("origin", "window", "temporal_mode", "seed"):
            if key in expected and result[key].item() != expected[key]:
                raise RuntimeError(f"Predicción incompatible {path}: identidad {key}")
        for key in ("pcap_uid", "query_date", "y_true", "candidate_labels"):
            if key in expected and not _array_equal(result[key], expected[key]):
                raise RuntimeError(f"Predicción incompatible {path}: alineación {key}")
    result["probs"] = result["probs"].astype(np.float64)
    result["y_true"] = result["y_true"].astype(np.int64)
    return result


def _validate_prediction_payload(path: Path, fingerprint: str, expected: dict[str, Any], provenance: dict[str, Any], expected_output_sha256: str) -> dict[str, Any]:
    if sha256(path) != expected_output_sha256:
        raise RuntimeError(f"Predicción incompatible {path}: SHA256 NPZ")
    loaded = _load_prediction(path, fingerprint, expected=expected)
    probs = np.asarray(loaded["probs"])
    pcap_uid = np.asarray(loaded["pcap_uid"])
    query_date = np.asarray(loaded["query_date"])
    y_true = np.asarray(loaded["y_true"])
    candidate_labels = np.asarray(loaded["candidate_labels"])
    if probs.ndim != 2 or candidate_labels.ndim != 1:
        raise RuntimeError(f"Predicción incompatible {path}: shape de metadata")
    if len(probs) != len(pcap_uid) or len(probs) != len(query_date) or len(probs) != len(y_true) or probs.shape[1] != len(candidate_labels):
        raise RuntimeError(f"Predicción incompatible {path}: filas/clases")
    for field in ("checkpoint_sha256", "scaler_sha256", "config_sha256", "input_fingerprint", "code_fingerprint"):
        if str(loaded[field].item()) != str(provenance[field]):
            raise RuntimeError(f"Predicción incompatible {path}: provenance {field}")
    return loaded


def _validate_legacy_cache(cache_dir: Path, origin: str, mode: str, seed: int, expected_windows: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """Valida un cache histórico completo sin escribir ni reparar artefactos."""
    manifest_path = cache_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        scaler = json.loads((cache_dir / "scaler.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cache legacy ilegible en {cache_dir}") from exc

    if manifest.get("fingerprint") != LEGACY_CACHE_SOURCE_FINGERPRINT:
        raise RuntimeError(f"Cache legacy incompatible {cache_dir}: fingerprint")
    if manifest.get("training", {}).get("git_commit") != LEGACY_CACHE_SOURCE_COMMIT:
        raise RuntimeError(f"Cache legacy incompatible {cache_dir}: training.git_commit")
    provenance = manifest.get("provenance", {})
    expected_provenance = {
        "config_sha256": LEGACY_CACHE_SOURCE_CONFIG_SHA256,
        "input_fingerprint": LEGACY_CACHE_SOURCE_INPUT_FINGERPRINT,
        "code_fingerprint": LEGACY_CACHE_SOURCE_CODE_FINGERPRINT,
    }
    for field, expected in expected_provenance.items():
        if provenance.get(field) != expected:
            raise RuntimeError(f"Cache legacy incompatible {cache_dir}: {field}")
    identity = {"origin": origin, "temporal_mode": mode, "seed": seed}
    if manifest.get("identity") != identity:
        raise RuntimeError(f"Cache legacy incompatible {cache_dir}: identidad manifest")
    if manifest.get("architecture") != EXPECTED_MICRO_ARCHITECTURE:
        raise RuntimeError(f"Cache legacy incompatible {cache_dir}: arquitectura manifest")
    if scaler.get("origin") != origin or scaler.get("temporal_mode") != mode or scaler.get("seed") != seed:
        raise RuntimeError(f"Cache legacy incompatible {cache_dir}: identidad scaler")
    if scaler.get("architecture") != EXPECTED_MICRO_ARCHITECTURE:
        raise RuntimeError(f"Cache legacy incompatible {cache_dir}: arquitectura scaler")
    for field, expected in expected_provenance.items():
        if scaler.get(field) != expected:
            raise RuntimeError(f"Cache legacy incompatible {cache_dir}: metadata scaler {field}")

    scaler_sha256 = provenance.get("scaler_sha256")
    checkpoint_sha256 = provenance.get("checkpoint_sha256")
    if not isinstance(scaler_sha256, str) or not isinstance(checkpoint_sha256, str):
        raise RuntimeError(f"Cache legacy incompatible {cache_dir}: hashes de artefactos ausentes")
    validate_file_sha256(cache_dir / "scaler.json", scaler_sha256, "scaler")
    validate_file_sha256(cache_dir / "model.pt", checkpoint_sha256, "checkpoint")
    try:
        checkpoint = _load_checkpoint_metadata(cache_dir / "model.pt")
    except Exception as exc:
        raise RuntimeError(f"Cache legacy incompatible {cache_dir}: checkpoint ilegible") from exc
    checkpoint_expected = {**identity, "fingerprint": LEGACY_CACHE_SOURCE_FINGERPRINT, **expected_provenance, "scaler_sha256": scaler_sha256}
    for field, expected in checkpoint_expected.items():
        if checkpoint.get(field) != expected:
            raise RuntimeError(f"Cache legacy incompatible {cache_dir}: metadata checkpoint {field}")
    if checkpoint.get("architecture") != EXPECTED_MICRO_ARCHITECTURE:
        raise RuntimeError(f"Cache legacy incompatible {cache_dir}: arquitectura checkpoint")

    outputs_sha256 = manifest.get("outputs_sha256", {})
    for window in WINDOWS:
        path = cache_dir / f"{window}.npz"
        expected = {**identity, "window": window}
        if expected_windows and window in expected_windows:
            expected.update(expected_windows[window])
        loaded = _validate_prediction_payload(path, LEGACY_CACHE_SOURCE_FINGERPRINT, expected, {**expected_provenance, "checkpoint_sha256": checkpoint_sha256, "scaler_sha256": scaler_sha256}, outputs_sha256.get(window, ""))
        if loaded["origin"].item() != origin or loaded["window"].item() != window or loaded["temporal_mode"].item() != mode or loaded["seed"].item() != seed:
            raise RuntimeError(f"Cache legacy incompatible {path}: metadata identity")
    return manifest


def classify_expert_cache_state(cache_dir: Path, origin: str, mode: str, seed: int, expected_fingerprint: str, expected_windows: dict[str, dict[str, Any]] | None = None) -> CacheState:
    state = classify_cache_state(_expert_expected_paths(cache_dir), cache_dir / "manifest.json", expected_fingerprint)
    if state == CacheState.COMPLETE_INCOMPATIBLE:
        try:
            _validate_legacy_cache(cache_dir, origin, mode, seed, expected_windows=expected_windows)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError):
            return state
        return CacheState.COMPLETE_COMPATIBLE_LEGACY
    if state != CacheState.COMPLETE_COMPATIBLE:
        return state
    try:
        for window in WINDOWS:
            expected = {"origin": origin, "window": window, "temporal_mode": mode, "seed": seed}
            if expected_windows and window in expected_windows:
                expected.update(expected_windows[window])
            _load_prediction(cache_dir / f"{window}.npz", expected_fingerprint, expected=expected)
    except (OSError, RuntimeError, ValueError, KeyError, TypeError):
        return CacheState.COMPLETE_INCOMPATIBLE
    return state


def _planned_phase13_fingerprint(config: BatteryConfig) -> str | None:
    root = Path(__file__).resolve().parents[2]
    input_paths = {"micro_source": Path(data_path("historical", "CLEAN_final_vectors_sites.csv")), "historical_source": Path(data_path("historical", "CLEAN_final_features_sites.csv")), "config": config.path}
    if not all(path.is_file() for path in input_paths.values()):
        return None
    return _artifact_fingerprint(_config_hash(config), _hash_paths(input_paths), _hash_paths(_phase13_code_paths(root)))


def _print_plan(only: str | None, config: BatteryConfig) -> dict[str, Any]:
    matrix = build_experiment_matrix(config)
    reusable = []
    trainable = []
    states: dict[str, str] = {}
    planned_fingerprint = _planned_phase13_fingerprint(config)
    metadata_context = None
    if planned_fingerprint is not None:
        metadata_context = _load_micro_context()
    print("P13-B1 — DRY-RUN; no se entrenará ningún modelo")
    print("Configuración: PHASE13_BATTERY_V1.yaml")
    print("Origins/windows: ORIGIN14, ORIGIN28 × NEAR, MID, FAR")
    print("Micro training (18):")
    for item in matrix["micro_training"]:
        cache_dir = CACHE_ROOT / item["origin"] / item["temporal_mode"] / str(item["seed"])
        expected_windows = None
        if metadata_context is not None:
            expected_windows = {window: _window_metadata(metadata_context, item["origin"], window) for window in tuple(config.payload["windows"])}
        state = classify_expert_cache_state(cache_dir, item["origin"], item["temporal_mode"], int(item["seed"]), planned_fingerprint or "__INPUTS_UNAVAILABLE__", expected_windows=expected_windows)
        states[item["id"]] = state.value
        if state == CacheState.COMPLETE_COMPATIBLE:
            reusable.append(item["id"])
            state_text = "COMPLETE_COMPATIBLE — reutilizable bajo --resume"
        elif state == CacheState.COMPLETE_COMPATIBLE_LEGACY:
            reusable.append(item["id"])
            state_text = "COMPLETE_COMPATIBLE_LEGACY — reutilizable bajo --resume"
        elif state == CacheState.ABSENT:
            trainable.append(item["id"])
            state_text = "ABSENT — requiere entrenamiento"
        else:
            state_text = f"{state.value} — no se reutiliza; la ejecución fallará explícitamente"
        print(f"  - {item['id']} — {state_text}")
    print("Ensembles derivados: " + ", ".join(matrix["micro_candidates"]))
    print("13C factorial: " + ", ".join(matrix["hybrid_factorial"]))
    macro_states = {}
    macro_context = {"historical_path": data_path("historical", "CLEAN_final_features_sites.csv")}
    for origin in tuple(config.payload["origins"]):
        if Path(macro_context["historical_path"]).is_file():
            macro_fingerprint, _, _ = _macro_provenance(config, macro_context, origin)
        else:
            macro_fingerprint = "__INPUTS_UNAVAILABLE__"
        macro_states[origin] = classify_cache_state(_macro_expected_paths(origin), _macro_expected_paths(origin)[1], macro_fingerprint).value
    print("Estados Macro: " + ", ".join(f"{origin}={state}" for origin, state in macro_states.items()))
    print("Macro reutilizable bajo --resume: " + (", ".join(origin for origin, state in macro_states.items() if state == CacheState.COMPLETE_COMPATIBLE.value) or "ninguno detectado"))
    print("Micro reutilizable detectado: " + (", ".join(reusable) if reusable else "ninguno detectado"))
    print(f"Entrenamientos Micro requeridos: {len(trainable) if only in (None, '13B') else 0}")
    print(f"Resultados 13B: {OUTPUT_13B}")
    print(f"Resultados 13C: {OUTPUT_13C}")
    print("Future diagnostics: no training; carril separado y POSTHOC_DIAGNOSTIC_ONLY")
    factorial_state = classify_cache_state(_factorial_expected_paths(), OUTPUT_13C / "13C_manifest.json", "__INPUTS_UNAVAILABLE__")
    print(f"Estados outputs 13C: {factorial_state.value}")
    return {"dry_run": True, "only": only, "matrix": matrix, "reusable": reusable, "trainable": trainable if only in (None, "13B") else [], "cache_states": states, "macro_states": macro_states, "factorial_state": factorial_state.value}


def train_micro_expert(*args, **kwargs):
    """Punto de extensión que carga el entrenador histórico solo en Zeus."""
    raise RuntimeError("El entrenador se debe resolver mediante _train_micro_expert")


def run_phase13(config_path: str | Path | None = None, *, resume: bool = False, dry_run: bool = False, only: str | None = None) -> dict[str, Any]:
    if only not in (None, "13B", "13C"):
        raise ValueError("--only debe ser 13B o 13C")
    config = load_battery_config(config_path)
    if dry_run:
        return _print_plan(only, config)
    if only in (None, "13B"):
        result = run_13b(config, resume=resume)
    else:
        result = load_13b_result(config=config, resume=True)
    if only in (None, "13C"):
        result["13C"] = run_13c(config, result.get("13B", result), resume=resume)
    return result


def _load_micro_context():
    root = Path(__file__).resolve().parents[2]
    micro_module = _load_module(root / "src/models/micro/08B_clean_temporal_micro_baseline.py", "p13_micro_08b")
    refit_module = _load_module(root / "src/models/final/10A_refit_full_historical.py", "p13_refit_10a")
    temporal_module = _load_module(root / "src/features/macro_v2/11H_frozen_temporal_environment_ensemble.py", "p13_temporal_11h")
    from src.features.macro_v2._common import extract_capture_dates
    micro_df, x_dir, x_size, y, labels, micro_source = refit_module.load_full_micro(micro_module)
    captures, historical_path = temporal_module.load_historical_65()
    captures["pcap_uid"] = captures["pcap_uid"].astype(str)
    captures["_query_date"] = extract_capture_dates(captures)
    micro_df["pcap_uid"] = micro_df["pcap_uid"].astype(str)
    micro_df = micro_df.merge(captures[["pcap_uid", "_query_date"]], on="pcap_uid", how="left", validate="one_to_one")
    if micro_df["_query_date"].isna().any():
        raise RuntimeError("El puente pcap_uid → fecha Historical está incompleto")
    dates = np.array(sorted(micro_df["_query_date"].unique()))
    if len(dates) != 52:
        raise RuntimeError(f"Se esperaban 52 fechas Historical; se encontraron {len(dates)}")
    origins = {
        "ORIGIN14": {"train": dates[:14], "windows": {"NEAR": dates[14:21], "MID": dates[28:35], "FAR": dates[42:49]}},
        "ORIGIN28": {"train": dates[:28], "windows": {"NEAR": dates[28:35], "MID": dates[35:42], "FAR": dates[45:52]}},
    }
    return {"micro": micro_module, "refit": refit_module, "temporal": temporal_module, "micro_df": micro_df, "x_dir": x_dir, "x_size": x_size, "y": y, "labels": labels, "source": micro_source, "historical_path": historical_path, "origins": origins}


def _train_micro_expert(context: dict[str, Any], origin: str, mode: str, seed: int, device: Any, epochs: int, edge_weight_ratio: float = EDGE_WEIGHT_RATIO):
    import torch
    micro = context["micro"]
    temporal = context["temporal"]
    df = context["micro_df"]
    train_dates = context["origins"][origin]["train"]
    train_idx = np.flatnonzero(df["_query_date"].isin(train_dates).to_numpy())
    micro.SEED = int(seed)
    full_weights = np.ones(len(df), dtype=np.float32)
    if mode != "UNIFORM":
        full_weights[train_idx] = temporal_weights(df.iloc[train_idx]["_query_date"], mode, ratio=edge_weight_ratio).astype(np.float32)
    source13a = Path(__file__).resolve().parents[2] / "src/models/micro/13A_frozen_temporal_micro_ensemble.py"
    mod13a = _load_module(source13a, f"p13a_{origin}_{mode}_{seed}")
    model, min_size, max_size, curve = mod13a.train_weighted_micro(micro, context["x_dir"], context["x_size"], context["y"], train_idx, full_weights, device, epochs, mode)
    return model, min_size, max_size, curve, train_idx


def _evaluate_model(context: dict[str, Any], model: Any, min_size: float, max_size: float, origin: str, device: Any) -> dict[str, dict[str, Any]]:
    source13a = Path(__file__).resolve().parents[2] / "src/models/micro/13A_frozen_temporal_micro_ensemble.py"
    mod13a = _load_module(source13a, f"p13a_eval_{origin}_{time.time_ns()}")
    spec = context["origins"][origin]
    return mod13a.evaluate_windows(context["micro"], model, min_size, max_size, context["micro_df"], context["x_dir"], context["x_size"], context["y"], spec["windows"], device)


def _window_metadata(context: dict[str, Any], origin: str, window: str) -> dict[str, np.ndarray]:
    dates = context["origins"][origin]["windows"][window]
    idx = np.flatnonzero(context["micro_df"]["_query_date"].isin(dates).to_numpy())
    frame = context["micro_df"].iloc[idx]
    return {
        "pcap_uid": frame["pcap_uid"].astype(str).to_numpy(),
        "query_date": pd.to_datetime(frame["_query_date"]).to_numpy(dtype="datetime64[D]"),
        "y_true": np.asarray(context["y"])[idx].astype(np.int64),
        "candidate_labels": np.asarray(context["labels"]).astype(str),
    }


def run_13b(config: BatteryConfig, *, resume: bool = False) -> dict[str, Any]:
    """Ejecuta el carril Historical 13B completo en Zeus."""
    import gc
    import torch

    context = _load_micro_context()
    OUTPUT_13B.mkdir(parents=True, exist_ok=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    config_hash = _config_hash(config)
    root = Path(__file__).resolve().parents[2]
    input_paths = {
        "micro_source": Path(context["source"]),
        "historical_source": Path(context["historical_path"]),
        "config": config.path,
    }
    input_hashes = _hash_paths(input_paths)
    code_hashes = _hash_paths(_phase13_code_paths(root))
    fingerprint = _artifact_fingerprint(config_hash, input_hashes, code_hashes)
    input_fingerprint = _fingerprint_digest(input_hashes)
    code_fingerprint = _fingerprint_digest(code_hashes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    expert_probs: dict[tuple[str, str, int], dict[str, Any]] = {}
    training_rows: list[dict[str, Any]] = []
    legacy_cache_count = 0
    protocol = config.payload
    for item in build_experiment_matrix(config)["micro_training"]:
        origin, mode, seed = item["origin"], item["temporal_mode"], int(item["seed"])
        cache_dir = CACHE_ROOT / origin / mode / str(seed)
        prediction_paths = {window: cache_dir / f"{window}.npz" for window in tuple(protocol["windows"])}
        cache_manifest = cache_dir / "manifest.json"
        checkpoint_path = cache_dir / "model.pt"
        scaler_path = cache_dir / "scaler.json"
        expected_paths = _expert_expected_paths(cache_dir)
        state = classify_expert_cache_state(cache_dir, origin, mode, seed, fingerprint)
        if state == CacheState.PARTIAL:
            _raise_cache_state(cache_dir, state, expected_paths)
        if state == CacheState.COMPLETE_INCOMPATIBLE:
            _raise_cache_state(cache_dir, state, expected_paths)
        if state in (CacheState.COMPLETE_COMPATIBLE, CacheState.COMPLETE_COMPATIBLE_LEGACY) and not resume:
            raise RuntimeError(f"Cache {state.value} en {cache_dir}; usar --resume para reutilizar")
        if resume and state in (CacheState.COMPLETE_COMPATIBLE, CacheState.COMPLETE_COMPATIBLE_LEGACY):
            identity = {"origin": origin, "temporal_mode": mode, "seed": seed}
            expected_windows = {window: _window_metadata(context, origin, window) for window in tuple(protocol["windows"])}
            if state == CacheState.COMPLETE_COMPATIBLE_LEGACY:
                cached = _validate_legacy_cache(cache_dir, origin, mode, seed, expected_windows=expected_windows)
                provenance = cached["provenance"]
                prediction_fingerprint = LEGACY_CACHE_SOURCE_FINGERPRINT
                legacy_cache_count += 1
            else:
                cached = _validate_cache(cache_manifest, fingerprint)
                if cached.get("identity") != identity:
                    raise RuntimeError(f"Cache incompatible {cache_dir}: identidad de entrenamiento")
                provenance = cached.get("provenance", {})
                if provenance.get("config_sha256") != config_hash or provenance.get("input_fingerprint") != input_fingerprint or provenance.get("code_fingerprint") != code_fingerprint:
                    raise RuntimeError(f"Cache incompatible {cache_dir}: provenance")
                validate_file_sha256(checkpoint_path, provenance.get("checkpoint_sha256", ""), "checkpoint")
                validate_file_sha256(scaler_path, provenance.get("scaler_sha256", ""), "scaler")
                scaler = json.loads(scaler_path.read_text(encoding="utf-8"))
                if scaler.get("origin") != origin or scaler.get("temporal_mode") != mode or int(scaler.get("seed")) != seed:
                    raise RuntimeError(f"Cache incompatible {cache_dir}: metadata scaler")
                expected_architecture = {"max_len": int(protocol["micro"]["max_len"]), "d_model": int(protocol["micro"]["d_model"]), "heads": int(protocol["micro"]["heads"]), "layers": int(protocol["micro"]["layers"]), "epochs": int(protocol["micro"]["epochs"]), "edge_weight_ratio": float(protocol["micro"]["edge_weight_ratio"])}
                if scaler.get("architecture") != expected_architecture or cached.get("architecture") != expected_architecture:
                    raise RuntimeError(f"Cache incompatible {cache_dir}: arquitectura")
                try:
                    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                except Exception as exc:
                    raise RuntimeError(f"Cache incompatible {cache_dir}: checkpoint ilegible") from exc
                for field, expected_value in {"origin": origin, "temporal_mode": mode, "seed": seed, "fingerprint": fingerprint}.items():
                    if checkpoint.get(field) != expected_value:
                        raise RuntimeError(f"Cache incompatible {cache_dir}: metadata checkpoint {field}")
                if checkpoint.get("architecture") != expected_architecture or checkpoint.get("config_sha256") != config_hash or checkpoint.get("input_fingerprint") != input_fingerprint or checkpoint.get("code_fingerprint") != code_fingerprint or checkpoint.get("scaler_sha256") != provenance.get("scaler_sha256"):
                    raise RuntimeError(f"Cache incompatible {cache_dir}: provenance checkpoint")
                prediction_fingerprint = fingerprint
            values = {}
            for window, path in prediction_paths.items():
                metadata = expected_windows[window]
                expected = {**identity, "window": window, **metadata}
                loaded = _load_prediction(path, prediction_fingerprint, expected=expected)
                for field in ("checkpoint_sha256", "scaler_sha256", "config_sha256", "input_fingerprint", "code_fingerprint"):
                    if str(loaded[field].item()) != str(provenance[field]):
                        raise RuntimeError(f"Cache incompatible {path}: provenance {field}")
                values[window] = loaded
            training_rows.append(cached["training"])
        else:
            started = time.perf_counter()
            model, min_size, max_size, curve, train_idx = _train_micro_expert(context, origin, mode, seed, device, int(protocol["micro"]["epochs"]), float(protocol["micro"]["edge_weight_ratio"]))
            evaluated = _evaluate_model(context, model, min_size, max_size, origin, device)
            fit_seconds = time.perf_counter() - started
            scaler_payload = {"origin": origin, "temporal_mode": mode, "seed": seed, "min_size": float(min_size), "max_size": float(max_size), "architecture": {"max_len": int(protocol["micro"]["max_len"]), "d_model": int(protocol["micro"]["d_model"]), "heads": int(protocol["micro"]["heads"]), "layers": int(protocol["micro"]["layers"]), "epochs": int(protocol["micro"]["epochs"]), "edge_weight_ratio": float(protocol["micro"]["edge_weight_ratio"])}, "config_sha256": config_hash, "input_fingerprint": input_fingerprint, "code_fingerprint": code_fingerprint}
            _write_json(scaler_payload, scaler_path)
            scaler_sha256 = sha256(scaler_path)
            torch.save({"state_dict": model.state_dict(), "min_size": min_size, "max_size": max_size, "seed": seed, "origin": origin, "temporal_mode": mode, "fingerprint": fingerprint, "architecture": scaler_payload["architecture"], "config_sha256": config_hash, "input_fingerprint": input_fingerprint, "code_fingerprint": code_fingerprint, "scaler_sha256": scaler_sha256}, checkpoint_path)
            checkpoint_sha256 = sha256(checkpoint_path)
            values = {}
            for window in tuple(protocol["windows"]):
                current = evaluated[window]
                metadata = _window_metadata(context, origin, window)
                if not _array_equal(current["y"], metadata["y_true"]) or not _array_equal(current["dates"], metadata["query_date"]):
                    raise RuntimeError(f"Alineación interna Micro rota en {origin}/{window}: evaluación y metadata")
                values[window] = {"probs": current["probs"], "y_true": current["y"], "query_date": current["dates"], **metadata, "origin": origin, "window": window, "temporal_mode": mode, "seed": seed}
                _save_prediction(prediction_paths[window], probs=values[window]["probs"], pcap_uid=values[window]["pcap_uid"], query_date=values[window]["query_date"], y=values[window]["y_true"], origin=origin, window=window, temporal_mode=mode, seed=seed, candidate_labels=values[window]["candidate_labels"], fingerprint=fingerprint, checkpoint_sha256=checkpoint_sha256, scaler_sha256=scaler_sha256, config_sha256=config_hash, input_fingerprint=input_fingerprint, code_fingerprint=code_fingerprint)
            training = {"origin": origin, "temporal_mode": mode, "seed": seed, "epochs": int(protocol["micro"]["epochs"]), "n_train": int(len(train_idx)), "fit_seconds": fit_seconds, "device": str(device), "git_commit": _git_commit()}
            _write_json({"fingerprint": fingerprint, "identity": {"origin": origin, "temporal_mode": mode, "seed": seed}, "training": training, "architecture": scaler_payload["architecture"], "provenance": {"config_sha256": config_hash, "input_fingerprint": input_fingerprint, "code_fingerprint": code_fingerprint, "checkpoint_sha256": checkpoint_sha256, "scaler_sha256": scaler_sha256}, "outputs_sha256": {window: sha256(path) for window, path in prediction_paths.items()}}, cache_manifest)
            training_rows.append(training)
            del model, curve
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        expert_probs[(origin, mode, seed)] = values

    summary_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    diversity_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    candidate_store: dict[tuple[str, str, str], dict[str, Any]] = {}
    seed_to_name = {11: "S11", 42: "S42", 73: "S73"}
    for origin in tuple(protocol["origins"]):
        for window in tuple(protocol["windows"]):
            records = []
            for mode in tuple(protocol["micro"]["temporal_modes"]):
                for seed in tuple(protocol["micro"]["seeds"]):
                    record = expert_probs[(origin, mode, seed)][window]
                    records.append({**record, "origin": origin, "window": window, "temporal_mode": mode, "seed": seed})
            validate_expert_prediction_alignment(records)
            y = expert_probs[(origin, "UNIFORM", 42)][window]["y_true"]
            dates = expert_probs[(origin, "UNIFORM", 42)][window]["query_date"]
            lookup = {(mode, seed): expert_probs[(origin, mode, seed)][window]["probs"] for mode in tuple(protocol["micro"]["temporal_modes"]) for seed in tuple(protocol["micro"]["seeds"])}
            candidates = _ensemble_from_config(lookup, protocol)
            for name, probs in candidates.items():
                metrics = metric_values(probs, y)
                summary_rows.append({"origin": origin, "window": window, "candidate": name, **metrics})
                candidate_store[(origin, window, name)] = {"probs": probs, "y": y, "dates": dates, "pcap_uid": records[0]["pcap_uid"], "query_date": records[0]["query_date"], "candidate_labels": records[0]["candidate_labels"], "origin": origin, "window": window}
                ranks = rank_values(probs, y)
                for group in ("TOP1_CORRECT", "TOP5_RECOVERABLE", "TOP5_MISSED"):
                    mask = np.asarray([rank_gap_group(value) == group for value in ranks])
                    confidence = entropy_margin(probs)
                    rank_rows.append({"origin": origin, "window": window, "candidate": name, "category": group, "count": int(mask.sum()), "fraction": float(mask.mean()), "confidence_top1": float(confidence["top1_confidence"][mask].mean()) if mask.any() else 0.0, "top1_top2_margin": float(confidence["top1_top2_margin"][mask].mean()) if mask.any() else 0.0, "entropy": float(confidence["entropy"][mask].mean()) if mask.any() else 0.0, "true_class_probability": float(probs[np.arange(len(y)), y][mask].mean()) if mask.any() else 0.0})
            pairs = (("EARLY", "UNIFORM"), ("UNIFORM", "LATE"), ("EARLY", "LATE"))
            for left, right in pairs:
                diversity_rows.append({"origin": origin, "window": window, "measure": f"top1_disagreement_{left}_{right}", "value": float(np.mean(np.argmax(lookup[(left, 42)], axis=1) != np.argmax(lookup[(right, 42)], axis=1)))})
            for candidate, reference in (("T3_S42", "U3"), ("T3_S11", "U3"), ("T3_S73", "U3"), ("U3", "U1")):
                first, second = candidate_store[(origin, window, reference)]["probs"], candidate_store[(origin, window, candidate)]["probs"]
                first_metrics, second_metrics = metric_values(first, y), metric_values(second, y)
                row = {"origin": origin, "window": window, "candidate": candidate, "reference": reference}
                for metric in ("accuracy", "macro_f1", "top5_accuracy", "mrr", "mean_true_rank"):
                    row[f"delta_{metric}"] = second_metrics[metric] - first_metrics[metric]
                delta_rows.append(row)
                stable_offset = int(_sha256_bytes(f"{origin}|{window}|{candidate}".encode("utf-8"))[:8], 16) % 100000
                boot = bootstrap_delta(y, dates, first, second, seed_offset=stable_offset)
                for metric, values in boot.items():
                    bootstrap_rows.append({"origin": origin, "window": window, "candidate": candidate, "reference": reference, "metric": metric, "observed_delta": row[f"delta_{metric}"], "bootstrap_mean": float(values.mean()), "ci95_low": float(np.quantile(values, 0.025)), "ci95_high": float(np.quantile(values, 0.975)), "fraction_delta_gt_0": float(np.mean(values > 0))})
    summary = _metrics_frame(summary_rows)
    deltas = pd.DataFrame(delta_rows)
    bootstrap = pd.DataFrame(bootstrap_rows)
    diversity = pd.DataFrame(diversity_rows)
    rank_gap = pd.DataFrame(rank_rows)
    _write_frame(summary, OUTPUT_13B / "13B_model_summary.csv")
    _write_frame(deltas, OUTPUT_13B / "13B_pairwise_deltas.csv")
    _write_frame(bootstrap, OUTPUT_13B / "13B_day_block_bootstrap.csv")
    _write_frame(diversity, OUTPUT_13B / "13B_expert_diversity.csv")
    _write_frame(pd.DataFrame(training_rows), OUTPUT_13B / "13B_training_summary.csv")
    _write_frame(rank_gap, OUTPUT_13B / "13B_rank_gap_historical.csv")
    outputs = {path.name: sha256(path) for path in OUTPUT_13B.glob("13B_*.csv")}
    primary = deltas[(deltas["candidate"] == "T3_S42") & (deltas["reference"] == "U3") & (deltas["window"] == "FAR")]
    primary_boot = bootstrap[(bootstrap["candidate"] == "T3_S42") & (bootstrap["reference"] == "U3") & (bootstrap["window"] == "FAR") & (bootstrap["metric"] == "macro_f1")]
    primary_accuracy = float(primary["delta_accuracy"].mean()) if not primary.empty else float("nan")
    primary_macro_f1 = float(primary["delta_macro_f1"].mean()) if not primary.empty else float("nan")
    primary_gate = {
        "far_macro_f1_positive_both_origins": bool(len(primary) == 2 and (primary["delta_macro_f1"] > 0).all()),
        "mean_far_macro_f1_gain": primary_macro_f1,
        "mean_far_macro_f1_gain_pass": bool(primary_macro_f1 >= 0.005),
        "mean_far_accuracy_gain": primary_accuracy,
        "mean_far_accuracy_gain_pass": bool(primary_accuracy > 0),
        "far_bootstrap_fraction_delta_gt_0_by_origin": primary_boot[["origin", "fraction_delta_gt_0"]].to_dict(orient="records"),
        "far_bootstrap_fraction_delta_gt_0_pass": bool(len(primary_boot) == 2 and (primary_boot["fraction_delta_gt_0"] >= 0.90).all()),
    }
    primary_gate["pass"] = bool(primary_gate["far_macro_f1_positive_both_origins"] and primary_gate["mean_far_macro_f1_gain_pass"] and primary_gate["mean_far_accuracy_gain_pass"] and primary_gate["far_bootstrap_fraction_delta_gt_0_pass"])
    manifest = build_manifest(stage="13B", research_question="¿El temporal weighting aporta un beneficio específico frente al ensemble genérico de seeds?", candidate_matrix=build_experiment_matrix(config), inputs=input_hashes, code_hashes=code_hashes, config_hash=config_hash, outputs=outputs)
    manifest["fingerprint"] = fingerprint
    manifest["provenance"] = {"config_sha256": config_hash, "input_fingerprint": input_fingerprint, "code_fingerprint": code_fingerprint}
    if legacy_cache_count:
        if legacy_cache_count != len(build_experiment_matrix(config)["micro_training"]):
            raise RuntimeError("No se permite mezclar caches Micro legacy y actuales en un único manifest 13B")
        manifest["training_cache_source_commit"] = LEGACY_CACHE_SOURCE_COMMIT
        manifest["training_cache_source_fingerprint"] = LEGACY_CACHE_SOURCE_FINGERPRINT
        manifest["training_cache_reuse"] = "VERIFIED_LEGACY_CACHE"
    else:
        manifest["training_cache_source_commit"] = _git_commit()
        manifest["training_cache_source_fingerprint"] = fingerprint
        manifest["training_cache_reuse"] = "NONE"
    manifest["postprocessing_git_commit"] = _git_commit()
    manifest["execution"] = {"device": str(device), "resume": resume, "bootstrap_iterations": BOOTSTRAP_ITERATIONS, "fraction_delta_gt_0_is_not_p_value": True}
    manifest["gate_evaluation"] = {"primary_T3_S42_vs_U3": primary_gate, "replications": "descriptivas; sin umbral inventado después del resultado", "generic_ensemble_gain": "U3_vs_U1_reported_separately"}
    _write_json(manifest, OUTPUT_13B / "13B_manifest.json")
    return {"13B": {"summary": summary, "deltas": deltas, "bootstrap": bootstrap, "candidate_store": candidate_store, "manifest": manifest}}


def load_13b_result(config: BatteryConfig | None = None, *, resume: bool = True) -> dict[str, Any]:
    manifest_path = OUTPUT_13B / "13B_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("13C requiere primero el resultado cacheado de 13B")
    summary = pd.read_csv(OUTPUT_13B / "13B_model_summary.csv")
    battery_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_battery_fingerprint = battery_manifest.get("fingerprint")
    if not expected_battery_fingerprint:
        raise RuntimeError("Manifest 13B no contiene fingerprint global")
    candidate_store: dict[tuple[str, str, str], dict[str, Any]] = {}
    config = config or load_battery_config()
    protocol = config.payload
    for origin in tuple(protocol["origins"]):
        for window in tuple(protocol["windows"]):
            paths = {(mode, seed): CACHE_ROOT / origin / mode / str(seed) / f"{window}.npz" for mode in tuple(protocol["micro"]["temporal_modes"]) for seed in tuple(protocol["micro"]["seeds"])}
            loaded = {}
            records = []
            for (mode, seed), path in paths.items():
                cache_manifest = path.parent / "manifest.json"
                expected_paths = _expert_expected_paths(path.parent)
                state = classify_expert_cache_state(path.parent, origin, mode, seed, expected_battery_fingerprint)
                if state not in (CacheState.COMPLETE_COMPATIBLE, CacheState.COMPLETE_COMPATIBLE_LEGACY):
                    _raise_cache_state(path.parent, state, expected_paths)
                fingerprint = json.loads(cache_manifest.read_text(encoding="utf-8"))["fingerprint"]
                if state == CacheState.COMPLETE_COMPATIBLE_LEGACY:
                    fingerprint = LEGACY_CACHE_SOURCE_FINGERPRINT
                metadata = {"origin": origin, "window": window, "temporal_mode": mode, "seed": seed}
                loaded[(mode, seed)] = _load_prediction(path, fingerprint, expected={**metadata})
                records.append({**loaded[(mode, seed)], **metadata})
            validate_expert_prediction_alignment(records)
            reference = loaded[("UNIFORM", 42)]
            y = reference["y_true"]
            dates = reference["query_date"]
            probabilities = {(mode, seed): loaded[(mode, seed)]["probs"] for mode in tuple(protocol["micro"]["temporal_modes"]) for seed in tuple(protocol["micro"]["seeds"])}
            candidates = _ensemble_from_config(probabilities, protocol)
            for name, probs in candidates.items():
                candidate_store[(origin, window, name)] = {"probs": probs, "y": y, "dates": dates, "pcap_uid": reference["pcap_uid"], "query_date": reference["query_date"], "candidate_labels": reference["candidate_labels"], "origin": origin, "window": window}
    return {"13B": {"summary": summary, "manifest": battery_manifest, "candidate_store": candidate_store}}


def _load_or_train_macro_candidates(context: dict[str, Any], origin: str, device: Any, resume: bool, config: BatteryConfig) -> dict[str, Any]:
    """Reconstrucción determinista bajo implementación y configuración Macro congeladas."""
    cache_path = CACHE_ROOT / "macro" / f"{origin}.npz"
    cache_manifest = cache_path.with_suffix(".json")
    root = Path(__file__).resolve().parents[2]
    fingerprint, input_hashes, code_hashes = _macro_provenance(config, context, origin)
    expected_paths = _macro_expected_paths(origin)
    state = classify_cache_state(expected_paths, cache_manifest, fingerprint)
    if state == CacheState.PARTIAL or state == CacheState.COMPLETE_INCOMPATIBLE:
        _raise_cache_state(cache_path.parent / origin, state, expected_paths)
    if state == CacheState.COMPLETE_COMPATIBLE and not resume:
        raise RuntimeError(f"Cache Macro COMPLETE_COMPATIBLE para {origin}; usar --resume para reutilizar")
    if resume and state == CacheState.COMPLETE_COMPATIBLE:
        cached_manifest = _validate_cache(cache_manifest, fingerprint)
        if cached_manifest.get("identity") != {"origin": origin}:
            raise RuntimeError(f"Cache Macro incompatible {origin}: identidad")
        with np.load(cache_path, allow_pickle=False) as payload:
            result = {"M0": {window: payload[f"M0_{window}"] for window in WINDOWS}, "M1": {window: payload[f"M1_{window}"] for window in WINDOWS}, "y": {window: payload[f"y_{window}"] for window in WINDOWS}, "dates": {window: payload[f"dates_{window}"] for window in WINDOWS}, "pcap_uid": {window: payload[f"pcap_uid_{window}"] for window in WINDOWS}, "candidate_labels": payload["candidate_labels"].astype(str)}
        return result
    mod = _load_module(root / "src/features/macro_v2/07A_candidate_conditioned_temporal_encoder.py", f"p13_macro_{origin}")
    helpers = _load_module(root / "src/features/macro_v2/11B_multiscale_longitudinal_memory.py", f"p13_macro_helpers_{origin}")
    m11h = context["temporal"]
    m11f = _load_module(root / "src/features/macro_v2/11F_oracle_memory_refresh.py", f"p13_macro_refresh_{origin}")
    m11l = _load_module(root / "src/features/macro_v2/11L_factorial_frozen_combination.py", f"p13_macro_11l_{origin}")
    features, _ = mod.load_frozen_features()
    captures, _ = m11h.load_historical_65()
    captures["site_label"] = captures["site_label"].astype(str)
    from src.features.macro_v2._common import extract_capture_dates
    captures["_query_date"] = extract_capture_dates(captures)
    labels = np.array(sorted(captures["site_label"].unique()), dtype=str)
    mapping = mod.make_label_mapping(labels)
    daily_all = helpers.build_daily(captures, features)
    dates = np.array(sorted(captures["_query_date"].unique()))
    specs = {"ORIGIN14": {"train": dates[:14], "windows": {"NEAR": dates[14:21], "MID": dates[28:35], "FAR": dates[42:49]}}, "ORIGIN28": {"train": dates[:28], "windows": {"NEAR": dates[28:35], "MID": dates[35:42], "FAR": dates[45:52]}}}
    data = m11h.prepare_origin(mod, helpers, m11f, captures, daily_all, labels, mapping, features, specs[origin], origin, device, [])
    multiscale = m11l.train_multiscale_ltd(mod, helpers, data["train"], data["tests"], daily_all, labels, mapping, features, specs[origin], origin, device, [])
    out = {"M0": {}, "M1": {}, "y": data["y_tests"], "dates": data["test_dates"], "pcap_uid": {}, "candidate_labels": labels}
    for window in WINDOWS:
        out["M0"][window] = helpers.fuse(data["xgb_candidates"]["UNIFORM"][window], data["ltd_probs"][window])
        out["M1"][window] = helpers.fuse(data["xgb_candidates"]["TEMPORAL_SYMMETRIC3"][window], multiscale[window])
        out["pcap_uid"][window] = data["tests"][window]["pcap_uid"].astype(str).to_numpy()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **{f"{macro}_{window}": out[macro][window] for macro in HYBRID_MACRO_CANDIDATES for window in WINDOWS}, **{f"y_{window}": out["y"][window] for window in WINDOWS}, **{f"dates_{window}": out["dates"][window] for window in WINDOWS}, **{f"pcap_uid_{window}": out["pcap_uid"][window] for window in WINDOWS}, candidate_labels=out["candidate_labels"])
    _write_json({"fingerprint": fingerprint, "identity": {"origin": origin}, "config_sha256": _config_hash(config), "inputs": input_hashes, "code": code_hashes, "scientific_status": "DETERMINISTIC_RECONSTRUCTION_FROZEN_IMPLEMENTATION", "outputs_sha256": {"cache": sha256(cache_path)}}, cache_manifest)
    return out


def run_13c(config: BatteryConfig, result_13b: dict[str, Any], *, resume: bool = False) -> dict[str, Any]:
    """Evalúa el factorial Hybrid fijo sin promover ningún candidato."""
    import torch
    if not (OUTPUT_13B / "13B_manifest.json").exists():
        raise FileNotFoundError("13C requiere 13B precomputado")
    factorial_fingerprint, factorial_inputs, factorial_code = _factorial_fingerprint(config)
    factorial_paths = _factorial_expected_paths()
    factorial_state = classify_cache_state(factorial_paths, OUTPUT_13C / "13C_manifest.json", factorial_fingerprint)
    if factorial_state in (CacheState.PARTIAL, CacheState.COMPLETE_INCOMPATIBLE):
        _raise_cache_state(OUTPUT_13C, factorial_state, factorial_paths)
    if factorial_state == CacheState.COMPLETE_COMPATIBLE:
        if not resume:
            raise RuntimeError("Outputs 13C COMPLETE_COMPATIBLE; usar --resume para reutilizar")
        manifest = json.loads((OUTPUT_13C / "13C_manifest.json").read_text(encoding="utf-8"))
        for path in factorial_paths:
            if path.suffix == ".csv":
                validate_file_sha256(path, manifest.get("outputs_sha256", {}).get(path.name, ""), f"output 13C {path.name}")
        return {"summary": pd.read_csv(OUTPUT_13C / "13C_factorial_summary.csv"), "effects": pd.read_csv(OUTPUT_13C / "13C_factorial_effects.csv"), "manifest": manifest}
    context = _load_micro_context()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidate_store = result_13b.get("candidate_store")
    if candidate_store is None:
        raise RuntimeError("El resume de 13C necesita las matrices de probabilidad 13B cacheadas")
    rows, deltas, boots, rank_rows = [], [], [], []
    product_store: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    protocol = config.payload
    for origin in tuple(protocol["origins"]):
        macro = _load_or_train_macro_candidates(context, origin, device, resume, config)
        for window in tuple(protocol["windows"]):
            for micro_name in tuple(protocol["hybrid_micro_candidates"]):
                for macro_name in tuple(protocol["macro_candidates"]):
                    key = (origin, window, micro_name)
                    micro_record = candidate_store[key]
                    y = micro_record["y"]
                    dates = micro_record["dates"]
                    macro_probs = macro[macro_name][window]
                    validate_hybrid_alignment(micro_record, {"pcap_uid": macro["pcap_uid"][window], "query_date": macro["dates"][window], "y_true": macro["y"][window], "candidate_labels": macro["candidate_labels"], "origin": origin, "window": window})
                    micro_weight = float(protocol["hybrid_weights"]["micro"])
                    macro_weight = float(protocol["hybrid_weights"]["macro"])
                    hybrid = np.exp(micro_weight * np.log(np.clip(micro_record["probs"], EPS, 1.0)) + macro_weight * np.log(np.clip(macro_probs, EPS, 1.0)))
                    hybrid /= hybrid.sum(axis=1, keepdims=True)
                    product_store[(origin, window, micro_name, macro_name)] = {"probs": hybrid, "y": y, "dates": dates}
                    rows.append({"origin": origin, "window": window, "micro_candidate": micro_name, "macro_candidate": macro_name, **metric_values(hybrid, y)})
                    ranks = rank_values(hybrid, y)
                    for group in ("TOP1_CORRECT", "TOP5_RECOVERABLE", "TOP5_MISSED"):
                        mask = np.asarray([rank_gap_group(value) == group for value in ranks])
                        rank_rows.append({"origin": origin, "window": window, "candidate": f"{micro_name}x{macro_name}", "category": group, "count": int(mask.sum()), "fraction": float(mask.mean())})
    for origin in tuple(protocol["origins"]):
        for window in tuple(protocol["windows"]):
            reference = product_store[(origin, window, "U1", "M0")]
            for micro_name in tuple(protocol["hybrid_micro_candidates"]):
                for macro_name in tuple(protocol["macro_candidates"]):
                    if micro_name == "U1" and macro_name == "M0":
                        continue
                    candidate = product_store[(origin, window, micro_name, macro_name)]
                    left_metrics = metric_values(reference["probs"], reference["y"])
                    right_metrics = metric_values(candidate["probs"], candidate["y"])
                    delta = {"origin": origin, "window": window, "candidate": f"{micro_name}x{macro_name}", "reference": "U1xM0"}
                    for metric in ("accuracy", "macro_f1", "top5_accuracy", "mrr", "mean_true_rank"):
                        delta[f"delta_{metric}"] = right_metrics[metric] - left_metrics[metric]
                    deltas.append(delta)
                    bootstrap = bootstrap_delta(reference["y"], reference["dates"], reference["probs"], candidate["probs"], seed_offset=int(_sha256_bytes(f"13C|{origin}|{window}|{micro_name}|{macro_name}".encode("utf-8"))[:8], 16) % 100000)
                    for metric, values in bootstrap.items():
                        boots.append({"origin": origin, "window": window, "candidate": f"{micro_name}x{macro_name}", "reference": "U1xM0", "metric": metric, "observed_delta": delta[f"delta_{metric}"], "bootstrap_mean": float(values.mean()), "ci95_low": float(np.quantile(values, 0.025)), "ci95_high": float(np.quantile(values, 0.975)), "fraction_delta_gt_0": float(np.mean(values > 0))})
    summary = pd.DataFrame(rows)
    effects = []
    for origin in tuple(protocol["origins"]):
        for window in tuple(protocol["windows"]):
            pivot = summary[(summary.origin == origin) & (summary.window == window)].set_index(["micro_candidate", "macro_candidate"])
            for metric in tuple(protocol["metrics"]):
                effects.append({"origin": origin, "window": window, "metric": metric, "micro_robustness_at_M0": float(pivot.loc[("U3", "M0"), metric] - pivot.loc[("U1", "M0"), metric]), "micro_robustness_at_M1": float(pivot.loc[("U3", "M1"), metric] - pivot.loc[("U1", "M1"), metric]), "macro_robustness_at_U1": float(pivot.loc[("U1", "M1"), metric] - pivot.loc[("U1", "M0"), metric]), "macro_robustness_at_U3": float(pivot.loc[("U3", "M1"), metric] - pivot.loc[("U3", "M0"), metric]), "interaction_vs_canonical": float(pivot.loc[("U3", "M1"), metric] - pivot.loc[("U3", "M0"), metric] - pivot.loc[("U1", "M1"), metric] + pivot.loc[("U1", "M0"), metric])})
    _write_frame(summary, OUTPUT_13C / "13C_factorial_summary.csv")
    _write_frame(pd.DataFrame(effects), OUTPUT_13C / "13C_factorial_effects.csv")
    _write_frame(pd.DataFrame(deltas), OUTPUT_13C / "13C_pairwise_deltas.csv")
    _write_frame(pd.DataFrame(boots), OUTPUT_13C / "13C_day_block_bootstrap.csv")
    _write_frame(pd.DataFrame(rank_rows), OUTPUT_13C / "13C_rank_gap_historical.csv")
    outputs = {path.name: sha256(path) for path in factorial_paths if path.suffix == ".csv"}
    manifest = build_manifest(stage="13C", research_question="¿Cómo interactúan los candidatos Micro y Macro robustos bajo la fusión Hybrid congelada?", candidate_matrix=build_experiment_matrix(config), inputs=factorial_inputs, code_hashes=factorial_code, config_hash=_config_hash(config), outputs=outputs)
    manifest["fingerprint"] = factorial_fingerprint
    manifest["macro_reconstruction"] = "deterministic reconstruction under frozen implementation/configuration"
    manifest["data_policy"]["promotion_allowed"] = False
    _write_json(manifest, OUTPUT_13C / "13C_manifest.json")
    return {"summary": summary, "effects": pd.DataFrame(effects), "manifest": manifest}
