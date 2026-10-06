import numpy as np
import pandas as pd
import json
from pathlib import Path
import pytest
import sys
import types

from src.experiments.phase13_battery import (
    BATTERY_MODES,
    CacheState,
    EXPECTED_MICRO_ARCHITECTURE,
    LEGACY_CACHE_SOURCE_COMMIT,
    LEGACY_CACHE_SOURCE_CODE_FINGERPRINT,
    LEGACY_CACHE_SOURCE_CONFIG_SHA256,
    LEGACY_CACHE_SOURCE_FINGERPRINT,
    LEGACY_CACHE_SOURCE_INPUT_FINGERPRINT,
    MICRO_SEEDS,
    ORIGINS,
    build_experiment_matrix,
    build_manifest,
    build_macro_candidate_definitions,
    classify_cache_state,
    geometric_probability_mean,
    validate_expert_prediction_alignment,
    validate_hybrid_alignment,
    validate_phase13_config,
    validate_file_sha256,
    temporal_weights,
)


def test_geometric_probability_mean_normaliza_y_preserva_forma():
    probs = [
        np.array([[0.8, 0.2], [0.1, 0.9]]),
        np.array([[0.5, 0.5], [0.4, 0.6]]),
    ]

    result = geometric_probability_mean(probs)

    assert result.shape == (2, 2)
    np.testing.assert_allclose(result.sum(axis=1), [1.0, 1.0])
    assert result[0, 0] > result[0, 1]


def test_temporal_weights_fija_early_uniform_late():
    dates = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"])

    early = temporal_weights(dates, "EARLY")
    uniform = temporal_weights(dates, "UNIFORM")
    late = temporal_weights(dates, "LATE")

    np.testing.assert_allclose(uniform, np.ones(3))
    assert early[0] > early[-1]
    assert late[-1] > late[0]
    np.testing.assert_allclose(early.mean(), 1.0)
    np.testing.assert_allclose(late.mean(), 1.0)


def test_matriz_13b_es_exacta_y_predeclarada():
    matrix = build_experiment_matrix()
    micro = matrix["micro_training"]

    assert len(micro) == 18
    assert {row["origin"] for row in micro} == set(ORIGINS)
    assert {row["temporal_mode"] for row in micro} == set(BATTERY_MODES)
    assert {row["seed"] for row in micro} == set(MICRO_SEEDS)
    assert matrix["micro_candidates"] == ["U1", "U3", "T3_S11", "T3_S42", "T3_S73", "T9"]
    assert matrix["hybrid_micro_candidates"] == ["U1", "U3", "T3_S42", "T9"]
    assert matrix["hybrid_macro_candidates"] == ["M0", "M1"]


def test_manifest_incluye_guardrails_y_no_seleccion():
    manifest = build_manifest(
        stage="13B",
        research_question="pregunta congelada",
        candidate_matrix=build_experiment_matrix(),
        inputs={"entrada": "abc"},
        code_hashes={"codigo": "def"},
        config_hash="ghi",
        outputs={"salida": "jkl"},
    )

    assert manifest["data_policy"]["future_b_scores_used"] is False
    assert manifest["data_policy"]["selection_allowed"] is False
    assert manifest["data_policy"]["training_inference_updates"] is False
    assert manifest["predeclared_gates"]


def test_dry_run_no_entrena(monkeypatch, capsys):
    from src.experiments import phase13_battery

    def fail_if_called(*args, **kwargs):
        raise AssertionError("dry-run no debe entrenar")

    monkeypatch.setattr(phase13_battery, "train_micro_expert", fail_if_called, raising=False)
    result = phase13_battery.run_phase13(dry_run=True, only="13B")

    output = capsys.readouterr().out
    assert result["dry_run"] is True
    assert "ORIGIN14/" in output
    assert "ORIGIN28/" in output
    assert "U3" in output
    assert "M1" in output
    assert "reutilizable" in output
    assert "FUTURE-B" not in output


def test_cache_incompatible_falla_explicitamente(tmp_path):
    from src.experiments.phase13_battery import _validate_cache

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"fingerprint": "antiguo"}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="Cache incompatible"):
        _validate_cache(manifest, "nuevo")


def _prediction(identity, *, pcap_uid=None, dates=None, y=None, labels=None):
    return {
        "probs": np.full((2, 3), 1 / 3, dtype=float),
        "pcap_uid": np.asarray(["p1", "p2"] if pcap_uid is None else pcap_uid),
        "query_date": np.asarray(["2026-01-01", "2026-01-02"] if dates is None else dates),
        "y_true": np.asarray([0, 1] if y is None else y),
        "candidate_labels": np.asarray(["a", "b", "c"] if labels is None else labels),
        **identity,
    }


def _nine_expert_predictions():
    return [
        _prediction({"origin": "ORIGIN14", "window": "FAR", "temporal_mode": mode, "seed": seed})
        for mode in BATTERY_MODES
        for seed in MICRO_SEEDS
    ]


def test_nueve_expertos_predeclarados_con_mismos_captures_pasan():
    validate_expert_prediction_alignment(_nine_expert_predictions())


def test_expert_alignment_permite_modes_y_seeds_validos_diferentes():
    records = _nine_expert_predictions()
    records[1]["query_date"] = np.asarray(["2026-01-01T00:00:00.000000000", "2026-01-02T00:00:00.000000000"], dtype="datetime64[ns]")

    validate_expert_prediction_alignment(records)


def test_hybrid_alignment_acepta_fechas_del_mismo_dia_con_dtypes_distintos():
    identity = {"origin": "ORIGIN14", "window": "FAR"}
    micro = _prediction(identity, dates=np.asarray(["2025-12-02", "2025-12-03"], dtype="datetime64[D]"))
    macro = _prediction(identity, dates=np.asarray(["2025-12-02T00:00:00.000000000", "2025-12-03T00:00:00.000000000"], dtype="datetime64[ns]"))

    validate_hybrid_alignment(micro, macro)


def test_hybrid_alignment_acepta_strings_y_datetime_del_mismo_dia():
    identity = {"origin": "ORIGIN14", "window": "FAR"}
    micro = _prediction(identity, dates=["2025-12-02", "2025-12-03"])
    macro = _prediction(identity, dates=np.asarray(["2025-12-02", "2025-12-03"], dtype="datetime64[D]"))

    validate_hybrid_alignment(micro, macro)


def _run_13c_schema_case(monkeypatch, tmp_path, *, macro_y):
    from src.experiments import phase13_battery

    fake_torch = types.ModuleType("torch")
    fake_torch.device = lambda name: "cpu"
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    output_13b = tmp_path / "13B"
    output_13c = tmp_path / "13C"
    output_13b.mkdir()
    (output_13b / "13B_manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(phase13_battery, "OUTPUT_13B", output_13b)
    monkeypatch.setattr(phase13_battery, "OUTPUT_13C", output_13c)
    monkeypatch.setattr(phase13_battery, "_factorial_fingerprint", lambda config: ("fp", {}, {}))
    monkeypatch.setattr(phase13_battery, "classify_cache_state", lambda *args: CacheState.ABSENT)
    monkeypatch.setattr(phase13_battery, "_load_micro_context", lambda: {})
    monkeypatch.setattr(phase13_battery, "bootstrap_delta", lambda *args, **kwargs: {metric: np.zeros(1) for metric in ("accuracy", "macro_f1", "top5_accuracy", "mrr")})

    origin = "ORIGIN14"
    window = "FAR"
    labels = np.asarray(["a", "b", "c"])
    pcap_uid = np.asarray(["p1", "p2"])
    dates = np.asarray(["2026-01-01", "2026-01-02"], dtype="datetime64[D]")
    micro_y = np.asarray([0, 1], dtype=np.int64)
    probs = np.asarray([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]], dtype=float)
    candidate_store = {}
    for micro_name in ("U1", "U3"):
        candidate_store[(origin, window, micro_name)] = {
            "probs": probs.copy(),
            "y": micro_y.copy(),
            "dates": dates.copy(),
            "pcap_uid": pcap_uid.copy(),
            "query_date": dates.copy(),
            "candidate_labels": labels.copy(),
            "origin": origin,
            "window": window,
        }
    macro = {
        "M0": {window: probs.astype(np.float32)},
        "M1": {window: probs.astype(np.float32)},
        "y": {window: np.asarray(macro_y, dtype=np.int64)},
        "dates": {window: dates.copy()},
        "pcap_uid": {window: pcap_uid.copy()},
        "candidate_labels": labels.copy(),
    }
    monkeypatch.setattr(phase13_battery, "_load_or_train_macro_candidates", lambda *args: macro)
    config = phase13_battery.BatteryConfig(
        tmp_path / "config.yaml",
        {
            "origins": [origin],
            "windows": [window],
            "hybrid_micro_candidates": ["U1", "U3"],
            "macro_candidates": ["M0", "M1"],
            "hybrid_weights": {"micro": 0.45, "macro": 0.55},
            "metrics": ["accuracy"],
        },
    )
    return phase13_battery, config, {"candidate_store": candidate_store}


def test_run_13c_normaliza_y_del_candidate_store_antes_de_validar(monkeypatch, tmp_path):
    phase13_battery, config, result_13b = _run_13c_schema_case(monkeypatch, tmp_path, macro_y=[0, 1])
    captured = []
    strict_validator = phase13_battery.validate_hybrid_alignment

    def capture_alignment(micro, macro):
        captured.append(micro)
        strict_validator(micro, macro)

    monkeypatch.setattr(phase13_battery, "validate_hybrid_alignment", capture_alignment)
    phase13_battery.run_13c(config, result_13b)

    assert len(captured) == 4
    for micro_alignment in captured:
        assert set(micro_alignment) == {"pcap_uid", "query_date", "y_true", "candidate_labels", "origin", "window"}
        np.testing.assert_array_equal(micro_alignment["y_true"], [0, 1])


def test_run_13c_rechaza_y_micro_distinto_de_y_true_macro(monkeypatch, tmp_path):
    phase13_battery, config, result_13b = _run_13c_schema_case(monkeypatch, tmp_path, macro_y=[1, 1])

    with pytest.raises(RuntimeError, match="Micro/Macro desalineados: y_true"):
        phase13_battery.run_13c(config, result_13b)


def test_hybrid_alignment_rechaza_nat():
    identity = {"origin": "ORIGIN14", "window": "FAR"}
    micro = _prediction(identity, dates=["2025-12-02", "2025-12-03"])
    macro = _prediction(identity, dates=np.asarray(["2025-12-02", "NaT"], dtype="datetime64[D]"))
    with pytest.raises(RuntimeError, match="query_date"):
        validate_hybrid_alignment(micro, macro)


@pytest.mark.parametrize(
    "dates",
    [
        ["2025-12-04", "2025-12-03"],
        ["2025-12-03", "2025-12-02"],
    ],
)
def test_hybrid_alignment_falla_si_fecha_distinta_o_permutada(dates):
    identity = {"origin": "ORIGIN14", "window": "FAR"}
    micro = _prediction(identity, dates=["2025-12-02", "2025-12-03"])
    macro = _prediction(identity, dates=dates)
    with pytest.raises(RuntimeError, match="query_date"):
        validate_hybrid_alignment(micro, macro)


def test_expert_alignment_falla_con_filas_permutadas_uid_fecha_y_label():
    for field, value in (
        ("pcap_uid", ["p2", "p1"]),
        ("query_date", ["2026-01-02", "2026-01-01"]),
        ("y_true", [1, 0]),
        ("candidate_labels", ["b", "a", "c"]),
    ):
        records = _nine_expert_predictions()
        records[1] = dict(records[1])
        records[1][field] = np.asarray(value)
        with pytest.raises(RuntimeError, match="alineación"):
            validate_expert_prediction_alignment(records)


@pytest.mark.parametrize("field,value", [("origin", "ORIGIN28"), ("window", "NEAR")])
def test_expert_alignment_falla_con_identidad_incorrecta(field, value):
    records = _nine_expert_predictions()
    records[1][field] = value
    with pytest.raises(RuntimeError, match="identidad"):
        validate_expert_prediction_alignment(records)


def test_expert_alignment_falla_con_seed_inesperado():
    records = _nine_expert_predictions()
    records[0]["seed"] = 999
    with pytest.raises(RuntimeError, match="seed inesperado"):
        validate_expert_prediction_alignment(records)


def test_expert_alignment_falla_con_mode_inesperado():
    records = _nine_expert_predictions()
    records[0]["temporal_mode"] = "UNKNOWN"
    with pytest.raises(RuntimeError, match="mode inesperado"):
        validate_expert_prediction_alignment(records)


def test_expert_alignment_falla_con_duplicado():
    records = _nine_expert_predictions()
    records[-1]["temporal_mode"] = records[0]["temporal_mode"]
    records[-1]["seed"] = records[0]["seed"]
    with pytest.raises(RuntimeError, match="duplicado"):
        validate_expert_prediction_alignment(records)


def test_expert_alignment_falla_con_experto_faltante():
    records = _nine_expert_predictions()[:-1]
    with pytest.raises(RuntimeError, match="faltante"):
        validate_expert_prediction_alignment(records)


def test_expert_alignment_falla_con_shape_de_probabilidades_distinta():
    records = _nine_expert_predictions()
    records[1]["probs"] = np.full((2, 2), 0.5, dtype=float)
    with pytest.raises(RuntimeError, match="shape"):
        validate_expert_prediction_alignment(records)


def test_micro_macro_alignment_falla_si_uid_se_intercambia_con_misma_etiqueta():
    identity = {"origin": "ORIGIN14", "window": "FAR"}
    micro = _prediction(identity)
    macro = _prediction(identity, pcap_uid=["p2", "p1"])
    with pytest.raises(RuntimeError, match="Micro/Macro"):
        validate_hybrid_alignment(micro, macro)


@pytest.mark.parametrize(
    "field,altered",
    [
        ("y_true", [1, 1]),
        ("candidate_labels", ["a", "c", "b"]),
    ],
)
def test_micro_macro_alignment_falla_si_y_true_o_labels_cambian(field, altered):
    identity = {"origin": "ORIGIN14", "window": "FAR"}
    micro = _prediction(identity)
    macro = dict(micro)
    macro[field] = np.asarray(altered)
    with pytest.raises(RuntimeError, match=field):
        validate_hybrid_alignment(micro, macro)


@pytest.mark.parametrize("field,value", [("origin", "ORIGIN28"), ("window", "NEAR")])
def test_micro_macro_alignment_falla_si_origin_o_window_cambian(field, value):
    identity = {"origin": "ORIGIN14", "window": "FAR"}
    micro = _prediction(identity)
    macro_identity = {**identity, field: value}
    macro = _prediction(macro_identity)
    with pytest.raises(RuntimeError, match="identidad"):
        validate_hybrid_alignment(micro, macro)


def test_cache_state_machine_distingue_parcial_y_incompatible(tmp_path):
    expected = [tmp_path / "manifest.json", tmp_path / "model.pt", tmp_path / "FAR.npz"]
    assert classify_cache_state(expected, tmp_path / "manifest.json", "fp") == CacheState.ABSENT
    expected[0].write_text('{"fingerprint":"fp"}', encoding="utf-8")
    assert classify_cache_state(expected, tmp_path / "manifest.json", "fp") == CacheState.PARTIAL
    expected[1].write_bytes(b"model")
    expected[2].write_bytes(b"prediction")
    assert classify_cache_state(expected, tmp_path / "manifest.json", "fp") == CacheState.COMPLETE_COMPATIBLE
    expected[0].write_text('{"fingerprint":"old"}', encoding="utf-8")
    assert classify_cache_state(expected, tmp_path / "manifest.json", "fp") == CacheState.COMPLETE_INCOMPATIBLE


def test_yaml_debe_coincidir_con_guardrails_congelados(tmp_path):
    from src.experiments.phase13_battery import load_battery_config

    config = load_battery_config()
    altered = dict(config.payload)
    altered["hybrid_weights"] = {"micro": 0.4, "macro": 0.6}
    with pytest.raises(ValueError, match="guardrail"):
        validate_phase13_config(altered)


def test_definiciones_macro_m0_m1_son_exactas():
    assert build_macro_candidate_definitions() == {
        "M0": "UNIFORM_XGB+RECENT5_LTD",
        "M1": "TEMPORAL_SYMMETRIC3_XGB+MULTISCALE5_LTD",
    }


def test_matriz_factorial_tiene_exactamente_ocho_productos():
    assert build_experiment_matrix()["hybrid_factorial"] == [
        "U1xM0", "U1xM1", "U3xM0", "U3xM1",
        "T3_S42xM0", "T3_S42xM1", "T9xM0", "T9xM1",
    ]


def test_hash_checkpoint_o_scaler_incompatible_falla(tmp_path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"original")
    import hashlib

    expected = hashlib.sha256(b"original").hexdigest()
    validate_file_sha256(artifact, expected, "checkpoint")
    artifact.write_bytes(b"corrupto")
    with pytest.raises(RuntimeError, match="SHA256 checkpoint"):
        validate_file_sha256(artifact, expected, "checkpoint")
    with pytest.raises(RuntimeError, match="SHA256 scaler"):
        validate_file_sha256(artifact, expected, "scaler")


def test_bootstrap_es_determinista():
    from src.experiments.phase13_battery import bootstrap_delta

    y = np.array([0, 1, 0, 1])
    dates = np.array(["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02"])
    first = np.tile([[0.8, 0.2]], (4, 1))
    second = np.tile([[0.7, 0.3]], (4, 1))
    left = bootstrap_delta(y, dates, first, second, seed_offset=9)
    right = bootstrap_delta(y, dates, first, second, seed_offset=9)
    for metric in left:
        np.testing.assert_array_equal(left[metric], right[metric])


def test_dry_run_expone_estado_de_cache(monkeypatch, tmp_path):
    from src.experiments import phase13_battery

    monkeypatch.setattr(phase13_battery, "CACHE_ROOT", tmp_path / "cache")
    result = phase13_battery.run_phase13(dry_run=True, only="13B")
    assert set(result["cache_states"].values()) == {CacheState.ABSENT.value}
    assert result["factorial_state"] == CacheState.ABSENT.value


def _macro_output_fixture():
    dates = np.asarray(["2025-12-02", "2025-12-03"], dtype="datetime64[ns]")
    return {
        "M0": {window: np.asarray([[0.8, 0.2], [0.3, 0.7]], dtype=float) for window in ("NEAR", "MID", "FAR")},
        "M1": {window: np.asarray([[0.7, 0.3], [0.4, 0.6]], dtype=float) for window in ("NEAR", "MID", "FAR")},
        "y": {window: np.asarray([0, 1], dtype=np.int32) for window in ("NEAR", "MID", "FAR")},
        "dates": {window: dates.copy() for window in ("NEAR", "MID", "FAR")},
        "pcap_uid": {window: np.asarray(["p1", "p2"], dtype=object) for window in ("NEAR", "MID", "FAR")},
        "candidate_labels": np.asarray(["a", "b"], dtype=object),
    }


def test_macro_cache_nuevo_es_pickle_free_y_tiene_dtypes_seguros(tmp_path):
    from src.experiments.phase13_battery import _load_macro_cache_arrays, _write_macro_cache

    path = tmp_path / "ORIGIN14.npz"
    _write_macro_cache(path, _macro_output_fixture())

    with np.load(path, allow_pickle=False) as payload:
        arrays = {key: payload[key] for key in payload.files}
    validated = _load_macro_cache_arrays(path)
    assert all(array.dtype.kind != "O" for array in arrays.values())
    assert all(validated[f"M0_{window}"].dtype == np.dtype(np.float32) for window in ("NEAR", "MID", "FAR"))
    assert all(validated[f"M1_{window}"].dtype == np.dtype(np.float32) for window in ("NEAR", "MID", "FAR"))
    assert all(validated[f"y_{window}"].dtype == np.dtype(np.int64) for window in ("NEAR", "MID", "FAR"))
    assert all(validated[f"dates_{window}"].dtype == np.dtype("datetime64[D]") for window in ("NEAR", "MID", "FAR"))
    assert all(validated[f"pcap_uid_{window}"].dtype.kind in {"U", "S"} for window in ("NEAR", "MID", "FAR"))
    assert validated["candidate_labels"].dtype.kind in {"U", "S"}


def test_macro_cache_corrupto_falla(tmp_path):
    from src.experiments.phase13_battery import _load_macro_cache_arrays, _write_macro_cache

    path = tmp_path / "ORIGIN14.npz"
    _write_macro_cache(path, _macro_output_fixture())
    path.write_bytes(b"corrupto")
    with pytest.raises(RuntimeError, match="Macro cache"):
        _load_macro_cache_arrays(path)


def test_macro_cache_sha_mismatch_falla(tmp_path):
    import hashlib

    from src.experiments.phase13_battery import _load_macro_cache, _write_macro_cache

    path = tmp_path / "ORIGIN14.npz"
    manifest = tmp_path / "ORIGIN14.json"
    _write_macro_cache(path, _macro_output_fixture())
    manifest.write_text(json.dumps({"fingerprint": "fp", "identity": {"origin": "ORIGIN14"}, "outputs_sha256": {"cache": hashlib.sha256(b"otro").hexdigest()}}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="SHA256 Macro cache"):
        _load_macro_cache(path, manifest, "fp", "ORIGIN14")


def test_launcher_declara_resume_directo_desde_13c_y_skip_independiente():
    script = Path("scripts/run_p13_b1_background.sh").read_text(encoding="utf-8")
    assert "--from-13c" in script
    assert "--only 13C" in script
    assert "--skip-14a" in script


def _write_legacy_cache(tmp_path, *, manifest_overrides=None, corrupt=None):
    import hashlib

    cache_dir = tmp_path / "ORIGIN14" / "EARLY" / "11"
    cache_dir.mkdir(parents=True)
    scaler = {
        "origin": "ORIGIN14",
        "temporal_mode": "EARLY",
        "seed": 11,
        "architecture": EXPECTED_MICRO_ARCHITECTURE,
        "config_sha256": LEGACY_CACHE_SOURCE_CONFIG_SHA256,
        "input_fingerprint": LEGACY_CACHE_SOURCE_INPUT_FINGERPRINT,
        "code_fingerprint": LEGACY_CACHE_SOURCE_CODE_FINGERPRINT,
    }
    (cache_dir / "scaler.json").write_text(json.dumps(scaler), encoding="utf-8")
    scaler_sha256 = hashlib.sha256((cache_dir / "scaler.json").read_bytes()).hexdigest()
    checkpoint = {
        "state_dict": {},
        "origin": "ORIGIN14",
        "temporal_mode": "EARLY",
        "seed": 11,
        "fingerprint": LEGACY_CACHE_SOURCE_FINGERPRINT,
        "architecture": EXPECTED_MICRO_ARCHITECTURE,
        "config_sha256": LEGACY_CACHE_SOURCE_CONFIG_SHA256,
        "input_fingerprint": LEGACY_CACHE_SOURCE_INPUT_FINGERPRINT,
        "code_fingerprint": LEGACY_CACHE_SOURCE_CODE_FINGERPRINT,
        "scaler_sha256": scaler_sha256,
    }
    checkpoint_path = cache_dir / "model.pt"
    try:
        import torch
    except ModuleNotFoundError:
        checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    else:
        torch.save(checkpoint, checkpoint_path)
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    outputs = {}
    for window in ("NEAR", "MID", "FAR"):
        np.savez_compressed(
            cache_dir / f"{window}.npz",
            probs=np.asarray([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]], dtype=np.float32),
            pcap_uid=np.asarray(["p1", "p2"]),
            query_date=np.asarray(["2026-01-01", "2026-01-02"], dtype="datetime64[D]"),
            y_true=np.asarray([0, 1], dtype=np.int64),
            origin=np.asarray("ORIGIN14"),
            window=np.asarray(window),
            temporal_mode=np.asarray("EARLY"),
            seed=np.asarray(11),
            candidate_labels=np.asarray(["a", "b", "c"]),
            fingerprint=np.asarray(LEGACY_CACHE_SOURCE_FINGERPRINT),
            checkpoint_sha256=np.asarray(checkpoint_sha256),
            scaler_sha256=np.asarray(scaler_sha256),
            config_sha256=np.asarray(LEGACY_CACHE_SOURCE_CONFIG_SHA256),
            input_fingerprint=np.asarray(LEGACY_CACHE_SOURCE_INPUT_FINGERPRINT),
            code_fingerprint=np.asarray(LEGACY_CACHE_SOURCE_CODE_FINGERPRINT),
        )
        outputs[window] = hashlib.sha256((cache_dir / f"{window}.npz").read_bytes()).hexdigest()
    manifest = {
        "fingerprint": LEGACY_CACHE_SOURCE_FINGERPRINT,
        "identity": {"origin": "ORIGIN14", "temporal_mode": "EARLY", "seed": 11},
        "training": {"git_commit": LEGACY_CACHE_SOURCE_COMMIT},
        "architecture": EXPECTED_MICRO_ARCHITECTURE,
        "provenance": {
            "config_sha256": LEGACY_CACHE_SOURCE_CONFIG_SHA256,
            "input_fingerprint": LEGACY_CACHE_SOURCE_INPUT_FINGERPRINT,
            "code_fingerprint": LEGACY_CACHE_SOURCE_CODE_FINGERPRINT,
            "checkpoint_sha256": checkpoint_sha256,
            "scaler_sha256": scaler_sha256,
        },
        "outputs_sha256": outputs,
    }
    if manifest_overrides:
        for path, value in manifest_overrides.items():
            target = manifest
            parts = path.split(".")
            for part in parts[:-1]:
                target = target[part]
            target[parts[-1]] = value
    (cache_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    if corrupt:
        (cache_dir / corrupt).write_bytes(b"corrupto")
    return cache_dir


def test_legacy_cache_exacto_aceptado(tmp_path):
    from src.experiments.phase13_battery import classify_expert_cache_state

    cache_dir = _write_legacy_cache(tmp_path)
    assert classify_expert_cache_state(cache_dir, "ORIGIN14", "EARLY", 11, "nuevo") == CacheState.COMPLETE_COMPATIBLE_LEGACY


@pytest.mark.parametrize(
    "override",
    [
        ("training.git_commit", "otro-commit"),
        ("provenance.config_sha256", "otro-config"),
        ("provenance.input_fingerprint", "otro-input"),
    ],
)
def test_legacy_cache_con_provenance_distinta_rechazado(tmp_path, override):
    from src.experiments.phase13_battery import classify_expert_cache_state

    cache_dir = _write_legacy_cache(tmp_path, manifest_overrides={override[0]: override[1]})
    assert classify_expert_cache_state(cache_dir, "ORIGIN14", "EARLY", 11, "nuevo") == CacheState.COMPLETE_INCOMPATIBLE


@pytest.mark.parametrize("corrupt", ["model.pt", "FAR.npz"])
def test_legacy_cache_corrupto_rechazado(tmp_path, corrupt):
    from src.experiments.phase13_battery import classify_expert_cache_state

    cache_dir = _write_legacy_cache(tmp_path, corrupt=corrupt)
    assert classify_expert_cache_state(cache_dir, "ORIGIN14", "EARLY", 11, "nuevo") == CacheState.COMPLETE_INCOMPATIBLE
