import numpy as np
import pandas as pd
import json
import pytest

from src.experiments.phase13_battery import (
    BATTERY_MODES,
    CacheState,
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
        "pcap_uid": np.asarray(pcap_uid or ["p1", "p2"]),
        "query_date": np.asarray(dates or ["2026-01-01", "2026-01-02"]),
        "y_true": np.asarray(y or [0, 1]),
        "candidate_labels": np.asarray(labels or ["a", "b", "c"]),
        **identity,
    }


def test_expert_alignment_falla_con_filas_permutadas_uid_fecha_y_label():
    identity = {"origin": "ORIGIN14", "window": "FAR", "temporal_mode": "UNIFORM", "seed": 42}
    reference = _prediction(identity)
    for field, value in (
        ("pcap_uid", ["p2", "p1"]),
        ("dates", ["2026-01-02", "2026-01-01"]),
        ("y", [1, 0]),
        ("labels", ["b", "a", "c"]),
    ):
        altered = _prediction(identity, **{field: value})
        with pytest.raises(RuntimeError, match="alineación"):
            validate_expert_prediction_alignment([reference, altered])


@pytest.mark.parametrize("field,value", [("origin", "ORIGIN28"), ("window", "NEAR"), ("seed", 73)])
def test_expert_alignment_falla_con_identidad_incorrecta(field, value):
    reference = _prediction({"origin": "ORIGIN14", "window": "FAR", "temporal_mode": "UNIFORM", "seed": 42})
    identity = {"origin": "ORIGIN14", "window": "FAR", "temporal_mode": "UNIFORM", "seed": 42}
    identity[field] = value
    with pytest.raises(RuntimeError, match="identidad"):
        validate_expert_prediction_alignment([reference, _prediction(identity)])


def test_micro_macro_alignment_falla_si_uid_se_intercambia_con_misma_etiqueta():
    identity = {"origin": "ORIGIN14", "window": "FAR"}
    micro = _prediction(identity)
    macro = _prediction(identity, pcap_uid=["p2", "p1"])
    with pytest.raises(RuntimeError, match="Micro/Macro"):
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
