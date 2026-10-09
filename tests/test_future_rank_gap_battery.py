import hashlib

import numpy as np
import pandas as pd
import pytest

from src.analysis.future_rank_gap_battery import (
    FUTURE_DIAGNOSTIC_FLAGS,
    build_confusion_edges,
    classify_rank_gap,
    entropy_margin,
    rank_distribution,
    rank_values,
)


def _write_recovery_fixture(tmp_path, *, y_true=None, query_date=None, macro=None, micro=None, extra_npz=None, uid_shape=None):
    from src.analysis import future_rank_gap_battery as battery

    tmp_path.mkdir(parents=True, exist_ok=True)
    labels = np.array(["site-a", "site-b"], dtype="<U6")
    macro = macro if macro is not None else pd.DataFrame(
        {
            "pcap_uid": ["u-1", "u-2", "u-3"],
            "site_label": ["site-a", "site-b", "site-a"],
            "date": ["2026-01-01", "2026-01-02", "2026-01-03"],
        }
    )
    micro = micro if micro is not None else pd.DataFrame({"pcap_uid": macro["pcap_uid"]})
    macro_path = tmp_path / "future_macro.csv"
    micro_path = tmp_path / "future_micro.csv"
    macro.to_csv(macro_path, index=False)
    micro.to_csv(micro_path, index=False)

    n = len(macro)
    y_true = np.array([0, 1, 0] if y_true is None else y_true, dtype=np.int64)
    query_date = np.array(
        ["2026-01-01", "2026-01-02", "2026-01-03"] if query_date is None else query_date,
        dtype="datetime64[D]",
    )
    probabilities = {
        "micro_probs": np.arange(n * 2, dtype=np.float32).reshape(n, 2),
        "macro_xgb_probs": (np.arange(n * 2, dtype=np.float32) + 10).reshape(n, 2),
        "macro_ltd_probs": (np.arange(n * 2, dtype=np.float32) + 20).reshape(n, 2),
        "macro_final_probs": (np.arange(n * 2, dtype=np.float32) + 30).reshape(n, 2),
        "hybrid_final_probs": (np.arange(n * 2, dtype=np.float32) + 40).reshape(n, 2),
    }
    values = {
        "pcap_uid": np.asarray(macro["pcap_uid"].tolist(), dtype=object),
        "query_date": query_date,
        "y_true": y_true,
        "candidate_labels": labels,
        **probabilities,
    }
    if uid_shape is not None:
        values["pcap_uid"] = np.empty(uid_shape, dtype=object)
    if extra_npz:
        values.update(extra_npz)
    score_path = tmp_path / "scores.npz"
    np.savez_compressed(score_path, **values)
    expected_hash = hashlib.sha256(score_path.read_bytes()).hexdigest()
    return battery, score_path, macro_path, micro_path, expected_hash, values


def _date_deriver(frame):
    return pd.to_datetime(frame["date"]), "date"


def test_rank_computation_and_distribucion_exacta():
    probs = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.2, 0.7],
        ]
    )
    y = np.array([0, 2, 1])

    ranks = rank_values(probs, y)
    assert ranks.tolist() == [1, 3, 2]
    distribution = rank_distribution(ranks)
    assert distribution["rank_1"] == 1
    assert distribution["rank_2"] == 1
    assert distribution["rank_3"] == 1
    assert distribution["rank_6_10"] == 0


def test_top5_recoverable_grouping():
    assert classify_rank_gap(1) == "TOP1_CORRECT"
    assert classify_rank_gap(2) == "TOP5_RECOVERABLE"
    assert classify_rank_gap(5) == "TOP5_RECOVERABLE"
    assert classify_rank_gap(6) == "TOP5_MISSED"


def test_entropy_y_margin_top1_top2():
    probs = np.array([[0.7, 0.2, 0.1]])
    values = entropy_margin(probs)

    np.testing.assert_allclose(values["top1_confidence"], [0.7])
    np.testing.assert_allclose(values["top1_top2_margin"], [0.5])
    assert values["entropy"][0] > 0


def test_confusion_edges_conserva_top5_y_true():
    probs = np.array(
        [
            [0.1, 0.8, 0.1],
            [0.6, 0.2, 0.2],
            [0.1, 0.2, 0.7],
        ]
    )
    y = np.array([0, 1, 1])
    edges = build_confusion_edges(y, probs, ["a", "b", "c"])

    assert len(edges) == 3
    assert set(edges["true_site"]) == {"a", "b"}
    assert edges.loc[edges["true_site"] == "a", "predicted_site"].iloc[0] == "b"
    assert float(edges.loc[edges["true_site"] == "a", "true_in_top5_fraction"].iloc[0]) == 1.0


def test_future_diagnostic_flags_never_habilitan_seleccion():
    assert FUTURE_DIAGNOSTIC_FLAGS["future_b_open"] is True
    assert FUTURE_DIAGNOSTIC_FLAGS["selection_allowed"] is False
    assert FUTURE_DIAGNOSTIC_FLAGS["training_allowed"] is False


def test_future_manifest_flags_never_habilitan_entrenamiento_ni_seleccion():
    from src.analysis.future_rank_gap_battery import build_future_manifest

    manifest = build_future_manifest(
        inputs={"scores": "x"},
        code_hashes={"analysis": "y"},
        config_hash="z",
        outputs={},
    )
    assert manifest["selection_allowed"] is False
    assert manifest["training_allowed"] is False
    assert manifest["git_commit"]


def test_future_dry_run_no_lee_npz_ni_entrena(monkeypatch, capsys):
    from src.analysis import future_rank_gap_battery

    monkeypatch.setattr(
        future_rank_gap_battery,
        "_load_npz",
        lambda: (_ for _ in ()).throw(AssertionError("dry-run no debe leer el NPZ")),
    )
    result = future_rank_gap_battery.run_future_rank_gap(dry_run=True)

    assert result["dry_run"] is True
    assert result["selection_allowed"] is False
    assert "no entrena" in capsys.readouterr().out


def test_recovery_reconstructs_identity_and_preserves_probabilities(tmp_path):
    battery, score_path, macro_path, micro_path, expected_hash, original = _write_recovery_fixture(tmp_path)

    before = hashlib.sha256(score_path.read_bytes()).hexdigest()
    recovered = battery._load_npz(
        score_path,
        future_macro_path=macro_path,
        historical_macro_path=tmp_path / "unused_historical.csv",
        future_micro_path=micro_path,
        expected_sha256=expected_hash,
        elite_sites={"site-a", "site-b"},
        date_deriver=_date_deriver,
        expected_n=3,
        expected_classes=2,
    )

    np.testing.assert_array_equal(recovered["pcap_uid"], ["u-1", "u-2", "u-3"])
    np.testing.assert_array_equal(recovered["query_date"], original["query_date"])
    np.testing.assert_array_equal(recovered["y_true"], original["y_true"])
    for name in ("micro_probs", "macro_xgb_probs", "macro_ltd_probs", "macro_final_probs", "hybrid_final_probs"):
        np.testing.assert_array_equal(recovered[name], original[name])
    assert hashlib.sha256(score_path.read_bytes()).hexdigest() == before


def test_recovery_uses_historical_site_mapping_only_when_site_label_missing(tmp_path):
    macro = pd.DataFrame(
        {
            "pcap_uid": ["u-1", "u-2", "u-3"],
            "site": ["raw-a", "raw-b", "raw-a"],
            "date": ["2026-01-01", "2026-01-02", "2026-01-03"],
        }
    )
    historical = pd.DataFrame({"site": ["raw-a", "raw-b"], "site_label": ["site-a", "site-b"]})
    battery, score_path, macro_path, micro_path, expected_hash, _ = _write_recovery_fixture(tmp_path, macro=macro)
    historical_path = tmp_path / "historical.csv"
    historical.to_csv(historical_path, index=False)

    recovered = battery._load_npz(
        score_path,
        future_macro_path=macro_path,
        historical_macro_path=historical_path,
        future_micro_path=micro_path,
        expected_sha256=expected_hash,
        elite_sites={"site-a", "site-b"},
        date_deriver=_date_deriver,
        expected_n=3,
        expected_classes=2,
    )

    np.testing.assert_array_equal(recovered["pcap_uid"], ["u-1", "u-2", "u-3"])


def test_recovery_ignora_micro_fuera_de_cohorte_y_preserva_scores(tmp_path):
    # Igual que 10B: UID ajenos (incluso duplicados) no entran a la cohorte.
    micro = pd.DataFrame({"pcap_uid": ["15713", "u-3", "u-1", "15714", "u-2", "15713"]})
    battery, score_path, macro_path, micro_path, expected_hash, values = _write_recovery_fixture(
        tmp_path, micro=micro,
    )
    recovered = battery._load_npz(
        score_path,
        future_macro_path=macro_path,
        future_micro_path=micro_path,
        expected_sha256=expected_hash,
        elite_sites={"site-a", "site-b"},
        date_deriver=_date_deriver,
        expected_n=3,
        expected_classes=2,
    )
    np.testing.assert_array_equal(recovered["pcap_uid"], ["u-1", "u-2", "u-3"])
    for field in ("micro_probs", "macro_xgb_probs", "macro_ltd_probs", "macro_final_probs", "hybrid_final_probs"):
        np.testing.assert_array_equal(recovered[field], values[field])


@pytest.mark.parametrize(
    ("micro_uids", "error"),
    [
        (["u-1", "u-2", "15713"], "Cobertura Micro Future-B"),
        (["u-1", "u-2", "u-2", "u-3", "15713"], "duplicado"),
    ],
)
def test_recovery_detecta_faltantes_y_duplicados_dentro_de_cohorte(tmp_path, micro_uids, error):
    battery, score_path, macro_path, micro_path, expected_hash, _ = _write_recovery_fixture(
        tmp_path, micro=pd.DataFrame({"pcap_uid": micro_uids}),
    )
    with pytest.raises(RuntimeError, match=error):
        battery._load_npz(
            score_path,
            future_macro_path=macro_path,
            future_micro_path=micro_path,
            expected_sha256=expected_hash,
            elite_sites={"site-a", "site-b"},
            date_deriver=_date_deriver,
            expected_n=3,
            expected_classes=2,
        )


def test_recovery_fails_on_label_discrepancy(tmp_path):
    battery, score_path, macro_path, micro_path, expected_hash, _ = _write_recovery_fixture(tmp_path, y_true=[1, 1, 0])

    with pytest.raises(RuntimeError, match="y_true"):
        battery._load_npz(
            score_path,
            future_macro_path=macro_path,
            future_micro_path=micro_path,
            expected_sha256=expected_hash,
            elite_sites={"site-a", "site-b"},
            date_deriver=_date_deriver,
            expected_n=3,
            expected_classes=2,
        )


def test_recovery_fails_on_date_discrepancy(tmp_path):
    battery, score_path, macro_path, micro_path, expected_hash, _ = _write_recovery_fixture(
        tmp_path,
        query_date=["2026-01-01", "2026-02-02", "2026-01-03"],
    )

    with pytest.raises(RuntimeError, match="query_date"):
        battery._load_npz(
            score_path,
            future_macro_path=macro_path,
            future_micro_path=micro_path,
            expected_sha256=expected_hash,
            elite_sites={"site-a", "site-b"},
            date_deriver=_date_deriver,
            expected_n=3,
            expected_classes=2,
        )


def test_recovery_fails_on_duplicate_missing_or_permuted_identity(tmp_path):
    duplicate_macro = pd.DataFrame(
        {
            "pcap_uid": ["u-1", "u-1", "u-3"],
            "site_label": ["site-a", "site-b", "site-a"],
            "date": ["2026-01-01", "2026-01-02", "2026-01-03"],
        }
    )
    battery, score_path, macro_path, micro_path, expected_hash, _ = _write_recovery_fixture(tmp_path / "duplicate", macro=duplicate_macro)
    with pytest.raises(RuntimeError, match="duplicado"):
        battery._load_npz(score_path, future_macro_path=macro_path, future_micro_path=micro_path, expected_sha256=expected_hash, elite_sites={"site-a", "site-b"}, date_deriver=_date_deriver, expected_n=3, expected_classes=2)

    missing_micro = pd.DataFrame({"pcap_uid": ["u-1", "u-2"]})
    battery, score_path, macro_path, micro_path, expected_hash, _ = _write_recovery_fixture(tmp_path / "missing", micro=missing_micro)
    with pytest.raises(RuntimeError, match="Micro Future-B"):
        battery._load_npz(score_path, future_macro_path=macro_path, future_micro_path=micro_path, expected_sha256=expected_hash, elite_sites={"site-a", "site-b"}, date_deriver=_date_deriver, expected_n=3, expected_classes=2)

    permuted_macro = pd.DataFrame(
        {
            "pcap_uid": ["u-2", "u-1", "u-3"],
            "site_label": ["site-b", "site-a", "site-a"],
            "date": ["2026-01-02", "2026-01-01", "2026-01-03"],
        }
    )
    battery, score_path, macro_path, micro_path, expected_hash, _ = _write_recovery_fixture(tmp_path / "permuted", macro=permuted_macro)
    with pytest.raises(RuntimeError, match="y_true|query_date"):
        battery._load_npz(score_path, future_macro_path=macro_path, future_micro_path=micro_path, expected_sha256=expected_hash, elite_sites={"site-a", "site-b"}, date_deriver=_date_deriver, expected_n=3, expected_classes=2)


def test_recovery_rejects_unexpected_object_field_and_bad_uid_header(tmp_path):
    battery, score_path, macro_path, micro_path, expected_hash, _ = _write_recovery_fixture(
        tmp_path / "extra-object",
        extra_npz={"other_object": np.array(["unexpected"], dtype=object)},
    )
    with pytest.raises(RuntimeError, match="object|inesperado"):
        battery._load_npz(score_path, future_macro_path=macro_path, future_micro_path=micro_path, expected_sha256=expected_hash, elite_sites={"site-a", "site-b"}, date_deriver=_date_deriver, expected_n=3, expected_classes=2)

    battery, score_path, macro_path, micro_path, expected_hash, _ = _write_recovery_fixture(tmp_path / "bad-shape", uid_shape=(3, 1))
    with pytest.raises(RuntimeError, match="pcap_uid"):
        battery._load_npz(score_path, future_macro_path=macro_path, future_micro_path=micro_path, expected_sha256=expected_hash, elite_sites={"site-a", "site-b"}, date_deriver=_date_deriver, expected_n=3, expected_classes=2)
