import numpy as np

from src.analysis.future_rank_gap_battery import (
    FUTURE_DIAGNOSTIC_FLAGS,
    build_confusion_edges,
    classify_rank_gap,
    entropy_margin,
    rank_distribution,
    rank_values,
)


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
