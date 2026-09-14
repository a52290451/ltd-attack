import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _probe(environment: str, cwd: Path) -> dict[str, str]:
    code = (
        "from src.utils.paths import PROJECT_ROOT, load_environment, data_path, "
        "artifact_path, result_path; "
        "p=load_environment(); "
        "print(PROJECT_ROOT); print(p['data_root']); print(p['artifact_root']); "
        "print(p['result_root']); print(data_path('sample.csv')); "
        "print(artifact_path('model.pth')); print(result_path('run'))"
    )
    env = os.environ.copy()
    env["LTD_ENV"] = environment
    env["PYTHONPATH"] = os.pathsep.join(
        [str(PROJECT_ROOT), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=cwd,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    values = completed.stdout.strip().splitlines()
    return {
        "project": values[0],
        "data": values[1],
        "artifact": values[2],
        "result": values[3],
        "sample": values[4],
        "model": values[5],
        "run": values[6],
    }


def test_project_root_and_local_paths_are_cwd_independent(tmp_path):
    values = _probe("local", tmp_path)
    assert Path(values["project"]) == PROJECT_ROOT
    assert Path(values["data"]) == PROJECT_ROOT / "data/local"
    assert Path(values["artifact"]) == PROJECT_ROOT / "artifacts"
    assert Path(values["result"]) == PROJECT_ROOT / "results"
    assert Path(values["sample"]) == PROJECT_ROOT / "data/local/sample.csv"
    assert Path(values["model"]) == PROJECT_ROOT / "artifacts/model.pth"
    assert Path(values["run"]) == PROJECT_ROOT / "results/run"


def test_zeus_paths_are_loaded_without_local_cwd_assumptions(tmp_path):
    values = _probe("zeus", tmp_path)
    assert Path(values["project"]) == PROJECT_ROOT
    assert Path(values["data"]) == Path("/home/bsierra/ltd-storage/datasets")
    assert Path(values["artifact"]) == Path("/home/bsierra/ltd-storage/artifacts")
    assert Path(values["result"]) == Path("/home/bsierra/ltd-storage/results")
    assert Path(values["sample"]) == Path("/home/bsierra/ltd-storage/datasets/sample.csv")
