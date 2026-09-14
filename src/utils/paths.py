"""Centralized, environment-aware paths for active LTD-Attack code.

The module intentionally resolves paths without creating directories. Data,
checkpoints, and other large artifacts must be provisioned separately.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def _find_project_root() -> Path:
    """Find the repository root from this module's location."""
    module_path = Path(__file__).resolve()
    for candidate in module_path.parents:
        if (candidate / "src").is_dir() and (candidate / "configs").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate project root from {module_path}")


PROJECT_ROOT = _find_project_root()
ENVIRONMENT_DIR = PROJECT_ROOT / "configs" / "environments"
_REQUIRED_KEYS = ("data_root", "artifact_root", "result_root")
_VALID_ENVIRONMENTS = {"local", "zeus"}


def _read_simple_yaml(path: Path) -> dict[str, Any]:
    """Read the small scalar YAML contract used by environment files.

    PyYAML is used when available. The fallback keeps path configuration
    usable in minimal environments and deliberately accepts only scalars.
    """
    try:
        import yaml  # type: ignore
    except ImportError:
        values: dict[str, Any] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip().strip("'\"")
        return values

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Environment config must be a mapping: {path}")
    return payload


def load_environment(environment: str | None = None) -> dict[str, Path | str]:
    """Load an environment config and resolve relative values to the root."""
    name = environment or os.environ.get("LTD_ENV", "local")
    if name not in _VALID_ENVIRONMENTS:
        raise ValueError(
            f"Unsupported LTD_ENV={name!r}; expected one of {sorted(_VALID_ENVIRONMENTS)}"
        )

    config_path = ENVIRONMENT_DIR / f"{name}.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing environment configuration: {config_path}")

    raw = _read_simple_yaml(config_path)
    missing = [key for key in _REQUIRED_KEYS if key not in raw]
    if missing:
        raise ValueError(f"Missing keys in {config_path}: {', '.join(missing)}")

    resolved: dict[str, Path | str] = {"environment": name}
    for key in _REQUIRED_KEYS:
        value = Path(str(raw[key])).expanduser()
        resolved[key] = value if value.is_absolute() else PROJECT_ROOT / value
    return resolved


def _root_path(root_key: str, *parts: str | os.PathLike[str], required: bool = False) -> Path:
    config = load_environment()
    if root_key not in _REQUIRED_KEYS:
        raise KeyError(f"Unknown configured root: {root_key}")
    path = config[root_key].joinpath(*(Path(part) for part in parts))
    if required and not path.exists():
        raise FileNotFoundError(f"Required path does not exist: {path}")
    return path


def data_path(*parts: str | os.PathLike[str], required: bool = False) -> Path:
    return _root_path("data_root", *parts, required=required)


def artifact_path(*parts: str | os.PathLike[str], required: bool = False) -> Path:
    return _root_path("artifact_root", *parts, required=required)


def result_path(*parts: str | os.PathLike[str], required: bool = False) -> Path:
    return _root_path("result_root", *parts, required=required)


def require_path(path: str | os.PathLike[str]) -> Path:
    """Validate an explicitly resolved path without creating anything."""
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Required path does not exist: {resolved}")
    return resolved


ENVIRONMENT = load_environment()
DATA_ROOT = ENVIRONMENT["data_root"]
ARTIFACT_ROOT = ENVIRONMENT["artifact_root"]
RESULT_ROOT = ENVIRONMENT["result_root"]
