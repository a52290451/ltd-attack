#!/usr/bin/env python3
"""Ejecuta el diagnóstico Future-B 14A sin entrenamiento ni selección."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.future_rank_gap_battery import run_future_rank_gap


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnóstico post-hoc Future-B 14A")
    parser.add_argument("--config", default="configs/experiments/FUTURE_RANK_GAP_V1.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Muestra fuentes y guardrails sin leer resultados")
    args = parser.parse_args()
    run_future_rank_gap(args.config, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
