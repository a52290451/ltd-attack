#!/usr/bin/env python3
"""Ejecuta o planifica la batería Historical P13-B1."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.phase13_battery import run_phase13


def main() -> int:
    parser = argparse.ArgumentParser(description="Batería P13-B1 Historical")
    parser.add_argument("--config", default="configs/experiments/PHASE13_BATTERY_V1.yaml")
    parser.add_argument("--resume", action="store_true", help="Reutiliza solo cache compatible por hash")
    parser.add_argument("--dry-run", action="store_true", help="Muestra la matriz sin entrenar")
    parser.add_argument("--only", choices=["13B", "13C"], help="Ejecuta un carril predeclarado")
    args = parser.parse_args()
    run_phase13(args.config, resume=args.resume, dry_run=args.dry_run, only=args.only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
