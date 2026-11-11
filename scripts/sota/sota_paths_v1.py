#!/usr/bin/env python3
"""Paths for one reproducible LTD original-SOTA battery on Perseo or Zeus.

Environment variables change LOCATION only, never labels, classes, metrics,
upstream commits, split population or training hyperparameters.
Frozen S1/S5 sources are separately hash-verified by the existing gates.

LTD_SOTA_OPS_ROOT defaults to ~/ltd-ops/sota
LTD_SOTA_ARTIFACT_ROOT defaults to /data/ltd-attack/sota
LTD_SOTA_STORAGE_ROOT defaults to ~/ltd-storage/datasets
LTD_SOTA_S1_MANIFEST defaults to <OPS>/s1_dev_frozen_20261102T130232Z/s1_manifest.json
LTD_SOTA_S5_MANIFEST defaults to <ARTIFACT>/s5/frozen_s5_refits_and_labels_v1.json
LTD_SOTA_FUTURE_MANIFEST defaults to docs/evidence/FINAL/FUTURE-B/10B_manifest_future_b.json

IMPORTANT: Existing S1/S5 JSON can contain embedded absolute paths. A change
of the *manifest* location does not rewrite embedded references. The Zeus
bootstrap verifies embedded source paths and checksums before allowing fit.
"""
from __future__ import annotations
import os
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[2]
OPS=Path(os.environ.get("LTD_SOTA_OPS_ROOT",str(Path.home()/"ltd-ops/sota"))).expanduser()
ARTIFACT=Path(os.environ.get("LTD_SOTA_ARTIFACT_ROOT","/data/ltd-attack/sota")).expanduser()
STORAGE=Path(os.environ.get("LTD_SOTA_STORAGE_ROOT",str(Path.home()/"ltd-storage/datasets"))).expanduser()
S1=Path(os.environ.get("LTD_SOTA_S1_MANIFEST",
    str(OPS/"s1_dev_frozen_20261102T130232Z/s1_manifest.json"))).expanduser()
S5=Path(os.environ.get("LTD_SOTA_S5_MANIFEST",
    str(ARTIFACT/"s5/frozen_s5_refits_and_labels_v1.json"))).expanduser()
FUTURE=Path(os.environ.get("LTD_SOTA_FUTURE_MANIFEST",
    str(PROJECT_ROOT/"docs/evidence/FINAL/FUTURE-B/10B_manifest_future_b.json"))).expanduser()
ORIGINAL=OPS/"upstream_original_2026"
MODERN=OPS/"upstream_2026"
