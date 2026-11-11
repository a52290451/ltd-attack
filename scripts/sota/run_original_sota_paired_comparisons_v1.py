#!/usr/bin/env python3
"""Post-frozen all-model statistical comparisons to LTD reference predictions.

Requires explicit 09B and Future-B LTD reference NPZ paths in environment.
No predictions are created and no model retraining or selection occurs.
A model's original score NPZ must pass its SHA manifest before analysis.
Output refuses overwrite, and all models must be complete.
"""
from __future__ import annotations
import json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.sota.original_sota_orchestrator_v2 import MODELS,ROOTS,COHORTS,EVAL_TYPES,check_manifest
from scripts.sota.sota_paths_v1 import ARTIFACT
SCRIPT=ROOT/"scripts/sota/original_sota_paired_date_stats_v1.py"
PY=os.environ.get("LTD_SOTA_PYTORCH_PY","/data/ltd-sota-bsierra/venv-cu128/bin/python")
FROZEN_REFERENCE={
 "09B":os.environ.get("LTD_SOTA_LTD_REFERENCE_09B"),
 "10B_DEV_FROZEN":os.environ.get("LTD_SOTA_LTD_REFERENCE_FUTURE_B")}
def main():
    for cohort,path in FROZEN_REFERENCE.items():
        if not path or not Path(path).is_file():
            raise RuntimeError("SOTA_PAIRED_BATTERY_BLOCKED missing "+
                               cohort+" frozen LTD reference NPZ")
    base=ARTIFACT/"authors_source_frozen_results_v1/comparisons"
    for model in MODELS:
        for cohort in COHORTS:
            folder=ROOTS[model]/"evaluation"/cohort
            check_manifest(folder/"manifest.json",EVAL_TYPES[model],cohort)
            target=base/("LTD_vs_"+model+"_"+cohort+".json")
            if target.exists():
                raise RuntimeError("SOTA_PAIRED_BATTERY_BLOCKED existing "+
                                   str(target)+" - no overwrites")
            cmd=[PY,str(SCRIPT),"--cohort",cohort,
                 "--reference",FROZEN_REFERENCE[cohort],
                 "--candidate",str(folder/"scores_sota_aligned.npz"),
                 "--candidate-name",model,
                 "--reference-name","LTD-Hybrid",
                 "--replicates","2500","--output",str(target)]
            print("SOTA_PAIRWISE_BEGIN",model,cohort,flush=True)
            subprocess.run(cmd,cwd=ROOT,check=True)
    print("SOTA_ALL_AUTHORS_PAIRED_09B_FUTURE_B_PASS",len(MODELS)*len(COHORTS),
          flush=True)
if __name__=="__main__":main()
