#!/usr/bin/env python3
"""Read-only Phase 15A availability and provenance preflight (no Future-B access)."""
from pathlib import Path
import hashlib
import json
import os

home=Path.home()
base=home/"ltd-storage/results"
targets={
    "13B":base/"micro/MICRO-ROBUSTNESS-PHASE13-BATTERY",
    "13C":base/"hybrid/LTD-ROBUSTNESS-PHASE13-BATTERY",
}
forbidden=("future","concept_drift","dataset_c")
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as stream:
        for b in iter(lambda:stream.read(1024*1024),b""):h.update(b)
    return h.hexdigest()
def main():
    if os.getenv("LTD_ENV","perseo")!="perseo":
        print("NOTE: preflight is designed for Perseo storage paths")
    for stage,path in targets.items():
        print("STAGE",stage,"PATH",path)
        manifest=path/(stage+"_manifest.json")
        if not manifest.is_file():
            raise SystemExit("MISSING_MANIFEST: "+str(manifest))
        doc=json.loads(manifest.read_text())
        hashes=doc.get("outputs_sha256",{})
        if not hashes:raise SystemExit("NO_OUTPUT_HASHES: "+stage)
        for name,expected in hashes.items():
            p=path/name
            if not p.is_file() or sha(p)!=expected:
                raise SystemExit("OUTPUT_HASH_FAILED: "+str(p))
        print(stage+"_HASHES_OK",len(hashes))
        for pattern in (stage+"_rank_gap_historical.csv",stage+"_model_summary.csv",stage+"_factorial_summary.csv"):
            p=path/pattern
            if p.is_file():
                import pandas as pd
                frame=pd.read_csv(p,nrows=4)
                print("DATA_SCHEMA",p.name,"FIELDS",",".join(frame.columns))
    import yaml
    cfg=Path("configs/experiments/PHASE15A_HIST_RANK_MECHANISMS_V1.yaml")
    plan=yaml.safe_load(cfg.read_text())
    assert plan["data_policy"]["historical_only"] is True
    assert plan["data_policy"]["allow_future_b"] is False
    assert plan["data_policy"]["training_allowed"] is False
    print("P15A_PREFLIGHT_OK; schema inventory only; no training was run")
if __name__=="__main__":main()
