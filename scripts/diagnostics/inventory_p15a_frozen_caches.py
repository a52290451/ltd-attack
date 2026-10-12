#!/usr/bin/env python3
"""Read-only cache inventory for Historical Phase 15A; never loads tensors or runs inference."""
from pathlib import Path
import json
import zipfile
import numpy as np

ROOT=Path.home()/"ltd-storage/artifacts/experiments/PHASE13_BATTERY_V1"
LABELS={"ORIGIN14","ORIGIN28"}
def inspect_npz(path):
    # Inspect ZIP member headers only; do not load data or unpickle object arrays.
    try:
        with zipfile.ZipFile(path) as z:
            rows=[]
            for member in sorted(z.namelist()):
                if not member.endswith(".npy"):continue
                with z.open(member) as f:
                    version=np.lib.format.read_magic(f)
                    if version==(1,0):
                        shape,_,dtype=np.lib.format.read_array_header_1_0(f)
                    elif version in {(2,0),(3,0)}:
                        shape,_,dtype=np.lib.format.read_array_header_2_0(f)
                    else:raise ValueError("unsupported NPY version")
                rows.append(f"{member}:{shape}:{dtype}:object={dtype.hasobject}")
        return rows
    except Exception as exc:
        return ["ERROR:"+type(exc).__name__+":"+str(exc)]
def main():
    print("CACHE_ROOT",ROOT,"EXISTS",ROOT.is_dir())
    if not ROOT.is_dir():
        print("NO_CACHE_AT_DEFAULT_PATH: inspect src/experiments/phase13_battery.py CACHE_ROOT and environment mapping")
        return
    paths=sorted(ROOT.rglob("*.npz"))
    print("NPZ_COUNT",len(paths))
    for path in paths:
        if "future" in str(path).lower() or "concept_drift" in str(path).lower():
            raise SystemExit("ABORT unexpected future-related path")
        print("CACHE",str(path),"BYTES",path.stat().st_size)
        for row in inspect_npz(path):print("  ARRAY",row)
    print("NOTE: HEADER-ONLY inventory. No training, predictions, or Future-B access.")
if __name__=="__main__":main()
