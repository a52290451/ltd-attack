#!/usr/bin/env python3
"""CPU-only audit of authors' NetAugment against frozen Historical DEV training.

Uses original SPIN-UMass notebook Augmentor class without substituting WFlib.
Recomputes the authors' 1000-trace burst-size histogram from DEV_EARLY/MIDDLE,
runs deterministic synthetic/real-trace smoke, and quantifies CDF/bin indexing.
Never trains, opens 09B/Future-B/C, or writes model predictions.

This is a diagnostic, NOT proof of paper-level augmentation equivalence.
"""
from __future__ import annotations
import argparse
import ast
import bisect
from collections import Counter
import importlib.util
import json
from pathlib import Path
import random
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
GATE_PATH=ROOT/"scripts/sota/s9_netclr_authors_cpu_gate_v1.py"
PREP_PATH=ROOT/"scripts/sota/s9_netclr_supervised_b_v1.py"
AUDIT_OUT=Path("/data/ltd-attack/sota/s9_netclr_author_original_v1/diagnostics")

def module(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj

def require(ok, msg):
    if not ok:raise RuntimeError("S9_AUTHOR_AUGMENT_AUDIT_BLOCKED "+msg)

def run():
    gate=module(GATE_PATH,"s9_original_gate")
    b=module(PREP_PATH,"s9_supervised_cache")
    provenance, _, aug_src=gate.inspect()
    # This cache is historical data only, with UID/SHA and chronological gates.
    X,y,split,cache=b.prepare_cache()
    idx=np.flatnonzero(split!=2)
    require(len(idx)==39258,"training split must contain only 39258 samples")
    require(int(np.sum(split==2))==18658,"DEV_LATE must remain unused")
    require(X.shape==(57916,5000),"cache tensor shape differs")
    # Keep the notebook's sampling population and count (1,000 sampled traces).
    # Preserve a separate seed and record the exact 1,000 cache row indices.
    rng=np.random.RandomState(42)
    selected=np.asarray(rng.choice(idx,size=1000,replace=False),dtype=np.int64)
    # Original find_bursts function is extracted from pinned authors' source,
    # without running the original cell's AWF global data sampling.
    cells=gate.notebook(gate.PRE)
    finder=None
    for txt in cells:
        try: tree=ast.parse(txt)
        except SyntaxError:continue
        for node in tree.body:
            if isinstance(node,ast.FunctionDef) and node.name=="find_bursts":
                finder=ast.get_source_segment(txt,node)
                break
        if finder:break
    require(finder is not None,"original find_bursts function missing")
    namespace={"np":np,"random":random,"bisect":bisect}
    exec(compile(ast.parse(finder),"s9_authors_find_bursts","exec"),namespace)
    burst_sizes=[]
    observed_lengths=[]
    for i in selected:
        trace=np.asarray(X[int(i)],dtype=np.int32)
        observed_lengths.append(int(np.count_nonzero(trace)))
        burst_sizes.extend([int(q[2]) for q in namespace["find_bursts"](trace)
                            if q[2]>0])
    require(bool(burst_sizes),"historical train has no positive bursts")
    maximum=max(burst_sizes)
    require(maximum>1,"historical train max burst too small")
    # Exactly as in notebook code cells 9-10.
    count,bins=np.histogram(burst_sizes,bins=maximum-1)
    pdf=count/count.sum()
    cdf=np.zeros_like(bins)
    cdf[1:]=np.cumsum(pdf)
    require(np.isclose(cdf[-1],1.0),"authors CDF malformed")
    require(len(cdf)==maximum,"unexpected notebook CDF shape")
    require(np.all(np.diff(cdf)>=0),"authors CDF not monotonic")
    namespace.update({
        "outgoing_burst_sizes":burst_sizes,
        "max_outgoing_burst_size":maximum,
        "OUTGOING_BURST_SIZE_CDF":cdf,
    })
    aug_ast=ast.parse(aug_src)
    require(len(aug_ast.body)==1 and isinstance(aug_ast.body[0],ast.ClassDef),
            "unexpected Augmentor source contents")
    exec(compile(aug_ast,"s9_authors_NetAugment","exec"),namespace)
    aug=namespace["Augmentor"]()
    # Quantify whether the notebook's range(max) index corresponds to the
    # histogram's actual support. This audit does not silently change the CDF.
    support=Counter(burst_sizes)
    probs=[]
    for raw in np.linspace(0,1-1e-12,num=1000):
        j=bisect.bisect_left(cdf,float(raw))
        if j<len(aug.outgoing_burst_sizes):
            probs.append(int(aug.outgoing_burst_sizes[j]))
    invalid_possible=sum(int(v<=0 or v not in support) for v in probs)
    random.seed(42);np.random.seed(42)
    statuses=Counter()
    errors=Counter()
    preview=selected[:128]
    for k,i in enumerate(preview):
        trace=np.asarray(X[int(i)],dtype=np.int32).copy()
        try:
            out=np.asarray(aug.augment(trace))
            if out.shape!=(5000,):
                statuses["invalid_shape"]+=1
            elif not np.isfinite(out).all():
                statuses["nonfinite"]+=1
            elif not np.isin(out,[-1,0,1]).all():
                statuses["invalid_direction"]+=1
            else:
                statuses["valid_direction_shape"]+=1
        except Exception as exc:
            statuses["exception"]+=1
            errors[type(exc).__name__]+=1
    report={
      "classification":"S9_AUTHOR_NETCLR_HISTORICAL_AUGMENT_CPU_DIAGNOSTIC",
      "source_commit":gate.COMMIT,
      "source_notebook_sha256":provenance["pretrain_notebook_sha256"],
      "cache_manifest_sha256":b.P.sha(b.CACHE/"manifest.json"),
      "historical_training_only":True,
      "training_population":39258,
      "sampled_historical_train_rows":1000,
      "sample_seed":42,
      "sample_row_sha256":__import__("hashlib").sha256(selected.tobytes()).hexdigest(),
      "bursts_positive":len(burst_sizes),
      "unique_burst_sizes":len(support),
      "max_burst_size":maximum,
      "observed_trace_min":min(observed_lengths),
      "observed_trace_max":max(observed_lengths),
      "outgoing_cdf_len":len(cdf),
      "non_positive_or_unobserved_bins_in_1000_quantiles":invalid_possible,
      "original_augmentation_on_128_training_traces":dict(statuses),
      "exception_types":dict(errors),
      "augmentation_fidelity_approved":False,
      "cpu_only":True,"gpu_training_started":False,
      "external_datasets_opened":False,"dataset_c_opened":False,
    }
    print("S9_AUTHORS_AUGMENT_DIAGNOSTIC",json.dumps(report,indent=2),flush=True)
    if statuses["exception"] or statuses["invalid_shape"] or statuses["nonfinite"] or statuses["invalid_direction"]:
        print("S9_AUTHORS_AUGMENT_RUNTIME_BLOCKED",flush=True)
    elif invalid_possible:
        print("S9_AUTHORS_AUGMENT_CDF_SEMANTICS_REVIEW_REQUIRED",flush=True)
    else:
        print("S9_AUTHORS_AUGMENT_RUNTIME_SMOKE_PASS_NOT_FIDELITY_PROOF",flush=True)
    if args.write:
        require(not AUDIT_OUT.exists() or AUDIT_OUT.is_dir(),
                "unexpected audit path")
        AUDIT_OUT.mkdir(parents=True,exist_ok=True)
        p=AUDIT_OUT/"authors_augment_audit_v1.json"
        require(not p.exists(),"audited report exists; no overwrite")
        p.write_text(json.dumps(report,indent=2)+"\n")
        print("S9_AUTHOR_AUGMENT_AUDIT_SAVED",str(p),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--write",action="store_true")
    args=parser.parse_args()
    run()
