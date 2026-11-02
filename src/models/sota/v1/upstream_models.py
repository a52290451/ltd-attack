"""Load the EXACT upstream WFlib DF / VarCNN architectures from a pinned checkout.

No copying third-party model code into LTD-Attack, preventing stale duplicates.
Inspect git commit+blob before executing upstream Python. External clone resides
outside repository in ~/ltd-ops/sota/upstream/WFlib; never opens A/B/C datasets.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import types

WFLIB_COMMIT="537ec8e03c805540b04b95b144fa158ce50aa255"
WFLIB_REMOTE="https://github.com/FIND-Lab/Website-Fingerprinting-Library"
DF_ORIGINAL_COMMIT="38df0c15a089f13228e4df06bd5269c8a37340fa"
WFLIB_BLOBS={
    "DF":"8e084f556dd8dadac8fc5a7260b96d6de9f517f8",
    "VarCNN":"c93fb5d73497fabc9e5065d9ca82ea6d881298db"
}
WFLIB_PATHS={
    "DF":"WFlib/models/DF.py",
    "VarCNN":"WFlib/models/VarCNN.py"
}
# LTD repo is the only code umbrella. During the no-copy migration we
# support the existing verified checkout; once it is moved into the repo
# workspace, resolve it there without needing two clones.
LTD_ROOT=Path(__file__).resolve().parents[4]
LEGACY_CHECKOUT=Path.home()/"ltd-ops"/"sota"/"upstream"/"WFlib"

def resolve_checkout(root:Path=LTD_ROOT,legacy:Path=LEGACY_CHECKOUT)->Path:
    target=Path(root)/".vendor"/"WFlib"
    if (target/".git").is_dir():
        return target
    if (Path(legacy)/".git").is_dir():
        return Path(legacy)
    return target

DEFAULT_CHECKOUT=resolve_checkout()

def _run(args):
    r=subprocess.run(args,capture_output=True,text=True,timeout=20)
    if r.returncode!=0:
        raise RuntimeError("SOTA_S3_UPSTREAM_GIT_FAILED "+r.stderr[-250:])
    return r.stdout.strip()

def verify_checkout(checkout:Path=DEFAULT_CHECKOUT):
    checkout=Path(checkout)
    if not (checkout/".git").exists():
        raise RuntimeError("SOTA_S3_WFLIB_MISSING "+str(checkout))
    head=_run(["git","-C",str(checkout),"rev-parse","HEAD"])
    if head!=WFLIB_COMMIT:
        raise RuntimeError("SOTA_S3_UPSTREAM_COMMIT_NOT_PINNED "+head)
    dirty=_run(["git","-C",str(checkout),"status","--porcelain",
                "--untracked-files=no"])
    if dirty:raise RuntimeError("SOTA_S3_UPSTREAM_TRACKED_FILES_DIRTY")
    evidence={}
    for model,relpath in WFLIB_PATHS.items():
        p=checkout/relpath
        if not p.is_file() or p.is_symlink():
            raise RuntimeError("SOTA_S3_UPSTREAM_SOURCE_MISSING "+relpath)
        blob=_run(["git","-C",str(checkout),"hash-object","--",relpath])
        if blob!=WFLIB_BLOBS[model]:
            raise RuntimeError("SOTA_S3_UPSTREAM_BLOB_MISMATCH "+relpath)
        evidence[model]={"path":str(p),"blob_sha1_git":blob}
    return {"commit":head,"models":evidence,
            "source":"FIND-Lab/Website-Fingerprinting-Library",
            "faithful_original_paper_implementation_proven":False}

def get_model(name:str,num_classes:int=65,checkout:Path=DEFAULT_CHECKOUT):
    if name not in WFLIB_PATHS:raise ValueError("model must be DF or VarCNN")
    if num_classes!=65:raise ValueError("frozen SOTA requires exactly 65 classes")
    verified=verify_checkout(checkout)
    # Execute the exact pinned upstream file, after verifying its git blob.
    # This is deliberately not dynamically downloaded at runtime.
    source=Path(verified["models"][name]["path"])
    module=types.ModuleType("ltd_external_sota_"+name)
    module.__file__=str(source)
    code=compile(source.read_bytes(),str(source),"exec")
    exec(code,module.__dict__)
    cls=getattr(module,name)
    return cls(num_classes=num_classes)

def audit_signatures():
    """Static configuration contract, no torch imports, no model loading."""
    return {
        "upstream_commit":WFLIB_COMMIT,
        "upstream_blobs":dict(WFLIB_BLOBS),
        "model_inputs":{
            "DF":{"shape":["N",1,5000],"feature":"DIR","channels":1},
            "VarCNN":{"shape":["N",2,5000],"feature":"DT2","channels":2}
        },
        "output":{"shape":["N",65],"type":"logits_unormalized"},
        "no_network_or_training":True,
        "faithful_original_df_paper":"NOT_ASSERTED_WFLIB_VARIANT",
        "varcnn_dt2_semantic_gate":"PENDING_TIME_VECTOR_AUDIT"
    }
