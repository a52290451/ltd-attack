#!/usr/bin/env python3
"""Zeus portability readiness for LTD original-source SOTA battery.

Checks only immutable evidence and environment paths. No training, GPU use,
data mutation, or automatic copying of large traces. This gate prevents using
S1/S5 JSON with absolute Perseo-only pointers on Zeus by requiring all embedded
referenced files to exist and match their frozen SHA256.

Run on *each* host after rsync and source checkout fetch.
"""
from __future__ import annotations
import argparse,hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.sota import sota_paths_v1 as P

def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda:f.read(4*1024*1024),b""):h.update(block)
    return h.hexdigest()
def require(ok,msg):
    if not ok:raise RuntimeError("SOTA_PORTABILITY_BLOCKED "+msg)
def evidence(p,expected,why):
    p=Path(p)
    require(p.is_file(),"missing "+why+": "+str(p))
    result=sha(p)
    require(result==expected,"sha256 mismatch "+why+": "+str(p))
    return {"path":str(p),"sha256":result,"size_bytes":p.stat().st_size}

def inspect():
    require(P.S1.is_file(),"missing S1 manifest "+str(P.S1))
    require(P.S5.is_file(),"missing S5 manifest "+str(P.S5))
    require(P.FUTURE.is_file(),"missing Future-B manifest "+str(P.FUTURE))
    s1=json.loads(P.S1.read_text())
    s5=json.loads(P.S5.read_text())
    f=json.loads(P.FUTURE.read_text())
    require(s1.get("classification")=="SOTA_S1_HISTORICAL_DEV_ONLY_FROZEN_INDEX",
            "S1 experiment classification mismatch")
    require(s1.get("uid_count")==57916 and s1.get("site_count")==65 and
            s1.get("future_b_not_opened") is True,
            "S1 57,916/65/chronological provenance invalid")
    out={"classification":"LTD_SOTA_HOST_EVIDENCE_READINESS_V1",
         "host":os.uname().nodename,"project_root":str(P.PROJECT_ROOT),
         "ops_root":str(P.OPS),"artifact_root":str(P.ARTIFACT),
         "storage_root":str(P.STORAGE),"s1":str(P.S1),
         "s5":str(P.S5),"future_manifest":str(P.FUTURE),
         "files":{}}
    out["files"]["dev_index"]=evidence(
        s1["dev_index_file"],s1["dev_index_sha256"],"S1 DEV index")
    out["files"]["label_order"]=evidence(
        s1["label_order_file"],s1["label_order_sha256"],"S1 65-class order")
    raw=s1["historical_checksums"]["historical_vectors"]
    out["files"]["historical_vectors"]=evidence(
        raw["path"],raw["sha256"],"S1 canonical Historical vectors")
    ref=s5["label_alignment"]
    for cohort in ("09B","10B_DEV_FROZEN"):
        require(ref[cohort].get("identity_order") is True,
                cohort+" canonical 65-class identity-order mismatch")
        out["files"][cohort+"_reference"]=evidence(
            ref[cohort]["source"],ref[cohort]["source_sha256"],
            cohort+" external reference")
    p=P.STORAGE/"future/CLEAN_final_vectors_sites_concept_drift.csv"
    expected=f["future_inputs"]["micro"]["sha256"]
    out["files"]["future_vectors"]=evidence(p,expected,
                                            "Future-B canonical vectors")
    require((P.ORIGINAL/"S9_NetCLR_Authors/.git").is_dir(),
            "missing pinned legacy author checkouts")
    for model in ("S10_RoFiRe","S11_Swallow","S12_WFCAT"):
        require((P.MODERN/model/".git").is_dir(),
                "missing pinned "+model)
    print("SOTA_HOST_DATA_PROVENANCE_PASS",json.dumps(out,indent=2),flush=True)
    return out

def relocate(a):
    """Make HOST-LOCAL copies of S1/S5 JSON; frozen input bytes never mutate.

    All byte hashes are checked BEFORE writing. S1 and S5 hash identities are
    retained, only absolute path pointers are adjusted. No raw data copied.
    """
    def need(p):require(p is not None and Path(p).is_file(),
                        "missing relocation input "+str(p));return Path(p)
    old1=json.loads(need(a.source_s1).read_text())
    old5=json.loads(need(a.source_s5).read_text())
    future=json.loads(need(a.future_manifest).read_text())
    paths={"index":need(a.dev_index),"order":need(a.label_order),
           "historical":need(a.historical_vectors),
           "09B":need(a.ref_09b),"10B_DEV_FROZEN":need(a.ref_future),
           "future":need(a.future_vectors)}
    evidence(paths["index"],old1["dev_index_sha256"],"DEV index")
    evidence(paths["order"],old1["label_order_sha256"],"label order")
    evidence(paths["historical"],
             old1["historical_checksums"]["historical_vectors"]["sha256"],
             "Historical DEV source")
    for c in ("09B","10B_DEV_FROZEN"):
        evidence(paths[c],old5["label_alignment"][c]["source_sha256"],
                 c+" frozen reference")
    evidence(paths["future"],future["future_inputs"]["micro"]["sha256"],
             "Future-B vectors")
    old1["dev_index_file"]=str(paths["index"].resolve())
    old1["label_order_file"]=str(paths["order"].resolve())
    old1["historical_checksums"]["historical_vectors"]["path"]=str(
        paths["historical"].resolve())
    for c in ("09B","10B_DEV_FROZEN"):
        old5["label_alignment"][c]["source"]=str(paths[c].resolve())
    out=Path(a.output_dir).resolve()
    require(not out.exists(),"relocation output exists; refuse overwrite")
    out.mkdir(parents=True,mode=0o700)
    s1=out/"s1_host_local.json";s5=out/"s5_host_local.json"
    s1.write_text(json.dumps(old1,indent=2)+"\n")
    s5.write_text(json.dumps(old5,indent=2)+"\n")
    env=out/"host_environment.sh"
    root=str(paths["historical"].parent.parent)
    # The evaluator expects a /historical and /future sibling directory tree.
    storage=paths["historical"].parent.parent
    expected_future=storage/"future/CLEAN_final_vectors_sites_concept_drift.csv"
    require(expected_future.resolve()==paths["future"].resolve(),
            "relocated Historical/Future vector paths lack shared storage root")
    import shlex
    overrides={"LTD_SOTA_OPS_ROOT":str(P.OPS),
               "LTD_SOTA_ARTIFACT_ROOT":str(P.ARTIFACT),
               "LTD_SOTA_STORAGE_ROOT":str(storage),
               "LTD_SOTA_S1_MANIFEST":str(s1),
               "LTD_SOTA_S5_MANIFEST":str(s5),
               "LTD_SOTA_FUTURE_MANIFEST":str(need(a.future_manifest).resolve())}
    env.write_text("# Source this host-local, non-secret file before LTD SOTA runs.\n"+
                   "".join("export "+k+"="+shlex.quote(v)+"\n"
                           for k,v in overrides.items()))
    print("SOTA_RELOCATED_FROZEN_METADATA_ONLY",str(out),flush=True)
    print("SOURCE_THIS_FILE_BEFORE_RUNNING",str(env),flush=True)
    return env

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--stage",choices=("inspect","relocate"),required=True)
    ap.add_argument("--source-s1")
    ap.add_argument("--source-s5")
    ap.add_argument("--future-manifest")
    ap.add_argument("--dev-index")
    ap.add_argument("--label-order")
    ap.add_argument("--historical-vectors")
    ap.add_argument("--future-vectors")
    ap.add_argument("--ref-09b")
    ap.add_argument("--ref-future")
    ap.add_argument("--output-dir")
    a=ap.parse_args()
    if a.stage=="inspect":inspect()
    else:
        require(bool(a.output_dir),"--output-dir required")
        relocate(a)
if __name__=="__main__":main()
