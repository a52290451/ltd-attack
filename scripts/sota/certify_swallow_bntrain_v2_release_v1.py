#!/usr/bin/env python3
"""Validate Swallow BNtrain v2 against immutable V1, and enforce complete
historical-only staged artifacts before allowing use as a new baseline.
Does not change any model or pre-existing evidence.
"""
from __future__ import annotations
import argparse,ast,hashlib,json,sys
from pathlib import Path
import numpy as np
def require(ok,msg):
    if not ok:raise RuntimeError("SWALLOW_V2_RELEASE_BLOCKED "+msg)
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--repo",type=Path,required=True)
    p.add_argument("--base",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    require(not a.output.exists(),"output exists")
    sys.path.insert(0,str(a.repo))
    from scripts.sota import s11_swallow_authors_bntrain_v2 as v2
    from scripts.sota import s11_swallow_authors_v1 as v1
    c1=dict(v1.CONFIG);c2=dict(v2.CONFIG)
    for key in ("target_mode","historical_v2","method"):
        c1.pop(key,None);c2.pop(key,None)
    require(c1==c2,"V2 changed other frozen pretrain parameters unexpectedly")
    code_v2=(a.repo/"scripts/sota/s11_swallow_authors_bntrain_v2.py").read_text()
    code_v1=(a.repo/"scripts/sota/s11_swallow_authors_v1.py").read_text()
    require("target.eval();t1=target(v1);t2=target(v2)" in code_v1,
            "V1 target mode unavailable")
    require("target.train();t1=target(v1);t2=target(v2)" in code_v2,
            "V2 target train mode unavailable")
    require(v2.OUT!=v1.OUT and v2.CACHE==v1.CACHE,
            "v2 not isolated or reused cached original source")
    target=v2.OUT
    checks={}
    for stage,folder in (("pretrain","pretrain_B"),
                         ("B","supervised_B"),
                         ("C","supervised_C")):
        mpath=target/folder/"manifest.json"
        require(mpath.exists(),"missing "+str(mpath))
        m=json.loads(mpath.read_text())
        require(m["config"]==v2.CONFIG,"V2 "+stage+" config mismatch")
        ck=Path(m["checkpoint"])
        require(ck.exists() and sha(ck)==m["checkpoint_sha256"],
                "V2 "+stage+" checkpoint digest invalid")
        require(m.get("external_data_opened") is False and
                m.get("dataset_c_opened") is False,
                "V2 external data opened flag")
        expected=39258 if stage in ("pretrain","B") else 57916
        require(m["training_n"]==expected,"V2 "+stage+" population")
        checks[stage]={"checkpoint_sha256":sha(ck),
                       "training_n":expected,
                       "epoch":m.get("selected_epoch",100)}
        if stage=="pretrain":
            probe=m.get("probe_monitor",[])
            require(len(probe)==100 and all(z["feature_std"]>=1e-7 and
                    z["unique"]>=3 for z in probe[2:]),
                    "V2 pretrain collapsed during historical probe")
    require(checks["B"]["epoch"]==checks["C"]["epoch"],
            "B/C epoch selection mismatch")
    for cohort,n in (("09B",14687),("10B_DEV_FROZEN",18543)):
        score=target/"evaluation"/cohort/"scores_sota_aligned.npz"
        manifest=target/"evaluation"/cohort/"manifest.json"
        require(score.is_file() and manifest.is_file(),
                "V2 evaluation incomplete "+cohort)
        m=json.loads(manifest.read_text())
        require(m["score_sha256"]==sha(score),"V2 cohort score SHA mismatch")
        with np.load(score,allow_pickle=False) as z:
            probs=z["sota_probs"]
            uid=z["pcap_uid"].astype(str)
            names=z["candidate_labels"].astype(str)
            y=z["y_true"].astype(int)
            dates=z["query_date"].astype("datetime64[D]").astype(str)
        require(probs.shape==(n,65) and len(set(uid))==n and
                len(set(names))==65 and len(np.unique(dates))>0,
                "V2 score shape/population")
        require(np.isfinite(probs).all() and
                np.allclose(probs.sum(1),1,atol=2e-4,rtol=0),
                "V2 probs invalid")
        divers=int(len(np.unique(probs.argmax(1))))
        acc=float(np.mean(probs.argmax(1)==y))
        require(divers>=20 and acc>1/65*1.5,
                "V2 predictions low-diversity or approximately chance on "+cohort)
        checks[cohort]={"n":n,"accuracy":acc,"predicted_classes":divers,
                        "score_sha256":sha(score)}
    output={"classification":"LTD_SWALLOW_TARGET_BNTRAIN_V2_HISTORICAL_RELEASE_GATE",
            "status":"PASS_ADAPTED_SWALLOW_V2_NONCOLLAPSED_TECHNICAL_GATE_NOT_PAPER_EXACT",
            "frozen_v1_unchanged":True,"v2_distinct_output":str(target),
            "method_change":"only target mode train() vs V1 eval(); pretrain batch32/100 epochs retain adapted protocol",
            "checks":checks,"paper_exact":False,
            "limitations":["Historical model selection only; Future-B never tuned.",
                           "Pretraining batch, epochs and augmentation multiplier differ from original Swallow paper.",
                           "Acceptance thresholds are operational noncollapse heuristics, not fidelity proof.",
                           "Recheck canonical S5/S6 label and row parity and paired CIs separately."]}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(output,indent=2)+"\n")
    print("SWALLOW_V2_RELEASE_STATUS",output["status"])
    print("SWALLOW_V2_RELEASE_REPORT",a.output)
if __name__=="__main__":main()
