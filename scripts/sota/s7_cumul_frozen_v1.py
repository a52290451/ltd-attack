#!/usr/bin/env python3
"""S7 CUMUL-style 104-feature SVM-RBF on Historical DEV (no feature leakage).

One command each for B, C, evaluate; DOES NOT use old 94-feature list.
B: DEV_EARLY+MIDDLE -> DEV_LATE; C: refit all Historical DEV.
Evaluate only after C frozen: 09B and Future-B DEV_FROZEN by UID.
Uses real full TCP frame lengths and signed directions from the original
Historical/Future CSV. Described as CUMUL-style until packet semantics and
reference source implementation are independently audited for paper fidelity.
No Dataset C access. No training or tuning from external scores.
"""
from __future__ import annotations
import argparse, ast, csv, hashlib, json, math, sys, time
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.models.sota.v1.data_adapter import load_s1_dev_index,parse_direction
from src.models.sota.v1.s4_protocol import sha256_file

S1=Path.home()/"ltd-ops/sota/s1_dev_frozen_20261102T130232Z/s1_manifest.json"
S5=Path("/data/ltd-attack/sota/s5/frozen_s5_refits_and_labels_v1.json")
OUTPUT=Path("/data/ltd-attack/sota/s7_cumul_svm_rbf_seed42")
CUMUL_FEATURES=104
COUNTS={"DEV_EARLY":19098,"DEV_MIDDLE":20160,"DEV_LATE":18658}
COHORT_N={"09B":14687,"10B_DEV_FROZEN":18543}
CONFIG={"classification":"CUMUL_STYLE_104_RBF_SVM_DEV_ONLY_V1",
        "feature_definition":"4 signed-size count/sum + 100 cumulative byte interpolation",
        "frame_length":"scapy_len_packet_TCP_IP_only","sign":"incoming-positive",
        "C":1.0,"gamma":"scale","kernel":"rbf","probability":True,
        "random_state":42,"scaler":"StandardScaler","seed":42,
        "validation":"early+middle to late; no hyperparameter selection",
        "input":"full recorded packet sequences; no invented padding"}

def check(ok,msg):
    if not ok:raise RuntimeError("S7_CUMUL_BLOCKED "+msg)

def parse_sizes(s):
    check(isinstance(s,str) and s.startswith("[") and s.endswith("]"),
          "size_vector is not a bracketed list")
    a=ast.literal_eval(s)
    check(isinstance(a,(list,tuple)) and len(a)>0,"empty/invalid size_vector")
    check(all(type(v) in (int,float) and math.isfinite(v)
              and v>0 and float(v).is_integer() for v in a),
          "nonpositive/noninteger original packet size")
    return np.asarray(a,dtype=np.float64)

def cumul104(direction_text,size_text):
    directions=np.asarray(parse_direction(direction_text),dtype=np.int8)
    sizes=parse_sizes(size_text)
    check(len(directions)==len(sizes) and
          np.all(np.isin(directions,[-1,1])),
          "misaligned direction and original TCP packet sizes")
    # Dataset: +1 is outgoing. CUMUL original definition: p>0 incoming.
    incoming=directions==-1
    sign=np.where(incoming,1.0,-1.0)
    signed=sizes*sign
    # Start curve at (0,0), x: cumulative absolute packet bytes,
    # y: cumulative signed packet bytes, then interpolate 100 samples.
    cumulative_bytes=np.r_[0.0,np.cumsum(sizes)]
    cumulative_signed=np.r_[0.0,np.cumsum(signed)]
    samples=np.linspace(0.0,float(cumulative_bytes[-1]),100)
    interp=np.interp(samples,cumulative_bytes,cumulative_signed)
    stats=np.array([incoming.sum(),(~incoming).sum(),
                    sizes[incoming].sum(),sizes[~incoming].sum()],
                   dtype=np.float64)
    out=np.concatenate([stats,interp])
    check(out.shape==(CUMUL_FEATURES,) and np.isfinite(out).all(),
          "invalid CUMUL feature shape/values")
    return out.astype(np.float32)

def historic_arrays():
    uids,classes,s1=load_s1_dev_index(S1)
    source=Path(s1["historical_checksums"]["historical_vectors"]["path"])
    check(source.is_file(),"historical source absent")
    check(sha256_file(source)==s1["historical_checksums"]["historical_vectors"]["sha256"],
          "historical source bytes not frozen")
    vectors=np.empty((57916,104),dtype=np.float32)
    y=np.empty(57916,dtype=np.int64)
    splits=[]
    seen=set()
    csv.field_size_limit(sys.maxsize)
    with source.open(newline="",encoding="utf-8") as f:
        reader=csv.DictReader(f)
        check({"pcap_uid","site_label","direction_vector","size_vector"}<=set(
            reader.fieldnames or []),"historical vector columns absent")
        for row in reader:
            uid=row["pcap_uid"]
            if uid not in uids:continue
            check(uid not in seen,"duplicate historical UID")
            site,date,split=uids[uid]
            check(site==row["site_label"],"historical UID/site mismatch")
            i=len(seen)
            vectors[i]=cumul104(row["direction_vector"],row["size_vector"])
            y[i]=classes[site]
            splits.append(split);seen.add(uid)
            if len(seen)%10000==0:
                print("S7_CUMUL_DEV",len(seen),57916,flush=True)
    check(len(seen)==57916 and seen==set(uids),"Historical DEV population mismatch")
    check(dict(Counter(splits))==COUNTS,"Historical DEV chronological counts mismatch")
    return vectors,y,np.asarray(splits),s1,source

def train(stage):
    import joblib
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.metrics import f1_score
    check(stage in ("B","C"),"invalid training stage")
    target=OUTPUT/stage
    check(not target.exists(),"refuse to overwrite scientific run")
    if stage=="C":
        b=OUTPUT/"B"/"manifest.json"
        check(b.is_file(),"B validation evidence required before C")
        m=json.loads(b.read_text())
        check(m["stage"]=="B" and m["config"]==CONFIG,
              "B settings differ from frozen CUMUL configuration")
    t0=time.monotonic()
    X,y,split,s1,src=historic_arrays()
    train_mask=np.isin(split,["DEV_EARLY","DEV_MIDDLE"]) if stage=="B" \
               else np.ones(len(y),dtype=bool)
    val_mask=split=="DEV_LATE" if stage=="B" \
             else np.zeros(len(y),dtype=bool)
    check(int(train_mask.sum())==(39258 if stage=="B" else 57916),
          "wrong train size")
    check(int(val_mask.sum())==(18658 if stage=="B" else 0),
          "wrong validation size")
    model=Pipeline([
        ("scaler",StandardScaler()),
        ("svm",SVC(C=1.0,kernel="rbf",gamma="scale",probability=True,
                   random_state=42,cache_size=1024,decision_function_shape="ovr"))
    ])
    print("S7_CUMUL_REAL_TRAIN_BEGIN",stage,"N",int(train_mask.sum()),flush=True)
    model.fit(X[train_mask],y[train_mask])
    check(np.array_equal(model.named_steps["svm"].classes_,np.arange(65)),
          "missing classes")
    val_metrics=None
    if stage=="B":
        pred=model.predict(X[val_mask])
        truth=y[val_mask]
        val_metrics={"accuracy":float((pred==truth).mean()),
                     "macro_f1":float(f1_score(truth,pred,
                                             labels=np.arange(65),average="macro"))}
        print("S7_CUMUL_DEV_LATE",val_metrics,flush=True)
    target.mkdir(parents=True,mode=0o700)
    ck=target/"model.joblib"
    joblib.dump(model,ck,compress=3)
    m={"classification":"S7_CUMUL_STYLE_FROZEN_SVM_V1",
       "stage":stage,"config":CONFIG,
       "created_utc":datetime.now(timezone.utc).isoformat(),
       "training_samples":int(train_mask.sum()),
       "validation_samples":int(val_mask.sum()),
       "validation":val_metrics,"checkpoint":str(ck),
       "checkpoint_sha256":sha256_file(ck),
       "historical_source_sha256":sha256_file(src),
       "S1_label_sha256":s1["label_order_sha256"],
       "elapsed_seconds":time.monotonic()-t0,
       "used_external_samples_for_fit":False,"dataset_c_opened":False,
       "paper_fidelity":"CUMUL_STYLE_NOT_YET_VERIFIED_ORIGINAL"}
    (target/"manifest.json").write_text(json.dumps(m,indent=2)+"\n")
    print("S7_CUMUL_REAL_TRAIN_PASS",stage,ck,flush=True)

def evaluate(cohort):
    import joblib
    check(cohort in COHORT_N,"invalid external cohort")
    target=OUTPUT/"evaluation"/cohort
    check(not target.exists(),"evaluation already exists: refuse overwrite")
    m=json.loads((OUTPUT/"C"/"manifest.json").read_text())
    check(m["stage"]=="C" and m["config"]==CONFIG,
          "C checkpoint/config mismatch")
    check(sha256_file(Path(m["checkpoint"]))==m["checkpoint_sha256"],
          "C checkpoint SHA mismatch")
    frozen=json.loads(S5.read_text())
    reference=frozen["label_alignment"][cohort]
    original=Path(reference["source"])
    check(sha256_file(original)==reference["source_sha256"],
          "canonical NPZ altered")
    with np.load(original,allow_pickle=True) as z:
        uid=z["pcap_uid"].astype(str)
        day=z["query_date"].astype("datetime64[D]").astype(str)
        true=z["y_true"].astype(np.int64)
        labels=z["candidate_labels"].astype(str)
    check(len(uid)==COHORT_N[cohort] and len(set(uid))==len(uid),
          "canonical UID count/uniqueness mismatch")
    check(reference["identity_order"] is True,
          "this CLI needs explicit class permutation before inference")
    actual={str(k):str(labels[int(v)]) for k,v in zip(uid,true)}
    source=Path.home()/("ltd-storage/datasets/historical/CLEAN_final_vectors_sites.csv"
                        if cohort=="09B" else
                        "ltd-storage/datasets/future/CLEAN_final_vectors_sites_concept_drift.csv")
    if cohort=="09B":
        source_sha=m["historical_source_sha256"]
    else:
        f=json.loads((ROOT/"docs/evidence/FINAL/FUTURE-B/10B_manifest_future_b.json").read_text())
        source_sha=f["future_inputs"]["micro"]["sha256"]
    check(sha256_file(source)==source_sha,"external packet source SHA mismatch")
    by_uid={}
    csv.field_size_limit(sys.maxsize)
    with source.open(newline="",encoding="utf-8") as f:
        reader=csv.DictReader(f)
        check({"pcap_uid","site_label","size_vector","direction_vector"}<=
              set(reader.fieldnames or []),"packet feature schema mismatch")
        for row in reader:
            key=row["pcap_uid"]
            if key not in actual:continue
            check(key not in by_uid and row["site_label"]==actual[key],
                  "external UID duplicate or label divergence")
            by_uid[key]=cumul104(row["direction_vector"],row["size_vector"])
    check(set(by_uid)==set(uid),"missing canonical external UIDs")
    X=np.stack([by_uid[k] for k in uid])
    model=joblib.load(Path(m["checkpoint"]))
    check(np.array_equal(model.named_steps["svm"].classes_,np.arange(65)),
          "model class order differs")
    prob=model.predict_proba(X).astype(np.float32)
    check(prob.shape==(len(uid),65) and np.isfinite(prob).all() and
          np.allclose(prob.sum(axis=1),1.,atol=2e-4,rtol=0),
          "invalid probabilities")
    top=np.argsort(-prob,axis=1)
    rank=(top==true[:,None]).argmax(axis=1)+1
    tp=np.bincount(true[top[:,0]==true],minlength=65)
    pred=np.bincount(top[:,0],minlength=65)
    support=np.bincount(true,minlength=65)
    f1=np.divide(2*tp,support+pred,out=np.zeros(65),
                 where=support+pred>0).mean()
    result={"accuracy":float((top[:,0]==true).mean()),
            "macro_f1":float(f1),"top5":float((rank<=5).mean()),
            "mrr":float(np.mean(1./rank))}
    target.mkdir(parents=True,mode=0o700)
    artifact=target/"scores_sota_aligned.npz"
    np.savez_compressed(artifact,pcap_uid=uid,query_date=day,y_true=true,
                        candidate_labels=labels,sota_probs=prob)
    report={"classification":"S7_CUMUL_STYLE_FROZEN_EXTERNAL_V1",
            "cohort":cohort,"model":"CUMUL_STYLE_SVM",
            "checkpoint_sha256":m["checkpoint_sha256"],
            "source_sha256":source_sha,
            "canonical_score_sha256":reference["source_sha256"],
            "score_sha256":sha256_file(artifact),
            "n":len(uid),"metrics":result,
            "no_external_tuning":True,"dataset_c_opened":False}
    (target/"manifest.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S7_CUMUL_EXTERNAL_PASS",cohort,result,flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--stage",choices=["B","C","evaluate"],required=True)
    p.add_argument("--cohort",choices=list(COHORT_N))
    a=p.parse_args()
    if a.stage=="evaluate":
        check(a.cohort is not None,"--cohort required for evaluate")
        evaluate(a.cohort)
    else:train(a.stage)

if __name__=="__main__":
    try:main()
    except Exception as e:
        print("S7_CUMUL_ERROR",type(e).__name__,str(e)[:800],
              file=sys.stderr,flush=True)
        sys.exit(4)
