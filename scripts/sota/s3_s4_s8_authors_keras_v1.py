#!/usr/bin/env python3
"""Author-code Keras SOTA S3 DF, S4 VarCNN(direction), S8 TikTok.

Imports exact class/function bodies extracted by AST from PINNED authors'
source files, not WFlib. Legacy Keras compatibility is applied outside the
layer definitions. Each run is isolated and prints source-version provenance.
Mode B: Early+Middle fit / Late macro-F1; C: all Historical DEV fixed B epochs;
eval: UID-aligned frozen 09B/Future-B. No Dataset C.
S4 is explicitly ORIGINAL VARCNN DIRECTION-BRANCH ONLY, not full ensemble.
GPU training is blocked if other compute jobs are present.
"""
from __future__ import annotations
import argparse,ast,csv,hashlib,importlib.util,json,os,random,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
BASE=Path.home()/"ltd-ops/sota/upstream_original_2026"
OUT=Path("/data/ltd-attack/sota")
REGISTRY={
 "DF":{"repo":"S3_DF_Authors","commit":"38df0c15a089f13228e4df06bd5269c8a37340fa",
       "path":"src/Model_NoDef.py","names":["DFNet"],"input":"direction",
       "architecture":"authors_DF_original_direction"},
 "VarCNN":{"repo":"S4_VarCNN_Authors","commit":"e5db76a86fcbf8839f764b9aca241749d1c1f700",
       "path":"var_cnn.py","names":["dilated_basic_1d","basic_1d","ResNet18","get_model"],
       "input":"direction","architecture":"authors_VarCNN_direction_branch_only"},
 "TikTok":{"repo":"S8_TikTok_Authors","commit":"0aa56660b435d88f3542939b3f08c0ba1409dbf5",
       "path":"DL_Experiments/cw_attack.py","names":["ConvNet"],
       "input":"directional_timing","architecture":"authors_TikTok_DT_original_classifier"},
}
COHORTS={"09B":14687,"10B_DEV_FROZEN":18543}
S5=Path("/data/ltd-attack/sota/s5/frozen_s5_refits_and_labels_v1.json")

def require(ok,msg):
    if not ok:raise RuntimeError("SOTA_AUTHORS_KERAS_BLOCKED "+msg)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
    return h.hexdigest()
def model_source(name):
    x=REGISTRY[name]
    base=BASE/x["repo"]
    r=subprocess.run(["git","-C",str(base),"rev-parse","HEAD"],
                     capture_output=True,text=True,check=True)
    require(r.stdout.strip()==x["commit"],"unpinned author source")
    require(not subprocess.run(["git","-C",str(base),"status","--porcelain",
                                "--untracked-files=no"],
                 capture_output=True,text=True,check=True).stdout.strip(),
            "authors checkout modified")
    path=base/x["path"]
    text=path.read_text(encoding="utf-8-sig")
    tree=ast.parse(text)
    chosen=[node for node in tree.body
            if isinstance(node,(ast.ClassDef,ast.FunctionDef))
            and node.name in x["names"]]
    require({z.name for z in chosen}==set(x["names"]),
            "missing original author class/function")
    code="\n\n".join(ast.get_source_segment(text,z) for z in chosen)
    return code,sha(path)

def keras_space():
    import keras
    from keras import layers
    from keras.models import Sequential,Model
    from keras.initializers import glorot_uniform
    from keras.callbacks import ReduceLROnPlateau,EarlyStopping,ModelCheckpoint
    def adamax(*args,**kwargs):
        kwargs.pop("decay",None)
        if "lr" in kwargs:kwargs["learning_rate"]=kwargs.pop("lr")
        return keras.optimizers.Adamax(*args,**kwargs)
    return {"keras":keras,"np":np,"Sequential":Sequential,"Model":Model,
            "Input":keras.Input,"Conv1D":layers.Conv1D,"MaxPooling1D":layers.MaxPooling1D,
            "BatchNormalization":layers.BatchNormalization,"Activation":layers.Activation,
            "Flatten":layers.Flatten,"Dense":layers.Dense,"Dropout":layers.Dropout,
            "ELU":layers.ELU,"glorot_uniform":glorot_uniform,"Adamax":adamax,
            "Adam":keras.optimizers.Adam,
            "ZeroPadding1D":layers.ZeroPadding1D,
            "GlobalAveragePooling1D":layers.GlobalAveragePooling1D,
            "Add":layers.Add,"Concatenate":layers.Concatenate,
            "ReduceLROnPlateau":ReduceLROnPlateau,
            "EarlyStopping":EarlyStopping,"ModelCheckpoint":ModelCheckpoint,
            "parameters":{"kernel_initializer":"he_normal"}}

def make_model(name):
    src,digest=model_source(name)
    ns=keras_space()
    if name=="VarCNN":
        # Authors' get_model creates legacy val_acc callbacks and a .h5
        # checkpoint not supported by modern Keras; create the authors'
        # original direction branch using their unmodified ResNet18 function.
        # This is NOT the full two-branch/timing/metadata ensemble.
        selected="\n\n".join(ast.get_source_segment(src,node)
                            for node in ast.parse(src).body
                            if getattr(node,"name",None)!="get_model")
        exec(compile(ast.parse(selected),"authors_varcnn_body","exec"),ns)
        keras=ns["keras"]
        inp=keras.Input(shape=(5000,1),name="dir_input")
        features=ns["ResNet18"](inp,"dir",block=ns["dilated_basic_1d"])
        logits=ns["Dense"](65,activation="softmax",name="model_output")(features)
        return keras.Model(inp,logits),digest
    tree=ast.parse(src)
    require(all(isinstance(x,ast.ClassDef) for x in tree.body),
            "unexpected executable source statement")
    exec(compile(tree,"pinned_authors_original","exec"),ns)
    if name=="DF":
        return ns["DFNet"].build(input_shape=(5000,1),classes=65),digest
    return ns["ConvNet"].build(classes=65,input_shape=(5000,1)),digest

def seed():
    random.seed(42);np.random.seed(42)
    import keras
    keras.utils.set_random_seed(42)
def guard_gpu():
    r=subprocess.run(["nvidia-smi","--query-compute-apps=pid",
                      "--format=csv,noheader,nounits"],
                     capture_output=True,text=True,timeout=15)
    require(r.returncode==0 and not r.stdout.strip(),
            "GPU busy: "+r.stdout.strip())

def local_data(name):
    if name=="TikTok":
        from scripts.sota import s8_tiktok_train_b_v1 as D
        X,y,s,_,m=D.load_data()
        return X,y,s,{"source_cache_sha256":sha(D.OUT/"historical_cache_v1/manifest.json")}
    from scripts.sota import s9_netclr_supervised_b_v1 as D
    X,y,s,m=D.prepare_cache()
    return X,y,s,{"source_cache_sha256":sha(D.CACHE/"manifest.json")}

def config(name,source_sha):
    return {"model":name,"author_architecture":REGISTRY[name]["architecture"],
            "source_repository":REGISTRY[name]["repo"],
            "commit":REGISTRY[name]["commit"],"source_sha256":source_sha,
            "feature":REGISTRY[name]["input"],"classes":65,"seed":42,
            "batch_size":64,"max_epochs":30,"train_splits":["DEV_EARLY","DEV_MIDDLE"],
            "selection":"DEV_LATE_macro_f1_max_earliest_tie",
            "full_author_ensemble":False if name=="VarCNN" else True,
            "no_external_tuning":True,"dataset_c_opened":False}

def train(name,stage):
    guard_gpu()
    X,y,s,prov=local_data(name)
    require(X.shape==(57916,5000) and y.shape==(57916,)
            and s.shape==(57916,),"Historical source shape")
    require(dict(zip(*np.unique(s,return_counts=True)))==
            {0:19098,1:20160,2:18658},"historical split count")
    require(set(map(int,np.unique(y)))==set(range(65)),
            "historical label coverage")
    folder=OUT/("authors_"+name.lower()+"_seed42")/stage
    require(not folder.exists(),"output directory already exists")
    if stage=="C":
        prior=OUT/("authors_"+name.lower()+"_seed42")/"B/manifest.json"
        require(prior.is_file(),"complete B required")
        evidence=json.loads(prior.read_text())
        require(evidence["classification"]=="AUTHOR_SOTA_HISTORICAL_B_V1",
                "B provenance mismatch")
        epochs=evidence["selected_epoch"]
        require(1<=epochs<=30,"B epoch invalid")
    else:
        epochs=30
    seed()
    model,digest=make_model(name)
    cfg=config(name,digest)
    model.compile(optimizer=keras_space()["keras"].optimizers.Adamax(
                    learning_rate=0.002) if name!="VarCNN"
                  else keras_space()["keras"].optimizers.Adam(learning_rate=0.001),
                  loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    tr=np.flatnonzero(s!=2) if stage=="B" else np.arange(57916)
    va=np.flatnonzero(s==2) if stage=="B" else np.empty(0,dtype=int)
    import keras
    class Best(keras.callbacks.Callback):
        def __init__(self):
            super().__init__();self.records=[];self.best=-1;self.epoch=0;self.weights=None
        def on_epoch_end(self,epoch,logs=None):
            probs=self.model.predict(np.asarray(X[va],dtype=np.float32)[:,:,None],
                                     batch_size=256,verbose=0)
            preds=probs.argmax(1)
            gold=np.asarray(y[va],dtype=int)
            from sklearn.metrics import f1_score
            f1=float(f1_score(gold,preds,labels=np.arange(65),average="macro"))
            rec={"epoch":epoch+1,"validation_accuracy":float(np.mean(preds==gold)),
                 "validation_macro_f1":f1}
            self.records.append(rec)
            print("AUTHORS_SOTA_B_EPOCH",json.dumps(rec),flush=True)
            if f1>self.best:
                self.best=f1;self.epoch=epoch+1
                self.weights=self.model.get_weights()
    callbacks=[]
    best=Best()
    if stage=="B":callbacks.append(best)
    folder.mkdir(parents=True,mode=0o700)
    input_x=np.asarray(X[tr],dtype=np.float32)[:,:,None]
    input_y=np.asarray(y[tr],dtype=np.int64)
    print("AUTHORS_SOTA_TRAIN_BEGIN",name,stage,len(tr),epochs,flush=True)
    history=model.fit(input_x,input_y,batch_size=64,epochs=epochs,
                      shuffle=True,verbose=2,callbacks=callbacks)
    if stage=="B":
        require(best.epoch>0,"no validation epoch selected")
        model.set_weights(best.weights)
        selected=best.epoch
    else:selected=epochs
    checkpoint=folder/"model.keras"
    model.save(checkpoint)
    manifest={"classification":"AUTHOR_SOTA_HISTORICAL_"+stage+"_V1",
              "config":cfg,"training_n":len(tr),
              "validation_n":len(va),"selected_epoch":selected,
              "checkpoint":str(checkpoint),"checkpoint_sha256":sha(checkpoint),
              "history":best.records if stage=="B" else history.history,
              "historical_cache":prov,"external_data_opened":False,
              "dataset_c_opened":False,"created_utc":datetime.now(timezone.utc).isoformat()}
    (folder/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print("AUTHORS_SOTA_TRAIN_PASS",name,stage,selected,sha(checkpoint),flush=True)

def evaluate(name,cohort):
    require(cohort in COHORTS,"unrecognized external cohort")
    import keras
    basedir=OUT/("authors_"+name.lower()+"_seed42")
    dest=basedir/"evaluation"/cohort
    require(not dest.exists(),"cannot overwrite existing evaluation")
    mp=basedir/"C/manifest.json"
    require(mp.is_file(),"frozen C checkpoint required")
    c=json.loads(mp.read_text())
    require(c["classification"]=="AUTHOR_SOTA_HISTORICAL_C_V1",
            "C source mismatch")
    ck=Path(c["checkpoint"])
    require(sha(ck)==c["checkpoint_sha256"],"C checkpoint altered")
    frozen=json.loads(S5.read_text())
    ref=frozen["label_alignment"][cohort]
    require(ref["identity_order"] is True,"external class order changed")
    p=Path(ref["source"])
    require(sha(p)==ref["source_sha256"],"canonical cohort source hash")
    with np.load(p,allow_pickle=True) as z:
        uid=z["pcap_uid"].astype(str)
        dates=z["query_date"].astype("datetime64[D]").astype(str)
        y=z["y_true"].astype(np.int64)
        labels=z["candidate_labels"].astype(str)
    n=COHORTS[cohort]
    require(len(uid)==n and len(set(uid))==n and len(labels)==65,
            "external UID/label coverage")
    expected={str(k):str(labels[int(v)]) for k,v in zip(uid,y)}
    src=(Path.home()/"ltd-storage/datasets/historical/CLEAN_final_vectors_sites.csv"
         if cohort=="09B" else
         Path.home()/"ltd-storage/datasets/future/CLEAN_final_vectors_sites_concept_drift.csv")
    if cohort=="09B":
        from scripts.sota.s9_netclr_supervised_b_v1 import source
        _,_,s1,_=source()
        reference_hash=s1["historical_checksums"]["historical_vectors"]["sha256"]
    else:
        fm=json.loads((ROOT/"docs/evidence/FINAL/FUTURE-B/10B_manifest_future_b.json").read_text())
        reference_hash=fm["future_inputs"]["micro"]["sha256"]
    require(sha(src)==reference_hash,"external vector source changed")
    from src.models.sota.v1.data_adapter import parse_direction,fixed_length
    from src.models.sota.v1.timing_adapter import parse_times
    from src.models.sota.v1.tiktok_adapter import tiktok_dt_from_sequences
    vectors={}
    csv.field_size_limit(sys.maxsize)
    with src.open(newline="",encoding="utf-8") as fh:
        rd=csv.DictReader(fh)
        for row in rd:
            key=row["pcap_uid"]
            if key not in expected:continue
            require(key not in vectors and row["site_label"]==expected[key],
                    "UID/label mismatch")
            ds=parse_direction(row["direction_vector"])
            if name=="TikTok":
                ts=parse_times(row["time_vector"])
                values=tiktok_dt_from_sequences(ds,ts)
            else:values=fixed_length(ds)
            vectors[key]=np.asarray(values,dtype=np.float32)
    require(len(vectors)==n,"missing cohort vectors")
    model=keras.models.load_model(ck,compile=False)
    out=np.empty((n,65),dtype=np.float32)
    for start in range(0,n,256):
        stop=min(n,start+256)
        b=np.stack([vectors[k] for k in uid[start:stop]])[:,:,None]
        out[start:stop]=model.predict(b,batch_size=128,verbose=0)
    require(np.isfinite(out).all() and
            np.allclose(out.sum(1),1,atol=2e-4,rtol=0),
            "invalid probabilities")
    rank=np.argsort(-out,axis=1)
    top=rank[:,0]
    pos=(rank==y[:,None]).argmax(axis=1)+1
    tp=np.bincount(y[top==y],minlength=65)
    support=np.bincount(y,minlength=65)
    pc=np.bincount(top,minlength=65)
    macro=float(np.divide(2*tp,support+pc,out=np.zeros(65),
                         where=support+pc>0).mean())
    metrics={"accuracy":float((top==y).mean()),"macro_f1":macro,
             "top5":float((pos<=5).mean()),"mrr":float((1/pos).mean())}
    dest.mkdir(parents=True,mode=0o700)
    scores=dest/"scores_sota_aligned.npz"
    np.savez_compressed(scores,pcap_uid=uid,query_date=dates,y_true=y,
                        candidate_labels=labels,sota_probs=out)
    info={"classification":"AUTHOR_SOTA_EXTERNAL_FROZEN_V1",
          "model":name,"cohort":cohort,"n":n,
          "architecture":REGISTRY[name]["architecture"],
          "metrics":metrics,"score_sha256":sha(scores),
          "C_manifest_sha256":sha(mp),"no_external_tuning":True,
          "dataset_c_opened":False}
    (dest/"manifest.json").write_text(json.dumps(info,indent=2)+"\n")
    print("AUTHOR_SOTA_EXTERNAL_PASS",name,cohort,json.dumps(metrics),flush=True)

def main():
    a=argparse.ArgumentParser()
    a.add_argument("--model",choices=list(REGISTRY),required=True)
    a.add_argument("--stage",choices=("smoke","B","C","evaluate"),required=True)
    a.add_argument("--cohort",choices=list(COHORTS))
    args=a.parse_args()
    if args.stage=="smoke":
        seed()
        m,s=make_model(args.model)
        require(m.output_shape[-1]==65,"65-class output requirement")
        print("AUTHOR_KERAS_MODEL_SMOKE_PASS",args.model,
              "sha256",s,"output_shape",m.output_shape,flush=True)
    elif args.stage in ("B","C"):train(args.model,args.stage)
    else:
        require(args.cohort is not None,"cohort required")
        evaluate(args.model,args.cohort)

if __name__=="__main__":
    main()
