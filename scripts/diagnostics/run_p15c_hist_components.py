#!/usr/bin/env python3
"""P15B independent closeout and P15C Historical-only frozen component study.
No training, Future-B, Dataset C, or modification of frozen cache artifacts.
"""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from report_p15a_capture_mechanisms import BASE,CACHE,RESULT,EPS,ORIGINS,WINDOWS,geometric,ranks,read_npz,sha,verify_manifest

P15B=RESULT/"diagnostics/P15B-FROZEN-FUSION-V1"
OUTPUT=RESULT/"diagnostics/P15C-HIST-COMPONENTS-V1"
CONFIG=Path(__file__).resolve().parents[2]/"configs/experiments/P15C_HIST_COMPONENTS_V1.yaml"

def check_15b():
    mf=json.loads((P15B/"P15B_manifest.json").read_text())
    if mf.get("stage")!="P15B_FROZEN_FUSION_ABLATION" or mf.get("future_b_opened") is not False:
        raise RuntimeError("15B_MANIFEST_POLICY_FAILURE")
    required={"P15B_summary.csv","P15B_bootstrap.csv"}
    if set(mf.get("outputs_sha256",{}))!=required:raise RuntimeError("15B_OUTPUT_SET_FAILURE")
    for name,expected in mf["outputs_sha256"].items():
        if sha(P15B/name)!=expected:raise RuntimeError("15B_SHA_FAILURE "+name)
    summary=pd.read_csv(P15B/"P15B_summary.csv")
    if len(summary)!=24:raise RuntimeError("15B_WRONG_ROWCOUNT")
    for variant in ("B1_arithmetic","B2_rank_rrf_k60"):
        actual=summary[(summary.variant==variant)&(summary.window=="FAR")]
        if len(actual)!=2 or not (actual.delta_macro_f1<0).all():
            raise RuntimeError("15B_CONCLUSION_MISMATCH "+variant)
        if mf["gate"][variant]["passed"] is not False:raise RuntimeError("15B_GATE_MISMATCH")
    print("P15B_INDEPENDENT_CLOSEOUT_OK 2/2 SHA; 24 rows; both promotion gates FAIL")

def metric(scores,y):
    r=ranks(scores,y)
    return {"accuracy":float(np.mean(r==1)),
       "macro_f1":float(f1_score(y,np.argmax(scores,axis=1),labels=np.arange(65),average="macro",zero_division=0)),
       "top5_accuracy":float(np.mean(r<=5)),"mrr":float(np.mean(1/r))}

def main():
    import yaml
    conf=yaml.safe_load(CONFIG.read_text())
    assert conf["historical_only"] and not conf["training_allowed"]
    assert not conf["future_b_access"] and not conf["dataset_c_access"]
    check_15b()
    verify_manifest("13B",RESULT/"micro/MICRO-ROBUSTNESS-PHASE13-BATTERY")
    verify_manifest("13C",RESULT/"hybrid/LTD-ROBUSTNESS-PHASE13-BATTERY")
    prior=pd.read_csv(RESULT/"hybrid/LTD-ROBUSTNESS-PHASE13-BATTERY/13C_factorial_summary.csv").set_index(["origin","window","micro_candidate","macro_candidate"])
    rows=[]; byday=[]; bysite=[]; files={}
    for origin in ORIGINS:
        macrofile=CACHE/"macro"/(origin+".npz")
        macro_mf=macrofile.with_suffix(".json")
        metadata=json.loads(macro_mf.read_text())
        if metadata.get("identity")!={"origin":origin} or metadata.get("outputs_sha256",{}).get("cache")!=sha(macrofile):
            raise RuntimeError("MACRO_CACHE_SHA_IDENTITY_MISMATCH "+origin)
        files[str(macrofile)]=sha(macrofile)
        with np.load(macrofile,allow_pickle=False) as archive:
            macro={k:archive[k].copy() for k in archive.files}
        for window in WINDOWS:
            expert=[]
            identity=None
            for seed in (11,42,73):
                p=CACHE/origin/"UNIFORM"/str(seed)/(window+".npz")
                d=read_npz(p,["pcap_uid","query_date","y_true","candidate_labels","probs","origin","window","seed","temporal_mode"])
                if (str(d["origin"]),str(d["window"]),int(d["seed"]),str(d["temporal_mode"]))!=(origin,window,seed,"UNIFORM"):
                    raise RuntimeError("MICRO_ID_MISMATCH")
                current=[d[k] for k in ("pcap_uid","query_date","y_true","candidate_labels")]
                if identity is None:identity=current
                elif not all(np.array_equal(x,y) for x,y in zip(identity,current)):raise RuntimeError("MICRO_ALIGNMENT")
                expert.append(d["probs"])
                files[str(p)]=sha(p)
            uid,dates,y,labels=identity
            if len(labels)!=65 or len(np.unique(uid))!=len(uid):raise RuntimeError("COHORT_FAILURE")
            for key,v in [(f"pcap_uid_{window}",uid),(f"dates_{window}",dates),(f"y_{window}",y),("candidate_labels",labels)]:
                if key not in macro or not np.array_equal(macro[key],v):raise RuntimeError("MACRO_ALIGNMENT "+key)
            micro=geometric(expert)
            m0=macro[f"M0_{window}"].astype(np.float64)
            m1=macro[f"M1_{window}"].astype(np.float64)
            def fuse(a):
                scores=np.exp(.45*np.log(np.clip(micro,EPS,1.))+.55*np.log(np.clip(a,EPS,1.)))
                return scores/scores.sum(axis=1,keepdims=True)
            scores={"U3":micro,"M0":m0,"M1":m1,"U3xM0":fuse(m0),"U3xM1":fuse(m1)}
            for name,s in scores.items():
                vals=metric(s,y)
                rows.append({"origin":origin,"window":window,"candidate":name,"n":len(y),**vals})
                if name.startswith("U3x"):
                    expected=prior.loc[(origin,window,"U3",name.split("x")[1])]
                    for key in vals:
                        if abs(vals[key]-float(expected[key]))>1e-5:
                            raise RuntimeError(f"13C_RECONSTRUCTION_FAIL {origin}/{window}/{name}/{key}")
            for name in ("M0","M1","U3xM0","U3xM1"):
                rk=ranks(scores[name],y); base=ranks(micro,y)
                df=pd.DataFrame({"date":dates.astype("datetime64[D]").astype(str),"site":labels[y].astype(str),
                   "is_correct":rk==1,"in_top5":rk<=5,"rescue_from_micro":(base!=1)&(rk==1),
                   "harm_to_micro":(base==1)&(rk!=1)})
                groupers=[(["date"],byday),(["site"],bysite)]
                for columns,dest in groupers:
                    grouped=df.groupby(columns,as_index=False).agg(n=("is_correct","size"),top1=("is_correct","sum"),
                     top5=("in_top5","sum"),rescues=("rescue_from_micro","sum"),harms=("harm_to_micro","sum"))
                    for row in grouped.to_dict("records"):dest.append({"origin":origin,"window":window,"candidate":name,**row})
            print("15C_COMPONENT_RECONSTRUCTED",origin,window,"N",len(y),"M0",f"{metric(m0,y)['accuracy']:.5f}",
              "M1",f"{metric(m1,y)['accuracy']:.5f}","HYBRID_M0",f"{metric(scores['U3xM0'],y)['accuracy']:.5f}",
              "HYBRID_M1",f"{metric(scores['U3xM1'],y)['accuracy']:.5f}")
    if OUTPUT.exists():raise RuntimeError("OUTPUT_ALREADY_EXISTS; never overwrite")
    OUTPUT.mkdir(parents=True)
    payload={"P15C_components.csv":pd.DataFrame(rows),
      "P15C_day_counts.csv":pd.DataFrame(byday),"P15C_site_counts.csv":pd.DataFrame(bysite)}
    oh={}
    for name,df in payload.items():
        path=OUTPUT/name;df.to_csv(path,index=False);oh[name]=sha(path)
    manifest={"stage":"P15C_HIST_COMPONENTS","status":"DESCRIPTIVE_ONLY",
      "future_b_opened":False,"dataset_c_opened":False,"training_allowed":False,
      "config_sha256":sha(CONFIG),"script_sha256":sha(Path(__file__)),
      "inputs_sha256":files,"outputs_sha256":oh}
    (OUTPUT/"P15C_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True))
    table=pd.DataFrame(rows)
    print("P15C_FAR_COMPONENT_RESULTS")
    print(table[table.window=="FAR"].to_string(index=False,float_format=lambda v:f"{v:.5f}"))
    print("P15C_COMPONENTS_COMPLETE",len(rows),"rows",OUTPUT)
if __name__=="__main__":main()
