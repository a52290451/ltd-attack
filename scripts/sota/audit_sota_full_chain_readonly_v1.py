#!/usr/bin/env python3
"""Read-only chained forensic audit: status, Swallow collapse, WFCAT feature shifts.

No writes to checkpoints, score NPZ, cached arrays, or git worktree. CPU-only;
uses existing provenance-certified scores and local original author sources.
Outputs ONE JSON report in a separate audit directory. External CSV sampling is
enabled by --deep-external and requires a full sequential scan of source files.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,sys,time
from collections import Counter
from pathlib import Path
import numpy as np

def need(ok,msg):
    if not ok:raise RuntimeError("SOTA_CHAIN_AUDIT_BLOCKED "+str(msg))

def desc(a):
    a=np.asarray(a,dtype=np.float64)
    if not a.size:return {}
    return {"n":int(a.size),"mean":float(np.mean(a)),"median":float(np.median(a)),
            "p05":float(np.percentile(a,5)),"p95":float(np.percentile(a,95)),
            "min":float(np.min(a)),"max":float(np.max(a))}

def safe_json(path):
    return json.loads(Path(path).read_text())

def latest_run(root):
    track=root/"background_battery_v1";pointer=track/"latest_run"
    if not pointer.is_file():return {"status":"NO_RUN"}
    run_id=pointer.read_text().strip();p=track/run_id
    out={"run_id":run_id,"status":(p/"status").read_text().strip() if (p/"status").exists() else "UNKNOWN",
         "exit_code":(p/"exit_code").read_text().strip() if (p/"exit_code").exists() else None,
         "pid":(p/"pid").read_text().strip() if (p/"pid").exists() else None}
    log=p/"console.log"
    if log.is_file():
        with log.open("rb") as stream:
            stream.seek(0,2);stream.seek(max(0,stream.tell()-131072))
            lines=stream.read().decode("utf-8",errors="replace").splitlines()
        out["last_stage_calls"]=[x for x in lines if x.startswith("SOTA_BATTERY_V2_EXEC ")][-5:]
        out["last_completion_events"]=[x for x in lines if
            ("_PASS" in x or "_FAILED" in x or "BATTERY_" in x)][-12:]
        out["last_log_lines"]=lines[-8:]
    return out

def record_frozen_forensic(report):
    need(report.get("overall_status")=="PASS_PREDICTION_INTEGRITY_ONLY",
         "input forensic JSON has not passed integrity")
    data={}
    for model,item in report["models"].items():
        training=item["training"]; evaluations=item["evaluation"]
        data[model]={
          "B_selected_epoch":training["B"].get("selected_epoch"),
          "B_best_dev_late_accuracy":training["B"].get("best_dev_late_accuracy"),
          "B_best_dev_late_macro_f1":training["B"].get("best_dev_late_macro_f1"),
          "09B":{k:evaluations["09B"].get(k) for k in (
             "accuracy","macro_f1","predicted_classes_present","majority_predicted_fraction",
             "normalized_prediction_entropy","mean_max_confidence")},
          "Future-B":{k:evaluations["10B_DEV_FROZEN"].get(k) for k in (
             "accuracy","macro_f1","predicted_classes_present","majority_predicted_fraction",
             "normalized_prediction_entropy","mean_max_confidence")},
        }
    return data

def cache_summaries(root):
    out={}
    for name,folder,feat in (
      ("Swallow",root/"s11_swallow_authors_seed42/historical_cif_cache","cif_uint16.npy"),
      ("WFCAT",root/"s12_wfcat_authors_iat_seed42/historical_cache","iat_uint16.npy")):
        mp=folder/"manifest.json"
        need(mp.is_file(),"missing "+str(mp))
        manifest=safe_json(mp)
        arr=np.load(folder/feat,mmap_mode="r")
        split=np.load(folder/"split.npy",mmap_mode="r")
        y=np.load(folder/"labels.npy",mmap_mode="r")
        need(len(arr)==len(split)==len(y)==57916,name+" inconsistent cache size")
        need(manifest.get("dataset_c_opened") is False,name+" cache C claim not false")
        by={}
        rng=np.random.default_rng(42)
        for sid,split_name in enumerate(("DEV_EARLY","DEV_MIDDLE","DEV_LATE")):
            positions=np.flatnonzero(split==sid)
            need(len(positions)>0,name+" missing "+split_name)
            indices=rng.choice(positions,size=min(128,len(positions)),replace=False)
            X=np.asarray(arr[indices],dtype=np.float32)
            by[split_name]={
               "n_total":int(len(positions)),
               "sampled":int(len(X)),
               "feature_nonzero_ratio":float(np.mean(X!=0)),
               "sample_fraction_all_zero":float(np.mean(np.all(X==0,axis=tuple(range(1,X.ndim))))),
               "feature_per_capture_sum":desc(X.reshape(len(X),-1).sum(1)),
               "feature_per_capture_max":desc(X.reshape(len(X),-1).max(1)),
               "feature_per_capture_nonzero":desc((X.reshape(len(X),-1)!=0).sum(1)),
               "feature_saturated_uint16_ratio":float(np.mean(X==65535)),
               "sampled_label_count":int(len(np.unique(y[indices]))),
            }
        out[name]={"source_manifest":str(mp),"feature_shape":list(arr.shape),
                   "splits":by,
                   "timestamp_regressions":manifest.get("timestamp_regressions"),
                   "traces_with_timestamp_regressions":manifest.get("traces_with_timestamp_regressions")}
    return out

def swallow_probe(repo,root):
    import torch
    torch.set_num_threads(2)
    sys.path.insert(0,str(repo))
    from scripts.sota import s11_swallow_authors_v1 as S
    cache=root/"s11_swallow_authors_seed42/historical_cif_cache"
    X=np.load(cache/"cif_uint16.npy",mmap_mode="r")
    y=np.load(cache/"labels.npy",mmap_mode="r")
    splits=np.load(cache/"split.npy",mmap_mode="r")
    result={}
    for stage,folder,filename in (
        ("B","supervised_B","best.pt"),("C","supervised_C","model.pt")):
        mp=root/"s11_swallow_authors_seed42"/folder/"manifest.json"
        meta=safe_json(mp)
        ck=Path(meta["checkpoint"])
        need(ck.name==filename and ck.exists(),stage+" checkpoint absent")
        state=torch.load(ck,map_location="cpu",weights_only=True)
        full,_,_=S.network()
        encoder=torch.nn.Sequential(*list(full.children())[:-1])
        classifier=torch.nn.Linear(512,65)
        encoder.load_state_dict(state["encoder"],strict=True)
        classifier.load_state_dict(state["classifier"],strict=True)
        encoder.eval();classifier.eval()
        groups={}
        for sid,label in ((0,"DEV_EARLY"),(1,"DEV_MIDDLE"),(2,"DEV_LATE")):
            indices=np.flatnonzero(splits==sid)
            chosen=np.random.default_rng(42).choice(indices,size=min(64,len(indices)),replace=False)
            xv=torch.from_numpy(np.asarray(X[chosen],dtype=np.float32)[:,None,:,:].copy())
            with torch.inference_mode():
                z=encoder(xv).flatten(1)
                logits=classifier(z)
                p=torch.softmax(logits,dim=1)
            q=p.numpy(); preds=np.argmax(q,axis=1)
            groups[label]={
              "sample_n":int(len(chosen)),
              "sampled_accuracy":float(np.mean(preds==y[chosen])),
              "predicted_unique_classes":int(len(set(preds))),
              "dominant_predicted_fraction":float(np.bincount(preds,minlength=65).max()/len(preds)),
              "mean_top_probability":float(np.max(q,axis=1).mean()),
              "embedding_across_samples_std":float(np.mean(z.numpy().std(axis=0))),
              "embedding_absolute_mean":float(np.abs(z.numpy()).mean()),
              "logit_across_samples_std":float(np.mean(logits.numpy().std(axis=0))),
              "logit_absolute_mean":float(np.abs(logits.numpy()).mean()),
            }
        result[stage]={
           "selected_epoch":meta.get("selected_epoch"),
           "classifier_weight_norm":float(classifier.weight.norm().item()),
           "classifier_weight_std":float(classifier.weight.std().item()),
           "classifier_bias_std":float(classifier.bias.std().item()),
           "groups":groups,
        }
    return result

def source_deviations(repo):
    sys.path.insert(0,str(repo))
    from scripts.sota import s11_swallow_authors_v1 as S
    from scripts.sota import s12_wfcat_authors_v1 as W
    from scripts.sota import s9_netclr_authors_pretrain_b_v1 as N
    return {"Swallow":S.CONFIG,"WFCAT":W.CFG,"NetCLR":N.CONFIG,
            "scope":"Effective declared configs are evidence of adaptation, not an exact upstream equivalence certificate."}

def external_samples(repo,root,storage,limit=64):
    """Sample external source vectors, rebuilding original-source WFCAT/Swallow
    features for each selected UID. Does not write caches or model outputs.
    Selects deterministic every-nth aligned UID. Iterates full CSV once per
    cohort, so run with nice/ionice during GPU training.
    """
    sys.path.insert(0,str(repo))
    from src.models.sota.v1.data_adapter import parse_direction
    from src.models.sota.v1.timing_adapter import parse_times
    from scripts.sota import s11_swallow_authors_v1 as S
    from scripts.sota import s12_wfcat_authors_v1 as W
    # Swallow and WFCAT both define top-level packages named "models".
    # Loading the WFCAT model here is unnecessary: this phase only needs its
    # upstream feature extractor. Import features.py explicitly by file path,
    # bypassing the conflicting models package cached by Swallow.
    import importlib.util
    import subprocess
    upstream_root=W.SOURCE
    source_file=upstream_root/"src/features.py"
    need(source_file.is_file(),"WFCAT upstream feature module missing")
    git_rev=subprocess.run(
        ["git","-C",str(upstream_root),"rev-parse","HEAD"],
        text=True,capture_output=True,check=True).stdout.strip()
    need(git_rev==W.COMMIT,"WFCAT upstream source commit mismatch")
    modified=subprocess.run(
        ["git","-C",str(upstream_root),"status","--porcelain",
         "--untracked-files=no"],text=True,capture_output=True,check=True).stdout.strip()
    need(not modified,"WFCAT upstream source tracked modifications")
    spec=importlib.util.spec_from_file_location(
        "ltd_wfcat_isolated_feature_module",source_file)
    need(spec is not None and spec.loader is not None,
         "WFCAT feature module could not be loaded")
    feature_module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(feature_module)
    wtransform=feature_module.feature_transform
    cif_fun,_,_,_=S.origin()
    out={}
    for cohort,folder,path in (
       ("09B","09B",storage/"historical/CLEAN_final_vectors_sites.csv"),
       ("Future-B","10B_DEV_FROZEN",storage/"future/CLEAN_final_vectors_sites_concept_drift.csv")):
        ref=root/"s12_wfcat_authors_iat_seed42/evaluation"/folder/"scores_sota_aligned.npz"
        with np.load(ref,allow_pickle=False) as z:
            uid=z["pcap_uid"].astype(str)
            labels=z["candidate_labels"].astype(str)
            yt=z["y_true"].astype(int)
        positions=np.unique(np.linspace(0,len(uid)-1,min(limit,len(uid)),dtype=int))
        expected={str(uid[i]):str(labels[yt[i]]) for i in positions}
        got={}; issues=Counter()
        csv.field_size_limit(sys.maxsize)
        with path.open(newline="",encoding="utf-8") as stream:
            reader=csv.DictReader(stream)
            need({"pcap_uid","site_label","direction_vector","time_vector"} <= set(reader.fieldnames or ()),
                 cohort+" source columns missing")
            for row in reader:
                ident=row["pcap_uid"]
                if ident not in expected:continue
                need(ident not in got,cohort+" duplicate UID in source")
                need(row["site_label"]==expected[ident],cohort+" UID class mismatch")
                d=parse_direction(row["direction_vector"])
                t=np.asarray(parse_times(row["time_vector"]),dtype=np.float64)
                need(len(d)==len(t),cohort+" trace length mismatch")
                w=W.create_feature(d,t,wtransform)
                c,load,dur=S.cif(d,t,cif_fun)
                got[ident]={
                    "packet_n":len(d),"timestamp_regression_count":int(np.sum(np.diff(t)<0)),
                    "duration_max_minus_min":float(t.max()-t.min()),
                    "time_last":float(t[-1]),
                    "iat_nonzero":int(np.count_nonzero(w)),
                    "iat_sum":float(np.sum(w,dtype=np.float64)),
                    "iat_max":int(w.max()),
                    "iat_saturated":int(np.sum(w==65535)),
                    "cif_nonzero":int(np.count_nonzero(c)),
                    "cif_sum":float(np.sum(c,dtype=np.float64)),
                    "cif_max":int(c.max()),
                }
                if len(got)==len(expected):break
        need(len(got)==len(expected),cohort+" sample UID missing in source")
        out[cohort]={"sample_n":len(got),
           "sample_uid_label_alignment":"PASS",
           "raw_vectors_and_original_features":{
               key:desc([rec[key] for rec in got.values()])
               for key in next(iter(got.values())).keys()
           }}
    return out

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--repo",type=Path,default=Path.home()/"ltd-attack")
    parser.add_argument("--base",type=Path,default=Path("/data/datasets_brayan/ltd-sota-2026"))
    parser.add_argument("--forensic-json",type=Path,required=True)
    parser.add_argument("--deep-external",action="store_true")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    os.environ.setdefault("LTD_SOTA_OPS_ROOT",str(args.base/"ops/sota"))
    os.environ.setdefault("LTD_SOTA_ARTIFACT_ROOT",str(args.base/"artifacts/sota"))
    os.environ.setdefault("LTD_SOTA_STORAGE_ROOT",str(Path.home()/"ltd-storage/datasets"))
    os.environ.setdefault("LTD_SOTA_S1_MANIFEST",str(args.base/"host_manifests/s1_host_local.json"))
    os.environ.setdefault("LTD_SOTA_S5_MANIFEST",str(args.base/"host_manifests/s5_host_local.json"))
    root=args.base/"artifacts/sota"
    started=time.time()
    result={"classification":"LTD_SOTA_READONLY_CHAINED_DIAGNOSTICS_V1",
            "started_utc":__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
            "limitations":["Descriptive forensic audit, not an original-paper exact equivalence certificate.",
                           "Sampling does not validate all 57,916 Historical vector feature transformations.",
                           "Future-B must not be used to select a new frozen model or tune parameters."],
            "battery":latest_run(root)}
    print("AUDIT_CHAIN battery",result["battery"].get("status"),flush=True)
    report=safe_json(args.forensic_json)
    result["reconstructed_score_audit"]=record_frozen_forensic(report)
    result["cross_model_alignment"]=report.get("cross_model_alignment")
    print("AUDIT_CHAIN scores aligned",flush=True)
    result["original_author_adaptation_configs"]=source_deviations(args.repo)
    print("AUDIT_CHAIN effective config retrieved",flush=True)
    result["historical_cache_feature_profiles"]=cache_summaries(root)
    print("AUDIT_CHAIN historical cache profiles",flush=True)
    result["swallow_cpu_checkpoint_probe"]=swallow_probe(args.repo,root)
    print("AUDIT_CHAIN Swallow frozen B/C inference probe",flush=True)
    if args.deep_external:
        result["external_original_feature_samples"]=external_samples(
             args.repo,root,Path(os.environ["LTD_SOTA_STORAGE_ROOT"]))
        print("AUDIT_CHAIN external original feature samples",flush=True)
    result["elapsed_seconds"]=round(time.time()-started,2)
    result["status"]="CHAIN_AUDIT_DONE_INSPECT_FINDINGS"
    need(not args.output.exists(),"output exists (refusing overwrite) "+str(args.output))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print("LTD_SOTA_CHAIN_AUDIT_WRITTEN",args.output,flush=True)
if __name__=="__main__":main()
