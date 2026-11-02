"""SOTA S4 protocol validation, chronological gates, and macro-F1 selection.

Pure stdlib. No torch, data access, remote API, evaluation holdouts, or
training code. Configuration is versioned and must never be tuned using
Internal 09B / Future-B / Dataset C.
"""
from __future__ import annotations
from collections import Counter,defaultdict
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Iterable

SPLIT_ORDER=("DEV_EARLY","DEV_MIDDLE","DEV_LATE")
EXPECTED_COUNTS={"DEV_EARLY":19098,"DEV_MIDDLE":20160,"DEV_LATE":18658}
EXPECTED_DAYS={"DEV_EARLY":14,"DEV_MIDDLE":14,"DEV_LATE":13}
EXPECTED_BOUNDS={
    "DEV_EARLY":("2025-11-18","2025-12-01"),
    "DEV_MIDDLE":("2025-12-02","2025-12-15"),
    "DEV_LATE":("2025-12-16","2025-12-28")
}
EXPECTED_UPSTREAM="537ec8e03c805540b04b95b144fa158ce50aa255"
EXPECTED_INPUTS={"DF":(1,128,0.002,"Adamax"),
                 "VarCNN":(2,50,0.001,"Adam")}
EXPECTED_IDS={"A":(("DEV_EARLY",),("DEV_MIDDLE",)),
              "B":(("DEV_EARLY","DEV_MIDDLE"),("DEV_LATE",)),
              "C":(SPLIT_ORDER,())}
FROZEN_METRICS=["accuracy","macro_f1","top5","mrr"]

def insist(cond:bool,message:str)->None:
    if not cond:raise ValueError("SOTA_S4_PROTOCOL_BLOCKED "+message)

def canonical_json(obj)->bytes:
    return (json.dumps(obj,ensure_ascii=False,sort_keys=True,
                       separators=(",",":"))+"\n").encode("utf-8")

def sha256_file(path:Path)->str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()

def validate_config(c:dict)->None:
    insist(c.get("classification")=="LTD_SOTA_S4_EXANTE_PROTOCOL_V1",
           "wrong protocol type")
    pop=c["population"]
    insist(pop["s1_uid_count"]==57916 and pop["classes"]==65,
           "training population changed")
    insist(pop["vector_length"]==5000 and pop["counts"]==EXPECTED_COUNTS,
           "historical split counts/length not frozen")
    insist(tuple(pop["training_splits"])==SPLIT_ORDER,"chronological DEV order changed")
    date_pairs=pop["dates"]
    for name in SPLIT_ORDER:
        insist(len(date_pairs[name])==2
               and tuple(date_pairs[name])==EXPECTED_BOUNDS[name],
               "DEV exact chronological bounds mismatch "+name)
    insist(c["model_source"]["commit"]==EXPECTED_UPSTREAM,
           "WFlib upstream source changed")
    insist(set(c["model_configs"])==set(EXPECTED_INPUTS),"unexpected baselines")
    for name,(channels,batch,lr,opt) in EXPECTED_INPUTS.items():
        m=c["model_configs"][name]
        insist((m["channels"],m["batch_size"],m["learning_rate"],m["optimizer"])
               ==(channels,batch,lr,opt),name+" WFlib hyperparameters changed")
        insist(m["epochs_max"]==30 and m["loss"]=="CrossEntropyLoss",
               name+" maximum budget or objective changed")
    insist(c["seeds"]["primary"]==[42],"primary seed not predeclared")
    insist(c["epoch_selection"]=={
        "metric":"macro_f1",
        "include_all_65_classes":True,
        "select":"maximum",
        "tie_break":"earliest_epoch",
        "evaluation_frequency":"every_epoch_on_DEV_only",
        "max_epoch":30,
        "early_stopping":False
    },"macro-F1 selection protocol changed")
    stages=c["schedule"]
    insist([s["id"] for s in stages]==["A","B","C"],"stage order not canonical")
    for stage in stages:
        tid=stage["id"]
        train,valid=EXPECTED_IDS[tid]
        insist(tuple(stage["train"])==train and tuple(stage["validate"])==valid,
               tid+" chronology not frozen")
        insist(not set(train)&set(valid),tid+" leaked training into validation")
        insist(stage["train_count"]==sum(EXPECTED_COUNTS[x] for x in train) and
               stage["val_count"]==sum(EXPECTED_COUNTS[x] for x in valid),
               tid+" expected population mismatch")
    insist(stages[1]["use_for_final_epoch_selection"] is True
           and stages[2]["epochs_from"]=="B",
           "final epoch must come from DEV Late validation")
    insist(c["metrics"]==FROZEN_METRICS,"frozen common metrics changed")
    insist(c["external_holdout"]=={
        "INTERNAL_TEST":"NO_TRAIN_NO_VALIDATION_NO_TUNING",
        "FUTURE_B":"NO_READ_BEFORE_CHECKPOINT_FROZEN",
        "DATASET_C":"EXCLUDED"
    },"external test leakage guard disabled")
    rc=c["resource_controls"]
    insist(rc["max_parallel_gpu_training_jobs"]==1 and rc["num_workers"]==0
           and rc["num_threads"]==2 and rc["gpu"]=="cuda:0",
           "Dataset C protected GPU/worker budget changed")
    insist(rc["output_root"]=="/data/ltd-attack/sota" and
           rc["minimum_data_free_gib"]>=120 and
           rc["do_not_modify_dataset_c_cron"] is True and
           rc["do_not_install_new_env"] is True and
           rc["do_not_modify_existing_ltd_venv"] is True,
           "unsafe data path or environment policy")
    insist(c["promotion_gates"]["G2_external_label_order"] is False and
           c["promotion_gates"]["checkpoints_frozen"] is False,
           "external predictions are not yet cleared")

def examine_s1_index(by_uid:dict)->dict:
    """No payload reads: only UID, class, date and split from S1 frozen CSV."""
    count=Counter()
    labels=defaultdict(set)
    dates=defaultdict(set)
    for uid,(site,date,split) in by_uid.items():
        insist(bool(uid) and bool(site) and split in SPLIT_ORDER,
               "invalid S1 UID/label/split")
        day=dt.date.fromisoformat(date)
        count[split]+=1
        labels[split].add(site)
        dates[split].add(day)
    insist(dict(count)==EXPECTED_COUNTS,"S1 population changed")
    insist(all(len(labels[s])==65 for s in SPLIT_ORDER),
           "not all DEV splits contain 65 classes")
    insist([len(dates[s]) for s in SPLIT_ORDER]==[14,14,13],
           "DEV date count not frozen")
    insist(all((min(dates[s]).isoformat(),max(dates[s]).isoformat())==
               EXPECTED_BOUNDS[s] for s in SPLIT_ORDER),
           "DEV exact date bounds not frozen")
    insist(max(dates["DEV_EARLY"])<min(dates["DEV_MIDDLE"])
           and max(dates["DEV_MIDDLE"])<min(dates["DEV_LATE"]),
           "temporal train/validation leakage detected")
    return {
        "split_counts":dict(count),
        "classes_per_split":{x:len(labels[x]) for x in SPLIT_ORDER},
        "days_per_split":{x:len(dates[x]) for x in SPLIT_ORDER},
        "date_bounds":{x:[min(dates[x]).isoformat(),
                          max(dates[x]).isoformat()] for x in SPLIT_ORDER},
        "unique_uids":len(by_uid)
    }

def score_macro_f1(y_true:Iterable[int],y_pred:Iterable[int],
                   classes:int=65)->float:
    true=list(y_true)
    pred=list(y_pred)
    insist(len(true)==len(pred) and bool(true),"empty or unaligned predictions")
    insist(classes==65,"macro-F1 must always include all 65 classes")
    insist(all(type(y) is int and 0<=y<65 for y in true+pred),
           "invalid 65-class label")
    tp=Counter((a for a,b in zip(true,pred) if a==b))
    support=Counter(true)
    predicted=Counter(pred)
    return sum(2*tp[k]/(support[k]+predicted[k])
               if support[k]+predicted[k] else 0.0
               for k in range(classes))/classes

def select_epoch(metric_by_epoch:dict[int,float],max_epochs:int=30)->int:
    """Highest DEV-only Macro-F1. Earliest epoch wins exact metric ties."""
    insist(bool(metric_by_epoch),"no DEV validation metrics")
    insist(all(type(epoch) is int and 1<=epoch<=max_epochs
               and type(score) in (int,float) and
               score==score and 0<=score<=1
               for epoch,score in metric_by_epoch.items()),
           "invalid epoch or DEV Macro-F1")
    return min(metric_by_epoch,key=lambda epoch:(-metric_by_epoch[epoch],epoch))
