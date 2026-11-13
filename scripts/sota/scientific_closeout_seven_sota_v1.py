#!/usr/bin/env python3
"""Reproducible, non-destructive scientific closeout for LTD vs seven adapted SOTA.

Inputs: frozen final article and 168 paired-date Monte Carlo p-value reports.
Outputs: paper-ready Markdown, LaTeX tables, complete CSV tables, exact
two-sided date sign-flip p-values for the 14 focal Hybrid accuracy comparisons,
Holm exact adjustment across those 14 tests, temporal degradation table,
provenance and methodological claim ledger. Never trains or tunes a model.

The 14-test family was defined during the analysis: call it focused/posthoc,
NOT prospectively preregistered or confirmatory. The 168-family Holm adjustment
remains a separate conservative exploratory sensitivity.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,shutil,subprocess,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.sota.sota_paths_v1 import S5
from scripts.sota.compare_all_seven_with_original_ltd_s5_s6_v1 import SOTA_FOLDERS
OURS=("LTD_MICRO","LTD_MACRO","LTD_HYBRID")
SIX=("SOTA_DF","SOTA_VarCNN_direction","SOTA_TikTok","SOTA_NetCLR","SOTA_RoFiRe","SOTA_WFCAT")
SV2="SOTA_Swallow_v2_BNtrain"
MODELS=OURS+SIX+(SV2,)
METRICS=("accuracy","macro_f1","top5","mrr")
COHORTS=("09B","10B_DEV_FROZEN")
SIZE={"09B":(14687,11),"10B_DEV_FROZEN":(18543,16)}
NAME={"LTD_MICRO":"LTD-Micro","LTD_MACRO":"LTD-Macro","LTD_HYBRID":"LTD-Hybrid",
      "SOTA_DF":"DF (adapted)","SOTA_VarCNN_direction":"VarCNN direction only",
      "SOTA_TikTok":"Tik-Tok (adapted)","SOTA_NetCLR":"NetCLR (adapted)",
      "SOTA_RoFiRe":"RoFiRe DyWin 10x","SOTA_WFCAT":"WFCAT IAT",
      SV2:"Swallow BNtrain v2"}
def check(cond,msg):
    if not cond:raise RuntimeError("SCIENTIFIC_CLOSEOUT_BLOCKED "+msg)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()
def j(path):return json.loads(Path(path).read_text())
def csvrows(path):
    with Path(path).open(newline="") as f:return list(csv.DictReader(f))
def fmt(x,scale=1,places=2):return f"{float(x)*scale:.{places}f}"
def holm(p):
    order=np.argsort(p,kind="stable")
    result=np.empty(len(p),dtype=float)
    prev=0.
    for i,ix in enumerate(order):
        prev=max(prev,min(1.,(len(p)-i)*p[ix]))
        result[ix]=prev
    return result
def exact_sign_test(date_differences):
    # Exact enumeration of all 2^D paired whole-date model-label swaps.
    # Null: methods exchangeable for each DATE, not independent per packet.
    d=np.asarray(date_differences,dtype=np.int64)
    D=len(d)
    check(D in (11,16),"unexpected number of date clusters")
    state=np.arange(1<<D,dtype=np.uint32)
    bits=((state[:,None] >> np.arange(D,dtype=np.uint32)) & 1).astype(np.int8)
    signs=2*bits-1
    observed=abs(int(d.sum()))
    perm=np.abs(signs.astype(np.int64)@d)
    count=int(np.count_nonzero(perm>=observed))
    check(count>=2,"complementary outcomes expected")
    return float(count/len(perm)),count,int(len(perm))
def primary_exact(base,stats,reference,cohort,baseline):
    n,days=SIZE[cohort]
    src=Path(j(S5)["label_alignment"][cohort]["source"])
    frozen_s5=j(S5)["label_alignment"][cohort]
    check(sha(src)==frozen_s5["source_sha256"],"canonical LTD source changed")
    with np.load(src,allow_pickle=False) as z:
        y=z["y_true"].astype(np.int64)
        dates=z["query_date"].astype("datetime64[D]").astype(str)
        labels=z["candidate_labels"].astype(str)
        hybrid=z["hybrid_final_probs"].astype(np.float32)
    folder="s11_swallow_authors_BNtrain_v2_seed42" if baseline==SV2 else SOTA_FOLDERS[baseline]
    score=base/"artifacts/sota"/folder/"evaluation"/cohort/"scores_sota_aligned.npz"
    evalmeta=j(score.parent/"manifest.json")
    check(sha(score)==evalmeta["score_sha256"],"SOTA score SHA mismatch")
    check(sha(score)==stats["source_sha256"][cohort][baseline],
          "statistical p-value source not same score file")
    with np.load(score,allow_pickle=False) as z:
        ys=z["y_true"].astype(np.int64)
        days_s=z["query_date"].astype("datetime64[D]").astype(str)
        names=z["candidate_labels"].astype(str)
        uid=z["pcap_uid"].astype(str)
        p=z["sota_probs"].astype(np.float32)
    check(len(y)==n and len(np.unique(dates))==days and
          np.array_equal(y,ys) and np.array_equal(dates,days_s) and
          np.array_equal(labels,names) and len(set(uid))==n,
          "cohort row, date, UID or label mismatch")
    check(hybrid.shape==(n,65) and p.shape==(n,65),"65-way score missing")
    d=(hybrid.argmax(1)==y).astype(np.int8)-(p.argmax(1)==y).astype(np.int8)
    actual=np.unique(dates)
    per_date=np.asarray([d[dates==date].sum() for date in actual],dtype=np.int64)
    observed=float(d.sum()/n)
    check(abs(observed-float(reference["delta"]))<1e-8,
          "observed signed accuracy incompatible with frozen CI")
    pv,count,total=exact_sign_test(per_date)
    return pv,count,total,per_date.tolist()
def tex(s):
    return str(s).replace("\\","\\textbackslash{}").replace("_","\\_").replace("%","\\%").replace("&","\\&")
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base",required=True,type=Path)
    ap.add_argument("--article-dir",required=True,type=Path)
    ap.add_argument("--statistical-dir",required=True,type=Path)
    ap.add_argument("--output-dir",required=True,type=Path)
    args=ap.parse_args()
    check(not args.output_dir.exists(),"existing destination cannot be overwritten")
    metrics_path=args.article_dir/"article_metrics_10models.csv"
    pairs_path=args.article_dir/"article_paired_168_ci95.csv"
    article_cert_path=args.article_dir/"article_certificate.json"
    stats_path=args.statistical_dir/"article_168_pvalues_and_ci95.csv"
    stats_cert_path=args.statistical_dir/"statistical_certificate.json"
    for path in (metrics_path,pairs_path,article_cert_path,stats_path,stats_cert_path,S5):
        check(path.is_file(),"missing source "+str(path))
    article=j(article_cert_path);stats=j(stats_cert_path)
    check(article["status"]=="FINAL_RETROSPECTIVE_ADAPTED_METHODS_COMPARISON_PASS" and
          article["models_per_cohort"]==10 and article["paired_metric_ci_count"]==168,
          "article comparison certificate invalid")
    check(stats["status"]=="PVALUES_COMPUTED_AND_LINKED_TO_FROZEN_168_CIS" and
          stats["n_metric_tests"]==168 and stats["primary_hybrid_accuracy_tests"]==14,
          "statistical certificate invalid")
    check(stats["source_sha256"]["final_cert"]==sha(article_cert_path) and
          stats["source_sha256"]["previous_168_ci_csv"]==sha(pairs_path) and
          stats["source_sha256"]["canonical_S5"]==sha(S5),
          "publication files detached from statistical provenance")
    metric_rows=csvrows(metrics_path);stat_rows=csvrows(stats_path);ci_rows=csvrows(pairs_path)
    check(len(metric_rows)==20 and len(stat_rows)==len(ci_rows)==168,
          "source rows incomplete")
    results={}
    for row in metric_rows:
        key=(row["cohort"],row["method"])
        check(key not in results,"duplicate metric row")
        check(key[0] in COHORTS and key[1] in MODELS,"unexpected metric model")
        results[key]={m:float(row[m]) for m in METRICS}
    check(set(results)=={(co,m) for co in COHORTS for m in MODELS},
          "10 models x2 not present")
    pindex={};ciindex={}
    for row in stat_rows:
        key=tuple(row[k] for k in ("cohort","own","baseline","metric"))
        check(key not in pindex and key[0] in COHORTS and key[1] in OURS and
              key[2] in SIX+(SV2,) and key[3] in METRICS,
              "unexpected or duplicate p comparison")
        check(abs(results[(key[0],key[1])][key[3]]-
                  results[(key[0],key[2])][key[3]]-float(row["delta"]))<1e-8,
              "p-value delta inconsistent with article model metrics")
        pindex[key]=row
    for row in ci_rows:
        key=tuple(row[k] for k in ("cohort","own","baseline","metric"))
        check(key not in ciindex,"duplicate frozen CI")
        ciindex[key]=row
        check(key in pindex and all(abs(float(row[k])-float(pindex[key][k]))<1e-12
                for k in ("delta","ci95_lower","ci95_upper")),
              "published p-value table does not preserve original CI")
    check(len(pindex)==168 and set(pindex)==set(ciindex),"missing comparison key")
    primary=[]
    for co in COHORTS:
        for sota in SIX+(SV2,):
            key=(co,"LTD_HYBRID",sota,"accuracy")
            row=pindex[key]
            pv,count,total,daily=primary_exact(args.base,stats,row,co,sota)
            primary.append({
                "cohort":co,"baseline":sota,
                "delta_accuracy_pp":100*float(row["delta"]),
                "ci95_lower_pp":100*float(row["ci95_lower"]),
                "ci95_upper_pp":100*float(row["ci95_upper"]),
                "exact_p_two_sided":pv,"exact_extreme_assignments":count,
                "exact_total_assignments":total,
                "monte_carlo_p_two_sided":float(row["p_value_two_sided"]),
                "monte_carlo_holm_14":float(row["p_holm_primary_14"]),
                "monte_carlo_holm_168":float(row["p_holm_168"]),
                "date_correct_count_deltas":daily})
    exact_adjusted=holm(np.array([r["exact_p_two_sided"] for r in primary]))
    for r,p in zip(primary,exact_adjusted):
        r["exact_holm_14"]=float(p)
        r["exact_holm_14_significant_0_05"]=bool(p<.05)
        r["ci95_excludes_zero"]=bool(r["ci95_lower_pp"]>0 or r["ci95_upper_pp"]<0)
    drift=[]
    for model in MODELS:
        earlier=results[("09B",model)];later=results[("10B_DEV_FROZEN",model)]
        drift.append({"method":model,**{
            m+"_09B":earlier[m] for m in METRICS},
            **{m+"_future_B":later[m] for m in METRICS},
            **{m+"_change_pp":100*(later[m]-earlier[m]) for m in METRICS}})
    # Deliberately do not call temporal changes statistically significant:
    # those are cross-cohort differences, not within-date paired comparisons.
    final={"classification":"LTD_SOTA_SCIENTIFIC_CLOSURE_V1",
           "status":"SCIENTIFIC_TABLES_INFERENCE_AND_METHOD_SCOPE_FROZEN",
           "cohorts":{c:{"n":SIZE[c][0],"dates":SIZE[c][1]} for c in COHORTS},
           "models_per_cohort":10,"sota_adapted":7,
           "confidence_intervals_and_monte_carlo_p_values":168,
           "exact_focused_accuracy_date_signflip_tests":14,
           "exact_focused_holm_significant":sum(x["exact_holm_14_significant_0_05"] for x in primary),
           "family_14":"focused POSTHOC family, not prospectively preregistered",
           "family_168":"exploratory 168 tests, Holm FWER sensitivity; MC p-values",
           "test_assumption":"exchangeability of the two methods' entire daily outcome vectors under the null",
           "exact_test":"two-sided sign reversal of paired day-level correct-count differences; all 2^D sign vectors",
           "bootstrap_ci":"frozen 2500 paired day-block percentile bootstrap 95%; does not share an identical test inversion with permutation p",
           "source_sha256":{"article_cert":sha(article_cert_path),
                            "stat_cert":sha(stats_cert_path),
                            "article_metrics":sha(metrics_path),
                            "article_168_CIs":sha(pairs_path),
                            "stat_168_p":sha(stats_path),
                            "S5":sha(S5)},
           "caveats":[
               "Author-source-adapted model implementations are not full-paper exact reproductions",
               "14-test focused family introduced during analysis; no prospective confirmatory designation",
               "Only 11 and 16 date blocks, so inference relies on date-level exchangeability",
               "Future-B 65 monitored sites possibly conditioned on future availability; not a prospectively blind sample",
               "Single seed; no multiple-seed variability",
               "VarCNN direction-only; RoFiRe DyWin 10x; NetCLR altered training/CDF; Swallow 100 epochs, batch 32",
               "WFCAT external timestamp unit provenance not fully certified",
               "Top-5 Future-B VarCNN direction (80.98%) slightly exceeds LTD-Hybrid (80.56%)",
               "Temporal performance changes are descriptive and not separately tested."]}
    # All checks happen BEFORE creating the output directory.
    args.output_dir.mkdir(parents=True)
    def savecsv(name,rs,fields=None):
        with (args.output_dir/name).open("x",newline="") as f:
            w=csv.DictWriter(f,fieldnames=fields or list(rs[0]))
            w.writeheader();w.writerows(rs)
    primary_flat=[{k:v for k,v in row.items() if k!="date_correct_count_deltas"}
                  for row in primary]
    savecsv("focused_14_exact_p_and_ci95.csv",primary_flat)
    savecsv("temporal_changes_10_models.csv",drift)
    shutil.copy2(metrics_path,args.output_dir/"all_10_model_metrics.csv")
    shutil.copy2(stats_path,args.output_dir/"all_168_p_values_and_ci95.csv")
    # Main results tables in Markdown and LaTeX.
    md=["# LTD-ATTACK: scientific closeout of seven source-adapted SOTA baselines",
        "","## Experimental scope",
        "Two retrospective frozen cohorts contain 65 monitored classes each: "
        "Historical 09B (n=14,687; 11 capture dates) and Future-B "
        "(n=18,543; 16 capture dates). Historical-only B selected each "
        "model's epoch on DEV_LATE; C refit used the historical development "
        "population. Future-B was not used for model fitting or selection. "
        "These are source-adapted variants, NOT full numerically identical "
        "published original implementations.",""]
    main_tex=[r"\begin{table*}[t]",r"\centering",r"\small",
       r"\caption{Retrospective performance on historical and future capture cohorts. "
       r"All SOTA baselines are source-adapted variants; values are percentages "
       r"(including MRR multiplied by 100).}",
       r"\label{tab:ltd-seven-sota-performance}",
       r"\begin{tabular}{llrrrr}",r"\hline",
       r"Cohort & Method & Accuracy & Macro-F1 & Top-5 & MRR \\",
       r"\hline"]
    for co in COHORTS:
        md+=["## "+co+" ("+str(SIZE[co][0])+" captures, "+str(SIZE[co][1])+" days)","",
             "| Method | Accuracy (%) | Macro-F1 (%) | Top-5 (%) | MRR (x100) |",
             "|---|---:|---:|---:|---:|"]
        for model in MODELS:
            v=results[(co,model)]
            md.append("| "+NAME[model]+" | "+" | ".join(fmt(v[m],100) for m in METRICS)+" |")
            main_tex.append(tex(co)+" & "+tex(NAME[model])+" & "+
                " & ".join(fmt(v[m],100) for m in METRICS)+r" \\")
        main_tex.append(r"\hline")
        md+=[""]
    main_tex +=[r"\end{tabular}",r"\end{table*}"]
    md +=["## Date-paired inference on Hybrid accuracy","",
         "The 14 accuracy contrasts constitute a focused posthoc family. "
         "The table reports the previously frozen 95% percentile "
         "date-cluster bootstrap CIs and newly enumerated exact two-sided "
         "date-block sign-flip p-values with Holm adjustment for this "
         "14-test family. These procedures are not test/CI inversions.","",
         "| Cohort | Adapted SOTA | Hybrid minus SOTA (pp) | 95% CI (pp) | Exact two-sided p | Exact Holm-14 p |",
         "|---|---|---:|---|---:|---:|"]
    inference_tex=[r"\begin{table*}[t]",r"\centering",r"\small",
        r"\caption{LTD-Hybrid minus seven source-adapted SOTA variants in "
        r"accuracy (percentage points). Confidence intervals are date-block "
        r"bootstrap percentiles; exact two-sided p-values enumerate "
        r"all paired date sign reversals. Holm adjustment over 14 "
        r"FOCUSED POSTHOC accuracy tests; not prospectively preregistered.}",
        r"\label{tab:ltd-seven-sota-exact-inference}",
        r"\begin{tabular}{llrrr}",r"\hline",
        r"Cohort & Adapted comparator & Difference (pp) & 95\% CI (pp) & Exact Holm-14 p \\",
        r"\hline"]
    for row in primary:
        ci="["+fmt(row["ci95_lower_pp"])+", "+fmt(row["ci95_upper_pp"])+"]"
        md.append("| "+row["cohort"]+" | "+NAME[row["baseline"]]+" | +"+
                  fmt(row["delta_accuracy_pp"])+" | "+ci+" | "+
                  fmt(row["exact_p_two_sided"],1,6)+" | "+
                  fmt(row["exact_holm_14"],1,6)+" |")
        inference_tex.append(tex(row["cohort"])+" & "+tex(NAME[row["baseline"]])+" & "+
            fmt(row["delta_accuracy_pp"])+" & "+ci+" & "+
            fmt(row["exact_holm_14"],1,6)+r" \\")
        if row["baseline"]==SV2:inference_tex.append(r"\hline")
    inference_tex +=[r"\end{tabular}",r"\end{table*}"]
    md +=["","## Descriptive temporal change (percentage points)","",
          "| Method | Accuracy historical | Accuracy future | Future minus historical |",
          "|---|---:|---:|---:|"]
    for row in drift:
        md.append("| "+NAME[row["method"]]+" | "+
                  fmt(row["accuracy_09B"],100)+" | "+
                  fmt(row["accuracy_future_B"],100)+" | "+
                  fmt(row["accuracy_change_pp"])+" |")
    md+=["","## Interpretation",
         "LTD-Hybrid achieved the highest accuracy among the evaluated "
         "adapted variants in both cohorts. Historical Hybrid accuracy "
         "was 97.44%; Future-B was 49.45%. The next highest SOTA "
         "accuracy was RoFiRe in 09B (94.21%) and VarCNN's direction-only "
         "branch in Future-B (44.55%).",
         "These findings are bounded by the model implementations, "
         "site selection, frozen cohorts, and single-seed setup. "
         "In Future-B, VarCNN direction Top-5 slightly exceeded Hybrid's "
         "(80.98% vs 80.56%); avoid claiming universal dominance across metrics.",
         "Observed temporal degradation is DESCRIPTIVE; its statistical "
         "significance was not tested as a cross-cohort contrast.",
         "","## Statistical interpretation and multiplicity",
         "The 168 originally computed Monte Carlo date-swap p-values "
         "and their Holm-168 adjustments are retained in the supplemental "
         "CSV. The focused 14-test Holm family was specified after "
         "inspection of project results, so its significance cannot be "
         "presented as prospectively confirmatory. Exact sign-flip "
         "results are supplied as a more precise sensitivity for the "
         "focused accuracy comparisons. Only 11/16 dates and the "
         "exchangeability assumption constrain these tests.",
         "","## Threats to validity",
         "1. Author-source adapters differ from full published methods; "
         "most notably direction-only VarCNN, reduced DyWin RoFiRe, "
         "modified NetCLR pretraining and Swallow BNtrain v2.",
         "2. Frozen 65-site Future-B population is retrospective and "
         "may be conditional on future collection availability.",
         "3. Training uses single seeds; bootstrap over days is not "
         "a substitute for seed variability.",
         "4. WFCAT timestamp-unit provenance is not fully closed; "
         "do not explain its future deterioration mechanistically.",
         "5. Date-block label-swap tests are conditional/exchangeability "
         "inference, not prospectively randomized tests.",
         "6. Swallow v1 collapsed and was excluded; v2 is a "
         "separately versioned adapted BNtrain run, not paper exact.",
         "","## Publication decision",
         "The retrospective benchmark is reproducible and ready for "
         "reporting AS AN ADAPTED-SOTA COMPARISON. A claim of superiority "
         "over exact original papers or unseen future datasets is not supported.",""]
    (args.output_dir/"scientific_closeout.md").write_text("\n".join(md)+"\n")
    (args.output_dir/"table_performance.tex").write_text("\n".join(main_tex)+"\n")
    (args.output_dir/"table_accuracy_inference.tex").write_text("\n".join(inference_tex)+"\n")
    exact_sig=sum(row["exact_holm_14_significant_0_05"] for row in primary)
    mc_global_sig=sum(row["monte_carlo_holm_168"]<.05 for row in primary)
    best09="RoFiRe (adapted)"
    bestfuture="VarCNN direction-only"
    section=[
      r"\\subsection{Comparison with source-adapted state-of-the-art methods}",
      r"All methods were evaluated on the same frozen 65-class monitored-site "
      r"cohorts: Historical 09B (14,687 captures over 11 dates) and "
      r"Future-B (18,543 captures over 16 dates). Historical model "
      r"selection used DEV\\_LATE; final refitting used the historical "
      r"development population. The Future-B cohort was reserved for "
      r"evaluation and did not determine fitted hyperparameters.",
      r"\\input{table_performance.tex}",
      r"LTD-Hybrid obtained accuracies of 97.44\\% on Historical 09B and "
      r"49.45\\% on Future-B, exceeding the best adapted SOTA model "
      r"in each cohort: RoFiRe (94.21\\%) and the direction-only "
      r"VarCNN variant (44.55\\%), respectively. The corresponding "
      r"accuracy differences were 3.23 and 4.90 percentage points.",
      r"\\subsection{Paired uncertainty and focused statistical tests}",
      r"The 95\\% uncertainty intervals for paired differences were "
      r"estimated from 2,500 bootstrap resamples of date clusters. "
      r"The 14 focused LTD-Hybrid accuracy comparisons additionally "
      r"underwent exact two-sided sign-reversal tests over the 2^D "
      r"possible swaps of whole-day paired outcomes (D=11 and 16). "
      r"Family-wise error was adjusted by Holm over these 14 contrasts.",
      r"\\input{table_accuracy_inference.tex}",
      "The exact Holm-adjusted focused analysis yielded "+
      str(exact_sig)+r"/14 comparisons with p<0.05. "
      r"These 14 contrasts were assembled during the analysis and "
      r"must be interpreted as posthoc focused tests rather than "
      r"prospectively registered confirmatory hypotheses. "
      r"The broader 168-test Monte Carlo Holm family retained "+
      str(mc_global_sig)+r"/14 focused comparisons below p<0.05, "
      r"highlighting sensitivity of significance claims to "
      r"the multiplicity family.",
      r"\\subsection{Temporal change and methodological scope}",
      r"The historical-to-future decline in accuracy was 47.99 "
      r"percentage points for LTD-Hybrid. Cross-cohort declines "
      r"are descriptive and are not paired date-level inference. "
      r"Future-B VarCNN direction-only had slightly higher Top-5 "
      r"accuracy than LTD-Hybrid (80.98\\% versus 80.56\\%), "
      r"so the superiority statement is restricted to accuracy.",
      r"These baselines reuse identifiable author-source implementations "
      r"with protocol adaptations rather than the exact complete "
      r"published configurations. VarCNN uses the direction branch, "
      r"RoFiRe uses reduced DyWin augmentation, and NetCLR and "
      r"Swallow use adapted training schedules. Swallow v1's "
      r"collapsed representation was excluded and replaced with an "
      r"independently versioned, noncollapsed target-BNtrain v2 "
      r"run. Only one seed was evaluated. The selected 65 sites "
      r"may depend retrospectively on future collection availability, "
      r"and the WFCAT time-unit acquisition provenance is not "
      r"fully established. These conditions limit generalization "
      r"to original full-paper methods and prospective data."]
    (args.output_dir/"manuscript_sota_results.tex").write_text(
        "\n\n".join(section)+"\n")
    final["primary_comparisons"]=primary
    source=ROOT/"scripts/sota/scientific_closeout_seven_sota_v1.py"
    final["script_sha256"]=sha(source)
    final["generated_sha256"]={p.name:sha(p) for p in args.output_dir.iterdir() if p.is_file()}
    (args.output_dir/"scientific_closeout_certificate.json").write_text(
        json.dumps(final,indent=2,ensure_ascii=False)+"\n")
    print("SCIENTIFIC_CLOSEOUT_STATUS",final["status"])
    print("EXACT_HOLM14_SIGNIFICANT",final["exact_focused_holm_significant"],"OF",14)
    for co in COHORTS:
        for row in primary:
            if row["cohort"]==co:
                print("EXACT_PRIMARY",co,row["baseline"],
                      "DELTA_PP",fmt(row["delta_accuracy_pp"]),
                      "CI95_PP",fmt(row["ci95_lower_pp"]),fmt(row["ci95_upper_pp"]),
                      "P",fmt(row["exact_p_two_sided"],1,6),
                      "HOLM14",fmt(row["exact_holm_14"],1,6))
    print("CLOSURE_OUTPUT",args.output_dir)
if __name__=="__main__":main()
