#!/usr/bin/env python3
"""LTD-Attack: read-only supplementary sensitivity of FROZEN-PAIRWISE-EVIDENCE-V1.

INPUT: The V1 *aggregated* CSV and manifest exports, NEVER raw PCAP/NPZ or models.
OUTPUT: A new directory with additive paired-day sensitivity, exact conditional
label-swap enumeration, site diagnostics and SHA provenance.
This is retrospective/descriptive; DO NOT use it to tune/select a model on Future-B.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = 'FROZEN-PAIRWISE-SENSITIVITY-V2'
SEED = 20261026
DRAWS = 20000
ADDITIVE = ('accuracy', 'top5', 'mrr')
PAIRS = [('MICRO','HYBRID'), ('MACRO','HYBRID'), ('XGB','HYBRID'),
         ('LTD','HYBRID'), ('XGB','MACRO'), ('LTD','MACRO'), ('LTD','XGB')]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_inputs(base: Path):
    mani = json.loads((base/'manifest.json').read_text())
    assert mani['stage']=='FROZEN-PAIRWISE-EVIDENCE-V1'
    for filename, digest in mani['outputs_sha256'].items():
        assert sha(base/filename)==digest, f'V1 output SHA mismatch: {filename}'
    tabs = {key:pd.read_csv(base/f'{key}.csv') for key in (
        '00_model_metrics','01_paired_comparisons','02_per_day_metrics',
        '03_per_site_metrics','04_existing_10C_ablation_summary',
        '05_existing_10C_ablation_bootstrap','06_existing_10C_rescues')}
    comparisons=tabs['01_paired_comparisons']
    for rec in comparisons.itertuples():
        if rec.metric=='accuracy':
            assert abs(rec.delta-(rec.rescued-rec.harmed)/rec.n) < 1e-11
            assert rec.both_correct+rec.rescued+rec.harmed+rec.both_wrong==rec.n
    models=tabs['00_model_metrics']; days=tabs['02_per_day_metrics']; sites=tabs['03_per_site_metrics']
    assert len(comparisons)==56 and len(models)==10 and len(sites)==650
    for row in models.itertuples():
        subset=days[(days.period==row.period)&(days.model==row.model)]
        assert len(subset)==row.days and int(subset.n.sum())==row.n
        ss=sites[(sites.period==row.period)&(sites.model==row.model)]
        assert len(ss)==65 and int(ss.n.sum())==row.n
        for met in ADDITIVE:
            assert abs((subset.n*subset[met]).sum()/row.n-getattr(row,met))<1e-12
        assert abs((ss.n*ss.recall).sum()/row.n-row.accuracy)<1e-12
        assert abs(ss.class_f1.mean()-row.macro_f1)<1e-12
        assert abs((ss.n*ss.top5).sum()/row.n-row.top5)<1e-12
    return mani, tabs


def exact_signflip(delta_counts: np.ndarray):
    """Enumerate all 2^D day-sign patterns, counting both tails, equality included.
    Valid ONLY conditionally on *whole-day* model-label exchangeability.
    """
    count = len(delta_counts)
    if count > 20:
        raise ValueError('Exact enumeration limited to <=20 days')
    states = np.arange(1 << count, dtype=np.uint32)
    signs = 2*((states[:, None] >> np.arange(count,dtype=np.uint32)) & 1).astype(np.int8)-1
    null = signs.astype(np.float64)@delta_counts.astype(np.float64)
    obs = delta_counts.sum()
    # Numerical equality tolerance on additive weighted daily aggregates.
    return float(np.count_nonzero(np.abs(null)>=abs(obs)-1e-8)/len(states)), len(states)


def circular_block_bootstrap(ns, ds, block_len, iterations, seed):
    """Percentile sensitivity, paired through day blocks; NOT confirmatory."""
    rng=np.random.default_rng(seed)
    D=len(ns)
    groups=int(np.ceil(D/block_len))
    starts=rng.integers(0,D,size=(iterations,groups))
    offs=np.arange(block_len)
    ix=(starts[:,:,None]+offs[None,None,:])%D
    ix=ix.reshape(iterations,-1)[:,:D]
    vals=(ns[ix]*ds[ix]).sum(axis=1)/ns[ix].sum(axis=1)
    return float(np.quantile(vals,.025)),float(np.quantile(vals,.975)),float((vals>0).mean())


def run(base: Path, out: Path):
    if out.exists():
        raise FileExistsError(f'Refusing overwrite: {out}')
    manifest,t=verify_inputs(base)
    c=t['01_paired_comparisons']; days=t['02_per_day_metrics']; sites=t['03_per_site_metrics']
    daily_rows=[]; sensitivity_rows=[]; sites_rows=[]; group_rows=[]; exact_rows=[]; observations=[]
    for period in sorted(days.period.unique()):
        d=days[days.period==period]
        by=d.pivot(index='date',columns='model',values=['n',*ADDITIVE]).sort_index()
        dates=by.index.to_list(); D=len(dates)
        ns=by['n']['MICRO'].to_numpy(dtype=float)
        assert all(np.array_equal(ns,by['n'][m].to_numpy()) for m in by['n'].columns)
        N=int(ns.sum())
        for a,b in PAIRS:
            cname=f'{b}_vs_{a}'
            for met in ADDITIVE:
                dv=(by[met][b]-by[met][a]).to_numpy(dtype=float)
                weighted=ns*dv
                obs=float(weighted.sum()/N)
                cr=c[(c.period==period)&(c.reference==a)&(c.candidate==b)&(c.metric==met)]
                assert len(cr)==1 and abs(float(cr.iloc[0].delta)-obs)<1e-11
                p_exact, nenum=exact_signflip(weighted)
                exact_rows.append(dict(period=period,comparison=cname,metric=met,days=D,
                    observed_delta=obs,exact_conditional_p_two_sided=p_exact,
                    enumerations=nenum,assumption='whole-day model-label exchangeability'))
                for i,date in enumerate(dates):
                    daily_rows.append(dict(period=period,comparison=cname,metric=met,date=date,
                        n=int(ns[i]),daily_delta=float(dv[i])))
                for deletion in range(1,min(4,D-2)+1):
                    vals=[]
                    for start in range(D-deletion+1):
                        take=np.ones(D,dtype=bool);take[start:start+deletion]=False
                        vals.append((float(weighted[take].sum()/ns[take].sum()),start))
                    minv,imin=min(vals); maxv,imax=max(vals)
                    sensitivity_rows.append(dict(period=period,comparison=cname,metric=met,
                        analysis='delete_consecutive_days',window_days=deletion,
                        observed_delta=obs,min_delta=minv,max_delta=maxv,
                        min_deleted_start=dates[imin],min_deleted_end=dates[imin+deletion-1],
                        max_deleted_start=dates[imax],max_deleted_end=dates[imax+deletion-1],
                        all_positive=minv>0,all_negative=maxv<0))
                for k in range(1,5):
                    low,up,fr=circular_block_bootstrap(ns,dv,k,DRAWS,SEED+len(exact_rows)*101+k)
                    sensitivity_rows.append(dict(period=period,comparison=cname,metric=met,
                        analysis='circular_moving_block_bootstrap',window_days=k,
                        observed_delta=obs,ci95_low=low,ci95_high=up,
                        draws=DRAWS,bootstrap_fraction_positive_NOT_P=fr,
                        all_positive=low>0,all_negative=up<0))
            # Per-period sign stability for each metric, and halves for ADDITIVE.
            dv=by['accuracy'][b].to_numpy()-by['accuracy'][a].to_numpy()
            group_rows.append(dict(period=period,comparison=cname,unit='date',metric='accuracy',
                improved=int((dv>1e-12).sum()),tied=int((abs(dv)<=1e-12).sum()),
                harmed=int((dv<-1e-12).sum()),total=D))
            for cut, inds in [('earlier_days',np.arange(D)<D//2),('later_days',np.arange(D)>=D//2)]:
                observations.append(dict(period=period,comparison=cname,analysis=cut,n=int(ns[inds].sum()),
                    days=int(inds.sum()),accuracy_delta=float(np.dot(ns[inds],dv[inds])/ns[inds].sum())))
            site=sites[sites.period==period].pivot(index='site_label',columns='model',values=['n','recall','top5','class_f1','precision']).sort_index()
            assert (site['n'][a]==site['n'][b]).all()
            for met,col in [('accuracy','recall'),('top5','top5'),('class_f1','class_f1')]:
                site_delta=(site[col][b]-site[col][a])
                group_rows.append(dict(period=period,comparison=cname,unit='site',metric=met,
                    improved=int((site_delta>1e-12).sum()),tied=int((abs(site_delta)<=1e-12).sum()),
                    harmed=int((site_delta<-1e-12).sum()),total=len(site_delta)))
            for label in site.index:
                n=int(site['n'][a].loc[label]); rec_a=float(site['recall'][a].loc[label]); rec_b=float(site['recall'][b].loc[label])
                site_delta=rec_b-rec_a
                gain=int(round(site_delta*n))
                assert abs(gain/n-site_delta)<1e-10
                sites_rows.append(dict(period=period,comparison=cname,site_label=str(label),n=n,
                    reference_recall=rec_a,candidate_recall=rec_b,recall_delta=site_delta,
                    net_correct_gain=gain,
                    top5_delta=float(site['top5'][b].loc[label]-site['top5'][a].loc[label]),
                    class_f1_delta=float(site['class_f1'][b].loc[label]-site['class_f1'][a].loc[label])))
    exact=pd.DataFrame(exact_rows)
    # This 9-test family is a sensitivity alternative only; it is NOT a replacement
    # for the original 12-test family with non-additive Macro-F1.
    exact['holm9_exact_additive_posthoc']=np.nan
    for period in exact.period.unique():
        mask=(exact.period==period)&(exact.comparison.isin(['HYBRID_vs_MICRO','HYBRID_vs_MACRO','HYBRID_vs_XGB']))
        ix=exact.index[mask]; pp=exact.loc[ix,'exact_conditional_p_two_sided'].to_numpy()
        order=np.argsort(pp); adj=np.zeros(len(pp)); largest=0.
        for rank,ind in enumerate(order):
            largest=max(largest,min(1.0,pp[ind]*(len(pp)-rank)));adj[ind]=largest
        exact.loc[ix,'holm9_exact_additive_posthoc']=adj
    out.mkdir(parents=True)
    outputs={
      '01_exact_day_label_swaps.csv':exact,
      '02_temporal_sensitivity.csv':pd.DataFrame(sensitivity_rows),
      '03_daily_paired_differences.csv':pd.DataFrame(daily_rows),
      '04_site_paired_differences.csv':pd.DataFrame(sites_rows),
      '05_day_site_signs.csv':pd.DataFrame(group_rows),
      '06_early_late_descriptive.csv':pd.DataFrame(observations),
    }
    for fname,table in outputs.items(): table.to_csv(out/fname,index=False,float_format='%.12g')
    # Short reproducible check summary; no ungrounded claims of independence.
    nm=exact[(exact.period=='FUTURE_10B_DEV_FROZEN')&(exact.comparison=='HYBRID_vs_MICRO')]
    lines=[f'# {VERSION} — audit supplement', '',
       'Data: frozen V1 aggregated outputs, never raw PCAP/NPZ. No fitting/selection, no FUTURE tuning.',
       f'Input V1 manifest SHA256: `{sha(base/"manifest.json")}`.',
       f'Bootstrap: circular day-block lengths 1–4, {DRAWS} draws, seed base {SEED}.',
       'Block-bootstrap intervals are exploratory sensitivity checks; few (11/16) possibly dependent dates.',
       'Exact day-label sign-flips enumerate all 2^D patterns for accuracy/Top-5/MRR. They are',
       'exact only CONDITIONAL ON WHOLE-DAY MODEL-LABEL EXCHANGEABILITY; temporal serial dependence',
       'is a threat. Holm9 uses 3 primary comparisons × 3 additive metrics; it DOES NOT replace',
       'V1 Holm12 (which also includes nonlinear Macro-F1). Never call these confirmatory p-values.',
       'Per-class deltas are descriptive marginals, NOT paired site×date uncertainty or causal attribution.',
       '', '## Future B Hybrid vs Micro exact conditional sign flip', '']
    for r in nm.itertuples():
        lines.append(f'- {r.metric}: delta={r.observed_delta:+.8f}, exact conditional p={r.exact_conditional_p_two_sided:.8g}, Holm9={r.holm9_exact_additive_posthoc:.8g}; {r.enumerations} permutations')
    lines.extend(['', '## Scope and unfinished evidence',
      '- Macro-F1 cannot be exactly recomputed from per-day macro-F1 averages: requires classwise TP, predicted totals and support in original NPZ.',
      '- Neither paired site×date errors nor cross-site temporal interaction is recoverable from these marginal CSVs.',
      '- Site label to domain mapping must be validated against the canonical frozen encoder; legacy site_dictionary.csv is not proven identical.',
      '- A/B acquisition equivalence and B-conditioned 65-site coverage must be independently documented.',
      '- 10C Historical-Final rescue table is a different checkpoint; no interval for it in the V1 10C bootstrap subset.',
      '- Training/model/weight promotion is forbidden on opened Future-B; next blinded set requires prospective protocol.',
      '- The corrected V2 output manifest records script hash and V1 SHA; preserves all V1 files unchanged.',
      ''])
    (out/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    manifest2={'stage':VERSION,'read_only':True,'training':False,'future_model_selection':False,
        'parent_v1_manifest_sha256':sha(base/'manifest.json'),
        'parent_v1_source_score_sha256':{r['period']:r['score_sha256'] for r in manifest['source_checks']},
        'method':{'seed':SEED,'circular_block_bootstrap_draws':DRAWS,'block_lengths_days':[1,2,3,4],
            'exact_day_swap_max_days':20,'holm9_only_additive_posthoc':True,
            'v1_holm12_preserved':True},
        'source_code_sha256':sha(Path(__file__)),
        'outputs_sha256':{f.name:sha(f) for f in sorted(out.iterdir()) if f.is_file()}}
    (out/'manifest.json').write_text(json.dumps(manifest2,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('CLOSEOUT_V2_COMPLETE',out)
    print('PARENT_V1_MANIFEST_SHA256',manifest2['parent_v1_manifest_sha256'])
    print('SCRIPT_SHA256',manifest2['source_code_sha256'])
    print('FUTURE_B_EXACT_HYBRID_MICRO')
    print(nm[['metric','observed_delta','exact_conditional_p_two_sided','holm9_exact_additive_posthoc']].to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--v1',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    run(args.v1,args.out)

