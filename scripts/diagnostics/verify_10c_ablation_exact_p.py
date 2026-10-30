#!/usr/bin/env python3
"""Read-only exact day-label swaps for archived 10C DEV_FROZEN ablations.

Usage: python scripts/diagnostics/verify_10c_ablation_exact_p.py
       /home/bsierra/ltd-storage/results/final/FUTURE-B
       /home/bsierra/ltd-storage/results/reports/FROZEN-10C-PVALUES-V1

Only additive metrics can be tested from day-level aggregated CSV.
No retroactive registration; day exchangeability is an assumption.
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

PAIRS = [
    ('ACTUAL_HYBRID_vs_NO_XGB_RENORM', 'ACTUAL_HYBRID', 'HYBRID_NO_XGB_RENORM'),
    ('ACTUAL_HYBRID_vs_NO_LTD_RENORM', 'ACTUAL_HYBRID', 'HYBRID_NO_LTD_RENORM'),
]
METRICS = ('accuracy', 'top5_accuracy', 'mrr')


def hash_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def csv_rows(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def exact_two_sided(values):
    count=0
    observed=abs(sum(values))
    tolerance=1e-10*max(1.,sum(abs(x) for x in values))
    total=1<<len(values)
    for signs in range(total):
        statistic=sum(-v if (signs>>i)&1 else v for i,v in enumerate(values))
        if abs(statistic)+tolerance>=observed:
            count+=1
    return count/total,count,total


def holm(values):
    indices=sorted(range(len(values)),key=lambda i:values[i])
    adjusted=[0.]*len(values)
    prior=0.
    for rank,i in enumerate(indices):
        prior=max(prior,min(1.,values[i]*(len(values)-rank)))
        adjusted[i]=prior
    return adjusted


def main():
    source=Path(sys.argv[1]).resolve()
    destination=Path(sys.argv[2]).resolve()
    if destination.exists(): raise FileExistsError(destination)
    manifest=json.loads((source/'10C_manifest.json').read_text())
    assert manifest['status']=='POSTHOC_MECHANISTIC_ANALYSIS'
    for key,filename in [('per_day','10C_per_day_ablation.csv'),('bootstrap','10C_mechanistic_bootstrap.csv')]:
        assert hash_file(source/filename)==manifest['outputs'][key],filename
    days=[r for r in csv_rows(source/'10C_per_day_ablation.csv') if r['track']=='DEV_FROZEN']
    ci=[r for r in csv_rows(source/'10C_mechanistic_bootstrap.csv') if r['track']=='DEV_FROZEN']
    results=[]
    for label,candidate,reference in PAIRS:
        a={r['date']:r for r in days if r['variant']==candidate}
        b={r['date']:r for r in days if r['variant']==reference}
        dates=sorted(a)
        assert len(dates)==16 and set(dates)==set(b)
        counts=[int(a[d]['n']) for d in dates]
        assert counts==[int(b[d]['n']) for d in dates] and sum(counts)==18543
        for metric in METRICS:
            values=[counts[i]*(float(a[d][metric])-float(b[d][metric])) for i,d in enumerate(dates)]
            difference=sum(values)/sum(counts)
            p,extreme,total=exact_two_sided(values)
            archived=[r for r in ci if r['comparison']==label and r['metric']==metric]
            assert len(archived)==1
            assert abs(float(archived[0]['observed_delta'])-difference)<1e-9
            results.append({'comparison':label,'metric':metric,'n':sum(counts),'days':16,
                'delta':difference,'p_exact_two_sided_posthoc':p,'extreme':extreme,
                'enumerations':total,'ci95_low_archived':archived[0]['ci95_low'],
                'ci95_high_archived':archived[0]['ci95_high']})
    adjusted=holm([r['p_exact_two_sided_posthoc'] for r in results])
    for i,r in enumerate(results): r['holm6_exact_posthoc']=adjusted[i]
    destination.mkdir(parents=True,exist_ok=False)
    output=destination/'10C_ablation_exact_day_swaps.csv'
    with output.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    provenance={
        'source_10C_manifest_sha256':hash_file(source/'10C_manifest.json'),
        'source_10C_per_day_sha256':hash_file(source/'10C_per_day_ablation.csv'),
        'source_10C_bootstrap_sha256':hash_file(source/'10C_mechanistic_bootstrap.csv'),
        'output_sha256':hash_file(output),
        'method':'posthoc conditional whole-day sign swaps; Holm6 (2 x 3 additive endpoints)',
        'limitations':'day dependence may invalidate p; macro-F1 permutation p not calculated'}
    (destination/'manifest.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('10C_EXACT_ABLATION_COMPLETE',destination)
    for r in results:
        print(r['comparison'],r['metric'],round(r['delta'],6),
            'p=',r['p_exact_two_sided_posthoc'],'Holm6=',r['holm6_exact_posthoc'])


if __name__=='__main__':
    main()
