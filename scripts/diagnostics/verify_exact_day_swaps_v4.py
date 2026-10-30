#!/usr/bin/env python3
"""Verify the archived exact post-hoc day-label swap p-values without training.

Usage: python verify_exact_day_swaps_v4.py /path/to/FROZEN-PAIRWISE-SENSITIVITY-V2
Assumes whole-day paired-label exchangeability; does not prove it.
"""
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path


def file_hash(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(1048576),b''):
            h.update(data)
    return h.hexdigest()


def read_csv(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def exact_p(values):
    # Count all whole-day model label swaps, two-sided, including observed.
    d=len(values)
    total=1<<d
    observed=abs(sum(values))
    tol=1e-10*max(1.,sum(map(abs, values)))
    extreme=0
    for assignment in range(total):
        statistic=sum(-v if assignment & (1<<i) else v for i,v in enumerate(values))
        if abs(statistic)+tol>=observed:
            extreme+=1
    return extreme/total,extreme,total


def main():
    directory=Path(sys.argv[1])
    manifest=json.loads((directory/'manifest.json').read_text())
    for filename,digest in manifest['outputs_sha256'].items():
        assert file_hash(directory/filename)==digest,filename
    daily=read_csv(directory/'03_daily_paired_differences.csv')
    archived=read_csv(directory/'01_exact_day_label_swaps.csv')
    grouped=defaultdict(list)
    for row in daily:
        grouped[(row['period'],row['comparison'],row['metric'])].append(row)
    n=0
    for row in archived:
        key=(row['period'],row['comparison'],row['metric'])
        days=sorted(grouped[key], key=lambda x:x['date'])
        assert len(days)==int(row['days'])
        vals=[int(x['n'])*float(x['daily_delta']) for x in days]
        p,extreme,total=exact_p(vals)
        weighted_delta=sum(vals)/sum(int(x['n']) for x in days)
        assert abs(p-float(row['exact_conditional_p_two_sided']))<1e-11,key
        assert abs(weighted_delta-float(row['observed_delta']))<1e-11,key
        n+=1
    assert n==42,n
    print('EXACT_DAY_SWAP_VERIFIED',n,'of 42')
    print('Whole-day exchangeability must be justified; this is a POST-HOC audit.')


if __name__=='__main__':
    main()
