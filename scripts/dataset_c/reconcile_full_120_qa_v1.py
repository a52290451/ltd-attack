#!/usr/bin/env python3
"""Dataset C: reconcile original QA12 + exactly 5 retry visits + remaining QA108.
Strict slot/URL mapping and all 65 frozen domains. Does NOT infer or train.
Only writes a compact summary alongside QA108 diagnostic evidence.
"""
from __future__ import annotations

import collections
import csv
import json
import pathlib
import sys
from urllib.parse import urlsplit

BASE = pathlib.Path('/data/ltd-c-qa-bsierra/C_DIRECT_V1')
ORIGINAL = BASE/'diagnostics/pinned_qa12_20261101T203050Z_136527/qa12_summary.json'
RECONCILED_12 = BASE/'diagnostics/pinned_retry5_20261101T204311Z_239428/combined_readiness.json'
READY = BASE/'diagnostics/scientific_readiness_20261101T205205Z_312672/scientific_readiness.json'
APP = pathlib.Path.home()/'ltd-collector-c-smoke-v22/P16C_RECOLECTOR_V2'
PAIRS = {
 'inst1':('belleza','turismo'),
 'inst2':('bancos','salud'),
 'inst3':('deportes','politica'),
}

def domain(url):
 try:
  v=urlsplit(str(url)).hostname
  return (v or '').lower().removeprefix('www.')
 except ValueError:
  return ''

def read(p):
 if not p.is_file() or p.is_symlink():
  raise RuntimeError('SOURCE_MISSING_OR_SYMLINK '+str(p))
 return json.loads(p.read_text())

def key(row):
 return (str(row['instance']),str(row['category']),int(row['slot']))

def make_index(records,kind):
 d={}
 for row in records:
  k=key(row)
  if k in d:
   raise RuntimeError('DUPLICATE_QA_SLOT '+kind+':'+str(k))
  d[k]=row
 return d

def main():
 if len(sys.argv)!=2:raise RuntimeError('Usage: reconcile_full_120_qa_v1.py /data/.../diagnostics/pinned_qa108_... ')
 folder=pathlib.Path(sys.argv[1]).resolve()
 if not folder.is_relative_to((BASE/'diagnostics').resolve()) or not folder.name.startswith('pinned_qa108_'):
  raise RuntimeError('QA108_DIAG_PATH_NOT_ALLOWED')
 qa108=read(folder/'qa108_summary.json')
 base=read(ORIGINAL); old=read(RECONCILED_12); readiness=read(READY)
 if base.get('attempts')!=12 or base.get('valid')!=7 or old.get('combined_valid')!=8:
  raise RuntimeError('ORIGINAL_QA_PROVENANCE_MISMATCH')
 if len(old.get('sites',[]))!=12 or old.get('remaining_access_constrained')!=4:
  raise RuntimeError('BASELINE_RETRY_PROVENANCE_MISMATCH')
 if qa108.get('attempts')!=108 or len(qa108.get('visits',[]))!=108:
  raise RuntimeError('REMAINING_QA_NOT_108')
 frozen=set(readiness['c_inventory']['frozen_matched_domains'])
 if len(frozen)!=65 or readiness['c_inventory']['positions']!=120 or readiness['c_inventory']['distinct_urls']!=118:
  raise RuntimeError('FROZEN_COHORT_INVALID')
 historical=make_index(old['sites'],'original12')
 recent=make_index(qa108['visits'],'qa108')
 if any(k[2]>2 for k in historical) or any(k[2]<=2 for k in recent):
  raise RuntimeError('OLD_NEW_SLOT_SEPARATION_BROKEN')
 if len(historical)!=12 or len(recent)!=108 or set(historical)&set(recent):
  raise RuntimeError('QA120_SLOT_COVERAGE_INCONSISTENT')
 combined=dict(historical)
 combined.update(recent)
 statuses=collections.Counter()
 by_category=collections.defaultdict(lambda:{'total':0,'ok':0,'http_restricted':0,'technical_or_other':0})
 frozen_success=set()
 frozen_blocked=set()
 observed=[]
 for inst,cats in PAIRS.items():
  for cat in cats:
   f=APP/'site_lists'/inst/(cat+'.txt')
   if not f.is_file() or f.is_symlink():
    raise RuntimeError('MISSING_CANONICAL_SITES '+str(f))
   urls=[s.strip() for s in f.read_text().splitlines() if s.strip() and not s.lstrip().startswith('#')]
   if len(urls)!=20:raise RuntimeError('WRONG_CANONICAL_SLOT_COUNT '+str(f))
   for i,url in enumerate(urls,1):
    k=(inst,cat,i)
    if k not in combined:raise RuntimeError('QA120_MISSING_SLOT '+str(k))
    r=combined[k]
    if r['url']!=url:raise RuntimeError('QA120_SITE_URL_MISMATCH '+str(k))
    allowed=bool(r.get('any_qa_valid')) if i<=2 else bool(r.get('qa_valid'))
    state=(r.get('retry_status') or r.get('baseline_status')) if i<=2 else r.get('status')
    state=state or 'UNKNOWN'
    d=domain(url)
    if not d:raise RuntimeError('MALFORMED_URL')
    frozen_label=d in frozen
    state_type='OK' if allowed else 'HTTP_RESTRICTED' if state in ('BLOCKED','HTTP_ERROR','CHALLENGE','AUTH_REQUIRED') else 'OTHER_FAILURE'
    statuses[state_type]+=1
    by_category[(inst,cat)]['total']+=1
    if allowed:
     by_category[(inst,cat)]['ok']+=1
     if frozen_label:frozen_success.add(d)
    elif state_type=='HTTP_RESTRICTED':
     by_category[(inst,cat)]['http_restricted']+=1
     if frozen_label:frozen_blocked.add(d)
    else:
     by_category[(inst,cat)]['technical_or_other']+=1
    observed.append({'instance':inst,'category':cat,'slot':i,'url':url,'domain':d,
      'frozen_65':frozen_label,'qa_valid_any':allowed,'status':state,
      'evidence_source':'baseline12_with_controlled_retry' if i<=2 else 'remaining108'})
 if len(observed)!=120 or len(set(x['url'] for x in observed))!=118:
  raise RuntimeError('FULL_INVENTORY_BAD')
 inv_domains={x['domain'] for x in observed}
 if len(inv_domains&frozen)!=65:
  raise RuntimeError('CANONICAL_65_CLASS_MAPPING_BROKEN')
 report={'classification':'C_DIRECT_V1_QA_ONLY_NO_SCIENTIFIC_RELEASE',
    'historical_AB_preserved':True,
    'inventory_slots':120,'inventory_distinct_urls':118,'frozen_classes_present':65,
    'frozen_65_successful_site_count':len(frozen_success),
    'frozen_65_missing_success':sorted(frozen-frozen_success),
    'frozen_65_restricted':sorted(frozen_blocked),
    'attempted_slots_with_evidence':len(observed),
    'slots_qa_valid':sum(x['qa_valid_any'] for x in observed),
    'slots_access_restricted':statuses['HTTP_RESTRICTED'],
    'slots_other_failure':statuses['OTHER_FAILURE'],
    'category_breakdown':{inst+'/'+cat:val for (inst,cat),val in sorted(by_category.items())},
    'status_counts':dict(statuses),
    'sources':[str(ORIGINAL),str(RECONCILED_12),str(folder/'qa108_summary.json'),str(READY)],
    'scientific_released':False,
    'blocking_gates':['original_AB_capture_parity','PCAP_background_noise_and_vantage',
       'trusted_UTC_clock','approved_storage_retention','scheduler_validation'],
    'records':observed}
 target=folder/'qa120_coverage_summary.json'
 target.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
 print('QA120_SLOTS_WITH_EVIDENCE',len(observed),'/120')
 print('QA120_VALID',report['slots_qa_valid'],'/120')
 print('QA120_HTTP_RESTRICTED',report['slots_access_restricted'])
 print('QA120_OTHER_FAILURE',report['slots_other_failure'])
 print('QA120_FROZEN65_WITH_VALID_CAPTURE',len(frozen_success),'/65')
 print('QA120_FROZEN65_RESTRICTED',len(frozen_blocked))
 for (inst,cat),v in sorted(by_category.items()):
  print('QA120_CATEGORY',inst,cat,str(v['ok'])+'/20','HTTP_RESTRICTED',v['http_restricted'],'OTHER',v['technical_or_other'])
 print('QA120_COVERAGE_REPORT',target)
 print('SCIENTIFIC_CAPTURE=NOT_AUTHORIZED')
 return 0

if __name__=='__main__':
 try:sys.exit(main())
 except Exception as exc:
  print('QA120_RECONCILIATION_INCONCLUSIVE',type(exc).__name__,str(exc)[:350],file=sys.stderr)
  sys.exit(4)
