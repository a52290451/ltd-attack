#!/usr/bin/env python3
"""Offline revalidation of final29 existing artifacts; no network or Docker calls."""
import json
import collections
import pathlib
import final29_recovery_qa_v1 as audit

run='qa_final29_20261102T111428Z_12857'
d=audit.BASE/'diagnostics'/run
files=audit.BASE/'qa_only'/run/'outputs'
old=json.loads((d/'final29_summary.json').read_text())
rows=old['records']
if len(rows)!=58:
 raise SystemExit('Expected exactly 58 existing visits')
rechecked=[]
for row in rows:
 item=audit.evaluate(row,files)
 rechecked.append(item)
 print('RECHECK',row['phase'],row['domain'],'HTTP',item['http_status'],'STATUS',item['status'],
       'PCAP_PACKETS',item['pcap_packets'],'QA_VALID',item['qa_valid'],flush=True)
keys={(r['instance'],r['category'],r['slot']) for r in rechecked if r['qa_valid']}
statuses=collections.Counter(x['status'] for x in rechecked)
out={
 'status':'QA_ONLY_RECONCILED_EXISTING_CAPTURES',
 'original_report':str(d/'final29_summary.json'),
 'network_requests_performed':0,
 'records_evaluated':len(rechecked),
 'new_valid_positions':len(keys),
 'overall_valid_positions':73+len(keys),
 'historical_frozen65_valid':47,
 'counts':dict(statuses),
 'passing_sites':[list(k) for k in sorted(keys)],
 'records':rechecked,
 'scientific_authorized':False
}
p=d/'final29_reconciled_v1.json'
p.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
print('RECONCILED_VALID',len(keys),'/29')
print('RECONCILED_TOTAL',str(73+len(keys))+'/120')
print('RECONCILED_STATUS_COUNTS',dict(statuses))
print('RECONCILED_REPORT',p)
print('SCIENTIFIC_CAPTURE=NOT_AUTHORIZED')
