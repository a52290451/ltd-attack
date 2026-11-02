#!/usr/bin/env python3
"""QA-only action plan from the five known C failures. No website requests."""
import csv,json,pathlib,sys
root=pathlib.Path('/data/ltd-c-qa-bsierra/C_DIRECT_V1')
qa=root/'qa_only/pinned_qa108_20261101T211133Z_465267'
diag=root/'diagnostics/pinned_qa108_20261101T211133Z_465267'
cases={('inst1','turismo','8'):'consent',('inst2','bancos','8'):'consent',
('inst2','salud','6'):'consent',('inst3','politica','14'):'navigation',
('inst3','politica','15'):'navigation'}
rows=list(csv.DictReader((diag/'visits.tsv').open(),delimiter='\t'))
seen=set()
for r in rows:
 k=(r['instance'],r['category'],r['slot'])
 if k not in cases:continue
 seen.add(k)
 meta=qa/r['instance']/r['category']/r['host']/r['day']/(r['uid']+'.json')
 data=json.loads(meta.read_text())
 print('RECOVERY_REVIEW',*k,r['url'],'STATUS',data.get('status'),
       'HTTP',data.get('http_status'),'ERRORS',str(data.get('errors',[]))[:220],
       'CONSENT',str(data.get('consent',{}))[:220])
if len(seen)!=5:raise SystemExit('RECOVERY_REVIEW_INCOMPLETE')
print('NEXT: 3 consent 200 sites need a consent-handler source patch with legitimate button handling; 2 timeouts need one controlled 60s browser check.')
print('SCIENTIFIC_CAPTURE=NOT_AUTHORIZED')
