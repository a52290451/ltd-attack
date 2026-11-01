#!/usr/bin/env bash
# Dataset C 2026-11: read-only triage of actual pinned QA12 attempts.
# Does not rerun websites or manipulate PCAP/Tor, A/B, clock or canonical inventory.
set -euo pipefail
BASE=/data/ltd-c-qa-bsierra/C_DIRECT_V1
if [[ $# -gt 1 ]]; then echo "Usage: $0 [diagnostics/pinned_qa12_RUN directory]"; exit 2; fi
if [[ $# -eq 1 ]]; then
 RUN="$1"
else
 RUN="$(find "$BASE/diagnostics" -mindepth 1 -maxdepth 2 -path '*/pinned_qa12_*/qa12_summary.json' -type f \
        -printf '%T@ %h\n' | sort -nr | head -1 | cut -d' ' -f2-)"
fi
[[ -n "$RUN" && -d "$RUN" && -f "$RUN/qa12_summary.json" && -f "$RUN/visits.tsv" ]] || {
 echo 'TRIAGE_STOP: pinned QA12 run missing'; exit 3;
}
python3 - "$RUN" "$BASE" <<'PY'
import sys,json,pathlib,csv,collections,re,urllib.parse
run=pathlib.Path(sys.argv[1]).resolve()
base=pathlib.Path(sys.argv[2]).resolve()
if not run.is_relative_to(base/'diagnostics') or not run.name.startswith('pinned_qa12_'):
 raise SystemExit('TRIAGE_STOP: untrusted QA diagnostic folder')
root=base/'qa_only'/run.name
summary=json.loads((run/'qa12_summary.json').read_text())
plans=list(csv.DictReader((run/'visits.tsv').open(newline=''),delimiter='\t'))
if len(plans)!=12:raise SystemExit('TRIAGE_STOP: expected exactly 12 planned visits')
def safe_value(obj,max_len=450):
 text=json.dumps(obj,ensure_ascii=False) if isinstance(obj,(dict,list)) else str(obj)
 # Avoid disclosing secrets inadvertently included in redirected URLs or browser errors.
 text=re.sub(r'(?i)\b(authorization|cookie|token|password|secret|api[_-]?key)=\S+',r'\1=[REDACTED]',text)
 text=re.sub(r'https?://[^\s"\\]+', lambda m:urllib.parse.urlunsplit((
  urllib.parse.urlsplit(m.group()).scheme,urllib.parse.urlsplit(m.group()).netloc,
  urllib.parse.urlsplit(m.group()).path,'','')),text)
 return text.replace('\n',' ')[:max_len]
def safe_url(s):
 try:
  u=urllib.parse.urlsplit(str(s))
  return urllib.parse.urlunsplit((u.scheme,u.netloc,u.path,'',''))
 except Exception:return '[unparseable]'
categories=collections.defaultdict(lambda:{'success':0,'total':0})
statuses=collections.Counter()
print('PINNED_QA12_TRIAGE_BEGIN',run)
print('ORIGINAL_SUMMARY', 'attempts',summary.get('attempts'),'valid',summary.get('valid'),'passed',summary.get('passed'))
for entry in summary.get('visits',[]):
 categories[(entry.get('instance'),entry.get('category'))]['total']+=1
 categories[(entry.get('instance'),entry.get('category'))]['success']+=bool(entry.get('qa_valid'))
 statuses[entry.get('status','UNSPECIFIED')]+=1
print('STATUS_BREAKDOWN',json.dumps(statuses,ensure_ascii=False))
for (inst,cat),st in sorted(categories.items()):
 print('CATEGORY_COVERAGE',inst,cat,str(st['success'])+'/'+str(st['total']))
print('CATEGORY_ANY_SUCCESS',sum(int(x['success']>0) for x in categories.values()),'/',len(categories))
valid=0
for p in plans:
 relative=pathlib.Path(p['instance'])/p['category']/p['host']/p['day']/p['uid']
 file=root/(str(relative)+'.json')
 try:data=json.loads(file.read_text())
 except (ValueError,OSError) as exc:data={'metadata_unreadable':type(exc).__name__}
 match=[x for x in summary.get('visits',[]) if x.get('instance')==p['instance'] and
  x.get('category')==p['category'] and str(x.get('slot'))==str(p['slot'])]
 if len(match)!=1:raise SystemExit('TRIAGE_STOP: plan/results mapping ambiguous')
 rec=match[0]
 if rec.get('qa_valid'):
  valid+=1
  continue
 print('=== FAILED_SITE',p['instance'],p['category'],'slot='+p['slot'],'===')
 print('REQUESTED_URL',safe_url(p['url']))
 print('ORIGINAL_STATUS',rec.get('status'),'NODE_EXIT',rec.get('node_exit'))
 for field in ('status','http_status','final_url','requested_url','navigation_error','error',
    'errors','reason','title','visible_chars','blocked_reason','blocked_by','redirect_chain',
    'cloudflare','captcha_detected','bot_detection','elapsed_ms','duration_ms'):
  if field in data:
   v=safe_url(data[field]) if field.endswith('_url') else safe_value(data[field])
   print('METADATA',field,v)
 print('METADATA_KEYS',','.join(sorted(data.keys())))
 print('PCAP','bytes='+str(rec.get('pcap_bytes')),'packets='+str(rec.get('packet_count')),
       'parsed='+str(rec.get('pcap_parsed')))
 log=run/('browser_'+p['uid']+'.log')
 if log.is_file():
  terms=re.compile('HTTP|error|blocked|captcha|challenge|status|403|429|503|cloudflare|timeout|redirect|navigation|failed',re.I)
  lines=[x.strip() for x in log.read_text(errors='replace').splitlines() if terms.search(x)]
  print('BROWSER_LOG_MATCHES',len(lines),'showing_last',min(8,len(lines)))
  for line in lines[-8:]:print('BROWSER_LOG',safe_value(line,350))
 else:print('BROWSER_LOG','NOT_FOUND')
print('QA_VALID_COUNT',valid,'/',len(plans))
print('DIAGNOSTIC_COMPLETE; NO_RETRIES_NO_CAPTURE_NO_MUTATIONS')
PY
