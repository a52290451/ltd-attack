#!/usr/bin/env python3
"""P16C failure review and conservative selective recovery.
No anti-bot bypass, no CAPTCHA handling, no 403/429 retries, no edits to frozen A/B.
Usage:
    python3 scripts/dataset_c/review_and_recover_qa.py report
    python3 scripts/dataset_c/review_and_recover_qa.py recover --max-per-category 3
"""
from __future__ import annotations
import argparse, collections, csv, datetime, json, os, pathlib, re, subprocess, sys

BASE=pathlib.Path("/data/ltd-c-qa-bsierra")
APP=pathlib.Path.home()/"ltd-collector-c-smoke-v22/P16C_RECOLECTOR_V2"
SPECS={1:("belleza","turismo"),2:("bancos","salud"),3:("deportes","politica")}
RETRY={"NAVIGATION_ERROR","TIMEOUT","LOW_CONTENT","CONSENT_UNRESOLVED","SERVER_ERROR"}
DO_NOT_RETRY={"BLOCKED"}
def source_run():
    runs=sorted((r for r in BASE.glob("three_hourly_*") if (r/"VALIDATED_THREE_FULL.txt").is_file()),
                key=lambda p:p.stat().st_mtime,reverse=True)
    if not runs: raise RuntimeError("no full 120 QA run found")
    return runs[0]
def rows_from(root):
    for i,cats in SPECS.items():
        for cat in cats:
            d=root/f"inst{i}"/cat
            for p in sorted(d.glob("attempts_*.tsv")):
                with p.open(newline="",encoding="utf-8-sig") as f:
                    for r in csv.DictReader(f,delimiter="\t"):
                        r["instance"]=str(i);r["category"]=cat
                        raw=(r.get("json") or "").strip()
                        if raw.startswith("/audit/") or raw.startswith("/audit120/"):
                            rel=pathlib.Path(*pathlib.Path(raw).parts[2:])
                            candidate=(root/rel).resolve()
                            if candidate.is_relative_to(root.resolve()) and candidate.is_file():
                                try:
                                    meta=json.loads(candidate.read_text(encoding="utf-8"))
                                    r["_http_meta"]=str(meta.get("http_status") or "")
                                except (ValueError,OSError): pass
                        yield r
def slot(r):
    s=str(r.get("slot_index") or r.get("slot") or "")
    if not s.isdigit() or not 1<=int(s)<=20: return None
    return int(s)
def safe_url(r):
    u=r.get("url") or ""
    return u.split("#",1)[0].split("?",1)[0][:180]
def http(r):
    # Different collector releases have different field names.
    v=str(r.get("http_status") or r.get("http") or r.get("_http_meta") or "")
    return int(v) if v.isdigit() else None
def recoverable(r):
    st=(r.get("status") or "").strip()
    if st in DO_NOT_RETRY or st=="OK": return False
    h=http(r)
    if h in (401,403,407,429): return False
    if st=="HTTP_ERROR": return h is not None and 500<=h<=599
    return st in RETRY
def status_report(run):
    rows=list(rows_from(run))
    if len(rows)!=120: raise RuntimeError("STOP: source not a 120-slot run")
    status=collections.Counter(r.get("status") for r in rows)
    print("SOURCE_RUN",run)
    print("STATUS_COUNTS",json.dumps(status,ensure_ascii=False,sort_keys=True))
    choices=[r for r in rows if recoverable(r)]
    excluded=[r for r in rows if r.get("status")!="OK" and not recoverable(r)]
    print("RECOVERABLE_CANDIDATES",len(choices),"EXCLUDED_BLOCKED_OR_POLICY",len(excluded))
    for r in rows:
        if (r.get("status") or "")=="OK": continue
        print("FAIL",f'inst{r["instance"]}/{r["category"]}',"slot",slot(r),
              "status",r.get("status"),"http",http(r),"retry_candidate",recoverable(r),
              "url",safe_url(r))
    return choices
def exec_cmd(args, **kwargs):
    return subprocess.run(args,check=True,text=True,**kwargs)
def tor_true(name):
    p=subprocess.run(["docker","exec",name,"curl","-fsS","--max-time","18","--socks5-hostname","127.0.0.1:9050","https://check.torproject.org/api/ip"],
        text=True,capture_output=True,timeout=25)
    if p.returncode: return False
    try:return json.loads(p.stdout).get("IsTor") is True
    except json.JSONDecodeError:return False
def recover(choices, limit):
    cron=exec_cmd(["crontab","-l"],capture_output=True).stdout
    if "LTD_C_THREE_QA" in cron or "LTD_C_QA_HOURLY" in cron:
        raise RuntimeError("STOP: pause hourly Cron before targeted recovery to avoid overlap")
    if subprocess.run(["docker","ps","--format","{{.Names}}"],text=True,capture_output=True,check=True).stdout.find("ltd-c-capture-")>=0:
        raise RuntimeError("STOP: collector already running")
    free=os.statvfs(BASE).f_bavail*os.statvfs(BASE).f_frsize
    if free<120*1024**3:raise RuntimeError("STOP: storage under 120 GiB free")
    if not APP.joinpath("run_category.sh").is_file():raise RuntimeError("STOP: collector missing")
    for i in SPECS:
        if not tor_true(f"ltd-c-tor-inst{i}"):raise RuntimeError(f"STOP: Tor {i} not ready")
    groups=collections.defaultdict(list)
    for r in choices:
        s=slot(r)
        if s is not None: groups[(int(r["instance"]),r["category"])].append(s)
    for k in list(groups): groups[k]=sorted(set(groups[k]))[:limit]
    root=BASE/("three_targeted_recovery_"+datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+str(os.getpid()))
    root.mkdir(mode=0o700)
    with (root/"PROTOCOL.txt").open("w") as f:
        f.write("TECHNICAL_QA_ONLY; independent re-visit with extended timeout; original observations unmodified.\n")
        f.write("NO_BYPASS: excluded blocked/access-denied/rate-limited, no CAPTCHA or WAF circumvention.\n")
    processes=[]
    for i,cats in SPECS.items():
        specs=[(c,groups[(i,c)]) for c in cats if groups[(i,c)]]
        if not specs: continue
        out=root/f"inst{i}";out.mkdir()
        commands=[]
        for cat,slots in specs:
            csv_slots=",".join(map(str,slots))
            commands.append(f'SITE_SLOT_FILTER="{csv_slots}" bash /opt/ltd-c-v2/run_category.sh inst{i} {cat} || exit 1')
        # Instance/category/slot values are fixed manifest names and bounded integers.
        script='set -euo pipefail; trap \'chown -R "$HOST_UID:$HOST_GID" /audit 2>/dev/null || true\' EXIT; '+'; '.join(commands)
        command=[
            "docker","run","--rm","--name",f"ltd-c-recovery-inst{i}-{os.getpid()}",
            "--network",f"container:ltd-c-tor-inst{i}",
            "--cap-add","NET_RAW","--cap-add","NET_ADMIN",
            "--security-opt","no-new-privileges","--shm-size","1g",
            "--mount",f"type=bind,src={APP},dst=/opt/ltd-c-v2,readonly",
            "--mount",f"type=bind,src={root},dst=/audit",
            "-e","APP_HOME=/opt/ltd-c-v2","-e","NODE_PATH=/app/node_modules",
            "-e","CAPTURE_ROOT=/audit","-e","MAX_URLS=0",
            "-e","CONSENT_POLICY=accept_all","-e","SESSION_POLICY=fresh",
            "-e","CHROME_NO_SANDBOX=1","-e","CHROME_TIMEOUT=95",
            "-e","RUN_WALL_LIMIT=130","-e","MIN_DISK_FREE_GB=120",
            "-e",f"HOST_UID={os.getuid()}","-e",f"HOST_GID={os.getgid()}",
            "--entrypoint","/bin/bash","tor_1site_noise:final","-lc",script]
        handle=(out/"recovery.log").open("w")
        processes.append((i,subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT),handle))
    error=False
    for i,p,h in processes:
        code=p.wait();h.close()
        observed=[r for r in rows_from(root) if int(r["instance"])==i]
        expected=sum(len(groups[(i,cat)]) for cat in SPECS[i])
        ok=sum(r.get("status")=="OK" for r in observed)
        print("RECOVERY",f"inst{i}",f"slots={len(observed)}/{expected}",f"OK={ok}","exit",code)
        if code or len(observed)!=expected:error=True
    print("RECOVERY_OUT",root)
    print("COMPARE_WITH_SOURCE",source_run())
    if error: raise RuntimeError("STOP: targeted recovery incomplete; inspect per-instance recovery logs")
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("action",choices=("report","recover"))
    ap.add_argument("--max-per-category",type=int,default=3)
    args=ap.parse_args()
    if not 1<=args.max_per_category<=5: raise SystemExit("STOP: max-per-category must be 1..5")
    choices=status_report(source_run())
    if args.action=="recover":
        if not choices: print("NO_RECOVERABLE_CANDIDATES");return
        recover(choices,args.max_per_category)
if __name__=="__main__":
    try: main()
    except Exception as e: print("ERROR",str(e),file=sys.stderr);sys.exit(2)
