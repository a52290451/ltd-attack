#!/usr/bin/env python3
"""Bounded Tor readiness check for the three independent Dataset-C QA networks.

Keep Tor running during transient bridge outages; never restart or overwrite its
state. This script only verifies SOCKS connectivity and reports health. A round
is launched only after all three Tor endpoints passed IsTor=true.
"""
import argparse
import concurrent.futures
import json
import subprocess
import sys
import time

NAMES = tuple("ltd-c-tor-inst"+str(i) for i in (1,2,3))

def running(name):
    r=subprocess.run(
        ["docker","inspect","--format","{{.State.Running}}",name],
        capture_output=True,text=True,timeout=10)
    return r.returncode==0 and r.stdout.strip()=="true"

def tor_ready(name):
    try:
        r=subprocess.run(
            ["docker","exec",name,"curl","-fsS","--connect-timeout","6",
             "--max-time","14","--socks5-hostname","127.0.0.1:9050",
             "https://check.torproject.org/api/ip"],
            capture_output=True,text=True,timeout=20)
        return r.returncode==0 and json.loads(r.stdout).get("IsTor") is True
    except (OSError,subprocess.TimeoutExpired,ValueError):
        return False

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--max-wait",type=int,default=360)
    p.add_argument("--interval",type=int,default=12)
    x=p.parse_args()
    if not (0<=x.max_wait<=1200 and 1<=x.interval<=120):
        raise SystemExit("STOP: invalid Tor readiness parameters")
    deadline=time.monotonic()+x.max_wait
    attempt=0
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        while True:
            attempt+=1
            up={name:running(name) for name in NAMES}
            if not all(up.values()):
                for name,state in up.items():
                    if not state:print("TOR_CONTAINER_DOWN",name,flush=True)
                return 3
            statuses=list(pool.map(tor_ready,NAMES))
            for name,ok in zip(NAMES,statuses):
                print("TOR_READY" if ok else "TOR_TEMPORARY_UNAVAILABLE",
                      name,"round="+str(attempt),flush=True)
            if all(statuses):
                print("ALL_THREE_TOR_READY",flush=True)
                return 0
            remaining=deadline-time.monotonic()
            if remaining<=0:
                print("TOR_WAIT_EXPIRED: no capture attempted, Tor state untouched",flush=True)
                return 4
            time.sleep(min(x.interval,remaining))
if __name__=="__main__":
    try:sys.exit(main())
    except Exception as e:
        print("TOR_HEALTH_ERROR",type(e).__name__,str(e)[:160],file=sys.stderr)
        sys.exit(2)
