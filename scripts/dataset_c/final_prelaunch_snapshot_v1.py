#!/usr/bin/env python3
"""Dataset C: final pre-launch evidence snapshot. NO production or A/B writes.
No model input, no cron scheduling, no capture, no container changes.
Research cohort and site-list hashes are not modified.
"""
from __future__ import annotations
import collections
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from urllib.parse import urlsplit

ROOT = Path("/data/ltd-c-qa-bsierra/C_DIRECT_V1")
QA12 = ROOT / "diagnostics/pinned_qa12_20261101T203050Z_136527/qa12_summary.json"
RECONCILE = ROOT / "diagnostics/pinned_retry5_20261101T204311Z_239428/combined_readiness.json"
EXPECTED_REFERENCE = ROOT / "diagnostics/scientific_readiness_20261101T205205Z_312672/scientific_readiness.json"
REPORT_SUBDIR = "prelaunch_snapshot_"
RESERVE_GB = 120

def exec_readonly(args, timeout=10):
    try:
        cp = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
        return {"rc": cp.returncode, "out": cp.stdout.strip()[:2200],
                "error": cp.stderr.strip()[:350]}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"rc": None, "error": type(e).__name__}

def hostname(url):
    try:
        x = urlsplit(str(url)).hostname
        return x.lower().removeprefix("www.") if x else ""
    except ValueError:
        return ""

def safe_load(path):
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("Missing read-only QA evidence: " + str(path))
    return json.loads(path.read_text())

def main():
    for p in (ROOT, ROOT/"diagnostics"):
        if not p.is_dir() or p.is_symlink():
            raise RuntimeError("Missing safe C-only diagnostics root")
    b = safe_load(QA12)
    combo = safe_load(RECONCILE)
    readiness = safe_load(EXPECTED_REFERENCE)
    inv = readiness.get("c_inventory", {})
    frozen = set(inv.get("frozen_matched_domains", []))
    if (len(frozen)!=65 or len(inv.get("missing_frozen_65", []))!=0
        or inv.get("positions")!=120 or inv.get("distinct_urls")!=118):
        raise RuntimeError("Frozen 65-of-120 mapping not confirmed")
    if (b.get("attempts")!=12 or b.get("valid")!=7
        or combo.get("combined_valid")!=8 or combo.get("remaining_access_constrained")!=4):
        raise RuntimeError("Expected QA12 + controlled retry evidence changed")
    rows=combo.get("sites",[])
    if len(rows)!=12:
        raise RuntimeError("Expected 12 immutable reconciled QA records")
    restricted=[]
    for r in rows:
        if r.get("unavailable_after_retry"):
            domain=hostname(r["url"])
            restricted.append({"site":domain,"instance":r["instance"],
                               "category":r["category"],"slot":r["slot"],
                               "in_frozen_65":domain in frozen,
                               "baseline_status":r["baseline_status"],
                               "retry_status":r["retry_status"]})
    if len(restricted)!=4:
        raise RuntimeError("Unexpected HTTP restriction count")
    frozen_blocked=sum(x["in_frozen_65"] for x in restricted)
    qa_good_frozen=sum(bool(x.get("any_qa_valid")) and hostname(x["url"]) in frozen for x in rows)
    # Retain observed QA volumes ONLY, do not claim they estimate full collection reliably.
    base_success = [int(x["pcap_bytes"]) for x in b["visits"] if x.get("qa_valid") and isinstance(x.get("pcap_bytes"),int)]
    # The successful retry is AS.com, explicitly documented in original evidence.
    retry_path=ROOT/"diagnostics/pinned_retry5_20261101T204311Z_239428/qa12_summary.json"
    retry=safe_load(retry_path)
    retry_success=[int(x["pcap_bytes"]) for x in retry["visits"] if x.get("qa_valid") and isinstance(x.get("pcap_bytes"),int)]
    observed=base_success+retry_success
    if len(observed)!=8 or min(observed)<=0:
        raise RuntimeError("Missing eight actual measured QA PCAP sizes")
    usage=shutil.disk_usage(ROOT)
    available_after_reserve=max(0,usage.free-RESERVE_GB*10**9)
    # Only hypothetical: 120 slots per hour as in the original protocol description.
    # This is NOT a storage forecast and excludes blocked pages, HTML, metadata and logs.
    mean_bytes=statistics.mean(observed)
    scenarios=[]
    for days in (7,30,60):
        n=120*24*days
        scenarios.append({"days":days,"hypothetical_visits":n,
                          "pcap_bytes_if_every_visit_matches_8_QA_mean":round(n*mean_bytes),
                          "assumption":"120 visits/hour; mean of only 8 successful QA captures; highly uncertain"})
    clock={"timedatectl":exec_readonly(["timedatectl","show","-p","NTP","-p","NTPSynchronized"]),
           "chronyc_tracking":exec_readonly(["chronyc","tracking"]),
           "clock_offset_proven_with_two_sources":False}
    clock["ntp_yes"] = "NTPSynchronized=yes" in clock["timedatectl"].get("out","")
    tor={}
    for i in (1,2,3):
        name=f"ltd-c-tor-inst{i}"
        tor[name]={
            "running":exec_readonly(["docker","inspect","-f","{{.State.Running}}",name]),
            "direct_no_bridges":exec_readonly(["docker","exec",name,"grep","-qx","UseBridges 0","/tmp/direct-torrc"]),
            "tor_version":exec_readonly(["docker","exec",name,"tor","--version"]),
        }
    user_cron=exec_readonly(["crontab","-l"])
    container_list=exec_readonly(["docker","ps","--format","{{.Names}} {{.Image}} {{.Status}}"])
    warnings=[
        "A/B source collectors, original BPF, capture interface and PCAP noise control not independently reconciled",
        "A/B vs C browser/Tor/version/session/navigation feature contract not signed off",
        "Full 120-slot unattended QA and per-label availability rates not established",
        "Frozen 65 sites are present as URLs; QA only sampled 12 slots",
        "No scientific C clock release until time source calibration verified",
        "No scientific C cron, retention, space or disk backpressure policy authorized",
    ]
    gates={
        "frozen_65_c_labels_present":True,
        "pinned_browser_native_tor_navigation_validated":True,
        "qa_pcaps_valid_at_eight_of_twelve_slots":True,
        "full_120_slot_operational_QA":False,
        "historical_AB_capture_method_parity":False,
        "pcap_background_noise_isolated":False,
        "clock_synchronized_and_offset_verified":False, # NTP alone is not an external offset audit
        "storage_and_retention_plan_approved":False,
        "scheduler_release_signed":False,
    }
    stamp=dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out=ROOT/"diagnostics"/(REPORT_SUBDIR+stamp+"_"+str(os.getpid()))
    out.mkdir(mode=0o700)
    result={
        "classification":"QA_ONLY_PRELAUNCH_INSPECTION","scientific_authorization":False,
        "frozen_cohort":{"sites":65,"matched_to_C":65,"extra_C_domains":len(inv.get("extra_C_domains",[]))},
        "qa":{"accepted":8,"attempted_unique_slots":12,"restricted":restricted,
              "restricted_in_frozen_65":frozen_blocked,
              "accepted_in_frozen_65":qa_good_frozen},
        "measured_successful_QA_pcap_bytes":observed,
        "observed_QA_pcap_mean_bytes":round(mean_bytes),
        "storage":{"free_bytes":usage.free,"used_bytes":usage.used,
                   "reserve_policy_bytes":RESERVE_GB*10**9,
                   "available_after_reserve_bytes":available_after_reserve,
                   "hypothetical_unverified_scenarios":scenarios},
        "clock":clock,"tor":tor,"user_cron":user_cron,"docker_running":container_list,
        "gates":gates,"pending":warnings,
        "references":[str(QA12),str(RECONCILE),str(EXPECTED_REFERENCE)],
    }
    dest=out/"prelaunch_snapshot.json"
    dest.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print("FROZEN_A_B_CLASSES_IN_C=65/65")
    print("C_120_SLOTS=UNCHANGED; EXTRA_OUTSIDE_65="+str(len(inv.get("extra_C_domains",[]))))
    print("QA_ACCEPTED=8/12")
    print("QA_ACCEPTED_IN_FROZEN_65="+str(qa_good_frozen)+"/12")
    print("RESTRICTED_4_IN_FROZEN_65="+str(frozen_blocked)+"/4")
    for rec in restricted:
        print("RESTRICTED_SITE",rec["site"],"FROZEN_65",rec["in_frozen_65"],
              "STATUS",rec["baseline_status"],rec["retry_status"])
    print("NTP_SYNCHRONIZED",clock["ntp_yes"])
    print("DISK_FREE_GB_DECIMAL",round(usage.free/10**9,1))
    print("DISK_AFTER_120GB_RESERVE_GB_DECIMAL",round(available_after_reserve/10**9,1))
    print("OBSERVED_8_QA_PCAP_MEAN_MB",round(mean_bytes/10**6,2))
    for sc in scenarios:
        print("ILLUSTRATIVE_120_PER_HOUR",sc["days"],"DAYS",
              round(sc["pcap_bytes_if_every_visit_matches_8_QA_mean"]/10**9,1),"GB_PCAP_ONLY",
              "NOT_A_FORECAST")
    for k,v in gates.items():
        print("RELEASE_GATE",k,"PASS" if v else "PENDING")
    print("SCIENTIFIC_CAPTURE=NOT_AUTHORIZED")
    print("SNAPSHOT_REPORT",dest)
    return 0

if __name__=="__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print("PRELAUNCH_INCONCLUSIVE",type(e).__name__,str(e)[:340],file=sys.stderr)
        sys.exit(4)
