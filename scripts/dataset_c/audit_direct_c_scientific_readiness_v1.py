#!/usr/bin/env python3
"""Dataset C scientific readiness: frozen 65-site crosswalk + time + QA + protocol.
Reads B frozen source and QA only. Writes *one report* in Dataset C diagnostics.
No model fitting/inference, no updates to A/B, no Tor writes, no capture.
"""
from __future__ import annotations
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from collections import Counter

BASE = Path("/data/ltd-c-qa-bsierra/C_DIRECT_V1")
HOME = Path.home()
REPO = Path(__file__).resolve().parents[2]
APP = HOME / "ltd-collector-c-smoke-v22/P16C_RECOLECTOR_V2"
B_FROZEN_CSV = HOME / "ltd-storage/datasets/future/CLEAN_final_features_sites_concept_drift.csv"
BASELINE = BASE / "diagnostics/pinned_qa12_20261101T203050Z_136527/qa12_summary.json"
RETRY = BASE / "diagnostics/pinned_retry5_20261101T204311Z_239428/combined_readiness.json"
AB_AUDIT = REPO / "docs/protocols/P16C_CAPTURE_SOURCE_AUDIT.md"
RUNBOOK = REPO / "docs/operations/DATASET_C_DIRECT_RECOLLECTION_RUNBOOK_20261030.md"
EXPECTED_HASHES = {
    "capture_dom.js": "137c163bfed44d17d702dffb8e4c15541469ea1d5c41f3c1e96c80cc8c9c6d4c",
    "quality.js": "2e95a90ea77115320d37aaff0ae14157727f7b47deb28a9b18de7237e84f59e1",
    "run_category.sh": "e55edd86abdc4c4f424d9e5b367b5cf360aaf189490195ac484bd44d1f1a948c",
}
PAIRS = {
    "inst1": ("belleza", "turismo"),
    "inst2": ("bancos", "salud"),
    "inst3": ("deportes", "politica"),
}

def run(args, seconds=12):
    try:
        p = subprocess.run(args, check=False, capture_output=True, text=True, timeout=seconds)
        return {"returncode": p.returncode, "stdout": p.stdout.strip()[:1600], "stderr": p.stderr.strip()[:300]}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"returncode": None, "error": type(e).__name__}

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def load_frozen_b_auditor():
    source = REPO / "scripts/dataset_c/audit_c_overlap_future_b.py"
    spec = importlib.util.spec_from_file_location("frozen_b_validator", source)
    if not spec or not spec.loader:
        raise RuntimeError("Cannot load frozen B audit module")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def main():
    if not BASE.is_dir() or BASE.is_symlink():
        raise RuntimeError("Missing safe C QA root")
    if not RETRY.is_file() or not BASELINE.is_file():
        raise RuntimeError("Missing immutable QA12 and controlled retry evidence")
    if not AB_AUDIT.is_file() or not RUNBOOK.is_file():
        raise RuntimeError("Missing A/B capture-source audit and direct protocol runbook")
    actual_hashes = {}
    for name, expected in EXPECTED_HASHES.items():
        p = APP/name
        if not p.is_file() or p.is_symlink():
            raise RuntimeError("Missing or linked pristine collector component: " + name)
        actual = sha(p)
        actual_hashes[name] = actual
        if actual != expected:
            raise RuntimeError("Collector baseline changed: " + name)
    base = json.loads(BASELINE.read_text())
    retry = json.loads(RETRY.read_text())
    if base.get("attempts") != 12 or base.get("valid") != 7:
        raise RuntimeError("Baseline QA12 evidence changed")
    if retry.get("distinct_slots") != 12 or retry.get("combined_valid") != 8 or retry.get("remaining_access_constrained") != 4:
        raise RuntimeError("Controlled retry reconciliation changed")
    frozen = load_frozen_b_auditor()
    if not B_FROZEN_CSV.is_file() or B_FROZEN_CSV.is_symlink():
        raise RuntimeError("Frozen Future-B source is absent or symlink")
    b_domains, label_column, b_raw, b_retained = frozen.b_class_set(B_FROZEN_CSV)
    if len(b_domains) != 65 or b_retained != 18543:
        raise RuntimeError("Frozen 65-site / 18,543-sample contract invalid")
    inventory = []
    for inst, cats in PAIRS.items():
        for cat in cats:
            path = APP/"site_lists"/inst/(cat+".txt")
            if not path.is_file() or path.is_symlink():
                raise RuntimeError("Missing canonical C site list: " + str(path))
            urls = [s.strip() for s in path.read_text(encoding="utf-8").splitlines() if s.strip() and not s.lstrip().startswith("#")]
            if len(urls) != 20:
                raise RuntimeError("C canonical category no longer has 20 slots: " + str(path))
            for slot, url in enumerate(urls, 1):
                d = frozen.domain(url)
                if not d:
                    raise RuntimeError("Cannot normalize C URL: " + url)
                inventory.append({"instance": inst, "category": cat, "slot": slot, "domain": d, "url": url})
    unique_urls = {r["url"] for r in inventory}
    domains = {r["domain"] for r in inventory}
    if len(inventory) != 120 or len(unique_urls) != 118:
        raise RuntimeError("C 120-slot/118-URL frozen inventory invariant failed")
    matched = sorted(domains & b_domains)
    missing = sorted(b_domains-domains)
    outside = sorted(domains-b_domains)
    status = Counter()
    valid_c_domains = set()
    for row in retry.get("sites", []):
        status[row.get("retry_status") or row.get("baseline_status") or "UNKNOWN"] += 1
        if row.get("any_qa_valid"):
            valid_c_domains.add(frozen.domain(row["url"]))
    if len(valid_c_domains) != 8:
        raise RuntimeError("Combined 8 valid QA visits do not map to 8 distinct hosts; review aliases before proceeding")
    ntp = run(["timedatectl", "show", "-p", "NTPSynchronized", "-p", "NTP"])
    sync = "NTPSynchronized=yes" in ntp.get("stdout", "")
    time_status = {
        "host_time_utc_unverified": dt.datetime.now(dt.timezone.utc).isoformat(),
        "timedatectl": ntp,
        "chronyc_tracking": run(["chronyc", "tracking"]),
        "ntp_sync_pass": sync,
        "external_offset_attested": False,
    }
    containers = {}
    for i in (1,2,3):
        n=f"ltd-c-tor-inst{i}"
        inspect=run(["docker", "inspect", "-f",
                     "{{.State.Running}}|{{.HostConfig.RestartPolicy.Name}}|{{.Config.Image}}",
                     n])
        tor_version=run(["docker", "exec", n, "tor", "--version"])
        tor_conf=run(["docker", "exec", n, "grep", "-x", "UseBridges 0", "/tmp/direct-torrc"])
        containers[n] = {
            "inspect": inspect, "tor_version": tor_version,
            "direct_tor_config_confirmed": tor_conf.get("returncode")==0,
        }
    checks = {
      "frozen_65_labels_confirmed": len(b_domains)==65 and b_retained==18543,
      "c_inventory_120_118_unchanged": True,
      "qa_8_of_12_and_4_access_restrictions_documented": True,
      "historical_AB_full_65_domains_present_in_C": len(missing)==0,
      "host_ntp_synchronized": sync,
      # Primary original A/B acquisition artifacts must still be inspected;
      # a descriptive audit/report is not a replacement for original logs/PCAP.
      "ab_capture_vantage_filter_timing_sessions_verified": False,
      "pcap_background_traffic_isolation_verified": False,
      "tor_and_chrome_version_or_methodological_shift_approved": False,
      "scientific_scheduler_deployment_authorized": False,
    }
    results = {
      "classification":"DATASET_C_SCIENTIFIC_READINESS_NOT_RELEASE",
      "frozen_b":{"source":str(B_FROZEN_CSV),"label_column":label_column,
                  "raw_rows":b_raw,"canonical_rows":b_retained,"canonical_domains":len(b_domains)},
      "c_inventory":{"positions":len(inventory),"distinct_urls":len(unique_urls),
                     "distinct_normalized_domains":len(domains),
                     "matched_frozen_65":len(matched),
                     "missing_frozen_65":missing,"extra_C_domains":outside,
                     "frozen_matched_domains":matched},
      "qa":{"original":str(BASELINE),"reconciliation":str(RETRY),
            "valid_slots":retry["combined_valid"],"restricted_sites":retry["remaining_access_constrained"],
            "valid_c_hosts":sorted(valid_c_domains),
            "qa_valid_hosts_in_frozen_b":sorted(valid_c_domains&b_domains),
            "qa_valid_hosts_outside_frozen_b":sorted(valid_c_domains-b_domains),
            "after_retry_status":dict(status)},
      "time":time_status,"tor":containers,"collector_sha256":actual_hashes,
      "guards":checks,
      "readiness":"BLOCKED" if not all(checks.values()) else "REVIEW_REQUIRED",
      "next_actions":["review original A/B capture source/logs for exact vantage/BPF/feature contract",
                      "resolve missing site classes without changing frozen A/B labels",
                      "validate host NTP/UTC or a preregistered traceable clock calibration",
                      "audit Tor background traffic and PCAP per-visit isolation",
                      "preregister irrecoverable HTTP 403/405 handling and comparability",
                      "only then approve recurrent C_DIRECT_V1 acquisition"]
    }
    output_root=BASE/"diagnostics"/("scientific_readiness_"+dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+str(os.getpid()))
    output_root.mkdir(mode=0o700)
    report=output_root/"scientific_readiness.json"
    report.write_text(json.dumps(results,indent=2,ensure_ascii=False)+"\n")
    print("FROZEN_B_COHORT",len(b_domains),"sites",b_retained,"rows")
    print("C_CANONICAL_INVENTORY",len(inventory),"slots",len(unique_urls),"unique_URLs",len(domains),"normalized_domains")
    print("C_B_DOMAIN_INTERSECTION",len(matched),"/65")
    print("MISSING_FROZEN_B_CLASSES_IN_C",len(missing))
    print("EXTRA_C_DOMAINS_OUTSIDE_FROZEN_B",len(outside))
    print("QA_RECONCILED_VALID",retry["combined_valid"],"/12","ACCESS_RESTRICTED",retry["remaining_access_constrained"])
    print("QA_VALID_HOSTS_IN_FROZEN_B",len(valid_c_domains&b_domains),"/8")
    print("NTP_SYNC", "yes" if sync else "no")
    for n, r in containers.items():
        print("TOR_INSTANCE",n,"direct_config",r["direct_tor_config_confirmed"],
              "version",r["tor_version"].get("stdout","unavailable").splitlines()[:1])
    for name, value in checks.items():
        print("READINESS_GATE",name,"PASS" if value else "PENDING")
    print("SCIENTIFIC_READINESS",results["readiness"])
    print("READINESS_REPORT",report)
    print("A_B_DATA_AND_MODELS=UNMODIFIED")
    return 0  # Read-only inventory completed even when scientific gates are blocked.

if __name__=="__main__":
    try:sys.exit(main())
    except Exception as exc:
        print("READINESS_AUDIT_INCONCLUSIVE",type(exc).__name__,str(exc)[:320],file=sys.stderr)
        sys.exit(4)
