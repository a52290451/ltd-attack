#!/usr/bin/env python3
"""Direct-Tor Dataset C operational QA. Never promotes data to science or installs cron.
- Uses the same collector/browser binary as legacy pilot, with direct Tor containers.
- All new output under /data/ltd-c-direct-v1/qa_only.
- Host clock may be unsynchronised: all output explicitly QA-only.
- Old bridge-capture scripts, URLs and datasets are NEVER reused as proof of A/B parity.
"""
from __future__ import annotations
import argparse, csv, datetime as dt, json, os, pathlib, subprocess, sys, hashlib, fcntl

ROOT = pathlib.Path("/data/ltd-c-direct-v1")
QA = ROOT / "qa_only"
APP = pathlib.Path.home() / "ltd-collector-c-smoke-v22/P16C_RECOLECTOR_V2"
BROWSER = "tor_1site_noise:final"
SPECS = {1: ("belleza", "turismo"), 2: ("bancos", "salud"), 3: ("deportes", "politica")}

def run_cmd(args, timeout=25):
    return subprocess.run(args, text=True, capture_output=True, timeout=timeout)

def current_ntp():
    try:
        return run_cmd(["timedatectl", "show", "-p", "NTPSynchronized", "--value"]).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"

def tor_ready(i):
    n = "ltd-c-tor-inst" + str(i)
    try:
        meta = run_cmd(["docker", "inspect", "--format",
                       "{{.State.Running}} {{.HostConfig.RestartPolicy.Name}}", n])
        if meta.returncode or not meta.stdout.startswith("true "):
            return False
        conf = run_cmd(["docker", "exec", n, "cat", "/tmp/direct-torrc"])
        if conf.returncode or "UseBridges 0" not in conf.stdout:
            return False
        if "UseBridges 1" in conf.stdout or "ClientTransportPlugin" in conf.stdout:
            return False
        r = run_cmd(["docker", "exec", n, "curl", "-fsS",
                     "--connect-timeout", "6", "--max-time", "20",
                     "--socks5-hostname", "127.0.0.1:9050",
                     "https://check.torproject.org/api/ip"], timeout=24)
        return r.returncode == 0 and json.loads(r.stdout).get("IsTor") is True
    except (OSError, subprocess.TimeoutExpired, ValueError, KeyError):
        return False

def verify_preflight():
    if not APP.joinpath("run_category.sh").is_file():
        raise RuntimeError("STOP: original collector source missing")
    if not ROOT.is_dir() or ROOT.is_symlink() or ROOT.resolve() != ROOT or ROOT.stat().st_uid != os.getuid():
        raise RuntimeError("STOP: /data/ltd-c-direct-v1 must exist and be owned by operator")
    free = os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize
    if free < 120 * 1024**3:
        raise RuntimeError("STOP: under 120 GiB free on capture filesystem")
    cron = run_cmd(["crontab", "-l"])
    if cron.returncode == 0 and any(("LTD_C_THREE_QA" in line or "LTD_C_QA_HOURLY" in line)
                                   for line in cron.stdout.splitlines()
                                   if line.strip() and not line.lstrip().startswith("#")):
        raise RuntimeError("STOP: obsolete C cron still active")
    missing = [i for i in SPECS if not tor_ready(i)]
    if missing:
        raise RuntimeError("STOP: direct Tor is not verified in instances: " + str(missing))

def collect(i, limit, root):
    n = "ltd-c-tor-inst" + str(i)
    out = root / ("inst" + str(i))
    out.mkdir(mode=0o700, exist_ok=False)
    cats = SPECS[i]
    body = ('set -uo pipefail; '
            'trap \'chown -R "$HOST_UID:$HOST_GID" /audit 2>/dev/null || true\' EXIT; '
            'rc=0; for category in ' + " ".join(cats) + '; do '
            'if bash /opt/ltd-c-v2/run_category.sh inst' + str(i) + ' "$category"; then '
            'echo "CATEGORY_SUCCESS $category"; else '
            'echo "CATEGORY_FAILED $category"; rc=1; fi; done; exit "$rc"')
    args = ["docker", "run", "--rm", "--name", "ltd-c-direct-qa-inst" + str(i) + "-" + str(os.getpid()),
            "--network", "container:" + n, "--cap-add", "NET_RAW", "--cap-add", "NET_ADMIN",
            "--security-opt", "no-new-privileges", "--shm-size", "1g",
            "--mount", "type=bind,src=" + str(APP) + ",dst=/opt/ltd-c-v2,readonly",
            "--mount", "type=bind,src=" + str(root) + ",dst=/audit",
            "-e", "APP_HOME=/opt/ltd-c-v2", "-e", "NODE_PATH=/app/node_modules",
            "-e", "CAPTURE_ROOT=/audit", "-e", "MAX_URLS=" + str(limit),
            "-e", "CONSENT_POLICY=accept_all", "-e", "SESSION_POLICY=fresh",
            "-e", "CHROME_NO_SANDBOX=1", "-e", "MIN_DISK_FREE_GB=120",
            "-e", "HOST_UID=" + str(os.getuid()), "-e", "HOST_GID=" + str(os.getgid()),
            "--entrypoint", "/bin/bash", BROWSER, "-lc", body]
    log = (out / "capture.log").open("w", encoding="utf-8")
    try:
        return subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT), log
    except Exception:
        log.close()
        raise

def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["--check", "--smoke"], help="QA only; never scientific or scheduled")
    args = p.parse_args()
    verify_preflight()
    print("DIRECT_TOR_3_INSTANCE_PREFLIGHT_PASS", flush=True)
    print("HOST_NTP_SYNC=" + current_ntp(), flush=True)
    if args.mode == "--check":
        return 0
    QA.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (QA / ".run.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("STOP: another QA is still running")
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        rid = "direct_smoke_" + stamp + "_" + str(os.getpid())
        root = QA / rid
        root.mkdir(mode=0o700)
        ntp = current_ntp()
        meta = {
            "identity": rid, "classification": "QA_ONLY_NOT_FOR_MODEL_TRAINING_OR_INFERENCE",
            "transport": "DIRECT_TOR_NO_BRIDGES", "host_ntp_sync": ntp,
            "host_utc_unverified": dt.datetime.now(dt.timezone.utc).isoformat(),
            "capture_protocol": "legacy_c_browser_collector_unverified_vs_A_B",
            "max_urls_per_category": 2, "expected_slots": 12,
            "disclaimer": "Labels/timestamps/vantage not yet certified comparable to Historical A/B",
        }
        (root / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        tasks = [collect(i, 2, root) for i in SPECS]
        status = {}
        for i, (proc, stream) in zip(SPECS, tasks):
            status[str(i)] = proc.wait()
            stream.close()
        slots = 0
        for path in root.glob("inst*/**/attempts_*.tsv"):
            with path.open(newline="", encoding="utf-8") as f:
                slots += sum(1 for _ in csv.DictReader(f, delimiter="\t"))
        meta["exit_codes"] = status
        meta["observed_slots"] = slots
        meta["status"] = "QA_PASS" if slots == 12 and all(v == 0 for v in status.values()) else "QA_INCOMPLETE"
        (root / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        print("DIRECT_C_QA_OUTPUT=" + str(root), flush=True)
        print("DIRECT_C_QA_SLOTS=" + str(slots) + "/12", flush=True)
        print("DIRECT_C_QA_STATUS=" + meta["status"], flush=True)
        print("SCIENTIFIC_DATASET_C_CAPTURE=NOT_AUTHORIZED", flush=True)
        return 0 if meta["status"] == "QA_PASS" else 5

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print("DIRECT_C_QA_STOP " + type(exc).__name__ + ": " + str(exc)[:250], file=sys.stderr)
        sys.exit(2)
