#!/usr/bin/env python3
"""Diagnóstico puntual del runtime obfs4. No modifica bridge ni los datasets.
Prueba ejecución en un directorio efímero, como debian-tor, con y sin shim.
Detiene los 3 Tor defectuosos después de capturar resultados (sin borrarlos).
"""
import json
import subprocess
import sys

CONTAINERS = ("ltd-c-manual-tor", "ltd-c-tor-inst1", "ltd-c-tor-inst2", "ltd-c-tor-inst3")
PROBE = r'''
import os, re, subprocess, tempfile
def drop():
    os.setgroups([])
    os.setgid(101)
    os.setuid(100)
def sanitize(s):
    s=re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\b", "[IP]", s)
    s=re.sub(r"(?i)\b(?:bridge|cert|fingerprint|iat-mode)\b\s*[=:]\s*\S+", "[REDACTED]", s)
    s=re.sub(r"[A-Za-z0-9+/]{48,}={0,2}", "[TOKEN]", s)
    return s[:450]
print("EXECUTABLE", "yes" if os.path.isfile("/usr/bin/obfs4proxy") else "no",flush=True)
for mode in ("clean", "with-shim"):
    with tempfile.TemporaryDirectory(prefix="ltd-pt-diagnostic-") as state:
        os.chown(state,100,101)
        os.chmod(state,0o700)
        env={
            "PATH":"/usr/local/bin:/usr/bin:/bin",
            "HOME":"/var/lib/tor-ltd",
            "TOR_PT_MANAGED_TRANSPORT_VER":"1",
            "TOR_PT_CLIENT_TRANSPORTS":"obfs4",
            "TOR_PT_STATE_LOCATION":state,
        }
        if mode=="with-shim":
            env["LD_PRELOAD"]="/usr/local/lib/libltd_realtime.so"
            env["LTD_CLOCK_OFFSET_SECONDS"]="-2963"
        try:
            p=subprocess.Popen(
                ["/usr/bin/obfs4proxy"],env=env,
                stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                stdin=subprocess.PIPE,text=True,preexec_fn=drop,
            )
            try:
                out,err=p.communicate(timeout=4)
                rc=str(p.returncode)
            except subprocess.TimeoutExpired:
                p.kill()
                out,err=p.communicate()
                rc="TIMEOUT"
            success="CMETHOD obfs4" in out and "CMETHODS DONE" in out
            print("PROBE",mode,"READY" if success else "FAIL","exit",rc,flush=True)
            if not success:
                errs=[sanitize(line) for line in (err+"\n"+out).splitlines() if line.strip()]
                for line in errs[:5]: print("DETAIL",line,flush=True)
        except Exception as exc:
            print("PROBE",mode,"ERROR",sanitize(str(exc)),flush=True)
'''
def call(*args, input_text=None, timeout=45):
    return subprocess.run(
        list(args), input=input_text, text=True, capture_output=True, timeout=timeout
    )
def main():
    failures=0
    for name in CONTAINERS:
        print("\n===== " + name + " =====",flush=True)
        state=call("docker","inspect",name,"--format",
            "image={{.Image}}; network={{.HostConfig.NetworkMode}}; user={{.Config.User}}; security={{json .HostConfig.SecurityOpt}}",timeout=10)
        print(state.stdout.strip() or ("INSPECT_ERROR "+str(state.returncode)))
        result=call("docker","exec","-i",name,"python3","-",input_text=PROBE)
        if result.returncode:
            print("PROBE_ERROR",result.returncode)
            print(result.stderr[:350])
            failures+=1
        else:
            print(result.stdout.strip())
            if "PROBE clean READY" not in result.stdout:
                failures+=1
    print("\n===== STOP BROKEN TOR CONTAINERS =====",flush=True)
    for i in (1,2,3):
        name="ltd-c-tor-inst"+str(i)
        result=call("docker","stop","--time","8",name,timeout=20)
        print(name,"STOPPED" if result.returncode==0 else "STOP_ERROR "+str(result.returncode),flush=True)
    print("ORIGINAL_TOR_UNTOUCHED")
    print("THREE_INSTANCE_CRON_NOT_INSTALLED_BY_THIS_SCRIPT")
    print("DIAG_RESULT=", "DIFFERENCE_FOUND_OR_FAILURE" if failures else "DIRECT_BINARY_READY_CHECK_TOR_INVOKER")
if __name__=="__main__":
    main()
