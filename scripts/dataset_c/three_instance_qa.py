#!/usr/bin/env python3
"""Prepara 3 procesos Tor aislados y QA paralelo por instancia. No activa cron."""
import argparse, os, pathlib, subprocess, sys, time, json, datetime
BASE=pathlib.Path("/data/ltd-c-qa-bsierra")
APP=pathlib.Path.home()/"ltd-collector-c-smoke-v22/P16C_RECOLECTOR_V2"
IMAGE="ltd-c-tor-nyx:trixie"
BROWSER="tor_1site_noise:final"
SOURCE="ltd-c-manual-tor"
SPECS={1:("belleza","turismo"),2:("bancos","salud"),3:("deportes","politica")}
def cmd(*a,check=True,capture=False):
    return subprocess.run(list(map(str,a)),check=check,text=True,stdout=subprocess.PIPE if capture else None,stderr=subprocess.PIPE if capture else None)
def exist(n):
    return cmd("docker","container","inspect",n,check=False,capture=True).returncode==0
def tor_ok(n):
    x=cmd("docker","exec",n,"curl","-fsS","--max-time","18","--socks5-hostname","127.0.0.1:9050","https://check.torproject.org/api/ip",check=False,capture=True)
    try: return x.returncode==0 and json.loads(x.stdout).get("IsTor") is True
    except ValueError: return False
def preflight():
    if BASE.resolve()!=BASE or not BASE.is_dir() or BASE.stat().st_uid != os.getuid(): raise RuntimeError("STOP: BASE debe ser directorio propio en /data")
    if not APP.joinpath("run_category.sh").is_file(): raise RuntimeError("STOP: falta APP")
    if not exist(SOURCE) or not tor_ok(SOURCE): raise RuntimeError("STOP: Tor fuente no disponible")
    if "LTD_C_QA_HOURLY" in cmd("crontab","-l",check=False,capture=True).stdout: raise RuntimeError("STOP: Cron anterior activo")
    free=os.statvfs(BASE).f_bavail*os.statvfs(BASE).f_frsize
    if free < 120*1024**3: raise RuntimeError("STOP: menos de 120 GiB libres")
    for i in SPECS:
        n="isolated_net_1site_inst"+str(i)
        x=cmd("docker","network","inspect",n,"--format","{{len .Containers}}",capture=True).stdout.strip()
        if x!="0": raise RuntimeError("STOP: red ocupada "+n)
def setup():
    preflight()
    confdir=BASE/"three_tor_config"; confdir.mkdir(mode=0o700,exist_ok=True)
    os.chmod(confdir,0o700)
    secret=confdir/"torrc-obfs4"
    if not secret.exists():
        cmd("docker","cp",SOURCE+":/etc/tor/torrc-obfs4",str(secret))
    os.chmod(secret,0o600)
    assert secret.stat().st_size>10
    # The working container has obfs4proxy installed in its writable layer;
    # the base image does not. Stage the executable for new instances.
    binary=confdir/"obfs4proxy"
    if not binary.exists():
        cmd("docker","cp","-L",SOURCE+":/usr/bin/obfs4proxy",str(binary))
    os.chmod(binary,0o755)
    assert binary.is_file() and binary.stat().st_size>1024
    for i in SPECS:
        name="ltd-c-tor-inst"+str(i)
        if exist(name): raise RuntimeError("STOP: "+name+" ya existe; no sobrescribir")
    for i in SPECS:
        name="ltd-c-tor-inst"+str(i)
        vol="ltd-c-tor-state-inst"+str(i)
        cmd("docker","volume","create",vol)
        # La unidad de reinicio contiene Tor como proceso principal supervisado,
        # sin utilizar el estado del Tor fuente.
        init='chown 100:101 /etc/tor/torrc-obfs4; chmod 600 /etc/tor/torrc-obfs4; chown -R 100:101 /var/lib/tor-ltd; chmod 700 /var/lib/tor-ltd; exec su -s /bin/sh debian-tor -c "exec env LTD_CLOCK_OFFSET_SECONDS=-2963 LD_PRELOAD=/usr/local/lib/libltd_realtime.so tor -f /etc/tor/torrc-obfs4"'
        cmd("docker","create","--name",name,"--network","isolated_net_1site_inst"+str(i),
            "--restart","unless-stopped","--security-opt","no-new-privileges",
            "--mount","type=volume,src="+vol+",dst=/var/lib/tor-ltd",
            "--entrypoint","/bin/sh",IMAGE,"-c",init)
        cmd("docker","cp",str(secret),name+":/etc/tor/torrc-obfs4")
        cmd("docker","cp",str(binary),name+":/usr/bin/obfs4proxy")
        cmd("docker","start",name)
        print("CREADO",name,vol,flush=True)
    for i in SPECS:
        name="ltd-c-tor-inst"+str(i)
        good=False
        for _ in range(18):
            if tor_ok(name): good=True;break
            time.sleep(10)
        print("TOR_OK" if good else "TOR_FAIL",name,flush=True)
        if not good: raise RuntimeError("STOP: Tor no alcanzó IsTor=true: "+name)
def collect(i,max_urls,root):
    n="ltd-c-tor-inst"+str(i)
    if not tor_ok(n): return 4
    output=root/("inst"+str(i));output.mkdir(parents=True,exist_ok=True)
    specs=SPECS[i]
    script='set -uo pipefail; trap \'chown -R "$HOST_UID:$HOST_GID" /audit 2>/dev/null || true\' EXIT; rc=0; for category in '+ " ".join(specs) +'; do echo "INICIO inst'+str(i)+'/$category"; if bash /opt/ltd-c-v2/run_category.sh inst'+str(i)+' "$category"; then echo "FIN inst'+str(i)+'/$category"; else echo "ERROR inst'+str(i)+'/$category"; rc=1; fi; done; exit "$rc"'
    args=["docker","run","--rm","--name","ltd-c-capture-inst"+str(i)+"-"+str(os.getpid()),
        "--network","container:"+n,"--cap-add","NET_RAW","--cap-add","NET_ADMIN",
        "--security-opt","no-new-privileges","--shm-size","1g",
        "--mount","type=bind,src="+str(APP)+",dst=/opt/ltd-c-v2,readonly",
        "--mount","type=bind,src="+str(root)+",dst=/audit",
        "-e","APP_HOME=/opt/ltd-c-v2","-e","NODE_PATH=/app/node_modules",
        "-e","CAPTURE_ROOT=/audit","-e","MAX_URLS="+str(max_urls),
        "-e","CONSENT_POLICY=accept_all","-e","SESSION_POLICY=fresh",
        "-e","CHROME_NO_SANDBOX=1","-e","MIN_DISK_FREE_GB=120",
        "-e","HOST_UID="+str(os.getuid()),"-e","HOST_GID="+str(os.getgid()),
        "--entrypoint","/bin/bash",BROWSER,"-lc",script]
    return subprocess.Popen(args,stdout=(output/"capture.log").open("w"),stderr=subprocess.STDOUT)
def run(max_urls):
    preflight_ready()
    import fcntl, concurrent.futures, csv
    lock=BASE/".three_qa.lock"
    with lock.open("a+") as f:
        try: fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError("STOP: ronda paralela anterior activa")
        rid=("three_smoke_" if max_urls<20 else "three_hourly_")+datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+str(os.getpid())
        root=BASE/rid;root.mkdir(mode=0o700)
        print("OUT="+str(root),flush=True)
        (root/"PILOT_METADATA.txt").write_text("QA_OPERATIVO_NO_CIENTIFICO; obfs4; reloj_host_no_sincronizado\\n")
        proc=[(i,collect(i,max_urls,root)) for i in SPECS]
        results={}
        for i,p in proc:
            if isinstance(p,int): results[i]=p
            else: results[i]=p.wait()
        expected=2*max_urls
        for i in SPECS:
            files=list((root/("inst"+str(i))).rglob("attempts_*.tsv"))
            count=0
            for path in files:
                with path.open() as fcsv: count+=sum(1 for _ in csv.DictReader(fcsv,delimiter="\t"))
            print("INST",i,"SLOTS",str(count)+"/"+str(expected),"EXIT",results[i],flush=True)
            if results[i]!=0 or count!=expected: results[i]=1
        overall=0 if all(x==0 for x in results.values()) else 1
        if overall==0 and max_urls==20:
            (root/"VALIDATED_THREE_FULL.txt").write_text("THREE_FULL_120_OK\\n")
        print("FIN THREE_EXIT="+str(overall)+" OUT="+str(root),flush=True)
        return overall
def preflight_ready():
    if BASE.resolve()!=BASE: raise RuntimeError("STOP: BASE no es /data")
    if "LTD_C_QA_HOURLY" in cmd("crontab","-l",check=False,capture=True).stdout: raise RuntimeError("STOP: antiguo Cron activo")
    if os.statvfs(BASE).f_bavail*os.statvfs(BASE).f_frsize<120*1024**3: raise RuntimeError("STOP: reserva 120 GiB")
    for i in SPECS:
        if not tor_ok("ltd-c-tor-inst"+str(i)): raise RuntimeError("STOP: Tor inst"+str(i)+" no OK")
def main():
    a=argparse.ArgumentParser()
    a.add_argument("action",choices=["setup","smoke","full","status"])
    x=a.parse_args()
    if x.action=="setup": setup()
    elif x.action=="smoke": sys.exit(run(2))
    elif x.action=="full": sys.exit(run(20))
    else:
        for i in SPECS:
            n="ltd-c-tor-inst"+str(i);print(n,"RUNNING" if exist(n) else "MISSING","IsTor",tor_ok(n) if exist(n) else False)
        print("FREE_GIB",round(os.statvfs(BASE).f_bavail*os.statvfs(BASE).f_frsize/1024**3,1))
if __name__=="__main__":
    try:main()
    except Exception as e: print("ERROR:",e,file=sys.stderr);sys.exit(2)
