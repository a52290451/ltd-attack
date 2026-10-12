#!/usr/bin/env python3
"""Create a private, minimal source-review ZIP for P16C browser collector.
Does not bundle bridge configuration, Tor state, PCAP, datasets or cookies.
Do not upload publicly before reviewing the included scripts.
"""
import hashlib
import os
from pathlib import Path
import re
import sys
from zipfile import ZipFile,ZIP_DEFLATED

APP=Path.home()/"ltd-collector-c-smoke-v22/P16C_RECOLECTOR_V2"
OUT=Path.home()/"ltd-ops/collector-source-review-v24.zip"
FILES=["capture_dom.js","run_category.sh","quality.js","quality.test.js","check_sites.py"]
def main():
    if not APP.is_dir():raise RuntimeError("STOP: collector missing")
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.parent.chmod(0o700)
    sources=[]
    for name in FILES:
        p=APP/name
        if not p.is_file() or p.is_symlink():raise RuntimeError("STOP: missing or symlink: "+name)
        data=p.read_bytes()
        if not 100<len(data)<500000:raise RuntimeError("STOP: unusual collector file size: "+name)
        # Avoid accidental packaging of secrets/certificates; user must inspect before sharing.
        if re.search(rb"-----BEGIN (?:RSA |EC |OPENSSH |)PRIVATE KEY-----",data):
            raise RuntimeError("STOP: private key pattern found in "+name)
        if re.search(rb"(?im)^[\s#]*(?:Bridge\s+obfs4|Cookie\s*:|Authorization\s*:)",data):
            raise RuntimeError("STOP: potentially sensitive source detected: "+name)
        sources.append((name,data))
    temp=OUT.with_suffix(".tmp")
    with ZipFile(temp,"w",compression=ZIP_DEFLATED) as z:
        for name,data in sources:
            z.writestr(name,data)
        z.writestr("README.txt","Source-only review of P16C collector. No datasets or Tor configuration. Review contents before sharing.\n")
    os.chmod(temp,0o600)
    temp.replace(OUT)
    print("SOURCE_REVIEW_ZIP",OUT)
    for name,data in sources:print("SOURCE",name,"bytes",len(data),"sha256",hashlib.sha256(data).hexdigest()[:16])
    print("Attach this ZIP in ChatGPT for an accurate collector V2.4 patch. Do not publish publicly.")
if __name__=="__main__":
    try:main()
    except Exception as e:print("ERROR",e,file=sys.stderr);sys.exit(2)
