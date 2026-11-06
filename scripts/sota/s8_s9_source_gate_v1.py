#!/usr/bin/env python3
"""Audit Tik-Tok/NetCLR PyTorch source availability in existing WFlib checkout.

Read-only: no GPU, downloads, external datasets, model training, output writes
or Dataset C. Print exact SHA and paths to pin before another scientific run.
"""
from __future__ import annotations
import subprocess
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.models.sota.v1.upstream_models import (
    resolve_checkout,verify_checkout,WFLIB_COMMIT
)

def git(cwd,*args):
    p=subprocess.run(["git","-C",str(cwd),*args],
                     text=True,capture_output=True,timeout=30)
    if p.returncode:raise RuntimeError(p.stderr[-350:])
    return p.stdout.strip()

def main():
    checkout=resolve_checkout()
    evidence=verify_checkout(checkout)
    assert evidence["commit"]==WFLIB_COMMIT
    tree=git(checkout,"ls-tree","-r","--name-only","HEAD").splitlines()
    print("S8_S9_WFLIB_COMMIT",WFLIB_COMMIT,flush=True)
    for name,patterns in (
        ("TikTok",("tiktok","tik_tok","tik-tok")),
        ("NetCLR",("netclr","netaugment","contrastive","clr_"))):
        paths=[x for x in tree if any(p in x.lower() for p in patterns)]
        print("S8_S9_SOURCE",name,"MATCHES",len(paths),flush=True)
        for rel in paths[:60]:
            blob=git(checkout,"rev-parse","HEAD:"+rel)
            print("S8_S9_FILE",name,blob,rel,flush=True)
        if not paths:
            print("S8_S9_MISSING",name,"source not in pinned checkout")
    for name in ("sklearn","torch"):
        try:
            m=__import__(name)
            print("S8_S9_PACKAGE",name,getattr(m,"__version__","unknown"))
        except ImportError as e:
            print("S8_S9_PACKAGE_MISSING",name,str(e))
    print("S8_S9_SOURCE_GATE=PASS_INSPECTION_ONLY",flush=True)

if __name__=="__main__":
    try:main()
    except Exception as e:
        print("S8_S9_SOURCE_GATE_ERROR",type(e).__name__,str(e)[:550],
              file=sys.stderr,flush=True)
        sys.exit(4)
