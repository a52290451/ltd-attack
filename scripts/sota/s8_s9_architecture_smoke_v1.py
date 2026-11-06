#!/usr/bin/env python3
"""Read-only, CPU-only WFlib source and architecture smoke for S8/S9.

No dataset access, model optimization, checkpoint writes or GPU use.
The upstream files must match the frozen git commit AND exact blob IDs.
"""
from pathlib import Path
import importlib.util
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.sota.v1.upstream_models import resolve_checkout, verify_checkout, WFLIB_COMMIT

SOURCES = {
    "TikTok": ("WFlib/models/TikTok.py", "8736ef4d7d4c00bb1559def532f511cb8e142625"),
    "NetCLR": ("WFlib/models/NetCLR.py", "f63cdf80046220baf0544e967ce7a6077a7096e4"),
    "NetCLR_augmentor": ("WFlib/tools/netclr_augmentor.py", "e28a587c6caa31b39397354316a7d22edeb4b414"),
    "NetCLR_pretrain": ("WFlib/tools/netclr_pretrain.py", "8d728f5ce324d0f6b59aa293e831a358e312c815"),
}

def git(checkout, *args):
    p = subprocess.run(["git", "-C", str(checkout), *args],
                       capture_output=True, text=True, timeout=30, check=True)
    return p.stdout.strip()

def load_verified(path, name):
    spec = importlib.util.spec_from_file_location("ltd_sota_source_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main():
    import torch
    torch.set_num_threads(min(torch.get_num_threads(), 2))
    checkout = resolve_checkout()
    verify_checkout(checkout)
    assert git(checkout, "rev-parse", "HEAD") == WFLIB_COMMIT
    for name, (rel, expected) in SOURCES.items():
        actual = git(checkout, "hash-object", rel)
        if actual != expected:
            raise RuntimeError(f"{name}: source blob mismatch {actual} != {expected}")
        print(f"S8_S9_SOURCE_PIN_PASS {name} {actual}", flush=True)

    # All-zero input tests only architecture tensor compatibility, not timing semantics.
    with torch.no_grad():
        tiktok = load_verified(checkout / SOURCES["TikTok"][0], "tiktok").TikTok(65).eval()
        y = tiktok(torch.zeros(2, 1, 5000))
        if tuple(y.shape) != (2, 65) or not torch.isfinite(y).all():
            raise RuntimeError("TikTok logits shape/finite mismatch")
        print("S8_TIKTOK_ARCHITECTURE_PASS (2,1,5000)->(2,65)", flush=True)
        del tiktok, y

        netclr = load_verified(checkout / SOURCES["NetCLR"][0], "netclr").NetCLR(65).eval()
        z = netclr(torch.zeros(2, 1, 5000))
        if tuple(z.shape) != (2, 65) or not torch.isfinite(z).all():
            raise RuntimeError("NetCLR logits shape/finite mismatch")
        print("S9_NETCLR_ARCHITECTURE_PASS (2,1,5000)->(2,65)", flush=True)
    print("S8_S9_ARCHITECTURE_SMOKE_PASS_INSPECTION_ONLY", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("S8_S9_ARCHITECTURE_SMOKE_ERROR", type(exc).__name__, str(exc)[:600],
              file=sys.stderr, flush=True)
        sys.exit(4)
