#!/usr/bin/env python3
"""Append-only QA reconciliation; never promote, merge, delete or rewrite captures."""
import collections
import json
import pathlib
import sys

if len(sys.argv) != 4:
    raise SystemExit("Usage: reconcile_pinned_qa_attempts_v1.py baseline.json retry5.json output.json")
base_path, retry_path, out_path = map(pathlib.Path, sys.argv[1:])
base = json.loads(base_path.read_text())
retry = json.loads(retry_path.read_text())
if base.get("attempts") != 12 or retry.get("attempts") != 5:
    raise SystemExit("RECONCILE_BLOCKED: unexpected 12+5 attempt counts")


def index(data):
    rows = data.get("visits")
    if not isinstance(rows, list):
        raise SystemExit("RECONCILE_BLOCKED: invalid visits")
    indexed = {}
    for r in rows:
        k = (r["instance"], r["category"], str(r["slot"]))
        if k in indexed:
            raise SystemExit("RECONCILE_BLOCKED: duplicate slot in same run")
        indexed[k] = r
    if len(indexed) != data["attempts"]:
        raise SystemExit("RECONCILE_BLOCKED: incomplete manifest")
    return indexed


b, t = index(base), index(retry)
failed = {k for k, r in b.items() if not r.get("qa_valid")}
if len(b) != 12 or len(failed) != 5 or set(t) != failed:
    raise SystemExit("RECONCILE_BLOCKED: retries not identical to five baseline failures")
results = []
for k, r in sorted(b.items()):
    n = t.get(k)
    if n and r.get("url") != n.get("url"):
        raise SystemExit("RECONCILE_BLOCKED: retry URL changed")
    success = bool(r.get("qa_valid")) or bool(n and n.get("qa_valid"))
    statuses = [r.get("status", "UNKNOWN")]
    if n:
        statuses.append(n.get("status", "UNKNOWN"))
    results.append({
        "instance": k[0], "category": k[1], "slot": k[2], "url": r.get("url"),
        "baseline_status": statuses[0],
        "retry_status": statuses[1] if len(statuses) == 2 else None,
        "any_qa_valid": success,
        "unavailable_after_retry": bool(n) and not success,
    })
total = sum(r["any_qa_valid"] for r in results)
persistent = [r for r in results if r["unavailable_after_retry"]]
status_counts = dict(collections.Counter((r["retry_status"] or r["baseline_status"]) for r in results))
output = {
    "classification": "QA_ONLY_NOT_SCIENTIFIC",
    "source_baseline": str(base_path),
    "source_retry": str(retry_path),
    "distinct_slots": 12,
    "baseline_valid": sum(bool(r["qa_valid"]) for r in b.values()),
    "retry_valid": sum(bool(r["qa_valid"]) for r in t.values()),
    "combined_valid": total,
    "remaining_access_constrained": len(persistent),
    "status_counts_after_retry": status_counts,
    "scientific_capture_authorized": False,
    "requires_ab_method_comparability": True,
    "requires_trusted_clock_gate": True,
    "sites": results,
}
out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
print("QA_COMBINED_VALID", str(total) + "/12")
print("QA_PERSISTENT_ACCESS_RESTRICTIONS", len(persistent))
for rec in persistent:
    print("ACCESS_REVIEW_REQUIRED", rec["instance"], rec["category"], "slot=" + rec["slot"],
          rec["url"], "baseline=" + rec["baseline_status"], "retry=" + str(rec["retry_status"]))
print("QA_READINESS_REPORT", out_path)
print("SCIENTIFIC_CAPTURE=NOT_AUTHORIZED")
