"""Offline contracts for S3 pinned source and Dataset C 3x20 summaries."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"scripts"/"dataset_c"))
sys.path.insert(0,str(ROOT/"scripts"/"sota"))

from src.models.sota.v1.upstream_models import (
    WFLIB_BLOBS,WFLIB_COMMIT,audit_signatures,get_model
)
from scripts.dataset_c.audit_halfsplit_operations_v1 import parse_batch,parse_start
from scripts.sota.s3_time_channel_audit_v1 import parse_numeric

BATCHES={
    "first":{"inst1":"belleza","inst2":"bancos","inst3":"deportes"},
    "second":{"inst1":"turismo","inst2":"salud","inst3":"politica"},
}

def batch_records(batch):
    return [{"instance":inst,"category":cat,"slot":slot,
             "status":"OK" if slot<=15 else "BLOCKED",
             "qa_valid":slot<=15}
            for inst,cat in BATCHES[batch].items() for slot in range(1,21)]

def make_report(path,batch,rows=None):
    rows=rows if rows is not None else batch_records(batch)
    path.write_text(json.dumps({
        "classification":"C_DIRECT_120_HALFHOUR_QA_ONLY",
        "batch":batch,"attempts":60,
        "records":rows,"valid_pcap_qa":sum(bool(x["qa_valid"]) for x in rows),
        "run_id":"synthetic_test_only"
    }))

class PinnedSotaTests(unittest.TestCase):
    def test_git_commit_and_blobs_are_explicit(self):
        self.assertEqual(len(WFLIB_COMMIT),40)
        self.assertEqual(set(WFLIB_BLOBS),{"DF","VarCNN"})
        self.assertTrue(all(len(x)==40 for x in WFLIB_BLOBS.values()))

    def test_df_and_varcnn_tensor_contract(self):
        plan=audit_signatures()
        self.assertEqual(plan["model_inputs"]["DF"]["shape"],["N",1,5000])
        self.assertEqual(plan["model_inputs"]["VarCNN"]["shape"],["N",2,5000])
        self.assertEqual(plan["output"]["shape"],["N",65])
        self.assertEqual(plan["varcnn_dt2_semantic_gate"],
                         "PENDING_TIME_VECTOR_AUDIT")

    def test_unsupported_model_rejected_without_imports(self):
        with self.assertRaises(ValueError):
            get_model("UNKNOWN",65)

    def test_65_classes_required_without_imports(self):
        with self.assertRaises(ValueError):
            get_model("DF",64)

    def test_time_vector_parse_accepts_numbers(self):
        self.assertEqual(parse_numeric("time","[0.0,0.25,0.5]"),
                         [0.0,0.25,0.5])

    def test_time_vector_parse_rejects_nonfinite(self):
        for raw in ('[1,float("inf")]', '[1, None]', '[1e999]','[]'):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    parse_numeric("time",raw)

class DatasetCOpsTests(unittest.TestCase):
    def test_real_run_id_minutes_parsed(self):
        original=parse_start("direct120_half_first_20261102T120108Z_451","first")
        alternate=parse_start("direct120_half_second_20261102T123006Z_452","second")
        self.assertIsNotNone(original)
        self.assertEqual(original.strftime("%Y%m%dT%H"),"20261102T12")
        self.assertEqual(alternate.strftime("%Y%m%dT%H"),"20261102T12")

    def test_invalid_or_cross_batch_run_id_rejected(self):
        self.assertIsNone(parse_start("direct120_half_second_20261102T123006Z_452",
                                      "first"))
        self.assertIsNone(parse_start("synthetic_test_only","second"))

    def test_original_3x20_per_batch_passes(self):
        with TemporaryDirectory() as d:
            p=Path(d)/"report.json"
            make_report(p,"first")
            r=parse_batch(p,"first")
            self.assertEqual(r["slots"],60)
            self.assertEqual(r["valid"],45)
            self.assertEqual(len(r["slot_ids"]),60)

    def test_second_3x20_per_batch_passes(self):
        with TemporaryDirectory() as d:
            p=Path(d)/"report.json"
            make_report(p,"second")
            r=parse_batch(p,"second")
            self.assertEqual(r["slots"],60)
            self.assertEqual(r["valid"],45)

    def test_duplicate_slot_is_rejected(self):
        with TemporaryDirectory() as d:
            p=Path(d)/"report.json"
            rows=batch_records("first")
            rows[1]=dict(rows[0])
            make_report(p,"first",rows)
            with self.assertRaisesRegex(ValueError,"duplicate"):
                parse_batch(p,"first")

    def test_wrong_category_is_rejected(self):
        with TemporaryDirectory() as d:
            p=Path(d)/"report.json"
            rows=batch_records("first")
            rows[0]["category"]="turismo"
            make_report(p,"first",rows)
            with self.assertRaisesRegex(ValueError,"category/instance"):
                parse_batch(p,"first")

    def test_wrong_valid_count_is_rejected(self):
        with TemporaryDirectory() as d:
            p=Path(d)/"report.json"
            make_report(p,"first")
            obj=json.loads(p.read_text())
            obj["valid_pcap_qa"]=46
            p.write_text(json.dumps(obj))
            with self.assertRaisesRegex(ValueError,"PCAP QA total"):
                parse_batch(p,"first")

if __name__=="__main__":
    unittest.main()
