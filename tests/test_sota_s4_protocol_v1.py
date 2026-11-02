"""Offline, no data / torch tests for ex-ante SOTA S4 protocol."""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import unittest

from src.models.sota.v1.s4_protocol import (
    EXPECTED_BOUNDS,EXPECTED_COUNTS,examine_s1_index,
    score_macro_f1,select_epoch,validate_config
)

CONFIG=Path(__file__).resolve().parents[1]/"configs/sota/sota_s4_frozen_protocol_v1.json"

def config():
    return json.loads(CONFIG.read_text(encoding="utf-8"))

class S4ProtocolTests(unittest.TestCase):
    def test_frozen_ex_ante_config_passes(self):
        validate_config(config())

    def test_external_holdout_leakage_rejected(self):
        c=config()
        c["external_holdout"]["FUTURE_B"]="ALLOWED_FOR_SELECTION"
        with self.assertRaisesRegex(ValueError,"external test leakage"):
            validate_config(c)

    def test_temporal_overlap_rejected(self):
        c=config()
        c["schedule"][1]["validate"]=["DEV_MIDDLE","DEV_LATE"]
        with self.assertRaises(ValueError):
            validate_config(c)

    def test_wflib_hyperparameters_fixed(self):
        c=config()
        c["model_configs"]["VarCNN"]["learning_rate"]=0.01
        with self.assertRaisesRegex(ValueError,"WFlib hyperparameters"):
            validate_config(c)

    def test_exact_date_windows_immutable(self):
        c=config()
        c["population"]["dates"]["DEV_EARLY"]=["2025-11-19","2025-12-02"]
        with self.assertRaisesRegex(ValueError,"exact chronological bounds"):
            validate_config(c)

    def test_gpu_budget_and_storage_policy_locked(self):
        c=config()
        c["resource_controls"]["max_parallel_gpu_training_jobs"]=2
        with self.assertRaises(ValueError):validate_config(c)
        c=config()
        c["resource_controls"]["output_root"]="/home/bsierra/ltd-attack/results"
        with self.assertRaises(ValueError):validate_config(c)

    def test_primary_seed_fixed(self):
        c=config()
        c["seeds"]["primary"]=[11,42,73]
        with self.assertRaises(ValueError):validate_config(c)

    def test_macro_f1_65_includes_absent_classes(self):
        # Perfect predictions on two classes still yield 2/65 Macro-F1.
        actual=score_macro_f1([0,1,1],[0,1,1],65)
        self.assertAlmostEqual(actual,2/65)

    def test_macro_f1_full_denominator(self):
        actual=score_macro_f1([0,1,2],[0,0,0],65)
        self.assertAlmostEqual(actual,(2/(1+3))/65)

    def test_validation_epoch_tie_breaks_earliest(self):
        self.assertEqual(select_epoch({2:0.4,1:0.4,3:0.2}),1)
        self.assertEqual(select_epoch({1:0.3,2:0.6,3:0.6}),2)

    def test_invalid_macro_f1_rejected(self):
        with self.assertRaises(ValueError):
            score_macro_f1([65],[0])
        with self.assertRaises(ValueError):
            select_epoch({1:float("nan")})
        with self.assertRaises(ValueError):
            select_epoch({31:0.5})

    def test_examine_s1_metadata_all_dates_without_payload(self):
        from datetime import date,timedelta
        by_uid={}
        for split,n in EXPECTED_COUNTS.items():
            lo,hi=EXPECTED_BOUNDS[split]
            first=date.fromisoformat(lo)
            last=date.fromisoformat(hi)
            ndays=(last-first).days+1
            for i in range(n):
                day=first+timedelta(days=i%ndays)
                by_uid[f"{split}:{i}"]=(f"site_{i%65}",day.isoformat(),split)
        x=examine_s1_index(by_uid)
        self.assertEqual(x["unique_uids"],57916)
        self.assertEqual(x["split_counts"],EXPECTED_COUNTS)
        self.assertTrue(all(c==65 for c in x["classes_per_split"].values()))
        self.assertEqual(x["days_per_split"],{"DEV_EARLY":14,"DEV_MIDDLE":14,"DEV_LATE":13})

if __name__=="__main__":
    unittest.main()
