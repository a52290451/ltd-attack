#!/usr/bin/env python3
"""Fast offline SOTA battery integration-contract checks: NO CUDA, NO datasets.

Compile original-source Python adapters, verify shell parsing, test orchestration
stage mappings, original-data safety flags, all-four-metrics math and synthetic
date-level pairing. These checks do not substitute for CPU mini-optimization,
real data preparation, or training on the target server.
"""
from __future__ import annotations
import ast,subprocess,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.sota import original_sota_orchestrator_v2 as o
from scripts.sota import original_sota_paired_date_stats_v1 as st
import numpy as np

SOTA=ROOT/"scripts/sota"
PYFILES=[
 "sota_paths_v1.py","sota_zeus_evidence_bootstrap_v1.py",
 "original_sota_orchestrator_v2.py","original_sota_paired_date_stats_v1.py",
 "s9_netclr_authors_cpu_gate_v1.py","s9_netclr_authors_pretrain_b_v1.py",
 "s9_netclr_authors_supervised_b_v1.py",
 "s9_netclr_authors_frozen_c_eval_v1.py",
 "s9_netclr_authors_results_gate_v1.py",
 "s3_s4_s8_authors_keras_v1.py",
 "s11_swallow_authors_v1.py","s12_wfcat_authors_v1.py",
 "original_authors_results_integrity_v1.py",
]
SHELLFILES=[
 "run_original_sota_battery_v1.sh",
 "run_s9_netclr_authors_v1.sh","run_s11_swallow_authors_v1.sh",
 "run_s12_wfcat_authors_v1.sh","run_authors_keras_s3_s4_s8_v1.sh",
 "setup_isolated_authors_keras_v1.sh",
 "report_perseo_gpu_readiness_v1.sh","smoke_perseo_gpu_runtime_v1.sh",
]
class ContractTests(unittest.TestCase):
    def test_all_scripts_compile(self):
        for filename in PYFILES:
            with self.subTest(file=filename):
                source=(SOTA/filename).read_text()
                compile(source,str(SOTA/filename),"exec")
    def test_shell_syntax(self):
        for filename in SHELLFILES:
            with self.subTest(file=filename):
                p=subprocess.run(["bash","-n",str(SOTA/filename)],
                                 capture_output=True,text=True)
                self.assertEqual(p.returncode,0,p.stderr)
    def test_stages_have_immutable_manifest_contract(self):
        self.assertEqual(len(o.MODELS),7)
        self.assertEqual(len(set(o.MODELS)),7)
        for model in o.MODELS:
            self.assertIn(model,o.ROOTS)
            self.assertIn(model,o.DIRS)
            self.assertIn(model,o.EVAL_TYPES)
            self.assertIn(model,o.RUNNERS)
            self.assertTrue(all(x in ("09B","Future-B")
                for x,d,k,c in o.stages(model,"evaluate")))
            kinds=[a for a,d,k in o.DIRS[model]]
            self.assertIn("C",kinds)
            self.assertEqual(len(kinds),3 if model in ("NetCLR","Swallow") else 2)
    def test_cli_stage_coverage(self):
        sources={
          "NetCLR":{
             "pretrain-B":"s9_netclr_authors_pretrain_b_v1.py",
             "supervised-B":"s9_netclr_authors_supervised_b_v1.py",
             "C":"s9_netclr_authors_frozen_c_eval_v1.py",
             "09B":"s9_netclr_authors_frozen_c_eval_v1.py",
             "Future-B":"s9_netclr_authors_frozen_c_eval_v1.py"},
          "Swallow":{k:"s11_swallow_authors_v1.py"
                     for k in ("pretrain-B","supervised-B","C","09B","Future-B")},
          "WFCAT":{k:"s12_wfcat_authors_v1.py"
                   for k in ("B","C","09B","Future-B")}}
        for model,parts in sources.items():
            for stage,filename in parts.items():
                with self.subTest(model=model,stage=stage):
                    body=(SOTA/filename).read_text()
                    self.assertIn('if __name__=="__main__"',body)
                    self.assertIn("argparse",body)
        self.assertTrue((SOTA/"s3_s4_s8_authors_keras_v1.py").is_file())
    def test_metrics_perfect_classifier(self):
        y=np.arange(65,dtype=np.int64)
        p=np.eye(65,dtype=np.float32)
        x=st.profile(p,y)
        r=st.stats(x,y,np.arange(65))
        np.testing.assert_allclose(r,[1,1,1,1],rtol=0,atol=1e-10)
    def test_metrics_single_wrong_classifier(self):
        y=np.arange(65,dtype=np.int64)
        p=np.eye(65,dtype=np.float32)
        p[0,:]=np.arange(65,dtype=np.float32)
        p[0,:]/=p[0,:].sum()
        x=st.profile(p,y)
        r=st.stats(x,y,np.arange(65))
        self.assertLess(r[0],1)
        self.assertLess(r[1],1)
        self.assertLess(r[2],1)
        self.assertLess(r[3],1)
    def test_legacy_data_provenance_is_not_scored_as_original(self):
        s=(SOTA/"s3_s4_s8_authors_keras_v1.py").read_text()
        self.assertIn("future_b_opened",s)
        self.assertIn("no_external_tuning",s)
        self.assertIn("authors_VarCNN_direction_branch_only",s)
        self.assertIn('source_repository',s)
    def test_gpu_all_delegates_to_resumable_orchestrator(self):
        source=(SOTA/"run_original_sota_battery_v1.sh").read_text()
        self.assertIn('original_sota_orchestrator_v2.py',source)
        self.assertIn('SOTA_HOST', (SOTA/"sota_zeus_evidence_bootstrap_v1.py").read_text())
if __name__=="__main__":
    unittest.main(verbosity=2)
