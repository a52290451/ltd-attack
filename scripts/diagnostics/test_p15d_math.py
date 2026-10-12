#!/usr/bin/env python3
"""Small deterministic mathematical regression checks; no scientific data accessed."""
import unittest
import numpy as np
from sklearn.metrics import f1_score
from run_p15d_and_milestone import batch_f1, bootstrap_paired, NCLASS

class P15DMath(unittest.TestCase):
    def test_confusion_matrix_macro_f1(self):
        y=np.array([0,1,2,1,0,2],dtype=int)
        pred=np.array([0,1,1,2,0,2],dtype=int)
        cm=np.bincount(NCLASS*y+pred,minlength=NCLASS*NCLASS).reshape(NCLASS,NCLASS)
        actual=batch_f1(cm[None,:,:])[0]
        expected=f1_score(y,pred,labels=np.arange(NCLASS),average="macro",zero_division=0)
        self.assertAlmostEqual(actual,expected,places=12)

    def test_identical_predictions_zero_bootstrap_delta(self):
        y=np.array([0,1,2,3,0,1,2,3],dtype=int)
        dates=np.array(["2026-01-01"]*4+["2026-01-02"]*4,dtype="datetime64[D]")
        scores=np.full((8,NCLASS),1e-6)
        scores[np.arange(8),y]=0.99
        result=bootstrap_paired(y,dates,scores,scores,40,42)
        self.assertEqual(result["day_blocks"],2)
        for name in ("accuracy","macro_f1","top5_accuracy"):
            self.assertAlmostEqual(result[name+"_ci_low"],0.0,places=12)
            self.assertAlmostEqual(result[name+"_ci_high"],0.0,places=12)
            self.assertEqual(result[name+"_fraction_positive"],0.0)

if __name__=="__main__":
    unittest.main()
