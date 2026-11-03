"""Synthetic-only checks of paired S6 statistics; no historical data reads."""
import unittest
import numpy as np
from scripts.sota.s6_compare_frozen_sota_ltd_v1 import (
    block_sign_p,holm,value
)

class PairedS6Tests(unittest.TestCase):
    def test_macro_f1_nonadditive_uses_65_classes(self):
        sizes=np.array([2.,2.])
        sup=np.zeros((2,65))
        sup[0,0]=2
        sup[1,1]=2
        s={"tp":sup.copy(),"pc":sup.copy(),
           "acc":sizes.copy(),"top5":sizes.copy(),"rr":sizes.copy()}
        results=value(np.ones(2),sizes,sup,s)
        self.assertAlmostEqual(results[0],1.)
        self.assertAlmostEqual(results[1],2/65)
        self.assertAlmostEqual(results[2],1.)
        self.assertAlmostEqual(results[3],1.)

    def test_day_resampling_changes_macro_f1_recomputed(self):
        sizes=np.array([1.,1.])
        sup=np.zeros((2,65))
        sup[0,0]=1;sup[1,1]=1
        st={"tp":sup.copy(),"pc":sup.copy(),"acc":sizes.copy(),
            "top5":sizes.copy(),"rr":sizes.copy()}
        normal=value(np.array([1.,1.]),sizes,sup,st)[1]
        repeated=value(np.array([2.,0.]),sizes,sup,st)[1]
        self.assertAlmostEqual(normal,2/65)
        self.assertAlmostEqual(repeated,1/65)

    def test_exact_day_swap_symmetric_difference_has_p_one(self):
        sizes=np.array([2.,2.])
        sup=np.zeros((2,65))
        sup[:,0]=2.
        same={"tp":sup.copy(),"pc":sup.copy(),
              "acc":sizes.copy(),"top5":sizes.copy(),"rr":sizes.copy()}
        for metric in (0,2,3):
            p,method=block_sign_p(sizes,sup,same,same,metric,
                                  np.random.default_rng(42))
            self.assertEqual(method,"exact_day_swap")
            self.assertAlmostEqual(p,1.0)

    def test_holm8_monotonic_and_order_preserved(self):
        vals=[.01,.04,.03,.002,.3,.5,.8,1.]
        rows=[{"p_posthoc_two_sided":p} for p in vals]
        holm(rows)
        self.assertEqual(len(rows),8)
        self.assertTrue(all(0<=r["holm8_adjusted_p_posthoc"]<=1 for r in rows))
        ordered=sorted(rows,key=lambda r:r["p_posthoc_two_sided"])
        self.assertEqual([r["holm8_adjusted_p_posthoc"] for r in ordered],
                         sorted(r["holm8_adjusted_p_posthoc"] for r in ordered))

if __name__=="__main__":
    unittest.main()
