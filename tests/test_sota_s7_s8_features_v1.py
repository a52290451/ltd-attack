"""Mathematical feature contracts; no training, future files or Dataset C."""
import unittest
from array import array
import numpy as np
from scripts.sota.s7_cumul_frozen_v1 import cumul104
from src.models.sota.v1.tiktok_adapter import tiktok_dt_from_sequences

class SotaFeatureTests(unittest.TestCase):
    def test_cumul_bytes_signed_order(self):
        # +1 is outgoing, CUMUL signed p<0 for outgoing;
        # -1 is incoming and contributes positive cumulative bytes.
        x=cumul104("[1,-1,1]","[100,200,300]")
        self.assertEqual(x.shape,(104,))
        self.assertTrue(np.allclose(x[:4],[1,2,200,400]))
        self.assertAlmostEqual(float(x[4]),0.)
        self.assertAlmostEqual(float(x[-1]),-200.)
        self.assertTrue(np.isfinite(x).all())

    def test_cumul_rejects_size_direction_disalignment(self):
        with self.assertRaises(RuntimeError):
            cumul104("[1,-1]","[100]")

    def test_cumul_rejects_zero_byte_packet(self):
        with self.assertRaises(RuntimeError):
            cumul104("[1,-1]","[100,0]")

    def test_dt_not_dt2(self):
        d=array("b",[1,-1,1])
        t=[0.,.2,.6]
        result=tiktok_dt_from_sequences(d,t,5)
        self.assertEqual(list(result[:3]),[0.,-float(result[1]*-1),float(result[2])])
        self.assertAlmostEqual(result[1],-.2,places=6)
        self.assertAlmostEqual(result[2],.6,places=6)
        self.assertEqual(list(result[3:]),[0.,0.])

    def test_dt_not_adding_timestamp_offset(self):
        x=tiktok_dt_from_sequences(array("b",[-1,1]),[0.,2.],2)
        self.assertEqual(list(x),[0.,2.])

    def test_dt_strict_alignment(self):
        with self.assertRaises(ValueError):
            tiktok_dt_from_sequences(array("b",[1,-1]),[0.],5000)

if __name__=="__main__":
    unittest.main()
