"""Offline S3 full-timing audit contract tests, pure stdlib."""
from __future__ import annotations
import unittest
from scripts.sota.s3_full_dev_timing_audit_v1 import summarize_trace

class S3FullTimingContract(unittest.TestCase):
    def test_normal_three_packet_trace(self):
        r=summarize_trace([1,-1,1],[0.0,0.1,0.3])
        self.assertEqual(r["packets"],3)
        self.assertEqual(r["negative_steps"],0)
        self.assertEqual(r["negative_steps_used"],0)

    def test_backwards_microseconds_counted_without_drop(self):
        r=summarize_trace([1,-1,1],[0.0,0.2,0.19997])
        self.assertEqual(r["negative_steps"],1)
        self.assertEqual(r["negative_steps_used"],1)
        self.assertAlmostEqual(r["max_backward_seconds"],0.00003,places=8)

    def test_out_of_window_rollback_reported(self):
        d=[1]*5003
        t=[i*0.01 for i in range(5003)]
        t[5002]=t[5001]-0.005
        r=summarize_trace(d,t)
        self.assertEqual(r["negative_steps"],1)
        self.assertEqual(r["negative_steps_used"],0)
        self.assertEqual(r["negative_steps_after_window"],1)

    def test_first_timestamp_not_zero_is_invalid(self):
        with self.assertRaises(ValueError):
            summarize_trace([1,-1],[10.,10.1])

    def test_direction_time_alignment_is_required(self):
        with self.assertRaises(ValueError):
            summarize_trace([1,-1],[0.])

    def test_nonfinite_timestamp_is_invalid(self):
        with self.assertRaises(ValueError):
            summarize_trace([1,-1],[0.,float("nan")])

    def test_nonfinite_infinity_timestamp_is_invalid(self):
        with self.assertRaisesRegex(ValueError,"non-finite"):
            summarize_trace([1,-1],[0.0,float("inf")])

    def test_nonnumeric_and_boolean_times_are_invalid(self):
        for times in ([0.0,"0.1"],[0.0,True],[0.0,None]):
            with self.subTest(times=times):
                with self.assertRaisesRegex(ValueError,"non-finite"):
                    summarize_trace([1,-1],times)

    def test_invalid_direction_is_rejected(self):
        with self.assertRaises(ValueError):
            summarize_trace([1,0],[0.,0.2])

    def test_zero_interarrival_is_not_negative(self):
        r=summarize_trace([-1,1,-1],[0.,0.,0.5])
        self.assertEqual(r["negative_steps"],0)

if __name__=="__main__":
    unittest.main()
