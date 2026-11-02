"""S3 DT2 equivalence tests; stdlib only, no torch, no A/B/C data."""
from __future__ import annotations
from array import array
import math
import unittest

from src.models.sota.v1.timing_adapter import (
    dt2_from_sequences, dt2_from_text, parse_times,
    reference_signed_timestamps
)

class VarCNNDT2Contract(unittest.TestCase):
    def test_basic_gaps_and_directions(self):
        v=dt2_from_text("[1,-1,1,-1]","[0.0,0.1,0.1,0.8]",6)
        self.assertEqual(list(v.direction),[1.,-1.,1.,-1.,0.,0.])
        self.assertAlmostEqual(v.interarrival[0],0.1,places=6)
        self.assertEqual(v.interarrival[1],0.)
        self.assertAlmostEqual(v.interarrival[2],0.7,places=6)
        self.assertEqual(list(v.interarrival)[3:],[0.,0.,0.])
        self.assertEqual(v.original_packets,4)

    def test_first_direction_not_erased_by_zero_time(self):
        v=dt2_from_text("[-1,1]","[0.0,0.5]",4)
        self.assertEqual(v.direction[0],-1.)
        self.assertAlmostEqual(v.interarrival[0],0.5,places=6)

    def test_signed_offset_equivalence_to_wflib_dt2(self):
        d=array("b",[-1,1,-1,1,1])
        t=[0.,0.01,0.21,0.21,0.75]
        shifted=reference_signed_timestamps(d,t)
        self.assertEqual([1 if x>0 else -1 for x in shifted],list(d))
        gaps=[max(0,abs(shifted[i+1])-abs(shifted[i]))
              for i in range(len(shifted)-1)]
        channels=dt2_from_sequences(d,t,8)
        for i,ref in enumerate(gaps):
            self.assertAlmostEqual(channels.interarrival[i],ref,places=6)

    def test_truncation_needs_packet_5001_time(self):
        d=array("b",[1]*5001)
        t=[float(i)*0.01 for i in range(5001)]
        v=dt2_from_sequences(d,t,5000)
        self.assertEqual(len(v.interarrival),5000)
        self.assertAlmostEqual(v.interarrival[4999],0.01,places=5)
        self.assertEqual(v.gaps_considered,5000)
        self.assertEqual(v.dropped_packets,1)

    def test_exactly_5000_packets_last_gap_padded(self):
        d=array("b",[1]*5000)
        t=[float(i)*0.01 for i in range(5000)]
        v=dt2_from_sequences(d,t,5000)
        self.assertEqual(v.gaps_considered,4999)
        self.assertEqual(v.interarrival[-1],0.)

    def test_less_than_5000_right_padding(self):
        d=array("b",[1,-1,1])
        t=[0.0,0.1,0.5]
        v=dt2_from_sequences(d,t,5)
        self.assertEqual(list(v.direction),[1.,-1.,1.,0.,0.])
        self.assertEqual(v.interarrival[2],0.)
        self.assertEqual(v.interarrival[3],0.)
        self.assertEqual(v.interarrival[4],0.)

    def test_mismatched_lengths_rejected(self):
        with self.assertRaisesRegex(ValueError,"lengths"):
            dt2_from_sequences(array("b",[1,-1]),[0.])

    def test_invalid_packet_direction_rejected(self):
        with self.assertRaisesRegex(ValueError,"packet direction"):
            dt2_from_sequences(array("b",[1,0]),[0.,0.1])

    def test_time_monotonicity_rejected(self):
        for raw in ("[0.0,0.2,0.1]","[0.,float('nan')]"):
            with self.subTest(raw=raw),self.assertRaises(ValueError):
                dt2_from_text("[1,-1,1]",raw)

    def test_timestamp_needs_zero_start(self):
        with self.assertRaisesRegex(ValueError,"relative"):
            dt2_from_text("[1,-1]","[100.0,100.5]")

    def test_time_parser_rejects_nonnumeric(self):
        for raw in ("[True,0.2]","[0.,1e309]","[]","[0, None]","0,1"):
            with self.subTest(raw=raw),self.assertRaises(ValueError):
                parse_times(raw)

    def test_no_timing_inference_from_size_or_direction(self):
        with self.assertRaises(ValueError):
            dt2_from_text("[1,-1,1]","",5000)

if __name__=="__main__":
    unittest.main()
