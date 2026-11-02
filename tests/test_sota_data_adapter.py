"""Offline SOTA input adapter contract tests; Python stdlib only, no A/B/C."""
from __future__ import annotations

from array import array
import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.models.sota.v1.data_adapter import (
    DEFAULT_LENGTH,
    digest,
    fixed_length,
    iter_dev_vectors,
    parse_direction,
)

class DirectionAdapterTests(unittest.TestCase):
    def test_original_df_length(self):
        self.assertEqual(DEFAULT_LENGTH,5000)

    def test_signed_direction_exact(self):
        parsed=parse_direction("[1, -1, 0, +1, -1.0, 1.0]")
        self.assertEqual(parsed,array("b",[1,-1,0,1,-1,1]))

    def test_padding_is_right_only(self):
        self.assertEqual(fixed_length(array("b",[1,-1,1]),7),
                         array("b",[1,-1,1,0,0,0,0]))

    def test_truncation_is_first_n_packets(self):
        self.assertEqual(fixed_length(array("b",[1,-1,0,1,-1]),3),
                         array("b",[1,-1,0]))

    def test_full_length_is_unchanged(self):
        original=array("b",[1,-1]*2500)
        self.assertEqual(fixed_length(original,5000),original)
        self.assertEqual(len(original),5000)

    def test_input_not_modified(self):
        original=array("b",[1,-1,1])
        self.assertEqual(len(fixed_length(original,5000)),5000)
        self.assertEqual(original,array("b",[1,-1,1]))

    def test_rejects_invalid_content(self):
        for raw in ('', '1,-1', '[]', '[True,1]', '[2,1]', '[1,float("nan")]',
                    '["-1",1]', '[1,None]', '[1/0]', '__import__("os")',
                    '[1,float("inf")]'):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    parse_direction(raw)

    def test_rejects_invalid_length(self):
        for length in (0,-1,1.5,True):
            with self.subTest(length=length):
                with self.assertRaises(ValueError):
                    fixed_length(array("b",[1,-1]),length)

    def test_stream_dev_metadata_aligns_by_uid(self):
        with TemporaryDirectory() as folder:
            source=Path(folder)/"vectors.csv"
            with source.open("w",newline="",encoding="utf-8") as f:
                writer=csv.DictWriter(f,fieldnames=[
                    "pcap_uid","site_label","direction_vector","size_vector"
                ])
                writer.writeheader()
                writer.writerows([
                    {"pcap_uid":"B","site_label":"class-2",
                     "direction_vector":"[-1,1,-1]","size_vector":"[20,30,40]"},
                    {"pcap_uid":"INTERNAL","site_label":"other",
                     "direction_vector":"[0]","size_vector":"[1]"},
                    {"pcap_uid":"A","site_label":"class-1",
                     "direction_vector":"[1,-1]","size_vector":"[4,5]"},
                ])
            cohort={
                "A":("class-1","2025-11-18","DEV_EARLY"),
                "B":("class-2","2025-12-02","DEV_MIDDLE")
            }
            labels={"class-1":0,"class-2":1}
            got=list(iter_dev_vectors(source,cohort,labels,length=5))
            self.assertEqual([x.pcap_uid for x in got],["B","A"])
            self.assertEqual([x.label_id for x in got],[1,0])
            self.assertEqual([x.temporal_split for x in got],
                             ["DEV_MIDDLE","DEV_EARLY"])
            self.assertEqual(got[0].directions,array("b",[-1,1,-1,0,0]))
            self.assertEqual(got[0].original_length,3)

    def test_stream_duplicate_uid_rejected(self):
        with TemporaryDirectory() as folder:
            source=Path(folder)/"data.csv"
            with source.open("w",newline="",encoding="utf-8") as f:
                writer=csv.DictWriter(f,fieldnames=[
                    "pcap_uid","site_label","direction_vector"
                ])
                writer.writeheader()
                for _ in range(2):
                    writer.writerow({"pcap_uid":"A","site_label":"C",
                                     "direction_vector":"[1,-1]"})
            cohort={"A":("C","2025-11-18","DEV_EARLY")}
            gen=iter_dev_vectors(source,cohort,{"C":0},length=3)
            self.assertEqual(next(gen).pcap_uid,"A")
            with self.assertRaisesRegex(ValueError,"duplicate DEV"):
                list(gen)

    def test_stream_wrong_label_rejected(self):
        with TemporaryDirectory() as folder:
            source=Path(folder)/"data.csv"
            with source.open("w",newline="",encoding="utf-8") as f:
                writer=csv.DictWriter(f,fieldnames=[
                    "pcap_uid","site_label","direction_vector"
                ])
                writer.writeheader()
                writer.writerow({"pcap_uid":"A","site_label":"WRONG",
                                 "direction_vector":"[1]"})
            with self.assertRaisesRegex(ValueError,"label differs"):
                list(iter_dev_vectors(
                    source,{"A":("GOOD","2025-11-18","DEV_EARLY")},
                    {"GOOD":0},length=5))

    def test_future_split_not_accepted(self):
        with TemporaryDirectory() as folder:
            path=Path(folder)/"unused.csv"
            path.write_text("pcap_uid,site_label,direction_vector\n")
            with self.assertRaisesRegex(ValueError,"DEV-only"):
                list(iter_dev_vectors(path,{},{"X":0},required_splits=("FUTURE_B",)))

if __name__=="__main__":
    unittest.main()
