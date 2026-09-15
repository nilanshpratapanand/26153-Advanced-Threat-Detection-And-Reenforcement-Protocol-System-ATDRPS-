import datetime
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from atdrps.data.datasets import (
    CANONICAL_ALIASES, detect_dataset, load_flow_csv, normalise_name,
)
from atdrps.data.schema import FLOW_FEATURES, PACKET_FEATURES


def write_csv(frame: pd.DataFrame) -> str:
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "flows.csv")
    frame.to_csv(path, index=False)
    return path


CIC2018 = pd.DataFrame({
    "Dst Port": [80, 445, 53],
    "Protocol": [6, 6, 17],
    "Timestamp": ["14/02/2018 08:31:01", "14/02/2018 08:31:05", "14/02/2018 08:31:09"],
    "Flow Duration": [1_200_000, 50, 3_000],          # microseconds
    "Tot Fwd Pkts": [10, 1, 2],
    "Tot Bwd Pkts": [8, 0, 2],
    "TotLen Fwd Pkts": [1500, 60, 120],
    "TotLen Bwd Pkts": [9000, 0, 300],
    "Flow Byts/s": [8750.0, np.inf, 140000.0],        # CIC really does emit Infinity
    "Flow Pkts/s": [15.0, np.inf, 1333.0],
    "Flow IAT Mean": [70588.0, 0, 1500.0],
    "SYN Flag Cnt": [1, 1, 0],
    "RST Flag Cnt": [0, 1, 0],
    "ACK Flag Cnt": [16, 0, 0],
    "Init Fwd Win Byts": [64240, 1024, 0],
    "Label": ["Benign", "Bot", "Benign"],
})

UNSW = pd.DataFrame({
    "sport": [1, 2], "dsport": [80, 53], "proto": [6, 17],
    "dur": [1.5, 0.2],                                 # seconds
    "Spkts": [5, 2], "Dpkts": [4, 2],
    "sbytes": [500, 100], "dbytes": [4000, 200],
    "sttl": [64, 128], "dttl": [64, 64],
    "Stime": [1_518_597_061, 1_518_597_065],
    "attack_cat": ["", "Exploits"],
})


class TestNameNormalisation(unittest.TestCase):
    def test_whitespace_and_punctuation_ignored(self):
        self.assertEqual(normalise_name(" Flow IAT Mean "), "flowiatmean")
        self.assertEqual(normalise_name("flow_iat_mean"), "flowiatmean")
        self.assertEqual(normalise_name("Flow Byts/s"), "flowbytss")

    def test_every_alias_target_is_a_canonical_feature(self):
        canonical = set(FLOW_FEATURES) | set(PACKET_FEATURES)
        for name in CANONICAL_ALIASES:
            self.assertIn(name, canonical, f"{name} is not a canonical feature")


class TestDetection(unittest.TestCase):
    def test_detects_cicids2018(self):
        self.assertEqual(detect_dataset(CIC2018.columns), "cicids2018")

    def test_detects_unsw(self):
        self.assertEqual(detect_dataset(UNSW.columns), "unswnb15")

    def test_unknown_falls_back_to_generic(self):
        self.assertEqual(detect_dataset(["a", "b"]), "generic")

    def test_bad_dataset_name_rejected(self):
        with self.assertRaises(ValueError):
            load_flow_csv(write_csv(CIC2018), dataset="nope")


class TestCicLoading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ff = load_flow_csv(write_csv(CIC2018))
        cls.frame = cls.ff.frame

    def test_duration_converted_from_microseconds(self):
        self.assertAlmostEqual(self.frame.flow_duration.iloc[0], 1.2, places=9)
        self.assertAlmostEqual(self.frame.flow_duration.iloc[2], 0.003, places=9)

    def test_timestamps_are_epoch_seconds(self):
        expected = datetime.datetime(2018, 2, 14, 8, 31, 1,
                                     tzinfo=datetime.timezone.utc).timestamp()
        self.assertAlmostEqual(self.frame.start_ts.iloc[0], expected, delta=1.0)
        self.assertAlmostEqual(
            self.frame.start_ts.iloc[1] - self.frame.start_ts.iloc[0], 4.0, places=3
        )

    def test_infinity_is_scrubbed(self):
        values = self.frame[list(FLOW_FEATURES) + list(PACKET_FEATURES)].to_numpy(float)
        self.assertTrue(np.isfinite(values).all())

    def test_derived_totals(self):
        self.assertEqual(self.frame.total_packets.iloc[0], 18)
        self.assertEqual(self.frame.total_bytes.iloc[0], 10500)
        self.assertAlmostEqual(self.frame.bytes_ratio.iloc[0], 1500 / 10500, places=6)

    def test_missing_features_reported_not_faked(self):
        # a NetFlow CSV cannot supply TTL variance or retransmission counts
        self.assertIn("retransmission_count", self.ff.missing)
        self.assertIn("ttl_std", self.ff.missing)
        self.assertNotIn("flow_duration", self.ff.missing)
        self.assertLess(len(self.ff.available & set(PACKET_FEATURES)), 5)
        self.assertIn("Flow-level run", self.ff.summary())

    def test_schema_is_complete_even_when_source_is_not(self):
        for col in list(FLOW_FEATURES) + list(PACKET_FEATURES):
            self.assertIn(col, self.frame.columns)

    def test_labels_preserved(self):
        self.assertEqual(list(self.frame.label), ["Benign", "Bot", "Benign"])


class TestUnswLoading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ff = load_flow_csv(write_csv(UNSW))
        cls.frame = cls.ff.frame

    def test_duration_left_in_seconds(self):
        self.assertAlmostEqual(self.frame.flow_duration.iloc[0], 1.5, places=9)

    def test_epoch_timestamps_pass_through(self):
        self.assertEqual(self.frame.start_ts.iloc[0], 1_518_597_061.0)

    def test_empty_label_becomes_benign(self):
        self.assertEqual(list(self.frame.label), ["Benign", "Exploits"])

    def test_ttl_columns_mapped(self):
        self.assertIn("ttl_mean", self.ff.available)
        self.assertEqual(self.frame.ttl_mean.iloc[1], 128)


class TestMisc(unittest.TestCase):
    def test_nrows_limit(self):
        self.assertEqual(len(load_flow_csv(write_csv(CIC2018), nrows=2)), 2)

    def test_rows_sorted_by_time(self):
        shuffled = CIC2018.iloc[::-1].reset_index(drop=True)
        frame = load_flow_csv(write_csv(shuffled)).frame
        self.assertTrue((frame.start_ts.diff().dropna() >= 0).all())

    def test_generic_csv_without_timestamps_still_loads(self):
        frame = load_flow_csv(write_csv(pd.DataFrame({
            "Flow Duration": [1.0, 2.0], "Tot Fwd Pkts": [1, 2], "Tot Bwd Pkts": [1, 1],
        }))).frame
        self.assertEqual(len(frame), 2)
        self.assertTrue(np.isfinite(frame.start_ts.to_numpy()).all())


if __name__ == "__main__":
    unittest.main()
