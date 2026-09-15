"""The synthetic generator is development data, so it has to be trustworthy.

These tests check the three properties the rest of the pipeline relies on:
determinism, a well-formed ground-truth timeline, and traffic that actually
carries the signatures each stage is supposed to have.
"""

import os
import tempfile
import unittest
from collections import Counter

from atdrps.data.pcap import read_pcap
from atdrps.data.schema import PROTO_TCP, TCP_RST, TCP_SYN
from atdrps.data.synth import generate_capture, write_scenario

CHAIN_ORDER = ["Reconnaissance", "InitialAccess", "LateralMovement",
               "CommandAndControl", "Exfiltration"]


class TestGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cap = generate_capture(seed=11, duration_s=900, n_campaigns=3,
                                   background_intensity=0.2)

    def test_deterministic_for_a_given_seed(self):
        a = generate_capture(seed=5, duration_s=300, n_campaigns=1, background_intensity=0.1)
        b = generate_capture(seed=5, duration_s=300, n_campaigns=1, background_intensity=0.1)
        self.assertEqual(len(a.packets), len(b.packets))
        self.assertEqual(
            [(p.ts, p.src_ip, p.dst_port, p.tcp_flags) for p in a.packets[:500]],
            [(p.ts, p.src_ip, p.dst_port, p.tcp_flags) for p in b.packets[:500]],
        )
        self.assertEqual([iv.stage for iv in a.timeline], [iv.stage for iv in b.timeline])

    def test_different_seeds_differ(self):
        a = generate_capture(seed=5, duration_s=300, n_campaigns=1, background_intensity=0.1)
        b = generate_capture(seed=6, duration_s=300, n_campaigns=1, background_intensity=0.1)
        self.assertNotEqual(len(a.packets), len(b.packets))

    def test_packets_are_time_ordered(self):
        ts = [p.ts for p in self.cap.packets]
        self.assertEqual(ts, sorted(ts))

    def test_timeline_respects_the_kill_chain_order(self):
        per_campaign = {}
        for iv in self.cap.timeline:
            per_campaign.setdefault(iv.campaign, []).append(iv)
        self.assertTrue(per_campaign, "generator produced no campaigns")
        for campaign, intervals in per_campaign.items():
            with self.subTest(campaign=campaign):
                intervals.sort(key=lambda i: i.start)
                stages = [i.stage for i in intervals]
                # stages must be a prefix of the kill chain, in order
                self.assertEqual(stages, CHAIN_ORDER[:len(stages)])
                # and must not overlap in time
                for earlier, later in zip(intervals, intervals[1:]):
                    self.assertLessEqual(earlier.end, later.start)

    def test_intervals_are_non_empty(self):
        for iv in self.cap.timeline:
            self.assertGreater(iv.end, iv.start, f"{iv.stage} has zero duration")

    def test_stage_at_matches_the_timeline(self):
        for iv in self.cap.timeline:
            mid = (iv.start + iv.end) / 2
            resolved = self.cap.stage_at(mid)
            self.assertIn(resolved, CHAIN_ORDER)
        before_everything = min(p.ts for p in self.cap.packets) - 10
        self.assertEqual(self.cap.stage_at(before_everything), "Benign")

    def test_all_five_stages_appear_across_seeds(self):
        seen = Counter()
        for seed in range(6):
            cap = generate_capture(seed=seed, duration_s=1800, n_campaigns=3,
                                   background_intensity=0.15)
            for iv in cap.timeline:
                seen[iv.stage] += 1
        for stage in CHAIN_ORDER:
            with self.subTest(stage=stage):
                self.assertGreater(seen[stage], 0, f"{stage} never generated")

    def test_not_every_campaign_completes(self):
        """A model must not be able to learn 'scan implies exfiltration'."""
        started = completed = 0
        for seed in range(12):
            cap = generate_capture(seed=seed, duration_s=1800, n_campaigns=3,
                                   background_intensity=0.1)
            stages = Counter(iv.stage for iv in cap.timeline)
            started += stages["Reconnaissance"]
            completed += stages["Exfiltration"]
        self.assertGreater(started, 0)
        self.assertGreater(completed, 0, "no campaign ever completes")
        self.assertLess(completed, started, "every campaign completes - too easy")

    def test_reconnaissance_traffic_looks_like_a_scan(self):
        recon = [iv for iv in self.cap.timeline if iv.stage == "Reconnaissance"]
        self.assertTrue(recon)
        iv = recon[0]
        probes = [p for p in self.cap.packets
                  if iv.start <= p.ts <= iv.end and p.src_ip == iv.attacker
                  and p.protocol == PROTO_TCP and p.tcp_flags & TCP_SYN]
        self.assertGreater(len(probes), 30)
        self.assertEqual(sum(p.payload_len for p in probes), 0, "scans carry no payload")
        self.assertGreater(len({p.dst_port for p in probes}), 20, "not enough distinct ports")
        resets = [p for p in self.cap.packets
                  if iv.start <= p.ts <= iv.end and p.tcp_flags & TCP_RST]
        self.assertGreater(len(resets), 10, "closed ports should answer with RST")

    def test_exfiltration_is_strongly_outbound(self):
        exfil = [iv for iv in self.cap.timeline if iv.stage == "Exfiltration"]
        if not exfil:
            self.skipTest("this seed produced no exfiltration stage")
        iv = exfil[0]
        out = sum(p.payload_len for p in self.cap.packets
                  if iv.start <= p.ts <= iv.end and p.src_ip == iv.victim)
        back = sum(p.payload_len for p in self.cap.packets
                   if iv.start <= p.ts <= iv.end and p.dst_ip == iv.victim)
        self.assertGreater(out, 500_000)
        self.assertGreater(out, 20 * max(back, 1))

    def test_survives_a_pcap_roundtrip(self):
        cap = generate_capture(seed=2, duration_s=240, n_campaigns=1, background_intensity=0.1)
        with tempfile.TemporaryDirectory() as tmp:
            info = write_scenario(tmp, cap, name="unit")
            self.assertTrue(os.path.exists(info["pcap"]))
            self.assertTrue(os.path.exists(info["timeline"]))
            table = read_pcap(info["pcap"])
        self.assertEqual(len(table), len(cap.packets))
        self.assertAlmostEqual(float(table.ts[0]), cap.packets[0].ts, places=4)


if __name__ == "__main__":
    unittest.main()
