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
               "CommandAndControl", "Exfiltration", "Impact"]


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
            # campaign -1 is the standalone-incident bucket: a worm outbreak or
            # a SYN flood is not a stage of anybody's kill chain, so it is not
            # held to the chain ordering.
            if iv.campaign < 0:
                continue
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


class TestAdditionalAttackFamilies(unittest.TestCase):
    """The attacks added from the team's taxonomy (docs/ATTACK_COVERAGE.md).

    Each is checked for the signature it is supposed to have, because a
    generator that produces the right *label* but the wrong *traffic* teaches
    the model nothing -- and would look fine in every other test.
    """

    @classmethod
    def setUpClass(cls):
        cls.captures = [
            generate_capture(seed=s, duration_s=3600, n_campaigns=2,
                             background_intensity=0.12, n_incidents=6)
            for s in range(5)
        ]

    def _incidents(self, needle):
        out = []
        for cap in self.captures:
            for iv in cap.timeline:
                if needle in iv.note.lower():
                    out.append((cap, iv))
        return out

    def test_every_family_is_generated(self):
        for needle in ("worm", "syn flood", "dns tunnel", "credential stuffing",
                       "web application", "machine-in-the-middle", "ransomware"):
            with self.subTest(family=needle):
                self.assertTrue(self._incidents(needle), f"{needle} never generated")

    def test_incidents_span_more_than_one_window(self):
        """A 30 s window cannot see an attack that lasts 5 s."""
        for needle in ("worm", "syn flood", "dns tunnel", "ransomware"):
            for cap, iv in self._incidents(needle):
                with self.subTest(family=needle):
                    self.assertGreater(iv.end - iv.start, 30.0)

    def test_worm_fans_out_on_a_single_port(self):
        """A scan is many ports on one host; a worm is many hosts on one port.
        Without that distinction the two are the same thing in aggregate."""
        cap, iv = self._incidents("worm")[0]
        pkts = [p for p in cap.packets if iv.start <= p.ts <= iv.end
                and p.src_ip == iv.attacker and p.tcp_flags & TCP_SYN]
        self.assertGreater(len({p.dst_ip for p in pkts}), 5)
        self.assertLessEqual(len({p.dst_port for p in pkts}), 2)

    def test_syn_flood_never_completes_a_handshake(self):
        cap, iv = self._incidents("syn flood")[0]
        window = [p for p in cap.packets if iv.start <= p.ts <= iv.end
                  and p.dst_ip == iv.victim]
        syns = [p for p in window if p.tcp_flags & TCP_SYN]
        self.assertGreater(len(syns), 200)
        self.assertGreater(len({p.src_ip for p in syns}), 10)   # distributed
        self.assertEqual(sum(p.payload_len for p in syns), 0)

    def test_dns_tunnel_queries_are_abnormally_large(self):
        """Ordinary lookups are tens of bytes; encoded data needs hundreds."""
        cap, iv = self._incidents("dns tunnel")[0]
        tunnel = [p.payload_len for p in cap.packets
                  if iv.start <= p.ts <= iv.end and p.dst_port == 53]
        benign = [p.payload_len for p in cap.packets
                  if p.ts < iv.start and p.dst_port == 53]
        self.assertTrue(tunnel and benign)
        self.assertGreater(sum(tunnel) / len(tunnel), 2 * sum(benign) / len(benign))

    def test_credential_stuffing_is_distributed_not_concentrated(self):
        """This is what separates it from brute force: many sources, few
        attempts each, so per-source rate limits never trigger."""
        cap, iv = self._incidents("credential stuffing")[0]
        window = [p for p in cap.packets if iv.start <= p.ts <= iv.end
                  and p.dst_ip == iv.victim and p.tcp_flags & TCP_SYN]
        sources = {p.src_ip for p in window}
        self.assertGreater(len(sources), 8)
        self.assertLess(len(window) / max(len(sources), 1), 25)

    def test_mitm_makes_one_source_arrive_with_several_ttls(self):
        cap, iv = self._incidents("machine-in-the-middle")[0]
        ttls = {}
        for p in cap.packets:
            if iv.start <= p.ts <= iv.end:
                ttls.setdefault(p.src_ip, set()).add(p.ip_ttl)
        self.assertTrue(any(len(v) > 1 for v in ttls.values()),
                        "no source shows TTL inconsistency")

    def test_benign_sources_have_a_stable_ttl(self):
        """The counterpart to the test above: if ordinary hosts jitter their
        TTL, the MitM signal is worthless."""
        cap = self.captures[0]
        quiet_end = min(iv.start for iv in cap.timeline)
        ttls = {}
        for p in cap.packets:
            if p.ts < quiet_end:
                ttls.setdefault(p.src_ip, set()).add(p.ip_ttl)
        inconsistent = [k for k, v in ttls.items() if len(v) > 1]
        self.assertEqual(inconsistent, [], f"unstable TTL for {inconsistent}")

    def test_ransomware_writes_back_at_mtu_over_smb(self):
        cap, iv = self._incidents("ransomware")[0]
        writes = [p for p in cap.packets if iv.start <= p.ts <= iv.end
                  and p.dst_port == 445 and p.payload_len > 0]
        self.assertGreater(len(writes), 200)
        at_mtu = [p for p in writes if p.payload_len >= 1400]
        self.assertGreater(len(at_mtu) / len(writes), 0.2)
        self.assertGreater(len({p.dst_ip for p in writes}), 1)   # several shares
