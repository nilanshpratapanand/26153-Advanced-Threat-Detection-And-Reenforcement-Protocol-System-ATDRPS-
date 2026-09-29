"""Real-background overlay: the ground truth has to be exactly what was injected."""

import os
import tempfile
import unittest

import numpy as np

from atdrps.data.flows import assemble_flows
from atdrps.data.overlay import load_background, overlay_campaigns, synthetic_background
from atdrps.data.pcap import write_pcap
from atdrps.data.schema import PacketTable
from atdrps.data.windows import build_windows

VALID = {"Reconnaissance", "InitialAccess", "LateralMovement", "CommandAndControl", "Exfiltration"}


class TestOverlay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bg = synthetic_background(seed=11, duration_s=3000.0)

    def test_deterministic_for_a_seed(self):
        a = overlay_campaigns(self.bg, seed=5, n_campaigns=2)
        b = overlay_campaigns(self.bg, seed=5, n_campaigns=2)
        self.assertEqual([(i.start, i.stage) for i in a.timeline], [(i.start, i.stage) for i in b.timeline])
        self.assertEqual(len(a.packets), len(b.packets))
        c = overlay_campaigns(self.bg, seed=6, n_campaigns=2)
        self.assertNotEqual([(i.start, i.stage) for i in a.timeline], [(i.start, i.stage) for i in c.timeline])

    def test_background_is_preserved_untouched(self):
        cap = overlay_campaigns(self.bg, seed=2, n_campaigns=1)
        self.assertEqual(len(cap.packets), len(self.bg.records) + cap.meta["n_injected_packets"])
        ids = {id(r) for r in cap.packets}
        self.assertTrue(all(id(r) in ids for r in self.bg.records))
        ts = [p.ts for p in cap.packets]
        self.assertEqual(ts, sorted(ts))

    def test_timeline_is_well_formed(self):
        for seed in range(12):
            cap = overlay_campaigns(self.bg, seed=seed, n_campaigns=2)
            for iv in cap.timeline:
                self.assertIn(iv.stage, VALID)
                self.assertGreaterEqual(iv.start, self.bg.t0 - 1e-6)
                self.assertLessEqual(iv.end, self.bg.t1 + 1e-6)
                self.assertLessEqual(iv.start, iv.end)

    def test_decoys_add_traffic_but_no_labels(self):
        cap = overlay_campaigns(self.bg, seed=3, n_campaigns=0, decoys_per_hour=40)
        self.assertEqual(cap.timeline, [])
        self.assertGreater(cap.meta["n_decoys"], 0)
        self.assertGreater(cap.meta["n_injected_packets"], 0)

    def test_direct_access_skips_reconnaissance(self):
        for seed in range(10):
            direct = overlay_campaigns(self.bg, seed=seed, n_campaigns=1, direct_access_prob=1.0)
            self.assertNotIn("Reconnaissance", {i.stage for i in direct.timeline})
            recon = overlay_campaigns(self.bg, seed=seed, n_campaigns=1, direct_access_prob=0.0)
            if recon.timeline:
                self.assertEqual(recon.timeline[0].stage, "Reconnaissance")

    def test_dwell_between_recon_and_access_is_heavy_tailed(self):
        dwell = []
        for seed in range(200):
            cap = overlay_campaigns(self.bg, seed=seed, n_campaigns=1, direct_access_prob=0.0,
                                    continue_probs=(1.0, 0.0, 0.0, 0.0), decoys_per_hour=0)
            st = {i.stage: i for i in cap.timeline}
            if "Reconnaissance" in st and "InitialAccess" in st:
                dwell.append(st["InitialAccess"].start - st["Reconnaissance"].end)
        self.assertGreater(len(dwell), 60)
        med, p90 = float(np.median(dwell)), float(np.percentile(dwell, 90))
        self.assertGreater(med, 100)
        self.assertLess(med, 500)
        self.assertGreater(p90, 2 * med, "dwell should have a long tail, not a narrow uniform gap")
        self.assertGreater(max(dwell), 600)

    def test_windows_carry_the_injected_labels(self):
        cap = overlay_campaigns(self.bg, seed=4, n_campaigns=1, direct_access_prob=1.0,
                                continue_probs=(1, 1, 0, 0))
        flows = assemble_flows(PacketTable.from_records(cap.packets))
        st = build_windows(flows, window_size_s=30.0, stage_timeline=cap.stage_at)
        self.assertGreater(int(st.infiltration.sum()), 0)
        self.assertLess(int(st.infiltration.sum()), len(st) // 2)

    def test_load_background_picks_a_private_victim(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bg.pcap")
            write_pcap(path, self.bg.records[:5000])
            bg = load_background(path)
        self.assertTrue(bg.victim.startswith(("10.", "192.168.", "172.")))
        self.assertGreaterEqual(len(bg.peers), 3)
        self.assertNotIn(bg.victim, bg.peers)

    def test_segments_are_disjoint_in_time(self):
        a, b = self.bg.segment(0.0, 0.5), self.bg.segment(0.5, 1.0)
        self.assertLess(max(r.ts for r in a.records), min(r.ts for r in b.records))
        self.assertEqual(a.t1, b.t0)


if __name__ == "__main__":
    unittest.main()
