import unittest

import numpy as np
import pandas as pd

from atdrps.data.flows import FLOW_META_COLUMNS
from atdrps.data.schema import (
    BENIGN_STAGE, FLOW_FEATURES, PACKET_FEATURES, STAGE_INDEX, STAGES,
    PacketTable,
)
from atdrps.data.synth import generate_capture
from atdrps.data.flows import assemble_flows
from atdrps.data.windows import build_windows, is_internal, state_feature_names


def make_flows(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal flow table; unspecified features default to zero."""
    columns = list(FLOW_META_COLUMNS) + list(FLOW_FEATURES) + list(PACKET_FEATURES) + ["label"]
    frame = pd.DataFrame(0.0, index=range(len(rows)), columns=columns)
    frame["src_ip"] = ""
    frame["dst_ip"] = ""
    frame["label"] = "Benign"
    for i, row in enumerate(rows):
        for key, value in row.items():
            frame.loc[i, key] = value
    return frame


def flow(ts, src="10.0.0.1", dst="10.0.0.2", dport=80, sport=1111,
         proto=6, total_bytes=1000.0, fwd_bytes=600.0, bwd_bytes=400.0,
         total_packets=10.0, label="Benign"):
    return {
        "start_ts": ts, "end_ts": ts + 0.5, "src_ip": src, "dst_ip": dst,
        "dst_port_raw": dport, "src_port_raw": sport, "protocol_raw": proto,
        "total_bytes": total_bytes, "fwd_bytes": fwd_bytes, "bwd_bytes": bwd_bytes,
        "total_packets": total_packets, "label": label,
    }


class TestFeatureSpace(unittest.TestCase):
    def test_names_are_unique_and_stable(self):
        names = state_feature_names()
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(names, state_feature_names())

    def test_state_dimension_matches_names(self):
        frame = make_flows([flow(0.0), flow(1.0)])
        states = build_windows(frame, window_size_s=30.0)
        self.assertEqual(states.n_features, len(state_feature_names()))
        self.assertEqual(states.feature_names, state_feature_names())

    def test_empty_input(self):
        states = build_windows(pd.DataFrame(), window_size_s=30.0)
        self.assertEqual(len(states), 0)
        self.assertEqual(states.X.shape[1], len(state_feature_names()))


class TestInternalDetection(unittest.TestCase):
    def test_private_ranges(self):
        for addr in ("10.0.0.1", "192.168.1.1", "172.16.5.5", "172.31.255.255", "127.0.0.1"):
            self.assertTrue(is_internal(addr), addr)

    def test_public_ranges(self):
        for addr in ("8.8.8.8", "93.184.216.34", "172.32.0.1", "185.1.2.3"):
            self.assertFalse(is_internal(addr), addr)

    def test_garbage_is_not_internal(self):
        self.assertFalse(is_internal("nonsense"))
        self.assertFalse(is_internal("v6:abcd"))


class TestWindowing(unittest.TestCase):
    def test_window_boundaries_are_half_open(self):
        frame = make_flows([flow(0.0), flow(29.999), flow(30.0), flow(59.0)])
        states = build_windows(frame, window_size_s=30.0)
        self.assertEqual(len(states), 2)
        self.assertEqual(len(states.flow_index[0]), 2)
        self.assertEqual(len(states.flow_index[1]), 2)

    def test_quiet_window_is_emitted_as_zeros(self):
        """A gap in traffic is an observation, not a missing row -- dropping it
        would break the uniform time axis the sequence model assumes."""
        frame = make_flows([flow(0.0), flow(120.0)])
        states = build_windows(frame, window_size_s=30.0)
        self.assertEqual(len(states), 5)
        self.assertEqual(len(states.flow_index[1]), 0)
        self.assertTrue(np.allclose(states.X[1], 0.0))
        self.assertTrue(np.isfinite(states.X).all())

    def test_stride_smaller_than_window(self):
        frame = make_flows([flow(float(i)) for i in range(0, 60, 5)])
        states = build_windows(frame, window_size_s=30.0, stride_s=10.0)
        self.assertGreater(len(states), 5)

    def test_timestamps_are_monotonic(self):
        frame = make_flows([flow(float(i) * 7) for i in range(20)])
        states = build_windows(frame, window_size_s=30.0)
        self.assertTrue((np.diff(states.ts_start) > 0).all())


class TestStateValues(unittest.TestCase):
    def index(self, states, name):
        return states.feature_names.index(name)

    def test_counts_and_distinct_ports(self):
        rows = [flow(1.0, dport=p) for p in (80, 443, 22)]
        states = build_windows(make_flows(rows), window_size_s=30.0)
        x = states.X[0]
        self.assertAlmostEqual(x[self.index(states, "log_n_flows")], np.log1p(3), places=5)
        self.assertAlmostEqual(
            x[self.index(states, "log_distinct_dst_ports")], np.log1p(3), places=5
        )

    def test_scan_detector_counts_ports_per_pair(self):
        rows = [flow(1.0 + i * 0.01, dport=1000 + i) for i in range(25)]
        states = build_windows(make_flows(rows), window_size_s=30.0)
        self.assertEqual(states.X[0][self.index(states, "max_ports_per_src_dst")], 25.0)
        # consecutive ports -> a deliberate sequential sweep
        self.assertGreater(states.X[0][self.index(states, "sequential_port_ratio")], 0.9)

    def test_randomised_scan_is_not_flagged_sequential(self):
        rng = np.random.default_rng(0)
        ports = rng.choice(np.arange(1, 60000), size=25, replace=False)
        rows = [flow(1.0 + i * 0.01, dport=int(p)) for i, p in enumerate(ports)]
        states = build_windows(make_flows(rows), window_size_s=30.0)
        self.assertLess(states.X[0][self.index(states, "sequential_port_ratio")], 0.2)
        self.assertEqual(states.X[0][self.index(states, "max_ports_per_src_dst")], 25.0)

    def test_outbound_bytes_ratio(self):
        rows = [
            # internal -> external, 900 bytes leaving
            flow(1.0, src="10.0.0.5", dst="93.1.1.1", fwd_bytes=900, bwd_bytes=100, total_bytes=1000),
            # internal -> internal, nothing leaves
            flow(2.0, src="10.0.0.5", dst="10.0.0.6", fwd_bytes=500, bwd_bytes=500, total_bytes=1000),
        ]
        states = build_windows(make_flows(rows), window_size_s=30.0)
        self.assertAlmostEqual(
            states.X[0][self.index(states, "outbound_bytes_ratio")], 900 / 2000, places=5
        )

    def test_beacon_detector_needs_history_not_just_the_window(self):
        """A 60 s beacon puts one flow in a 30 s window. Measured over the
        lookback it is obvious; measured inside the window it is invisible."""
        rows = [flow(float(i) * 60.0, src="10.0.0.5", dst="45.1.1.1", dport=443)
                for i in range(12)]
        states = build_windows(make_flows(rows), window_size_s=30.0, history_s=600.0)
        col = self.index(states, "beacon_regularity")
        self.assertGreater(states.X[-1][col], 0.85)

    def test_bursty_traffic_is_not_a_beacon(self):
        rng = np.random.default_rng(3)
        times = np.cumsum(rng.exponential(60.0, size=12))
        rows = [flow(float(t), src="10.0.0.5", dst="45.1.1.1", dport=443) for t in times]
        states = build_windows(make_flows(rows), window_size_s=30.0, history_s=900.0)
        col = self.index(states, "beacon_regularity")
        self.assertLess(states.X[:, col].max(), 0.85)

    def test_scan_rate_traffic_is_not_a_beacon(self):
        """A port scan is metronomic but it is not beaconing.

        nmap -T4 emits probes at an almost perfectly constant ~2.4 ms interval,
        which scores a coefficient of variation near zero. Before the interval
        floor existed that scored 1.00 regularity and tripped the C2 beacon
        rule -- and because regularity is measured over the history window, a
        2.4-second scan poisoned the C2 score for the next ten minutes.
        """
        # 400 probes, perfectly regular, 2.4 ms apart: a scan, not an implant
        times = np.arange(400) * 0.0024
        rows = [flow(float(t), src="192.168.1.77", dst="10.0.0.9", dport=443)
                for t in times]
        states = build_windows(make_flows(rows), window_size_s=30.0, history_s=900.0)
        col = self.index(states, "beacon_regularity")
        self.assertEqual(
            states.X[:, col].max(), 0.0,
            "scan-rate traffic must not register as a beacon at any regularity",
        )

    def test_a_real_slow_beacon_still_registers(self):
        """The floor must not cost us the thing the detector is for."""
        times = np.arange(20) * 60.0          # a 60 s implant, the classic case
        rows = [flow(float(t), src="10.0.0.5", dst="45.1.1.1", dport=443)
                for t in times]
        states = build_windows(make_flows(rows), window_size_s=30.0, history_s=900.0)
        col = self.index(states, "beacon_regularity")
        self.assertGreater(states.X[:, col].max(), 0.9)

    def test_new_destination_ratio_falls_as_hosts_repeat(self):
        rows = [flow(float(i) * 31.0, dst="10.0.0.2") for i in range(5)]
        states = build_windows(make_flows(rows), window_size_s=30.0)
        col = self.index(states, "new_dst_ratio")
        self.assertEqual(states.X[0][col], 1.0)
        self.assertEqual(states.X[-1][col], 0.0)

    def test_all_states_finite_on_real_traffic(self):
        cap = generate_capture(seed=6, duration_s=900, n_campaigns=2, background_intensity=0.2)
        flows = assemble_flows(PacketTable.from_records(cap.packets))
        states = build_windows(flows, window_size_s=30.0, stage_timeline=cap.stage_at)
        self.assertTrue(np.isfinite(states.X).all())
        self.assertGreater(len(states), 20)


class TestLabelling(unittest.TestCase):
    def test_timeline_labels_take_the_most_severe_stage(self):
        def timeline(ts):
            return "LateralMovement" if 30 <= ts <= 45 else BENIGN_STAGE

        frame = make_flows([flow(float(i) * 10) for i in range(9)])
        states = build_windows(frame, window_size_s=30.0, stage_timeline=timeline)
        names = states.stage_names()
        self.assertEqual(names[0], BENIGN_STAGE)
        self.assertEqual(names[1], "LateralMovement")
        self.assertTrue(states.stage_mask.all())
        self.assertEqual(states.infiltration[1], 1)
        self.assertEqual(states.infiltration[0], 0)

    def test_dataset_labels_pick_the_most_severe_flow(self):
        frame = make_flows([
            flow(1.0, label="Benign"),
            flow(2.0, label="SSH-Bruteforce"),
            flow(3.0, label="Benign"),
        ])
        states = build_windows(frame, window_size_s=30.0)
        self.assertEqual(states.stage_names()[0], "InitialAccess")
        self.assertEqual(states.label_source, "dataset labels")

    def test_out_of_scope_only_window_is_masked_out(self):
        """Unrecognised families must not supervise the stage head."""
        frame = make_flows([flow(1.0, label="Fuzzers"),
                            flow(2.0, label="some-2027-attack")])
        states = build_windows(frame, window_size_s=30.0)
        self.assertFalse(states.stage_mask[0])
        self.assertEqual(states.infiltration[0], 0)

    def test_denial_of_service_window_is_labelled_impact(self):
        """DoS is a real stage now, so it trains the model instead of being
        masked away -- that is what teaches it a flood is not a scan."""
        frame = make_flows([flow(1.0, label="DDoS attacks-LOIC-HTTP"),
                            flow(2.0, label="DoS attacks-Hulk")])
        states = build_windows(frame, window_size_s=30.0)
        self.assertTrue(states.stage_mask[0])
        self.assertEqual(states.stage_names()[0], "Impact")
        self.assertEqual(states.infiltration[0], 1)

    def test_unlabelled_capture_trains_nothing(self):
        frame = make_flows([flow(1.0), flow(2.0)]).drop(columns=["label"])
        states = build_windows(frame, window_size_s=30.0)
        self.assertEqual(states.label_source, "unlabelled")
        self.assertFalse(states.stage_mask.any())

    def test_infiltration_flag_follows_the_stage(self):
        for stage, expected in (("Reconnaissance", 0), ("InitialAccess", 1),
                                ("Exfiltration", 1), (BENIGN_STAGE, 0)):
            with self.subTest(stage=stage):
                frame = make_flows([flow(1.0)])
                states = build_windows(frame, window_size_s=30.0,
                                       stage_timeline=lambda ts, s=stage: s)
                self.assertEqual(states.infiltration[0], expected)
                self.assertEqual(states.stage[0], STAGE_INDEX[stage])


if __name__ == "__main__":
    unittest.main()


class TestAttackFamilyDetectors(unittest.TestCase):
    """The seven detectors added for the attack taxonomy.

    Each is driven with hand-built traffic that has exactly the property the
    detector is named after, and with a benign control that does not -- so a
    detector that fires on everything fails here.
    """

    def index(self, states, name):
        return states.feature_names.index(name)

    def value(self, rows, name, window=0, **kw):
        states = build_windows(make_flows(rows), window_size_s=30.0, **kw)
        return float(states.X[window][self.index(states, name)]), states

    def test_worm_fanout_separates_from_a_port_scan(self):
        """One source, many hosts, ONE port -- versus one source, one host,
        many ports. In aggregate these look alike; this is what splits them."""
        worm = [flow(1.0 + i * 0.1, dst=f"10.0.0.{i + 2}", dport=445) for i in range(12)]
        scan = [flow(1.0 + i * 0.1, dst="10.0.0.2", dport=1000 + i) for i in range(12)]
        worm_v, _ = self.value(worm, "log_same_port_fanout")
        scan_v, _ = self.value(scan, "log_same_port_fanout")
        self.assertGreater(worm_v, np.log1p(10))
        self.assertLess(scan_v, np.log1p(2))

    def test_half_open_ratio_catches_a_flood(self):
        flood = [dict(flow(1.0 + i * 0.01, src=f"9.9.9.{i % 50}"),
                      syn_count=1.0, ack_count=0.0) for i in range(60)]
        normal = [dict(flow(1.0 + i * 0.1), syn_count=1.0, ack_count=8.0) for i in range(30)]
        self.assertGreater(self.value(flood, "half_open_ratio")[0], 0.9)
        self.assertEqual(self.value(normal, "half_open_ratio")[0], 0.0)

    def test_many_sources_on_one_service(self):
        stuffing = [flow(1.0 + i * 0.1, src=f"9.9.9.{i}", dst="10.0.0.5", dport=443)
                    for i in range(20)]
        spread = [flow(1.0 + i * 0.1, src="10.0.0.1", dst=f"10.0.0.{i + 2}", dport=443)
                  for i in range(20)]
        self.assertGreater(self.value(stuffing, "log_max_srcs_per_dst_service")[0],
                           np.log1p(15))
        self.assertLess(self.value(spread, "log_max_srcs_per_dst_service")[0], np.log1p(2))

    def test_dns_query_size_separates_tunnelling_from_lookups(self):
        """The response/query ratio does not work -- real lookups already return
        far more than they ask. The query size does."""
        tunnel = [flow(1.0 + i * 0.1, dport=53, proto=17, fwd_bytes=200.0,
                       bwd_bytes=700.0, total_bytes=900.0) for i in range(20)]
        lookups = [flow(1.0 + i * 0.1, dport=53, proto=17, fwd_bytes=45.0,
                        bwd_bytes=280.0, total_bytes=325.0) for i in range(20)]
        self.assertGreater(self.value(tunnel, "log_dns_query_size")[0],
                           self.value(lookups, "log_dns_query_size")[0] + 1.0)
        self.assertAlmostEqual(self.value(tunnel, "dns_share")[0], 1.0, places=5)

    def test_ttl_inconsistency_is_a_fraction_not_a_count(self):
        """A raw count of distinct TTLs grows with traffic volume and would just
        re-measure how busy the window is."""
        relayed = [dict(flow(1.0 + i * 0.1, src="10.0.0.5", dst=f"10.0.0.{i + 10}"),
                        fwd_ttl_mean=float(64 - (i % 2) * 4)) for i in range(10)]
        stable = [dict(flow(1.0 + i * 0.1, src="10.0.0.5", dst=f"10.0.0.{i + 10}"),
                       fwd_ttl_mean=64.0) for i in range(10)]
        self.assertGreater(self.value(relayed, "ttl_inconsistency")[0], 0.5)
        self.assertEqual(self.value(stable, "ttl_inconsistency")[0], 0.0)

    def test_rst_injection_ignores_refused_connections(self):
        """A failed login is handshake, request, reply, reset -- six packets.
        Counting that as injection would flag every brute-force window."""
        injected = [dict(flow(1.0 + i * 0.1, total_packets=40.0), rst_count=1.0)
                    for i in range(10)]
        refused = [dict(flow(1.0 + i * 0.1, total_packets=6.0), rst_count=1.0)
                   for i in range(10)]
        self.assertGreater(self.value(injected, "rst_injection_ratio")[0], 0.9)
        self.assertEqual(self.value(refused, "rst_injection_ratio")[0], 0.0)

    def test_detectors_stay_finite_on_an_empty_window(self):
        states = build_windows(make_flows([flow(1.0), flow(200.0)]), window_size_s=30.0)
        self.assertTrue(np.isfinite(states.X).all())
