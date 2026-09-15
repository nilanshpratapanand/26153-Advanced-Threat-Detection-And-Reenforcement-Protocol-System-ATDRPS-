"""Flow-assembly tests.

Expected values here are computed by hand in the test, not read back out of the
implementation, so a change in the extractor shows up as a failure rather than
as a quietly different number.
"""

import unittest

import numpy as np

from atdrps.data.flows import assemble_flows
from atdrps.data.schema import (
    FLOW_FEATURES, IP_FLAG_MF, PACKET_FEATURES, PROTO_TCP, PROTO_UDP,
    TCP_ACK, TCP_FIN, TCP_PSH, TCP_RST, TCP_SYN,
    PacketRecord, PacketTable,
)
from atdrps.data.synth import generate_capture


def pkt(ts, src, dst, sport, dport, **kw):
    kw.setdefault("protocol", PROTO_TCP)
    kw.setdefault("ip_len", 40 + kw.get("payload_len", 0))
    return PacketRecord(ts=ts, src_ip=src, dst_ip=dst, src_port=sport, dst_port=dport, **kw)


def table(records):
    return PacketTable.from_records(records)


class TestFlowKeys(unittest.TestCase):
    def test_both_directions_are_one_flow(self):
        recs = [
            pkt(0.0, "10.0.0.1", "10.0.0.2", 1111, 80, tcp_flags=TCP_SYN),
            pkt(0.1, "10.0.0.2", "10.0.0.1", 80, 1111, tcp_flags=TCP_SYN | TCP_ACK),
            pkt(0.2, "10.0.0.1", "10.0.0.2", 1111, 80, tcp_flags=TCP_ACK),
        ]
        df = assemble_flows(table(recs))
        self.assertEqual(len(df), 1)
        row = df.iloc[0]
        self.assertEqual(row.fwd_packets, 2)   # direction of the first packet
        self.assertEqual(row.bwd_packets, 1)
        self.assertEqual(row.total_packets, 3)

    def test_different_ports_are_different_flows(self):
        recs = [
            pkt(0.0, "10.0.0.1", "10.0.0.2", 1111, 80),
            pkt(0.1, "10.0.0.1", "10.0.0.2", 2222, 80),
        ]
        self.assertEqual(len(assemble_flows(table(recs))), 2)

    def test_different_protocols_are_different_flows(self):
        recs = [
            pkt(0.0, "10.0.0.1", "10.0.0.2", 1111, 53, protocol=PROTO_TCP),
            pkt(0.1, "10.0.0.1", "10.0.0.2", 1111, 53, protocol=PROTO_UDP),
        ]
        self.assertEqual(len(assemble_flows(table(recs))), 2)


class TestFlowLifetime(unittest.TestCase):
    def test_idle_timeout_splits(self):
        recs = [
            pkt(0.0, "10.0.0.1", "10.0.0.2", 1111, 80),
            pkt(1.0, "10.0.0.1", "10.0.0.2", 1111, 80),
            pkt(500.0, "10.0.0.1", "10.0.0.2", 1111, 80),
        ]
        df = assemble_flows(table(recs), idle_timeout_s=120.0, active_timeout_s=10_000)
        self.assertEqual(len(df), 2)
        self.assertEqual(list(df.total_packets), [2.0, 1.0])

    def test_active_timeout_splits(self):
        recs = [pkt(float(i), "10.0.0.1", "10.0.0.2", 1111, 80) for i in range(0, 60, 5)]
        df = assemble_flows(table(recs), idle_timeout_s=120.0, active_timeout_s=20.0)
        self.assertGreater(len(df), 1)
        for dur in df.flow_duration:
            self.assertLessEqual(dur, 25.0)

    def test_rst_closes_the_flow(self):
        recs = [
            pkt(0.0, "10.0.0.1", "10.0.0.2", 1111, 80, tcp_flags=TCP_SYN),
            pkt(0.1, "10.0.0.2", "10.0.0.1", 80, 1111, tcp_flags=TCP_RST | TCP_ACK),
            pkt(0.2, "10.0.0.1", "10.0.0.2", 1111, 80, tcp_flags=TCP_SYN),
        ]
        self.assertEqual(len(assemble_flows(table(recs))), 2)

    def test_graceful_teardown_keeps_the_final_ack(self):
        """FIN, FIN, ACK is one flow -- not a flow plus a stray one-packet flow."""
        recs = [
            pkt(0.0, "10.0.0.1", "10.0.0.2", 1111, 80, tcp_flags=TCP_SYN),
            pkt(0.1, "10.0.0.2", "10.0.0.1", 80, 1111, tcp_flags=TCP_SYN | TCP_ACK),
            pkt(0.2, "10.0.0.1", "10.0.0.2", 1111, 80, tcp_flags=TCP_ACK),
            pkt(1.0, "10.0.0.1", "10.0.0.2", 1111, 80, tcp_flags=TCP_FIN | TCP_ACK),
            pkt(1.1, "10.0.0.2", "10.0.0.1", 80, 1111, tcp_flags=TCP_FIN | TCP_ACK),
            pkt(1.2, "10.0.0.1", "10.0.0.2", 1111, 80, tcp_flags=TCP_ACK),
        ]
        df = assemble_flows(table(recs))
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0].total_packets, 6)

    def test_teardown_can_be_disabled(self):
        recs = [
            pkt(0.0, "10.0.0.1", "10.0.0.2", 1111, 80, tcp_flags=TCP_SYN),
            pkt(0.1, "10.0.0.2", "10.0.0.1", 80, 1111, tcp_flags=TCP_RST),
            pkt(0.2, "10.0.0.1", "10.0.0.2", 1111, 80, tcp_flags=TCP_SYN),
        ]
        self.assertEqual(len(assemble_flows(table(recs), close_on_teardown=False)), 1)


class TestFlowFeatureValues(unittest.TestCase):
    def setUp(self):
        # forward: t=0,1,3 (payload 100,200,300) ; backward: t=2,6 (payload 50,50)
        self.recs = [
            pkt(0.0, "10.0.0.1", "10.0.0.2", 1111, 80, payload_len=100, tcp_flags=TCP_SYN, ip_ttl=64, tcp_window=1000),
            pkt(1.0, "10.0.0.1", "10.0.0.2", 1111, 80, payload_len=200, tcp_flags=TCP_PSH | TCP_ACK, ip_ttl=64, tcp_window=1000),
            pkt(2.0, "10.0.0.2", "10.0.0.1", 80, 1111, payload_len=50, tcp_flags=TCP_ACK, ip_ttl=128, tcp_window=500),
            pkt(3.0, "10.0.0.1", "10.0.0.2", 1111, 80, payload_len=300, tcp_flags=TCP_PSH | TCP_ACK, ip_ttl=64, tcp_window=1000),
            pkt(6.0, "10.0.0.2", "10.0.0.1", 80, 1111, payload_len=50, tcp_flags=TCP_ACK, ip_ttl=128, tcp_window=500),
        ]
        self.row = assemble_flows(table(self.recs)).iloc[0]

    def test_counts_and_volumes(self):
        self.assertEqual(self.row.fwd_packets, 3)
        self.assertEqual(self.row.bwd_packets, 2)
        self.assertEqual(self.row.flow_duration, 6.0)
        # ip_len == 40 + payload
        self.assertEqual(self.row.fwd_bytes, (140 + 240 + 340))
        self.assertEqual(self.row.bwd_bytes, (90 + 90))
        self.assertEqual(self.row.total_bytes, 720 + 180)

    def test_rates_use_duration(self):
        self.assertAlmostEqual(self.row.flow_packets_per_s, 5 / 6.0, places=6)
        self.assertAlmostEqual(self.row.flow_bytes_per_s, 900 / 6.0, places=6)

    def test_ratios(self):
        self.assertAlmostEqual(self.row.packets_ratio, 3 / 5, places=6)
        self.assertAlmostEqual(self.row.bytes_ratio, 720 / 900, places=6)
        self.assertAlmostEqual(self.row.mean_packet_size, 900 / 5, places=6)

    def test_iat_statistics(self):
        # all packets: 0,1,2,3,6 -> diffs 1,1,1,3
        diffs = np.array([1.0, 1.0, 1.0, 3.0])
        self.assertAlmostEqual(self.row.iat_mean, diffs.mean(), places=6)
        self.assertAlmostEqual(self.row.iat_std, diffs.std(), places=6)
        self.assertAlmostEqual(self.row.iat_max, 3.0, places=6)
        # forward only: 0,1,3 -> diffs 1,2
        fwd = np.array([1.0, 2.0])
        self.assertAlmostEqual(self.row.fwd_iat_mean, fwd.mean(), places=6)
        # backward only: 2,6 -> diff 4
        self.assertAlmostEqual(self.row.bwd_iat_mean, 4.0, places=6)

    def test_flag_counts_and_ratios(self):
        self.assertEqual(self.row.syn_count, 1)
        self.assertEqual(self.row.ack_count, 4)
        self.assertEqual(self.row.psh_count, 2)
        self.assertAlmostEqual(self.row.syn_ratio, 1 / 5, places=6)

    def test_ttl_and_window_statistics(self):
        ttl = np.array([64, 64, 128, 64, 128], dtype=float)
        self.assertAlmostEqual(self.row.ttl_mean, ttl.mean(), places=6)
        self.assertAlmostEqual(self.row.ttl_std, ttl.std(), places=6)
        self.assertEqual(self.row.ttl_unique, 2)
        self.assertEqual(self.row.window_min, 500)
        self.assertEqual(self.row.window_max, 1000)

    def test_payload_statistics(self):
        payload = np.array([100, 200, 50, 300, 50], dtype=float)
        self.assertAlmostEqual(self.row.payload_mean, payload.mean(), places=6)
        self.assertEqual(self.row.payload_max, 300)
        self.assertAlmostEqual(self.row.payload_p50, float(np.percentile(payload, 50)), places=6)
        self.assertEqual(self.row.payload_zero_ratio, 0.0)


class TestTcpAnomalies(unittest.TestCase):
    def test_retransmission_detected(self):
        recs = [
            pkt(0.0, "10.0.0.1", "10.0.0.2", 1111, 80, payload_len=100, tcp_seq=1000, tcp_flags=TCP_PSH | TCP_ACK),
            pkt(0.5, "10.0.0.1", "10.0.0.2", 1111, 80, payload_len=100, tcp_seq=1100, tcp_flags=TCP_PSH | TCP_ACK),
            pkt(1.0, "10.0.0.1", "10.0.0.2", 1111, 80, payload_len=100, tcp_seq=1100, tcp_flags=TCP_PSH | TCP_ACK),
        ]
        row = assemble_flows(table(recs)).iloc[0]
        self.assertEqual(row.retransmission_count, 1)
        self.assertAlmostEqual(row.retransmission_ratio, 1 / 3, places=6)

    def test_bare_acks_are_not_retransmissions(self):
        recs = [pkt(float(i) / 10, "10.0.0.1", "10.0.0.2", 1111, 80,
                    payload_len=0, tcp_seq=1000, tcp_flags=TCP_ACK) for i in range(5)]
        self.assertEqual(assemble_flows(table(recs)).iloc[0].retransmission_count, 0)

    def test_duplicate_acks_counted(self):
        recs = [
            pkt(0.0, "10.0.0.1", "10.0.0.2", 1111, 80, payload_len=100, tcp_seq=1, tcp_flags=TCP_PSH | TCP_ACK),
            pkt(0.1, "10.0.0.2", "10.0.0.1", 80, 1111, tcp_flags=TCP_ACK, tcp_ack=101),
            pkt(0.2, "10.0.0.2", "10.0.0.1", 80, 1111, tcp_flags=TCP_ACK, tcp_ack=101),
            pkt(0.3, "10.0.0.2", "10.0.0.1", 80, 1111, tcp_flags=TCP_ACK, tcp_ack=101),
        ]
        self.assertEqual(assemble_flows(table(recs)).iloc[0].dup_ack_count, 2)

    def test_out_of_order_counted(self):
        recs = [
            pkt(0.0, "10.0.0.1", "10.0.0.2", 1111, 80, payload_len=100, tcp_seq=2000, tcp_flags=TCP_PSH),
            pkt(0.1, "10.0.0.1", "10.0.0.2", 1111, 80, payload_len=100, tcp_seq=1000, tcp_flags=TCP_PSH),
        ]
        self.assertEqual(assemble_flows(table(recs)).iloc[0].out_of_order_count, 1)

    def test_fragments_counted(self):
        recs = [
            pkt(0.0, "10.0.0.1", "10.0.0.2", 1111, 80, payload_len=100, ip_flags=IP_FLAG_MF),
            pkt(0.1, "10.0.0.1", "10.0.0.2", 1111, 80, payload_len=100),
        ]
        row = assemble_flows(table(recs)).iloc[0]
        self.assertEqual(row.frag_packet_count, 1)
        self.assertAlmostEqual(row.mf_ratio, 0.5, places=6)


class TestFrameContract(unittest.TestCase):
    def test_empty_input_gives_empty_frame_with_full_schema(self):
        df = assemble_flows(PacketTable.empty(0))
        self.assertEqual(len(df), 0)
        for col in list(FLOW_FEATURES) + list(PACKET_FEATURES):
            self.assertIn(col, df.columns)

    def test_no_nan_or_inf_on_real_traffic(self):
        cap = generate_capture(seed=4, duration_s=300, n_campaigns=1, background_intensity=0.15)
        df = assemble_flows(PacketTable.from_records(cap.packets))
        self.assertGreater(len(df), 20)
        values = df[list(FLOW_FEATURES) + list(PACKET_FEATURES)].to_numpy(dtype=float)
        self.assertTrue(np.isfinite(values).all())

    def test_rows_are_time_sorted(self):
        cap = generate_capture(seed=4, duration_s=300, n_campaigns=1, background_intensity=0.15)
        df = assemble_flows(PacketTable.from_records(cap.packets))
        self.assertTrue((df.start_ts.diff().dropna() >= 0).all())

    def test_scan_flows_carry_the_scan_signature(self):
        cap = generate_capture(seed=3, duration_s=600, n_campaigns=2, background_intensity=0.1)
        recon = [iv for iv in cap.timeline if iv.stage == "Reconnaissance"]
        self.assertTrue(recon)
        iv = recon[0]
        df = assemble_flows(PacketTable.from_records(cap.packets))
        scan = df[(df.start_ts >= iv.start) & (df.end_ts <= iv.end) & (df.src_ip == iv.attacker)]
        self.assertGreater(len(scan), 30)
        self.assertEqual(scan.payload_zero_ratio.min(), 1.0)      # probes carry no data
        self.assertGreater(scan.dst_port_raw.nunique(), 20)       # spread across ports
        self.assertGreater(scan.syn_ratio.mean(), 0.3)


if __name__ == "__main__":
    unittest.main()
