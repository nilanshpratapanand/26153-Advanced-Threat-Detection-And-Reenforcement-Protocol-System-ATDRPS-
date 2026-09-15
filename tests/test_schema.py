import unittest

import numpy as np

from atdrps.data.schema import (
    FLOW_FEATURES, PACKET_FEATURES, STAGES, STAGE_INDEX,
    TCP_ACK, TCP_SYN, V6_SPACE_BASE, V6_SPACE_END,
    PacketRecord, PacketTable, flags_to_str, int_to_ip, ip_to_int,
)


class TestAddresses(unittest.TestCase):
    def test_ipv4_roundtrip_including_reserved_and_broadcast(self):
        # 255.255.255.255 (DHCP) and 224.0.0.0/4 (multicast) really do appear in
        # captures, so they must survive the IPv6 hashing scheme untouched.
        for addr in ("0.0.0.0", "10.0.0.5", "192.168.1.10", "172.16.0.1",
                     "8.8.8.8", "224.0.0.251", "239.255.255.250",
                     "255.255.255.255", "127.0.0.1"):
            with self.subTest(addr=addr):
                self.assertEqual(int_to_ip(ip_to_int(addr)), addr)

    def test_ipv6_lands_in_reserved_block_and_is_stable(self):
        v = ip_to_int("fe80::1")
        self.assertTrue(V6_SPACE_BASE <= v <= V6_SPACE_END)
        self.assertEqual(v, ip_to_int("fe80::1"))
        self.assertTrue(int_to_ip(v).startswith("v6:"))

    def test_ipv6_ids_are_distinct(self):
        ids = {ip_to_int(f"2001:db8::{i:x}") for i in range(4000)}
        # 20 bits of space; a handful of collisions is acceptable, a pile is not
        self.assertGreater(len(ids), 3980)

    def test_invalid_address_rejected(self):
        for bad in ("not.an.ip", "10.0.0", "300.1.1.1", ""):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                ip_to_int(bad)


class TestFlags(unittest.TestCase):
    def test_flag_rendering(self):
        self.assertEqual(flags_to_str(TCP_SYN | TCP_ACK), "SYN|ACK")
        self.assertEqual(flags_to_str(0), "-")
        self.assertEqual(flags_to_str(TCP_SYN), "SYN")


class TestPacketTable(unittest.TestCase):
    def _records(self):
        return [
            PacketRecord(ts=3.0, src_ip="10.0.0.1", dst_ip="10.0.0.2", src_port=1, dst_port=80),
            PacketRecord(ts=1.0, src_ip="10.0.0.3", dst_ip="10.0.0.2", src_port=2, dst_port=443),
            PacketRecord(ts=2.0, src_ip="10.0.0.1", dst_ip="10.0.0.9", src_port=3, dst_port=22),
        ]

    def test_roundtrip_records(self):
        table = PacketTable.from_records(self._records())
        self.assertEqual(len(table), 3)
        back = table.to_records()
        self.assertEqual([r.src_ip for r in back], ["10.0.0.1", "10.0.0.3", "10.0.0.1"])
        self.assertEqual([r.dst_port for r in back], [80, 443, 22])

    def test_sort_by_time(self):
        table = PacketTable.from_records(self._records()).sort_by_time()
        self.assertTrue(np.all(np.diff(table.ts) >= 0))
        self.assertEqual(table.to_records()[0].src_ip, "10.0.0.3")

    def test_slicing_keeps_name_map(self):
        table = PacketTable.from_records(self._records())
        sub = table[np.array([0, 2])]
        self.assertEqual(len(sub), 2)
        self.assertEqual(sub.ip_str(sub.src_ip[0]), "10.0.0.1")

    def test_missing_column_rejected(self):
        with self.assertRaises(ValueError):
            PacketTable({"ts": np.zeros(3)})

    def test_empty_table(self):
        self.assertEqual(len(PacketTable.empty(0)), 0)


class TestFeatureNames(unittest.TestCase):
    def test_no_duplicates(self):
        self.assertEqual(len(FLOW_FEATURES), len(set(FLOW_FEATURES)))
        self.assertEqual(len(PACKET_FEATURES), len(set(PACKET_FEATURES)))

    def test_flow_and_packet_namespaces_are_disjoint(self):
        self.assertEqual(set(FLOW_FEATURES) & set(PACKET_FEATURES), set())

    def test_stage_index_matches_order(self):
        self.assertEqual(STAGES[0], "Benign")
        for i, name in enumerate(STAGES):
            self.assertEqual(STAGE_INDEX[name], i)


if __name__ == "__main__":
    unittest.main()
