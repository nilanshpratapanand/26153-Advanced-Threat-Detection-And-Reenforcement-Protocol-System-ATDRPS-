"""Format-level tests for the in-tree pcap/pcapng reader.

The pcapng and big-endian cases are built byte by byte here rather than
checked in as binary fixtures, so the expectations are readable and the test
fails loudly if the reader drifts from the spec.
"""

import os
import struct
import tempfile
import unittest

from atdrps.data.pcap import PcapFormatError, read_pcap, write_pcap
from atdrps.data.schema import (
    PROTO_ICMP, PROTO_TCP, PROTO_UDP, TCP_ACK, TCP_PSH, TCP_SYN,
    PacketRecord,
)


def _sample_records():
    return [
        PacketRecord(ts=1_700_000_000.123456, src_ip="10.0.0.5", dst_ip="10.0.0.9",
                     src_port=44321, dst_port=80, protocol=PROTO_TCP,
                     tcp_flags=TCP_SYN, tcp_window=64240, ip_ttl=64, tcp_seq=1000),
        PacketRecord(ts=1_700_000_000.223456, src_ip="10.0.0.9", dst_ip="10.0.0.5",
                     src_port=80, dst_port=44321, protocol=PROTO_TCP,
                     tcp_flags=TCP_SYN | TCP_ACK, tcp_window=29200, ip_ttl=128,
                     tcp_seq=5000, tcp_ack=1001),
        PacketRecord(ts=1_700_000_000.323456, src_ip="10.0.0.5", dst_ip="10.0.0.9",
                     src_port=44321, dst_port=80, protocol=PROTO_TCP,
                     tcp_flags=TCP_PSH | TCP_ACK, tcp_window=64240,
                     tcp_seq=1001, tcp_ack=5001, payload_len=1460),
        PacketRecord(ts=1_700_000_001.0, src_ip="10.0.0.5", dst_ip="8.8.8.8",
                     src_port=53211, dst_port=53, protocol=PROTO_UDP,
                     payload_len=44, ip_ttl=64),
        PacketRecord(ts=1_700_000_002.0, src_ip="10.0.0.5", dst_ip="10.0.0.1",
                     protocol=PROTO_ICMP, src_port=8, dst_port=0,
                     payload_len=56, ip_ttl=64),
    ]


def _eth_ipv4_tcp_frame(src="10.0.0.1", dst="10.0.0.2", sport=1234, dport=80,
                        payload=b"", vlan=False, ttl=57):
    sb = bytes(int(x) for x in src.split("."))
    db = bytes(int(x) for x in dst.split("."))
    tcp = struct.pack(">HHIIBBHHH", sport, dport, 7, 9, 5 << 4, TCP_ACK, 512, 0, 0)
    total = 20 + len(tcp) + len(payload)
    ip = struct.pack(">BBHHHBBH4s4s", 0x45, 0, total, 1, 0x4000, ttl, PROTO_TCP, 0, sb, db)
    eth = b"\x02" * 6 + b"\x03" * 6
    if vlan:
        eth += struct.pack(">HHH", 0x8100, 0x0064, 0x0800)
    else:
        eth += struct.pack(">H", 0x0800)
    return eth + ip + tcp + payload


class TestClassicPcap(unittest.TestCase):
    def test_write_read_roundtrip(self):
        records = _sample_records()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "t.pcap")
            self.assertEqual(write_pcap(path, records), len(records))
            table = read_pcap(path)

        self.assertEqual(len(table), len(records))
        for original, parsed in zip(records, table.to_records()):
            self.assertAlmostEqual(original.ts, parsed.ts, places=5)
            self.assertEqual(original.src_ip, parsed.src_ip)
            self.assertEqual(original.dst_ip, parsed.dst_ip)
            self.assertEqual(original.src_port, parsed.src_port)
            self.assertEqual(original.dst_port, parsed.dst_port)
            self.assertEqual(original.protocol, parsed.protocol)
            self.assertEqual(original.ip_ttl, parsed.ip_ttl)
            self.assertEqual(original.tcp_flags, parsed.tcp_flags)
            self.assertEqual(original.payload_len, parsed.payload_len)
            self.assertEqual(original.tcp_window, parsed.tcp_window)
            self.assertEqual(original.tcp_seq, parsed.tcp_seq)

    def test_written_file_is_a_real_little_endian_pcap(self):
        """The magic must be stored in the file's own byte order.

        Getting this backwards still round-trips through our own reader -- it
        only shows up when Wireshark, tcpdump or CICFlowMeter opens the file.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "t.pcap")
            write_pcap(path, _sample_records()[:1])
            with open(path, "rb") as fh:
                head = fh.read(24)
        self.assertEqual(head[:4], bytes.fromhex("d4c3b2a1"))
        magic, major, minor, _zone, _sig, snaplen, linktype = struct.unpack("<IHHiIII", head)
        self.assertEqual(magic, 0xA1B2C3D4)
        self.assertEqual((major, minor), (2, 4))
        self.assertEqual(linktype, 1)          # Ethernet
        self.assertEqual(snaplen, 65535)

    def test_max_packets_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "t.pcap")
            write_pcap(path, _sample_records())
            self.assertEqual(len(read_pcap(path, max_packets=2)), 2)

    def test_output_is_time_sorted(self):
        records = _sample_records()
        records.reverse()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "t.pcap")
            write_pcap(path, records)
            ts = read_pcap(path).ts
        self.assertTrue(all(ts[i] <= ts[i + 1] for i in range(len(ts) - 1)))

    def test_big_endian_and_nanosecond_variants(self):
        frame = _eth_ipv4_tcp_frame(payload=b"hello")
        for magic, endian, div in ((0xA1B2C3D4, ">", 10**6), (0xA1B23C4D, ">", 10**9),
                                   (0xA1B2C3D4, "<", 10**6), (0xA1B23C4D, "<", 10**9)):
            with self.subTest(magic=hex(magic), endian=endian):
                header = struct.pack(endian + "IHHiIII", magic, 2, 4, 0, 0, 65535, 1)
                frac = int(0.5 * div)
                body = struct.pack(endian + "IIII", 1_700_000_000, frac, len(frame), len(frame)) + frame
                with tempfile.TemporaryDirectory() as tmp:
                    path = os.path.join(tmp, "t.pcap")
                    with open(path, "wb") as fh:
                        fh.write(header + body)
                    table = read_pcap(path)
                self.assertEqual(len(table), 1)
                rec = table.to_records()[0]
                self.assertAlmostEqual(rec.ts, 1_700_000_000.5, places=5)
                self.assertEqual(rec.payload_len, 5)

    def test_vlan_tag_is_stripped(self):
        frame = _eth_ipv4_tcp_frame(payload=b"abc", vlan=True)
        header = struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1)
        body = struct.pack("<IIII", 1, 0, len(frame), len(frame)) + frame
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "t.pcap")
            with open(path, "wb") as fh:
                fh.write(header + body)
            table = read_pcap(path)
        self.assertEqual(len(table), 1)
        self.assertEqual(table.to_records()[0].dst_port, 80)

    def test_truncated_trailing_record_is_ignored(self):
        frame = _eth_ipv4_tcp_frame()
        header = struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1)
        good = struct.pack("<IIII", 1, 0, len(frame), len(frame)) + frame
        truncated = struct.pack("<IIII", 2, 0, len(frame), len(frame)) + frame[:10]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "t.pcap")
            with open(path, "wb") as fh:
                fh.write(header + good + truncated)
            self.assertEqual(len(read_pcap(path)), 1)

    def test_bad_magic_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.pcap")
            with open(path, "wb") as fh:
                fh.write(b"NOTAPCAP" + bytes(40))
            with self.assertRaises(PcapFormatError):
                read_pcap(path)


class TestPcapng(unittest.TestCase):
    @staticmethod
    def _build(frames, tsresol=6):
        def block(btype, body):
            total = len(body) + 12
            return struct.pack("<II", btype, total) + body + struct.pack("<I", total)

        shb = block(0x0A0D0D0A,
                    struct.pack("<IHHq", 0x1A2B3C4D, 1, 0, -1))
        # idb: linktype 1, reserved, snaplen, then option 9 (if_tsresol)
        opts = struct.pack("<HH", 9, 1) + bytes([tsresol]) + bytes(3) + struct.pack("<HH", 0, 0)
        idb = block(0x00000001, struct.pack("<HHI", 1, 0, 65535) + opts)

        out = shb + idb
        for ts_ticks, frame in frames:
            pad = (4 - len(frame) % 4) % 4
            body = struct.pack("<IIIII", 0, ts_ticks >> 32, ts_ticks & 0xFFFFFFFF,
                               len(frame), len(frame)) + frame + bytes(pad)
            out += block(0x00000006, body)
        return out

    def test_reads_enhanced_packet_blocks(self):
        frame = _eth_ipv4_tcp_frame(src="192.168.5.5", dst="192.168.5.9",
                                    sport=5555, dport=443, payload=b"x" * 100, ttl=57)
        ticks = 1_700_000_000_500_000  # microseconds
        data = self._build([(ticks, frame), (ticks + 250_000, frame)])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "t.pcapng")
            with open(path, "wb") as fh:
                fh.write(data)
            table = read_pcap(path)

        self.assertEqual(len(table), 2)
        rec = table.to_records()[0]
        self.assertAlmostEqual(rec.ts, 1_700_000_000.5, places=4)
        self.assertEqual(rec.src_ip, "192.168.5.5")
        self.assertEqual(rec.dst_port, 443)
        self.assertEqual(rec.payload_len, 100)
        self.assertEqual(rec.ip_ttl, 57)

    def test_honours_if_tsresol_nanoseconds(self):
        frame = _eth_ipv4_tcp_frame()
        ticks = 1_700_000_000_500_000_000  # nanoseconds
        data = self._build([(ticks, frame)], tsresol=9)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "t.pcapng")
            with open(path, "wb") as fh:
                fh.write(data)
            rec = read_pcap(path).to_records()[0]
        self.assertAlmostEqual(rec.ts, 1_700_000_000.5, places=4)


if __name__ == "__main__":
    unittest.main()
