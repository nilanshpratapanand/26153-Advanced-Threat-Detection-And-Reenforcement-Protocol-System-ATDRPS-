"""A pcap / pcapng reader and writer written against the file formats directly.

ATDRPS does not depend on ``scapy`` or ``pyshark``.  That is a deliberate
decision, not laziness: the tool is meant to be installable inside air-gapped
Critical Information Infrastructure, where every extra package is another thing
to vet, ship and patch.  The subset of the format we actually need -- Ethernet
(with VLAN tags), Linux cooked capture, raw IP, IPv4/IPv6, TCP/UDP/ICMP -- is
small enough to implement exactly.

Supported on read:

* classic ``pcap``, both endiannesses, microsecond and nanosecond resolution
* ``pcapng``: Section Header, Interface Description and Enhanced Packet blocks,
  honouring each interface's ``if_tsresol``
* link types 1 (Ethernet), 101/12/14 (raw IPv4/IPv6), 113 (Linux SLL), 276 (SLL2)

Supported on write: classic ``pcap`` with Ethernet framing, with correct IPv4
and TCP/UDP checksums so the output opens cleanly in Wireshark.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import BinaryIO, Iterable, Iterator

import numpy as np

from .schema import (
    IP_FLAG_DF,
    IP_FLAG_MF,
    PROTO_ICMP,
    PROTO_ICMPV6,
    PROTO_TCP,
    PROTO_UDP,
    PACKET_COLUMNS,
    PacketRecord,
    PacketTable,
    ip_to_int,
)

__all__ = ["read_pcap", "write_pcap", "iter_pcap_records", "PcapFormatError"]


class PcapFormatError(ValueError):
    """Raised when a file is not a capture we can read."""


# ---------------------------------------------------------------- constants
# The magic is stored in the file's own byte order: read the first four bytes
# with "<I" and with ">I"; whichever spelling yields one of these values tells
# you both the endianness and the timestamp resolution.
PCAP_MAGIC_US = 0xA1B2C3D4   # seconds + microseconds
PCAP_MAGIC_NS = 0xA1B23C4D   # seconds + nanoseconds
PCAPNG_SHB = 0x0A0D0D0A

LINKTYPE_ETHERNET = 1
LINKTYPE_RAW = 101
LINKTYPE_RAW_BSD = 12
LINKTYPE_RAW_OPENBSD = 14
LINKTYPE_LINUX_SLL = 113
LINKTYPE_LINUX_SLL2 = 276

ETHERTYPE_IPV4 = 0x0800
ETHERTYPE_IPV6 = 0x86DD
ETHERTYPE_VLAN = 0x8100
ETHERTYPE_QINQ = 0x88A8


# ================================================================== decoding
def _decode_ipv4(buf: memoryview, ts: float) -> PacketRecord | None:
    if len(buf) < 20:
        return None
    vihl = buf[0]
    if (vihl >> 4) != 4:
        return None
    ihl = (vihl & 0x0F) * 4
    if ihl < 20 or len(buf) < ihl:
        return None
    total_len, ident, flags_frag, ttl, proto = struct.unpack_from(">HHHBB", buf, 2)
    src = ".".join(str(b) for b in buf[12:16])
    dst = ".".join(str(b) for b in buf[16:20])
    ip_flags = (flags_frag >> 13) & 0x07
    frag_offset = flags_frag & 0x1FFF

    rec = PacketRecord(
        ts=ts, src_ip=src, dst_ip=dst, protocol=proto, ip_ttl=ttl,
        ip_id=ident, ip_flags=ip_flags, frag_offset=frag_offset,
        ip_len=total_len or len(buf), ihl=ihl, tcp_doff=0,
    )
    # A non-initial fragment carries no usable L4 header.
    if frag_offset != 0:
        rec.payload_len = max(0, (total_len or len(buf)) - ihl)
        return rec
    _decode_l4(buf[ihl:], rec, l4_len=max(0, (total_len or len(buf)) - ihl))
    return rec


def _decode_ipv6(buf: memoryview, ts: float) -> PacketRecord | None:
    if len(buf) < 40:
        return None
    if (buf[0] >> 4) != 6:
        return None
    payload_len, next_hdr, hop_limit = struct.unpack_from(">HBB", buf, 4)
    src = _fmt_v6(buf[8:24])
    dst = _fmt_v6(buf[24:40])
    rec = PacketRecord(
        ts=ts, src_ip=src, dst_ip=dst, protocol=next_hdr, ip_ttl=hop_limit,
        ip_id=0, ip_flags=0, frag_offset=0, ip_len=payload_len + 40, ihl=40,
        tcp_doff=0,
    )
    _decode_l4(buf[40:], rec, l4_len=payload_len)
    return rec


def _fmt_v6(raw: memoryview) -> str:
    groups = [f"{(raw[i] << 8) | raw[i + 1]:x}" for i in range(0, 16, 2)]
    return ":".join(groups)


def _decode_l4(buf: memoryview, rec: PacketRecord, l4_len: int) -> None:
    proto = rec.protocol
    if proto == PROTO_TCP and len(buf) >= 20:
        sport, dport, seq, ack = struct.unpack_from(">HHII", buf, 0)
        doff_flags = struct.unpack_from(">H", buf, 12)[0]
        doff = ((doff_flags >> 12) & 0x0F) * 4
        window = struct.unpack_from(">H", buf, 14)[0]
        rec.src_port, rec.dst_port = sport, dport
        rec.tcp_seq, rec.tcp_ack = seq, ack
        rec.tcp_flags = doff_flags & 0x01FF & 0xFF
        rec.tcp_window = window
        rec.tcp_doff = doff if doff >= 20 else 20
        rec.payload_len = max(0, l4_len - rec.tcp_doff)
    elif proto == PROTO_UDP and len(buf) >= 8:
        sport, dport, ulen, _ = struct.unpack_from(">HHHH", buf, 0)
        rec.src_port, rec.dst_port = sport, dport
        rec.tcp_doff = 8
        rec.payload_len = max(0, (ulen if ulen >= 8 else l4_len) - 8)
    elif proto in (PROTO_ICMP, PROTO_ICMPV6) and len(buf) >= 4:
        # ICMP has no ports; type/code are stashed there so scan/sweep
        # signatures stay visible to the feature extractor.
        rec.src_port = buf[0]
        rec.dst_port = buf[1]
        rec.tcp_doff = 8
        rec.payload_len = max(0, l4_len - 8)
    else:
        rec.payload_len = max(0, l4_len)


def _decode_link(linktype: int, data: bytes, ts: float) -> PacketRecord | None:
    buf = memoryview(data)
    if linktype == LINKTYPE_ETHERNET:
        if len(buf) < 14:
            return None
        ethertype = struct.unpack_from(">H", buf, 12)[0]
        off = 14
        # strip up to two stacked VLAN tags
        for _ in range(2):
            if ethertype in (ETHERTYPE_VLAN, ETHERTYPE_QINQ) and len(buf) >= off + 4:
                ethertype = struct.unpack_from(">H", buf, off + 2)[0]
                off += 4
            else:
                break
        payload = buf[off:]
    elif linktype == LINKTYPE_LINUX_SLL:
        if len(buf) < 16:
            return None
        ethertype = struct.unpack_from(">H", buf, 14)[0]
        payload = buf[16:]
    elif linktype == LINKTYPE_LINUX_SLL2:
        if len(buf) < 20:
            return None
        ethertype = struct.unpack_from(">H", buf, 0)[0]
        payload = buf[20:]
    elif linktype in (LINKTYPE_RAW, LINKTYPE_RAW_BSD, LINKTYPE_RAW_OPENBSD):
        if not len(buf):
            return None
        version = buf[0] >> 4
        ethertype = ETHERTYPE_IPV4 if version == 4 else ETHERTYPE_IPV6 if version == 6 else 0
        payload = buf
    else:
        raise PcapFormatError(f"unsupported link type {linktype}")

    if ethertype == ETHERTYPE_IPV4:
        rec = _decode_ipv4(payload, ts)
    elif ethertype == ETHERTYPE_IPV6:
        rec = _decode_ipv6(payload, ts)
    else:
        return None
    if rec is not None:
        rec.frame_len = len(data)
    return rec


# ==================================================================== readers
def _iter_classic(fh: BinaryIO, max_packets: int | None) -> Iterator[PacketRecord]:
    header = fh.read(24)
    if len(header) < 24:
        raise PcapFormatError("truncated pcap global header")
    as_le = struct.unpack("<I", header[:4])[0]
    as_be = struct.unpack(">I", header[:4])[0]
    if as_le in (PCAP_MAGIC_US, PCAP_MAGIC_NS):
        endian, ts_div = "<", (1e6 if as_le == PCAP_MAGIC_US else 1e9)
    elif as_be in (PCAP_MAGIC_US, PCAP_MAGIC_NS):
        endian, ts_div = ">", (1e6 if as_be == PCAP_MAGIC_US else 1e9)
    else:
        raise PcapFormatError(
            f"unrecognised pcap magic (0x{as_le:08x} little-endian / 0x{as_be:08x} big-endian)"
        )

    linktype = struct.unpack(endian + "I", header[20:24])[0]
    rec_hdr = struct.Struct(endian + "IIII")
    count = 0
    while True:
        raw = fh.read(16)
        if len(raw) < 16:
            return
        ts_sec, ts_frac, incl_len, _orig_len = rec_hdr.unpack(raw)
        data = fh.read(incl_len)
        if len(data) < incl_len:
            return
        rec = _decode_link(linktype, data, ts_sec + ts_frac / ts_div)
        if rec is not None:
            yield rec
            count += 1
            if max_packets is not None and count >= max_packets:
                return


def _iter_pcapng(fh: BinaryIO, max_packets: int | None) -> Iterator[PacketRecord]:
    fh.seek(0)
    endian = "<"
    interfaces: list[tuple[int, float]] = []  # (linktype, seconds-per-tick)
    count = 0
    while True:
        head = fh.read(8)
        if len(head) < 8:
            return
        block_type = struct.unpack(endian + "I", head[:4])[0]
        if block_type == PCAPNG_SHB:
            bom = fh.read(4)
            if len(bom) < 4:
                return
            endian = "<" if struct.unpack("<I", bom)[0] == 0x1A2B3C4D else ">"
            block_len = struct.unpack(endian + "I", head[4:8])[0]
            fh.seek(block_len - 12, 1)
            interfaces = []
            continue
        block_len = struct.unpack(endian + "I", head[4:8])[0]
        if block_len < 12:
            raise PcapFormatError(f"pcapng block length {block_len} is impossible")
        body = fh.read(block_len - 12)
        fh.read(4)  # trailing length

        if block_type == 0x00000001:  # Interface Description Block
            linktype = struct.unpack_from(endian + "H", body, 0)[0]
            interfaces.append((linktype, _sll_tsresol(body[8:], endian)))
        elif block_type == 0x00000006:  # Enhanced Packet Block
            iface_id, ts_hi, ts_lo, cap_len, _orig = struct.unpack_from(endian + "IIIII", body, 0)
            linktype, tick = interfaces[iface_id] if iface_id < len(interfaces) else (LINKTYPE_ETHERNET, 1e-6)
            ts = ((ts_hi << 32) | ts_lo) * tick
            rec = _decode_link(linktype, bytes(body[20:20 + cap_len]), ts)
            if rec is not None:
                yield rec
                count += 1
                if max_packets is not None and count >= max_packets:
                    return


def _sll_tsresol(options: bytes, endian: str) -> float:
    """Read ``if_tsresol`` (option code 9) from an Interface Description Block."""
    off = 0
    while off + 4 <= len(options):
        code, length = struct.unpack_from(endian + "HH", options, off)
        off += 4
        if code == 0:  # opt_endofopt
            break
        value = options[off:off + length]
        off += length + ((4 - length % 4) % 4)
        if code == 9 and value:
            raw = value[0]
            if raw & 0x80:
                return 2.0 ** -(raw & 0x7F)
            return 10.0 ** -raw
    return 1e-6


def iter_pcap_records(path: str | Path, max_packets: int | None = None) -> Iterator[PacketRecord]:
    """Stream :class:`PacketRecord` objects out of a capture file."""
    path = Path(path)
    with open(path, "rb") as fh:
        magic = fh.read(4)
        fh.seek(0)
        if len(magic) < 4:
            raise PcapFormatError(f"{path} is too short to be a capture")
        if struct.unpack("<I", magic)[0] == PCAPNG_SHB:
            yield from _iter_pcapng(fh, max_packets)
        else:
            yield from _iter_classic(fh, max_packets)


def read_pcap(path: str | Path, max_packets: int | None = None) -> PacketTable:
    """Read a whole capture into a columnar :class:`PacketTable`.

    Packets are returned in capture order, then stably sorted by timestamp --
    merged captures are not always monotonic and every downstream windowing
    assumption depends on time order.
    """
    records = list(iter_pcap_records(path, max_packets))
    table = PacketTable.from_records(records)
    return table.sort_by_time() if len(table) else table


# ==================================================================== writing
def _checksum(data: bytes) -> int:
    if len(data) % 2:
        data += b"\x00"
    total = 0
    for i in range(0, len(data), 2):
        total += (data[i] << 8) | data[i + 1]
    while total >> 16:
        total = (total & 0xFFFF) + (total >> 16)
    return (~total) & 0xFFFF


def _build_frame(rec: PacketRecord) -> bytes:
    """Serialise a :class:`PacketRecord` as an Ethernet/IPv4 frame."""
    payload = bytes(rec.payload_len)

    if rec.protocol == PROTO_TCP:
        doff = max(20, rec.tcp_doff)
        l4 = struct.pack(
            ">HHIIBBHHH",
            rec.src_port, rec.dst_port, rec.tcp_seq & 0xFFFFFFFF,
            rec.tcp_ack & 0xFFFFFFFF, (doff // 4) << 4, rec.tcp_flags & 0xFF,
            min(rec.tcp_window, 0xFFFF), 0, 0,
        ) + bytes(doff - 20)
    elif rec.protocol == PROTO_UDP:
        l4 = struct.pack(">HHHH", rec.src_port, rec.dst_port, 8 + len(payload), 0)
    elif rec.protocol in (PROTO_ICMP, PROTO_ICMPV6):
        l4 = struct.pack(">BBHHH", rec.src_port & 0xFF, rec.dst_port & 0xFF, 0, 0, 0)
    else:
        l4 = b""

    l4_total = l4 + payload
    src_b = bytes(int(p) for p in rec.src_ip.split("."))
    dst_b = bytes(int(p) for p in rec.dst_ip.split("."))
    total_len = 20 + len(l4_total)
    flags_frag = ((rec.ip_flags & 0x07) << 13) | (rec.frag_offset & 0x1FFF)

    ip_hdr = struct.pack(
        ">BBHHHBBH4s4s", 0x45, 0, total_len, rec.ip_id & 0xFFFF, flags_frag,
        rec.ip_ttl & 0xFF, rec.protocol & 0xFF, 0, src_b, dst_b,
    )
    ip_hdr = ip_hdr[:10] + struct.pack(">H", _checksum(ip_hdr)) + ip_hdr[12:]

    # L4 checksum over the pseudo-header, so Wireshark shows the frame as valid
    if rec.protocol in (PROTO_TCP, PROTO_UDP):
        pseudo = src_b + dst_b + struct.pack(">BBH", 0, rec.protocol, len(l4_total))
        csum = _checksum(pseudo + l4_total)
        pos = 16 if rec.protocol == PROTO_TCP else 6
        l4_total = l4_total[:pos] + struct.pack(">H", csum) + l4_total[pos + 2:]

    src_mac = b"\x02\x00" + src_b
    dst_mac = b"\x02\x00" + dst_b
    frame = dst_mac + src_mac + struct.pack(">H", ETHERTYPE_IPV4) + ip_hdr + l4_total
    return frame if len(frame) >= 60 else frame + bytes(60 - len(frame))


def write_pcap(path: str | Path, records: Iterable[PacketRecord], snaplen: int = 65535) -> int:
    """Write records to a classic pcap with Ethernet framing.  Returns the count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(path, "wb") as fh:
        fh.write(struct.pack("<IHHiIII", PCAP_MAGIC_US, 2, 4, 0, 0, snaplen, LINKTYPE_ETHERNET))
        for rec in records:
            frame = _build_frame(rec)
            ts_sec = int(rec.ts)
            ts_usec = int(round((rec.ts - ts_sec) * 1e6))
            if ts_usec >= 1_000_000:  # rounding can carry
                ts_sec += 1
                ts_usec -= 1_000_000
            fh.write(struct.pack("<IIII", ts_sec, ts_usec, len(frame), len(frame)))
            fh.write(frame)
            written += 1
    return written
