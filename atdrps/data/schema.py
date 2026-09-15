"""Canonical data schema for ATDRPS.

Everything downstream -- the pcap parser, the CSV adapters, the feature
extractors, the world model and the dashboard -- agrees on the structures
defined here.  Changing a feature name in one place and forgetting another is
the classic way these pipelines rot, so the *names* live here too, as ordered
tuples, and the extractors are tested against them.

Three levels of representation:

``PacketRecord`` / ``PacketTable``
    One row per packet.  ``PacketTable`` is the columnar (numpy) form used on
    the hot path; ``PacketRecord`` is the readable per-packet form used in
    tests and in the synthetic generator.

``FlowRecord`` / flow feature vectors
    One row per bidirectional 5-tuple conversation.

``state vector``
    One row per *time window* -- this is what the world model actually consumes.
    Assembled in :mod:`atdrps.data.windows`.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, fields
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "TCP_FIN", "TCP_SYN", "TCP_RST", "TCP_PSH", "TCP_ACK", "TCP_URG",
    "TCP_ECE", "TCP_CWR", "TCP_FLAG_NAMES", "flags_to_str",
    "PROTO_ICMP", "PROTO_TCP", "PROTO_UDP", "PROTO_NAMES",
    "IP_FLAG_DF", "IP_FLAG_MF",
    "PacketRecord", "PacketTable", "PACKET_COLUMNS",
    "FlowRecord", "FLOW_FEATURES", "PACKET_FEATURES",
    "STAGES", "STAGE_INDEX", "BENIGN_STAGE", "DEFAULT_INFILTRATION_STAGES",
    "ip_to_int", "int_to_ip", "V6_SPACE_BASE", "V6_SPACE_END", "stage_indices",
]

# --------------------------------------------------------------------- flags
TCP_FIN = 0x01
TCP_SYN = 0x02
TCP_RST = 0x04
TCP_PSH = 0x08
TCP_ACK = 0x10
TCP_URG = 0x20
TCP_ECE = 0x40
TCP_CWR = 0x80

TCP_FLAG_NAMES: tuple[tuple[int, str], ...] = (
    (TCP_FIN, "FIN"),
    (TCP_SYN, "SYN"),
    (TCP_RST, "RST"),
    (TCP_PSH, "PSH"),
    (TCP_ACK, "ACK"),
    (TCP_URG, "URG"),
    (TCP_ECE, "ECE"),
    (TCP_CWR, "CWR"),
)


def flags_to_str(flags: int) -> str:
    """``0x12`` -> ``'SYN|ACK'`` -- used in the flagged-flow table in the UI."""
    parts = [name for bit, name in TCP_FLAG_NAMES if flags & bit]
    return "|".join(parts) if parts else "-"


PROTO_ICMP = 1
PROTO_TCP = 6
PROTO_UDP = 17
PROTO_ICMPV6 = 58
PROTO_NAMES = {PROTO_ICMP: "ICMP", PROTO_TCP: "TCP", PROTO_UDP: "UDP", PROTO_ICMPV6: "ICMPv6"}

IP_FLAG_DF = 0x02  # don't fragment  (bit 1 of the 3-bit flags field)
IP_FLAG_MF = 0x01  # more fragments  (bit 0)


# ------------------------------------------------------------------ addresses
# Hashed IPv6 identities live in 240.0.0.0/12 -- inside IANA's "reserved for
# future use" block, and deliberately clear of 255.255.255.255 (DHCP broadcast)
# and of 224.0.0.0/4 multicast, both of which do appear in real captures.
V6_SPACE_BASE = 0xF0000000
V6_SPACE_MASK = 0x000FFFFF
V6_SPACE_END = V6_SPACE_BASE | V6_SPACE_MASK

def ip_to_int(addr: str) -> int:
    """Dotted-quad -> uint32.

    IPv6 addresses are folded into the 32-bit space with an FNV-1a hash placed
    inside ``240.0.0.0/12`` (IANA "reserved for future use", so it never appears
    as a real source or destination).  Packet-level features depend on
    address *identity*, not address arithmetic, so this is lossless for our
    purposes, and the original string is kept by the parser for display.
    """
    if ":" in addr:
        h = 0x811C9DC5
        for byte in addr.encode("utf-8"):
            h ^= byte
            h = (h * 0x01000193) & 0xFFFFFFFF
        return V6_SPACE_BASE | (h & V6_SPACE_MASK)
    try:
        parts = [int(p) for p in addr.split(".")]
    except ValueError as exc:
        raise ValueError(f"not an IPv4 address: {addr!r}") from exc
    if len(parts) != 4 or any(p < 0 or p > 255 for p in parts):
        raise ValueError(f"not an IPv4 address: {addr!r}")
    return struct.unpack(">I", bytes(parts))[0]


def int_to_ip(value: int) -> str:
    """uint32 -> dotted quad.

    Ids inside the reserved ``240.0.0.0/12`` block are hashed IPv6 addresses and
    render as ``v6:<hex>``; the readable address is recovered through
    :meth:`PacketTable.ip_str`, which consults the parser's name map first.
    """
    value = int(value) & 0xFFFFFFFF
    if V6_SPACE_BASE <= value <= V6_SPACE_END:
        return f"v6:{value & V6_SPACE_MASK:05x}"
    return ".".join(str(b) for b in struct.pack(">I", value))


# -------------------------------------------------------------------- packets
PACKET_COLUMNS: tuple[tuple[str, str], ...] = (
    ("ts", "f8"),             # capture timestamp, seconds since epoch
    ("src_ip", "u4"),
    ("dst_ip", "u4"),
    ("src_port", "u2"),
    ("dst_port", "u2"),
    ("protocol", "u1"),
    ("ip_ttl", "u1"),
    ("ip_id", "u2"),
    ("ip_flags", "u1"),       # DF / MF bits
    ("frag_offset", "u2"),    # in 8-byte units, as on the wire
    ("ip_len", "u4"),         # total length of the IP datagram
    ("ihl", "u1"),            # IP header length in bytes
    ("tcp_flags", "u1"),
    ("tcp_window", "u4"),
    ("tcp_seq", "u4"),
    ("tcp_ack", "u4"),
    ("tcp_doff", "u1"),       # TCP header length in bytes
    ("payload_len", "u4"),    # L4 payload bytes
    ("frame_len", "u4"),      # bytes actually on the wire
)


@dataclass(slots=True)
class PacketRecord:
    """A single parsed packet.  Ports are 0 for protocols that have none."""

    ts: float
    src_ip: str
    dst_ip: str
    src_port: int = 0
    dst_port: int = 0
    protocol: int = PROTO_TCP
    ip_ttl: int = 64
    ip_id: int = 0
    ip_flags: int = IP_FLAG_DF
    frag_offset: int = 0
    ip_len: int = 40
    ihl: int = 20
    tcp_flags: int = 0
    tcp_window: int = 0
    tcp_seq: int = 0
    tcp_ack: int = 0
    tcp_doff: int = 20
    payload_len: int = 0
    frame_len: int = 54

    def key(self) -> tuple:
        """Directional 5-tuple."""
        return (self.src_ip, self.dst_ip, self.src_port, self.dst_port, self.protocol)


class PacketTable:
    """Columnar packet store.

    Parsing straight into numpy arrays keeps a multi-million-packet capture
    tractable; a dataclass per packet would not be.
    """

    __slots__ = ("cols", "_n", "ip_names")

    def __init__(self, cols: dict[str, np.ndarray], ip_names: dict[int, str] | None = None):
        expected = {name for name, _ in PACKET_COLUMNS}
        missing = expected - set(cols)
        if missing:
            raise ValueError(f"PacketTable missing columns: {sorted(missing)}")
        lengths = {len(v) for v in cols.values()}
        if len(lengths) > 1:
            raise ValueError(f"PacketTable columns have differing lengths: {lengths}")
        self.cols = cols
        self._n = lengths.pop() if lengths else 0
        # uint32 id -> original textual address, for display only
        self.ip_names: dict[int, str] = ip_names or {}

    # ------------------------------------------------------------ constructors
    @classmethod
    def empty(cls, n: int = 0) -> "PacketTable":
        return cls({name: np.zeros(n, dtype=dt) for name, dt in PACKET_COLUMNS})

    @classmethod
    def from_records(cls, records: Iterable[PacketRecord]) -> "PacketTable":
        records = list(records)
        table = cls.empty(len(records))
        names: dict[int, str] = {}
        for i, rec in enumerate(records):
            for name, _ in PACKET_COLUMNS:
                if name in ("src_ip", "dst_ip"):
                    text = getattr(rec, name)
                    value = ip_to_int(text)
                    names[value] = text
                else:
                    value = getattr(rec, name)
                table.cols[name][i] = value
        table.ip_names = names
        return table

    # ---------------------------------------------------------------- protocol
    def __len__(self) -> int:
        return self._n

    def __getattr__(self, name: str) -> np.ndarray:
        cols = object.__getattribute__(self, "cols")
        if name in cols:
            return cols[name]
        raise AttributeError(name)

    def __getitem__(self, idx) -> "PacketTable":
        return PacketTable({k: v[idx] for k, v in self.cols.items()}, self.ip_names)

    def ip_str(self, value: int) -> str:
        return self.ip_names.get(int(value), int_to_ip(value))

    def sort_by_time(self) -> "PacketTable":
        order = np.argsort(self.cols["ts"], kind="stable")
        return self[order]

    def to_records(self) -> list[PacketRecord]:
        out: list[PacketRecord] = []
        field_names = [f.name for f in fields(PacketRecord)]
        for i in range(self._n):
            kwargs = {}
            for name in field_names:
                value = self.cols[name][i]
                if name in ("src_ip", "dst_ip"):
                    kwargs[name] = self.ip_str(value)
                elif name == "ts":
                    kwargs[name] = float(value)
                else:
                    kwargs[name] = int(value)
            out.append(PacketRecord(**kwargs))
        return out


# ---------------------------------------------------------------------- flows
@dataclass(slots=True)
class FlowRecord:
    """A bidirectional conversation, keyed on the canonical 5-tuple.

    ``fwd`` is the direction of the first packet seen.
    """

    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int
    start_ts: float
    end_ts: float
    fwd_packets: int = 0
    bwd_packets: int = 0
    fwd_bytes: int = 0
    bwd_bytes: int = 0
    fwd_flags: int = 0          # OR of every TCP flag seen forward
    bwd_flags: int = 0
    flag_counts: tuple = ()     # per-flag counts, order of TCP_FLAG_NAMES
    fwd_iat: tuple = ()         # (mean, std, min, max)
    bwd_iat: tuple = ()
    iat: tuple = ()
    payload_stats: tuple = ()   # (mean, std, min, max)
    ttl_stats: tuple = ()       # (mean, std, min, max)
    window_stats: tuple = ()    # (mean, std, min, max)
    retransmissions: int = 0
    dup_acks: int = 0
    frag_packets: int = 0
    label: str = "Benign"
    stage: str = "Benign"

    @property
    def duration(self) -> float:
        return max(0.0, self.end_ts - self.start_ts)

    @property
    def total_packets(self) -> int:
        return self.fwd_packets + self.bwd_packets

    @property
    def total_bytes(self) -> int:
        return self.fwd_bytes + self.bwd_bytes


# The canonical, ordered feature names.  Extractors emit exactly these, in this
# order; the dataset adapters map foreign column names onto them; the model's
# input dimension is derived from them.  Tested in tests/test_schema.py.
FLOW_FEATURES: tuple[str, ...] = (
    # --- volume
    "flow_duration",
    "fwd_packets", "bwd_packets", "total_packets",
    "fwd_bytes", "bwd_bytes", "total_bytes",
    # --- rates
    "flow_bytes_per_s", "flow_packets_per_s",
    "fwd_packets_per_s", "bwd_packets_per_s",
    # --- ratios
    "bytes_ratio", "packets_ratio", "mean_packet_size",
    "fwd_mean_packet_size", "bwd_mean_packet_size",
    # --- inter-arrival timing
    "iat_mean", "iat_std", "iat_min", "iat_max",
    "fwd_iat_mean", "fwd_iat_std", "fwd_iat_min", "fwd_iat_max",
    "bwd_iat_mean", "bwd_iat_std", "bwd_iat_min", "bwd_iat_max",
    # --- TCP flag counts
    "fin_count", "syn_count", "rst_count", "psh_count",
    "ack_count", "urg_count", "ece_count", "cwr_count",
    # --- flag ratios (the shape of the handshake, normalised for flow size)
    "syn_ratio", "rst_ratio", "ack_ratio", "psh_ratio",
    # --- derived endpoint facts
    "dst_port", "src_port", "protocol", "is_wellknown_dst_port",
)

# Packet-level statistics that are *not* derivable from NetFlow-style records.
# These are what let ATDRPS see slow reconnaissance that stays under every
# flow-level threshold.
PACKET_FEATURES: tuple[str, ...] = (
    "ttl_mean", "ttl_std", "ttl_min", "ttl_max", "ttl_unique",
    "window_mean", "window_std", "window_min", "window_max",
    "window_zero_count",
    "payload_mean", "payload_std", "payload_min", "payload_max",
    "payload_p50", "payload_p90", "payload_zero_ratio",
    "header_len_mean", "tcp_doff_mean",
    "retransmission_count", "retransmission_ratio",
    "dup_ack_count", "out_of_order_count",
    "frag_packet_count", "df_ratio", "mf_ratio",
)


# ------------------------------------------------------------- MITRE stages
STAGES: tuple[str, ...] = (
    "Benign",
    "Reconnaissance",
    "InitialAccess",
    "LateralMovement",
    "CommandAndControl",
    "Exfiltration",
)
STAGE_INDEX: dict[str, int] = {name: i for i, name in enumerate(STAGES)}
BENIGN_STAGE = "Benign"
DEFAULT_INFILTRATION_STAGES: tuple[str, ...] = (
    "InitialAccess",
    "LateralMovement",
    "CommandAndControl",
    "Exfiltration",
)


def stage_indices(names: Sequence[str]) -> np.ndarray:
    return np.array([STAGE_INDEX[n] for n in names], dtype=np.int64)
