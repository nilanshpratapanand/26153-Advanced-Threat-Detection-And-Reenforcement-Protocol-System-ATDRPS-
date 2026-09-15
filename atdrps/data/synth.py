"""Synthetic attack-chain traffic generator.

CIC-IDS2018 is ~220 GB of PCAP.  Developing against it directly means every
change costs a download and an hour of I/O, and it makes the pipeline
untestable in CI.  So ATDRPS ships a generator that produces *small, labelled,
deterministic* captures with the same structure the real data has: benign
background traffic, plus intrusion campaigns that walk the MITRE ATT&CK chain

    Reconnaissance -> Initial Access -> Lateral Movement -> Command & Control
    -> Exfiltration

Each campaign carries a ground-truth timeline, so every stage of the pipeline
(flow assembly, windowing, stage labelling, forecasting) can be checked against
what was actually injected.

Two properties matter and are deliberately engineered in:

*   **Not every campaign completes.**  Some stop after reconnaissance, some
    after initial access.  A model that simply memorises "scan => exfiltration"
    is wrong on this data, which is exactly the memorisation failure the
    problem statement warns about.
*   **Stage signatures overlap.**  Benign bulk transfer looks like exfiltration
    on volume alone; the difference is in the trajectory that preceded it.
    That is the whole thesis of the project, so the generator must not make it
    trivially separable per-flow.

This is development and test data.  Reported benchmark numbers come from
CIC-IDS2018, never from here.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .schema import (
    IP_FLAG_DF,
    IP_FLAG_MF,
    PROTO_ICMP,
    PROTO_TCP,
    PROTO_UDP,
    TCP_ACK,
    TCP_FIN,
    TCP_PSH,
    TCP_RST,
    TCP_SYN,
    PacketRecord,
)
from .pcap import write_pcap

__all__ = ["StageInterval", "SyntheticCapture", "generate_capture", "write_scenario"]

# Operating-system TTL fingerprints; the extractor's ttl_* features key off these.
TTL_LINUX = 64
TTL_WINDOWS = 128
TTL_NETDEV = 255

WELL_KNOWN = (80, 443, 53, 22, 25, 110, 143, 3389, 445, 3306, 8080)


@dataclass
class StageInterval:
    """Ground truth: one stage of one campaign, over a closed time interval."""

    start: float
    end: float
    stage: str
    attacker: str
    victim: str
    campaign: int
    note: str = ""


@dataclass
class SyntheticCapture:
    packets: list[PacketRecord] = field(default_factory=list)
    timeline: list[StageInterval] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def sorted_packets(self) -> list[PacketRecord]:
        return sorted(self.packets, key=lambda p: p.ts)

    def stage_at(self, ts: float) -> str:
        """Ground-truth stage covering ``ts``; later stages win on overlap."""
        best = "Benign"
        best_rank = -1
        order = {
            "Reconnaissance": 1, "InitialAccess": 2, "LateralMovement": 3,
            "CommandAndControl": 4, "Exfiltration": 5,
        }
        for iv in self.timeline:
            if iv.start <= ts <= iv.end:
                rank = order.get(iv.stage, 0)
                if rank > best_rank:
                    best, best_rank = iv.stage, rank
        return best


# ------------------------------------------------------------------ helpers
class _Gen:
    """Packet emitter with per-host TTL and sequence-number bookkeeping."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.packets: list[PacketRecord] = []
        self._ttl: dict[str, int] = {}
        self._hops: dict[str, int] = {}
        self._seq: dict[tuple, int] = {}

    def ttl_for(self, host: str) -> int:
        if host not in self._ttl:
            base = self.rng.choice([TTL_LINUX, TTL_WINDOWS, TTL_NETDEV], p=[0.6, 0.35, 0.05])
            self._ttl[host] = int(base)
        # A real source's TTL is stable: it is the initial value minus a fixed
        # hop count for that path. Jittering it per packet, as the first version
        # did, made every host look like it was being relayed and destroyed the
        # only header-level evidence of a machine-in-the-middle.
        if host not in self._hops:
            self._hops[host] = int(self.rng.integers(0, 4))
        return max(1, self._ttl[host] - self._hops[host])

    def next_seq(self, key: tuple, advance: int) -> int:
        cur = self._seq.get(key)
        if cur is None:
            cur = int(self.rng.integers(1_000, 4_000_000_000))
        self._seq[key] = (cur + advance) & 0xFFFFFFFF
        return cur

    def emit(
        self, ts: float, src: str, dst: str, sport: int, dport: int,
        *, proto: int = PROTO_TCP, flags: int = 0, payload: int = 0,
        window: int | None = None, ack: int = 0, retransmit: bool = False,
        fragment: bool = False,
    ) -> PacketRecord:
        key = (src, dst, sport, dport, proto)
        if retransmit:
            seq = self._seq.get(key, 1) - payload if payload else self._seq.get(key, 1)
            seq &= 0xFFFFFFFF
        else:
            seq = self.next_seq(key, max(payload, 1 if flags & (TCP_SYN | TCP_FIN) else 0))
        if window is None:
            window = int(self.rng.choice([64240, 65535, 29200, 8192, 5840, 1024]))
        header = 20 if proto == PROTO_TCP else 8 if proto == PROTO_UDP else 8
        rec = PacketRecord(
            ts=float(ts), src_ip=src, dst_ip=dst, src_port=int(sport), dst_port=int(dport),
            protocol=proto, ip_ttl=self.ttl_for(src),
            ip_id=int(self.rng.integers(0, 65536)),
            ip_flags=(IP_FLAG_MF if fragment else IP_FLAG_DF),
            frag_offset=0, ip_len=20 + header + payload, ihl=20,
            tcp_flags=flags if proto == PROTO_TCP else 0,
            tcp_window=window if proto == PROTO_TCP else 0,
            tcp_seq=seq if proto == PROTO_TCP else 0,
            tcp_ack=ack if proto == PROTO_TCP else 0,
            tcp_doff=20 if proto == PROTO_TCP else 8,
            payload_len=payload, frame_len=14 + 20 + header + payload,
        )
        self.packets.append(rec)
        return rec

    def handshake(self, ts: float, src: str, dst: str, sport: int, dport: int) -> float:
        """SYN / SYN-ACK / ACK.  Returns the time the connection is usable."""
        rtt = float(self.rng.uniform(0.0004, 0.06))
        self.emit(ts, src, dst, sport, dport, flags=TCP_SYN)
        self.emit(ts + rtt, dst, src, dport, sport, flags=TCP_SYN | TCP_ACK)
        self.emit(ts + 1.5 * rtt, src, dst, sport, dport, flags=TCP_ACK)
        return ts + 2 * rtt

    def teardown(self, ts: float, src: str, dst: str, sport: int, dport: int) -> None:
        rtt = float(self.rng.uniform(0.0004, 0.05))
        self.emit(ts, src, dst, sport, dport, flags=TCP_FIN | TCP_ACK)
        self.emit(ts + rtt, dst, src, dport, sport, flags=TCP_FIN | TCP_ACK)
        self.emit(ts + 2 * rtt, src, dst, sport, dport, flags=TCP_ACK)

    def ephemeral(self) -> int:
        return int(self.rng.integers(32768, 60999))


# ------------------------------------------------------------------ benign
def _benign_session(g: _Gen, ts: float, client: str, server: str) -> None:
    """A short HTTP(S)-shaped exchange: request up, content down."""
    sport, dport = g.ephemeral(), int(g.rng.choice([80, 443, 443, 8080]))
    t = g.handshake(ts, client, server, sport, dport)
    n_req = int(g.rng.integers(1, 6))
    for _ in range(n_req):
        req = int(g.rng.integers(180, 900))
        g.emit(t, client, server, sport, dport, flags=TCP_PSH | TCP_ACK, payload=req)
        t += float(g.rng.uniform(0.01, 0.35))
        remaining = int(g.rng.integers(800, 90_000))
        while remaining > 0:
            chunk = min(remaining, int(g.rng.integers(500, 1460)))
            g.emit(t, server, client, dport, sport, flags=TCP_PSH | TCP_ACK, payload=chunk)
            remaining -= chunk
            t += float(g.rng.uniform(0.0008, 0.02))
            # real TCP occasionally retransmits; the extractor must not treat
            # that alone as hostile
            if g.rng.random() < 0.01:
                g.emit(t, server, client, dport, sport, flags=TCP_PSH | TCP_ACK,
                       payload=chunk, retransmit=True)
                t += float(g.rng.uniform(0.2, 0.6))
        g.emit(t, client, server, sport, dport, flags=TCP_ACK)
        t += float(g.rng.uniform(0.05, 1.2))
    g.teardown(t, client, server, sport, dport)


def _benign_dns(g: _Gen, ts: float, client: str, resolver: str) -> None:
    sport = g.ephemeral()
    g.emit(ts, client, resolver, sport, 53, proto=PROTO_UDP, payload=int(g.rng.integers(28, 60)))
    g.emit(ts + float(g.rng.uniform(0.002, 0.09)), resolver, client, 53, sport,
           proto=PROTO_UDP, payload=int(g.rng.integers(60, 300)))


def _benign_backup(g: _Gen, ts: float, client: str, server: str) -> None:
    """Bulk internal transfer.  Looks like exfiltration on volume alone --
    on purpose, so volume cannot be the whole story."""
    sport = g.ephemeral()
    t = g.handshake(ts, client, server, sport, 445)
    remaining = int(g.rng.integers(500_000, 3_000_000))
    while remaining > 0:
        chunk = min(remaining, 1460)
        g.emit(t, client, server, sport, 445, flags=TCP_ACK, payload=chunk, window=65535)
        remaining -= chunk
        t += float(g.rng.uniform(0.0004, 0.0035))
        if g.rng.random() < 0.06:
            g.emit(t, server, client, 445, sport, flags=TCP_ACK, window=65535)
            t += 0.0002
    g.teardown(t, client, server, sport, 445)


def _background(g: _Gen, t0: float, t1: float, hosts: list[str], externals: list[str],
                resolver: str, intensity: float) -> None:
    span = t1 - t0
    n_sessions = max(1, int(span * intensity))
    for _ in range(n_sessions):
        ts = float(g.rng.uniform(t0, t1))
        _benign_session(g, ts, str(g.rng.choice(hosts)), str(g.rng.choice(externals)))
    for _ in range(max(1, int(span * intensity * 2.5))):
        _benign_dns(g, float(g.rng.uniform(t0, t1)), str(g.rng.choice(hosts)), resolver)
    for _ in range(max(0, int(span / 900))):
        _benign_backup(g, float(g.rng.uniform(t0, t1)),
                       str(g.rng.choice(hosts)), str(g.rng.choice(hosts)))


# ------------------------------------------------------------ attack stages
def _recon_scan(g: _Gen, t0: float, attacker: str, victim: str, slow: bool) -> float:
    """Port scan.

    ``slow`` spreads the same number of probes over minutes instead of seconds:
    per-flow it is indistinguishable from noise, and only the *timing structure
    across windows* gives it away.  That is the case the world model is for.
    """
    n_ports = int(g.rng.integers(60, 260))
    # fast scan spans seconds-to-a-minute; slow scan spans several minutes, so
    # both straddle window boundaries and are visible as a trajectory
    gap = float(g.rng.uniform(1.0, 4.0)) if slow else float(g.rng.uniform(0.15, 0.9))
    open_ports = set(g.rng.choice(WELL_KNOWN, size=3, replace=False).tolist())
    t = t0
    for _ in range(n_ports):
        dport = int(g.rng.integers(1, 65535))
        sport = g.ephemeral()
        # scanners use a small, constant window and send no payload
        g.emit(t, attacker, victim, sport, dport, flags=TCP_SYN, payload=0, window=1024)
        rtt = float(g.rng.uniform(0.0003, 0.01))
        if dport in open_ports:
            g.emit(t + rtt, victim, attacker, dport, sport, flags=TCP_SYN | TCP_ACK)
            g.emit(t + 2 * rtt, attacker, victim, sport, dport, flags=TCP_RST, window=0)
        else:
            g.emit(t + rtt, victim, attacker, dport, sport, flags=TCP_RST | TCP_ACK, window=0)
        t += gap
    return t


def _initial_access(g: _Gen, t0: float, attacker: str, victim: str) -> float:
    """Credential brute force against a remote-access service, ending in success."""
    service = int(g.rng.choice([22, 3389, 445]))
    attempts = int(g.rng.integers(25, 90))
    t = t0
    for i in range(attempts):
        sport = g.ephemeral()
        t = g.handshake(t, attacker, victim, sport, service)
        g.emit(t, attacker, victim, sport, service, flags=TCP_PSH | TCP_ACK,
               payload=int(g.rng.integers(60, 200)))
        t += float(g.rng.uniform(0.02, 0.2))
        g.emit(t, victim, attacker, service, sport, flags=TCP_PSH | TCP_ACK,
               payload=int(g.rng.integers(40, 90)))
        t += float(g.rng.uniform(0.01, 0.1))
        if i == attempts - 1:  # the one that works
            g.emit(t, victim, attacker, service, sport, flags=TCP_PSH | TCP_ACK,
                   payload=int(g.rng.integers(600, 2000)))
            t += 0.05
            g.teardown(t, attacker, victim, sport, service)
        else:
            g.emit(t, victim, attacker, service, sport, flags=TCP_RST | TCP_ACK, window=0)
        # paced to stay under lockout thresholds, so the campaign spans minutes
        # rather than seconds -- and therefore spans many state windows
        t += float(g.rng.uniform(1.0, 9.0))
    return t


def _lateral_movement(g: _Gen, t0: float, foothold: str, targets: list[str]) -> float:
    """Foothold reaches into peers over SMB/RDP/SSH, with real losses."""
    t = t0
    for target in targets:
        service = int(g.rng.choice([445, 3389, 22, 135]))
        sport = g.ephemeral()
        t = g.handshake(t, foothold, target, sport, service)
        for _ in range(int(g.rng.integers(6, 30))):
            payload = int(g.rng.integers(120, 1400))
            g.emit(t, foothold, target, sport, service, flags=TCP_PSH | TCP_ACK, payload=payload)
            t += float(g.rng.uniform(0.004, 0.09))
            if g.rng.random() < 0.12:  # loss -> retransmit + duplicate ACKs
                g.emit(t, foothold, target, sport, service, flags=TCP_PSH | TCP_ACK,
                       payload=payload, retransmit=True)
                t += float(g.rng.uniform(0.2, 0.5))
                ack_val = int(g.rng.integers(1, 4_000_000_000))
                for _ in range(3):
                    g.emit(t, target, foothold, service, sport, flags=TCP_ACK, ack=ack_val)
                    t += 0.001
            g.emit(t, target, foothold, service, sport, flags=TCP_PSH | TCP_ACK,
                   payload=int(g.rng.integers(60, 900)))
            t += float(g.rng.uniform(0.004, 0.06))
        g.teardown(t, foothold, target, sport, service)
        # operators dwell on a host before pivoting to the next
        t += float(g.rng.uniform(20.0, 120.0))
    return t


def _c2_beacon(g: _Gen, t0: float, victim: str, c2: str, duration: float) -> float:
    """Periodic beacon with jitter.

    The tell is not any single flow -- each is tiny and unremarkable -- it is
    the near-constant inter-arrival time across many windows.
    """
    interval = float(g.rng.choice([30.0, 45.0, 60.0, 90.0]))
    jitter = interval * float(g.rng.uniform(0.02, 0.12))
    port = int(g.rng.choice([443, 8080, 53, 8443]))
    t = t0
    while t < t0 + duration:
        sport = g.ephemeral()
        tt = g.handshake(t, victim, c2, sport, port)
        g.emit(tt, victim, c2, sport, port, flags=TCP_PSH | TCP_ACK,
               payload=int(g.rng.integers(120, 190)), window=8192)
        tt += float(g.rng.uniform(0.05, 0.3))
        if g.rng.random() < 0.25:  # a task comes back
            g.emit(tt, c2, victim, port, sport, flags=TCP_PSH | TCP_ACK,
                   payload=int(g.rng.integers(200, 1400)), window=8192)
            tt += 0.05
        else:
            g.emit(tt, c2, victim, port, sport, flags=TCP_ACK, window=8192)
        g.teardown(tt + 0.02, victim, c2, sport, port)
        t += max(1.0, interval + float(g.rng.normal(0, jitter)))
    return t


def _exfiltration(g: _Gen, t0: float, victim: str, sink: str) -> float:
    """Sustained, strongly asymmetric outbound transfer, in chunks."""
    port = int(g.rng.choice([443, 21, 22, 8080]))
    total = int(g.rng.integers(2_000_000, 10_000_000))
    t = t0
    sent = 0
    while sent < total:
        sport = g.ephemeral()
        t = g.handshake(t, victim, sink, sport, port)
        chunk_total = min(total - sent, int(g.rng.integers(400_000, 1_500_000)))
        chunk_sent = 0
        while chunk_sent < chunk_total:
            size = min(chunk_total - chunk_sent, 1460)
            frag = g.rng.random() < 0.02
            g.emit(t, victim, sink, sport, port, flags=TCP_PSH | TCP_ACK,
                   payload=size, window=65535, fragment=frag)
            chunk_sent += size
            sent += size
            # throttled: a competent operator paces exfiltration to stay under
            # volumetric alarms, which also makes it span several windows
            t += float(g.rng.uniform(0.006, 0.050))
            if g.rng.random() < 0.05:
                g.emit(t, sink, victim, port, sport, flags=TCP_ACK, window=65535)
                t += 0.0002
        g.teardown(t, victim, sink, sport, port)
        t += float(g.rng.uniform(5.0, 40.0))
    return t




# ------------------------------------------------- additional attack families
# These come from the team's attack taxonomy (docs/ATTACK_COVERAGE.md). Each one
# is generated because it has a distinct, network-visible signature that the
# five-stage chain alone does not produce -- and because a model that has never
# seen a SYN flood will happily call one "reconnaissance".

def _worm_propagation(g: _Gen, t0: float, patient_zero: str, targets: list[str]) -> float:
    """Self-propagation: one host, many hosts, ONE port.

    This is what separates a worm from a scan in aggregate. A scan is one source
    touching many ports on one host; a worm is one source touching many hosts on
    one port. Most attempts fail, a few succeed and the payload is pushed.
    """
    port = int(g.rng.choice([445, 139, 3389, 22]))
    t = t0
    infected = []
    for target in targets:
        sport = g.ephemeral()
        g.emit(t, patient_zero, target, sport, port, flags=TCP_SYN, window=8192)
        rtt = float(g.rng.uniform(0.0004, 0.02))
        if g.rng.random() < 0.35:                      # host is up and vulnerable
            g.emit(t + rtt, target, patient_zero, port, sport, flags=TCP_SYN | TCP_ACK)
            g.emit(t + 1.5 * rtt, patient_zero, target, sport, port, flags=TCP_ACK)
            tt = t + 2 * rtt
            remaining = int(g.rng.integers(180_000, 420_000))   # the worm body
            while remaining > 0:
                chunk = min(remaining, 1460)
                g.emit(tt, patient_zero, target, sport, port,
                       flags=TCP_PSH | TCP_ACK, payload=chunk, window=8192)
                remaining -= chunk
                tt += float(g.rng.uniform(0.0004, 0.004))
            g.teardown(tt, patient_zero, target, sport, port)
            infected.append(target)
        else:                                          # closed or patched
            g.emit(t + rtt, target, patient_zero, port, sport, flags=TCP_RST | TCP_ACK, window=0)
        t += float(g.rng.uniform(1.5, 7.0))

    # second generation: newly infected hosts start scanning too
    for host in infected[:3]:
        peers = [x for x in targets if x != host]
        for target in peers[: int(g.rng.integers(3, 8))]:
            sport = g.ephemeral()
            g.emit(t, host, target, sport, port, flags=TCP_SYN, window=8192)
            g.emit(t + 0.004, target, host, port, sport, flags=TCP_RST | TCP_ACK, window=0)
            t += float(g.rng.uniform(1.0, 5.0))
    return t


def _ransomware_encryption(g: _Gen, t0: float, host: str, shares: list[str]) -> float:
    """Mass SMB access followed by write-back of encrypted content.

    The network-level indicator is an SMB connection spike: one host touching
    many shares in a short window, then pushing back near-MTU writes with no
    compressibility. Read then write, share after share.
    """
    t = t0
    for share in shares:
        sport = g.ephemeral()
        t = g.handshake(t, host, share, sport, 445)
        n_files = int(g.rng.integers(180, 420))
        for _ in range(n_files):
            # read the original
            g.emit(t, host, share, sport, 445, flags=TCP_PSH | TCP_ACK,
                   payload=int(g.rng.integers(90, 200)), window=65535)
            t += float(g.rng.uniform(0.004, 0.03))
            size = int(g.rng.integers(400, 1460))
            g.emit(t, share, host, 445, sport, flags=TCP_PSH | TCP_ACK,
                   payload=size, window=65535)
            t += float(g.rng.uniform(0.004, 0.03))
            # write the encrypted replacement -- consistently near MTU
            g.emit(t, host, share, sport, 445, flags=TCP_PSH | TCP_ACK,
                   payload=1460, window=65535)
            t += float(g.rng.uniform(0.004, 0.03))
        g.teardown(t, host, share, sport, 445)
        t += float(g.rng.uniform(0.5, 4.0))
    return t


def _ddos_flood(g: _Gen, t0: float, victim: str, sources: list[str], duration: float) -> float:
    """Volumetric SYN flood: many sources, one destination, no handshake ever
    completes. Impact (TA0040), not a step toward infiltration."""
    port = int(g.rng.choice([80, 443, 53]))
    t = t0
    end = t0 + duration
    while t < end:
        src = str(g.rng.choice(sources))
        g.emit(t, src, victim, g.ephemeral(), port, flags=TCP_SYN, payload=0, window=512)
        if g.rng.random() < 0.25:                      # victim still answering
            g.emit(t + 0.001, victim, src, port, g.ephemeral(),
                   flags=TCP_SYN | TCP_ACK, window=0)
        t += float(g.rng.uniform(0.0015, 0.012))
    return t


def _dns_tunnel(g: _Gen, t0: float, victim: str, resolver: str, duration: float) -> float:
    """Exfiltration over DNS: a query-rate spike where responses dwarf queries.

    Encoded data rides in long subdomains and comes back in oversized TXT
    records -- so the query payload is large for DNS, and the response is larger
    still, which is the opposite of ordinary lookups.
    """
    t = t0
    end = t0 + duration
    while t < end:
        sport = g.ephemeral()
        g.emit(t, victim, resolver, sport, 53, proto=PROTO_UDP,
               payload=int(g.rng.integers(120, 250)))       # long encoded subdomain
        g.emit(t + float(g.rng.uniform(0.002, 0.03)), resolver, victim, 53, sport,
               proto=PROTO_UDP, payload=int(g.rng.integers(400, 900)))   # TXT reply
        t += float(g.rng.uniform(0.02, 0.18))
    return t


def _credential_stuffing(g: _Gen, t0: float, sources: list[str], victim: str) -> float:
    """Many sources, a handful of attempts each.

    Per-source rate limiting sees nothing here; only the window-level view --
    how many distinct sources converged on one service -- shows it.
    """
    service = int(g.rng.choice([443, 80, 22]))
    t = t0
    for src in sources:
        for _ in range(int(g.rng.integers(1, 4))):
            sport = g.ephemeral()
            t = g.handshake(t, src, victim, sport, service)
            g.emit(t, src, victim, sport, service, flags=TCP_PSH | TCP_ACK,
                   payload=int(g.rng.integers(180, 420)))
            t += float(g.rng.uniform(0.05, 0.4))
            g.emit(t, victim, src, service, sport, flags=TCP_PSH | TCP_ACK,
                   payload=int(g.rng.integers(120, 300)))
            t += float(g.rng.uniform(0.02, 0.2))
            g.teardown(t, src, victim, sport, service)
            t += float(g.rng.uniform(0.3, 2.5))
    return t


def _web_attack(g: _Gen, t0: float, attacker: str, victim: str) -> float:
    """Injection probing: repeated requests to one endpoint with unusual request
    sizes and a raised rate of short, uniform error replies.

    The payload itself is invisible without deep inspection; the shape is not.
    """
    service = int(g.rng.choice([80, 443, 8080]))
    t = t0
    for _ in range(int(g.rng.integers(60, 200))):
        sport = g.ephemeral()
        t = g.handshake(t, attacker, victim, sport, service)
        g.emit(t, attacker, victim, sport, service, flags=TCP_PSH | TCP_ACK,
               payload=int(g.rng.integers(400, 2400)))     # long injected query string
        t += float(g.rng.uniform(0.01, 0.12))
        if g.rng.random() < 0.8:                            # rejected: short, uniform
            g.emit(t, victim, attacker, service, sport, flags=TCP_PSH | TCP_ACK, payload=180)
        else:
            g.emit(t, victim, attacker, service, sport, flags=TCP_PSH | TCP_ACK,
                   payload=int(g.rng.integers(1200, 9000)))
        t += float(g.rng.uniform(0.01, 0.1))
        g.teardown(t, attacker, victim, sport, service)
        t += float(g.rng.uniform(0.05, 0.9))
    return t


def _mitm_injection(g: _Gen, t0: float, victim: str, peer: str, duration: float) -> float:
    """The ARP spoofing itself is below IP, so it is not in the capture. Its
    consequences are: the victim's packets arrive with inconsistent TTLs because
    some are relayed, and the attacker injects RSTs into live conversations."""
    t = t0
    end = t0 + duration
    while t < end:
        sport = g.ephemeral()
        t = g.handshake(t, victim, peer, sport, 80)
        for _ in range(int(g.rng.integers(6, 20))):
            rec = g.emit(t, victim, peer, sport, 80, flags=TCP_PSH | TCP_ACK,
                         payload=int(g.rng.integers(200, 1200)))
            if g.rng.random() < 0.5:
                # relayed through the attacker: one hop fewer than the rest
                rec.ip_ttl = max(1, rec.ip_ttl - int(g.rng.integers(2, 6)))
            t += float(g.rng.uniform(0.01, 0.09))
            g.emit(t, peer, victim, 80, sport, flags=TCP_PSH | TCP_ACK,
                   payload=int(g.rng.integers(200, 1400)))
            t += float(g.rng.uniform(0.01, 0.06))
        if g.rng.random() < 0.6:                            # injected reset
            g.emit(t, peer, victim, 80, sport, flags=TCP_RST | TCP_ACK, window=0)
        else:
            g.teardown(t, victim, peer, sport, 80)
        t += float(g.rng.uniform(1.0, 8.0))
    return t


# --------------------------------------------------------------- generator
def generate_capture(
    seed: int = 1337,
    duration_s: float = 3600.0,
    n_campaigns: int = 4,
    start_ts: float = 1_767_225_600.0,   # 2026-01-01T00:00:00Z
    n_internal: int = 12,
    background_intensity: float = 0.35,
    continue_probs: tuple[float, float, float, float] = (0.80, 0.80, 0.75, 0.75),
    n_incidents: int = 3,
    ransomware_prob: float = 0.35,
) -> SyntheticCapture:
    """Build a labelled capture.

    ``n_incidents`` injects standalone attacks that are not part of a
    five-stage campaign -- worm outbreaks, denial of service, DNS tunnelling,
    credential stuffing, web probing, machine-in-the-middle. They come from the
    team's attack taxonomy (``docs/ATTACK_COVERAGE.md``) and exist so the model
    is trained on traffic that contains them: a model that has never seen a SYN
    flood will cheerfully classify one as reconnaissance.

    ``ransomware_prob`` is the chance a fully-developed campaign ends in mass
    SMB encryption (Impact) rather than stopping at exfiltration.

    ``continue_probs`` is the chance a campaign advances past each stage
    (recon->access, access->lateral, lateral->C2, C2->exfil).  The defaults give
    a full chain roughly 36% of the time, so "saw a scan" is a genuinely weak
    predictor of "will exfiltrate" and the model has to learn the trajectory
    rather than the first symptom.
    """
    rng = np.random.default_rng(seed)
    g = _Gen(rng)

    internal = [f"10.20.{int(rng.integers(0, 4))}.{i + 10}" for i in range(n_internal)]
    externals = [f"93.184.{int(rng.integers(0, 255))}.{int(rng.integers(1, 254))}" for _ in range(8)]
    attackers = [f"185.{int(rng.integers(1, 254))}.{int(rng.integers(1, 254))}.{int(rng.integers(1, 254))}"
                 for _ in range(n_campaigns)]
    c2_hosts = [f"45.{int(rng.integers(1, 254))}.{int(rng.integers(1, 254))}.{int(rng.integers(1, 254))}"
                for _ in range(n_campaigns)]
    resolver = "10.20.0.2"

    t_end = start_ts + duration_s
    _background(g, start_ts, t_end, internal, externals, resolver, background_intensity)

    timeline: list[StageInterval] = []
    # Campaigns start throughout the capture, not only in the first third.
    # Clustering them early makes any chronological split degenerate: the tail
    # of every capture is then pure background, so a held-out final slice
    # contains almost no positives and every metric computed on it is noise.
    # A campaign that starts late simply runs out of capture, which is also
    # what happens in real collection.
    # A very short capture has no room for a 60-second lead-in; clamp rather
    # than let numpy raise on an inverted range.
    first_start = start_ts + min(60.0, duration_s * 0.05)
    last_start = max(first_start + 1e-3, start_ts + duration_s * 0.80)
    starts = np.sort(rng.uniform(first_start, last_start, size=n_campaigns))
    p_access, p_lateral, p_c2, p_exfil = continue_probs

    for c, camp_start in enumerate(starts):
        attacker = attackers[c]
        victim = str(rng.choice(internal))
        slow = bool(rng.random() < 0.4)

        t = float(camp_start)
        t_recon_end = _recon_scan(g, t, attacker, victim, slow=slow)
        timeline.append(StageInterval(t, t_recon_end, "Reconnaissance", attacker, victim, c,
                                      "slow scan" if slow else "fast port scan"))
        if rng.random() > p_access:
            continue  # scanned and moved on -- a real and common outcome

        t = t_recon_end + float(rng.uniform(5, 60))
        t_ia_end = _initial_access(g, t, attacker, victim)
        timeline.append(StageInterval(t, t_ia_end, "InitialAccess", attacker, victim, c,
                                      "credential brute force"))
        if rng.random() > p_lateral:
            continue

        peers = [h for h in internal if h != victim]
        targets = [str(x) for x in rng.choice(peers, size=int(rng.integers(3, 7)), replace=False)]
        t = t_ia_end + float(rng.uniform(10, 90))
        t_lm_end = _lateral_movement(g, t, victim, targets)
        timeline.append(StageInterval(t, t_lm_end, "LateralMovement", victim, ",".join(targets), c,
                                      "SMB/RDP/SSH spread"))
        if rng.random() > p_c2:
            continue

        t = t_lm_end + float(rng.uniform(5, 45))
        beacon_for = min(float(rng.uniform(300, 900)), max(60.0, t_end - t - 120))
        t_c2_end = _c2_beacon(g, t, victim, c2_hosts[c], beacon_for)
        timeline.append(StageInterval(t, t_c2_end, "CommandAndControl", c2_hosts[c], victim, c,
                                      "jittered beacon"))
        if rng.random() > p_exfil or t_c2_end > t_end - 60:
            continue

        t = t_c2_end + float(rng.uniform(2, 30))
        t_ex_end = _exfiltration(g, t, victim, c2_hosts[c])
        timeline.append(StageInterval(t, t_ex_end, "Exfiltration", c2_hosts[c], victim, c,
                                      "bulk outbound transfer"))

        # double extortion: steal first, then encrypt
        if rng.random() < ransomware_prob and t_ex_end < t_end - 120:
            t = t_ex_end + float(rng.uniform(5, 60))
            shares = [str(x) for x in rng.choice(
                [h for h in internal if h != victim],
                size=int(rng.integers(2, 5)), replace=False)]
            t_rw_end = _ransomware_encryption(g, t, victim, shares)
            timeline.append(StageInterval(t, t_rw_end, "Impact", victim, ",".join(shares), c,
                                          "ransomware: mass SMB encryption"))

    # ---- standalone incidents, unrelated to the campaigns above
    botnet = [f"{int(rng.integers(1, 223))}.{int(rng.integers(0, 255))}."
              f"{int(rng.integers(0, 255))}.{int(rng.integers(1, 254))}"
              for _ in range(40)]
    catalogue = ["worm", "ddos", "dns_tunnel", "cred_stuffing", "web_attack", "mitm",
                 "ransomware"]
    for _ in range(max(0, n_incidents)):
        kind = str(rng.choice(catalogue))
        t = float(rng.uniform(start_ts + 30, start_ts + duration_s * 0.9))
        victim = str(rng.choice(internal))
        peers = [h for h in internal if h != victim]

        if kind == "worm":
            targets = [str(x) for x in rng.choice(
                peers, size=min(len(peers), int(rng.integers(8, 12))), replace=False)]
            end = _worm_propagation(g, t, victim, targets)
            timeline.append(StageInterval(t, end, "LateralMovement", victim,
                                          ",".join(targets), -1, "worm propagation"))
        elif kind == "ddos":
            end = _ddos_flood(g, t, victim, botnet, float(rng.uniform(45, 150)))
            timeline.append(StageInterval(t, end, "Impact", "botnet", victim, -1,
                                          "volumetric SYN flood"))
        elif kind == "dns_tunnel":
            end = _dns_tunnel(g, t, victim, resolver, float(rng.uniform(120, 420)))
            timeline.append(StageInterval(t, end, "Exfiltration", victim, resolver, -1,
                                          "DNS tunnelling"))
        elif kind == "cred_stuffing":
            sources = [str(x) for x in rng.choice(botnet, size=int(rng.integers(12, 30)),
                                                  replace=False)]
            end = _credential_stuffing(g, t, sources, victim)
            timeline.append(StageInterval(t, end, "InitialAccess", "distributed", victim, -1,
                                          "credential stuffing"))
        elif kind == "web_attack":
            attacker = str(rng.choice(botnet))
            end = _web_attack(g, t, attacker, victim)
            timeline.append(StageInterval(t, end, "InitialAccess", attacker, victim, -1,
                                          "web application probing"))
        elif kind == "ransomware":
            # not every ransomware incident is preceded by a campaign we
            # observed -- the intrusion may predate the capture entirely
            shares = [str(x) for x in rng.choice(
                peers, size=min(len(peers), int(rng.integers(2, 5))), replace=False)]
            end = _ransomware_encryption(g, t, victim, shares)
            timeline.append(StageInterval(t, end, "Impact", victim, ",".join(shares), -1,
                                          "ransomware: mass SMB encryption"))
        else:  # mitm
            peer = str(rng.choice(peers))
            end = _mitm_injection(g, t, victim, peer, float(rng.uniform(90, 300)))
            timeline.append(StageInterval(t, end, "Reconnaissance", "on-path", victim, -1,
                                          "machine-in-the-middle"))

    timeline.sort(key=lambda iv: iv.start)

    # Collection stops when the capture window ends, so anything a campaign
    # would have done afterwards simply is not in the file -- and a stage that
    # was still running is truncated, not recorded in full. Without this a
    # capture asked for as 60 seconds could span twenty minutes, and
    # duration_s would mean nothing.
    packets = sorted((p for p in g.packets if start_ts <= p.ts <= t_end),
                     key=lambda p: p.ts)
    clipped: list[StageInterval] = []
    for iv in timeline:
        if iv.start > t_end:
            continue
        clipped.append(StageInterval(
            start=iv.start, end=min(iv.end, t_end), stage=iv.stage,
            attacker=iv.attacker, victim=iv.victim, campaign=iv.campaign,
            note=iv.note + (" (truncated by end of capture)" if iv.end > t_end else ""),
        ))
    timeline = clipped
    meta = {
        "seed": seed,
        "start_ts": start_ts,
        "duration_s": duration_s,
        "n_packets": len(packets),
        "n_campaigns": n_campaigns,
        "continue_probs": list(continue_probs),
        "n_incidents": n_incidents,
        "internal_hosts": internal,
        "external_hosts": externals,
        "attackers": attackers,
        "c2_hosts": c2_hosts,
        "resolver": resolver,
        "stage_counts": {
            s: sum(1 for iv in timeline if iv.stage == s)
            for s in ("Reconnaissance", "InitialAccess", "LateralMovement",
                      "CommandAndControl", "Exfiltration")
        },
    }
    return SyntheticCapture(packets=packets, timeline=timeline, meta=meta)


def write_scenario(outdir: str | Path, capture: SyntheticCapture, name: str = "capture") -> dict:
    """Write ``<name>.pcap`` and ``<name>.timeline.json`` into ``outdir``."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pcap_path = outdir / f"{name}.pcap"
    json_path = outdir / f"{name}.timeline.json"
    n = write_pcap(pcap_path, capture.packets)
    payload = {
        "meta": capture.meta,
        "timeline": [asdict(iv) for iv in capture.timeline],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"pcap": str(pcap_path), "timeline": str(json_path), "packets": n}
