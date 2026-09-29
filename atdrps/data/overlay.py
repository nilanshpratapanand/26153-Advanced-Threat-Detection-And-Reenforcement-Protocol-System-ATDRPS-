"""Real background traffic + injected, labelled attack campaigns.

Why this exists (see docs/RESEARCH_AND_PLAN_V2.md): a detector trained on pure simulation
scored a benign laptop capture as an attack in most windows, because real traffic lies far
outside anything the simulator produces.  Public labelled datasets cannot be fetched in the
build sandbox, so the closest honest substitute is *semi-synthetic* data: keep a real capture
as the background (real protocols, real timing, real keepalives and idle stretches) and
inject attack campaigns into it, with exact ground truth.  False alarms are then measured on
real traffic, and onsets still have known labels.

What it adds over ``synth.generate_capture`` (each item is a documented weakness of v1):

* **Look-alikes that never escalate**, unlabelled and therefore benign: Internet background
  probes, a flaky client hammering a closed port, password-typo bursts, an internal
  vulnerability scan.  Without them "someone is scanning" is a giveaway.
* **Heavy-tailed dwell** between reconnaissance and access (log-normal, minutes to an hour)
  instead of a 5-60 s uniform gap.
* **Campaigns with no precursor** (``direct_access_prob``): an onset nothing forecasts.
* **A reconnaissance variant that imitates background probing** (``noise_like``), so
  recon alone is not decisive.

Caveats, deliberately visible: one background capture is one network and one vantage point;
attack traffic is still simulated; overlays built from the same background share it, so
statistics must be resampled by background segment, not by variant.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pcap import read_pcap
from .schema import PROTO_TCP, TCP_ACK, TCP_PSH, TCP_RST, TCP_SYN, PacketRecord
from .synth import (
    StageInterval, SyntheticCapture, WELL_KNOWN, _Gen, _c2_beacon, _exfiltration,
    _initial_access, _lateral_movement,
)

__all__ = ["Background", "load_background", "overlay_campaigns", "synthetic_background"]

_SERVICE_PORTS = (21, 22, 23, 25, 53, 80, 110, 135, 139, 143, 443, 445, 993, 995, 1433, 3306,
                  3389, 5432, 5900, 5985, 8080, 8443)


@dataclass
class Background:
    """Real (or benign-simulated) packets that the overlay leaves untouched."""

    records: list[PacketRecord]
    t0: float
    t1: float
    victim: str                 # the host attacks are aimed at / launched from
    peers: list[str]            # other internal hosts seen (lateral-movement targets)

    def segment(self, frac0: float, frac1: float) -> "Background":
        """A contiguous time slice -- used to keep train/validation/test disjoint in time."""
        a = self.t0 + (self.t1 - self.t0) * frac0
        b = self.t0 + (self.t1 - self.t0) * frac1
        recs = [r for r in self.records if a <= r.ts < b]
        return Background(recs, a, b, self.victim, list(self.peers))


def _is_private(ip: str) -> bool:
    if ip.startswith(("10.", "192.168.")):
        return True
    if ip.startswith("172."):
        try:
            return 16 <= int(ip.split(".")[1]) <= 31
        except ValueError:
            return False
    return False


def load_background(path, victim: str | None = None) -> Background:
    """Read a capture and pick the victim host (the busiest private address by default)."""
    recs = [r for r in read_pcap(path).to_records() if ":" not in r.src_ip]
    if not recs:
        raise ValueError(f"{path}: no IPv4 packets to use as a background")
    counts: dict[str, int] = {}
    for r in recs:
        for ip in (r.src_ip, r.dst_ip):
            if _is_private(ip):
                counts[ip] = counts.get(ip, 0) + 1
    if not counts:
        raise ValueError(f"{path}: no private addresses, cannot place a victim")
    ranked = sorted(counts, key=counts.get, reverse=True)
    victim = victim or ranked[0]
    peers = [ip for ip in ranked if ip != victim and not ip.endswith((".0", ".255"))][:12]
    if len(peers) < 3:                       # a laptop sees few peers; invent /24 neighbours
        base = ".".join(victim.split(".")[:3])
        peers += [f"{base}.{n}" for n in (2, 20, 31, 44, 57) if f"{base}.{n}" not in peers]
    ts = [r.ts for r in recs]
    return Background(recs, min(ts), max(ts), victim, peers[:12])


def synthetic_background(seed: int = 0, duration_s: float = 3000.0,
                         intensity: float = 0.2) -> Background:
    """A benign simulated background, for tests and for experiments without a real capture."""
    from .synth import generate_capture
    cap = generate_capture(seed=seed, duration_s=duration_s, n_campaigns=0, n_incidents=0,
                           background_intensity=intensity)
    ts = [p.ts for p in cap.packets]
    internal = sorted({p.src_ip for p in cap.packets if _is_private(p.src_ip)})
    return Background(list(cap.packets), min(ts), max(ts), internal[0], internal[1:13])


# ----------------------------------------------------------- reconnaissance
def _recon_targeted(g: _Gen, t0: float, attacker: str, victim: str) -> float:
    """A scan of the ports a person would try against one host: fast, service-shaped."""
    ports = g.rng.choice(_SERVICE_PORTS, size=int(g.rng.integers(8, len(_SERVICE_PORTS) + 1)), replace=False)
    open_ports = set(g.rng.choice(WELL_KNOWN, size=3, replace=False).tolist())
    t = t0
    for dport in ports:
        sport = g.ephemeral()
        g.emit(t, attacker, victim, sport, int(dport), flags=TCP_SYN, window=1024)
        rtt = float(g.rng.uniform(0.0004, 0.02))
        if int(dport) in open_ports:
            g.emit(t + rtt, victim, attacker, int(dport), sport, flags=TCP_SYN | TCP_ACK)
            g.emit(t + 2 * rtt, attacker, victim, sport, int(dport), flags=TCP_RST, window=0)
        else:
            g.emit(t + rtt, victim, attacker, int(dport), sport, flags=TCP_RST | TCP_ACK, window=0)
        t += float(g.rng.uniform(0.2, 1.5))
    return t


def _probe(g: _Gen, t: float, src: str, dst: str, dport: int, opened: bool = False) -> None:
    sport = g.ephemeral()
    g.emit(t, src, dst, sport, dport, flags=TCP_SYN, window=int(g.rng.choice([1024, 29200, 65535])))
    rtt = float(g.rng.uniform(0.0004, 0.03))
    if opened:
        g.emit(t + rtt, dst, src, dport, sport, flags=TCP_SYN | TCP_ACK)
        g.emit(t + 2 * rtt, src, dst, sport, dport, flags=TCP_RST, window=0)
    else:
        g.emit(t + rtt, dst, src, dport, sport, flags=TCP_RST | TCP_ACK, window=0)


def _recon_noise_like(g: _Gen, t0: float, attacker: str, victim: str, peers: list[str]) -> float:
    """Low-rate single-port sweep -- indistinguishable, per flow, from Internet background."""
    port = int(g.rng.choice([22, 23, 445, 3389, 80, 443]))
    targets = [victim] + list(g.rng.choice(peers, size=min(len(peers), int(g.rng.integers(1, 4))),
                                           replace=False))
    t = t0
    for _ in range(int(g.rng.integers(6, 20))):
        _probe(g, t, attacker, str(g.rng.choice(targets)), port, opened=port in (22, 80, 443) and g.rng.random() < 0.3)
        t += float(g.rng.uniform(4.0, 25.0))
    return t


# ------------------------------------------------------------------- decoys
def _decoy_internet_noise(g: _Gen, t0: float, ext: str, victim: str) -> None:
    port = int(g.rng.choice([22, 23, 445, 3389, 5900, 8080]))
    t = t0
    for _ in range(int(g.rng.integers(3, 14))):
        _probe(g, t, ext, victim, port)
        t += float(g.rng.uniform(3.0, 20.0))


def _decoy_flaky_client(g: _Gen, t0: float, client: str, server: str) -> None:
    port = int(g.rng.choice([443, 8443, 5432, 3306]))
    t = t0
    for _ in range(int(g.rng.integers(5, 20))):
        _probe(g, t, client, server, port)
        t += float(g.rng.uniform(1.0, 6.0))


def _decoy_typo_burst(g: _Gen, t0: float, src: str, dst: str) -> None:
    t = t0
    for _ in range(int(g.rng.integers(3, 8))):
        sport = g.ephemeral()
        t = g.handshake(t, src, dst, sport, 22)
        g.emit(t, src, dst, sport, 22, flags=TCP_PSH | TCP_ACK, payload=int(g.rng.integers(60, 140)))
        g.emit(t + 0.05, dst, src, 22, sport, flags=TCP_RST | TCP_ACK, window=0)
        t += float(g.rng.uniform(4.0, 30.0))


def _decoy_vuln_scan(g: _Gen, t0: float, scanner: str, victim: str) -> None:
    """An authorised internal scan: fast, wide -- and benign."""
    t = t0
    for dport in g.rng.choice(_SERVICE_PORTS, size=int(g.rng.integers(10, len(_SERVICE_PORTS) + 1)), replace=False):
        _probe(g, t, scanner, victim, int(dport), opened=g.rng.random() < 0.1)
        t += float(g.rng.uniform(0.05, 0.6))


# ----------------------------------------------------------------- overlay
def overlay_campaigns(bg: Background, seed: int, n_campaigns: int = 1, *,
                      direct_access_prob: float = 0.25, dwell_median_s: float = 240.0,
                      dwell_sigma: float = 1.0, decoys_per_hour: float = 10.0,
                      continue_probs=(0.70, 0.75, 0.70, 0.70),
                      recon_kinds=(("targeted", 0.35), ("slow", 0.25), ("noise_like", 0.40)),
                      ) -> SyntheticCapture:
    """Inject campaigns and look-alike decoys into ``bg``; return packets plus ground truth."""
    from .synth import _recon_scan
    rng = np.random.default_rng([seed, 7919])
    g = _Gen(rng)
    span = bg.t1 - bg.t0
    victim, peers = bg.victim, bg.peers
    pub = lambda: f"{int(rng.choice([45, 91, 103, 185, 193, 203]))}.{int(rng.integers(1, 254))}." \
                  f"{int(rng.integers(0, 254))}.{int(rng.integers(1, 254))}"
    timeline: list[StageInterval] = []
    p_access, p_lateral, p_c2, p_exfil = continue_probs
    kinds, weights = zip(*recon_kinds)
    weights = np.asarray(weights, dtype=float) / sum(weights)

    # decoys first: Poisson arrivals, none of them labelled
    n_decoys = int(rng.poisson(decoys_per_hour * span / 3600.0))
    kinds_decoy = ("noise", "flaky", "typo", "vuln")
    for _ in range(n_decoys):
        t = float(rng.uniform(bg.t0, bg.t1 - 30))
        kind = str(rng.choice(kinds_decoy, p=[0.45, 0.2, 0.2, 0.15]))
        if kind == "noise":
            _decoy_internet_noise(g, t, pub(), victim)
        elif kind == "flaky":
            _decoy_flaky_client(g, t, victim, str(rng.choice(peers)))
        elif kind == "typo":
            _decoy_typo_burst(g, t, str(rng.choice(peers + [pub()])), victim)
        else:
            _decoy_vuln_scan(g, t, str(rng.choice(peers)), victim)

    lo, hi = bg.t0 + 0.08 * span, bg.t0 + 0.80 * span
    starts = np.sort(rng.uniform(lo, max(lo + 1, hi), size=n_campaigns))
    for c, t_start in enumerate(starts):
        attacker, c2 = pub(), pub()
        t = float(t_start)
        direct = bool(rng.random() < direct_access_prob)
        if not direct:
            kind = str(rng.choice(kinds, p=weights))
            if kind == "targeted":
                t_r = _recon_targeted(g, t, attacker, victim); note = "targeted service scan"
            elif kind == "slow":
                t_r = _recon_scan(g, t, attacker, victim, slow=True); note = "slow scan"
            else:
                t_r = _recon_noise_like(g, t, attacker, victim, peers); note = "noise-like probing"
            timeline.append(StageInterval(t, t_r, "Reconnaissance", attacker, victim, c, note))
            if rng.random() > p_access:
                continue                                    # scanned, then went away
            dwell = float(np.clip(rng.lognormal(np.log(dwell_median_s), dwell_sigma), 20.0, 3600.0))
            t = t_r + dwell
        if t > bg.t1 - 120:
            continue                                        # ran out of capture
        t_ia = _initial_access(g, t, attacker, victim)
        timeline.append(StageInterval(t, t_ia, "InitialAccess", attacker, victim, c,
                                      "direct credential attack" if direct else "credential brute force"))
        if rng.random() > p_lateral:
            continue
        targets = [str(x) for x in rng.choice(peers, size=min(len(peers), int(rng.integers(2, 5))),
                                              replace=False)]
        t = t_ia + float(rng.lognormal(np.log(40.0), 0.8))
        t_lm = _lateral_movement(g, t, victim, targets)
        timeline.append(StageInterval(t, t_lm, "LateralMovement", victim, ",".join(targets), c,
                                      "SMB/RDP/SSH spread"))
        if rng.random() > p_c2:
            continue
        t = t_lm + float(rng.uniform(5, 45))
        t_c2 = _c2_beacon(g, t, victim, c2, min(float(rng.uniform(300, 900)), max(60.0, bg.t1 - t - 90)))
        timeline.append(StageInterval(t, t_c2, "CommandAndControl", c2, victim, c, "jittered beacon"))
        if rng.random() > p_exfil or t_c2 > bg.t1 - 60:
            continue
        t = t_c2 + float(rng.uniform(2, 30))
        t_ex = _exfiltration(g, t, victim, c2)
        timeline.append(StageInterval(t, t_ex, "Exfiltration", c2, victim, c, "bulk outbound transfer"))

    timeline.sort(key=lambda iv: iv.start)
    injected = [p for p in g.packets if bg.t0 <= p.ts <= bg.t1]
    packets = sorted(bg.records + injected, key=lambda p: p.ts)
    clipped = [StageInterval(iv.start, min(iv.end, bg.t1), iv.stage, iv.attacker, iv.victim,
                             iv.campaign, iv.note + (" (truncated)" if iv.end > bg.t1 else ""))
               for iv in timeline if iv.start <= bg.t1]
    meta = {"seed": seed, "victim": victim, "n_background_packets": len(bg.records),
            "n_injected_packets": len(injected), "n_decoys": n_decoys, "n_campaigns": n_campaigns,
            "start_ts": bg.t0, "duration_s": span}
    return SyntheticCapture(packets=packets, timeline=clipped, meta=meta)
