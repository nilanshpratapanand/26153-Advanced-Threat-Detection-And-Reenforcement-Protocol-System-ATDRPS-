"""Flow assembly and feature extraction.

Packets go in, one row per bidirectional conversation comes out, carrying both
the NetFlow/IPFIX-style aggregates a CICFlowMeter record would have *and* the
packet-level statistics that only a full capture can give you.

Why both levels, in one row:

*   Flow aggregates catch volumetric behaviour -- a SYN flood is obvious in
    packets-per-second and flag ratios.
*   Packet statistics catch what flow records erase.  A slow port scan sits
    under every flow-level threshold; what gives it away is a constant TTL, a
    constant tiny TCP window, zero payload, and a sequential walk across
    destination ports.  None of that survives aggregation into a NetFlow
    record.

The problem statement asks for the combination explicitly, and this is where
the combination is formed.

Flow lifetime follows the usual rules: an idle timeout, a hard active timeout,
and -- for TCP -- teardown on FIN in both directions or on RST.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import (
    FLOW_FEATURES,
    IP_FLAG_DF,
    IP_FLAG_MF,
    PACKET_FEATURES,
    PROTO_TCP,
    TCP_ACK,
    TCP_CWR,
    TCP_ECE,
    TCP_FIN,
    TCP_PSH,
    TCP_RST,
    TCP_SYN,
    TCP_URG,
    PacketTable,
)

__all__ = ["assemble_flows", "FLOW_META_COLUMNS", "EPS"]

# CICFlowMeter divides by a microsecond floor rather than guarding each rate
# individually; we do the same so rate features stay comparable with published
# CIC-IDS2018 numbers.
EPS = 1e-6

FLOW_META_COLUMNS: tuple[str, ...] = (
    "flow_id", "src_ip", "dst_ip", "src_port_raw", "dst_port_raw",
    "protocol_raw", "start_ts", "end_ts",
)

WELL_KNOWN_PORTS = frozenset(
    (20, 21, 22, 23, 25, 53, 67, 68, 69, 80, 110, 123, 135, 137, 138, 139, 143,
     161, 389, 443, 445, 465, 514, 587, 631, 636, 993, 995, 1433, 1521, 3306,
     3389, 5432, 5900, 8080, 8443)
)

_FLAG_BITS = (
    ("fin_count", TCP_FIN), ("syn_count", TCP_SYN), ("rst_count", TCP_RST),
    ("psh_count", TCP_PSH), ("ack_count", TCP_ACK), ("urg_count", TCP_URG),
    ("ece_count", TCP_ECE), ("cwr_count", TCP_CWR),
)


# --------------------------------------------------------------------- keys
def _canonical_keys(table: PacketTable) -> np.ndarray:
    """One int64 id per bidirectional conversation.

    Both directions of a conversation must land on the same id, so the endpoint
    pair is sorted before hashing.  A 64-bit mix over (lo_ip, hi_ip, lo_port,
    hi_port, proto) is collision-safe at capture scale and vectorises, which a
    tuple-keyed dict would not.
    """
    src_ip = table.src_ip.astype(np.int64)
    dst_ip = table.dst_ip.astype(np.int64)
    src_port = table.src_port.astype(np.int64)
    dst_port = table.dst_port.astype(np.int64)
    proto = table.protocol.astype(np.int64)

    # order endpoints by (ip, port) so direction does not matter
    swap = (src_ip > dst_ip) | ((src_ip == dst_ip) & (src_port > dst_port))
    lo_ip = np.where(swap, dst_ip, src_ip)
    hi_ip = np.where(swap, src_ip, dst_ip)
    lo_port = np.where(swap, dst_port, src_port)
    hi_port = np.where(swap, src_port, dst_port)

    h = np.zeros(len(table), dtype=np.uint64)
    for part, shift in ((lo_ip, 0), (hi_ip, 17), (lo_port, 33), (hi_port, 45), (proto, 57)):
        h ^= (part.astype(np.uint64) << np.uint64(shift))
        # splitmix-style avalanche keeps low-entropy fields from clustering
        h ^= h >> np.uint64(30)
        h = (h * np.uint64(0xBF58476D1CE4E5B9)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    h ^= h >> np.uint64(31)
    return h.astype(np.int64)


def _stat4(values: np.ndarray) -> tuple[float, float, float, float]:
    """(mean, std, min, max) with empty input mapped to zeros."""
    if values.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    return (float(values.mean()), float(values.std()),
            float(values.min()), float(values.max()))


def _iat_stats(ts: np.ndarray) -> tuple[float, float, float, float]:
    """Inter-arrival statistics.

    A single-packet direction has no inter-arrival time at all.  Reporting
    zeros there is a deliberate convention (it matches CICFlowMeter) and is
    what makes ``iat_std == 0`` meaningful as a beacon signature: a beacon has
    *many* packets with near-constant spacing, not one packet.
    """
    if ts.size < 2:
        return 0.0, 0.0, 0.0, 0.0
    return _stat4(np.diff(ts))


# ---------------------------------------------------------------- extraction
def _episode_features(table: PacketTable, idx: np.ndarray, fwd_mask: np.ndarray) -> dict:
    ts = table.ts[idx]
    ip_len = table.ip_len[idx].astype(np.float64)
    payload = table.payload_len[idx].astype(np.float64)
    ttl = table.ip_ttl[idx].astype(np.float64)
    window = table.tcp_window[idx].astype(np.float64)
    flags = table.tcp_flags[idx].astype(np.int64)
    ihl = table.ihl[idx].astype(np.float64)
    doff = table.tcp_doff[idx].astype(np.float64)
    ip_flags = table.ip_flags[idx].astype(np.int64)
    frag_off = table.frag_offset[idx].astype(np.int64)
    proto = int(table.protocol[idx[0]])

    bwd_mask = ~fwd_mask
    duration = float(ts[-1] - ts[0])
    span = max(duration, EPS)

    n_fwd = int(fwd_mask.sum())
    n_bwd = int(bwd_mask.sum())
    n_tot = n_fwd + n_bwd
    b_fwd = float(ip_len[fwd_mask].sum())
    b_bwd = float(ip_len[bwd_mask].sum())
    b_tot = b_fwd + b_bwd

    feats: dict[str, float] = {
        "flow_duration": duration,
        "fwd_packets": float(n_fwd),
        "bwd_packets": float(n_bwd),
        "total_packets": float(n_tot),
        "fwd_bytes": b_fwd,
        "bwd_bytes": b_bwd,
        "total_bytes": b_tot,
        "flow_bytes_per_s": b_tot / span,
        "flow_packets_per_s": n_tot / span,
        "fwd_packets_per_s": n_fwd / span,
        "bwd_packets_per_s": n_bwd / span,
        # ratios are bounded in [0, 1] so a 40 MB exfil and a 4 kB beacon are
        # on the same scale; asymmetry is the signal, not magnitude
        "bytes_ratio": b_fwd / b_tot if b_tot else 0.0,
        "packets_ratio": n_fwd / n_tot if n_tot else 0.0,
        "mean_packet_size": b_tot / n_tot if n_tot else 0.0,
        "fwd_mean_packet_size": b_fwd / n_fwd if n_fwd else 0.0,
        "bwd_mean_packet_size": b_bwd / n_bwd if n_bwd else 0.0,
    }

    for prefix, subset in (("", ts), ("fwd_", ts[fwd_mask]), ("bwd_", ts[bwd_mask])):
        mean, std, lo, hi = _iat_stats(subset)
        feats[f"{prefix}iat_mean"] = mean
        feats[f"{prefix}iat_std"] = std
        feats[f"{prefix}iat_min"] = lo
        feats[f"{prefix}iat_max"] = hi

    for name, bit in _FLAG_BITS:
        feats[name] = float(np.count_nonzero(flags & bit))
    denom = float(n_tot) if n_tot else 1.0
    feats["syn_ratio"] = feats["syn_count"] / denom
    feats["rst_ratio"] = feats["rst_count"] / denom
    feats["ack_ratio"] = feats["ack_count"] / denom
    feats["psh_ratio"] = feats["psh_count"] / denom

    dst_port = int(table.dst_port[idx[0]])
    src_port = int(table.src_port[idx[0]])
    feats["dst_port"] = float(dst_port)
    feats["src_port"] = float(src_port)
    feats["protocol"] = float(proto)
    feats["is_wellknown_dst_port"] = 1.0 if dst_port in WELL_KNOWN_PORTS else 0.0

    # ---------------------------------------------------------- packet level
    ttl_mean, ttl_std, ttl_min, ttl_max = _stat4(ttl)
    feats.update({
        "ttl_mean": ttl_mean, "ttl_std": ttl_std, "ttl_min": ttl_min,
        "ttl_max": ttl_max, "ttl_unique": float(np.unique(ttl).size),
    })
    win_mean, win_std, win_min, win_max = _stat4(window)
    feats.update({
        "window_mean": win_mean, "window_std": win_std, "window_min": win_min,
        "window_max": win_max,
        "window_zero_count": float(np.count_nonzero(window == 0)),
    })
    pay_mean, pay_std, pay_min, pay_max = _stat4(payload)
    feats.update({
        "payload_mean": pay_mean, "payload_std": pay_std,
        "payload_min": pay_min, "payload_max": pay_max,
        "payload_p50": float(np.percentile(payload, 50)) if payload.size else 0.0,
        "payload_p90": float(np.percentile(payload, 90)) if payload.size else 0.0,
        "payload_zero_ratio": float(np.count_nonzero(payload == 0)) / denom,
    })
    feats["header_len_mean"] = float(ihl.mean()) if ihl.size else 0.0
    feats["tcp_doff_mean"] = float(doff.mean()) if doff.size else 0.0

    retrans = dup_ack = out_of_order = 0
    if proto == PROTO_TCP:
        retrans, dup_ack, out_of_order = _tcp_anomalies(
            table.tcp_seq[idx], table.tcp_ack[idx], payload, flags, fwd_mask
        )
    feats["retransmission_count"] = float(retrans)
    feats["retransmission_ratio"] = retrans / denom
    feats["dup_ack_count"] = float(dup_ack)
    feats["out_of_order_count"] = float(out_of_order)

    is_frag = (ip_flags & IP_FLAG_MF).astype(bool) | (frag_off > 0)
    feats["frag_packet_count"] = float(np.count_nonzero(is_frag))
    feats["df_ratio"] = float(np.count_nonzero(ip_flags & IP_FLAG_DF)) / denom
    feats["mf_ratio"] = float(np.count_nonzero(ip_flags & IP_FLAG_MF)) / denom
    return feats


def _tcp_anomalies(seq: np.ndarray, ack: np.ndarray, payload: np.ndarray,
                   flags: np.ndarray, fwd_mask: np.ndarray) -> tuple[int, int, int]:
    """Count retransmissions, duplicate ACKs and out-of-order segments.

    Each direction is tracked separately -- sequence spaces are independent --
    and only data-bearing segments can be retransmissions, which keeps bare
    ACKs from inflating the count.
    """
    retrans = dup_ack = out_of_order = 0
    for mask in (fwd_mask, ~fwd_mask):
        if not mask.any():
            continue
        d_seq = seq[mask].astype(np.int64)
        d_ack = ack[mask].astype(np.int64)
        d_pay = payload[mask]
        d_flags = flags[mask]

        seen: set[tuple[int, int]] = set()
        highest = -1
        last_ack = None
        repeat = 0
        for s, a, p, f in zip(d_seq, d_ack, d_pay, d_flags):
            if p > 0:
                key = (int(s), int(p))
                if key in seen:
                    retrans += 1
                elif highest >= 0 and s < highest:
                    out_of_order += 1
                seen.add(key)
                highest = max(highest, int(s))
            elif f & TCP_ACK:
                if last_ack is not None and a == last_ack:
                    repeat += 1
                    if repeat >= 1:
                        dup_ack += 1
                else:
                    repeat = 0
                last_ack = int(a)
    return retrans, dup_ack, out_of_order


# ------------------------------------------------------------------ assembly
def assemble_flows(
    table: PacketTable,
    idle_timeout_s: float = 120.0,
    active_timeout_s: float = 300.0,
    close_on_teardown: bool = True,
) -> pd.DataFrame:
    """Turn a :class:`PacketTable` into one row per flow.

    Returns a DataFrame with :data:`FLOW_META_COLUMNS` followed by
    :data:`FLOW_FEATURES` and :data:`PACKET_FEATURES`, in that order.
    """
    columns = list(FLOW_META_COLUMNS) + list(FLOW_FEATURES) + list(PACKET_FEATURES)
    if len(table) == 0:
        return pd.DataFrame(columns=columns)

    table = table.sort_by_time()
    keys = _canonical_keys(table)
    order = np.argsort(keys, kind="stable")   # stable => time order kept inside a group
    sorted_keys = keys[order]
    boundaries = np.flatnonzero(np.diff(sorted_keys)) + 1
    groups = np.split(order, boundaries)

    src_ip = table.src_ip
    dst_ip = table.dst_ip
    src_port = table.src_port
    dst_port = table.dst_port
    ts = table.ts
    flags = table.tcp_flags
    proto_col = table.protocol

    rows: list[dict] = []
    for group in groups:
        if group.size == 0:
            continue
        first = group[0]
        a_ip, a_port = src_ip[first], src_port[first]
        # forward == direction of the first packet in this conversation
        fwd = (src_ip[group] == a_ip) & (src_port[group] == a_port)
        is_tcp = int(proto_col[first]) == PROTO_TCP

        episode: list[int] = []
        ep_fwd: list[bool] = []
        ep_start = ts[first]
        last_ts = ts[first]
        fin_seen = {True: False, False: False}
        closed = False
        # A graceful TCP close is FIN, FIN, then a final ACK. Closing the flow
        # the instant the second FIN lands strands that ACK in a spurious
        # one-packet flow, which then pollutes the per-window flow counts.
        grace = -1

        def flush() -> None:
            if not episode:
                return
            idx = np.asarray(episode, dtype=np.int64)
            rows.append(_row(table, idx, np.asarray(ep_fwd, dtype=bool)))

        for pos, direction in zip(group, fwd):
            t = ts[pos]
            gap = t - last_ts
            if episode and (gap > idle_timeout_s
                            or (t - ep_start) > active_timeout_s
                            or closed):
                flush()
                episode, ep_fwd = [], []
                ep_start = t
                fin_seen = {True: False, False: False}
                closed = False
                grace = -1
            episode.append(int(pos))
            ep_fwd.append(bool(direction))
            last_ts = t
            if grace > 0:
                grace -= 1
                if grace == 0:
                    closed = True
            if is_tcp and close_on_teardown:
                f = int(flags[pos])
                if f & TCP_RST:
                    closed = True
                elif f & TCP_FIN:
                    fin_seen[bool(direction)] = True
                    if fin_seen[True] and fin_seen[False] and grace < 0:
                        grace = 1   # admit the final ACK, then close
        flush()

    frame = pd.DataFrame(rows, columns=columns)
    return frame.sort_values("start_ts", kind="stable").reset_index(drop=True)


def _row(table: PacketTable, idx: np.ndarray, fwd_mask: np.ndarray) -> dict:
    first = int(idx[0])
    feats = _episode_features(table, idx, fwd_mask)
    src = table.ip_str(table.src_ip[first])
    dst = table.ip_str(table.dst_ip[first])
    sport = int(table.src_port[first])
    dport = int(table.dst_port[first])
    proto = int(table.protocol[first])
    meta = {
        "flow_id": f"{src}:{sport}-{dst}:{dport}-{proto}-{table.ts[first]:.6f}",
        "src_ip": src, "dst_ip": dst,
        "src_port_raw": sport, "dst_port_raw": dport, "protocol_raw": proto,
        "start_ts": float(table.ts[first]), "end_ts": float(table.ts[int(idx[-1])]),
    }
    meta.update(feats)
    return meta
