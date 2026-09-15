"""Time-windowing: flows in, network *states* out.

This is where ATDRPS stops looking like an IDS.  A classifier asks a question
per flow; the world model asks a question per *window* -- what is the network
doing right now, and what does that imply about the next window -- so the unit
of representation has to be a time slice, not a conversation.

Each window ``S_t`` is a fixed-length vector with three kinds of component:

``volume and shape``
    How much traffic, from how many hosts, to how many ports, in which
    direction.  The obvious part.

``distributional aggregates``
    Mean, spread and extreme of the per-flow features across the window.  The
    spread matters as much as the mean: a window of beacons has a *tiny*
    ``iat_std``, and that is the whole tell.

``behavioural detectors``
    Structure that no single flow contains -- how many ports one source swept
    on one host, how regular the interval between repeated contacts is, how
    many internal peers a host reached, how much of the destination set is new
    relative to everything seen before.

The third group is the reason this project is not a feature-engineering
exercise dressed up in a transformer.  Those quantities only exist once traffic
is grouped in time, and they are what a forecast of ``S_{t+1}`` is actually
made of.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .mitre import OUT_OF_SCOPE, stage_from_label
from .schema import (
    BENIGN_STAGE, DEFAULT_INFILTRATION_STAGES, PROTO_ICMP, PROTO_TCP,
    PROTO_UDP, STAGE_INDEX, STAGES,
)

__all__ = [
    "WindowedStates", "build_windows", "AGGREGATED_FLOW_FEATURES",
    "state_feature_names", "is_internal",
]

# Per-flow features aggregated into every window as (mean, std, max).
AGGREGATED_FLOW_FEATURES: tuple[str, ...] = (
    "flow_duration", "total_packets", "total_bytes",
    "flow_bytes_per_s", "flow_packets_per_s",
    "bytes_ratio", "packets_ratio", "mean_packet_size",
    "iat_mean", "iat_std",
    "syn_ratio", "rst_ratio", "psh_ratio",
    "ttl_std", "ttl_unique", "window_mean",
    "payload_mean", "payload_zero_ratio",
    "retransmission_ratio", "frag_packet_count",
)

SCALAR_FEATURES: tuple[str, ...] = (
    "log_n_flows", "log_n_packets", "log_n_bytes",
    "log_bytes_per_s", "log_packets_per_s", "log_flows_per_s",
    "log_distinct_src", "log_distinct_dst",
    "log_distinct_dst_ports", "log_distinct_src_ports",
    "tcp_share", "udp_share", "icmp_share",
    "internal_flow_share", "external_flow_share", "inbound_flow_share",
    "outbound_bytes_ratio",
    "top1_src_flow_share", "top1_dst_flow_share", "top1_src_byte_share",
    "log_mean_flow_bytes", "mean_flow_packets",
)

DETECTOR_FEATURES: tuple[str, ...] = (
    "max_ports_per_src_dst", "log_max_flows_per_src_dst_port",
    "sequential_port_ratio",
    "auth_service_share", "admin_service_share", "web_service_share",
    "distinct_internal_dsts", "beacon_regularity", "log_beacon_flow_count",
    "repeat_external_dst_count", "new_dst_ratio", "new_dst_port_ratio",
    "graph_mean_degree", "graph_max_degree",
    # --- added to carry attack families the first fourteen could not separate;
    #     see docs/ATTACK_COVERAGE.md for which attack each one is for
    "log_same_port_fanout",      # worm: one source, many hosts, one port
    "half_open_ratio",           # DoS / scanning: handshakes that never complete
    "log_max_srcs_per_dst_service",  # credential stuffing, DDoS: many -> one
    "dns_share",                 # DNS tunnelling / spoofing
    "log_dns_query_size",        # tunnelling: data rides in long encoded subdomains
    "ttl_inconsistency",         # MitM: one source arriving with several TTLs
    "rst_injection_ratio",       # MitM: RSTs injected mid-session
)

DNS_PORTS = frozenset((53, 5353, 853))

AUTH_PORTS = frozenset((21, 22, 23, 3389, 445, 1433, 3306, 5432, 5900))
ADMIN_PORTS = frozenset((135, 139, 445, 3389, 22, 5985, 5986))
WEB_PORTS = frozenset((80, 443, 8080, 8443))

_PRIVATE_BLOCKS = (
    ("10.0.0.0", 8), ("172.16.0.0", 12), ("192.168.0.0", 16),
    ("127.0.0.0", 8), ("169.254.0.0", 16),
)


def _ip_to_int(addr: str) -> int:
    try:
        parts = [int(p) for p in str(addr).split(".")]
    except (ValueError, AttributeError):
        return -1
    if len(parts) != 4:
        return -1
    return (parts[0] << 24) | (parts[1] << 16) | (parts[2] << 8) | parts[3]


_PRIVATE_RANGES = []
for _net, _bits in _PRIVATE_BLOCKS:
    _base = _ip_to_int(_net)
    _mask = (0xFFFFFFFF << (32 - _bits)) & 0xFFFFFFFF
    _PRIVATE_RANGES.append((_base & _mask, _mask))


def is_internal(addr: str) -> bool:
    """RFC1918 / loopback / link-local.  Used to tell lateral movement
    (internal -> internal) from command and control (internal -> outside)."""
    value = _ip_to_int(addr)
    if value < 0:
        return False
    return any((value & mask) == base for base, mask in _PRIVATE_RANGES)


def state_feature_names() -> list[str]:
    names: list[str] = list(SCALAR_FEATURES)
    for feature in AGGREGATED_FLOW_FEATURES:
        names.extend((f"{feature}__mean", f"{feature}__std", f"{feature}__max"))
    names.extend(DETECTOR_FEATURES)
    return names


@dataclass
class WindowedStates:
    """A time-ordered sequence of network states with their labels."""

    X: np.ndarray                      # (T, F) float32
    feature_names: list[str]
    ts_start: np.ndarray               # (T,) window start, epoch seconds
    window_size_s: float
    stage: np.ndarray                  # (T,) int index into STAGES
    stage_mask: np.ndarray             # (T,) bool -- False => exclude from stage loss
    infiltration: np.ndarray           # (T,) int 0/1
    flow_index: list[np.ndarray] = field(default_factory=list)
    summaries: list[dict] = field(default_factory=list)
    label_source: str = "unlabelled"

    def __len__(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    def stage_names(self) -> list[str]:
        return [STAGES[i] if 0 <= i < len(STAGES) else OUT_OF_SCOPE for i in self.stage]

    def describe(self) -> str:
        counts = {}
        for name, valid in zip(self.stage_names(), self.stage_mask):
            key = name if valid else OUT_OF_SCOPE
            counts[key] = counts.get(key, 0) + 1
        parts = ", ".join(f"{k}={v}" for k, v in sorted(counts.items(), key=lambda x: -x[1]))
        return (f"{len(self)} windows of {self.window_size_s:g}s, "
                f"{self.n_features} features, labels from {self.label_source} [{parts}]")


# ------------------------------------------------------------------ helpers
def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def _agg(values: np.ndarray) -> tuple[float, float, float]:
    if values.size == 0:
        return 0.0, 0.0, 0.0
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 0.0, 0.0
    return float(finite.mean()), float(finite.std()), float(finite.max())


def _beacon_regularity(starts_by_triple: dict[tuple, list[float]]) -> tuple[float, int]:
    """How metronomic is the most regular repeated contact in this window?

    For each (src, dst, dst_port) seen at least four times, take the coefficient
    of variation of the gaps between contacts and turn it into a regularity in
    ``[0, 1]``.  A human clicking around produces bursty, high-variance gaps;
    an implant on a timer does not.  Returns the best regularity and how many
    flows that triple contributed.
    """
    best, best_count = 0.0, 0
    for starts in starts_by_triple.values():
        if len(starts) < 4:
            continue
        gaps = np.diff(np.sort(np.asarray(starts, dtype=float)))
        gaps = gaps[gaps > 0]
        if gaps.size < 3:
            continue
        mean = float(gaps.mean())
        if mean <= 0:
            continue
        cv = float(gaps.std() / mean)
        regularity = 1.0 / (1.0 + cv)
        if regularity > best:
            best, best_count = regularity, len(starts)
    return best, best_count


def _sequential_port_ratio(ports: np.ndarray) -> float:
    """Fraction of a sorted port list that advances by exactly one.

    Separates a deliberate sequential sweep from randomised scanning; both are
    reconnaissance, but they look different and the model should be able to see
    which one it is.
    """
    if ports.size < 3:
        return 0.0
    unique = np.unique(ports)
    if unique.size < 3:
        return 0.0
    return float(np.count_nonzero(np.diff(unique) == 1) / (unique.size - 1))


# -------------------------------------------------------------------- build
def build_windows(
    flows: pd.DataFrame,
    window_size_s: float = 30.0,
    stride_s: float | None = None,
    stage_timeline=None,
    infiltration_stages=DEFAULT_INFILTRATION_STAGES,
    use_labels: bool = True,
    history_s: float = 600.0,
    t_end: float | None = None,
) -> WindowedStates:
    """Aggregate a flow table into a time-ordered sequence of state vectors.

    ``history_s`` is the lookback the periodicity detectors use.  It has to be
    longer than one window: a beacon on a 60 s timer contributes at most one
    flow to a 30 s window, so regularity measured inside the window alone is
    blind to exactly the behaviour it is named after.  Measured over the last
    ten minutes instead, the same beacon is unmistakable.

    ``stage_timeline`` is an optional callable ``ts -> stage name`` (the
    synthetic generator's ground truth, or an attack-timeline annotation).  If
    absent, stage labels come from the flows' own ``label`` column.  If that is
    absent too, every window is labelled benign and ``stage_mask`` is cleared,
    so an unlabelled capture can still be *scored* but never silently trains
    anything.

    ``t_end`` anchors the last window edge to a caller-supplied time instead of
    the last flow's *start* time.  A finished capture has no "now" -- the last
    flow to start is as good an endpoint as any -- but a live rolling buffer
    does, and it matters: a single long-lived flow (one persistent connection,
    a steady beacon) starts once and then never moves ``start_ts`` again, so
    without an explicit ``t_end`` the window count would stay frozen at 1
    forever even as real wall-clock time keeps passing and new packets keep
    arriving on that same flow. Live mode passes ``time.time()`` every tick;
    offline analysis leaves this ``None`` and keeps its original behaviour.
    """
    stride_s = float(stride_s or window_size_s)
    names = state_feature_names()
    if flows is None or len(flows) == 0:
        empty = np.zeros((0, len(names)), dtype=np.float32)
        return WindowedStates(
            X=empty, feature_names=names, ts_start=np.zeros(0),
            window_size_s=window_size_s, stage=np.zeros(0, dtype=np.int64),
            stage_mask=np.zeros(0, dtype=bool), infiltration=np.zeros(0, dtype=np.int64),
        )

    flows = flows.sort_values("start_ts", kind="stable").reset_index(drop=True)
    start_ts = flows["start_ts"].to_numpy(dtype=float)
    t0 = float(start_ts[0])
    t1 = float(t_end) if t_end is not None else float(start_ts[-1])
    t1 = max(t1, t0)
    n_windows = max(1, int(np.floor((t1 - t0) / stride_s)) + 1)
    edges = t0 + np.arange(n_windows + 1) * stride_s

    # which window does each flow belong to (by its start time)
    assign = np.clip(((start_ts - t0) // stride_s).astype(np.int64), 0, n_windows - 1)
    order = np.argsort(assign, kind="stable")
    sorted_assign = assign[order]
    bounds = np.searchsorted(sorted_assign, np.arange(n_windows + 1))

    src = flows["src_ip"].to_numpy()
    dst = flows["dst_ip"].to_numpy()
    dport = flows["dst_port_raw"].to_numpy()
    sport = flows["src_port_raw"].to_numpy()
    proto = flows["protocol_raw"].to_numpy()
    fwd_bytes = flows["fwd_bytes"].to_numpy(dtype=float)
    bwd_bytes = flows["bwd_bytes"].to_numpy(dtype=float)
    tot_bytes = flows["total_bytes"].to_numpy(dtype=float)
    tot_packets = flows["total_packets"].to_numpy(dtype=float)
    agg_source = {f: flows[f].to_numpy(dtype=float) for f in AGGREGATED_FLOW_FEATURES}
    # extra per-flow columns the detectors need but the aggregates do not carry
    det_source = {
        name: (flows[name].to_numpy(dtype=float) if name in flows.columns
               else np.zeros(len(flows), dtype=float))
        for name in ("syn_count", "ack_count", "rst_count", "fwd_ttl_mean")
    }

    src_internal = np.array([is_internal(a) for a in src], dtype=bool)
    dst_internal = np.array([is_internal(a) for a in dst], dtype=bool)

    has_labels = use_labels and "label" in flows.columns
    flow_stage = (
        np.array([stage_from_label(v) for v in flows["label"].to_numpy()])
        if has_labels else None
    )

    rows = np.zeros((n_windows, len(names)), dtype=np.float32)
    stages = np.zeros(n_windows, dtype=np.int64)
    masks = np.ones(n_windows, dtype=bool)
    infil = np.zeros(n_windows, dtype=np.int64)
    flow_index: list[np.ndarray] = []
    summaries: list[dict] = []

    seen_dsts: set = set()
    seen_dst_ports: set = set()

    for w in range(n_windows):
        idx = order[bounds[w]:bounds[w + 1]]
        flow_index.append(idx)
        # flows from the recent past, for the periodicity detectors
        hist_lo = np.searchsorted(start_ts, edges[w + 1] - history_s, side="left")
        hist_hi = np.searchsorted(start_ts, edges[w + 1], side="left")
        hist_idx = np.arange(hist_lo, hist_hi, dtype=np.int64)

        summary = _window_summary(
            idx, hist_idx, window_size_s, src, dst, sport, dport, proto,
            fwd_bytes, bwd_bytes, tot_bytes, tot_packets,
            src_internal, dst_internal, agg_source, det_source,
            seen_dsts, seen_dst_ports, start_ts,
        )
        summaries.append(summary)
        rows[w] = np.asarray([summary["_vector"][n] for n in names], dtype=np.float32)

        stage_name, valid = _window_stage(
            idx, edges[w], edges[w + 1], stage_timeline, flow_stage
        )
        stages[w] = STAGE_INDEX.get(stage_name, 0)
        masks[w] = valid
        infil[w] = 1 if (valid and stage_name in infiltration_stages) else 0

    label_source = (
        "ground-truth timeline" if stage_timeline is not None
        else "dataset labels" if has_labels else "unlabelled"
    )
    if label_source == "unlabelled":
        masks[:] = False

    return WindowedStates(
        X=rows, feature_names=names, ts_start=edges[:-1].copy(),
        window_size_s=window_size_s, stage=stages, stage_mask=masks,
        infiltration=infil, flow_index=flow_index, summaries=summaries,
        label_source=label_source,
    )


def _window_stage(idx, w_start, w_end, stage_timeline, flow_stage):
    """Stage for one window, plus whether it should train the stage head.

    Severity order matters: if any part of the window contains lateral
    movement, the window is lateral movement even though most of its flows are
    somebody reading email.  Attacks are rare and would otherwise be averaged
    away by the background they hide in.
    """
    if stage_timeline is not None:
        # Windows are half-open, [w_start, w_end). Sampling the timeline *at*
        # w_end reads the next window's label and drags its stage one window
        # early -- which, for a forecaster, silently turns a real prediction
        # into a peek at the answer.
        mid = (w_start + w_end) / 2.0
        last = w_end - 1e-6
        candidates = [stage_timeline(w_start), stage_timeline(mid), stage_timeline(last)]
        best = BENIGN_STAGE
        for name in candidates:
            if STAGE_INDEX.get(name, 0) > STAGE_INDEX.get(best, 0):
                best = name
        return best, True

    if flow_stage is None or len(idx) == 0:
        return BENIGN_STAGE, True

    present = flow_stage[idx]
    in_scope = [s for s in present if s != OUT_OF_SCOPE]
    out_of_scope = int(np.count_nonzero(present == OUT_OF_SCOPE))
    if not in_scope:
        # nothing but denial-of-service style traffic: real, but not a step on
        # the infiltration chain, so it must not supervise the stage head
        return BENIGN_STAGE, out_of_scope == 0

    best = BENIGN_STAGE
    for name in in_scope:
        if STAGE_INDEX.get(name, 0) > STAGE_INDEX.get(best, 0):
            best = name
    if best == BENIGN_STAGE and out_of_scope > 0:
        return BENIGN_STAGE, False
    return best, True


def _window_summary(
    idx, hist_idx, window_size_s, src, dst, sport, dport, proto,
    fwd_bytes, bwd_bytes, tot_bytes, tot_packets,
    src_internal, dst_internal, agg_source, det_source,
    seen_dsts, seen_dst_ports, start_ts,
) -> dict:
    n = int(idx.size)
    vec: dict[str, float] = {name: 0.0 for name in state_feature_names()}
    summary: dict[str, float] = {}

    if n == 0:
        # An empty window is a real observation -- the network went quiet --
        # so it is emitted as a zero state rather than dropped, which keeps the
        # time axis uniform for the sequence model.
        summary["_vector"] = vec
        return summary

    w_src, w_dst = src[idx], dst[idx]
    w_sport, w_dport = sport[idx], dport[idx]
    w_proto = proto[idx]
    w_fwd, w_bwd, w_bytes, w_pkts = fwd_bytes[idx], bwd_bytes[idx], tot_bytes[idx], tot_packets[idx]
    w_src_int, w_dst_int = src_internal[idx], dst_internal[idx]

    total_bytes = float(w_bytes.sum())
    total_packets = float(w_pkts.sum())
    span = max(window_size_s, 1e-6)

    vec["log_n_flows"] = float(np.log1p(n))
    vec["log_n_packets"] = float(np.log1p(total_packets))
    vec["log_n_bytes"] = float(np.log1p(total_bytes))
    vec["log_bytes_per_s"] = float(np.log1p(total_bytes / span))
    vec["log_packets_per_s"] = float(np.log1p(total_packets / span))
    vec["log_flows_per_s"] = float(np.log1p(n / span))

    uniq_src = np.unique(w_src)
    uniq_dst = np.unique(w_dst)
    uniq_dport = np.unique(w_dport)
    vec["log_distinct_src"] = float(np.log1p(uniq_src.size))
    vec["log_distinct_dst"] = float(np.log1p(uniq_dst.size))
    vec["log_distinct_dst_ports"] = float(np.log1p(uniq_dport.size))
    vec["log_distinct_src_ports"] = float(np.log1p(np.unique(w_sport).size))

    vec["tcp_share"] = _safe_div(np.count_nonzero(w_proto == PROTO_TCP), n)
    vec["udp_share"] = _safe_div(np.count_nonzero(w_proto == PROTO_UDP), n)
    vec["icmp_share"] = _safe_div(np.count_nonzero(w_proto == PROTO_ICMP), n)

    internal_flow = w_src_int & w_dst_int
    external_flow = w_src_int & ~w_dst_int
    inbound_flow = ~w_src_int & w_dst_int
    vec["internal_flow_share"] = _safe_div(np.count_nonzero(internal_flow), n)
    vec["external_flow_share"] = _safe_div(np.count_nonzero(external_flow), n)
    vec["inbound_flow_share"] = _safe_div(np.count_nonzero(inbound_flow), n)

    # bytes leaving the network, as a share of all bytes -- the exfiltration axis
    out_bytes = float(w_fwd[external_flow].sum() + w_bwd[inbound_flow].sum())
    vec["outbound_bytes_ratio"] = _safe_div(out_bytes, total_bytes)

    src_counts = pd.Series(w_src).value_counts()
    dst_counts = pd.Series(w_dst).value_counts()
    vec["top1_src_flow_share"] = _safe_div(float(src_counts.iloc[0]), n)
    vec["top1_dst_flow_share"] = _safe_div(float(dst_counts.iloc[0]), n)
    byte_by_src = pd.Series(w_bytes).groupby(pd.Series(w_src)).sum()
    vec["top1_src_byte_share"] = _safe_div(float(byte_by_src.max()), total_bytes)

    vec["log_mean_flow_bytes"] = float(np.log1p(total_bytes / n))
    vec["mean_flow_packets"] = float(total_packets / n)

    for feature in AGGREGATED_FLOW_FEATURES:
        mean, std, mx = _agg(agg_source[feature][idx])
        vec[f"{feature}__mean"] = mean
        vec[f"{feature}__std"] = std
        vec[f"{feature}__max"] = mx

    # ---------------------------------------------------------- detectors
    pair = pd.DataFrame({"src": w_src, "dst": w_dst, "dport": w_dport})
    ports_per_pair = pair.groupby(["src", "dst"])["dport"].nunique()
    vec["max_ports_per_src_dst"] = float(ports_per_pair.max()) if len(ports_per_pair) else 0.0
    flows_per_triple = pair.groupby(["src", "dst", "dport"]).size()
    vec["log_max_flows_per_src_dst_port"] = float(
        np.log1p(flows_per_triple.max()) if len(flows_per_triple) else 0.0
    )

    busiest = ports_per_pair.idxmax() if len(ports_per_pair) else None
    if busiest is not None:
        sel = (w_src == busiest[0]) & (w_dst == busiest[1])
        vec["sequential_port_ratio"] = _sequential_port_ratio(w_dport[sel])

    vec["auth_service_share"] = _safe_div(
        np.count_nonzero(np.isin(w_dport, list(AUTH_PORTS))), n)
    vec["admin_service_share"] = _safe_div(
        np.count_nonzero(np.isin(w_dport, list(ADMIN_PORTS))), n)
    vec["web_service_share"] = _safe_div(
        np.count_nonzero(np.isin(w_dport, list(WEB_PORTS))), n)

    if np.any(internal_flow):
        internal_pairs = pd.DataFrame(
            {"src": w_src[internal_flow], "dst": w_dst[internal_flow]}
        ).groupby("src")["dst"].nunique()
        vec["distinct_internal_dsts"] = float(internal_pairs.max())

    # Periodicity is measured over the lookback, not the window: a 60 s beacon
    # leaves at most one flow inside a 30 s window and would otherwise be
    # invisible to the very feature meant to catch it.
    h_src, h_dst, h_dport = src[hist_idx], dst[hist_idx], dport[hist_idx]
    h_starts = start_ts[hist_idx]
    h_src_int, h_dst_int = src_internal[hist_idx], dst_internal[hist_idx]

    triples: dict[tuple, list[float]] = {}
    for a, b, p, t in zip(h_src, h_dst, h_dport, h_starts):
        triples.setdefault((a, b, int(p)), []).append(float(t))
    regularity, beacon_count = _beacon_regularity(triples)
    vec["beacon_regularity"] = regularity
    vec["log_beacon_flow_count"] = float(np.log1p(beacon_count))

    h_external = h_src_int & ~h_dst_int
    if np.any(h_external):
        ext_counts = pd.Series(h_dst[h_external]).value_counts()
        vec["repeat_external_dst_count"] = float(np.count_nonzero(ext_counts.to_numpy() >= 3))

    new_dsts = [a for a in uniq_dst if a not in seen_dsts]
    new_ports = [int(p) for p in uniq_dport if int(p) not in seen_dst_ports]
    vec["new_dst_ratio"] = _safe_div(len(new_dsts), uniq_dst.size)
    vec["new_dst_port_ratio"] = _safe_div(len(new_ports), uniq_dport.size)
    seen_dsts.update(uniq_dst.tolist())
    seen_dst_ports.update(int(p) for p in uniq_dport)

    degree = pd.Series(np.concatenate([w_src, w_dst])).value_counts()
    vec["graph_mean_degree"] = float(degree.mean())
    vec["graph_max_degree"] = float(degree.max())

    # ---------------------------------------------- attack-family detectors
    # Worm: the signature is fan-out on a *fixed* port. A scan is one source
    # touching many ports on one host; a worm is one source touching many hosts
    # on one port. Without this they look alike in aggregate.
    fanout = pair.assign(dport=w_dport).groupby(["src", "dport"])["dst"].nunique()
    same_port_fanout = float(fanout.max()) if len(fanout) else 0.0
    vec["log_same_port_fanout"] = float(np.log1p(same_port_fanout))

    # Denial of service and scanning both leave handshakes unfinished.
    syn = det_source["syn_count"][idx]
    ack = det_source["ack_count"][idx]
    vec["half_open_ratio"] = _safe_div(np.count_nonzero((syn > 0) & (ack == 0)), n)

    # Credential stuffing and DDoS both converge many sources on one service;
    # a per-source rate limit sees nothing, the window-level view does.
    srcs_per_service = pd.DataFrame(
        {"dst": w_dst, "dport": w_dport, "src": w_src}
    ).groupby(["dst", "dport"])["src"].nunique()
    max_srcs = float(srcs_per_service.max()) if len(srcs_per_service) else 0.0
    vec["log_max_srcs_per_dst_service"] = float(np.log1p(max_srcs))

    # DNS tunnelling. The response/query size *ratio* does not work: ordinary
    # lookups already return far more than they ask (a 40-byte query, a 300-byte
    # answer), and a tunnel's ratio is if anything lower. What separates them is
    # the query itself -- encoded data rides in long subdomains, so a tunnelled
    # query is several times the size of a real one.
    is_dns = np.isin(w_dport, list(DNS_PORTS)) | np.isin(w_sport, list(DNS_PORTS))
    vec["dns_share"] = _safe_div(np.count_nonzero(is_dns), n)
    if np.any(is_dns):
        vec["log_dns_query_size"] = float(np.log1p(np.mean(w_fwd[is_dns])))

    # MitM: the spoofing is below IP, but a source suddenly arriving with
    # several different TTLs is the consequence, and that is in the header.
    # Expressed as the *fraction of sources* that are inconsistent, because a
    # raw count grows with traffic volume and would just re-measure busyness.
    ttl = np.round(det_source["fwd_ttl_mean"][idx])
    if n > 1:
        per_src = pd.DataFrame({"src": w_src, "ttl": ttl}).groupby("src")["ttl"].nunique()
        vec["ttl_inconsistency"] = float((per_src > 1).mean()) if len(per_src) else 0.0

    # An RST on a conversation that was already flowing is injection; an RST on
    # a short exchange is a refused connection. The threshold has to clear a
    # failed authentication attempt -- handshake, request, reply, reset is six
    # packets -- or every brute-force window scores as machine-in-the-middle.
    rst = det_source["rst_count"][idx]
    vec["rst_injection_ratio"] = _safe_div(np.count_nonzero((rst > 0) & (w_pkts > 12)), n)

    # ------------- plain-language summary used by the rule engine and the UI
    summary.update({
        "n_flows": float(n),
        "total_bytes": total_bytes,
        "distinct_dst_ports": float(uniq_dport.size),
        "mean_payload_zero_ratio": vec["payload_zero_ratio__mean"],
        "mean_flow_packets": vec["mean_flow_packets"],
        "rst_share": vec["rst_ratio__mean"],
        "max_ports_per_src_dst": vec["max_ports_per_src_dst"],
        "max_flows_per_src_dst_port": float(flows_per_triple.max()) if len(flows_per_triple) else 0.0,
        "auth_service_share": vec["auth_service_share"],
        "admin_service_share": vec["admin_service_share"],
        "internal_flow_share": vec["internal_flow_share"],
        "external_flow_share": vec["external_flow_share"],
        "distinct_internal_dsts": vec["distinct_internal_dsts"],
        "mean_retransmission_ratio": vec["retransmission_ratio__mean"],
        "beacon_regularity": regularity,
        "mean_flow_bytes": total_bytes / n,
        "repeat_external_dst_count": vec["repeat_external_dst_count"],
        "outbound_bytes_ratio": vec["outbound_bytes_ratio"],
        "mean_payload_mean": vec["payload_mean__mean"],
        "same_port_fanout": same_port_fanout,
        "half_open_ratio": vec["half_open_ratio"],
        "max_srcs_per_dst_service": max_srcs,
        "dns_share": vec["dns_share"],
        "dns_query_size": float(np.mean(w_fwd[is_dns])) if np.any(is_dns) else 0.0,
        "ttl_inconsistency": vec["ttl_inconsistency"],
        "rst_injection_ratio": vec["rst_injection_ratio"],
    })
    summary["_vector"] = vec
    return summary
