"""Plain-English names for state features.

"High ``syn_ratio__mean`` and low ``iat_std__mean``" is not an explanation to
anyone who has not read this repository.  The problem statement asks for output
of the form *"high SYN ratio + low inter-arrival variance + sequential port
pattern -> Reconnaissance likelihood 0.82"*, so the vocabulary has to come out
of the model in words a SOC analyst already uses.
"""

from __future__ import annotations

__all__ = ["describe_feature", "FEATURE_GLOSSARY", "AGGREGATE_SUFFIXES"]

AGGREGATE_SUFFIXES = {
    "__mean": "average",
    "__std": "variability of",
    "__max": "peak",
}

BASE_GLOSSARY: dict[str, str] = {
    # --- window scalars
    "log_n_flows": "number of conversations",
    "log_n_packets": "packet volume",
    "log_n_bytes": "byte volume",
    "log_bytes_per_s": "throughput",
    "log_packets_per_s": "packet rate",
    "log_flows_per_s": "connection rate",
    "log_distinct_src": "number of distinct sources",
    "log_distinct_dst": "number of distinct destinations",
    "log_distinct_dst_ports": "number of distinct destination ports",
    "log_distinct_src_ports": "number of distinct source ports",
    "tcp_share": "share of TCP traffic",
    "udp_share": "share of UDP traffic",
    "icmp_share": "share of ICMP traffic",
    "internal_flow_share": "share of internal-to-internal traffic",
    "external_flow_share": "share of traffic leaving the network",
    "inbound_flow_share": "share of traffic entering from outside",
    "outbound_bytes_ratio": "proportion of bytes leaving the network",
    "top1_src_flow_share": "concentration of activity on one source host",
    "top1_dst_flow_share": "concentration of activity on one destination host",
    "top1_src_byte_share": "concentration of volume on one source host",
    "log_mean_flow_bytes": "typical conversation size",
    "mean_flow_packets": "typical packets per conversation",
    # --- detectors
    "max_ports_per_src_dst": "most ports one host probed on a single target",
    "log_max_flows_per_src_dst_port": "most repeated connections to one service",
    "sequential_port_ratio": "how sequentially ports were walked",
    "auth_service_share": "share of traffic to remote-access services",
    "admin_service_share": "share of traffic to SMB/RDP/SSH/WMI",
    "web_service_share": "share of traffic to web services",
    "distinct_internal_dsts": "internal peers reached by one host",
    "beacon_regularity": "how metronomic repeated outbound contacts are",
    "log_beacon_flow_count": "size of the most regular contact pattern",
    "repeat_external_dst_count": "external hosts contacted repeatedly",
    "new_dst_ratio": "proportion of destinations never seen before",
    "new_dst_port_ratio": "proportion of destination ports never seen before",
    "graph_mean_degree": "average host connectivity",
    "graph_max_degree": "connectivity of the busiest host",
    # --- aggregated per-flow features
    "flow_duration": "conversation duration",
    "total_packets": "packets per conversation",
    "total_bytes": "bytes per conversation",
    "flow_bytes_per_s": "per-conversation throughput",
    "flow_packets_per_s": "per-conversation packet rate",
    "bytes_ratio": "byte asymmetry (how one-sided a conversation is)",
    "packets_ratio": "packet asymmetry",
    "mean_packet_size": "packet size",
    "iat_mean": "gap between packets",
    "iat_std": "irregularity of packet timing",
    "syn_ratio": "SYN flag ratio",
    "rst_ratio": "RST flag ratio (refused connections)",
    "psh_ratio": "PSH flag ratio",
    "ttl_std": "TTL variation",
    "ttl_unique": "number of distinct TTL values",
    "window_mean": "TCP receive-window size",
    "payload_mean": "payload size",
    "payload_zero_ratio": "proportion of packets carrying no payload",
    "retransmission_ratio": "retransmission rate",
    "frag_packet_count": "fragmented packets",
}

FEATURE_GLOSSARY = dict(BASE_GLOSSARY)


def describe_feature(name: str) -> str:
    """``'syn_ratio__std'`` -> ``'variability of SYN flag ratio'``."""
    for suffix, prefix in AGGREGATE_SUFFIXES.items():
        if name.endswith(suffix):
            base = name[: -len(suffix)]
            described = BASE_GLOSSARY.get(base, base.replace("_", " "))
            return f"{prefix} {described}"
    return BASE_GLOSSARY.get(name, name.replace("_", " "))
