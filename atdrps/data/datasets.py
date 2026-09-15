"""Adapters that map public flow datasets onto the ATDRPS canonical schema.

Every dataset spells its columns differently -- ``Dst Port`` in CIC-IDS2018,
`` Destination Port`` (with a leading space) in CIC-IDS2017, ``dsport`` in
UNSW-NB15.  Rather than scatter that knowledge through the pipeline, all of it
lives here, keyed on a normalised form of the column name so that whitespace
and punctuation differences stop mattering.

An honest note about what CSV flow records can and cannot give you
--------------------------------------------------------------------
The published CSVs are NetFlow-style aggregates.  They carry most of
:data:`FLOW_FEATURES` and almost none of :data:`PACKET_FEATURES` -- there is no
TTL variance, no per-packet window trace, no retransmission count, because
those were discarded when the flows were built.

So a CSV-only run is a *flow-level* run.  ``load_flow_csv`` returns the set of
features it could actually populate, the missing ones are left at zero, and the
trainer is told which columns are real so it never reports a packet-level
feature as a driver on a run that never saw a packet.  To get the full dual-level
state the problem statement asks for, the matching PCAPs have to be ingested
through :mod:`atdrps.data.flows`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import FLOW_FEATURES, PACKET_FEATURES
from .flows import FLOW_META_COLUMNS

__all__ = [
    "load_flow_csv", "FlowFrame", "normalise_name", "detect_dataset",
    "CANONICAL_ALIASES", "KNOWN_DATASETS",
]


def normalise_name(name: str) -> str:
    """``' Flow IAT Mean '`` and ``'flow_iat_mean'`` both become ``flowiatmean``."""
    return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())


# canonical feature -> source column spellings seen in the wild
CANONICAL_ALIASES: dict[str, tuple[str, ...]] = {
    "flow_duration": ("Flow Duration", "dur", "Duration"),
    "fwd_packets": ("Tot Fwd Pkts", "Total Fwd Packets", "Spkts", "spkts"),
    "bwd_packets": ("Tot Bwd Pkts", "Total Backward Packets", "Dpkts", "dpkts"),
    "fwd_bytes": ("TotLen Fwd Pkts", "Total Length of Fwd Packets", "sbytes", "SrcBytes"),
    "bwd_bytes": ("TotLen Bwd Pkts", "Total Length of Bwd Packets", "dbytes", "DstBytes"),
    "flow_bytes_per_s": ("Flow Byts/s", "Flow Bytes/s"),
    "flow_packets_per_s": ("Flow Pkts/s", "Flow Packets/s"),
    "fwd_packets_per_s": ("Fwd Pkts/s", "Fwd Packets/s", "Sload"),
    "bwd_packets_per_s": ("Bwd Pkts/s", "Bwd Packets/s", "Dload"),
    "mean_packet_size": ("Pkt Size Avg", "Average Packet Size"),
    "fwd_mean_packet_size": ("Fwd Pkt Len Mean", "Fwd Packet Length Mean", "Fwd Seg Size Avg"),
    "bwd_mean_packet_size": ("Bwd Pkt Len Mean", "Bwd Packet Length Mean", "Bwd Seg Size Avg"),
    "iat_mean": ("Flow IAT Mean",),
    "iat_std": ("Flow IAT Std",),
    "iat_min": ("Flow IAT Min",),
    "iat_max": ("Flow IAT Max",),
    "fwd_iat_mean": ("Fwd IAT Mean",),
    "fwd_iat_std": ("Fwd IAT Std",),
    "fwd_iat_min": ("Fwd IAT Min",),
    "fwd_iat_max": ("Fwd IAT Max",),
    "bwd_iat_mean": ("Bwd IAT Mean",),
    "bwd_iat_std": ("Bwd IAT Std",),
    "bwd_iat_min": ("Bwd IAT Min",),
    "bwd_iat_max": ("Bwd IAT Max",),
    "fin_count": ("FIN Flag Cnt", "FIN Flag Count"),
    "syn_count": ("SYN Flag Cnt", "SYN Flag Count"),
    "rst_count": ("RST Flag Cnt", "RST Flag Count"),
    "psh_count": ("PSH Flag Cnt", "PSH Flag Count"),
    "ack_count": ("ACK Flag Cnt", "ACK Flag Count"),
    "urg_count": ("URG Flag Cnt", "URG Flag Count"),
    "ece_count": ("ECE Flag Cnt", "ECE Flag Count"),
    "cwr_count": ("CWR Flag Count", "CWE Flag Count", "CWE Flag Cnt"),
    "dst_port": ("Dst Port", "Destination Port", "dsport", "dport"),
    "src_port": ("Src Port", "Source Port", "sport", "srcport"),
    "protocol": ("Protocol", "proto"),
    # packet-level statistics that some datasets do partially expose
    "window_mean": ("Init Fwd Win Byts", "Init_Win_bytes_forward", "swin"),
    "payload_min": ("Pkt Len Min", "Min Packet Length"),
    "payload_max": ("Pkt Len Max", "Max Packet Length"),
    "payload_mean": ("Pkt Len Mean", "Packet Length Mean"),
    "payload_std": ("Pkt Len Std", "Packet Length Std"),
    "header_len_mean": ("Fwd Header Len", "Fwd Header Length"),
    "ttl_mean": ("sttl", "Sttl"),
    "ttl_max": ("dttl", "Dttl"),
}

META_ALIASES: dict[str, tuple[str, ...]] = {
    "src_ip": ("Src IP", "Source IP", "srcip", "SrcAddr", "Src_IP"),
    "dst_ip": ("Dst IP", "Destination IP", "dstip", "DstAddr", "Dst_IP"),
    "start_ts": ("Timestamp", "StartTime", "Stime", "ts"),
    "label": ("Label", "label", "attack_cat", "Attack", "class"),
}

KNOWN_DATASETS = ("cicids2018", "cicids2017", "unswnb15", "ctu13", "generic")

# duration units differ: CIC reports microseconds, UNSW reports seconds
_DURATION_SCALE = {"cicids2018": 1e-6, "cicids2017": 1e-6, "unswnb15": 1.0,
                   "ctu13": 1.0, "generic": 1.0}


@dataclass
class FlowFrame:
    """Flow features plus an honest record of which of them are real."""

    frame: pd.DataFrame
    dataset: str
    available: set[str] = field(default_factory=set)
    source: str = ""

    @property
    def missing(self) -> list[str]:
        return [f for f in list(FLOW_FEATURES) + list(PACKET_FEATURES)
                if f not in self.available]

    def __len__(self) -> int:
        return len(self.frame)

    def summary(self) -> str:
        total = len(FLOW_FEATURES) + len(PACKET_FEATURES)
        pkt_have = len(self.available & set(PACKET_FEATURES))
        return (
            f"{self.dataset}: {len(self.frame)} flows, "
            f"{len(self.available)}/{total} canonical features populated "
            f"({pkt_have}/{len(PACKET_FEATURES)} packet-level). "
            f"{'Flow-level run.' if pkt_have < 5 else 'Dual-level run.'}"
        )


def detect_dataset(columns) -> str:
    """Guess which public dataset a header came from."""
    norm = {normalise_name(c) for c in columns}
    if {"totfwdpkts", "flowbytss"} & norm and "dstport" in norm:
        return "cicids2018"
    if {"totalfwdpackets"} & norm or " destination port" in {str(c).lower() for c in columns}:
        return "cicids2017"
    if {"sbytes", "dbytes", "sttl"} <= norm:
        return "unswnb15"
    if {"srcaddr", "dstaddr"} <= norm:
        return "ctu13"
    return "generic"


def _build_lookup(columns) -> dict[str, str]:
    return {normalise_name(c): c for c in columns}


def load_flow_csv(
    path: str | Path,
    dataset: str = "auto",
    nrows: int | None = None,
    label_column: str | None = None,
) -> FlowFrame:
    """Read a flow CSV into the canonical schema.

    Features the file does not contain are present as zero columns and are
    reported in :attr:`FlowFrame.missing`, so nothing downstream mistakes an
    absent measurement for a measured zero.
    """
    path = Path(path)
    raw = pd.read_csv(path, nrows=nrows, low_memory=False)
    raw.columns = [str(c) for c in raw.columns]
    if dataset == "auto":
        dataset = detect_dataset(raw.columns)
    if dataset not in KNOWN_DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; expected one of {KNOWN_DATASETS}")

    lookup = _build_lookup(raw.columns)
    out = pd.DataFrame(index=raw.index)
    available: set[str] = set()

    for canonical in list(FLOW_FEATURES) + list(PACKET_FEATURES):
        source_col = None
        for alias in CANONICAL_ALIASES.get(canonical, ()):
            source_col = lookup.get(normalise_name(alias))
            if source_col is not None:
                break
        if source_col is None:
            out[canonical] = 0.0
            continue
        values = pd.to_numeric(raw[source_col], errors="coerce")
        out[canonical] = values.astype("float64")
        available.add(canonical)

    # CIC stores durations in microseconds; the rest of ATDRPS speaks seconds
    if "flow_duration" in available:
        out["flow_duration"] = out["flow_duration"] * _DURATION_SCALE[dataset]

    _derive_missing(out, available)

    # ---- metadata
    for meta, aliases in META_ALIASES.items():
        col = None
        for alias in aliases:
            col = lookup.get(normalise_name(alias))
            if col is not None:
                break
        out[meta] = raw[col] if col is not None else ("" if meta != "start_ts" else np.nan)

    if label_column and label_column in raw.columns:
        out["label"] = raw[label_column]
    # pandas >= 3 keeps NA through astype(str) instead of producing "nan", so
    # fill before casting rather than trying to catch the string afterwards.
    labels = out["label"].astype("object")
    labels = labels.where(pd.notna(labels), "Benign")
    out["label"] = (
        labels.astype(str).str.strip()
        .replace({"": "Benign", "nan": "Benign", "NaN": "Benign", "None": "Benign",
                  "-": "Benign", "normal": "Benign", "BENIGN": "Benign"})
    )

    out["start_ts"] = _parse_timestamps(out["start_ts"])
    out["end_ts"] = out["start_ts"] + out["flow_duration"].fillna(0.0)
    for raw_col, feature in (("src_port_raw", "src_port"), ("dst_port_raw", "dst_port"),
                             ("protocol_raw", "protocol")):
        out[raw_col] = pd.to_numeric(out[feature], errors="coerce").fillna(0).astype("int64")
    out["flow_id"] = [f"{i}" for i in range(len(out))]

    # CIC files contain literal "Infinity" in the rate columns for zero-duration
    # flows; leaving those in poisons every scaler downstream.
    numeric = out.select_dtypes(include=[np.number]).columns
    out[numeric] = out[numeric].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    ordered = list(FLOW_META_COLUMNS) + list(FLOW_FEATURES) + list(PACKET_FEATURES) + ["label"]
    for col in ordered:
        if col not in out.columns:
            out[col] = 0.0
    out = out[ordered].sort_values("start_ts", kind="stable").reset_index(drop=True)
    return FlowFrame(frame=out, dataset=dataset, available=available, source=str(path))


def _derive_missing(out: pd.DataFrame, available: set[str]) -> None:
    """Fill in canonical features that follow arithmetically from present ones."""
    def has(*names: str) -> bool:
        return all(n in available for n in names)

    if has("fwd_packets", "bwd_packets") and "total_packets" not in available:
        out["total_packets"] = out["fwd_packets"] + out["bwd_packets"]
        available.add("total_packets")
    if has("fwd_bytes", "bwd_bytes") and "total_bytes" not in available:
        out["total_bytes"] = out["fwd_bytes"] + out["bwd_bytes"]
        available.add("total_bytes")

    tot_p = out["total_packets"].replace(0, np.nan)
    tot_b = out["total_bytes"].replace(0, np.nan)
    if "total_bytes" in available and "total_packets" in available:
        if "bytes_ratio" not in available:
            out["bytes_ratio"] = (out["fwd_bytes"] / tot_b).fillna(0.0)
            available.add("bytes_ratio")
        if "packets_ratio" not in available:
            out["packets_ratio"] = (out["fwd_packets"] / tot_p).fillna(0.0)
            available.add("packets_ratio")
        if "mean_packet_size" not in available:
            out["mean_packet_size"] = (out["total_bytes"] / tot_p).fillna(0.0)
            available.add("mean_packet_size")

    for ratio, count in (("syn_ratio", "syn_count"), ("rst_ratio", "rst_count"),
                         ("ack_ratio", "ack_count"), ("psh_ratio", "psh_count")):
        if count in available and ratio not in available:
            out[ratio] = (out[count] / tot_p).fillna(0.0)
            available.add(ratio)

    if "dst_port" in available and "is_wellknown_dst_port" not in available:
        from .flows import WELL_KNOWN_PORTS
        out["is_wellknown_dst_port"] = out["dst_port"].isin(WELL_KNOWN_PORTS).astype(float)
        available.add("is_wellknown_dst_port")


def _parse_timestamps(series: pd.Series) -> pd.Series:
    """Coerce a timestamp column to epoch seconds.

    CIC files are ``DD/MM/YYYY HH:MM:SS`` with an inconsistent 12/24-hour clock;
    UNSW and CTU use epoch seconds directly.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() > 0.9:
        return numeric.astype("float64").fillna(0.0)
    try:
        parsed = pd.to_datetime(series, errors="coerce", dayfirst=True, format="mixed")
    except (TypeError, ValueError):
        return pd.Series(np.arange(len(series), dtype="float64"), index=series.index)
    if parsed.isna().all():
        # no parseable clock at all: fall back to row order, which still gives
        # the windowing a monotonic axis to work with
        return pd.Series(np.arange(len(series), dtype="float64"), index=series.index)
    # pandas infers the datetime64 resolution from the input, so it can be us,
    # ms or ns depending on the file. Pin it to nanoseconds before converting,
    # otherwise the epoch comes out scaled by a factor of 1000 and every window
    # boundary downstream is silently wrong.
    epoch_ns = parsed.astype("datetime64[ns]").astype("int64")
    seconds = (epoch_ns / 1e9).astype("float64")
    return seconds.where(parsed.notna(), 0.0)
