"""MITRE ATT&CK stage mapping.

Two jobs:

1.  **Dataset label -> stage.**  Public datasets label flows with attack family
    names ("SSH-Bruteforce", "Infilteration", "Bot"), not with ATT&CK tactics.
    :func:`stage_from_label` does that translation.

2.  **Traffic -> stage, without labels.**  When ATDRPS is pointed at an
    unlabelled capture there is nothing to translate, so
    :func:`score_stages` scores a window against interpretable rules derived
    from the tactic definitions.  These rules are also what make the weak
    supervision in :mod:`atdrps.data.windows` possible, which is how we get
    fine-grained stage labels out of datasets that only publish coarse ones.

An honest scope note
--------------------
The problem statement names five stages: Reconnaissance, Initial Access,
Lateral Movement, Command & Control, Exfiltration.  Real captures also contain
denial of service and ransomware encryption, which are ATT&CK **Impact**
(TA0040) and are not steps on the path to infiltration.  Those get their own
stage rather than being forced onto the nearest of the five -- a SYN flood is
not reconnaissance, however many SYNs it sends, and training a forecaster to
believe otherwise would corrupt exactly the transition dynamics we are trying
to learn.

:data:`OUT_OF_SCOPE` remains for what genuinely does not belong in a
traffic-based kill chain, and for unrecognised labels.  Windows dominated by
those are excluded from the stage loss by a mask rather than mislabelled.

See ``docs/ATTACK_COVERAGE.md`` for the full attack-by-attack breakdown of what
is visible in network traffic and what is not.
"""

from __future__ import annotations

import re

import numpy as np

from .schema import BENIGN_STAGE, DEFAULT_INFILTRATION_STAGES, STAGES

__all__ = [
    "_rule_support",
    "OUT_OF_SCOPE", "MITRE_TACTICS", "STAGE_TECHNIQUES", "LABEL_PATTERNS",
    "stage_from_label", "describe_stage", "score_stages", "RULES",
]

OUT_OF_SCOPE = "OutOfScope"

# Tactic identifiers, so the UI and the report can cite ATT&CK properly.
MITRE_TACTICS: dict[str, str] = {
    "Reconnaissance": "TA0043",
    "InitialAccess": "TA0001",
    "LateralMovement": "TA0008",
    "CommandAndControl": "TA0011",
    "Exfiltration": "TA0010",
    "Impact": "TA0040",
    OUT_OF_SCOPE: "-",
    BENIGN_STAGE: "-",
}

# The techniques each stage's rules are actually looking for.
STAGE_TECHNIQUES: dict[str, tuple[tuple[str, str], ...]] = {
    "Reconnaissance": (
        ("T1595.001", "Active Scanning: Scanning IP Blocks"),
        ("T1046", "Network Service Discovery"),
    ),
    "InitialAccess": (
        ("T1110", "Brute Force"),
        ("T1190", "Exploit Public-Facing Application"),
    ),
    "LateralMovement": (
        ("T1021.002", "Remote Services: SMB/Windows Admin Shares"),
        ("T1021.001", "Remote Services: Remote Desktop Protocol"),
        ("T1021.004", "Remote Services: SSH"),
    ),
    "CommandAndControl": (
        ("T1071.001", "Application Layer Protocol: Web Protocols"),
        ("T1573", "Encrypted Channel"),
        ("T1008", "Fallback Channels"),
    ),
    "Exfiltration": (
        ("T1041", "Exfiltration Over C2 Channel"),
        ("T1048", "Exfiltration Over Alternative Protocol"),
        ("T1030", "Data Transfer Size Limits"),
    ),
    "Impact": (
        ("T1486", "Data Encrypted for Impact"),
        ("T1498", "Network Denial of Service"),
        ("T1499", "Endpoint Denial of Service"),
    ),
}

# Dataset attack-family names -> stage.  Ordered: the first pattern that matches
# wins, so put the specific ones first.
LABEL_PATTERNS: tuple[tuple[str, str], ...] = (
    # --- benign
    (r"^(benign|normal|background|-)$", BENIGN_STAGE),
    # --- reconnaissance
    (r"portscan|port scan|port-scan|reconnaissance|^scan|probe|analysis|heartbleed", "Reconnaissance"),
    (r"nmap|ipsweep|portsweep|satan|mscan|saint", "Reconnaissance"),
    (r"mitm|man.?in.?the.?middle|arp.?spoof|arp.?poison|dns.?spoof|cache.?poison",
     "Reconnaissance"),
    # --- self-propagating SMB exploits: these must be matched BEFORE the
    #     generic "exploit" pattern below, or EternalBlue resolves to initial
    #     access. First match wins, so specific goes first.
    (r"worm|conficker|sasser|blaster|wannacry.?spread|eternalblue|smb.?exploit",
     "LateralMovement"),
    # --- initial access
    (r"brute ?-?force|bruteforce|patator|password|guess", "InitialAccess"),
    (r"credential.?stuff|cred.?stuff|account.?takeover|password.?spray", "InitialAccess"),
    (r"web attack|xss|sql ?injection|sqli|infilt|exploit|shellcode|backdoor(?!.*c2)", "InitialAccess"),
    (r"^r2l$|warezclient|warezmaster|imap|multihop|phf|spy", "InitialAccess"),
    # --- lateral movement (worms propagate laterally by definition)
    (r"lateral|^u2r$|rootkit|buffer_overflow|loadmodule|perl|smb|psexec", "LateralMovement"),
    # --- command and control
    (r"\bbot\b|botnet|^bot$|c&?c|command.?and.?control|beacon|irc|mirai|zeus|neris|rbot|virut",
     "CommandAndControl"),
    (r"spyware|keylog|rat\b|remote.?access.?tro", "CommandAndControl"),
    (r"trojan|worm(?!.*dos)", "CommandAndControl"),
    # --- exfiltration (DNS tunnelling is exfiltration over an alternative protocol)
    (r"exfil|data ?theft|data ?leak", "Exfiltration"),
    (r"dns.?tunnel|tunnel|iodine|dnscat|covert.?channel", "Exfiltration"),
    # --- Impact (TA0040): real attacks, but not steps toward infiltration
    (r"ransom|wannacry|locky|cryptolocker|encrypt.?for.?impact|wiper", "Impact"),
    (r"ddos|dos ?attack|^dos|goldeneye|slowloris|slowhttptest|hulk|\bloic\b|\bhoic\b", "Impact"),
    (r"flood|syn.?flood|udp.?lag|^apache", "Impact"),
    # --- genuinely outside a traffic-based kill chain
    (r"fuzzers|generic|analysis|shellcode\b(?!.*access)", OUT_OF_SCOPE),
)

_COMPILED = tuple((re.compile(pattern, re.IGNORECASE), stage) for pattern, stage in LABEL_PATTERNS)


def stage_from_label(label: str) -> str:
    """Map a dataset attack-family label onto a stage.

    Unrecognised non-benign labels return :data:`OUT_OF_SCOPE`, never a guess --
    a wrong stage is worse than an excluded one when the model is learning
    transitions between stages.
    """
    text = str(label).strip()
    if not text:
        return BENIGN_STAGE
    for pattern, stage in _COMPILED:
        if pattern.search(text):
            return stage
    return OUT_OF_SCOPE


def describe_stage(stage: str) -> str:
    """One-line human description, for the dashboard and the CLI report."""
    tactic = MITRE_TACTICS.get(stage, "-")
    if stage == BENIGN_STAGE:
        return "Benign - no attack behaviour indicated"
    if stage == OUT_OF_SCOPE:
        return ("Out of scope - not observable as a kill-chain stage in network "
                "traffic (see docs/ATTACK_COVERAGE.md)")
    techniques = ", ".join(f"{tid} {name}" for tid, name in STAGE_TECHNIQUES.get(stage, ()))
    return f"{stage} ({tactic}) - {techniques}"


# --------------------------------------------------------------------- rules
# Each rule is (stage, feature, comparison, threshold, weight, human phrasing).
# Kept as data rather than code so the dashboard can print exactly the rules
# that fired, and so thresholds are auditable in one place.
RULES: tuple[tuple[str, str, str, float, float, str], ...] = (
    # ---- Reconnaissance: many short, empty, one-sided probes across many ports
    ("Reconnaissance", "distinct_dst_ports", ">", 20, 1.4, "probing more than 20 distinct destination ports"),
    ("Reconnaissance", "mean_payload_zero_ratio", ">", 0.7, 1.2, "probe packets carry no payload"),
    ("Reconnaissance", "mean_flow_packets", "<", 6, 0.8, "conversations end after a handful of packets"),
    ("Reconnaissance", "rst_share", ">", 0.2, 1.0, "a high share of connections are refused with RST"),
    ("Reconnaissance", "max_ports_per_src_dst", ">", 15, 1.3, "one source sweeping many ports on one host"),

    # ---- Initial Access: repeated connections to one auth service, small payloads
    ("InitialAccess", "max_flows_per_src_dst_port", ">", 12, 1.4, "repeated connections to a single service"),
    ("InitialAccess", "auth_service_share", ">", 0.3, 1.2, "traffic concentrated on remote-access services"),
    ("InitialAccess", "rst_share", ">", 0.15, 0.7, "many attempts refused"),
    ("InitialAccess", "distinct_dst_ports", "<", 12, 0.6, "targeting few ports, not sweeping"),

    # ---- Lateral Movement: internal-to-internal, admin services, retransmissions
    ("LateralMovement", "internal_flow_share", ">", 0.6, 1.3, "traffic is internal host to internal host"),
    ("LateralMovement", "admin_service_share", ">", 0.25, 1.4, "SMB/RDP/SSH/WMI activity"),
    ("LateralMovement", "distinct_internal_dsts", ">", 2, 1.1, "one host reaching several internal peers"),
    ("LateralMovement", "mean_retransmission_ratio", ">", 0.02, 0.6, "retransmissions typical of pushed tooling"),

    # ---- Command and Control: low-volume, regular, long-lived outbound
    ("CommandAndControl", "beacon_regularity", ">", 0.88, 2.0, "near-constant interval between outbound contacts"),
    ("CommandAndControl", "mean_flow_bytes", "<", 6000, 0.8, "each contact moves very little data"),
    ("CommandAndControl", "external_flow_share", ">", 0.4, 0.7, "destination is outside the network"),
    ("CommandAndControl", "repeat_external_dst_count", ">", 3, 1.2, "the same external host contacted repeatedly"),

    # ---- Exfiltration: sustained, strongly outbound bulk transfer
    ("Exfiltration", "outbound_bytes_ratio", ">", 0.85, 1.6, "almost all bytes are leaving the network"),
    ("Exfiltration", "total_bytes", ">", 2_000_000, 1.3, "bulk volume moved in a single window"),
    ("Exfiltration", "mean_payload_mean", ">", 800, 0.9, "packets are consistently near MTU"),
    ("Exfiltration", "external_flow_share", ">", 0.4, 0.6, "destination is outside the network"),

    # ---- Impact: volumetric denial of service, or mass encryption over SMB
    ("Impact", "half_open_ratio", ">", 0.5, 1.6, "most connections never complete a handshake"),
    ("Impact", "max_srcs_per_dst_service", ">", 8, 1.4, "many sources converging on one service"),
    ("Impact", "n_flows", ">", 400, 1.0, "an extreme number of conversations in one window"),
    ("Impact", "mean_flow_packets", "<", 5, 0.6, "conversations end almost immediately"),
)


def _rule_support(value: float, op: str, threshold: float, band: float = 0.18) -> float:
    """How well one rule is satisfied, in ``[0, 1]`` -- a ramp, not a cliff.

    A hard ``value > threshold`` makes the whole stage score flicker when a
    noisy feature sits near its cutoff. Measured on benign background traffic,
    ``beacon_regularity`` ranges over 0.72-0.79 window to window; against a
    binary cutoff anywhere in that range the command-and-control score
    oscillates between 0.26 and 0.68 on identical traffic, which then decides
    whether a forecast is called CONFIRMED or not.

    Ramping over a band around the threshold keeps the ordering ("more is more
    suspicious") while making the score continuous, so a borderline value
    contributes borderline evidence instead of all or nothing.
    """
    if threshold == 0:
        lo, hi = -band, band
    else:
        span = abs(threshold) * band
        lo, hi = threshold - span, threshold + span
    if op == ">":
        if hi <= lo:
            return 1.0 if value > threshold else 0.0
        return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))
    # "<" -- support grows as the value falls below the threshold
    if hi <= lo:
        return 1.0 if value < threshold else 0.0
    return float(np.clip((hi - value) / (hi - lo), 0.0, 1.0))


def score_stages(window: dict[str, float], graded: bool = True) -> dict[str, float]:
    """Score one window's summary statistics against :data:`RULES`.

    Returns a score per stage in ``[0, 1]`` (fraction of that stage's rule
    weight satisfied).  This is a weak labeller and a sanity check, not the
    model -- the world model learns transition dynamics that no threshold set
    can express.  Keeping it around matters anyway: it is the fallback when a
    capture has no labels, and it is the thing a sceptical analyst can read.
    """
    totals: dict[str, float] = {}
    hits: dict[str, float] = {}
    for stage, feature, op, threshold, weight, _text in RULES:
        totals[stage] = totals.get(stage, 0.0) + weight
        value = window.get(feature)
        if value is None or not np.isfinite(value):
            continue
        if graded:
            support = _rule_support(float(value), op, float(threshold))
        else:
            support = 1.0 if ((value > threshold) if op == ">" else (value < threshold)) else 0.0
        if support > 0:
            hits[stage] = hits.get(stage, 0.0) + weight * support
    return {stage: hits.get(stage, 0.0) / total for stage, total in totals.items() if total}


def explain_rules(window: dict[str, float], stage: str) -> list[str]:
    """The human-readable rules that fired for ``stage`` on this window."""
    out = []
    for rule_stage, feature, op, threshold, _weight, text in RULES:
        if rule_stage != stage:
            continue
        value = window.get(feature)
        if value is None or not np.isfinite(value):
            continue
        if (value > threshold) if op == ">" else (value < threshold):
            out.append(f"{text} ({feature}={value:.4g} {op} {threshold:g})")
    return out


def infiltration_flag(stage: str, infiltration_stages=DEFAULT_INFILTRATION_STAGES) -> int:
    return 1 if stage in infiltration_stages else 0
