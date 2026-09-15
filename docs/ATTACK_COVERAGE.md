# Attack coverage — what ATDRPS can and cannot see

ATDRPS observes **network traffic**: flow records and packet headers. That is the input the
problem statement specifies, and it decides what is detectable. An attack is visible here if
it *changes how the network behaves*. An attack that lives entirely inside a payload, an
inbox, or a human being is not.

Overstating this would be the easiest way to lose credibility in a review, so this document
is explicit about the three cases:

| | meaning |
|---|---|
| **Full** | the attack has a distinct signature in flow + packet features, and ATDRPS models it as a stage |
| **Partial** | the attack is visible only through its *side effects* or *consequences*, not the act itself |
| **Out of scope** | not observable from traffic headers; needs payload inspection, email security, or endpoint telemetry |

A separate and more important point runs through the whole table: **ATDRPS forecasts
trajectories, not signatures.** Several attacks below are only "partial" as isolated events
yet are handled well as *stages in a chain* — a zero-day has no signature by definition, but
the lateral movement that follows it does, and that is what gets predicted.

---

## 1. Social Engineering

| Attack | What it looks like in traffic | Coverage | ATT&CK |
|---|---|---|---|
| **Phishing** | Nothing in the delivery itself — SMTP/IMAP flows look ordinary. The *click* shows as a new external destination never seen before, often a freshly registered domain, followed by a credential POST. | **Partial** — `new_dst_ratio`, `new_dst_port_ratio`, outbound POST-shaped flows | T1566 |
| **Spear phishing** | Same as above; targeting is invisible at network level. | **Partial** | T1566.001 |
| **Baiting / pretexting** (malicious USB) | Nothing at delivery — the device is physical. Becomes visible the moment the implant calls home: a beacon from a host that has never made that kind of connection. | **Partial → Full once it beacons** — `beacon_regularity`, `repeat_external_dst_count` | T1091 |

**Honest position.** ATDRPS does not detect phishing. It detects *what happens after
someone falls for it*, which is the part a network sensor can see and the part where
intervention is still possible. Phishing itself belongs to email security and user training —
both listed in the defensive controls section of the source document.

---

## 2. Malware

| Attack | What it looks like in traffic | Coverage | ATT&CK |
|---|---|---|---|
| **Ransomware** | A three-part trajectory: mass SMB access from one host to many shares, a burst of write-heavy flows with near-MTU payloads and high entropy, then often exfiltration before encryption (double extortion). SMB connection spikes are the standard network-level indicator. | **Full** | T1486 (Impact), preceded by T1021.002 |
| **Trojans** | Periodic, low-volume outbound contact to a stable external host — the beacon. Individually unremarkable; the tell is the near-constant interval across many windows. | **Full** | T1071 |
| **Spyware / keyloggers** | Small, regular outbound transfers; low byte volume, high regularity, strongly asymmetric. | **Full** | T1041 |
| **Worms** | The most network-visible attack there is: one source contacting **many destinations on the same port** in a short window, with a high proportion of connections refused. Fan-out on a fixed port is the classic signature. | **Full** | T1210 |

Ransomware and worms are why the state vector carries `same_port_fanout`,
`distinct_internal_dsts` and `admin_service_share` rather than volume alone.

---

## 3. Network & Infrastructure

| Attack | What it looks like in traffic | Coverage | ATT&CK |
|---|---|---|---|
| **DoS / DDoS** | Unmistakable: extreme packet rate, many sources converging on one destination, very high SYN-to-ACK ratio (half-open connections), tiny payloads. | **Full** — but see the note below | T1498 (Impact) |
| **Man-in-the-Middle** | ARP-level spoofing is not in IP headers, but the consequences are: inconsistent TTLs for the same source, injected RSTs, and duplicated sequence numbers across a session. | **Partial** | T1557 |
| **DNS spoofing / cache poisoning / tunnelling** | Spoofing shows as responses from unexpected resolvers and a spike in DNS volume. Tunnelling is clearer still: thousands of queries per minute from one host, abnormally long subdomains, heavy TXT/NULL record use, unusually large responses, and many NXDOMAIN replies. | **Partial → Full for tunnelling** — `dns_share`, `dns_response_ratio`, query rate | T1071.004 / T1048 |

**The DoS note, and why it matters.** Denial of service is **Impact (TA0040)**, not a step
on the infiltration path. It would be easy — and wrong — to map a SYN flood onto
"Reconnaissance" because both involve many SYNs. ATDRPS keeps `Impact` as its own stage and
**trains on floods deliberately**, so the model learns the difference instead of confusing
volumetric noise with an intrusion in progress. Mislabelling here would corrupt the
transition dynamics the whole project rests on.

---

## 4. Application & Web

| Attack | What it looks like in traffic | Coverage | ATT&CK |
|---|---|---|---|
| **SQL injection** | The payload is invisible without deep inspection. The *shape* is not: many requests to a single endpoint from one source, unusual request-size distribution, and a raised error-response rate (short, uniform replies). | **Partial** | T1190 |
| **Cross-site scripting** | Executes in the victim's browser. Almost nothing at the network layer beyond an unusual request pattern. | **Out of scope** | T1059.007 |
| **Zero-day exploit** | No signature exists — that is the definition. But the exploit is never the goal. What follows it is: a new process reaching out, lateral movement, a beacon, exfiltration. | **Out of scope for the exploit, Full for the aftermath** | varies |

**This is the strongest argument for the approach.** A signature-based IDS cannot detect a
zero-day, ever. A world model does not need to: it forecasts from the *trajectory* the
network is on. Initial access by an unknown exploit still produces lateral movement that
looks like lateral movement — and ATDRPS flags it a window later, which is well before
exfiltration.

WAFs and parameterised queries are the right control for SQLi and XSS, exactly as the source
document says. ATDRPS is a layer behind them, not a replacement.

---

## 5. Password & Authentication

| Attack | What it looks like in traffic | Coverage | ATT&CK |
|---|---|---|---|
| **Brute force** | Many repeated connections from one source to one authentication service, small uniform payloads, a high refusal rate, then one connection that does not get refused. | **Full** — already the `InitialAccess` stage | T1110 |
| **Credential stuffing** | Distinguishable from brute force by its shape: **many sources, few attempts each**, spread across one service. Per-source rate limits miss it; the window-level view does not. | **Full** | T1110.004 |
| **Session hijacking** | A stolen cookie replayed from a different host. Detectable only if the same session is seen from two sources — which needs application logs, not headers. | **Out of scope** | T1539 |

---

## Summary

| Coverage | Count | Attacks |
|---|---|---|
| **Full** | 9 | Ransomware, Trojans, Spyware/Keyloggers, Worms, DoS/DDoS, DNS tunnelling, Brute force, Credential stuffing, Lateral movement / post-exploitation |
| **Partial** | 6 | Phishing, Spear phishing, Baiting, MitM, DNS spoofing, SQL injection |
| **Out of scope** | 3 | XSS, Zero-day exploit *(the exploit itself)*, Session hijacking |

Nine of eighteen are modelled as stages with their own generated training data. Six are
detected through consequences rather than the act. Three genuinely require a different sensor,
and ATDRPS says so rather than claiming them.

## What this changed in the build

The attack list above is not a wish list — it drove concrete work:

* **`Impact` added as a sixth stage**, so denial of service and ransomware encryption have an
  honest home instead of being forced onto the infiltration chain or silently dropped.
* **Seven new attack generators**: worm fan-out, ransomware (SMB burst + encryption + double
  extortion), DDoS flood, DNS tunnelling, credential stuffing, web-application probing, and
  MitM/RST injection. The model now trains on traffic that contains all of them.
* **New state features** carrying the signatures the existing 96 did not:
  `same_port_fanout`, `half_open_ratio`, `max_srcs_per_dst_service`, `dns_share`,
  `dns_response_ratio`, `ttl_inconsistency`, `rst_injection_ratio`.
* **Extended label mapping**, so CIC-IDS2018, CIC-IDS2017, UNSW-NB15 and CTU-13 attack
  families for these categories resolve to the right stage instead of `OutOfScope`.

## Sources

- [DNS Tunneling Detection — Six Techniques, ADHDecode](https://adhdecode.com/network-security/dns-security/dns-tunneling-detection-techniques/) — query-rate spikes, subdomain length, TXT/NULL record use, NXDOMAIN volume
- [Server Message Block (SMB) traffic connection spikes, Splunk Lantern](https://lantern.splunk.com/Security/UCE/Guided_Insights/Threat_hunting/Detecting_a_ransomware_attack/Server_Message_Block_(SMB)_traffic_connection_spikes) — SMB connection spikes as a ransomware indicator
- [Detecting Malicious SMB Activity Using Bro, SANS](https://www.sans.org/reading-room/whitepapers/detection/detecting-malicious-smb-activity-bro-37472) — SMB behaviour at the network layer
- [Traffic Signature Detection for Unknown Internet Worms, ResearchGate](https://www.researchgate.net/publication/238758454_Traffic_Signature_Detection_for_Unknown_Internet_Worms) — worm fan-out and scanning signatures
- [Worm Propagation overview, ScienceDirect](https://www.sciencedirect.com/topics/computer-science/worm-propagation) — propagation dynamics
- [Ransomware detection methods, Vectra AI](https://www.vectra.ai/topics/ransomware-detection) — network-level ransomware behaviour
- [MITRE ATT&CK Enterprise Matrix](https://attack.mitre.org/) — tactic and technique identifiers throughout
- Source attack taxonomy: `attackcybersecurity.docx`, supplied by the team
