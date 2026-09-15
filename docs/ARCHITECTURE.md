# ATDRPS — Architecture

**Advanced Threat Detection and Reinforcement Protocol System**
SIH 2026 · PS 26153 (NTRO) · Team Bell Labs, Lloyd Institute of Engineering & Technology

---

## 1. The problem, restated

A conventional IDS asks *"is this flow malicious?"* — one flow at a time, discarding
temporal structure. A slow port scan stays under every per-flow threshold; a beacon on a
60-second timer is, flow by flow, indistinguishable from a health check.

ATDRPS asks a different question: **where is this network heading?** It learns the
transition dynamics of network state, `P(S_t+1 | S_≤t)`, and rolls that model forward K
steps to estimate infiltration probability *before* the kill chain completes.

## 2. Pipeline

```
PCAP/PCAPNG or flow CSV → packet parse → flow assembly → dual-level features
   → 30s time windowing → 103-dim state S_t → temporal transformer (16-window
   context) → {next-state Δ, stage head, infiltration head} → K-step
   autoregressive rollout → {probability timeline, ATT&CK stage, SHAP +
   attention} → offline Flask dashboard / CLI
```

Parsing, flow assembly and feature extraction are all in-tree (no scapy/pyshark). MITRE
stage labels for training come from dataset labels, attack timelines, or rules.

## 3. State representation

Each 30-second window becomes a **103-dimensional vector**: volume/shape features (flow,
packet and byte counts and rates, protocol mix, internal/external/inbound shares,
top-talker concentration — 22 dims); distributional aggregates — mean, std and max of 20
per-flow features across the window, since *spread* carries signal a mean alone does not
(a beacon window has near-zero `iat_std`, and that is the whole tell — 60 dims); and 21
behavioural detectors that no single flow contains — ports swept per src–dst pair, beacon
regularity, admin/auth service share, never-before-seen destination ratio, plus seven
built for specific families: same-port fan-out (worms), half-open ratio (denial of
service), converging sources (credential stuffing/DDoS), DNS share/query size
(tunnelling), TTL inconsistency and RST injection (machine-in-the-middle). Full list in
`docs/ATTACK_COVERAGE.md`.

Per-flow features are dual-level by construction: NetFlow-style aggregates (duration,
byte/packet counts, TCP flag ratios, IAT statistics) **plus** packet-level statistics only
a full capture supplies (TTL variance, TCP window trace, retransmissions, duplicate ACKs,
out-of-order segments, fragmentation). Measured separation, benign vs. attack, on
generated traffic: `max_ports_per_src_dst` 1.3 vs 22.8 (port scan), `beacon_regularity`
0.66 vs 0.92 (C2), `outbound_bytes_ratio` 0.01 vs 0.52 (exfiltration), `half_open_ratio`
0.00 vs 0.80 (SYN flood).

## 4. The world model

A **temporal transformer** over sequences of states: linear input projection to
`d_model = 128`, learned positional encoding, 3 pre-norm encoder blocks with 4-head
self-attention, and three heads on the final position.

* **Dynamics head** — predicts `S_t+1 − S_t`, a *residual*, so regularisation means
  "nothing changes" — the correct prior for network state. Predicting the absolute state
  instead pulls a regularised model toward the *mean* state, and a rollout that feeds mean
  states back into its own context oscillates within a few steps. Observed and fixed.
* **Stage head** — MITRE ATT&CK stage of the **next** window. Six stages: the five the
  problem statement names, plus **Impact** (TA0040) for denial of service and ransomware
  encryption, which are real, are in every dataset, and are not steps toward infiltration.
* **Infiltration head** — binary probability for the next window.

Training all three together is the point: forcing one representation to also reconstruct
the next state is what makes the encoder learn dynamics rather than a signature lookup.
Attention is bidirectional **within the observed context** only — the target window is
never fed to the encoder, so there is no context-to-target leakage.

**K-step forward simulation** is autoregressive: predict, append, re-predict. Predicted
states are clamped to the range the training data covered, so the simulation cannot
wander into states no network has ever produced.

A **linear-dynamics backend** (ridge + logistic heads) implements the same interface,
keeps the pipeline runnable without PyTorch, and answers the question a sceptical judge
should ask: *does the transformer buy anything over a well-specified linear model?* This
is the backend the shipped `artifacts/model-linear` and the benchmark table below use.

## 5. Explainability

Mandatory, answered as two questions. **What drove it** — grouped KernelSHAP over the
state features, masking a feature's whole trace across the context against a background
of real windows, so attributions are per *measurement*, at the level an analyst asks
about. Implemented in-tree; verified against the closed-form Shapley values of a linear
model to 1.2 × 10⁻⁵. **When it came from** — transformer attention over the context
windows, with model-agnostic temporal occlusion as a cross-check and as the answer for the
linear backend.

Output is plain language: *"share of traffic to remote-access services +0.216"*, not
`auth_service_share=0.36`. Interpretable ATT&CK rules are scored alongside as
corroboration.

## 6. Results

Held-out captures; train and test share no window, flow or campaign. Thresholds chosen on
validation, applied unchanged to test.

| model | F1 | Precision | Recall | FPR | Stage acc. |
|---|---|---|---|---|---|
| logistic regression (static) | 0.8477 | 0.8275 | 0.8688 | 0.1918 | 0.7619 |
| logistic regression (context) | 0.8946 | 0.9075 | 0.8821 | 0.0952 | 0.7956 |
| **ATDRPS world model** | **0.9294** | **0.9306** | **0.9283** | **0.0734** | **0.8400** |

Wins at **every** horizon step (step 5: F1 0.844 vs 0.815) at less than half the
false-positive rate of the static classifier. Weakest on **Impact** (F1 0.491, 32 test
windows — too rare in this corpus to learn well, reported rather than hidden). Full
per-stage and per-horizon breakdown in `docs/BENCHMARKS.md`.

**Dynamics check.** Next-state error 0.611 against the persistence floor of 1.007 — a
model that beats a baseline on stage F1 but cannot beat "assume nothing changes" has
learned a classifier, not dynamics; this one clears the floor.

## 7. Deployment posture

Fully offline: no cloud APIs, no telemetry, no runtime downloads. ATDRPS ships **its own
pcap/pcapng parser** and **its own KernelSHAP**, so `scapy`, `pyshark` and `shap` are not
required — one less supply-chain surface inside an air-gapped CII network. The dashboard
is a single Flask process with every asset inlined, and input is a PCAP/PCAPNG capture or
a flow CSV, matching the problem statement's stated input format directly.

Runtime on an 88,767-packet capture: parse, assemble 4,737 flows, window, score 80
windows, forward-simulate and explain in under 4 seconds.
