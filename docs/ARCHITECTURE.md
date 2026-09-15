# ATDRPS — Architecture

**Advanced Threat Detection and Reinforcement Protocol System**
SIH 2026 · PS 26153 (NTRO) · Team Bell Labs, Lloyd Institute of Engineering & Technology

---

## 1. The problem, restated

A conventional IDS asks *"is this flow malicious?"* — one flow at a time, discarding the
temporal and causal structure. Infiltration is not a packet; it is a **process** that
unfolds across time. A slow port scan stays under every per-flow threshold. A beacon on a
60-second timer is, flow by flow, indistinguishable from a health check.

ATDRPS asks a different question: **where is this network heading?** It learns the
transition dynamics of network state, `P(S_t+1 | S_≤t)`, and rolls that model forward K
steps to estimate infiltration probability *before* the kill chain completes.

## 2. Pipeline

```
PCAP / PCAPNG ──┐
                ├─► packet parse ─► flow assembly ─► dual-level features ─┐
flow CSV      ──┘   (in-tree)       (5-tuple,          (flow + packet)    │
                                     timeouts, teardown)                  │
                                                                          ▼
                                                            30 s time windowing
                                                                          │
                                                          96-dim network state S_t
                                                                          │
                          ┌───────────────────────────────────────────────┤
                          ▼                                               ▼
              temporal transformer encoder                     MITRE stage labels
              (L = 16 context windows)                     (dataset labels, attack
                          │                                 timelines, or rules)
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
  next-state Δ       stage head      infiltration head
        │                 │                 │
        └────────► K-step autoregressive rollout ◄─────┘
                          │
        ┌─────────────────┼──────────────────┐
        ▼                 ▼                  ▼
 probability timeline   ATT&CK stage    SHAP + attention
                          │
                          ▼
                 offline Flask dashboard / CLI
```

## 3. State representation

Each 30-second window becomes a **103-dimensional vector** with three kinds of component.

**Volume and shape (22)** — flow/packet/byte counts, rates, distinct sources,
destinations and ports, protocol mix, internal/external/inbound shares, outbound byte
ratio, top-talker concentration.

**Distributional aggregates (60)** — mean, standard deviation and maximum of 20 per-flow
features across the window. The *spread* carries as much signal as the mean: a window of
beacons has a very small `iat_std`, and that is the whole tell.

**Behavioural detectors (21)** — structure that no single flow contains: ports swept per
source–destination pair, repeated connections to one service, sequential-port ratio,
admin/auth service share, internal peers reached by one host, **beacon regularity**,
repeat external destinations, proportion of never-before-seen destinations, graph degree —
plus seven built for specific attack families: same-port fan-out (worms), half-open ratio
(denial of service), sources converging on one service (credential stuffing, DDoS), DNS
share and DNS query size (tunnelling), TTL inconsistency and RST injection (machine-in-the-
middle). See `docs/ATTACK_COVERAGE.md`.

Per-flow features are dual-level by construction: NetFlow-style aggregates (durations,
byte/packet counts, TCP flag counts and ratios, IAT statistics) **plus** packet-level
statistics that only a full capture can supply (TTL variance, TCP window trace, payload
distribution, retransmissions, duplicate ACKs, out-of-order segments, fragmentation).

Measured separation on generated traffic:

| feature | benign | attack |
|---|---|---|
| `max_ports_per_src_dst` | 1.3 | 22.8 (port scan) |
| `admin_service_share` | 0.00 | 0.36 (brute force) |
| `beacon_regularity` | 0.66 | 0.92 (C2 beacon) |
| `outbound_bytes_ratio` | 0.01 | 0.52 (exfiltration) |
| `same_port_fanout` (log) | 0.79 | 1.89 (worm) |
| `half_open_ratio` | 0.00 | 0.80 (SYN flood) |
| `dns_query_size` (log) | 4.29 | 5.39 (DNS tunnelling) |
| `ttl_inconsistency` | 0.001 | 0.104 (machine-in-the-middle) |

## 4. The world model

A **temporal transformer** over sequences of states: linear input projection to
`d_model = 128`, learned positional encoding, 3 pre-norm encoder blocks with 4-head
self-attention, and three heads on the final position.

* **Dynamics head** — predicts `S_t+1 − S_t`, a *residual*. Regularisation then means
  "nothing changes", the correct prior for network state. Predicting the absolute state
  instead pulls a regularised model toward the *mean* state, and a rollout that feeds
  mean states back into its own context oscillates within a few steps. This was observed
  and fixed, not assumed.
* **Stage head** — MITRE ATT&CK stage of the **next** window. Six stages: the five the
  problem statement names, plus **Impact** (TA0040) for denial of service and ransomware
  encryption, which are real, are in every dataset, and are not steps toward infiltration.
  Forcing a SYN flood onto "Reconnaissance" because both send many SYNs would corrupt the
  transition dynamics the model exists to learn.
* **Infiltration head** — binary probability for the next window.

Training all three together is the point. The stage head alone is a classifier with extra
steps; forcing one representation to also reconstruct the next state is what makes the
encoder learn dynamics rather than a signature lookup.

Attention is bidirectional **within the observed context**, which is safe: every context
position already precedes the target. The leakage that matters — context to target — is
prevented by construction, since the target window is never fed to the encoder.

**K-step forward simulation** is autoregressive: predict, append, re-predict. Predicted
states are clamped to the range the training data covered, so the simulation cannot
wander into states no network has ever produced.

A **linear-dynamics backend** (ridge + logistic heads) implements the same interface. It
keeps the whole pipeline runnable without PyTorch and, more importantly, answers the
question a sceptical judge should ask: *does the transformer buy anything over a
well-specified linear model?*

## 5. Explainability

Mandatory, and answered as two separate questions.

**What drove it** — grouped KernelSHAP over the 96 features. Masking a feature substitutes
its whole trace across the context from a background of real windows, so attributions are
per *measurement*, at the level an analyst asks about. Implemented in-tree, with the
efficiency constraint imposed exactly by variable elimination; verified against the
closed-form Shapley values of a linear model to 1.2 × 10⁻⁵.

**When it came from** — transformer attention over the 16 context windows, with
model-agnostic temporal occlusion as a cross-check and as the answer for the linear
backend.

Output is in plain language: *"share of traffic to remote-access services +0.216"*, not
`auth_service_share=0.36`. Interpretable ATT&CK rules are scored alongside as
corroboration.

## 6. Results

Held-out captures; train and test share no window, flow or campaign. Thresholds chosen on
validation, applied unchanged to test.

| model | F1 | Precision | Recall | FPR | Stage acc. |
|---|---|---|---|---|---|
| logistic regression (static) | 0.8376 | 0.8417 | 0.8335 | 0.1660 | 0.7556 |
| logistic regression (context) | 0.8947 | 0.9128 | 0.8773 | 0.0888 | 0.8013 |
| **ATDRPS world model** | **0.9214** | **0.9317** | **0.9113** | **0.0708** | **0.8400** |

The world model wins at **every** horizon step (step 5: F1 0.842 vs 0.811) and less than
half the false-positive rate of the static classifier — precisely what a snapshot
classifier cannot do.

Per stage, one window ahead: Command & Control 0.848, Exfiltration 0.859, Benign 0.896,
Reconnaissance 0.798, Initial Access 0.790, Lateral Movement 0.784, **Impact 0.500**. The
last is the weak one, on 32 test windows — too rare in this corpus to learn well, and
reported rather than hidden.

**Dynamics check.** Next-state error 0.611 against the persistence floor of 1.006. A model
that beats a baseline on stage F1 but cannot beat "assume nothing changes" has learned a
classifier, not dynamics; this one clears the floor.

## 7. Deployment posture

Fully offline: no cloud APIs, no telemetry, no runtime downloads. ATDRPS ships **its own
pcap/pcapng parser** and **its own KernelSHAP**, so `scapy`, `pyshark` and `shap` are not
required — one less supply-chain surface inside an air-gapped CII network. The dashboard
is a single Flask process with every asset inlined.

Runtime on a 79,485-packet capture: parse, assemble 2,012 flows, window, score 82 windows,
forward-simulate and explain in **3.9 seconds**.
