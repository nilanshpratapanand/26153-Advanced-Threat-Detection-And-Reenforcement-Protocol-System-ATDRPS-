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

PCAP/PCAPNG or flow CSV → packet parse → flow assembly → dual-level features → 30s
windowing → 103-dim state `S_t` → temporal transformer over a 16-window context →
{next-state Δ, stage head, infiltration head} → K-step autoregressive rollout →
{probability timeline, ATT&CK stage, SHAP + attention} → offline dashboard or CLI.

Parsing, flow assembly and feature extraction are all in-tree (no scapy/pyshark). MITRE
stage labels for training come from dataset labels, attack timelines, or rules.

## 3. State representation

Each 30-second window becomes a **103-dimensional vector**: volume and shape (22 dims);
distributional aggregates — mean, std and max of 20 per-flow features, since *spread*
carries signal a mean alone does not, a beacon window having near-zero `iat_std` (60);
and 21 behavioural detectors no single flow contains — ports swept per src–dst pair,
beacon regularity, admin-service share, plus seven built for specific families
(same-port fan-out for worms, half-open ratio for DoS, converging sources for credential
stuffing, DNS query size for tunnelling, TTL inconsistency for MitM). Full list in
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
  "nothing changes", the correct prior for network state. Predicting the absolute state
  pulls a regularised model toward the *mean* state, and a rollout feeding mean states
  back into its own context oscillates within a few steps. Observed and fixed.
* **Stage head** — MITRE ATT&CK stage of the **next** window: the five the problem
  statement names, plus **Impact** (TA0040) for denial of service and ransomware, which
  are real, in every dataset, and not steps toward infiltration.
* **Infiltration head** — binary probability for the next window.

Training all three together is the point: forcing one representation to also reconstruct
the next state is what makes the encoder learn dynamics rather than a signature lookup.
Attention is bidirectional **within the observed context** only — the target window is
never fed to the encoder, so there is no leakage.

**K-step forward simulation** is autoregressive: predict, append, re-predict. Predicted
states are clamped to the range the training data covered, so the simulation cannot
wander into states no network has ever produced.

A **linear-dynamics backend** (ridge + logistic heads) implements the same interface,
keeps the pipeline runnable without PyTorch, and answers the question a sceptical judge
should ask: *does the transformer buy anything over a well-specified linear model?* It is
what ships as `artifacts/model-linear`.

## 5. Explainability

Mandatory, answered as two questions. **What drove it** — grouped KernelSHAP over the
state features, masking a feature's whole trace across the context against a background
of real windows, so attributions are per *measurement*. Implemented in-tree; verified
against the closed-form Shapley values of a linear model to 1.2 × 10⁻⁵. **When it came
from** — attention over the context windows, with model-agnostic occlusion as a
cross-check and as the answer for the linear backend. Output is plain language: *"share
of traffic to remote-access services +0.216"*, not `auth_service_share=0.36`, with
interpretable ATT&CK rules scored alongside as corroboration.

## 6. Results

Held-out captures; train and test share no window, flow or campaign. Thresholds chosen on
validation, applied unchanged to test.

| model | F1 | FPR | Stage acc. | Next-state MSE |
|---|---|---|---|---|
| logreg (static) | 0.8477 | 0.1918 | 0.7625 | — |
| logreg (context) | 0.8948 | 0.0965 | 0.7950 | — |
| **linear dynamics** | **0.9294** | **0.0734** | 0.8400 | **0.6114** |
| temporal transformer | 0.9163 | 0.0901 | **0.8556** | 0.7544 |

Both world models beat every static baseline and both clear the persistence floor of
1.0068 — a model that wins on stage F1 but cannot beat "assume nothing changes" has
learned a classifier, not dynamics.

**The transformer does not straightforwardly win.** It takes MITRE stage classification,
the harder seven-class problem, by +1.6 accuracy and +2.2 macro-F1 (0.8030 vs 0.7807),
and loses infiltration F1, FPR and next-state error. The training curve says why:
validation loss bottomed at epoch 3 and rose for eight more while training loss fell
36%, early-stopping at epoch 11 of 40. 4,960 sequences is not enough for a 128-dim
three-layer encoder — an overfitting result, not a capacity ceiling. So the linear
backend ships as default and the transformer is reported beside it, not dropped.

Weakest stage is **Impact** (F1 0.491 on 32 test windows, too rare here to learn —
reported rather than hidden). Full breakdown in `docs/BENCHMARKS.md`.

## 7. Deployment posture

Fully offline: no cloud APIs, no telemetry, no runtime downloads. ATDRPS ships **its own
pcap/pcapng parser** and **its own KernelSHAP**, so `scapy`, `pyshark` and `shap` are not
required — one less supply-chain surface inside an air-gapped CII network. Input is a
PCAP/PCAPNG capture or a flow CSV, exactly as the problem statement specifies. An
88,767-packet capture parses, assembles 4,737 flows, windows, scores, forward-simulates
and explains in under 4 seconds; flow assembly shards across cores.
