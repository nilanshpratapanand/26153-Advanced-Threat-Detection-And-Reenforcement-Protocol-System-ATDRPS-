# ATDRPS v2 — research digest and plan for reliable attack forecasting

Status: living document. Only items marked **[done]** were built and measured in this
repository; everything else is planned. Numbers here are only ever copied from a
command that was actually run (see the `Evidence` column).

## 1. Why v1 was not reliable (measured, not assumed)

Diagnostic run on the v1 pipeline (16 synthetic captures, group split, 180 test sequences):

| Question | Result |
|---|---|
| F1 from copying the *current* label forward (oracle persistence, uses the answer) | 0.898 (agreement 0.928) |
| Share of test windows that are attack windows | 35.6 % |
| Onset-only subset (no attack in the 16-window context): onsets available | **5** of 70 windows |
| Model AUC when no attack in the *last* window | 0.461 (chance), 7 onsets |

Reading: the headline F1 mostly measures detecting an attack that is already under way;
the genuine forecasting cases are too few to support any claim; and the data are far more
attack-dense than any real network. The sample is small, so this is a warning, not a proof
that forecasting skill is zero — but it is enough to say the v1 numbers do not establish
forecasting skill.

## 2. What the literature says (sources verified by search in this session)

| Finding | Source | Consequence for the design |
|---|---|---|
| Ten recurring evaluation pitfalls in security ML: sampling bias, label inaccuracy, data snooping, inappropriate baselines/measures, **base-rate neglect**, lab-only evaluation | Arp et al., *Dos and Don'ts of ML in Computer Security*, USENIX Sec 2022 | Evaluate at realistic prevalence, with strong baselines, on held-out real data |
| Temporal bias (test data older than training data) and spatial bias inflate results; propose time-aware constraints and the AUT robustness metric | Pendlebury et al., *TESSERACT*, USENIX Sec 2019 | Train on earlier captures, test on later; report performance decay |
| The false alarm rate, not the detection rate, limits an IDS because attacks are rare | Axelsson, *The base-rate fallacy…*, ACM TISSEC 2000 | Report precision at assumed real prevalence and alerts/day, not just F1 |
| Six reasons ML is hard in intrusion detection: cost of errors, semantic gap, variability of normal, hard evaluation, adversaries | Sommer & Paxson, *Outside the Closed World*, IEEE S&P 2010 | Prefer detecting *changes in behaviour with explanations*; treat anomaly scores as evidence, not verdicts |
| CIC-IDS2017 has labelling errors and artefact flows (>25 % of flows meaningless in some classes; ~10 % mislabelled) and packet misordering | Engelen et al. 2021; Lanvin et al., CRiSIS 2022 | If real data is used, use the corrected versions and re-derive flows from PCAP |
| Attack prediction is a distinct task family (projection, intrusion prediction, situation forecasting); methods span Markov/Bayesian, time-series, ML | Husák et al., IEEE Commun. Surv. Tutor. 21(1), 2019 | Frame the task as *hazard of onset*, and include Markov and time-series baselines |
| Online, unsupervised ensembles of autoencoders reach detection comparable to offline detectors on CPU-only hardware | Mirsky et al., *Kitsune*, NDSS 2018 | Use a Kitsune-style anomaly baseline |
| Conformal risk control can map an alert budget to a false-positive-bounded threshold under exchangeability | CALIBURN and related 2025–26 preprints (arXiv) | Calibrate the alert threshold on held-out benign data; state the exchangeability assumption |
| LANL multi-source dataset: 58 days, 1.65 B events, red-team labels with ~0.00007 % malicious | csr.lanl.gov | Real prevalence is many orders of magnitude below the synthetic 35 % |
| CALDERA/LADEMU generate labelled data from real adversary emulation; sim-to-real gap is a known limit of synthetic traffic | MITRE CALDERA; LADEMU (FFI); TempoNet, GAN-based generators | Add a real-kernel-stack lab generator and *measure* the sim-to-real gap |

## 3. Design

**What can honestly be promised.** No system can say with certainty that an attack "is going
to happen". What can be built and *measured* is a calibrated **hazard**: P(infiltration onset
within the next K windows | recent behaviour), reported together with the false-alarm rate
and the warning lead time achieved at a stated alert budget.

1. **Evaluation protocol first [done — Phase A].** Onset-conditional task (only windows with no
   attack in the recent past), event-level recall and lead time, prevalence-adjusted
   precision, cluster-bootstrap confidence intervals, time/family-aware splits, and baselines
   including the deployable v1 model.
2. **Harder, more honest simulation [Phase B].** Benign look-alikes that never escalate
   (vulnerability scanners, backup bursts, admin sweeps), aborted campaigns, heavy-tailed
   dwell times between stages, low-and-slow/distributed tempo, per-capture domain
   randomisation, benign-only captures to estimate false-alarm rates precisely.
3. **Real-kernel lab [Phase D].** Real client/server processes and attacker tools on loopback
   addresses, captured with `AF_PACKET`, so the packets come from a real TCP stack. Used as a
   *held-out realism test* (train on simulator, test on lab) to measure the sim-to-real gap.
4. **Hazard model [Phase C].** Multi-scale features of the recent windows, gradient-boosted
   trees, isotonic calibration, conformal alert threshold; compared against persistence,
   generic anomaly detectors (Mahalanobis, isolation forest), and v1.
5. **Real data [Phase E].** Adapters and an `eval-real` command so the same protocol runs on
   CIC-IDS (corrected), UNSW-NB15, CTU-13 and LANL flows as soon as the files are supplied.

## 4. Hard limits (stated up front)

* The build sandbox cannot reach any real-dataset host (CIC, UNSW, CTU, Zenodo, Kaggle,
  HuggingFace all refused). Until real files are provided, **no accuracy claim on real
  networks can be made**, and this document will not make one.
* Simulator results measure the simulator. They are useful for finding bugs and comparing
  methods under controlled difficulty, not for predicting field performance.
* A calibrated hazard at a 1-false-alert-per-day budget will miss a large share of
  campaigns; the protocol reports that trade-off instead of hiding it.

## 5. Evidence log

(Each row names the command that produced it.)

### Phase A — v1 vs simple detectors on the quiet-state onset task

Command: `python -m atdrps.cli corpus --captures 64 --duration 3600 --campaigns 3 --out DIR/corpus_v1sim.npz --workers 4`
then `python scripts/forecast_baselines_v1sim.py DIR`. Context 16 windows of 30 s, horizon K=5,
quiet gap 3, group split by capture (train 31 / val 13 / test 19 captures). Test set: 637 quiet
samples, 165 positive (25.9 %), **35 distinct onsets**, 472 negatives. Thresholds chosen on
validation at a 1 % FPR budget. CIs: 200 capture-level bootstrap draws.

| model | AUC [95 % CI] | AP (chance 0.259) | TPR @ 1 % FPR | event recall | median lead | precision @ onset rate 1e-3 |
|---|---|---|---|---|---|---|
| constant | 0.500 | 0.259 | 0.000 | 0.000 | – | – |
| CUSUM/EWMA on recon indicators | 0.765 [0.694, 0.847] | 0.642 | 0.224 | 0.600 | 90 s | 0.018 |
| Mahalanobis (last window) | 0.685 [0.621, 0.757] | 0.460 | 0.067 | 0.257 | 90 s | 0.005 |
| Isolation Forest (last window) | 0.714 [0.651, 0.785] | 0.528 | 0.073 | 0.314 | 90 s | 0.008 |
| ATDRPS v1 world model | 0.663 [0.579, 0.743] | 0.405 | 0.030 | 0.343 | 90 s | 0.003 |

Reading, with caveats:
* On genuine forecasting samples the deployed v1 model is **worse than a hand-built
  reconnaissance detector** and worse than a stock Isolation Forest. The v1 and CUSUM
  intervals overlap slightly (35 onsets is a small sample), so this is strong evidence,
  not proof, that v1's learned dynamics add nothing here.
* At a realistic onset rate of 1 in 1000 windows every model's precision is below 2 %:
  more than 98 % of alerts would be false. This is the base-rate problem (Axelsson 2000)
  and it is why later phases report alerts/day and precision at assumed prevalence.
* Median lead is 90 s for all models because the simulator's reconnaissance-to-access gap is
  5–60 s: the simulator itself caps the warning time. Phase B replaces that.
* The corpus is 52 % attack windows. This is not a realistic prevalence (Phase B).

### Real-traffic check of v1 (user-supplied laptop capture on hostel Wi-Fi)

Command: `python scripts/real_capture_check.py SCRATCH CAPTURES` with a v1 linear model trained
on the 64-capture simulator corpus above (`atdrps train --backend numpy`). The real capture is
145 minutes, 96,124 frames (54 % IPv4, 46 % ARP, no IPv6), 291 windows of 30 s, one host = 54 %
of packets. It has **no attack labels**; it is assumed mostly benign (not verified).

| capture | windows | raw attack prob. mean / p90 / max | windows >= 0.85 | "CONFIRMED" | median abs z vs simulator benign |
|---|---|---|---|---|---|
| real laptop, 145 min | 291 | 0.94 / 1.00 / 1.00 | 258 | 57 | 1.64 |
| real laptop, 24 s | 1 | (too short for a 16-window context) | – | – | 1.62 |
| crafted "benign_safe" (simulator address plan) | 40 | 0.07 / 0.14 / 0.27 | 0 | 0 | 1.14 |
| crafted "preattack_threshold_085" (simulator address plan) | 80 | 1.00 / 1.00 / 1.00 | 65 | 36 | 1.23 |
| reference: simulator benign windows | – | – | – | – | 0.34 |

Reading: v1 saturates on real traffic. The user's own dashboard run (a different training run)
showed the same failure in milder form (probability 0.3-0.87 across all windows; peak window
= a ZeroTier UDP keepalive plus a 3-packet HTTPS exchange, read as beacon/exfiltration, with
the rule engine "corroborating" because it keys on the same periodicity). Two crafted files
that use the simulator's address plan separate as intended, which shows the tool responds to
traffic built like its training data, and nothing about real attacks.

Consequences for the design: (1) sim-trained absolute thresholds must not be used on a new
network; (2) score windows relative to that network's own recent normal; (3) calibrate the alert
threshold on the network's own benign data; (4) evaluate with real background traffic plus
injected, labelled attacks (semi-synthetic overlay) so the false-alarm side is measured on
real data; (5) ARP (46 % of this capture) is currently discarded by the flow pipeline.

### Phase B/C — real laptop background + injected campaigns (two tasks, both reported)

Command: `python scripts/forecast_overlay_benchmark.py --background CAPTURE --task onset|escalation`.
Background: the user's real 145-minute hostel Wi-Fi capture (52,010 IPv4 packets, assumed
benign). Train / validation / test = disjoint time segments 0-55 % / 55-72 % / 72-100 % of
the background; 60 / 24 / 30 overlays (one injected campaign each) plus benign look-alike
decoys (Internet probes, flaky client, password typos, internal vulnerability scan) at 10/hour.
Campaigns: 25 % skip reconnaissance, recon is targeted (35 %), slow (25 %) or noise-like
(40 %, built to resemble background probing), recon-to-access dwell log-normal (median 240 s).
Context 16 windows of 30 s, horizon K = 5. Thresholds are chosen on validation at 1 % FPR.
Both tasks were declared before the second was run; neither was tuned on test data.

**Task 1 — from-silence onset** (will infiltration begin within 5 windows, given no
infiltration in the last 3?). Test: 1,448 quiet samples, 93 positive, **20 onsets**.

| model | AUC [95 % CI] | AP (chance 0.064) | event recall at val. threshold |
|---|---|---|---|
| constant | 0.500 | 0.064 | 0 |
| CUSUM/EWMA on recon indicators | 0.606 [0.494, 0.708] | 0.091 | 5 % |
| Mahalanobis | 0.532 [0.454, 0.604] | 0.068 | 5 % |
| Isolation Forest | 0.547 [0.467, 0.622] | 0.079 | 5 % |
| v2 hazard (profile-relative GBT) | 0.518 [0.428, 0.600] | 0.085 | 5 % |
| v1 world model | 0.570 [0.485, 0.645] | 0.076 | 0 |

No model forecasts an onset from silence better than chance on this benchmark. This is
expected from the construction, not only a modelling failure: about 25 % of onsets have no
precursor, 40 % of reconnaissance mimics background noise, and the median recon-to-access gap
(4 min) exceeds the 2.5-minute horizon, so few onsets have any observable precursor inside the
horizon. It also matches the literature's warning that low-and-slow / patient adversaries
are hard to foresee from network aggregates alone.

**Task 2 — escalation** (given early-stage activity may already be visible, will lateral
movement, C2, exfiltration or impact begin within 5 windows?). Test: 1,666 quiet samples,
54 positive, **11 onsets**.

| model | AUC [95 % CI] | AP (chance 0.032) | TPR / FPR at val. threshold | event recall | precision at onset rate 1e-3 |
|---|---|---|---|---|---|
| constant | 0.500 | 0.032 | 0 / 0 | 0 | – |
| CUSUM/EWMA | 0.732 [0.670, 0.799] | 0.057 | 0 / 0.006 | 0 | 0 |
| Mahalanobis | 0.592 [0.532, 0.659] | 0.038 | 0 / 0.012 | 0 | 0 |
| Isolation Forest | 0.592 [0.513, 0.671] | 0.038 | 0 / 0.002 | 0 | 0 |
| **v2 hazard** | **0.941 [0.909, 0.966]** | **0.263** | 0.037 / 0.002 | 0.18 (2 of 11), lead 60 s | 0.015 |
| v1 world model | 0.880 [0.804, 0.941] | 0.223 | 0.093 / 0.007 | 0.18 (2 of 11), lead 135 s | 0.012 |

Reading, with the limits kept visible:
* On the escalation task the v2 hazard model ranks best (AUC 0.94, AP 8x chance). Its interval
  overlaps v1's, so "v2 beats v1" is suggested but **not** established with 11 test onsets.
* Ranking skill did not translate into a good operating point. At the threshold chosen on
  validation data the test false-positive rate was 0.2 % (budget 1 %) and only 2 of 11
  escalations were flagged in advance. The validation negatives (a different time segment)
  did not represent the test negatives well: an example of the drift the literature warns about.
* At a realistic onset rate of 1 in 1,000 windows, precision is about 1.5 %: over 98 % of alerts
  would be false. No claim of field-ready accuracy is supported.
* False alerts on the untouched real background: 0 observed in 66 clean test windows, but the
  exact 95 % upper bound is 6.5 alerts/hour. Too little clean real data to say more.
* The injected attacks are simulated and loud (a credential brute-force burst), and whether a
  campaign escalates is independent of anything observable by construction. The escalation
  skill therefore largely reflects detecting that intrusion activity is under way. One
  background network, one vantage point.

### Phase C (continued) — more data, score smoothing, and operating points

Commands: `python scripts/forecast_improve.py --background CAPTURE --cache STATES.pkl`, then
`python scripts/forecast_operating_points.py ...` (same arguments). Escalation task, real
laptop background, **fresh test overlays** (seeds 3000-3039, same held-out time segment, so
the background is shared with the earlier benchmark): 2,169 quiet samples, 66 positive,
**15 distinct escalation events**. Validation: 47 overlays, 14 events. All 12 configurations
are shown; the choice was made on validation AUC only.

| training overlays | smoothing (windows) | validation AUC | test AUC [95 % CI] | event recall at the 5 %-FPR validation threshold |
|---|---|---|---|---|
| 60 | 1 | 0.889 | 0.960 [0.939, 0.979] | 0.67 |
| 60 | 2 | 0.910 | 0.963 [0.945, 0.981] | 0.67 |
| 60 | 3 | 0.924 | 0.967 [0.950, 0.983] | 0.73 |
| 60 | 4 | 0.927 | 0.966 [0.950, 0.982] | 0.73 |
| 120 | 1 | 0.892 | 0.966 [0.949, 0.981] | 0.73 |
| 120 | 2 | 0.919 | 0.970 [0.954, 0.983] | 0.73 |
| 120 | 3 | 0.934 | 0.972 [0.956, 0.983] | 0.73 |
| 120 | 4 | 0.936 | 0.970 [0.951, 0.983] | 0.73 |
| 240 | 1 | 0.913 | 0.967 [0.949, 0.981] | 0.73 |
| 240 | 2 | 0.936 | 0.971 [0.957, 0.984] | 0.73 |
| **240** | **3** | **0.946** | **0.973 [0.957, 0.985]** | **0.73** |
| 240 | 4 | 0.946 | 0.971 [0.952, 0.984] | 0.73 |

More training campaigns and smoothing help a little and consistently, but every interval
overlaps every other: the differences are within noise for 15 events. The chosen setting
(240 overlays, 3-window smoothing) at stricter false-alarm budgets, thresholds from validation:

| budget (FPR) | test FPR | events flagged in advance | median lead | false alerts/hour on test negatives | precision if onsets are 1 in 1,000 windows |
|---|---|---|---|---|---|
| 0.5 % | 0.7 % | 8 / 15 | 150 s | 0.8 | 4.6 % |
| 1 % | 1.1 % | 9 / 15 | 150 s | 1.3 | 3.6 % |
| 2 % | 1.5 % | 10 / 15 | 150 s | 1.8 | 2.8 % |
| 5 % | 3.7 % | 11 / 15 | 150 s | 4.5 | 1.8 % |

* On the untouched real background: 0 alerts in 64 clean windows; the exact 95 % upper bound
  is 6.7 alerts/hour, so this cannot show the false-alarm rate is low.
* Lead time is the full 150 s horizon whenever an escalation is caught: alerts come from
  visible early-stage activity (the simulated brute-force burst) well before lateral movement.
* **Bottom line, stated plainly:** on this benchmark the system flags roughly half to three
  quarters of simulated escalations about 2.5 minutes ahead at roughly one to four false alerts
  per hour. At a realistic base rate most alerts would still be false (precision under 5 %).
  It is a triage aid whose alerts need a human, not an accurate predictor of attacks, and
  nothing here shows it would work on real attacks: the attacks are simulated and loud, there is
  a single background network, and the test has 15 events.

### What was not done

* Phase D (real-kernel loopback lab) and Phase E (`eval-real` for CIC-IDS/UNSW/CTU-13/LANL):
  not built. Real-dataset hosts are unreachable from the build sandbox; the `datasets.py`
  adapters exist but have not been evaluated with this protocol.
* The dashboard still serves the v1 engine. `atdrps train-local` / `atdrps forecast` are
  command-line only.
* ARP and IPv6 are ignored by the flow pipeline (46 % of the supplied laptop capture was ARP).
* Self-updating launchers: blocked by the session's safety classifier; awaiting the user's decision.
