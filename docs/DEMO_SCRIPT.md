# Demo video script — 2 minutes

**Constraint:** SIH allows a maximum of 2 minutes. This is timed at ~1:55 with pauses.
Record at 1920×1080, screen capture with voice-over. No cuts to slides — the whole video
is the working system, because the point is that it works.

Nothing in this recording touches the network. Say so, on camera, once.

---

## 0:00 – 0:15 · The reframe

> *[Terminal open, repo visible.]*

"Every intrusion detection system asks the same question: is this flow malicious? It looks
at one conversation at a time and throws away time itself.

But infiltration isn't a packet. It's a process. So ATDRPS asks a different question:
**where is this network heading?**"

---

## 0:15 – 0:35 · Ingest, live

> *Run:*
> ```
> python -m atdrps.cli predict data/demo/capture.pcap --model artifacts/model-transformer
> ```
> *Let the header scroll.*

"This is a raw packet capture — eighty thousand packets. No CICFlowMeter, no scapy: ATDRPS
parses pcap itself, assembles two thousand flows, and turns every thirty seconds of traffic
into a ninety-six dimensional network **state**.

Flow-level features and packet-level features together — because a slow port scan sits
under every flow threshold. What gives it away is a constant TTL, a tiny fixed window, and
a walk across ports. Aggregate that into a NetFlow record and the evidence is gone."

---

## 0:35 – 1:00 · The forecast

> *The risk timeline prints. Point at the alert block.*

"For every window, the model forecasts one step ahead. And here's the capture's story,
recovered in order: reconnaissance — initial access — lateral movement.

The model never saw this capture, or anything from this campaign. Eighty percent exact
stage match against ground truth, at a false-positive rate of point one one nine.

And it caught the *slow* scan — the one that looks like nothing, flow by flow."

---

## 1:00 – 1:20 · Forward simulation and explanation

> *Scroll to the K-step block, then the explanation.*

"Then it rolls the learned dynamics forward five windows — a real simulation, not a lookup.
Confidence decays with distance, which is what an honest forecast does.

And nothing here is a black box. Every prediction carries Shapley attributions: *share of
traffic to remote-access services, plus zero point two one six*. In plain language. Plus
the flows behind the call — there's the brute force, straight at port twenty-two."

---

## 1:20 – 1:45 · The dashboard

> *Switch to the browser. Upload the same capture. Let it render.*

"Same engine, offline dashboard. Upload to result in under four seconds. Risk timeline,
the ATT&CK stage ribbon, the forward simulation, the drivers, the flows.

No cloud API. No telemetry. Nothing leaves this machine — which is the only way this
deploys inside critical infrastructure."

---

## 1:45 – 2:00 · The number that matters

> *Cut to `docs/BENCHMARKS.md` on screen.*

"Against logistic regression on identical features and identical splits: F1 point nine
five six against point nine three eight, and the false-positive rate roughly halved.

More importantly, the margin **widens** the further ahead we forecast. A static classifier
has nothing to extrapolate from. A learned transition model does.

That's ATDRPS."

---

## Recording checklist

- [ ] `python -m atdrps.cli benchmark --corpus data/corpus.npz` run beforehand so
      `artifacts/model-transformer` exists
- [ ] Terminal font large enough to read at 1080p (16pt+), dark theme
- [ ] Browser zoomed so the whole dashboard fits without scrolling mid-sentence
- [ ] Disconnect the network before recording and say so — it is the most persuasive
      twenty seconds in the video
- [ ] Keep to 2:00. Overrunning is a disqualification, not a style note.
