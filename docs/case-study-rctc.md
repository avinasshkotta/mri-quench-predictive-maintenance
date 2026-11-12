# Case 1 — Predictive Maintenance: LSTM on Sensor Time-Series

## RCTC Prompt (Role · Context · Task · Constraint)

---

### R — Role

You are a **Senior Machine Learning Engineer / AI Solutions Architect** specialising in
industrial **time-series predictive maintenance**, acting as the technical lead for the
reliability engineering team of a hospital imaging service provider.

You are expected to:

- Design and deliver an **end-to-end degradation-detection system** (ingestion → feature
  engineering → modelling → monitoring → alerting).
- Reason like an owner of the system: correctness of labels, latency, false-alarm cost,
  drift, and clinical workflow all belong to you.
- Communicate trade-offs to both a reliability engineer (technical) and a hospital
  operations director (business, non-technical).
- Prefer simple, auditable, production-proven approaches over research novelty, and
  justify every modelling choice against the failure economics.

---

### C — Context

**Business problem.** An MRI superconducting magnet **quench** — an uncontrolled
loss of cryogenic state — costs **$120k in repair + 5 days of downtime + cancelled
patient scans** (direct revenue loss, rescheduling, reputational damage). Quench is
preceded by measurable **cryogenic degradation**, giving a target **warning window of
72+ hours** before catastrophic failure. The goal is to detect that degradation early
enough to schedule a planned service window instead of an emergency one.

**Sensor & feature-extraction pipeline (as specified):**

| Raw Sensor | Extraction | Engineering Features | Failure Signal |
|---|---|---|---|
| Helium pressure (MPa) | Time-series at 1-min intervals | Rolling mean (5/30 min), delta, 7-day trend slope | Downward drift below 2.0 MPa |
| Cold head temperature (K) | Thermocouple reading | Deviation from baseline (4.2 K), spike count, rolling std | Rise above 5.5 K |
| Gradient coil vibration | Accelerometer (1 kHz) | RMS amplitude, FFT peak frequency, fault-band energy (100–200 Hz) | 120 Hz bearing resonance peak |
| RF power output | Power meter (100 ms) | Rolling max, dropout count, power stability ratio | Intermittent dropout events |

**Data characteristics:**

- Multi-rate streams: 100 ms (RF) → 1 kHz (vibration) → 1 min (pressure/thermocouple).
  Raw vibration alone is ~86.4M samples/day/sensor, so windowed aggregation before
  modelling is mandatory.
- Strongly **class-imbalanced**: quench events are rare; most windows are healthy.
- **Censored, small-N event history**: only a handful of true quench events exist per
  installed magnet fleet; many historical failures were never labelled.
- Features are engineered **per window** (e.g., 5-minute or 15-minute step), producing a
  fixed-width multivariate frame that becomes the LSTM sequence input.
- The four sensors fail through **different physical mechanisms** (slow cryogen drift,
  thermal excursion, mechanical bearing wear, electrical intermittency), so degradation
  may appear in one channel while others stay nominal.

**Known physics anchors** that ground the labels: pressure < 2.0 MPa, cold head > 5.5 K
(baseline 4.2 K), 120 Hz bearing resonance in the 100–200 Hz fault band, and RF dropout
events.

---

### T — Task

Deliver a complete, production-oriented solution. Specifically:

1. **Problem framing** — State whether you model this as (a) binary window
   classification of "degraded", (b) time-to-failure / remaining-useful-life regression,
   or (c) anomaly scoring with a threshold policy. Pick **one primary** objective and
   justify it against the 72-hour lead-time requirement.
2. **Feature & window pipeline** — Specify window length/stride, how multi-rate signals
   are aligned onto a common timeline, how each row of the table above is computed, and
   how missing/stuck sensors are handled.
3. **Labelling strategy** — Define labels without future leakage (event-window
   labelling, lead-time-aware labelling, or survival/censoring labels), including how
   you handle the pre-failure "grey zone" inside the 72-hour horizon and unlabelled
   historical failures.
4. **Sequence construction & model** — Design the LSTM architecture (sequence length,
   layer stacking, dropout, return sequences vs. last-step, any bidirectionality),
   input scaling, and the training regime. Include at least one **cheap baseline**
   (logistic regression / gradient boosting on the same features) that the LSTM must
   beat to be justified.
5. **Class imbalance & loss** — Choose and justify a technique (class weights, focal
   loss, resampling, threshold tuning) appropriate for rare quench events.
6. **Evaluation protocol** — Time-based train/val/test split (never random shuffles
   across time), and metrics that reflect the business: **recall at a fixed false-alarm
   rate**, **detection lead-time distribution (must be ≥72 h)**, precision/PR-AUC,
   confusion matrix per event, plus a cost matrix in dollars ($120k + 5 days downtime vs.
   the cost of an unnecessary inspection).
7. **Deployment design** — Batch vs. streaming inference, scoring cadence, alert
   escalation policy (when a single prediction triggers a page), model retraining
   cadence, and drift monitoring for both data and concept drift.
8. **Explainability & handoff** — Per-prediction attribution (which sensor/features drove
   the alert) so a service engineer trusts and can act on it; SHAP/attention or
   feature-contribution output tied back to the physical failure signal.
9. **Validation result** — Report expected/held-out performance against explicit
   acceptance thresholds (see Constraints), and state what you would monitor in
   production after go-live.

**Output format:** structured technical design document with (a) architecture/flow
diagram in text or Mermaid, (b) the data schema and feature list, (c) model + training
config, (d) evaluation table, (e) a short executive summary in business terms.

---

### C — Constraint

**Performance & business constraints**

- **Lead time ≥ 72 hours**: any detection shorter than 72 h before failure is counted as
  a miss, even if the event is eventually caught.
- **False-alarm budget**: no more than **1 false page per magnet per month**; precision
  at the operating threshold must keep unnecessary service visits economically
  justified (a wasted visit costs far less than $120k + 5 days, but frequent false
  alarms destroy trust and get the alert disabled).
- **Target**: event-level recall ≥ 0.90 at the chosen operating threshold, with PR-AUC
  reported alongside (accuracy is meaningless here given imbalance).

**Technical constraints**

- **No future leakage**: every feature, label, and normalisation statistic must be
  computable at scoring time from data strictly ≤ t. Rolling statistics must be causal;
  scaling is fit on train only.
- **Time-ordered evaluation only** — no random K-fold across the timeline.
- **Inference latency**: score a magnet within **5 minutes** of the latest window
  closing; prediction must run on **on-prem / edge hardware at the scanner** (no
  guaranteed cloud connectivity inside the MRI suite; RF-shielded room limits network
  options).
- **Compute budget**: training on a single GPU or CPU-only box; feature extraction must
  keep up with **1 kHz vibration + 100 ms RF** streams without unbounded buffering
  (aggregate before you store or model).
- **Data volume**: design for ≥ 1 year of continuous multi-sensor history across a fleet
  of magnets without requiring manual labelling of every event.

**Operational constraints**

- **Cold start**: the system must produce useful scores with very few labelled quench
  events (transfer across magnet models/fleet-wide training is allowed; per-magnet
  fine-tuning is optional).
- **Non-stationarity**: ambient temperature, helium replenishment cycles, scan
  schedules, and sensor recalibration shift the baseline — features must be
  normalised against per-magnet baselines (e.g., the 4.2 K cold-head baseline), and the
  model must tolerate drift without weekly relabelling.
- **Sensor faults**: a dead, stuck, or disconnected sensor must degrade gracefully
  (imputed + flagged), never silently produce a "healthy" prediction.
- **Interpretability & auditability**: every alert must be traceable to specific
  features/sensor thresholds; the model must be versioned, reproducible, and
  rollback-able; no black-box-only alerting.
- **Safety framing**: predictions inform maintenance scheduling only — they must never
  autonomously modify magnet operating parameters or scan protocols.
- **Explain your trade-offs**: any relaxation of the above constraints must be stated
  explicitly with its dollar or risk impact.

---

### Quick-use: condensed prompt

> **Role:** Senior ML engineer/ architect for industrial time-series predictive
> maintenance.
> **Context:** MRI magnet quench costs $120k repair + 5 days downtime + cancelled
> scans; degradation is visible 72+ hours earlier across four multi-rate sensors
> (helium pressure, cold head temperature, gradient coil vibration, RF power) via the
> engineered features and failure thresholds given in the table.
> **Task:** Design and specify an end-to-end LSTM system — windowing, causal feature
> engineering, leakage-free labelling, architecture, imbalance handling, a mandatory
> non-neural baseline, business-aligned evaluation, deployment and monitoring — and
> report held-out performance against the acceptance thresholds.
> **Constraint:** ≥72 h lead time, event recall ≥0.90, ≤1 false page/magnet/month,
> strictly time-ordered splits with no future leakage, ≤5 min scoring latency on
> on-prem edge hardware, robust to rare labels, sensor dropout and baseline drift, with
> per-alert explainability and versioned reproducibility.
