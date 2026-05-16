# Backtest Trust Checklist

Use this checklist before trusting any sports betting backtest, simulation, or forward-test claim.

The goal is to separate three different questions:

1. Is the model using only point-in-time information?
2. Is the model actually ahead of the market?
3. Is the simulated P&L remotely achievable in the real world?

If the answer to any of those is "not sure", the backtest is not ready to trust.

---

## 1. Point-in-Time Data Integrity

These are non-negotiable.

- [ ] Every feature is built only from information available before the event start time.
- [ ] Rolling stats use explicit lagging or `shift(1)` style logic before window aggregation.
- [ ] Season-to-date stats do not use full-season totals.
- [ ] Elo, ratings, and H2H values are recorded pre-match and updated only after known outcomes.
- [ ] Future scheduled matches can flow through inference without updating historical state.
- [ ] Missing-value imputation at inference uses artifacts derived from the training set only.
- [ ] Join keys for odds, rankings, and schedule data cannot silently attach the wrong event.
- [ ] Ambiguous joins are rejected, not guessed through.
- [ ] Date logic is based on actual or carefully estimated event chronology, not coarse season or tournament bucket dates.

Minimum evidence:
- [ ] At least one explicit leakage audit has been run.
- [ ] At least one regression test exists for future-row inference safety.

---

## 2. Validation Architecture

The most common failure mode is evaluating on data that influenced the model indirectly.

- [ ] Holdout evaluation is chronological, not randomized.
- [ ] Each holdout season or window uses an independent model artifact.
- [ ] Hyperparameter tuning never touches holdout data.
- [ ] Threshold selection never touches holdout data.
- [ ] Governance rules were either fixed prospectively or clearly labeled as retroactive.
- [ ] Calibration data is truly held out from model fitting, ideally out-of-fold.
- [ ] Metrics are reported per holdout period, not only in aggregate.

Preferred standard:
- [ ] One model artifact per evaluation year or season.
- [ ] One fixed policy per evaluation period.
- [ ] A clear table of train / validation / calibration / test ranges.

---

## 3. Market Methodology

Backtests fail when "model probability", "market probability", and "price actually bet" are mixed together carelessly.

- [ ] Edge is defined against a clear market baseline.
- [ ] The market baseline is explicitly one of:
  - no-vig median retail composite
  - best-book no-vig
  - sharp close
  - prediction-market probability
- [ ] The price used for P&L is the actual bettable price, not the no-vig probability source.
- [ ] The odds source and bookmaker set are documented.
- [ ] Historical odds quality was audited for stale, broken, live-contaminated, or implausible values.
- [ ] Low-confidence odds matches are excluded from pick generation.

Preferred standard:
- [ ] Use no-vig market probability for edge.
- [ ] Use best actually available raw odds for simulated execution.
- [ ] Track both retail-composite close and sharp-close validation when possible.

---

## 4. Model Validation Metrics

Accuracy is not enough, and bankroll growth is not a validity test.

- [ ] Discrimination is reported with AUC or similar ranking metrics.
- [ ] Calibration is reported with log loss and Brier score.
- [ ] Calibration plots or bucket summaries are reviewed.
- [ ] Tail probabilities are audited for saturation or clipping artifacts.
- [ ] The model is compared against a naive baseline.

Backtest trust should weight these metrics in this order:

1. Point-in-time integrity
2. CLV / beat-close evidence
3. Calibration quality
4. Flat-bet ROI
5. Kelly-style compounded outcomes

---

## 5. CLV And Market-Efficiency Evidence

This is the strongest evidence of real edge.

- [ ] CLV is computed for every pick where later market data is available.
- [ ] Average CLV is reported by holdout period.
- [ ] Beat-close rate is reported by holdout period.
- [ ] CLV is positive across multiple independent holdout periods.
- [ ] A negative-P&L / positive-CLV season is treated as plausible variance, not automatic invalidation.
- [ ] A positive-P&L / negative-CLV season is treated as a red flag, not a success.

Preferred standard:
- [ ] Make CLV the primary evidence of market edge.
- [ ] Treat bankroll growth as secondary and execution-sensitive.

---

## 6. Governance And Threshold Discipline

Rules are often where hidden look-ahead sneaks in.

- [ ] Every edge threshold has a documented origin.
- [ ] Every governance rule has a documented origin.
- [ ] Rules discovered from later seasons are not silently applied backward.
- [ ] Blocked-bet cohorts are analyzed separately from retained bets.
- [ ] Governance effects are reported as a delta, not hidden inside final ROI.

If a rule is retroactive:
- [ ] It must be labeled as retroactive.
- [ ] Performance should be restated both with and without it.

---

## 7. Simulation Realism

This is where fantasy bankrolls usually come from.

- [ ] Bets are ordered in real chronology.
- [ ] Same-day bankroll reuse is handled explicitly.
- [ ] Position sizing uses the same probability definition as edge selection.
- [ ] Daily and per-bet exposure caps are documented.
- [ ] The simulation does not assume impossible multi-book access without saying so.
- [ ] The simulation distinguishes between:
  - theoretical price
  - observed close
  - achievable execution price
- [ ] Slippage and stale-line risk are either modeled or clearly acknowledged.
- [ ] Account limits and scaling constraints are acknowledged for compounding results.

Hard rule:
- [ ] Never treat Kelly terminal bankroll as primary proof of validity.

---

## 8. Live And Forward Testing

Historical backtests can only earn provisional trust.

- [ ] There is a shadow-mode pipeline that generates picks before outcomes are known.
- [ ] Raw source snapshots are archived for each live run.
- [ ] Model version, config version, and source timestamps are captured in a manifest.
- [ ] Live picks are evaluated later against realized closes and outcomes.
- [ ] Live CLV is tracked separately from historical CLV.
- [ ] Live execution price, if different from modeled price, is recorded separately.

Best practice:
- [ ] Require at least 30 days of clean shadow runs before stronger claims.

---

## 9. Reporting Standards

The write-up should make optimistic assumptions obvious, not invisible.

- [ ] Every odds source is named.
- [ ] Every holdout period is named.
- [ ] Every retroactive rule is disclosed.
- [ ] Every known methodological weakness is disclosed.
- [ ] Flat ROI and Kelly results are both labeled with execution assumptions.
- [ ] Confidence should be stated in levels, not just raw percentages.

Minimum outputs for any serious review:
- [ ] Per-period AUC
- [ ] Per-period log loss
- [ ] Per-period CLV
- [ ] Per-period beat-close rate
- [ ] Per-period flat ROI
- [ ] Governance block counts
- [ ] A written limitations section

---

## 10. Sport-Specific Reminders

### Tennis

- [ ] Within-tournament chronology is handled carefully.
- [ ] Rematches in the same month or event cannot share the wrong odds row.
- [ ] Schedule rows with missing stat depth can still be scored without leakage.
- [ ] Surface and round are treated as essential match-join context.

### Baseball

- [ ] Probable pitcher data is point-in-time and not silently replaced with later-confirmed starters.
- [ ] Early-season features do not overstate current-year signal.
- [ ] Bullpen workload and lineup context are timestamp-safe.
- [ ] October, lineup rest, and clinching scenarios are governed explicitly.

---

## 11. Trust Levels

Use this scale when summarizing any system.

### Level 0 — Not Trustworthy
- leakage unresolved
- market joins heuristic and unvalidated
- no independent holdouts

### Level 1 — Research Only
- basic holdouts exist
- some PIT controls exist
- CLV evidence weak or absent

### Level 2 — Backtest Credible
- PIT controls pass
- independent holdouts pass
- CLV is positive across multiple periods
- limitations are disclosed honestly

### Level 3 — Shadow-Operational
- reproducible daily pipeline exists
- live snapshots archived
- forward CLV is being tracked

### Level 4 — Real-Money Candidate
- at least 30 days of clean shadow operation
- live CLV consistent with backtest CLV
- data feed reliability and execution assumptions proven in practice

---

## Bottom Line

Do not trust a sports betting backtest because:
- the AUC looks decent
- the win rate looks high
- the Kelly curve explodes

Trust starts only when:
- point-in-time integrity is clean
- independent holdouts are clean
- CLV is consistently positive
- the simulation assumptions are explicitly realistic
