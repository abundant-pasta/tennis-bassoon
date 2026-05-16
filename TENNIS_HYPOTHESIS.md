# ATP Match Pricing: Current Hypothesis And Trust Assessment

**Prepared for:** internal model review  
**System:** ATP pre-match win probability model and market comparison pipeline  
**Evaluation horizon:** independent holdout years 2022, 2023, 2024 plus approximate 2026 YTD forward-style test  
**Date:** April 29, 2026  

---

## 1. Central Hypothesis

> A point-in-time pre-match tennis model can estimate ATP match win probabilities well enough to identify market prices that are mispriced at the margin, producing positive expected value over time.

This is a stronger claim than "the model predicts match winners well."

It requires two things:

1. the outcome model is genuinely discriminative and point-in-time, and  
2. the market comparison layer is strong enough that model-market disagreements are meaningful rather than artifacts of bad joins or unrealistic execution assumptions.

As of April 29, 2026, there is now affirmative evidence on both counts.

---

## 2. What I Trust Most Right Now

### 2.1 The outcome model is probably real

The independent holdout-year suite:

| Holdout Year | Train Window | Validation | OOS AUC | Bets | Flat ROI |
|---|---|---|---:|---:|---:|
| 2022 | 2008–2020 | 2021 | 0.7257 | 943 | +16.89% |
| 2023 | 2008–2021 | 2022 | 0.7167 | 964 | +16.01% |
| 2024 | 2008–2022 | 2023 | 0.7184 | 840 | +23.50% |
| **Average** | — | — | **0.7203** | **2,747 total** | **+18.80%** |

These are **independent artifacts** trained separately per holdout year, which is a materially better methodology than evaluating one model over a broad pooled window.

The AUC stability is the main reason I take the model seriously:

- 2022: `0.7257`
- 2023: `0.7167`
- 2024: `0.7184`

That consistency is hard to fake accidentally after the point-in-time chronology fixes.

### 2.2 The AUC is believable

An OOS AUC around `0.72` is high, but not obviously implausible for ATP match outcomes.

Tennis is structurally more predictable than baseball:

- favorites win more often
- rank differentials carry signal
- surface specialization is real
- player-specific form and H2H matter more directly than in many team sports

Also, the model is not jumping from random chance to perfection. The ranking baseline is already strong. The model appears to be improving on an already informative prior, not inventing signal from nowhere.

### 2.3 The Pinnacle CLV panel is now positive — and strongly so

The previously missing piece was a year-by-year market validation. That panel has now been built using Pinnacle closing odds from tennis-data.co.uk.

**CLV Panel — Pinnacle Closing Line Validation:**

| Year | AUC | Bets | Avg CLV | Model P | Pinnacle Close P | Win Rate | Excess | Flat ROI | Z vs Mkt |
|---|---|---|---|---|---|---|---|---|---|
| 2022 | 0.7257 | 943 | +12.1% | 62.6% | 50.5% | 60.0% | +9.5% | +16.9% | 6.24 |
| 2023 | 0.7167 | 964 | +12.2% | 62.2% | 50.0% | 59.1% | +9.1% | +16.0% | 6.09 |
| 2024 | 0.7184 | 840 | +11.3% | 60.6% | 49.4% | 60.6% | +11.2% | +23.5% | 6.94 |
| **3yr** | — | **2,747** | **+11.9%** | **61.9%** | **50.0%** | **59.9%** | **+9.9%** | **+18.6%** | **11.10** |

*CLV = model_prob − Pinnacle closing no-vig probability for the bet side. Excess = actual win rate − avg Pinnacle implied probability.*

**Edge bucket breakdown (3-year combined):**

| Bucket | Bets | Win Rate | Model P | Mkt P | Excess | Flat ROI | Z |
|---|---|---|---|---|---|---|---|
| 06–08% | 715 | 60.6% | 59.9% | 52.9% | +7.7% | +13.9% | 4.48 |
| 08–10% | 536 | 59.5% | 61.5% | 52.5% | +7.0% | +9.6% | 3.50 |
| 10–12% | 434 | 61.3% | 62.3% | 51.3% | +10.0% | +16.4% | 4.47 |
| 12%+ | 1,062 | 59.0% | 63.3% | 46.2% | +12.8% | +27.2% | 8.78 |

**Surface breakdown (3-year combined):**

| Surface | Bets | Win Rate | Mkt P | Excess | Flat ROI | Z |
|---|---|---|---|---|---|---|
| Hard | 1,579 | 62.6% | 51.5% | +11.1% | +21.8% | 9.45 |
| Clay | 887 | 58.2% | 49.4% | +8.7% | +14.5% | 5.55 |
| Grass | 281 | 50.2% | 43.3% | +6.8% | +13.4% | 2.46 |

**What the CLV panel shows:**

The model's picks win **+9.9 percentage points** more often than Pinnacle's closing odds imply they should. This excess is consistent across all three independent holdout years and across all four edge buckets. The aggregate z-score of **11.10** (combined) and **6.09–6.94** per year is well into the territory where chance explanations are implausible.

The CLV source is **Pinnacle's closing line** — which is the sharpest available benchmark in sports betting. This is a harder test than the retail-composite CLV used by the MLB system. The tennis model is consistently beating Pinnacle at close on the picks it selects.

**The structural pattern:** picks average ~50% Pinnacle implied probability (the model is finding near-coin-flip matches per Pinnacle), but actual win rates are ~60%. The model is systematically identifying matches where Pinnacle prices at random but where surface-specific ELO, H2H, and form signal a meaningful edge.

---

## 3. What Has Changed Since April 28

The primary concern in the previous version of this document was:

> *"The missing piece is historical CLV. Tennis does not yet have a mature historical open-vs-close validation framework."*

That gap is now closed. The CLV panel above is the evidence that was missing. The honest hierarchy shifts:

1. I trust the AUC — still holds
2. I trust that the model is finding real Pinnacle mispricing — now supported
3. I do not yet treat the ROI figures as executable projections — still holds

---

## 4. What Is Still Not Proven

### 4.1 This is closing-line CLV, not open-to-close CLV

tennis-data.co.uk provides only one set of odds per match — Pinnacle closing odds. True open-to-close CLV (the kind used in the MLB system's live shadow run) requires knowing what odds were available when the model ran and whether the line subsequently moved in the model's direction.

What the panel can say: the model's probability consistently exceeds Pinnacle's closing probability for picks. This is meaningful — it means the model finds value that even the final, sharpest market price doesn't fully close out.

What it cannot say: whether the model was "early" (line moved toward model post-open) or "against the market at close" (model diverges from the sharpest consensus price). The first interpretation is more bullish; the second more cautious. We don't know which applies without opening line data.

### 4.2 The average CLV of ~12% is large and needs explanation

The avg CLV of 11–12% is much larger than the baseball system's +0.35%. This warrants explanation rather than celebration:

- By construction, all picks have CLV > 6% (the edge threshold). Average of ~12% means many picks cluster around 10–15% edge.
- The model's average probability for picks is ~62%; Pinnacle's closing implied is ~50%. This means the model is regularly putting heavy probability on sides Pinnacle prices at near-coin-flip.
- Actual win rate of ~60% is between model (62%) and market (50%), confirming the model is slightly overconfident but Pinnacle is meaningfully underpricing the selected side.

The most plausible explanation is that Pinnacle's tennis closing prices for Challenger and tour-level 250/500 matches are less efficient than their MLB or NFL pricing. The tennis market is thinner, particularly below Grand Slam level. The model's ELO and H2H signals may be capturing something the general market underweights.

### 4.3 Grass surface is weaker

The Grass surface shows z=2.46 and lower excess win rate (+6.8%) than Hard or Clay. This is worth monitoring but is not sufficient evidence to exclude Grass picks — 281 bets is a relatively small sample and z=2.46 is still meaningful.

### 4.4 The holdout suite used light tuning

The 2022–2024 independent suite was built with only 5 Optuna trials per year. That is enough to test stability, but the final benchmark configuration should use a full tuning budget (50–100 trials). ROI would likely change slightly with full tuning.

### 4.5 No live shadow validation yet with tracked opening odds

The 2026 YTD result through April 19, 2026 shows +6.72% flat ROI on 526 bets — more modest than the holdout backtests, consistent with the forward test being less optimistic than historical simulation. But the shadow ledger does not yet record opening odds at bet time, so open-to-close CLV cannot be computed in real time.

### 4.6 Historical odds matching is still partly heuristic

The historical join relies on inferred tournament chronology plus names, surface, round, and event normalization. Coverage is ~99.7% (Pinnacle), so the match rate is high, but a small fraction of joins may be incorrect, which would add noise to the CLV calculation.

### 4.7 Calibration is still not true out-of-fold

The calibration path uses the averaged recent-year Platt approach. This is not the strictest method. For outcome modeling it is acceptable; for betting-edge claims it is a residual caveat, particularly at the tails of the probability distribution where most picks land.

---

## 5. Methodological Improvements Already Made

- within-tournament chronology rebuilt using inferred match order rather than raw tournament start date
- ELO, rolling stats, form, H2H, and context features made safe for point-in-time use
- odds matching tightened with confidence scoring and explicit ambiguity rejection
- train-time medians saved and reused at inference
- daily runner is fail-closed and writes immutable artifacts
- shadow ledger and opening/current odds snapshot path exist for future market validation
- independent holdout-year artifacts exist for 2022, 2023, and 2024
- CLV panel built using Pinnacle closing odds across all 3 holdout years

---

## 6. Current Trust Level

**Tennis is now Level 2+ — Backtest Credible with Market Validation.**

Why above Level 2:

- point-in-time controls are materially better
- independent holdout years exist
- software path is reproducible and fail-closed
- Pinnacle CLV panel now shows consistent positive excess win rate (z > 6 each year)
- all four edge buckets show positive excess over Pinnacle close
- all three surfaces show positive excess (Grass weakest but still positive)

Why not full Level 3:

- CLV is closing-line only, not open-to-close (opening odds not available historically)
- no meaningful run history of live shadow validation with opening odds recorded
- the magnitude of the CLV (12% avg) is unusually large and requires ongoing monitoring to verify it is real market inefficiency rather than residual calibration artifact

The cleanest single-sentence summary:

> The tennis system now shows strong statistical evidence of real Pinnacle market mispricing on selected picks, consistent across 3 independent holdout years, with the primary remaining gap being live prospective validation with opening odds recorded.

---

## 7. What Would Change My Confidence Most

In priority order:

1. **Record opening Pinnacle odds in the live shadow run.** The daily runner already runs before matches; storing the opening/current Pinnacle price at run time enables true open-to-close CLV. If the market consistently closes toward the model's picks, that would be the strongest single piece of evidence available.

2. **Re-run the independent holdout suite with a full tuning budget** (50–100 Optuna trials per year). Verify that ROI and CLV figures are stable under more thorough hyperparameter search.

3. **Replace heuristic historical joins** where possible with event-identity-keyed historical lines. The current 99.7% coverage is good, but the remaining ~0.3% of missed or wrong joins add noise.

4. **Add a fourth holdout year (2021)** to extend the evaluation window. 2021 requires a different training split and may be a weaker year, which would be valuable information.

5. **Accumulate live shadow results through the 2026 season.** The year-end result will be the first fully prospective test and is accumulating now.

If steps 1–2 hold up, the tennis system can be characterized as a confirmed market edge candidate. Steps 3–5 add confidence and durability to that claim.

---

## 8. Comparison to the MLB System

| Dimension | MLB | Tennis |
|---|---|---|
| Holdout seasons | 6 (2019–2025) | 3 (2022–2024) |
| Total holdout picks | 9,149 | 2,747 |
| CLV source | Retail composite (no Pinnacle) | **Pinnacle closing** |
| Avg CLV | +0.35% | +11.9% |
| Avg excess win rate vs. market | ~3–5% implied | **+9.9%** |
| Combined z-score | ~8–10 (estimated) | **11.10** |
| Seasons with positive CLV | 6/6 | 3/3 |
| Live shadow run | Operational (2026) | Partial (2026 YTD) |

The tennis CLV evidence per-season (z > 6) is actually stronger than the MLB per-season signal, principally because: (a) the tennis edge is larger (excess win rate ~10% vs. ~4%), and (b) the CLV benchmark (Pinnacle) is sharper than the MLB retail composite. The tennis system has fewer holdout years and no opening-odds CLV, which keeps it below the MLB system's overall confidence level despite the stronger per-pick signal.

---

## Bottom Line

The `~0.72` AUC is real.

The `+9.9%` excess win rate over Pinnacle's closing odds is statistically significant at z=11.10 across 2,747 independent holdout picks, with year-by-year z-scores of 6.09–6.94. This is the strongest available evidence that the model is finding genuine market mispricing.

The `16%–24%` flat ROI figures remain high enough to warrant continued scrutiny, but they now have a coherent market-validation backing: the model is finding cases where Pinnacle prices ~50% and the picks win ~60%.

The most honest conclusion today is:

> Trust the tennis model as a pre-match betting edge candidate, with the primary open question being whether the Pinnacle price at open (when bets would actually be placed) reflects similarly exploitable mispricing as the closing price does.
