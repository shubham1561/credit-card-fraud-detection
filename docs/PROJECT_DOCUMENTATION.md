# Credit Card Fraud Detection: Project Documentation

| | |
|---|---|
| **Author** | Shubham Pandey |
| **Repository** | `credit-card-fraud-detection` |
| **Version** | v2 (modular rebuild of the first version), October 2026 |
| **Stack** | Python 3.10+ (results produced on 3.13.18), pandas, NumPy, scikit-learn, XGBoost, matplotlib; FastAPI |
| **Status** | Complete: both evaluation protocols run end to end, and every number below comes from them |

Every number in this document is copied from `reports/results.md`, which `python -m src.report` generates from `reports/metrics/*.json`.

## Contents

1. [Summary](#1-summary)
2. [Problem framing](#2-problem-framing)
3. [Data](#3-data)
4. [Exploratory analysis](#4-exploratory-analysis)
5. [Features](#5-features)
6. [Evaluation design](#6-evaluation-design)
7. [Models](#7-models)
8. [Metrics](#8-metrics)
9. [Choosing the alert threshold](#9-choosing-the-alert-threshold)
10. [Experiments](#10-experiments)
11. [Results](#11-results)
12. [Explanations and error analysis](#12-explanations-and-error-analysis)
13. [Model card](#13-model-card)
14. [Reproducibility](#14-reproducibility)
15. [Serving](#15-serving)
16. [Limitations](#16-limitations)
17. [Future work](#17-future-work)
18. [References](#18-references)
19. [Appendix: changes from the first version](#19-appendix-changes-from-the-first-version)

---

## 1. Summary

The goal is to rank card transactions by fraud risk and decide which ones a review team should check. On the ULB credit-card dataset (283,726 transactions after removing duplicates, 473 of them fraud), XGBoost, the model chosen on the validation block, reached PR-AUC 0.785 (95% CI 0.693–0.868) on a time-ordered test block of 56,744 transactions with 74 frauds, where random guessing scores 0.0013. On a stratified split it reached 0.821. At the cost-optimal threshold for a missed fraud costing 20 reviews, chosen on validation data and applied once, it caught **57 of 74 test frauds with 35 false alerts**.

Four findings shape how the results are reported.

1. **ROC-AUC and PR-AUC rank the models in opposite orders.** On the time-ordered test block ROC-AUC puts logistic regression first (0.981) and random forest last (0.965); PR-AUC puts random forest first (0.812) and logistic regression last (0.745). PR-AUC is the headline because it is the metric that counts false alerts.
2. **The best model depends on the evaluation block.** XGBoost led in cross-validation (0.854 against 0.844 for random forest), on the validation block (0.789 against 0.768) and, within noise, on the stratified test set (+0.009, 95% CI -0.004 to +0.025). Random forest led on the time-ordered test block (0.812 against 0.785; paired difference -0.027, 95% CI -0.055 to -0.006). With 57 to 95 frauds per block, the two tree ensembles are best described as tied at the top; both clearly beat logistic regression on the stratified split.
3. **For XGBoost, class weighting and threshold choice act on the same decision.** Removing the weights moved the cost-optimal threshold from 0.136 to 0.053 and changed validation PR-AUC by +0.009 (95% CI -0.006 to +0.026); both thresholds catch 45 of 57 validation frauds. Logistic regression is different: on the time-ordered protocol the weights changed its ranking, raising validation PR-AUC by 0.143.
4. **A random split flatters XGBoost, not every model.** XGBoost's PR-AUC is about 0.035 lower on the time-ordered test block than on the stratified test set, close to the 0.056 gap the first version measured, while random forest holds steady and class-weighted logistic regression even scores higher on the later block. All of these changes are within the intervals, and the later block's lower fraud rate (0.130% against 0.167%) makes PR-AUC harder to earn there anyway.

---

## 2. Problem framing

### 2.1 The decision

Each transaction gets a score. Transactions scoring at or above a threshold *t* are flagged for manual review; the rest go through. The model ranks; the threshold decides how many frauds are caught and how much review work is created.

### 2.2 Costs

The two errors have different prices.

| Outcome | What happens | Cost used here |
|---|---|---|
| Missed fraud (false negative) | Money lost, chargeback handling, an unhappy customer | `C_miss` per fraud |
| False alert (false positive) | Analyst time, a call to or block on a genuine customer | `C_review` per alert |

Total cost at threshold *t*:

```
Cost(t) = C_miss · FN(t) + C_review · FP(t)
```

Costs are expressed as the ratio r = C_miss / C_review, with C_review = 1. **The main analysis uses r = 20.** The first version costed a missed fraud at its own amount and a review at 5. The mean fraud amount in Protocol B's training block is 113 (notebook 01, §4), so on that basis a miss costs about 23 reviews; r = 20 is a round value close to that and is also a point of the sweep. Section 9.3 repeats the analysis for r from 2 to 500, and experiment E6 drops the constant ratio and costs each miss at its own amount, so no conclusion rests on this one choice.

If every alert costs a review, true frauds included, replace C_miss with C_miss − C_review. The extra term is a constant, so the best threshold is otherwise unchanged.

### 2.3 Why accuracy fails

About 1 transaction in 600 is fraud. Predicting "legitimate" for everything scores 99.83% accuracy and catches nothing. Every metric used here is built on the fraud class (Section 8).

### 2.4 Success criteria, fixed before looking at test data

- PR-AUC on the time-ordered test block, with a 95% bootstrap interval.
- Frauds caught and false alerts raised at the chosen threshold.
- No decision about features, model settings, threshold or calibration uses test data.

---

## 3. Data

### 3.1 Source

Card transactions made by European cardholders over two days in September 2013, collected during a research collaboration between Worldline and the Machine Learning Group of the Université Libre de Bruxelles (ULB) [1, 2]. Available on Kaggle (`mlg-ulb/creditcardfraud`) and OpenML (dataset 1597). The file is not redistributed in this repository.

The committed results were produced from a public GitHub mirror of the file, which `python -m src.data --download` fetches. The mirror stores the values rounded to nine decimal places, so it is smaller than the Kaggle original; its row, fraud and duplicate counts match the published figures (3.3). Its SHA-256 is recorded in `reports/data_checks.json`, and the data check reports whether a local copy is this exact file. Running on the Kaggle original should give the same counts, but the numbers in Section 11 may move in their last digit.

### 3.2 Schema

| Column | Type | Meaning |
|---|---|---|
| `Time` | float | Seconds elapsed since the first transaction in the file |
| `V1`–`V28` | float | Principal components of the original features, which are confidential |
| `Amount` | float | Transaction amount, untransformed |
| `Class` | int | 1 = fraud, 0 = legitimate |

### 3.3 Data checks (`python -m src.data`)

| Check | Expected | Observed |
|---|---|---|
| Rows | 284,807 | 284,807 |
| Frauds | 492 (0.173%) | 492 (0.173%) |
| Missing values | 0 | 0 |
| Exact duplicate rows removed | 1,081 | 1,081 (19 of them fraud) |
| After de-duplication | 283,726 rows, 473 frauds (0.167%) | 283,726 rows, 473 frauds (0.167%) |
| Rows with identical features but different labels | count reported | 0 |
| SHA-256 of `creditcard.csv` (mirror) | | `33a178be65174943a07d1736acf5898f6c8e1fa80950dbbe598c17eda968ec59` |

De-duplication runs once, before any split. A transaction on one side of a split and its exact copy on the other would let the model be graded on a row it had memorised.

The removed copies are disproportionately fraud: 19 of 492 fraud rows (3.9%) against 1,062 of 284,315 legitimate rows (0.37%). The duplicated frauds cluster in time: they form 13 groups, six of which fall within 26 seconds of each other around hour 26, and two of which are transactions of 1.00 repeated four and six times, a pattern consistent with repeated attempts (notebook 01, §2).

### 3.4 What the data can't tell us

- **Two days only.** The time-ordered split tests generalisation hours ahead, not drift over weeks or months.
- **Few frauds per block.** Protocol B's validation and test blocks hold 57 and 74 frauds, so model rankings and thresholds come with wide intervals.
- **Anonymised features.** There are no card IDs, merchants or customer histories, so the aggregate "velocity" features that drive real fraud systems can't be built, and explanations are in units of V-components.
- **Relative time.** `Time` counts from the start of the file; the hour of day is inferred from it (Section 5).
- **No customer attributes.** Fairness across customer groups can't be assessed.
- **Labels are taken as correct and complete.** In practice, fraud labels arrive days or weeks late and depend on which transactions were investigated.

---

## 4. Exploratory analysis

`notebooks/01_eda.ipynb` explores the de-duplicated data. It does not choose models, features or thresholds; those choices are made in `src/` on training data only.

| Question | Output | Finding |
|---|---|---|
| How rare is fraud? | class counts | 473 of 283,726 after de-duplication, about 1 in 600 |
| Do fraud amounts differ? | `amount_by_class.png` (log scale) | Yes. 36% of frauds are 1.00 or less (11% of legitimate transactions) and 5.3% are exactly 0; the fraud upper tail is also heavier (95th percentile 656 against 365) |
| Is there a daily cycle in volume and in fraud rate? | `volume_and_fraud_rate_by_hour.png` | Volume is lowest 2–5 and 26–29 hours after the file starts, and the fraud rate is highest in those quiet hours (above 1.3% in hours 2, 26 and 28, against about 0.1–0.2% in the daytime) |
| Which components separate fraud from legitimate? | `top_components_by_class.png` | V17, V14, V12, V10, V16 and V3 (\|d\| from 8.1 down to 4.5), which includes the four most often reported for this dataset |
| Are V1–V28 uncorrelated with each other? | `correlation_heatmap.png` | Yes: the largest \|r\| is 0.019. `Amount` and `Time` are not uncorrelated with them (V2 -0.53 and V7 0.40 with `Amount`, V3 -0.42 with `Time`) |
| What do the duplicate rows look like? | table | 1,081 extra copies in 773 groups; 19 fraud copies in 13 groups that cluster in time |

---

## 5. Features

`src/features.py` defines one function, `build_features`, used unchanged by training and by the API.

| Feature | Definition | Why |
|---|---|---|
| `V1`–`V28` | As provided | Already-centred PCA components |
| `log_amount` | log(1 + Amount) | Amount is heavily right-skewed; the log helps logistic regression and changes nothing for trees |
| `hour_sin`, `hour_cos` | sin and cos of 2π · hour / 24, with hour = ⌊Time / 3600⌋ mod 24 | Encodes the daily cycle without a false jump between hour 23 and hour 0 |

**Dropped: raw `Time`.** It is a position inside a two-day file. Under a time-ordered split every test value lies beyond anything seen in training; under a random split it lets a model memorise when bursts of fraud happened. Neither generalises. Dropping it doesn't remove every trace of time, though: V3 correlates -0.42 with `Time` (notebook 01, §6).

**The hour is relative.** It counts hours since the file began, folded onto a 24-hour cycle. Volume falls to its lowest 2–5 hours after the start of each day, so the file probably starts near midnight; the dataset doesn't say.

**Scaling.** `StandardScaler` is used for logistic regression only, inside a scikit-learn `Pipeline`, so it is fitted on training folds alone. Trees don't need it.

**No resampling.** Class imbalance is handled with class weights (Section 7). SMOTE and undersampling are not used. If they are tried later, they must run inside each training fold: resampling before the split is a documented cause of inflated fraud-detection results [9].

---

## 6. Evaluation design

### 6.1 Leakage rules

1. De-duplicate before splitting.
2. Fit every transformation (scaler, calibrator) on training data only, through pipelines.
3. Choose thresholds and calibration on out-of-fold or validation predictions, never on test data.
4. Score each test set once, after every choice is frozen.
5. Never use raw `Time` as a feature.

`tests/test_pipeline.py` checks rule 3 directly: it recomputes every model's threshold from the saved selection scores and requires an exact match.

### 6.2 Two protocols

| | Protocol A: stratified | Protocol B: time-ordered (headline) |
|---|---|---|
| **Split** | 80% train / 20% test, stratified by class, seed 42 | Sorted by `Time`: first 60% train, next 20% validation, last 20% test. Cut points move forward past tied timestamps, so no second appears in two blocks |
| **Model selection and threshold** | Out-of-fold predictions from 5-fold stratified cross-validation on the training 80% | Validation block |
| **Final model fitted on** | The whole training 80% | The training block |
| **Question answered** | How well does the model rank transactions like those it was trained on? Comparable with most published work | How well does it rank *later* transactions, which is the only situation production faces? |
| **Size** | Train 226,980 rows, 378 frauds; test 56,746 rows, 95 frauds | 170,236 / 56,746 / 56,744 rows; 342 / 57 / 74 frauds (0.201%, 0.100%, 0.130%); hours 0.0–33.4, 33.4–40.3, 40.3–48.0 |

The time-ordered blocks are not stratified, so their fraud rates differ; each block's rate is reported and is the no-skill PR-AUC for that block. The training block holds both nights of the file, when fraud is most frequent (Section 4); the validation and test blocks cover the daytime and evening of the second day.

### 6.3 Why the Protocol B model isn't refitted on train + validation

The threshold is read from validation scores of the model trained on the training block. Refitting on more data would change the score scale, and the threshold would no longer match it. A production system would refit on a rolling window and re-derive the threshold from a fresh validation window.

---

## 7. Models

`src/models.py` builds every model through one factory. The `weighted` flag switches class weighting on or off with nothing else changed.

| Model | Configuration | Class weighting, on / off | Scaling |
|---|---|---|---|
| Logistic regression | lbfgs, C = 1.0, max_iter = 5,000 | `class_weight="balanced"` / none | StandardScaler in a pipeline |
| Random forest | 300 trees, min_samples_leaf = 2, max_features = √p | `class_weight="balanced"` / none | None |
| XGBoost | 400 trees, learning rate 0.05, max depth 5, subsample 0.8, colsample_bytree 0.8, `hist` | `scale_pos_weight` = legitimate ÷ fraud in the training data / 1 | None |

Both schemes give the frauds, taken together, the same total weight as all legitimate transactions together, so the model families are weighted comparably. The ratio is recomputed on each training set: 599.5 for Protocol A's training 80% and 496.8 for Protocol B's training block, which contains both nights and so a higher fraud rate. XGBoost is described in [6].

**Tuning: none.** Every model uses the configuration above under both protocols. With 57 frauds in the validation block, each extra configuration tried on it is another chance to fit noise, and Section 11.3 shows that even the top two models can't be separated reliably at this sample size. A small grid, chosen inside the selection data of each protocol, is listed under future work.

---

## 8. Metrics

`src/evaluate.py`.

### 8.1 Ranking quality

**PR-AUC (average precision), the headline.** Walk down the list ranked by score and add up precision at each step in recall:

```
AP = Σₙ (Rₙ − Rₙ₋₁) · Pₙ        (scikit-learn: average_precision_score)
```

A random scorer gets the fraud rate, about 0.0017. This is not the trapezoidal area `auc(recall, precision)`, which interpolates linearly and reads optimistic.

**ROC-AUC, reported alongside.** The probability that a random fraud scores above a random legitimate transaction. Its false-positive rate divides by every legitimate transaction, which hides the cost a review team pays [4]. On a test set of about 56,650 legitimate transactions, 567 false alerts is only 1% FPR. If they came with 85% recall, about 81 of 95 frauds, precision would be 81 / (81 + 567) ≈ 12.5%: seven of every eight alerts wasted.

### 8.2 At the chosen threshold

True and false positives, missed frauds, precision, recall, alerts raised, total cost, and recall at 1% FPR. The headline sentence is always in counts: "catches *x* of *y* frauds with *z* false alerts".

### 8.3 Uncertainty

- **Stratified bootstrap** with 2,000 resamples: frauds and legitimate rows are resampled separately, so every resample keeps the same fraud count. The 95% percentile interval is reported for PR-AUC. Because the scores never change, only how often each row is counted, the scores are sorted once and each resample is a weighted cumulative sum; a unit test checks the result against scikit-learn's `average_precision_score` with sample weights.
- **Paired comparisons:** two models are scored on the same resampled rows; the interval for their difference, and the share of resamples the first model wins, are reported.
- **Cross-validation spread** (Protocol A): mean ± sample standard deviation of fold PR-AUC.

With 74 to 95 frauds in a test set, one fraud crossing the threshold moves recall by one to one and a half points, and the PR-AUC intervals in Section 11 are 14 to 22 points wide.

These intervals cover the test rows only. They don't include the variation from retraining with another seed or on another sample, so they understate the total uncertainty.

### 8.4 Calibration

The Brier score, and expected versus actual frauds in score bands (0–0.01, 0.01–0.1, 0.1–0.5, 0.5–1). For calibrated scores, the scores in a band add up to roughly the number of frauds in it. Class weighting shifts scores upward by design, so weighted scores are not probabilities until recalibrated; E4 measures by how much.

---

## 9. Choosing the alert threshold

`src/threshold.py`.

### 9.1 The cost curve

For every distinct score *t* in the selection set (out-of-fold or validation predictions), compute Cost(t) = C_miss · FN(t) + C_review · FP(t), along with the option of flagging nothing. Transactions with tied scores are flagged together, because no threshold can separate them. The chosen threshold has the lowest cost; among equal costs, the highest threshold, which raises the fewest alerts. The curve is computed exactly in one pass after sorting, not on a grid, and a unit test checks it against a brute-force count.

### 9.2 A reference point: calibrated scores

If scores were calibrated probabilities *p*, flagging pays when p · C_miss > (1 − p) · C_review, that is when

```
p > t* = C_review / (C_review + C_miss) = 1 / (1 + r)
```

This is the standard cost-sensitive decision rule [5]. The cost-optimal thresholds found on data are compared with this value; the gap is a calibration diagnostic.

### 9.3 Sensitivity to the cost ratio

The optimisation is repeated for r ∈ {2, 5, 10, 20, 50, 100, 200, 500}. Output: `reports/metrics/<protocol>.json` (key `sweep`) and `reports/figures/threshold_vs_cost_ratio.png`, which plots the threshold, alerts raised and frauds caught against r.

### 9.4 Alternative: a capacity limit

If the team can review only *k* alerts, the threshold becomes the *k*-th highest score, and the useful metric is precision among the top *k*. Reported for k = 50, 100 and 200 (Section 11.4).

---

## 10. Experiments

Each experiment writes its results to `reports/metrics/`.

**E1. Which model ranks fraud best, and do the metrics agree?**
All three models, class-weighted and not, under both protocols. PR-AUC and ROC-AUC side by side, plus paired bootstraps of XGBoost against random forest and logistic regression. If the interval for a difference includes zero, the two are tied on this data and the write-up says so.

**E2. Class weights vs threshold tuning.**
Every model trained with and without class weights on the same split, compared on the selection set (`src/experiments.py`): PR-AUC with a paired bootstrap of the difference, the cost-optimal threshold at the main cost ratio, and the Spearman rank correlation between the two versions' scores.
Theory: weighting each fraud by *w* under log loss multiplies a flexible model's odds by about *w*, so an unweighted threshold *t* maps to *w·t* / (1 − *t* + *w·t*). The observed weighted threshold is reported next to this prediction. The conclusion is stated per model, not as a general law.

**E3. How much does a random split flatter the models?**
The same models and settings under Protocols A and B: the change in PR-AUC, and whether the model ranking survives. The first version measured 0.857 (stratified) against 0.801 (time-ordered) for XGBoost; this version re-measures both after de-duplication and the fixes in Section 6.1.

**E4. Are the scores probabilities?**
Brier score and expected-versus-actual frauds per score band, for weighted and unweighted XGBoost, before and after sigmoid (Platt) calibration fitted on the validation block (`CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid")`). Isotonic calibration is avoided: it is prone to overfitting with the few dozen frauds a validation block holds.

**E5. What drives the scores, and what does the model miss?**
TreeSHAP values [7] from XGBoost's own `pred_contribs=True` (no extra package), summarised as mean |SHAP| per feature. Then the test frauds missed at the chosen threshold: their amounts, hours, scores and top SHAP features, compared with the frauds that were caught, and the highest-scoring false alerts.

**E6. Amount-dependent costs.**
The dataset description suggests using `Amount` for example-dependent cost-sensitive learning. Here a miss costs the transaction amount plus a handling cost (0 by default, so a miss costs exactly its amount, as in the first version), and every alert costs 5 to review. Four policies are scored in money on the test block, every choice made on validation: flag nothing; the constant-ratio threshold of the main analysis; one threshold chosen on validation by money cost; and a per-transaction rule that flags when calibrated *p* · (amount + handling) > review cost.

---

## 11. Results

All values come from `reports/results.md`; the figures are listed in 11.10.

### 11.1 Protocol B: time-ordered test block (56,744 rows, 74 frauds)

| Model | Class weights | PR-AUC (95% CI) | ROC-AUC | Recall at 1% FPR |
| --- | --- | ---: | ---: | ---: |
| Logistic regression | balanced | 0.745 (0.637–0.859) | 0.981 | 0.878 |
| Logistic regression | none | 0.699 (0.600–0.796) | 0.971 | 0.851 |
| Random forest | balanced | 0.812 (0.726–0.890) | 0.965 | 0.865 |
| Random forest | none | 0.794 (0.703–0.874) | 0.950 | 0.865 |
| XGBoost | scale_pos_weight | 0.785 (0.693–0.868) | 0.970 | 0.878 |
| XGBoost | none | 0.796 (0.707–0.877) | 0.978 | 0.851 |
| No-skill baseline |  | 0.0013 | 0.500 | 0.010 |

### 11.2 Protocol A: stratified test set (56,746 rows, 95 frauds)

| Model | Class weights | PR-AUC, test (95% CI) | ROC-AUC, test | PR-AUC, 5-fold CV (mean ± SD) |
| --- | --- | ---: | ---: | ---: |
| Logistic regression | balanced | 0.679 (0.584–0.786) | 0.963 | 0.739 ± 0.025 |
| Logistic regression | none | 0.699 (0.608–0.797) | 0.957 | 0.755 ± 0.033 |
| Random forest | balanced | 0.812 (0.736–0.884) | 0.961 | 0.844 ± 0.037 |
| Random forest | none | 0.802 (0.726–0.873) | 0.947 | 0.841 ± 0.040 |
| XGBoost | scale_pos_weight | 0.821 (0.750–0.890) | 0.975 | 0.854 ± 0.035 |
| XGBoost | none | 0.830 (0.760–0.901) | 0.974 | 0.854 ± 0.033 |
| No-skill baseline |  | 0.0017 | 0.500 |  |

Do the two metrics agree on the order of the models?

| Protocol | Class weights | Order by PR-AUC | Order by ROC-AUC | Same order? |
| --- | --- | --- | --- | --- |
| Protocol B: time-ordered | on | Random forest > XGBoost > Logistic regression | Logistic regression > XGBoost > Random forest | **no** |
| Protocol B: time-ordered | off | XGBoost > Random forest > Logistic regression | XGBoost > Logistic regression > Random forest | **no** |
| Protocol A: stratified | on | XGBoost > Random forest > Logistic regression | XGBoost > Logistic regression > Random forest | **no** |
| Protocol A: stratified | off | XGBoost > Random forest > Logistic regression | XGBoost > Logistic regression > Random forest | **no** |

The two metrics disagree in all four comparisons, and always about the same pair: ROC-AUC puts logistic regression ahead of random forest, PR-AUC puts it far behind (on the stratified test set, ROC-AUC 0.963 against 0.961, PR-AUC 0.679 against 0.812). Logistic regression places almost every fraud above the bulk of legitimate transactions, which is what ROC-AUC measures, but its highest scores also include more legitimate transactions: its precision never rises above about 0.9 on the stratified test set (`pr_curves_random.png`), and precision is exactly what a review team pays for. With class weights on the time-ordered block, the two metrics give exactly opposite orders.

### 11.3 Paired model comparisons (difference in test PR-AUC, class-weighted models)

| Comparison (class-weighted models) | Protocol | Difference (95% CI) | Resamples the first model wins |
| --- | --- | ---: | ---: |
| XGBoost − random forest | Protocol B | -0.027 (-0.055 to -0.006) | 0.4% |
| XGBoost − logistic regression | Protocol B | +0.040 (-0.027 to +0.111) | 73.0% |
| random forest − logistic regression | Protocol B | +0.067 (+0.003 to +0.139) | 98.4% |
| XGBoost − random forest | Protocol A | +0.009 (-0.004 to +0.025) | 89.6% |
| XGBoost − logistic regression | Protocol A | +0.142 (+0.066 to +0.215) | 100.0% |
| random forest − logistic regression | Protocol A | +0.133 (+0.058 to +0.203) | 100.0% |

On the stratified test set XGBoost and random forest are tied (+0.009, interval spanning zero), and both beat logistic regression by about 0.13–0.14 with intervals well clear of zero. On the time-ordered test block random forest is ahead of XGBoost (-0.027, interval -0.055 to -0.006), although XGBoost was ahead on the validation block, where the model choice was made (0.789 against 0.768); there only random forest is clearly better than logistic regression. These intervals cover test-row sampling only (8.3). Taken together: the two tree ensembles are tied at the top, and which one "wins" depends on which few dozen frauds land in the evaluation block.

### 11.4 Operating point: Protocol B, r = 20, threshold from the validation block

| Model | Class weights | Threshold | Alerts | Frauds caught | False alerts | Frauds missed | Precision | Recall | Cost (reviews) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic regression | balanced | 0.999751 | 72 | 53 | 19 | 21 | 0.736 | 0.716 | 439 |
| Logistic regression | none | 0.0356 | 165 | 61 | 104 | 13 | 0.370 | 0.824 | 364 |
| Random forest | balanced | 0.1330 | 105 | 59 | 46 | 15 | 0.562 | 0.797 | 346 |
| Random forest | none | 0.1421 | 80 | 57 | 23 | 17 | 0.713 | 0.770 | 363 |
| XGBoost | scale_pos_weight | 0.1362 | 92 | 57 | 35 | 17 | 0.620 | 0.770 | 375 |
| XGBoost | none | 0.0533 | 87 | 58 | 29 | 16 | 0.667 | 0.784 | 349 |

At r = 20, class-weighted random forest has the lowest test cost (346 reviews: 59 of 74 frauds caught with 46 false alerts), unweighted XGBoost is next (349), and the served model, weighted XGBoost, catches 57 with 35 false alerts (375). Weighted logistic regression's threshold lands at 0.9998, where it raises the fewest alerts (72) but misses 21 frauds; unweighted logistic regression goes the other way (165 alerts, 61 caught). Every threshold was chosen on 57 validation frauds, so differences of a few frauds between models are within what that set can resolve.

The same operating point under Protocol A, with thresholds from out-of-fold predictions on the training 80%:

| Model | Class weights | Threshold | Alerts | Frauds caught | False alerts | Frauds missed | Precision | Recall | Cost (reviews) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic regression | balanced | 0.9718 | 137 | 76 | 61 | 19 | 0.555 | 0.800 | 441 |
| Logistic regression | none | 0.0180 | 123 | 75 | 48 | 20 | 0.610 | 0.789 | 448 |
| Random forest | balanced | 0.2164 | 94 | 76 | 18 | 19 | 0.809 | 0.800 | 398 |
| Random forest | none | 0.1114 | 91 | 74 | 17 | 21 | 0.813 | 0.779 | 437 |
| XGBoost | scale_pos_weight | 0.1596 | 101 | 77 | 24 | 18 | 0.762 | 0.811 | 384 |
| XGBoost | none | 0.0620 | 86 | 75 | 11 | 20 | 0.872 | 0.789 | 411 |

If the team can review only *k* alerts, the frauds among the *k* highest test scores:

| Model | Class weights | Frauds in top 50 | Frauds in top 100 | Frauds in top 200 |
| --- | --- | ---: | ---: | ---: |
| Logistic regression | balanced | 48 (96.0%) | 59 (59.0%) | 63 (31.5%) |
| Logistic regression | none | 37 (74.0%) | 55 (55.0%) | 61 (30.5%) |
| Random forest | balanced | 49 (98.0%) | 59 (59.0%) | 64 (32.0%) |
| Random forest | none | 49 (98.0%) | 58 (58.0%) | 63 (31.5%) |
| XGBoost | scale_pos_weight | 48 (96.0%) | 57 (57.0%) | 60 (30.0%) |
| XGBoost | none | 49 (98.0%) | 59 (59.0%) | 60 (30.0%) |

### 11.5 Threshold against the cost ratio (XGBoost, validation block)

Validation frauds: 57. "Alerts" and "frauds caught" are counted on the validation block, where the thresholds were chosen.

| r | 1 / (1 + r) | Threshold, weighted | Alerts | Frauds caught | Threshold, unweighted | Alerts | Frauds caught |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.333 | 0.9761 | 44 | 42 | 0.8130 | 42 | 41 |
| 5 | 0.167 | 0.9049 | 47 | 43 | 0.4565 | 48 | 43 |
| 10 | 0.091 | 0.9049 | 47 | 43 | 0.4565 | 48 | 43 |
| 20 | 0.048 | 0.1362 | 78 | 45 | 0.0533 | 72 | 45 |
| 50 | 0.020 | 0.0352 | 134 | 47 | 0.0533 | 72 | 45 |
| 100 | 0.010 | 0.0148 | 224 | 48 | 0.0063 | 163 | 46 |
| 200 | 0.005 | 0.0148 | 224 | 48 | 0.0063 | 163 | 46 |
| 500 | 0.002 | 0.0053 | 450 | 49 | 4.39e-04 | 1260 | 49 |

At r = 20 the unweighted model's threshold (0.053) sits close to the calibrated-theory value 1 / (1 + r) = 0.048, consistent with its scores being roughly probabilities (11.7). The weighted model's threshold (0.136) is higher, but nowhere near the 0.966 that the odds theory predicts for a weight of 497 (11.6). With 57 validation frauds the curve moves in steps: nothing changes between r = 5 and r = 10, or between 100 and 200. Raising r from 2 to 500 buys 7 more frauds for the weighted model (42 → 49) at about ten times the alerts (44 → 450), and 8 more for the unweighted one (41 → 49) at thirty times the alerts (42 → 1,260).

### 11.6 Class weights vs threshold (E2, selection scores)

| Model | Protocol | PR-AUC weighted | PR-AUC unweighted | Difference (95% CI) | Threshold weighted | Threshold unweighted | Odds-theory prediction | Spearman ρ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| XGBoost | Protocol B | 0.789 | 0.780 | +0.009 (-0.006 to +0.026) | 0.1362 | 0.0533 | 0.9655 | 0.863 |
| XGBoost | Protocol A | 0.852 | 0.852 | -0.001 (-0.006 to +0.005) | 0.1596 | 0.0620 | 0.9754 | 0.840 |
| Logistic regression | Protocol B | 0.754 | 0.610 | +0.143 (+0.070 to +0.222) | 0.999751 | 0.0356 | 0.9483 | 0.838 |
| Logistic regression | Protocol A | 0.738 | 0.754 | -0.016 (-0.034 to +0.003) | 0.9718 | 0.0180 | 0.9166 | 0.713 |
| Random forest | Protocol B | 0.768 | 0.765 | +0.003 (-0.017 to +0.024) | 0.1330 | 0.1421 | 0.9880 | 0.439 |
| Random forest | Protocol A | 0.843 | 0.840 | +0.003 (-0.006 to +0.013) | 0.2164 | 0.1114 | 0.9869 | 0.425 |

For XGBoost the weights left PR-AUC unchanged within noise under both protocols (+0.009 on the validation block, -0.001 out of fold) while moving the cost-optimal threshold by a factor of about 2.5 (0.053 → 0.136 and 0.062 → 0.160). At those thresholds the two versions catch the same frauds to within one (45 and 45 of 57; 319 and 320 of 378): for this model, weighting and threshold choice are two routes to the same operating point. The odds theory, which assumes a model flexible enough to shift its odds by the full weight, predicts weighted thresholds near 0.97; the observed ones are 0.14 and 0.16, so the boosted trees shift far less than that.

Logistic regression is different: the weights changed its ranking, by +0.143 on the time-ordered validation block and -0.016 (-0.034 to +0.003) out of fold, so for a linear model class weighting is a modelling choice, not just a shift of the threshold. Random forest's PR-AUC barely moves either way, and on the time-ordered protocol even its threshold barely moves (0.142 → 0.133), most likely because its leaves are nearly pure at `min_samples_leaf = 2`, and a pure leaf scores the same with or without weights. The low Spearman correlations for random forest come from the mass of legitimate transactions with near-zero scores, whose order shuffles freely, not from the top of the ranking.

### 11.7 Calibration (E4, Protocol B test block)

| Model | Brier, raw | Brier, sigmoid | Expected frauds, raw | Expected frauds, sigmoid | Actual frauds |
| --- | ---: | ---: | ---: | ---: | ---: |
| XGBoost, weighted | 0.000506 | 0.000428 | 87.6 | 69.3 | 74 |
| XGBoost, unweighted | 0.000410 | 0.000392 | 67.7 | 69.8 | 74 |

Unweighted XGBoost's raw scores are already close to probabilities: they add up to 67.7 expected frauds against 74 actual, and sigmoid calibration barely changes the Brier score (0.000410 → 0.000392). Weighted scores are inflated: they add up to 87.6, and the 31 transactions scored between 0.1 and 0.5 were expected to hold 7.3 frauds and held 1. Sigmoid calibration on the validation block brings their Brier score from 0.000506 to 0.000428 and their total to 69.3. Both models put 11–14 frauds in the lowest band, below 0.01: misses that no sensible threshold reaches (Section 12). The served model returns the calibrated probability next to the raw score (Section 15).

### 11.8 Random vs time-ordered split (E3)

| Model | Class weights | Protocol A (stratified) | Protocol B (time-ordered) | Change |
| --- | --- | ---: | ---: | ---: |
| Logistic regression | balanced | 0.679 | 0.745 | +0.066 |
| Logistic regression | none | 0.699 | 0.699 | 0.000 |
| Random forest | balanced | 0.812 | 0.812 | 0.000 |
| Random forest | none | 0.802 | 0.794 | -0.008 |
| XGBoost | scale_pos_weight | 0.821 | 0.785 | -0.037 |
| XGBoost | none | 0.830 | 0.796 | -0.035 |
| No-skill baseline |  | 0.0017 | 0.0013 |  |

A random split doesn't uniformly flatter the models. XGBoost loses about 0.035 PR-AUC on the time-ordered block, close to the 0.056 gap the first version measured (0.857 against 0.801), but random forest holds steady (0.812 and 0.812) and weighted logistic regression scores higher on the later block (0.745 against 0.679). Every change is smaller than the intervals around it (8.3), and the later block also has a lower fraud rate, which on its own makes PR-AUC harder to earn. Two days of data can't show drift convincingly; what the time-ordered protocol does show is how fragile the model ranking is (11.3).

### 11.9 Amount-dependent costs (E6, Protocol B test block, money units)

| Policy | Total cost | Fraud lost | Review spend | Alerts | Frauds caught | False alerts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Flag nothing | 7,728 | 7,728 | 0 | 0 | 0 | 0 |
| Constant cost ratio (r = 20) | 3,096 | 2,636 | 460 | 92 | 57 | 35 |
| One threshold, chosen on validation by money cost | 3,425 | 2,635 | 790 | 158 | 58 | 100 |
| Per transaction: calibrated p × (amount + handling) > review cost | 2,824 | 2,679 | 145 | 29 | 23 | 6 |

Costing each miss at its own amount changes the ranking of policies. The constant-ratio threshold cuts the cost of fraud from 7,728 (flag nothing) to 3,096. The per-transaction rule is cheaper still, 2,824, but it gets there by catching only 23 of 74 frauds: it lets small frauds through, because reviewing one costs more than the fraud is worth, and 36% of frauds are 1.00 or less (Section 4). A real team would probably want a floor under that, since tiny charges can be card testing that precedes larger fraud. The single money-optimal threshold chosen on validation (0.035) did worse on test (3,425) than the constant-ratio one: validation frauds were about twice as large on average as test frauds (215 against 104, notebook 01), so the threshold it chose bought more reviews than the test block repaid.

### 11.10 Figures (`reports/figures/`)

| File | Shows |
|---|---|
| `pr_curves_{random,time}.png` | Precision-recall curves for the three class-weighted models |
| `roc_curves_{random,time}.png` | ROC curves for the same models |
| `cost_curve_time.png` | Cost against threshold on the validation block, weighted and unweighted XGBoost, chosen points marked |
| `threshold_vs_cost_ratio.png` | Table 11.5 as a chart: threshold, alerts raised and frauds caught against r |
| `score_bands_time.png` | Expected vs actual frauds per score band, weighted and unweighted, raw and calibrated |
| `shap_bar_time.png` | Mean \|SHAP\| per feature |
| `score_hist_time.png` | Score distribution by class, log-scaled y-axis |
| `amount_by_class.png`, `volume_and_fraud_rate_by_hour.png`, `top_components_by_class.png`, `correlation_heatmap.png` | Exploratory figures from notebook 01 |

---

## 12. Explanations and error analysis

- **Most influential features** (mean \|SHAP\|, weighted XGBoost, test block): V4 (1.84), V14 (1.68), V12 (1.27), V11 (0.89) and V10 (0.81), then V1, V8, V19, V17 and `log_amount`. Three of the top five (V14, V12, V10) are among the six largest class differences in the EDA, and V4 and V11 are in its top ten. V17, the largest effect size in the EDA, ranks ninth here, most likely because it repeats what V14 and V12 already tell the model.
- **Missed frauds** at the chosen threshold: 17 of 74. They aren't near misses: the highest-scoring one scored 0.038 against a threshold of 0.136, and the median missed fraud scored 0.0003, so lowering the threshold to reach them would flood the queue. Their median amount is 12.31 against 2.00 for the 57 frauds caught, and in most of them the components that usually push a score up (V12, V14, V4) push it down instead. The full list is in `reports/results.md`.
- **Highest-scoring false alerts:** four of the top five are transactions of 0.77 with strongly fraud-like V14, V4 and V10, scoring above 0.99. Tiny, fraud-shaped transactions like these are either legitimate look-alikes of card testing or frauds the labels missed; the data can't tell which.

Because V1–V28 are anonymised, explanations say which components matter, not what they mean in business terms.

---

## 13. Model card

Following Mitchell et al. (2019) [8].

| | |
|---|---|
| **Model** | XGBoost classifier with `scale_pos_weight`, version `xgb_w-time-v2`, trained on the training block of Protocol B |
| **Intended use** | Ranking card transactions for manual review, in a research and portfolio setting |
| **Not intended for** | Live decisions about real customers; data without the same 30 input columns; other periods without re-validation |
| **Training data** | ULB credit-card dataset (two days, September 2013), de-duplicated, earliest 60% by time: 170,236 transactions, 342 frauds |
| **Evaluation data** | Latest 20% by time: 56,744 transactions, 74 frauds |
| **Headline metrics** | PR-AUC 0.785 (95% CI 0.693–0.868); at the chosen threshold, 57 of 74 frauds caught with 35 false alerts |
| **Decision threshold** | 0.1362 on the raw score, minimising cost at r = 20 on the validation block |
| **Outputs** | Raw score (compared with the threshold) and a sigmoid-calibrated probability (fitted on the validation block) |
| **Known limitations** | Section 16. On the same test block, random forest ranked frauds better (11.3) |
| **Fairness** | Cannot be assessed: the data has no customer attributes |

The served model was fixed in `configs/config.yaml` before the test block was scored and was not changed afterwards. On the validation block XGBoost had the highest PR-AUC, so switching to random forest because it did better on the test block would be choosing on test data.

---

## 14. Reproducibility

- **Environment:** results produced with Python 3.13.18, scikit-learn 1.9.1, XGBoost 3.4.1, pandas 3.0.5 and NumPy 2.5.3. Exact versions of every direct dependency are in `requirements-lock.txt`; `requirements.txt` gives the minimums.
- **Seeds:** 42 for splits and models, 0 for the bootstrap. Re-running the time-ordered protocol reproduced every number in its metrics JSON exactly, except one random-forest Spearman correlation that moved in the ninth decimal place (random forest averages its trees in parallel, so floating-point sums can come out in a different order).
- **Configuration:** every setting is in `configs/config.yaml`; each metrics JSON stores the configuration it ran with, the package versions and the git commit.
- **Data:** the SHA-256 of `creditcard.csv` is written to `reports/data_checks.json`.
- **Tests:** `pytest` runs 35 tests on synthetic data, so they need neither the dataset nor a trained model: de-duplication, features, both splits (time blocks never share a timestamp), the cost curve and the money-cost threshold against brute-force counts, the fast bootstrap against scikit-learn, threshold provenance (rule 3 of 6.1), an end-to-end run of both protocols and the report, and the API, including its demo endpoints.
- **Runtime:** on a 2-core machine, the time-ordered protocol took about 8 minutes and the stratified one about 44 minutes, most of it random forest (five cross-validation fits plus a final fit, with and without weights).

```bash
python -m src.data --download
python -m src.train --protocol random
python -m src.train --protocol time
python -m src.report
pytest
```

---

## 15. Serving

`app/main.py` is a FastAPI service. It loads the bundle `{model, calibrated_model, threshold, version}` saved by `python -m src.train --protocol time` and applies the same `build_features` used in training, so training and serving can't drift apart.

**`POST /predict`.** The request body has `Time` (≥ 0), `Amount` (≥ 0) and all of `V1`–`V28` (floats, required). Response for a fraud from the test block (amount 1.18):

```json
{
  "fraud_score": 0.99997,
  "flag_for_review": true,
  "threshold": 0.1362,
  "model_version": "xgb_w-time-v2",
  "fraud_probability": 0.8766
}
```

`fraud_score` is the raw model score that the threshold applies to; `fraud_probability` is the sigmoid-calibrated estimate (E4). A missing field, an unknown field, a negative amount or a NaN returns HTTP 422. FastAPI's default 422 response echoes the offending value back, which crashes on NaN because NaN isn't valid JSON, so the service uses its own validation handler that reports where and why without the value; a test covers this. `GET /health` returns the model version. Run locally with `uvicorn app.main:app --reload`; interactive docs are at `/docs`.

**Demo page and endpoints.** `http://127.0.0.1:8000/` serves a one-page demo (`app/static/index.html`) built on these endpoints (`app/demo.py`), all drawing on Protocol B's test block, which the model never saw: the `/predict` page offers a real test-block fraud and a real legitimate transaction as ready-made examples; `GET /demo/sample` scores a random mix of real frauds and legitimate transactions and labels each decision *fraud caught*, *fraud missed*, *false alert* or *correctly approved*; `GET /demo/test-block` scores the whole block and returns the operating point of Section 11.4 (57 of 74 frauds caught, 35 false alerts); `GET /demo/transaction` returns one real transaction to paste into `/predict`. They read `data/raw/creditcard.csv` (or `DATA_PATH`) and answer 503 if it is missing.

---

## 16. Limitations

1. Two days of 2013 data: no evidence about drift over weeks or months.
2. Anonymised PCA features: no card-level history, merchants or velocity features, and no business-language explanations.
3. Few positives: 57 frauds in the validation block and 74 to 95 in the test sets, so intervals are wide and the top two models can't be separated (11.3).
4. Assumed costs: real costs depend on amount, chargeback rules and review capacity. The sweep (9.3) and E6 show the sensitivity.
5. Static evaluation: no label delay, no feedback loop from investigated transactions.
6. The bootstrap intervals cover test-row sampling only, not retraining variability.
7. Fairness can't be assessed: no customer attributes.

---

## 17. Future work

- A small hyperparameter grid chosen inside each protocol's selection data, and seed-to-seed variation added to the intervals.
- Amount-dependent costs as the main decision rule (E6), with better-calibrated probabilities and a floor for card-testing patterns, which a pure money rule lets through.
- Rolling-origin evaluation across the two days, to see how stable the threshold is over time.
- Card- and terminal-level aggregate features, on data that has the IDs, such as the simulated datasets of the ULB/Worldline Fraud Detection Handbook [3].
- Monitoring once served: score-distribution drift and alert-rate checks.
- Bootstrap intervals for the chosen threshold itself, not only for PR-AUC.

---

## 18. References

1. A. Dal Pozzolo, O. Caelen, R. A. Johnson, G. Bontempi. *Calibrating Probability with Undersampling for Unbalanced Classification.* IEEE Symposium on Computational Intelligence and Data Mining (CIDM), 2015.
2. A. Dal Pozzolo, G. Boracchi, O. Caelen, C. Alippi, G. Bontempi. *Credit Card Fraud Detection: A Realistic Modeling and a Novel Learning Strategy.* IEEE Transactions on Neural Networks and Learning Systems, 29(8), 3784–3797, 2018.
3. Y.-A. Le Borgne, W. Siblini, B. Lebichot, G. Bontempi. *Reproducible Machine Learning for Credit Card Fraud Detection: Practical Handbook.* Université Libre de Bruxelles, 2022. https://fraud-detection-handbook.github.io/fraud-detection-handbook/
4. T. Saito, M. Rehmsmeier. *The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets.* PLOS ONE 10(3): e0118432, 2015. https://doi.org/10.1371/journal.pone.0118432
5. C. Elkan. *The Foundations of Cost-Sensitive Learning.* IJCAI, 2001.
6. T. Chen, C. Guestrin. *XGBoost: A Scalable Tree Boosting System.* KDD, 2016. https://arxiv.org/abs/1603.02754
7. S. M. Lundberg, S.-I. Lee. *A Unified Approach to Interpreting Model Predictions.* NeurIPS, 2017. https://arxiv.org/abs/1705.07874
8. M. Mitchell et al. *Model Cards for Model Reporting.* FAT\*, 2019. https://arxiv.org/abs/1810.03993
9. K. Hayat, B. Magnier. *Data Leakage and Deceptive Performance: A Critical Examination of Credit Card Fraud Detection Methodologies.* arXiv:2506.02703, 2025. https://arxiv.org/abs/2506.02703
10. Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud · OpenML: https://www.openml.org/d/1597

---

## 19. Appendix: changes from the first version

The first version was one training script (`src/train.py`) with a companion notebook. Its numbers are kept here so the change is traceable.

| Area | First version | This version |
|---|---|---|
| Headline number | XGBoost PR-AUC 0.857 on a stratified 75/25 split (0.862 ± 0.025 in 5-fold CV) | Time-ordered test block, with the stratified split reported alongside |
| Time-ordered check | XGBoost only, PR-AUC 0.801 on the last 25% by time | Protocol B, all models, with intervals |
| Duplicates | Kept (1,081 copies) | Removed before splitting |
| Features | All 30 raw columns, including raw `Time` and `Amount` | V1–V28, log amount, hour of day as sine and cosine; raw `Time` dropped |
| Threshold chosen on | The test set: the cost curve was minimised on the same rows it was reported on | Out-of-fold or validation predictions only |
| Cost model | A miss costs its amount, a review costs 5 | Constant ratio r = 20 for the main analysis; the amount-based model kept as E6 |
| Logistic regression scaling | StandardScaler fitted on the training split (already leak-free) | StandardScaler inside a pipeline |
| Uncertainty | Fold standard deviation | Stratified bootstrap intervals and paired comparisons |
| Code | One script and a notebook | Modules, configuration file, 35 tests, one command per protocol, an API |

**Re-running the first version.** The first version's `train.py`, run unchanged on the same file with the library versions in `requirements-lock.txt`, reproduces its logistic regression and random forest numbers exactly. XGBoost's numbers move slightly: PR-AUC 0.859 instead of 0.857, 5-fold CV 0.861 instead of 0.862, time-ordered 0.796 instead of 0.801. Its finding that removing the class weights left PR-AUC unchanged does not survive the rerun: unweighted XGBoost now scores 0.844 against 0.859 weighted, and the cost-optimal thresholds become 0.71 weighted and 0.06 unweighted instead of 0.52 and 0.055. A result that changes with a library upgrade was never strong evidence, which is why this version reports that comparison with an interval (11.6).
