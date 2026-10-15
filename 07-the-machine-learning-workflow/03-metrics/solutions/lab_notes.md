# metrics · lab notes

## experiment 1: 99% accuracy

**what people usually predict:** the dummy gets around 99% (most people see that coming), and a model that actually uses the feature does clearly better, maybe 99.5%+, with decent recall.

**what happens** (20,000 test transactions, 1% fraud):

| model | accuracy | precision | recall | F1 | ROC AUC | avg precision |
|---|---|---|---|---|---|---|
| dummy (most frequent) | 0.990 | 0.000 | 0.000 | 0.000 | 0.500 | 0.010 |
| stump ($x > 3.23$) | **0.992** | 0.755 | **0.192** | 0.306 | 0.964 | 0.453 |
| top 2% ($x > 2.22$) | 0.984 | 0.325 | **0.622** | 0.427 | 0.964 | 0.453 |

the stump beats the dummy's accuracy by 0.2 points and catches one fraud in five. the "top 2%" rule catches three in five, with more false alarms (249 against 120 frauds caught), and has *lower* accuracy than doing nothing.

**why:** accuracy weighs every mistake equally, and 99% of the examples are negatives. the stump does ERM for accuracy, so it puts the threshold where a fraud becomes more likely than not, which for a 1% class is far out in the tail ($\approx \mu/2 + \log(99)/\mu \approx 3.1$): it only flags transactions that are almost surely fraud. whether "top 2%" is better depends on costs the metric doesn't know: if a missed fraud costs 50 times a false alarm, it is far better. both thresholded models share the same score, so ROC AUC and AP are identical for them: those measure the ranking, not the threshold. AUC (0.96) looks excellent because it averages over all false positive rates, including ones irrelevant here; AP (0.45) says that at useful recall levels, about half the alarms are false. report the confusion matrix, precision and recall at the operating threshold, and a ranking metric that suits the class balance.

## experiment 2: change the prevalence

**what people usually predict:** both AUC and AP get worse as the positives get rarer, because "imbalance makes everything harder".

**what happens:**

| prevalence | ROC AUC | average precision |
|---|---|---|
| 50% | 0.962 | 0.963 |
| 10% | 0.962 | 0.813 |
| 1% | 0.962 | 0.472 |
| 0.1% | 0.949 | 0.161 |

ROC AUC stays at 0.95–0.96 and the ROC curves lie on top of each other (the 0.1% curve, at 0.949, is just noisier: it has only 189 positives). the precision–recall curves collapse towards the bottom left.

**why:** TPR $= P(\text{flag} \mid \text{positive})$ and FPR $= P(\text{flag} \mid \text{negative})$ condition on the class, so they don't depend on how common each class is; the ROC curve, and AUC, describe the score distributions alone. precision $= P(\text{positive} \mid \text{flag})$ conditions on the *prediction*, and by Bayes' rule it carries the prior: $\frac{\text{TPR}\,\pi}{\text{TPR}\,\pi + \text{FPR}(1 - \pi)}$. as $\pi$ shrinks, even a small FPR produces false alarms that swamp the true ones. so: ROC AUC to compare *models* across datasets with different balance, PR curves and AP to judge whether a model is *useful* at the prevalence you'll deploy at (the AP of a random model is the prevalence, the right baseline).

## experiment 3: same ranking, different probabilities

**what people usually predict:** all four metrics differ, or none do ("they're the same model").

**what happens** ($T = 3$):

| model | accuracy | ROC AUC | log loss | ECE |
|---|---|---|---|---|
| calibrated | 0.807 | 0.858 | **0.419** | **0.005** |
| overconfident (×3) | 0.807 | 0.858 | 0.643 | 0.121 |
| underconfident (÷3) | 0.807 | 0.858 | 0.514 | 0.160 |

accuracy and AUC are identical to the last digit. log loss and ECE separate them, and they disagree about which miscalibrated model is worse: ECE rates the underconfident model worse, log loss the overconfident one.

**why:** scaling the logit $z$ by $T$ preserves its sign (so every prediction at 0.5 is unchanged) and its order (so the ranking, and AUC, are unchanged). only the probability values move: towards 0 and 1 for the overconfident model, towards 0.5 for the underconfident one. on the reliability diagram the overconfident curve is too flat (when it says 0.97, the truth is more like 0.85) and the underconfident one too steep. ECE averages absolute gaps, and the underconfident model's gaps are larger on average, while log loss is driven by the tails: an overconfident model that says 0.99 for a negative pays $-\log 0.01 \approx 4.6$ nats, and those confident mistakes dominate. only the calibrated model, the true conditional probability, minimizes log loss (it's a proper scoring rule). calibration matters whenever someone acts on the probability itself: expected costs, risk scores shown to people, combining models. chapter 28 fixes miscalibration after training with temperature scaling, which is exactly this experiment run backwards.
