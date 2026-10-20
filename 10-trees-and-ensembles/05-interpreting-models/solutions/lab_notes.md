# interpreting models · lab notes

numbers are from the reference solutions (topics 02 and 04 for the models, this topic for the tools) with the lab's seeds. the forest draws random numbers, so your exact digits will differ; the picture shouldn't.

## experiment 1: which feature matters?

**what people usually predict:** that $x_1$, the only feature related to $y$, has by far the largest importance, with the noise features near zero; perhaps some suspicion that the continuous one gets "a bit" of importance.

**what happens:**

| feature | impurity | permutation (train) | permutation (test) |
|---|---|---|---|
| $x_1$ (signal) | 0.159 | 0.290 | **0.156** ± 0.002 |
| coin | 0.038 | 0.041 | 0.001 ± 0.001 |
| 4 levels | 0.086 | 0.107 | 0.003 ± 0.005 |
| 10 levels | 0.165 | 0.190 | 0.013 ± 0.007 |
| 20 levels | 0.213 | 0.165 | 0.006 ± 0.004 |
| continuous | **0.339** | 0.256 | 0.015 ± 0.006 |

by impurity importance the continuous noise feature is the most important, and the 20- and 10-level noise features also beat $x_1$, which comes **fourth**. permutation importance on the *training* rows is almost as wrong. permutation importance on the test rows gets it right: $x_1$ at 0.156 (shuffling it costs 15.6 points of accuracy, nearly all of the forest's edge over chance), every noise feature within about two standard deviations of 0. the forest's test accuracy is 0.663 against the best possible 0.70.

**why:** with fully grown trees, most splits happen deep down, in small nodes, where the job is fitting noise, and any feature can reduce training impurity there. a feature with more distinct values offers more thresholds, and the best of many candidate thresholds on noise reduces impurity more than the best of few (it's selection bias: a maximum over more random draws is larger). so high-cardinality noise features are chosen more often and collect more impurity decrease: the importance measures how useful a feature was for *memorizing the training set*. the binary $x_1$ offers one threshold, which is used near the root (where it really helps), but its total is diluted by the hundreds of noise splits below. permutation importance on the training rows has the same problem in a different form: the forest memorized the training labels using the noise features, so shuffling them breaks the memorization and the training score drops. on fresh rows that memorization was worthless anyway, so shuffling the noise doesn't hurt. this is Strobl et al. (2007)'s result; their fix for forests is conditional inference trees, the practical fix is: compute importances on held-out data.

## experiment 2: twins

**what people usually predict:** that $x_1$ and its copy $x_3$ split $x_1$'s importance roughly in half, and that dropping them both costs about what dropping $x_2$ costs.

**what happens:** the boosted model's test $R^2$ is 0.943. permutation importance: $x_2$ **0.966**, $x_1$ **0.528**, $x_3$ 0.062, noise 0.001. drop-column (refit) importance: $x_1$ alone 0.001, $x_3$ alone 0.000, noise 0.000, $x_2$ 0.567, and $x_1$ and $x_3$ together **0.508**.

**why:** the two signals matter equally ($y = x_1 + x_2$), so their true information content is equal, and the refits show it: losing $x_2$ or losing both copies of $x_1$ costs about half the $R^2$. but each copy alone can be dropped for free: the refitted model just uses the other one. permutation importance measures the *fitted* model's reliance, and the booster happened to put most of its $x_1$ splits on $x_1$ and some on $x_3$, so shuffling $x_1$ leaves part of the information in $x_3$: 0.528 instead of ~0.97. (the numbers don't add up: 0.528 + 0.062 is less than 0.966, because shuffling only $x_1$ also creates impossible rows where $x_1$ and $x_3$ disagree wildly, and the model's predictions there are an odd mixture.) the two methods answer two questions. permutation: "how much does *this model* rely on the feature?" (useful for debugging the model). drop-column: "how much would we lose if we didn't have the feature?" (useful for deciding what to collect). with correlated features, neither alone tells you how much the feature matters "in the world"; report groups of correlated features together.

## experiment 3: an effect that averages away

**what people usually predict:** a rising line or an S-curve for $x_1$'s partial dependence, since $x_1$ obviously affects the prediction.

**what happens:** the partial dependence of $x_1$ is almost flat (range **0.067** over $[-1, 1]$). the ICE curves form two fans: rising for rows with $x_2 > 0$ (red), falling for $x_2 < 0$ (blue), with slopes up to about $\pm1$. the SHAP values of $x_1$ are far from 0 (mean $|\phi|$ **0.124**), their sign set by the sign of $x_1x_2$; $x_2$ gets the same mean $|\phi|$ (0.124), and $x_3$ gets 0.525. efficiency holds to rounding ($2\times10^{-16}$). switching the dropdown to $x_3$, all three views agree: a straight rising PD line, parallel ICE curves, and SHAP values on a line.

**why:** partial dependence averages $f(v, x_{i,-1}) = vx_{i,2} + x_{i,3}$ over rows; since $x_2$ is symmetric around 0, the average slope is $\bar x_2 \approx 0$, and the plot says "$x_1$ does nothing" about a feature that drives half the model. ICE curves undo the averaging: each row's curve has slope $x_{i,2}$. Shapley values split the interaction $x_1x_2$ equally between the two features (on paper 4: $\phi_1 = \phi_2 = x_1x_2/2$ for centred, independent features), which is why both get the same mean $|\phi|$, and why plotting $\phi_1$ against $x_1$, coloured by $x_2$, shows the interaction as an X shape. $x_3$ enters additively, so averaging loses nothing and every method sees the same line. the lesson: always look at ICE curves (or coloured SHAP dependence plots) before trusting a PD plot, and read a flat PD as "no *average* effect", never as "no effect".
