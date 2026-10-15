# what is learning? · lab notes

## experiment 1: memorizer, stump, and baselines

**what people usually predict:** the lookup table is best on training data and "pretty good" on test data, maybe 70–80%, since it has seen 200 examples; the stump is mediocre everywhere because it can only use one of the two features.

**what happens** (15% of labels flipped, seed 0):

| model | train | test |
|---|---|---|
| dummy (most frequent) | 0.51 | 0.50 |
| dummy (stratified) | 0.54 | 0.51 |
| lookup table | **1.00** | **0.50** |
| stump | 0.78 | **0.75** |

the best possible test accuracy is 0.85 (the noise floor). the lookup table is perfect on the training set and *exactly* the `most_frequent` dummy on the test set. the stump, which ignores half the information, is far better than both, and its training accuracy predicts its test accuracy to within 3 points.

**why:** with continuous inputs, no test point ever equals a training point, so the table always falls back to the majority class: memorizing gave it zero information about new points. the stump's hypothesis class is tiny (about $2 \cdot 2 \cdot 200$ distinct rules), so its training error is an honest estimate of its test error (idea 5's bound). its accuracy is limited by *approximation* error instead: the best rule using only $x_0$ agrees with the true line on $1 - \arctan(0.5)/\pi \approx 85\%$ of inputs, and with 15% label noise that gives $0.85 \cdot 0.852 + 0.15 \cdot 0.148 \approx 0.75$, which is what you measure. at 45% noise there is almost no signal left (the best possible is 0.55): the stump's training accuracy is 0.61 but its test accuracy 0.52, a gap that is mostly the stump fitting noise (experiment 3 isolates this effect), while the lookup table's training accuracy stays at 1.00. memorizing the noise looks perfect however much noise there is.

## experiment 2: when does memorizing work?

**what people usually predict:** the lookup table never generalizes ("it's memorizing"), or it generalizes once $n$ is "large", independent of $d$.

**what happens** (mean test accuracy over 5 seeds):

| d | possible rows | n = 100 | n = 1,000 | n = 10,000 |
|---|---|---|---|---|
| 3 | 8 | 0.90 | 0.90 | 0.90 |
| 6 | 64 | 0.73 | 0.90 | 0.90 |
| 10 | 1,024 | 0.53 | 0.70 | 0.90 |
| 16 | 65,536 | 0.51 | 0.51 | 0.55 |

with $d = 3$ the table hits the best possible accuracy (0.90) by $n \approx 100$ and is the best model you could build. the curves move right by roughly a factor of $2^d$: each needs a few times $2^d$ examples. the right panel explains it: test accuracy tracks the fraction of test rows already seen in training. the stump (dashed) is stuck near 0.70 at every $n$: it can only look at one feature, and the label depends on three.

**why:** for each row it has seen, the table stores the majority label, which is a counting estimate of $\arg\max_y P(y \mid x)$, the Bayes-optimal prediction. when every possible row has been seen many times, that estimate is right and memorization is optimal. the catch is the number of possible inputs, which grows exponentially with $d$ (and is infinite for continuous features). this is the **curse of dimensionality** in its simplest form, and it's why every useful model makes an assumption that lets it generalize to *unseen* inputs: smoothness (nearest neighbours, chapter 09), linearity (chapter 08), a few relevant features (the stump), and so on.

## experiment 3: ERM on pure noise

**what people usually predict:** about 50% train accuracy (the labels are random), maybe a little more.

**what happens** ($n = 50$, mean of 20 runs):

| d | 1 | 10 | 100 | 200 | 500 |
|---|---|---|---|---|---|
| train | 0.62 | 0.69 | 0.73 | 0.75 | 0.77 |
| fresh data | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 |

even with a single feature the best stump "learns" 62% of coin flips; with 200 features it gets three quarters of them right. on new data every one of them is exactly a coin flip. the finite-class bound sits above the curve all the way and grows at about the same rate.

**why:** ERM picks the best of many candidates, and the best of many noisy scores is biased upwards. one feature already offers about $2n$ thresholds to choose from; each extra feature adds more chances to find a lucky split. the bound's penalty $\sqrt{(\log|\mathcal H| + \log(2/\delta))/(2n)}$ captures the $\sqrt{\log d}$ growth because it's exactly this "maximum of many" argument (a union bound), but it's conservative: it must hold with 95% probability for *every* rule, while the curve is an average. with a larger $n$ (try the slider) everything shrinks towards 0.5 like $1/\sqrt n$. the lesson that the rest of the chapter builds on: a score computed on the data you used to *choose* the model, whether the choice was a threshold, a feature or a hyperparameter, is optimistic. only data that played no part in any choice gives an honest estimate. topic 02 builds the machinery for that, and its lab shows the same effect hiding inside a cross-validation done wrong.
