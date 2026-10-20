# bounds in practice · lab notes

numbers are from the reference solution at the shipped seeds, on the cached datasets (California housing: 20,433 districts; MNIST: 60,000/10,000). offline (`MLFS_OFFLINE=1`) the labs run on synthetic stand-ins and the numbers differ. measured errors are on fixed held-out sets, and every bound is at $\delta = 0.05$.

## experiment 1: a learning curve with its bounds

**what people usually predict:** that with 16,000 examples and only 9 parameters, any reasonable bound should be within a few percentage points of the test error.

**what happens:**

| model | train error | test error | gap | VC bound | 32-bit bound | test-set bound |
|---|---|---|---|---|---|---|
| logistic, n = 100 | 0.110 | 0.167 | +0.057 | 1.928 | 1.117 | 0.177 |
| logistic, n = 1,000 | 0.155 | 0.158 | +0.003 | 0.859 | 0.473 | 0.168 |
| logistic, n = 3,000 | 0.156 | 0.157 | +0.001 | 0.594 | 0.340 | 0.167 |
| logistic, n = 10,000 | 0.161 | 0.156 | −0.005 | 0.418 | 0.262 | 0.166 |
| logistic, n = 16,000 | 0.164 | 0.156 | −0.008 | **0.372** | **0.243** | **0.165** |
| 15-NN, n = 16,000 | 0.136 | 0.153 | +0.018 | ∞ | – | 0.163 |

the measured gap is below 0.01 in absolute value from $n = 1000$ on (it even turns slightly negative, which is sampling noise in the 4,000-district test set: its standard error is about 0.006). the VC bound is vacuous (above 0.5, the coin-flip level, and above 1 at $n \le 300$) until about $n = 3000$, and at $n = 16{,}000$ it's still 0.21 above the test error. the crude 32-bit counting bound is tighter at every $n$. (the counting bound is for the model stored in float32; rounding the 9 fitted weights to float32 changes no training prediction at any $n$, so the float64 training error is the float32 model's.) the test-set bound is within 0.01 of the test error throughout. your `vc_sample_size(9, ε, 0.05)` says the VC complexity term alone needs 17,562 districts for ε = 0.2, 81,277 for 0.1 and 368,653 for 0.05: more than California has. 15-NN has the lowest test error but no uniform bound at all.

**why:** the gap between bound and measurement comes from stacked worst cases: a union bound over every labeling of a double sample (with Sauer's polynomial, which counts labelings no sensible fit would produce), the factor 8 from Hoeffding on a difference at scale $\varepsilon/2$, the $\ln n$ in $d\ln(2en/d)$, and the requirement to hold for every distribution and every hypothesis, not the one you got. under the square root, the 32-bit bound has $(9 \times 32\ln 2 + \ln 20)/(2n) \approx 101/n$, while the VC bound has $8(9\ln(2en/9) + \ln 80)/n \approx 696/n$ at $n = 16{,}000$, so the counting bound wins despite its crude reasoning. the test-set bound has none of these costs: it bounds one fixed classifier, the number of test errors is exactly binomial, and the tail is inverted exactly. its price is 4,000 districts withheld from training, and validity for that one classifier only. that's the practical answer to "is my model good?"; the uniform bounds answer "how does generalization scale?", and they get the scaling right ($\sqrt{d/n}$ up to logs) while getting the level badly wrong.

## experiment 2: count the bits

**what people usually predict:** that fewer bits per weight means a worse model and therefore a worse bound, or that 32 bits is needed for logistic regression to work.

**what happens:** (MNIST even vs odd, 60,000 training images, 785 weights)

| bits | train error | test error | counting bound | test-set bound |
|---|---|---|---|---|
| 2 | 0.222 | 0.211 | 0.317 | 0.218 |
| 3 | 0.148 | 0.146 | 0.264 | 0.152 |
| 4 | 0.115 | 0.112 | **0.250** | 0.117 |
| 6 | 0.098 | 0.100 | 0.263 | 0.105 |
| 8 | 0.098 | 0.100 | 0.288 | 0.105 |
| 32 | 0.098 | 0.100 | 0.479 | 0.105 |

the VC bound for the real-valued model is **0.893**. (every row is evaluated with the weights actually rounded to the counted precision: the fixed grid for 2–8 bits, float16 and float32 for 16 and 32 bits; rounding to float16 or float32 changes no error in the table.) the counting bound is below it at every bit width, and it's tightest at 4 bits (**0.250**), where the quantized classifier's test error has risen only from 0.100 to 0.112. from 6 bits up the classifier is indistinguishable from the float model, and the counting bound grows linearly with the bits it pays for but doesn't use.

**why:** the counting bound's complexity term is $\sqrt{(785\,b\ln 2 + \ln 20)/(2 \times 60000)}$: it shrinks like $\sqrt b$. quantizing trades complexity for training error: at 2 bits the weights are too coarse and the training error jumps, at 32 bits the bound pays for 25 bits per weight of precision the classifier doesn't need. the sweet spot says the learned classifier carries only a few bits of information per weight, a theme that returns in compression-based and PAC-Bayes bounds (topic 06) and in int8 inference (ch 26). the VC bound loses because its constant 8 and its $\ln(2en/d) \approx 6.0$ multiply $d = 785$: under the square root it has about $37{,}900/n$, against $(785 \times 4\ln 2 + \ln 20)/(2n) \approx 1{,}090/n$ for the 4-bit counting bound. validity requires the quantization grid ($2^b$ levels on $[-2, 2]$, values beyond clipped) to be fixed before seeing the data: choosing the grid from the learned weights would make the class depend on the sample, and you'd have to pay for that choice (with a union bound over the grids you considered, which is cheap if they're few). choosing the *best* $b$ after looking at the bounds costs a factor 7 in $\delta$ (7 values tried), which changes the 4-bit bound by less than 0.001.

## experiment 3: coin flips that look like classifiers

**what people usually predict:** that the best of many random classifiers still looks like about 50% error, or that a 95% bound fails about 5% of the time no matter how it was chosen.

**what happens:** (300 repetitions, 500 test examples)

| k | best test error (mean) | naive bound (mean) | naive misses | δ/k bound (mean) | δ/k misses |
|---|---|---|---|---|---|
| 1 | 0.499 | 0.537 | 0.05 | 0.537 | 0.05 |
| 10 | 0.467 | 0.504 | 0.36 | 0.525 | 0.02 |
| 100 | 0.443 | 0.481 | **1.00** | 0.518 | 0.04 |
| 1,000 | 0.428 | 0.466 | **1.00** | 0.516 | 0.04 |
| 10,000 | 0.413 | 0.451 | **1.00** | 0.513 | 0.06 |

with 100 or more coin flips, the best one shows 41–44% test error and its naive "95%" bound excludes the true error 0.5 in every single repetition. the $\delta/k$ bound misses 2–6% of the time, consistent with its 5% guarantee (with 300 repetitions a frequency has a standard error of about 0.013, so 0.06 is noise), and it's only 0.04 wider than the naive bound even at $k = 10{,}000$.

**why:** each coin has error count $\text{Binomial}(500, 0.5)$, with sd 11 errors (0.022 in error rate). the minimum of $k$ such counts sits about $\sqrt{2\ln k}$ standard deviations below the mean: 0.067 below 0.5 at $k = 1000$. the naive bound adds only about 1.7 standard deviations (0.037), so it can't reach back to 0.5. selection turned noise into an apparent signal, and the test set stopped being a test set. the union bound adds $\ln k$ to the confidence budget, which widens a binomial bound by roughly $\sqrt{2\ln k}$ standard deviations: exactly enough to compensate, and logarithmic in $k$, so a few thousand comparisons are affordable if you account for them. the same arithmetic explains overfitting to a public leaderboard, and why a test set used to choose among many models must be replaced by a fresh one for the final number (or protected by methods like the reusable holdout of Dwork et al., 2015).
