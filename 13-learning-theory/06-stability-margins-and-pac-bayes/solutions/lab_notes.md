# stability, margins & PAC-Bayes · lab notes

numbers are from the reference solution at the shipped seeds, on the cached datasets (MNIST 60,000/10,000; California housing 20,433 districts). offline (`MLFS_OFFLINE=1`) the labs run on synthetic stand-ins and the numbers differ. every bound uses $\delta = 0.05$.

## experiment 1: a non-vacuous bound, actually computed

**what people usually predict:** after topic 04, that any bound computed from training data alone will be vacuous for a 785-parameter model, or, at the other extreme, that a bound for "the same" classifier should land within a point of its test error.

**what happens:**

| | empirical | test | bound |
|---|---|---|---|
| deterministic $w$ | train error 0.0979 | test error 0.1000 | – |
| VC bound ($d = 785$) | – | – | **0.893** |
| PAC-Bayes, prior at 0 ($\sigma = 0.184$, KL = 1,297) | Gibbs train 0.1478 | Gibbs test 0.1480 | **0.232** |
| PAC-Bayes, data-dependent prior ($\sigma = 0.144$, KL = 278) | Gibbs on the bound half 0.1291 | Gibbs test 0.1310 | **0.180** |
| test-set bound (10,000 images) | – | test error 0.1000 | **0.1051** |

the PAC-Bayes bound is non-vacuous with either prior: 0.232 with the prior at 0, 0.180 with a prior trained on the other half of the data, against 0.893 for the VC bound. the bound-versus-$\sigma$ curve is U-shaped: at small $\sigma$ the KL term ($\|w\|^2/2\sigma^2$, up to 4.9 million nats at $\sigma = 0.003$) makes the bound vacuous; at large $\sigma$ the Gibbs risk itself climbs towards 0.4. the Gibbs classifier's training and test errors agree to within 0.002 (no overfitting of the stochastic classifier), and both are about 0.03–0.05 above $w$'s.

**why:** the VC bound charges for the ability to realize every labeling with 785 free parameters. PAC-Bayes charges $\|w - \mu_P\|^2/(2\sigma^2)$, the distance the data moved the weights from the prior in units of the weight noise the classifier can tolerate. that tolerance is a margin statement: the Gibbs risk is small only for examples whose normalized margin $y\,w^\top x/\|x\|$ is large compared with $\sigma$. for MNIST odd-vs-even most examples are classified with a comfortable normalized margin, so $\sigma \approx 0.15$ costs little Gibbs risk and buys a KL of a few hundred to a thousand nats, about 0.005–0.02 per example. the kl-inversion turns that into the final bound (experiment 2 shows how much it helps at these error rates). the data-dependent prior cuts the KL by a factor of 4.7, because $\|w - w_0\| = 3.4$ while $\|w\| = 9.4$: most of where the weights end up is predictable from half the data. that more than pays for computing the bound on 30,000 examples instead of 60,000. to say something about $w$ itself, use $L(w) \le 2L(Q)$, giving 0.36: valid but loose; the gap between the Gibbs classifier and $w$ (0.13 vs 0.10 on test) is the main remaining slack, together with the gap between the bound and the Gibbs test error (0.18 vs 0.13). the test-set bound is still the tightest number in the table, because it uses 10,000 fresh examples and bounds one fixed classifier.

## experiment 2: why invert the kl?

**what people usually predict:** that the kl form and the square-root form differ by a constant factor, or only at large empirical risk.

**what happens:** with $c = 0.01$:

| $\hat L$ | kl inverse | McAllester | Bernstein-like |
|---|---|---|---|
| 0.0 | **0.0100** | 0.0707 | 0.0200 |
| 0.01 | 0.0311 | 0.0807 | 0.0441 |
| 0.05 | 0.0869 | 0.1207 | 0.1016 |
| 0.2 | 0.2603 | 0.2707 | 0.2832 |
| 0.5 | 0.5704 | 0.5707 | 0.6200 |

at zero empirical risk the kl bound's slack is $c = 0.01$, McAllester's is $\sqrt{c/2} = 0.071$, seven times larger. the gap shrinks as $\hat L$ grows and is gone at $\hat L = 0.5$ (0.5704 vs 0.5707). the kl inverse is the tightest of the three everywhere.

**why:** the slack in a confidence bound is set by the standard deviation of the empirical risk, $\sqrt{L(1 - L)/n}$. near $L = 0$ that's tiny, so the true risk can't be much above a zero empirical risk; the kl divergence captures this, giving slack $\approx c$, while Pinsker's inequality throws the variance information away and pays the worst case $\sqrt{c/2}$, which is the right answer only near $\frac12$. the "Bernstein-like" form $\hat L + \sqrt{2\hat Lc} + 2c$ is a common closed-form approximation that keeps the variance dependence; it's within a factor 2 of the kl inverse at small risk. experiment 1's Gibbs risks (0.13–0.15) with $c \approx 0.01–0.02$ are in between: there the kl inversion gives bounds about 0.02 tighter than McAllester's form.

## experiment 3: stability, measured

**what people usually predict:** that 1-NN, a method with no parameters to estimate, becomes more stable with more data like everything else, and that ridge's stability improves like $1/\sqrt n$.

**what happens:** largest change in prediction on 500 held-out districts:

| n | ridge, λ = 0.001 | ridge, λ = 0.1 | 1-NN (random removals) | 1-NN (removing neighbours) |
|---|---|---|---|---|
| 100 | 0.53 | 0.15 | 3.09 | 3.09 |
| 1,000 | 0.15 | 0.033 | 1.06 | 2.27 |
| 3,000 | 0.025 | 0.010 | 0.30 | 1.81 |
| 10,000 | 0.0017 | 0.0016 | **0.00** | **2.02** |

ridge's change falls like $1/n$ (log-log slopes −1.22 for λ = 0.001 and −1.04 for λ = 0.1), and the stronger regularization is more stable at every $n$ until both reach the same small value at 10,000. 1-NN with random removals *appears* to become perfectly stable: at $n = 10{,}000$ none of the 20 removed districts was the nearest neighbour of any of the 500 evaluation districts, so nothing changed. removing the districts that *are* nearest neighbours of evaluation points shows the truth: the change stays around 2 standard deviations of house value at every $n$ (slope −0.11).

**why:** ridge minimizes a $\lambda$-strongly convex objective (per example), and removing one of $n$ points perturbs its gradient by $O(1/n)$, so the minimizer moves by $O(1/(\lambda n))$ (on paper 4); with $\beta \propto 1/n$ the Bousquet–Elisseeff bound decays like $1/\sqrt n$. 1-NN's prediction at a query is one training label: remove the nearest neighbour and the prediction jumps to the next neighbour's label, a change of the order of the label noise, however large $n$ is. its uniform stability is $O(1)$, and the stability bound says nothing. that doesn't mean 1-NN fails to generalize (its test error is fine, and it's consistent in the limit as $k$ grows); it means stability, like VC and Rademacher bounds, can't explain it. and the random-removal column is the warning: uniform stability is a supremum over every removal, and averaging over random removals can make an unstable algorithm look stable. measuring a supremum requires looking where it's attained.
