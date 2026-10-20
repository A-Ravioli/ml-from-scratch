# online learning · lab notes

numbers are from the reference solution at the shipped seeds. experiment 3 uses the cached California housing data (20,433 districts); offline it runs on the 1,000-row synthetic stand-in and the numbers differ.

## experiment 1: regret grows like √T

**what people usually predict:** that when every expert is equally good, the regret should be about zero ("there's nothing to regret"), and that the adversarial-looking switching sequence will be the hardest.

**what happens:** regret at $T$ (mean of 5 seeds), $N = 10$:

| T | coin flips | one better expert | switching | guarantee |
|---|---|---|---|---|
| 100 | 8.0 | 7.7 | 7.9 | 10.7 |
| 1,000 | 24.1 | 21.3 | 20.8 | 33.9 |
| 10,000 | 76.8 | 57.4 | 63.7 | 107.3 |
| 30,000 | **140.6** | 96.3 | 121.7 | 185.8 |

every curve is a straight line on the log-log plot with slope close to $\frac12$ (fitted for $T \ge 1000$: **0.52** for coin flips, 0.44 with one better expert, 0.51 for switching), and every one stays below the guarantee $\sqrt{(T/2)\ln N}$. the *pure noise* sequence has the largest regret, at about 76% of the guarantee.

**why:** on coin flips every algorithm, Hedge included, has expected loss exactly $T/2$, but the best of 10 experts *in hindsight* has a cumulative loss about $\sqrt{(T/4)\cdot 2\ln N}$ below $T/2$ by sheer luck (the maximum of 10 nearly independent random walks). the regret measures the comparator's luck, which nobody can anticipate; that's why $\sqrt{T\ln N}$ is a lower bound for every algorithm and why the noise sequence comes so close to the guarantee. with one genuinely better expert, Hedge locks onto it after a few hundred rounds and regret grows more slowly (the comparator is no longer the luckiest of ten equals, and Hedge pays mainly the start-up cost $\ln N/\eta$, which grows like $\sqrt T$ only because $\eta^\star$ shrinks like $1/\sqrt T$). the switching sequence is gentler than it looks: the best expert in hindsight is good only a tenth of the time, so the comparator is weak too.

## experiment 2: the greedy leader

**what people usually predict:** that Hedge, the algorithm with the theorem, beats follow-the-leader on both sequences.

**what happens:** with $T = 10{,}000$:

| sequence | FTL regret at T | Hedge regret at T |
|---|---|---|
| stochastic (means 0.4, 0.6) | **1.0** | 30.0 |
| alternating | **5000.0** | 29.7 |

on iid losses FTL finds the better expert within a few rounds and its regret stays at 1. Hedge's regret climbs to 30 over the first 1,000 rounds and then stays flat. on the alternating sequence FTL loses every round from round 2 on, regret exactly $T/2 = 5000$, while Hedge stays below 30 (its guarantee is 58.9).

**why:** with iid losses and a gap of 0.2, the empirical leader is the true best expert almost immediately, so greedy is optimal; Hedge, tuned for the worst case with $\eta = \sqrt{8\ln 2/T} = 0.024$, keeps hedging until the gap in cumulative loss is several times $1/\eta \approx 42$, which at an expected gap of 0.2 per round takes several hundred rounds (the curve flattens around round 1,000), paying up to 0.1 per round meanwhile: hence the plateau at 30. on the alternating sequence the adversary exploits FTL's determinism: it always knows which expert FTL will play and makes that one lose. Hedge's softmax changes its distribution by only $O(\eta)$ per round, so there's nothing to exploit. the knob between the two is $\eta$: $\eta \to \infty$ is FTL. algorithms that adapt $\eta$ to the data (AdaHedge, "follow the regularized leader" with adaptive regularization) get the best of both: $O(1)$ regret on easy data and $O(\sqrt T)$ on hard data.

## experiment 3: online regression on a real stream

**what people usually predict:** that regret is always non-negative ("you can't beat the best in hindsight"), and that the theory's step size is the right one to use.

**what happens:** (20,433 rounds; the best fixed predictor in the ball $\|w\| \le 2$ has norm 1.55, so the constraint doesn't bind)

| stream | step size | mean loss (online) | mean loss (best fixed) | regret at T |
|---|---|---|---|---|
| shuffled | theory $(D/G)/\sqrt t$ | 0.1817 | 0.1722 | +193.7 |
| shuffled | constant 0.01 | 0.1820 | 0.1722 | +200.5 |
| sorted by latitude | theory $(D/G)/\sqrt t$ | 0.1714 | 0.1722 | −17.0 |
| sorted by latitude | constant 0.01 | **0.1404** | 0.1722 | **−651.4** |

the guarantee $\frac32GD\sqrt T$ is **36,157** ($G = 42.2$, $D = 4$), about 190 times the measured regret on the shuffled stream. on the shuffled stream the regret curve rises like $\sqrt t$ and both step sizes end near +200, an average of 0.01 per round. on the latitude-sorted stream, the constant step size ends at −651: its average loss is 18% *lower* than the best fixed linear model's, fitted to the whole stream in hindsight. the decaying step size ends slightly negative after spending most of the stream above zero.

**why:** the bound uses worst-case constants. $G$ bounds the gradient norm $|w^\top x - y|\,\|x\|$ by assuming the largest feature norm, the largest $|y|$ and a weight vector on the boundary all at once; at the best fixed predictor the gradient norm has median 0.79 and 99th percentile 5.4, against $G = 42.2$, and the bound is linear in $G$. the shuffled stream is effectively iid, so the online learner is doing stochastic gradient descent on least squares and pays the usual $O(\sqrt T)$ for estimating $w$ on the fly. sorted by latitude, the stream drifts: southern and northern California have different relationships between features and prices. the comparator must use one $w$ for the whole state; the online learner with a constant step keeps adapting and tracks a *local* model, which fits each region better. nothing is contradicted: the regret theorem bounds regret from above, and negative regret just means the fixed comparator is a weak yardstick for non-stationary data. the decaying step size $\eta/\sqrt t$ freezes as $t$ grows (it's designed to converge to the best fixed $w$), so it can't track the drift late in the stream, which is why it gains much less. against a comparator allowed to change slowly over time (dynamic regret), beating it is impossible again, and the right step size would stay bounded below.
