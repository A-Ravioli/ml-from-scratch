# privacy & secure aggregation · lab notes

numbers are from the reference solution on the cached MNIST data, one CPU thread (the lab takes 70–115 s depending on machine load). experiment 3 uses two seeds per configuration; per-seed values are in the table, and differences under about 0.01 are within seed noise.

## experiment 1: secure aggregation at scale

**what people usually predict:** the secure sum is "approximately" right, with an error that grows with the number of clients and the size of the random masks (people picture adding and subtracting numbers around $10^9$ in floating point). most guess that key overhead matters only for thousands of clients.

**what happens:**

| clients | dropped | largest error vs plain sum | rounding bound $n\cdot0.5/2^{16}$ | seconds |
|---|---|---|---|---|
| 5 | 1 | 2.7e-05 | 3.1e-05 | 0.00 |
| 20 | 4 | 6.3e-05 | 1.2e-04 | 0.02 |
| 50 | 10 | 1.2e-04 | 3.1e-04 | 0.12 |

the error never exceeds the fixed-point rounding bound, with or without dropouts: the masks themselves contribute exactly nothing. the masked upload's histogram is flat on $[0, 2^{32})$, while the raw encoding sits in two spikes (small positive values near 0, small negative ones wrapped to just below $2^{32}$).

upload overhead per client (keys and shares relative to the masked update):

| parameters | n = 10 | n = 100 | n = 1,000 | n = 10,000 |
|---|---|---|---|---|
| 6,634 | 0.3% | 3.0% | 30.1% | 301.5% |
| 50,890 | 0.0% | 0.4% | 3.9% | 39.3% |
| 1,000,000 | 0.0% | 0.0% | 0.2% | 2.0% |

**why:** everything happens in $\mathbb Z_{2^{32}}$, where addition is exact and each pairwise mask enters once with each sign, so the masks cancel bit for bit; the only error is the rounding in `encode`, at most $0.5/2^{16}$ per client. a uniform mask added mod $M$ makes the message exactly uniform whatever the update (a one-time pad), which is why the masked histogram carries no trace of the update. the overhead is $8(n + |D|)$ bytes per client, so linear per client and quadratic per round: for the project's small model, a cohort of 1,000 adds 30% to every upload, and 10,000 quadruples it. Bell et al. (2020) fix this by masking with $O(\log n)$ random neighbours instead of everyone. (the simulation's runtime also grows like $n^2$ because every pair expands a mask; a real deployment pays this on the clients in parallel.)

## experiment 2: what does the accountant buy?

**what people usually predict:** with 1% sampling, 1,000 rounds cost about as much as 10 full rounds, so ε is "about 10 rounds' worth", maybe 10–20. people rarely expect three orders of magnitude between methods.

**what happens:** (σ = 1, q = 0.01, δ = 1e-5)

| rounds | your accountant (best order) | no amplification (q = 1) | (ε, δ) basic composition |
|---|---|---|---|
| 1 | 1.32 (10) | 5.3 | 1.3 |
| 10 | 1.46 (9) | 20.8 | 15.7 |
| 100 | 1.62 (9) | 111.5 | 182.9 |
| 1,000 | 2.54 (8) | 1,011.5 | 2,085.0 |
| 10,000 | 7.47 (4) | 10,011.5 | 23,408.6 |
| 100,000 | 28.69 (2) | 100,011.5 | 259,669.8 |

your accountant reports ε = 2.5 after 1,000 rounds; ignoring sampling gives over 1,000, and adding up per-round ε's gives over 2,000. the accountant's curve has two regimes: a plateau (1.3 → 1.6 over the first 100 rounds), then growth of about 3–4× per decade of rounds from 10³ to 10⁵.

**why:** two effects multiply. sampling amplification is roughly *quadratic* in $q$ at the RDP level (for moderate orders the per-round RDP is close to $q^2\alpha/(2\sigma^2)$, not $q\alpha/(2\sigma^2)$), so 1% sampling cuts the per-round cost by orders of magnitude, far more than the 100× fewer uses suggests. the plateau comes from the *cliff* in $\rho(\alpha)$: amplification only works up to a critical order (here $\rho(10) \approx 0.04$ but $\rho(12) \approx 1$ and $\rho(15) \approx 2.6$, where the term with all $\alpha$ draws including the user, weighted by $e^{\alpha^2/(2\sigma^2)}$, takes over). the conversion term $\log(1/\delta)/(\alpha - 1)$ wants the largest usable order, so for small $T$ the best order sits just below the cliff (9–10), $T\rho$ is negligible, and ε ≈ $\log(1/\delta)/(\alpha^\star - 1) \approx 1.3$, set by δ and the cliff rather than by $T$. once $T\rho$ matters the best order slides down (8, 4, 2), and in the middle range ε grows roughly like $\sqrt T$ (check question 6); when the best order reaches 2, the smallest tracked, growth turns linear. converting every round and adding pays the conversion $T$ times and is linear from the start. both crude methods are valid upper bounds, just useless ones. a tighter accountant (better conversion, fractional orders, or a numerical privacy-loss-distribution accountant) would lower the blue curve somewhat and smooth the plateau; it would not change the conclusion that sampling plus RDP composition is what makes hundreds of private rounds affordable. the (ε, δ)-composition curve matches the accountant at $T = 1$ and starts *below* the no-amplification one because it does use the amplified single-round ε; it loses because of how it composes.

## experiment 3 (project): the privacy–accuracy–communication triangle

**what people usually predict:** privacy is expensive: ε = 1 costs several points of accuracy, ε = 0.1 makes the model useless; and more users per round helps only through "more data".

**what happens:** (1,000 users × 30 images, MLP 196 → 32 → 10, 40 rounds, clip 0.5, δ = 1e-4)

| ε | q (≈ users/round) | σ | accuracy (per seed) | MB uploaded |
|---|---|---|---|---|
| ∞ | 0.025 (25) | 0 | 0.892 (0.890, 0.895) | 26.8 |
| ∞ | 0.1 (100) | 0 | 0.930 (0.934, 0.926) | 109.3 |
| ∞ | 0.4 (400) | 0 | 0.952 (0.957, 0.947) | 480.7 |
| 1 | 0.025 | 1.34 | 0.884 (0.872, 0.896) | 26.8 |
| 1 | 0.1 | 3.15 | 0.930 (0.932, 0.928) | 109.3 |
| 1 | 0.4 | 11.36 | 0.952 (0.953, 0.951) | 480.7 |
| 0.25 | 0.025 | 3.14 | 0.847 (0.837, 0.857) | 26.8 |
| 0.25 | 0.1 | 11.25 | 0.913 (0.922, 0.905) | 109.3 |
| 0.25 | 0.4 | 43.93 | 0.944 (0.948, 0.941) | 480.7 |
| 0.1 | 0.025 | 7.17 | 0.756 (0.770, 0.742) | 26.8 |
| 0.1 | 0.1 | 27.53 | 0.852 (0.854, 0.850) | 109.3 |
| 0.1 | 0.4 | 109.08 | 0.900 (0.893, 0.908) | 480.7 |

reading the table as a triangle:

- **user-level ε = 1 is essentially free** at every cohort size (within seed noise of no noise at all). privacy starts to cost at ε = 0.25 (−4.5 points with 25 users per round, −0.8 with 400) and bites at ε = 0.1 (−13.6 points with 25 users, −5.2 with 400).
- **under DP it's cohort size, not bytes, that buys accuracy.** going from 25 to 400 users per round (18× the upload) gains 6 points without noise, 10 at ε = 0.25 and 14 at ε = 0.1 here, but experiment 3b shows the no-noise gain is entirely "more data": at equal uploads, cohort size doesn't matter without noise and decides everything at ε = 0.1.
- the communication numbers include secure aggregation: at 400 users per round each upload carries 3.2 kB of keys and shares on top of the 26.5 kB update (12%).

**why (the four write-up questions):**

1. **the mechanism.** the noise in the averaged update has standard deviation $\sigma S/(qN)$. at ε = 1, $\sigma/q$ falls from 54 to 28 as $q$ goes from 0.025 to 0.4: amplification weakens as $q$ grows, so $\sigma$ needs to grow less than proportionally, and the noise in the average drops by half. at ε = 0.1, $\sigma/q$ is nearly constant (287, 275, 273): deep in the amplification regime, ε is roughly proportional to $q/\sigma$, so the per-round noise in the average is the same for all cohort sizes. the large cohorts still win there because the *signal* is better: each round averages 400 clipped updates instead of 25, and the model being noised has made more progress per round, so the same absolute noise is a smaller fraction of a larger, cleaner step.
2. **the triangle.** at fixed communication, privacy costs accuracy (a column of the table); at fixed privacy, communication buys accuracy (a block); at fixed accuracy, more privacy costs communication (≈ 0.85 accuracy costs 27 MB with ε = 0.25 but 109 MB with ε = 0.1). if uploads bind, take ε = 1 with small cohorts: nearly no accuracy cost. if privacy binds (ε ≤ 0.25), pay for large cohorts. privacy is nearly free whenever the population is large relative to the noise, which is the regime McMahan et al. (2018) target with millions of users.
3. **the baseline.** without noise, 400 users per round beats 25 by 6 points in experiment 3 only because it processes 16× more client updates in the same 40 rounds. experiment 3b fixes the budget at about 16,000 client updates (and so about 430–480 MB of uploads) per configuration. measured with two seeds (the lab shows seed 0):

   | ε | q | rounds | σ | σ / q | accuracy (seeds 0, 1) | MB |
   |---|---|---|---|---|---|---|
   | ∞ | 0.025 | 640 | 0 | – | 0.948 (0.951, 0.945) | 427 |
   | ∞ | 0.1 | 160 | 0 | – | 0.953 (0.952, 0.954) | 439 |
   | ∞ | 0.4 | 40 | 0 | – | 0.952 (0.957, 0.947) | 474 |
   | 0.1 | 0.025 | 640 | 27.3 | 1092 | 0.145 (0.152, 0.138) | 427 |
   | 0.1 | 0.1 | 160 | 54.6 | 546 | 0.644 (0.716, 0.572) | 439 |
   | 0.1 | 0.4 | 40 | 109.1 | 273 | 0.900 (0.893, 0.908) | 474 |

   without noise the three are indistinguishable: experiment 3's no-noise gap was all data per round. at ε = 0.1 and equal megabytes, cohort size decides everything, from chance level to 90%. the σ / q column says why: more rounds cost privacy (ε grows with $T$), so the small cohort, which needs 640 rounds, must use a σ whose noise in the average ($\propto\sigma/q$) is 4× that of the large cohort. under DP, spend your communication on fewer, bigger rounds.
4. **honesty.** held fixed and worth tuning: the clipping norm (0.5, chosen once; too small biases the update, too large wastes the noise budget), the learning rates (the server step could be larger for large cohorts), and the number of rounds (ε grows with rounds, so fewer, better rounds can be cheaper). the accountant is an upper bound under Poisson sampling, which real client check-ins don't follow; and the hyperparameters were not chosen privately. "user-level ε = 1" promises that the trained model's distribution would be nearly the same (within a factor $e^1$ on every event, plus $\delta = 10^{-4}$) whether or not any one user took part; it does not promise that the model reveals nothing about groups of users or about population-level facts, and with δ = 1e-4 and 1,000 users the δ term is not negligible (δ should be well below $1/N$; here it's 10× below).
