# vector spaces · lab notes

## experiment 1: the angle between two random vectors

**what people usually predict:** angles spread out uniformly (anything from 0° to 180° is equally likely), or clustered around 45°, or "random vectors could point anywhere".

**what happens:** in 2-D the angles really are spread out. as $d$ grows, the histogram collapses onto **90°**. the standard deviation of $\cos\theta$ matches $1/\sqrt d$ almost exactly: 0.32 at $d = 10$, 0.032 at $d = 1000$. at $d = 1000$, about 99.7% of pairs have $|\cos\theta| < 0.1$ (within about 6° of perpendicular), and the most parallel pair out of 2,000 is still about 83° apart. at $d = 10{,}000$ every pair sits within a couple of degrees of 90°.

**why:** $u^\top v = \sum_{i=1}^d u_iv_i$ is a sum of $d$ independent terms with mean 0 and variance 1, so it's typically of size $\sqrt d$ (the central limit theorem), while $\|u\|\,\|v\| \approx \sqrt d\cdot\sqrt d = d$. their ratio is about $1/\sqrt d$. in high dimensions there are so many directions that two random ones almost never line up. consequences you'll meet later: random projections approximately preserve distances (Johnson–Lindenstrauss, used for random features in chapter 09), and a $d$-dimensional space holds exponentially many *nearly* orthogonal directions, which is one reason networks can store more features than they have neurons (superposition, chapter 28). the flip side is the curse of dimensionality: distances concentrate, which hurts nearest-neighbour methods.

## experiment 2: what is the rank of a noisy matrix?

**what people usually predict:** rank 3 (the noise is tiny), or rank 20 (the noise makes it full rank), with confidence either way, and with no dependence on tol.

**what happens:** every curve is a staircase. for noise $10^{-8}$, `rank` returns 20 for tol $\lesssim 10^{-8}$ and drops to 3 for tol $\gtrsim 10^{-6}$, with a brief 4 or 5 in between. the step moves with the noise level, sitting just above it: noise $10^{-11}$ steps down near $10^{-10}$, noise $10^{-2}$ near $10^{-1}$. even the **noise-free** matrix reports rank 20 when tol is $10^{-16}$: rounding errors during elimination are themselves noise at the $10^{-15}$ level.

**why:** after eliminating the three "real" directions, the leftover entries aren't zero. they're the noise, transformed, at about the noise's size. a tolerance above that level calls them zero (rank 3). below it, every leftover direction becomes a pivot (rank 20). so the rank of a measured matrix isn't a property of the matrix alone. it's a property of the matrix *and of how much noise you're willing to call zero*. the useful information is the gap between the "signal" scale (order 1 here) and the noise scale. topic 06 shows that gap directly as a cliff in the singular values, which is how PCA decides how many components are real.

## experiment 3: when are random vectors a basis?

**what people usually predict:** almost always a basis, like Gaussian vectors. maybe occasionally not for tiny $d$, and less and less often as $d$ grows.

**what happens:** Gaussian vectors are a basis every time. coin-flip vectors fail **about half the time at $d = 2$**, and the failure rate *rises* to roughly 65% at $d = 3$–$5$, before falling slowly: still about 50% at $d = 8$ and about 40% at $d = 10$.

**why:** at $d = 2$, the second vector is dependent on the first exactly when it equals $\pm$ the first: 2 of the 4 equally likely possibilities, so probability 1/2. a discrete distribution puts positive probability on exact coincidences: two equal rows or columns, a row equal to the sum or difference of others, and so on. as $d$ grows there are more ways to coincide, which is why the rate first goes up. eventually independence wins, but slowly. Tikhomirov (2020) proved the singular probability is $(1/2 + o(1))^d$, so it does go to zero, and the convergence is slow at small $d$: two particular rows (or columns) are equal or opposite with probability $2^{1-d}$, and there are $d(d-1)$ such row-or-column pairs, so that one failure mode alone contributes roughly $d(d-1)\,2^{1-d}$ (about 0.4 at $d = 8$, 0.2 at $d = 10$). sums and differences of three or more rows account for the rest of what you measured. only for large $d$ does the probability approach its $(1/2)^d$ asymptotics. continuous distributions put zero probability on any exact linear relation, which is why "random matrices are invertible" is safe for Gaussian initializations but not for binary or low-precision data. the lesson: **independence depends on the distribution, not just the count**. quantized weights and one-hot features (chapter 07) produce exactly-dependent columns surprisingly often.
