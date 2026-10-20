# VC dimension

## why this matters

topic 01 ended with a paradox. thresholds on the line are an infinite class, yet they need *fewer* examples than a finite grid of a thousand thresholds. and a float32 linear model is a finite class with $2^{25120}$ members, whose finite-class bound asks for 348,298 examples, while logistic regression is perfectly happy with far fewer. counting hypotheses is the wrong measure of complexity.

the right measure counts something else: not how many hypotheses there are, but how many different ways they can **label a sample**. two thresholds between the same pair of data points are the same hypothesis as far as the data can tell. if a class can only produce a few labelings of $m$ points, then $m$ examples pin it down, however many members it has. if it can produce *every* labeling, then fitting the data tells you nothing, because it would have fitted any labels.

that idea, due to Vapnik and Chervonenkis (1971), gives the **VC dimension**: the size of the largest set of points a class can label in every possible way. it's the quantity that decides whether a class is learnable at all, and it replaces $\ln|\mathcal H|$ in the bounds. this topic computes it the honest way, by brute force: you'll write the oracles that decide whether a class can produce a labeling, enumerate labelings, and search for the largest shattered set, and then meet the lemma that makes the whole theory work.

## warm-up: predict first

1. thresholds $\mathbb 1[x \ge t]$ can label two points $x_1 < x_2$ in how many of the 4 possible ways?
2. axis-aligned rectangles in the plane have 4 parameters. can they produce all 16 labelings of the four corners of a square?
3. lines in the plane (halfspaces) have 3 parameters. what's the largest number of points they can label in every possible way?
4. a class has one real parameter. what's the most points it could possibly shatter?
5. a class can't shatter any set of 3 points. on 100 points, roughly how many of the $2^{100}$ labelings can it produce: almost all of them, about half, or a vanishing fraction?

## prerequisites check

- [ ] you know the PAC framework, the realizable and agnostic cases, and the finite-class bound (→ topic 01)
- [ ] you know binomial coefficients and $\sum_{i}\binom mi = 2^m$ (→ ch 01)
- [ ] you can state a linear program and solve one with `scipy.optimize.linprog` (→ ch 06 · topic 06)
- [ ] you know what linear separability means for a perceptron (→ ch 08 · topic 04)

## core ideas

### idea 1: shattering

fix a finite set of points $C = \{x_1, \dots, x_m\}$. each hypothesis $h \in \mathcal H$ induces a **labeling** $(h(x_1), \dots, h(x_m)) \in \{0, 1\}^m$, and the **restriction** $\mathcal H_C$ is the set of distinct labelings the class produces on $C$. there are at most $2^m$. the class **shatters** $C$ if it produces all of them: $|\mathcal H_C| = 2^m$. the **VC dimension** $\text{VCdim}(\mathcal H)$ is the size of the largest set the class shatters (infinite if there's no largest).

the definition has an asymmetry that trips everyone up. to show $\text{VCdim} \ge d$ you need **one** set of $d$ points that is shattered (you get to choose the points). to show $\text{VCdim} < d + 1$ you need to show that **no** set of $d + 1$ points is shattered (an argument about every configuration).

to compute with a class you need an **oracle**: given points $X$ and labels $y$, can some $h \in \mathcal H$ produce exactly $y$ on $X$? for simple classes the oracle is a one-liner, because there's a canonical candidate:

- **thresholds** $\mathbb 1[x \ge t]$: every positive must lie to the right of every negative. two points: 3 labelings of 4 ($y = (1, 0)$ with $x_1 < x_2$ is impossible, warm-up 1). VC dimension 1.
- **intervals** $\mathbb 1[a \le x \le b]$: the smallest interval containing the positives is $[\min_{y_i = 1}x_i,\ \max_{y_i = 1}x_i]$, and any interval containing the positives contains this one. so $y$ is realizable iff no negative falls inside it (the all-negative labeling uses an empty interval). two points are shattered; three are not, since $(1, 0, 1)$ fails. VC dimension 2.
- **axis-aligned boxes** in $\mathbb R^d$, $\mathbb 1[a \le x \le b]$ coordinatewise: the same argument with the bounding box of the positives. in the plane the four points of a diamond are shattered; the four corners of a square are not (two diagonal corners force the box to contain the others, warm-up 2). no five points are shattered: take the leftmost, rightmost, top and bottom points (at most four); the fifth lies in their bounding box, so "the four extremes positive, the fifth negative" fails. VC dimension 4.

> **spec.** `interval_realizable(x, y)`: points `x` (n,) and labels `y` (n,) in {0, 1}; `True` iff no negative lies in $[\min, \max]$ of the positives (always `True` with no positives). `rectangle_realizable(X, y)`: points `X` (n, d) (any $d$); `True` iff no negative lies in the bounding box of the positives (boundaries included). return Python bools.

> **▶ build: stage 1.** implement `interval_realizable` and `rectangle_realizable`.

### idea 2: halfspaces, by linear programming

a **halfspace** classifier is $h_{w, b}(x) = \mathbb 1[w^\top x + b \ge 0]$, with $w \in \mathbb R^d$, $b \in \mathbb R$: a line in the plane, a plane in 3-D. there's no canonical candidate now, but deciding whether a labeling is realizable is a **linear program**. with signs $s_i = +1$ for positives and $-1$ for negatives, $y$ is realizable iff the inequalities

$$s_i\,(w^\top x_i + b) \ge 1, \qquad i = 1, \dots, n$$

have a solution $(w, b)$. why 1 and not "$> 0$"? on a finite set, any strict separator can be rescaled until its smallest margin is 1, and any separator with "$\ge 0$" for positives and "$< 0$" for negatives can be shifted to a strict one (on paper 2 asks you to check). so feasibility of this LP, with no objective, is exactly linear separability. it's also exactly what a perceptron needs to converge (chapter 08), but an LP gives a definite *no*, while a perceptron that hasn't converged yet might just need more epochs.

halfspaces in $\mathbb R^d$ have VC dimension $d + 1$. lower bound: the origin and the $d$ unit vectors are shattered (pick $w_j = \pm1$ to put each unit vector on the side you want, and $b$ for the origin). upper bound: **Radon's theorem** says any $d + 2$ points can be split into two groups whose convex hulls intersect; label one group positive and the other negative, and no halfspace can realize it, because a halfspace is convex and would contain a point of the other group's hull. in the plane ($d = 2$): any 4 points are either in convex position (label the diagonals alternately, the XOR pattern) or one lies in the triangle of the others (label it alone). VC dimension 3 (warm-up 3), equal to the number of parameters. that coincidence is the source of a persistent myth, which the lab demolishes.

> **spec.** `halfspace_realizable(X, y)`: points `X` (n, d), labels `y` (n,). return `True` if all labels are equal (no LP needed); otherwise solve the feasibility LP above with `scipy.optimize.linprog` (zero objective, `A_ub` $= -s_i[x_i, 1]$, `b_ub` $= -1$, **`bounds=(None, None)`**, because the default bounds force every variable to be non-negative) and return whether it found a feasible point (`res.status == 0`).

> **▶ build: stage 2.** implement `halfspace_realizable`.

### idea 3: counting labelings by brute force

with an oracle, $|\mathcal H_C|$ is a finite computation: run through all $2^m$ labelings of $C$ and ask the oracle about each. `itertools.product((0, 1), repeat=m)` generates them. checking whether $C$ is shattered is the same loop with an early exit at the first labeling the class can't produce.

this is exponential, and you should feel it. $m = 10$ is 1,024 oracle calls; $m = 20$ is a million; $m = 30$ is a billion. with an LP per call (around half a millisecond), $m = 20$ already takes several minutes. so brute-force code needs a **guard**: refuse inputs that would run forever instead of hanging. in this topic the guard is a `max_points` argument (16 by default, 65,536 labelings) that raises `ValueError` *before* any work is done, and the caller can raise it deliberately.

counting also gives you a sharper picture than shattering alone. intervals on $m$ distinct points produce exactly $1 + m + \binom m2$ labelings (empty, or a contiguous block determined by its two ends): 16 for $m = 5$, out of 32. halfspaces in the plane on $m$ points in general position produce exactly $m^2 - m + 2$ (Cover, 1965): 22 for $m = 5$.

> **spec.** `count_labelings(X, realizable, *, max_points=16)` returns the number of labelings `y` (int arrays of shape (n,)) of the `n = len(X)` points for which `realizable(X, y)` is true. `is_shattered(X, realizable, *, max_points=16)` returns whether all $2^n$ are, stopping at the first failure. both raise `ValueError` if `n > max_points`, before calling the oracle.

> **🔮 predict:** how many of the 64 labelings of 6 points in general position in the plane can a line produce?

> **▶ build: stage 3.** implement `count_labelings` and `is_shattered`.

### idea 4: searching for the VC dimension

given a pool of candidate points, you can search for the largest shattered subset: for $d = 1, 2, \dots$, try every $d$-subset of the pool until one is shattered. two facts make this search finite and honest.

**monotonicity.** every subset of a shattered set is shattered (restrict the labelings). so if *no* $d$-subset of the pool is shattered, no larger subset is either, and the search can stop. this is the only reason a brute-force search terminates before trying every subset of every size.

**it's a lower bound.** the search finds the largest shattered subset *of the pool*. the VC dimension is a supremum over all point sets in the domain, which no finite search can cover; if the pool happens to contain no shattered configuration of the right shape (four points in convex "diamond" position for boxes, say), the search reports less than the truth. the lab's random pools usually contain a witness, and the upper bounds come from arguments like Radon's theorem, not from code.

the cost is $\sum_d \binom{N}{d}2^d$ oracle calls in the worst case, for a pool of $N$ points: with $N = 12$ and $d$ up to 5 that's up to 35,000 calls, fine for intervals and boxes (microseconds each), noticeable for LPs. hence `max_d`.

> **spec.** `vc_dimension_search(pool, realizable, *, max_d=5)`: `pool` is (N,) or (N, d). for `d = 1, ..., min(max_d, N)`, go through `itertools.combinations(range(N), d)` and stop at the first subset that `is_shattered`; if none is, stop the search. return `(best_d, witness)`, the largest $d$ found and the tuple of pool indices of its shattered subset (`(0, ())` if not even one point is shattered).

> **▶ build: stage 4.** implement `vc_dimension_search`.

### idea 5: the growth function and Sauer's lemma

the **growth function** $\Pi_{\mathcal H}(m) = \max_{|C| = m}|\mathcal H_C|$ is the largest number of labelings the class produces on any $m$ points. if the VC dimension is $d$, then $\Pi_{\mathcal H}(m) = 2^m$ for $m \le d$, and the remarkable fact is what happens after:

$$\textbf{Sauer–Shelah lemma:}\qquad \Pi_{\mathcal H}(m) \le \sum_{i=0}^{d}\binom mi \le \Big(\frac{em}{d}\Big)^d \quad (m \ge d).$$

once $m$ passes $d$, the growth function stops being exponential and becomes a **polynomial** of degree $d$ (warm-up 5: a class that can't shatter 3 points produces at most $1 + 100 + 4950 = 5051$ of the $2^{100}$ labelings of 100 points). there's no middle ground: a class either shatters arbitrarily large sets or grows polynomially.

the polynomial is what learning needs. the finite-class argument of topic 01 used a union bound over $|\mathcal H|$ hypotheses; the VC argument uses a union bound over the $\Pi_{\mathcal H}(2m)$ labelings of a double sample (a "ghost sample" trick, Vapnik & Chervonenkis 1971). one standard form is

$$P\Big(\sup_{h \in \mathcal H}|L_{\mathcal D}(h) - L_S(h)| > \varepsilon\Big) \le 4\,\Pi_{\mathcal H}(2m)\,e^{-m\varepsilon^2/8},$$

so $\ln|\mathcal H|$ is replaced by $\ln\Pi_{\mathcal H}(2m) \le d\ln(2em/d)$. setting the right side to $\delta$ gives the explicit bound you'll compute in topic 04. this leads to the **fundamental theorem of statistical learning** (Shalev-Shwartz & Ben-David 2014, Theorems 6.7–6.8): for binary classification with the 0-1 loss, a class is PAC learnable (realizable or agnostic) if and only if its VC dimension is finite, and then, for absolute constants $C_1, C_2$,

$$C_1\frac{d + \ln(1/\delta)}{\varepsilon^2} \le m_{\mathcal H}^{\text{agnostic}}(\varepsilon, \delta) \le C_2\frac{d + \ln(1/\delta)}{\varepsilon^2}, \qquad C_1\frac{d + \ln(1/\delta)}{\varepsilon} \le m_{\mathcal H}^{\text{realizable}} \le C_2\frac{d\ln(1/\varepsilon) + \ln(1/\delta)}{\varepsilon}.$$

the VC dimension plays the role $\ln|\mathcal H|$ played for finite classes, and ERM achieves the upper bounds. the theorem doesn't name the constants; topic 04 uses a version that does.

> **spec.** `sauer_bound(m, d)` returns $\sum_{i=0}^{\min(d, m)}\binom mi$ as an exact Python int (`math.comb`). `growth_lower_bound(pool, realizable, m, *, n_subsets=20, rng=None)`: with `gen = as_rng(rng)`, draw `n_subsets` subsets of `m` distinct pool indices (`gen.choice(N, size=m, replace=False)`) and return the largest `count_labelings` among them, a lower bound on $\Pi_{\mathcal H}(m)$.

> **🔮 predict:** Sauer's bound for $d = 3$ at $m = 10$ is 176. halfspaces in the plane have VC dimension 3. will they reach 176 labelings of 10 points?

> **▶ build: stage 5.** implement `sauer_bound` and `growth_lower_bound`.

## worked example

**counting by hand.** intervals on three points $0.1 < 0.5 < 0.9$: the realizable labelings are $000$, the three singletons $100, 010, 001$, and the two adjacent pairs $110, 011$, plus $111$: seven, which is $1 + 3 + \binom32$. the eighth, $101$, fails because the bounding interval $[0.1, 0.9]$ contains the negative 0.5. `count_labelings` must return 7 and `is_shattered` must return `False`; on the first two points, both 4 labelings work, so they're shattered.

**boxes.** for the diamond $(0, 1), (1, 0), (0, -1), (-1, 0)$ and labels "top and bottom positive", the bounding box of the positives is the segment $x = 0$, $-1 \le y \le 1$, which contains neither side point: realizable. for the square corners $(0,0), (1,0), (0,1), (1,1)$ with the diagonal $(0,0), (1,1)$ positive, the bounding box is the whole square: not realizable.

**Sauer against reality.** $m = 5$, $d = 3$: $\binom50 + \binom51 + \binom52 + \binom53 = 1 + 5 + 10 + 10 = 26$. halfspaces in the plane realize $5^2 - 5 + 2 = 22$ labelings of five points in general position; intervals ($d = 2$) realize $1 + 5 + 10 = 16 = $ `sauer_bound(5, 2)`, meeting the bound exactly.

## on paper

1. **boxes can't shatter 5 points.** complete the argument from idea 1: for any 5 points in the plane, exhibit a labeling no axis-aligned box produces. (careful when several points share an extreme coordinate.)

<details><summary>solution</summary>

pick one point attaining the minimum $x$, one attaining the maximum $x$, one the minimum $y$ and one the maximum $y$ (ties broken arbitrarily). that's a set $E$ of at most 4 points, so some fifth point $p \notin E$ exists. label $E$ positive and everything else negative. any box containing $E$ contains $[\min x, \max x] \times [\min y, \max y]$, the bounding box of *all* five points, which contains $p$. so this labeling is unrealizable.

</details>

2. **why 1 in the LP?** show that a labeling is realizable by $\mathbb 1[w^\top x + b \ge 0]$ on a finite set iff the system $s_i(w^\top x_i + b) \ge 1$ is feasible.

<details><summary>solution</summary>

if the system is feasible, positives have $w^\top x_i + b \ge 1 \ge 0$ and negatives $\le -1 < 0$: realized. conversely, suppose positives have $w^\top x_i + b \ge 0$ and negatives $w^\top x_i + b < 0$. let $\gamma = \min_{\text{neg}}|w^\top x_i + b| > 0$ (finitely many negatives). shifting $b \to b + \gamma/2$ gives every positive value $\ge \gamma/2$ and every negative $\le -\gamma/2$; dividing $(w, b)$ by $\gamma/2$ gives margins $\ge 1$. (with no negatives, take $w = 0$, $b = 1$.)

</details>

3. **Radon in the plane.** show that lines can't shatter any 4 points in the plane.

<details><summary>solution</summary>

if three of the points are collinear, label the middle one differently from the outer two; a halfspace is convex, so if it contains both outer points it contains the middle. otherwise, either the convex hull is a triangle with the fourth point inside (label the inside point alone; any halfspace containing the three corners contains the triangle), or the four points are in convex position, a quadrilateral whose diagonals cross (label one diagonal positive, the other negative; both halfspaces would contain the crossing point, a contradiction).

</details>

4. **from the growth function to a sample size.** using $P(\sup_h|L - L_S| > \varepsilon) \le 4\Pi(2m)e^{-m\varepsilon^2/8}$ and Sauer's lemma, show that with probability $\ge 1 - \delta$ every $h$ satisfies $L(h) \le L_S(h) + \sqrt{\frac8m\big(d\ln\frac{2em}{d} + \ln\frac4\delta\big)}$.

<details><summary>solution</summary>

set $4\Pi(2m)e^{-m\varepsilon^2/8} = \delta$: $\varepsilon^2 = \frac8m(\ln\Pi(2m) + \ln\frac4\delta)$. by Sauer, $\ln\Pi(2m) \le d\ln(2em/d)$ for $2m \ge d$, which only makes $\varepsilon$ larger. with probability $\ge 1 - \delta$ the deviation is at most $\varepsilon$ for every $h$ simultaneously, in particular $L(h) \le L_S(h) + \varepsilon$. this is the `vc_bound` you'll implement in topic 04.

</details>

## experiment

`lab.py` investigates, with your code:

1. the VC dimensions of intervals, boxes and halfspaces in the plane, found by brute-force search on random pools, with the oracle calls each search costs.
2. the measured growth function of halfspaces and intervals against Sauer's lemma and $2^m$.
3. how many points a class with a single real parameter can shatter.

## check your understanding

1. why can't you prove $\text{VCdim}(\mathcal H) \ge d$ by showing that *some* labelings of many $d$-sets are realizable?

<details><summary>answer</summary>

shattering is about one set and *all* of its labelings. many sets each missing a different labeling prove nothing; you need a single set on which every one of the $2^d$ labelings is realized.

</details>

2. your search on a pool of random points reports 3 for a class. what can you conclude?

<details><summary>answer</summary>

$\text{VCdim} \ge 3$ (you have a witness). you can't conclude it's exactly 3: the pool may lack a shattered 4-set of the right shape. an upper bound needs an argument about every configuration.

</details>

3. is the VC dimension always the number of parameters?

<details><summary>answer</summary>

no. it coincides for halfspaces, and it's often in that ballpark for "linear-ish" models, but $\mathbb 1[\sin(\omega x) > 0]$ has one parameter and infinite VC dimension (lab experiment 3), while a class with many redundant parameters can have a small one. it measures expressiveness on data, not parameter count.

</details>

4. why does monotonicity (subsets of shattered sets are shattered) matter for the search?

<details><summary>answer</summary>

it lets the search stop at the first size $d$ with no shattered subset; without it you'd have to try every size up to $N$, which is hopeless. it also shows the set of shattered sizes is an interval $\{0, 1, \dots, \text{VCdim}\}$.

</details>

5. Sauer's lemma bounds the number of labelings by a polynomial in $m$. why is that the key to learnability?

<details><summary>answer</summary>

the uniform-convergence argument multiplies a per-labeling failure probability that decays exponentially in $m$ by the number of labelings. a polynomial times $e^{-cm\varepsilon^2}$ goes to 0; $2^m$ times it doesn't (for small $\varepsilon$). so finite VC dimension makes the union bound work, and infinite VC dimension breaks it, which the fundamental theorem shows is unavoidable.

</details>

6. VC dimension is "worst case over points, best case over the class". what does each half mean, and why might it be pessimistic for your data?

<details><summary>answer</summary>

$\text{VCdim}$ asks for the *best* configuration of points (the one that's easiest to shatter) and whether the class realizes *every* labeling. real data may never take such configurations and real labels aren't adversarial, so data-dependent quantities (the growth on your sample, Rademacher complexity in topic 03, margins in topic 06) can be much smaller.

</details>

## common bugs

- **`linprog` with default bounds.** symptom: labelings that need a negative weight or bias come back infeasible; simple separable sets fail. pass `bounds=(None, None)`.
- **forgetting the all-negative labeling.** symptom: counts one short of the closed forms; nothing is ever shattered. with no positives, the empty interval or box works.
- **strict inequalities at the box boundary.** symptom: points on the edge of the positives' bounding box counted as outside. boxes are closed.
- **the guard after the loop.** symptom: `count_labelings` on 25 points runs for minutes before complaining. check `n > max_points` first.
- **searching every size.** symptom: searches that take minutes. stop as soon as some size has no shattered subset.
- **treating the search result as the VC dimension.** symptom: wrong conclusions from an unlucky pool. it's a lower bound.

## going deeper

- Vapnik & Chervonenkis (1971), "On the uniform convergence of relative frequencies of events to their probabilities", *Theory of Probability and Its Applications* 16(2). the original. [doi:10.1137/1116025](https://doi.org/10.1137/1116025)
- Sauer (1972), "On the density of families of sets", *Journal of Combinatorial Theory A* 13(1). [doi:10.1016/0097-3165(72)90019-2](https://doi.org/10.1016/0097-3165(72)90019-2)
- Cover (1965), "Geometrical and statistical properties of systems of linear inequalities with applications in pattern recognition", *IEEE Transactions on Electronic Computers* EC-14(3). the function counting theorem. [doi:10.1109/PGEC.1965.264137](https://doi.org/10.1109/PGEC.1965.264137)
- Blumer, Ehrenfeucht, Haussler & Warmuth (1989), "Learnability and the Vapnik–Chervonenkis dimension", *Journal of the ACM* 36(4). [doi:10.1145/76359.76371](https://doi.org/10.1145/76359.76371)
- Shalev-Shwartz & Ben-David (2014), *Understanding Machine Learning*, ch. 6 (VC dimension, Sauer's lemma, the fundamental theorem) and ch. 28 (proofs). [free PDF](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/)
- Mohri, Rostamizadeh & Talwalkar (2018), *Foundations of Machine Learning*, 2nd ed., ch. 3.3–3.4 (growth function, VC dimension, Radon's theorem). [cs.nyu.edu/~mohri/mlbook](https://cs.nyu.edu/~mohri/mlbook/)

## connections

- ← builds on: the PAC framework and finite-class bounds (topic 01), linear programming (ch 06 · topic 06), linear separability and the perceptron (ch 08 · topic 04).
- → used in: the VC generalization bound you'll compute for real models (topic 04), Rademacher complexity, whose finite-class version uses the same restriction $\mathcal H_C$ (topic 03), margins, which explain why high-dimensional linear classifiers generalize anyway (topic 06), and the "why do overparameterized networks generalize?" question of deep learning theory (ch 19).
