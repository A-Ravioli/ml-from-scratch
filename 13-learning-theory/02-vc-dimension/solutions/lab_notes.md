# VC dimension · lab notes

numbers are from the reference solution at the shipped seeds. oracle-call counts depend on the order in which you enumerate labelings and subsets; the VC dimensions and labeling counts don't.

## experiment 1: three VC dimensions, by brute force

**what people usually predict:** that the VC dimension equals the number of parameters (2 for intervals, 4 for boxes, 3 for halfspaces), or, for boxes, that the four corners of a square are the obvious shattered set.

**what happens:** the searches report **2** for intervals (660 oracle calls), **4** for boxes (6,773 calls, 0.05 s) and **3** for halfspaces (2,661 calls, about 1 s, because each call is an LP). the box witness is four points that are each the unique extreme in one direction (leftmost, rightmost, lowest, highest), a squashed diamond. `is_shattered` says the square's corners are **not** shattered and the diamond **is**.

**why:** for the square, labeling two diagonal corners positive forces any box containing them to contain the whole square, so the other two corners can't be negative. the diamond has no such problem: every subset of its points has a bounding box that misses the others. five points are never shattered (on paper 1): the extremes in the four directions are at most four points, and the fifth sits in their bounding box. so the search's 4 is the true VC dimension. the search can only report a lower bound in general: with an unlucky pool (all points on a line, say, or no four points in "diamond" position) it would report less, and no finite search proves an upper bound. the parameter counts happen to match here; experiment 3 shows that's a coincidence. the halfspace search is slowest despite the smallest answer because every call is an LP (about 0.4 ms): it has to try all $\binom{12}{4} = 495$ four-point subsets, each failing within its first 16 labelings (Radon), before it may stop. the box search makes more calls (6,773, including all $\binom{12}{5} = 792$ five-point subsets) but each costs microseconds.

## experiment 2: the growth function against Sauer's lemma

**what people usually predict:** that halfspaces in the plane, with VC dimension 3, realize "most" labelings of 10 points, or that Sauer's lemma is tight.

**what happens:**

| m | $2^m$ | halfspaces | Sauer $d = 3$ | intervals | Sauer $d = 2$ |
|---|---|---|---|---|---|
| 3 | 8 | 8 | 8 | 7 | 7 |
| 4 | 16 | 14 | 15 | 11 | 11 |
| 5 | 32 | 22 | 26 | 16 | 16 |
| 6 | 64 | 32 | 42 | 22 | 22 |
| 8 | 256 | 58 | 93 | 37 | 37 |
| 10 | 1,024 | **92** | 176 | 56 | 56 |

halfspaces realize 92 of 1,024 labelings of 10 points (9%), exactly Cover's $m^2 - m + 2$ at every $m$; intervals realize exactly $1 + m + \binom m2$, which *is* Sauer's bound for $d = 2$. the halfspace counts fall below Sauer's bound from $m = 4$ on, reaching about half of it at $m = 10$. on the log scale, both measured curves peel away from $2^m$ right after $m = d$ and bend into polynomials.

**why:** Sauer's lemma is tight for some class of every VC dimension (intervals, or "subsets of size at most $d$"), but not for every class. its proof counts every labeling that could be "generated" by subsets of size at most $d$; for halfspaces, labelings are cut by a line, and a line through $m$ points in general position separates them in only $m^2 - m + 2$ ways (rotate a line around the plane: each pair of points gives two critical directions). the exact constant doesn't matter for learning: what matters is that the count is polynomial ($\sim m^d$), so $\ln\Pi(m) \approx d\ln m$, and the union bound over labelings costs a factor $d\ln m$ instead of $m\ln 2$. with $2^m$ labelings the union bound would be useless.

## experiment 3: one parameter, infinite VC dimension

**what people usually predict:** that a class with one real parameter can shatter at most one or two points ("VC dimension ≈ number of parameters").

**what happens:** $\mathbb 1[\sin(\omega x) > 0]$ shatters the points $10^{-1}, \dots, 10^{-n}$ for every $n$ from 1 to **15**: all $2^{15} = 32{,}768$ labelings are realized by the recipe $\omega = \pi(1 + \sum_i(1 - y_i)10^i)$. at $n = 16$ only 52,697 of 65,536 labelings (80%) come out right.

**why:** the recipe encodes the labeling in the decimal digits of $\omega$: $\omega x_j$ is $\pi$ times a number whose integer part's parity is set by $y_j$ (the larger terms contribute even multiples of $\pi$, the smaller ones less than $\pi$ in total). a single real number holds infinitely many digits, so in exact arithmetic the class shatters any finite set and has **infinite VC dimension**: it's not PAC learnable, despite having one parameter. the failure at $n = 16$ is floating point, not mathematics: $\omega$ reaches about $10^{16}$, where float64 spacing exceeds 1, so the digits that encode the labels are lost. that doesn't rescue the class in practice (15 shattered points already means the VC bound needs far more than 15 examples, and the class can fit essentially any labeling of a few points), but it does show the sense in which a float64 parameter is a finite class: $2^{64}$ values, $\ln|\mathcal H| = 44$, and the finite-class bound of topic 01 applies. the VC dimension measures how many labelings a class can *produce on data*, not how many numbers it takes to describe a hypothesis.
