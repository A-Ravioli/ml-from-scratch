# exact inference · lab notes

timings below are best-of-3 measurements from laptop runs of the reference solution on a machine shared with other jobs; they vary by tens of percent between runs, and yours will differ in absolute terms, but the shapes should be similar.

## experiment 1: the order is everything

**what people usually predict:** that a random order is perhaps 2–4 times worse than a good one, i.e. tables of $2^8$ or $2^9$ entries.

**what happens:** min-degree and row-by-row both reach the minimum for this grid, a largest table of $2^7 = 128$ entries. the 300 random orders have a median largest table of $2^{13}$ (64 times bigger) and the worst one $2^{20}$, about a million entries, 8,000 times bigger. the best random order was $2^9$; not one random order matched min-degree. all orders give the same probability, $P(g_{0202} = 1) = 0.717679$, to every printed digit. in wall-clock time the difference is still small at this size: about 1.2 ms with min-degree and about 2 ms with the most expensive random order we ran ($2^{17}$ entries, 1 MB), because NumPy multiplies a million numbers in well under a millisecond. the cost is in the exponent: the $2^{20}$ order needs 8 MB tables, and every extra row of fill doubles them. on a $10\times10$ grid a bad order is out of reach while min-degree stays at $2^{11}$.

**why:** eliminating a variable joins all its current neighbours into one table, and they stay joined. a random order soon eliminates variables in the middle of the grid, whose neighbourhoods have already grown through fill-in, so the frontier of "variables tied together" spreads in two dimensions. min-degree eats the grid from a corner, keeping a frontier of about one row ($n$ variables, plus the one being eliminated: $2^{n+1} = 2^7$ for $n = 6$). that's essentially optimal: the treewidth of an $n \times n$ grid is $n$, so no order can do better than tables of about $2^{n+1}$ entries. for a $100 \times 100$ binary image that's $2^{101}$ entries: exact inference is impossible however cleverly you order, which is why image models use sampling (topic 05), variational methods (topic 07) or loopy BP. finding the best order is NP-hard in general (Arnborg, Corneil & Proskurowski, 1987), but greedy heuristics like min-degree and min-fill are good enough in practice.

## experiment 2: brute force versus elimination

**what people usually predict:** a large but fixed factor, often "1,000×", or that NumPy's speed makes brute force competitive.

**what happens:** for short chains brute force is as fast or faster (around $n = 10$ the two cross over: NumPy multiplies a few thousand numbers faster than elimination does its bookkeeping). after that, a straight-line fit of log time against $n$ over $n = 14..22$ gives a growth factor of about **2.2× per extra variable** for brute force (theory: 2, the table doubles) and about 1.08× for elimination (linear growth looks nearly flat on this scale). at $n = 22$ brute force took about 180 ms and elimination about 0.3 ms, a ratio of about 500× (single-step ratios like "$n = 20 \to 22$" are dominated by timer noise for the sub-millisecond elimination runs, which is why we fit a slope). the ratio itself grows exponentially: at $n = 30$ brute force would need $2^{30}$ entries (8 GB), at $n = 40$ 8 TB, while elimination would still take about a millisecond.

**why:** brute force materializes a table with $2^n$ entries, so each variable doubles its cost; elimination on a chain carries a single 2-vector from one end to the other ($\tau(x_{i+1}) = \sum_{x_i}\tau(x_i)\phi(x_i, x_{i+1})$), so each variable adds a constant. that vector is exactly the forward message $\alpha_t$ of an HMM (topic 04); the comparison is the reason HMMs are usable at all. the moral: whenever a computation enumerates joint configurations, look for a sum you can push inside.

## experiment 3: all the marginals at once

**what people usually predict:** 2–10× faster, or "the same, because both are exact".

**what happens:** 80 variable-elimination calls took about 0.4–0.5 s; one `sum_product` call (best of 3) took about 2 ms: roughly **200–300×** faster across runs, with the two agreeing to $2\times10^{-16}$.

**why:** each elimination costs $O(n)$ table operations on a tree, and doing it for every variable costs $O(n^2)$, recomputing the same intermediate tables over and over: the message from a subtree to its parent is identical in every query that lies outside that subtree. sum-product computes each of the $2E \approx 2n$ directed messages once and assembles every marginal from them: $O(n)$ for all marginals together, so the speedup grows linearly with $n$ (from the operation count alone it would be about $n/2 = 40$; the rest is per-call overhead in variable elimination, which builds an order and reduces every factor on each call). this caching of shared subcomputations is the same idea as dynamic programming, and the same reason reverse-mode autodiff computes all partial derivatives in one backward pass (ch 14). on graphs with loops the analogous exact algorithm is the junction tree, which runs sum-product on a tree of cliques; its cost is exponential in the treewidth, which brings back experiment 1.
