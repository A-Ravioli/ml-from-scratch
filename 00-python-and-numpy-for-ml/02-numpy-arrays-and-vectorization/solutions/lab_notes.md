# numpy arrays & vectorization · lab notes

numbers below are from an Apple-silicon laptop; yours will differ, but the ratios and shapes of the curves are typical.

## experiment 1: how much does vectorization buy?

**what people usually predict:** 10–100×, roughly constant across sizes.

**what happens:** at $n = 10$ the speedup is about 20×. it climbs to about 140× at $n = 30$, 500× at 100, and **roughly 900× at $n = 300$** (0.13 s for the loop against 0.15 ms). beyond that it levels off in the high hundreds: at $n = 3{,}000$ your function takes about 17 ms for nine million distances, where the loop would need about 13 s. a bigger $d$ makes the gap larger still, because each loop iteration does more work in many small NumPy calls, while the vectorized version just feeds a bigger matrix product.

**why:** the loop pays a fixed interpreter cost per pair, about 1.5 µs here (each iteration makes several small NumPy calls, each with its own overhead, which is why this is even slower than topic 01's pure-Python 20 ns per multiply-add would suggest). its time grows exactly like $n^2$. your version pays a fixed cost per *call* (a few microseconds of setup) and then runs compiled loops. at tiny $n$ that setup is a large part of the total, so the speedup is modest. as $n$ grows, the setup is amortized and the speedup rises until both methods are limited by their per-pair costs: interpreter overhead for the loop, raw arithmetic and memory bandwidth for yours. the plateau is the ratio of those two, and that ratio, hundreds to a thousand, is the everyday payoff of vectorization.

## experiment 2: three loop-free ways to compute distances

**what people usually predict:** "broadcast" is fastest, because it has no Python loop at all. the row loop must be slowest.

**what happens:** broadcast and the row loop are **about equally fast**: roughly 3–4 ms at small $d$ and 23–25 ms at $d = 128$, with the row loop sometimes ahead. your expansion version takes 0.4–0.6 ms at every $d$, **6–40× faster than both**. the broadcast method's largest array grows to 256 MB at $d = 128$ (and squaring it makes a second one), while the expansion method never needs more than the 2 MB result.

**why:** all three do about $n \cdot m \cdot d$ multiply-adds. what differs is **memory traffic**. the broadcast method writes $n \cdot m \cdot d$ differences to memory, reads them back to square them (writing another array of the same size), and reads them again to sum. arrays that big don't fit in the CPU's caches, so the processor spends most of its time waiting for main memory, not computing. the row loop's 500 Python iterations cost only about 500 × a few µs, and each works on a small array that fits in cache, which is why it keeps up. the expansion method spends almost all of its time in one matrix product, which BLAS libraries organize so that each number loaded from memory is reused many times from cache and registers, using vector instructions and several cores. the lesson: "no Python loops" is a good start, but **the fastest vectorized code also avoids creating big intermediate arrays**, and reduces as much of the work as possible to matrix products.

## experiment 3: the fast formula, far from the origin

**what people usually predict:** no change at all, since distances don't depend on where the points are. or a slow growth of tiny errors.

**what happens:** the worst relative error of your `pairwise_distances` grows by about a factor of 100 for every factor of 10 in $c$: about $10^{-14}$ at $c = 1$, $10^{-8}$ at $c = 1{,}000$, $10^{-4}$ at $c = 10^5$, 2% at $c = 10^6$, and **over 100% from $c = 10^7$**: the distances are garbage. the same function applied to the centered points stays at about $10^{-14}$ for every $c$.

**why:** with $c = 10^8$, each coordinate is about $10^8$, so $\|x\|^2 \approx 3 \times 10^{16}$, and $2\,x\cdot y$ is about the same size. their difference, the squared distance, is only about 5. a double holds about 16 significant digits, so each of those huge numbers is only known to within about 4 (one unit in the 16th digit of $3 \times 10^{16}$), and the subtraction leaves an answer that is pure rounding noise. this is **catastrophic cancellation**: subtracting two nearly equal large numbers destroys the significant digits, and the error scales like $c^2 \cdot 10^{-16} / \text{distance}^2$, exactly the factor-of-100-per-decade you saw. the difference-array method doesn't suffer from it, because it subtracts the coordinates *first*, while they're still small relative to their difference. centering fixes the fast formula for the same reason: distances are unchanged by shifting all points, and after subtracting the mean, the coordinates are small again. topic 03 is all about this kind of error, and the habit to keep is simple: **center (or standardize) your data before computing distances**, which most ML pipelines do anyway (chapter 07).
