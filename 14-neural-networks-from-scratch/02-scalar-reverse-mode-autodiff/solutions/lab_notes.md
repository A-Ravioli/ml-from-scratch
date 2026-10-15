# scalar reverse-mode autodiff · lab notes

timings vary by machine and run (both engines are pure Python, so garbage collection adds noise); the growth rates and the orders of magnitude are stable.

## experiment 1: one gradient, two modes

**what people usually predict:** that the two modes cost about the same, or that forward mode is faster because it needs no graph.

**what happens:** both give the same gradient (to $10^{-15}$). reverse mode's time grows linearly with $n$ (about 0.06 ms at $n = 10$, 2–3 ms at $n = 300$); forward mode's grows quadratically (0.2 ms at $n = 10$, about **450 ms** at $n = 300$). at $n = 300$ reverse mode is roughly **150–200 times** faster, and the ratio keeps growing with $n$. on the log-log plot the two curves follow slopes 1 and 2.

**why:** forward mode seeds one input's dual part per pass, so the gradient takes $n$ passes, each of which evaluates the whole function ($\propto n$ operations): $O(n^2)$ in total. reverse mode evaluates the function once while recording the graph ($\propto n$ nodes) and sweeps it backwards once, each `_backward` doing a constant amount of work: $O(n)$ in total. the measured ratio at $n = 300$ is close to on paper 4's estimate of 150. forward mode would still be the right tool for a function with few inputs and many outputs, or when the graph wouldn't fit in memory. for a network with a million weights and one loss, forward mode would need a million passes per training step; reverse mode needs one backward pass, which is why every deep learning framework is built on it.

## experiment 2: the price of thinking in scalars

**what people usually predict:** that a scalar engine is "maybe 10 or 100 times slower" than NumPy.

**what happens:** one forward + backward pass through one $784 \to 32$ layer, for one image, builds a graph of **76,145** `Value`s and takes about **0.8 s**; NumPy does the same arithmetic, with the hand-written backward pass, in about **60–75 µs**. that's a slowdown of about **11,000–14,000×**. the gradients agree to $10^{-15}$. at that speed, one epoch of MNIST through this single layer would take the scalar engine about **13–14 hours**.

**why:** the arithmetic is tiny (about 25,000 multiply-adds forward, twice that backward). the scalar engine spends its time on bookkeeping: every multiply-add creates two Python objects, two closures and two tuples, each operation goes through several Python function calls, and the backward pass walks 76,000 nodes one at a time. NumPy does each layer's multiply-adds in one call into compiled, vectorized (BLAS) code, with no per-number objects at all. the fix keeps reverse mode and changes the *unit* of the graph: let each node hold a whole array and each operation be a whole matrix product, so the graph for this layer has a handful of nodes instead of 76,000 and the Python overhead is paid per layer, not per number. that's topic 03's `Tensor`.
