# backprop in depth · lab notes

timings vary by machine and run; the memory numbers (in units of one $2048 \times 128$ float64 activation, 2 MB) and the pass counts are stable.

## experiment 1: tall and wide Jacobians

**what people usually predict:** that reverse mode always wins ("backprop is what everyone uses"), or that the two are about the same since the Jacobian has 400 entries in every case.

**what happens:**

| shape ($n \times m$) | forward passes | backward passes | forward mode | reverse mode |
|---|---|---|---|---|
| 400 × 1 | 400 | 1 | 528 ms | 0.2 ms |
| 40 × 10 | 40 | 10 | 22 ms | 0.2 ms |
| 20 × 20 | 20 | 20 | 5.5 ms | 0.7 ms |
| 10 × 40 | 10 | 40 | 13 ms | 0.5 ms |
| 1 × 400 | 1 | 400 | 0.4 ms | 8.0 ms |

the two Jacobians agree to $10^{-16}$. for the wide case (400 inputs, one output) reverse mode is about **2,500 times** faster; for the tall case (one input, 400 outputs) forward mode is about **20 times** faster. the crossover is near the tall end, not at the square shape.

**why:** forward mode needs one pass per input and reverse mode one backward pass per output, so the pass counts swap as the shape goes from wide to tall. the wide case is exactly what training looks like: many parameters, one scalar loss, so one backward pass gives the whole gradient while forward mode needs one pass per parameter. the crossover isn't at $n = m$ because the cost *per pass* is very different: your `Dual` runs on NumPy object arrays, where every multiplication is a Python method call, so one forward-mode pass costs about 100 times a NumPy pass of the same size (compare the 20 × 20 row: 20 passes each way, 5.5 ms against 0.7 ms). a constant factor can't overcome a factor of $n$ when $n$ is a million parameters, though: for a scalar loss reverse mode wins by a margin that grows with the model, whatever the implementation. a forward mode on raw arrays (JAX's `jvp`, PyTorch's forward-mode AD) removes the constant, and then the crossover moves to $n \approx m$.

## experiment 2: memory for compute

**what people usually predict:** that any checkpointing saves memory, that more segments (more checkpoints) save more, and that recomputation doubles the training time.

**what happens:**

| segments | held after forward | graph_nbytes | peak | layer evaluations | time |
|---|---|---|---|---|---|
| none | 50.0 | 33.0 | 54.0 | 16 | ~110 ms |
| 1 | 2.0 | 1.0 | **56.0** | 32 | ~150 ms |
| 2 | 3.0 | 2.0 | 33.5 | 32 | ~135 ms |
| 4 | 5.0 | 4.0 | 23.2 | 32 | ~160 ms |
| 6 | 7.0 | 6.0 | 22.1 | 32 | ~165 ms |
| 8 | 9.0 | 8.0 | **21.1** | 32 | ~155 ms |
| 16 | 17.0 | 16.0 | 25.1 | 32 | ~175 ms |

with one segment the graph holds almost nothing after the forward pass (2 activations instead of 50), but the peak is **56**, *higher* than without checkpointing (54). the peak falls to a minimum of **21.1** at 8 segments, 2.6 times lower than without checkpointing, and rises again to 25.1 at 16. every checkpointed run evaluates exactly 32 layers, twice the plain run, and takes about 20–50% longer.

**why:** what matters is the peak, and the peak happens during backward. with one segment the backward pass recomputes the whole network with a full graph, holding everything the plain run held plus the segment's saved input and output, so it's slightly worse. with $s$ segments, the backward pass holds the $s$ saved segment outputs plus one segment's rebuilt graph, about $3L/s$ activations (three arrays per layer here: the transposed weight is small, but the matrix product, the tanh output and the tanh's saved output in its closure are each a full activation. the third exists only because the reference `from_op` copies `data` into the new Tensor, so the closure's array and the node's `.data` are separate; an engine that doesn't copy holds about two per layer, and the absolute numbers 50, 54 and 21.1 shrink accordingly while the shape of the curve stays the same), so memory $\approx s + 3L/s$, minimized near $s = \sqrt{3L} \approx 7$. the model curve has its minimum in the right place (6–8 segments) but sits about 7 activations below the measurement: it ignores the gradients themselves (the incoming gradient, the outgoing one, and temporaries inside each VJP), which a real memory budget must include. at 16 segments, every layer is checkpointed: 16 saved outputs plus one layer's graph, back up to 25. the compute cost is one extra forward evaluation of every layer, hence exactly 32 evaluations for any $s \ge 1$. since forward + backward costs about three forward passes, one extra forward pass should add about a third; the measured 20–50% matches that within the timing noise.
