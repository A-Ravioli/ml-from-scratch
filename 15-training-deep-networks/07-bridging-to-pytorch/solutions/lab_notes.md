# bridging to PyTorch · lab notes

numbers are for real MNIST (standardized pixels), measured with the reference solution on an Apple-silicon laptop whose best accelerator is `mps`, while other jobs were running. experiment 2's timings depend on the machine and its load (two runs differed by up to 30% at the largest width); the ratios and the crossover points are what to compare. offline (the synthetic stand-in that `verify.py` uses), experiments 1 and 2 give the same picture, and experiment 3 scores 100% for both models.

## experiment 1: two engines, one trajectory

**what people usually predict:** the float64 copies agree to about $10^{-16}$ at first and then drift apart; the float32 copy drifts until the two models are unrelated.

**what happens:** the largest weight difference between your engine and its PyTorch copy, training in lockstep:

| torch dtype | step 1 | step 10 | step 30 | step 100 | step 300 |
|---|---|---|---|---|---|
| float64 | 1.4e-17 | 4.2e-17 | 1.4e-16 | 2.2e-16 | 4.7e-16 |
| float32 | 1.0e-08 | 2.5e-08 | 6.0e-08 | 9.5e-08 | 2.0e-07 |

in float64 the two engines stay within a few units in the last place for all 300 steps. the float32 copy starts $10^{-8}$ away and its gap grows only slowly, by a factor of 20 in 300 steps, to $2 \times 10^{-7}$.

**why:** the float64 differences come from summing the same products in a different order inside the matrix multiplications (NumPy's and PyTorch's BLAS kernels block and parallelize differently; on paper 1), so they start at the rounding level and stay there: 300 steps of SGD on this smooth, well-conditioned problem don't amplify them noticeably. the float32 copy starts with every weight rounded to 7 significant digits ($\sim 10^{-8}$ absolute for weights of size $\sim 0.05$), and every subsequent operation rounds again; those errors accumulate roughly like a random walk, which is the slow growth in the table. over many thousands of steps, especially through unstable phases, the trajectories would eventually separate (training is chaotic, topic 06's lab), and that doesn't matter: a different float32 rounding is like a different seed, not a bug. what this experiment has shown that no gradient check could: your engine computes *the same function and the same gradients as PyTorch*, step after step, through the whole training loop, including your `SGD`.

## experiment 2: how fast is each engine?

**what people usually predict:** PyTorch is 10–100× faster than a NumPy engine everywhere, and the GPU is faster still.

**what happens:** milliseconds per training step (batch 256, best of 5):

| engine | w = 64 | w = 256 | w = 1024 | w = 2048 |
|---|---|---|---|---|
| mlfs (NumPy, float64) | 0.77 | 2.61 | 20.4 | 125 |
| torch CPU float32 | 0.89 | 2.35 | 12.4 | 34.6 |
| torch CPU float64 | 1.82 | 6.27 | 24.1 | 87.9 |
| torch mps float32 | 1.87 | 2.11 | 3.86 | 8.63 |

for small networks your engine is as fast as PyTorch on the CPU (0.9× to 1.1×). at width 1,024 it's *faster* than PyTorch in float64 (20 versus 24 ms) and only 1.7× slower than PyTorch in float32; at 2,048, 3.6× slower than float32. the GPU is the *slowest* option at width 64, about even at 256, 5× faster than the CPU at 1,024 and 15× faster than your engine at 2,048. (offline: 0.8×, 1.1×, 1.6× and 2.6×.)

**why:** for large layers almost all the time goes into a handful of matrix products, and both NumPy and PyTorch hand those to an optimized BLAS library (Accelerate on a Mac, MKL or OpenBLAS elsewhere). so at large widths the comparison is mostly float64 BLAS versus float32 BLAS (twice the numbers per vector register, half the memory traffic), plus your engine's extra copies (every `Tensor` stores a fresh array; `unbroadcast` and `from_op` allocate). at small widths the matrix products take microseconds and the time is overhead: a few dozen Python-level operations per step in your engine, PyTorch's dispatcher and autograd bookkeeping in C++, comparable costs. the GPU adds a fixed cost per kernel launch (and a synchronization at the end of the step); with tiny matrices that's all there is, so it loses. it wins once each kernel has enough arithmetic to fill the GPU, here somewhere between widths 256 and 1,024. the practical lesson: profile before you move a small model to a GPU, and know that your own engine was never the bottleneck for chapter 14's models.

## experiment 3: an exponential moving average of the weights

**what people usually predict:** the EMA lags behind, so it scores a bit worse than the trained model, or the same.

**what happens:** test accuracy after each epoch:

| epoch | model | EMA (decay 0.99) | EMA − model |
|---|---|---|---|
| 1 | 0.9442 | 0.9508 | +0.0066 |
| 2 | 0.9525 | 0.9666 | +0.0141 |
| 3 | 0.9611 | 0.9724 | +0.0113 |
| 4 | 0.9618 | 0.9734 | +0.0116 |

the untrained-by-itself average beats the trained model at every epoch, by about 1.1–1.4 points after the first; the gap is several times the binomial standard error of these accuracies (about 0.0018 on 10,000 images), and similar to what a decayed learning rate would buy.

**why:** with a constant learning rate of 0.1 and momentum 0.9, SGD's weights keep jumping around the region of a minimum (the noise floor of topic 04, on paper 1) instead of settling. averaging the last ~100 positions ($1/(1 - 0.99)$) lands near the middle of that cloud, where the loss is lower than at a typical point of it (on paper 3: the averaged weights' variance shrinks by a factor of about $(1 - \delta)/(1 + \delta)$ for uncorrelated noise). it's the same effect a learning-rate decay achieves by shrinking the jumps, obtained instead by averaging them, with no change to training at all. early in training the EMA lags (its first epoch includes the first ~100 steps' worse weights), which is why the gap is smallest after epoch 1, and why large-scale recipes use decays like 0.999–0.9999 only for long runs. buffers are copied, not averaged, because running statistics must match the weights they're used with; this MLP has no buffers, but the batch-norm networks of later chapters do.
