# sets · lab notes

numbers measured with the reference solution (seed 0, 800 Adam steps per model, CPU, about 20 s for the whole lab). the data are synthetic, so `verify.py`'s offline run gives the same numbers. we also retrained every model with seeds 1 and 2, and the sum-pooled ones with seeds 1–4: the failures (near-yardstick RMSEs) and the perfect max-pooled max are identical across seeds; the in-distribution RMSEs of the partly successful pairs vary (sum pooling on the max task: 0.028–0.077; on the count: 0.33–0.53), and so does how badly sum pooling extrapolates on the max task (below).

## experiment 1: which aggregation can learn which task?

**what people usually predict:** that all three aggregations do fine on all three tasks ("it's a universal approximator"), or that max is the obvious winner for the max task and sum for the rest, with no strong opinion about the failures.

**what happens** (test RMSE, sets of 1–10; predicting the mean target gives 1.59 / 0.21 / 1.86):

| aggregation | sum | max | count > 0.5 |
|---|---|---|---|
| sum | **0.003** | 0.057 | **0.329** |
| mean | 1.277 | 0.081 | 1.432 |
| max | 1.078 | **0.002** | 1.395 |

sum pooling learns all three tasks (the count least well: it needs a step function at 0.5, which $\phi$ only approximates in 800 steps). mean and max pooling fail the sum and the count outright, staying within 20–30% of the do-nothing yardstick. max pooling learns the max almost perfectly; mean pooling only roughly.

**why:** the failures are information-theoretic, not optimization problems (on paper 2). mean pooling sees the *distribution* of $\phi(x_i)$, which is the same for a set and for that set repeated twice, so it can't recover $n$, and both the sum and the count scale with $n$. the residual it does get right is the part predictable from the distribution alone (the average element, the fraction above 0.5). max pooling sees only *which* values occur, so it's blind to multiplicities too. the max task is different: the max is a function of the support, so max pooling with $\phi(x) = x$ solves it exactly, and mean pooling can approximate it by making $\phi$ steeply emphasize large values, which is hard to do precisely. sum pooling is the only one that keeps everything (the injectivity argument of on paper 1), but notice that "can represent" didn't mean "learns as well": its max is 30× worse than max pooling's.

## experiment 2: bigger sets than you trained on

**what people usually predict:** that a model that learned a task well keeps working on bigger sets, maybe with some degradation; most expect the sum-pooled network to be the most robust, since it was the best in experiment 1.

**what happens** (RMSE, sizes 1–10 → sizes 50–100):

| aggregation | sum | max | count > 0.5 |
|---|---|---|---|
| sum | 0.003 → 0.017 | 0.057 → **0.362** | 0.329 → 1.585 |
| mean | 1.277 → 34.7 | 0.081 → 0.103 | 1.432 → 34.8 |
| max | 1.078 → 34.1 | 0.002 → **0.001** | 1.395 → 34.3 |

the max-pooled network's max predictions stay perfect on sets 5 to 10 times larger than any it has seen. the sum-pooled network, which had learned the max task respectably, gets six times worse: in the plot its predictions scatter between 0.2 and 1.05 when every true max is above 0.9. across seeds 0–4 this degradation ranges from 4× to 32× (0.12 to 2.5 RMSE on sets of 50–100; on sets of 20–30 it's milder, 1.1× to 7×, which is why the lab uses the bigger sets). the sum task itself extrapolates almost perfectly with sum pooling (0.017 when the targets are around 37), and the count degrades but stays meaningful (1.6 when the targets are around 37). mean and max pooling, already failing the sum and the count, now miss by about 34, which is just the growth of the target with $n$.

**why:** to compute a max from a sum, the network has to arrange that the sum is dominated by the largest element, for example $\phi(x) = e^{\beta x}$ and $\rho(s) = \log(s)/\beta$ (a "soft max"), and that trick's error depends on how many other elements contribute to the sum. it learned a version that works for $n \le 10$; at $n = 75$ the pooled vector is far outside anything $\rho$ was trained on, and $\rho$ extrapolates whatever its last ReLU pieces happen to do, which differs from seed to seed. max pooling's representation of a 75-element set is the same kind of object as a 5-element set's (a vector of per-feature maxima, all in the training range), so nothing is out of distribution. the sum task extrapolates for the opposite reason: $\phi(x) \approx x$ and a nearly linear $\rho$ is the exact solution at every size, and a ReLU network extrapolates linearly. the rule: pick the aggregation whose structure matches the task's, and the model inherits the task's behaviour on sizes it hasn't seen.

## experiment 3: does attention fix it?

**what people usually predict:** that learned pooling is at least as good as the best fixed aggregation on every task, since it could learn to be any of them.

**what happens** (RMSE, sets of 1–10):

| model | sum | max | count > 0.5 |
|---|---|---|---|
| attention pooling (4 heads) | 1.170 | 0.003 | 1.445 |
| best fixed aggregation | 0.003 (sum) | 0.002 (max) | 0.329 (sum) |

attention pooling matches max pooling on the max (each head can put nearly all its weight on the largest element) and fails the sum and the count exactly as badly as mean pooling does. seeds 1 and 2 give the same picture (sum 1.17–1.20, max 0.003, count 1.45).

**why:** attention weights are a softmax, so attention pooling is a *weighted average*: the stage 4 check showed that a set listed twice gives exactly the same output. a weighted average can be as sharp as a max or as flat as a mean, but it can never be a sum, because the weights always add up to 1. self-attention blocks before the pooling don't change this on their own (each block is again a normalized average); the cheap fixes are to concatenate a sum-pooled vector, to gate with sigmoids instead of a softmax (a learned weighted *sum*), or to append $\log n$ as a feature. "learned" doesn't mean "more expressive" when the learned thing has a normalization built in.
