# a tensor autograd engine · lab notes

## experiment 1: a wrong gradient that still trains

**what people usually predict:** a wrong gradient either breaks training completely (the loss blows up or stalls) or, if it's "close enough", makes no visible difference.

**what happens:** with $n = 200$, all three versions start at a loss of 9.97 and all three go down. the correct gradient reaches **0.009** (the noise level) within about 60 steps and learns the bias as **3.00**. the mean bug ends at **6.34** with a bias of **0.39**, the first-row bug at **6.09** with **0.44**. both buggy curves look like ordinary slow training: the loss is still falling at step 300. your `gradcheck` flags both (the bias gradient it prints is off by a factor of about 8 here, since the check uses 8 rows) and passes the correct one. moving the slider makes the mean bug worse in proportion to $n$: at $n = 1000$ the bias barely moves at all.

**why:** the weight gradient is untouched, so $w$ trains normally; only the bias is wrong. the mean bug's bias gradient points the right way but is $n$ times too small, so the bias crawls at an effective learning rate of $0.05/n$; the first-row bug uses one example's residual, a noisy but on-average-correct direction scaled down by $n$ as well. a descent direction that's merely too short still decreases the loss, so nothing looks broken: the loss curve just looks like a network that "needs more epochs" or "a higher learning rate". in a real network with millions of parameters this kind of bug can survive for weeks, and the only reliable way to catch it is to compare against finite differences on a small case. that's why every operation in the engine gets a `gradcheck`, and why chapter 15 checks the whole engine against PyTorch.

## experiment 2: the bug gradcheck can't see

**what people usually predict:** "`pred - y` is fine, NumPy will line them up", or that the model learns something slightly worse.

**what happens:** `pred - y` has shape **(200, 200)**. the model with the shape bug learns **nothing**: its weights end at about $10^{-12}$, its bias at **2.82** (exactly the mean of $y$), and its train $R^2$ is **0.000**. the scatter plot is a flat line. the correct model reaches $R^2 = 0.989$. and **`gradcheck` passes on both losses.**

**why:** broadcasting $(n, 1) - (n,)$ gives the matrix $D_{ij} = \hat y_i - y_j$, so the buggy loss is

$$\frac{1}{n^2}\sum_{i,j}(\hat y_i - y_j)^2 = \frac1n\sum_i(\hat y_i - \bar y)^2 + \operatorname{Var}(y),$$

the average squared distance from each prediction to *every* target. it's minimized by predicting $\bar y$ for every input, which the model can do exactly with $W = 0$, $b = \bar y$. there's no gradient bug at all: the engine differentiates this function perfectly, so `gradcheck` passes. a gradient check verifies that backward matches the function you wrote, not that you wrote the function you meant. the cure is a shape assertion in the loss (`assert pred.shape == target.shape`), which is why your `mse_loss` in topic 04 refuses mismatched shapes instead of broadcasting them.

## experiment 3: what does a gradient cost?

**what people usually predict:** that backward is "about as expensive as the forward pass" (ratio 2), or that it gets relatively more expensive as there are more weights to differentiate.

**what happens:** forward plus backward takes about **1.5–5 times** a forward pass at every width, with no trend in the number of weights, which grows from 1,024 to 4.2 million (the exact ratios move by ±1 between runs; typical values are 1.6, 1.5–3, 1.8–2.4 and 3.9–4.5 for widths 16, 64, 256, 1024). finite differences would need $2n$ forward passes: about **0.3 ms** for the smallest network but around **47 hours** for the largest, where backprop takes under 0.1 s.

**why:** each forward matrix product $H W^\top$ becomes two backward products of the same size ($\bar Y W$ for the input and $\bar Y^\top H$ for the weights), and each elementwise operation becomes one elementwise VJP. so the backward pass does about twice the forward pass's arithmetic, and the total is about 3 forward passes, whatever $n$ is: the "cheap gradient principle" (on paper 4). at small widths both passes are dominated by Python overhead per operation rather than arithmetic, which is why the ratio there is closer to the ratio of operation counts; at the largest width memory traffic (allocating and writing a fresh $1024 \times 1024$ gradient per layer) pushes it well above 3 (about 4). for the capstone's network (about 200,000 weights, a few milliseconds per forward pass on a batch), finite differences would take on the order of 20 minutes *per step*; backprop takes a few milliseconds.
