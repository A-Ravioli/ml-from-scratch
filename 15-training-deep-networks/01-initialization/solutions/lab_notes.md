# initialization · lab notes

numbers are for real MNIST (standardized pixels), measured with the reference solution. offline (the synthetic stand-in that `verify.py` uses) experiments 1 and 2 look the same; experiment 3 differs, as noted.

## experiment 1: fifty layers of signal

**what people usually predict:** that kaiming keeps the std near 1, that xavier is "close enough" for ReLU, and that small random weights are the safe default. few people put a number on the failures.

**what happens:**

| network | std, layer 1 | std, layer 10 | std, layer 50 | ‖∂L/∂W‖, layer 1 | ‖∂L/∂W‖, layer 51 |
|---|---|---|---|---|---|
| tanh · N(0, 0.01²) | 0.259 | 1.7e-08 | 2.5e-40 | 5.8e-40 | 5.9e-40 |
| tanh · xavier | 0.677 | 0.226 | 0.101 | 0.544 | 0.257 |
| relu · xavier | 0.702 | 0.037 | 3.5e-08 | 8.2e-08 | 4.1e-08 |
| relu · kaiming | 0.808 | 0.968 | 0.958 | 3.39 | 7.75 |
| relu · torch default | 0.343 | 0.023 | 0.020 | 1.3e-19 | 0.025 |

only kaiming keeps the scale. the small init loses a factor of about 6 in std per layer and reaches $10^{-40}$; relu · xavier loses $\sqrt2$ per layer, $3.5 \times 10^{-8}/0.70 = 5 \times 10^{-8} \approx 2^{-49/2}$, exactly the law; tanh · xavier decays slowly (0.68 → 0.23 → 0.10); and the torch default drops fast and then *stops* at 0.02.

**why:** each line is the variance law $\operatorname{Var}(z_{l+1}) = n\operatorname{Var}(W)E[h_l^2]$. with $N(0, 0.01^2)$ and $n = 256$ the factor on the second moment is 0.0256, a std factor of 0.16 per layer; xavier on square ReLU layers gives a factor ½ on the second moment; kaiming gives exactly 1. tanh is linear near zero, so xavier is nearly variance-preserving for small activations, and the decay comes only from tanh squashing the larger values, which matters less and less as the signal shrinks: a slow, roughly power-law decay instead of a geometric one. the torch default's floor is its random bias: the input's share of the signal shrinks by a factor 6 per layer, but every layer adds a fresh bias of variance $1/(3n)$, and the recursion settles at $q^* = 2/(5n)$, an activation std of about 0.023 (on paper 4). so the floor is there, but it's made of biases, not of the input. the gradients mirror the forward pass, because $\partial L/\partial W_l = \bar z_l h_{l-1}^\top$ multiplies a backward signal by a forward one: with the default init the backward signal shrinks by the same factors on its way down, so the first layer's gradient is $10^{-19}$ while the last layer's is a healthy 0.025; with the small init both factors are tiny everywhere.

## experiment 2: can the network still tell two images apart?

**what people usually predict:** that kaiming, having "fixed" the signal, keeps images as distinguishable as they were (about 0.3), and that only the failing inits lose information.

**what happens:** the mean cosine similarity between 500 images' activation vectors:

| network | layer 1 | layer 5 | layer 10 | layer 25 | layer 50 |
|---|---|---|---|---|---|
| tanh · xavier | 0.274 | 0.268 | 0.250 | 0.173 | 0.141 |
| relu · xavier | 0.471 | 0.773 | 0.910 | 0.965 | 0.993 |
| relu · kaiming | 0.471 | 0.773 | 0.910 | 0.965 | 0.993 |
| relu · torch default | 0.495 | 0.962 | 1.000 | 1.000 | 1.000 |

(pixels: 0.29.) relu · kaiming's representations converge: by layer 50 every pair of images is at cosine 0.993, nearly the same direction. relu · xavier gives *exactly* the same numbers. the torch default collapses completely by layer 10. tanh · xavier is the only one that keeps images apart, and even pushes them slightly further apart.

**why:** xavier and kaiming draw the same random matrices up to a constant factor per layer (both use `normal(0, σ)` with the same generator, and their σ differ by a fixed factor per layer, $\sqrt 2$ on the square layers), and ReLU is positively homogeneous, $\operatorname{relu}(cz) = c\operatorname{relu}(z)$ for $c > 0$. so the two networks' activations differ by a constant factor at every layer, and a cosine ignores constant factors. the *scale* is what xavier gets wrong; the *geometry* is identical. the geometry is the problem: a ReLU zeroes negative coordinates, and two vectors that disagreed in sign on some coordinates agree after the ReLU (both zero there), so every ReLU layer makes vectors more aligned, and the effect compounds with depth. tanh is odd-symmetric and keeps signs, so it has no such pull (in the language of Poole et al. 2016, this tanh network sits near the "edge of chaos", where correlations are preserved). the torch default's 1.000 is the bias floor of experiment 1 seen from another angle: at layer 10 the activations are the biases' contribution, the same vector for every image, so the network's output no longer depends on its input. the lesson: variance preservation is necessary but not sufficient. a plain ReLU network much deeper than about 20 layers starts from a nearly input-independent representation, however well you choose the scale, and that's one reason residual connections (ch 16) and normalization (topic 02) were needed to go deeper.

## experiment 3: does it train?

**what people usually predict:** kaiming first, then xavier, then the default, with tanh somewhere in the middle; that nothing stays exactly at chance.

**what happens:** validation accuracy after 3 epochs (20 layers, 128 wide, SGD 0.01 with momentum 0.9), seeds 0, 1, 2:

| network | validation accuracy | mean |
|---|---|---|
| relu · torch default | 0.102 · 0.102 · 0.102 | 0.102 |
| relu · xavier | 0.383 · 0.199 · 0.362 | 0.315 |
| relu · kaiming | 0.845 · 0.880 · 0.870 | 0.865 |
| tanh · xavier | 0.902 · 0.916 · 0.923 | 0.914 |

the torch default stays at chance with a training loss of 2.301 for all three seeds (it predicts the most common class of the held-out set). relu · xavier learns slowly and erratically. kaiming trains. and tanh · xavier, which wasn't variance-preserving, trains best. (offline, on the synthetic stand-in: 0.102, 0.095, 0.365 and 1.000: the same order, with xavier stuck at chance and tanh solving the easy synthetic task outright.)

**why:** the torch default's input signal (and its gradient at the first layers) is $10^{-19}$-ish: SGD can only fit the last layer's bias to the label frequencies, which is what a loss of exactly $\log 10$ looks like. relu · xavier starts with a signal and gradients shrunk by $2^{-10}$ in std; momentum SGD slowly grows the weights until the scale recovers, and how fast that happens depends on the random draw, hence the spread across seeds. kaiming starts at the right scale and trains, but its representations start correlated (experiment 2), and its gradients are large (the norms of 3–8 in experiment 1), which at this learning rate makes the first epoch noisy. tanh · xavier at 20 layers keeps both a usable scale (about 0.2) and well-separated inputs, and its bounded activations make large gradients less likely. none of this says tanh beats ReLU in general: with normalization (topic 02) or residual connections (ch 16), ReLU networks train far deeper. what it does say is that "initialization" means getting both the scale and the geometry right, and that the default init of a framework is tuned for shallow networks.
