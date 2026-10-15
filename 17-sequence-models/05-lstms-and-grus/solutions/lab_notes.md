# LSTMs & GRUs · lab notes

numbers are measured with the reference solution on a laptop CPU; the lab uses only synthetic data, so it gives the same results offline. it takes about 1.5 minutes on an idle machine (experiments 1 and 2 about 45 s each); runs are single-seed. we repeated every configuration of experiment 2 with a second seed, and the GRU at delay 80 with two more: every pair agreed to within 0.05 (solved or not solved, never in between).

## experiment 1: how long can each cell remember?

**what people usually predict:** the RNN manages 5–10 steps, the LSTM much longer (it was designed for this), the GRU somewhere in between.

**what happens** (accuracy after 400 steps; chance 0.25):

| cell | delay 5 | delay 10 | delay 20 | delay 40 | delay 80 |
|---|---|---|---|---|---|
| RNN | 0.27 | 0.29 | 0.24 | 0.26 | 0.20 |
| GRU | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| LSTM | 1.00 | 1.00 | 1.00 | 0.28 | 0.20 |

the RNN fails at every delay, even 5. the LSTM solves 20 but not 40. the GRU, with fewer parameters, solves all of them, up to 80.

**why:** to recall the symbol, the network must learn to *store* it on the first step and *protect* it from the distractors, and the only teacher is the gradient of the last step's loss, which has to travel back `delay` steps. for the RNN that gradient passes through $\operatorname{diag}(1 - h^2)W_{hh}$ at every step and is already weak after 5 (in 400 steps with a 3e-3 learning rate; with 1,000 steps it reached 0.50 at delay 5 on one seed of two, and still nothing at 10). in the LSTM, the cell-state path multiplies the gradient by the forget gate, about $\sigma(1) = 0.73$ at initialization: $0.73^{20} = 2 \times 10^{-3}$ is enough to get learning started, $0.73^{40} = 4 \times 10^{-6}$ isn't (experiment 2 makes this exact). the GRU result is the surprise. at initialization its gradient path is actually *weaker* than the LSTM's (experiment 3), so it isn't the starting point. tracking the gates during training at delay 80, the GRU's mean update gate on distractors only rises from 0.51 to about 0.62, but by step 100 three units have $z > 0.95$ on every distractor, by step 150 five do (the largest at 0.9999), and accuracy jumps to 1.0: a handful of units become exact copiers. the LSTM at delay 40 also pushes 27 forget gates above 0.95, but none past 0.98, and it doesn't solve the task. the reason is the GRU's coupled gate: $z \approx 1$ means "keep *and* don't write", so a unit that learns it is immune to distractors, while an LSTM unit has to learn $f \approx 1$ and, separately, $i \approx 0$, and $0.98^{40} = 0.45$ of a memory that is also being overwritten isn't enough. an ablation supports this: an LSTM with the input gate tied to the forget gate, $i = 1 - f$ (the "coupled input and forget gate" variant, CIFG, which Greff et al. (2017) found to perform as well as the full LSTM), trained exactly like the lab's LSTM (forget bias 1, 400 steps, 4 seeds), solves delay 40 on 4 of 4 seeds (accuracy 1.00 each) and delay 80 on 3 of 4 (1.00, 0.87, 1.00, 1.00), where the standard LSTM stays at chance on all 8 runs (0.20–0.28). the single change that couples writing to forgetting is enough to match the GRU. that's a measured result on this task and budget, not a general ranking: published comparisons (Chung et al., 2014; Jozefowicz et al., 2015; Greff et al., 2017, "LSTM: a search space odyssey", [arXiv:1503.04069](https://arxiv.org/abs/1503.04069)) find GRUs and LSTM variants close, with the winner depending on the task.

## experiment 2: the forget-gate bias

**what people usually predict:** a small effect: the bias is one learnable number per unit and training can move it.

**what happens:**

| forget bias | initial forget gate | delay 20 | delay 40 | delay 80 |
|---|---|---|---|---|
| 0 | σ = 0.50 | 0.26 | 0.28 | 0.23 |
| 1 | σ = 0.73 | 1.00 | 0.28 | 0.20 |
| 3 | σ = 0.95 | 1.00 | 1.00 | 0.20 |

the initialization alone decides which delays are learnable: bias 0 fails even at 20, bias 1 reaches 20, bias 3 reaches 40, and none reaches 80 in 400 steps.

**why:** at initialization, the fraction of the cell state (and of its gradient) that survives $d$ steps is $\sigma(b)^d$: for $b = 0$, $10^{-6}$ at $d = 20$; for $b = 1$, $1.9 \times 10^{-3}$ at 20 and $3.6 \times 10^{-6}$ at 40; for $b = 3$, $0.14$ at 40 and $0.021$ at 80. training can raise the bias, but only if a gradient tells it to, and that gradient travels along the same decaying path, so a starting point below the threshold stays stuck. this is why Jozefowicz et al. (2015) found that a forget bias of 1 closed most of the gap between the LSTM and the best architectures in their search, and why `nn.LSTM`'s default (biases near 0) is a known trap. a bias of 3 isn't free either: a cell that starts out remembering everything has to learn to forget, and on tasks with short dependencies it can train more slowly.

## experiment 3: the gradient's path, measured

**what people usually predict:** the RNN's influence decays fast, and the LSTM and GRU stay near 1.

**what happens:** $\|\partial(\sum h_T)/\partial e_1\|$ at initialization (64 sequences):

| cell | T = 1 | T = 2 | T = 5 | T = 10 | T = 20 | T = 50 |
|---|---|---|---|---|---|---|
| RNN | 4.2e+00 | 1.6e+00 | 1.4e-01 | 2.4e-03 | 7.2e-07 | 2.8e-17 |
| GRU | 2.0e+00 | 1.1e+00 | 2.5e-01 | 3.1e-02 | 9.2e-04 | 3.2e-08 |
| LSTM | 1.0e+00 | 7.5e-01 | 3.5e-01 | 1.2e-01 | 1.8e-02 | 2.0e-04 |

all three decay exponentially at initialization, but at very different rates: over 50 steps the RNN loses 17 orders of magnitude, the GRU 8, the LSTM 4. the LSTM has the best-connected gradient path of the three at the start, yet experiment 1's winner was the GRU.

**why:** the slopes match the per-step factors. the RNN's factor, $\operatorname{diag}(1 - h^2)W_{hh}$, combines a derivative below 1 with a matrix of norm about 1, roughly 0.45 per step here. the LSTM's cell path is scaled by the forget gate, about 0.73 per step with bias 1 (and the measured slope, about 0.84, is a little better because the hidden-state paths add to it). the GRU's is scaled by the update gate $z$, about $\sigma(0) = 0.5$ at initialization, plus the candidate path through the reset gate; the measured slope is about 0.69 per step. the GRU starts with a worse gradient path than the LSTM, so its success in experiment 1 is not about the starting point but about how easily training can turn a few units into copiers (see experiment 1's notes). gates are *values*, recomputed from the input at each step and learnable through their biases and weights; a matrix-times-derivative product is not something a network can hold near the identity. that difference is what gated cells buy.
