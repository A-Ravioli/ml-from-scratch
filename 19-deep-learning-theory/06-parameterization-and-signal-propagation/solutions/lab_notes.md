# parameterization & signal propagation · lab notes

numbers below are from the reference solution on a laptop CPU, on the real MNIST (8,192 digits pooled to 14 × 14), with the seeds in the notebook. the lab took 181 s on a heavily shared laptop (load average above 20); expect about a minute on an idle one. offline, the synthetic stand-in shows the same patterns with different losses.

## experiment 1: the edge of chaos

**what people predict.** most people expect a broad optimum ("anything reasonable trains, the optimizer adapts"), or that larger weights are better because they give bigger gradients.

**what happens.** the critical scale for $\sigma_b = 0.3$, from bisection on your $\chi_1$: $\sigma_w = 1.396$.

| $\sigma_w$ | 0.50 | 1.00 | 1.25 | 1.50 | 2.00 | 2.50 | 3.00 |
|---|---|---|---|---|---|---|---|
| $\chi_1$ at $q^*$ | 0.21 | 0.70 | 0.91 | 1.06 | 1.33 | 1.59 | 1.84 |
| correlation after 30 layers (started at 0.6) | 1.000 | 1.000 | 0.983 | 0.781 | 0.325 | 0.043 | 0.159 |

and the 20-layer, width-128 tanh networks after 200 SGD steps at learning rate 0.01 (training loss on 2,048 digits, two seeds; chance is $\ln 10 = 2.30$):

| $\sigma_w$ | 0.6 | 1.0 | 1.15 | 1.25 | **1.4 (edge)** | 1.55 | 1.7 | 2.0 | 3.0 |
|---|---|---|---|---|---|---|---|---|---|
| $\chi_1$ | 0.30 | 0.70 | 0.83 | 0.91 | 1.00 | 1.09 | 1.17 | 1.33 | 1.84 |
| training loss | 2.301, 2.301 | 2.267, 2.290 | 1.248, 1.367 | 0.884, 0.948 | **0.747, 0.803** | 0.804, 0.846 | 1.015, 1.013 | 1.532, 1.538 | 2.245, 2.229 |

the networks train well in a window around the edge, roughly $\chi_1 \approx 0.9$–$1.1$ (losses below 0.95), with the best value at the edge itself; the loss climbs steeply on both sides (1.25–1.37 at $\chi_1 = 0.83$, 1.01 at 1.17) and at $\sigma_w = 0.6$ the network hasn't moved from chance at all. the two seeds agree closely everywhere. in the ordered phase the correlation after 30 layers is 1.000 (every pair of inputs has become indistinguishable); in the chaotic phase it falls to 0.04–0.33 and fluctuates (a single width-500 network, so the deep-layer values are noisy).

**why.** the same $\chi_1$ is the mean squared singular value of each layer's Jacobian. below 1, both the forward differences between inputs and the backward gradients shrink by about $\chi_1$ per layer: after 20 layers at $\sigma_w = 0.6$ the first layers' gradients are smaller by a factor around $0.3^{20}$, so they don't train, and the last layers see a nearly constant input. above 1, gradients grow and the network's output becomes a chaotic function of its input, which is hard to fit. at the edge the depth scale over which information survives diverges, so all 20 layers receive usable signal (Schoenholz et al., 2017). the window has a finite width because a 20-layer network only needs signal to survive 20 layers: the depth scale $\xi \approx -1/\ln\chi_1$ is about 10 layers at $\chi_1 = 0.91$ and 5 at 0.83, so the loss rises as $\xi$ drops below the depth (Schoenholz et al. found trainability up to a few times $\xi$); a deeper network would have a narrower window; with residual connections or normalization, the sensitivity to $\sigma_w$ largely disappears.

## experiment 2: the coordinate check

**what people predict.** that the changes are similar at all widths ("same learning rate, same steps"), or that wider networks change *less* because each weight matters less.

**what happens.** RMS change of each layer's activations after 3 full-batch Adam steps at learning rate $10^{-2}$, mean of 2 seeds:

| width | SP hidden 1 | SP hidden 2 | SP output | μP hidden 1 | μP hidden 2 | μP output |
|---|---|---|---|---|---|---|
| 64 | 0.34 | 0.39 | 0.74 | 0.22 | 0.33 | 0.30 |
| 256 | 0.31 | 0.72 | 2.52 | 0.22 | 0.34 | 0.31 |
| 1,024 | 0.23 | 1.12 | 7.13 | 0.22 | 0.35 | 0.29 |
| 2,048 | 0.21 | 1.97 | 20.4 | 0.22 | 0.34 | 0.29 |

under SP the output's change grows 27-fold from width 64 to 2,048 and the second hidden layer's 5-fold; the first layer's change even shrinks slightly. under μP all three are flat to within a few percent across a 32-fold range of widths.

**why.** Adam moves each weight by about the learning rate per step, and those moves are aligned with the activations they multiply, so a unit with fan-in $n$ sees an input change of order $\eta n$ (on paper 4). in SP this grows with width in every layer whose fan-in grows: the second hidden layer and, most strongly, the readout (whose change is then amplified by the preceding layer's changes too). the first layer's fan-in is fixed at 196, so it doesn't grow (it even shrinks a little, which the counting argument doesn't explain; one plausible cause is that the wider network's fast-moving readout reduces the loss, and so the gradient reaching the first layer, within the three steps). μP divides the hidden learning rate by the width multiplier and scales the readout's input by $1/m$, which exactly cancels the $n$ in each case. flat lines are the operational definition of a correct μP implementation (Yang et al., 2022).

## experiment 3: μTransfer

**what people predict.** under SP, many people expect the best learning rate to be roughly width-independent (it's Adam, after all); under μP, that it will be exactly constant.

**what happens.** best Adam learning rate (grid $2^{-12}, \dots, 2^{-4}$ in factors of 4), 200 steps, mean of 2 seeds:

| parameterization | width | best learning rate | its training loss | loss at width 64's best rate |
|---|---|---|---|---|
| SP | 64 | $2^{-6}$ | 0.101 | 0.101 |
| SP | 256 | $2^{-8}$ | 0.071 | 0.162 |
| SP | 1,024 | $2^{-10}$ | 0.051 | 0.245 |
| μP | 64 | $2^{-6}$ | 0.111 | 0.111 |
| μP | 256 | $2^{-6}$ | 0.093 | 0.093 |
| μP | 1,024 | $2^{-6}$ | 0.077 | 0.077 |

under SP the optimum moves left by a factor of 4 for every 4× in width, and using width 64's best rate at width 1,024 gives a loss almost five times worse than its own optimum. under μP the optimum stays at $2^{-6}$ at all three widths and the loss curves nearly coincide, with wider always better at every learning rate. two honest caveats: the grid has factor-4 resolution, so "the same optimum" means within a factor of 2 either way (an earlier version of this experiment, with 300 steps and a different readout scaling, found the μP optimum moving from $2^{-8}$ to $2^{-6}$ between widths 64 and 1,024); and at width 1,024 the best SP run (0.051, at its own tuned rate) beats the best μP run (0.077). μP saved the sweep; it didn't find a better model in 200 steps.

**why.** the coordinate check explains the SP shift: at a fixed rate, wider SP networks take larger steps in function space, so the stable, efficient rate shrinks as the width grows (roughly $\propto 1/n$ here, as the counting argument predicts for the hidden and readout layers). μP makes the per-step change of every layer width-independent, so the whole training trajectory, and therefore the best learning rate, converges as $n$ grows. SP's lower loss at its tuned rate is a finite-budget effect: SP's random readout and its larger effective readout learning rate make faster early progress; μP's zero readout starts from scratch. Yang et al.'s evidence for μTransfer at scale is that transferred hyperparameters match or beat directly tuned SP ones after full training of large models, which this experiment is far too small to test.
