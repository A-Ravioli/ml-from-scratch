# finding questions · lab notes

numbers below are from the reference solution on an Apple M-series laptop (CPU, `OMP_NUM_THREADS=1`), real MNIST from the cache. experiment 1's runs take about 0.15 s each and are deterministic on the same machine; experiment 2's throughput is a measurement and changes with the machine and its load; experiment 3 is a simulation with a fixed seed. the lab takes about 8 seconds. offline, experiments 1 and 2 run on a synthetic stand-in and the throughput measurement still works.

## experiment 1: how many seeds does a small question need?

**what people predict.** that 5–10 seeds are plenty for a 0.2-point effect, and that a 3-seed comparison has decent power.

**what happens.** the 8-seed pilot (a $784 \to 128 \to 10$ MLP, 3 epochs of SGD with momentum on 5,000 images) gives 92.96% without and 94.37% with label smoothing: a paired difference of **+1.40 points** (95% CI [+0.86, +1.95]), with a standard deviation of the paired differences of **0.65 points**. from that sd:

| true effect (points) | seeds for 80% power | power with 3 seeds | with 5 |
|---|---|---|---|
| 0.1 | 264 | 0.07 | 0.09 |
| 0.2 | 67 | 0.10 | 0.14 |
| 0.3 | 31 | 0.14 | 0.22 |
| 0.5 | 12 | 0.24 | 0.41 |
| 1.0 | 5 | 0.54 | 0.88 |

detecting a 0.2-point improvement takes **67 paired seeds**; a 3-seed comparison would detect it **10%** of the time, barely above the 5% false-positive rate.

**why.** power depends on $\Delta\sqrt n/\sigma_d$, so the seeds needed scale as $(\sigma_d/\Delta)^2$: each halving of the effect quadruples them (67 for 0.2, 264 for 0.1). most "small trick" papers report effects of a few tenths of a point with 1–5 runs, which is why so many of them don't replicate: their experiments couldn't reliably see the effects they report. "no significant difference" from 3 seeds says almost nothing about a 0.2-point effect.

as for label smoothing itself: the pilot already answers *this* question (an effect of 1.4 points, far above the noise), which is a good outcome for a pilot. but its estimate shouldn't become the `min_effect` of the real experiment. it's an 8-seed estimate (the interval spans 0.9 to 2.0), and if this question had been picked *because* its pilot looked best among several, it would be inflated by the winner's curse of experiment 3. the `min_effect` should be the smallest effect that would matter to someone choosing a loss function, say 0.3 points: 31 seeds.

## experiment 2: what can a laptop afford?

**what people predict.** a week of laptop compute sounds like a lot: guesses for the largest compute-optimal model run from 100M to a billion parameters; GPT-2 small "a few days".

**what happens.** the measured float32 matrix-multiply throughput on one thread was about **1,150 GFLOP/s** (Apple's matrix coprocessor through Accelerate; a laptop without one gets tens of GFLOP/s per core), so 460 GFLOP/s at 40% utilization:

| budget | largest compute-optimal model | tokens |
|---|---|---|
| an hour | 3.7M | 74M |
| a day | 18.2M | 364M |
| a week | **48.2M** | 963M |

GPT-2 small at 20 tokens per parameter (124M parameters, 2.5B tokens) would take about **1,100 hours (7 weeks)**, and Chinchilla (70B parameters, 1.4T tokens) about 40,000 years.

**why.** compute-optimal size grows only as the square root of compute: 168× more time (a week vs an hour) buys 13× more parameters. and these are optimistic numbers: the 6ND rule ignores attention, data loading, evaluation and checkpointing, and a real training loop on a laptop rarely sustains 40% of a matmul benchmark (ch 26), so real runs take longer. the explain prompt's answer: questions that fit this budget are about mechanisms you can see at small scale (training dynamics, optimization, data efficiency, evaluation methodology, interpretability of small models, anything on MNIST/CIFAR/tiny Shakespeare-sized data) and about *trends* across a range of small sizes; questions that need the behaviour of large models (emergent abilities, large-scale pretraining recipes) don't fit, and a proposal that needs them must say how it scales down (topic 02's step 4).

## experiment 3: the most promising pilot

**what people predict.** that the winner's true effect is somewhat smaller than its pilot, maybe half; and that a proper follow-up confirms most winners.

**what happens.** with 20 ideas (true effects sd 0.2 points, pilot noise sd 0.5): the winner's one-seed pilot showed **+1.01 points** on average; its true effect is **+0.14**, and the 20-seed follow-up estimates +0.14. the follow-up's 95% interval lies above 0 in only **35%** of projects, exactly what your `paired_power`, averaged over the winners' true effects, predicts (35%). with the slider at 2 ideas the winner's pilot shows +0.31 against a true +0.04 (follow-up confirms 21%); at 50 ideas, +1.21 against +0.17 (40%).

**why.** the winner was selected for having a large *observed* effect, and with pilot noise (0.5) much larger than the spread of true effects (0.2), a large observation is mostly a large noise draw. the expected true effect given an observation is $\tau^2/(\tau^2 + \sigma^2) = 0.04/0.29 = 0.14$ times the observation (on paper, problem 3): $0.14 \times 1.01 = 0.14$, matching the simulation. the follow-up uses fresh seeds, so its noise is independent of the selection and it lands on the truth. more ideas make the winner's pilot look better (the maximum of more noise draws), while the true effect improves only a little, so the exaggeration grows. for a lab that tries many ideas and publishes the best-looking one, this is the mechanism behind effects that shrink on replication (Button et al., 2013; Sculley et al., 2018). the defence is the protocol of this topic: treat the pilot as a pilot, state the hypothesis with a `min_effect` that matters, and confirm on fresh seeds, sized by power.
