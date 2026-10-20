# MAML · lab notes

numbers measured with the reference solution: the 1–40–40–1 ReLU network, 10-shot sine tasks, meta-batches of 25, Adam at $10^{-3}$ for 3,000 meta-iterations, inner learning rate 0.01, scored on 1,000 fresh test tasks with 100 query points each. uncertainties are standard deviations over meta-training seeds 0–2. times are wall-clock seconds on one CPU thread of an idle laptop (read them as ratios: under load they double); the whole lab took 71 seconds. the data is synthetic, so `verify.py` runs exactly this lab.

## experiment 1: what does meta-training buy?

**what people usually predict:** that MAML's init is better than pretraining's from the start, and that both improve with steps, MAML faster.

**what happens:**

| init | k = 0 | k = 1 | k = 10 | meta-training time |
|---|---|---|---|---|
| MAML (1 step) | 2.966 ± 0.024 | 0.710 ± 0.097 | 0.284 ± 0.046 | 5.2 s |
| pretraining | 2.955 ± 0.018 | 2.599 ± 0.106 | 1.819 ± 0.068 | 1.5 s |

before adaptation the two inits are indistinguishable, and both sit at the loss of the best single function for all tasks, about 3.0 (on paper 4: $-1.62\cos x$, against 4.25 for predicting zero). one gradient step on ten points then cuts MAML's error by a factor of four and pretraining's by 12%; after ten steps MAML is at 0.28 and pretraining at 1.82, six times worse. MAML keeps improving well past the single step it was trained with.

**why:** the loss at $\theta$ and the effect of a step from $\theta$ are different properties. pretraining minimizes the first, and its minimizer is the flat compromise; from there, the support gradient of a task points in a direction that changes the function only slightly per step at $\alpha = 0.01$. MAML minimizes the loss *after* the step, so it picks, among roughly equally good compromises, weights whose gradient for a task's ten points moves the function towards that task's wave by the right amount. dragging the slider shows MAML's $k = 0$ curve is also a low-amplitude wave; the information is in the gradient field around it, not in the curve. later steps keep helping because they keep moving in the right directions; the init was shaped for one step of size 0.01, and nothing about that stops a tenth step from helping. MAML costs about 3.5× pretraining's time: a second backward pass through the inner step, plus the inner step itself.

## experiment 2: how many inner steps should training use?

**what people usually predict:** that training with 3 inner steps is better at every test-time $k$, and costs about 3× as much.

**what happens:**

| training | k = 1 | k = 5 | k = 10 | meta-training time |
|---|---|---|---|---|
| MAML, 1 step (exp. 1) | 0.710 ± 0.097 | 0.342 ± 0.063 | 0.284 ± 0.046 | 5.2 s |
| MAML, 3 steps | 0.728 ± 0.056 | 0.213 ± 0.039 | 0.163 ± 0.031 | 10.7 s |
| first-order MAML, 1 step | 0.905 ± 0.055 | 0.438 ± 0.056 | 0.358 ± 0.045 | 2.6 s |

at $k = 1$ the two inits are level (0.73 vs 0.71, well inside the spread). from $k = 5$ on, the 3-step init is ahead by more than the seed-to-seed spread: 0.21 vs 0.34 at $k = 5$, 0.16 vs 0.28 at $k = 10$. it costs 2–3× the meta-training time (machine- and load-dependent). more is not always better, though: an earlier version of this lab trained with **5** inner steps (same seeds, 3,000 iterations) and found that init clearly *worse* at $k = 1$ (1.08 ± 0.31), tied at $k = 5$ (0.33 ± 0.12), and ahead at $k = 10$ only within its own spread (0.24 ± 0.09), for 2.5× the time. so the answer to "how many inner steps does adaptation need" here is: a few during training (3 beat 1 and 5 at this budget), and as many at test time as you can afford, since test-time steps are cheap and help every init. first-order MAML loses about a quarter of 1-step MAML's accuracy at every $k$ and is still five times better than pretraining at $k = 10$; it runs in half the time.

**why:** an init trained for $k$ steps is only ever judged after $k$; it gets better at $k$ and nearby, and nothing rewards it for being good after fewer. with 3 steps that freedom costs nothing at $k = 1$ here; with 5 the optimizer uses it (it can place $\theta$ where the first step overshoots, as long as the next four correct it), and five unrolled second-order steps are a deeper, noisier graph to differentiate through, which shows in the larger seed spread and in Antoniou et al.'s list of MAML instabilities. the time grows less than linearly in the number of steps because the query pass, the optimizer step and Python overhead don't scale with $k$. first-order MAML still works because most of its signal is the query gradient at the adapted weights, which already says "from here, move this way for this task"; what it drops is the correction for how moving $\theta$ changes the adaptation itself. for ReLU networks part of that Hessian term vanishes, but not all of it (warm-up 4), which is the quarter you lose.

## experiment 3: adapting from half a wave

**what people usually predict:** that predictions on the unseen half will be poor for both, perhaps flat for pretraining and wild for MAML.

**what happens** (5 support points on $[-5, 0]$, scored on $[0, 5]$ after 10 steps):

| init | query MSE on [0, 5] |
|---|---|
| predict 0 everywhere | 4.330 |
| MAML | 1.603 ± 0.839 |
| pretraining | 4.308 ± 0.407 |

MAML's predictions on the half it never saw are a third of the zero baseline's error; pretraining is no better than predicting zero. the plots show MAML continuing the wave into the grey region, with the right period and roughly the right phase, while pretraining bends towards the points it saw and then relaxes back to its compromise curve. the spread over seeds is large (± 0.84), which is what 3,000 meta-iterations buy; the plots show seed 0 only.

**why:** a sine task has two degrees of freedom. MAML's init, together with a few gradient steps, behaves like a two-parameter family: the support points select amplitude and phase, and the network's shape outside the data comes from the init, which has learned what every member of the family looks like. Finn et al. show the same figure (their figure 2). pretraining's init knows only the average wave; the gradient steps fit the five points locally and nothing propagates the fit to $x > 0$, so more steps can't help there. the result is the clearest sign that MAML learned a prior over functions, which is how Grant et al. (2018) formalize it.
