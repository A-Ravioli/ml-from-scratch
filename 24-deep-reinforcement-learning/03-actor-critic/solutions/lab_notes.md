# actor-critic · lab notes

numbers are from the reference solution with the lab's defaults (one CPU thread).

## experiment 1: what λ buys, and what it costs

**what people usually predict:** with a perfect critic λ doesn't matter much; with a bad critic, λ = 1 (no reliance on the critic) wins; somewhere in the middle is a smooth compromise.

**what happens:** (per-state squared bias / variance of the targets, averaged over the 19 states)

| critic | λ = 0 | λ = 1 | best λ (MSE) |
|---|---|---|---|
| perfect | 0.000 / 0.010 | 0.001 / 0.715 | 0 (0.010) |
| shrunk (0.5 V) | 0.058 / 0.012 | 0.001 / 0.715 | 0.75 (0.062) |
| noisy (± 0.3) | 0.012 / 0.026 | 0.001 / 0.715 | 0.5 (0.021) |

at λ = 1 the targets are the same for every critic, unbiased, and enormously noisy: their variance (0.715) is 10 to 70 times the total error of the best λ. the variance climbs slowly up to λ ≈ 0.9 and then explodes. with a perfect critic every λ is unbiased and λ = 0 is best. with the shrunk critic the bias falls as λ grows, and the best compromise is at 0.75; with the noisy critic, at 0.5. no critic makes λ = 1 the best choice on this task.

**why:** at λ = 1 the target is the Monte Carlo return, $\pm1$ depending on which end the walk falls off: the critic's values cancel out of the telescoping sum entirely (on paper 2), so the critic's quality is irrelevant and the variance is that of a coin flip, $1 - V(s)^2$, averaged over states. at λ = 0 the target is $r + V(s')$: one random step's worth of variance (the walk moves to one of two neighbours), but all the critic's error at the neighbours. the shrunk critic is wrong by the same *sign* everywhere (too close to 0), so its one-step targets are systematically biased, and only mixing in many real steps (large λ) washes that out. the noisy critic's errors are independent across states, so the one-step target averages the errors of the two neighbours and partly cancels them: its bias at λ = 0 is smaller than that of the shrunk critic, even though its per-state errors are of a similar size. a systematic critic error is much more harmful than a random one. in deep RL the critic is wrong in both ways, and wrong most of all early in training, which is why practical λ values sit at 0.9–0.97: enough real reward to dilute a bad critic, not so much that the return's variance takes over.

## experiment 2: sweeping λ in A2C

**what people usually predict:** a smooth inverted U in λ, with a clear winner around 0.95, and λ = 0 a bit worse than the rest.

**what happens:** (4 seeds, mean ± standard error)

| λ | final return (last 20 episodes) | per seed | mean return over training |
|---|---|---|---|
| 0 | 69 ± 9 | 81, 88, 63, 46 | 23 ± 3 |
| 0.5 | 304 ± 90 | 138, 478, 440, 161 | 42 ± 2 |
| 0.9 | 286 ± 63 | 462, 187, 290, 203 | 132 ± 8 |
| 0.97 | 315 ± 42 | 354, 242, 413, 250 | 131 ± 11 |
| 1 | 277 ± 73 | 478, 283, 138, 211 | 138 ± 6 |

λ = 0 barely learns in 100,000 steps. λ = 0.5 spends the first 25,000 steps stuck near the random policy's 20, then catches up. 0.9, 0.97 and 1 learn at the same speed from the start; by the end every λ ≥ 0.5 is somewhere between 140 and 480 depending on the seed, and the differences between them are well inside the seed-to-seed spread.

**why:** for most of training the critic is far from right (it starts random and chases a moving policy), which is experiment 1's "bad critic" regime, and with $\gamma = 0.99$ a one-step target has to propagate the value of balancing for 100+ steps one step at a time. with λ = 0 the actor is graded almost entirely by the critic's errors, so it learns slowly; with λ = 0.5 the effective horizon $1/(1 - \gamma\lambda) \approx 2$ steps is still far too short at first, until the critic improves. from λ = 0.9 the advantages contain tens of real rewards, which is enough. λ = 1 isn't clearly worse here, because a 32-step rollout already truncates the Monte Carlo sum (the last steps bootstrap from `last_value`): with short rollouts the variance penalty of experiment 1 is capped. to rank 0.9 against 0.97 you'd need many more seeds: with standard errors of about 10 on the training mean and a difference of 1, even 100 seeds wouldn't settle it. the honest conclusion from four seeds is "λ = 0 and 0.5 learn slower; λ ≥ 0.9 are indistinguishable".
