# policy gradients · lab notes

numbers are from the reference solution with the lab's defaults (one CPU thread). yours will differ in detail.

## experiment 1: how noisy is one episode's gradient?

**what people usually predict:** a modest positive cosine (0.2–0.5) for single episodes, improving somewhat with the baseline, and a variance reduction of "maybe 2×".

**what happens:** the frozen policy balances for 75 steps on average; its network has 226 parameters.

| estimator | mean cosine with the true gradient | total variance | variance / ‖true gradient‖² |
|---|---|---|---|
| total return | −0.161 | $5.7\times10^5$ | 21 |
| reward-to-go | 0.045 | $2.0\times10^5$ | 8 |
| reward-to-go − V | 0.817 | $4.5\times10^4$ | 2 |

the histogram is the striking part: the cosines are not spread around a small positive value but pile up at $+1$ and $-1$. without a baseline, roughly half the episodes point almost exactly *along* the true gradient and half almost exactly *against* it, so the mean cosine is near zero (negative, for total return). with the value baseline most episodes point the right way. reward-to-go cuts the variance by 2.8×, and the baseline by another 4.5× (12× overall).

**why:** CartPole's per-episode gradients live almost on a line. in the reference run the top singular direction of the 300 single-episode estimates carries 84% of their energy, and it sits in the output layer: the policy's overall preference for "left" versus "right" (and the hidden features that say which way the pole leans). an episode's estimate is (weight) × (that episode's score), and the score says "make the actions I happened to take more likely": an episode that pushed left a bit more than the policy expects has a score pointing towards "more left".

without a baseline every weight $G_t$ is positive, so every episode votes for its own actions. the sign of its vote therefore depends only on which way its random actions happened to lean, not on whether they were good. the true gradient is the small *difference* between long episodes' votes (whose leaning was good) and short episodes' votes (whose leaning was bad). and short episodes are the majority: in the reference run the episodes pointing against the true gradient lasted 48 steps on average and those pointing along it 94. the estimator is still unbiased, because the long episodes carry larger weights, but an unweighted average of cosines comes out negative. with $G_t - V(s_t)$ the weight is negative exactly when the episode did worse than usual, so a short episode votes *against* its own leaning, and nearly every episode's vote agrees in sign with the truth.

the last column is the number of episodes to average before the noise drops to the size of the signal (variance of the mean $\approx$ variance$/N$): about 21 without anything, 8 with reward-to-go, 2 with the baseline. that's why policy-gradient methods average over many episodes per update, and why a better baseline (topic 03's critic, with GAE) buys as much as a much bigger batch.

## experiment 2: does the baseline speed up learning?

**what people usually predict:** a big improvement everywhere, because experiment 1 showed a 4.5× variance reduction.

**what happens:** (5 seeds, mean ± standard error)

| variant | episodes 100–200 | last 100 episodes | mean over all 600 |
|---|---|---|---|
| no baseline | 41 | 246 ± 15 | 101 ± 7 |
| value baseline | 40 | 355 ± 9 | 142 ± 11 |

the two curves are indistinguishable for about 250 episodes; then the baseline variant pulls away, and after 600 episodes it balances about 45% longer. neither has solved the task (500) yet at this learning rate.

**why:** a baseline only helps once it predicts returns. the value network starts at random and is trained on returns produced by a policy that is itself changing, so for the first couple of hundred episodes $V$ is a poor guess and $G_t - V(s_t)$ isn't much better than $G_t$. experiment 1's 4.5× reduction used a value network fitted to a *frozen* policy for 300 Adam steps on 300 episodes; during training the critic always lags. the gap also grows with episode length: longer episodes have larger, more variable returns, so the variance problem (and the baseline's benefit) gets worse as the agent gets better. to learn faster you'd raise the learning rate (lr $10^{-2}$ solves CartPole within 600 episodes in most seeds, at the cost of runs that sometimes collapse), use larger batches (fewer updates per episode collected), or replace the Monte Carlo return by a bootstrapped advantage (topic 03), trading variance for bias.
