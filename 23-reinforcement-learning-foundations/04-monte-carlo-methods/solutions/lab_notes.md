# monte carlo methods · lab notes

## the exact answer

the DP cell agrees with your environment: the optimal policy is worth **−0.0466** per game exactly, and 100,000 games of it in `Blackjack` average **−0.0504 ± 0.0030**, 1.3 standard errors away. (the exact figure is for this rule set: no doubling, splitting or bonus for naturals, so the house edge is larger than in a casino's basic-strategy tables.)

## experiment 1: Monte Carlo control on blackjack

**what people usually predict:** after half a million games, the learned policy matches the optimal one almost everywhere, or else that it's clearly worse; the training return should be about the policy's value.

**what happens:** the learned greedy policy differs from the optimal one in **15 of the 200** decision states, yet its exact value is **−0.0510**, only 0.0044 per game worse than optimal (**−0.0466**). "stick on 20 or 21" is worth **−0.354**. the average return during training is **−0.0885**, much worse than the learned greedy policy. the maps look like Sutton & Barto's Figure 5.2: without a usable ace, stick from 13 against a dealer's 2–6 and from 17 against 7–ace; with a usable ace, hit up to soft 18 against 9, 10 or an ace.

**why:** the disagreements come in three kinds, and the gap table plus a count of how often each action was tried in each state (not shown in the lab; easy to add) separates them.

- **the ε-soft optimum really prefers the other action.** in two states (12 against a 3, and 16 against a 10, both without a usable ace) MC control isn't wrong at all. it learns the values of the ε-greedy policy it follows, and when every *later* decision is random 10% of the time, hitting is worth slightly less: exactly, the ε-soft values favour sticking by 0.006 and 0.011, while the greedy optimum favours hitting by 0.019 and 0.006. this is idea 3's "converges to the best ε-soft policy" made visible.
- **near ties.** in three states (12 against a 2, 16 against a 9, soft 18 against a 10) the two exact values differ by 0.03–0.04, and both actions were tried at least 220 times. with returns of standard deviation near 1, the sign of a difference that small is still uncertain, and a wrong choice costs almost nothing.
- **exploration starvation.** in the other ten (soft 12–16 against a 2, 4 or 5, and soft 18 against an ace, a 3, 5, 6 or 9, with gaps of 0.04–0.32) one action became greedy early, perhaps on a few unlucky returns, and from then on the other action is tried only when ε picks it: half of 10% of the visits. these states come up only a few hundred times in 500,000 games (214–782 visits), so the neglected action has just 15–104 samples, collected while the policy *after* it was still poor: a sample average of a non-stationary target lags. the error is never corrected because the evidence that would correct it arrives at a trickle.

all three cost little, because the states are either nearly tied or rarely reached; that's the trade-off sample-based control makes. the training return is lower than the learned greedy policy's value because the behaviour is ε-greedy: on 10% of decisions it picks a random action, including hitting on 20.

the shape of the policy: a usable ace is insurance, since a hit can't bust you (the ace can drop to 1), so you can afford to hit more. against a dealer's 2–6 the dealer is likely to bust, so you stick on low totals and let them.

## experiment 2: ordinary versus weighted importance sampling

**what people usually predict:** weighted IS is better throughout, by a constant factor.

**what happens:** with 1–3 episodes ordinary IS has a mean squared error of **2.2–3.9**; weighted IS **0.2–0.35**, ten times lower. by 30 episodes they're equal (**0.33** versus **0.38**), and from 100 episodes on they're indistinguishable (**0.009** each at 1,000), falling as $1/n$ (slope −1 on the log–log plot). weighted IS's error first *rises* from 1 to 10 episodes.

**why:** with a uniform behaviour policy and a target that is deterministic from state 13 onwards, an episode's ratio is $2^k$ if the behaviour happened to follow the target for its $k$ remaining decisions, else 0. so ordinary IS averages values like $0, \pm2, \pm4, \pm8$ to estimate $-0.277$: its few-sample estimates are wild. weighted IS divides by the sum of the weights, so its estimate is always a weighted average of actual returns in $[-1, 1]$: bounded error from the start. it's biased, though, and its error at tiny $n$ is dominated by the cases with no usable episode (estimate 0, error 0.077) or one (estimate ±1); as more runs collect a sample or two, the error first rises. once there are many matching episodes the bias vanishes and both behave like ordinary averages with similar variance (here, because the episodes are short and the ratios bounded). in long-horizon problems the ratios aren't bounded and the gap persists; that's experiment 3.

## experiment 3: right on average, and useless

**what people usually predict:** that an estimator whose terms all have the right expectation scatters symmetrically around the true value, so the median is about 1.

**what happens:** over 200 runs of 1,000 episodes, ordinary IS has mean **0.756**, median **0.519**, and **92%** of these 200 runs fall below the true value 1 (a 20,000-run check puts the fraction near 87%: either way, the large majority); one run reports **19.8**, and the sample standard deviation is 1.5. weighted IS gives exactly **1.000** in every run.

**why:** the estimate is an average of $\rho G$, which is $2^k$ for an episode of $k$ lefts ending in the reward and 0 otherwise. long all-left episodes are exponentially rare under the behaviour policy but carry exponentially large weights, and their contribution to the mean never dies out: the second moment is infinite (on paper 3). so the expectation of 1 is made up of a typical value well below 1 plus rare enormous runs; in most samples of 1,000 episodes the rare event simply hasn't happened, and when it does the estimate jumps far above 1. the sample mean still converges (the mean is finite) but with no $1/\sqrt n$ rate and no meaningful standard error: Sutton & Barto's Figure 5.4 shows ten runs still far from 1 after 10 million episodes. weighted IS is exact here only because every episode with a nonzero weight has the same return; with varying returns it would be biased and noisy, but bounded. this is the variance problem that makes long-horizon off-policy learning hard, and the reason for per-decision importance sampling, truncated ratios (Retrace, V-trace) and PPO's clipped ratio in ch 24.
