# continuous control · lab notes

numbers are from the reference solution with the lab's defaults (one CPU thread).

## experiment 1: what the temperature does to a policy

**what people usually predict:** α = 0 picks one peak; a moderate α gives a policy that covers both peaks; a large α gives a broad blob centred between them.

**what happens:**

| α | mean reward per action | fraction of actions > 0 | entropy estimate |
|---|---|---|---|
| 0 | 1.000 | 1.00 | −5.84 |
| 0.05 | 0.974 | 1.00 | −2.29 |
| 0.2 | 0.303 | 0.49 | 0.63 |
| 1 | 0.285 | 0.50 | 0.68 |

with α = 0 the policy collapses onto the right-hand peak (a spike of width about 0.01; a negative differential entropy just means a narrow density). α = 0.05 still sits on that peak, a little wider, and gives up 3% of the reward. from α = 0.2 on, the policy is essentially **uniform** on $(-1, 1)$: its entropy (0.63–0.68) is close to the maximum possible, $\log2 \approx 0.69$, and it earns only 0.3 per action, because most of its mass is in the valley and at the edges. no setting covers the two peaks and avoids the valley.

**why:** with α = 0 the actor maximizes $Q$; the two peaks are equally good, and which one wins is decided by the random initialization and the warm-up data (a symmetric tie broken by noise). as α grows, the objective trades reward against entropy. the ideal max-entropy policy is the Boltzmann distribution $\pi \propto e^{Q/\alpha}$, which for α = 0.2 is bimodal: about 150 times denser on the peaks than in the valley. but a squashed Gaussian is unimodal in $u$; the only way it can put mass on both peaks is to become very wide, and a wide Gaussian pushed through tanh is close to uniform on $(-1, 1)$. so the actor faces a choice between "one peak, low entropy" and "nearly uniform, high entropy", and somewhere between α = 0.05 and α = 0.2 the entropy bonus starts paying more than the reward lost. the lesson: the policy class limits what maximum entropy can express. mixtures of Gaussians, normalizing flows (ch 21), or energy-based policies sampled with Langevin dynamics or diffusion (ch 22) can represent the bimodal Boltzmann policy; SAC's actor is a deliberate compromise for speed.

## experiment 2: the temperature on the pendulum

**what people usually predict:** α = 0 fails for lack of exploration; something around 0.2 works best; the automatic temperature beats every fixed one.

**what happens:** (return over the last 5 of 30 episodes, two seeds)

| α | seed 0 | seed 1 | α at the end |
|---|---|---|---|
| 0 | −194 | −161 | 0 |
| 0.2 | −183 | −161 | 0.2 |
| 1 | −189 | −188 | 1 |
| 5 | −511 | −314 | 5 |
| auto | −192 | −162 | 0.191 |

everything from α = 0 to α = 1, and the automatic temperature, swings the pendulum up and balances it within about 15 episodes (3,000 steps) and ends near −150 to −190, which is about as good as this task gets. only α = 5 fails to settle: its returns climb but stay noisy around −300 to −600. the automatic temperature starts at 1 and falls steadily; after 6,000 steps it is at 0.19 and still falling.

**why:** the pendulum is a forgiving task for exploration. the 1,000 random warm-up steps already cover most of the state and action space; the dynamics are deterministic and smooth; the reward is dense (every step says how far from upright you are); and the actor's own sampling noise is enough, even when α = 0 lets the policy's standard deviation shrink. so the entropy bonus isn't needed here, and it barely hurts as long as it's small compared with the rewards. at α = 5 it isn't: the entropy term $-\alpha\log\pi$ is worth several reward units per step, comparable to the cost of hanging slightly off-vertical, so the policy is paid to keep thrashing around and can't hold the pendulum still at the top.

the automatic temperature keeps the policy's entropy near the target ($-1$ nat for one action dimension), and α is whatever multiplier achieves that: large early, when the policy would otherwise collapse onto a poor solution, and smaller as the policy's precision starts to pay. its advantage isn't that it beats a well-chosen fixed α on one task (here it doesn't), it's that "−1 nat per action dimension" transfers across tasks with different reward scales, while a fixed α of 0.2 that's harmless here is 5 when the rewards are 25 times smaller. two seeds per setting is also only enough to see the large effect (α = 5); the differences among the others are noise.
