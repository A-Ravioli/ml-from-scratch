# offline RL & imitation · lab notes

numbers are from the reference solution with the lab's defaults (one CPU thread).

## experiment 1: errors compound

**what people usually predict:** a clone of one 500-step episode that fits its training data well balances for a good fraction of 500 steps; more data helps gradually; DAgger helps a little.

**what happens:** (5 seeds × 20 test episodes, sampled actions)

| policy | mean return | median | fraction reaching 500 |
|---|---|---|---|
| BC, 1 episode | 135 | 112 | 0.01 |
| BC, 2 episodes | 493 | 500 | 0.94 |
| BC, 3 episodes | 500 | 500 | 1.00 |
| DAgger (1 expert + 3 learner episodes) | 500 | 500 | 1.00 |

the one-episode clone survives the first 70 steps every time and then falls steadily: half its episodes are over by step 110, nearly all by 300. a second demonstration changes everything, and three are enough for a perfect clone. DAgger, given a single expert episode, reaches the expert's level after three iterations.

**why:** one expert episode is a thin tube through state space. the expert keeps the pole nearly upright and the cart near the middle, so its 500 states cover a tiny region; the clone matches the expert inside it, but its sampled actions sometimes differ, the cart drifts a little further, and soon it is in states (cart off-centre, moving) where the training data says nothing and its extrapolation is wrong. each mistake makes the next more likely, which is why failures keep accumulating over time rather than all happening at once: the survival curve is the picture of compounding error. two or three episodes start from different random states and contain the expert's corrections from a wider spread of situations, which on a task this smooth is enough to cover where the clone goes. DAgger gets the same coverage more cheaply in *expert* effort: its learner visits exactly the off-track states it would fail in, and the expert labels those (it labelled up to 1,500 extra states but demonstrated only one episode). on a task where states that matter are rare under the expert (think of a car drifting towards the kerb), more demonstrations of the same expert help far less, and DAgger's advantage grows.

## experiment 2: offline RL from a noisy controller

**what people usually predict:** offline DQN does reasonably (the data is good and contains both actions), CQL a bit better, behavior cloning of noisy data worst.

**what happens:** (2 seeds per learner, 10 greedy episodes each: enough to see a collapse, not to rank close results)

| learner | mean return (greedy) | largest predicted Q on the data |
|---|---|---|
| behaviour policy (the data) | 475 | — |
| offline double DQN | 9 | $5.6\times10^5$, $3.4\times10^4$ |
| CQL (α = 1) | 500 | 72, 71 |
| behavior cloning of the data (greedy) | 500 | — |

offline DQN collapses completely: its greedy policy falls immediately, and its Q-values reach hundreds of thousands on a task whose values can't exceed 100. CQL learns a policy that balances every time, with values around 72, *below* the true value of balancing forever. greedy cloning of the noisy data also balances every time.

**why:** the data is narrow even though it contains both actions: it comes from a controller that keeps the pole near upright, so the states where the pole is falling, and especially the "wrong" action in states the controller handles confidently, are rare or absent. the target's max queries $Q(s', a')$ for every action at every next state, including those rarely covered pairs; their values are whatever the network extrapolates, the max picks the most optimistic, and bootstrapping copies that optimism into everything upstream. online, the agent would act on an inflated value, fail, and get corrected; offline, nothing ever corrects it, so the errors grow without bound (the deadly triad again, now with no fresh data at all). the greedy policy then follows the inflated, unsupported actions and fails.

CQL's penalty pushes down whatever has a high soft-max value and pushes the dataset's actions back up, so unsupported actions can't win the max, and the values stay on the scale the data supports. its estimates (about 72) are conservative, as the theory promises: a lower bound on the policy's true value (about 100). behavior cloning works here because the noise is unbiased: in each state the controller's action is still the most common one in the data, so the clone's argmax recovers the controller, and greedy action selection throws the 30% noise away. cloning would fail on data whose *most common* action is bad (a mix of a few good and many poor demonstrators, or noise that's correlated with the state), and it can never improve on the best behaviour in the data; offline RL, with rewards, can stitch good fragments of different trajectories together. that's when CQL earns its complexity.
