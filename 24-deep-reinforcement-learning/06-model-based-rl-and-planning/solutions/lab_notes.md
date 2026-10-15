# model-based RL & planning · lab notes

numbers are from the reference solution with the lab's defaults (one CPU thread).

## experiment 1: the Dyna maze

**what people usually predict:** planning helps a bit, maybe twice as fast; some expect it to shorten the first episode too.

**what happens:** (mean of 10 seeds)

| planning steps | episode 1 | episode 2 | episode 5 | first episode under 20 steps |
|---|---|---|---|---|
| 0 | 1196 | 768 | 226 | 27 |
| 5 | 437 | 145 | 22 | 6 |
| 50 | 936 | 42 | 16 | 3 |

with 50 planning steps per real step the agent walks a near-shortest path (14 steps) from the third episode on; Q-learning needs about 27 episodes. this matches Sutton & Barto's Figure 8.2. the first-episode lengths differ a lot too, but that is noise: every Q-value is 0 until the goal is found, so all three agents perform the same kind of random walk, and the length of a random walk to a far corner of a maze varies by hundreds of steps from seed to seed (planning only changes which random numbers each agent draws).

**why:** with a reward only at the goal, Q-learning learns one more step of the path per episode: the episode that first reaches the goal updates only the last state–action pair, the next one the pair before it, and so on (on paper 4). Dyna-Q's model remembers every transition it has seen, so after the goal is found once, the planning updates replay the whole explored part of the maze and push the goal's value back along all the stored transitions within a few hundred updates. the price is computation: 50 updates per real step is 50 times the arithmetic, which is cheap when real experience is expensive (a robot, a user) and pointless when it isn't (a fast simulator). the model also has to be right: the maze is deterministic, so the last observed outcome is the true outcome.

## experiment 2: planning budget versus strength

**what people usually predict:** a steady, roughly linear gain with budget; the equal-budget match exactly 50%.

**what happens:** (16 games per budget against a 100-simulation opponent)

| our simulations | wins / draws / losses | win rate [95% CI] |
|---|---|---|
| 10 | 0 / 0 / 16 | 0.00 [0.00, 0.19] |
| 30 | 2 / 0 / 14 | 0.12 [0.03, 0.36] |
| 100 | 10 / 1 / 5 | 0.62 [0.39, 0.82] |
| 400 | 14 / 0 / 2 | 0.88 [0.64, 0.97] |

strength rises steeply with the budget: 10 simulations never win, 30 rarely, equal budgets about half the time, and 4× the opponent's budget wins seven games in eight. with 16 games per point that is all these numbers can say; the shape of the curve is not pinned down.

**why:** each extra simulation refines the statistics of the most promising moves, and in a tactical game like Connect Four a deeper, better-informed search spots threats and wins that a shallow one misses; the gain per simulation shrinks (the next simulation mostly re-confirms what the tree already knows); in computer chess and Go, strength is usually found to grow roughly per *doubling* of search effort, and these four points are consistent with that, but far too few games to show it. the equal-budget match is symmetric in expectation (first-move advantage is balanced by alternating colours), so 0.62 is a 50% match plus noise: its interval runs from 0.39 to 0.82. with 16 games an interval is about ±0.2 wide, enough to separate 10 from 400 simulations but not 100 from 200. telling 55% from 50% needs the standard error of the win rate to be about 0.025 or less, which takes around 400 games: that's why engine developers run tens of thousands of games per change.

## experiment 3: AlphaZero-lite

**what people usually predict:** that a learned network quickly beats random rollouts, or at least that more self-play steadily helps.

**what happens:** (20 games per entry, score with draws counted half; standard error about 0.1 per entry)

| training seed | untrained network | after 25 iterations (about 30 s) |
|---|---|---|
| 0 | 0.15 | 0.35 |
| 1 | 0.17 | 0.40 |
| 2 | 0.35 | 0.38 |
| mean | 0.22 | 0.38 |

in the lab's three seeds the trained networks score a little better than the untrained ones, but never reach 0.5: a search guided by AlphaZero-lite still loses to plain rollouts. a larger check with the reference solution (5 training seeds, 40 iterations, 40 games per entry) shows how weak the effect is: untrained networks scored between 0.11 and 0.53 (mean 0.30), trained ones between 0.25 and 0.49 (mean 0.34), so the seed-to-seed spread is larger than the gain. and longer training doesn't help: seeds 0 and 1 trained for 160 iterations (about four minutes) scored 0.40 and 0.34. the training loss keeps falling throughout. at this scale, AlphaZero-lite plateaus below rollouts.

**why:** an untrained network returns near-uniform priors and values near 0 for every position, so the search knows nothing beyond the positions it can reach in its tree; how badly that goes depends on the random initialization (a network whose arbitrary value outputs happen to correlate with good positions does better), which is the large untrained spread. a random rollout, crude as it is, plays to the end of the game and brings back real information. self-play training improves the network only as fast as its targets improve: the visit distributions and game outcomes come from a weak search playing itself, with 50 simulations, a two-layer MLP, sampling at temperature 1 for the whole game, and 10 games per iteration. the network fits those targets (the loss falls), but they are a poor teacher, and the bootstrap (better net, better search, better targets) never gets going. that's the honest lesson: AlphaZero's loop works at scale, and its ingredients are not optional. what to change first, and measure over several seeds: many more simulations per self-play move (better targets), a temperature schedule (explore for the first few moves, then play the best move, so outcomes reflect good play), Dirichlet noise at the root, left–right symmetry augmentation, a convolutional network that shares weights across the board, and an arena that only keeps a new network if it beats the old one. the capstone's stretch goal asks for exactly this.
