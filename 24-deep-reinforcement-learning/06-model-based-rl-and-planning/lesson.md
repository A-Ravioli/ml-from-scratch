# model-based RL & planning

## why this matters

every agent so far has been **model-free**: it learned values or policies directly from experience and threw the transitions away (or kept them in a buffer to replay as they were). but experience says more than "this was good": it says how the world works. an agent that learns a **model**, a prediction of the next state and reward, can then improve its policy by *imagining* experience instead of collecting it. that's **planning**, and it buys sample efficiency: Dyna-Q (Sutton, 1990), the simplest version, learns a maze in a fraction of the real steps Q-learning needs.

when the model is *given* (the rules of a game), planning becomes **search**, and search plus learning is how AlphaGo, AlphaZero and MuZero beat the best humans at Go, chess and shogi (Silver et al., 2016, 2017, 2018). their search is **Monte Carlo tree search**: grow a tree of possible futures from the current position, spend more simulations on moves that look promising, and read the move to play off the visit counts. AlphaZero adds a network that tells the search which moves to consider (a policy prior) and how good a position is (a value), and trains that network on the search's own output: search improves the network, the network improves the search. in this topic you build Dyna-Q, Connect Four, PUCT search, and the pieces of AlphaZero-lite; in the lab you'll measure what more planning buys, in a maze and on a Connect Four board.

## warm-up: predict first

1. Dyna-Q does 50 planning updates per real step. the environment is deterministic. how much faster than Q-learning, in real steps, do you expect it to learn a maze?
2. a learned model is slightly wrong. what happens to an agent that plans with it for 100 steps into the future?
3. in a two-player zero-sum game, your opponent's best move is worth $+0.7$ *to your opponent*. what is it worth to you?
4. MCTS has 200 simulations. one move has been visited 150 times, another 50. which do you play, the one with more visits or the one with the higher average value?
5. AlphaZero trains its policy network to imitate MCTS's visit counts, not to maximize reward directly. why would imitating its own search make it stronger?

## prerequisites check

- [ ] you know tabular Q-learning, `epsilon_greedy` and the `Env` protocol (→ ch 23 · topics 04–05)
- [ ] you know UCB and the exploration–exploitation trade-off (→ ch 23 · topic 01)
- [ ] you know value iteration as repeated Bellman backups on a model (→ ch 23 · topic 03)
- [ ] you can train a network with a cross-entropy and an MSE head (→ ch 14–15)

## core ideas

### idea 1: Dyna: learn a model, plan with it

**Dyna** (Sutton, 1990) interleaves three things at every real step: *act* in the environment, *learn* from the real transition (a Q-learning update), and *model-learn*: record what happened. then it *plans*: $n$ times, pick a previously seen state–action pair, ask the model what happens, and do a Q-learning update on the imagined transition. for a deterministic environment the tabular model is just a table: $(s, a) \mapsto (r, s', \text{terminated})$, the last outcome observed.

planning updates are the same Bellman backups as value iteration, applied to the part of the world the agent has seen, at no cost in real experience. in a maze, Q-learning propagates the goal's value back one state per *episode*; Dyna-Q propagates it all the way back in the planning steps after the *first* time the goal is reached (warm-up 1). with $n = 50$, Sutton & Barto (§8.2) report a near-optimal maze policy after 3 episodes instead of about 25.

two cautions. the model must store `terminated`, not `done`: an imagined transition into a time-limit cutoff must bootstrap like a real one. and the model is only as good as the experience behind it: if the world changes, or the model is wrong, planning confidently propagates the error (Dyna-Q+ adds an exploration bonus for pairs not tried in a long time). with learned *neural* models the problem is worse, because errors compound over a multi-step imagined rollout (warm-up 2); modern model-based agents (Janner et al.'s MBPO, Hafner et al.'s Dreamer world models) keep imagined rollouts short or plan in a learned latent space, and use ensembles to know where the model is unsure.

> **spec.** `dyna_q(env, n_episodes, *, n_planning=10, alpha=0.1, gamma=0.95, epsilon=0.1, rng=None)` on a Discrete-observation, Discrete-action env: $Q$ zeros; the first episode resets with `env.reset(seed=int(rng.integers(2**31)))` and actions come from `epsilon_greedy(Q[s], epsilon, rng=rng)`, exactly as in ch 23's `q_learning` (with `n_planning=0` the two must agree exactly); after each real Q-learning update, store `model[(s, a)] = (r, s', terminated)`, then do `n_planning` updates, each on a uniformly random previously seen pair (drawn from `rng`) with the stored outcome (target $r$ if terminated, else $r + \gamma\max_{a'}Q(s', a')$). returns `(Q, episode_returns)`.

> **🔮 predict:** a deterministic chain $0 \to 1 \to 2 \to 3 \to 4$ (goal, $+1$), $\gamma = 0.9$, α = 1. after one real episode, Q-learning has learned the value of only the last step. what does Dyna-Q with hundreds of planning updates per step know? (the worked example has it.)

> **▶ build: stage 1.** implement `dyna_q`.

### idea 2: games, and whose point of view

for games the model is given exactly: the rules. MCTS only needs a small protocol: the starting position, whose turn it is, the legal moves, the position after a move, whether the game is over, and its value. two conventions make two-player zero-sum games easy:

- **players are $+1$ and $-1$**, and they alternate;
- **values are from the point of view of the player to move**. a position worth $v$ to the player to move is worth $-v$ to the other one (warm-up 3), so a search never needs to know who is "us": at every level it maximizes the value for whoever moves there, and flips the sign on the way up (**negamax**). in Connect Four the game ends right after a winning move, so at a terminal position the player to move has just lost: value $-1$ (or 0 for a full board).

**Connect Four**: a board of 6 rows and 7 columns, stones dropped into a column fall to the lowest empty cell, four in a row (horizontally, vertically or diagonally) wins. it's small enough to play in a notebook and big enough to be interesting: about $4.5\times10^{12}$ positions, first-player win with perfect play (Allis, 1988). for networks a state is encoded as two binary planes, the stones of the player to move and those of the opponent: the "canonical" view, so one network plays both sides.

> **spec.** `ConnectFour(rows=6, cols=7)` with attributes `rows`, `cols`, `n_actions = cols`. states are immutable (use tuples or bytes, never a shared mutable array). `initial_state()`: the empty board, player $+1$ to move. `current_player(state)`: $\pm1$. `legal_actions(state)`: the columns that aren't full, ascending (empty at a terminal state). `next_state(state, action)`: a *new* state with the mover's stone at the lowest free row of that column (a column outside `0..cols-1`, a full column or a finished game raise `ValueError`). `is_terminal(state)`: four in a row, or a full board. `terminal_value(state)`: $-1$ if the game was won (the player to move at a terminal state is the loser), 0 for a draw. `encode(state)`: float32 array `(2, rows, cols)`, plane 0 the stones of the player to move and plane 1 the opponent's, row 0 at the top (so the first stone in a column lands in row `rows - 1`).

> **▶ build: stage 2.** implement `ConnectFour`.

### idea 3: Monte Carlo tree search

MCTS builds a search tree one **simulation** at a time. each node stores, for every move $a$, a visit count $N(a)$, a total value $W(a)$ (so $Q(a) = W(a)/N(a)$, from the point of view of the player to move at that node), and a prior $P(a)$. a simulation has four phases:

1. **select**: from the root, repeatedly pick the move maximizing the **PUCT** score until reaching a node not yet in the tree,
$$a^* = \arg\max_a\Big[Q(a) + c_{\text{puct}}\,P(a)\,\frac{\sqrt{\sum_bN(b)}}{1 + N(a)}\Big],$$
(unvisited moves have $Q = 0$). the second term is an exploration bonus in the spirit of UCB (ch 23 · topic 01): large for moves with high prior and few visits, shrinking as a move is explored.
2. **expand**: add the new node, with priors over its legal moves.
3. **evaluate**: estimate the new position's value $v$ for the player to move there: the result of a **random rollout** to the end of the game (classic MCTS), or a network's value (AlphaZero). at a terminal node, use the true value.
4. **back up**: walk back to the root; at each step up, flip the sign ($v \leftarrow -v$) and add it to the parent's $W(a)$ for the move taken, and add 1 to $N(a)$.

after all simulations, the visit counts are the result: $\pi(a) \propto N(a)$. counts are more robust than averages (warm-up 4): a move with a high average from few visits may be lucky, while the search keeps visiting a move exactly as long as it keeps looking good. with random rollouts and uniform priors this is plain MCTS; it needs no knowledge beyond the rules, and with enough simulations it converges to minimax play (Kocsis & Szepesvári, 2006).

> **spec.** `MCTS(game, *, n_simulations=100, c_puct=1.0, evaluate=None, rng=None)` searches any game with `ConnectFour`'s protocol. `evaluate(state) -> (prior (n_actions,), value)` gives the prior over moves (renormalize it over the legal ones) and the value for the player to move; with `evaluate=None`, use a uniform prior over legal moves and the result of a uniformly random rollout (moves drawn with `as_rng(rng)`), converted to the leaf player's point of view. `search(state)` builds a fresh tree, runs exactly `n_simulations` simulations with PUCT selection and negamax backup, and returns the root's visit counts divided by their sum, an ndarray of shape `(n_actions,)` with zeros for illegal moves. the same `rng` must give the same result.

> **🔮 predict:** a node has visits $N = (10, 5, 0)$, total values $W = (6, -1, 0)$, priors $P = (0.4, 0.4, 0.2)$, $c_{\text{puct}} = 1$. which move does PUCT select next? (the worked example has it.)

> **▶ build: stage 3.** implement `MCTS`.

### idea 4: AlphaZero-lite

random rollouts are a poor evaluator (they assume both players play randomly), and uniform priors waste simulations on silly moves. **AlphaZero** replaces both with one network $f_\theta(s) = (\mathbf p, v)$: move logits for the prior, and a value in $[-1, 1]$ for the player to move. then it trains that network on games the search plays against itself:

- **self-play**: from the start position, run MCTS with $f_\theta$ as `evaluate`, record the encoded state and the visit distribution $\pi$, play a move sampled from $\pi$ (with a temperature: $\pi^{1/T}$, so early games explore), repeat until the game ends. then label every recorded state with the final result $z$ **from the point of view of the player to move at that state**: $+1$ if they went on to win, $-1$ if they lost, 0 for a draw.
- **training**: minimize $\ell = -\sum_a\pi(a)\log\operatorname{softmax}(\mathbf p)_a + (v - z)^2$ (plus weight decay) over recent games.

the logic (warm-up 5): the search with a prior is *stronger* than the prior alone, so its visit distribution is a better policy, and imitating it improves the network; the improved network then makes the search stronger still. it's policy iteration, with MCTS as the policy-improvement operator and the outcomes $z$ as Monte Carlo policy evaluation. "lite" here means a small MLP instead of a deep residual network, tens of simulations instead of 800, and minutes of CPU instead of thousands of TPUs; the algorithm is the same.

> **spec.** `PolicyValueNet(game, *, hidden_sizes=(128, 128))` is an `nn.Module` taking encodings `(B, *encode shape)` (flattened inside) to `(logits (B, n_actions), value (B,))` with a tanh value head; `evaluate(state)` returns `(softmax prior as an ndarray, value as a float)` without gradients, ready to pass to `MCTS`. `alphazero_loss(logits, values, target_pi, target_z)` returns the mean cross-entropy between `target_pi` and the softmax of `logits` plus the mean squared error between `values` and `target_z`. `self_play(game, mcts, *, temperature=1.0, rng=None)` plays one game as above (temperature 0 means play the most visited move; otherwise sample from $\pi^{1/T}$ with `as_rng(rng)`) and returns `(encodings (T, ...), pis (T, n_actions), z (T,))` as arrays.

> **▶ build: stage 4.** implement `PolicyValueNet`, `alphazero_loss` and `self_play`.

### idea 5: the training loop

the notebook's `train_alphazero` puts your pieces together: for each iteration, play a batch of self-play games with the current network inside MCTS, keep the most recent games in a buffer, and train for a few epochs on them. a full AlphaZero also pits each new network against the old one, uses Dirichlet noise on the root prior to keep exploring, and augments data with the board's left–right symmetry; the capstone's stretch goal adds those for full-size Connect Four.

## worked example

**Dyna-Q on a chain.** states 0 → 1 → 2 → 3 → 4 (terminal, reward $+1$), $\gamma = 0.9$, α = 1. after one real episode that reaches the goal, Q-learning has updated only the last pair: $Q(3, \text{right}) = 1$. Dyna-Q's planning replays the stored transitions until the values match the model's fixed point: $Q(3, \text{right}) = 1$, $Q(2, \text{right}) = 0.9$, $Q(1, \text{right}) = 0.81$, $Q(0, \text{right}) = 0.729$, and every observed "left" pair one discount further, e.g. $Q(1, \text{left}) = 0.9\max Q(0, \cdot) = 0.656$.

**a PUCT step.** $N = (10, 5, 0)$, $W = (6, -1, 0)$, $P = (0.4, 0.4, 0.2)$, $c_{\text{puct}} = 1$, so $\sqrt{\sum N} = \sqrt{15} = 3.873$. $Q = (0.6, -0.2, 0)$; the bonuses are $0.4\cdot3.873/11 = 0.141$, $0.4\cdot3.873/6 = 0.258$, $0.2\cdot3.873/1 = 0.775$. scores $(0.741, 0.058, 0.775)$: the search tries the unvisited third move, despite its low prior, before visiting the best-so-far move an eleventh time.

**a backup.** a simulation goes root → (move 2) → $s_1$ → (move 0) → $s_2$, a new leaf whose rollout ends in a win for the player to move at $s_2$: $v = +1$. at $s_1$, move 0 gets $W \mathrel{+}= -1$ (it led to a position good for the opponent); at the root, move 2 gets $W \mathrel{+}= +1$. both counts go up by one.

**the AlphaZero loss.** logits $((0, 0), (2, 0))$, values $(0.5, -0.5)$, targets $\pi = ((1, 0), (0.25, 0.75))$, $z = (1, -1)$. cross-entropies: $-\log0.5 = 0.693$ and $-(0.25\log0.881 + 0.75\log0.119) = 1.627$, mean 1.160; value errors $(0.5 - 1)^2 = (-0.5 + 1)^2 = 0.25$; total 1.410.

your code must reproduce these.

## on paper

1. **negamax.** for a game tree where leaves have values from the point of view of the player to move there, show that "each player maximizes $-$(child value)" computes the same root value as minimax from a fixed player's point of view.

<details><summary>solution</summary>

let $m(s)$ be the minimax value for player $+1$. at a node where $+1$ moves, $m(s) = \max_am(s_a)$; where $-1$ moves, $m(s) = \min_am(s_a)$. define $n(s) = \text{player}(s)\cdot m(s)$, the value for the player to move. at a $+1$ node: $n(s) = \max_am(s_a) = \max_a(-n(s_a))$, since $\text{player}(s_a) = -1$ gives $n(s_a) = -m(s_a)$. at a $-1$ node: $n(s) = -\min_am(s_a) = \max_a(-m(s_a)) = \max_a(-n(s_a))$. same recursion at every node. MCTS's sign flip in the backup is this, with averages in place of max.

</details>

2. **why the prior bonus decays like $1/(1 + N(a))$.** in PUCT, show that if all moves had equal $Q$, the visit counts would end up roughly proportional to the priors. why is that a sensible default?

<details><summary>solution</summary>

with equal $Q$, the selected move maximizes $P(a)/(1 + N(a))$ (the $\sqrt{\sum N}$ factor is common). greedy selection keeps these ratios balanced, so $N(a) + 1 \approx P(a)\cdot\text{const}$: visits follow the prior. that's what you want before the search has learned anything: trust the network's opinion, then let the values $Q$ override it as evidence accumulates. (UCB1's $\sqrt{\ln t/N}$ bonus is designed for regret guarantees without a prior; AlphaGo's variant trades those guarantees for using the prior.)

</details>

3. **the self-play labels.** a game ends after $T$ moves with a win for the player who made the last move. show that $z_T, z_{T-1}, \dots$ alternate $+1, -1, \dots$, starting from $+1$ at the last recorded state.

<details><summary>solution</summary>

the last recorded state is the one before the final move; the player to move there made the winning move, so $z = +1$. the state before it had the other player to move, who lost: $z = -1$. players alternate, so the labels alternate. (the checks test exactly this.)

</details>

4. **Dyna's value of a model.** in a deterministic chain of length $L$ with the goal at the end, Q-learning with α = 1 and a greedy policy that always moves right updates one more state per episode. how many episodes does it need until $Q(0, \text{right})$ is correct? how many does Dyna-Q with unlimited planning need?

<details><summary>solution</summary>

Q-learning: in episode $k$, the update at state $L - k$ is the first to see a nonzero $\max Q$ downstream, so $Q(0, \text{right})$ becomes correct in episode $L$. Dyna-Q: after the first episode the model contains the whole chain, and enough planning updates reach the fixed point, so one episode. the general lesson: planning substitutes computation for experience, up to the limit of what the experience has revealed.

</details>

## experiment

`lab.py` investigates planning budgets:

1. Dyna-Q on Sutton & Barto's maze with 0, 5 and 50 planning updates per step: steps per episode.
2. MCTS against MCTS on Connect Four: how often does a search with more simulations beat one with fewer?
3. AlphaZero-lite on a small board: a few minutes of self-play against plain MCTS with the same number of simulations.

## check your understanding

1. Dyna-Q's planning updates pick pairs uniformly at random. what would be a better order, and why?

<details><summary>answer</summary>

**prioritized sweeping** (Moore & Atkeson, 1993): update the pairs whose values would change most, starting from where something just changed (the goal, just discovered) and working backwards through predecessors. uniform sampling wastes most updates on pairs whose targets haven't changed. in large state spaces the difference is orders of magnitude.

</details>

2. why does MCTS return visit counts rather than the root's $Q$-values?

<details><summary>answer</summary>

visits integrate both the value estimates and their reliability: the search allocates simulations to moves that keep looking good, so a high count means "good and well-tested". a high $Q$ from two lucky visits is unreliable. counts also give a full distribution, which is what AlphaZero trains its policy on.

</details>

3. (the predict callout in idea 3.) which move does PUCT select?

<details><summary>answer</summary>

the third: its score $0 + 0.775$ narrowly beats the first's $0.6 + 0.141$. an unvisited move's bonus is $c\,P\sqrt{\sum N}$, which grows with the total visits, so every legal move with nonzero prior gets tried eventually.

</details>

4. a random rollout is an unbiased estimate of what?

<details><summary>answer</summary>

of the value of the position *if both players then play uniformly at random*: not of its value under good play. MCTS still converges to good play because the tree part of the search becomes more and more selective, and the rollouts start deeper and deeper; but early in the search, and in positions where random play is misleading (a forced win that requires precise moves), rollouts mislead it. that's exactly the weakness AlphaZero's value network removes.

</details>

5. why does AlphaZero play self-play moves by sampling from $\pi$ instead of always playing the most visited move?

<details><summary>answer</summary>

to generate diverse training positions. if both sides always played the argmax, every self-play game would be nearly identical, and the network would learn about a narrow set of positions and nothing about the rest. sampling (and the root's Dirichlet noise) explores openings; real AlphaZero samples for the first 30 moves and plays greedily afterwards.

</details>

6. Dyna-Q's model stores only the *last* outcome of each pair. when is that wrong, and what would you do instead?

<details><summary>answer</summary>

in a stochastic environment: a pair with two possible outcomes is modelled as whichever happened last, so planning backs up a random one of them. store counts of outcomes (an empirical distribution) and plan with expected updates, or sample from it.

</details>

## common bugs

- **mutable states.** symptom: the search corrupts positions it explored earlier. `next_state` must return a new object.
- **no sign flip in the backup.** symptom: MCTS plays the moves that are best for the *opponent*.
- **rollout values from the wrong point of view.** symptom: the search prefers losing moves in odd-length lines. convert the terminal value to the leaf player's view with the product of the two players.
- **illegal moves in the prior.** symptom: the search tries to drop a stone into a full column. mask and renormalize the network's prior over legal moves.
- **self-play labels from a fixed player's view.** symptom: the value head learns that every position is worth about 0 (half the labels have the wrong sign). label each state with the result for the player to move there.
- **Dyna planning on unseen pairs.** symptom: `KeyError`, or updates from made-up transitions. sample only pairs in the model.

## going deeper

- Sutton (1990), "Integrated architectures for learning, planning, and reacting based on approximating dynamic programming", *ICML*. Dyna. [doi:10.1016/B978-1-55860-141-3.50030-4](https://doi.org/10.1016/B978-1-55860-141-3.50030-4)
- Sutton & Barto (2018), *Reinforcement Learning: An Introduction*, 2nd ed., chapter 8 (planning and learning, Dyna, prioritized sweeping, MCTS). [incompleteideas.net/book/the-book-2nd.html](http://incompleteideas.net/book/the-book-2nd.html)
- Kocsis & Szepesvári (2006), "Bandit based Monte-Carlo planning", *ECML*. UCT. [doi:10.1007/11871842_29](https://doi.org/10.1007/11871842_29)
- Silver et al. (2017), "Mastering the game of Go without human knowledge", *Nature* 550. AlphaGo Zero. [doi:10.1038/nature24270](https://doi.org/10.1038/nature24270)
- Silver et al. (2018), "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play", *Science* 362. AlphaZero. [doi:10.1126/science.aar6404](https://doi.org/10.1126/science.aar6404)
- Schrittwieser et al. (2020), "Mastering Atari, Go, chess and shogi by planning with a learned model", *Nature* 588. MuZero. [arXiv:1911.08265](https://arxiv.org/abs/1911.08265)
- Janner, Fu, Zhang & Levine (2019), "When to trust your model: model-based policy optimization", *NeurIPS*. MBPO. [arXiv:1906.08253](https://arxiv.org/abs/1906.08253)
- Hafner, Pasukonis, Ba & Lillicrap (2023), "Mastering diverse domains through world models". DreamerV3. [arXiv:2301.04104](https://arxiv.org/abs/2301.04104)
- Allis (1988), *A knowledge-based approach of connect-four*, MSc thesis, Vrije Universiteit Amsterdam. the first-player win.

## connections

- ← builds on: Q-learning and `epsilon_greedy` (ch 23 · topics 04–05), value iteration as backups on a model (ch 23 · topic 03), UCB (ch 23 · topic 01), cross-entropy and MSE heads (ch 14–15).
- → used in: the capstone's stretch goal (AlphaZero-lite on full-size Connect Four, with Dirichlet noise and symmetry augmentation); search at inference time over a learned model reappears with language models (best-of-n, tree search over reasoning steps, ch 27); offline RL's distribution shift (topic 07) is the model-error problem of idea 1 in another guise.
