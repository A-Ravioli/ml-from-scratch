# model-based RL & planning · hints

one hint per stage. open it only after you've been stuck for a while.

## stage 1

<details><summary>hint</summary>

start from your ch 23 `q_learning` and add two things after the real update: `model[(s, a)] = (r, s2, terminated)` (plus a list of the pairs seen, in insertion order, so you can sample one with `seen[int(rng.integers(len(seen)))]`), and a loop of `n_planning` updates that each pick a seen pair, look up its outcome, and apply the same Q-learning update. draw nothing from `rng` when `n_planning == 0`, so the run matches `q_learning` exactly.

</details>

## stage 2

<details><summary>hint</summary>

one convenient immutable state: `(board.tobytes(), player_to_move, winner, n_moves)` with `board` an int8 array of shape `(rows, cols)`. `next_state` copies the board (`np.frombuffer(...).reshape(...).copy()`), finds the lowest free row `r = np.flatnonzero(board[:, c] == 0)[-1]`, places the stone, and checks only the four lines through `(r, c)`: count same-coloured stones in both directions along each of `(0, 1)`, `(1, 0)`, `(1, 1)`, `(1, -1)`; four or more wins.

</details>

## stage 3

<details><summary>hint</summary>

store per node: `N`, `W` (arrays over all actions), `prior`, the legal moves, a dict of children, and whether it's terminal. a simulation:

```python
node, s, path = root, state, []
while node is expanded and not terminal:
    a = argmax over legal moves of W/N (0 if unvisited) + c * P * sqrt(N.sum()) / (1 + N)
    path.append((node, a)); s = game.next_state(s, a); node = child of node for a
v = expand(node, s)            # terminal value, or evaluate / rollout; for the player to move at s
for parent, a in reversed(path):
    v = -v
    parent.N[a] += 1; parent.W[a] += v
```

for a rollout from `s`, play random legal moves to the end and multiply the terminal value by `current_player(end) * current_player(s)`.

</details>

## stage 4

<details><summary>hint</summary>

the loss is `-(target_pi * F.log_softmax(logits, -1)).sum(-1).mean() + ((values - target_z) ** 2).mean()`. in `self_play`, keep the player to move with each record; at the end, `final = terminal_value(end) * current_player(end)` is $+1$ if player $+1$ won, and each record's label is `final * player`. sample moves with `rng.choice(n_actions, p=p / p.sum())` where `p = pi ** (1 / temperature)`.

</details>
