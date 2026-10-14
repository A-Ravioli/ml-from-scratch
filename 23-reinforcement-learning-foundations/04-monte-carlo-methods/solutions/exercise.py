import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.rl import Discrete, Env  # yours, from topic 02
    from mlfs.utils import as_rng  # yours, from ch 00 · topic 05


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # monte carlo methods · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson, gives you
    code to write, and checks it live as you edit. from here on there's no model: your
    agents learn from episodes they generate themselves.

    | stage | you build |
    |---|---|
    | 1 | `Blackjack`: the card game as a gym-style environment → `mlfs.rl` |
    | 2 | `epsilon_greedy`, `mc_prediction`: the exploration rule, and values from averaged returns → `mlfs.rl` |
    | 3 | `mc_control`: on-policy ε-soft Monte Carlo control → `mlfs.rl` |
    | 4 | `off_policy_mc_prediction`: evaluate one policy from another's episodes, by importance sampling → `mlfs.rl` |

    every learner here seeds its environment once from `rng`: the first episode calls
    `env.reset(seed=int(rng.integers(2**31)))`, later ones `env.reset()`. you can also run
    the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.class_definition
class Blackjack(Env):
    """Sutton & Barto's blackjack (infinite deck, dealer sticks on 17+); actions 0 stick, 1 hit.

    obs = encode(player_sum 4..21, dealer_card 1..10, usable_ace) = (player_sum - 4) * 20 +
    (dealer_card - 1) * 2 + usable_ace, so Discrete(360).
    """

    def __init__(self):
        """stage 1: observation_space = Discrete(360), action_space = Discrete(2)."""
        self.observation_space = Discrete(360)
        self.action_space = Discrete(2)

    @staticmethod
    def encode(player_sum, dealer_card, usable_ace):
        """stage 1: (player_sum 4..21, dealer_card 1..10, usable_ace) -> obs int in [0, 360)."""
        return (int(player_sum) - 4) * 20 + (int(dealer_card) - 1) * 2 + int(bool(usable_ace))

    @staticmethod
    def decode(obs):
        """stage 1: obs -> (player_sum, dealer_card, usable_ace) as (int, int, bool)."""
        obs = int(obs)
        return obs // 20 + 4, (obs % 20) // 2 + 1, bool(obs % 2)

    @staticmethod
    def hand_value(cards):
        """stage 1: (total, usable_ace) for a list of cards, counting one ace as 11 when that doesn't bust."""
        total = sum(cards)
        if 1 in cards and total + 10 <= 21:
            return total + 10, True
        return total, False

    def draw(self):
        """stage 1: one card, min(self.np_random.integers(1, 14), 10) (face cards count 10, an ace 1)."""
        return min(int(self.np_random.integers(1, 14)), 10)

    def observe(self):
        """stage 1: the obs for the current hands (player sum clipped to 21 after a bust)."""
        total, usable = self.hand_value(self.player)
        return self.encode(min(total, 21), self.dealer[0], usable)

    def reset(self, *, seed=None):
        """stage 1: seed, deal player, player, dealer, dealer (dealer[0] is showing); returns (obs, {})."""
        super().reset(seed=seed)
        self.player = [self.draw(), self.draw()]
        self.dealer = [self.draw(), self.draw()]
        return self.observe(), {}

    def step(self, action):
        """stage 1: hit (1) draws a card, -1 on a bust; stick (0) plays the dealer out and scores +1/0/-1."""
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action!r}: 0 (stick) or 1 (hit)")
        if action == 1:
            self.player.append(self.draw())
            total, _ = self.hand_value(self.player)
            if total > 21:
                return self.observe(), -1.0, True, False, {"player_sum": total}
            return self.observe(), 0.0, False, False, {"player_sum": total}
        while self.hand_value(self.dealer)[0] < 17:
            self.dealer.append(self.draw())
        mine, theirs = self.hand_value(self.player)[0], self.hand_value(self.dealer)[0]
        reward = 1.0 if theirs > 21 or mine > theirs else (0.0 if mine == theirs else -1.0)
        return self.observe(), reward, True, False, {"player_sum": mine, "dealer_sum": theirs}


@app.cell(hide_code=True)
def _():
    checks.run(1, Blackjack=Blackjack)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def epsilon_greedy(q_values, epsilon, *, rng=None):
    """stage 2: a random action with probability epsilon, else a greedy one (random ties); a Python int."""
    rng = as_rng(rng)
    q = np.asarray(q_values)
    if rng.random() < epsilon:
        return int(rng.integers(len(q)))
    return int(rng.choice(np.flatnonzero(q == q.max())))


@app.function
def mc_prediction(env, policy, n_episodes, *, gamma=1.0, first_visit=True, rng=None):
    """stage 2: V (S,) for policy (callable obs -> action) by averaging first- or every-visit returns."""
    rng = as_rng(rng)
    S = env.observation_space.n
    V, N = np.zeros(S), np.zeros(S)
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(2**31))) if ep == 0 else env.reset()
        states, rewards, done = [], [], False
        while not done:
            obs2, r, terminated, truncated, _ = env.step(policy(obs))
            states.append(obs)
            rewards.append(r)
            obs, done = obs2, terminated or truncated
        first = {}
        for t, s in enumerate(states):
            first.setdefault(s, t)
        G = 0.0
        for t in range(len(states) - 1, -1, -1):
            G = gamma * G + rewards[t]
            s = states[t]
            if not first_visit or first[s] == t:
                N[s] += 1
                V[s] += (G - V[s]) / N[s]
    return V


@app.cell(hide_code=True)
def _():
    checks.run(2, epsilon_greedy=epsilon_greedy, mc_prediction=mc_prediction, Blackjack=Blackjack)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def mc_control(env, n_episodes, *, gamma=1.0, epsilon=0.1, first_visit=True, rng=None):
    """stage 3: on-policy epsilon-soft MC control; returns (Q (S, A), list of episode returns)."""
    rng = as_rng(rng)
    S, A = env.observation_space.n, env.action_space.n
    Q, N = np.zeros((S, A)), np.zeros((S, A))
    episode_returns = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(2**31))) if ep == 0 else env.reset()
        pairs, rewards, done = [], [], False
        while not done:
            a = epsilon_greedy(Q[obs], epsilon, rng=rng)
            obs2, r, terminated, truncated, _ = env.step(a)
            pairs.append((obs, a))
            rewards.append(r)
            obs, done = obs2, terminated or truncated
        episode_returns.append(float(sum(rewards)))
        first = {}
        for t, sa in enumerate(pairs):
            first.setdefault(sa, t)
        G = 0.0
        for t in range(len(pairs) - 1, -1, -1):
            G = gamma * G + rewards[t]
            if not first_visit or first[pairs[t]] == t:
                s, a = pairs[t]
                N[s, a] += 1
                Q[s, a] += (G - Q[s, a]) / N[s, a]
    return Q, episode_returns


@app.cell(hide_code=True)
def _():
    checks.run(3, mc_control=mc_control, epsilon_greedy=epsilon_greedy, Blackjack=Blackjack)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def off_policy_mc_prediction(env, target_policy, behavior_policy, n_episodes, *, gamma=1.0, weighted=True, rng=None):
    """stage 4: Q^target (S, A) from behavior-policy episodes, ordinary or weighted importance sampling (every visit)."""
    rng = as_rng(rng)
    S, A = env.observation_space.n, env.action_space.n
    Q, C = np.zeros((S, A)), np.zeros((S, A))
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(2**31))) if ep == 0 else env.reset()
        pairs, rewards, done = [], [], False
        while not done:
            a = int(rng.choice(A, p=behavior_policy[obs]))
            obs2, r, terminated, truncated, _ = env.step(a)
            pairs.append((obs, a))
            rewards.append(r)
            obs, done = obs2, terminated or truncated
        G, W = 0.0, 1.0
        for t in range(len(pairs) - 1, -1, -1):
            s, a = pairs[t]
            G = gamma * G + rewards[t]
            if weighted:
                if W == 0.0:
                    break  # every earlier pair gets weight 0: nothing changes
                C[s, a] += W
                Q[s, a] += W / C[s, a] * (G - Q[s, a])
            else:
                C[s, a] += 1
                Q[s, a] += (W * G - Q[s, a]) / C[s, a]
            W *= target_policy[s, a] / behavior_policy[s, a]
    return Q


@app.cell(hide_code=True)
def _():
    checks.run(4, off_policy_mc_prediction=off_policy_mc_prediction, Blackjack=Blackjack)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, `Blackjack`, `epsilon_greedy`, `mc_control` and
    `off_policy_mc_prediction` are part of `mlfs.rl`. open the lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
