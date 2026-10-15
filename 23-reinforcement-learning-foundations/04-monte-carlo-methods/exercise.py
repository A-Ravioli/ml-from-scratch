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
    # monte carlo methods · build

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
        raise NotImplementedError("stage1: implement Blackjack.__init__")

    @staticmethod
    def encode(player_sum, dealer_card, usable_ace):
        """stage 1: (player_sum 4..21, dealer_card 1..10, usable_ace) -> obs int in [0, 360)."""
        raise NotImplementedError("stage1: implement Blackjack.encode")

    @staticmethod
    def decode(obs):
        """stage 1: obs -> (player_sum, dealer_card, usable_ace) as (int, int, bool)."""
        raise NotImplementedError("stage1: implement Blackjack.decode")

    @staticmethod
    def hand_value(cards):
        """stage 1: (total, usable_ace) for a list of cards, counting one ace as 11 when that doesn't bust."""
        raise NotImplementedError("stage1: implement Blackjack.hand_value")

    def draw(self):
        """stage 1: one card, min(self.np_random.integers(1, 14), 10) (face cards count 10, an ace 1)."""
        raise NotImplementedError("stage1: implement Blackjack.draw")

    def observe(self):
        """stage 1: the obs for the current hands (player sum clipped to 21 after a bust)."""
        raise NotImplementedError("stage1: implement Blackjack.observe")

    def reset(self, *, seed=None):
        """stage 1: seed, deal player, player, dealer, dealer (dealer[0] is showing); returns (obs, {})."""
        raise NotImplementedError("stage1: implement Blackjack.reset")

    def step(self, action):
        """stage 1: hit (1) draws a card, -1 on a bust; stick (0) plays the dealer out and scores +1/0/-1."""
        raise NotImplementedError("stage1: implement Blackjack.step")


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
    raise NotImplementedError("stage2: implement epsilon_greedy")


@app.function
def mc_prediction(env, policy, n_episodes, *, gamma=1.0, first_visit=True, rng=None):
    """stage 2: V (S,) for policy (callable obs -> action) by averaging first- or every-visit returns."""
    raise NotImplementedError("stage2: implement mc_prediction")


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
    raise NotImplementedError("stage3: implement mc_control")


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
    raise NotImplementedError("stage4: implement off_policy_mc_prediction")


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
