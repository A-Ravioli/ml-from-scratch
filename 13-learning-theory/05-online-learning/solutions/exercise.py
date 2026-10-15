import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.optim import project_l2_ball  # yours, from ch 06 · topic 06
    from mlfs.utils import logsumexp  # yours, from chapter 00


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # online learning · build (reference solution)

    work through the stages in order. no more iid samples: data arrives one round at a time,
    possibly chosen by an adversary, and you're judged by your **regret**, how much worse you
    did than the best fixed decision in hindsight. you'll build the two workhorse algorithms
    of online learning, Hedge (multiplicative weights over experts) and online gradient
    descent, the follow-the-leader baseline they improve on, and their regret bounds, then run
    OGD on a stream of real regression data.

    | stage | you build |
    |---|---|
    | 1 | `Hedge` → `mlfs.theory`: exponential weights over $N$ experts |
    | 2 | `OnlineGradientDescent` → `mlfs.theory`: a gradient step per round, $\eta_t = \eta/\sqrt t$, projected |
    | 3 | `follow_the_leader`, `regret_curve`: the greedy baseline and the yardstick |
    | 4 | `run_hedge`, `hedge_optimal_lr`, `hedge_regret_bound`: Hedge's $\sqrt{(T/2)\ln N}$ guarantee |
    | 5 | `ogd_regret_bound`, `online_least_squares`: OGD's $\frac32GD\sqrt T$ guarantee, on a data stream |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.class_definition
class Hedge:
    """multiplicative weights over n_experts: play p_t proportional to exp(-lr * cumulative loss)."""

    def __init__(self, n_experts, lr):
        """stage 1: store lr and start with zero cumulative loss for every expert (shape (n_experts,))."""
        self.n_experts = int(n_experts)
        self.lr = float(lr)
        self.cum_loss = np.zeros(self.n_experts)

    def predict(self):
        """stage 1: the current distribution over experts, softmax(-lr * cum_loss), shape (n_experts,), computed stably."""
        z = -self.lr * self.cum_loss
        return np.exp(z - logsumexp(z))

    def update(self, losses):
        """stage 1: add this round's losses (n_experts,) in [0, 1] to the cumulative losses; return self."""
        losses = np.asarray(losses, dtype=float)
        if losses.shape != (self.n_experts,):
            raise ValueError(f"losses must have shape ({self.n_experts},), got {losses.shape}")
        self.cum_loss = self.cum_loss + losses
        return self


@app.cell(hide_code=True)
def _():
    checks.run(1, Hedge=Hedge)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class OnlineGradientDescent:
    """projected online gradient descent: x_{t+1} = project(x_t - eta_t g_t), from x_1 = 0."""

    def __init__(self, dim, lr=1.0, *, project=None, decay="sqrt"):
        """stage 2: start at x = zeros(dim), t = 0; decay must be 'sqrt' (eta_t = lr / sqrt(t)) or None (eta_t = lr)."""
        if decay not in ("sqrt", None):
            raise ValueError("decay must be 'sqrt' or None")
        self.dim, self.lr, self.project, self.decay = int(dim), float(lr), project, decay
        self.x = np.zeros(self.dim)
        self.t = 0

    def predict(self):
        """stage 2: a copy of the current point x_t, shape (dim,)."""
        return self.x.copy()

    def update(self, grad):
        """stage 2: t += 1; x <- x - eta_t grad, then x <- project(x) if project is given; return self."""
        self.t += 1
        eta = self.lr / math.sqrt(self.t) if self.decay == "sqrt" else self.lr
        x = self.x - eta * np.asarray(grad, dtype=float)
        self.x = np.asarray(self.project(x), dtype=float) if self.project is not None else x
        return self


@app.cell(hide_code=True)
def _():
    checks.run(2, OnlineGradientDescent=OnlineGradientDescent)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def follow_the_leader(expert_losses):
    """stage 3: losses (T,) of the learner that plays, each round, the expert with the smallest cumulative loss so far (ties: lowest index)."""
    L = np.asarray(expert_losses, dtype=float)
    cum = np.vstack([np.zeros(L.shape[1]), np.cumsum(L, axis=0)[:-1]])  # cumulative loss BEFORE each round
    picks = np.argmin(cum, axis=1)
    return L[np.arange(len(L)), picks]


@app.function
def regret_curve(learner_losses, expert_losses):
    """stage 3: R_t = sum_{s<=t} learner_s - min_k sum_{s<=t} expert_losses[s, k], shape (T,)."""
    return np.cumsum(np.asarray(learner_losses, dtype=float)) - np.min(np.cumsum(np.asarray(expert_losses, dtype=float), axis=0),
                                                                        axis=1)


@app.cell(hide_code=True)
def _():
    checks.run(3, follow_the_leader=follow_the_leader, regret_curve=regret_curve)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def run_hedge(expert_losses, lr):
    """stage 4: Hedge's expected loss p_t . l_t in every round of expert_losses (T, N), shape (T,)."""
    L = np.asarray(expert_losses, dtype=float)
    h = Hedge(L.shape[1], lr)
    out = np.empty(len(L))
    for t, losses in enumerate(L):
        out[t] = h.predict() @ losses
        h.update(losses)
    return out


@app.function
def hedge_optimal_lr(T, n_experts):
    """stage 4: the learning rate sqrt(8 ln N / T) that minimizes the regret bound."""
    return math.sqrt(8 * math.log(n_experts) / T)


@app.function
def hedge_regret_bound(T, n_experts, lr):
    """stage 4: ln(N) / lr + lr T / 8, Hedge's regret bound for losses in [0, 1]."""
    return math.log(n_experts) / lr + lr * T / 8


@app.cell(hide_code=True)
def _():
    checks.run(4, run_hedge=run_hedge, hedge_optimal_lr=hedge_optimal_lr, hedge_regret_bound=hedge_regret_bound,
               regret_curve=regret_curve, Hedge=Hedge)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def ogd_regret_bound(T, diameter, grad_bound):
    """stage 5: (3/2) G D sqrt(T), the regret bound of OGD with lr = D / G and decay='sqrt' (Hazan, Thm 3.1)."""
    return 1.5 * grad_bound * diameter * math.sqrt(T)


@app.function
def online_least_squares(X, y, *, lr=1.0, radius=1.0, decay="sqrt"):
    """stage 5: OGD on f_t(w) = (w . x_t - y_t)^2 / 2 over the ball ||w|| <= radius, one row per round -> (losses (T,), final w)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    ogd = OnlineGradientDescent(X.shape[1], lr, project=lambda w: project_l2_ball(w, radius), decay=decay)
    losses = np.empty(len(y))
    for t in range(len(y)):
        w = ogd.predict()
        r = w @ X[t] - y[t]
        losses[t] = 0.5 * r * r
        ogd.update(r * X[t])
    return losses, ogd.predict()


@app.cell(hide_code=True)
def _():
    checks.run(5, ogd_regret_bound=ogd_regret_bound, online_least_squares=online_least_squares,
               OnlineGradientDescent=OnlineGradientDescent)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, `Hedge` and `OnlineGradientDescent` are part of `mlfs.theory`
    (try `from mlfs.theory import Hedge` in a scratch cell). then open the lab:
    `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
