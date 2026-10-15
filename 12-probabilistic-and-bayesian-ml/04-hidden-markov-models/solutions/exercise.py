import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.base import Estimator  # yours, from chapter 07
    from mlfs.utils import as_rng  # yours, from chapter 00


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # hidden markov models · build (reference solution)

    work through the stages in order. an HMM has $K$ hidden states and $M$ observable
    symbols: a start distribution `startprob_` $\pi$ (K,), a transition matrix `transmat_`
    $A$ (K, K) with $A_{ij} = P(z_{t+1} = j \mid z_t = i)$, and an emission matrix
    `emissionprob_` $B$ (K, M) with $B_{jm} = P(x_t = m \mid z_t = j)$. `HMM` becomes part of
    `mlfs.bayes`.

    | stage | you build |
    |---|---|
    | 1 | `HMM.__init__`, `HMM.sample`: the generative story |
    | 2 | `HMM.forward_backward`: the likelihood and the posterior over each hidden state (scaled) |
    | 3 | `HMM.viterbi`: the single most probable hidden path (in log space) |
    | 4 | `HMM.fit` (Baum–Welch) and `HMM.score` → `mlfs.bayes`: learning the parameters with EM |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def as_sequences(X):
    """one int sequence -> [array]; a list of sequences -> list of arrays."""
    if isinstance(X, np.ndarray) and X.ndim == 1:
        return [X.astype(np.int64)]
    if isinstance(X, (list, tuple)) and len(X) and np.ndim(X[0]) == 0:
        return [np.asarray(X, dtype=np.int64)]
    return [np.asarray(s, dtype=np.int64) for s in X]


@app.class_definition
class HMM(Estimator):
    """a discrete-emission hidden Markov model with K states and M symbols.

    parameters (set by fit, or directly): startprob_ (K,), transmat_ (K, K), emissionprob_ (K, M).
    observations are int sequences with values in 0..M-1.
    """

    def __init__(self, n_states, n_symbols, *, max_iters=100, tol=1e-06, rng=None):
        """stage 1: store the hyperparameters unchanged."""
        self.n_states = n_states
        self.n_symbols = n_symbols
        self.max_iters = max_iters
        self.tol = tol
        self.rng = rng

    def sample(self, n_steps, *, rng=None):
        """stage 1: draw (obs (n_steps,), states (n_steps,)) int arrays from the model."""
        rng = as_rng(rng)
        K, M = self.emissionprob_.shape
        states = np.empty(n_steps, dtype=np.int64)
        obs = np.empty(n_steps, dtype=np.int64)
        z = rng.choice(K, p=self.startprob_)
        for t in range(n_steps):
            if t > 0:
                z = rng.choice(K, p=self.transmat_[z])
            states[t] = z
            obs[t] = rng.choice(M, p=self.emissionprob_[z])
        return obs, states

    def forward_backward(self, obs):
        """stage 2: (log p(obs), posteriors (T, K) with row t = P(z_t | obs)), with scaled messages."""
        alpha, beta, c = self.scaled_messages(np.asarray(obs, dtype=np.int64))
        post = alpha * beta
        post /= post.sum(axis=1, keepdims=True)
        return float(np.sum(np.log(c))), post

    def scaled_messages(self, obs):
        """stage 2: scaled messages (alpha (T, K), beta (T, K), c (T,)): alpha[t] = P(z_t | x_1..t), c[t] = P(x_t | x_1..t-1)."""
        T = len(obs)
        K = self.transmat_.shape[0]
        B = self.emissionprob_[:, obs].T  # (T, K): P(x_t | z_t = k)
        alpha = np.empty((T, K))
        c = np.empty(T)
        a = self.startprob_ * B[0]
        c[0] = a.sum()
        alpha[0] = a / c[0]
        for t in range(1, T):
            a = (alpha[t - 1] @ self.transmat_) * B[t]
            c[t] = a.sum()
            alpha[t] = a / c[t]
        beta = np.empty((T, K))
        beta[-1] = 1.0
        for t in range(T - 2, -1, -1):
            beta[t] = self.transmat_ @ (B[t + 1] * beta[t + 1]) / c[t + 1]
        return alpha, beta, c

    def viterbi(self, obs):
        """stage 3: the most probable state path (T,) and its log joint probability log p(obs, path)."""
        obs = np.asarray(obs, dtype=np.int64)
        T = len(obs)
        with np.errstate(divide="ignore"):
            logA = np.log(self.transmat_)
            logB = np.log(self.emissionprob_[:, obs].T)
            delta = np.log(self.startprob_) + logB[0]
        back = np.empty((T, len(delta)), dtype=np.int64)
        for t in range(1, T):
            scores = delta[:, None] + logA  # (from, to)
            back[t] = np.argmax(scores, axis=0)
            delta = scores[back[t], np.arange(len(delta))] + logB[t]
        path = np.empty(T, dtype=np.int64)
        path[-1] = int(np.argmax(delta))
        for t in range(T - 1, 0, -1):
            path[t - 1] = back[t, path[t]]
        return path, float(delta[path[-1]])

    def score(self, X, y=None):
        """stage 4: the total log-likelihood of one int sequence or a list of them."""
        return float(sum(self.forward_backward(seq)[0] for seq in as_sequences(X)))

    def fit(self, X, y=None):
        """stage 4: Baum-Welch (EM) on one sequence or a list; starts from the current parameters if all three are set."""
        seqs = as_sequences(X)
        K, M = self.n_states, self.n_symbols
        if not all(hasattr(self, a) for a in ("startprob_", "transmat_", "emissionprob_")):
            rng = as_rng(self.rng)
            self.startprob_ = rng.dirichlet(np.ones(K))
            self.transmat_ = rng.dirichlet(np.ones(K), size=K)
            self.emissionprob_ = rng.dirichlet(np.ones(M), size=K)
        self.history_ = []
        self.n_iter_ = 0
        for _ in range(self.max_iters):
            start = np.zeros(K)
            trans = np.zeros((K, K))
            emit = np.zeros((K, M))
            total = 0.0
            for obs in seqs:
                alpha, beta, c = self.scaled_messages(obs)
                total += float(np.sum(np.log(c)))
                gamma = alpha * beta
                gamma /= gamma.sum(axis=1, keepdims=True)
                B = self.emissionprob_[:, obs].T
                # xi[t, i, j] = P(z_t = i, z_t+1 = j | obs), summed over t
                xi = (alpha[:-1, :, None] * self.transmat_[None] * (B[1:] * beta[1:])[:, None, :]
                      / c[1:, None, None])
                start += gamma[0]
                trans += xi.sum(axis=0)
                np.add.at(emit.T, obs, gamma)
            self.history_.append(total)
            self.n_iter_ += 1
            self.startprob_ = start / start.sum()
            self.transmat_ = trans / np.maximum(trans.sum(axis=1, keepdims=True), 1e-300)
            self.emissionprob_ = emit / np.maximum(emit.sum(axis=1, keepdims=True), 1e-300)
            if len(self.history_) > 1 and self.history_[-1] - self.history_[-2] < self.tol:
                break
        return self


@app.cell(hide_code=True)
def _():
    checks.run(1, HMM=HMM)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    *`forward_backward` is a method of `HMM` above.*
    """)
    return


@app.cell(hide_code=True)
def _():
    checks.run(2, HMM=HMM)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    *`viterbi` is a method of `HMM` above.*
    """)
    return


@app.cell(hide_code=True)
def _():
    checks.run(3, HMM=HMM)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    *`fit` and `score` are methods of `HMM` above.*
    """)
    return


@app.cell(hide_code=True)
def _():
    checks.run(4, HMM=HMM)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, `HMM` is part of `mlfs.bayes` (try
    `from mlfs.bayes import HMM` in a scratch cell). then open the lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
